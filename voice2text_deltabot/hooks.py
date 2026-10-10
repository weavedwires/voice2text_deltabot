"""Event handlers and hooks."""

import logging
import os
import time
from argparse import Namespace

from deltabot_cli import BotCli
from deltachat2 import (
    Bot,
    ChatType,
    EventType,
    EventTypeError,
    EventTypeInfo,
    EventTypeSecurejoinInviterProgress,
    EventTypeWarning,
    Message,
    MessageData,
    NewMsgEvent,
    Viewtype,
    events,
)
from faster_whisper import WhisperModel
from rich.logging import RichHandler

from ._version import __version__

cli = BotCli("voice2text-bot")
cli.add_generic_option("-v", "--version", action="version", version=__version__)
cli.add_generic_option(
    "--no-time",
    help="do not display date timestamp in log messages",
    action="store_false",
)
cli.add_generic_option(
    "--max-duration",
    type=int,
    default=600,
    help="maximum amount of seconds to transcribe (default: %(default)s)",
)
cli.add_generic_option(
    "--model",
    help="set the whisper model to use, for example: small, medium, large. (default: %(default)s)",
    default="large",
)
cli.add_generic_option(
    "--compute-type",
    help="set the compute type (default: %(default)s)",
    choices=["int8", "float16", "int8_float16"],
    default="int8",
)
cli.add_generic_option(
    "--cpu-threads",
    type=int,
    default=0,
    help=(
        "number of CPU threads for the whisper model, 0 = all CPUs available to the"
        " process (default: %(default)s). CTranslate2 caps its own default at 4"
        " threads, which is why the model may use only a few cores on a bigger"
        " machine."
    ),
)
HELP = (
    "I'm a Delta Chat bot, send me any voice message to convert it to text,"
    " you can also use me in groups.\n\nNo 3rd party service is involved,"
    " only I will have access for a short period of time to the voice messages you send to me."
)
MODEL: WhisperModel = None  # noqa
ARGS = Namespace()


def physical_cpu_count(cpuinfo: str = "/proc/cpuinfo") -> int:
    """Number of physical cores, or 0 when it cannot be determined.

    Only Linux exposes this: `os.process_cpu_count()` counts logical CPUs, so with
    SMT enabled (and on some virtual machines) the physical count has to be read
    from /proc/cpuinfo, where it is the number of distinct (physical id, core id)
    pairs.
    """
    try:
        with open(cpuinfo, encoding="utf-8") as fp:
            lines = fp.read().splitlines()
    except OSError:
        return 0
    cores = set()
    physical_id = "0"
    for line in lines:
        if not line.strip():
            physical_id = "0"
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "physical id":
            physical_id = value
        elif key == "core id":
            cores.add((physical_id, value))
    return len(cores)


def resolve_cpu_threads(requested: int) -> int:
    """Threads to give the model: the flag if set, else every usable CPU."""
    if requested > 0:
        return requested
    return os.process_cpu_count() or os.cpu_count() or 1


@cli.on_init
def _on_init(bot: Bot, args: Namespace) -> None:
    global ARGS
    ARGS = args
    level = logging.DEBUG if bot.logger.level == logging.DEBUG else logging.ERROR
    logging.basicConfig(level=level)
    bot.logger.handlers = [
        RichHandler(show_path=False, omit_repeated_times=False, show_time=args.no_time)
    ]
    for accid in bot.rpc.get_all_account_ids():
        if not bot.rpc.get_config(accid, "displayname"):
            bot.rpc.set_config(accid, "displayname", "Voice To Text")
            status = (
                "I'm a Delta Chat bot, send me any voice message to convert it to text"
            )
            bot.rpc.set_config(accid, "selfstatus", status)
            bot.rpc.set_config(accid, "delete_device_after", str(60 * 60 * 24))


@cli.on_start
def on_start(bot: Bot, args: Namespace) -> None:
    global MODEL  # pylint: disable=W0603
    threads = resolve_cpu_threads(args.cpu_threads)
    bot.logger.info(
        "whisper: model=%s compute_type=%s cpu_threads=%d (logical=%s, physical=%s)",
        args.model,
        args.compute_type,
        threads,
        os.process_cpu_count(),
        physical_cpu_count() or "unknown",
    )
    MODEL = WhisperModel(
        args.model,
        device="auto",
        compute_type=args.compute_type,
        cpu_threads=threads,
    )


@cli.on(events.RawEvent)
def _log_event(bot: Bot, accid: int, event: EventType) -> None:
    if isinstance(event, EventTypeInfo):
        bot.logger.debug(event.msg)
    elif isinstance(event, EventTypeWarning):
        bot.logger.warning(event.msg)
    elif isinstance(event, EventTypeError):
        bot.logger.error(event.msg)
    elif isinstance(event, EventTypeSecurejoinInviterProgress):
        if event.progress == 1000:
            if not bot.rpc.get_contact(accid, event.contact_id).is_bot:
                bot.logger.debug("QR scanned by contact id=%s", event.contact_id)
                chatid = bot.rpc.create_chat_by_contact_id(accid, event.contact_id)
                bot.rpc.send_msg(accid, chatid, MessageData(text=HELP))


@cli.after(events.NewMessage)
def delete_msgs(bot: Bot, accid: int, event: NewMsgEvent) -> None:
    bot.rpc.delete_messages(accid, [event.msg.id])


@cli.on(events.NewMessage(is_info=False, is_bot=None))
def on_newmsg(bot: Bot, accid: int, event: NewMsgEvent) -> None:
    msg = event.msg
    chat = bot.rpc.get_basic_chat_info(accid, msg.chat_id)
    if chat.chat_type == ChatType.SINGLE:
        bot.rpc.markseen_msgs(accid, [msg.id])
        if msg.is_bot:
            return

    if msg.view_type in (Viewtype.VOICE, Viewtype.AUDIO):
        _process_audio(bot, accid, msg)
    elif chat.chat_type == ChatType.SINGLE:
        reply = MessageData(text=HELP)
        bot.rpc.send_msg(accid, msg.chat_id, reply)


def _process_audio(bot: Bot, accid: int, msg: Message) -> None:
    reply = MessageData(text="⏳", quoted_message_id=msg.id)
    msgid = bot.rpc.send_msg(accid, msg.chat_id, reply)
    start = time.time()
    segments, info = MODEL.transcribe(msg.file)
    lines = []
    last_time = 0
    for seg in segments:
        if seg.end > ARGS.max_duration:
            lines.append("[...]")
            break
        if seg.avg_logprob < -0.7 or seg.no_speech_prob > 0.5:
            continue
        text = seg.text.strip()
        if text.strip("."):
            lines.append(text)
        if seg.end - last_time > 10 and info.duration - seg.end > 10:
            last_time = seg.end
            bot.rpc.send_edit_request(accid, msgid, "\n".join(lines) + "\n\n⏳")
    took = time.time() - start
    percent = int(info.language_probability * 100)
    bot.logger.info(
        f"[chat={msg.chat_id}, msg={msg.id}] Voice extracted: duration={info.duration:.1f}"
        f" language={info.language} (probability={percent}%) took {took:.1f} seconds"
    )
    if lines:
        bot.rpc.send_edit_request(accid, msgid, "\n".join(lines))
    else:
        bot.rpc.delete_messages_for_all(accid, [msgid])
        bot.rpc.send_reaction(accid, msg.id, ["🎶"])
