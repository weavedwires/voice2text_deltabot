"""Unit tests for voice2text_deltabot.hooks."""

import dataclasses
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from deltachat2 import (
    BasicChat,
    ChatType,
    EventTypeError,
    EventTypeInfo,
    EventTypeSecurejoinInviterProgress,
    EventTypeWarning,
    Message,
    MessageData,
    Viewtype,
)

from voice2text_deltabot import hooks

BLOB = "/tmp/blob"


def test_hook_attribute_names_still_exist_in_deltachat2():
    """Guard the deltachat2 2.x rename (CoreEvent/MessageViewtype/MsgData -> ...)."""
    assert {"id", "chat_id", "file", "is_bot", "text", "view_type"} <= {
        f.name for f in dataclasses.fields(Message)
    }
    assert "chat_type" in {f.name for f in dataclasses.fields(BasicChat)}
    assert {"text", "quoted_message_id"} <= {
        f.name for f in dataclasses.fields(MessageData)
    }
    assert hasattr(ChatType, "SINGLE")
    assert hasattr(Viewtype, "VOICE") and hasattr(Viewtype, "AUDIO")


def make_msg(**kwargs):
    """Build a stand-in for deltachat2.Message with just the used attributes."""
    attrs = {
        "id": 1,
        "chat_id": 10,
        "file": BLOB,
        "is_bot": False,
        "text": "hello",
        "view_type": Viewtype.TEXT,
    }
    attrs.update(kwargs)
    return SimpleNamespace(**attrs)


def make_bot(chat_type=ChatType.SINGLE):
    """Build a stand-in for deltachat2.Bot with a mocked rpc."""
    bot = SimpleNamespace()
    bot.rpc = MagicMock()
    bot.rpc.send_msg.return_value = 42
    bot.rpc.get_basic_chat_info.return_value = SimpleNamespace(chat_type=chat_type)
    bot.logger = MagicMock()
    return bot


def seg(end, text="привет", avg_logprob=-0.1, no_speech_prob=0.05):
    """Build a stand-in for a faster-whisper segment."""
    return SimpleNamespace(
        end=end, text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob
    )


class FakeModel:
    """Stand-in for faster_whisper.WhisperModel."""

    def __init__(self, segments, duration=5.0, language="ru", probability=0.99):
        self.segments = segments
        self.info = SimpleNamespace(
            duration=duration, language=language, language_probability=probability
        )
        self.transcribed = []

    def transcribe(self, path):
        self.transcribed.append(path)
        return self.segments, self.info


@pytest.fixture(autouse=True)
def isolate_module_state():
    """Keep MODEL/ARGS changes out of other tests."""
    old_model, old_args = hooks.MODEL, hooks.ARGS
    hooks.MODEL = FakeModel([seg(1.0, "test")])
    hooks.ARGS = Namespace(max_duration=600)
    yield
    hooks.MODEL, hooks.ARGS = old_model, old_args


def newmsg_event(msg):
    return SimpleNamespace(msg=msg)


# --- on_newmsg -----------------------------------------------------------------


def test_single_chat_text_message_gets_help():
    bot = make_bot(ChatType.SINGLE)
    msg = make_msg(text="привет")
    hooks.on_newmsg(bot, 1, newmsg_event(msg))

    bot.rpc.markseen_msgs.assert_called_once_with(1, [msg.id])
    bot.rpc.send_msg.assert_called_once()
    accid, chat_id, data = bot.rpc.send_msg.call_args.args
    assert (accid, chat_id) == (1, msg.chat_id)
    assert isinstance(data, MessageData)
    assert data.text == hooks.HELP
    assert bot.rpc.send_edit_request.call_count == 0


def test_group_text_message_is_ignored():
    bot = make_bot(ChatType.GROUP)
    hooks.on_newmsg(bot, 1, newmsg_event(make_msg(text="привет")))

    bot.rpc.markseen_msgs.assert_not_called()
    bot.rpc.send_msg.assert_not_called()


def test_single_chat_message_from_bot_is_seen_but_not_answered():
    bot = make_bot(ChatType.SINGLE)
    msg = make_msg(text="я бот", is_bot=True)
    hooks.on_newmsg(bot, 1, newmsg_event(msg))

    bot.rpc.markseen_msgs.assert_called_once_with(1, [msg.id])
    bot.rpc.send_msg.assert_not_called()


def test_voice_message_is_transcribed():
    hooks.MODEL = FakeModel([seg(1.0, "привет"), seg(2.0, "как дела")], duration=3.0)
    bot = make_bot()

    hooks.on_newmsg(bot, 1, newmsg_event(make_msg(view_type=Viewtype.VOICE)))

    assert hooks.MODEL.transcribed == [BLOB]
    _, _, placeholder = bot.rpc.send_msg.call_args.args
    assert placeholder.text == "⏳"
    assert placeholder.quoted_message_id == 1
    bot.rpc.send_edit_request.assert_called_once_with(1, 42, "привет\nкак дела")


def test_audio_viewtype_is_transcribed_too():
    hooks.MODEL = FakeModel([seg(1.0, "текст")])
    bot = make_bot()

    hooks.on_newmsg(bot, 1, newmsg_event(make_msg(view_type=Viewtype.AUDIO)))

    assert hooks.MODEL.transcribed == [BLOB]
    bot.rpc.send_edit_request.assert_called_once_with(1, 42, "текст")


def test_voice_in_group_is_transcribed():
    bot = make_bot(ChatType.GROUP)
    hooks.on_newmsg(bot, 1, newmsg_event(make_msg(view_type=Viewtype.VOICE)))

    bot.rpc.markseen_msgs.assert_not_called()
    bot.rpc.send_edit_request.assert_called_once_with(1, 42, "test")


# --- _process_audio ------------------------------------------------------------


def test_noisy_segments_are_dropped():
    hooks.MODEL = FakeModel(
        [
            seg(1.0, "нормально"),
            seg(2.0, "мусор по вероятности", avg_logprob=-0.9),
            seg(3.0, "тишина", no_speech_prob=0.9),
            seg(4.0, "..."),
            seg(5.0, "   "),
            seg(6.0, "тоже нормально"),
        ],
        duration=7.0,
    )
    bot = make_bot()

    hooks._process_audio(bot, 1, make_msg(view_type=Viewtype.VOICE))

    bot.rpc.send_edit_request.assert_called_once_with(
        1, 42, "нормально\nтоже нормально"
    )


def test_empty_transcription_deletes_placeholder_and_reacts():
    hooks.MODEL = FakeModel([seg(1.0, "...", avg_logprob=-1.0)], duration=2.0)
    bot = make_bot()
    msg = make_msg(view_type=Viewtype.VOICE)

    hooks._process_audio(bot, 1, msg)

    bot.rpc.delete_messages_for_all.assert_called_once_with(1, [42])
    bot.rpc.send_reaction.assert_called_once_with(1, msg.id, ["🎶"])
    bot.rpc.send_edit_request.assert_not_called()


def test_max_duration_stops_and_marks_truncation():
    hooks.ARGS = Namespace(max_duration=30)
    hooks.MODEL = FakeModel([seg(10.0, "первое"), seg(40.0, "второе")], duration=90.0)
    bot = make_bot()

    hooks._process_audio(bot, 1, make_msg(view_type=Viewtype.VOICE))

    bot.rpc.send_edit_request.assert_called_once_with(1, 42, "первое\n[...]")


def test_long_transcription_sends_progress_edits():
    hooks.MODEL = FakeModel([seg(30.0, "раз"), seg(90.0, "два")], duration=100.0)
    bot = make_bot()

    hooks._process_audio(bot, 1, make_msg(view_type=Viewtype.VOICE))

    assert bot.rpc.send_edit_request.call_args_list == [
        call(1, 42, "раз\n\n⏳"),
        call(1, 42, "раз\nдва"),
    ]


# --- other hooks ---------------------------------------------------------------


def test_delete_msgs_post_hook_deletes_the_message():
    bot = make_bot()
    hooks.delete_msgs(bot, 1, newmsg_event(make_msg(id=7)))

    bot.rpc.delete_messages.assert_called_once_with(1, [7])


@pytest.mark.parametrize(
    "event,method",
    [
        (EventTypeInfo(msg="info-msg"), "debug"),
        (EventTypeWarning(msg="warning-msg"), "warning"),
        (EventTypeError(msg="error-msg"), "error"),
    ],
)
def test_raw_event_is_logged_with_the_matching_level(event, method):
    bot = make_bot()

    hooks._log_event(bot, 1, event)

    getattr(bot.logger, method).assert_called_once_with(event.msg)


def test_securejoin_progress_sends_help_to_humans():
    bot = make_bot()
    bot.rpc.get_contact.return_value = SimpleNamespace(is_bot=False)
    bot.rpc.create_chat_by_contact_id.return_value = 55
    event = EventTypeSecurejoinInviterProgress(
        chat_id=0, chat_type=ChatType.SINGLE, contact_id=7, progress=1000
    )

    hooks._log_event(bot, 1, event)

    bot.rpc.create_chat_by_contact_id.assert_called_once_with(1, 7)
    bot.rpc.send_msg.assert_called_once()
    accid, chat_id, data = bot.rpc.send_msg.call_args.args
    assert (accid, chat_id) == (1, 55)
    assert data.text == hooks.HELP


def test_securejoin_progress_skips_bots():
    bot = make_bot()
    bot.rpc.get_contact.return_value = SimpleNamespace(is_bot=True)
    event = EventTypeSecurejoinInviterProgress(
        chat_id=0, chat_type=ChatType.SINGLE, contact_id=7, progress=1000
    )

    hooks._log_event(bot, 1, event)

    bot.rpc.create_chat_by_contact_id.assert_not_called()
    bot.rpc.send_msg.assert_not_called()


def test_securejoin_incomplete_progress_is_ignored():
    bot = make_bot()
    bot.rpc.get_contact.return_value = SimpleNamespace(is_bot=False)
    event = EventTypeSecurejoinInviterProgress(
        chat_id=0, chat_type=ChatType.SINGLE, contact_id=7, progress=500
    )

    hooks._log_event(bot, 1, event)

    bot.rpc.get_contact.assert_not_called()
    bot.rpc.send_msg.assert_not_called()
