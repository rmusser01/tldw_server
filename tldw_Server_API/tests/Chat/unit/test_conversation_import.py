"""Rules for importing an "On this device" chat into the signed-in account (D7 P8).

These cover the pure half of ``POST /api/v1/chats/import``: the request is
validated, ordered parents-first and fingerprinted before any database work, so
a rejected import never opens a transaction.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat.conversation_import import (
    CHAT_DELETED,
    CHAT_ID_CONFLICT,
    ChatImportError,
    ChatImportLimits,
    import_fingerprint_from_authority,
    import_message_authority,
    normalize_client_conversation_id,
    prepare_chat_import,
    resolve_import_replay,
)

pytestmark = pytest.mark.unit

CHAT_ID = "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
LIMITS = ChatImportLimits(max_content_chars=1_000, max_image_bytes=64, max_total_image_bytes=100)
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _at(minutes: int) -> datetime:
    return datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes)


def _message(mid: str, parent: str | None, role: str = "user", minute: int = 0, **extra: Any) -> dict[str, Any]:
    return {
        "id": mid,
        "parent_message_id": parent,
        "role": role,
        "content": f"text of {mid}",
        "timestamp": _at(minute),
        "images": [],
        "metadata": None,
        **extra,
    }


def _request(messages: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "id": CHAT_ID,
        "title": "Trip planning",
        "state": None,
        "created_at": _at(0),
        "last_modified": None,
        "character_id": None,
        "assistant_kind": None,
        "assistant_id": None,
        "persona_memory_mode": None,
        "parent_conversation_id": None,
        "forked_from_message_id": None,
        "messages": messages if messages is not None else [_message("m1", None), _message("m2", "m1", "assistant", 1)],
        **extra,
    }


def _inspect(data: bytes) -> str:
    if not data.startswith(b"\x89PNG"):
        raise ChatImportError("invalid_image", 422, "not an image")
    return "image/png"


def _prepare(request: dict[str, Any], *, limits: ChatImportLimits = LIMITS, now: datetime = NOW):
    return prepare_chat_import(request, limits=limits, now=now, inspect_image=_inspect)


def _code(request: dict[str, Any], **kwargs: Any) -> tuple[str, int]:
    with pytest.raises(ChatImportError) as raised:
        _prepare(request, **kwargs)
    return raised.value.code, raised.value.status_code


# ---------------------------------------------------------------------------
# Conversation id
# ---------------------------------------------------------------------------


def test_conversation_id_is_stored_as_a_lowercase_canonical_uuid() -> None:
    assert normalize_client_conversation_id(CHAT_ID.upper()) == CHAT_ID


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-uuid",
        "",
        "6b0f8c1e2d4a4f3b9a715c2e8d9f0a14",
        "{6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14}",
        " 6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14",
        "00000000-0000-0000-0000-000000000000",
        "ffffffff-ffff-ffff-ffff-ffffffffffff",
    ],
)
def test_conversation_id_must_be_a_usable_uuid(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_client_conversation_id(bad)


# ---------------------------------------------------------------------------
# Message graph
# ---------------------------------------------------------------------------


def test_messages_are_ordered_parents_first_and_otherwise_keep_request_order() -> None:
    # A regenerated reply (m3) and an edited first message (m4) make two branches and two roots.
    request = _request(
        [
            _message("m3", "m1", "assistant", 2),
            _message("m2", "m1", "assistant", 1),
            _message("m5", "m4", "assistant", 4),
            _message("m1", None, "user", 0),
            _message("m4", None, "user", 3),
        ]
    )
    prepared = _prepare(request)
    assert [message["id"] for message in prepared.messages] == ["m1", "m3", "m2", "m4", "m5"]
    assert {message["id"]: message["parent_message_id"] for message in prepared.messages} == {
        "m1": None, "m2": "m1", "m3": "m1", "m4": None, "m5": "m4",
    }


def test_a_long_linear_chat_is_ordered_without_recursion() -> None:
    count = 5_000
    messages = [_message(f"m{index}", f"m{index - 1}" if index else None) for index in reversed(range(count))]
    prepared = _prepare(_request(messages), limits=ChatImportLimits(1_000, 64, 100))
    assert [message["id"] for message in prepared.messages] == [f"m{index}" for index in range(count)]


def test_parent_outside_the_import_is_rejected() -> None:
    assert _code(_request([_message("m1", "elsewhere")])) == ("missing_parent", 422)


def test_self_parent_is_rejected_as_a_cycle() -> None:
    assert _code(_request([_message("m1", "m1")])) == ("cyclic_ancestry", 422)


def test_longer_cycle_is_rejected() -> None:
    cycle = [_message("m1", "m3"), _message("m2", "m1"), _message("m3", "m2"), _message("root", None)]
    assert _code(_request(cycle)) == ("cyclic_ancestry", 422)


def test_duplicate_message_ids_are_rejected() -> None:
    assert _code(_request([_message("m1", None), _message("m1", None)])) == ("duplicate_message_id", 422)


def test_graph_errors_name_the_offending_message() -> None:
    with pytest.raises(ChatImportError) as raised:
        _prepare(_request([_message("m1", None), _message("m2", "gone")]))
    assert raised.value.detail() == {
        "error_code": "missing_parent",
        "message": raised.value.message,
        "message_id": "m2",
    }


# ---------------------------------------------------------------------------
# Timestamps are kept, never re-stamped
# ---------------------------------------------------------------------------


def test_timestamps_are_stored_in_utc_at_millisecond_precision() -> None:
    offset = timezone(timedelta(hours=2))
    request = _request(
        [_message("m1", None, timestamp=datetime(2026, 9, 1, 12, 30, 15, 123_456, tzinfo=offset))],
        created_at=datetime(2026, 9, 1, 12, 0, tzinfo=offset),
    )
    prepared = _prepare(request)
    assert prepared.messages[0]["timestamp"] == "2026-09-01T10:30:15.123Z"
    assert prepared.conversation["created_at"] == "2026-09-01T10:00:00.000Z"


def test_last_modified_defaults_to_the_newest_message_not_to_now() -> None:
    prepared = _prepare(_request())
    assert prepared.conversation["last_modified"] == "2026-09-01T10:01:00.000Z"
    explicit = _prepare(_request(last_modified=_at(90)))
    assert explicit.conversation["last_modified"] == "2026-09-01T11:30:00.000Z"


def test_last_modified_is_never_before_created_at() -> None:
    prepared = _prepare(_request([_message("m1", None, minute=-30)]))
    assert prepared.conversation["last_modified"] == prepared.conversation["created_at"]


@pytest.mark.parametrize("field", ["created_at", "last_modified", "message"])
def test_timestamps_beyond_the_clock_skew_allowance_are_rejected(field: str) -> None:
    future = NOW + timedelta(minutes=6)
    request = _request([_message("m1", None, timestamp=future)]) if field == "message" else _request(**{field: future})
    assert _code(request) == ("timestamp_in_future", 422)
    within = NOW + timedelta(minutes=4)
    allowed = _request([_message("m1", None, timestamp=within)]) if field == "message" else _request(**{field: within})
    assert _prepare(allowed).fingerprint


def test_naive_timestamp_is_rejected() -> None:
    assert _code(_request(created_at=datetime(2026, 9, 1, 10, 0))) == ("invalid_timestamp", 422)


@pytest.mark.parametrize(
    "extreme",
    [
        datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone(timedelta(hours=-10))),
        datetime(1, 1, 1, tzinfo=timezone(timedelta(hours=10))),
    ],
    ids=["past-the-last-year", "before-the-first-year"],
)
def test_timestamp_that_cannot_be_expressed_in_utc_is_rejected(extreme: datetime) -> None:
    """Converting these to UTC leaves the range of dates; that is an invalid timestamp, not a crash."""
    assert _code(_request([_message("m1", None, timestamp=extreme)])) == ("invalid_timestamp", 422)
    assert _code(_request(created_at=extreme)) == ("invalid_timestamp", 422)
    assert _code(_request(last_modified=extreme)) == ("invalid_timestamp", 422)


def test_pre_epoch_timestamp_is_rejected() -> None:
    old = datetime(1969, 12, 31, tzinfo=timezone.utc)
    assert _code(_request([_message("m1", None, timestamp=old)])) == ("invalid_timestamp", 422)


# ---------------------------------------------------------------------------
# Content, metadata and images
# ---------------------------------------------------------------------------


def test_content_is_kept_verbatim() -> None:
    text = "  leading and trailing spaces kept \n\n{{char}} <b>raw</b>  "
    prepared = _prepare(_request([_message("m1", None, content=text)]))
    assert prepared.messages[0]["content"] == text


def test_message_needs_text_or_an_image() -> None:
    assert _code(_request([_message("m1", None, content="   ")])) == ("empty_message", 422)
    image_only = _message("m1", None, content="", images=[base64.b64encode(PNG).decode()])
    assert _prepare(_request([image_only])).messages[0]["images"] == [{"data": PNG, "mime": "image/png"}]


def test_content_over_the_message_limit_is_rejected() -> None:
    assert _code(_request([_message("m1", None, content="x" * 1_001)])) == ("message_content_too_large", 413)
    assert _prepare(_request([_message("m1", None, content="x" * 1_000)])).fingerprint


def test_content_length_uses_the_callers_measure() -> None:
    request = _request([_message("m1", None, content="{{char}}" * 200)])
    assert _code(request) == ("message_content_too_large", 413)
    prepared = prepare_chat_import(
        request, limits=LIMITS, now=NOW, inspect_image=_inspect, content_length=lambda text: text.count("{{char}}")
    )
    assert prepared.messages[0]["content"] == "{{char}}" * 200


def test_nul_characters_are_rejected() -> None:
    assert _code(_request([_message("m1", None, content="a\x00b")])) == ("invalid_text", 422)
    assert _code(_request(title="a\x00b")) == ("invalid_text", 422)


def test_text_that_cannot_be_stored_as_utf8_is_rejected() -> None:
    """JSON may carry half of a surrogate pair (a cut emoji); the database cannot store it."""
    assert _code(_request([_message("m1", None, content="cut \ud83d emoji")])) == ("invalid_text", 422)
    assert _code(_request(title="cut \ud83d emoji")) == ("invalid_text", 422)
    whole = _prepare(_request([_message("m1", None, content="whole \U0001f600 emoji")]))
    assert whole.messages[0]["content"] == "whole \U0001f600 emoji"


def test_blank_title_is_rejected() -> None:
    assert _code(_request(title="   ")) == ("invalid_title", 422)


def test_allow_listed_metadata_is_stored_with_the_sender_role() -> None:
    metadata = {
        "model_id": "gpt-4o-mini",
        "provider": "openai",
        "finish_reason": "stop",
        "generation_status": "complete",
        "usage": {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42},
    }
    prepared = _prepare(_request([_message("m1", None), _message("m2", "m1", "assistant", 1, metadata=metadata)]))
    assert prepared.messages[0]["extra_metadata"] == {"sender_role": "user"}
    assert prepared.messages[1]["extra_metadata"] == {
        "sender_role": "assistant",
        "model_id": "gpt-4o-mini",
        "provider": "openai",
        "finish_reason": "stop",
        "generation_status": "complete",
        "usage": {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42},
    }


@pytest.mark.parametrize(
    "metadata",
    [
        {"sources": ["https://example.com"]},
        {"generation_status": "complete", "sender_role": "system"},
        {"generation_status": "finished"},
        {"usage": {"prompt_tokens": 1, "cost": 2}},
        {"model_id": ""},
        {"model_id": " gpt-4o "},
        {"provider": "openai "},
        {"finish_reason": "\tstop"},
    ],
    ids=[
        "unknown-key", "reserved-key", "bad-status", "bad-usage", "empty-model",
        "padded-model", "padded-provider", "padded-finish-reason",
    ],
)
def test_metadata_outside_the_allow_list_is_rejected(metadata: dict[str, Any]) -> None:
    request = _request([_message("m1", None), _message("m2", "m1", "assistant", 1, metadata=metadata)])
    with pytest.raises(ChatImportError) as raised:
        _prepare(request)
    assert (raised.value.code, raised.value.status_code) == ("unsupported_metadata", 422)
    assert raised.value.detail()["message_id"] == "m2"


def test_unknown_metadata_keys_are_named_briefly_and_in_plain_ascii() -> None:
    """The refusal names the keys, but never echoes long or unencodable text back."""
    metadata = {"x" * 5_000: 1, "cut \ud83d key": 2, **{f"key{index}": index for index in range(40)}}
    request = _request([_message("m1", None), _message("m2", "m1", "assistant", 1, metadata=metadata)])
    with pytest.raises(ChatImportError) as raised:
        _prepare(request)
    detail = raised.value.detail()
    assert detail["error_code"] == "unsupported_metadata"
    encoded = json.dumps(detail, ensure_ascii=False).encode("utf-8")
    assert len(encoded) < 600
    assert detail["message"].isascii()
    assert "key0" in detail["message"]


def test_generation_metadata_is_only_accepted_on_assistant_messages() -> None:
    request = _request([_message("m1", None, "user", metadata={"model_id": "gpt-4o"})])
    assert _code(request) == ("unsupported_metadata", 422)
    assert _prepare(_request([_message("m1", None, "user", metadata={})])).messages[0]["extra_metadata"] == {
        "sender_role": "user"
    }


def test_images_accept_data_urls_and_bare_base64_and_keep_their_order() -> None:
    second = PNG + b"\x01"
    images = [f"data:image/png;base64,{base64.b64encode(PNG).decode()}", base64.b64encode(second).decode()]
    prepared = _prepare(_request([_message("m1", None, images=images)]))
    assert prepared.messages[0]["images"] == [
        {"data": PNG, "mime": "image/png"},
        {"data": second, "mime": "image/png"},
    ]


@pytest.mark.parametrize("bad", ["data:image/png;base64,", "not base64 !!", "data:image/png,plain", "%%%%"])
def test_undecodable_images_are_rejected(bad: str) -> None:
    assert _code(_request([_message("m1", None, images=[bad])])) == ("invalid_image", 422)


def test_unsupported_image_bytes_are_rejected() -> None:
    not_image = base64.b64encode(b"just some text bytes").decode()
    assert _code(_request([_message("m1", None, images=[not_image])])) == ("invalid_image", 422)


def test_image_over_the_per_image_limit_is_rejected() -> None:
    big = base64.b64encode(PNG + b"\x00" * 64).decode()
    assert _code(_request([_message("m1", None, images=[big])])) == ("image_too_large", 413)


def test_padded_image_text_is_refused_on_its_length_before_it_is_processed() -> None:
    """Whitespace inside base64 is allowed, but not as a way to send megabytes of padding."""
    wrapped = "\n".join(base64.b64encode(PNG).decode()[index:index + 8] for index in range(0, 32, 8))
    assert _prepare(_request([_message("m1", None, images=[wrapped])])).messages[0]["images"][0]["data"] == PNG
    padded = base64.b64encode(PNG).decode() + " " * 10_000
    assert _code(_request([_message("m1", None, images=[padded])])) == ("image_too_large", 413)


def test_images_over_the_import_total_are_rejected() -> None:
    image = base64.b64encode(PNG + b"\x00" * 30).decode()  # 54 bytes each, 100 allowed in total
    messages = [_message("m1", None, images=[image]), _message("m2", "m1", images=[image])]
    assert _code(_request(messages)) == ("images_too_large", 413)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_a_sha256_hex_digest() -> None:
    fingerprint = _prepare(_request()).fingerprint
    assert len(fingerprint) == 64 and set(fingerprint) <= set("0123456789abcdef")


def test_equivalent_requests_share_a_fingerprint() -> None:
    base = _prepare(_request()).fingerprint
    offset = timezone(timedelta(hours=-5))
    shifted = _request(created_at=_at(0).astimezone(offset))
    assert _prepare(shifted).fingerprint == base
    # The same image bytes sent as a data URL or as bare base64 are the same image.
    bare = _request([_message("m1", None, images=[base64.b64encode(PNG).decode()])])
    url = _request([_message("m1", None, images=[f"data:image/png;base64,{base64.b64encode(PNG).decode()}"])])
    assert _prepare(bare).fingerprint == _prepare(url).fingerprint
    # A later retry is fingerprinted the same: the server clock is not part of it.
    assert _prepare(_request(), now=NOW + timedelta(days=30)).fingerprint == base


def test_fingerprint_is_bound_to_the_conversation_id() -> None:
    """The fingerprint is stored with the messages, so it must not match under another chat id."""
    other = _request(id="9d2f6a70-1c3b-4e5d-8f90-a1b2c3d4e5f6")
    assert _prepare(other).fingerprint != _prepare(_request()).fingerprint
    assert _prepare(_request(id=CHAT_ID.upper())).fingerprint == _prepare(_request()).fingerprint


@pytest.mark.parametrize(
    "change",
    [
        {"title": "Another title"},
        {"state": "resolved"},
        {"created_at": _at(5)},
        {"last_modified": _at(50)},
        {"character_id": 7, "assistant_kind": "character", "assistant_id": "7"},
        {"parent_conversation_id": "parent"},
    ],
    ids=["title", "state", "created_at", "last_modified", "character", "parent"],
)
def test_conversation_changes_change_the_fingerprint(change: dict[str, Any]) -> None:
    assert _prepare(_request(**change)).fingerprint != _prepare(_request()).fingerprint


@pytest.mark.parametrize(
    "messages",
    [
        [_message("m1", None), _message("m2", "m1", "assistant", 1, content="edited")],
        [_message("m1", None), _message("m2", None, "assistant", 1)],
        [_message("m1", None), _message("m2", "m1", "assistant", 2)],
        [_message("m1", None), _message("m2", "m1", "user", 1)],
        [_message("m1", None), _message("m2", "m1", "assistant", 1, metadata={"model_id": "gpt-4o"})],
        [_message("m1", None), _message("m2", "m1", "assistant", 1, images=[base64.b64encode(PNG).decode()])],
        [_message("m1", None), _message("m2", "m1", "assistant", 1), _message("m3", "m2", "user", 2)],
        [_message("m1", None)],
    ],
    ids=["content", "parent", "timestamp", "role", "metadata", "image", "added", "removed"],
)
def test_message_changes_change_the_fingerprint(messages: list[dict[str, Any]]) -> None:
    assert _prepare(_request(messages)).fingerprint != _prepare(_request()).fingerprint


def test_fingerprint_follows_the_stored_order_not_the_order_of_the_request() -> None:
    """Listing the same messages in another order stores the same chat, so it is the same import."""
    listed_backwards = _request([_message("m2", "m1", "assistant", 1), _message("m1", None)])
    assert _prepare(listed_backwards).fingerprint == _prepare(_request()).fingerprint
    # Equal timestamps are stored in the order they were sent, so there the order is part of the chat.
    tied = [_message("a", None, minute=3), _message("b", None, minute=3)]
    assert _prepare(_request(tied)).fingerprint != _prepare(_request(tied[::-1])).fingerprint
    # A parent is always stored before its child, however a tie between them was listed.
    family = [_message("parent", None, minute=3), _message("child", "parent", minute=3)]
    assert _prepare(_request(family)).fingerprint == _prepare(_request(family[::-1])).fingerprint


def test_omitted_state_is_the_default_state() -> None:
    assert _prepare(_request(state="in-progress")).fingerprint == _prepare(_request()).fingerprint
    assert _prepare(_request()).conversation["state"] == "in-progress"


# ---------------------------------------------------------------------------
# Provenance and replay
# ---------------------------------------------------------------------------


def test_import_authority_marks_messages_as_a_settled_parent_graph() -> None:
    fingerprint = "a" * 64
    authority = import_message_authority(fingerprint)
    assert authority["version"] == 1
    assert authority["interpretation"] == {"kind": "parent_graph_v1"}
    assert authority["settled"] is True
    assert import_fingerprint_from_authority(json.dumps(authority)) == fingerprint
    assert import_fingerprint_from_authority(authority) == fingerprint


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "not json",
        "[]",
        json.dumps({"version": 1, "settled": True}),
        json.dumps({"import": {"version": 1, "request_fingerprint": "A" * 64}}),
        json.dumps({"import": {"version": 1, "request_fingerprint": "a" * 63}}),
        json.dumps({"import": {"version": 2, "request_fingerprint": "a" * 64}}),
        json.dumps({"selection": {"import": {"version": 1, "request_fingerprint": "a" * 64}}}),
    ],
)
def test_other_provenance_never_reads_as_an_import(raw: Any) -> None:
    assert import_fingerprint_from_authority(raw) is None


def test_replay_is_only_for_the_same_owner_and_the_same_request() -> None:
    fingerprint = "a" * 64
    row = {"id": CHAT_ID, "client_id": "1", "deleted": 0}
    assert resolve_import_replay(None, owner_id="1", stored_fingerprint=None, fingerprint=fingerprint) is None
    assert resolve_import_replay(row, owner_id="1", stored_fingerprint=fingerprint, fingerprint=fingerprint) is row
    assert resolve_import_replay(row, owner_id=1, stored_fingerprint=fingerprint, fingerprint=fingerprint) is row

    def refused(**kwargs: Any) -> tuple[str, int]:
        with pytest.raises(ChatImportError) as raised:
            resolve_import_replay(**{"owner_id": "1", "stored_fingerprint": fingerprint, "fingerprint": fingerprint, **kwargs})
        return raised.value.code, raised.value.status_code

    assert refused(existing=row, owner_id="2") == (CHAT_ID_CONFLICT, 409)
    assert refused(existing=row, fingerprint="b" * 64) == (CHAT_ID_CONFLICT, 409)
    assert refused(existing=row, stored_fingerprint=None) == (CHAT_ID_CONFLICT, 409)
    assert refused(existing={**row, "deleted": 1}) == (CHAT_DELETED, 410)
    # A trashed chat that is not this import is still only a conflict: nothing about it is revealed.
    assert refused(existing={**row, "deleted": 1}, fingerprint="b" * 64) == (CHAT_ID_CONFLICT, 409)
    assert refused(existing={**row, "deleted": 1}, owner_id="2") == (CHAT_ID_CONFLICT, 409)


def test_refusals_do_not_suggest_that_a_new_chat_id_alone_is_enough() -> None:
    """Message ids of the existing chat stay taken, so only new ids throughout can be imported."""
    row = {"id": CHAT_ID, "client_id": "1", "deleted": 1}
    for kwargs in ({"fingerprint": "a" * 64}, {"fingerprint": "b" * 64}):
        with pytest.raises(ChatImportError) as raised:
            resolve_import_replay(row, owner_id="1", stored_fingerprint="a" * 64, **kwargs)
        assert "new id" not in raised.value.message
        assert "message" in raised.value.message


def test_refusals_carry_no_detail_of_the_existing_chat() -> None:
    row = {"id": CHAT_ID, "client_id": "1", "deleted": 0, "title": "Owner one private plans"}
    with pytest.raises(ChatImportError) as raised:
        resolve_import_replay(row, owner_id="2", stored_fingerprint="a" * 64, fingerprint="a" * 64)
    assert set(raised.value.detail()) == {"error_code", "message"}
    assert "private" not in json.dumps(raised.value.detail())
