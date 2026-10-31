from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock

import pytest

from tldw_Server_API.app.core.RAG.rag_service.result_model import RAGResult
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.Slides import standalone_html_sources as sources_module
from tldw_Server_API.app.core.Slides.standalone_html_config import (
    StandaloneHtmlInputLimits,
)
from tldw_Server_API.app.core.Slides.standalone_html_sources import (
    StandaloneHtmlSourceError,
    resolve_standalone_html_source,
)

pytestmark = pytest.mark.unit


def _limits(*, chars: int = 200, tokens: int = 100) -> StandaloneHtmlInputLimits:
    return StandaloneHtmlInputLimits(
        max_request_bytes=4_194_304,
        max_source_chars=chars,
        max_source_tokens=tokens,
        max_audience_chars=500,
        max_source_identifier_bytes=256,
        max_note_ids=100,
        max_rag_query_chars=20_000,
        max_rag_top_k=100,
    )


class _TokenCounter:
    def __init__(self, result: int = 1) -> None:
        self.result = result
        self.calls: list[str] = []

    def __call__(self, text: str) -> int:
        self.calls.append(text)
        return self.result


class _MessageStore:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        conversation_exists: bool = True,
        projection_truncated: bool = False,
        projection_invalid: bool = False,
    ) -> None:
        self.rows = rows
        self.conversation_exists = conversation_exists
        self.projection_truncated = projection_truncated
        self.projection_invalid = projection_invalid
        self.projection_calls: list[tuple[str, int, str | None]] = []

    def get_source_message_projection(
        self,
        conversation_id: str,
        *,
        max_chars: int,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        self.projection_calls.append((conversation_id, max_chars, owner_user_id))
        if not self.conversation_exists:
            return {
                "rows": [],
                "conversation_exists": False,
                "invalid": self.projection_invalid,
                "truncated": False,
            }
        budget = max_chars + 1
        projected: list[dict[str, str]] = []
        truncated = self.projection_truncated
        for row in self.rows:
            full_text = row.get("source_text")
            if full_text is None:
                full_text = f"{row.get('sender') or 'unknown'}: {row.get('content') or ''}"
            separator_chars = int(bool(projected))
            available = budget - separator_chars
            if available <= 0:
                truncated = True
                break
            bounded_text = full_text[:available]
            projected.append({"source_text": bounded_text})
            budget -= separator_chars + len(bounded_text)
            if len(full_text) > len(bounded_text):
                truncated = True
                break
        return {
            "rows": projected,
            "conversation_exists": True,
            "invalid": self.projection_invalid,
            "truncated": truncated,
        }


class _NoteStore:
    def __init__(self, rows: dict[str, dict[str, Any] | None]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, int, str | None]] = []

    def get_source_note_projection(
        self,
        note_id: str,
        *,
        max_chars: int,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | None:
        self.calls.append((note_id, max_chars, owner_user_id))
        row = self.rows.get(note_id)
        if row is None:
            return None
        title = str(row.get("title") or "")
        content = str(row.get("content") or "")
        source_text = f"# {title}\n\n{content}" if title and content else f"# {title}" if title else content
        return {
            "id": row["id"],
            "source_text": source_text[: max_chars + 1],
            "source_invalid": "\0" in title or "\0" in content,
        }


class _ChaChaDb:
    def __init__(
        self,
        *,
        messages: _MessageStore | None = None,
        notes: _NoteStore | None = None,
    ) -> None:
        self.message_store = messages or _MessageStore([])
        self.note_store = notes or _NoteStore({})
        self.db_path_str = "/owner/chacha.db"


class _MediaDb:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row
        self.calls: list[tuple[int, int, str | None]] = []
        self.db_path_str = "/owner/media.db"

    def get_media_source_projection(
        self,
        media_id: int,
        *,
        max_chars: int,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | None:
        self.calls.append((media_id, max_chars, owner_user_id))
        if self.row is None:
            return None
        row = dict(self.row)
        row.setdefault("source_invalid", False)
        if isinstance(row.get("source_text"), str):
            row["source_text"] = row["source_text"][: max_chars + 1]
        return row


@pytest.mark.asyncio
async def test_prompt_snapshot_preserves_exact_text_and_provenance() -> None:
    counter = _TokenCounter(result=3)
    prompt = "  Keep intentional outer whitespace.  "

    snapshot = await resolve_standalone_html_source(
        {"kind": "prompt", "prompt": prompt},
        owner_user_id="owner-1",
        limits=_limits(),
        token_counter=counter,
    )

    assert snapshot.text == prompt
    assert snapshot.char_count == len(prompt)
    assert snapshot.byte_count == len(prompt.encode("utf-8"))
    assert snapshot.token_count == 3
    assert snapshot.provenance.source_kind == "prompt"
    assert snapshot.provenance.source_ref is None
    assert snapshot.provenance.reference_hmac_input is None
    assert snapshot.provenance.summary == {
        "source_kind": "prompt",
        "source_ref": None,
    }
    assert counter.calls == [prompt]


@pytest.mark.asyncio
@pytest.mark.parametrize(("size", "accepted"), [(10, True), (11, False)])
async def test_prompt_character_boundary_precedes_tokenization(
    size: int,
    accepted: bool,
) -> None:
    counter = _TokenCounter()
    prompt = "x" * size

    if accepted:
        snapshot = await resolve_standalone_html_source(
            {"kind": "prompt", "prompt": prompt},
            owner_user_id="owner-1",
            limits=_limits(chars=10),
            token_counter=counter,
        )
        assert snapshot.text == prompt
        assert counter.calls == [prompt]
    else:
        with pytest.raises(StandaloneHtmlSourceError) as exc_info:
            await resolve_standalone_html_source(
                {"kind": "prompt", "prompt": prompt},
                owner_user_id="owner-1",
                limits=_limits(chars=10),
                token_counter=counter,
            )
        assert exc_info.value.code == "input_too_large"
        assert counter.calls == []


@pytest.mark.asyncio
async def test_chat_uses_bounded_text_only_projection_and_stable_order() -> None:
    store = _MessageStore(
        [
            {
                "sender": "user",
                "content": "First",
                "image_data": b"must-not-be-used",
                "images": [b"must-not-be-used"],
            },
            {"sender": "assistant", "content": "Second"},
        ]
    )
    counter = _TokenCounter(result=5)

    snapshot = await resolve_standalone_html_source(
        {"kind": "chat", "conversation_id": "conversation-1"},
        owner_user_id="owner-1",
        limits=_limits(),
        token_counter=counter,
        chacha_db=_ChaChaDb(messages=store),
    )

    assert snapshot.text == "user: First\nassistant: Second"
    assert snapshot.provenance.source_ref == "conversation-1"
    assert snapshot.provenance.reference_hmac_input is None
    assert store.projection_calls == [("conversation-1", 200, "owner-1")]
    assert counter.calls == [snapshot.text]


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_exists", [False])
async def test_missing_or_deleted_chat_is_indistinguishable_and_skips_tokenizer(
    conversation_exists: bool,
) -> None:
    store = _MessageStore([], conversation_exists=conversation_exists)
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "chat", "conversation_id": "missing"},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            chacha_db=_ChaChaDb(messages=store),
        )

    assert exc_info.value.code == "conversation_not_found"
    assert exc_info.value.status_code == 404
    assert store.projection_calls == [("missing", 200, "owner-1")]
    assert counter.calls == []


@pytest.mark.asyncio
async def test_chat_character_max_plus_one_fails_before_tokenizer() -> None:
    store = _MessageStore([{"sender": "u", "content": "x" * 20}])
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "chat", "conversation_id": "conversation-1"},
            owner_user_id="owner-1",
            limits=_limits(chars=10),
            token_counter=counter,
            chacha_db=_ChaChaDb(messages=store),
        )

    assert exc_info.value.code == "input_too_large"
    assert exc_info.value.status_code == 413
    assert counter.calls == []
    assert store.projection_calls == [("conversation-1", 10, "owner-1")]


@pytest.mark.asyncio
async def test_chat_projection_race_truncation_fails_before_tokenizer() -> None:
    store = _MessageStore(
        [{"sender": "user", "content": "short"}],
        projection_truncated=True,
    )
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "chat", "conversation_id": "conversation-1"},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            chacha_db=_ChaChaDb(messages=store),
        )

    assert exc_info.value.code == "input_too_large"
    assert exc_info.value.status_code == 413
    assert counter.calls == []


@pytest.mark.asyncio
async def test_chat_invalid_projection_fails_before_tokenizer() -> None:
    store = _MessageStore(
        [{"sender": "user", "content": "invalid"}],
        projection_invalid=True,
    )
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "chat", "conversation_id": "conversation-1"},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            chacha_db=_ChaChaDb(messages=store),
        )

    assert exc_info.value.code == "source_invalid"
    assert exc_info.value.status_code == 422
    assert counter.calls == []


@pytest.mark.asyncio
async def test_invalid_stored_unicode_fails_before_tokenizer_without_echo() -> None:
    store = _MessageStore([{"sender": "user", "content": "\ud800"}])
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "chat", "conversation_id": "conversation-1"},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            chacha_db=_ChaChaDb(messages=store),
        )

    assert exc_info.value.code == "source_invalid"
    assert str(exc_info.value) == "source_invalid"
    assert counter.calls == []


@pytest.mark.asyncio
async def test_notes_preserve_request_order_and_exact_separators() -> None:
    notes = _NoteStore(
        {
            "note-b": {"id": "note-b", "title": "Beta", "content": "Body B"},
            "note-a": {"id": "note-a", "title": "Alpha", "content": "Body A"},
        }
    )
    counter = _TokenCounter(result=7)

    snapshot = await resolve_standalone_html_source(
        {"kind": "notes", "note_ids": ["note-b", "note-a"]},
        owner_user_id="owner-1",
        limits=_limits(),
        token_counter=counter,
        chacha_db=_ChaChaDb(notes=notes),
    )

    assert snapshot.text == "# Beta\n\nBody B\n\n# Alpha\n\nBody A"
    assert [note_id for note_id, _, _ in notes.calls] == ["note-b", "note-a"]
    assert [max_chars for _, max_chars, _ in notes.calls] == [200, 184]
    assert all(owner_user_id == "owner-1" for _, _, owner_user_id in notes.calls)
    assert snapshot.provenance.source_ref is None
    assert snapshot.provenance.reference_hmac_input == b'{"note_ids":["note-b","note-a"]}'
    assert snapshot.provenance.summary == {"source_kind": "notes", "source_ref": None}


@pytest.mark.asyncio
async def test_notes_missing_or_deleted_refuses_entire_snapshot() -> None:
    notes = _NoteStore(
        {
            "present": {"id": "present", "title": "Present", "content": "text"},
            "deleted": None,
        }
    )
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "notes", "note_ids": ["present", "deleted"]},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            chacha_db=_ChaChaDb(notes=notes),
        )

    assert exc_info.value.code == "notes_not_found"
    assert exc_info.value.status_code == 404
    assert counter.calls == []


@pytest.mark.asyncio
async def test_notes_at_character_limit_still_resolve_later_empty_note() -> None:
    notes = _NoteStore(
        {
            "full": {"id": "full", "title": "T", "content": "12345"},
            "empty": {"id": "empty", "title": "", "content": ""},
        }
    )
    counter = _TokenCounter()

    snapshot = await resolve_standalone_html_source(
        {"kind": "notes", "note_ids": ["full", "empty"]},
        owner_user_id="owner-1",
        limits=_limits(chars=10),
        token_counter=counter,
        chacha_db=_ChaChaDb(notes=notes),
    )

    assert snapshot.text == "# T\n\n12345"
    assert notes.calls == [
        ("full", 10, "owner-1"),
        ("empty", 0, "owner-1"),
    ]
    assert counter.calls == [snapshot.text]


@pytest.mark.asyncio
async def test_notes_at_character_limit_still_report_later_missing_note() -> None:
    notes = _NoteStore(
        {
            "full": {"id": "full", "title": "T", "content": "12345"},
            "missing": None,
        }
    )
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "notes", "note_ids": ["full", "missing"]},
            owner_user_id="owner-1",
            limits=_limits(chars=10),
            token_counter=counter,
            chacha_db=_ChaChaDb(notes=notes),
        )

    assert exc_info.value.code == "notes_not_found"
    assert notes.calls == [
        ("full", 10, "owner-1"),
        ("missing", 0, "owner-1"),
    ]
    assert counter.calls == []


@pytest.mark.asyncio
async def test_notes_nul_projection_fails_before_tokenizer() -> None:
    notes = _NoteStore(
        {
            "invalid": {
                "id": "invalid",
                "title": "Title",
                "content": "prefix\0" + ("secret" * 1000),
            }
        }
    )
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "notes", "note_ids": ["invalid"]},
            owner_user_id="owner-1",
            limits=_limits(chars=20),
            token_counter=counter,
            chacha_db=_ChaChaDb(notes=notes),
        )

    assert exc_info.value.code == "source_invalid"
    assert counter.calls == []


@pytest.mark.asyncio
async def test_notes_missing_validation_marker_fails_closed() -> None:
    class _MarkerlessNoteStore:
        @staticmethod
        def get_source_note_projection(*_args, **_kwargs):
            return {"id": "note-1", "source_text": "apparently safe"}

    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "notes", "note_ids": ["note-1"]},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            chacha_db=SimpleNamespace(note_store=_MarkerlessNoteStore()),
        )

    assert exc_info.value.code == "source_invalid"
    assert counter.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "note_ids",
    [
        [],
        ["duplicate", "duplicate"],
        ["x"] * 101,
        ["\N{PILE OF POO}" * 65],
        ["   "],
    ],
)
async def test_note_identifier_bounds_fail_before_repository_or_tokenizer(
    note_ids: list[str],
) -> None:
    notes = _NoteStore({})
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "notes", "note_ids": note_ids},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            chacha_db=_ChaChaDb(notes=notes),
        )

    assert exc_info.value.status_code == 422
    assert notes.calls == []
    assert counter.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_id", ["", "   ", "\N{PILE OF POO}" * 65, "\ud800"])
async def test_chat_identifier_bounds_fail_before_repository_or_tokenizer(
    conversation_id: str,
) -> None:
    store = _MessageStore([])
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "chat", "conversation_id": conversation_id},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            chacha_db=_ChaChaDb(messages=store),
        )

    assert exc_info.value.status_code == 422
    assert store.projection_calls == []
    assert counter.calls == []


@pytest.mark.asyncio
async def test_media_uses_single_preformatted_bounded_projection() -> None:
    rows = [
        {"id": 7, "source_text": " transcript "},
        {"id": 7, "source_text": " document "},
        {"id": 7, "source_text": " media "},
    ]

    resolved: list[str] = []
    for row in rows:
        snapshot = await resolve_standalone_html_source(
            {"kind": "media", "media_id": 7},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=_TokenCounter(),
            media_db=_MediaDb(row),
        )
        resolved.append(snapshot.text)
        assert snapshot.provenance.source_ref == "7"

    assert resolved == ["transcript", "document", "media"]


@pytest.mark.asyncio
async def test_media_projection_is_bounded_before_tokenization() -> None:
    media_db = _MediaDb(
        {
            "id": 7,
            "source_text": "short transcript",
        }
    )

    snapshot = await resolve_standalone_html_source(
        {"kind": "media", "media_id": 7},
        owner_user_id="owner-1",
        limits=_limits(chars=100),
        token_counter=_TokenCounter(),
        media_db=media_db,
    )

    assert snapshot.text == "short transcript"
    assert media_db.calls == [(7, 100, "owner-1")]


@pytest.mark.asyncio
async def test_media_whitespace_overflow_sentinel_fails_before_tokenization() -> None:
    media_db = _MediaDb(
        {
            "id": 7,
            "source_text": ("x" * 10) + " hidden suffix",
        }
    )
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "media", "media_id": 7},
            owner_user_id="owner-1",
            limits=_limits(chars=10),
            token_counter=counter,
            media_db=media_db,
        )

    assert exc_info.value.code == "input_too_large"
    assert counter.calls == []
    assert media_db.calls == [(7, 10, "owner-1")]


@pytest.mark.asyncio
async def test_media_without_text_returns_not_found_before_tokenization() -> None:
    media_db = _MediaDb(
        {
            "id": 7,
            "source_text": None,
            "source_invalid": False,
        }
    )
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "media", "media_id": 7},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            media_db=media_db,
        )

    assert exc_info.value.code == "media_content_not_found"
    assert exc_info.value.status_code == 404
    assert counter.calls == []


@pytest.mark.asyncio
async def test_media_nul_projection_fails_before_tokenizer() -> None:
    media_db = _MediaDb(
        {
            "id": 7,
            "source_text": "prefix",
            "source_invalid": True,
        }
    )
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "media", "media_id": 7},
            owner_user_id="owner-1",
            limits=_limits(chars=20),
            token_counter=counter,
            media_db=media_db,
        )

    assert exc_info.value.code == "source_invalid"
    assert counter.calls == []


@pytest.mark.asyncio
async def test_media_missing_validation_marker_fails_closed() -> None:
    class _MarkerlessMediaDb:
        db_path_str = "/owner/media.db"

        @staticmethod
        def get_media_source_projection(*_args, **_kwargs):
            return {"id": 7, "source_text": "apparently safe"}

    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "media", "media_id": 7},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            media_db=_MarkerlessMediaDb(),
        )

    assert exc_info.value.code == "source_invalid"
    assert counter.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("media_id", [0, -1, 2**63, True, "7"])
async def test_media_id_bounds_fail_before_repository_or_tokenizer(media_id: Any) -> None:
    media_db = _MediaDb(None)
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "media", "media_id": media_id},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            media_db=media_db,
        )

    assert exc_info.value.status_code == 422
    assert media_db.calls == []
    assert counter.calls == []


@pytest.mark.asyncio
async def test_token_max_plus_one_refuses_snapshot_after_character_check() -> None:
    counter = _TokenCounter(result=11)

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "prompt", "prompt": "in bounds"},
            owner_user_id="owner-1",
            limits=_limits(chars=20, tokens=10),
            token_counter=counter,
        )

    assert exc_info.value.code == "input_too_large"
    assert exc_info.value.status_code == 413
    assert counter.calls == ["in bounds"]


@pytest.mark.asyncio
async def test_owner_bound_dependencies_do_not_cross_read_same_identifier() -> None:
    owner_a = _ChaChaDb(notes=_NoteStore({"same": {"id": "same", "title": "A", "content": "owner A"}}))
    owner_b = _ChaChaDb(notes=_NoteStore({"same": {"id": "same", "title": "B", "content": "owner B"}}))

    first = await resolve_standalone_html_source(
        {"kind": "notes", "note_ids": ["same"]},
        owner_user_id="owner-a",
        limits=_limits(),
        token_counter=_TokenCounter(),
        chacha_db=owner_a,
    )
    second = await resolve_standalone_html_source(
        {"kind": "notes", "note_ids": ["same"]},
        owner_user_id="owner-b",
        limits=_limits(),
        token_counter=_TokenCounter(),
        chacha_db=owner_b,
    )

    assert first.text == "# A\n\nowner A"
    assert second.text == "# B\n\nowner B"


@pytest.mark.asyncio
async def test_rag_formats_only_documents_and_never_generated_answer() -> None:
    rag_result = RAGResult(
        query="bounded query",
        documents=[
            Document(
                id="doc-1",
                content="First body",
                source=DataSource.MEDIA_DB,
                metadata={"title": "First"},
                score=0.9,
            ),
            Document(
                id="doc-2",
                content="Second body",
                source=DataSource.NOTES,
                metadata={},
                score=0.8,
            ),
        ],
        generated_answer="must never be used",
    )
    retriever = AsyncMock(return_value=rag_result)

    snapshot = await resolve_standalone_html_source(
        {"kind": "rag", "query": "bounded query", "top_k": 2},
        owner_user_id="owner-1",
        limits=_limits(),
        token_counter=_TokenCounter(),
        chacha_db=_ChaChaDb(),
        media_db=_MediaDb(None),
        rag_retriever=retriever,
    )

    assert snapshot.text == "# First\n\nFirst body\n\n# doc-2\n\nSecond body"
    assert "must never be used" not in snapshot.text
    assert snapshot.provenance.reference_hmac_input == (b'{"query":"bounded query","top_k":2}')
    retriever.assert_awaited_once_with(
        query="bounded query",
        owner_user_id="owner-1",
        top_k=2,
        max_source_chars=200,
        media_db=ANY,
        chacha_db=ANY,
    )


@pytest.mark.asyncio
async def test_rag_preserves_preformatted_projection_without_duplicate_title() -> None:
    retriever = AsyncMock(
        return_value=RAGResult(
            query="query",
            documents=[
                Document(
                    id="note-1",
                    content="# Note title\n\nNote body",
                    source=DataSource.NOTES,
                    metadata={
                        "title": "must not be repeated",
                        "_standalone_source_preformatted": True,
                    },
                    score=0.8,
                ),
                Document(
                    id="chat-1",
                    content="user: question\nassistant: answer",
                    source=DataSource.CHAT_HISTORY,
                    metadata={"_standalone_source_preformatted": True},
                    score=0.7,
                ),
            ],
        )
    )

    snapshot = await resolve_standalone_html_source(
        {"kind": "rag", "query": "query", "top_k": 2},
        owner_user_id="owner-1",
        limits=_limits(),
        token_counter=_TokenCounter(),
        chacha_db=_ChaChaDb(),
        media_db=_MediaDb(None),
        rag_retriever=retriever,
    )

    assert snapshot.text == ("# Note title\n\nNote body\n\nuser: question\nassistant: answer")
    assert "must not be repeated" not in snapshot.text


@pytest.mark.asyncio
async def test_rag_no_documents_does_not_fall_back_to_generated_answer() -> None:
    retriever = AsyncMock(
        return_value=RAGResult(
            query="query",
            documents=[],
            generated_answer="unsafe fallback",
        )
    )
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "rag", "query": "query", "top_k": 8},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            chacha_db=_ChaChaDb(),
            media_db=_MediaDb(None),
            rag_retriever=retriever,
        )

    assert exc_info.value.code == "rag_no_results"
    assert exc_info.value.status_code == 404
    assert counter.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "top_k"),
    [
        ("", 8),
        (" " * 2, 8),
        ("q" * 20_001, 8),
        ((" " * 20_000) + "q", 8),
        ("query", 0),
        ("query", 101),
        ("query", True),
    ],
)
async def test_rag_bounds_fail_before_retrieval_or_tokenizer(
    query: str,
    top_k: Any,
) -> None:
    retriever = AsyncMock()
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "rag", "query": query, "top_k": top_k},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            chacha_db=_ChaChaDb(),
            media_db=_MediaDb(None),
            rag_retriever=retriever,
        )

    assert exc_info.value.status_code == 422
    retriever.assert_not_awaited()
    assert counter.calls == []


@pytest.mark.asyncio
async def test_rag_document_aggregate_max_plus_one_fails_before_tokenizer() -> None:
    retriever = AsyncMock(
        return_value=RAGResult(
            query="query",
            documents=[
                SimpleNamespace(
                    id="doc",
                    content="x" * 11,
                    metadata={},
                )
            ],
        )
    )
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "rag", "query": "query", "top_k": 1},
            owner_user_id="owner-1",
            limits=_limits(chars=10),
            token_counter=counter,
            chacha_db=_ChaChaDb(),
            media_db=_MediaDb(None),
            rag_retriever=retriever,
        )

    assert exc_info.value.code == "input_too_large"
    assert counter.calls == []


@pytest.mark.asyncio
async def test_rag_stored_truncation_marker_fails_before_tokenizer() -> None:
    retriever = AsyncMock(
        return_value=RAGResult(
            query="query",
            documents=[
                Document(
                    id="doc",
                    content="bounded prefix",
                    source=DataSource.MEDIA_DB,
                    metadata={"_standalone_source_projection_truncated": True},
                    score=0.5,
                )
            ],
        )
    )
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "rag", "query": "query", "top_k": 1},
            owner_user_id="owner-1",
            limits=_limits(chars=100),
            token_counter=counter,
            chacha_db=_ChaChaDb(),
            media_db=_MediaDb(None),
            rag_retriever=retriever,
        )

    assert exc_info.value.code == "input_too_large"
    assert exc_info.value.status_code == 413
    assert counter.calls == []


@pytest.mark.asyncio
async def test_rag_dependency_failure_is_fixed_and_redacted_before_tokenizer() -> None:
    retriever = AsyncMock(side_effect=RuntimeError("secret backend detail"))
    counter = _TokenCounter()

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "rag", "query": "query", "top_k": 1},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            chacha_db=_ChaChaDb(),
            media_db=_MediaDb(None),
            rag_retriever=retriever,
        )

    assert exc_info.value.code == "source_dependency_unavailable"
    assert exc_info.value.status_code == 503
    assert str(exc_info.value) == "source_dependency_unavailable"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert counter.calls == []


@pytest.mark.asyncio
async def test_default_rag_retriever_load_failure_is_fixed_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "secret import detail"
    counter = _TokenCounter()

    def fail_to_load():
        raise RuntimeError(secret)

    monkeypatch.setattr(sources_module, "_default_rag_retriever", fail_to_load)

    with pytest.raises(StandaloneHtmlSourceError) as exc_info:
        await resolve_standalone_html_source(
            {"kind": "rag", "query": "query", "top_k": 1},
            owner_user_id="owner-1",
            limits=_limits(),
            token_counter=counter,
            chacha_db=_ChaChaDb(),
            media_db=_MediaDb(None),
        )

    assert exc_info.value.code == "source_dependency_unavailable"
    assert exc_info.value.status_code == 503
    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert counter.calls == []


def test_limits_fixture_can_be_lowered_without_mutation() -> None:
    original = _limits()
    lowered = replace(original, max_source_chars=10)

    assert original.max_source_chars == 200
    assert lowered.max_source_chars == 10
