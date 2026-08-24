"""Bounded, owner-scoped source snapshots for standalone HTML generation."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_media_source_projection,
)
from tldw_Server_API.app.core.RAG.rag_service.result_model import RAGResult
from tldw_Server_API.app.core.Slides.standalone_html_config import (
    StandaloneHtmlInputLimits,
)
from tldw_Server_API.app.core.Utils.tokenizer import count_tokens

StandaloneHtmlSourceKind = Literal["prompt", "chat", "media", "notes", "rag"]


class StandaloneHtmlSourceError(RuntimeError):
    """Fixed source-resolution failure that never carries source material."""

    __slots__ = ("code", "status_code")

    def __init__(self, code: str, *, status_code: int) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class StandaloneHtmlSourceProvenance:
    """Source-only provenance inputs completed with HMAC/provider data later."""

    source_kind: StandaloneHtmlSourceKind
    source_ref: str | None
    reference_hmac_input: bytes | None = field(default=None, repr=False)

    @property
    def summary(self) -> dict[str, str | None]:
        """Return the exact safe source projection available at this stage."""

        return {
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True, slots=True)
class StandaloneHtmlSourceSnapshot:
    """One immutable, bounded source snapshot created before queue admission."""

    source_kind: StandaloneHtmlSourceKind
    text: str = field(repr=False)
    char_count: int
    byte_count: int
    token_count: int
    provenance: StandaloneHtmlSourceProvenance


class StandaloneHtmlRagRetriever(Protocol):
    """Closed retrieval-only dependency accepted by the source resolver."""

    def __call__(
        self,
        *,
        query: str,
        owner_user_id: str,
        top_k: int,
        max_source_chars: int,
        media_db: Any,
        chacha_db: Any,
    ) -> Awaitable[RAGResult]: ...


class _BoundedParts:
    """Accumulate at most the configured source size without partial output."""

    __slots__ = ("_length", "_max_chars", "_parts")

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max_chars
        self._length = 0
        self._parts: list[str] = []

    @property
    def remaining(self) -> int:
        return self._max_chars - self._length

    @property
    def has_value(self) -> bool:
        return bool(self._parts)

    def append(self, value: str) -> None:
        if not value:
            return
        if len(value) > self.remaining:
            raise StandaloneHtmlSourceError("input_too_large", status_code=413)
        self._parts.append(value)
        self._length += len(value)

    def append_separated(self, value: str, *, separator: str) -> None:
        if not value:
            return
        if self._parts:
            self.append(separator)
        self.append(value)

    def build(self) -> str:
        return "".join(self._parts)


def _require_scalar_text(value: Any) -> str:
    if not isinstance(value, str):
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    encoding_failed = False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        encoding_failed = True
    if encoding_failed:
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    return value


def _require_identifier(value: Any, *, max_bytes: int, trim: bool) -> str:
    text = _require_scalar_text(value)
    if not text.strip():
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    if len(text.encode("utf-8")) > max_bytes:
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    return text.strip() if trim else text


def _canonical_reference(value: Mapping[str, Any]) -> bytes:
    encoded: bytes | None = None
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError):
        pass
    if encoded is None:
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    return encoded


def _require_dependency(value: Any, attribute: str | None = None) -> Any:
    candidate = getattr(value, attribute, None) if attribute else value
    if candidate is None:
        raise StandaloneHtmlSourceError(
            "source_dependency_unavailable",
            status_code=503,
        )
    return candidate


def _source_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _document_field(document: Any, name: str) -> Any:
    if isinstance(document, Mapping):
        return document.get(name)
    return getattr(document, name, None)


def _rag_document_title(document: Any) -> str:
    metadata = _document_field(document, "metadata")
    if isinstance(metadata, Mapping):
        for name in ("title", "source_title"):
            value = metadata.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    document_id = _document_field(document, "id")
    if isinstance(document_id, (str, int)) and not isinstance(document_id, bool):
        text = str(document_id).strip()
        if text:
            return text
    return "source"


def _rag_document_is_truncated(document: Any) -> bool:
    metadata = _document_field(document, "metadata")
    return isinstance(metadata, Mapping) and bool(metadata.get("_standalone_source_projection_truncated"))


def _rag_document_is_preformatted(document: Any) -> bool:
    metadata = _document_field(document, "metadata")
    return isinstance(metadata, Mapping) and bool(metadata.get("_standalone_source_preformatted"))


def _default_rag_retriever() -> StandaloneHtmlRagRetriever:
    from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
        retrieve_slides_source_documents_v1,
    )

    return retrieve_slides_source_documents_v1


async def _resolve_chat(
    source: Mapping[str, Any],
    *,
    owner_user_id: str,
    limits: StandaloneHtmlInputLimits,
    chacha_db: Any,
) -> tuple[str, StandaloneHtmlSourceProvenance]:
    conversation_id = _require_identifier(
        source.get("conversation_id"),
        max_bytes=limits.max_source_identifier_bytes,
        trim=True,
    )
    store = _require_dependency(chacha_db, "message_store")
    project = getattr(store, "get_source_message_projection", None)
    if not callable(project):
        raise StandaloneHtmlSourceError(
            "source_dependency_unavailable",
            status_code=503,
        )
    projection_failed = False
    try:
        projection = project(
            conversation_id,
            max_chars=limits.max_source_chars,
            owner_user_id=owner_user_id,
        )
    except Exception:  # noqa: BLE001 - repository failures cross a redacted boundary
        projection_failed = True
        projection = None
    if projection_failed:
        raise StandaloneHtmlSourceError(
            "source_dependency_unavailable",
            status_code=503,
        )
    if not isinstance(projection, Mapping):
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    conversation_exists = projection.get("conversation_exists")
    if not isinstance(conversation_exists, bool):
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    if not conversation_exists:
        raise StandaloneHtmlSourceError("conversation_not_found", status_code=404)
    invalid = projection.get("invalid")
    if not isinstance(invalid, bool):
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    if invalid:
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    truncated = projection.get("truncated")
    if not isinstance(truncated, bool):
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    if truncated:
        raise StandaloneHtmlSourceError("input_too_large", status_code=413)
    rows = projection.get("rows")
    if not isinstance(rows, list):
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)

    parts = _BoundedParts(limits.max_source_chars)
    for row in rows:
        if not isinstance(row, Mapping):
            raise StandaloneHtmlSourceError("source_invalid", status_code=422)
        source_text = row.get("source_text")
        if source_text in (None, ""):
            continue
        if not isinstance(source_text, str):
            raise StandaloneHtmlSourceError("source_invalid", status_code=422)
        parts.append_separated(source_text, separator="\n")

    text = parts.build().strip()
    if not text:
        raise StandaloneHtmlSourceError("conversation_empty", status_code=404)
    return text, StandaloneHtmlSourceProvenance(
        source_kind="chat",
        source_ref=conversation_id,
    )


async def _resolve_notes(
    source: Mapping[str, Any],
    *,
    owner_user_id: str,
    limits: StandaloneHtmlInputLimits,
    chacha_db: Any,
) -> tuple[str, StandaloneHtmlSourceProvenance]:
    raw_note_ids = source.get("note_ids")
    if not isinstance(raw_note_ids, list):
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    if not 1 <= len(raw_note_ids) <= limits.max_note_ids:
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    note_ids = [
        _require_identifier(
            note_id,
            max_bytes=limits.max_source_identifier_bytes,
            trim=False,
        )
        for note_id in raw_note_ids
    ]
    if len(set(note_ids)) != len(note_ids):
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)

    store = _require_dependency(chacha_db, "note_store")
    project = getattr(store, "get_source_note_projection", None)
    if not callable(project):
        raise StandaloneHtmlSourceError(
            "source_dependency_unavailable",
            status_code=503,
        )

    parts = _BoundedParts(limits.max_source_chars)
    for note_id in note_ids:
        separator_chars = 2 if parts.has_value else 0
        projection_chars = max(0, parts.remaining - separator_chars)
        projection_failed = False
        try:
            row = project(
                note_id,
                max_chars=projection_chars,
                owner_user_id=owner_user_id,
            )
        except Exception:  # noqa: BLE001 - repository failures cross a redacted boundary
            projection_failed = True
            row = None
        if projection_failed:
            raise StandaloneHtmlSourceError(
                "source_dependency_unavailable",
                status_code=503,
            )
        if row is None:
            raise StandaloneHtmlSourceError("notes_not_found", status_code=404)
        if not isinstance(row, Mapping):
            raise StandaloneHtmlSourceError("source_invalid", status_code=422)
        invalid = row.get("source_invalid")
        if not isinstance(invalid, bool):
            raise StandaloneHtmlSourceError("source_invalid", status_code=422)
        if invalid:
            raise StandaloneHtmlSourceError("source_invalid", status_code=422)
        source_text = row.get("source_text")
        if not isinstance(source_text, str):
            raise StandaloneHtmlSourceError("source_invalid", status_code=422)
        parts.append_separated(source_text, separator="\n\n")

    text = parts.build().strip()
    if not text:
        raise StandaloneHtmlSourceError("notes_empty", status_code=404)
    return text, StandaloneHtmlSourceProvenance(
        source_kind="notes",
        source_ref=None,
        reference_hmac_input=_canonical_reference({"note_ids": note_ids}),
    )


async def _resolve_media(
    source: Mapping[str, Any],
    *,
    owner_user_id: str,
    limits: StandaloneHtmlInputLimits,
    media_db: Any,
) -> tuple[str, StandaloneHtmlSourceProvenance]:
    media_id = source.get("media_id")
    if isinstance(media_id, bool) or not isinstance(media_id, int) or not 1 <= media_id <= 9_223_372_036_854_775_807:
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    _require_dependency(media_db)
    projection_failed = False
    try:
        row = get_media_source_projection(
            media_db,
            media_id,
            max_chars=limits.max_source_chars,
            owner_user_id=owner_user_id,
        )
    except Exception:  # noqa: BLE001 - repository failures cross a redacted boundary
        projection_failed = True
        row = None
    if projection_failed:
        raise StandaloneHtmlSourceError(
            "source_dependency_unavailable",
            status_code=503,
        )
    if row is None:
        raise StandaloneHtmlSourceError("media_not_found", status_code=404)
    if not isinstance(row, Mapping):
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    invalid = row.get("source_invalid")
    if not isinstance(invalid, bool):
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    if invalid:
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)

    raw_text = row.get("source_text")
    if raw_text is not None and not isinstance(raw_text, str):
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    if isinstance(raw_text, str) and len(raw_text) > limits.max_source_chars:
        raise StandaloneHtmlSourceError("input_too_large", status_code=413)
    text = _source_text(raw_text)
    if text is None:
        raise StandaloneHtmlSourceError("media_content_not_found", status_code=404)
    return text, StandaloneHtmlSourceProvenance(
        source_kind="media",
        source_ref=str(media_id),
    )


async def _resolve_rag(
    source: Mapping[str, Any],
    *,
    owner_user_id: str,
    limits: StandaloneHtmlInputLimits,
    media_db: Any,
    chacha_db: Any,
    rag_retriever: StandaloneHtmlRagRetriever | None,
) -> tuple[str, StandaloneHtmlSourceProvenance]:
    raw_query = _require_scalar_text(source.get("query"))
    if not raw_query.strip() or len(raw_query) > limits.max_rag_query_chars:
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    query = raw_query.strip()
    top_k = source.get("top_k", 8)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= limits.max_rag_top_k:
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)

    _require_dependency(media_db)
    _require_dependency(chacha_db)
    retrieval_failed = False
    try:
        retrieve = rag_retriever or _default_rag_retriever()
        result = await retrieve(
            query=query,
            owner_user_id=owner_user_id,
            top_k=top_k,
            max_source_chars=limits.max_source_chars,
            media_db=media_db,
            chacha_db=chacha_db,
        )
    except Exception:  # noqa: BLE001 - retrieval failures cross a redacted boundary
        retrieval_failed = True
        result = None
    if retrieval_failed:
        raise StandaloneHtmlSourceError(
            "source_dependency_unavailable",
            status_code=503,
        )
    documents = getattr(result, "documents", None)
    if not isinstance(documents, (list, tuple)):
        raise StandaloneHtmlSourceError("rag_no_results", status_code=404)

    parts = _BoundedParts(limits.max_source_chars)
    document_count = 0
    for document in documents[:top_k]:
        if _rag_document_is_truncated(document):
            raise StandaloneHtmlSourceError("input_too_large", status_code=413)
        content = _document_field(document, "content")
        if not isinstance(content, str) or not content.strip():
            continue
        if _rag_document_is_preformatted(document):
            parts.append_separated(content, separator="\n\n")
        else:
            title = _rag_document_title(document)
            parts.append_separated(f"# {title}", separator="\n\n")
            parts.append_separated(content, separator="\n\n")
        document_count += 1
    text = parts.build().strip()
    if document_count == 0 or not text:
        raise StandaloneHtmlSourceError("rag_no_results", status_code=404)
    return text, StandaloneHtmlSourceProvenance(
        source_kind="rag",
        source_ref=None,
        reference_hmac_input=_canonical_reference({"query": query, "top_k": top_k}),
    )


def _finalize_snapshot(
    source_kind: StandaloneHtmlSourceKind,
    text: str,
    provenance: StandaloneHtmlSourceProvenance,
    *,
    limits: StandaloneHtmlInputLimits,
    token_counter: Callable[[str], int],
) -> StandaloneHtmlSourceSnapshot:
    if not text.strip():
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    if len(text) > limits.max_source_chars:
        raise StandaloneHtmlSourceError("input_too_large", status_code=413)
    encoding_failed = False
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError:
        encoding_failed = True
        encoded = b""
    if encoding_failed:
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    tokenization_failed = False
    try:
        token_count = token_counter(text)
    except Exception:  # noqa: BLE001 - tokenizer plugins are isolated behind a fixed error
        tokenization_failed = True
        token_count = -1
    if tokenization_failed:
        raise StandaloneHtmlSourceError(
            "source_tokenizer_unavailable",
            status_code=503,
        )
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise StandaloneHtmlSourceError(
            "source_tokenizer_unavailable",
            status_code=503,
        )
    if token_count > limits.max_source_tokens:
        raise StandaloneHtmlSourceError("input_too_large", status_code=413)
    return StandaloneHtmlSourceSnapshot(
        source_kind=source_kind,
        text=text,
        char_count=len(text),
        byte_count=len(encoded),
        token_count=token_count,
        provenance=provenance,
    )


async def resolve_standalone_html_source(
    source: Mapping[str, Any],
    *,
    owner_user_id: str,
    limits: StandaloneHtmlInputLimits,
    token_counter: Callable[[str], int] = count_tokens,
    chacha_db: Any | None = None,
    media_db: Any | None = None,
    rag_retriever: StandaloneHtmlRagRetriever | None = None,
) -> StandaloneHtmlSourceSnapshot:
    """Resolve one closed source variant into an immutable bounded snapshot."""

    if not isinstance(source, Mapping):
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    owner = _require_scalar_text(owner_user_id).strip()
    if not owner:
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)
    kind = source.get("kind")
    if kind == "prompt":
        text = _require_scalar_text(source.get("prompt"))
        if not text.strip():
            raise StandaloneHtmlSourceError("source_invalid", status_code=422)
        if len(text) > limits.max_source_chars:
            raise StandaloneHtmlSourceError("input_too_large", status_code=413)
        provenance = StandaloneHtmlSourceProvenance(
            source_kind="prompt",
            source_ref=None,
        )
    elif kind == "chat":
        text, provenance = await _resolve_chat(
            source,
            owner_user_id=owner,
            limits=limits,
            chacha_db=chacha_db,
        )
    elif kind == "media":
        text, provenance = await _resolve_media(
            source,
            owner_user_id=owner,
            limits=limits,
            media_db=media_db,
        )
    elif kind == "notes":
        text, provenance = await _resolve_notes(
            source,
            owner_user_id=owner,
            limits=limits,
            chacha_db=chacha_db,
        )
    elif kind == "rag":
        text, provenance = await _resolve_rag(
            source,
            owner_user_id=owner,
            limits=limits,
            media_db=media_db,
            chacha_db=chacha_db,
            rag_retriever=rag_retriever,
        )
    else:
        raise StandaloneHtmlSourceError("source_invalid", status_code=422)

    return _finalize_snapshot(
        provenance.source_kind,
        text,
        provenance,
        limits=limits,
        token_counter=token_counter,
    )


__all__ = [
    "StandaloneHtmlRagRetriever",
    "StandaloneHtmlSourceError",
    "StandaloneHtmlSourceProvenance",
    "StandaloneHtmlSourceSnapshot",
    "resolve_standalone_html_source",
]
