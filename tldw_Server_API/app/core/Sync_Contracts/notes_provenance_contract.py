"""Bounded portable Knowledge evidence; pointers confer no access authority."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from functools import partial
from typing import Annotated, Literal
from urllib.parse import quote, unquote

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

MAX_PROVENANCE_BYTES = 1_000_000


def _client_text_length(value: str, *, limit: int) -> str:
    """Match JavaScript string.length, including surrogate pairs."""
    if len(value.encode("utf-16-le")) // 2 > limit:
        raise ValueError("Knowledge provenance text exceeds the size limit")
    return value


ShortString = Annotated[
    str, Field(strict=True, min_length=1, max_length=512), AfterValidator(partial(_client_text_length, limit=512))
]
Excerpt = Annotated[str, Field(max_length=100_000), AfterValidator(partial(_client_text_length, limit=100_000))]
UrlString = Annotated[str, Field(max_length=4096), AfterValidator(partial(_client_text_length, limit=4096))]
PositiveId = Annotated[int, Field(strict=True, gt=0, le=9_007_199_254_740_991)]
ShortList = Annotated[list[ShortString], Field(max_length=100)]
TrustState = Literal[
    "cited_answer",
    "uncited_degraded_answer",
    "no_answer_insufficient_evidence",
    "no_results",
    "failed_search",
    "unsynced_local_result",
    "unknown_trust",
]
EvidenceOrigin = Literal["local_library", "web_fallback", "mixed", "unknown_origin"]
_MARKER = re.compile(r"^<!-- tldw-knowledge:v1:([^\r\n]+) -->$", re.MULTILINE)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NotesProvenanceScope(_StrictModel):
    """Explicit retrieval scope, never arbitrary metadata or permission grants."""

    sources: ShortList = None
    include_note_ids: ShortList = None
    include_media_ids: Annotated[list[PositiveId], Field(max_length=100)] = None
    enable_web_fallback: bool = None
    collection_id: ShortString | PositiveId | None = None
    keyword_filter: ShortString | ShortList | Literal[""] | None = None


class NotesProvenanceSource(_StrictModel):
    """Retained bounded citation with nullable original pointers."""

    originalId: ShortString | PositiveId | None
    excerpt: Excerpt
    mediaId: PositiveId | None
    title: ShortString
    type: Literal["pdf", "video", "audio", "website", "text", "document"]
    sourceType: ShortString | None
    url: UrlString | None = None
    snapshotMediaId: PositiveId | None = None
    originalVersion: PositiveId | None = None
    pageNumber: PositiveId | None = None
    citationIndex: PositiveId | None = None


SourceList = Annotated[list[NotesProvenanceSource], Field(max_length=100)]


class _Evidence(_StrictModel):
    importId: ShortString
    threadId: ShortString | None
    snapshot: bool
    sources: SourceList
    trustState: TrustState | None = None
    trustReasonCodes: ShortList | None = None
    evidenceOrigin: EvidenceOrigin | None = None
    scope: NotesProvenanceScope | None = None


class _ResearchSource(_StrictModel):
    mediaId: PositiveId
    evidence: _Evidence


class _Research(_StrictModel):
    workspace_id: ShortString
    import_id: ShortString
    sources: Annotated[list[_ResearchSource], Field(max_length=100)]


class NotesProvenancePayload(_StrictModel):
    """Knowledge and reviewed-source evidence, independent of note content."""

    origin: Literal["knowledge_qa", "reviewed_sources"]
    trust_state: TrustState | None = None
    evidence_origin: EvidenceOrigin | None = None
    thread_id: ShortString | None = None
    research: _Research | None = None
    question: Excerpt | None = None
    scope: NotesProvenanceScope | None = None
    trust_reason_codes: ShortList | None = None
    sources: SourceList | None = None


def canonical_notes_provenance_json(payload: Mapping[str, object]) -> str:
    """Serialize strict canonical payload with a portable one-megabyte ceiling."""
    value = NotesProvenancePayload.model_validate(payload).model_dump(mode="json", exclude_unset=True)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if (
        len(encoded.encode("utf-8")) > MAX_PROVENANCE_BYTES
        or len(quote(encoded, safe="~()*!.'-")) > MAX_PROVENANCE_BYTES
    ):
        raise ValueError("Knowledge provenance exceeds the size limit")
    return encoded


def validate_notes_provenance_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate every nested field without coercion or arbitrary extensions."""
    return json.loads(canonical_notes_provenance_json(payload))


def notes_provenance_object_hash(payload: Mapping[str, object], deleted: bool = False) -> str:
    """Hash the bounded canonical payload and independent lifecycle state."""
    if type(deleted) is not bool:
        raise ValueError("deleted must be a boolean")
    value = {"payload": validate_notes_provenance_payload(payload), "deleted": deleted}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_notes_provenance(content: str) -> dict[str, object] | None:
    """Read the first valid bounded portable marker; malformed comments are text."""
    for match in _MARKER.finditer(content):
        if len(match[1]) > MAX_PROVENANCE_BYTES or re.search(r"%(?![0-9a-fA-F]{2})", match[1]):
            continue
        try:
            return validate_notes_provenance_payload(json.loads(unquote(match[1], errors="strict")))
        except (ValueError, TypeError, RecursionError):
            continue
    return None


def strip_notes_provenance(content: str) -> str:
    """Remove only valid provenance markers and preserve ordinary user comments."""
    removed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal removed
        if read_notes_provenance(match[0]) is None:
            return match[0]
        removed = True
        return ""

    result = _MARKER.sub(replace, content)
    return result.rstrip() if removed else result


def retain_notes_provenance(content: str, metadata: Mapping[str, object] | None = None) -> str:
    """Append explicit metadata or retain the existing validated portable marker."""
    value = validate_notes_provenance_payload(metadata) if metadata is not None else read_notes_provenance(content)
    if value is None:
        return content
    encoded = quote(canonical_notes_provenance_json(value), safe="~()*!.'-")
    if len(encoded) > MAX_PROVENANCE_BYTES:
        raise ValueError("Knowledge provenance marker exceeds the size limit")
    return f"{strip_notes_provenance(content).rstrip()}\n\n<!-- tldw-knowledge:v1:{encoded} -->"
