"""Version-bound canonical content and deterministic chunks for Notes semantics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

from .semantic_settings import DEFAULT_SEMANTIC_INDEX_SETTINGS, SemanticIndexSettings
from .suggestion_content import (
    canonicalize_note_content,
    stored_text_utf8_bytes,
)

SEMANTIC_NORMALIZATION_VERSION = "notes-semantic-normalization-v1"
SEMANTIC_CHUNKER_VERSION = "notes-semantic-chunker-v1"

SemanticField = Literal["title", "content"]


class SemanticContentError(ValueError):
    """A stable, content-free semantic chunk admission failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class SemanticProviderInput:
    """Ephemeral text sent to an embedding provider for one evidence chunk."""

    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class SemanticChunkInput:
    """Opaque chunk identity, canonical evidence coordinates, and ephemeral input."""

    vector_id: str
    chunk_fingerprint: str
    content_fingerprint: str
    generation_id: str
    note_id: str
    content_version: int
    ordinal: int
    field: SemanticField
    start_offset: int
    end_offset: int
    normalization_version: str
    chunker_version: str
    provider_input: SemanticProviderInput = field(repr=False)


def _validated_version(value: int) -> int:
    if type(value) is not int or value <= 0:
        raise SemanticContentError("invalid_content_version")
    return value


def _hash_parts(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256()
    for value in (prefix, *parts):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def semantic_content_fingerprint(
    title: str | None,
    content: str | None,
    content_version: int,
    *,
    normalization_version: str = SEMANTIC_NORMALIZATION_VERSION,
) -> str:
    """Bind canonical title/body, normalization, and Note content version."""

    version = _validated_version(content_version)
    canonical = canonicalize_note_content(title, content)
    return _hash_parts(
        "notes-semantic-content-fingerprint-v1",
        normalization_version,
        canonical.title,
        canonical.content,
        version,
    )


def _provider_text(field: SemanticField, title: str, source_slice: str) -> str:
    if field == "content" and title.strip():
        return f"{title}\n\n{source_slice}"
    return source_slice


def _bounded_end(
    field_text: str,
    *,
    start: int,
    field: SemanticField,
    title: str,
    settings: SemanticIndexSettings,
) -> int:
    low = start + 1
    high = min(len(field_text), start + settings.max_chunk_code_points)
    accepted = start
    while low <= high:
        middle = (low + high) // 2
        provider_text = _provider_text(field, title, field_text[start:middle])
        if len(provider_text.encode("utf-8")) <= settings.max_provider_input_bytes:
            accepted = middle
            low = middle + 1
        else:
            high = middle - 1
    if accepted == start:
        raise SemanticContentError("provider_input_bytes_exceeded")
    return accepted


def _chunk_fingerprint(
    *,
    source_slice: str,
    content_fingerprint: str,
    content_version: int,
    ordinal: int,
    field: SemanticField,
    start_offset: int,
    end_offset: int,
    normalization_version: str,
    chunker_version: str,
) -> str:
    return _hash_parts(
        "notes-semantic-chunk-fingerprint-v1",
        normalization_version,
        chunker_version,
        content_fingerprint,
        content_version,
        field,
        start_offset,
        end_offset,
        ordinal,
        source_slice,
    )


def _vector_id(
    *,
    generation_id: str,
    note_id: str,
    chunk_fingerprint: str,
    content_fingerprint: str,
    content_version: int,
    ordinal: int,
    field: SemanticField,
    start_offset: int,
    end_offset: int,
    normalization_version: str,
    chunker_version: str,
) -> str:
    digest = _hash_parts(
        "notes-semantic-vector-id-v1",
        generation_id,
        note_id,
        content_fingerprint,
        chunk_fingerprint,
        content_version,
        field,
        start_offset,
        end_offset,
        ordinal,
        normalization_version,
        chunker_version,
    )
    return f"semchunk:{digest.removeprefix('sha256:')}"


def build_semantic_chunks(
    *,
    generation_id: str,
    note_id: str,
    title: str | None,
    content: str | None,
    content_version: int,
    settings: SemanticIndexSettings = DEFAULT_SEMANTIC_INDEX_SETTINGS,
    normalization_version: str = SEMANTIC_NORMALIZATION_VERSION,
    chunker_version: str = SEMANTIC_CHUNKER_VERSION,
) -> tuple[SemanticChunkInput, ...]:
    """Create complete field-relative chunks or reject the Note without truncation."""

    version = _validated_version(content_version)
    if stored_text_utf8_bytes(title, content) > settings.max_stored_note_bytes:
        raise SemanticContentError("stored_note_bytes_exceeded")
    canonical = canonicalize_note_content(title, content)
    if max(len(canonical.title), len(canonical.content)) > settings.max_canonical_field_code_points:
        raise SemanticContentError("canonical_field_code_points_exceeded")
    if not canonical.title.strip() and not canonical.content.strip():
        raise SemanticContentError("note_empty")

    field: SemanticField = "content" if canonical.content.strip() else "title"
    field_text = canonical.content if field == "content" else canonical.title
    fingerprint = semantic_content_fingerprint(
        canonical.title,
        canonical.content,
        version,
        normalization_version=normalization_version,
    )
    chunks: list[SemanticChunkInput] = []
    start = 0
    while start < len(field_text):
        end = _bounded_end(
            field_text,
            start=start,
            field=field,
            title=canonical.title,
            settings=settings,
        )
        ordinal = len(chunks)
        source_slice = field_text[start:end]
        chunk_fingerprint = _chunk_fingerprint(
            source_slice=source_slice,
            content_fingerprint=fingerprint,
            content_version=version,
            ordinal=ordinal,
            field=field,
            start_offset=start,
            end_offset=end,
            normalization_version=normalization_version,
            chunker_version=chunker_version,
        )
        chunks.append(
            SemanticChunkInput(
                vector_id=_vector_id(
                    generation_id=generation_id,
                    note_id=note_id,
                    chunk_fingerprint=chunk_fingerprint,
                    content_fingerprint=fingerprint,
                    content_version=version,
                    ordinal=ordinal,
                    field=field,
                    start_offset=start,
                    end_offset=end,
                    normalization_version=normalization_version,
                    chunker_version=chunker_version,
                ),
                chunk_fingerprint=chunk_fingerprint,
                content_fingerprint=fingerprint,
                generation_id=generation_id,
                note_id=note_id,
                content_version=version,
                ordinal=ordinal,
                field=field,
                start_offset=start,
                end_offset=end,
                normalization_version=normalization_version,
                chunker_version=chunker_version,
                provider_input=SemanticProviderInput(
                    _provider_text(field, canonical.title, source_slice)
                ),
            )
        )
        if len(chunks) > settings.max_chunks_per_note:
            raise SemanticContentError("chunks_per_note_exceeded")
        start = end
    return tuple(chunks)


def reconstruct_semantic_chunk(
    chunk: SemanticChunkInput,
    *,
    title: str | None,
    content: str | None,
    content_version: int,
) -> str | None:
    """Reconstruct a chunk only when current canonical source identity still matches."""

    try:
        fingerprint = semantic_content_fingerprint(
            title,
            content,
            content_version,
            normalization_version=chunk.normalization_version,
        )
    except SemanticContentError:
        return None
    if fingerprint != chunk.content_fingerprint or content_version != chunk.content_version:
        return None
    canonical = canonicalize_note_content(title, content)
    field_text = canonical.title if chunk.field == "title" else canonical.content
    if not 0 <= chunk.start_offset < chunk.end_offset <= len(field_text):
        return None
    source_slice = field_text[chunk.start_offset : chunk.end_offset]
    expected = _chunk_fingerprint(
        source_slice=source_slice,
        content_fingerprint=fingerprint,
        content_version=content_version,
        ordinal=chunk.ordinal,
        field=chunk.field,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        normalization_version=chunk.normalization_version,
        chunker_version=chunk.chunker_version,
    )
    return source_slice if expected == chunk.chunk_fingerprint else None


__all__ = [
    "SEMANTIC_CHUNKER_VERSION",
    "SEMANTIC_NORMALIZATION_VERSION",
    "SemanticChunkInput",
    "SemanticContentError",
    "SemanticProviderInput",
    "build_semantic_chunks",
    "reconstruct_semantic_chunk",
    "semantic_content_fingerprint",
]
