"""Selected-source retrieval helpers for Explainer generation jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from tldw_Server_API.app.core.Explainer.models import ExplainerSession


class ExplainerSourceAccessError(ValueError):
    """Raised when retrieved context is not from the session's selected sources."""


@dataclass(frozen=True)
class ExplainerSourceExcerpt:
    source_id: str
    source_type: str
    title: str
    excerpt: str
    location_label: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    url: str | None = None
    snapshot_hash: str | None = None
    metadata: dict[str, Any] | None = None

    def to_citation_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            "excerpt": self.excerpt,
            "location_label": self.location_label,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "url": self.url,
            "snapshot_hash": self.snapshot_hash,
        }


@dataclass(frozen=True)
class ExplainerSourceContext:
    excerpts: list[ExplainerSourceExcerpt | dict[str, Any]] = field(default_factory=list)
    insufficient: bool = False
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_excerpts(self) -> list[ExplainerSourceExcerpt]:
        return [_coerce_excerpt(excerpt) for excerpt in self.excerpts]


class ExplainerRetriever(Protocol):
    def __call__(self, *, session: ExplainerSession, owner_user_id: str) -> ExplainerSourceContext | dict[str, Any]:
        """Return selected-source context for an owned Explainer session."""


def retrieve_selected_source_context(
    *,
    session: ExplainerSession,
    owner_user_id: str,
) -> ExplainerSourceContext:
    """Return safe source context derived from selected-source snapshots.

    The default implementation intentionally does not reach into provider-heavy
    RAG/media modules at import time. Workers can inject a richer retriever, and
    this function still validates that all returned excerpts belong to the owned
    session's selected sources.
    """

    _validate_session_owner(session=session, owner_user_id=owner_user_id)
    excerpts: list[ExplainerSourceExcerpt] = []
    for source in session.selected_sources:
        metadata = source.metadata or {}
        excerpt = str(metadata.get("excerpt") or metadata.get("text") or "").strip()
        if not excerpt:
            continue
        excerpts.append(
            ExplainerSourceExcerpt(
                source_id=source.source_id,
                source_type=source.source_type,
                title=source.title,
                excerpt=excerpt,
                location_label=metadata.get("location_label"),
                start_offset=_coerce_optional_int(metadata.get("start_offset")),
                end_offset=_coerce_optional_int(metadata.get("end_offset")),
                url=metadata.get("url"),
                snapshot_hash=metadata.get("snapshot_hash"),
                metadata=metadata,
            )
        )
    context = ExplainerSourceContext(
        excerpts=excerpts,
        insufficient=bool(session.selected_sources and not excerpts),
        retrieval_metadata={
            "selectedSourceCount": len(session.selected_sources),
            "excerptCount": len(excerpts),
        },
    )
    return validate_source_context_ownership(
        session=session,
        owner_user_id=owner_user_id,
        source_context=context,
    )


def coerce_source_context(value: ExplainerSourceContext | dict[str, Any] | None) -> ExplainerSourceContext:
    """Normalize injected retriever output into an ExplainerSourceContext."""

    if value is None:
        return ExplainerSourceContext(insufficient=True)
    if isinstance(value, ExplainerSourceContext):
        return value
    if not isinstance(value, dict):
        raise TypeError("retriever output must be an ExplainerSourceContext or dict")
    excerpts = value.get("excerpts") or []
    if not isinstance(excerpts, list):
        raise TypeError("retriever excerpts must be a list")
    metadata = value.get("retrieval_metadata") or value.get("retrievalMetadata") or {}
    return ExplainerSourceContext(
        excerpts=excerpts,
        insufficient=bool(value.get("insufficient")),
        retrieval_metadata=metadata if isinstance(metadata, dict) else {},
    )


def validate_source_context_ownership(
    *,
    session: ExplainerSession,
    owner_user_id: str,
    source_context: ExplainerSourceContext,
) -> ExplainerSourceContext:
    """Ensure retrieved excerpts are scoped to the owned selected sources."""

    _validate_session_owner(session=session, owner_user_id=owner_user_id)
    selected = {
        (source.source_type, source.source_id)
        for source in session.selected_sources
    }
    normalized = source_context.normalized_excerpts()
    for excerpt in normalized:
        if (excerpt.source_type, excerpt.source_id) not in selected:
            raise ExplainerSourceAccessError("retrieved source is not selected for this session")
    return ExplainerSourceContext(
        excerpts=normalized,
        insufficient=source_context.insufficient,
        retrieval_metadata=dict(source_context.retrieval_metadata or {}),
    )


def _validate_session_owner(*, session: ExplainerSession, owner_user_id: str) -> None:
    if str(session.owner_user_id) != str(owner_user_id):
        raise ExplainerSourceAccessError("session does not belong to owner")


def _coerce_excerpt(value: ExplainerSourceExcerpt | dict[str, Any]) -> ExplainerSourceExcerpt:
    if isinstance(value, ExplainerSourceExcerpt):
        return value
    if not isinstance(value, dict):
        raise TypeError("source excerpt must be a dict or ExplainerSourceExcerpt")
    return ExplainerSourceExcerpt(
        source_id=_required_text(value.get("source_id") or value.get("sourceId"), "source_id"),
        source_type=_required_text(value.get("source_type") or value.get("sourceType"), "source_type"),
        title=_required_text(value.get("title"), "title"),
        excerpt=_required_text(value.get("excerpt"), "excerpt"),
        location_label=value.get("location_label") or value.get("locationLabel"),
        start_offset=_coerce_optional_int(_get_alias(value, "start_offset", "startOffset")),
        end_offset=_coerce_optional_int(_get_alias(value, "end_offset", "endOffset")),
        url=value.get("url"),
        snapshot_hash=value.get("snapshot_hash") or value.get("snapshotHash"),
        metadata=value.get("metadata") if isinstance(value.get("metadata"), dict) else None,
    )


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _get_alias(value: dict[str, Any], snake_key: str, camel_key: str) -> Any:
    return value[snake_key] if snake_key in value else value.get(camel_key)


__all__ = [
    "ExplainerRetriever",
    "ExplainerSourceAccessError",
    "ExplainerSourceContext",
    "ExplainerSourceExcerpt",
    "coerce_source_context",
    "retrieve_selected_source_context",
    "validate_source_context_ownership",
]
