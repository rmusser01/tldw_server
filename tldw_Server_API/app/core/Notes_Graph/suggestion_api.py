"""Application facade for the nested Notes graph suggestion API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable

from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_models import (
    NoteGraphSuggestion,
    NoteGraphSuggestionKind,
)
from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_store import (
    NotesGraphFTSNotReadyError,
    NotesGraphSourceTooLargeError,
)

from .suggestion_capabilities import (
    SuggestionCapabilities,
    build_unavailable_suggestion_capabilities,
)
from .suggestion_content import EvidenceReference, content_fingerprint, reconstruct_evidence
from .suggestion_provider import resolve_generation_capability


class SuggestionAPIError(RuntimeError):
    """Stable sanitized transport mapping emitted by the application facade."""

    def __init__(self, status_code: int, code: str) -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class SuggestionEvidenceExcerpt:
    side: str
    note_id: str
    field: str
    start_offset: int
    end_offset: int
    text: str


@dataclass(frozen=True, slots=True)
class SuggestionReviewItem:
    suggestion: NoteGraphSuggestion
    evidence: tuple[SuggestionEvidenceExcerpt, ...]
    target_title: str | None


@dataclass(frozen=True, slots=True)
class SuggestionReviewPage:
    items: tuple[SuggestionReviewItem, ...]
    next_cursor: str | None
    current_source_fingerprint: str
    rejection_set_revision: int
    rejection_count: int


@dataclass(frozen=True, slots=True)
class SuggestionRunPage:
    items: tuple[Any, ...]
    next_cursor: str | None


class OpaqueSuggestionCursorCodec:
    """Sign bounded keyset positions while keeping cursor logic out of DB code."""

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes) or len(secret) < 16:
            raise ValueError("notes_graph_cursor_secret_invalid")
        self._secret = secret

    @staticmethod
    def _segment(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @classmethod
    def _unsegment(cls, value: str) -> bytes:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        if cls._segment(decoded) != value:
            raise ValueError("notes_graph_cursor_invalid")
        return decoded

    def encode(self, *, binding: dict[str, object], position: tuple[str, str]) -> str:
        payload = {"v": 1, "binding": binding, "after": list(position)}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
        if len(raw) > 2048:
            raise ValueError("notes_graph_cursor_invalid")
        signature = hmac.digest(self._secret, raw, "sha256")
        return f"{self._segment(raw)}.{self._segment(signature)}"

    def decode(self, cursor: str, *, binding: dict[str, object]) -> tuple[str, str]:
        if not isinstance(cursor, str) or len(cursor.encode("ascii", "ignore")) > 4096:
            raise ValueError("notes_graph_cursor_invalid")
        try:
            payload_segment, signature_segment = cursor.split(".")
            raw = self._unsegment(payload_segment)
            signature = self._unsegment(signature_segment)
            if not hmac.compare_digest(signature, hmac.digest(self._secret, raw, "sha256")):
                raise ValueError("notes_graph_cursor_invalid")
            payload = json.loads(raw)
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("notes_graph_cursor_invalid") from exc
        after = payload.get("after") if isinstance(payload, dict) else None
        if (
            payload.get("v") != 1
            or payload.get("binding") != binding
            or not isinstance(after, list)
            or len(after) != 2
            or not all(isinstance(value, str) for value in after)
        ):
            raise ValueError("notes_graph_cursor_invalid")
        return after[0], after[1]


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


class NotesGraphSuggestionsAPI:
    """Compose owner-bound persistence and existing suggestion coordinators."""

    _ACTIVE_RUN_STATES = ("admitting", "queued", "running", "cancelling", "publishing")

    def __init__(
        self,
        *,
        store: Any,
        owner_user_id: str,
        dataset_id: str,
        admission_service: Any,
        cancellation_coordinator: Any,
        decision_service: Any,
        worker_ready: Callable[[], bool],
        feature_ready: Callable[[], bool],
        cursor_codec: Any,
        resolve_capability: Callable[..., Any] = resolve_generation_capability,
        clock: Callable[[], datetime] = _default_clock,
    ) -> None:
        self._store = store
        self._owner_user_id = owner_user_id
        self._dataset_id = dataset_id
        self._admission = admission_service
        self._cancellation = cancellation_coordinator
        self._decisions = decision_service
        self._worker_ready = worker_ready
        self._feature_ready = feature_ready
        self._cursor = cursor_codec
        self._resolve_capability = resolve_capability
        self._clock = clock

    @property
    def capability_resolver(self) -> Callable[..., Any]:
        return self._resolve_capability

    @staticmethod
    def _translate(exc: Exception) -> SuggestionAPIError:
        code = str(exc)
        if isinstance(exc, SuggestionAPIError):
            return exc
        if isinstance(exc, NotesGraphSourceTooLargeError):
            return SuggestionAPIError(422, "notes_graph_source_too_large")
        if isinstance(exc, NotesGraphFTSNotReadyError):
            return SuggestionAPIError(503, "notes_graph_fts_not_ready")
        if code in {
            "Notes graph source is unavailable",
            "notes_graph_run_not_found",
            "notes_graph_run_cancel_resource_missing",
            "notes_graph_suggestion_not_found",
        }:
            return SuggestionAPIError(404, "notes_graph_suggestion_not_found")
        if code == "notes_graph_cursor_invalid":
            return SuggestionAPIError(422, code)
        if "rate_limited" in code:
            return SuggestionAPIError(429, code)
        if "provider_model_disallowed" in code:
            return SuggestionAPIError(422, "notes_graph_provider_disallowed")
        if "contract_invalid" in code or "limit_invalid" in code or "filter_invalid" in code:
            return SuggestionAPIError(422, "notes_graph_invalid_request")
        if "not_ready" in code or "unavailable" in code:
            return SuggestionAPIError(503, code)
        if "conflict" in code or "mismatch" in code or "fingerprint_stale" in code:
            return SuggestionAPIError(409, code)
        return SuggestionAPIError(503, "notes_graph_suggestions_unavailable")

    def _source(self, note_id: str) -> Any:
        try:
            source = self._store.load_source_note(
                dataset_id=self._dataset_id,
                note_id=note_id,
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        if source.note_id != note_id:
            raise SuggestionAPIError(404, "notes_graph_suggestion_not_found")
        return source

    @staticmethod
    def _resolved_parts(resolved: Any) -> tuple[SuggestionCapabilities, Any]:
        if isinstance(resolved, tuple):
            return resolved
        return resolved.capabilities, resolved.provider

    def get_capabilities(
        self,
        *,
        note_id: str,
        provider: str | None,
        model: str | None,
    ) -> SuggestionCapabilities:
        self._source(note_id)
        try:
            resolved = self._resolve_capability(provider=provider, model=model)
            capabilities, _generation_provider = self._resolved_parts(resolved)
        except Exception as exc:
            if str(exc) == "notes_graph_provider_model_disallowed":
                capabilities = build_unavailable_suggestion_capabilities(
                    provider=getattr(exc, "provider", provider),
                    model=getattr(exc, "model", model),
                )
            else:
                raise self._translate(exc) from exc

        unavailable_reason = None
        try:
            self._store.ensure_fts_ready(dataset_id=self._dataset_id)
        except NotesGraphFTSNotReadyError:
            unavailable_reason = "notes_graph_fts_not_ready"
        if not self._feature_ready():
            unavailable_reason = "notes_graph_suggestions_disabled"
        elif not self._worker_ready():
            unavailable_reason = "notes_graph_suggestions_worker_unavailable"
        elif not capabilities.generation_available:
            unavailable_reason = capabilities.unavailable_reason or "notes_graph_provider_unavailable"
        if unavailable_reason is None:
            return capabilities
        return replace(
            capabilities,
            generation_available=False,
            unavailable_reason=unavailable_reason,
        )

    @staticmethod
    def _admission_status(capabilities: SuggestionCapabilities) -> int:
        if capabilities.unavailable_reason in {
            "notes_graph_provider_disallowed",
            "notes_graph_provider_model_disallowed",
        }:
            return 422
        return 503

    def admit_run(
        self,
        *,
        note_id: str,
        provider: str | None,
        model: str | None,
        capability_revision: str,
        idempotency_key: str,
    ) -> Any:
        prompt_contract_version = "notes-graph-suggestion-prompt-v1"
        replay = None
        replay_admission = getattr(self._admission, "replay", None)
        if replay_admission is not None:
            try:
                replay = replay_admission(
                    dataset_id=self._dataset_id,
                    source_note_id=note_id,
                    requested_provider=provider,
                    requested_model=model,
                    prompt_contract_version=prompt_contract_version,
                    idempotency_key=idempotency_key,
                )
            except Exception as exc:
                raise self._translate(exc) from exc
        if replay is not None:
            return replay
        source = self._source(note_id)
        capabilities = self.get_capabilities(note_id=note_id, provider=provider, model=model)

        def validate(run: Any) -> None:
            if capability_revision != capabilities.revision:
                raise SuggestionAPIError(412, "notes_graph_capabilities_changed")
            if not capabilities.generation_available:
                code = capabilities.unavailable_reason or "notes_graph_provider_unavailable"
                raise SuggestionAPIError(self._admission_status(capabilities), code)
            if (
                run.provider != capabilities.provider
                or run.model != capabilities.model
                or run.capability_revision != capabilities.revision
            ):
                raise SuggestionAPIError(412, "notes_graph_capabilities_changed")

        try:
            return self._admission.admit(
                dataset_id=self._dataset_id,
                source_note_id=note_id,
                source_fingerprint=content_fingerprint(source.title, source.content),
                provider=capabilities.provider,
                model=capabilities.model,
                requested_provider=provider,
                requested_model=model,
                capability_revision=capabilities.revision,
                prompt_contract_version=prompt_contract_version,
                idempotency_key=idempotency_key,
                now=self._clock(),
                validate_before_enqueue=validate,
            )
        except Exception as exc:
            raise self._translate(exc) from exc

    def _binding(self, kind: str, note_id: str, fingerprint: str, filters: object) -> dict[str, object]:
        return {
            "kind": kind,
            "owner": hashlib.sha256(self._owner_user_id.encode()).hexdigest(),
            "dataset": hashlib.sha256(self._dataset_id.encode()).hexdigest(),
            "source": note_id,
            "fingerprint": fingerprint,
            "filters": filters,
        }

    def list_runs(
        self,
        *,
        note_id: str,
        states: tuple[str, ...] | None,
        limit: int,
        cursor: str | None,
    ) -> SuggestionRunPage:
        source = self._source(note_id)
        fingerprint = content_fingerprint(source.title, source.content)
        selected = states or self._ACTIVE_RUN_STATES
        binding = self._binding("runs", note_id, fingerprint, sorted(set(selected)))
        try:
            after = self._cursor.decode(cursor, binding=binding) if cursor else None
            page = self._store.list_runs(
                dataset_id=self._dataset_id,
                source_note_id=note_id,
                states=selected,
                limit=limit,
                after=after,
            )
            next_cursor = (
                self._cursor.encode(binding=binding, position=page.next_position)
                if page.next_position is not None
                else None
            )
            return SuggestionRunPage(page.items, next_cursor)
        except Exception as exc:
            raise self._translate(exc) from exc

    def get_run(self, *, note_id: str, run_id: str) -> Any:
        self._source(note_id)
        try:
            run = self._store.get_run(dataset_id=self._dataset_id, run_id=run_id)
        except Exception as exc:
            raise self._translate(exc) from exc
        if run.source_note_id != note_id:
            raise SuggestionAPIError(404, "notes_graph_suggestion_not_found")
        return run

    def cancel_run(
        self,
        *,
        note_id: str,
        run_id: str,
        expected_revision: int,
        idempotency_key: str,
    ) -> Any:
        try:
            result = self._cancellation.cancel(
                dataset_id=self._dataset_id,
                run_id=run_id,
                expected_source_note_id=note_id,
                expected_state=None,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
                now=self._clock(),
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        return result

    def list_suggestions(
        self,
        *,
        note_id: str,
        states: tuple[str, ...] | None,
        limit: int,
        cursor: str | None,
    ) -> SuggestionReviewPage:
        source = self._source(note_id)
        fingerprint = content_fingerprint(source.title, source.content)
        selected = states or ("pending", "accepting")
        binding = self._binding("suggestions", note_id, fingerprint, sorted(set(selected)))
        try:
            after = self._cursor.decode(cursor, binding=binding) if cursor else None
            page = self._store.list_suggestions(
                dataset_id=self._dataset_id,
                source_note_id=note_id,
                source_fingerprint=fingerprint,
                states=selected,
                limit=limit,
                after=after,
            )
            evidence_rows = (
                self._store.list_suggestion_evidence(
                    dataset_id=self._dataset_id,
                    source_note_id=note_id,
                    source_fingerprint=fingerprint,
                    suggestion_ids=tuple(item.id for item in page.items),
                    limit=min(600, max(1, len(page.items) * 6)),
                )
                if page.items
                else ()
            )
            evidence_by_id: dict[str, list[SuggestionEvidenceExcerpt]] = {}
            suggestions_by_id = {item.id: item for item in page.items}
            target_title_by_id: dict[str, str] = {}
            for row in evidence_rows:
                evidence = row.evidence
                text = reconstruct_evidence(
                    EvidenceReference(
                        note_id=evidence.note_id,
                        field=evidence.field.value,
                        fingerprint=evidence.content_fingerprint,
                        start_offset=evidence.start_offset,
                        end_offset=evidence.end_offset,
                    ),
                    title=row.excerpt_note.title,
                    content=row.excerpt_note.content,
                )
                if text is None:
                    continue
                suggestion = suggestions_by_id.get(evidence.suggestion_id)
                if (
                    suggestion is not None
                    and suggestion.kind.value == "related_note"
                    and evidence.side.value == "target"
                    and evidence.note_id == suggestion.target_note_id
                    and evidence.content_fingerprint == suggestion.target_fingerprint
                ):
                    target_title_by_id[suggestion.id] = row.excerpt_note.title
                evidence_by_id.setdefault(evidence.suggestion_id, []).append(
                    SuggestionEvidenceExcerpt(
                        side=evidence.side.value,
                        note_id=evidence.note_id,
                        field=evidence.field.value,
                        start_offset=evidence.start_offset,
                        end_offset=evidence.end_offset,
                        text=text,
                    )
                )
            rejection_set = self._store.get_rejection_set(
                dataset_id=self._dataset_id,
                source_note_id=note_id,
                source_fingerprint=fingerprint,
            )
            next_cursor = (
                self._cursor.encode(binding=binding, position=page.next_position)
                if page.next_position is not None
                else None
            )
            return SuggestionReviewPage(
                items=tuple(
                    SuggestionReviewItem(
                        item,
                        tuple(evidence_by_id.get(item.id, ())),
                        target_title_by_id.get(item.id),
                    )
                    for item in page.items
                ),
                next_cursor=next_cursor,
                current_source_fingerprint=fingerprint,
                rejection_set_revision=rejection_set.revision if rejection_set else 0,
                rejection_count=rejection_set.rejection_count if rejection_set else 0,
            )
        except Exception as exc:
            raise self._translate(exc) from exc

    def _suggestion(self, *, note_id: str, suggestion_id: str) -> NoteGraphSuggestion:
        self._source(note_id)
        try:
            suggestion = self._store.get_suggestion(
                dataset_id=self._dataset_id,
                suggestion_id=suggestion_id,
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        if suggestion.source_note_id != note_id:
            raise SuggestionAPIError(404, "notes_graph_suggestion_not_found")
        return suggestion

    def accept_permission_requirements(
        self,
        *,
        note_id: str,
        suggestion_id: str,
        expected_revision: int | None = None,
        expected_source_fingerprint: str | None = None,
        expected_target_fingerprint: str | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[str, ...]:
        if (
            expected_revision is not None
            and expected_source_fingerprint is not None
            and idempotency_key is not None
            and hasattr(self._store, "get_acceptance_authorization_scope")
        ):
            try:
                scope = self._store.get_acceptance_authorization_scope(
                    dataset_id=self._dataset_id,
                    source_note_id=note_id,
                    suggestion_id=suggestion_id,
                    expected_revision=expected_revision,
                    expected_source_fingerprint=expected_source_fingerprint,
                    expected_target_fingerprint=expected_target_fingerprint,
                    idempotency_key=idempotency_key,
                )
            except Exception as exc:
                raise self._translate(exc) from exc
            if scope == "relationship":
                return ("notes.graph.write",)
            if scope == "existing_tag":
                return ("notes.link_keyword",)
            if scope == "new_tag":
                return ("notes.link_keyword", "keywords.create")
            raise SuggestionAPIError(503, "notes_graph_suggestions_unavailable")
        suggestion = self._suggestion(note_id=note_id, suggestion_id=suggestion_id)
        if suggestion.kind == NoteGraphSuggestionKind.RELATED_NOTE:
            return ("notes.graph.write",)
        required = ["notes.link_keyword"]
        if suggestion.keyword_sync_id is None:
            required.append("keywords.create")
        return tuple(required)

    def accept_suggestion(self, *, note_id: str, suggestion_id: str, **kwargs: Any) -> Any:
        probe = getattr(self._store, "get_terminal_acceptance_replay", None)
        if probe is not None:
            try:
                replay = probe(
                    dataset_id=self._dataset_id,
                    source_note_id=note_id,
                    suggestion_id=suggestion_id,
                    **kwargs,
                )
            except Exception as exc:
                raise self._translate(exc) from exc
            if replay is not None:
                return replay
        self._suggestion(note_id=note_id, suggestion_id=suggestion_id)
        try:
            return self._decisions.accept(
                dataset_id=self._dataset_id,
                suggestion_id=suggestion_id,
                **kwargs,
            )
        except Exception as exc:
            raise self._translate(exc) from exc

    def reject_suggestion(self, *, note_id: str, suggestion_id: str, **kwargs: Any) -> Any:
        probe = getattr(self._store, "get_terminal_rejection_replay", None)
        if probe is not None:
            try:
                replay = probe(
                    dataset_id=self._dataset_id,
                    source_note_id=note_id,
                    suggestion_id=suggestion_id,
                    **kwargs,
                )
            except Exception as exc:
                raise self._translate(exc) from exc
            if replay is not None:
                return replay
        self._suggestion(note_id=note_id, suggestion_id=suggestion_id)
        try:
            return self._decisions.reject(
                dataset_id=self._dataset_id,
                suggestion_id=suggestion_id,
                **kwargs,
            )
        except Exception as exc:
            raise self._translate(exc) from exc

    def reset_rejections(self, *, note_id: str, **kwargs: Any) -> Any:
        probe = getattr(self._store, "get_terminal_rejection_reset_replay", None)
        if probe is not None:
            try:
                replay = probe(
                    dataset_id=self._dataset_id,
                    source_note_id=note_id,
                    **kwargs,
                )
            except Exception as exc:
                raise self._translate(exc) from exc
            if replay is not None:
                return replay
        self._source(note_id)
        try:
            return self._decisions.reset_rejections(
                dataset_id=self._dataset_id,
                source_note_id=note_id,
                **kwargs,
            )
        except Exception as exc:
            raise self._translate(exc) from exc


class _UnavailableDecisionService:
    def __getattr__(self, _name: str) -> Any:
        raise RuntimeError("notes_graph_sync_not_ready")


def build_notes_graph_suggestions_api(
    *,
    note_db: Any,
    owner_user_id: str,
    dataset_id: str,
    jobs: Any,
) -> NotesGraphSuggestionsAPI:
    """Build the single owner/dataset-bound API facade used by nested routes."""

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.Notes_Graph.graph_service import NOTES_GRAPH_ENABLED
    from tldw_Server_API.app.core.Notes_Graph.suggestion_jobs import (
        SuggestionAdmissionService,
        SuggestionCancellationCoordinator,
    )
    from tldw_Server_API.app.core.Notes_Graph.suggestion_service import (
        build_suggestion_decision_service,
    )
    from tldw_Server_API.app.core.testing import env_flag_enabled

    settings = get_settings()
    secret_material = str(
        getattr(settings, "JWT_SECRET_KEY", "")
        or getattr(settings, "SINGLE_USER_API_KEY", "")
        or "notes-graph-cursor-local"
    ).encode()
    store = note_db.note_graph_suggestion_store
    decisions = build_suggestion_decision_service(
        note_db=note_db,
        owner_user_id=owner_user_id,
        dataset_id=dataset_id,
    )
    return NotesGraphSuggestionsAPI(
        store=store,
        owner_user_id=owner_user_id,
        dataset_id=dataset_id,
        admission_service=SuggestionAdmissionService(
            store=store,
            jobs=jobs,
            owner_user_id=owner_user_id,
        ),
        cancellation_coordinator=SuggestionCancellationCoordinator(
            store=store,
            jobs=jobs,
            owner_user_id=owner_user_id,
        ),
        decision_service=decisions or _UnavailableDecisionService(),
        worker_ready=lambda: jobs is not None
        and env_flag_enabled("NOTES_GRAPH_SUGGESTIONS_WORKER_ENABLED"),
        feature_ready=NOTES_GRAPH_ENABLED,
        cursor_codec=OpaqueSuggestionCursorCodec(hashlib.sha256(secret_material).digest()),
    )


__all__ = [
    "NotesGraphSuggestionsAPI",
    "OpaqueSuggestionCursorCodec",
    "SuggestionAPIError",
    "SuggestionEvidenceExcerpt",
    "SuggestionReviewItem",
    "SuggestionReviewPage",
    "SuggestionRunPage",
    "build_notes_graph_suggestions_api",
]
