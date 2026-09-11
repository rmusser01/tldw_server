"""Domain helpers for OSCE stations and self-assessed practice attempts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel

from tldw_Server_API.app.api.v1.schemas.osce import (
    OsceAttemptState,
    OsceAttemptSummary,
    OsceCandidateAttemptResponse,
    OsceCandidateCitation,
    OsceCandidatePatientContext,
    OsceCandidateStation,
    OsceRevealedAttemptResponse,
    OsceRubricResult,
    OsceStationCreateContent,
    OsceStationStoredContent,
    OsceStationSummary,
    OsceStationUpdateContent,
    OsceVerificationState,
)
from tldw_Server_API.app.core.exceptions import OsceStationIdentityError


@dataclass(frozen=True)
class ReconciledOsceStation:
    """Validated station content and its post-update verification state."""

    content: OsceStationStoredContent
    verification_state: OsceVerificationState


def utc_now() -> str:
    """Return a bounded UTC timestamp suitable for OSCE response models."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )
    return str(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("OSCE attempt snapshot contains invalid JSON") from exc
    return value


def _snapshot(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_snapshot = row.get("station_snapshot")
    if raw_snapshot is None:
        raw_snapshot = row.get("station_snapshot_json")
    snapshot = _json_value(raw_snapshot)
    if not isinstance(snapshot, Mapping):
        raise ValueError("OSCE attempt is missing its station snapshot")
    return snapshot


def _snapshot_content(row: Mapping[str, Any]) -> OsceStationStoredContent:
    snapshot = _snapshot(row)
    raw_content = snapshot.get("content", snapshot)
    return OsceStationStoredContent.model_validate(_json_value(raw_content))


def _selection_mapping(value: Any) -> Mapping[Any, Any]:
    parsed = _json_value(value)
    if parsed is None:
        return {}
    if not isinstance(parsed, Mapping):
        raise ValueError("OSCE assessment selections must be mappings")
    return parsed


def _uuid_text(value: Any, *, field: str) -> str:
    try:
        return str(value if isinstance(value, UUID) else UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"invalid {field} UUID") from exc


def _payload(content: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(content, BaseModel):
        return content.model_dump(mode="json")
    return dict(content)


def _strip_nested_ids(payload: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(payload)
    stripped["checklist_items"] = [
        {key: value for key, value in dict(item).items() if key != "id"}
        for item in payload.get("checklist_items", [])
    ]
    domains: list[dict[str, Any]] = []
    for domain_value in payload.get("rubric_domains", []):
        domain = {key: value for key, value in dict(domain_value).items() if key != "id"}
        domain["levels"] = [
            {key: value for key, value in dict(level).items() if key != "id"}
            for level in domain.get("levels", [])
        ]
        domains.append(domain)
    stripped["rubric_domains"] = domains
    stripped["expected_key_points"] = [
        {key: value for key, value in dict(point).items() if key != "id"}
        for point in payload.get("expected_key_points", [])
    ]
    return stripped


def materialize_station_content(
    content: OsceStationCreateContent | Mapping[str, Any],
) -> OsceStationStoredContent:
    """Validate create content and assign fresh UUIDs to every nested object."""

    create = OsceStationCreateContent.model_validate(_strip_nested_ids(_payload(content)))
    payload = create.model_dump(mode="json")
    for item in payload["checklist_items"]:
        item["id"] = str(uuid4())
    for domain in payload["rubric_domains"]:
        domain["id"] = str(uuid4())
        for level in domain["levels"]:
            level["id"] = str(uuid4())
    for point in payload["expected_key_points"]:
        point["id"] = str(uuid4())
    return OsceStationStoredContent.model_validate(payload)


def _prior_identities(
    content: OsceStationStoredContent,
) -> dict[UUID, tuple[str, UUID | None]]:
    identities: dict[UUID, tuple[str, UUID | None]] = {}
    for item in content.checklist_items:
        identities[item.id] = ("checklist item", None)
    for domain in content.rubric_domains:
        identities[domain.id] = ("rubric domain", None)
        for level in domain.levels:
            identities[level.id] = ("rubric level", domain.id)
    for point in content.expected_key_points:
        identities[point.id] = ("key point", None)
    return identities


def _resolve_id(
    raw_id: Any,
    *,
    kind: str,
    parent_domain_id: UUID | None,
    prior: Mapping[UUID, tuple[str, UUID | None]],
    seen: set[UUID],
) -> UUID:
    if raw_id is None:
        generated = uuid4()
        while generated in prior or generated in seen:
            generated = uuid4()
        seen.add(generated)
        return generated

    nested_id = raw_id if isinstance(raw_id, UUID) else UUID(str(raw_id))
    if nested_id in seen:
        raise OsceStationIdentityError(f"duplicate nested UUID: {nested_id}")
    seen.add(nested_id)

    previous = prior.get(nested_id)
    if previous is None:
        raise OsceStationIdentityError(f"unknown {kind} UUID: {nested_id}")
    previous_kind, previous_parent = previous
    if previous_kind != kind:
        raise OsceStationIdentityError(
            f"{kind} UUID {nested_id} belongs to a {previous_kind}"
        )
    if kind == "rubric level" and previous_parent != parent_domain_id:
        raise OsceStationIdentityError(
            f"rubric level UUID {nested_id} cannot move to another domain"
        )
    return nested_id


def _reconcile_collection_ids(
    payload: dict[str, Any],
    prior: Mapping[UUID, tuple[str, UUID | None]],
) -> None:
    seen: set[UUID] = set()
    if "checklist_items" in payload:
        for item in payload["checklist_items"]:
            item["id"] = str(
                _resolve_id(
                    item.get("id"),
                    kind="checklist item",
                    parent_domain_id=None,
                    prior=prior,
                    seen=seen,
                )
            )
    if "rubric_domains" in payload:
        for domain in payload["rubric_domains"]:
            domain_id = _resolve_id(
                domain.get("id"),
                kind="rubric domain",
                parent_domain_id=None,
                prior=prior,
                seen=seen,
            )
            domain["id"] = str(domain_id)
            for level in domain["levels"]:
                level["id"] = str(
                    _resolve_id(
                        level.get("id"),
                        kind="rubric level",
                        parent_domain_id=domain_id,
                        prior=prior,
                        seen=seen,
                    )
                )
    if "expected_key_points" in payload:
        for point in payload["expected_key_points"]:
            point["id"] = str(
                _resolve_id(
                    point.get("id"),
                    kind="key point",
                    parent_domain_id=None,
                    prior=prior,
                    seen=seen,
                )
            )


def evidence_fingerprint(content: OsceStationStoredContent) -> str:
    """Hash only evidence-bearing station content."""

    def canonical_list(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            values,
            key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")),
        )

    def citations(values: list[Any]) -> list[dict[str, Any]]:
        return canonical_list([value.model_dump(mode="json") for value in values])

    evidence = {
        "patient_context": {
            "text": content.patient_context.text,
            "citations": citations(content.patient_context.citations),
        },
        "checklist": canonical_list(
            [
                {
                    "rationale": item.rationale,
                    "citations": citations(item.citations),
                }
                for item in content.checklist_items
            ]
        ),
        "key_points": canonical_list(
            [
                {
                    "text": point.text,
                    "citations": citations(point.citations),
                }
                for point in content.expected_key_points
            ]
        ),
    }
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def reconcile_station_update(
    stored: OsceStationStoredContent,
    update: OsceStationUpdateContent | OsceStationStoredContent | Mapping[str, Any],
    verification_state: OsceVerificationState | str,
) -> ReconciledOsceStation:
    """Merge an update while preserving and validating protected nested UUIDs."""

    if isinstance(update, OsceStationStoredContent):
        update_payload = update.model_dump(mode="json")
    else:
        update_model = (
            update
            if isinstance(update, OsceStationUpdateContent)
            else OsceStationUpdateContent.model_validate(update)
        )
        update_payload = update_model.model_dump(mode="json", exclude_unset=True)

    _reconcile_collection_ids(update_payload, _prior_identities(stored))
    merged_payload = stored.model_dump(mode="json")
    merged_payload.update(update_payload)
    reconciled = OsceStationStoredContent.model_validate(merged_payload)

    current_state = OsceVerificationState(getattr(verification_state, "value", verification_state))
    if (
        current_state is OsceVerificationState.SOURCE_VERIFIED
        and evidence_fingerprint(stored) != evidence_fingerprint(reconciled)
    ):
        current_state = OsceVerificationState.MODIFIED_AFTER_VERIFICATION
    return ReconciledOsceStation(content=reconciled, verification_state=current_state)


def project_station_summary(row: Mapping[str, Any]) -> OsceStationSummary:
    """Project a station row to its compact, marking-guide-free summary."""

    raw_content = row["content"]
    content = (
        raw_content
        if isinstance(raw_content, OsceStationStoredContent)
        else OsceStationStoredContent.model_validate(raw_content)
    )
    return OsceStationSummary.model_validate(
        {
            "id": row["id"],
            "quiz_id": row["quiz_id"],
            "title": content.title,
            "recommended_duration_seconds": content.recommended_duration_seconds,
            "order_index": row["order_index"],
            "version": row["version"],
            "checklist_count": len(content.checklist_items),
            "rubric_domain_count": len(content.rubric_domains),
            "verification_state": row["verification_state"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }
    )


def validate_assessment_selections(
    station: OsceStationStoredContent | Mapping[str, Any],
    checklist_selections: Mapping[Any, Any] | str | None,
    rubric_selections: Mapping[Any, Any] | str | None,
    *,
    require_complete: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """Validate selection IDs against one immutable station snapshot."""

    content = (
        station
        if isinstance(station, OsceStationStoredContent)
        else OsceStationStoredContent.model_validate(station)
    )
    checklist_ids = {str(item.id) for item in content.checklist_items}
    normalized_checklist: dict[str, str] = {}
    for raw_id, raw_selection in _selection_mapping(checklist_selections).items():
        item_id = _uuid_text(raw_id, field="checklist item")
        if item_id not in checklist_ids:
            raise ValueError(f"unknown checklist item UUID: {item_id}")
        if raw_selection not in {"met", "not_met"}:
            raise ValueError(f"invalid checklist selection for {item_id}")
        normalized_checklist[item_id] = str(raw_selection)

    domains = {str(domain.id): domain for domain in content.rubric_domains}
    normalized_rubric: dict[str, str] = {}
    for raw_domain_id, raw_level_id in _selection_mapping(rubric_selections).items():
        domain_id = _uuid_text(raw_domain_id, field="rubric domain")
        level_id = _uuid_text(raw_level_id, field="rubric level")
        domain = domains.get(domain_id)
        if domain is None:
            raise ValueError(f"unknown rubric domain UUID: {domain_id}")
        if level_id not in {str(level.id) for level in domain.levels}:
            raise ValueError(
                f"rubric level UUID {level_id} does not belong to domain {domain_id}"
            )
        normalized_rubric[domain_id] = level_id

    if require_complete and set(normalized_checklist) != checklist_ids:
        raise ValueError("checklist selections are incomplete")
    if require_complete and set(normalized_rubric) != set(domains):
        raise ValueError("rubric selections are incomplete")
    return dict(sorted(normalized_checklist.items())), dict(sorted(normalized_rubric.items()))


def project_candidate_attempt(row: Mapping[str, Any]) -> OsceCandidateAttemptResponse:
    """Build the candidate response strictly from non-guide allowlisted fields."""

    if str(getattr(row.get("state"), "value", row.get("state"))) != "in_progress":
        raise ValueError("candidate projection requires an in-progress attempt")
    content = _snapshot_content(row)
    citations = [
        OsceCandidateCitation(
            source_type=citation.source_type,
            source_id=citation.source_id,
            label=citation.label,
        )
        for citation in content.patient_context.citations
    ]
    station = OsceCandidateStation(
        schema_version=content.schema_version,
        title=content.title,
        candidate_instructions=content.candidate_instructions,
        candidate_task=content.candidate_task,
        patient_context=OsceCandidatePatientContext(
            text=content.patient_context.text,
            citations=citations,
        ),
        recommended_duration_seconds=content.recommended_duration_seconds,
    )
    return OsceCandidateAttemptResponse(
        id=row["id"],
        quiz_id=row["quiz_id"],
        station_id=row["station_id"],
        client_attempt_id=row["client_attempt_id"],
        state=OsceAttemptState.IN_PROGRESS,
        version=row["version"],
        station=station,
        notes=str(row.get("candidate_notes") or ""),
        started_at=_timestamp(row.get("started_at")),
        last_modified_at=_timestamp(row.get("last_modified_at")),
        server_time=utc_now(),
    )


def project_revealed_attempt(row: Mapping[str, Any]) -> OsceRevealedAttemptResponse:
    """Project a revealed or completed attempt from its immutable snapshot."""

    state = str(getattr(row.get("state"), "value", row.get("state")))
    if state not in {"self_assessment", "completed"}:
        raise ValueError("revealed projection requires a self-assessment or completed attempt")
    elapsed = row.get("elapsed_seconds", row.get("frozen_elapsed_seconds"))
    return OsceRevealedAttemptResponse(
        id=row["id"],
        quiz_id=row["quiz_id"],
        station_id=row["station_id"],
        client_attempt_id=row["client_attempt_id"],
        state=state,
        version=row["version"],
        station=_snapshot_content(row),
        notes=str(row.get("candidate_notes") or ""),
        checklist_selections=_selection_mapping(row.get("checklist_selections")),
        rubric_selections=_selection_mapping(row.get("rubric_selections")),
        started_at=_timestamp(row.get("started_at")),
        self_assessment_started_at=_timestamp(row.get("self_assessment_started_at")),
        completed_at=_timestamp(row.get("completed_at")),
        elapsed_seconds=elapsed,
        last_modified_at=_timestamp(row.get("last_modified_at")),
        server_time=utc_now(),
    )


def summarize_osce_attempt(row: Mapping[str, Any]) -> OsceAttemptSummary:
    """Return a compact, note-free and score-free attempt summary."""

    content = _snapshot_content(row)
    state = str(getattr(row.get("state"), "value", row.get("state")))
    checklist, rubric = validate_assessment_selections(
        content,
        _selection_mapping(row.get("checklist_selections")),
        _selection_mapping(row.get("rubric_selections")),
        require_complete=state == "completed",
    )
    checklist_met_count: int | None = None
    checklist_total: int | None = None
    rubric_results: list[OsceRubricResult] = []
    if state == "completed":
        checklist_met_count = sum(selection == "met" for selection in checklist.values())
        checklist_total = len(content.checklist_items)
        for domain in content.rubric_domains:
            selected_level_id = rubric[str(domain.id)]
            selected_level = next(
                level for level in domain.levels if str(level.id) == selected_level_id
            )
            rubric_results.append(
                OsceRubricResult(
                    domain_id=domain.id,
                    domain_label=domain.label,
                    level_id=selected_level.id,
                    level_label=selected_level.label,
                )
            )

    return OsceAttemptSummary(
        id=row["id"],
        quiz_id=row["quiz_id"],
        station_id=row["station_id"],
        client_attempt_id=row["client_attempt_id"],
        station_title=content.title,
        state=state,
        version=row["version"],
        started_at=_timestamp(row.get("started_at")),
        self_assessment_started_at=_timestamp(row.get("self_assessment_started_at")),
        completed_at=_timestamp(row.get("completed_at")),
        last_modified_at=_timestamp(row.get("last_modified_at")),
        elapsed_seconds=row.get("elapsed_seconds", row.get("frozen_elapsed_seconds")),
        checklist_met_count=checklist_met_count,
        checklist_total=checklist_total,
        rubric_results=rubric_results,
    )
