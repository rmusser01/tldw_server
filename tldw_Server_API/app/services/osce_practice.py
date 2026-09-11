"""Domain helpers for OSCE station identity and response projection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel

from tldw_Server_API.app.api.v1.schemas.osce import (
    OsceStationCreateContent,
    OsceStationStoredContent,
    OsceStationSummary,
    OsceStationUpdateContent,
    OsceVerificationState,
)


class OsceStationIdentityError(ValueError):
    """Raised when an update violates server-owned nested identity."""


@dataclass(frozen=True)
class ReconciledOsceStation:
    """Validated station content and its post-update verification state."""

    content: OsceStationStoredContent
    verification_state: OsceVerificationState


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
