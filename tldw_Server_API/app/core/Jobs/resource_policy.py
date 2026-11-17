"""Closed opt-in registration for Jobs resource-required types."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.claims_cluster_rebuild_contract import (
    CLAIMS_REBUILD_CLUSTERS_JOB_TYPE,
    ClusterRebuildSettings,
    canonical_cluster_settings,
    validate_cluster_rebuild_payload,
)

if TYPE_CHECKING:
    from .operations.contracts import CreateJobCommand
    from .resource_contracts import ResourceAdmissionRequest

_CLUSTER_ALIAS = re.compile(re.escape(CLAIMS_REBUILD_CLUSTERS_JOB_TYPE) + r"\.v[1-9][0-9]*\Z")


def resource_requirement_for(domain: str, job_type: str) -> str | None:
    """Resolve the registered resource or fail closed on reserved version aliases."""
    if domain == "claims":
        if job_type == CLAIMS_REBUILD_CLUSTERS_JOB_TYPE:
            return "cluster-rebuild"
        if _CLUSTER_ALIAS.fullmatch(job_type):
            raise ValueError("unsupported cluster rebuild job type alias")
    return None


def reject_resource_required(domain: str, job_type: str) -> None:
    """Close ordinary and independent-receipt entry points, including aliases."""
    try:
        required = resource_requirement_for(domain, job_type)
    except ValueError:
        required = True
    if required:
        raise ValueError("resource_admission_required")


def validate_resource_command(
    command: CreateJobCommand, request: ResourceAdmissionRequest
) -> tuple[CreateJobCommand, ResourceAdmissionRequest]:
    """Bind declared scope/settings to the neutral, validated payload contract.

    Imports stay local to avoid a contracts/policy cycle. This validates facts,
    not caller authorization; callers establish their Jobs owner scope.
    """
    from .resource_contracts import ResourceAdmissionRequest, canonical_settings_json

    if not isinstance(request, ResourceAdmissionRequest):
        raise ValueError("resource admission request is required")
    request = replace(request)
    required = resource_requirement_for(command.domain, command.job_type)
    if (
        required is None
        or request.scope.domain != command.domain
        or request.scope.owner_user_id != command.owner_user_id
        or request.scope.resource != required
        or command.idempotency_key is not None
    ):
        raise ValueError("resource admission scope does not match command")
    payload = validate_cluster_rebuild_payload(command.payload)
    settings = canonical_cluster_settings(
        ClusterRebuildSettings(**{key: value for key, value in payload.items() if key != "owner_user_id"})
    )
    if payload["owner_user_id"] != command.owner_user_id or canonical_settings_json(
        settings
    ) != canonical_settings_json(request.canonical_settings):
        raise ValueError("resource admission settings do not match payload")
    return replace(command, payload=payload), request
