"""Admin endpoint for content-free Context Integrity boot-state inspection."""

from __future__ import annotations

from fastapi import APIRouter, Request

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminContextIntegrityFinding,
    AdminContextIntegrityResponse,
)


router = APIRouter()


@router.get(
    "/context-integrity",
    response_model=AdminContextIntegrityResponse,
)
async def get_context_integrity_status(request: Request) -> AdminContextIntegrityResponse:
    """Return current-process Context Integrity boot state."""
    boot_state = getattr(request.app.state, "context_integrity_boot_state", None)
    if boot_state is None:
        return AdminContextIntegrityResponse(
            mode="uninitialized",
            degraded=True,
            manifest_sequence=None,
            manifest_digest=None,
            findings_present=False,
            findings=[],
        )

    findings = [AdminContextIntegrityFinding.model_validate(finding) for finding in boot_state.findings]
    return AdminContextIntegrityResponse(
        mode=str(boot_state.mode),
        degraded=bool(boot_state.degraded),
        manifest_sequence=boot_state.manifest_sequence,
        manifest_digest=boot_state.manifest_digest,
        findings_present=bool(findings),
        findings=findings,
    )
