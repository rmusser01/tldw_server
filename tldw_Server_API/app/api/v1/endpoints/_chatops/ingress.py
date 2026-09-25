"""The request path shared by the Discord and Slack ChatOps endpoints (ADR-050 stage 3).

Per protocol, and injected by each endpoint: the request signature check, the command
parser, the tenant field names, and the objects tests patch on each endpoint module
(the job manager, org-membership lookup, metric emitter). Everything here is the same
for both: ingress rate-limit and duplicate responses, job submission, the ``status``
command's tenant/owner scoping, the job-status route, and workspace org resolution.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

Emit = Callable[..., None]


def rate_limited_response(retry_after: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(retry_after)},
        content={"ok": False, "error": "rate_limited", "retry_after_seconds": retry_after},
    )


def duplicate_response() -> JSONResponse:
    return JSONResponse(status_code=200, content={"ok": True, "status": "duplicate"})


def job_summary(job: dict[str, Any], job_id: int) -> dict[str, Any]:
    return {
        "id": job_id,
        "status": job.get("status"),
        "domain": job.get("domain"),
        "queue": job.get("queue"),
        "job_type": job.get("job_type"),
    }


def submit_job(
    jm: Any,
    *,
    domain: str,
    action: str,
    request_id: str,
    owner_user_id: str | None,
    payload: dict[str, Any],
    response_mode: str,
    safe_int: Callable[[Any], int | None],
) -> dict[str, Any]:
    """Queue a ``{domain}_{action}`` job; ``payload`` holds the protocol's routing fields."""
    job = jm.create_job(
        domain=domain,
        queue="default",
        job_type=f"{domain}_{action}",
        payload={"request_id": request_id, **payload, "response_mode": response_mode},
        owner_user_id=owner_user_id,
        request_id=request_id,
    )
    return {
        "job_id": safe_int(job.get("id")),
        "request_id": request_id,
        "response_mode": response_mode,
        "job_status": str(job.get("status") or "queued"),
    }


def status_command_response(
    jm: Any,
    *,
    parsed_command: dict[str, Any],
    policy: dict[str, Any],
    tenant_field: str,
    tenant_id: str | None,
    actor_user_id: str | None,
    emit: Emit,
    requests_metric: str,
    endpoint: str,
    coerce: Callable[[Any], str | None],
    safe_int: Callable[[Any], int | None],
) -> JSONResponse:
    """Answer ``status <job id>`` for a job in this tenant (and, if the policy says so, this user)."""
    requested_job_id = safe_int(parsed_command.get("input"))
    if requested_job_id is None:
        emit(requests_metric, endpoint=endpoint, outcome="invalid_status_query")
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": "invalid_status_query",
                "message": "Status command requires a numeric job id. Example: status 42",
            },
        )
    job = jm.get_job(requested_job_id)
    job_payload = job.get("payload") if isinstance(job, dict) and isinstance(job.get("payload"), dict) else {}
    job_tenant_id = coerce(job_payload.get(tenant_field))
    owner_user_id = coerce(job.get("owner_user_id")) if isinstance(job, dict) else None
    # "guild_and_user" / "workspace_and_user": restrict to the requesting user's own jobs.
    user_scoped = str(policy.get("status_scope") or "").strip().lower().endswith("_and_user")
    wrong_tenant = bool(job_tenant_id and tenant_id and job_tenant_id != tenant_id)
    wrong_user = bool(user_scoped and actor_user_id and owner_user_id and actor_user_id != owner_user_id)
    if not job or wrong_tenant or wrong_user:
        emit(requests_metric, endpoint=endpoint, outcome="status_denied")
        return JSONResponse(status_code=404, content={"ok": False, "error": "job_not_found", "job_id": requested_job_id})
    emit(requests_metric, endpoint=endpoint, outcome="accepted", action="status")
    return JSONResponse(
        status_code=200,
        content={"ok": True, "status": "accepted", "parsed": parsed_command, "job": job_summary(job, requested_job_id)},
    )


_ACTIVE_MEMBERSHIP_STATUSES = frozenset({"", "active", "member", "approved"})


async def job_status_payload(
    jm: Any,
    job_id: int,
    *,
    domain: str,
    tenant_field: str,
    user_id: int,
    list_memberships: Callable[[int], Awaitable[Any]],
    get_installations_repo: Callable[[], Awaitable[Any]],
    policy_for: Callable[[str | None], dict[str, Any]],
    coerce: Callable[[Any], str | None],
    auth_mode: str,
) -> dict[str, Any]:
    """The ``GET /{domain}/jobs/{job_id}`` body, for a caller allowed to see that job.

    Allowed: the job's owner, or an active member of an org that installed the job's
    guild/workspace -- unless that tenant's policy limits status to the job owner, as
    the in-platform ``status`` command does. Anything else is a 404, so ids do not
    reveal which jobs exist. In single-user mode the one user owns every installation.
    """
    not_found = HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job_not_found")
    job = jm.get_job(int(job_id))
    if not job or str(job.get("domain") or "").strip().lower() != domain:
        raise not_found
    body = {"ok": True, "job": job_summary(job, int(job.get("id") or job_id))}

    if auth_mode.strip().lower() == "single_user" or coerce(job.get("owner_user_id")) == str(int(user_id)):
        return body

    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    tenant_id = coerce(payload.get(tenant_field))
    if not tenant_id:
        raise not_found
    if str(policy_for(tenant_id).get("status_scope") or "").strip().lower().endswith("_and_user"):
        raise not_found

    org_ids: list[int] = []
    for membership in await list_memberships(int(user_id)) or []:
        membership = membership or {}
        if str(membership.get("status") or "").strip().lower() not in _ACTIVE_MEMBERSHIP_STATUSES:
            continue
        try:
            org_ids.append(int(membership.get("org_id")))
        except (TypeError, ValueError):
            continue
    if not org_ids:
        raise not_found
    repo = await get_installations_repo()
    for org_id in org_ids:
        # A disabled installation no longer grants its org's members access to the
        # tenant's jobs; the job owner keeps access through the check above.
        for installation in (
            await repo.list_installations(org_id=org_id, provider=domain, include_disabled=False) or []
        ):
            if installation.get("disabled"):
                continue
            if coerce(installation.get("external_id")) == tenant_id:
                return body
    raise not_found


async def resolve_workspace_org_id(
    request: Request | None,
    user_id: int,
    *,
    list_memberships: Callable[[int], Awaitable[Any]],
    safe_int: Callable[[Any], int | None],
    auth_mode: str,
) -> int:
    """The org an installation belongs to: request scope, then single-user org 1, then memberships."""
    if request is not None:
        active_org_id = safe_int(getattr(request.state, "active_org_id", None))
        if active_org_id is not None and active_org_id > 0:
            return active_org_id
        request_org_ids = getattr(request.state, "org_ids", None)
        if isinstance(request_org_ids, (list, tuple, set)):
            for candidate in request_org_ids:
                org_id = safe_int(candidate)
                if org_id is not None and org_id > 0:
                    return org_id

    if auth_mode.strip().lower() == "single_user":
        return 1

    memberships = await list_memberships(int(user_id)) or []
    fallback: int | None = None
    for membership in memberships:
        try:
            org_id = int((membership or {}).get("org_id"))
        except (TypeError, ValueError):
            continue
        if org_id <= 0:
            continue
        status_value = str((membership or {}).get("status") or "").strip().lower()
        if not status_value or status_value in {"active", "member", "approved"}:
            return org_id
        if fallback is None:
            fallback = org_id
    if fallback is not None:
        return fallback

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unable to resolve workspace organization for installation",
    )
