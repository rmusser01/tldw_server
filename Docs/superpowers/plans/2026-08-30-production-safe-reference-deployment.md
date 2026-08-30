# Production-Safe Reference Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a fail-closed production reference deployment whose only anonymous operational surface is exact minimal liveness, while readiness, health detail, metrics, data services, backup, upgrade, and rollback remain private or claim-first protected.

**Architecture:** Consolidate readiness into one typed service and project it through three boundaries: public liveness never calls it, loopback-only `/internal/ready` returns only ready/not-ready, and authenticated operator routes return sanitized detail. Add a standalone Caddy/app/PostgreSQL/Redis Compose profile, a deterministic static preflight, and a host-side operational gate that verifies immutable target and rollback images plus PostgreSQL, Redis, and app-data backups before starting the target app.

**Tech Stack:** Python 3.10+, FastAPI, existing AuthNZ `RequirePermission(SYSTEM_LOGS)`, pytest, PyYAML, Docker Compose, Caddy, PostgreSQL client tools, Redis recovery tools, Prometheus, Markdown.

**Spec:** `Docs/superpowers/specs/2026-08-30-production-safe-reference-deployment-design.md`

## Global Constraints

- Work only in the isolated `codex/task-13013-6-production-reference` worktree; preserve unrelated or concurrent changes.
- Follow test-driven development: establish each red test, implement the smallest complete behavior, then rerun the focused suite before committing.
- Use the existing project virtual environment; do not add Python or JavaScript dependencies.
- Keep `GET/HEAD /health` public, dependency-free, deterministic, and exactly `{"status":"ok"}` with `Cache-Control: no-store`.
- Make `/internal/ready` OpenAPI-hidden, loopback-only, forwarding-header agnostic, and limited to `{"status":"ready"}` or `{"status":"not_ready"}`.
- Protect every detailed health, readiness, security, and metrics route with existing claim-first `RequirePermission(SYSTEM_LOGS)` in every environment; retain the additional admin guard on metrics reset.
- Keep proxy denial as defense in depth: the application remains authoritative for authorization even when Caddy blocks private or legacy aliases.
- Use a standalone production Compose file. Only Caddy publishes host ports 80 and 443; PostgreSQL and Redis never publish host ports.
- Use explicit, configurable, private, non-overlapping `edge` and `backend` CIDRs. `backend` is `internal: true`; Caddy joins only `edge`; the app joins both; PostgreSQL and Redis join only `backend`.
- Align Uvicorn, AuthNZ, Setup, Resource Governor, and enabled MCP proxy trust to the edge CIDR until TASK-13144 supplies a global physical-peer contract.
- Keep the checked-in production env example names-only. Do not commit a deployable credential, domain, origin, image default, or backup path.
- Parse the raw operator env file without shell evaluation. Never print secret values or connection URLs.
- Require immutable `sha-*` tags or digests for current and rollback app images; require them to differ. Require exact version tags or digests for third-party images while leaving full provenance to TASK-13013.7.
- Static preflight is read-only and offline: no secret generation, file rewriting, image pulls, container starts, or network requests.
- Operational deployment verification runs on the host, never mounts the Docker socket into a container, and stops before target-app startup on any failed gate.
- Treat PostgreSQL, app data, and Redis as the durable recovery boundary. Configuration and secret backups remain operator-managed.
- Archive inspection is not restore proof. The runbook requires a disposable restore drill; extended recovery and soak evidence remain TASK-13013.9.
- Refresh `Docs/Published` only through `bash Helper_Scripts/refresh_docs_published.sh`; never edit generated mirrors directly.
- Run focused Bandit against every touched Python or shell execution surface before completion.
- A materially AI-authored PR cannot merge until the human requester supplies the repository-required Change summary explaining what changed and why.

---

## Stage Summary

### Stage 1: Control-plane boundary

**Goal:** Establish exact public liveness, loopback-only minimal readiness, one shared readiness calculation, and claim-first operator diagnostics.

**Success Criteria:** Route matrix, response minimization, no-store headers, loopback enforcement, shared-snapshot use, 401/403/permission/admin behavior, and metrics-reset admin behavior are covered by tests.

**Tests:** Focused Health, Services, AuthNZ, Monitoring, and Resource Governance tests.

**Status:** Not Started

### Stage 2: Standalone production topology

**Goal:** Add the standalone Compose, Caddy, and names-only environment contract.

**Success Criteria:** Only Caddy publishes 80/443; networks, trust, TLS, Redis/PostgreSQL auth, proxy header overwrite, route denial order, setup denial, immutable image variables, and the one-shot preflight dependency are statically enforced.

**Tests:** `test_docker_production_reference.py` plus optional read-only `docker compose config` rendering.

**Status:** Not Started

### Stage 3: Static fail-closed preflight

**Goal:** Reject unsafe environment, topology, secret, proxy, backup, setup, and image inputs without side effects.

**Success Criteria:** Every design invariant has a stable error code, failures aggregate without secret leakage, and one complete fixture passes.

**Tests:** `test_production_preflight.py` and focused Bandit.

**Status:** Not Started

### Stage 4: Operational backup, deployment, and rollback gate

**Goal:** Verify both app images and fresh PostgreSQL, Redis, and app-data artifacts before target startup, then support an explicit restore-backed rollback.

**Success Criteria:** Mocked command-order tests prove fail-closed sequencing, artifact validation, checksummed non-secret manifests, no Docker-socket mounts, and no target start after any failure.

**Tests:** `test_production_deploy.py`, archive/manifest tests, Make target contract tests, and focused Bandit.

**Status:** Not Started

### Stage 5: Probe migration, monitoring, documentation, and release verification

**Goal:** Move machine probes to the private endpoint, authenticate Prometheus, publish the operator runbook, and complete repository verification.

**Success Criteria:** Docker/Kubernetes probes, monitoring credentials, source/generated docs, recovery drill instructions, broader regression tests, Backlog evidence, PR checks, and the human Change summary gate all agree with the approved design.

**Tests:** Probe/docs contracts, docs refresh, OpenAPI check, focused and broader pytest, Bandit, `git diff --check`, and optional Docker render.

**Status:** Not Started

---

## File Responsibility Map

- `tldw_Server_API/app/services/readiness_service.py` — own the typed readiness snapshot, sanitized detail collection, and minimal/operator projections.
- `tldw_Server_API/app/main.py` — register exact public liveness, private internal readiness, protected legacy readiness aliases, and protected root/API metrics.
- `tldw_Server_API/app/api/v1/endpoints/health.py` — reuse the shared snapshot and protect all detailed API-v1 health surfaces with `SYSTEM_LOGS`.
- `tldw_Server_API/app/api/v1/endpoints/metrics.py` — protect all metrics reads with `SYSTEM_LOGS` while retaining the separate admin reset requirement.
- `tldw_Server_API/tests/Health/test_control_plane_access_contract.py` — prove route access, exact payloads, no-store headers, loopback behavior, and forwarding spoof resistance.
- `tldw_Server_API/tests/Health/test_shared_readiness_service.py` — prove all readiness projections consume one snapshot contract and sanitize failures.
- `tldw_Server_API/tests/AuthNZ_Unit/test_health_permissions_claims.py` — prove 401, 403, `system.logs`, and admin bypass for API-v1 health routes.
- `tldw_Server_API/tests/AuthNZ_Unit/test_metrics_permissions_claims.py` — prove the same claims matrix for metrics and the extra reset admin guard.
- `Dockerfiles/docker-compose.production.yml` — define the complete production reference and one-shot preflight dependency.
- `Dockerfiles/Production/Caddyfile` — terminate TLS, overwrite client-identity headers, deny private/legacy/setup paths, and proxy allowed paths.
- `Dockerfiles/production.env.example` — list required variable names with empty values and non-secret comments.
- `tldw_Server_API/tests/Utils/test_docker_production_reference.py` — statically enforce Compose, Caddy, and env-example invariants.
- `Helper_Scripts/Deployment/production_preflight.py` — parse raw env input and aggregate deterministic semantic/static failures without mutation.
- `tldw_Server_API/tests/Utils/test_production_preflight.py` — cover every preflight rejection and redaction branch.
- `Helper_Scripts/Deployment/production_artifacts.py` — create/verify checksums, safe archive metadata, and non-secret deployment manifests.
- `Helper_Scripts/Deployment/production_deploy.py` — orchestrate host-side verification, backup, deploy, and explicit restore-backed rollback with fixed argv subprocesses.
- `tldw_Server_API/tests/Utils/test_production_deploy.py` — prove command order, failure stops, artifact handling, rollback selection, and log redaction with a fake runner.
- `Makefile` — expose canonical `production-preflight`, `production-deploy`, and explicit `production-rollback` entry points.
- `Dockerfiles/Dockerfile.prod` and checked-in app Compose profiles — migrate container-local healthchecks from `/ready` to `/internal/ready`.
- `Helper_Scripts/Samples/Kubernetes/tldw-app-deployment.yaml` — replace network HTTP readiness with an in-container exec probe.
- `Dockerfiles/Monitoring/prometheus.yml` and `Dockerfiles/Monitoring/docker-compose.monitoring.yml` — send a scoped credential to the protected metrics route without embedding it in config.
- `Docs/Deployment/Production_Reference_Deployment.md` — canonical backup, upgrade, restore-drill, rollback, monitoring, and troubleshooting runbook.
- `Docs/Deployment/First_Time_Production_Setup.md`, `Docs/Deployment/Long_Term_Admin_Guide.md`, `Docs/Deployment/Reverse_Proxy_Examples.md`, and `Dockerfiles/README.md` — point production users to the reference and label legacy proxy overlays non-production.
- `tldw_Server_API/tests/Docs/test_production_reference_deployment_docs.py` — enforce commands, links, recovery boundaries, limitations, and generated mirror parity.
- `tldw_Server_API/tests/Utils/test_production_probe_contract.py` — enforce Docker, Kubernetes, Caddy, and Prometheus probe/access contracts.
- `backlog/tasks/task-13013.6 - Ship-a-production-safe-reference-deployment-and-health-surface.md` — record plan, touched files, verification, PR, review, and merge evidence through Backlog MCP.

---

### Task 1: Build the Shared Readiness and Claim-First Control Plane

**Files:**
- Create: `tldw_Server_API/app/services/readiness_service.py`
- Modify: `tldw_Server_API/app/main.py:2860-3165`
- Modify: `tldw_Server_API/app/api/v1/endpoints/health.py:1-360`
- Modify: `tldw_Server_API/app/api/v1/endpoints/metrics.py:1-300`
- Create: `tldw_Server_API/tests/Health/test_control_plane_access_contract.py`
- Create: `tldw_Server_API/tests/Health/test_shared_readiness_service.py`
- Create: `tldw_Server_API/tests/AuthNZ_Unit/test_health_permissions_claims.py`
- Modify: `tldw_Server_API/tests/AuthNZ_Unit/test_metrics_permissions_claims.py`
- Modify: `tldw_Server_API/tests/Services/test_main_readiness_shutdown.py`
- Modify: `tldw_Server_API/tests/Monitoring/test_metrics_surface_contracts.py`
- Modify: `tldw_Server_API/tests/Resource_Governance/test_health_policy_snapshot.py`
- Modify: `tldw_Server_API/tests/Resource_Governance/test_health_policy_snapshot_api_v1.py`

**Interfaces:**
- Consumes: `get_or_create_lifecycle_state(app: FastAPI) -> AppLifecycleState`, current AuthNZ/database/workflow/provider/OTEL/resource-governor checks, `RequirePermission(SYSTEM_LOGS)`, and `RequireRole("admin")`.
- Produces: immutable `ReadinessSnapshot(ready: bool, reason: str | None, details: Mapping[str, Any])`.
- Produces: `async collect_readiness_snapshot(app: FastAPI) -> ReadinessSnapshot`.
- Produces: `internal_readiness_payload(snapshot: ReadinessSnapshot) -> dict[str, str]`.
- Produces: `operator_readiness_payload(snapshot: ReadinessSnapshot) -> dict[str, Any]`.
- Produces: `is_loopback_peer(request: Request) -> bool`, based only on the ASGI client peer and never on forwarding headers.
- Produces: public `GET/HEAD /health`, private `GET/HEAD /internal/ready`, and protected detailed health/readiness/metrics routes.

- [ ] **Step 1: Write the red tests for exact public and internal responses**

Create `test_control_plane_access_contract.py` with direct route tests and request-scope unit cases:

```python
from starlette.requests import Request


def _request(peer: str, headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/internal/ready",
            "headers": headers or [],
            "client": (peer, 43210),
            "server": ("app", 8000),
            "scheme": "http",
            "query_string": b"",
        }
    )


def test_remote_peer_cannot_spoof_loopback_with_forwarding_headers() -> None:
    request = _request(
        "172.30.0.2",
        [(b"x-forwarded-for", b"127.0.0.1"), (b"x-real-ip", b"127.0.0.1")],
    )
    assert is_loopback_peer(request) is False


def test_loopback_peer_is_allowed_for_internal_probe() -> None:
    assert is_loopback_peer(_request("127.0.0.1")) is True
    assert is_loopback_peer(_request("::1")) is True
```

Add client assertions that `GET /health` returns status 200, body exactly `{"status":"ok"}`, content type JSON, and `Cache-Control: no-store`; `HEAD /health` returns the same status/headers with an empty body. For `/internal/ready`, use `httpx.ASGITransport(app=test_app, client=("127.0.0.1", 43100))` for loopback 200/503 and `client=("172.30.0.2", 43100)` for remote 404. Assert exact minimal bodies and no detail keys.

- [ ] **Step 2: Write the red tests for one shared readiness snapshot**

Create `test_shared_readiness_service.py` around the public interfaces:

```python
def test_internal_projection_discards_all_detail() -> None:
    snapshot = ReadinessSnapshot(
        ready=False,
        reason="database_unavailable",
        details={"database": {"type": "postgresql"}, "providers_initialized": False},
    )
    assert internal_readiness_payload(snapshot) == {"status": "not_ready"}


def test_operator_projection_keeps_only_sanitized_snapshot_detail() -> None:
    snapshot = ReadinessSnapshot(
        ready=True,
        reason=None,
        details={"database": {"status": "healthy", "type": "postgresql"}},
    )
    assert operator_readiness_payload(snapshot) == {
        "status": "ready",
        "database": {"status": "healthy", "type": "postgresql"},
    }
```

Monkeypatch `readiness_service.collect_readiness_snapshot` with one `AsyncMock` and call `/internal/ready`, `/ready`, `/health/ready`, `/api/v1/readyz`, and `/api/v1/health/ready` through small FastAPI test apps. Assert each readiness route calls the shared builder once per request; assert `/health` never calls it. Preserve current sanitized exception tests, including no raw database URL, provider exception, file path, or secret in operator responses.

- [ ] **Step 3: Write the red claims-matrix tests**

Use an isolated FastAPI app and override `auth_deps.get_auth_principal`. Define principals exactly as follows:

```python
def _principal(*, permissions: list[str], is_admin: bool = False) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="test-operator",
        token_type="access",
        jti=None,
        roles=["admin"] if is_admin else ["user"],
        permissions=permissions,
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )
```

Parameterize the detailed route set:

```python
DETAILED_HEALTH_PATHS = (
    "/ready",
    "/health/ready",
    "/api/v1/healthz",
    "/api/v1/readyz",
    "/api/v1/health",
    "/api/v1/health/live",
    "/api/v1/health/ready",
    "/api/v1/health/metrics",
    "/api/v1/health/security",
)

DETAILED_METRICS_PATHS = (
    "/metrics",
    "/api/v1/metrics",
    "/api/v1/metrics/text",
    "/api/v1/metrics/json",
    "/api/v1/metrics/health",
    "/api/v1/metrics/chat",
)
```

For every path assert anonymous 401, authenticated principal without `system.logs` 403, principal with `[SYSTEM_LOGS]` is not 401/403, and admin principal without the explicit permission is not 401/403. Keep `/api/v1/metrics/reset` 403 for a non-admin `system.logs` principal and 200 for an admin.

- [ ] **Step 4: Run the focused tests and confirm the intended failures**

Run:

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Health/test_control_plane_access_contract.py \
  tldw_Server_API/tests/Health/test_shared_readiness_service.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_health_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_metrics_permissions_claims.py \
  tldw_Server_API/tests/Resource_Governance/test_health_policy_snapshot.py -q
```

Expected: collection/import failures for the missing readiness service, public health contains obsolete metadata, `/internal/ready` is absent, and detailed routes do not yet enforce `SYSTEM_LOGS`.

- [ ] **Step 5: Extract the typed readiness service without changing diagnostic meaning**

Create the public types and projections:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from fastapi import FastAPI, Request


@dataclass(frozen=True)
class ReadinessSnapshot:
    ready: bool
    reason: str | None
    details: Mapping[str, Any]


def internal_readiness_payload(snapshot: ReadinessSnapshot) -> dict[str, str]:
    return {"status": "ready" if snapshot.ready else "not_ready"}


def operator_readiness_payload(snapshot: ReadinessSnapshot) -> dict[str, Any]:
    return {
        "status": "ready" if snapshot.ready else "not_ready",
        **dict(snapshot.details),
        **({"reason": snapshot.reason} if snapshot.reason else {}),
    }


def is_loopback_peer(request: Request) -> bool:
    import ipaddress

    client = request.scope.get("client")
    if not client:
        return False
    try:
        return ipaddress.ip_address(client[0]).is_loopback
    except ValueError:
        return False
```

Move `_public_database_health()` and the complete dependency-gathering body of `main.readiness_check()` into `collect_readiness_snapshot(app)`. Preserve these exact fields in `details`: `database`, `workflows_db`, `engine`, `providers_initialized`, `provider_health`, `otel_available`, and optional sanitized `rg_policy`. Convert the draining return to `ReadinessSnapshot(False, "shutdown_in_progress", {})`; convert unexpected guarded failures to `ReadinessSnapshot(False, "dependency_check_failed", {})`; never put exception text in the snapshot.

Replace `health.readyz()` and `health.api_readiness()` dependency gathering with the same builder and operator projection. Do not call `collect_readiness_snapshot()` from public liveness.

- [ ] **Step 6: Register public, internal, legacy, and operator routes with the exact guards**

In `main.py`, replace public health with a fixed JSON response and add the internal projection:

```python
_NO_STORE_HEADERS = {"Cache-Control": "no-store"}


async def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok"}, headers=_NO_STORE_HEADERS)


async def internal_readiness_check(request: Request) -> JSONResponse:
    if not is_loopback_peer(request):
        return JSONResponse({"detail": "Not Found"}, status_code=404, headers=_NO_STORE_HEADERS)
    snapshot = await readiness_service.collect_readiness_snapshot(request.app)
    return JSONResponse(
        readiness_service.internal_readiness_payload(snapshot),
        status_code=200 if snapshot.ready else 503,
        headers=_NO_STORE_HEADERS,
    )
```

Register `GET` and `HEAD` for `/health`; register `GET` and `HEAD` for `/internal/ready` with `include_in_schema=False`; register `/ready`, `/health/ready`, `/metrics`, and direct `/api/v1/metrics` with `dependencies=[Depends(RequirePermission(SYSTEM_LOGS))]`. Remove `openapi_extra={"security": []}` from protected legacy aliases.

Define both endpoint routers with the diagnostics dependency:

```python
router = APIRouter(
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
```

Keep `dependencies=[Depends(RequireRole("admin"))]` on metrics reset so it requires both claims-layer access and the mutating admin role. Add a lightweight dependency that sets `Cache-Control: no-store` on detailed health/metrics responses that currently return plain dictionaries; preserve the existing no-cache/no-store Prometheus headers.

- [ ] **Step 7: Update legacy tests to authenticate or use public liveness intentionally**

Run:

```bash
rg -n '(/ready|/health/ready|/api/v1/health|/api/v1/readyz|/metrics)' tldw_Server_API/tests \
  --glob '*.py'
```

For tests that only wait for server availability, change the probe to `/health`. For tests that verify readiness detail, health detail, security posture, or metrics, retain the route and provide an existing admin header or a principal with `system.logs`. Update `test_main_readiness_shutdown.py` to authenticate `/ready` and add a separate loopback `/internal/ready` draining assertion. Update `test_health_policy_snapshot.py` to assert exact minimal public liveness; keep resource-governor detail assertions on authenticated API-v1 health/readiness.

- [ ] **Step 8: Run focused control-plane verification**

Run:

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Health \
  tldw_Server_API/tests/AuthNZ_Unit/test_health_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_metrics_permissions_claims.py \
  tldw_Server_API/tests/Monitoring/test_metrics_surface_contracts.py \
  tldw_Server_API/tests/Services/test_main_readiness_shutdown.py \
  tldw_Server_API/tests/Services/test_drain_gate_middleware.py \
  tldw_Server_API/tests/Resource_Governance/test_health_policy_snapshot.py \
  tldw_Server_API/tests/Resource_Governance/test_health_policy_snapshot_api_v1.py -q
```

Expected: pass; no anonymous detailed route returns success, both permission and admin bypass work, and reset remains admin-only.

- [ ] **Step 9: Commit the control-plane boundary**

```bash
git add tldw_Server_API/app/services/readiness_service.py \
  tldw_Server_API/app/main.py \
  tldw_Server_API/app/api/v1/endpoints/health.py \
  tldw_Server_API/app/api/v1/endpoints/metrics.py \
  tldw_Server_API/tests/Health \
  tldw_Server_API/tests/AuthNZ_Unit/test_health_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_metrics_permissions_claims.py \
  tldw_Server_API/tests/Monitoring/test_metrics_surface_contracts.py \
  tldw_Server_API/tests/Services/test_main_readiness_shutdown.py \
  tldw_Server_API/tests/Services/test_drain_gate_middleware.py \
  tldw_Server_API/tests/Resource_Governance/test_health_policy_snapshot.py \
  tldw_Server_API/tests/Resource_Governance/test_health_policy_snapshot_api_v1.py
git commit -m "security: harden operational health surfaces"
```

---

### Task 2: Add the Standalone Production Compose and Caddy Boundary

**Files:**
- Create: `Dockerfiles/docker-compose.production.yml`
- Create: `Dockerfiles/Production/Caddyfile`
- Create: `Dockerfiles/production.env.example`
- Create: `tldw_Server_API/tests/Utils/test_docker_production_reference.py`

**Interfaces:**
- Consumes: Task 1 route matrix and `Helper_Scripts/Deployment/production_preflight.py` as the target-image one-shot command created in Task 3.
- Produces: Compose project `tldw-production` with services `preflight`, `caddy`, `app`, `postgres`, and `redis`.
- Produces: networks `edge` and `backend`, volumes `app-data`, `postgres_data`, `redis_data`, `caddy_data`, and `caddy_config`.
- Produces wrapper-only input `TLDW_ENV_FILE`, supplied by the CLI/Make invocation and intentionally absent from the raw file it names.
- Produces raw-file inputs: `TLDW_PUBLIC_DOMAIN`, `TLDW_ACME_EMAIL`, `ALLOWED_ORIGINS`, `JWT_SECRET_KEY`, `SESSION_ENCRYPTION_KEY`, `POSTGRES_USER`, `POSTGRES_DB`, `POSTGRES_PASSWORD`, `DATABASE_URL`, `REDIS_PASSWORD`, `REDIS_URL`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `ADMIN_EMAIL`, `TLDW_EXISTING_INSTALLATION`, `TLDW_SETUP_COMPLETED`, `TLDW_EDGE_SUBNET`, `TLDW_BACKEND_SUBNET`, `TLDW_APP_IMAGE`, `TLDW_ROLLBACK_IMAGE`, `CADDY_IMAGE`, `POSTGRES_IMAGE`, `REDIS_IMAGE`, and `TLDW_BACKUP_DIR`.

- [ ] **Step 1: Write the red static topology and environment-contract tests**

Create YAML/text tests with these exact assertions:

```python
def test_only_caddy_publishes_ports() -> None:
    services = _compose()["services"]
    assert services["caddy"]["ports"] == ["80:80", "443:443"]
    for name in ("preflight", "app", "postgres", "redis"):
        assert "ports" not in services[name]


def test_network_membership_and_backend_isolation_are_exact() -> None:
    compose = _compose()
    assert compose["networks"]["backend"]["internal"] is True
    assert set(compose["services"]["caddy"]["networks"]) == {"edge"}
    assert set(compose["services"]["app"]["networks"]) == {"edge", "backend"}
    assert set(compose["services"]["postgres"]["networks"]) == {"backend"}
    assert set(compose["services"]["redis"]["networks"]) == {"backend"}


def test_names_only_env_example_has_no_values() -> None:
    assignments = [
        line for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert assignments
    assert all(line.endswith("=") for line in assignments)
```

Also assert `AUTH_MODE=multi_user`, `tldw_production=true`, `TLDW_SETUP_ALLOW_REMOTE=0`, all five trust surfaces derive from `${TLDW_EDGE_SUBNET:?Set private TLDW_EDGE_SUBNET}`, Redis requires a password and healthcheck authentication, PostgreSQL has no default password, app images use required substitutions, third-party images use required substitutions, preflight has `network_mode: none`, app depends on preflight `service_completed_successfully`, no service mounts `/var/run/docker.sock`, and the app healthcheck calls `/internal/ready`.

- [ ] **Step 2: Write the red Caddy behavior tests**

Require deny matchers to occur before the catch-all reverse proxy and require exact forwarding-header overwrite directives:

```python
DENIED_PATHS = (
    "/internal/ready",
    "/ready",
    "/health/ready",
    "/api/v1/healthz",
    "/api/v1/readyz",
    "/setup",
    "/setup/*",
    "/api/v1/setup",
    "/api/v1/setup/*",
)


def test_caddy_denies_private_legacy_and_setup_routes_before_proxy() -> None:
    text = CADDYFILE.read_text(encoding="utf-8")
    assert text.index("respond @private_control 404") < text.index("reverse_proxy app:8000")
    for path in DENIED_PATHS:
        assert path in text


def test_caddy_overwrites_client_identity_headers() -> None:
    text = CADDYFILE.read_text(encoding="utf-8")
    assert "header_up X-Forwarded-For {remote_host}" in text
    assert "header_up X-Real-IP {remote_host}" in text
    assert "header_up X-Forwarded-Proto https" in text
```

Assert `/health`, `/metrics`, and `/api/v1/health` are not in the deny matcher: Caddy allows them through and the application makes the public/auth decision.

- [ ] **Step 3: Run the new tests and confirm missing-asset failures**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Utils/test_docker_production_reference.py -q
```

Expected: fail because the standalone Compose, Caddyfile, and env example do not exist.

- [ ] **Step 4: Create the standalone Compose contract**

Implement these security-critical shapes; include ordinary restart, healthcheck, volume, and dependency details around them:

```yaml
name: tldw-production

x-tldw-env-file: &tldw-env-file
  path: ${TLDW_ENV_FILE:?Set TLDW_ENV_FILE to the validated absolute raw env path}
  required: true
  format: raw

services:
  preflight:
    image: ${TLDW_APP_IMAGE:?Set immutable TLDW_APP_IMAGE}
    network_mode: none
    env_file: [*tldw-env-file]
    entrypoint: ["python", "/app/Helper_Scripts/Deployment/production_preflight.py"]
    command:
      - --env-file
      - /run/tldw/production.env
      - --compose-file
      - /run/tldw/docker-compose.production.yml
      - --proxy-file
      - /run/tldw/Caddyfile
      - --runtime-backup-dir
      - /backups
    volumes:
      - ${TLDW_ENV_FILE:?Set TLDW_ENV_FILE}:/run/tldw/production.env:ro
      - ./docker-compose.production.yml:/run/tldw/docker-compose.production.yml:ro
      - ./Production/Caddyfile:/run/tldw/Caddyfile:ro
      - ${TLDW_BACKUP_DIR:?Set absolute TLDW_BACKUP_DIR}:/backups
    restart: "no"

  caddy:
    image: ${CADDY_IMAGE:?Set exact CADDY_IMAGE version or digest}
    ports: ["80:80", "443:443"]
    networks: [edge]
    depends_on:
      app: {condition: service_healthy}

  app:
    image: ${TLDW_APP_IMAGE:?Set immutable TLDW_APP_IMAGE}
    expose: ["8000"]
    env_file: [*tldw-env-file]
    environment:
      AUTH_MODE: multi_user
      tldw_production: "true"
      TLDW_SETUP_ALLOW_REMOTE: "0"
      FORWARDED_ALLOW_IPS: ${TLDW_EDGE_SUBNET:?Set private TLDW_EDGE_SUBNET}
      AUTH_TRUST_X_FORWARDED_FOR: "true"
      AUTH_TRUSTED_PROXY_IPS: ${TLDW_EDGE_SUBNET:?Set private TLDW_EDGE_SUBNET}
      TLDW_TRUSTED_PROXIES: ${TLDW_EDGE_SUBNET:?Set private TLDW_EDGE_SUBNET}
      RG_CLIENT_IP_HEADER: X-Forwarded-For
      RG_TRUSTED_PROXIES: ${TLDW_EDGE_SUBNET:?Set private TLDW_EDGE_SUBNET}
      MCP_TRUST_X_FORWARDED: "true"
      MCP_TRUSTED_PROXY_IPS: ${TLDW_EDGE_SUBNET:?Set private TLDW_EDGE_SUBNET}
    networks: [edge, backend]
    depends_on:
      preflight: {condition: service_completed_successfully}
      postgres: {condition: service_healthy}
      redis: {condition: service_healthy}

  postgres:
    image: ${POSTGRES_IMAGE:?Set exact POSTGRES_IMAGE version or digest}
    env_file: [*tldw-env-file]
    networks: [backend]

  redis:
    image: ${REDIS_IMAGE:?Set exact REDIS_IMAGE version or digest}
    env_file: [*tldw-env-file]
    command:
      - /bin/sh
      - -ec
      - >-
        umask 077;
        printf 'appendonly yes\nrequirepass %s\n' "$$REDIS_PASSWORD" > /tmp/tldw-redis.conf;
        exec redis-server /tmp/tldw-redis.conf
    healthcheck:
      test: ["CMD-SHELL", "REDISCLI_AUTH=\"$$REDIS_PASSWORD\" redis-cli ping"]
    networks: [backend]

networks:
  edge:
    ipam:
      config: [{subnet: ${TLDW_EDGE_SUBNET:?Set private TLDW_EDGE_SUBNET}}]
  backend:
    internal: true
    ipam:
      config: [{subnet: ${TLDW_BACKEND_SUBNET:?Set private TLDW_BACKEND_SUBNET}}]
```

Caddy receives `TLDW_PUBLIC_DOMAIN` and `TLDW_ACME_EMAIL` through explicit environment entries and mounts `./Production/Caddyfile` read-only. The Redis password is inserted into a mode-private in-container config with `printf '%s'`, so it is not interpolated into the rendered Compose command or stored in Docker's configured argv. Persist the five named volumes. Do not add `build`, development defaults, `container_name`, or host bindings for app/data services.

- [ ] **Step 5: Create the production Caddyfile**

Use explicit denial before proxy and explicit header overwrite:

```caddyfile
{$TLDW_PUBLIC_DOMAIN} {
  encode zstd gzip

  @private_control path /internal/ready /ready /health/ready /api/v1/healthz /api/v1/readyz /setup /setup/* /api/v1/setup /api/v1/setup/*
  respond @private_control 404

  reverse_proxy app:8000 {
    header_up X-Forwarded-For {remote_host}
    header_up X-Real-IP {remote_host}
    header_up X-Forwarded-Proto https
    transport http {
      read_timeout 3600s
      write_timeout 3600s
      dial_timeout 60s
    }
  }

  header {
    X-Content-Type-Options nosniff
    X-Frame-Options DENY
    Referrer-Policy no-referrer
  }

  tls {$TLDW_ACME_EMAIL}
}
```

- [ ] **Step 6: Create the names-only environment contract**

Create comments explaining generation and permissions, then include only empty assignments:

```dotenv
TLDW_PUBLIC_DOMAIN=
TLDW_ACME_EMAIL=
ALLOWED_ORIGINS=
JWT_SECRET_KEY=
SESSION_ENCRYPTION_KEY=
POSTGRES_USER=
POSTGRES_DB=
POSTGRES_PASSWORD=
DATABASE_URL=
REDIS_PASSWORD=
REDIS_URL=
ADMIN_USERNAME=
ADMIN_PASSWORD=
ADMIN_EMAIL=
TLDW_EXISTING_INSTALLATION=
TLDW_SETUP_COMPLETED=
TLDW_EDGE_SUBNET=
TLDW_BACKEND_SUBNET=
TLDW_APP_IMAGE=
TLDW_ROLLBACK_IMAGE=
CADDY_IMAGE=
POSTGRES_IMAGE=
REDIS_IMAGE=
TLDW_BACKUP_DIR=
```

State in comments that the file must be copied outside version control, set to owner read/write, filled with independently generated values, and will fail preflight unchanged.

- [ ] **Step 7: Run and commit the topology tests**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Utils/test_docker_production_reference.py -q
git add Dockerfiles/docker-compose.production.yml \
  Dockerfiles/Production/Caddyfile \
  Dockerfiles/production.env.example \
  tldw_Server_API/tests/Utils/test_docker_production_reference.py
git commit -m "feat: add standalone production topology"
```

Expected: tests pass and the commit contains no populated env assignment.

---

### Task 3: Implement the Offline Static and Semantic Preflight

**Files:**
- Create: `Helper_Scripts/Deployment/production_preflight.py`
- Create: `tldw_Server_API/tests/Utils/test_production_preflight.py`

**Interfaces:**
- Consumes: Task 2 raw env names, Compose template, Caddyfile, and optional container-mounted runtime backup directory.
- Produces: `PreflightIssue(code: str, field: str, message: str)`.
- Produces: `PreflightReport(issues: tuple[PreflightIssue, ...])` with `ok: bool`.
- Produces: `load_raw_env(path: Path) -> dict[str, str]`.
- Produces: `validate_environment(values: Mapping[str, str], *, env_path: Path, runtime_backup_dir: Path | None = None) -> tuple[PreflightIssue, ...]`.
- Produces: `validate_compose(document: Mapping[str, Any]) -> tuple[PreflightIssue, ...]`.
- Produces: `validate_rendered_compose(document: Mapping[str, Any], values: Mapping[str, str]) -> tuple[PreflightIssue, ...]` for the host deployer's captured `docker compose config --format json` output.
- Produces: `validate_proxy(text: str) -> tuple[PreflightIssue, ...]`.
- Produces: `run_preflight(env_file: Path, compose_file: Path, proxy_file: Path, *, runtime_backup_dir: Path | None = None) -> PreflightReport`.
- Produces: `main(argv: Sequence[str] | None = None) -> int`, returning zero only for an empty issue tuple.

- [ ] **Step 1: Write red parser, aggregation, and redaction tests**

Cover LF/CRLF, quoted literal values, `#` inside values, duplicate names, malformed lines, `export KEY=`, shell substitution text, unreadable files, and deterministic ordering. The parser must not expand `$NAME`, backticks, command substitutions, escapes, or inline shell syntax.

```python
def test_report_aggregates_without_secret_values(tmp_path: Path) -> None:
    secret = "super-secret-value-that-must-not-leak"
    values = _valid_env(tmp_path)
    values["JWT_SECRET_KEY"] = "short"
    values["POSTGRES_PASSWORD"] = secret
    values["DATABASE_URL"] = "postgresql://app:different@postgres/app"
    issues = validate_environment(values, env_path=tmp_path / "production.env")
    rendered = "\n".join(f"{item.code}:{item.field}:{item.message}" for item in issues)
    assert "weak_secret:JWT_SECRET_KEY" in rendered
    assert "credential_mismatch:DATABASE_URL" in rendered
    assert secret not in rendered
    assert "postgresql://" not in rendered
```

- [ ] **Step 2: Write red environment validation cases for every invariant**

Build `_valid_env(tmp_path)` with synthetic non-production secrets, two non-overlapping RFC1918 networks, matching encoded database/Redis URLs, distinct immutable app references, exact third-party tags, completed setup, and a writable backup directory. Parameterize these stable codes:

```python
ENV_CASES = (
    ("JWT_SECRET_KEY", "short", "weak_secret"),
    ("POSTGRES_PASSWORD", "change-me", "placeholder_secret"),
    ("REDIS_PASSWORD", "redis", "placeholder_secret"),
    ("ALLOWED_ORIGINS", "*", "unsafe_origin"),
    ("TLDW_EDGE_SUBNET", "0.0.0.0/0", "unsafe_network"),
    ("TLDW_BACKEND_SUBNET", "172.30.0.0/24", "overlapping_network"),
    ("TLDW_SETUP_COMPLETED", "false", "setup_incomplete"),
    ("TLDW_APP_IMAGE", "registry/tldw:latest", "mutable_image"),
    ("TLDW_ROLLBACK_IMAGE", "registry/tldw:prod", "mutable_image"),
    ("CADDY_IMAGE", "caddy:2", "inexact_third_party_image"),
)
```

Add explicit cases for identical secrets, identical target/rollback image, PostgreSQL user/database/password mismatches after URL decoding, Redis password/host mismatch after URL decoding, sample domain/contact, origin/domain mismatch, non-absolute backup path, backup equal to a live-data path, missing directory, non-writable directory, wildcard proxy trust in the rendered Compose contract, missing bootstrap fields when `TLDW_EXISTING_INSTALLATION=false`, and populated bootstrap password when `TLDW_EXISTING_INSTALLATION=true`.

- [ ] **Step 3: Write red Compose and Caddy mutation tests**

Deep-copy the real YAML and mutate one invariant per case: add app/PostgreSQL/Redis ports, add a non-80/443 Caddy port, remove `internal: true`, overlap network membership, attach Caddy to backend, attach PostgreSQL to edge, remove the one-shot preflight condition, add Docker-socket volume, restore a production fallback, remove Redis auth, change any trust value away from the edge subnet, or use a populated image default. Mutate the Caddy text to remove each deny path, put denial after proxy, preserve incoming XFF, omit TLS, or proxy a stateful service. Require stable `topology_*` or `proxy_*` codes. Add a rendered-Compose fixture with concrete networks/images and assert `validate_rendered_compose` rejects resolved secret text in service `command`, final published ports outside Caddy 80/443, unexpected network attachments, incorrect image resolution, or a missing `internal` backend.

- [ ] **Step 4: Run the preflight tests and confirm missing-module failure**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Utils/test_production_preflight.py -q
```

Expected: import failure because `production_preflight.py` does not exist.

- [ ] **Step 5: Implement raw parsing and typed reports**

Use the standard library and PyYAML only:

```python
@dataclass(frozen=True, order=True)
class PreflightIssue:
    code: str
    field: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    issues: tuple[PreflightIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues
```

`load_raw_env` accepts blank/comment lines and a single `NAME=value` split, strips one matching pair of outer single or double quotes, rejects duplicate/invalid names and `export`, and returns literal text without expansion. Parser failures become `env_parse` issues through `run_preflight`; they do not escape as tracebacks.

- [ ] **Step 6: Implement secret, URL, image, origin, setup, network, and backup validation**

Use `urllib.parse.urlsplit()` plus `urllib.parse.unquote()` for credential comparisons. Use `ipaddress.ip_network(..., strict=True)` and require IPv4, `is_private`, no default route, and no overlap. Accept immutable app images only when they match one of:

```python
APP_IMAGE_PATTERNS = (
    re.compile(r"^.+@sha256:[0-9a-f]{64}$"),
    re.compile(r"^.+:sha-[0-9a-f]{7,64}$"),
)
```

Accept third-party images when they use a digest or a tag containing a full numeric version such as `2.10.2-alpine`; reject bare, `latest`, and major-only tags. Validate `TLDW_BACKUP_DIR` against the host path, or test the mounted `runtime_backup_dir` for existence/type/write access when the one-shot container supplies it. Never include the candidate value in an issue message.

- [ ] **Step 7: Implement static Compose and proxy validation**

Require the exact services, networks, memberships, port ownership, env substitutions, trust alignment, preflight dependency, no Docker socket, Redis/PostgreSQL authentication, and immutable-variable references from Task 2. `validate_rendered_compose` rechecks the resolved model against the parsed env values and rejects a rendered service command containing `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `DATABASE_URL`, or `REDIS_URL`. Proxy validation checks literal deny paths, order, 404 behavior, TLS, only `app:8000` upstream, and all three overwrite directives. Do not attempt to execute Caddy or Docker.

- [ ] **Step 8: Implement the CLI and one-shot service contract**

Support these arguments:

```text
--env-file PATH              required raw operator env file
--compose-file PATH          default Dockerfiles/docker-compose.production.yml
--proxy-file PATH            default Dockerfiles/Production/Caddyfile
--runtime-backup-dir PATH    optional container-mounted view of TLDW_BACKUP_DIR
```

Print success to stdout as `Production preflight passed.`. Print each failure to stderr as `ERROR [code] field: message`, sorted by `(code, field, message)`, and return 1. Return 2 for argument parsing. Log no environment mapping.

- [ ] **Step 9: Run tests, side-effect scan, and Bandit**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Utils/test_production_preflight.py \
  tldw_Server_API/tests/Utils/test_docker_production_reference.py -q
rg -n 'subprocess|requests|httpx|urlopen|docker|os\.system|shell=True' \
  Helper_Scripts/Deployment/production_preflight.py
../../.venv/bin/python -m bandit -q \
  Helper_Scripts/Deployment/production_preflight.py
```

Expected: tests and Bandit pass; the side-effect scan returns no executable process/network calls (documentation strings describing Docker are acceptable only when visibly non-executable).

- [ ] **Step 10: Commit the static gate**

```bash
git add Helper_Scripts/Deployment/production_preflight.py \
  tldw_Server_API/tests/Utils/test_production_preflight.py
git commit -m "feat: add fail-closed production preflight"
```

---

### Task 4: Add Verified Backup, Deployment, and Restore-Backed Rollback

**Files:**
- Create: `Helper_Scripts/Deployment/production_artifacts.py`
- Create: `Helper_Scripts/Deployment/production_deploy.py`
- Create: `tldw_Server_API/tests/Utils/test_production_deploy.py`
- Modify: `Makefile:1-210`
- Modify: `tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py`

**Interfaces:**
- Consumes: `run_preflight(env_file, compose_file, proxy_file) -> PreflightReport`, Compose project `tldw-production`, raw operator env, and the immutable image/backup values validated in Task 3.
- Produces: `ArtifactRecord(kind: str, path: str, sha256: str, size_bytes: int)`.
- Produces: `DeploymentManifest(created_at: str, target_image: str, rollback_image: str, compose_file_sha256: str, artifacts: tuple[ArtifactRecord, ...])`.
- Produces: `DeploymentConfig(env_file: Path, compose_file: Path, proxy_file: Path, backup_dir: Path, values: Mapping[str, str])`.
- Produces: `sha256_file(path: Path) -> str`, `verify_tar_archive(path: Path) -> tuple[str, ...]`, `write_manifest(path: Path, manifest: DeploymentManifest) -> None`, and `load_verified_manifest(path: Path) -> DeploymentManifest`.
- Produces: `CommandResult(returncode: int, stdout: bytes, stderr: bytes)` and injectable `CommandRunner = Callable[[Sequence[str], Mapping[str, str] | None, bytes | None], CommandResult]`.
- Produces: `deploy(config: DeploymentConfig, *, runner: CommandRunner) -> DeploymentManifest`.
- Produces: `rollback(config: DeploymentConfig, manifest_path: Path, *, runner: CommandRunner) -> None`.
- Produces CLI subcommands `deploy` and `rollback --manifest PATH --restore-artifacts`.

- [ ] **Step 1: Write red artifact and manifest tests**

Create deterministic fixtures and assert:

```python
def test_manifest_contains_checksums_but_no_secrets(tmp_path: Path) -> None:
    artifact = tmp_path / "postgres.dump"
    artifact.write_bytes(b"custom-dump-fixture")
    record = ArtifactRecord(
        kind="postgresql",
        path=artifact.name,
        sha256=sha256_file(artifact),
        size_bytes=artifact.stat().st_size,
    )
    manifest = DeploymentManifest(
        created_at="2026-08-30T00:00:00Z",
        target_image="registry/tldw:sha-1234567",
        rollback_image="registry/tldw:sha-7654321",
        compose_file_sha256="a" * 64,
        artifacts=(record,),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    text = path.read_text(encoding="utf-8")
    assert "password" not in text.lower()
    assert "database_url" not in text.lower()
    assert load_verified_manifest(path) == manifest
```

Add tests that reject path traversal in manifest artifact paths, checksum mismatch, zero-byte artifacts, malformed JSON, unknown artifact kinds, duplicate kinds, tar members with absolute/parent paths, unreadable tar, and an app-data tar without a readable regular-file member.

- [ ] **Step 2: Write red deploy command-order and fail-stop tests**

Use a fake runner that records argv tuples and returns fixture bytes. Assert the successful order is:

1. static `run_preflight` passes;
2. `docker compose --env-file ENV -f COMPOSE config --format json` renders, parses in memory, and passes `validate_rendered_compose` without printing the secret-bearing model;
3. target and rollback images are pulled;
4. both images run a network-disabled Python import smoke command;
5. PostgreSQL and Redis services become healthy without starting app/Caddy;
6. app is stopped before app-data/coordination backup;
7. PostgreSQL custom dump is captured and `pg_restore --list` succeeds;
8. Redis `SAVE`, copy-out, and `redis-check-rdb` succeed;
9. app-data volume archive is created and locally verified;
10. checksummed non-secret manifest is written;
11. only then does `docker compose --env-file ENV -f COMPOSE up -d --remove-orphans` start the target profile.

Parameterize a failure at each gate and assert no later command, especially the final `up`, is recorded. Assert stdout/stderr containing raw connection URLs or known secret fixtures are replaced by exception types and gate labels in user-visible errors.

- [ ] **Step 3: Write red restore-backed rollback tests**

Require `--restore-artifacts`; no implicit binary-only rollback is allowed. Assert rollback verifies every checksum, stops app/Caddy, restores PostgreSQL from the manifest dump, safely replaces app-data contents from the verified tar, restores the Redis RDB while Redis is stopped, selects `manifest.rollback_image` as `TLDW_APP_IMAGE`, reruns static preflight, and starts the prior profile. A missing artifact, checksum mismatch, failed restore command, or target/rollback mismatch prevents final startup.

- [ ] **Step 4: Run the deploy tests and confirm missing-module failures**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Utils/test_production_deploy.py \
  tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py -q
```

Expected: import/test failures because the artifact/deploy modules and Make targets do not exist.

- [ ] **Step 5: Implement safe artifact primitives**

Use `hashlib`, `json`, `tarfile`, and `pathlib`. Serialize the manifest with sorted keys and mode `0o600`; store only timestamps, image references, Compose checksum, relative artifact names, sizes, and SHA-256 digests. `verify_tar_archive` lists members, rejects unsafe names and links, opens one nonempty regular member to prove readability, and never extracts during verification.

- [ ] **Step 6: Implement the fixed-argv runner and deployment gates**

Call `subprocess.run` with `shell=False`, `check=False`, explicit argv, byte input/output, and a minimal inherited environment plus the parsed Compose variables. Never put `DATABASE_URL`, `REDIS_URL`, or passwords in argv. Use these command forms:

```text
docker compose --env-file ENV -f COMPOSE config --format json
docker pull TARGET_IMAGE
docker pull ROLLBACK_IMAGE
docker run --rm --network none --entrypoint python IMAGE -c IMPORT_SMOKE
docker compose --env-file ENV -f COMPOSE up -d postgres redis
docker compose --env-file ENV -f COMPOSE stop app caddy
docker compose --env-file ENV -f COMPOSE exec -T postgres pg_dump --format=custom --no-owner --username POSTGRES_USER --dbname POSTGRES_DB
docker run --rm --network none --entrypoint pg_restore -v BACKUP_DIR:/backup:ro POSTGRES_IMAGE --list /backup/postgres.dump
docker compose --env-file ENV -f COMPOSE exec -T redis /bin/sh -ec 'REDISCLI_AUTH="$REDIS_PASSWORD" exec redis-cli SAVE'
docker compose --env-file ENV -f COMPOSE cp redis:/data/dump.rdb BACKUP_DIR/redis.rdb
docker run --rm --network none --entrypoint redis-check-rdb -v BACKUP_DIR:/backup:ro REDIS_IMAGE /backup/redis.rdb
docker run --rm --network none --entrypoint python -v tldw-production_app-data:/data:ro -v BACKUP_DIR:/backup TARGET_IMAGE -c TAR_SCRIPT
docker compose --env-file ENV -f COMPOSE up -d --remove-orphans
```

Parse the rendered JSON in memory, pass it to `validate_rendered_compose`, discard it after validation, and never print or persist it because service environment data may contain secrets. Capture the binary PostgreSQL dump from stdout directly into a mode-`0o600` file. The app-data `TAR_SCRIPT` uses Python `tarfile` without a shell. Do not mount the Docker socket in any helper container.

- [ ] **Step 7: Implement explicit restore-backed rollback**

After manifest/checksum verification, use fixed argv plus binary stdin for `pg_restore --clean --if-exists --no-owner`. Stop Redis before copying the verified RDB into `tldw-production_redis_data`; replace app-data only through a Python helper container after `verify_tar_archive` succeeds. Set the subprocess environment override `TLDW_APP_IMAGE=manifest.rollback_image` for the final Compose render/start. Record rollback completion in a new non-secret manifest rather than modifying the source manifest.

- [ ] **Step 8: Add canonical Make targets**

Add variables and targets without changing quickstart defaults:

```make
PRODUCTION_COMPOSE ?= Dockerfiles/docker-compose.production.yml
PRODUCTION_ENV_FILE ?=
PRODUCTION_MANIFEST ?=

production-preflight:
	@test -n "$(PRODUCTION_ENV_FILE)" || (echo "Set PRODUCTION_ENV_FILE to an absolute raw env path" >&2; exit 2)
	$(PYTHON) Helper_Scripts/Deployment/production_preflight.py --env-file "$(PRODUCTION_ENV_FILE)" --compose-file "$(PRODUCTION_COMPOSE)"

production-deploy:
	@test -n "$(PRODUCTION_ENV_FILE)" || (echo "Set PRODUCTION_ENV_FILE to an absolute raw env path" >&2; exit 2)
	$(PYTHON) Helper_Scripts/Deployment/production_deploy.py deploy --env-file "$(PRODUCTION_ENV_FILE)" --compose-file "$(PRODUCTION_COMPOSE)"

production-rollback:
	@test -n "$(PRODUCTION_ENV_FILE)" || (echo "Set PRODUCTION_ENV_FILE to an absolute raw env path" >&2; exit 2)
	@test -n "$(PRODUCTION_MANIFEST)" || (echo "Set PRODUCTION_MANIFEST to a verified pre-upgrade manifest" >&2; exit 2)
	$(PYTHON) Helper_Scripts/Deployment/production_deploy.py rollback --restore-artifacts --env-file "$(PRODUCTION_ENV_FILE)" --compose-file "$(PRODUCTION_COMPOSE)" --manifest "$(PRODUCTION_MANIFEST)"
```

Add all three to `.PHONY` and extend the Makefile contract test to prove no `docker compose up` appears in `production-preflight`.

- [ ] **Step 9: Run deploy tests and Bandit**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Utils/test_production_deploy.py \
  tldw_Server_API/tests/Utils/test_production_preflight.py \
  tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py -q
../../.venv/bin/python -m bandit -q \
  Helper_Scripts/Deployment/production_artifacts.py \
  Helper_Scripts/Deployment/production_deploy.py \
  Helper_Scripts/Deployment/production_preflight.py
```

Expected: all pass with no shell-injection, unsafe archive-extraction, secret-log, or subprocess findings.

- [ ] **Step 10: Commit the operational gate**

```bash
git add Helper_Scripts/Deployment/production_artifacts.py \
  Helper_Scripts/Deployment/production_deploy.py \
  Makefile \
  tldw_Server_API/tests/Utils/test_production_deploy.py \
  tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py
git commit -m "feat: verify production backup and rollback artifacts"
```

---

### Task 5: Migrate Probes, Authenticate Monitoring, Publish the Runbook, and Verify

**Files:**
- Modify: `Dockerfiles/Dockerfile.prod`
- Modify every API container healthcheck returned by the Step 1 scan, including: `Dockerfiles/docker-compose.yml`, `Dockerfiles/docker-compose.single-user.yml`, `Dockerfiles/docker-compose.multi-user-postgres.yml`, and `Dockerfiles/docker-compose.host-storage.yml`
- Modify: `Helper_Scripts/Samples/Kubernetes/tldw-app-deployment.yaml`
- Modify: `Helper_Scripts/Samples/Kubernetes/README.md`
- Modify: `Dockerfiles/Monitoring/prometheus.yml`
- Modify: `Dockerfiles/Monitoring/docker-compose.monitoring.yml`
- Create: `Docs/Deployment/Production_Reference_Deployment.md`
- Modify: `Docs/Deployment/First_Time_Production_Setup.md`
- Modify: `Docs/Deployment/Long_Term_Admin_Guide.md`
- Modify: `Docs/Deployment/Reverse_Proxy_Examples.md`
- Modify: `Dockerfiles/README.md`
- Generate through refresh: `Docs/Published/**`
- Create: `tldw_Server_API/tests/Utils/test_production_probe_contract.py`
- Create: `tldw_Server_API/tests/Docs/test_production_reference_deployment_docs.py`
- Modify: `tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py`
- Modify through Backlog MCP: `backlog/tasks/task-13013.6 - Ship-a-production-safe-reference-deployment-and-health-surface.md`

**Interfaces:**
- Consumes: `/internal/ready`, protected `/metrics`, the Compose/preflight/deploy CLIs, and deployment manifests from Tasks 1-4.
- Produces: all container-local readiness probes against `/internal/ready`.
- Produces: Prometheus bearer credential file `TLDW_METRICS_API_KEY_FILE`, whose existing AuthNZ API-key principal has `system.logs`.
- Produces: source runbook plus deterministic generated published mirror.
- Produces: complete verification evidence and review-ready PR; merge remains gated on the requester-authored Change summary.

- [ ] **Step 1: Write red probe and monitoring contract tests**

First enumerate every checked-in API readiness probe:

```bash
rg -n "localhost:8000/ready|path: /ready|/api/v1/metrics/text" \
  Dockerfiles Helper_Scripts/Samples/Kubernetes --glob '*'
```

Create `test_production_probe_contract.py` asserting no application container healthcheck calls `/ready`, production and development Docker healthchecks call `http://localhost:8000/internal/ready`, and Kubernetes readiness uses an exec command containing `/internal/ready` rather than `httpGet`. Assert liveness remains HTTP `/health`.

Assert Prometheus uses a protected metrics path and reads a credential from a mounted file:

```python
def test_prometheus_uses_scoped_bearer_credential_file() -> None:
    config = yaml.safe_load(Path("Dockerfiles/Monitoring/prometheus.yml").read_text())
    job = next(item for item in config["scrape_configs"] if item["job_name"] == "tldw_server")
    assert job["metrics_path"] == "/api/v1/metrics/text"
    assert job["authorization"] == {
        "type": "Bearer",
        "credentials_file": "/run/secrets/tldw_metrics_api_key",
    }
```

Require the monitoring Compose service to mount `${TLDW_METRICS_API_KEY_FILE:?Set TLDW_METRICS_API_KEY_FILE}` read-only at that exact path. No key value may appear in either tracked file.

- [ ] **Step 2: Write red documentation contract tests**

Require the new source runbook to contain executable commands and these markers:

```python
RUNBOOK_MARKERS = (
    "make production-preflight",
    "make production-deploy",
    "make production-rollback",
    "chmod 600",
    "pg_restore --list",
    "redis-check-rdb",
    "system.logs",
    "TLDW_METRICS_API_KEY_FILE",
    "disposable restore drill",
    "TASK-13013.7",
    "TASK-13013.9",
    "TASK-13144",
)
```

Assert it documents: only Caddy publishes; public `/health` exact body; public denial of internal/legacy/setup aliases; authenticated health/metrics; raw-env creation without echoing secrets; immutable current/rollback images; PostgreSQL/Redis/app-data artifacts; operator-managed secret backup; upgrade ordering; restore-backed rollback; the limits of archive inspection; and environment-only skips. Assert first-time setup, long-term admin, reverse-proxy examples, and Docker README link to it and label legacy overlays non-production.

- [ ] **Step 3: Run the probe/docs tests and confirm intended failures**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Utils/test_production_probe_contract.py \
  tldw_Server_API/tests/Docs/test_production_reference_deployment_docs.py \
  tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py -q
```

Expected: fail on `/ready` probes, missing monitoring credential configuration, and missing runbook.

- [ ] **Step 4: Migrate Docker and Kubernetes probes**

Replace each API-image/container healthcheck URL with `http://localhost:8000/internal/ready`. In Kubernetes use an in-container exec probe so the network peer is loopback:

```yaml
readinessProbe:
  exec:
    command:
      - python
      - -c
      - >-
        import sys, urllib.request;
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/internal/ready', timeout=3).status == 200 else 1)
  initialDelaySeconds: 10
  periodSeconds: 10
```

Keep Kubernetes liveness on `/health`. Update sample README language to state that detailed readiness is not remotely anonymous.

- [ ] **Step 5: Authenticate Prometheus with an existing scoped principal**

Keep `metrics_path: /api/v1/metrics/text`, add:

```yaml
authorization:
  type: Bearer
  credentials_file: /run/secrets/tldw_metrics_api_key
```

Mount the operator-created file in the Prometheus service:

```yaml
volumes:
  - ${TLDW_METRICS_API_KEY_FILE:?Set a mode-0600 API-key file whose principal has system.logs}:/run/secrets/tldw_metrics_api_key:ro
```

Document creating a constrained existing API key/principal with only `system.logs`; do not introduce a new token type or auth system. Keep monitoring traffic on the private Compose network and do not publish application metrics anonymously through Caddy.

- [ ] **Step 6: Write the source operator runbook and correct legacy guidance**

Document this exact order:

1. copy the names-only env file to an absolute untracked path, fill it through the operator's secret tooling, and `chmod 600`;
2. create a separate absolute backup directory and independently back up the secret/config file;
3. set immutable target and rollback image references plus exact third-party versions;
4. set non-overlapping private edge/backend CIDRs and aligned trust inputs;
5. complete bootstrap through `ADMIN_*` or attest an existing completed installation, with remote setup disabled;
6. run `make production-preflight PRODUCTION_ENV_FILE=/absolute/path` and stop on any issue;
7. perform the documented disposable restore drill before first rollout or material migration;
8. run `make production-deploy PRODUCTION_ENV_FILE=/absolute/path/to/production.env`, which verifies both images and all three backup classes before target startup;
9. verify public `/health` exact body, public 404 denial for private/legacy/setup paths, and authenticated operator diagnostics/metrics;
10. upgrade only through a fresh verified manifest;
11. on uncertain migration compatibility, stop writes and use `make production-rollback PRODUCTION_ENV_FILE=/absolute/path/to/production.env PRODUCTION_MANIFEST=/absolute/path/to/manifest.json`, which restores matching artifacts before the prior image;
12. record the limitations assigned to TASK-13013.7, TASK-13013.9, and TASK-13144.

Correct proxy sample paths to `Helper_Scripts/Samples/Caddy/Caddyfile.compose` and `Helper_Scripts/Samples/Nginx/nginx.conf`, and label proxy overlay examples custom/non-production rather than a substitute for the standalone reference.

- [ ] **Step 7: Refresh generated public documentation**

```bash
bash Helper_Scripts/refresh_docs_published.sh
cmp Docs/Deployment/Production_Reference_Deployment.md \
  Docs/Published/Deployment/Production_Reference_Deployment.md
```

Expected: refresh succeeds and the source runbook matches its generated published mirror. Inspect the full generated diff; commit only deterministic changes resulting from the source docs.

- [ ] **Step 8: Run focused probe, docs, control-plane, deployment, and security verification**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Utils/test_production_probe_contract.py \
  tldw_Server_API/tests/Docs/test_production_reference_deployment_docs.py \
  tldw_Server_API/tests/Utils/test_docker_production_reference.py \
  tldw_Server_API/tests/Utils/test_production_preflight.py \
  tldw_Server_API/tests/Utils/test_production_deploy.py \
  tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py \
  tldw_Server_API/tests/Health \
  tldw_Server_API/tests/AuthNZ_Unit/test_health_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_metrics_permissions_claims.py \
  tldw_Server_API/tests/Monitoring \
  tldw_Server_API/tests/Services/test_main_readiness_shutdown.py -q
../../.venv/bin/python -m bandit -q \
  Helper_Scripts/Deployment/production_preflight.py \
  Helper_Scripts/Deployment/production_artifacts.py \
  Helper_Scripts/Deployment/production_deploy.py \
  tldw_Server_API/app/services/readiness_service.py \
  tldw_Server_API/app/main.py \
  tldw_Server_API/app/api/v1/endpoints/health.py \
  tldw_Server_API/app/api/v1/endpoints/metrics.py
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 9: Run broader affected tests and OpenAPI verification**

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/AuthNZ_Unit \
  tldw_Server_API/tests/Config/test_route_and_cors_guards.py \
  tldw_Server_API/tests/Services/test_main_router_contract.py \
  tldw_Server_API/tests/Services/test_router_groups_contract.py \
  tldw_Server_API/tests/Services/test_main_lifecycle_contract.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  tldw_Server_API/tests/Resource_Governance -q
../../.venv/bin/python Helper_Scripts/export_openapi_schema.py \
  --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

If the canonical OpenAPI check reports reviewed security-metadata drift, regenerate through the repository's existing OpenAPI target, inspect the diff, and include only the fingerprint/schema artifacts produced by that target. Do not hand-edit generated API artifacts.

- [ ] **Step 10: Render the production Compose file when Docker is already available**

Create a temporary synthetic env fixture outside the repository that satisfies preflight, then run only:

```bash
TLDW_ENV_FILE=/absolute/path/to/synthetic-production.env \
  docker compose --env-file /absolute/path/to/synthetic-production.env \
  -f Dockerfiles/docker-compose.production.yml config --quiet
```

Do not pull or start containers during repository verification. If Docker is unavailable, record the exact environment limitation; the static topology/preflight suites remain mandatory.

- [ ] **Step 11: Update Backlog through MCP and commit migration/docs evidence**

Use Backlog MCP with the isolated worktree project path. Keep TASK-13013.6 `In Progress`; set the implementation plan/documentation links, modified files, exact test/Bandit/OpenAPI/Docker-render results, known skips, and PR-readiness note. Do not manually edit the task file.

```bash
git add Dockerfiles/Dockerfile.prod \
  Dockerfiles/docker-compose.yml \
  Dockerfiles/docker-compose.single-user.yml \
  Dockerfiles/docker-compose.multi-user-postgres.yml \
  Dockerfiles/docker-compose.host-storage.yml \
  Dockerfiles/Monitoring/prometheus.yml \
  Dockerfiles/Monitoring/docker-compose.monitoring.yml \
  Dockerfiles/README.md \
  Helper_Scripts/Samples/Kubernetes/tldw-app-deployment.yaml \
  Helper_Scripts/Samples/Kubernetes/README.md \
  Docs/Deployment \
  Docs/Published \
  tldw_Server_API/tests/Utils/test_production_probe_contract.py \
  tldw_Server_API/tests/Docs/test_production_reference_deployment_docs.py \
  tldw_Server_API/tests/Utils/test_docker_public_profile_compose.py \
  'backlog/tasks/task-13013.6 - Ship-a-production-safe-reference-deployment-and-health-surface.md'
git commit -m "docs: publish production deployment runbook"
```

- [ ] **Step 12: Self-review, publish the PR, and honor the merge gate**

```bash
git status --short
git diff origin/dev...HEAD --stat
git diff origin/dev...HEAD --check
git log --oneline origin/dev..HEAD
rg -n '(change-me|changeme|TestPassword|example\.com|example\.invalid|:latest)' \
  Dockerfiles/docker-compose.production.yml \
  Dockerfiles/Production/Caddyfile \
  Dockerfiles/production.env.example \
  Docs/Deployment/Production_Reference_Deployment.md
```

The sentinel scan may find explanatory prose only; it must find no deployable assignment or image reference. Rebase on latest `origin/dev`, rerun exact-head focused gates, push with a lease, and open the PR against `dev`. Record the PR URL in Backlog. Wait for Qodo and trusted required checks, address all exact-head review findings, and do not merge until the human requester supplies the repository-required Change summary in their own words. After normal merge, verify the merge commit is reachable from `origin/dev` and finalize TASK-13013.6 through Backlog MCP.

---

## Self-Review Checklist

- [ ] **Spec coverage:** Map every design section to a task: topology and route contract (Tasks 1-2), shared readiness (Task 1), proxy/client identity and secrets (Tasks 2-3), static and operational gates (Tasks 3-4), backup/upgrade/rollback (Task 4), error/logging and compatibility (Tasks 1-5), verification and delivery boundaries (Task 5).
- [ ] **Authorization consistency:** Confirm every detailed route uses `RequirePermission(SYSTEM_LOGS)`, admin bypass is tested, and metrics reset retains `RequireRole("admin")` in addition to the permission guard.
- [ ] **Topology consistency:** Confirm Caddy/app share only `edge`, app/data services share `backend`, backend is internal, networks are private/non-overlapping, and only Caddy publishes 80/443.
- [ ] **Trust consistency:** Confirm Uvicorn, AuthNZ, Setup, Resource Governor, and MCP values all derive from the edge CIDR and wildcard trust is rejected.
- [ ] **Recovery consistency:** Confirm PostgreSQL custom dump, Redis RDB, app-data tar, checksums, manifest, disposable restore drill, and restore-backed rollback are all present; archive inspection is never described as restore proof.
- [ ] **Placeholder scan:** Run `rg -n '(T[B]D|T[O]DO|implement[[:space:]]+later|fill[[:space:]]+in[[:space:]]+details|similar[[:space:]]+to[[:space:]]+Task|add[[:space:]]+appropriate)' Docs/superpowers/plans/2026-08-30-production-safe-reference-deployment.md` and remove every plan-writing placeholder hit.
- [ ] **Type consistency:** Confirm Task 3 and Task 4 signatures, dataclass fields, CLI flags, Compose service names, env names, volume names, manifest keys, and Task 5 commands match exactly.
- [ ] **Generated-doc discipline:** Confirm all `Docs/Published` changes came from `refresh_docs_published.sh`, not manual edits.
- [ ] **Scope boundary:** Confirm TASK-13144, TASK-13013.7, and TASK-13013.9 remain explicit follow-ups rather than being silently absorbed.
