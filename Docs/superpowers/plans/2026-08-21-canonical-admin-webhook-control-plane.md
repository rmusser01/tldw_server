# Canonical Admin Webhook Control Plane And Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver upstream PR 1 of the approved outgoing-webhook design: a secure, database-portable canonical admin control plane, deterministic legacy migration, exclusive compatibility routing, and an admin UI that safely manages inactive registrations and one-time secrets.

**Architecture:** A new focused `Admin_Webhooks` package owns immutable domain types, validated configuration, contextual encryption, repository transactions, control-plane rules, key rotation, and legacy import. FastAPI exposes only the PR 1 catalog/status/CRUD/rotate contract through one selected router, while the admin UI consumes response ETags and keeps command idempotency keys and revealed secrets in memory. The data plane remains absent and activation fails closed until the later delivery PR supplies a healthy capability.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, aiosqlite/SQLite, asyncpg/PostgreSQL, repository AES-GCM JSON envelopes, Typer, pytest, React 19, Next.js 16, TypeScript, Vitest, Testing Library, Playwright, Ruff, Bandit.

**Spec:** `Docs/Design/2026-07-12-canonical-admin-outgoing-webhooks.md`

**Backlog task:** `TASK-13014`

## Global Constraints

- This plan implements only upstream PR 1, "Canonical Control Plane And Migration." No outbound HTTP, Jobs worker, automatic producer, test attempt, manual redelivery, delivery-history API, or final legacy deletion belongs here.
- Existing `services/admin_webhooks_service.py` remains reachable only from isolated compatibility routes. `core/AuthNZ/admin_webhook_secrets.py` remains only for historical migration 082 and may be imported solely by `Admin_Webhooks/legacy_import.py` behind an explicit migration CLI flag. No runtime canonical module imports either historical file; PR 3 deletes them after compatibility removal.
- `TLDW_ADMIN_WEBHOOKS_MODE` accepts exactly `off`, `migrate`, or `on` and defaults to `off`.
- `TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT` is temporary, defaults to `false`, and selects the isolated legacy webhook routes instead of canonical routes. It is valid only when canonical mode is `off`; combining it with `migrate` or `on` fails startup. An authenticated status selector is always mounted and reports the explicit selection; a process must never mount both implementations for the same method/path, and clients never infer compatibility from 404 or transport failure.
- Canonical/`off` with compatibility unset intentionally disables all historical webhook CRUD/test/delivery/incident-notify routes. Startup emits one fixed sanitized warning, status/UI show the explicit disabled state, and operator documentation calls out the upgrade break plus the only two choices: temporary explicit compatibility or reviewed migration.
- Canonical create always persists `active=false`; no create schema accepts a signing secret or wildcard event.
- Canonical activation returns `503 admin_webhook_delivery_unavailable` until a later PR injects a healthy data-plane capability. PR 1 must not claim canonical `on` is releaseable.
- The initial subscription catalog contains exactly `user.created`, `user.deleted`, `incident.created`, `incident.updated`, `incident.resolved`, and `incident.notify`. `webhook.test` is reserved and never subscribable.
- The initial payload API version is exactly `2026-07-01`.
- Signing secrets are 32 server-generated random bytes encoded as `whsec_` plus 64 lowercase hexadecimal characters.
- Full target URLs, signing secrets, exact event bytes, pending incident markers, and secret-bearing idempotency replay material use the dedicated webhook key ring. Runtime code never falls back to BYOK, session, JWT, API-key, or single-user credentials.
- `TLDW_ADMIN_WEBHOOK_KEYS_JSON` maps 1-64 character key IDs from `[A-Za-z0-9._-]` to strict base64 encodings of exactly 32 random bytes; `TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID` must name one configured key.
- Contextual envelopes validate purpose and stable row identity after decryption. Targets bind registration ID plus internal target version; secrets bind registration ID plus secret version. Cross-row, cross-version, and cross-purpose substitution must fail closed.
- Migration state records `active_primary_key_id`; every ordinary protected write checks in its transaction that the local configured primary matches it. A mismatch returns `503 admin_webhook_key_configuration_mismatch`.
- Create, rotate, test, and manual-redelivery idempotency keys use 16-255 characters from `[A-Za-z0-9._:-]`, are scoped by actor/operation/route/request, and expire after 24 hours. PR 1 implements create and rotate behavior while reserving the common model for later operations.
- Exact idempotent replay is evaluated before current-resource preconditions. Same-key/different-request returns `409 idempotency_conflict`; superseded secret replay returns `409 idempotency_result_superseded`.
- Idempotency stores a domain-separated SHA-256 lookup digest of scope plus key and an `hmac-sha256:` canonical-request fingerprint keyed by the presented raw idempotency key. It never stores a plain body hash, raw key, or canonical request.
- PATCH and DELETE require a strong current `If-Match`; missing returns `428 precondition_required`, stale returns `412 precondition_failed`, and an effective no-op PATCH changes no revision or delivery/secret version.
- Non-deleted registrations default to a limit of 100, active registrations default to 25, the active limit cannot exceed the non-deleted limit, and neither configured limit can exceed 1,000.
- Target URLs are at most 2,048 UTF-8 bytes; the request schema also caps them at 2,048 characters. Description is at most 500 characters. Timeout defaults to 10 seconds and is bounded to 1-30 seconds.
- Canonical event bodies are bounded to 64 KiB before encryption. PR 1 proves the shared protector and schema; event persistence/fanout begins in PR 2.
- Each canonical migration-state mapping or rejection JSON value is sorted compact UTF-8 and bounded to 1,048,576 bytes on both backends.
- Soft-deleted registrations are tombstones. PR 1 exposes no purge command and never hard-deletes canonical or legacy records.
- The legacy importer is dry-run first, imports all accepted records inactive, never truncates to satisfy limits, never silently merges conflicts, and is resumable after every documented crash window.
- Every imported registration preserves its legacy signing secret encrypted but sets `secret_rotation_required=true`; activation returns `409 admin_webhook_secret_rotation_required` until canonical rotation clears it.
- `system_ops.json` publication uses a same-directory temporary file, file fsync, atomic replace, and parent-directory fsync under the existing process and file locks.
- The encrypted full-file rollback backup is mode `0600`; its one-time key is mode `0600`, stored outside the application data directory, retained for 7 days by default, and never retained beyond 30 days.
- `TLDW_ADMIN_WEBHOOK_ROLLBACK_WINDOW_DAYS` is an integer from 1 through 30 and defaults to 7.
- Every effective canonical registration mutation compare-and-sets a first-canonical-activity marker in the same transaction. Replays, rejected requests, and effective no-ops do not. Backup extraction fails once that marker exists or the rollback window closes.
- Migration report, backup, rollback-key, and active data paths must be distinct after normalization. Existing report/backup/key/output parents are owned by the invoking effective UID and are not group/world writable. Backup/key creation is exclusive and no-follow; existing, symlinked, or non-regular targets fail before mutation. The redacted report uses atomic mode-`0600` publication.
- API, logs, audit, metrics, reports, and UI never expose full URL paths/queries, signing secrets after their bounded response, encrypted event contents, receiver contents, or incident narrative.
- Temporary legacy compatibility create/update audits retain only webhook ID, event count, and enabled state; they no longer write the historical full URL into new audit records. Compatibility response and dispatch behavior otherwise stays unchanged until PR 3 deletion.
- Every canonical API endpoint requires a platform-admin principal. Mutations additionally require a numeric user-backed principal and return `403 admin_webhook_user_principal_required` for service principals without one. Every mutation uses the mandatory unified-audit pattern with only actor, action, numeric webhook ID, target hostname, event type, outcome, request ID, and stable reason code. The control-plane transaction awaits an `accepted` or `no_op` audit before commit; audit unavailability rolls back and maps to `503 admin_webhook_audit_unavailable`. Because AuthNZ and unified audit are separate stores, the plan does not claim distributed atomicity or label a pre-commit event `succeeded`. Authorized catalog/status/list/get calls emit bounded best-effort access audits; read-audit failure never blocks recovery status.
- Host-side import apply/rejection, key-rotation state changes, rollback-backup extraction, and rollback-artifact retirement use a closed mandatory operational-audit adapter. It records only operator ID, fixed action, durable operation ID, generated request ID, `accepted`/`completed`/`failed` outcome, and stable reason code; no key IDs/material, source fingerprints/content, artifact paths, or free text. Audit unavailability before an invocation's first mutation aborts it; durable migration state remains the recovery authority across later audit/database/filesystem crash windows.
- SQLite and PostgreSQL must expose equivalent tables, constraints, indexes, transaction behavior, collision handling, and migration state.
- The admin UI does not persist idempotency keys or one-time secrets in local storage, session storage, cookies, URLs, logs, or analytics.
- Successful create/rotate responses and eligible secret-bearing replays set `Cache-Control: no-store` and `Pragma: no-cache`; the authenticated admin proxy preserves both.
- All implementation commits include the current Backlog task update. Touched Python passes focused pytest, Ruff, and Bandit before the PR review gate.

---

## Delivery Stages

1. Establish exact domain/config/crypto contracts and additive dual-backend schema.
2. Implement repository transactions, idempotency, control-plane lifecycle, and key rotation.
3. Implement crash-safe legacy import and rollback-key lifecycle.
4. Expose the exclusive canonical API and update the admin UI transport/workflow.
5. Complete documentation, OpenAPI review, database matrices, security gates, and human review.

## File Map

**Create**

- `tldw_Server_API/app/core/Admin_Webhooks/__init__.py` - public PR 1 exports only.
- `tldw_Server_API/app/core/Admin_Webhooks/domain.py` - enums, immutable records, stable errors, ETags, hashes, and idempotency types.
- `tldw_Server_API/app/core/Admin_Webhooks/config.py` - validated feature modes, limits, key-ring environment, and temporary route selection.
- `tldw_Server_API/app/core/Admin_Webhooks/catalog.py` - versioned six-event catalog and subscription validation.
- `tldw_Server_API/app/core/Admin_Webhooks/crypto.py` - dedicated contextual key ring and protected-byte envelopes.
- `tldw_Server_API/app/core/Admin_Webhooks/audit.py` - bounded mandatory unified-audit adapters and closed API/host-operation sink protocols.
- `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py` - all canonical webhook SQL and transaction/unit-of-work operations.
- `tldw_Server_API/app/core/Admin_Webhooks/control_plane.py` - registration, precondition, idempotency, status, and secret lifecycle rules.
- `tldw_Server_API/app/core/Admin_Webhooks/key_rotation.py` - durable resumable re-encryption scanner and verification pass.
- `tldw_Server_API/app/core/Admin_Webhooks/legacy_import.py` - dry-run plan, source fingerprints, import, readback, sanitization, non-destructive backup extraction, and backup/key destruction.
- `tldw_Server_API/app/api/v1/schemas/admin_webhooks.py` - canonical request/response schemas isolated from legacy admin schemas.
- `tldw_Server_API/cli/commands/admin_webhooks.py` - explicit dry-run/apply/rotation/backup-extraction/backup-destruction operator commands.
- `tldw_Server_API/cli/admin_webhooks_cli.py` - dedicated Typer application and process exit mapping for destructive webhook operations.
- `tldw_Server_API/tests/Admin_Webhooks/test_domain_config_catalog.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_crypto.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_audit.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_migration_sqlite.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_repository_sqlite.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_key_rotation.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_legacy_import.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_legacy_import_postgres.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_api.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_route_selection.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_openapi.py`
- `admin-ui/lib/idempotent-command.ts` - one-command in-memory retry state.
- `admin-ui/lib/__tests__/idempotent-command.test.ts`
- `admin-ui/tests/e2e/webhooks-control-plane.spec.ts`
- `Docs/Admin_Webhooks_Control_Plane.md` - API/operator behavior available in PR 1.
- `Docs/Admin_Webhooks_Migration_Runbook.md` - dry-run, import, readback, non-destructive backup extraction, structural offline recovery, rollback-key, and forward-fix procedure.
- `Docs/Admin_Webhooks_Key_Rotation_Runbook.md` - key provisioning, rotation, recovery, verification, and old-key removal.
- `Docs/Evidence/Admin_Webhooks_PR1_Verification.md` - exact backend, UI, OpenAPI, static-analysis, and security-gate evidence retained with the change.

**Modify**

- `tldw_Server_API/app/core/Security/egress.py` - expose a structured platform-webhook policy adapter that composes existing generic and webhook-specific global lists without changing tenant workflow callers.
- `tldw_Server_API/app/core/AuthNZ/migrations.py` - additive SQLite migration 094.
- `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py` - idempotent equivalent PostgreSQL ensure path.
- `tldw_Server_API/app/services/admin_system_ops_service.py` - strict bounded migration snapshots and crash-safe atomic JSON publication while preserving existing locking.
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py` - replace unmounted direct-SQL/delivery routes with the canonical PR 1 router.
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py` - move legacy webhook endpoints onto an isolated `legacy_webhooks_router`; keep incident endpoints on `router`.
- `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py` - mount exactly one webhook router from validated route selection.
- `tldw_Server_API/app/api/v1/schemas/admin_schemas.py` - remove the duplicate numeric webhook schema block after canonical imports are wired.
- `tldw_Server_API/tests/Admin/test_admin_system_ops_service.py` - prove strict reads and crash-safe atomic publication without changing permissive recovery callers.
- `tldw_Server_API/tests/Admin/test_admin_webhooks_service.py` - retire obsolete numeric-router assertions while retaining compatibility dispatcher/signing coverage until PR 3.
- `tldw_Server_API/tests/Admin/test_admin_webhooks_schemas.py` - replace obsolete numeric canonical-schema assertions while retaining explicit compatibility coverage.
- `tldw_Server_API/tests/Security/test_egress.py` - prove platform-webhook allow/deny composition and unchanged tenant helper behavior.
- `pyproject.toml` - register the dedicated `tldw-admin-webhooks` entry point without mixing these operations into `tldw-evals`.
- `admin-ui/lib/http.ts` - expose status/headers with parsed JSON without changing existing callers.
- `admin-ui/lib/server-auth.ts` - forward `If-Match` and `Idempotency-Key` through the authenticated proxy.
- `admin-ui/app/api/proxy/__tests__/route.test.ts` - prove conditional/idempotency forwarding and ETag preservation.
- `admin-ui/lib/api-client.ts` - canonical catalog/status/CRUD/rotate methods with ETag metadata.
- `admin-ui/types/webhooks.ts` - one canonical numeric-ID type family and no secret on ordinary registration types.
- `admin-ui/types/index.ts` - export only the canonical webhook types.
- `admin-ui/app/webhooks/page.tsx` - catalog-driven inactive create, ETag mutations, rotation, degraded states, and one-time secret handling.
- `admin-ui/app/webhooks/__tests__/page.test.tsx` - replace stale legacy fixtures and test the complete PR 1 workflow.
- `admin-ui/docs/feature-guides/webhooks.md` - document only available control-plane behavior and later delivery dependency.
- `apps/tldw-frontend/lib/api/openapi.fingerprint.json` - record the reviewed canonical PR 1 API delta.
- `backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md` - retain execution notes, verification evidence, and review state with each implementation unit.

### Task 0: Rebase, Attach The Plan, And Prove The Baseline

**Files:**
- Modify: `backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md`

**Interfaces:**
- Consumes: approved design at `Docs/Design/2026-07-12-canonical-admin-outgoing-webhooks.md`.
- Produces: a clean child worktree based on current `origin/dev`, with `TASK-13014` In Progress and this exact plan linked.

- [ ] **Step 1: Create an isolated implementation worktree from current `origin/dev`**

```bash
git fetch origin dev
git worktree add .worktrees/admin-webhooks-control-plane -b codex/admin-webhooks-control-plane origin/dev
cd .worktrees/admin-webhooks-control-plane
git log -1 --format='%H %s'
git status --short
rg -n "migration_[0-9]{3}|Migration\([0-9]+" tldw_Server_API/app/core/AuthNZ/migrations.py | tail -n 20
```

Expected: the first command output identifies current `origin/dev`; status is empty; migration 090 remains the highest registered SQLite migration, leaving 091 free. If the approved design commit is not yet reachable from `origin/dev`, stop and merge its review PR before implementation. If another change has claimed 091, stop and update this plan, both backend tests, and `TASK-13014` to the next free version before writing DDL; never reuse or renumber an already-merged migration.

- [ ] **Step 2: Attach execution notes to the existing task**

```bash
backlog task edit 13014 -s "In Progress" --plan $'1. Define canonical contracts and dedicated encryption.\n2. Add equivalent SQLite/PostgreSQL persistence.\n3. Implement repository, lifecycle, rotation, and importer.\n4. Expose exclusive API routing and update the admin UI.\n5. Run migration, security, UI, and OpenAPI gates.\nDetailed plan: Docs/superpowers/plans/2026-08-21-canonical-admin-webhook-control-plane.md'
backlog task 13014 --plain
```

Expected: `TASK-13014` is In Progress and links the approved design and this plan.

- [ ] **Step 3: Run the pre-change regression baseline**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin/test_admin_webhooks_schemas.py \
  tldw_Server_API/tests/Admin/test_admin_webhooks_service.py \
  tldw_Server_API/tests/Admin/test_admin_ops_webhooks_reports.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_admin_webhook_migration_sqlite.py \
  tldw_Server_API/tests/Security/test_egress.py \
  tldw_Server_API/tests/Workflows/test_webhook_admin_endpoints.py
cd admin-ui
bun run test -- app/webhooks/__tests__/page.test.tsx app/api/proxy/__tests__/route.test.ts
cd ..
```

Expected: record exact pass/fail counts and every pre-existing failure in `TASK-13014`; do not silently attribute baseline failures to later changes.

### Task 1: Freeze Domain, Configuration, Catalog, And Revision Contracts

**Files:**
- Create: `tldw_Server_API/app/core/Admin_Webhooks/__init__.py`
- Create: `tldw_Server_API/app/core/Admin_Webhooks/domain.py`
- Create: `tldw_Server_API/app/core/Admin_Webhooks/config.py`
- Create: `tldw_Server_API/app/core/Admin_Webhooks/catalog.py`
- Modify: `tldw_Server_API/app/core/Security/egress.py`
- Modify: `tldw_Server_API/tests/Security/test_egress.py`
- Test: `tldw_Server_API/tests/Admin_Webhooks/test_domain_config_catalog.py`

**Interfaces:**
- Consumes: an explicit environment mapping for canonical settings, plus the existing process-environment, DNS-resolution, port, private-address, and profile behavior already owned by `Security/egress.py` for the focused platform-webhook policy adapter.
- Produces: `evaluate_platform_webhook_url_policy()`, `AdminWebhookMode`, `WebhookRouteSelection`, `AdminWebhookSettings`, `WebhookRegistration`, `WebhookStatus`, `WebhookErrorCode`, `WebhookError`, `IdempotencyScope`, `IdempotencyClaim`, `EVENT_API_VERSION`, `EVENT_CATALOG`, `build_idempotency_scope()`, `build_registration_etag()`, `parse_registration_etag()`, `normalize_request_id()`, `idempotency_lookup_digest()`, `canonical_request_hash()`, `validate_webhook_target()`, and `redact_target()`.

- [ ] **Step 1: Write failing configuration, catalog, and pure-domain tests**

```python
def test_settings_default_off_and_validate_bounds() -> None:
    settings = AdminWebhookSettings.from_environment({})
    assert settings.mode is AdminWebhookMode.OFF
    assert settings.route_selection is WebhookRouteSelection.CANONICAL
    assert settings.registration_limit == 100
    assert settings.active_limit == 25

def test_catalog_is_explicit_and_rejects_wildcard() -> None:
    assert tuple(item.event_type for item in EVENT_CATALOG) == (
        "user.created", "user.deleted", "incident.created",
        "incident.updated", "incident.resolved", "incident.notify",
    )
    with pytest.raises(WebhookError, match="admin_webhook_event_unsupported"):
        validate_subscriptions(["*"])

def test_etag_is_strong_and_round_trips() -> None:
    value = build_registration_etag(webhook_id=41, revision=7)
    assert value == '"admin-webhook-41-r7"'
    assert parse_registration_etag(value, expected_webhook_id=41) == 7
```

Also cover invalid mode, invalid booleans, rejection of `legacy_compat=true` with mode `migrate` or `on`, zero/negative limits, active greater than non-deleted, either limit greater than 1,000, rollback-window values outside 1-30, timeout outside 1-30, URL over 2,048 UTF-8 bytes, URL control characters/backslashes/user-info/fragments/missing host/malformed IDNA or port, HTTPS default, explicit HTTP only in validated non-production mode, production HTTP-override startup refusal, central policy delegation with `sensitive_observability=True`, generic plus webhook-specific global allowlist union, all-denylist union/precedence, no policy value in logs, and unchanged `is_webhook_url_allowed_for_tenant()` behavior. Also cover duplicate subscriptions, reserved `webhook.test`, empty subscriptions, catalog-order normalization, reordered-set equality, weak/malformed idempotency keys, sanitized request IDs with generated fallback, deterministic domain-separated HMAC request fingerprints, changed-key/body/scope separation, absence of URL/query bytes from stored inputs, URL redaction, and no path/query in `target_display`.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_domain_config_catalog.py \
  tldw_Server_API/tests/Security/test_egress.py
```

Expected: FAIL during collection because the `Admin_Webhooks` package and the
platform-webhook policy adapter do not exist.

- [ ] **Step 3: Implement the exact public types and pure helpers**

```python
class AdminWebhookMode(str, Enum):
    OFF = "off"
    MIGRATE = "migrate"
    ON = "on"

class WebhookRouteSelection(str, Enum):
    CANONICAL = "canonical"
    LEGACY = "legacy"

@dataclass(frozen=True)
class AdminWebhookSettings:
    mode: AdminWebhookMode
    route_selection: WebhookRouteSelection
    registration_limit: int
    active_limit: int
    allow_http_dev: bool
    idempotency_ttl_seconds: int
    rollback_window_days: int

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> "AdminWebhookSettings":
        mode = AdminWebhookMode(environ.get("TLDW_ADMIN_WEBHOOKS_MODE", "off").strip().lower())
        legacy = parse_strict_bool(environ.get("TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT", "false"))
        registration_limit = parse_bounded_positive_int(environ, "TLDW_ADMIN_WEBHOOK_REGISTRATION_LIMIT", 100, 1000)
        active_limit = parse_bounded_positive_int(environ, "TLDW_ADMIN_WEBHOOK_ACTIVE_LIMIT", 25, 1000)
        if active_limit > registration_limit:
            raise ValueError("TLDW_ADMIN_WEBHOOK_ACTIVE_LIMIT cannot exceed registration limit")
        allow_http_dev = parse_strict_bool(environ.get("TLDW_ADMIN_WEBHOOKS_ALLOW_HTTP_DEV", "false"))
        if allow_http_dev and is_production_environment_mapping(environ):
            raise ValueError("Webhook HTTP development override is forbidden in production")
        if legacy and mode is not AdminWebhookMode.OFF:
            raise ValueError("Legacy webhook compatibility requires canonical mode off")
        rollback_window_days = parse_bounded_positive_int(
            environ,
            "TLDW_ADMIN_WEBHOOK_ROLLBACK_WINDOW_DAYS",
            7,
            30,
        )
        return cls(mode, WebhookRouteSelection.LEGACY if legacy else WebhookRouteSelection.CANONICAL, registration_limit, active_limit, allow_http_dev, 86400, rollback_window_days)
```

Define frozen dataclasses for public registration metadata, status, limits, migration summary, and idempotency outcomes. Define `WebhookErrorCode` as the closed set of stable codes named by this plan and one `WebhookError(code: WebhookErrorCode, http_status: int)` hierarchy instead of passing database/crypto exceptions to routes. Include `admin_webhook_user_principal_required` and `admin_webhook_audit_unavailable`. `is_production_environment_mapping(environ)` is the pure-mapping equivalent of `core.config.is_production_environment()`: it recognizes truthy `tldw_production` and `prod|production` in `ENV`/`APP_ENV`/`TLDW_ENV`/`ENVIRONMENT`; parity tests prevent the safety rules from diverging. `normalize_request_id(value, *, generator=uuid.uuid4) -> str` accepts only 1-128 characters from `[A-Za-z0-9._:-]` and otherwise returns a generated UUID string; routes pass the middleware-owned `request.state.request_id`, not a raw header. `canonical_request_hash(idempotency_key, *, scope, body, conditional_version)` must serialize version, normalized route/resource, body, actor, operation, and conditional version with sorted compact JSON, then return `hmac-sha256:` plus HMAC-SHA256 keyed by the validated raw idempotency key. `idempotency_lookup_digest(idempotency_key, scope)` returns a separate domain-separated SHA-256 digest. Compare stored and presented lookup/fingerprint values with `hmac.compare_digest()`.

In `Security/egress.py`, add the exact public adapter below after the existing list parsers. It composes both generic global families (`EGRESS_*` and `WORKFLOWS_EGRESS_*`) with the matching `WORKFLOWS_WEBHOOK_*` family, preserves deny precedence by passing both explicit unions to the existing evaluator, and forces destination-safe observability. Do not alter `is_webhook_url_allowed_for_tenant()`.

```python
def evaluate_platform_webhook_url_policy(url: str) -> URLPolicyResult:
    allowlist = list(dict.fromkeys([
        *_get_allowlist(os.getenv(GLOBAL_ALLOWLIST_ENV, "")),
        *_get_allowlist(os.getenv(ALLOWLIST_ENV, "")),
        *_parse_list_env(os.getenv(WEBHOOK_ALLOWLIST_ENV)),
    ]))
    denylist = list(dict.fromkeys([
        *_get_allowlist(os.getenv(GLOBAL_DENYLIST_ENV, "")),
        *_get_allowlist(os.getenv(DENYLIST_ENV, "")),
        *_parse_list_env(os.getenv(WEBHOOK_DENYLIST_ENV)),
    ]))
    return evaluate_url_policy(
        url,
        allowlist=allowlist,
        denylist=denylist,
        sensitive_observability=True,
    )
```

`validate_webhook_target()` performs the stricter syntax/HTTPS checks above, bounds the UTF-8 encoding to 2,048 bytes, then calls that adapter; it returns a normalized host/display plus the exact validated URL for encryption and never logs policy input/reason text. `redact_target()` must return only normalized scheme and IDNA hostname, plus an allowed non-default port.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_domain_config_catalog.py \
  tldw_Server_API/tests/Security/test_egress.py
git add tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/core/Security/egress.py \
  tldw_Server_API/tests/Admin_Webhooks/test_domain_config_catalog.py \
  tldw_Server_API/tests/Security/test_egress.py \
  "backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md"
git commit -m "feat(webhooks): define canonical control-plane contracts"
```

### Task 2: Add A Dedicated Contextual Encryption Key Ring

**Files:**
- Create: `tldw_Server_API/app/core/Admin_Webhooks/crypto.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/config.py`
- Test: `tldw_Server_API/tests/Admin_Webhooks/test_crypto.py`

**Interfaces:**
- Consumes: `encrypt_json_blob_with_key()` and `decrypt_json_blob_with_key()` from `tldw_Server_API.app.core.Security.crypto`.
- Produces: `ProtectedValue(ciphertext_json: str, key_id: str)`, strict `WebhookKeyRing.from_environment()`, `WebhookKeyRingLoadResult`, non-throwing runtime `load_webhook_key_ring()`, `encrypt_bytes()`, `decrypt_bytes()`, `encrypt_text()`, `decrypt_text()`, `can_decrypt()`, and rotation-only `reencrypt_to_key()`.

- [ ] **Step 1: Write failing dedicated-key and substitution tests**

```python
def test_context_prevents_cross_row_substitution(key_ring: WebhookKeyRing) -> None:
    protected = key_ring.encrypt_text(
        purpose="registration.secret",
        identity={"registration_id": 7, "secret_version": 1},
        plaintext="whsec_" + "a" * 64,
    )
    with pytest.raises(WebhookKeyError, match="admin_webhook_envelope_context_mismatch"):
        key_ring.decrypt_text(
            purpose="registration.secret",
            identity={"registration_id": 8, "secret_version": 1},
            protected=protected,
        )

def test_runtime_key_ring_ignores_unrelated_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", "unrelated")
    monkeypatch.delenv("TLDW_ADMIN_WEBHOOK_KEYS_JSON", raising=False)
    with pytest.raises(WebhookKeyError, match="admin_webhook_key_unavailable"):
        WebhookKeyRing.from_environment(os.environ)
```

Cover malformed JSON, non-object top-level values including an array of apparent key/value pairs, duplicate/blank/overlength/invalid IDs, duplicate IDs detected before JSON object construction, missing primary, primary absent from ring, invalid base64, decoded keys shorter or longer than 32 bytes, previous-key reads, primary-only writes, UTF-8 and arbitrary byte round trips, exact event-body acceptance at 65,536 bytes, rejection at 65,537 bytes, event ID/API-version substitution, domain-separated migration HMAC fingerprints, copied envelope purpose/version mismatch, unknown key ID, tampering, redacted exceptions, absence of every unrelated credential fallback, and `load_webhook_key_ring()` returning a closed code with no raw value/exception text for every invalid environment while strict `from_environment()` continues to raise.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_crypto.py
```

Expected: FAIL because `WebhookKeyRing` is not implemented.

- [ ] **Step 3: Implement the contextual envelope**

```python
@dataclass(frozen=True)
class ProtectedValue:
    ciphertext_json: str
    key_id: str

class _JSONObjectPairs(list[tuple[str, object]]):
    """Distinguish a JSON object from a top-level array before validation."""

class WebhookKeyRing:
    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> "WebhookKeyRing":
        raw_pairs = json.loads(
            environ.get("TLDW_ADMIN_WEBHOOK_KEYS_JSON", "{}"),
            object_pairs_hook=_JSONObjectPairs,
        )
        if not isinstance(raw_pairs, _JSONObjectPairs):
            raise WebhookKeyError("admin_webhook_key_configuration_invalid")
        primary_id = environ.get("TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID", "").strip()
        return cls(validate_key_pairs(raw_pairs), primary_id=primary_id)

    def encrypt_bytes(self, *, purpose: str, identity: Mapping[str, str | int], plaintext: bytes) -> ProtectedValue:
        payload = {
            "schema": 1,
            "purpose": purpose,
            "identity": normalize_identity(identity),
            "value_b64": base64.b64encode(plaintext).decode("ascii"),
        }
        envelope = encrypt_json_blob_with_key(payload, self._keys[self.primary_id])
        if envelope is None:
            raise WebhookKeyError("admin_webhook_encryption_failed")
        return ProtectedValue(json.dumps(envelope, sort_keys=True, separators=(",", ":")), self.primary_id)
```

`decrypt_bytes()` must select only the envelope's declared configured key, decrypt, compare exact schema/purpose/normalized identity, decode with strict base64 validation, and return bytes. It must raise stable, secret-free `WebhookKeyError` values. `reencrypt_to_key(protected, *, purpose, identity, target_key_id)` validates/decrypts the existing value and encrypts explicitly to a configured target without changing the ordinary-write primary; only `key_rotation.py` may call it. This module never imports the historical webhook-secret helper or reads unrelated credentials.

`WebhookKeyRingLoadResult` contains exactly `ring: WebhookKeyRing | None` and
a closed redacted availability/configuration code. `load_webhook_key_ring()`
catches only expected configuration/key errors from strict loading and never
retains raw environment text or an exception object. The API dependency passes
this result to status/control-plane composition so default `off` works with no
keys; importer/rotation CLI composition calls `require_ring()` before any audit
or mutation and exits with the stable code when unavailable.

Add `encrypt_event_body(event_id: str, api_version: str, body: bytes) -> ProtectedValue` and the matching decrypt helper. They enforce the 64 KiB limit before encryption and bind both event ID and API version into the envelope identity; PR 2 will consume these helpers rather than inventing a second event protector.

Add `fingerprint_migration_source(domain: str, canonical_bytes: bytes) -> tuple[str, str]`, returning `(primary_key_id, "hmac-sha256:" + hex_digest)`. It decodes the current 32-byte key internally and authenticates `b"tldw-admin-webhook-migration-v1\x00" + domain.encode("ascii") + b"\x00" + canonical_bytes`; `domain` must be one of the importer constants rather than caller-controlled free text. It never exposes raw key bytes. Plain SHA-256 is used only for ciphertext backup bytes and canonical redacted report bytes.

- [ ] **Step 4: Run GREEN, scan forbidden fallbacks, and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_crypto.py
if rg -n "BYOK_ENCRYPTION_KEY|SESSION_ENCRYPTION_KEY|JWT_SECRET_KEY|SINGLE_USER_API_KEY|API_KEY|admin_webhook_secrets" \
  tldw_Server_API/app/core/Admin_Webhooks \
  --glob '!legacy_import.py'; then
  printf 'runtime credential fallback found\n' >&2
  exit 1
fi
git add tldw_Server_API/app/core/Admin_Webhooks/crypto.py \
  tldw_Server_API/app/core/Admin_Webhooks/config.py \
  tldw_Server_API/tests/Admin_Webhooks/test_crypto.py \
  "backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md"
git commit -m "feat(webhooks): add dedicated contextual encryption"
```

Expected: tests PASS and the scan prints no match.

### Task 3: Add Equivalent Additive SQLite And PostgreSQL Schema

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_migration_sqlite.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py`

**Interfaces:**
- Consumes: SQLite migration registry at version 90 and PostgreSQL idempotent ensure helpers.
- Produces: SQLite migration 094 and `ensure_admin_webhook_canonical_tables_pg(conn)` with equivalent canonical tables, constraints, partial indexes, and migration-state seed.

- [ ] **Step 1: Write failing schema-contract tests before DDL**

```python
CANONICAL_TABLES = {
    "admin_webhook_sequences",
    "admin_webhook_registrations",
    "admin_webhook_events",
    "admin_webhook_deliveries",
    "admin_webhook_delivery_attempts",
    "admin_webhook_idempotency",
    "admin_webhook_migration_state",
}

def test_sqlite_091_is_additive_and_preserves_legacy_tables(legacy_082_db: Path) -> None:
    apply_authnz_migrations(legacy_082_db)
    with sqlite3.connect(legacy_082_db) as conn:
        names = table_names(conn)
        assert CANONICAL_TABLES <= names
        assert {"admin_webhooks", "admin_webhooks_delivery_log"} <= names
        assert current_schema_version(conn) == 91
```

SQLite tests cover fresh install and upgrades from pre-080, 080, and 082; exact columns; check constraints; foreign keys; partial unique source indexes; automatic-delivery uniqueness; idempotency scope uniqueness; sequence seed; migration-state singleton and revision; fingerprint key ID; paired nullable first-canonical-activity fields and closed kinds; canonical mapping/rejection JSON acceptance at 1,048,576 UTF-8 bytes and rejection at 1,048,577; mutually exclusive import/rotation state; rerun; and rollback of an injected DDL failure. PostgreSQL tests inspect the same logical schema on a disposable fixture, rerun the ensure helper, and prove representative legacy rows remain untouched.

- [ ] **Step 2: Run RED on SQLite and PostgreSQL fixtures**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_sqlite.py
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py
```

Expected: SQLite FAIL because migration 094 is absent. PostgreSQL either FAIL against the configured disposable fixture or SKIP with the repository's explicit no-test-database reason; never point it at staging or production.

- [ ] **Step 3: Implement migration 094 and the PostgreSQL parity helper**

The registration table stores numeric ID, description, encrypted URL/secret plus key IDs, `target_hostname`, `target_display`, canonical event JSON, `active`, timeout, target/delivery/secret versions, `secret_rotation_required`, revision, creator/updater, timestamps, and tombstone metadata. `target_version` starts at 1 and increments only when URL ciphertext changes, so disable and other non-URL updates never invalidate its envelope identity. `secret_rotation_required` is non-null and defaults false for canonical creates. The sequence row starts at 1 and is updated in the same transaction as insertion; IDs are allocated and inserted atomically and are never reused.

The migration-state singleton stores `state_revision`; import `phase` exactly `migration_pending`/`artifacts_pending`/`artifacts_ready`/`database_committed`/`complete`; import operation/operator/timestamps; `fingerprint_key_id`; durable `active_primary_key_id`; system-ops webhook-subtree and legacy-table fingerprints; canonical source mapping JSON; redacted report digest; protected backup ciphertext digest; source-fingerprint-bound rejection decisions JSON; and completion time. Private fields retain normalized active report/backup/key and staging identities, artifact owner/group/mode/file-identity evidence needed for resume, rollback window/expiry, `rollback_retirement_phase` exactly `not_applicable`/`retained`/`rollback_retirement_in_progress`/`retired`, retirement operator/start/completion times, and expected ciphertext digest; no API mapper exposes those paths. `first_canonical_activity_at` is nullable and paired with `first_canonical_activity_kind` constrained to `registration_mutation`/`event_capture`/`delivery_attempt`; it stores no resource ID or content. The same row stores the mutually exclusive key-rotation operation ID, source/target key IDs, `rotation_phase` exactly `rewriting`/`verifying`/`awaiting_primary_cutover`/`complete`, table/key cursor, processed/verified counts, and start/completion times. A non-complete import forbids an active rotation; `rewriting`, `verifying`, or `awaiting_primary_cutover` forbids import approval/apply. Constrain phases, nullable-field combinations, non-negative counts, digest/fingerprint formats, the exact 1,048,576-byte UTF-8 limit for each canonical JSON field, and valid field combinations equivalently on both backends. Use `length(CAST(value AS BLOB))` on SQLite and `octet_length(value)` on PostgreSQL so the bound is bytes rather than characters; repository compare-and-set uses `state_revision` rather than last-write-wins updates.

Create the remaining canonical tables exactly as defined by the design even though PR 1 only writes registration, idempotency, and migration-state rows. Include encrypted event-body columns now so key rotation and later data-plane work do not require a divergent schema. Use bounded text/check constraints and backend-equivalent partial indexes rather than nullable unique shortcuts.

```python
def migration_094_create_canonical_admin_webhook_tables(conn: sqlite3.Connection) -> None:
    for statement in CANONICAL_ADMIN_WEBHOOK_SQLITE_DDL:
        conn.execute(statement)
    conn.execute(
        "INSERT OR IGNORE INTO admin_webhook_sequences (name, next_value) VALUES (?, ?)",
        ("registration", 1),
    )
    conn.execute(
        "INSERT OR IGNORE INTO admin_webhook_migration_state (singleton_id, schema_version, phase) VALUES (?, ?, ?)",
        (1, 1, "migration_pending"),
    )
```

Define `CANONICAL_ADMIN_WEBHOOK_SQLITE_DDL` as an ordered tuple of one complete
statement per element. Do not use `executescript()`: it can implicitly commit
outside the `MigrationManager` transaction and make the injected-failure gate
lie. Add migration 094 to the ordered SQLite registry. Wire
`ensure_admin_webhook_canonical_tables_pg()` into the same PostgreSQL
startup/ensure path used by neighboring AuthNZ tables. Do not reinterpret
migrations 080/082 as canonical and do not drop/sanitize legacy data in DDL.
The three existing migration-082 `sqlite3.Row.keys()` membership checks are
intentional because direct `in row` tests values; add targeted `# noqa: SIM118`
annotations rather than applying Ruff's semantics-changing suggestion.

- [ ] **Step 4: Run migration matrices and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_admin_webhook_migration_sqlite.py
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py
git add tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py \
  "backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md"
git commit -m "feat(webhooks): add canonical dual-backend schema"
```

Expected: all configured matrices PASS; PostgreSQL absence is recorded as an environment skip and must be resolved in CI before merge.

### Task 4: Implement Repository-Owned Transactions And Idempotency

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_repository_sqlite.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py`

**Interfaces:**
- Consumes: `DatabasePool.transaction()`, canonical domain records from Task 1, and migration 094 from Task 3.
- Produces: `AdminWebhookRepository`, `AdminWebhookUnitOfWork`, `RegistrationInsert`, `RegistrationPatch`, `IdempotencyLookup`, `MigrationState`, `mark_first_canonical_activity(kind, at)`, and backend-neutral transaction methods used by control plane, rotation, and import.

- [ ] **Step 1: Write failing SQLite repository tests**

```python
async def test_create_commits_before_connection_close(sqlite_repo: AdminWebhookRepository) -> None:
    async with sqlite_repo.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        await tx.insert_registration(registration_insert(webhook_id))
    reopened = await open_repository(sqlite_repo.database_path)
    assert (await reopened.get_registration(webhook_id)).id == webhook_id

async def test_conditional_patch_is_noop_when_values_match(sqlite_repo: AdminWebhookRepository) -> None:
    original = await seed_registration(sqlite_repo, revision=4, delivery_config_version=2)
    async with sqlite_repo.transaction() as tx:
        result = await tx.patch_registration(
            original.id,
            expected_revision=4,
            patch=RegistrationPatch(description=original.description),
        )
    assert result.changed is False
    assert result.registration.revision == 4
    assert result.registration.delivery_config_version == 2
```

Cover numeric ID allocation under concurrent transactions, insertion/read/list pagination with deterministic `id DESC` ordering, event-set round trip, soft delete, stale revision, target/config/secret version increments, unchanged target version on event/timeout/active updates, description-only behavior, active and non-deleted counts, over-limit read state, tombstone exclusion, 30-day purge eligibility with dependent-delivery/unexpired-idempotency/migration-reference blockers, idempotency new/replay/conflict/in-progress/expiry, route scoping, HMAC request-fingerprint comparison without webhook-key decryption, absence of raw key/canonical request/URL query bytes in direct database inspection, replay-before-precondition access, superseded-resource metadata, migration-state compare-and-set, first-canonical-activity compare-and-set preserving the earliest timestamp/kind, key-rotation cursor persistence, and transaction rollback after injected exceptions.

- [ ] **Step 2: Run SQLite RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_sqlite.py
```

Expected: FAIL because the repository and transaction types do not exist.

- [ ] **Step 3: Implement the backend-neutral repository contract**

```python
class AdminWebhookRepository:
    def __init__(self, pool: DatabasePool) -> None:
        self._pool = pool

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["AdminWebhookUnitOfWork"]:
        async with self._pool.transaction() as connection:
            yield AdminWebhookUnitOfWork(connection, is_postgres=self._pool.pool is not None)

    async def get_registration(self, webhook_id: int, *, include_deleted: bool = False) -> WebhookRegistration | None:
        async with self._pool.acquire() as connection:
            return await read_registration(connection, webhook_id, include_deleted=include_deleted)

class AdminWebhookUnitOfWork:
    async def allocate_registration_id(self) -> int:
        row = await self._fetchrow(
            "UPDATE admin_webhook_sequences "
            "SET next_value = next_value + 1 WHERE name = ? "
            "RETURNING next_value - 1 AS allocated_id",
            ("registration",),
        )
        if row is None:
            raise WebhookRepositoryError("admin_webhook_sequence_unavailable")
        return int(row["allocated_id"])
```

Implement `claim_idempotency() -> IdempotencyLookup`, `complete_idempotency()`, `insert_registration()`, `patch_registration()`, `soft_delete_registration()`, `mark_first_canonical_activity(kind, at)`, and `find_purge_eligible_registration_ids()` with backend-adapted SQL local to this module. The activity marker uses one conditional update that writes only when both activity fields are null, validates the closed kind, and returns the retained earliest value under races. `claim_idempotency()` receives only the lookup digest and HMAC request fingerprint, performs lookup, fingerprint comparison, expiry handling, and insert in one transaction; a unique-scope race is reread and returned as replay/conflict/in-progress rather than leaking a database exception. Use the transaction connection for every write and write-followed-by-read. Normalize `?` parameters through the private `_fetchrow`, `_fetch`, and `_execute` adapters in this file; do not branch in services. Keep raw idempotency keys, canonical requests, plaintext URL/secret values out of repository records and SQL parameters. Complete idempotency, its protected replay result, the resource mutation, and the first-activity marker share one transaction.

- [ ] **Step 4: Turn SQLite tests GREEN and prove write paths do not use pool shortcuts**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_sqlite.py
rg -n "pool\.(execute|fetchone|fetchall|fetchval)" \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py
```

Expected: tests PASS and the scan finds no write performed outside `DatabasePool.transaction()`.

- [ ] **Step 5: Write and run the PostgreSQL parity matrix**

```python
@pytest.mark.postgres
async def test_postgres_idempotency_claim_has_one_winner(pg_repo: AdminWebhookRepository) -> None:
    outcomes = await asyncio.gather(
        claim_create(pg_repo, key="0123456789abcdef"),
        claim_create(pg_repo, key="0123456789abcdef"),
    )
    assert sorted(outcome.kind for outcome in outcomes) == ["new", "replay"]
```

Repeat the SQLite behavior matrix for durable commit, revision CAS, no-op patch, sequence allocation, idempotency races, soft delete, limits, migration-state/activity-marker CAS, and rollback. Create/rotate keep claim and completion in one transaction, so a concurrent identical request waits for the winner and then observes replay; a concurrent same-key/different-request observes conflict. It must never observe an uncommitted row or create a second resource. A bounded backend lock/statement timeout maps to `503 admin_webhook_database_busy`, not a fabricated `idempotency_in_progress`. Use only the disposable PostgreSQL fixture.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py
```

Expected: PASS when the disposable database is configured; a local SKIP must become a required CI pass before merge.

- [ ] **Step 6: Commit the repository unit**

```bash
git add tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py \
  "backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md"
git commit -m "feat(webhooks): add transactional canonical repository"
```

### Task 5: Implement Registration Lifecycle And Bounded Secret Replay

**Files:**
- Create: `tldw_Server_API/app/core/Admin_Webhooks/audit.py`
- Create: `tldw_Server_API/app/core/Admin_Webhooks/control_plane.py`
- Test: `tldw_Server_API/tests/Admin_Webhooks/test_audit.py`
- Test: `tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py`

**Interfaces:**
- Consumes: `AdminWebhookRepository`, `WebhookKeyRing`, `AdminWebhookSettings`, catalog validators, registration-time `evaluate_platform_webhook_url_policy()`, injected `DeliveryCapability.is_ready() -> bool`, and the established `MandatoryAuditWriteError`/isolated unified-audit service pattern from `AuthNZ/api_key_audit.py`.
- Produces: `DeliveryCapability.is_ready() -> bool`, default `UnavailableDeliveryCapability`, `AdminWebhookControlPlane.create()`, `list()`, `get()`, `patch()`, `delete()`, `rotate_secret()`, `catalog()`, and `status()`; `get_admin_webhook_control_plane()`; `CreateRegistrationCommand`, `PatchRegistrationCommand`, `RotateSecretCommand`, `SecretMutationResult`, `MutationResult`; redacted `MutationAudit` and `OperationalAudit`; callable `MutationAuditSink` and `OperationalAuditSink`; `validate_actor_principal_id()`, `validate_actor_kind()`, `validate_actor_roles()`; `emit_mandatory_webhook_audit()`; and `emit_mandatory_webhook_operation_audit()`.

- [ ] **Step 1: Write failing lifecycle, key-loss, and idempotency tests**

```python
async def test_create_is_inactive_and_replay_returns_same_secret(service: AdminWebhookControlPlane) -> None:
    command = create_command(idempotency_key="0123456789abcdef")
    first = await service.create(command, audit_sink=recording_audit_sink)
    replay = await service.create(command, audit_sink=recording_audit_sink)
    assert first.registration.active is False
    assert first.secret.startswith("whsec_") and len(first.secret) == 70
    assert replay.secret == first.secret
    assert replay.replayed is True
    assert replay.registration.id == first.registration.id

async def test_rotate_replay_is_checked_before_changed_revision(service: AdminWebhookControlPlane) -> None:
    registration = await create_inactive_registration(service)
    command = rotate_command(registration, idempotency_key="fedcba9876543210")
    first = await service.rotate_secret(command, audit_sink=recording_audit_sink)
    replay = await service.rotate_secret(command, audit_sink=recording_audit_sink)
    assert replay.secret == first.secret
    assert replay.registration.revision == first.registration.revision
```

Cover mode off/migrate/on endpoint gates; migration pending; no usable key; create bounds; caller secret/active/wildcard rejection; URL policy; target encryption/redaction; exact `whsec_` format; empty/duplicate events; catalog-order persistence/response; reordered create replay and PATCH no-op; timeout/description bounds; list/get without decryption where possible; missing/stale/malformed ETag; no-op PATCH; description-only revision; URL/event/timeout version changes; activation rejected while delivery capability is absent/unhealthy or `secret_rotation_required`; canonical rotation clearing that marker; soft delete; rotating only while inactive; exact create/rotate replay; same key/different body; concurrent claim; replay with missing key; replay after rotation/delete; secret-free errors; first-canonical-activity marking on every effective create/PATCH/delete/rotate but not replay/rejection/no-op; bounded audit metadata with hostname but no path/query/secret; `accepted`/`no_op` auditing before commit; audit failure rollback; and a correlated `failed` audit attempt when commit fails after acceptance.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_audit.py \
  tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py
```

Expected: FAIL because the audit adapter and `AdminWebhookControlPlane` are absent.

- [ ] **Step 3: Implement the bounded mandatory-audit adapter**

```python
@dataclass(frozen=True)
class MutationAudit:
    actor_id: int
    action: Literal["admin_webhook.create", "admin_webhook.patch", "admin_webhook.delete", "admin_webhook.rotate_secret"]
    webhook_id: int | None
    target_hostname: str | None
    event_types: tuple[str, ...]
    outcome: Literal["accepted", "no_op", "denied", "failed"]
    request_id: str
    reason_code: WebhookErrorCode | None

MutationAuditSink = Callable[[MutationAudit], Awaitable[None]]

@dataclass(frozen=True)
class OperationalAudit:
    operator_id: int
    action: Literal[
        "admin_webhook.import.apply",
        "admin_webhook.import.reject_source",
        "admin_webhook.key_rotation.start",
        "admin_webhook.key_rotation.resume",
        "admin_webhook.key_rotation.verify",
        "admin_webhook.key_rotation.finalize",
        "admin_webhook.rollback.extract",
        "admin_webhook.rollback.destroy",
    ]
    operation_id: str
    outcome: Literal["accepted", "completed", "failed"]
    request_id: str
    reason_code: WebhookOperationalReasonCode | None

OperationalAuditSink = Callable[[OperationalAudit], Awaitable[None]]

AUDIT_WRITE_TIMEOUT_SECONDS = 5.0
AUDIT_STOP_TIMEOUT_SECONDS = 1.0

async def emit_mandatory_webhook_audit(
    record: MutationAudit,
    *,
    actor_principal_id: str,
    actor_kind: str | None,
    actor_roles: tuple[str, ...],
) -> None:
    audit_service: UnifiedAuditService | None = None
    async def write_event() -> None:
        nonlocal audit_service
        audit_service = await _create_isolated_audit_service(record.actor_id)
        event_type, category = {
            "admin_webhook.create": (AuditEventType.DATA_WRITE, AuditEventCategory.DATA_MODIFICATION),
            "admin_webhook.patch": (AuditEventType.DATA_UPDATE, AuditEventCategory.DATA_MODIFICATION),
            "admin_webhook.delete": (AuditEventType.DATA_DELETE, AuditEventCategory.DATA_MODIFICATION),
            "admin_webhook.rotate_secret": (AuditEventType.DATA_UPDATE, AuditEventCategory.SECURITY),
        }[record.action]
        await audit_service.log_event(
            event_type=event_type,
            category=category,
            context=AuditContext(user_id=str(record.actor_id)),
            resource_type="admin_webhook",
            resource_id=str(record.webhook_id) if record.webhook_id is not None else None,
            action=record.action,
            metadata={
                "actor_principal_id": actor_principal_id,
                "actor_kind": actor_kind or "unknown",
                "actor_roles": list(actor_roles),
                "target_hostname": record.target_hostname,
                "event_types": list(record.event_types),
                "outcome": record.outcome,
                "request_id": record.request_id,
                "reason_code": record.reason_code,
            },
        )
        await audit_service.flush(raise_on_failure=True)

    try:
        await asyncio.wait_for(write_event(), timeout=AUDIT_WRITE_TIMEOUT_SECONDS)
    except MandatoryAuditWriteError:
        raise
    except Exception as exc:
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable") from exc
    finally:
        if audit_service is not None:
            with suppress(Exception):
                await asyncio.wait_for(audit_service.stop(), timeout=AUDIT_STOP_TIMEOUT_SECONDS)
```

Follow `AuthNZ/api_key_audit.py`: the private `_create_isolated_audit_service()` wrapper imports `_create_audit_service_for_user`, flushes with `raise_on_failure=True`, preserves `MandatoryAuditWriteError`, converts every other adapter failure to that exception without embedding original text, and always attempts a bounded stop. The adapters accept no arbitrary metadata dictionary. `emit_mandatory_webhook_operation_audit()` uses the same bounded lifecycle and stores only the exact `OperationalAudit` fields; key IDs, source fingerprints/content, artifact paths, and free text are impossible in its type and serialization mapping. Before constructing either sink, require every identity, action, outcome, request ID, and reason code to satisfy closed enums or safe-character/length bounds; reject rather than truncate. Unit tests inject create/log/flush/timeout/stop failures, assert the exception and five-second write-boundary contracts, recursively scan captured records for URL paths, queries, secrets, payloads, key material, filesystem paths, and free text, and prove an unavailable pre-operation audit prevents the corresponding service callback from running.

Define `DeliveryCapability` as a protocol with synchronous `is_ready() -> bool`
because PR 1 performs no network health probe. `UnavailableDeliveryCapability`
always returns false. `get_admin_webhook_control_plane()` wires that default and
the repository/settings/`WebhookKeyRingLoadResult` once through the repository's
existing application-scoped dependency pattern; tests override the provider
rather than mutating module globals. Missing or malformed keys never prevent
authorized status or permitted metadata operations from resolving the service.

- [ ] **Step 4: Implement create and exact replay ordering first**

```python
@dataclass(frozen=True)
class CreateRegistrationCommand:
    actor_id: int
    idempotency_key: str
    url: str
    event_types: tuple[str, ...]
    description: str
    timeout_seconds: int
    request_id: str
    now: datetime

@dataclass(frozen=True)
class PatchRegistrationCommand:
    actor_id: int
    webhook_id: int
    if_match: str
    patch: RegistrationPatch
    request_id: str
    now: datetime

@dataclass(frozen=True)
class RotateSecretCommand:
    actor_id: int
    webhook_id: int
    if_match: str
    idempotency_key: str
    request_id: str
    now: datetime

async def create(
    self,
    command: CreateRegistrationCommand,
    *,
    audit_sink: MutationAuditSink,
) -> SecretMutationResult:
    self._require_mode_and_migration_ready()
    normalized = normalize_create(command)
    scope = build_idempotency_scope(command.actor_id, "create", "/admin/webhooks")
    request_hash = canonical_request_hash(
        command.idempotency_key,
        scope=scope,
        body=normalized.request_body(),
        conditional_version=None,
    )
    async with self._repository.transaction() as tx:
        claim = await tx.claim_idempotency(build_create_claim(scope, request_hash, command))
        if claim.kind == "conflict":
            raise WebhookError("idempotency_conflict", 409)
        if claim.kind == "in_progress":
            raise WebhookError("idempotency_in_progress", 409)
        migration_state = await tx.lock_migration_state()
        if claim.kind == "replay":
            self._require_secret_replay_key_state(migration_state)
            replay = await self._resolve_secret_replay(tx, claim)
            await audit_sink(build_mutation_audit(command, replay.registration, outcome="no_op"))
            return replay
        self._require_protected_write_key_state(migration_state)
        await tx.enforce_registration_limit(self._settings.registration_limit)
        webhook_id = await tx.allocate_registration_id()
        secret = "whsec_" + secrets.token_hex(32)
        registration = await tx.insert_registration(
            build_encrypted_registration(webhook_id, normalized, secret, command.actor_id, command.now)
        )
        await tx.complete_idempotency(build_create_completion(scope, request_hash, registration, secret))
        await audit_sink(build_mutation_audit(command, registration, outcome="accepted"))
    return SecretMutationResult(registration=registration, secret=secret, replayed=False)
```

The actual implementation must claim a new idempotency row before allocating/inserting, keep claim/completion/mutation in one transaction, mark first canonical activity as `registration_mutation` in that transaction, await the mandatory sink before the transaction exits, and clear local secret references after response construction. Do not commit a separate create/rotate `in_progress` claim merely to make that state externally observable: normal concurrent identical calls resolve to one new result plus replay after the winner commits. Retain defensive handling for a durable `in_progress` row reserved for later network operations or recovery fixtures. Exact replay compares scope/request hash before any resource precondition, decrypts replay material only when resource ID and recorded secret version are still current, and never synthesizes a replacement secret. Key availability is checked after request-hash conflict detection: same-key/different-body still returns `409` without decrypting, while an exact secret-bearing replay with a missing key returns `503` and leaves the completed record replayable.

Wrap each public mutation in one audit-orchestration boundary after the route has established the numeric actor. Every deterministic validation, mode, authorization-state, precondition, limit, conflict, or policy rejection after service entry emits exactly one bounded `denied` record before the stable error is raised; dependency, repository, key-loading, and unexpected service failures emit `failed`. A failure detected inside a unit of work writes that audit before raising so incidental claim/expiry cleanup also rolls back. A failure detected before a transaction writes its audit before returning. Invalid caller event values or targets contribute no unvalidated event/host metadata. Audit persistence failure replaces the original response with `503 admin_webhook_audit_unavailable`; it must never create a second mutation-audit event in the route. Replays and effective no-op PATCH calls emit one `no_op` audit but do not alter the activity marker. `MandatoryAuditWriteError` escapes the unit of work so both backends leave no committed business change or activity marker; focused tests prove all pre-transaction, in-transaction, and commit-failure paths. If commit fails after `accepted`, attempt a `failed` event with the same request ID; do not mask the commit error if that follow-up audit also fails.

- [ ] **Step 5: Implement list/get/PATCH/delete/rotate/status**

PATCH accepts description, URL, event set, timeout, and active only. It never accepts a secret. Resolve current registration inside the mutation transaction, compare the parsed ETag revision, compute effective changes, encrypt a changed target with the next `target_version`, and increment only the versions required by the design. URL, event, timeout, and active changes increment `delivery_config_version`; only URL changes increment `target_version`. With no usable key, metadata reads, description/event/timeout changes, disable, and soft delete remain available; URL update, enable, create, rotate, and secret-bearing replay fail with `503 admin_webhook_key_unavailable`. Every protected write also locks/reads migration state in that transaction and requires no active rotation plus local primary equal to `active_primary_key_id`; mismatch returns `503 admin_webhook_key_configuration_mismatch`. Activation additionally requires `DeliveryCapability.is_ready()` and current active-count capacity; the PR 1 default capability always returns false.

Rotate performs idempotency inspection before ETag comparison, requires inactive/current/non-deleted state, increments revision/delivery-config/secret versions, clears `secret_rotation_required`, encrypts the new generated secret and replay material with their resulting identities, and makes old replay material superseded. Delete writes tombstone metadata and never hard-deletes. Every effective PATCH/delete/rotate marks first canonical activity in its mutation transaction; rejection, replay, and no-op do not. `status()` reports mode, selected route, schema/import/key state, imported registrations awaiting signing-secret rotation, limits/current counts, `legacy_file_restore_permitted`, rollback-window expiry, and `delivery_capability_ready=false` without artifact paths or invented worker/reconciler heartbeat data before PR 2. `legacy_file_restore_permitted` is true only when import is complete, retirement is `retained`, the rollback window is unexpired, and the first-activity marker is empty.

Enforce one explicit availability matrix: status is always available after platform-admin authorization; mode `off` makes every other canonical route return `503 admin_webhooks_disabled`; mode `migrate` or incomplete import makes every other route return `503 admin_webhook_migration_pending`; mode `on` plus completed migration permits catalog and metadata reads; key loss then applies the restricted mutation behavior above; key rotation returns `503 admin_webhook_key_rotation_in_progress` for create, URL update, signing-secret rotation, and secret-bearing replay while metadata-only mutation remains available. Use `404 admin_webhook_not_found`, `409 admin_webhook_registration_limit`, `409 admin_webhook_active_limit`, `409 admin_webhook_secret_rotation_required`, `409 idempotency_conflict`, `409 idempotency_in_progress`, `409 idempotency_result_superseded`, `412 precondition_failed`, `428 precondition_required`, `503 admin_webhook_audit_unavailable`, `503 admin_webhook_database_busy`, and `503 admin_webhook_key_configuration_mismatch` exactly.

- [ ] **Step 6: Run GREEN and encrypted-at-rest assertions**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_audit.py \
  tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_sqlite.py \
  -k 'create or replay or patch or rotate or delete or key or encrypted or limit or status'
```

Expected: PASS; direct database assertions cannot find target path/query, signing secret, or replay secret plaintext.

- [ ] **Step 7: Commit the lifecycle unit**

```bash
git add tldw_Server_API/app/core/Admin_Webhooks/control_plane.py \
  tldw_Server_API/app/core/Admin_Webhooks/audit.py \
  tldw_Server_API/tests/Admin_Webhooks/test_audit.py \
  tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py \
  "backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md"
git commit -m "feat(webhooks): implement canonical registration lifecycle"
```

### Task 6: Make System-Ops Publication Atomic And Implement Key Rotation

**Files:**
- Create: `tldw_Server_API/app/core/Admin_Webhooks/key_rotation.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/domain.py`
- Modify: `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py`
- Modify: `tldw_Server_API/app/services/admin_system_ops_service.py`
- Test: `tldw_Server_API/tests/Admin_Webhooks/test_key_rotation.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py`
- Modify: `tldw_Server_API/tests/Admin/test_admin_system_ops_service.py`

**Interfaces:**
- Consumes: `WebhookKeyRing`, repository protected-row paging/CAS methods, migration-state rotation fields, `OperationalAuditSink`, and existing `_STORE_LOCK` plus `_store_file_lock()`.
- Produces: `_load_store_strict(path, max_bytes=67_108_864)`, `_atomic_write_store(path, store)`, dormant `PendingIncidentWebhookMarker`, `WebhookKeyRotationService.start(operation_id, source_key_id, target_key_id, operator_id, request_id, audit_sink)`, `resume(operation_id, operator_id, request_id, audit_sink)`, `verify(operation_id, operator_id, request_id, audit_sink)`, `finalize(operation_id, operator_id, request_id, audit_sink)`, and `KeyRotationProgress`.

- [ ] **Step 1: Write atomic publication RED tests**

```python
def test_atomic_save_fsyncs_file_replace_and_parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    instrument_atomic_calls(monkeypatch, calls)
    write_store_at(tmp_path / "system_ops.json", {"incidents": [], "webhooks": []})
    assert calls == ["write-temp", "flush", "fsync-file", "replace", "fsync-parent"]
```

Also inject failure before write, after file fsync, and before replace; the old file must stay parseable and no partial destination may publish. Verify mode `0600`, same-directory temp placement, cleanup, existing process/file lock behavior, and unchanged JSON structure. Add strict-reader cases for missing and whitespace-only files, exact 64 MiB acceptance, 64 MiB plus one rejection, read errors, invalid JSON, non-object roots, and proof that no failure calls `_atomic_write_store()` or logs file content. Existing `_load_store()` behavior remains unchanged for its current callers.

- [ ] **Step 2: Implement atomic save and run focused GREEN**

```python
def _atomic_write_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(store, indent=2, sort_keys=False).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)
```

Implement `_load_store_strict()` by opening a regular file without following a
symlink where the platform supports it, checking the byte limit before and
during read, decoding UTF-8, parsing JSON without default injection, and
requiring an object root. Return `{}` only for a genuinely absent or
whitespace-only file. Use `_atomic_write_store()` only from the already-locked
`_save_store()` path. Preserve a platform-specific tested error path where
directory fsync is unsupported; do not silently skip ordinary read/write/replace
failures.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin/test_admin_system_ops_service.py \
  -k 'atomic or lock or store'
```

- [ ] **Step 3: Write failing rotation state-machine and crash tests**

```python
async def test_rotation_resumes_after_each_committed_batch(rotation_fixture: RotationFixture) -> None:
    operation = await rotation_fixture.service.start(
        "rotation-op-1",
        "key-2026-01",
        "key-2026-08",
        operator_id=9,
        request_id="rotation-start-1",
        audit_sink=recording_operation_audit_sink,
    )
    await rotation_fixture.fail_after_batches(operation.id, completed_batches=1)
    resumed = await rotation_fixture.service.resume(
        operation.id,
        operator_id=9,
        request_id="rotation-resume-1",
        audit_sink=recording_operation_audit_sink,
    )
    assert resumed.phase == "verifying"
    assert await rotation_fixture.count_envelopes("key-2026-01") == 0
    verified = await rotation_fixture.service.verify(
        operation.id,
        operator_id=9,
        request_id="rotation-verify-1",
        audit_sink=recording_operation_audit_sink,
    )
    assert verified.phase == "awaiting_primary_cutover"
    await rotation_fixture.configure_local_primary("key-2026-08")
    completed = await rotation_fixture.service.finalize(
        operation.id,
        operator_id=9,
        request_id="rotation-finalize-1",
        audit_sink=recording_operation_audit_sink,
    )
    assert completed.phase == "complete"
    assert completed.verified_count == completed.processed_count
```

Cover invalid same/missing source/target keys, source unequal to durable active primary, local primary unequal to source at start, one active operation, rejection while legacy import is in progress, durable operation ID/phase/table/cursor/counts/timestamps, bounded batches, registration targets, registration secrets, event bodies, pending incident markers, unexpired replay secrets, CAS rejection after concurrent change, skip of already-target envelopes, crash before/after row commit and cursor commit, complete readback, repeated verification until no source envelope remains, unavailable source key, `awaiting_primary_cutover`, finalize refusal before local primary changes to target, finalize full rescan, durable active-primary switch, lagging old-primary write rejection, forward-resume-only once any row moves, source-key removal refusal, and create/URL-update/secret-rotation/replay maintenance gating while metadata-only mutations remain available. For every mutating command, also prove a generated sanitized request ID, mandatory `accepted` audit before the first state/file mutation, correlated `completed`/`failed` outcome, no key ID/path/fingerprint in audit, and zero state change when the pre-operation audit is unavailable.

Extend `test_repository_postgres.py` with the same bounded protected-row page,
ciphertext compare-and-set, cursor/count atomicity, already-target accounting,
and rollback cases used by rotation. Those tests run against the disposable
PostgreSQL fixture and may not be satisfied by mocked SQL.

- [ ] **Step 4: Run rotation RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_key_rotation.py
```

Expected: FAIL because key rotation is absent.

- [ ] **Step 5: Add bounded repository scanners and compare-and-set replacement**

```python
@dataclass(frozen=True)
class ProtectedRow:
    table: str
    row_identity: str
    field: str
    protected: ProtectedValue
    purpose: str
    envelope_identity: Mapping[str, str | int]

async def page_protected_rows(self, *, table: str, after: str | None, limit: int) -> list[ProtectedRow]:
    return await self._page_known_protected_table(table=table, after=after, limit=min(limit, 500))
async def replace_protected_value(self, row: ProtectedRow, expected_ciphertext: str, replacement: ProtectedValue) -> bool:
    return await self._replace_known_protected_field(row, expected_ciphertext, replacement)
```

Implement concrete SQL for every schema field in the design. The scanner returns no decrypted values. Re-encryption decrypts and validates context, encrypts under the target primary, then replaces only when the old ciphertext still matches. Persist the new cursor/count in the same transaction as each bounded batch.

- [ ] **Step 6: Implement start/resume/verify and file-marker participation**

Start requires both keys configured, `source_key_id` equal to durable `active_primary_key_id`, and the local ordinary-write primary still equal to source; target must differ but need not yet be local primary. The CLI caller generates the durable operation ID and a separate sanitized request ID, then passes both with the required `OperationalAuditSink` into the service. The service emits mandatory `accepted` audit through that sink before entering its first locked migration-state transaction; audit failure leaves state untouched. In that transaction start sets phase `rewriting` and blocks every protected write. Resume follows a fixed table/field order, calls `reencrypt_to_key(row.protected, purpose=row.purpose, identity=row.envelope_identity, target_key_id=target_key_id)`, and sets `verifying` after the final rewrite page. `processed_count` means protected values durably observed under the target key, not only successful compare-and-set rewrites: a resumed page increments it when a state-owned cursor advances past either a value it rewrote or a value already on target. This makes the count deterministic when a locked file publication commits but the later database cursor update crashes; replay observes the target envelope, accounts for it once while advancing the cursor, and never double-counts a committed database row/cursor batch. `verified_count` uses the same inventory definition during the final full scan, so equality is meaningful.

Define dormant `PendingIncidentWebhookMarker` entries under `webhook_pending_events` with event ID/type/API version, aggregate or command source identity, encrypted body, body key ID, and creation time; no PR 1 code produces them. Marker scans and compare-and-set rewrites execute under the existing system-ops process/file lock and publish only through `_atomic_write_store()`. Never hold an AuthNZ transaction while acquiring the file lock: publish one locked file batch, then persist its cursor; a crash before cursor persistence safely rescans and accounts for envelopes already using target as described above. Verify is accepted only in `verifying`, performs full readback, and repeats database plus locked-file scans until a complete pass finds no source-key envelope, then sets `awaiting_primary_cutover` without unblocking protected writes. After deployment configuration changes every process primary to target, finalize requires `awaiting_primary_cutover` and local primary target, repeats a full zero-source pass, atomically sets `active_primary_key_id=target`, and marks rotation `complete`. Each invocation attempts a correlated `completed` or `failed` operational audit after its durable step; an absent terminal audit never substitutes for migration state during recovery. Every ordinary protected write checks that durable ID, so an old-primary process remains failed closed. Once `processed_count > 0`, only resume/verify/finalize is permitted; rollback is rejected.

- [ ] **Step 7: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_key_rotation.py \
  tldw_Server_API/tests/Admin_Webhooks/test_crypto.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_sqlite.py \
  tldw_Server_API/tests/Admin/test_admin_system_ops_service.py
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py \
  -k 'protected or rotation or cursor'
git add tldw_Server_API/app/core/Admin_Webhooks/key_rotation.py \
  tldw_Server_API/app/core/Admin_Webhooks/domain.py \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/services/admin_system_ops_service.py \
  tldw_Server_API/tests/Admin_Webhooks/test_key_rotation.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py \
  tldw_Server_API/tests/Admin/test_admin_system_ops_service.py \
  "backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md"
git commit -m "feat(webhooks): add resumable encryption-key rotation"
```

### Task 7: Import Legacy Sources Safely

**Files:**
- Create: `tldw_Server_API/app/core/Admin_Webhooks/legacy_import.py`
- Create: `tldw_Server_API/cli/commands/admin_webhooks.py`
- Create: `tldw_Server_API/cli/admin_webhooks_cli.py`
- Modify: `pyproject.toml`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_legacy_import.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_legacy_import_postgres.py`

**Interfaces:**
- Consumes: Task 6 `_load_store_strict()` and `_atomic_write_store()`, existing `_STORE_LOCK`, `_store_file_lock()`, and `_STORE_PATH`, canonical repository/key ring/catalog, historical `decrypt_admin_webhook_secret()` only inside this module, `OperationalAuditSink`, and operator-supplied backup/report/key/output paths.
- Produces: migration-only `LegacySecretDecryptor`, `LegacyImportRequest`, `LegacyImportService.build_plan()`, `apply_plan()`, `verify_and_sanitize()`, `extract_rollback_backup()`, `destroy_rollback_key()`, `LegacyImportPlan`, and CLI commands `import-legacy`, `reject-source`, `extract-rollback-backup`, `rotate-key start|resume|verify|finalize`, `rotation-status`, and `destroy-rollback-key`.

- [ ] **Step 1: Write legacy-import RED tests for both sources and every crash window**

```python
async def test_dry_run_is_deterministic_and_mutates_no_source_or_canonical_state(import_fixture: LegacyImportFixture) -> None:
    first = await import_fixture.service.build_plan(import_fixture.request)
    second = await import_fixture.service.build_plan(import_fixture.request)
    assert first.report_digest == second.report_digest
    assert first.source_mapping == second.source_mapping
    assert import_fixture.request.report_path.is_file()
    assert await import_fixture.canonical_registration_count() == 0
    assert import_fixture.system_ops_bytes == import_fixture.original_system_ops_bytes
```

Cover absent/whitespace-only, valid, oversized, unreadable, malformed, and non-object JSON sources without permissive default substitution; legacy `*` expansion to the six current catalog values; byte-for-byte preservation and dedicated re-encryption of plaintext/old encrypted DB secrets; `secret_rotation_required=true` on every imported row; explicit migration-only fallback; duplicate source identity; deterministic duplicate/existing/nonnumeric numeric-ID collision handling; sequence advancement above the maximum imported ID; positive-64-bit exhaustion with no partial import; semantically similar but conflicting rows; timeout bounds; rejected URL; undecryptable secret; projected limit overflow with no partial import; deterministic source-to-canonical mapping; atomic mode-`0600` redacted report; distinct normalized output/source paths; rejection of unsafe report/backup/key/output parents and unclaimed existing/symlink/non-regular backup, key, staging, and plaintext-extraction targets; domain-separated keyed source-table/record/webhook-subtree fingerprints plus recorded key ID; source-fingerprint-bound rejection decisions and source-drift invalidation; unresolved records blocking completion; an empty fresh-install plan completing without backup files; database-only import completing without a file backup; unrelated file changes between lock windows; webhook-subtree changes that halt sanitization; crash before/after `artifacts_pending`, each staging write/fsync/readback/link/directory-fsync/unlink, `artifacts_ready`, DB commit, canonical readback, and active-file replace; state-owned staging/key-only/key-plus-backup resume; refusal to adopt or delete unclaimed artifacts; rerun after commit; preservation/removal of exactly `webhooks` and `webhook_deliveries`; encrypted full-file backup readback; verified non-destructive backup extraction to a separate mode-`0600` file with no plaintext stdout/logging; extraction refusal after first canonical activity, expiry, or retirement; one-time key outside data directory; rollback window 7-30 days; mandatory operational audit before apply/reject/extract/destroy mutation; zero side effects when that audit is unavailable; correlated terminal audit; and auditable key destruction.

The focused PostgreSQL import suite repeats the accepted/rejected mapping
transaction, preserved-ID collision allocation, sequence advancement,
projected-limit rollback, source-drift rejection, database-only no-artifact
path, post-commit rerun, and canonical decrypt/readback against a disposable
backend. It must inspect committed PostgreSQL rows and sequence state rather
than replacing the repository with a mock.

- [ ] **Step 2: Implement dry-run planning and explicit apply approval**

```python
@dataclass(frozen=True)
class LegacyImportRequest:
    report_path: Path
    backup_path: Path | None
    rollback_key_path: Path | None
    operator_id: int
    now: datetime
    allow_legacy_credential_decryption: bool = False

@dataclass(frozen=True)
class LegacyImportPlan:
    operation_id: str
    fingerprint_key_id: str
    legacy_credential_decryption_enabled: bool
    source_fingerprints: Mapping[str, str]
    accepted: tuple[LegacyAcceptedRecord, ...]
    unresolved: tuple[LegacyUnresolvedRecord, ...]
    explicitly_rejected: tuple[LegacyRejectedRecord, ...]
    projected_non_deleted_count: int
    source_mapping: Mapping[str, int]
    requires_system_ops_backup: bool
    report_digest: str
```

`LegacyImportPlan` contains whether legacy credential decryption was explicitly enabled, source fingerprints, accepted/unresolved/explicitly-rejected records with stable reason codes, projected counts, deterministic source mapping, `requires_system_ops_backup`, and `report_digest`; output paths remain only on `LegacyImportRequest`. The plan contains no full target path/query or secret. Define one versioned `canonical_report_payload(plan) -> bytes` encoder over sorted compact redacted content that excludes the outer `report_digest` field, presentation timestamps, and filesystem paths. SHA-256 that payload and encode the result exactly as `sha256:` plus 64 lowercase hexadecimal characters, then publish an envelope containing the payload fields plus `report_digest`; dry-run and apply call the same encoder, so the digest is deterministic rather than self-referential. Compare the approved and recomputed values with `hmac.compare_digest()`. Tests mutate every payload field, the outer digest, excluded presentation fields, and key order to prove the boundary. Reports are published mode `0600`.

Build the ID mapping in two deterministic passes over stable
`(source_kind, source_identity)` order. First reserve existing canonical IDs and
the first source owning each valid positive legacy 64-bit ID that is not already
canonical. Then assign every
duplicate, colliding, or nonnumeric source the next unreserved positive ID from
a cursor initialized from the persisted sequence; skip all reserved IDs and
fail on 64-bit exhaustion. Similar rows never merge automatically. In the import
transaction, compare the current sequence/source state with the approved plan,
insert the mapping, and set `next_value` to at least one greater than every
inserted ID. A rerun reads the durable mapping and never advances twice.

Define `LegacySecretDecryptor` in `legacy_import.py` as the only wrapper around
historical `decrypt_admin_webhook_secret()`. Construct it only when
`allow_legacy_credential_decryption=true`, which is reachable solely through
the exact CLI flag `--allow-legacy-credential-decryption`. Without the flag,
an encrypted legacy row that needs unrelated credentials remains unresolved;
with it, decrypt failures map to a stable redacted reason and no candidate name,
value, exception text, or plaintext enters logs, reports, or audit. Dry-run and
apply must receive the same flag; a mismatch invalidates approval.

Derive `operation_id` as `"whmig_"` plus the first 32 lowercase hex
characters of HMAC-SHA256 over
`b"tldw-admin-webhook-migration-operation-v1\x00"`, fingerprint key ID,
and sorted source-kind/fingerprint pairs. Use the dedicated fingerprint key and
one canonical encoder; never include source bytes. Include `operation_id` in
the canonical report payload. Tests prove unchanged source/key dry-runs retain
the same ID/digest while any source fingerprint or fingerprint-key change
changes the ID and invalidates approval.

`build_plan(request) -> LegacyImportPlan` reads source locations only from the
configured repository and `_STORE_PATH`; the CLI cannot redirect it to an
arbitrary source file. `apply_plan(request, *, approved_report_digest) ->
MigrationState` reparses the mode-`0600` report and a fresh source snapshot
rather than trusting an in-memory plan object. Normalize all paths without
requiring absent outputs to exist, reject aliases with active data or each
other, reject symlink/non-regular inputs, create first-attempt backup/key outputs
with `O_CREAT|O_EXCL` plus `O_NOFOLLOW` where supported, fsync each file and
parent directory, and read back mode, ownership, file identity, digest, and
decryptability before continuing. A resume may open an existing artifact only
when durable `artifacts_pending`/`artifacts_ready` state claims that exact
operation and normalized path; all unclaimed existing targets fail closed and
are never adopted or deleted.
Publish the report with the same atomic file/replace/parent-fsync contract as
system ops but never overwrite a backup or key. Exercise the documented
platform-specific directory-fsync error path for all three artifact kinds.

Add a `reject-source` command that requires `--source-kind system_ops|database`, the report's source identity and keyed record fingerprint, one reason code from `receiver_decommissioned|duplicate_external_config|invalid_legacy_record|operator_excluded`, and `--operator-id`. Persist the decision and fingerprint key ID in migration state and audit it without free text or source content. A later source fingerprint mismatch automatically makes the decision stale. `--apply` refuses any unresolved source. For apply and rejection, the service receives a generated sanitized request ID plus `OperationalAuditSink`, emits mandatory `accepted` before its first state/file mutation, and attempts correlated `completed`/`failed` afterward. A failed pre-operation audit leaves report/source/database state untouched; durable migration state, not the presence of a terminal audit, controls resume.

The CLI requires this sequence:

```bash
tldw-admin-webhooks import-legacy --dry-run \
  --allow-legacy-credential-decryption \
  --report ./admin-webhook-import-report.json \
  --backup ./admin-webhook-system-ops.backup.enc \
  --rollback-key-file ~/.config/tldw/admin-webhook-rollback.key \
  --operator-id 9
tldw-admin-webhooks import-legacy --apply \
  --all-writers-quiesced \
  --allow-legacy-credential-decryption \
  --approved-report-digest sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --report ./admin-webhook-import-report.json \
  --backup ./admin-webhook-system-ops.backup.enc \
  --rollback-key-file ~/.config/tldw/admin-webhook-rollback.key \
  --operator-id 9
```

The sample digest is deliberately fake; the operator supplies the literal digest recorded during review and never computes it from the report inside the apply command. The implementation validates that approved report digest against the report and a fresh locked source snapshot before any write. It refuses `--apply` unless local settings are canonical/`migrate` with compatibility false, or without a matching dry-run artifact, usable dedicated key ring, explicit operator ID, and migration state proving no encryption-key rotation or competing import operation is in progress. The CLI requires an explicit `--all-writers-quiesced` acknowledgement; the runbook permits it only after every app process is drained/stopped or verified on canonical/`migrate` status with legacy CRUD/test/delivery/incident-notify routes unreachable. This acknowledgement is recorded by operator/request/operation ID but is not treated as technical proof; source fingerprint rechecks still fail on drift. It requires every report/backup/key/output parent to exist, be owned by the invoking effective UID, and have no group/world write bits. It additionally requires a backup path and rollback-key path outside the application data directory when `requires_system_ops_backup=true`; those flags are optional and no backup/key files are created when the plan will not alter `system_ops.json`.

- [ ] **Step 3: Implement commit, readback, structural sanitization, and resumability**

When `requires_system_ops_backup=true`, after the mandatory pre-operation audit, acquire the file lock only long enough to snapshot and compute the keyed fingerprint of the canonicalized webhook subtree, then release it. In a separate AuthNZ transaction compare-and-set migration state to `artifacts_pending` with operation/operator, approved report digest, source fingerprints/fingerprint-key ID, the approved canonical source mapping and rejection decisions, and normalized private final plus deterministic same-directory staging identities for backup/key. That state-owned redacted plan snapshot, not the mutable report file or terminal audit, is the recovery authority after reservation. Initial apply must parse and approve the report; resume from `artifacts_pending` or later still requires the same literal approved digest and a matching fresh source snapshot, but a missing report file cannot strand an already-reserved operation. If a report remains at the recorded path, a digest mismatch fails closed. Reacquire the file lock, require the same source fingerprint, and only then publish the mode-`0600` key first and encrypted full-file backup second; never hold an AuthNZ transaction while waiting for or holding the file lock.

Implement one private `publish_exclusive_artifact(final_path, staging_path, payload)` primitive. It creates the absent state-owned staging path with `O_CREAT|O_EXCL`/`O_NOFOLLOW`, writes/fsyncs/readbacks complete bytes, publishes to the absent final path with `os.link(..., follow_symlinks=False)` as an atomic no-overwrite step, fsyncs the parent, unlinks staging, and fsyncs the parent again. Resume validates owner/mode/regular-file identity and either finishes a complete staging link, removes/recreates only an incomplete state-owned staging inode while final is absent, or verifies final and removes a same-inode staging name left after link. Any unclaimed/existing final path or mismatched inode fails closed without deletion. The key payload is versioned JSON containing operation ID, source fingerprint, approved report digest, and strict-base64 encoding of exactly 32 key bytes. Encrypt the backup with purpose `legacy.system_ops.backup` and identity containing import operation ID plus source fingerprint. A resumed `artifacts_pending` operation validates contextual key, envelope context, decryptability, and source snapshot; a key-only crash creates the missing backup, and a complete pair is read back rather than overwritten. Persist the ciphertext digest and `artifacts_ready` in a later state CAS after releasing the file lock.

Apply requires the same fingerprint key ID and configured primary that produced the approved report. When file sanitization is required, the database insertion transaction starts only from `artifacts_ready`. For database-only or empty fresh-install plans it starts directly from `migration_pending` while locking the singleton, validates the approved plan in that same transaction, and never exposes an intermediate artifact phase. The transaction verifies the approved ID mapping against current canonical IDs/sequence, inserts all accepted inactive registrations, advances the sequence above every inserted ID, and advances migration `phase=database_committed` with `active_primary_key_id` set to that primary, fingerprint key ID, source fingerprints, mapping/report digest, backup ciphertext digest when applicable, rejection decisions, and `rollback_retirement_phase=retained` only when rollback artifacts exist. A resumed import requires the same durable active primary. After commit, decrypt and compare every canonical record. If file sanitization is required, reacquire the file lock, require the same webhook-subtree fingerprint, remove only top-level `webhooks` and `webhook_deliveries`, preserve current unrelated fields, and publish atomically. Mark migration complete only after a final database/file verification. A zero-record/no-legacy-field apply executes the same fresh source/key checks and `database_committed`/completion CAS but inserts no registration, creates no backup, and retains `rollback_retirement_phase=not_applicable`; this is the required fresh-install bootstrap path. Database-only imports likewise skip artifact phases and use `not_applicable`.

Rerun reads durable mapping/state before inserting. A crash after DB commit resumes readback/sanitization without allocating or inserting again. A changed webhook subtree returns `admin_webhook_legacy_source_changed`; unrelated incident changes survive. Rejected or undecryptable records leave state degraded until `reject-source` records an explicit current-fingerprint decision or the source is repaired and dry-run repeated.

- [ ] **Step 4: Implement bounded backup extraction, artifact destruction, and rotation CLI wiring**

`extract-rollback-backup` requires completed migration, `rollback_retirement_phase=retained`, an unexpired rollback window, no first-canonical-activity marker, `--backup`, `--rollback-key-file`, a distinct `--output`, `--operator-id`, and explicit confirmation. It returns stable `admin_webhook_rollback_window_closed` before artifact access when any restore condition is false. After mandatory `accepted` audit it verifies the paths against normalized migration-state identities, ciphertext digest, key file mode/readback, envelope purpose/operation/source-fingerprint context, and object-shaped JSON. The output must be outside application data; its existing parent must be owned by the invoking account and have no group/world write bits. It creates the plaintext output with `O_CREAT|O_EXCL`, `O_NOFOLLOW` where supported, mode `0600`, file fsync, parent fsync, and exact readback. It never prints/logs plaintext, writes the active data path, or overwrites an output. On failure it removes only an output inode created by that invocation and reports a stable reason; it attempts correlated `completed`/`failed` audit. Tests use secret canaries to scan stdout, stderr, logs, audit, exceptions, and state, prove expiry/activity/retirement and unsafe-parent refusal, and prove the active file is byte-for-byte unchanged.

`destroy-rollback-key` branches on durable retirement state before artifact access. `not_applicable` returns the stable no-op result `admin_webhook_rollback_artifacts_not_applicable`; `retired` returns `admin_webhook_rollback_artifacts_already_retired`. Neither path writes audit/state or touches the filesystem. `retained` requires completed import, the active backup ciphertext digest, rollback-key readback, requested window expiry measured from migration completion, and operator confirmation. After mandatory `accepted` audit and before file deletion it compare-and-sets `rollback_retirement_phase=rollback_retirement_in_progress` with operator, normalized artifact identities, and expected digest. A rerun already in `rollback_retirement_in_progress` validates those state-owned identities and resumes before requiring files that a prior attempt may already have deleted; an absent key or backup counts as completed only at its corresponding recorded post-unlink recovery boundary. It unlinks the key first, then the active encrypted backup, fsyncs each distinct parent directory, and finally sets `rollback_retirement_phase=retired` with retirement time/outcome plus the retired ciphertext digest. It does not claim filesystem overwrite guarantees and does not delete separately managed infrastructure backups. Tests inject crashes before/after accepted audit, marker commit, each unlink, each parent fsync, completion CAS, and terminal audit; cover `not_applicable`, `retired`, retained-before-expiry, and identity mismatch; and prove rerun reaches one completed state without treating audit as the recovery authority. The `rotate-key` subcommands and `rotation-status` call Task 6 services; mutating subcommands use the operational audit sink and require source plus target keys through finalize.

- [ ] **Step 5: Run importer suite and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_legacy_import.py
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_legacy_import_postgres.py
git add tldw_Server_API/app/core/Admin_Webhooks/legacy_import.py \
  tldw_Server_API/cli/commands/admin_webhooks.py \
  tldw_Server_API/cli/admin_webhooks_cli.py \
  pyproject.toml \
  tldw_Server_API/tests/Admin_Webhooks/test_legacy_import.py \
  tldw_Server_API/tests/Admin_Webhooks/test_legacy_import_postgres.py \
  "backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md"
git commit -m "feat(webhooks): add crash-safe legacy migration"
```

### Task 8: Expose One Authorized Canonical PR 1 Router

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/admin_webhooks.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/admin_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py`
- Modify: `tldw_Server_API/tests/Admin/test_admin_webhooks_service.py`
- Modify: `tldw_Server_API/tests/Admin/test_admin_webhooks_schemas.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_api.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_route_selection.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_openapi.py`

**Interfaces:**
- Consumes: `AdminWebhookControlPlane`, `get_auth_principal`, platform-admin enforcement, the existing bounded best-effort admin read-audit helper, `MutationAuditSink`, `emit_mandatory_webhook_audit()`, and canonical errors from Tasks 1-7.
- Produces: always-mounted `GET /admin/webhooks/status`; canonical-only `GET /admin/webhooks/catalog`, `GET/POST /admin/webhooks`, `GET/PATCH/DELETE /admin/webhooks/{webhook_id}`, and `POST /admin/webhooks/{webhook_id}/rotate-secret`; strong `ETag`, `If-Match`, and `Idempotency-Key` transport; `WebhookErrorResponse`; router-scoped redacted validation handling; route-local `_require_webhook_mutation_actor(principal) -> int`; and `_build_webhook_audit_sink(request_id, principal, actor_id) -> MutationAuditSink`.

- [ ] **Step 1: Write failing schema and OpenAPI contract tests**

```python
def test_create_schema_rejects_secret_active_and_wildcard() -> None:
    with pytest.raises(ValidationError):
        WebhookCreateRequest.model_validate({
            "url": "https://receiver.example/hooks/private",
            "event_types": ["*"],
            "secret": "caller-controlled",
            "active": True,
        })

def test_pr1_openapi_has_no_delivery_operations(app: FastAPI) -> None:
    paths = app.openapi()["paths"]
    assert "/api/v1/admin/webhooks/catalog" in paths
    assert "/api/v1/admin/webhooks/status" in paths
    assert "/api/v1/admin/webhooks/{webhook_id}/rotate-secret" in paths
    assert "/api/v1/admin/webhooks/{webhook_id}/test" not in paths
    assert "/api/v1/admin/webhooks/{webhook_id}/deliveries" not in paths
```

Define tests for exact field names/types, numeric IDs, redacted target fields, absence of secret from ordinary responses, `signing_secret` only on create/rotate response, revision/version fields, pagination, catalog limits/version, status degradation, `legacy_file_restore_permitted` plus nullable rollback-window expiry without artifact paths, PATCH requiring at least one recognized non-null field while allowing an effective same-value no-op, and declared 401/403/404/409/412/422/428/429/500/503 stable error bodies. Assert every canonical operation's 422 response references `WebhookErrorResponse`, no canonical operation references FastAPI's `HTTPValidationError`, and schema examples contain only reserved `.example` hosts and fake secrets.

- [ ] **Step 2: Define canonical Pydantic schemas and remove the duplicate numeric block**

```python
class WebhookCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=1, max_length=2048)
    event_types: list[str] = Field(min_length=1, max_length=6)
    description: str = Field(default="", max_length=500)
    timeout_seconds: int = Field(default=10, ge=1, le=30)

class WebhookRegistrationResponse(BaseModel):
    id: int
    description: str
    target_display: str
    target_hostname: str
    event_types: list[str]
    active: bool
    timeout_seconds: int
    revision: int
    delivery_config_version: int
    secret_version: int
    secret_rotation_required: bool
    created_by: int | None
    updated_by: int | None
    created_at: datetime
    updated_at: datetime

class WebhookSecretResponse(BaseModel):
    registration: WebhookRegistrationResponse
    signing_secret: str
    replayed: bool
```

Add patch, delete, catalog, status, paginated list, and the exact bounded error schemas in this focused file. `WebhookPatchRequest` uses `extra="forbid"`, rejects explicit null for every field, and has an after-validator requiring at least one field from description/URL/events/timeout/active; service comparison, not schema normalization, decides whether it is an effective no-op. List query validation and response metadata use `limit=50` by default, `1 <= limit <= 100`, `0 <= offset <= 1000`, and deterministic numeric `id DESC` ordering:

```python
class WebhookErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str

class WebhookErrorResponse(BaseModel):
    error: WebhookErrorDetail
```

Remove `AdminWebhookCreateRequest` through `AdminWebhookTestResponse` from `admin_schemas.py` after all canonical imports are moved. Do not delete string-ID legacy schemas used by the compatibility router.

- [ ] **Step 3: Run schema RED/GREEN boundary**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_openapi.py \
  tldw_Server_API/tests/Admin/test_admin_webhooks_schemas.py
```

Expected: canonical schema tests PASS. Update or retire old numeric-schema tests only where they assert the replaced contract; legacy string-schema tests remain. In `test_admin_webhooks_service.py`, remove route tests that import the rewritten canonical endpoint, retain service-level signing/dispatch tests needed by compatibility-only incident notification, and label that dependency for PR 3 deletion.

- [ ] **Step 4: Write failing authorization, header, error, and audit tests**

```python
@pytest.mark.parametrize("path", (
    "/api/v1/admin/webhooks/status",
    "/api/v1/admin/webhooks/catalog",
    "/api/v1/admin/webhooks",
))
def test_every_canonical_route_denies_non_platform_admin(client: TestClient, path: str) -> None:
    response = client.get(path, headers=user_headers())
    assert response.status_code == 403

def test_patch_requires_current_etag_and_returns_new_etag(admin_client: TestClient) -> None:
    created = create_registration(admin_client)
    missing = admin_client.patch(f"/api/v1/admin/webhooks/{created.id}", json={"description": "new"})
    assert missing.status_code == 428
    updated = admin_client.patch(
        f"/api/v1/admin/webhooks/{created.id}",
        headers={"If-Match": created.etag},
        json={"description": "new"},
    )
    assert updated.status_code == 200
    assert updated.headers["etag"] != created.etag
```

Cover anonymous/non-admin/platform-admin access for every route; user-backed API-key admins; service principals with an admin role but no `user_id` receiving `403 admin_webhook_user_principal_required` on mutation; static route ordering; missing/malformed/stale ETag; idempotency key validation; create/rotate replay; stable error mapping including `admin_webhook_audit_unavailable`; mode/migration/key/delivery-capability gates; response ETags on create/get/PATCH/rotate; `no-store`/`no-cache` on every secret-bearing success/replay; no ETag synthesis for list; pagination; no URL/secret in list/get/status/errors/audit; mandatory denied/failed audit calls for user-backed principals; bounded catalog/status/list/get access audits whose injected failure does not change the response; request ID/reason code; exception redaction; compatibility create/update audit canaries proving URL/path/query/secret omission while event count and enabled state remain; canonical-package import scans that reject `admin_webhooks_service`, `admin_webhook_secrets`, and `httpx`; and service-level proof that a failed mandatory audit leaves no registration, revision change, tombstone, rotation, or completed idempotency row.

Add request-level redaction cases that submit unique canaries in a complete destination URL, its query credential, a forbidden caller `secret`, malformed `If-Match`, and malformed `Idempotency-Key`. Each must return the fixed envelope below, with the normalized middleware request ID repeated in `X-Request-ID` and `Cache-Control: no-store`; no canary or Pydantic error detail may occur in the response body, response headers, `caplog`, or captured audit records. Inject auth dependencies that raise 401/403 with canaries in `detail` and arbitrary headers; assert fixed canonical auth codes/messages, no reflected detail, only exact `WWW-Authenticate: Bearer` on 401, and exactly the existing auth audit count.

Repeat the exception-header cases for 429 and 503: `Retry-After` survives 429
only when it contains ASCII decimal digits and parses to 0-86,400 seconds, then
is reserialized from that integer. Decimal overflow, signs, whitespace, dates,
and every other exception header are dropped. A 503 retains no injected headers,
and an unmapped HTTP status receives `admin_webhook_request_rejected` without
its detail.

```json
{"error":{"code":"admin_webhook_validation_failed","message":"Webhook request validation failed","request_id":"4aa1324c-7fb7-49cf-9058-ce0df25d5932"}}
```

- [ ] **Step 5: Rewrite `admin_webhooks.py` as a thin canonical router**

Expose `/webhooks/status` on `status_router` and declare `/webhooks/catalog` before `/webhooks/{webhook_id}` on `canonical_router`. Construct both with `route_class=AdminWebhookRoute`. `AdminWebhookRoute.get_route_handler()` wraps the original handler and catches three closed classes: `RequestValidationError`, canonical `WebhookError`, and `HTTPException`. Request validation returns `422 admin_webhook_validation_failed` without calling `exc.errors()`, formatting `exc`, or logging it. Known domain errors map through a fixed code/status/message registry. Audited auth 401/403/429/503 maps to fixed `authentication_required`/`platform_admin_required`/`authentication_rate_limited`/`authentication_unavailable` codes. Any other HTTP exception maps to `admin_webhook_request_rejected` with its status only when the integer is 400-599, otherwise 500. Never serialize `detail`. Preserve only exact `WWW-Authenticate: Bearer` on 401. On 429, preserve `Retry-After` only after ASCII-decimal parsing, a 0-86,400 bound, and canonical integer reserialization. Drop every other exception header.

All mapped cases call `_webhook_error_response()`, which accepts only a closed code, status, sanitized request ID, and fixed server-owned message; it returns `WebhookErrorResponse`, sets matching `X-Request-ID`, and sets `Cache-Control: no-store`. Do not pass exception text, submitted values, request bodies, or raw header values into the helper or logs. Register `WebhookErrorResponse` explicitly for every applicable canonical 4xx/5xx OpenAPI response, including 401, 403, and 422.

Status requires platform-admin authorization and reports explicit route selection even when legacy compatibility is selected. Every canonical handler accepts `Request`, `Response`, and `AuthPrincipal = Depends(get_auth_principal)`, invokes platform-admin enforcement before service access, and constructs a bounded `MutationAuditSink` from the sanitized middleware request ID and principal. Existing authentication and platform-role dependencies still own their audit events; route-level response translation must not duplicate them. Unexpected exceptions remain owned by the global sanitized 500 handler. The control plane invokes the sink at the required transaction boundary; the route must not add a post-commit success audit. Neither router performs SQL or imports `httpx` or the legacy service.

```python
def _require_webhook_mutation_actor(principal: AuthPrincipal) -> int:
    _require_platform_admin(principal)
    if principal.user_id is None:
        raise WebhookError("admin_webhook_user_principal_required", 403)
    return int(principal.user_id)

def _build_webhook_audit_sink(
    *,
    request_id: str,
    principal: AuthPrincipal,
    actor_id: int,
) -> MutationAuditSink:
    principal_id = validate_actor_principal_id(principal.principal_id)
    actor_kind = validate_actor_kind(principal.kind)
    actor_roles = validate_actor_roles(tuple(principal.roles))

    async def sink(record: MutationAudit) -> None:
        if record.actor_id != actor_id or record.request_id != request_id:
            raise MandatoryAuditWriteError("Mandatory audit identity mismatch")
        await emit_mandatory_webhook_audit(
            record,
            actor_principal_id=principal_id,
            actor_kind=actor_kind,
            actor_roles=actor_roles,
        )

    return sink

@router.post("/webhooks", response_model=WebhookSecretResponse, status_code=201)
async def create_webhook(
    payload: WebhookCreateRequest,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> WebhookSecretResponse:
    actor_id = _require_webhook_mutation_actor(principal)
    request_id = normalize_request_id(getattr(request.state, "request_id", None))
    audit_sink = _build_webhook_audit_sink(
        request_id=request_id,
        principal=principal,
        actor_id=actor_id,
    )
    result = await get_admin_webhook_control_plane().create(
        CreateRegistrationCommand(
            actor_id=actor_id,
            idempotency_key=idempotency_key,
            url=payload.url,
            event_types=tuple(payload.event_types),
            description=payload.description,
            timeout_seconds=payload.timeout_seconds,
            request_id=request_id,
            now=datetime.now(timezone.utc),
        ),
        audit_sink=audit_sink,
    )
    response.headers["ETag"] = build_registration_etag(
        webhook_id=result.registration.id,
        revision=result.registration.revision,
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return WebhookSecretResponse(
        registration=to_registration_response(result.registration),
        signing_secret=result.secret,
        replayed=result.replayed,
    )
```

Authentication and platform-role denials remain owned by the existing authentication/authorization audit path because no webhook mutation starts. Validation, service, and commit failures after a user-backed mutation actor is established are audited by the control-plane sink protocol, so the route does not duplicate them. PATCH and DELETE pass raw `If-Match` to domain parsing. Rotate requires both headers. Canonical routes for test, deliveries, and redelivery do not exist in PR 1.

After authorization, status/catalog/list/get handlers invoke the existing
best-effort admin audit helper with `resource_type="admin_webhook"`, a closed
read action, sanitized request ID, outcome, and only numeric resource ID,
target hostname for a single get, or result count for a list. They never pass
the response object, event set, full target, or arbitrary metadata, and audit
failure does not alter the read response.

- [ ] **Step 6: Extract legacy routes and mount exactly one implementation**

Keep `admin_ops.router` for non-webhook system operations. Add `admin_ops.legacy_webhooks_router` and move the existing string-ID CRUD, delivery-list, test, and `/incidents/{incident_id}/notify-webhooks` decorators onto it without changing compatibility API or dispatch behavior. Narrow the legacy create/update audit metadata to integer `event_count` and boolean `enabled`; it must not contain the URL, target host/path/query, event values, secret, or payload. In `admin/__init__.py`, always include `admin_ops.router` and `admin_webhooks.status_router`, then include either `legacy_webhooks_router` or `admin_webhooks.canonical_router` from one startup selector.

```python
settings = AdminWebhookSettings.from_environment(os.environ)
router.include_router(admin_webhooks_endpoints.status_router)
if settings.route_selection is WebhookRouteSelection.LEGACY:
    router.include_router(admin_ops_endpoints.legacy_webhooks_router)
else:
    router.include_router(admin_webhooks_endpoints.canonical_router)
```

The default is canonical routing with mode `off`, which exposes authorized status and fails closed elsewhere. Do not mount legacy incident notification alongside canonical mode; PR 3 later replaces it with the durable producer.

For canonical/`off`, emit one fixed startup warning stating that historical
webhook CRUD, test, delivery, and incident-notify routes are disabled and that
the operator must explicitly select temporary compatibility or migration. The
warning contains no source count, ID, URL, path, secret, environment value, or
exception text and is asserted once per application construction.

- [ ] **Step 7: Prove route uniqueness in all mode combinations**

```python
def test_selected_router_has_no_duplicate_method_path_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    for legacy in ("false", "true"):
        app = build_admin_app(monkeypatch, mode="off", legacy_compat=legacy)
        pairs = [(method, route.path) for route in app.routes for method in route.methods or set()]
        assert len(pairs) == len(set(pairs))
```

Also assert canonical static routes resolve to their handlers rather than integer parsing; both valid selections expose exactly one authorized status route with the correct explicit `route_selection`; legacy selection has its old method/path set and no canonical catalog/rotate route; canonical selection has no legacy test/delivery/notify handler; canonical/`off` emits exactly one fixed redacted compatibility warning; `legacy_compat=true` with `migrate` or `on` and every other invalid environment fail startup with a sanitized configuration error.

- [ ] **Step 8: Run API suites and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_route_selection.py \
  tldw_Server_API/tests/Admin_Webhooks/test_openapi.py \
  tldw_Server_API/tests/Admin/test_admin_webhooks_service.py \
  tldw_Server_API/tests/Admin/test_admin_ops_webhooks_reports.py \
  tldw_Server_API/tests/Workflows/test_webhook_admin_endpoints.py
if rg -n "admin_webhooks_service" \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py; then
  printf 'forbidden legacy service import found\n' >&2
  exit 1
fi
if rg -n "admin_webhook_secrets" \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py \
  --glob '!legacy_import.py'; then
  printf 'historical secret helper escaped migration boundary\n' >&2
  exit 1
fi
if rg -n '^[[:space:]]*(from|import)[[:space:]]+(aiohttp|httpx|requests|urllib|http\.client|socket)\b|tldw_Server_API\.app\.core\.Jobs' \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py; then
  printf 'PR 1 network or Jobs dependency found\n' >&2
  exit 1
fi
git add tldw_Server_API/app/api/v1/schemas/admin_webhooks.py \
  tldw_Server_API/app/api/v1/schemas/admin_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py \
  tldw_Server_API/app/api/v1/endpoints/admin/__init__.py \
  tldw_Server_API/tests/Admin/test_admin_webhooks_service.py \
  tldw_Server_API/tests/Admin/test_admin_webhooks_schemas.py \
  tldw_Server_API/tests/Admin_Webhooks/test_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_route_selection.py \
  tldw_Server_API/tests/Admin_Webhooks/test_openapi.py \
  "backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md"
git commit -m "feat(webhooks): expose canonical admin control plane"
```

Expected: API/compatibility tests PASS and the forbidden-dependency scans print no matches. `legacy_import.py` is the only permitted explicit bridge to the historical secret decryptor, but it receives no exception from the PR 1 network/Jobs scan.

### Task 9: Carry ETags And Idempotency Through The Admin UI Transport

**Files:**
- Modify: `admin-ui/lib/http.ts`
- Modify: `admin-ui/lib/server-auth.ts`
- Modify: `admin-ui/app/api/proxy/__tests__/route.test.ts`
- Modify: `admin-ui/lib/api-client.ts`
- Modify: `admin-ui/types/webhooks.ts`
- Modify: `admin-ui/types/index.ts`
- Create: `admin-ui/lib/idempotent-command.ts`
- Create: `admin-ui/lib/__tests__/idempotent-command.test.ts`

**Interfaces:**
- Consumes: canonical API JSON and headers from Task 8, browser `crypto.getRandomValues()`, and the existing authenticated Next.js proxy.
- Produces: `JsonResponse<T>`, `requestJsonWithMetadata<T>()`, one canonical webhook type family, ETag-aware API methods, `detectWebhookApi()`, `IdempotentCommand<T>`, and `createIdempotentCommand()`.

- [ ] **Step 1: Write failing proxy-header and metadata tests**

```typescript
it('forwards conditional/idempotency headers and preserves ETag', async () => {
  backendFetch.mockResolvedValue(new Response('{"id":41}', {
    status: 200,
    headers: { 'content-type': 'application/json', etag: '"admin-webhook-41-r2"' },
  }));
  const response = await PATCH(proxyRequest({
    'if-match': '"admin-webhook-41-r1"',
    'idempotency-key': '0123456789abcdef',
  }));
  expect(backendFetch.mock.calls[0][1].headers.get('if-match')).toBe('"admin-webhook-41-r1"');
  expect(backendFetch.mock.calls[0][1].headers.get('idempotency-key')).toBe('0123456789abcdef');
  expect(response.headers.get('etag')).toBe('"admin-webhook-41-r2"');
});
```

Cover both request headers independently, no client override of backend authorization, ETag exposure on success, typed contract failure when get/create/PATCH/rotate omits or malforms its required strong ETag, preservation of `Cache-Control: no-store`, `Pragma: no-cache`, and `X-Request-ID` on success and errors, exact `WebhookErrorResponse` parsing on 412/422/428/409/503, and unchanged existing `requestJson()` behavior. Send URL/query/forbidden-secret/idempotency-header canaries through the proxy's 422 fixture and prove none appears in its response headers/body or captured server logs; the proxy must not log forwarded conditional or idempotency header values.

- [ ] **Step 2: Forward the two safe request headers and expose response metadata**

Add `if-match` and `idempotency-key` to the explicit `appendProxyHeaders()` allowlist. Keep cookie-derived authorization authoritative. Add:

```typescript
export type JsonResponse<T> = {
  data: T;
  status: number;
  etag: string | null;
  requestId: string | null;
};

export const requestJsonWithMetadata = async <T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<JsonResponse<T>> => {
  const response = await requestResponse(endpoint, options);
  const data = await parseJsonResponse<T>(response);
  return { data, status: response.status, etag: response.headers.get('etag'), requestId: response.headers.get('x-request-id') };
};
```

Refactor shared fetch/error parsing once so `requestJson()` still returns only `T`; do not duplicate auth/logout/CSRF behavior. Parse the canonical error envelope into a bounded typed client error carrying status, code, fixed message, and sanitized request ID without retaining raw response text. Preserve existing non-canonical error behavior for other admin endpoints.

- [ ] **Step 3: Consolidate canonical and compatibility DTOs**

`admin-ui/types/webhooks.ts` becomes authoritative for canonical `WebhookRegistration`, `WebhookCreateRequest`, `WebhookPatchRequest`, `WebhookSecretResponse`, `WebhookCatalog`, `WebhookStatus`, and list response. `WebhookStatus` includes `legacy_file_restore_permitted: boolean` and nullable `rollback_window_expires_at`, but no artifact path. Registration has numeric ID and no URL/secret property. Remove the duplicate `AdminWebhook` interfaces from `types/index.ts` and re-export the focused file.

Keep `LegacyWebhookDto` private to `api-client.ts` with string ID, URL, events, and enabled fields. Its adapter produces a separate `LegacyWebhookView`; never cast it to `WebhookRegistration`, derive a canonical ETag, or expose rotate as supported.

- [ ] **Step 4: Write failing command-key lifecycle tests**

```typescript
it('reuses one key only when retrying the same command object', async () => {
  const request = vi.fn()
    .mockRejectedValueOnce(new TypeError('network'))
    .mockResolvedValueOnce({ registration: fixture, signing_secret: `whsec_${'a'.repeat(64)}`, replayed: true });
  const command = createIdempotentCommand('create', canonicalBody, request);
  await expect(command.run()).rejects.toThrow('network');
  await command.retry();
  expect(request.mock.calls[0][0].idempotencyKey).toBe(request.mock.calls[1][0].idempotencyKey);
  expect(createIdempotentCommand('create', canonicalBody, request).idempotencyKey).not.toBe(command.idempotencyKey);
});
```

Also prove 16 random bytes become 32 lowercase hex characters, normalized request cannot change on retry, non-transport HTTP errors are not automatically retried, completion clears retry eligibility, and the module never calls local/session storage or writes the key into a URL.

- [ ] **Step 5: Implement canonical API client methods**

```typescript
getWebhookStatus: () => requestJson<WebhookStatus>('/admin/webhooks/status'),
getWebhookCatalog: () => requestJson<WebhookCatalog>('/admin/webhooks/catalog'),
getWebhook: (id: number) => requestJsonWithMetadata<WebhookRegistration>(`/admin/webhooks/${id}`),
createWebhook: (body: WebhookCreateRequest, key: string) => requestJsonWithMetadata<WebhookSecretResponse>(
  '/admin/webhooks', { method: 'POST', headers: { 'Idempotency-Key': key }, body: JSON.stringify(body) },
),
rotateWebhookSecret: (id: number, etag: string, key: string) => requestJsonWithMetadata<WebhookSecretResponse>(
  `/admin/webhooks/${id}/rotate-secret`, { method: 'POST', headers: { 'If-Match': etag, 'Idempotency-Key': key } },
),
```

PATCH/DELETE require caller-provided ETag. The client has no canonical test/deliveries/redelivery methods in PR 1. Add an explicit `detectWebhookApi()` adapter: a successful validated status response selects the canonical or typed legacy client only from its `route_selection` field. A 404, transport failure, malformed body, unknown selection, or any other error remains visible and never downgrades to legacy behavior.

Wrap get/create/PATCH/rotate with `requireStrongWebhookEtag()`: accept only the
regular expression `^"admin-webhook-([1-9][0-9]*)-r([1-9][0-9]*)"$`, with the
two captured integers matching the response registration ID and revision. A missing, weak, malformed, or mismatched
ETag is a typed contract error; do not cache it or enable a follow-up mutation.
DELETE validates success status but has no response ETag.

- [ ] **Step 6: Run transport/type tests and commit**

```bash
cd admin-ui
bun run test -- app/api/proxy/__tests__/route.test.ts lib/__tests__/idempotent-command.test.ts
bun run typecheck
cd ..
git add admin-ui/lib/http.ts admin-ui/lib/server-auth.ts \
  admin-ui/app/api/proxy/__tests__/route.test.ts \
  admin-ui/lib/api-client.ts admin-ui/types/webhooks.ts admin-ui/types/index.ts \
  admin-ui/lib/idempotent-command.ts admin-ui/lib/__tests__/idempotent-command.test.ts \
  "backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md"
git commit -m "feat(admin-ui): support conditional webhook commands"
```

### Task 10: Rebuild The Admin Webhooks Page Around The Canonical Workflow

**Files:**
- Modify: `admin-ui/app/webhooks/page.tsx`
- Modify: `admin-ui/app/webhooks/__tests__/page.test.tsx`
- Create: `admin-ui/tests/e2e/webhooks-control-plane.spec.ts`

**Interfaces:**
- Consumes: Task 9 API adapter/types/command helper and existing permission, dialog, toast, form, table, alert, badge, and icon components.
- Produces: a catalog-driven canonical control-plane page with explicit degraded/legacy states, inactive creation, fresh-ETag mutation, rotation, and one-time secret acknowledgement.

- [ ] **Step 1: Replace stale fixtures with failing canonical workflow tests**

```typescript
it('creates inactive from the server catalog and requires secret acknowledgement', async () => {
  apiMock.getWebhookStatus.mockResolvedValue(readyStatus);
  apiMock.getWebhookCatalog.mockResolvedValue(canonicalCatalog);
  apiMock.createWebhook.mockResolvedValue({ data: secretResponse, etag: '"admin-webhook-41-r1"', status: 201, requestId: 'req-1' });
  render(<WebhooksPage />);
  await user.click(await screen.findByRole('button', { name: /add webhook/i }));
  await user.type(screen.getByLabelText(/destination url/i), 'https://receiver.example/private-hook');
  await user.click(screen.getByLabelText('user.created'));
  await user.click(screen.getByRole('button', { name: /^create$/i }));
  expect(apiMock.createWebhook.mock.calls[0][0]).not.toHaveProperty('active');
  expect(screen.getByText(secretResponse.signing_secret)).toBeVisible();
  expect(screen.getByRole('button', { name: /done/i })).toBeDisabled();
});
```

Cover catalog loading/no hardcoded event list, migration/key/limit/delivery-unavailable status, imported-secret-rotation status/badge/action, redacted target display, explicit blank destination replacement with no redacted-display prefill/submission, metadata-only PATCH omitting `url`, server-total-driven previous/next pagination with deterministic page contents, no delivery history/test controls in canonical PR 1, typed legacy compatibility banner and old controls, create description/timeout, one-time copy and acknowledgement, close/navigation warning, no secret after dismissal, `pagehide` plus persisted back-forward-cache `pageshow` clearing, same-command retry after transport loss, page reload discovery and rotate guidance, rotate inactive only, fresh GET before PATCH/delete/rotate, 428 handling, 412 reload plus explicit re-review, no automatic mutation against the new ETag, activation failure while delivery capability unavailable or signing-secret rotation is required, no secret/URL in toast/error/log, and platform-admin permission guard.

- [ ] **Step 2: Implement status-first mode selection and catalog-driven form**

Load status first. In canonical mode, render actionable full-width status alerts for `off`, `migrate`, `migration_pending`, `key_unavailable`, over-limit, and delivery capability unavailable. Show the rollback-window state in the migration/status area without exposing artifact paths; after the first effective canonical mutation, label legacy restore unavailable and direct operators to forward-fix documentation. Fetch catalog/list only when their mode gates permit. Use catalog event descriptions and effective limits from the server. Render compact previous/next controls from the returned `total/limit/offset`, reset to the first page after create/delete, and refetch the current page after metadata mutation without client-side reordering. Creation contains URL, description, explicit event checkboxes, and timeout; it has no active, secret, retry-count, custom-header, method, payload, or wildcard control.

When status explicitly reports `route_selection=legacy`, use the dedicated legacy adapter and show `Legacy compatibility mode`; do not present ETags, rotate, or canonical readiness as available. Any status failure or unknown selection renders the actual bounded error and retry control without probing legacy CRUD routes.

- [ ] **Step 3: Implement one-time create/rotate secret state**

Keep `signing_secret`, command object, copied state, and acknowledgement state only in component memory. The dialog uses the existing copy icon/button and a required acknowledgement checkbox before dismissal. One idempotent `clearSensitiveCommandState()` clears both React state and mutable references. On acknowledged close, use the normal state path. While sensitive state exists, its `pagehide` listener must clear references and synchronously flush the state removal before returning; a persisted `pageshow` listener clears again as defense in depth. Do not rely on an asynchronous state update after the page enters the back-forward cache. Component and Playwright tests dispatch `pagehide` and assert the secret text/retry action is absent immediately, before dispatching `pageshow`, then repeat the assertion after a persisted restore. During an unresolved transport failure, offer `Retry same command`, which calls only `command.retry()`. On reload/navigation loss, list the inactive record and provide `Generate a new secret` through rotate; never promise retrieval of the original.

- [ ] **Step 4: Implement fresh-ETag mutation review**

Before PATCH, DELETE, or rotate, call `getWebhook(id)` and retain its response ETag only for that visible command. Show the fetched current metadata in the existing privileged-action dialog. Metadata edits omit `url`. Destination replacement is a distinct command whose URL field starts blank, labels the current redacted host as context, and requires a complete new URL plus explicit confirmation; never put `target_display` into that field or request body. Send exactly the fresh ETag. On 412, discard it, refetch, display what changed, and require a new click/confirmation. Do not auto-retry a conditional mutation. Activation remains visible but disabled with the server's `delivery_capability_ready=false` reason until PR 2/3 readiness exists.

- [ ] **Step 5: Run component tests and repair accessibility failures**

```bash
cd admin-ui
bun run test -- app/webhooks/__tests__/page.test.tsx
bun run typecheck
bun run lint -- app/webhooks/page.tsx app/webhooks/__tests__/page.test.tsx \
  lib/idempotent-command.ts lib/api-client.ts types/webhooks.ts
cd ..
```

Expected: PASS with no React state-update warnings. Labels, dialog focus, keyboard copy/acknowledgement, destructive confirmations, status announcements, and icon-button accessible names are covered.

- [ ] **Step 6: Add a mocked-browser control-plane journey**

The Playwright test uses the existing `setAuthenticatedSession()` and admin
API route-helper pattern, then intercepts
`/api/proxy/admin/webhooks/status`, `/catalog`,
list/get/create/rotate/PATCH/delete and models ETag changes. It verifies
create-inactive -> copy -> acknowledge -> list -> fetch-current -> edit; a
pagehide/persisted-pageshow cycle cannot restore secret text or retry state; an
injected 412 forces re-review; a lost rotate response retries with the same
idempotency key and reveals the same secret. Inspect captured requests to prove
no secret enters URL/query/localStorage/sessionStorage and no canonical
test/delivery route is requested.

```bash
cd admin-ui
bunx playwright test tests/e2e/webhooks-control-plane.spec.ts --reporter=line
cd ..
```

- [ ] **Step 7: Commit the page workflow**

```bash
git add admin-ui/app/webhooks/page.tsx \
  admin-ui/app/webhooks/__tests__/page.test.tsx \
  admin-ui/tests/e2e/webhooks-control-plane.spec.ts \
  "backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md"
git commit -m "feat(admin-ui): manage canonical webhook registrations"
```

### Task 11: Document Operations And Run The PR 1 Release Gates

**Files:**
- Create: `Docs/Admin_Webhooks_Control_Plane.md`
- Create: `Docs/Admin_Webhooks_Migration_Runbook.md`
- Create: `Docs/Admin_Webhooks_Key_Rotation_Runbook.md`
- Create: `Docs/Evidence/Admin_Webhooks_PR1_Verification.md`
- Modify: `admin-ui/docs/feature-guides/webhooks.md`
- Modify: `apps/tldw-frontend/lib/api/openapi.fingerprint.json`
- Modify: `backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md`

**Interfaces:**
- Consumes: all PR 1 implementation and verification output.
- Produces: deployable operator documentation, reviewed OpenAPI fingerprint, recorded evidence, and a review-ready branch that remains default-off.

- [ ] **Step 1: Write the control-plane and environment reference**

Document exact mode/compatibility/limit/key environment variables, default values, canonical endpoint tables, header/error contracts, one-time-secret behavior, redaction, platform-admin requirement, and the explicit statement that PR 1 cannot activate delivery. The environment table includes `TLDW_ADMIN_WEBHOOK_ROLLBACK_WINDOW_DAYS` (default 7, range 1-30), `TLDW_ADMIN_WEBHOOKS_ALLOW_HTTP_DEV`, and the composed `EGRESS_ALLOWLIST`/`EGRESS_DENYLIST`, `WORKFLOWS_EGRESS_ALLOWLIST`/`WORKFLOWS_EGRESS_DENYLIST`, and `WORKFLOWS_WEBHOOK_ALLOWLIST`/`WORKFLOWS_WEBHOOK_DENYLIST` policy families with deny precedence. Include an upgrade warning that the default canonical/`off` selector disables every historical webhook and incident-notify route, with explicit temporary-compatibility and migration choices; never imply 404 fallback. Include one create and rotate example using generated 32-hex idempotency keys and response ETags; examples use reserved `.example` hosts and fake secrets only.

- [ ] **Step 2: Write the migration and rollback runbook**

Document preflight, all-node drain or canonical/`migrate` rollout, per-node status and legacy-route-unreachability evidence, the `--all-writers-quiesced` acknowledgement, dry-run command, report review, separately recorded literal digest, digest-approved apply that never derives its approval argument from the report at execution time, encrypted backup/readback evidence, source mapping, sanitization, rerun after each crash boundary, unresolved/rejected records, seven-day rollback key lifecycle, 30-day ceiling, backup extraction, backup-key destruction, and rollback boundaries. State explicitly that a single-node environment change is insufficient in a multi-process deployment and fingerprint checks detect but do not authorize concurrent legacy writes. Before file recovery, require status to report `legacy_file_restore_permitted=true`; extraction independently enforces the same retained/unexpired/no-activity conditions. The recovery procedure must stop/quiesce all writers, set canonical mode off, use `extract-rollback-backup` to a new `0600` file, compare it with the current strict snapshot, and merge only top-level `webhooks`/`webhook_deliveries` under the existing lock/atomic writer; it explicitly forbids whole-file replacement, stdout extraction, reuse of the plaintext output, or deletion of canonical tables. Require a second-person review before the structural merge and document secure deletion limitations for the extracted file. State that after any canonical mutation or delivery the durable activity marker closes extraction and the operator disables and forward-fixes rather than restoring the legacy writer.

- [ ] **Step 3: Write the key provisioning and rotation runbook**

Document JSON key-ring syntax using fake random values, durable active-primary state, start/status/resume/verify commands while the old key remains local primary, the `awaiting_primary_cutover` deployment step, finalize after every process uses the target primary, lagging-node mismatch behavior, maintenance impact on all protected writes, source-key retention through final zero-envelope verification, file-marker scanning, crash recovery, and old-key removal. Explicitly prohibit BYOK/session/JWT/API-key fallback and plaintext emergency writes.

- [ ] **Step 4: Refresh and review OpenAPI drift**

```bash
make openapi-fingerprint
make openapi-drift-check
git diff -- apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

Expected: only the canonical PR 1 webhook schema/path changes and removals of the duplicate numeric/delivery endpoints appear. Any unrelated API drift is investigated before staging.

- [ ] **Step 5: Run the complete Python PR 1 matrix**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks \
  tldw_Server_API/tests/Admin/test_admin_system_ops_service.py \
  tldw_Server_API/tests/Admin/test_admin_webhooks_service.py \
  tldw_Server_API/tests/Admin/test_admin_webhooks_schemas.py \
  tldw_Server_API/tests/Admin/test_admin_ops_webhooks_reports.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_admin_webhook_migration_sqlite.py \
  tldw_Server_API/tests/Security/test_egress.py \
  tldw_Server_API/tests/Workflows/test_webhook_admin_endpoints.py
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_legacy_import_postgres.py
```

Expected: all SQLite/API/import/crypto tests PASS and the required disposable PostgreSQL command PASS without a database-availability skip. Record exact counts and any unrelated justified environment skips in `TASK-13014`; PostgreSQL absence blocks merge.

- [ ] **Step 6: Run static and secret-leak gates**

```bash
../../.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/core/Security/egress.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/services/admin_system_ops_service.py \
  tldw_Server_API/app/api/v1/schemas/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py \
  tldw_Server_API/app/api/v1/endpoints/admin/__init__.py \
  tldw_Server_API/cli/admin_webhooks_cli.py \
  tldw_Server_API/cli/commands/admin_webhooks.py
../../.venv/bin/python -m bandit -q -r \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/core/Security/egress.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/services/admin_system_ops_service.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py \
  tldw_Server_API/app/api/v1/endpoints/admin/__init__.py \
  tldw_Server_API/cli/admin_webhooks_cli.py \
  tldw_Server_API/cli/commands/admin_webhooks.py
if rg -n "logger\..*(url|secret|payload|response_body)|metadata=.*(url|secret|payload)" \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/core/Security/egress.py \
  tldw_Server_API/app/services/admin_system_ops_service.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py; then
  printf 'sensitive canonical log/audit pattern found\n' >&2
  exit 1
fi
if git diff --unified=0 origin/dev -- \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py | \
  rg '^\+.*(logger\..*(url|secret|payload|response_body)|metadata=.*(url|secret|payload))'; then
  printf 'new sensitive compatibility log/audit pattern found\n' >&2
  exit 1
fi
```

Expected: Ruff and Bandit PASS; the scan finds no full URL, secret, payload, response body, or incident narrative passed to logs/audit.

- [ ] **Step 7: Run the complete admin UI gate**

```bash
cd admin-ui
bun run test
bun run typecheck
bun run lint
bun run build
bunx playwright test tests/e2e/webhooks-control-plane.spec.ts --reporter=line
if rg -n 'localStorage|sessionStorage|document\.cookie|console\.' \
  app/webhooks/page.tsx lib/idempotent-command.ts lib/api-client.ts types/webhooks.ts; then
  printf 'webhook UI persistence or console sink found\n' >&2
  exit 1
fi
cd ..
```

Expected: every command PASS and the sensitive-state sink scan prints no matches. Do not waive production build or the focused Playwright journey because component mocks alone cannot prove proxy/header transport.

- [ ] **Step 8: Record exact evidence and update the Backlog task**

Create `Docs/Evidence/Admin_Webhooks_PR1_Verification.md` with the tested commit SHA, UTC timestamp, OS/Python/Node/Bun/PostgreSQL versions, every command from Steps 4-7, exact pass counts, zero unexpected skips, OpenAPI paths reviewed, Ruff/Bandit results, secret-log scan result, and any resolved failure with its fixing commit. Use `Not run` only while the branch remains In Progress; no acceptance criterion may be checked while its required gate says `Not run`.

```bash
git diff --check
git status --short
git log --oneline --decorate origin/dev..HEAD
backlog task edit 13014 --append-notes $'PR 1 verification evidence: Docs/Evidence/Admin_Webhooks_PR1_Verification.md\nAll required SQLite, disposable PostgreSQL, API, admin UI, OpenAPI, Ruff, Bandit, and secret-redaction gates are recorded there with exact results.'
backlog task 13014 --plain
```

Keep the task In Progress until review is resolved and all acceptance criteria are demonstrably complete.

- [ ] **Step 9: Commit documentation and verification records**

```bash
git add Docs/Admin_Webhooks_Control_Plane.md \
  Docs/Admin_Webhooks_Migration_Runbook.md \
  Docs/Admin_Webhooks_Key_Rotation_Runbook.md \
  Docs/Evidence/Admin_Webhooks_PR1_Verification.md \
  admin-ui/docs/feature-guides/webhooks.md \
  apps/tldw-frontend/lib/api/openapi.fingerprint.json \
  "backlog/tasks/task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md"
git commit -m "docs(webhooks): add control-plane operations runbooks"
git status --short
```

Expected: clean status.

- [ ] **Step 10: Request human review before opening the stacked delivery PR**

Prepare a human-written `Change summary:` describing behavior and risk, not a file list. Request review of migration crash safety, dedicated-key isolation, SQLite/PostgreSQL parity, idempotency ordering, conditional mutation behavior, route exclusivity, authorization/audit redaction, and browser secret handling. Resolve findings and rerun affected gates. Do not start PR 2 from an unreviewed or failing PR 1 branch.

## Completion Boundary

PR 1 is complete only when the canonical control-plane and migration behavior above passes on SQLite and disposable PostgreSQL, the admin UI passes its production build and focused browser journey, the OpenAPI delta is reviewed, and the branch still defaults to canonical mode `off`. Completion does not authorize outbound delivery or public release. PR 2 must implement and prove the delivery substrate; PR 3 must connect durable producers, remove compatibility routing, and pass the final activation gate.
