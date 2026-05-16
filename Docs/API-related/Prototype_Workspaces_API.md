# Prototype Workspaces API

## Overview

`Prototype Workspace` is the app-prototyping collaboration surface built on top of the existing ACP and Sandbox infrastructure.

The current slice supports:

- owner-created prototype workspaces with a canonical snapshot
- isolated owner and external collaborator branch sessions
- private-link exchange into a `Prototype Shared Actor`
- brokered preview grants and renewals
- promotion request submission and owner review
- a minimal WebUI route at `/prototype-workspaces`

The product model stays explicit:

- `Prototype Workspace`: the canonical artifact and promotion boundary
- `Prototype Session`: an isolated owner or collaborator branch
- `Prototype Snapshot`: a saved revision in the workspace lineage
- `Prototype Shared Actor`: an external collaborator identity created from a private link
- `Promotion Request`: a request to move a candidate snapshot into the canonical line

ACP and Sandbox remain execution backends, not the user-facing product model.

## Lifecycle

1. The owner creates a prototype workspace with a title, prompt, and runtime/share policy.
2. The service seeds the workspace with an initial canonical snapshot.
3. The owner can request an owner branch session for guided work on top of the canonical snapshot.
4. The owner can create a private share token for `prototype_workspace`.
5. A stakeholder opens `/share/:token`, verifies any password, and exchanges the token for a `Prototype Shared Actor` session token.
6. The collaborator uses that session token to create or reuse an isolated branch session.
7. The collaborator produces a candidate snapshot and submits a promotion request.
8. The owner reviews the request and either rejects it or promotes the candidate after validation.
9. Preview access is always brokered through a preview handle rather than exposing raw runtime URLs.

## Runtime Profiles

The design references several runtime policy profiles:

- `template_demo`: constrained demo-oriented runtime for seeded prototypes
- `repo_bootstrap`: profile for repository-backed scaffolds or richer bootstrap flows
- `locked_collab`: the default external collaborator runtime profile

The current code path actively uses these runtime-policy keys:

- `runtime_policy.owner_profile`
- `runtime_policy.external_collaborator_profile`
- preview-broker defaults such as `canonical_preview` and `owner_collab`

Important implementation note:

- external collaborator exchange defaults to `locked_collab` when no explicit external profile is set
- preview brokering resolves owner vs external runtime profiles from workspace/runtime context instead of trusting the client

## API Surface

### Create workspace

`POST /api/v1/prototype-workspaces`

Creates a new prototype workspace and its seed canonical snapshot.

Request highlights:

- `title`
- `creation_source`
- `description`
- `prompt`
- `preview_policy`
- `share_policy`
- `runtime_policy`
- `designated_promoter_ids`

Response highlights:

- workspace metadata
- canonical and last-known-good snapshot ids
- canonical preview and publish-validation state

### Get owner detail

`GET /api/v1/prototype-workspaces/{prototype_workspace_id}`

Owner-only detail response used by the minimal WebUI.

Response includes:

- workspace metadata
- `viewer_role`
- branch-session inventory
- snapshot inventory
- canonical and last-known-good markers on snapshots

### Create owner branch session

`POST /api/v1/prototype-workspaces/{prototype_workspace_id}/sessions`

Owner-only route that creates or reuses a branch session rooted at the canonical snapshot and enqueues bootstrap work.

Response highlights:

- `job_id`
- `job_type = "branch_session_bootstrap"`
- `prototype_session_id`
- `actor_type = "owner"`

### Create collaborator branch session

`POST /api/v1/prototype-sessions`

Consumes a collaborator `session_token` minted by the private-link exchange flow and creates or reuses the stakeholder branch session.

Response highlights:

- `prototype_workspace_id`
- `prototype_session_id`
- `shared_actor_id`
- `actor_type = "external_collaborator"`

### Submit promotion request

`POST /api/v1/prototype-promotions`

Creates a promotion request for a candidate snapshot.

Server-side validation binds the request to:

- the prototype workspace in the session token
- the branch session referenced by the request
- the candidate snapshot lineage
- the `Prototype Shared Actor` that owns the branch and snapshot

### Review promotion request

`POST /api/v1/prototype-promotions/{promotion_request_id}/review`

Owner-only approval or rejection flow.

Approval path:

- validates the requested baseline
- runs promotion validation
- updates canonical pointers only on success
- can return a refreshed preview handle for the promoted snapshot

Rejection path:

- records reviewer id and notes
- does not advance canonical state

### Renew preview grant

`POST /api/v1/prototype-previews/{preview_handle}/renew`

Renews a brokered preview grant for an active preview handle.

Response highlights:

- short-lived preview token
- preview URL
- expiry
- resolved runtime policy profile

### Private-link exchange

`POST /api/v1/sharing/public/{token}/prototype-session`

Turns a private share link into a `Prototype Shared Actor` and collaborator session token.

Request highlights:

- `display_name`
- `password` when the token is protected

Response highlights:

- `shared_actor_id`
- `actor_type = "external_collaborator"`
- `session_token`
- `runtime_policy_profile`

Expected failure states:

- Invalid, expired, exhausted, archived, or mismatched links: non-enumerating 404 or 403 `PrototypeErrorResponse`.
- Missing required FastAPI request fields: 422 `HTTPValidationError`.
- Domain-invalid request shape, such as a missing first-use `display_name`: 422 `PrototypeErrorResponse`.
- Public exchange rate-limit exhaustion: 429.

## Prototype Shared Actor Model

External stakeholders do not become first-class AuthNZ users.

Instead, the exchange flow creates or resumes a `Prototype Shared Actor` that is scoped to:

- one prototype workspace
- one share-link id
- a session binding / resume cookie chain
- an external runtime policy profile

This preserves auditability and keeps owner-only controls separate from collaborator capabilities.

## Risk Gate 1 Security Contract

The current productionization security model is documented in `Docs/Security/Prototype_Workspaces_Threat_Model.md`.

The short version:

- owner APIs must re-check AuthNZ owner or designated-promoter authority server-side
- public share exchange must keep invalid, expired, revoked, exhausted, missing, and mismatched link states non-enumerating where possible
- external shared actors are scoped to one prototype workspace and one share-link id
- collaborator session tokens must be rechecked against active shared-actor state, not trusted only because their signature is valid
- resume cookies are signed browser-binding hints and rotate their stored binding secret on successful resume
- preview grants are short-lived broker-issued grants behind opaque handles
- promotion submission must bind the token, active shared actor, branch session, candidate snapshot, and workspace before creating a request

Frontend/backend state names, HTTP status expectations, retryability, and frozen Risk Gate 4 decisions live in `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`.

## Risk Gate 4 Error Contract

Prototype-specific endpoint failures use the normal FastAPI `detail` envelope with a stable structured payload:

```json
{
  "detail": {
    "category": "inactive_session",
    "message": "Prototype session token is no longer active",
    "frontend_state": "session_inactive",
    "retryable": false
  }
}
```

Contract rules:

- `category` is the machine-readable backend error category.
- `frontend_state` is the machine-readable UI state bucket.
- `retryable` is the only field frontend retry affordances should branch on.
- `message` is safe fallback copy, but clients should not parse it.
- Invalid, expired, revoked, exhausted, missing, and owner-mismatched public links remain non-enumerating as `invalid_or_unavailable_link`.
- Active collaborator session-token paths use `inactive_session` when the signed token maps to a revoked or expired shared actor/session.
- Preview renewal returns `preview_unavailable` with 404 for missing/revoked handles and 409 for renewal conflicts.

The generated OpenAPI contract references `PrototypeErrorResponse` for prototype route 403, 404, and 409 responses where those statuses are expected. Prototype 422 entries allow either `PrototypeErrorResponse` or FastAPI's `HTTPValidationError` because malformed request bodies are rejected before endpoint code can wrap them in the domain error envelope.

## Lifecycle Examples

### Owner creates a workspace and branch session

1. `POST /api/v1/prototype-workspaces` with title, source, prompt, and policy objects.
2. The service creates the workspace and seed canonical snapshot in one transaction.
3. `POST /api/v1/prototype-workspaces/{prototype_workspace_id}/sessions` creates or reuses an owner branch session.
4. The response returns a `branch_session_bootstrap` job id and the branch session id.
5. Runtime bootstrap completion updates branch-session runtime fields and preview state through the Jobs-backed runtime path.

Expected failure states:

- Missing workspace: 404 `missing`, `frontend_state = "missing"`, not retryable.
- Non-owner caller: 403 `unauthorized`, not retryable.
- Missing canonical snapshot or stale bootstrap precondition: 409 `conflict`, not retryable.
- Runtime bootstrap failure: 409 `bootstrap_failed`; retry only when `retryable = true`.

### Owner creates a private link and collaborator enters

1. Owner creates a share token with `resource_type = "prototype_workspace"`.
2. Collaborator opens `/share/:token` and submits `POST /api/v1/sharing/public/{token}/prototype-session`.
3. Password-protected links require `password` unless a valid same-browser resume cookie exists.
4. First-time sessions require `display_name`.
5. The exchange returns a scoped `shared_actor_id`, `session_token`, and runtime policy profile.
6. The browser stores only the signed resume cookie; passwords and tokens should not be persisted in durable route state.

Expected failure states:

- Invalid, expired, revoked, exhausted, missing, or owner-mismatched public link: 404 `invalid_or_unavailable_link`.
- Missing password: 403 `password_required`, retryable.
- Bad password: 403 `invalid_password`, retryable.
- Archived workspace: 403 `workspace_unavailable`, not retryable.

### Collaborator creates a branch and submits promotion

1. Collaborator calls `POST /api/v1/prototype-sessions` with the exchange `session_token`.
2. The service rechecks that the token maps to an active shared actor and share-link id.
3. Collaborator work saves a candidate snapshot tied to the branch session.
4. Collaborator calls `POST /api/v1/prototype-promotions` with workspace, session, candidate snapshot, and session token.
5. The backend verifies workspace, session, snapshot lineage, share-link id, and shared actor ownership before creating the request.

Expected failure states:

- Expired or revoked actor/session: 403 `inactive_session`, not retryable.
- Token/workspace/session/snapshot mismatch: 403 `unauthorized` or 422 `invalid_request` depending on whether the caller is unauthorized or the request shape is inconsistent.
- Missing session or snapshot: 404 `missing`, not retryable.

### Owner reviews promotion and renews preview

1. Owner or designated promoter calls `POST /api/v1/prototype-promotions/{promotion_request_id}/review`.
2. Rejection records reviewer notes without changing canonical pointers.
3. Approval validates the candidate against the requested baseline and current canonical state.
4. Successful promotion advances canonical and last-known-good pointers and may return a preview handle.
5. Owner calls `POST /api/v1/prototype-previews/{preview_handle}/renew` to rotate the short-lived preview grant token.

Expected states:

- Unauthorized reviewer: 403 `unauthorized`, not retryable.
- Missing promotion request/workspace: 404 `missing`, not retryable.
- Stale candidate: 200 response with `status = "stale"` and `failure_code = "stale_candidate"`.
- Validation failure: 200 response with `status = "failed"` and `failure_code = "publish_validation_failed"`.
- Missing/revoked preview handle: 404 `preview_unavailable`, not retryable.
- Preview renewal conflict: 409 `preview_unavailable`, retryable.

## Risk Gate 2 Persistence Contract

Prototype workspace persistence is owned by the AuthNZ repository boundary in `PrototypeWorkspacesRepo`.

Transaction rules:

- workspace creation and seed snapshot creation run in one AuthNZ transaction
- session snapshot save and session pointer update run in one AuthNZ transaction
- publish promotion still compensates around preview-broker side effects because preview grants include in-memory broker state plus persisted handles
- repository transaction adapters preserve the existing `?` placeholder style and convert to PostgreSQL `$N` placeholders when the underlying `DatabasePool` is PostgreSQL-backed

Cleanup and retention rules:

- archived workspaces are soft-archived first and can be deleted after the configured archive-retention cutoff; cascading foreign keys remove related prototype rows
- expired shared actors are soft-revoked by setting `revoked_at`; they are retained for audit until their workspace is deleted
- expired sessions are soft-revoked by setting `revoked_at`, `runtime_status = "revoked"`, and `preview_status = "revoked"`
- active preview handles attached to archived workspaces or revoked/expired sessions are deactivated and receive `revoked_at`
- inactive preview handles can be deleted after their inactive-preview cutoff
- old pending promotion requests can be marked `stale`; approved, rejected, promoted, or already stale requests are not rewritten by cleanup

SQLite/PostgreSQL behavior:

- migration 086 is the SQLite migration source for the prototype tables
- PostgreSQL compatibility is through the AuthNZ `DatabasePool` execution contract and repository table discovery via `information_schema`
- all repository SQL remains parameterized; no user input is interpolated into SQL strings
- cleanup accepts explicit cutoff timestamps so callers own retention policy and scheduling

Query/index review:

| Path | Query shape | Index coverage |
| --- | --- | --- |
| owner workspace detail | `prototype_workspaces.id`, workspace snapshots by `prototype_workspace_id` | primary key plus `idx_prototype_snapshots_workspace_created` |
| branch-session inventory | sessions by `prototype_workspace_id`, active first, ordered by update time | `idx_prototype_sessions_workspace_active_updated` |
| active session reuse | workspace, base snapshot, actor type, actor identity, share link, revoked/expiry/runtime filters | `idx_prototype_sessions_active_lookup` |
| active shared actor check | workspace-scoped actor activity and expiry checks | primary key plus `idx_prototype_shared_actors_active_lookup` |
| promotion listings/review | workspace and status with update-time cleanup | `idx_prototype_promotion_requests_workspace_status_updated` |
| cleanup retention sweep | global archived, revoked, expired, and inactive-preview cutoffs | `idx_prototype_workspaces_archived_at_cleanup`, `idx_prototype_sessions_revoked_at_cleanup`, `idx_prototype_sessions_expires_at_cleanup`, `idx_prototype_shared_actors_expires_revoked_cleanup`, `idx_prototype_preview_handles_inactive_revoked_cleanup` |
| preview handle lookup | handle id, active scope replacement, workspace/session cleanup scans | primary key plus preview handle workspace/session/scope indexes |

## Risk Gate 3 Runtime Job Contract

Prototype runtime orchestration uses the shared Jobs module with domain `prototype_workspaces` and queue `default`.

Runtime job types:

- `branch_session_bootstrap`
- `preview_boot`
- `snapshot_save`
- `publish_validate_and_promote`

Worker result shape:

- successful job handlers return `status`, `job_type`, and `retryable`
- terminal publish outcomes also return stable `failure_code` and relevant snapshot ids
- payload errors use `failure_code = "invalid_job_payload"` and `retryable = false`
- expected permission failures use `failure_code = "permission_denied"` and `retryable = false`
- retryable runtime failures use `failure_code = "runtime_retryable"` and `retryable = true`
- terminal runtime failures such as archived workspaces, revoked actors, expired sessions, and missing resources use `failure_code = "runtime_terminal"` and `retryable = false`

Retry/idempotency guarantees:

- branch bootstrap jobs use workspace, actor, canonical snapshot, and request nonce in the idempotency key; repeated execution reuses an active compatible branch session
- preview boot jobs use scope, snapshot, runtime profile version, and target fingerprint in the idempotency key; repeated execution for the same active scope/snapshot/target/profile renews the existing handle instead of minting a second handle
- preview boot replacement with a different target revokes the previous active handle for the scope and rolls back to the previous active handle only when the scope is still unclaimed after persistence failure
- snapshot-save jobs use session and save request id in the idempotency key; repeated execution with the same explicit snapshot id returns the existing session-owned snapshot and preserves a single saved snapshot row
- publish validation and promotion jobs use workspace, candidate snapshot, and baseline snapshot in the idempotency key; stale or failed validation returns a terminal result without advancing canonical or last-known-good pointers

Cancellation and timeout boundary:

- queued prototype jobs inherit the shared Jobs cancellation behavior and can be cancelled before acquisition
- processing cancellation is best-effort at the shared worker boundary; the current Risk Gate 3 handlers are short transactional units and rely on idempotent retry/compensation rather than mid-operation interruption
- worker shutdown uses `WorkerSDK.stop()` via the injected stop event and does not acquire new prototype jobs after shutdown starts
- lease expiry and retry are owned by the shared Jobs manager; retry-safe service operations are the prototype-specific protection against duplicate effects after worker restart or completion-ack failure
- runtime host process termination, long-running sandbox teardown, and operator-facing timeout controls remain later runtime-hosting work; this gate documents that boundary rather than introducing a parallel timeout system

## Configuration Requirements

Prototype collaboration requires stable secrets and runtime policy configuration:

- `PrototypeAccessService` requires a stable signing secret from `JWT_SECRET_KEY` or `SINGLE_USER_API_KEY`; startup/test paths fail closed when neither is configured.
- `PrototypePreviewBroker` requires a stable preview signing secret from explicit configuration or the same stable server secret fallback.
- `share_policy.allow_browser_session_resume` controls same-browser resume behavior for public prototype links.
- `runtime_policy.owner_profile` and `runtime_policy.external_collaborator_profile` choose server-side runtime profiles; clients cannot request arbitrary runtime targets.
- Prototype runtime jobs use Jobs domain `prototype_workspaces` on queue `default`; deployments enabling collaboration must run the shared Jobs worker infrastructure for async bootstrap, preview boot, snapshot save, and publish validation work.
- Operators should set quotas for share-token use counts, session expiry, preview grant TTLs, and cleanup retention before exposing public links outside trusted testers.

## Migration And Rollback Notes

Migration requirements:

- Apply AuthNZ migration 086 for prototype workspace tables and migration 087 for the `prototype_workspace` share-token resource type before enabling the route set.
- Confirm foreign-key enforcement and transaction behavior on the configured AuthNZ database backend before creating public links.
- Deploy backend contract changes before Risk Gate 5/6 frontend consumption so clients can rely on `PrototypeErrorResponse` and the version 2 contract fixture.
- Existing Risk Gate 1 draft fixture consumers must tolerate the structured `detail` object before switching routes to production data.

Rollback guidance:

- Disable or hide frontend entry points for `/prototype-workspaces` and `/share/:token` prototype links before rolling back backend route behavior.
- Revoke active prototype share tokens when rolling back public collaborator access.
- Let existing prototype sessions expire or explicitly revoke them; do not delete workspace rows unless the rollback is destructive by operator intent.
- Keep migration 086/087 tables in place during rollback. Removing tables should be reserved for a later planned data-migration task because share-token and audit history may reference prototype resources.
- If Jobs workers are rolled back first, pause new collaborator entry and owner branch-session creation so bootstrap jobs are not queued without handlers.

## Preview Broker Guarantees

Preview access is brokered through `preview_handle` records and signed preview grants.

Guardrails:

- raw runtime target URLs are not exposed to clients
- stale or revoked preview handles do not renew successfully
- grants are short-lived and scoped to a workspace/session/snapshot tuple
- runtime policy resolution happens server-side

## Promotion And Publish Guarantees

Promotion is intentionally explicit and validation-gated.

Guardrails:

- only the owner or a designated promoter can approve
- candidate snapshots must belong to the same workspace
- candidate snapshots must belong to the referenced branch session
- stale baselines do not silently overwrite canonical state
- failed validation does not update canonical or last-known-good pointers

## Minimal WebUI

Current frontend surfaces:

- option route: `/prototype-workspaces`
- public share route: `/share/:token`

Owner flow includes:

- workspace creation
- canonical preview state
- owner branch-session creation
- private-link generation
- branch inventory
- candidate snapshot inventory

Collaborator flow includes:

- private-link exchange
- collaborator branch-session creation
- candidate snapshot selection
- promotion request submission

## Verification Notes

Verified in the current implementation worktree with:

- `python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q`
- `python -m pytest tldw_Server_API/tests/Sharing/test_sharing_endpoints.py -q`
- `bunx vitest run src/hooks/__tests__/usePrototypeWorkspaces.test.tsx src/hooks/__tests__/useSharing.auth.test.tsx src/routes/__tests__/option-prototype-workspaces.route.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx`
- `python -m bandit -f json -o /tmp/bandit_prototype_task6.json tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py tldw_Server_API/app/api/v1/schemas/prototype_workspace_schemas.py`
