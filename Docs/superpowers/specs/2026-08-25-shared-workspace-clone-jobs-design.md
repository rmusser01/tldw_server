# Durable Shared Workspace Clone Jobs Design

**Status:** Approved in chat on 2026-08-25

**Backlog:** TASK-12020.41

**Issue:** https://github.com/rmusser01/tldw_server/issues/2734

## Problem

The recipient clone endpoint currently invents an in-memory job identifier, runs the clone through a FastAPI background task, and immediately returns `pending`. No durable job record or status API exists, application restart loses the operation, failures are reduced to server logs, and the Shared With Me UI announces success before any copy has completed.

The replacement must use the existing Jobs subsystem, keep authorization server-authoritative, prevent same-request duplicates, survive reloads and restarts, and avoid exposing the generic Jobs administrative contract to recipients.

## Goals

- Enqueue an owner-scoped, idempotent `sharing/workspace_clone` Job.
- Expose one bounded clone-operation envelope from both enqueue and status APIs.
- Execute cloning through one reusable handler used by application-managed and standalone WorkerSDK lifecycles.
- Revalidate current share access and `allow_clone` when the worker begins.
- Report queued, running, completed, completed-with-warnings, and failed outcomes truthfully.
- Resume recipient polling after WebUI reload without adding a page-level status banner.
- Prevent repeated requests with the same idempotency key from creating duplicate target workspaces.

## Non-Goals

- Exposing generic Jobs rows, payloads, stack traces, or administrative controls.
- Making cross-database clone work transactionally resumable in this slice.
- Automatically retrying a clone after a worker failure.
- Adding route aliases or redirects.
- Changing Research Workspace and Shared With Me into one route or one product surface.

## Architecture

### Jobs Contract

Clone work uses:

- domain: `sharing`
- queue: configurable, default `workspace-clone`
- job type: `workspace_clone`
- owner: recipient user ID
- batch group: a bounded share correlation value
- idempotency key: a stable digest derived from recipient user ID and the required client `Idempotency-Key`
- maximum automatic retries: `0`

The payload contains only bounded identifiers and normalized request data: share ID, recipient user ID, requested name, and a canonical request fingerprint. It does not contain copied content, credentials, authorization claims, or owner database paths.

JobManager remains the sole durable status source. No clone projection table is introduced.

### Idempotency

`POST /api/v1/sharing/shared-with-me/{share_id}/clone` requires an `Idempotency-Key` header. Replaying the same recipient/key/request returns the existing operation. Reusing the same recipient/key for a different share or normalized name returns `409`.

The worker derives the target workspace UUID deterministically from the durable job UUID. Therefore one job cannot create multiple target workspace identities. The initial clone implementation is not transactionally resumable across ChaChaNotes and Media databases, so automatic retries are disabled. A fatal failure best-effort archives an incomplete target and returns a recoverable failed operation. An explicit user retry uses a new idempotency key and a new job.

### Authorization

Enqueue resolves the canonical active recipient share and checks `allow_clone` before creating a Job. The worker repeats that resolution before opening owner content databases, so a revoked share or removed clone permission fails closed while queued.

Status lookup first requires an authenticated job owner match, then verifies domain, type, and share correlation. Missing, foreign, malformed, or mismatched operations return the same neutral not-found response. Status remains readable by its recipient owner after later share revocation so a previously accepted operation does not become undiscoverable.

### Clone Operation Envelope

POST and GET return the same typed projection:

- schema version
- operation/job UUID
- share ID
- state: `pending`, `running`, `completed`, or `failed`
- progress percent and bounded progress message
- created, started, updated, and completed timestamps
- result only when completed: target workspace ID/name and attempted/copied/failed counts
- bounded warnings for item-level partial failures
- safe error code/message and `retryable` only when failed

Raw Jobs payloads, internal database IDs, worker IDs, exception text, and stack traces are never serialized.

### Worker

`shared_workspace_clone_jobs_worker.py` owns payload validation, canonical access revalidation, database acquisition and release, progress translation, clone execution, result normalization, and safe failure classification. `run_shared_workspace_clone_jobs_worker()` wraps that handler with WorkerSDK and supports an optional stop event. The module's standalone entry point invokes the same function.

The application lifecycle registers the worker in the existing primary Jobs poller inventory with shutdown phase `JOB_POLLER_QUIESCE`, a five-second bounded shutdown, and `SHARED_WORKSPACE_CLONE_JOBS_WORKER_ENABLED`. The stable default is enabled when the Sharing route is enabled; operators may disable it explicitly.

### Clone Service Boundary

`CloneService.clone_workspace()` accepts a caller-provided target workspace ID and continues to expose progress callbacks. It loads source collections before creating the target workspace where practical, reducing fatal partial-target windows. Item-level media, source, note, and artifact failures remain bounded result warnings rather than unqualified success.

The worker owns operation policy and lifecycle; CloneService remains a synchronous content-copy service and does not depend on JobManager.

### WebUI

Sharing hooks send the required idempotency key, consume the typed operation envelope, and poll the operation status while pending or running. A bounded local-storage record keeps the share ID, operation ID, and idempotency key so polling resumes after reload.

Shared With Me renders status inline with the relevant row:

- pending/running: progress text and a disabled clone command
- completed: copy counts or warning summary plus `Open copy`
- failed: safe failure copy plus `Try again`

Success is never announced at enqueue time. Toasts may supplement terminal transitions, but durable inline state is authoritative. No page-level banner or additional navigation surface is added.

## Failure Semantics

- Jobs unavailable during enqueue: typed `503`; no success state.
- Idempotency payload mismatch: typed `409`; existing operation remains unchanged.
- Revoked or unauthorized share at enqueue/worker start: fail closed.
- Foreign operation status request: neutral `404`.
- Worker fatal error: bounded failed operation with no raw exception details.
- Item-level copy failures: completed result with warnings and exact counts.
- Browser storage unavailable or corrupt: discard the local pointer; cloning and server status remain correct.

## Verification

- Unit tests for idempotency derivation, request fingerprints, status projection bounds, authorization, and payload validation.
- CloneService tests for deterministic target identity and partial result counts.
- API tests for enqueue replay, mismatch conflict, owner isolation, status projection, and share revocation behavior.
- Worker tests for permission revalidation, progress, completion, failure, cleanup, and standalone/application handler parity.
- Lifecycle tests for enabled/disabled defaults, inventory registration, and bounded shutdown.
- Frontend tests for header propagation, polling, terminal states, retry, open-copy routing, corrupt storage, and reload continuity.
- Live backend + WebUI CDP acceptance covering completion, failure, authorization, idempotent replay, and reload.
- Ruff/ESLint, focused type checks, OpenAPI drift, Bandit on touched Python, and `git diff --check`.

## Security And Privacy

Jobs are recipient-owned and payloads contain no copied content or secrets. Worker authorization is resolved from current server state rather than trusted payload claims. Recipient APIs return bounded domain projections and neutral authorization failures. Logs use IDs and exception classes only; they do not include source content, credentials, paths, or raw exception messages.
