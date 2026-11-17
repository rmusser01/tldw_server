# Workflows Debugging Guide

Targeted troubleshooting tips for Workflows runs, artifacts, webhooks, and human-in-loop flows. Use alongside the Runbook.

## Enable Debug Logs

Set environment flags before starting the server:

- `WORKFLOWS_DEBUG=1` - broad debug logs for endpoints and engine
- `WORKFLOWS_ARTIFACTS_DEBUG=1` - artifact endpoints (IDs, file paths, Range parsing, containment decisions)
- `WORKFLOWS_DLQ_DEBUG=1` - webhook DLQ list/replay endpoints and worker

Check application logs for lines prefixed with `Workflows:` or `Artifacts:` hints.

## WebSocket connection failures

For `/api/v1/workflows/ws`, check application logs for `Workflows WS closed: code=... reason=... error_type=...`. Expected authentication/ownership refusals are warnings; internal lookup or streaming failures are errors. Structured records also include `component=workflows`, `endpoint=workflows_ws`, `close_code`, `reason`, and `error_type`.

| Code | Stable reason | Action |
| --- | --- | --- |
| 4401 | `authentication_required` | Supply a valid supported credential. |
| 4401 | `invalid_token` | Refresh or correct the bearer token or API key; a revoked token lands here. |
| 4403 | `insufficient_scope` | Use an API key with the `read` scope. |
| 1011 | `auth_unavailable` | Investigate AuthNZ database availability. |
| 4401 / 4403 | `cookie_auth_rejected` | Check session validity, Origin, and cookie authentication configuration. |
| 4404 | `run_not_found` | Verify that the run exists in this server's workflow database. |
| 4403 | `not_run_owner` | Connect as the run's creator; the existing owner-only rule still applies. |
| 1011 | `run_lookup_error` | Investigate database availability and the logged exception class. |
| 1011 | `stream_error` | Investigate event reads or stream delivery using the exception class. |

Refusals before WebSocket acceptance may appear in browsers as an HTTP 403 handshake failure or code 1006 rather than a close frame; use the server record to distinguish them. Accepted streams that fail internally close with 1011 and the fixed reason. A peer disconnect is logged at INFO with its close code and `peer_disconnected`.

These diagnostic records contain fixed labels and exception class names, without credentials, run/owner identifiers, raw exception messages, peer-supplied reasons, or traceback locals. Do not add tokens to diagnostic captures. Since TASK-13950.7 the route accepts the same credentials as HTTP (AuthNZ access tokens, API keys, the single-user cookie) and no longer reads a `token` query parameter or MCP JWTs.

## Investigation-First Triage

For failed or flaky runs, start with the derived diagnostics endpoints before reading raw events:

1. `GET /api/v1/workflows/runs/{run_id}/investigation`
2. `GET /api/v1/workflows/runs/{run_id}/steps`
3. `GET /api/v1/workflows/runs/{run_id}/steps/{step_id}/attempts`

Focus on the structured fields first:

- `reason_code_core` and `reason_code_detail`
- `category`
- `blame_scope`
- `retryable`
- `retry_recommendation`

If investigation generation is incomplete, fall back to `GET /runs/{run_id}/events?limit=200` and artifacts/log excerpts. The attempt ledger is the preferred source for retry history.

## Preflight & Replay Safety

- Use `POST /api/v1/workflows/preflight` before saving or rerunning a changed definition.
- `validation_mode=block` returns blocking validation errors.
- `validation_mode=non-block` demotes definition validation issues to warnings so authors can inspect the rest of the payload.
- Replay safety warnings come from step capability metadata:
  - `replay_safe`
  - `idempotency_strategy`
  - `compensation_supported`
  - `requires_human_review_for_rerun`
- Treat `unsafe_replay_step` as a real warning. Side-effecting steps such as `webhook`, `notify`, or external tool calls should be reviewed before rerun even when the original failure looks transient.

## Artifact Downloads

- Containment: Enforced only when a `workdir` is recorded on the run/step metadata. If strict mode blocks unexpected paths, ensure your step recorded the correct workdir.
- Strict vs Non-block:
  - Strict: `WORKFLOWS_ARTIFACT_VALIDATE_STRICT=true` (default) or run `validation_mode='block'`
  - Non-block: per-run `validation_mode='non-block'` or env set to false (proceeds with warnings)
- Range: Only a single `Range` header is honored; responses use `206 Partial Content` and `Content-Range`.
- Checksums: When present, checksum mismatches trigger `409 Conflict` in strict mode; non-block logs a warning and continues.

Checklist:
- Confirm the artifact is listed under `GET /runs/{run_id}/artifacts`.
- Ensure the API and your test harness point to the same DB (check `DATABASE_URL_WORKFLOWS`).
- Inspect debug logs with `WORKFLOWS_ARTIFACTS_DEBUG=1` for resolved path and containment decisions.

## Webhooks

- Signing scheme (v1): The server signs the body with HMAC-SHA256 using `WORKFLOWS_WEBHOOK_SECRET`.
  - Headers: `X-Workflows-Signature-Version: v1`, `X-Workflows-Signature`, `X-Hub-Signature-256`, `X-Signature-Timestamp`, `X-Webhook-ID`, `X-Workflow-Id`, `X-Run-Id`
  - Compute signature over `f"{ts}.{body}"` using the `X-Signature-Timestamp` value.

Python verifier snippet:

```python
import hmac, hashlib

def verify(secret: str, ts: str, body: str, received: str) -> bool:
    mac = hmac.new(secret.encode(), f"{ts}.{body}".encode(), hashlib.sha256)
    return hmac.compare_digest(mac.hexdigest(), received)
```

- Egress policy: If deliveries show `blocked`, check allow/deny lists and private IP blocking configuration.
- DLQ: Inspect `workflow_webhook_dlq` for failures; enable `WORKFLOWS_DLQ_DEBUG=1` and run the worker to replay.

## Pause/Resume/Cancel & Human-in-Loop

- Pause: sets `paused` and emits `run_paused`; engine cooperatively idles and maintains leases.
- Resume: sets `running` and emits `run_resumed`.
- Cancel: sets cancel flag, best-effort terminates subprocesses, updates run to `cancelled`.
- Human steps: `wait_for_human`/`wait_for_approval` set `waiting_human`/`waiting_approval` status; resume flow by POSTing to:
  - `POST /api/v1/workflows/runs/{run_id}/steps/{step_id}/approve`
  - `POST /api/v1/workflows/runs/{run_id}/steps/{step_id}/reject`

Debug checklist:
- Inspect events stream (`GET /runs/{run_id}/events?limit=200`) for `run_paused`, `run_resumed`, `run_cancelled`, `step_*`.
- If resume stalls, verify `after_step_id` handling via logs (enabled by `WORKFLOWS_DEBUG`).

## Retry Decisions

- `retryable=true` means the latest failure looked transient, not that replay is automatically safe.
- Use attempt metadata plus step capability metadata together:
  - Safe replay: transient failure + replay-safe step.
  - Conditional replay: transient failure on a side-effecting or human-reviewed step.
  - Fix-before-rerun: definition/policy/input failures; run preflight after the change.
- If you need DB-level inspection, look at `workflow_step_attempts` before `workflow_events`; it is the canonical retry ledger.

## Idempotency

- Provide `idempotency_key` in run requests to deduplicate repeated submissions.
- With `WORKFLOWS_DEBUG=1`, logs show reuse decisions for existing runs.
- Expect the same `run_id` to be returned if a matching run exists for your key and user.

## DB & Environment

- Ensure `DATABASE_URL_WORKFLOWS` is consistent across app/test contexts.
- SQLite: watch for `database is locked`; reduce concurrent writers or move to Postgres for heavier loads.
- Postgres: monitor connection pool saturation and autovacuum on `workflow_events`.
