# Startup Warning Framework Design

**Date:** 2026-04-30
**Status:** Draft for review
**Scope:** `tldw_Server_API/app/services/`, admin API schemas/endpoints, sandbox diagnostics integration, and startup/lifespan wiring

## Summary

This design adds a minimal reusable startup warning framework for the current
process, with `sandbox.vz_linux` reconciliation as the first producer.

The framework exists to make startup-time operator findings visible in three
places without mutating runtime state:

- structured startup logs
- a generic admin API surface for current-process startup warnings
- additive component-local summaries, starting with sandbox diagnostics

The same framework also carries startup blockers. In this slice,
`macos_virtualization_helper_protocol_mismatch` is promoted to a startup
blocking condition, while stale/unhealthy/orphaned `vz_linux` reconciliation
findings remain warnings only.

## Why A New Slice Is Needed

The current merged sandbox lifecycle work already provides:

- read-only reconciliation of persisted VZ session rows against helper VM truth
- explicit admin repair
- ownership-gated orphan classification and termination
- helper protocol mismatch handling during runtime operations

What is still missing is startup-time operator visibility. Today, operators only
learn about stale or unhealthy `vz_linux` state after calling diagnostics, and
there is no shared shape for startup warnings that other subsystems can reuse.

This design closes that gap without turning startup into a mutating cleanup
path.

## Corrected Design Constraints

This design intentionally tightens the earlier brainstorming draft in four ways:

1. The generic admin surface is app-level, not sandbox-router-owned.
2. The warning registry is explicitly current-process and current-boot scoped.
3. Startup blockers are log-first failures; the admin endpoint is only available
   for successful boots.
4. The sandbox startup producer uses a bounded, lighter-weight startup probe
   instead of re-running the full admin diagnostics path.

## Goals

1. Add one reusable startup warning record shape that later subsystems can
   adopt.
2. Surface startup warnings for the current process through a generic admin
   endpoint.
3. Surface an additive startup warning summary in sandbox diagnostics.
4. Emit structured logs from the same warning records stored in memory.
5. Block startup on helper protocol mismatch for `vz_linux`.
6. Keep startup non-mutating and operator-first.

## Non-Goals

1. Cross-process or cluster-wide startup warning aggregation.
2. Persistence of startup warnings across restarts.
3. A general runtime health platform.
4. Automatic sandbox repair at startup.
5. UI/admin notification fan-out in this PR.
6. New producers outside sandbox reconciliation.

## Current-Process Scope

The new registry is in-memory and lives in the current API process only.

That means:

- in single-process dev/operator workflows, the endpoint reflects the real boot
  warnings for that process
- in multi-worker deployments, each worker has its own startup warning state
- the generic API must describe itself as current-process startup warnings, not
  global application truth

This limitation is acceptable for the first slice because the operator goal is
to make startup findings inspectable without inventing shared persistence or
distributed coordination.

## Architecture

### Shared Record Shape

Add a small startup warning model owned by the startup services layer:

```python
{
    "component": "sandbox.vz_linux",
    "severity": "warning" | "error",
    "startup_action": "warn" | "block_startup",
    "code": "vz_stale_session_controls_detected",
    "summary": "Startup detected stale persisted vz_linux session bindings.",
    "remediation": "Review sandbox diagnostics and run explicit reconciliation repair after confirming no active work.",
    "details": {...},
    "detected_at": "2026-04-30T20:10:00Z",
}
```

Rules:

- `severity` is operator-facing urgency.
- `startup_action` is the startup policy decision.
- `code` is stable and machine-readable.
- `details` stays small and summary-oriented, not a full diagnostic dump.

### Startup Warning Registry

Add an in-process registry owned by app startup state, not by the sandbox
module.

Responsibilities:

- clear state at process startup
- store startup warning records
- expose warning records for later admin reads
- answer whether any record requires `block_startup`
- return grouped summaries by component and severity

The preferred ownership seam is `app.state.startup_warning_registry`, with a
small service helper around it, rather than a module-global singleton. That
keeps startup lifecycle control explicit and makes tests less fragile.

The registry should be created by the lifespan startup flow and attached to
`app.state`. Admin endpoints may read it from `request.app.state`. Lower-level
diagnostics helpers must not reach into `app.state` directly.

### Producer Boundary

The first producer is sandbox-specific, but it only translates existing truth.

It should:

- inspect helper reachability/compatibility with bounded calls
- inspect reconciliation results with bounded calls
- create shared startup warning records
- never mutate persisted rows
- never call repair
- never terminate VMs

It should not reuse `collect_macos_diagnostics()` wholesale because that path is
broader than the startup decision and may perform heavier or redundant checks.
Instead, startup should use a focused producer that pulls only:

- helper compatibility status
- reconciliation status/counts

The producer must not import or depend on the sandbox endpoint module singleton
(`tldw_Server_API.app.api.v1.endpoints.sandbox._service`). Startup code should
depend on an explicit sandbox/orchestrator seam owned by app startup, not an
endpoint-owned global.

### Startup Ordering

The sandbox startup warning producer must run only after the sandbox service and
its orchestrator dependencies are available.

It should run once during the lifespan startup sequence, after core sandbox
state is ready but before startup is considered complete.

Concretely, the startup producer should receive either:

- an orchestrator reference passed from startup-owned initialization, or
- a small startup-safe sandbox service reference attached to `app.state`

It must not obtain startup dependencies by importing endpoint modules.

Policy:

- if the producer emits only `warn` records, startup continues
- if the producer emits any `block_startup` record, startup raises immediately
  using the strongest blocking record

### Blocking Behavior

For this PR, only one sandbox condition blocks startup:

- `macos_virtualization_helper_protocol_mismatch`

Why:

- it signals a known incompatibility between the trusted Python control plane
  and the helper protocol
- continuing would make later diagnostics and repair behavior misleading or
  unsafe

Non-blocking sandbox startup findings:

- helper unavailable
- stale persisted rows
- unhealthy persisted/live VM bindings
- skipped-active reconciliation items
- owned/unknown/foreign orphan VM findings

### Logging

Structured logs must be emitted from the same record objects that are stored in
the registry.

This avoids log/API drift and ensures startup blockers are still visible even
when the app never becomes available enough to serve the new admin endpoint.

Startup blockers are therefore log-first by design.

## Sandbox Producer Policy

Recommended startup warning codes:

- `vz_stale_session_controls_detected`
- `vz_unhealthy_session_controls_detected`
- `vz_orphaned_vms_detected`
- `vz_skipped_active_reconciliation_items_detected`
- `vz_helper_unavailable_at_startup`
- `vz_helper_protocol_mismatch`

Recommended actions:

| Code | Action | Severity |
|---|---|---|
| `vz_stale_session_controls_detected` | `warn` | `warning` |
| `vz_unhealthy_session_controls_detected` | `warn` | `warning` |
| `vz_orphaned_vms_detected` | `warn` | `warning` |
| `vz_skipped_active_reconciliation_items_detected` | `warn` | `warning` |
| `vz_helper_unavailable_at_startup` | `warn` | `warning` |
| `vz_helper_protocol_mismatch` | `block_startup` | `error` |

The producer should summarize counts, for example:

```json
{
  "stale_session_controls": 2,
  "unhealthy_session_controls": 1,
  "orphaned_vms": 3,
  "skipped_active_sessions": 1
}
```

It should not embed full reconciliation item lists into the generic startup
warning endpoint.

## API Surfaces

### Generic Admin Endpoint

Add a new admin-only endpoint outside the sandbox router:

```text
GET /api/v1/admin/startup-warnings
```

Response shape:

```json
{
  "startup_id": "2026-04-30T20:10:00Z",
  "scope": "current_process",
  "warnings_present": true,
  "blocking_present": false,
  "summary": {
    "total": 3,
    "by_component": {
      "sandbox.vz_linux": 3
    },
    "by_severity": {
      "warning": 3
    }
  },
  "items": [
    {
      "component": "sandbox.vz_linux",
      "severity": "warning",
      "startup_action": "warn",
      "code": "vz_stale_session_controls_detected",
      "summary": "Startup detected stale persisted vz_linux session bindings.",
      "remediation": "Review admin macOS diagnostics and run explicit reconciliation repair after confirming no active work.",
      "details": {
        "stale_session_controls": 2
      },
      "detected_at": "2026-04-30T20:10:00Z"
    }
  ]
}
```

This endpoint is only meaningful for a successful boot.

Implementation note: this endpoint must be wired into the existing admin router
assembly under `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py` and the
admin router group, not only created as a standalone module.

### Sandbox Diagnostics Additive Summary

Extend the existing sandbox diagnostics payload with a compact startup summary:

```json
"startup_warning_summary": {
  "present": true,
  "blocking": false,
  "codes": [
    "vz_stale_session_controls_detected",
    "vz_orphaned_vms_detected"
  ]
}
```

This keeps sandbox operators in their existing workflow while the generic admin
endpoint establishes the reusable app-level pattern.

Because `collect_macos_diagnostics(orchestrator)` is intentionally app-agnostic,
the startup warning summary must be injected one layer above it. The preferred
seam is:

- keep `macos_diagnostics.py` pure and independent of `app.state`
- have `SandboxService.macos_diagnostics()` or the sandbox admin endpoint obtain
  the shared startup warning summary from a provider attached to app startup
  state and merge that additive summary into the response payload

The design must not assume that low-level diagnostics helpers can read
`app.state` directly.

## Timeout And Startup Cost Rules

The sandbox startup producer must use bounded helper calls.

Requirements:

- reuse existing helper client timeout behavior if already bounded
- avoid heavyweight template validation during startup warning production
- avoid broad admin diagnostics aggregation
- skip the producer entirely when the sandbox/helper path is not configured
  enough to be actionable

The intent is to prevent non-essential helper lag from turning startup warnings
into startup drag.

## Failure Handling

If the sandbox startup warning producer encounters incidental internal errors
that are not classified compatibility failures:

- do not crash startup by default
- emit a bounded warning record or structured log describing producer failure
  only if it is actionable
- prefer fail-open for incidental producer errors

The only explicit fail-closed startup path in this slice is helper protocol
mismatch.

## Testing Strategy

### Unit Tests

- startup warning registry add/list/clear behavior
- grouped summary generation
- `block_startup` detection
- sandbox producer translation from reconciliation/helper inputs to warning
  records

### Startup Integration Tests

- registry is initialized and cleared per startup
- sandbox producer runs once during startup
- warning records appear on the generic admin endpoint
- sandbox diagnostics include `startup_warning_summary`
- protocol mismatch aborts startup with the expected blocking code

### Regression Tests

- healthy/empty reconciliation produces no startup warning records
- startup warning production does not delete session-control rows
- startup warning production does not terminate VMs
- helper unavailable remains non-blocking

## Documentation Updates

Update:

- `Docs/Sandbox/macos-runtime-operator-notes.md`
- `tldw_Server_API/app/core/Sandbox/README.md`
- admin API docs/schema references for the new startup-warning surface

Docs should explicitly say:

- startup warnings are current-process only in this PR
- startup blockers are guaranteed in logs, not in the admin endpoint
- sandbox startup remains non-mutating
- helper protocol mismatch blocks startup

## Future Follow-Up

Later work can extend this framework with:

- additional producers outside sandbox
- persistence for startup warnings across restarts
- multi-worker/shared warning aggregation
- UI/admin notification consumption
- audit integration if startup warnings become an operator workflow primitive
