# Sandbox Status Reason Details Design

**Status:** Approved design for a narrow Phase 3 sandbox runtime-taxonomy slice.
**Date:** 2026-05-08.
**Task:** TASK-122.

## Goal

Expose structured metadata for existing sandbox `status_reason_code` values so
clients and operator surfaces can present stable severity, category, retry, and
action guidance without parsing raw runner messages.

This is an additive API contract. It must not change run phases, raw messages,
stored run rows, runner behavior, or existing `status_reason_code` literals.

## Context

The sandbox module already exposes `status_reason_code` on public run status and
admin run summary responses. The code is derived from existing status facts in
`tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py`.

The remaining Phase 3 gap in
`Docs/Sandbox/sandbox-runtime-capability-inventory.md` is richer structured
error metadata beyond the first normalized alias pass. This design closes the
first practical part of that gap for run status responses.

## Non-Goals

- Do not introduce a new persisted state machine.
- Do not rename or remove `status_reason_code`.
- Do not change runner message text or phase transitions.
- Do not add per-runtime special cases in API schemas.
- Do not create a general FastAPI error-envelope migration.
- Do not claim richer metadata for runtime discovery `normalized_reasons`; that
  can be a later slice if needed.

## API Shape

Add a nullable `status_reason_details` field next to `status_reason_code` on:

- `SandboxRunStatus`
- `SandboxAdminRunSummary`
- `SandboxAdminRunDetails` through inheritance

The field is present when `status_reason_code` can be derived. It is `None` only
when no reason code is available.

Proposed details object:

```json
{
  "code": "runtime_unavailable",
  "category": "runtime",
  "severity": "error",
  "terminal": true,
  "retryable": true,
  "operator_action": "check_runtime_readiness",
  "user_message_key": "sandbox.status.runtime_unavailable"
}
```

### Field Meanings

| Field | Meaning |
| --- | --- |
| `code` | Mirrors the existing `status_reason_code` literal. |
| `category` | Stable grouping for client UI and diagnostics. |
| `severity` | Presentation-level severity, not a security guarantee. |
| `terminal` | Whether the run is in a terminal lifecycle outcome. |
| `retryable` | Whether retry may be reasonable after conditions change. |
| `operator_action` | Stable action hint key for operator UX. |
| `user_message_key` | Stable localization/display key, not full prose. |

Initial categories:

- `lifecycle`
- `success`
- `limits`
- `policy`
- `runtime`
- `timeout`
- `cancellation`
- `execution`
- `unknown`

Initial severities:

- `info`
- `warning`
- `error`

Initial operator actions:

- `none`
- `inspect_logs`
- `review_limits`
- `review_policy`
- `check_runtime_readiness`
- `retry_later`
- `review_exit_code`
- `unknown`

## Implementation Design

Centralize the contract in `run_status_taxonomy.py`:

- Define `RunStatusReasonCategory`, `RunStatusReasonSeverity`, and
  `RunStatusOperatorAction` literal types.
- Define a frozen dataclass `RunStatusReasonDetails`.
- Add a completeness-checked `RUN_STATUS_REASON_METADATA` map keyed by every
  `RunStatusReasonCode`.
- Add `run_status_reason_details(code)` to return the metadata for a code.
- Add `normalize_run_status_reason_details(...)` as a convenience wrapper that
  derives the code and returns details.

The API endpoint should call the same helper currently used to compute
`status_reason_code`, then derive `status_reason_details` from that code. To
avoid double normalization and future drift, the endpoint should use one local
helper that returns both values.

The Pydantic schema should define a `SandboxRunStatusReasonDetails` model with
literal fields matching the taxonomy module. The endpoint converts the dataclass
to this schema model.

## Metadata Policy

The metadata should describe normalized outcomes, not runtime internals.

Examples:

- `policy_failed` is terminal, non-retryable by default, category `policy`,
  operator action `review_policy`.
- `runtime_unavailable` is terminal for that run, retryable after host/runtime
  readiness changes, category `runtime`, operator action
  `check_runtime_readiness`.
- `limits_applied` is terminal, warning severity, category `limits`, operator
  action `review_limits`.
- `queued`, `starting`, and `running` are non-terminal, info severity,
  category `lifecycle`.
- `unknown` uses category `unknown`, warning severity, and action `unknown`.

## Error Handling

Missing metadata for a known code is a developer error. The module should fail
fast at import time if `RUN_STATUS_REASON_METADATA` does not exactly cover every
`RunStatusReasonCode` literal.

Unexpected external input should not crash response serialization. If an
unknown string reaches `run_status_reason_details()`, it should return the
`unknown` metadata with `code="unknown"` rather than raising from API paths.

## Testing

Use TDD for implementation.

Required focused tests:

- `run_status_taxonomy.py` has metadata for every `RunStatusReasonCode`.
- representative metadata values are correct for lifecycle, success, limits,
  policy, runtime-unavailable, timeout, cancellation, nonzero, and unknown.
- public and admin schemas expose `status_reason_details`.
- public run status response includes both `status_reason_code` and matching
  `status_reason_details`.
- admin run list/details paths include matching details.
- the portable runtime capability gate or inventory guard references the new
  metadata contract so docs and code do not drift.

## Documentation

Update:

- `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- `Docs/API-related/Sandbox_API.md` if it already documents
  `status_reason_code`
- TASK-122 implementation notes and final summary

The inventory should describe this as a first structured metadata pass. It
should not remove the Phase 3 gap entirely if runtime discovery
`normalized_reasons` still lacks equivalent rich metadata.

## Design Review Notes

Potential issue: duplicating `status_reason_code` inside details can drift.

Resolution: details are derived from the code in the same helper and the schema
model includes the mirrored `code` to make client rendering self-contained.

Potential issue: retryability can be interpreted as a guarantee.

Resolution: document `retryable` as guidance only. It means retry may be
reasonable after conditions change; it does not promise automatic recovery.

Potential issue: long prose in the API becomes compatibility-sensitive.

Resolution: expose stable keys, not full human prose. UI/client layers can map
keys to copy.

Potential issue: adding metadata to all admin list rows could make response
construction noisy.

Resolution: metadata is static and derived in-process. It does not require DB
queries or helper calls.
