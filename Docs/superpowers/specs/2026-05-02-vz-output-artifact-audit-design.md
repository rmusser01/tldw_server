# VZ Linux Output And Artifact Limit Audit Design

## Status

Design approved for planning. Implementation is not started in this document.

## Context

The `vz_linux` runtime now has a helper-backed `exec_guest` contract, but the
next stability gap is bounded result handling. Today `VZLinuxRunner` publishes
the helper-returned stdout/stderr without passing the global sandbox log cap, and
artifact collection reads every matching file fully into memory. The sandbox
stream hub already supports capped log publishing, and sandbox run completion
already has an audit hook. This slice should reuse those seams instead of
creating a parallel logging or audit subsystem.

This work follows `Docs/Sandbox/sandbox-architecture-doctrine.md`: Python owns
artifact metadata and public API behavior, while the native helper owns the host
protocol boundary and live runtime execution facts.

## Goals

- Bound `vz_linux` helper-returned stdout/stderr before Python receives an
  unbounded response.
- Publish `vz_linux` stdout/stderr through the existing sandbox stream cap.
- Bound `vz_linux` artifact collection by per-file and total bytes.
- Preserve run success/failure semantics when artifact capture exceeds policy;
  oversized artifacts are skipped, not fatal.
- Record limit outcomes in existing completion audit metadata.
- Emit aggregated truncation audit events: one for output caps and one for
  artifact cap/skips per run when applicable.
- Add a shared Python limit helper that future runtimes can adopt, while wiring
  only `vz_linux` in this PR.

## Non-Goals

- Do not add command allowlisting.
- Do not change `docker`, `seatbelt`, `worktree`, `firecracker`, or `lima`
  behavior in this PR.
- Do not add durable artifact manifests beyond the existing orchestrator/store
  behavior.
- Do not stream guest stdout/stderr incrementally from the guest agent.
- Do not kill the guest process when the output cap is reached.

## Chosen Approach

Add a shared Python module under `tldw_Server_API/app/core/Sandbox/` that owns
limit math and metadata shapes for:

- output cap requests
- output cap observations
- safe artifact collection with per-file and total caps
- structured, path-minimized audit metadata

Wire that module into `VZLinuxRunner` only. The runner will:

- read `SANDBOX_MAX_LOG_BYTES`
- pass that value to helper `exec_guest` as `max_output_bytes`, clamped to the
  helper protocol ceiling of 256 MiB
- publish stdout/stderr with `max_log_bytes`
- collect artifacts through the shared helper with explicit limits
- put integer-only limit counters into `RunStatus.resource_usage`
- let `SandboxService` derive richer audit metadata from those counters

The native Swift helper will extend the `exec_guest` protocol with an optional
`max_output_bytes` request field. It will validate the value and cap returned
stdout/stderr before JSON encoding the host response. The helper response should
include details indicating stdout/stderr byte counts and truncation flags.
Because the current Swift helper response models use `details: [String: String]`,
byte counts and booleans in helper details must be encoded as strings and parsed
defensively by Python.

## Output Cap Semantics

`max_output_bytes` is a combined stdout+stderr cap with per-stream metadata.
The helper should preserve deterministic behavior:

- when both streams exceed the cap, each stream receives a fair minimum share so
  noisy stdout cannot completely hide stderr diagnostics
- unused budget from one stream may be used by the other stream
- the sum of returned stdout and stderr bytes must not exceed
  `max_output_bytes`
- response details include original and returned byte counts for each stream
- response details include whether each stream was truncated
- invalid `max_output_bytes` shape returns `invalid_request`
- semantic violations return `exec_output_limit_invalid`

This protects the Python helper-client boundary from unbounded helper responses.
It does not fully protect the Swift bridge or guest agent if either layer
buffers command output internally before the service-level cap is applied.
Guest-protocol streaming or guest-side kill-on-cap remains a follow-up.

`SANDBOX_MAX_LOG_BYTES` remains the WebSocket/log publication cap and becomes
the default `vz_linux` helper-returned output cap. For `vz_linux`, runtime
discovery advertises the effective helper cap when `SANDBOX_MAX_LOG_BYTES`
exceeds the helper protocol ceiling. Documentation should state that this knob
now bounds both surfaces for `vz_linux`, subject to that helper ceiling.

## Artifact Cap Semantics

The shared artifact collector should enforce:

- traversal stays under the workspace root
- symlinks are skipped
- non-matching files are ignored
- matching files over the per-file cap are skipped
- matching files that would exceed the total cap are skipped
- skipped files do not fail the run
- collected artifact bytes remain accurate in `resource_usage.artifact_bytes`

The first wired runtime is `vz_linux`. The shared collector should read explicit
settings for per-file and total caps. The implementation plan should add
conservative sandbox settings such as `SANDBOX_MAX_ARTIFACT_FILE_BYTES` and
`SANDBOX_MAX_ARTIFACT_TOTAL_BYTES`, with defaults and tests.

Audit metadata must avoid raw artifact paths by default because file names may
contain sensitive information. Completion metadata may include counts, byte
totals, cap values, and reason codes. If path-level metadata is already exposed
in normal artifact APIs, it should still not be added to audit events in this
slice.

## Audit Design

Reuse the existing sandbox run completion audit path and add compact metadata:

- `output_limit_applied`
- `output_truncated`
- `stdout_bytes_returned`
- `stderr_bytes_returned`
- `stdout_bytes_original` when provided by helper details
- `stderr_bytes_original` when provided by helper details
- `artifact_limit_applied`
- `artifact_files_collected`
- `artifact_files_skipped`
- `artifact_bytes_collected`
- `artifact_skip_reasons`

`RunStatus.resource_usage` must remain compatible with the current public and
admin schemas, which type it as `dict[str, int]`. Store only integer counters
there, for example:

- `output_limit_bytes`
- `stdout_bytes_returned`
- `stderr_bytes_returned`
- `stdout_bytes_original`
- `stderr_bytes_original`
- `stdout_truncated`
- `stderr_truncated`
- `artifact_limit_file_bytes`
- `artifact_limit_total_bytes`
- `artifact_files_collected`
- `artifact_files_skipped`
- `artifact_bytes_collected`
- `artifact_skip_file_limit`
- `artifact_skip_total_limit`

`SandboxService` should convert those integer counters into boolean flags and
reason-code lists for audit metadata. This avoids breaking the run status API
while still making audit records readable.

Add separate aggregated audit events only when limits affect the run:

- one `sandbox.run.output_truncated` style event when stdout/stderr was capped
- one `sandbox.run.artifacts_limited` style event when artifact files were
  skipped because of per-file or total caps

These events must be per-run aggregates, not per-stream or per-file spam.
`SandboxService` should emit these from `RunStatus.resource_usage` metadata
alongside the existing completion audit path, so runtime runners do not open
their own audit clients.
Implementation should use the existing audit enum values, for example
`AuditEventType.API_RESPONSE` with actions `output_truncated` and
`artifacts_limited`, instead of adding new audit enum values in this slice.

## Error Handling

- Malformed `max_output_bytes` in socket JSON returns `invalid_request`.
- Non-positive or excessive `max_output_bytes` returns
  `exec_output_limit_invalid`.
- Artifact collection exceptions should fail closed for collection and preserve
  the run outcome, matching existing runner behavior.
- Audit failures remain non-fatal and should use existing best-effort audit
  handling.

## Testing Strategy

Swift helper tests:

- service-level `exec_guest` caps stdout/stderr and returns details.
- socket-level malformed `max_output_bytes` returns `invalid_request`.
- socket-level semantic invalid `max_output_bytes` returns
  `exec_output_limit_invalid`.

Python helper-client tests:

- request passes `max_output_bytes` to the helper.
- fake TEST_MODE helper mirrors max-output validation and response details.

Python runner tests:

- `VZLinuxRunner` passes `SANDBOX_MAX_LOG_BYTES` as `max_output_bytes`.
- stdout/stderr publication uses the same cap.
- output truncation metadata appears in `resource_usage`.
- artifact collection skips oversized files and preserves completed run status.
- artifact skip metadata appears in `resource_usage`.
- audit metadata/event helpers are invoked with aggregated metadata.

## Risks And Open Issues

- Helper-side caps do not solve guest-agent internal buffering. This PR must
  document that guest-agent streaming or kill-on-cap is follow-up work.
- Reusing `SANDBOX_MAX_LOG_BYTES` for helper response caps is pragmatic but
  changes the knob's meaning for `vz_linux`; docs and tests need to make this
  explicit.
- Audit event names should match existing audit conventions during
  implementation. If the audit system lacks a good category/action vocabulary,
  prefer stable metadata on completion over inventing a broad new taxonomy.
- Artifact caps should avoid changing other runtime behavior. The shared helper
  must be low-level and opt-in.

## Follow-Up Work

- Extend the guest protocol so the guest agent can stream output or enforce
  kill-on-cap without buffering all output first.
- Adopt the shared artifact/output helper in `seatbelt` and `worktree` once
  `vz_linux` behavior is proven.
- Add operator docs for tuning artifact caps if this becomes an operational
  issue.
