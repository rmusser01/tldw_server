# Sandbox Operator Status Consolidation Design

**Date:** 2026-06-18
**Status:** Proposed design
**Backlog:** TASK-2384
**Scope:** Read-only sandbox operator/admin status consolidation across runtime
discovery, macOS diagnostics, image-store health, reconciliation posture, and
host-gated VZ evidence signals.

## Summary

The sandbox module now has several mature operator surfaces:

- `/api/v1/sandbox/runtimes` for public runtime discovery.
- `/api/v1/sandbox/admin/runtime-diagnostics` for cross-runtime admin
  readiness and warning summaries.
- `/api/v1/sandbox/admin/macos-diagnostics` for macOS helper, template,
  image-store, VZ reconciliation, observability, startup warning, and recovery
  posture.
- Separate dry-run-first repair and image-store cleanup endpoints.
- Host-gated VZ smoke evidence artifacts and an advisory evidence summary.

The next pragmatic slice is a read-only operator status consolidation layer.
It should help an operator answer "what is the sandbox subsystem's current
operational posture?" without parsing multiple endpoint payloads, workflow
artifacts, and docs by hand. It must not become another source of truth, start
helpers, load launchd services, mutate image-store files, run repair, reboot
hosts, or run real VMs.

## Goals

- Provide one read-only status projection for operator dashboards and CLI
  status commands.
- Reuse existing runtime discovery, admin diagnostics, image-store, recovery,
  and evidence-summary facts.
- Classify findings into stable status buckets that clients can render without
  matching runtime-specific strings.
- Preserve current VZ/manual-host-gated boundaries.
- Keep normal CI portable; real VZ execution remains manual or trusted
  host-gated only.

## Non-Goals

- Do not introduce a new helper lifecycle manager.
- Do not start, stop, restart, or install the macOS helper.
- Do not bootstrap, kickstart, bootout, or infer ownership of launchd agents.
- Do not run reconciliation repair or image-store cleanup.
- Do not create image-store roots or evidence directories from read-only
  status collection.
- Do not run host-gated smoke, launchd-drill, host-reboot-drill, stale socket
  drills, or real VMs.
- Do not expose raw serial logs, helper stdout/stderr contents, environment
  dumps, secrets, user workspace paths beyond already-exposed bounded pointers,
  or full workflow logs.
- Do not make `seatbelt` or `worktree` appear VM-grade.
- Do not require Apple `container` or change current VZ boot behavior.

## Existing Sources Of Truth

The consolidated status must be a projection over these existing sources:

| Source | Role | Mutation allowed |
| --- | --- | --- |
| `SandboxService.feature_discovery()` | Runtime availability, implementation state, normalized reasons, isolation, network, and session contract metadata. | No |
| `SandboxService.runtime_diagnostics_summary()` | Cross-runtime admin readiness summary derived from feature discovery. | No |
| `collect_macos_diagnostics()` | macOS host/helper/template/image-store/reconciliation/observability/recovery details. | No |
| `probe_image_store(create_root=False)` behavior | Image-store existence, manifests, and GC candidate posture. | No |
| `summarize_recovery()` | Recovery and repair-plan posture derived from reconciliation and image-store diagnostics. | No |
| Host-gated evidence summary artifact | Advisory prepared-host smoke evidence status when a caller provides a bounded, operator-owned artifact path or the workflow exposes one. | No |
| Startup warning summary | Last startup-time helper/reconciliation warning posture when available in diagnostics. | No |

The projection should not call lower-level helper APIs directly when an
existing diagnostics function already owns that check. If a needed fact is
missing, add it to the owning diagnostics surface first or mark it unavailable
with a reason.

## Proposed Operator Status Shape

Add a status projection named `sandbox_operator_status` at the service layer,
then expose it through an admin-only read-only endpoint if implementation
proceeds.

Top-level fields:

- `source`: fixed string such as `sandbox_operator_status`.
- `generated_at`: ISO timestamp from the server process.
- `overall_status`: one of `ready`, `degraded`, `action_required`,
  `unavailable`, or `unknown`.
- `overall_severity`: one of `info`, `warning`, or `error`.
- `summary`: small counts and high-level booleans.
- `sections`: structured section records.
- `recommended_actions`: ordered action records.
- `notes`: bounded operator-readable notes.

Section records should be stable and small:

| Section | Content |
| --- | --- |
| `runtime_readiness` | Counts and per-runtime status from `runtime-diagnostics`. |
| `macos_vz` | Helper, template, protocol, generation, and startup warning posture from `macos-diagnostics`. |
| `image_store` | Configured/existing status, manifest count, GC candidate count, and cleanup-plan endpoint pointer from diagnostics. |
| `reconciliation` | Computed state, stale/unhealthy/active/orphan counts, repair dry-run endpoint pointer, and fail-closed reasons. |
| `evidence` | Advisory host-gated evidence summary presence, result, workflow/source, commit/ref, age, and expected skips when supplied. |
| `security_boundaries` | Host-local warning runtimes, untrusted eligibility summary, deny-all/allowlist strictness summary, and known unsupported states. |

Recommended actions should be action codes, not prose-only text. Examples:

- `restore_helper_readiness`
- `update_helper_protocol`
- `configure_vz_template`
- `inspect_reconciliation`
- `run_repair_dry_run`
- `inspect_image_store_cleanup_plan`
- `run_host_gated_smoke`
- `review_expected_skips`
- `use_different_runtime`
- `none`

The implementation should derive action order from existing normalized reason
details where possible. New action codes must be documented in the schema and
tests before clients depend on them.

## Status Classification

The projection should classify the overall status conservatively:

- `ready`: at least one configured runtime is ready, no blocking macOS/VZ
  recovery issues are present, and no evidence artifact reports a blocking
  failure.
- `degraded`: usable runtime coverage exists, but host-gated VZ evidence is
  missing/stale, optional drills were skipped, image-store cleanup candidates
  exist, startup warnings exist, or host-local runtime warnings need operator
  awareness.
- `action_required`: runtime diagnostics or macOS diagnostics found repairable
  stale/unhealthy state, helper/template setup is incomplete for a configured
  VZ path, or evidence reports a blocking smoke failure.
- `unavailable`: no runtime is currently usable or diagnostics cannot compute
  due to helper/protocol/runtime prerequisites that block configured paths.
- `unknown`: required status inputs were absent, malformed, or unavailable
  before a safer classification could be made.

This is an operator UX classification. It must not change sandbox admission,
runtime selection, repair eligibility, or security policy enforcement.

## Evidence Summary Handling

Evidence summary input is advisory and optional.

Accepted sources:

- a caller-provided path to a local summary JSON produced by the existing
  host-gated evidence summary tooling
- a future workflow artifact pointer that has already been validated by CI
  tooling

Safety rules:

- Treat the evidence summary as untrusted external input.
- Reject embedded NUL paths, non-files, symlinks when path policy requires a
  regular file, oversized JSON, malformed JSON, non-scalar fields, and
  unexpected schema versions.
- Do not recursively scan arbitrary artifact directories.
- Do not read raw logs.
- If the summary is absent or invalid, report `evidence.status=unknown` or
  `evidence.status=unavailable`; do not fail the whole operator status unless
  the caller explicitly requested strict evidence mode.

The first implementation can omit path-based evidence ingestion if that would
broaden the PR too much. In that case the spec still defines the section shape,
and the endpoint should return `evidence.status=not_configured` with an action
to run or attach host-gated smoke evidence.

## API Placement

Preferred endpoint for a future implementation:

```text
GET /api/v1/sandbox/admin/operator-status
```

Reasons:

- It is clearly admin-only.
- It does not overload runtime discovery.
- It can aggregate existing admin diagnostics while keeping detailed drill and
  repair endpoints separate.
- It leaves `/api/v1/sandbox/admin/macos-diagnostics` as the detailed macOS
  evidence source instead of replacing it.

The response should have a Pydantic schema in
`tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`. Keep field names
plain and stable; prefer additive changes over nested runtime-specific blobs.

## Implementation Slices

### Slice 1: Portable Status Projection

Add service-level projection and endpoint without evidence-file ingestion.

Deliverables:

- `SandboxService.operator_status()` or a focused helper module.
- Admin-only `GET /api/v1/sandbox/admin/operator-status`.
- Pydantic response schema.
- Unit tests with synthetic runtime diagnostics and macOS diagnostics.
- Docs update pointing operators at detailed sources.

### Slice 2: Advisory Evidence Summary Input

Add optional, bounded evidence summary ingestion.

Deliverables:

- Safe parser for existing host-gated evidence summary JSON.
- Optional config/env or request query path for operator-supplied evidence.
- Tests for malformed paths, malformed JSON, oversized JSON, stale evidence,
  blocking failure evidence, and expected skips.
- Docs describing evidence retention and privacy boundaries.

### Slice 3: Dashboard/CLI Consumers

Use the status projection in future UI or CLI surfaces.

Deliverables:

- Dashboard or CLI view that renders section statuses and recommended actions.
- No duplicated runtime-string matching in clients.
- Accessibility and copy review for operator messages.

## Testing Strategy

Normal CI should stay portable.

Required tests for Slice 1:

- Runtime ready plus clean macOS diagnostics returns `ready`.
- Helper/protocol/template blockers produce `action_required` or
  `unavailable` with stable action codes.
- Reconciliation stale/unhealthy/owned-orphan counts point to dry-run repair
  without mutating state.
- Image-store GC candidates point to cleanup-plan, not cleanup mutation.
- Host-local runtime warnings appear in `security_boundaries`.
- Missing macOS diagnostics degrade gracefully rather than crashing.
- No code path invokes helper lifecycle commands, launchd, repair mutation,
  image-store cleanup mutation, or real VZ execution.

Required tests for Slice 2:

- Valid summary produces advisory evidence status.
- Missing summary is non-blocking by default.
- Malformed JSON, embedded NUL paths, non-files, symlinks where disallowed, and
  oversized summaries are treated as unavailable/invalid without crashing.
- Blocking evidence failures affect status only through documented rules.
- Expected skips are preserved as advisory details.

## Security And Privacy Review

- Keep the endpoint admin-only and consistent with existing sandbox admin
  authorization.
- Do not expose secrets, environment dumps, raw logs, serial log contents, or
  user workspace contents.
- Preserve path-safety rules for any future evidence summary reads.
- Keep repair mutation, helper lifecycle mutation, and launchd mutation on
  their existing explicit endpoints/commands.
- Treat helper/evidence metadata as external input and validate types before
  reading nested fields.
- Keep host-local runtime warnings visible so operators do not overclaim
  isolation.

## Design Risks And Mitigations

- Risk: The consolidated endpoint becomes a second source of truth.
  Mitigation: make it a projection over existing diagnostics and document the
  owner for each section.
- Risk: Operators mistake advisory evidence for live runtime health.
  Mitigation: label evidence as advisory, include age/source/commit, and keep
  live readiness in runtime and macOS diagnostics.
- Risk: The status model pressures VZ-only concepts into every runtime.
  Mitigation: keep VZ details under `macos_vz`; cross-runtime fields stay based
  on discovery/session/security contracts.
- Risk: The endpoint grows mutating convenience features.
  Mitigation: no mutation in this endpoint; recommended actions point to
  existing explicit dry-run-first surfaces.
- Risk: Evidence ingestion probes arbitrary filesystem paths.
  Mitigation: make evidence ingestion a later optional slice with strict path,
  size, and schema validation.

## Open Questions

- Should evidence summary input be configured server-side only, or should the
  endpoint accept an admin query parameter in a later slice?
- Should `overall_status=degraded` be used for stale evidence, or should stale
  evidence remain section-local until a strict evidence mode is requested?
- Should the first implementation expose `generated_at`, or follow existing
  diagnostics surfaces and omit timestamps to keep tests deterministic?

## Acceptance Criteria

- The design keeps operator status read-only and advisory.
- Existing runtime discovery and diagnostics remain the source of truth.
- The first implementation slice is portable and does not require a prepared
  Apple silicon host.
- Security boundaries for helper lifecycle, launchd, repair, image-store
  mutation, and evidence parsing are explicit.
- Future client/dashboard work can consume stable section/action codes instead
  of parsing runtime-specific reason strings.
