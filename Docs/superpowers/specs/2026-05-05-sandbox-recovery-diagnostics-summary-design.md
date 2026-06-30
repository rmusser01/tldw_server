# Sandbox Recovery Diagnostics Summary Design

**Status:** Approved implementation design.
**Date:** 2026-05-05.
**Backlog:** `TASK-70`.
**Scope:** Additive recovery summary for macOS sandbox admin diagnostics.

## Related Guidance

- `Docs/Sandbox/sandbox-architecture-doctrine.md`
- `Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md`
- `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- `Docs/superpowers/specs/2026-05-05-sandbox-cleanup-recovery-contract-tests-design.md`

## Goal

Add a read-only `recovery_summary` block to
`GET /api/v1/sandbox/admin/macos-diagnostics` so operators can quickly answer:

- whether recovery data is healthy, degraded, or unavailable
- which cleanup or recovery issues are present
- how many stale sessions, unhealthy sessions, orphan VMs, and image-store
  cleanup candidates exist
- which explicit next admin action is safest

The summary is an operator projection over existing diagnostics. It must not
call the helper, scan the image store, mutate state, or introduce a generic
cross-runtime repair contract.

## Design

`collect_macos_diagnostics()` will continue collecting the existing blocks in
the current order:

- `reconciliation`
- `image_store`
- `observability`

After those blocks are built, a new pure helper will derive a summary from their
already-collected values. This keeps diagnostics truth layered and avoids a
second source of runtime state.

The summary shape is:

- `status`: `healthy`, `action_recommended`, or `unavailable`
- `severity`: `ok`, `warning`, or `error`
- `codes`: stable issue codes for clients and dashboards
- `counts`: stale sessions, unhealthy sessions, skipped active sessions, owned
  orphan VMs, unknown orphan VMs, foreign orphan VMs, total orphan VMs, image
  store GC candidates, and live VMs
- `recommended_action`: short operator action string
- `repair_endpoint`: existing `vz_linux` repair endpoint only when repair is a
  relevant next action
- `cleanup_plan_endpoint`: existing image-store cleanup-plan endpoint only when
  image-store candidates exist
- `notes`: bounded human-readable details for admin UX

## Status Rules

The summary is `unavailable` with `severity=error` when reconciliation is not
computed or has helper/protocol/unavailable reasons. The next action should
point operators back to helper/preflight diagnostics instead of repair.

The summary is `action_recommended` with `severity=warning` when diagnostics are
computed and any of these are non-zero:

- stale session controls
- unhealthy inactive session controls
- owned orphan helper VMs
- unknown or foreign orphan helper VMs
- image-store garbage-collection candidates

The summary is `healthy` with `severity=ok` when diagnostics are computed and no
actionable issue counts are present.

## Non-Goals

- Do not add a new repair endpoint.
- Do not generalize repair beyond existing `vz_linux` reconciliation repair.
- Do not classify host-local runtime cleanup as repairable.
- Do not read log file contents.
- Do not re-query helper, session store, or image store while building the
  summary.
- Do not remove or rename existing diagnostics fields.

## Risks And Mitigations

- Risk: summary codes drift from underlying diagnostics.
  Mitigation: derive codes from existing payload fields and cover representative
  combinations with focused tests.
- Risk: summary appears to authorize destructive repair.
  Mitigation: expose the repair endpoint only as a next-step pointer; existing
  repair remains explicit, admin-only, dry-run-first, and ownership-checked.
- Risk: clients treat `healthy` as runtime availability.
  Mitigation: keep host/runtime/template/helper availability in existing blocks;
  this summary only describes recovery posture.

## Test Strategy

Add focused tests in existing sandbox diagnostics suites:

- schema accepts the new `recovery_summary` block
- healthy reconciliation produces `healthy` and no endpoints
- helper-unavailable reconciliation produces `unavailable`
- stale/unhealthy/orphaned VM state produces `action_recommended` and repair
  endpoint guidance
- image-store GC candidates produce cleanup-plan guidance without repair-only
  claims
- admin endpoint preserves existing fields while returning `recovery_summary`
