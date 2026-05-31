# Sandbox Session Semantics Contract Design

## Goal

Expose stable session semantics metadata through runtime discovery so clients can
distinguish shared-workspace sessions, warm VM reuse, scaffolded session shapes,
and unsupported recovery/repair posture without parsing runtime notes. This is a
narrow Phase 4 contract slice and does not change session creation, dispatch,
VM reuse, cleanup, or repair behavior.

## Current State

`/api/v1/sandbox/runtimes` already exposes static isolation and network policy
contracts from `runtime_capabilities.py`. The inventory documents session
support in a table, but clients still cannot tell from discovery whether
`Sessions=supported` means a persistent workspace only or actual warm runtime
reuse. Today `vz_linux` is the only runtime with real same-session VM reuse and
helper health validation.

## Design

Add a `session_contract` object to every runtime discovery row. Keep the
contract static and roadmap-oriented, like isolation and network policy
metadata. Current host truth remains in `available`, `reasons`, and runtime
diagnostics.

The contract fields are:

- `support_state`: `supported`, `unsupported`, `scaffold`, `host_gated`, or
  `not_applicable`.
- `reuse_model`: `none`, `workspace_only`, `warm_vm`, or `scaffold`.
- `requires_live_health_check`: whether safe reuse depends on checking live
  runtime state before reuse.
- `recovery_state`: support state for cross-restart recovery semantics.
- `repair_state`: support state for operator/admin repair semantics.

Runtime classification:

- Docker: supported sessions with `workspace_only`; no runtime-owned warm
  container reuse or repair contract in this slice.
- Firecracker and Lima: scaffolded sessions.
- `vz_linux`: host-gated `warm_vm` sessions requiring live helper VM health
  checks, with host-gated recovery/repair posture.
- `vz_macos`: scaffolded sessions.
- `seatbelt` and `worktree`: scaffolded host-local workspace participation,
  not warm runtime reuse and no repair contract.

## Risks And Mitigations

- Risk: clients may treat `session_contract.support_state=supported` as proof
  of current host availability.
  Mitigation: document that `available` and preflight reasons remain host truth.
- Risk: contract could overclaim host-local session guarantees.
  Mitigation: mark host-local runtimes as scaffolded with workspace-only reuse,
  leaving policy/isolation warnings unchanged.
- Risk: fields may become too behavior-specific before all runtimes implement
  parity.
  Mitigation: keep the field set small and descriptive; do not add repair
  endpoints or runtime behavior in this PR.

## Tests

Add focused runtime inventory tests that fail before implementation:

- every runtime discovery row includes `session_contract`;
- `vz_linux` advertises host-gated `warm_vm` semantics and requires live health
  checks;
- Docker advertises workspace-only semantics;
- scaffold and host-local runtimes do not claim warm reuse or repair support;
- the metadata map covers all `RuntimeType` values and rejects unknown runtimes.

## Documentation

Update `Docs/Sandbox/sandbox-runtime-capability-inventory.md` to document the
new discovery field and replace the broad "session semantics are not
normalized" gap with narrower remaining behavior/recovery contract-test work.
