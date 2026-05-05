# Sandbox Host-Local Warning Metadata Design

## Goal

Expose additive, machine-readable runtime discovery warnings for host-local
sandbox runtimes so clients can distinguish availability from isolation
strength. This slice targets `seatbelt` and `worktree`.

## Current Gap

Runtime discovery already exposes `boundary_class`, `vm_grade_isolation`, and
`untrusted_eligible`, and admission now enforces static network policy
contracts. The remaining ambiguity is client UX: a runtime can be available
while still being host-local and not VM-grade. Existing prose notes are useful
for humans, but clients should not need to parse note strings to show safety
warnings.

## Design

Add an additive `isolation_warnings: list[str]` field to each
`/api/v1/sandbox/runtimes` item.

Warning codes are static advisory discovery metadata derived from
`RuntimeIsolationMetadata`; they do not replace admission policy, runtime
preflight, or admin diagnostics.

Initial warning codes:

- `host_local_boundary`: runtime executes with a host-local boundary rather
  than a VM boundary.
- `not_vm_grade_isolation`: runtime must not be described as VM-grade.
- `not_untrusted_eligible`: runtime is not eligible for untrusted workloads.

Only host-local runtimes receive this initial warning set. VM-grade runtimes do
not receive host-local warnings. `docker` remains described by its existing
`container` boundary metadata without host-local warnings.

## Non-Goals

- Do not change `SandboxPolicy` admission behavior.
- Do not change runtime preflight availability.
- Do not introduce new runtime execution checks.
- Do not generalize a full warning framework beyond static isolation warnings.

## Risks And Mitigations

- Risk: clients treat warnings as rejection reasons.
  Mitigation: schema and docs describe warnings as advisory discovery metadata.
- Risk: warnings drift from isolation metadata.
  Mitigation: derive warnings from the existing `RuntimeIsolationMetadata`
  helper instead of maintaining a parallel map.
- Risk: over-warning non-VM runtimes like Docker.
  Mitigation: this slice only emits host-local warnings for `seatbelt` and
  `worktree`.

## Tests

- Service discovery tests assert `seatbelt` and `worktree` receive the warning
  codes.
- Service discovery tests assert VM-grade runtimes do not receive host-local
  warnings.
- API shape tests assert `/api/v1/sandbox/runtimes` includes the additive field.
