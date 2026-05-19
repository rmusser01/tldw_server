# Sandbox Runtime Isolation Metadata Design

**Status:** Approved for TASK-36 implementation.
**Date:** 2026-05-04.
**Scope:** `/api/v1/sandbox/runtimes` discovery metadata for all sandbox runtimes.

## Problem

Sandbox runtime discovery already exposes availability, trust levels, network enforcement flags, implementation state, and human-readable notes. That is enough for operators, but clients still have to infer core security posture from prose. This is especially risky for host-local runtimes such as `seatbelt` and `worktree`, which may be available for trusted or standard workflows but must never be represented as VM-grade isolation or `untrusted`-eligible.

## Goals

- Add machine-readable isolation posture fields to every runtime discovery row.
- Keep the change additive and backward compatible with existing clients.
- Align the fields with `Docs/Sandbox/sandbox-security-policy-matrix.md`.
- Avoid treating host availability as proof of security readiness.

## Non-Goals

- No runtime admission behavior changes.
- No changes to runner execution, network enforcement, or helper protocols.
- No attempt to make Docker VM-grade or to make `vz_macos` production-ready.

## Proposed Contract

Each `SandboxRuntimeInfo` row gains:

- `boundary_class`: stable runtime boundary category.
- `vm_grade_isolation`: whether the runtime boundary is VM-grade for policy and UX purposes.
- `untrusted_eligible`: whether the runtime may be admitted for `untrusted` workloads when preflight and policy allow it.

Boundary values:

| Runtime | `boundary_class` | `vm_grade_isolation` | `untrusted_eligible` |
| --- | --- | --- | --- |
| `docker` | `container` | `false` | `true` |
| `firecracker` | `vm_grade` | `true` | `true` |
| `lima` | `vm_grade` | `true` | `true` |
| `vz_linux` | `vm_grade` | `true` | `true` |
| `vz_macos` | `vm_grade_scaffold` | `false` | `false` |
| `seatbelt` | `host_local` | `false` | `false` |
| `worktree` | `host_local` | `false` | `false` |

`untrusted_eligible` describes policy eligibility, not current availability. A runtime can be eligible but unavailable on the current host.

## Design Review Notes

- Docker remains `container`, not `vm_grade`, even though the current policy can admit it for `untrusted`; this avoids overstating macOS isolation guarantees.
- `vz_macos` is not marked VM-grade-ready because real execution is scaffolded. Its identity remains separate from its current readiness.
- The fields duplicate information available in docs and policy code intentionally, so API clients do not parse notes or trust-level arrays to decide safety.
- The runtime-specific values should be centralized in `runtime_capabilities.py` rather than embedded ad hoc in `SandboxService.feature_discovery()`.

## Testing

- Add a focused discovery contract test asserting all runtimes expose the three fields.
- Add host-local assertions for `seatbelt` and `worktree`.
- Add VM/container/scaffold assertions to prevent overclaiming Docker and `vz_macos`.
- Keep existing discovery shape tests passing.
