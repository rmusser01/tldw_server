# Sandbox Runtime Capability Inventory

**Status:** Phase 0 inventory baseline.
**Date:** 2026-05-02.
**Scope:** `docker`, `firecracker`, `lima`, `vz_linux`, `vz_macos`,
`seatbelt`, and `worktree`.

## Purpose

This document records the current support state for every sandbox runtime using
the roadmap vocabulary from
`Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md`.

It is an inventory, not a promise that every runtime should reach feature
parity. Runtime discovery and preflight remain the source of truth for the
current host. This document classifies subsystem guarantees so future plans can
avoid overclaiming isolation, networking, recovery, or CI coverage.

## State Vocabulary

| State | Meaning |
| --- | --- |
| `supported` | Implemented in normal code paths and not host-gated beyond ordinary runtime dependencies. |
| `unsupported` | Not implemented or intentionally rejected. |
| `scaffold` | Shape exists, but real execution or enforcement is incomplete. |
| `host_gated` | Implemented only on prepared hosts or with explicit opt-in prerequisites. |
| `not_applicable` | The capability does not apply to this runtime model. |

## Discovery Contract

`/api/v1/sandbox/runtimes` is the summarized client-facing discovery surface.
It should include every value from `RuntimeType` and expose two separate
concepts:

- `available`: current host/preflight truth.
- `implementation_state`: roadmap maturity label independent of whether this
  specific host has the required binaries, helper, VM support, or feature flags.

| Runtime | `implementation_state` | Discovery source |
| --- | --- | --- |
| `docker` | `supported` | `SandboxService.feature_discovery()` plus `docker_available()` |
| `firecracker` | `host_gated` | `SandboxService.feature_discovery()` plus `firecracker_available()` |
| `lima` | `host_gated` | `LimaRunner.preflight()` |
| `vz_linux` | `host_gated` | `VZLinuxRunner.preflight()` |
| `vz_macos` | `scaffold` | `VZMacOSRunner.preflight()` |
| `seatbelt` | `host_gated` | `SeatbeltRunner.preflight()` |
| `worktree` | `supported` | `WorktreeRunner.preflight()` |

Discovery is intentionally summarized. Admin/operator diagnostics can expose
helper, template, reconciliation, and image-store details that should not be
duplicated into the public discovery payload.

## Trust-Level Support

| Runtime | `trusted` | `standard` | `untrusted` | Notes |
| --- | --- | --- | --- | --- |
| `docker` | `supported` | `supported` | `supported` | Broad default runtime, but policy must not claim VM-grade isolation where that is required. |
| `firecracker` | `supported` | `supported` | `supported` | VM-grade path when host prerequisites and real execution are available. |
| `lima` | `supported` | `supported` | `supported` | VM isolation path with strict host preflight requirements. |
| `vz_linux` | `supported` | `supported` | `supported` | Primary Apple silicon Linux VM path. |
| `vz_macos` | `scaffold` | `scaffold` | `scaffold` | Trust levels are advertised through preflight, but real execution is not implemented. |
| `seatbelt` | `supported` | `host_gated` | `unsupported` | `standard` requires `TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED=1`; never VM-grade. |
| `worktree` | `supported` | `supported` | `unsupported` | Host-local VCS isolation only; never VM-grade. |

## Network Policy Support

| Runtime | `deny_all` | `allowlist` | Notes |
| --- | --- | --- | --- |
| `docker` | `supported` | `host_gated` | Deny-all can use container network isolation. Allowlist depends on configured egress enforcement. |
| `firecracker` | `host_gated` | `scaffold` | Real mode requires a prepared Linux/KVM host. Allowlist is advertised only behind explicit enforcement flags and remains planned. |
| `lima` | `host_gated` | `unsupported` | Deny-all depends on the Lima enforcer preflight. Service discovery intentionally forces allowlist false for execution. |
| `vz_linux` | `host_gated` | `unsupported` | Real helper path accepts `deny_all` only and attaches no guest network device. |
| `vz_macos` | `scaffold` | `unsupported` | No real execution yet. |
| `seatbelt` | `unsupported` | `unsupported` | Deny-all is best effort and must not be reported as strict enforcement. |
| `worktree` | `unsupported` | `unsupported` | Host-local process execution does not provide strict network isolation. |

## Execution And Lifecycle Support

| Runtime | Real execution | Interactivity | Sessions | Cancellation/timeouts | Artifacts |
| --- | --- | --- | --- | --- | --- |
| `docker` | `supported` | `supported` | `supported` | `supported` | `supported` |
| `firecracker` | `host_gated` | `unsupported` | `scaffold` | `supported` | `supported` |
| `lima` | `host_gated` | `unsupported` | `scaffold` | `supported` | `supported` |
| `vz_linux` | `host_gated` | `unsupported` | `supported` | `supported` | `supported` |
| `vz_macos` | `scaffold` | `unsupported` | `scaffold` | `scaffold` | `scaffold` |
| `seatbelt` | `host_gated` | `unsupported` | `scaffold` | `supported` | `supported` |
| `worktree` | `supported` | `unsupported` | `scaffold` | `supported` | `supported` |

Session support means a runtime can participate in the sandbox session API.
It does not imply a warm VM, warm container, or long-lived executor unless the
runtime notes say so. Today `vz_linux` is the only VM path with real same-session
VM reuse.

## Recovery And Diagnostics Support

| Runtime | Public discovery | Admin diagnostics | Reconciliation/repair | Resource usage | CI coverage |
| --- | --- | --- | --- | --- | --- |
| `docker` | `supported` | `scaffold` | `unsupported` | `supported` | `supported` |
| `firecracker` | `supported` | `scaffold` | `unsupported` | `scaffold` | `supported` |
| `lima` | `supported` | `scaffold` | `unsupported` | `scaffold` | `supported` |
| `vz_linux` | `supported` | `supported` | `supported` | `host_gated` | `host_gated` |
| `vz_macos` | `supported` | `supported` | `scaffold` | `scaffold` | `scaffold` |
| `seatbelt` | `supported` | `supported` | `unsupported` | `scaffold` | `supported` |
| `worktree` | `supported` | `scaffold` | `unsupported` | `supported` | `supported` |

`vz_linux` currently has the deepest operator path: helper diagnostics,
image-store correlation, reconciliation, dry-run repair, and host-gated real VM
smoke. Those details are intentionally not generalized until other runtimes have
an equally clear ownership model.

## Current Gaps

| Gap | Runtime(s) | Follow-up phase |
| --- | --- | --- |
| Public discovery lacks normalized state labels and reason taxonomy. | all | Phase 3 |
| Host-local runtimes need clearer docs/API warnings that they are not VM-grade. | `seatbelt`, `worktree` | Phase 2 |
| Allowlist support is inconsistent and often scaffold-only. | all except host-local unsupported paths | Phase 2 |
| Session semantics are not normalized across warm VM, container, and host-local reuse. | all | Phase 4 |
| Recovery/repair ownership exists only for `vz_linux`. | all except `vz_linux` | Phase 4 |
| CI has no single cross-runtime capability gate. | all | Phase 5 |

## Maintenance Rules

- Add a runtime to this inventory when it is added to `RuntimeType`.
- Keep `/api/v1/sandbox/runtimes` aligned with `RuntimeType`.
- Do not use `available=true` as proof of a security guarantee.
- Do not classify `seatbelt` or `worktree` as `untrusted`-eligible.
- Prefer `unsupported` over ambiguous wording when a guarantee cannot be
  proven.
- Update this document before expanding `vz_macos`, Apple `containerization`,
  vmnet networking, or new VM runtime support.
