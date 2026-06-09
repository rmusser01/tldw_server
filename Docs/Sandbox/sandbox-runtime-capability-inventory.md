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
- `reasons`: raw runtime/operator reason strings preserved for diagnostics.
- `normalized_reasons`: stable client-facing reason codes derived from raw
  reasons so clients can group failures without runtime-specific string
  matching.
- `boundary_class`: machine-readable isolation boundary category.
- `vm_grade_isolation`: whether the runtime boundary is VM-grade for isolation
  claims, independent of current host availability.
- `untrusted_eligible`: whether policy may admit this runtime for `untrusted`
  workloads when preflight and host readiness also pass.
- `isolation_warnings`: additive advisory warning codes derived from static
  isolation metadata. These are for client UX and do not replace admission,
  preflight, or diagnostics.
- `network_policy_contract`: static posture for `deny_all` and `allowlist`
  support, strict enforcement, and current-readiness source.
- `session_contract`: static posture for session participation, reuse model,
  live-health-check expectations, and recovery/repair posture. Current host
  truth remains in `available`, raw `reasons`, and admin diagnostics.

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

`/api/v1/sandbox/admin/runtime-diagnostics` is the cross-runtime operator
summary. It is read-only and derived from `/api/v1/sandbox/runtimes` discovery
rows, so it does not introduce another readiness source. The summary groups
runtimes by readiness posture, preserves raw and normalized reasons, reports
host-local isolation warnings, and identifies runtimes with explicit repair
support. Repair support remains scoped to runtimes whose session contract
advertises it; current generalized diagnostics must not imply generic repair
for Docker, Firecracker, Lima, `seatbelt`, `worktree`, or `vz_macos`.

## Portable Runtime Capability Gate

The portable gate in
`tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py` protects this
inventory and `/api/v1/sandbox/runtimes` from drifting apart. It injects
synthetic `RuntimePreflightResult` rows for `docker`, `firecracker`, `lima`,
`vz_linux`, `vz_macos`, `seatbelt`, and `worktree` so the check does not require
Docker, Lima, Firecracker, Apple Virtualization.framework, `sandbox-exec`, or a
prepared VM host.

The gate verifies every `RuntimeType` has implementation-state, isolation,
network-policy, session-contract, normalized-reason, run-status taxonomy, schema,
and inventory coverage. Host-gated smoke tests still own real runtime execution;
this gate only proves the portable capability contract remains complete.

## Runtime Isolation Metadata

Isolation posture is exposed as structured discovery metadata so clients do not
parse human-readable notes for security decisions. `untrusted_eligible` is a
policy eligibility signal, not a statement that the runtime is available or
healthy on the current host.

| Runtime | `boundary_class` | `vm_grade_isolation` | `untrusted_eligible` | Notes |
| --- | --- | --- | --- | --- |
| `docker` | `container` | `false` | `true` | Compatibility path; not VM-grade. |
| `firecracker` | `vm_grade` | `true` | `true` | Linux/KVM host-gated VM runtime. |
| `lima` | `vm_grade` | `true` | `true` | macOS/Linux VM runtime when host preflight passes. |
| `vz_linux` | `vm_grade` | `true` | `true` | Primary Apple silicon Linux VM path. |
| `vz_macos` | `vm_grade_scaffold` | `false` | `false` | Runtime identity exists, but real execution is scaffolded. |
| `seatbelt` | `host_local` | `false` | `false` | Host-local macOS process isolation only. |
| `worktree` | `host_local` | `false` | `false` | Host-local VCS/workspace isolation only. |

Host-local runtimes also expose `isolation_warnings`:

- `host_local_boundary`
- `not_vm_grade_isolation`
- `not_untrusted_eligible`

These warnings are advisory discovery metadata. Policy admission remains the
source of truth for whether a request is accepted, and runtime preflight/admin
diagnostics remain the source of truth for current host readiness.

## Normalized Reason Codes

Runtime discovery preserves raw `reasons` for operator diagnostics and exposes
additive `normalized_reasons` for client logic. The normalized vocabulary is
centralized in `runtime_capabilities.py`.

Runtime discovery also exposes additive `normalized_reason_details` derived
from `normalized_reasons`. Details include `category`, `severity`,
`availability_blocking`, `operator_action`, and `user_message_key`; they are
presentation and triage metadata, not a replacement for raw runtime preflight
facts. Admin runtime diagnostics reuse the same metadata for recommended
actions so clients do not need to duplicate raw reason matching.

| Normalized reason | Meaning |
| --- | --- |
| `runtime_unavailable` | The runtime itself is not available on the current host. |
| `unsupported_os` | The runtime requires a different host operating system. |
| `unsupported_arch` | The runtime requires a different host CPU architecture. |
| `helper_unavailable` | A required helper daemon or helper connection is unavailable. |
| `helper_protocol_mismatch` | A helper protocol or version check failed. |
| `helper_missing` | A configured helper binary/path is missing or unusable. |
| `template_missing` | A required VM image/template artifact is missing. |
| `template_unconfigured` | No template or image source has been configured. |
| `network_policy_unsupported` | The requested network policy is not supported by the runtime. |
| `trust_policy_denied` | Trust-level policy denies this runtime or request shape. |
| `host_prerequisite_missing` | A required host binary, device, or capability is missing. |
| `host_permission_denied` | Host permissions block a required enforcement check. |
| `feature_not_implemented` | Runtime shape exists, but the requested feature is not implemented. |
| `image_store_unavailable` | Image-store configuration or probing failed. |
| `unknown` | A raw reason has no stable normalized mapping yet. |

## Normalized Run Status Reason Codes

Sandbox run status responses preserve raw `phase`, `message`, and `exit_code`
while adding a derived `status_reason_code` for client grouping. The code is
computed from existing status data and aggregate limit counters; it is not a
separate persisted state machine.

| Status reason code | Meaning |
| --- | --- |
| `queued` | The run is queued and has not started. |
| `starting` | The run has been admitted and is starting. |
| `running` | The runtime is executing the run. |
| `completed` | The run completed without a non-zero exit or known limit signal. |
| `limits_applied` | The run completed but output or artifact limits were applied. |
| `nonzero_exit` | The runtime process completed with a non-zero exit code. |
| `policy_failed` | Runtime or network/trust policy admission failed. |
| `runtime_unavailable` | A runtime prerequisite was unavailable or missing. |
| `startup_timeout` | Provisioning/startup timed out before execution completed. |
| `execution_timeout` | Command execution exceeded the configured timeout. |
| `canceled_by_user` | A cancellation request moved the run to killed state. |
| `killed` | The run was killed without a more specific cancellation reason. |
| `queue_ttl_expired` | The run expired while queued. |
| `runtime_error` | The runtime failed without a more specific normalized reason. |
| `unknown` | Existing status data does not map to a known reason code. |

The first Phase 3 taxonomy pass centralizes known runtime aliases for policy
failures, runtime-unavailable failures, queue expiry, timeout, cancellation, and
limit signals. Raw runner messages remain available for operator diagnostics;
clients should group by `status_reason_code` instead of matching runtime
message strings.

Run status responses also expose additive `status_reason_details` derived from
the same code. Details include `category`, `severity`, `terminal`, `retryable`,
`operator_action`, and `user_message_key`; they are response metadata, not
additional persisted run state.

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

Runtime discovery exposes `network_policy_contract` as static posture metadata.
It does not replace `available`, raw `reasons`, `normalized_reasons`, or
`enforcement_ready`, which remain current host/preflight truth. The contract
uses the same support-state vocabulary as this inventory and records whether a
policy can ever be strictly enforced by the runtime.

Effective support is stricter than either static metadata or raw readiness by
itself. A runtime reports `strict_deny_all_supported` or
`strict_allowlist_supported` only when the static contract is `supported` or
`host_gated`, `strict_enforcement=true`, and current readiness for that policy
is true. `scaffold` and `unsupported` modes remain false even if an operator
sets experimental flags.

| Runtime | `deny_all` | `allowlist` | Notes |
| --- | --- | --- | --- |
| `docker` | `supported` | `host_gated` | Deny-all can use container network isolation. Allowlist is effective only when egress enforcement and granular enforcement are both configured; the `network=none` fallback is treated as deny-all, not allowlist. |
| `firecracker` | `host_gated` | `scaffold` | Real mode requires a prepared Linux/KVM host. Allowlist remains planned and is not advertised as effective support. |
| `lima` | `host_gated` | `unsupported` | Deny-all depends on the Lima enforcer preflight. Service discovery intentionally forces allowlist false for execution. |
| `vz_linux` | `host_gated` | `unsupported` | Real helper path accepts `deny_all` only and attaches no guest network device. |
| `vz_macos` | `scaffold` | `unsupported` | No real execution yet. |
| `seatbelt` | `unsupported` | `unsupported` | Deny-all is best effort and must not be reported as strict enforcement. |
| `worktree` | `unsupported` | `unsupported` | Host-local process execution does not provide strict network isolation. |

| Runtime | `deny_all` strict | `deny_all` readiness source | `allowlist` strict | `allowlist` readiness source |
| --- | --- | --- | --- | --- |
| `docker` | `true` | `config` | `true` | `config` |
| `firecracker` | `true` | `runtime_preflight` | `false` | `runtime_preflight` |
| `lima` | `true` | `runtime_preflight` | `false` | `not_applicable` |
| `vz_linux` | `true` | `runtime_preflight` | `false` | `not_applicable` |
| `vz_macos` | `false` | `runtime_preflight` | `false` | `not_applicable` |
| `seatbelt` | `false` | `not_applicable` | `false` | `not_applicable` |
| `worktree` | `false` | `not_applicable` | `false` | `not_applicable` |

## Session Semantics Contract

Runtime discovery exposes `session_contract` as static posture metadata. It
does not replace session create/run APIs or runtime diagnostics. `support_state`
describes the maturity of session participation, while `reuse_model` describes
what can actually be reused across session-backed runs.

| Runtime | `support_state` | `reuse_model` | Live health check required | `recovery_state` | `repair_state` | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `docker` | `supported` | `workspace_only` | `false` | `unsupported` | `unsupported` | Session-backed runs share a control-plane workspace, not a warm container. |
| `firecracker` | `scaffold` | `scaffold` | `false` | `unsupported` | `unsupported` | Session shape exists, but warm VM/session ownership is not normalized. |
| `lima` | `scaffold` | `scaffold` | `false` | `unsupported` | `unsupported` | Session shape exists, but warm VM/session ownership is not normalized. |
| `vz_linux` | `host_gated` | `warm_vm` | `true` | `host_gated` | `host_gated` | Same-session VM reuse requires helper VM health checks; repair is explicit/admin-only. |
| `vz_macos` | `scaffold` | `scaffold` | `false` | `scaffold` | `scaffold` | Runtime identity exists, but real execution and session reuse are scaffolded. |
| `seatbelt` | `scaffold` | `workspace_only` | `false` | `unsupported` | `unsupported` | Host-local workspace participation only; no warm runtime reuse. |
| `worktree` | `scaffold` | `workspace_only` | `false` | `unsupported` | `unsupported` | Host-local workspace participation only; no warm runtime reuse. |

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
image-store correlation, reconciliation, read-only recovery-summary projection,
dry-run repair, and host-gated real VM smoke. Those details are intentionally
not generalized until other runtimes have an equally clear ownership model.

Host-local isolation warnings are now carried through public discovery,
cross-runtime admin diagnostics, and the admin Monitoring page's
`Sandbox Runtime Isolation` card. This covers the operator-visible warning
surface for `seatbelt` and `worktree` while preserving the core policy rule:
host-local runtimes remain weaker than VM-grade isolation and are not eligible
for `untrusted` workloads.

## Current Gaps

| Gap | Runtime(s) | Follow-up phase |
| --- | --- | --- |
| Additional real allowlist implementations remain limited beyond Docker granular enforcement. | all except unsupported paths | Future |
| Host-gated recovery smoke covers `vz_linux` diagnostics, dry-run repair planning, drill-owned stale VM termination, and one manual smoke-owned helper restart drill; destructive repair, host reboot, launchd restart, and broader helper crash recovery remain manual/operator-verified. | all | Phase 4 |
| Recovery/repair ownership exists only for `vz_linux`. | all except `vz_linux` | Phase 4 |
| No single CI job proves real execution for every runtime; the portable session-contract gate covers static `session_contract` capability coverage only. | all | Phase 5 |

## Maintenance Rules

- Add a runtime to this inventory when it is added to `RuntimeType`.
- Keep `/api/v1/sandbox/runtimes` aligned with `RuntimeType`.
- Do not use `available=true` as proof of a security guarantee.
- Do not classify `seatbelt` or `worktree` as `untrusted`-eligible.
- Add network policy metadata for every runtime and keep it separate from
  current `enforcement_ready` host/preflight truth.
- Add session contract metadata for every runtime and keep it separate from
  current host availability and admin diagnostics.
- Keep admin runtime diagnostics session fields aligned with `session_contract`
  so operator surfaces do not invent separate reuse or repair claims.
- Add runtime reason metadata for every `RuntimeReasonCode` and keep it derived
  from `normalized_reasons`, not raw runtime-specific reason strings.
- Prefer `unsupported` over ambiguous wording when a guarantee cannot be
  proven.
- Update this document before expanding `vz_macos`, Apple `containerization`,
  vmnet networking, or new VM runtime support.
