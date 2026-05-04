# Sandbox Security Policy Matrix

**Status:** Active guidance for sandbox policy and runtime work.
**Date:** 2026-05-03.
**Scope:** `docker`, `firecracker`, `lima`, `vz_linux`, `vz_macos`,
`seatbelt`, and `worktree`.

## Purpose

This matrix defines the security contract that runtime plans, policy
admission, diagnostics, and operator docs should use when describing sandbox
guarantees. It complements:

- `Docs/Sandbox/sandbox-architecture-doctrine.md`
- `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- `tldw_Server_API/app/core/Sandbox/policy.py`
- `tldw_Server_API/app/core/Sandbox/runtime_capabilities.py`

This document is not a host-readiness report. Runtime preflight and
diagnostics remain the source of truth for whether a specific host can run a
specific runtime today.

## Contract Vocabulary

| Term | Meaning |
| --- | --- |
| `policy-admitted` | The Python policy layer can admit this runtime for the stated request when preflight permits it. |
| `VM-grade` | The runtime boundary is a VM or VM-like isolation boundary appropriate for `untrusted` workloads. |
| `host-local` | The runtime executes directly on the host with process, filesystem, VCS, or OS sandbox constraints. |
| `strict deny-all` | Runtime networking is blocked by an enforced runtime or VM boundary, not only by convention. |
| `strict allowlist` | Runtime networking permits only explicitly configured destinations and blocks all others. |
| `scaffold` | Shape exists, but the guarantee is not ready for normal operator use. |

Public runtime discovery exposes these concepts through `boundary_class`,
`vm_grade_isolation`, and `untrusted_eligible`. Those fields describe policy
posture and boundary category; they do not replace current-host `available`,
`reasons`, or runtime preflight checks.

## Trust-Level Eligibility

`trusted` and `standard` workloads may use VM-backed or host-local runtimes
when policy and preflight allow them. `untrusted` workloads must not silently
fall back to a weaker host-local runtime.

| Runtime | Boundary class | `trusted` | `standard` | `untrusted` | Notes |
| --- | --- | --- | --- | --- | --- |
| `docker` | container | policy-admitted | policy-admitted | policy-admitted, not VM-grade | Existing compatibility path. Do not describe as the macOS `untrusted` target. |
| `firecracker` | VM-grade | policy-admitted | policy-admitted | policy-admitted | Linux/KVM host-gated VM runtime. |
| `lima` | VM-grade | policy-admitted | policy-admitted | policy-admitted | macOS-host Linux VM path with strict preflight requirements. |
| `vz_linux` | VM-grade | policy-admitted | policy-admitted | policy-admitted | Primary Apple silicon Linux VM path. |
| `vz_macos` | VM-grade scaffold | scaffold | scaffold | scaffold | Runtime identity and preflight shape exist; real execution is not implemented. |
| `seatbelt` | host-local | policy-admitted | opt-in only | rejected | `standard` requires `TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED=1`; never VM-grade. |
| `worktree` | host-local | policy-admitted | policy-admitted | rejected | VCS/workspace isolation only; never VM-grade. |

Policy requirements:

- `seatbelt` and `worktree` must reject `untrusted` even if callers omit
  runtime preflight metadata.
- Runtime preflight `supported_trust_levels` may further restrict admission.
- No code path may downgrade from a requested VM-grade runtime to a host-local
  runtime to make execution succeed.

## Network Semantics

Network policy names are request semantics, not proof that every runtime can
enforce them. Public discovery and diagnostics must report enforcement
readiness separately from runtime availability.

| Runtime | `deny_all` semantics | `allowlist` semantics | Operator rule |
| --- | --- | --- | --- |
| `docker` | Supported through container network isolation where configured. | Supported only when Docker egress enforcement is enabled and healthy. | Do not advertise allowlist unless enforcement is configured. |
| `firecracker` | Host-gated strict deny-all when VM networking is absent or blocked. | Scaffold. | Fail closed when host enforcement cannot prove the requested policy. |
| `lima` | Host-gated strict deny-all through Lima enforcer readiness. | Unsupported for execution today. | `allowlist` requests must be rejected even if test overrides report readiness. |
| `vz_linux` | Strict deny-all by Python admission plus helper rejection of non-`deny_all` VM creation; current VM config attaches no network device. | Unsupported. | Helper metadata should echo the accepted policy for diagnostics. |
| `vz_macos` | Scaffold only. | Unsupported. | Do not claim real network enforcement until real execution exists. |
| `seatbelt` | Best effort only; not firewall-backed or VM-grade. | Unsupported. | Discovery may report availability while strict network support is false. |
| `worktree` | Unsupported as strict isolation. | Unsupported. | Host process execution must not be represented as strict egress control. |

## Workspace And Mount Model

| Runtime | Workspace model | Mount/write rule |
| --- | --- | --- |
| `docker` | Container workspace. | Writes should be limited to the run/session workspace and configured artifact paths. |
| `firecracker` | VM guest workspace. | Host paths must be passed through explicit VM devices or prepared images only. |
| `lima` | VM guest workspace. | Host sharing must be explicit and covered by Lima preflight. |
| `vz_linux` | Guest workspace exposed through the image/agent contract. | Workspace devices and guest agent readiness must be verified before execution or session reuse. |
| `vz_macos` | Scaffold. | Do not rely on workspace isolation until the real runner defines mount behavior. |
| `seatbelt` | Run-local host workspace, isolated `HOME`, and isolated temp dirs. | Seatbelt profile should allow writes only to workspace/home/temp paths and control files must stay outside writable workspace. |
| `worktree` | Temporary git worktree. | Repo path must be allowlisted; sensitive host environment must be stripped. |

## User And Process Model

| Runtime | User/process expectation |
| --- | --- |
| `docker` | Prefer non-root container users and runner hardening defaults where image support permits it. |
| `firecracker` | Guest process identity belongs inside the VM image and should not inherit host credentials. |
| `lima` | Guest process identity belongs inside the VM and must not be confused with host user permissions. |
| `vz_linux` | Guest agent owns in-guest command execution; host helper owns VM lifecycle, not command policy. |
| `vz_macos` | Undefined until real execution lands. |
| `seatbelt` | Host process runs under the API user's macOS account with seatbelt restrictions. Treat as trusted/standard only. |
| `worktree` | Host process runs under the API user's account. Treat as trusted/standard only. |

## Artifact And Output Exposure

Artifact handling is a trusted-control-plane responsibility. Runtime-specific
execution code may produce outputs, but API exposure must remain guarded by the
sandbox artifact store and policy limits.

Required rules:

- Enforce `SANDBOX_MAX_LOG_BYTES`, `SANDBOX_MAX_ARTIFACT_FILE_BYTES`, and
  `SANDBOX_MAX_ARTIFACT_TOTAL_BYTES` on runtime output and artifact capture.
- Never expose raw host paths in user-facing artifact URLs, audit rows, or
  startup warnings.
- Normalize and validate artifact paths before publication.
- Treat helper-provided run metadata as external input when correlating
  diagnostics, reconciliation, and image-store manifests.
- Prefer skip-and-record behavior for oversized artifacts over failing a
  successful run unless the runtime contract explicitly says otherwise.

## Helper And Request Allowlisting

Helper-backed runtimes must keep request admission in Python and live runtime
truth in the helper.

| Runtime | Allowlisting expectation |
| --- | --- |
| `vz_linux` | Python admits only supported trust/network/runtime requests; helper independently rejects unsupported VM creation, unsafe sockets, and unsupported protocol versions. |
| `vz_macos` | Must follow the same Python/helper split when real execution lands. |
| `lima` | Python must revalidate strict policy at execution time before dispatch. |
| `firecracker` | Runtime config paths must be validated before VM start. |
| `seatbelt` | Command path resolution and environment construction must be controlled and sanitized. |
| `worktree` | Allowed repo directories and safe environment construction are mandatory. |

## Audit Expectations

All runtimes should emit enough metadata to explain why a run was admitted,
rejected, or limited in capability reporting. Audit data should include:

- requested runtime and effective runtime
- trust level and network policy
- policy hash or equivalent policy material
- runtime preflight reasons when admission fails
- resource caps applied to the run
- artifact skip counters and output truncation state
- helper protocol/runtime version for helper-backed runtimes

Audit data must not include secrets, unredacted environment variables, raw
credentials, or raw host filesystem paths that are not already operator-facing
diagnostic paths.

## Maintenance Rules

- Update this matrix when adding a runtime, trust level, network policy, or
  runtime-specific artifact path.
- Keep `sandbox-runtime-capability-inventory.md` as the current support-state
  inventory and this document as the security contract.
- Add focused tests when a row describes a fail-closed admission rule.
- Prefer explicit `unsupported` reasons over ambiguous best-effort language.
- Do not use `available=true` as proof of a security guarantee.
