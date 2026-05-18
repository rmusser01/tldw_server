# Sandbox Module Roadmap Design

**Status:** Active roadmap spec, rebaselined 2026-05-16 after the initial
immediate PR queue landed.
**Date:** 2026-05-02.
**Scope:** Full sandbox module: API, orchestrator, store, artifacts, streaming,
admin surfaces, CI, security, and all current runtime families.

## Related Docs

- `Docs/Sandbox/sandbox-architecture-doctrine.md`
- `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- `Docs/Sandbox/sandbox-security-policy-matrix.md`
- `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
- `Docs/Sandbox/vz-linux-prepared-host-evidence.md`
- `Docs/Design/2026-05-02-apple-containerization-evaluation.md`
- `tldw_Server_API/app/core/Sandbox/README.md`

## Goal

Create a structured 6-12 month roadmap plus an immediate PR queue for taking the
sandbox module from feature-rich prototype paths to a stable, diagnosable, and
security-reviewable subsystem.

The roadmap uses `vz_linux` as the strongest current VM-backed proving ground,
then extracts stable contracts across Docker, Firecracker, Lima, `vz_macos`,
`seatbelt`, and `worktree`. It should guide future plans without forcing every
runtime to move at the same maturity level.

## Current Baseline

The sandbox module already has substantial production-shaped pieces:

- REST and WebSocket API for sessions, runs, logs, artifacts, cancellation, and
  runtime discovery.
- Orchestrator support for sessions, idempotency, queueing, artifact storage,
  and status persistence.
- Runtimes for Docker, Firecracker, Lima, `vz_linux`, `vz_macos`, `seatbelt`,
  and `worktree`.
- Docker hardening defaults, limited interactivity, WebSocket resume, signed
  URLs, artifact quotas, and resource usage reporting.
- Firecracker and Lima runtime surfaces with varying levels of real execution,
  strict admission, and host prerequisites.
- macOS runtime doctrine, operator notes, helper protocol, image store, helper
  lifecycle command, host-gated smoke path, reconciliation, repair, and startup
  warnings.
- `vz_linux` real Apple silicon VM execution with helper-backed boot, guest
  command execution, session VM reuse, output caps, artifact audit metadata,
  and guest capability readiness details.
- Apple `container`/`containerization` evaluation that recommends OCI-aware
  image-store metadata before deeper dependency or runtime rewrites.

The biggest remaining risk is not lack of features. It is uneven runtime
maturity: each runtime reports and enforces a different subset of guarantees,
and some operational behaviors are mature only for `vz_linux`.

2026-05-16 rebaseline:

- The original immediate PR queue is no longer a pending to-do list. It now
  serves as a completed-foundation checklist and evidence map.
- `vz_linux` remains the strongest proving ground, but real execution and
  failure drills are still prepared-host and operator-gated.
- The next pragmatic work should close evidence and lifecycle gaps, not reopen
  metadata, inventory, security, or public-docs slices that already landed.

## Strategy

Use a stability-first roadmap with a parity overlay.

1. Harden one VM path (`vz_linux`) until the lifecycle is boring.
2. Extract only proven contracts into shared sandbox capability, diagnostics,
   recovery, and security rules.
3. Apply those shared rules to other runtimes with explicit `supported`,
   `unsupported`, or `not_applicable` states.
4. Keep feature expansion gated behind diagnostics, tests, and clear operator
   behavior.

This avoids two failure modes:

- building abstractions from the least mature runtimes
- overfitting the whole module to macOS-specific implementation details

## Runtime Families

### Docker

Role: broadest default runtime and best place for fast developer feedback.

Roadmap stance:

- keep broad availability and interactive support
- harden cleanup, egress allowlist, artifact caps, and resource reporting
- do not claim VM-grade isolation for `untrusted` if the policy requires a VM
- preserve Docker as the baseline for API behavior and fast local tests

### Firecracker

Role: Linux VM-grade isolation path.

Roadmap stance:

- make runtime discovery and host prep as clear as `vz_linux`
- align cancellation, artifacts, metrics, and cleanup with shared contracts
- defer advanced features until real execution parity is demonstrably stable

### Lima

Role: compatibility VM path, especially where first-party macOS helper support
is unavailable or not desired.

Roadmap stance:

- keep fail-closed admission and explicit enforcement readiness
- document slower lifecycle and host dependency constraints
- avoid competing with `vz_linux` for Apple silicon production path unless a
  concrete capability gap requires it

### `vz_linux`

Role: primary macOS Apple silicon VM-backed Linux sandbox path.

Roadmap stance:

- remain the proving ground for helper lifecycle, image-store, diagnostics,
  guest protocol, recovery, and host-gated CI
- keep current repo-owned bundle path while making metadata OCI-aware
- do not require Apple `container`
- do not attach networking under `deny_all`
- do not replace `tldw-agent` until the current guest path is stable and a
  focused comparison proves clear benefit

### `vz_macos`

Role: future macOS guest VM path.

Roadmap stance:

- keep scaffold honest until real execution exists
- reuse helper-owned readiness and provenance rules from `vz_linux`
- avoid promising parity before image/template, guest control, and isolation
  semantics are concrete

### `seatbelt`

Role: host-local trusted-workflow isolation bridge.

Roadmap stance:

- keep weaker guarantee claims explicit
- never treat as VM-grade or acceptable for `untrusted`
- continue to make availability and `sandbox-exec` deprecation visible
- align audit, cleanup, and diagnostics with shared runtime contracts where
  applicable

### `worktree`

Role: host-local VCS workspace isolation for trusted and standard workflows.

Roadmap stance:

- keep weaker guarantee claims explicit
- never treat as VM-grade or acceptable for `untrusted`
- preserve explicit repository allowlisting, sensitive environment stripping,
  and Linux `unshare` readiness checks
- align cancellation, cleanup, artifacts, audit, and diagnostics with other
  host-local runtime contracts where applicable
- do not make it a long-lived session reuse path until health and recovery
  semantics are concrete

## Phased Roadmap

### Phase 0: Baseline Inventory

Goal: establish one authoritative view of current runtime capabilities, gaps,
risks, and tests.

Deliverables:

- runtime capability matrix covering Docker, Firecracker, Lima, `vz_linux`,
  `vz_macos`, `seatbelt`, and `worktree`
- current support states for trust levels, network policies, interactivity,
  session reuse, artifacts, cancellation, recovery, host readiness, and CI
- docs/code drift report for Sandbox README, API quick guide, operator notes,
  and runtime discovery
- test coverage map by runtime and behavior

Exit criteria:

- every runtime capability is classified as `supported`, `unsupported`,
  `scaffold`, `host_gated`, or `not_applicable`
- public docs do not imply stronger guarantees than discovery/preflight reports
- next PR queue is grounded in explicit gaps, not inferred roadmap memory

### Phase 1: `vz_linux` Production Stability

Goal: make the first-party Apple silicon Linux VM path repeatable, debuggable,
and safe enough to serve as the VM contract reference.

Deliverables:

- OCI-aware image-store metadata scaffolding while keeping bundle boot unchanged
- helper lifecycle version/compatibility checks and operator upgrade behavior
- richer helper diagnostics for boot logs, serial logs, resource stats, and VM
  health snapshots
- session reuse health checks that reject stale or unhealthy VMs before reuse
- recovery behavior for helper crash, host reboot, stale socket, stuck boot,
  stuck readiness, and guest-agent mismatch
- host-gated CI acceptance gates for manual and nightly real VM smoke

Exit criteria:

- a prepared Apple silicon host can run one command, same-session reuse, helper
  restart/recovery, and cleanup through documented operator commands
- diagnostics explain failures without requiring source-code inspection
- no other runtime is required to adopt macOS-only details

### Phase 2: Security Contract Hardening

Goal: make isolation guarantees explicit and testable across all runtimes.

Deliverables:

- trust-level policy matrix for runtime eligibility and required guarantees
- network policy matrix for `deny_all`, `allowlist`, and unsupported states
- workspace mount and user model contract per runtime
- artifact exposure and path-safety contract shared by all runtimes
- helper/request allowlisting and bounded input validation requirements
- audit event schema for lifecycle, policy decisions, artifacts, repair, and
  admin actions

Exit criteria:

- `untrusted` cannot silently fall back to a weaker runtime
- each runtime advertises and enforces only guarantees it can prove
- security-relevant behavior has focused tests or explicit host-gated coverage

### Phase 3: Runtime Parity Overlay

Goal: define a shared runtime capability contract without pretending every
runtime implements every feature.

Deliverables:

- normalized runtime discovery schema for capability states and reasons
- common preflight vocabulary for missing binaries, unsupported OS, helper
  mismatch, template missing, network unsupported, and policy denied
- shared status/error taxonomy for cancellation, timeout, runtime unavailable,
  policy failure, artifact truncation, and stale session
- common shape for resource usage, image/template provenance, and execution
  mode

Exit criteria:

- clients and ACP can make decisions from discovery without runtime-specific
  string guessing
- adding a new runtime means filling out capability contracts, not inventing
  a parallel readiness model

### Phase 4: Cross-Runtime Reliability

Goal: align lifecycle behavior where runtimes already share concepts.

Deliverables:

- cleanup/recovery contract tests for cancel, timeout, failed start, stuck run,
  and orphaned resources
- artifact quota and log cap tests across Docker, Firecracker, Lima, `vz_linux`,
  `vz_macos`, `seatbelt`, and `worktree` where applicable
- persistent store reconciliation rules for sessions/runs that outlive worker
  processes
- operator repair surfaces generalized only where the ownership model is
  strong enough
- warm-pool and session reuse behavior explicitly scoped to runtimes that can
  prove health

Exit criteria:

- users see consistent terminal states and error classes across runtimes
- failed runs do not leak process/VM/container resources under normal failure
  modes
- repair actions stay explicit, dry-run-first, and ownership-checked

### Phase 5: Operator And Admin UX

Goal: make sandbox operation understandable without reading implementation code.

Deliverables:

- admin runtime dashboard/API model for readiness, warnings, repair plans, image
  store health, queue pressure, and CI status
- documented operator playbooks for Docker, Firecracker, Lima, host-local
  runtimes, and macOS VZ hosts
- startup warning and diagnostics summaries that distinguish blocking from
  non-blocking conditions
- clear guidance for host-gated smoke, expected skips, and failure triage

Exit criteria:

- an operator can answer "why is this runtime unavailable?" from admin surfaces
- host preparation and smoke tests are copy/paste-safe
- docs match current runtime discovery and preflight behavior

### Phase 6: Expansion And Optimization

Goal: add larger capabilities only after stable contracts exist.

Candidates:

- `vz_macos` real execution
- vmnet-backed `vz_linux` allowlist networking
- direct Apple `containerization` Swift package spike
- APFS clone or filesystem snapshot execution
- Firecracker real execution parity and snapshots
- optimized Linux kernel/rootfs benchmark
- richer guest protocol inspired by `vminitd`
- resource stats and boot-log parity across VM runtimes

Exit criteria:

- each expansion has a focused design
- dependency additions are isolated behind adapters
- compatibility and rollback paths are explicit

## Immediate Queue Rebaseline

The initial immediate queue has been worked through as a foundation set. Treat
this table as the current status map before selecting new sandbox work.

| Item | Current status | Evidence / source of truth | Residual boundary |
| --- | --- | --- | --- |
| OCI-aware image-store metadata scaffolding | Complete for metadata-only bundle provenance. | `SandboxImageStore` now records `artifact_format`, current bundles use `tldw_bundle`, and macOS diagnostics expose template provenance. See `Docs/superpowers/plans/2026-05-02-oci-aware-image-store-metadata-implementation-plan.md`. | No helper boot, OCI pull, or Apple `containerization` runtime change. |
| Runtime capability inventory | Complete as the current support-state inventory. | `Docs/Sandbox/sandbox-runtime-capability-inventory.md` plus the portable runtime capability gate. | Real execution remains host-gated per runtime; the inventory is not itself a live-host proof. |
| `vz_linux` helper lifecycle compatibility | Complete for operator-owned direct helper, launchd-drill, signing, socket safety, status, and protocol/version checks. | `Docs/Sandbox/macos-runtime-operator-notes.md`, `tools/macos-vz-helper/scripts/vz-helperctl.py`, and the launchd/helper lifecycle tasks. | No automatic installation or hidden upgrade daemon. Host reboot remains a manual operator procedure. |
| Boot logs and resource diagnostics | Complete for current read-only diagnostics. | `TASK-404`, `Docs/superpowers/specs/2026-05-16-vz-boot-resource-diagnostics-design.md`, and `Docs/Sandbox/macos-runtime-operator-notes.md`. | Current resource snapshots are configured VM facts plus uptime, not live CPU/RSS/I/O telemetry. |
| Cross-runtime cleanup contract tests | Complete for portable cleanup/recovery contracts. | `TASK-62` and `Docs/superpowers/specs/2026-05-05-sandbox-cleanup-recovery-contract-tests-design.md`. | Real VM cleanup behavior remains host-gated where runtime hosts are required. |
| Security policy matrix | Complete as active guidance, with supporting runtime contracts. | `Docs/Sandbox/sandbox-security-policy-matrix.md`, isolation/network metadata tasks, audit metadata task, and VZ helper `create_vm` validation. | Future runtimes must update the matrix before claiming stronger guarantees. |
| Host-gated CI acceptance policy | Complete as policy/workflow scaffolding, with manual-only drills. | `.github/workflows/vz-linux-host-gated.yml`, `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`, `TASK-145`, `TASK-153`, and launchd-drill tasks. | Ongoing work is prepared-host evidence and regression triage, not PR/push-triggered CI. |
| Public docs reconciliation | Complete for the current diagnostics and runtime posture. | `TASK-89`, `Docs/API-related/Sandbox_API.md`, `tldw_Server_API/app/core/Sandbox/README.md`, and macOS operator notes. | Docs still need maintenance when runtime behavior changes; do not infer support beyond runtime discovery. |

## Next Pragmatic Queue

1. **Prepared-host acceptance evidence and gap tracking**

   Capture manual/nightly `vz_linux` host-gated run evidence from a prepared
   Apple silicon host, including command execution, same-session reuse, helper
   restart drill, launchd-drill where explicitly requested, artifact upload, and
   expected skips. Record results in
   `Docs/Sandbox/vz-linux-prepared-host-evidence.md`. This is the best next
   slice because it tests the stability path the preceding PRs built.

2. **Remaining `vz_linux` lifecycle drill gaps**

   Add narrow manual drills or documented operator checks for stale socket,
   stuck boot/readiness, and guest-agent mismatch behavior. Keep host reboot
   out of scheduled CI and avoid destructive repair defaults.

3. **Operator/admin status consolidation**

   After real prepared-host evidence exists, expose the resulting readiness,
   warning, repair-plan, and CI-status signals through admin/operator surfaces
   without duplicating runtime-specific logic in clients.

4. **Expansion design only after evidence (Phase 6)**

   Resume expansion candidates such as `vz_macos` real execution, vmnet
   allowlist networking, APFS clone provisioning, Firecracker parity, or direct
   Apple `containerization` adapters only after the evidence loop above is
   boring enough to catch regressions.

## Design Risks And Mitigations

- Risk: using `vz_linux` as the proving ground could overfit the whole sandbox
  module to macOS.
  Mitigation: only promote runtime-neutral contracts; keep macOS-specific
  helper details in macOS docs and tests.

- Risk: parity language could imply every runtime must implement every feature.
  Mitigation: parity means consistent reporting, not identical support.

- Risk: host-local convenience runtimes could be mistaken for untrusted-code
  isolation.
  Mitigation: `seatbelt` and `worktree` stay explicitly weaker than VM-backed
  runtimes and must not satisfy `untrusted`.

- Risk: direct Apple `containerization` reuse could increase dependency and
  macOS version risk before the current helper path is stable.
  Mitigation: require a focused adapter spike before adding the package graph.

- Risk: networking expansion could weaken `deny_all`.
  Mitigation: no network device under `deny_all`; vmnet work starts as a policy
  design with host-gated tests.

- Risk: repair endpoints can become destructive.
  Mitigation: dry-run-first, explicit mutation flags, ownership metadata checks,
  and fail-closed behavior when helper truth is unavailable.

- Risk: docs become stale faster than runtime code.
  Mitigation: Phase 0 inventory and Phase 3 discovery schema should make docs
  cite runtime capability states instead of hand-written assumptions.

## Non-Goals

- Do not replace the current helper with Apple `container`.
- Do not require Apple `container` for operators.
- Do not make `seatbelt` VM-grade.
- Do not make `worktree` VM-grade.
- Do not attach networking for `deny_all`.
- Do not implement `vz_macos` real execution until the shared contracts are
  stable enough to reuse.
- Do not make every runtime support every feature before stabilizing how support
  is reported.

## Current Handoff

The next implementation artifact should target prepared-host acceptance evidence
and remaining lifecycle drill gaps. Keep the slice operator-first and
host-gated: no PR-triggered real VM execution, no scheduled destructive drills,
no host reboot automation, no networking expansion, and no `vz_macos` real
execution until the shared contracts remain stable under real-host evidence.
