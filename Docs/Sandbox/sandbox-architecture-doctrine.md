# Sandbox Architecture Doctrine

**Status:** Active guidance for sandbox subsystem work as of 2026-03-11.

## Purpose

This document defines the durable architectural rules for the sandbox subsystem.
Future plans for `seatbelt`, `vz_linux`, `vz_macos`, diagnostics, image-store
work, session recovery, and new runtimes should reference this document instead
of re-deriving the same invariants per milestone.

The doctrine intentionally separates:

- stable subsystem rules
- runtime-specific implementation details
- milestone-specific design and implementation plans

## Scope

This doctrine applies to:

- `tldw_Server_API/app/core/Sandbox/`
- runtime helpers and guest agents under `tools/`
- sandbox diagnostics, operator docs, and runtime discovery surfaces
- image and template production paths used by VM-backed runtimes

## Source Material

This doctrine synthesizes lessons from:

- Bouvet
- Gobii sandbox compute architecture
- Nono
- CUA/Lume
- CodeRunner
- Apple [`container`](https://github.com/apple/container) and
  [`containerization`](https://github.com/apple/containerization)

It encodes the parts that are useful as enduring subsystem rules rather than
copying any one implementation wholesale.

## Apple Container Prior-Art Guidance

Apple's [`container`](https://github.com/apple/container) and
[`containerization`](https://github.com/apple/containerization) projects are
first-party prior art for macOS-hosted Linux workloads, not a drop-in
replacement for this sandbox subsystem. They validate several choices this
subsystem has adopted as foundational principles:

- Apple silicon as the primary serious macOS VM target
- per-workload lightweight Linux VMs instead of one shared Linux VM
- `Virtualization.framework` as the VM control surface
- host services managed through `launchd`, XPC-style helper boundaries, and
  unified logging
- OCI-compatible image flow for portable Linux workload artifacts
- vmnet-backed networking when network access is intentionally enabled
- a guest init/control service over vsock

Future macOS VM plans should compare against this prior art before inventing
new helper, image, network, or guest-control mechanics. That comparison must be
bounded by this repo's sandbox contract:

- Python still owns admission, run/session identity, policy, artifacts, queueing,
  and audit.
- The host helper still owns live VM truth and must expose a stable repo-owned
  protocol.
- The guest agent still owns in-guest execution readiness and must remain
  versioned separately from the host helper protocol.
- `untrusted` workloads still require VM-grade isolation and fail-closed
  admission.
- `seatbelt` and future `vz_macos` work must not be forced through a
  Linux-container abstraction.

Do not require the `container` CLI to be installed for the sandbox module.
Direct Swift package reuse from `containerization` is allowed only after a
focused evaluation shows that it preserves the runtime contract, works on the
intended macOS support window, accounts for upstream source-stability risk, and
reduces custom code without weakening diagnostics, audit, or security
reviewability.

The current evaluation record is:

- `Docs/Design/2026-05-02-apple-containerization-evaluation.md`

The current cross-runtime support inventory is:

- `Docs/Sandbox/sandbox-runtime-capability-inventory.md`

The current cross-runtime security policy matrix is:

- `Docs/Sandbox/sandbox-security-policy-matrix.md`

## Non-Negotiable Boundaries

### Trusted Control Plane vs Untrusted Compute

The Python sandbox service is trusted orchestration. Untrusted execution must
not run in the trusted service process.

Control-plane responsibilities:

- policy admission
- session identity and persistence
- run lifecycle and queueing
- artifact bookkeeping
- public API behavior
- ACP integration

Untrusted/runtime responsibilities:

- executing commands
- VM lifecycle and live runtime health
- template and boot validation
- guest-tool transport

### Canonical Runtime Layers

VM-backed runtimes should preserve three distinct layers:

- Python sandbox service
- host-side runtime helper
- guest-side agent or executor

Host-local runtimes like `seatbelt` may not need a guest layer, but they still
must keep execution logic outside policy description and diagnostics logic.

Host-side helpers should be decomposed by trust boundary and operational
responsibility rather than growing into one ambient daemon. Apple `container`
is useful prior art here: image/content management, network management, and
per-workload runtime management can have distinct lifecycle and permission
profiles. This repo should adopt that shape only when a split materially
improves safety, diagnostics, or operator control.

## Layered Readiness Model

Readiness must be reported in layers, not collapsed into one boolean.

Required layers:

- `host_ready`
  - required host OS, architecture, framework/tooling, helper reachability
- `template_ready`
  - image/template exists and passes runtime-owned validation
- `runtime_ready`
  - VM/process can actually be started with the requested guarantees
- `agent_ready`
  - guest tool endpoint or host-local executor is ready to accept work

Plans, diagnostics, and runtime discovery should use the same layered truth.
There should not be separate "preflight truth", "diagnostics truth", and
"operator docs truth".

## Source-Of-Truth Rules

### Python Owns

- trust-level admission
- session identity
- persisted run and session metadata
- artifact metadata and guardrails
- queueing and idempotency

### Runtime Helper Owns

- live runtime availability
- host readiness checks
- template compatibility validation
- VM/process status
- guest transport health
- reconciliation facts about running runtime state

### Diagnostics Must Reuse Runtime Truth

Admin diagnostics should surface the same truth the runtime uses internally.
Environment-only scaffolding is acceptable only as a temporary bridge and should
be removed once runtime truth exists.

## Fail-Closed Runtime Contract

- No silent fallback from a stronger runtime to a weaker runtime.
- Unsupported guarantees must return explicit unsupported or unavailable reasons.
- `untrusted` workloads require a VM-grade boundary.
- Host-local runtimes like `seatbelt` must document weaker guarantees explicitly.

Convenience paths may exist, but they must never redefine the guarantee model.

## Protocol Doctrine

### Separate Host And Guest Protocols

Use two protocols when a VM guest is involved:

- host helper protocol
- guest agent protocol

The helper bridges them. Python should not speak directly to the guest protocol.
Guest protocol evolution should not require rewriting the Python host contract.

### Version Everything

Both host and guest protocols must be versioned from day one.

### Readiness Is More Than "VM Started"

For VM runtimes, runtime health requires:

- boot success
- required mounts/devices configured
- transport connected
- agent readiness confirmed

Session reuse must check live health before reusing a stored runtime identifier.

## Image And Template Doctrine

### Canonical Path vs Compatibility Path

Every VM runtime should distinguish:

- canonical repo-owned artifact path
- weaker compatibility path for operator-supplied artifacts

Canonical artifacts should have stronger validation and deterministic metadata.
Compatibility paths may exist, but their weaker validation strength should be
reported explicitly.

The long-term canonical Linux artifact path should be OCI-aware even if the
current implementation remains a repo-owned bundle format. Image-store metadata
should be structured so templates can later map to OCI manifests, layers,
digests, registry provenance, or a dedicated `vz_oci_linux` runtime without
rewriting run/session/audit semantics.

The current bundle path remains valid for near-term `vz_linux` work. Do not
switch to OCI or `containerization` as a side effect of unrelated helper,
diagnostics, or policy hardening.

### Reproducible Builder Outputs

Repo-owned builders should emit inspectable outputs and provenance metadata.

Minimum expectations:

- explicit artifact layout
- deterministic manifest/metadata emission
- pinned defaults for suite/profile/kernel/toolchain choices
- clear distinction between source inputs and generated outputs

### Provenance And Audit

Builder outputs should emit deterministic metadata such as:

- suite
- architecture
- profile
- kernel package or image family
- selected package set or source artifact paths
- OCI source image, manifest digest, or registry provenance when applicable

Cryptographic signing or attestation can layer on top later, but provenance
metadata is mandatory even before signing exists.

## Debuggability Doctrine

Canonical VM images should include the minimum debug affordances needed to
understand readiness failures early:

- serial console access
- deterministic service startup
- required module loading for transport support
- clear boot artifact layout

Debuggability is part of the canonical path, not a separate afterthought.

## Policy Doctrine

Inspired by Nono, the subsystem should keep runtime enforcement mechanics and
policy description distinct.

- runtime code should enforce only the capabilities and guarantees it can prove
- policy/profile decisions should live in one auditable source of truth
- security-relevant config should be explicit and reviewable
- ambient environment variables should not become the long-term policy model

### Network Policy Doctrine

VM networking must be explicit policy, not an implementation side effect.
Apple `container` shows that vmnet can provide a first-party macOS networking
surface for Linux VMs, but this subsystem's strict baseline remains no attached
guest network for `vz_linux` unless a policy implementation can prove and
diagnose the requested guarantee.

`deny_all` should continue to mean no guest network device unless a runtime has
a separately reviewed enforcement layer. Future allowlist work must expose
host/helper truth in diagnostics and must not silently downgrade to best-effort
networking.

## Lifecycle And Recovery Doctrine

Session recovery must be designed explicitly.

Required principles:

- persisted metadata is necessary but not sufficient for reuse
- live runtime health must be checked before reuse
- startup and admin reconciliation should compare persisted control metadata
  against live runtime facts
- destroy and cleanup paths must tolerate already-gone runtime state without
  leaving stale control metadata behind

## Observability And Audit Doctrine

Runtime actions should emit structured lifecycle facts, including:

- create/start
- ready
- exec start and finish
- timeout/cancel
- stop/destroy
- reconciliation outcomes

Bounded stdout/stderr, timeouts, and resource caps belong at the sandbox
boundary rather than being left to ad hoc runtime behavior.

## Guidance For Specific Runtime Families

### `vz_linux`

- canonical VM path for macOS-hosted Linux isolation
- helper-owned runtime truth
- guest agent over vsock
- canonical repo-owned bundle path plus weaker compatibility path

### `vz_macos`

- should follow the same helper-owned readiness and provenance rules
- must not claim stronger guarantees than it can actually prove

### `seatbelt`

- stays a host-local runtime with weaker boundary guarantees than a VM
- must continue to expose conservative readiness and guarantee claims
- should adopt the same audit, policy-truth, and diagnostics rules where they
  apply

## Planning Requirement

Future runtime-related plans should explicitly answer:

- who owns readiness truth
- what the readiness layers are
- which path is canonical versus compatibility-only
- what provenance and audit metadata is emitted
- how session reuse or recovery is validated
- which guarantees are intentionally weaker than VM-grade isolation

If a plan does not answer those questions, it is incomplete.

The full-module roadmap that sequences these requirements across current and
future runtime families is:

- `Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md`
