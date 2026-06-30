# VZ Linux Helper Stability Design

**Date:** 2026-03-10

## Goal

Define the most pragmatic long-term stability path for macOS VM sandboxing by making `vz_linux` depend on a first-party, in-repo macOS helper daemon instead of env-driven scaffolding and a Python stub client.

## Current State

The repo already has the correct high-level split for VM-backed sandboxing:

- Python owns sandbox admission, trust policy, run/session persistence, artifacts, and ACP integration.
- `vz_linux` already assumes a helper-backed control plane.
- Session metadata already persists `runtime`, `vm_id`, `template_id`, `workspace_mount`, and `agent_ready`.

The unstable seams are the ones that are still fake or split across subsystems:

- `MacOSVirtualizationHelperClient` is still a stub outside `TEST_MODE`.
- `macos_diagnostics.py` still infers helper/template readiness from env flags.
- Session reuse currently trusts persisted `vm_id` metadata without authoritative VM health checks.
- `SandboxImageStore` is still manifest-oriented scaffolding and is not the source of truth for runnable templates.

## Recommendation

The long-term stable path is:

1. Build a narrow, first-party helper daemon in this repo for `vz_linux` only.
2. Make that helper the source of truth for host readiness, template registration, template validation, VM lifecycle, and VM health.
3. Keep Python authoritative for sandbox policy, session identity, persistence, artifacts, and ACP-facing behavior.
4. Move diagnostics and session reuse to consume real helper truth instead of env scaffolding and optimistic stored metadata.

This is intentionally not a broad "macOS control plane" yet. Stability comes from making one runtime boring before expanding scope.

## Ownership Boundary

### Python sandbox service owns

- runtime selection and policy admission
- trust-level enforcement
- run/session records
- artifact capture and export
- ACP integration
- persisted linkage from sandbox session to helper VM identity

### Native helper daemon owns

- macOS host readiness for `vz_linux`
- template registration and compatibility validation
- `Virtualization.framework` lifecycle
- VM create, exec, status, list, and terminate operations
- real runtime facts used by diagnostics and session reuse

### Guest agent owns

- in-guest command execution over vsock
- structured stdout/stderr and exit-code responses
- no shell injection and no guest-network control dependency

## First-Party Helper Shape

The helper should live in a dedicated native subproject in the main repo, for example:

- `tools/macos-vz-helper/`

It should run as a long-lived local daemon over a Unix domain socket. That is the most stable shape for:

- session VM reuse
- status and health checks
- structured diagnostics
- startup reconciliation
- protocol evolution without shelling out to a per-command CLI

## Protocol

The helper protocol should be explicit and versioned from the first real implementation.

Required operations:

- `ping`
- `validate_host`
- `register_template`
- `validate_template`
- `create_vm`
- `exec_guest`
- `get_vm_status`
- `list_vms`
- `terminate_vm`

Required protocol properties:

- explicit protocol version in every response
- helper build/version metadata
- stable machine-readable error codes
- request/response schemas narrow enough to test directly from Python

The repo should not keep a "best effort" contract here. Drift between Python and the helper is one of the main long-term instability risks.

## Template Source Of Truth

Template truth should move to the helper because the helper is what actually boots VMs.

Phase 1 should stay simple:

- register a direct base-image path plus template metadata
- validate that the path exists and is compatible with helper expectations
- return a helper-owned `template_id`

This is intentionally narrower than the eventual APFS clone story. APFS clone-backed provisioning should be layered on after helper truth, diagnostics, and session reuse are stable.

`SandboxImageStore` can remain a Python-side metadata seam for now, but it should stop pretending to be authoritative for runnable templates.

## Session State Model

Python remains authoritative for sandbox session identity and persistence.

The helper should not own product-level session identity. It should only own runtime VM state.

The persisted Python control row remains valuable, but reuse must stop being optimistic:

1. Python loads persisted `vm_id` for a sandbox session.
2. Python asks helper `get_vm_status(vm_id)`.
3. If helper confirms the VM is healthy, reuse it.
4. If helper reports missing or unhealthy state, Python fails closed or recreates explicitly.
5. Session destroy calls helper termination and then deletes persisted control state.

This keeps one durable source of truth for sandbox semantics while still making helper runtime facts authoritative.

## Diagnostics

Long-term stability requires diagnostics to stop deriving readiness from env flags.

The admin diagnostics surface should become a thin view over helper truth:

- `host`
  - driven by helper `validate_host`
- `helper`
  - driven by helper `ping`
  - includes protocol version and helper version
- `templates`
  - driven by helper template registration/validation state
- `runtimes.vz_linux`
  - driven by helper-backed preflight

`/api/v1/sandbox/runtimes` should keep a summarized shape, but it should now be summarizing real helper-backed readiness instead of placeholder metadata.

## Reconciliation And Restart Safety

One missing piece in the current design is restart-safe reconciliation.

For long-term stability, the system needs an explicit recovery loop:

- helper can list live VMs
- Python can load persisted `sandbox_vz_sessions`
- startup or admin reconciliation can compare persisted session control with live helper runtime state
- stale persisted rows can be repaired or surfaced clearly

Without this, a helper restart or service restart turns session reuse into guesswork.

This does not require the helper to become the persistence layer. It only requires the helper to expose authoritative runtime state.

## Review Corrections Applied

The reviewed design includes four corrections that materially reduce future churn:

1. Protocol versioning is required from day one.
   Without it, the Python client and native helper will drift immediately.

2. Template registration starts with direct base-image paths.
   This avoids coupling first-party helper bring-up to APFS clone/image-store work.

3. Startup reconciliation is a first-class requirement.
   Session reuse cannot rely only on stored `vm_id` values and destroy-time cleanup.

4. Helper state stays runtime-scoped, not product-scoped.
   The helper should not become a second sandbox/session database.

## Phased Roadmap

### Phase 1: Real helper protocol and host truth

- implement in-repo helper daemon for `vz_linux`
- replace Python helper stub with a real socket client
- add helper-backed host validation and version reporting

### Phase 2: Template truth and diagnostics

- add helper-owned template registration/validation
- wire admin diagnostics and `vz_linux` preflight to helper truth

### Phase 3: Stable session reuse

- add `get_vm_status` and `list_vms`
- make session reuse health-based
- add startup/admin reconciliation

### Phase 4: Provisioning improvements

- layer APFS clone-backed provisioning behind the stable helper/template contract

### Phase 5: Broaden scope only after `vz_linux` is boring

- richer operator tooling
- `vz_macos`
- broader helper capabilities if still justified

## Why This Is The Pragmatic Path

This design matches the architecture the repo already wants:

- Python already assumes a helper boundary.
- Python already owns sandbox session persistence.
- `vz_linux` is the runtime that matters for the `untrusted => VM only` policy.

So the most pragmatic long-term stability move is not more scaffolding. It is making the helper real, making helper truth authoritative, and tightening the runtime/persistence boundary before adding more features.
