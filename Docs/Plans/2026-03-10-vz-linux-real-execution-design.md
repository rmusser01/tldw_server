# vz_linux Real Execution Design

Date: 2026-03-10
Status: Approved for planning
Scope: `tldw_Server_API/app/core/Sandbox/` and related ACP/runtime operator surfaces

## 1. Summary

This design turns `vz_linux` into the first real VM-backed macOS sandbox runtime for Apple silicon hosts.

The milestone adds:

- real Linux guest boot via Apple's `Virtualization.framework`
- a native macOS helper that owns VM lifecycle
- a Linux guest agent reachable over vsock
- `virtiofs` workspace sharing between host and guest
- support for both ephemeral one-shot runs and ACP/session-style command reuse inside the same VM

The security contract stays explicit:

- `untrusted` may use `vz_linux`
- command transport is not guest networking or SSH
- allowlist networking is still out of scope
- operator-supplied template disks are acceptable for the first real-exec milestone

## 2. Current State

The merged macOS sandbox work already provides:

- `RuntimeType.vz_linux`
- preflight/discovery scaffolding
- admin diagnostics for macOS runtime readiness
- a helper/image-store contract stub
- fake execution gates for the VZ runtimes

What is still missing:

- real Linux guest boot
- real helper lifecycle control
- real guest command execution
- ACP/session reuse inside a live `vz_linux` guest

This design fills that gap before APFS clone-backed provisioning or macOS guest execution.

## 3. User-Confirmed Decisions

1. The next milestone should optimize for real `vz_linux` execution, not operator tooling or `vz_macos`.
2. The first real `vz_linux` milestone should include ACP/session-oriented command reuse inside a running VM.
3. Guest command transport should use a host-local control channel such as vsock, not SSH.
4. The initial bootstrap story should use an operator-supplied Linux template disk with the guest agent installed.
5. The workspace should be exposed to the guest with `virtiofs`.

## 4. Goals and Non-Goals

### Goals

1. Make `vz_linux` a real execution backend on Apple silicon macOS hosts.
2. Support both:
   - ephemeral one-command runs
   - session-oriented repeated command execution for ACP flows
3. Keep command transport independent from guest networking.
4. Preserve fail-closed sandbox policy and existing error taxonomy.
5. Reuse the existing helper/image-store seams rather than inventing a separate control plane.

### Non-Goals

1. Shipping APFS clone-backed provisioning in the same milestone.
2. Implementing allowlist networking.
3. Implementing `vz_macos` real execution.
4. Using SSH as the control channel.
5. Treating a Python-only `Virtualization.framework` bridge as the production control plane.

## 5. Selected Architecture

### 5.1 Runtime shape

`vz_linux` becomes the first real VM-backed macOS runtime with two execution modes sharing one control plane:

- ephemeral run mode
  - boot a fresh Linux guest from a prepared template disk
  - attach the run workspace over `virtiofs`
  - connect to the guest agent over vsock
  - execute one command, stream logs, collect artifacts, tear down
- ACP/session mode
  - create one running guest per sandbox session
  - keep the workspace mounted for the life of the session
  - reuse the same guest agent for multiple commands
  - destroy the VM when the session expires or is terminated

### 5.2 Responsibility split

- Python sandbox service
  - policy admission
  - queueing
  - run/session bookkeeping
  - stream publication
  - ACP integration
- native macOS helper
  - `Virtualization.framework` lifecycle
  - device configuration
  - vsock setup
  - `virtiofs` workspace mount
  - VM teardown
- Linux guest agent
  - in-guest command execution
  - stdout/stderr streaming
  - exit status and structured error reporting

## 6. Control Plane Design

### 6.1 Native helper

The native helper is the authoritative VM control plane for `vz_linux`.

Responsibilities:

- validate host support and template disk compatibility
- create ephemeral run VMs
- create and track session VMs
- mount `virtiofs` shares
- expose a local control API for:
  - create ephemeral VM
  - create session VM
  - exec inside session VM
  - query VM status
  - terminate VM

This helper should be the component that directly owns `Virtualization.framework`, not Python.

This repo currently only contains the Python client and contract scaffolding for that helper.
Unless a native helper source tree is added during implementation, the repo-side milestone should
be explicit that it integrates with an operator-installed helper binary or service and fails closed
when that helper is unavailable.

### 6.2 Guest agent

The Linux guest image must boot with a guest agent installed and enabled.

The agent:

- listens on vsock only
- accepts structured exec requests:
  - argv
  - cwd
  - env
  - timeout
  - capture settings
- executes without shell interpolation
- streams stdout/stderr back over the control channel
- returns exit codes and structured failure reasons

### 6.3 Data flow

1. Sandbox service admits a `vz_linux` request.
2. Python validates authoritative `vz_linux` preflight.
3. Python asks the helper to boot or reuse a VM.
4. Helper mounts the host workspace with `virtiofs`.
5. Helper waits for guest-agent readiness over vsock.
6. Python sends an exec request through the helper to the guest agent.
7. Guest agent runs the command and streams results back.
8. Python publishes logs/events and finalizes run/session state.
9. Helper destroys the VM or keeps it alive for session reuse.

## 7. Workspace and Session Model

### 7.1 Workspace access

The first real milestone uses `virtiofs` for workspace access.

Benefits:

- avoids copy-in/copy-out overhead
- fits repeated ACP/session commands
- keeps artifact handling aligned with the existing sandbox workspace model

The design does not require a separate guest-side artifact export channel in phase 1. Artifacts remain files under the mounted workspace and are collected through the existing sandbox artifact rules.

### 7.2 Session reuse

For ACP and session-oriented runs:

- one sandbox session maps to one live `vz_linux` guest VM
- session metadata stores:
  - helper VM id
  - template id
  - workspace mount metadata
  - guest-agent readiness state
- later commands reuse the VM only while that health state is still authoritative
- stale or unhealthy VMs fail closed and must be recreated explicitly by the session path
- destroying a sandbox session must terminate the stored VM even when no run is active

For ACP specifically, the first real `vz_linux` milestone should preserve the existing ACP model:

- ACP still uses one long-lived interactive sandbox run inside the session VM
- ACP prompt reuse stays inside that existing stream-oriented run path
- no ACP protocol redesign is required just to make the underlying `vz_linux` VM real

## 8. Template and Bootstrap Story

The first real `vz_linux` milestone uses an operator-supplied template disk.

The template must:

- boot on Apple `Virtualization.framework`
- include the Linux guest agent
- include the guest-agent service enablement required for automatic startup
- be compatible with the helper's expected Linux guest contract

The existing `SandboxImageStore` remains the registration and manifest seam, even if real APFS clone execution is deferred to a later milestone.

That means phase 1 should support:

- template registration
- deterministic template lookup
- helper-side template validation
- real execution from an operator-managed template

It should not promise a full managed image factory yet.

## 9. Preflight and Policy

Real-ready `vz_linux` preflight should stop being env-only scaffolding.

It must prove:

- macOS host
- Apple silicon host
- native helper reachable
- `Virtualization.framework` readiness validated by the helper
- configured template disk exists and is compatible
- guest agent expected for the configured template family
- requested network policy is supported

Policy rules for the first real milestone:

- `deny_all` is supported
- `allowlist` remains rejected
- `untrusted` is allowed only when the VM path is truly ready
- no silent fallback to another runtime

## 10. Error Semantics

Map helper and guest-agent failures into the existing sandbox taxonomy.

### `runtime_unavailable`

- `macos_required`
- `apple_silicon_required`
- `macos_virtualization_helper_unavailable`
- `virtualization_framework_unavailable`
- `vz_linux_template_missing`
- `vz_linux_template_invalid`
- `vz_linux_guest_agent_missing`

### `policy_unsupported`

- `strict_allowlist_not_supported`
- unsupported trust/runtime combinations

### `runtime_execution_failed`

- boot timeout
- guest-agent readiness timeout
- vsock channel failure
- guest exec protocol failure
- unexpected VM termination during run or session reuse

Structured failures should include the runtime, helper state, template identity when available, and whether the failing path was ephemeral or session-based.

## 11. Testing Strategy

### Unit tests

1. `vz_linux` preflight reason mapping
2. helper request/response translation
3. sandbox service routing into the real `vz_linux` runner path
4. ACP/session metadata and session reuse rules

### Fake integration tests

1. ephemeral boot/exec/teardown lifecycle using a fake helper client
2. session create/exec/exec/terminate lifecycle
3. stdout/stderr/artifact propagation through the guest-agent protocol
4. fail-closed behavior when a reused session VM is unhealthy

### macOS-gated integration tests

1. Apple silicon only
2. real helper handshake
3. real template validation
4. simple real command execution inside the guest
5. second ACP-style command inside the same running session VM

## 12. Rollout Order

1. Make the native helper protocol real for `vz_linux`.
2. Add guest-agent protocol and ephemeral single-command execution.
3. Add ACP/session VM reuse on the same helper/agent path.
4. Expand diagnostics and operator notes for the real execution path.
5. Add APFS clone-backed provisioning as the next milestone after execution stability.

## 13. Rationale

This order validates the most important boundary first:

- VM boot is real
- command execution is real
- ACP session reuse is real
- `untrusted` can use a real VM-backed path on macOS hosts

Only after that is proven should the project take on faster provisioning through APFS clone-backed template lifecycle work.
