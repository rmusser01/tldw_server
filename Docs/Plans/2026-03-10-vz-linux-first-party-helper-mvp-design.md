# VZ Linux First-Party Helper MVP Design

**Date:** 2026-03-10

## Goal

Define the most pragmatic long-term-stable path to real macOS VM-backed end-to-end
testing by migrating `tldw-agent` into this repo and using it as the first-party
guest agent for a new in-repo `vz_linux` helper daemon and reference Linux image.

## Current State

The repo now has the Python-side boundaries mostly in the right place:

- Python owns sandbox policy admission, run/session persistence, ACP integration,
  artifacts, and public API behavior.
- `vz_linux` already expects a helper-backed Unix-socket control plane.
- The helper protocol is frozen in `tools/macos-vz-helper/PROTOCOL.md`.
- Real-host E2E tests exist and now tell the truth about why they skip.

What is still missing is the part that actually turns those tests green:

- no native macOS helper daemon in this repo
- no in-guest agent/data plane under the same versioned ownership boundary
- no reproducible reference Linux image build path

There is also a relevant adjacent codebase at `../tldw-agent`. That repo was
already intended to become the long-term first-party agent for this purpose.

## Recommendation

The long-term-stable path is:

1. Migrate the full `tldw-agent` source tree into this repo under `tools/tldw-agent/`.
2. Preserve its current host/browser and ACP-facing modes first.
3. Add a new `vz_linux` guest mode to the migrated agent instead of creating a
   second guest-agent codebase.
4. Build a narrow Swift helper daemon in `tools/macos-vz-helper/` for
   `Virtualization.framework` lifecycle and host-side control.
5. Add one reproducible reference Linux image build path that installs the
   guest-mode `tldw-agent`.

This keeps helper, guest agent, protocol, reference image, and E2E tests under
one repo and one versioned contract.

## Ownership Boundary

### Python sandbox service owns

- runtime selection and policy admission
- sandbox run/session persistence
- ACP integration
- artifact capture and export
- public API behavior

### Swift helper daemon owns

- `Virtualization.framework` lifecycle
- host validation for `vz_linux`
- template validation and VM boot readiness
- `virtiofs` and vsock setup
- VM create, status, list, and terminate operations
- bridging host helper protocol requests to guest-agent protocol requests

### Guest-mode `tldw-agent` owns

- in-guest command and tool execution
- guest workspace awareness for the mounted `virtiofs` path
- stdout/stderr/exit status and later richer guest tools
- guest readiness reporting after boot

The helper is authoritative for VM lifecycle. The guest agent is authoritative
for in-guest execution. Python remains authoritative for sandbox semantics.

## Repo Architecture

The target in-repo layout should be:

- `tools/macos-vz-helper/`
  - Swift daemon for `vz_linux` only in the first slice
  - Unix-socket server implementing the frozen helper protocol
- `tools/tldw-agent/`
  - migrated Go codebase from `../tldw-agent`
  - existing host/browser-facing modes preserved
  - new guest mode added for Linux VM use
- `tools/vz-linux-image/`
  - reproducible reference Linux image assets/scripts
  - installs guest-mode `tldw-agent`
  - documents the image used by local E2E

The critical long-term rule is: do not create a separate guest-agent codebase.
The migrated `tldw-agent` becomes the in-guest execution/data-plane binary.

## `tldw-agent` Mode Split

After migration, `tldw-agent` should have explicit modes:

- host/native-messaging mode
  - preserves the current browser-extension/native-host use case
- ACP/runner mode
  - preserves the current downstream ACP/session behavior
- `vz_linux` guest mode
  - new in-guest mode used inside Linux VMs

Guest mode should not expose the full native-messaging surface. It should use
shared agent core pieces where appropriate, but it needs a VM-specific framing
and execution contract.

For the first slice, guest mode should support:

- readiness reporting
- workspace-root awareness
- exec requests with `argv`, `cwd`, `env`, and timeout
- stdout/stderr/exit status responses
- cancellation or terminate-on-timeout behavior if straightforward

## Control Plane And Data Flow

The stable data flow should be:

1. Python admits a `vz_linux` run or session command.
2. Python talks to the helper over the existing Unix-socket JSON helper protocol.
3. Helper validates host/template and boots or reuses a VM.
4. Helper establishes `virtiofs` and vsock connectivity for that VM.
5. Helper waits for explicit guest-agent readiness.
6. Helper bridges exec requests to guest-mode `tldw-agent` over vsock.
7. Guest-mode `tldw-agent` executes the request and returns output/status.
8. Helper translates the guest reply back into the host helper protocol.
9. Python updates run/session state and artifact bookkeeping.

Python must never talk directly to the guest agent. The guest agent must never
become the source of truth for VM lifecycle.

## Two-Protocol Model

Long-term stability requires two separate versioned protocols:

### Host helper protocol

- Python <-> Swift helper
- Unix socket transport
- frozen protocol version `1`
- public host-side contract already consumed by the Python client

### Guest agent protocol

- Swift helper <-> guest-mode `tldw-agent`
- vsock transport
- separate versioned protocol starting at `1`
- intentionally narrower and specific to guest execution/tooling

The helper bridges between those protocols. Python should remain insulated from
guest-agent protocol changes.

## Layered Readiness Model

Readiness should be explicit and layered:

### Host-ready

- helper daemon reachable
- `Virtualization.framework` usable
- required Apple silicon/macOS features available

### Template-ready

- reference image or operator image exists
- helper validates compatibility
- guest agent is expected in that image

### VM-ready

- VM booted
- `virtiofs` mount established
- vsock channel established

### Agent-ready

- guest-mode `tldw-agent` completed startup
- guest protocol version compatible
- workspace root known inside the guest

Session reuse is valid only when both VM health and agent readiness are true.

## Migration Strategy

To minimize risk, migrate `tldw-agent` intact first:

1. copy the full `../tldw-agent` source tree into `tools/tldw-agent/`
2. preserve its Go module and internal package layout initially
3. record source provenance and upstream commit in repo docs
4. prove existing host/browser and ACP behaviors still build and test
5. only then add guest mode and helper integration

This avoids a rewrite-during-migration failure mode.

## MVP Scope

The first helper MVP should stay narrow:

- `vz_linux` only
- Apple silicon macOS hosts only
- one Unix-socket helper daemon
- one guest-mode `tldw-agent` over vsock
- one `virtiofs` workspace mount
- one reproducible reference Linux image
- enough functionality to make the existing real-host `vz_linux` E2E tests pass

Out of scope:

- `vz_macos`
- `seatbelt`
- APFS clone provisioning
- multiple guest distros
- rich image registry UX
- CI-hosted full E2E in the first slice

## Testing Strategy

Testing should be staged:

### 1. Migration-preservation tests

- verify migrated `tools/tldw-agent/` still builds
- preserve current host/native-messaging behavior
- preserve current ACP-facing behavior

### 2. Helper protocol tests

- `ping`
- `validate_host`
- `validate_template`
- `create_vm`
- `get_vm_status`
- `list_vms`
- `terminate_vm`

### 3. Guest agent protocol tests

- readiness handshake
- exec request parsing
- timeout/cancel behavior
- stdout/stderr/exit status reporting

### 4. Integrated local E2E

- Apple silicon host only
- real helper daemon
- real reference image
- existing pytest E2E turns from skip into pass for:
  - ephemeral execution
  - session reuse with a second command
  - session destroy cleanup

## Rollout Order

1. migrate `tldw-agent` source unchanged
2. preserve existing builds/tests and add provenance docs
3. add Swift helper daemon skeleton matching the frozen host protocol
4. add guest-mode `tldw-agent` and the guest vsock protocol
5. add the reproducible reference image build/install path
6. connect helper boot to guest readiness
7. make the local real-host E2E tests pass
8. only then expand diagnostics, tooling, and CI-hosted E2E

## Review Corrections Applied

Reviewing the approved design surfaced three important corrections:

1. The helper-to-guest link needs its own versioned protocol.
   Reusing the Python-helper protocol inside the guest would couple the wrong
   layers and make future evolution brittle.

2. VM reuse must require explicit guest readiness, not just VM liveness.
   A running VM without a ready guest agent is not a reusable execution target.

3. `../tldw-agent` should be treated as the intended first-party agent being
   migrated into this repo, not as a runtime dependency or mere reference repo.

## Why This Is The Pragmatic Long-Term Path

This approach removes the current unstable split permanently:

- one repo owns helper, guest agent, image recipe, protocol, and tests
- one helper owns VM lifecycle truth
- one migrated `tldw-agent` owns in-guest execution truth
- Python stays focused on sandbox semantics instead of native VM control

That is the shortest path that both enables real local end-to-end testing and
reduces future architectural churn instead of increasing it.
