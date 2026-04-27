# VZ Linux VSock Transport Design

**Date:** 2026-03-11

## Goal

Define the next pragmatic slice required to move `vz_linux` from real VM boot to
real guest readiness and guest command execution on Apple silicon macOS hosts.
The target is one durable helper-owned vsock transport connection per VM, with
the migrated `tldw-agent` acting as the only in-guest transport and execution
endpoint.

## Current State

The repo now has:

- a real Swift helper daemon that serves the host-helper protocol over a Unix socket
- a real Python helper client that talks to that daemon
- canonical bundle and raw-disk template resolution
- a real `Virtualization.framework` boot-configuration builder plus boot-driver seam
- a migrated `tldw-agent` guest mode that still serves only a generic stdin/stdout
  JSON stream
- a `VSockBridge` protocol bridge in Swift with no real transport implementation

What is still missing is the real guest transport path:

- no host-side vsock listener owned by the helper
- no guest-mode `tldw-agent` vsock server/client mode
- no durable per-VM guest transport session
- no reconnect semantics when the guest-agent connection drops
- no real guest `ready` handshake or real `exec` over a VM transport

## Recommendation

Use one durable host-owned vsock transport connection per VM.

The helper should open a host-side vsock listener before boot, the guest-mode
`tldw-agent` should connect out to that listener on boot, and the helper should
bind that one long-lived connection to the helper `vm_id`. Control traffic and
sequential `exec` traffic should share the same newline-delimited JSON stream.

This is the most stable next step because it:

- keeps Python unchanged at the transport layer
- preserves the helper as the source of truth for VM lifecycle and guest transport
- keeps `tldw-agent` as the only in-guest execution stack
- supports session reuse and reconnects without inventing a second guest shim

## Transport Architecture

There are three roles:

- Python sandbox service
  - unchanged at this layer
  - still talks only to the Swift helper over the host-helper Unix socket protocol
- Swift helper
  - boots the VM
  - opens a host-side vsock listener
  - accepts one guest-agent connection per VM
  - binds that connection to the helper `vm_id`
  - owns reconnect and health state
- guest-mode `tldw-agent`
  - starts on boot via the baked-in `systemd` service
  - dials the helper’s host-side vsock listener
  - sends a handshake identifying protocol version and guest role
  - stays connected for readiness, health, and request handling

The transport should be long-lived and newline-delimited JSON, with message types
on one stream:

- control:
  - `handshake`
  - `handshake_ack`
  - `ready`
  - `heartbeat`
  - `reconnect`
  - `error`
- request/response:
  - `exec_request`
  - `exec_response`

Python should continue to see only helper protocol results. The guest stream stays
hidden behind the helper.

## VM Identity And Reconnect Model

To make reconnects stable, the helper boot path should inject two runtime values
into the guest:

- `vm_id`
- a per-boot connection token

The first guest message on the vsock stream should carry:

- guest protocol version
- `vm_id`
- connection token
- guest mode/version
- workspace root as seen in the guest

Helper behavior should be:

1. create VM record in `booting`
2. start vsock listener context for that `vm_id`
3. boot VM
4. accept guest connection
5. verify `vm_id` and token match the boot context
6. mark transport connected
7. wait for `ready` confirmation
8. mark VM healthy

Reconnect rule:

- if the socket drops but the VM is still alive, the guest agent reconnects with
  the same `vm_id` and token
- helper rebinds the new socket to the same VM transport state
- any in-flight request fails
- later requests may continue once the transport is healthy again

This keeps reconnect logic deterministic instead of guessing which VM a guest
belongs to.

## Request And Control Flow

The first stable transport slice should allow:

- one durable guest connection per VM
- control traffic at any time on that connection
- only one in-flight `exec_request` at a time per VM

Recommended flow:

1. guest connects
2. guest sends `handshake`
3. helper validates identity/token and sends `handshake_ack`
4. guest sends `ready`
5. helper marks VM healthy
6. when Python asks for exec, helper sends `exec_request`
7. guest runs argv directly without shell interpolation
8. guest sends `exec_response`
9. both sides continue heartbeats while idle

Important behavior:

- control messages can happen while no exec is active
- if heartbeats fail, helper marks transport unhealthy
- if an exec is in flight when disconnect happens, helper returns a transport
  failure for that exec
- concurrent exec multiplexing is explicitly out of scope for this slice

This keeps the first real transport boring and debuggable while leaving room for
future richer tool calls over the same stream.

## Implementation Shape

### `tools/tldw-agent/`

The migrated guest agent should gain a real vsock mode:

- keep the current request/response logic behind a transport interface
- preserve stdin/stdout serving for tests and existing narrow uses
- add a vsock client mode for guest use
- add handshake, ready, heartbeat, and reconnect handling around the existing exec path

The key rule is: do not create a second in-guest execution stack. `tldw-agent`
remains the only guest executor.

### `tools/macos-vz-helper/`

The helper should gain:

- a host-side vsock listener/connection manager
- one transport session per `vm_id`
- a real `GuestTransporting` implementation behind `VSockBridge`
- boot-driver support for creating listener state before VM start
- deterministic guest boot metadata injection for `vm_id` and connection token

The helper remains the bridge between the host-helper protocol and the guest
transport protocol.

### Python

Python should not need a transport redesign. It should only see:

- `create_vm` succeeding once the guest is ready
- `exec_guest` succeeding once the durable transport exists
- possibly a few new helper error codes mapped cleanly into existing sandbox error
  handling

## Testing Strategy

Testing should be layered around the transport boundary.

### Guest agent tests

- handshake encode/decode
- ready/heartbeat behavior
- reconnect acceptance
- existing exec behavior still works over the new transport abstraction

### Swift helper tests

- vsock listener accepts and binds a guest connection to `vm_id`
- wrong token or wrong `vm_id` is rejected
- reconnect replaces the prior socket for the same VM
- `VSockBridge` can drive `ready` and `exec` over a fake transport session

### Host-gated integration

- canonical bundle boot smoke moves from “past `boot_not_implemented`” to
  “guest ready connected”
- real host E2E moves from skip or transport ceiling to:
  - ephemeral `/bin/echo`
  - session reuse with a second command in the same VM

## Success Criteria

This slice is complete when:

- a booted canonical-bundle VM reaches real guest-agent readiness over vsock
- helper `create_vm` returns success for that VM on a prepared host
- helper `exec_guest` succeeds for at least one real command
- session reuse can execute a second command over the same durable connection
- reconnect behavior is covered in unit/integration tests, even if full host-side
  reconnect simulation remains limited

That is the first point where `vz_linux` on macOS becomes full end-to-end
execution instead of boot-only scaffolding.
