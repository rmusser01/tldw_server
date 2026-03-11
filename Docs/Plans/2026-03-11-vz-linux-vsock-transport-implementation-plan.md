# VZ Linux VSock Transport Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the real helper-to-guest vsock transport so `vz_linux` can reach guest-agent readiness and execute real commands inside a booted Linux VM.

**Architecture:** Keep Python unchanged at the transport boundary. Extend the migrated `tldw-agent` with a real guest vsock mode, add a host-side vsock session manager in the Swift helper, and wire `Virtualization.framework` boot so the helper injects `vm_id` plus a connection token and waits for a real guest handshake before treating the VM as healthy.

**Tech Stack:** Go, Swift Package Manager, `Virtualization.framework`, `Network` or native socket APIs for vsock transport, pytest, Swift Testing, Go test

---

### Task 1: Extend The Guest Protocol For Handshake, Heartbeats, And Reconnects

**Files:**
- Modify: `tools/tldw-agent/docs/vz-linux-guest-protocol.md`
- Modify: `tools/tldw-agent/internal/guest/types.go`
- Modify: `tools/tldw-agent/internal/guest/transport.go`
- Create: `tools/tldw-agent/internal/guest/handshake_test.go`
- Modify: `tools/tldw-agent/internal/guest/transport_test.go`

**Step 1: Write the failing Go tests**

Add tests like:

```go
func TestGuestTransportParsesHandshakeAndReadyMessages(t *testing.T) {}
func TestGuestTransportRejectsWrongProtocolVersion(t *testing.T) {}
func TestGuestTransportAcceptsHeartbeatWithoutExec(t *testing.T) {}
```

The handshake test should assert:

- `handshake` includes `protocol_version`
- `handshake` includes `vm_id`
- `handshake` includes `connection_token`
- `ready` still returns a workspace-root-aware response

**Step 2: Run the tests to verify they fail**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest -run 'Handshake|Heartbeat|Ready'
```

Expected: FAIL because the protocol types and message handlers do not exist yet.

**Step 3: Write the minimal implementation**

- Update `vz-linux-guest-protocol.md` with:
  - `handshake`
  - `handshake_ack`
  - `ready`
  - `heartbeat`
  - `reconnect`
  - `error`
- Extend `types.go` with typed request and response structs for those messages
- Update `transport.go` so the guest stream can parse and encode the new control messages
- Keep newline-delimited JSON and one connection

**Step 4: Run the tests to verify they pass**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest -run 'Handshake|Heartbeat|Ready'
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/tldw-agent/docs/vz-linux-guest-protocol.md tools/tldw-agent/internal/guest/types.go tools/tldw-agent/internal/guest/transport.go tools/tldw-agent/internal/guest/handshake_test.go tools/tldw-agent/internal/guest/transport_test.go
git commit -m "feat(vz_linux): extend guest protocol for handshake and heartbeats"
```

### Task 2: Add Real Guest VSock Mode To `tldw-agent-guest`

**Files:**
- Modify: `tools/tldw-agent/cmd/tldw-agent-guest/main.go`
- Create: `tools/tldw-agent/internal/guest/vsock_client.go`
- Create: `tools/tldw-agent/internal/guest/vsock_client_test.go`
- Modify: `tools/tldw-agent/internal/guest/server.go`
- Modify: `tools/tldw-agent/internal/guest/server_test.go`

**Step 1: Write the failing Go tests**

Add tests like:

```go
func TestGuestVSockClientSendsHandshakeAndReady(t *testing.T) {}
func TestGuestVSockClientReconnectsWithSameVMIDAndToken(t *testing.T) {}
```

The tests should assert:

- guest mode can create a transport client from env/config
- it sends `handshake` before `ready`
- reconnect reuses the same `vm_id` and `connection_token`

**Step 2: Run the tests to verify they fail**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest -run 'VSockClient|Reconnect'
```

Expected: FAIL because the vsock client mode does not exist yet.

**Step 3: Write the minimal implementation**

- Add `vsock_client.go` with a transport client abstraction for:
  - connect
  - send handshake
  - send ready
  - heartbeat loop
  - reconnect loop
- Update `main.go` so guest mode can choose:
  - stdin/stdout mode for tests or local narrow use
  - vsock mode for real guest execution
- Read runtime values from boot-injected envs:
  - `TLDW_AGENT_GUEST_VM_ID`
  - `TLDW_AGENT_GUEST_CONNECTION_TOKEN`
  - `TLDW_AGENT_GUEST_HOST_VSOCK_PORT`
  - `TLDW_AGENT_GUEST_WORKSPACE_ROOT`

**Step 4: Run the tests to verify they pass**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest -run 'VSockClient|Reconnect'
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/tldw-agent/cmd/tldw-agent-guest/main.go tools/tldw-agent/internal/guest/vsock_client.go tools/tldw-agent/internal/guest/vsock_client_test.go tools/tldw-agent/internal/guest/server.go tools/tldw-agent/internal/guest/server_test.go
git commit -m "feat(vz_linux): add guest vsock transport mode"
```

### Task 3: Add The Helper VSock Session Manager

**Files:**
- Create: `tools/macos-vz-helper/Sources/Guest/VSockSessionManager.swift`
- Create: `tools/macos-vz-helper/Sources/Guest/VSockSession.swift`
- Create: `tools/macos-vz-helper/Sources/Guest/VSockListener.swift`
- Modify: `tools/macos-vz-helper/Sources/Guest/VSockBridge.swift`
- Create: `tools/macos-vz-helper/Tests/VSockSessionManagerTests.swift`
- Modify: `tools/macos-vz-helper/Tests/VSockBridgeTests.swift`

**Step 1: Write the failing Swift tests**

Add tests like:

```swift
@Test func vsockSessionManagerBindsHandshakeToExpectedVMID() throws {}
@Test func vsockSessionManagerRejectsWrongConnectionToken() throws {}
@Test func vsockBridgeExecUsesBoundSessionTransport() throws {}
```

The session-manager tests should assert:

- guest handshake is matched to the helper `vm_id`
- wrong token is rejected
- reconnect replaces the prior socket/session for the same VM

**Step 2: Run the tests to verify they fail**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'VSockSessionManagerTests|VSockBridgeTests'
```

Expected: FAIL because the helper still has no real vsock transport/session layer.

**Step 3: Write the minimal implementation**

- Add `VSockSessionManager` to:
  - create listener state per `vm_id`
  - track `connection_token`
  - accept and rebind reconnects
  - expose a transport session for `VSockBridge`
- Add `VSockSession` and `VSockListener` abstractions so unit tests can use fake sessions without booting a VM
- Update `VSockBridge` to send:
  - `ready`
  - `exec_request`
  - decode `exec_response`
  - surface transport failures with existing `GuestBridgeError` mappings

**Step 4: Run the tests to verify they pass**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'VSockSessionManagerTests|VSockBridgeTests'
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/macos-vz-helper/Sources/Guest/VSockSessionManager.swift tools/macos-vz-helper/Sources/Guest/VSockSession.swift tools/macos-vz-helper/Sources/Guest/VSockListener.swift tools/macos-vz-helper/Sources/Guest/VSockBridge.swift tools/macos-vz-helper/Tests/VSockSessionManagerTests.swift tools/macos-vz-helper/Tests/VSockBridgeTests.swift
git commit -m "feat(vz_linux): add helper vsock session manager"
```

### Task 4: Inject VM Identity And Listener State During Boot

**Files:**
- Modify: `tools/macos-vz-helper/Sources/VM/VirtualizationLinuxBootDriver.swift`
- Modify: `tools/macos-vz-helper/Sources/VM/VZLinuxConfigurationBuilder.swift`
- Modify: `tools/macos-vz-helper/Sources/VM/VZLinuxVMManager.swift`
- Modify: `tools/macos-vz-helper/Sources/Server/HelperService.swift`
- Create: `tools/macos-vz-helper/Tests/VirtualizationLinuxBootDriverTransportTests.swift`

**Step 1: Write the failing Swift tests**

Add tests like:

```swift
@Test func bootDriverCreatesVSockListenerStateBeforeStart() throws {}
@Test func bootDriverInjectsVMIDAndConnectionTokenIntoGuestConfig() throws {}
```

The tests should assert:

- listener/session state exists before the VM starts
- per-boot connection token is generated
- guest boot config receives:
  - `vm_id`
  - connection token
  - host vsock port

**Step 2: Run the tests to verify they fail**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'VirtualizationLinuxBootDriverTransportTests'
```

Expected: FAIL because the boot path does not create listener state or inject guest transport metadata.

**Step 3: Write the minimal implementation**

- Update the boot driver to:
  - create helper listener state before VM start
  - generate a connection token per boot
  - store enough metadata to support reconnect
- Update the configuration builder so the guest can see:
  - `vm_id`
  - connection token
  - host vsock port
  - workspace root
- Update `VZLinuxVMManager` or helper service only as needed to preserve existing `create_vm` behavior

**Step 4: Run the tests to verify they pass**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'VirtualizationLinuxBootDriverTransportTests'
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/macos-vz-helper/Sources/VM/VirtualizationLinuxBootDriver.swift tools/macos-vz-helper/Sources/VM/VZLinuxConfigurationBuilder.swift tools/macos-vz-helper/Sources/VM/VZLinuxVMManager.swift tools/macos-vz-helper/Sources/Server/HelperService.swift tools/macos-vz-helper/Tests/VirtualizationLinuxBootDriverTransportTests.swift
git commit -m "feat(vz_linux): inject guest transport boot metadata"
```

### Task 5: Make `create_vm` Wait For Real Guest Readiness

**Files:**
- Modify: `tools/macos-vz-helper/Sources/Guest/VSockBridge.swift`
- Modify: `tools/macos-vz-helper/Sources/Server/HelperService.swift`
- Modify: `tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift`
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`
- Modify: `tools/macos-vz-helper/PROTOCOL.md`

**Step 1: Write the failing transport contract tests**

Add tests like:

```python
def test_socket_helper_create_vm_moves_past_boot_not_implemented(...): ...
def test_socket_helper_exec_guest_uses_real_transport_error_codes(...): ...
```

And add Swift tests that expect:

```swift
@Test func helperServiceCreateVMWaitsForGuestReadyBeforeReturning() throws {}
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py -q
cd tools/macos-vz-helper && swift test --filter 'HelperServiceVMTests|UnixSocketServerTests'
```

Expected: FAIL because the helper still does not wait on a real guest transport session.

**Step 3: Write the minimal implementation**

- Make `create_vm` succeed only after:
  - handshake accepted
  - `ready` received
- Update `exec_guest` to use the durable session transport instead of the stub
- Keep helper error codes explicit for:
  - guest handshake timeout
  - guest protocol mismatch
  - transport disconnect
- Update `PROTOCOL.md` only if helper-visible error shapes change

**Step 4: Run the tests to verify they pass**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py -q
cd tools/macos-vz-helper && swift test --filter 'HelperServiceVMTests|UnixSocketServerTests'
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/macos-vz-helper/Sources/Guest/VSockBridge.swift tools/macos-vz-helper/Sources/Server/HelperService.swift tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py tools/macos-vz-helper/PROTOCOL.md
git commit -m "feat(vz_linux): wait for real guest readiness"
```

### Task 6: Turn The Host-Gated Smokes Into Real End-To-End Proof

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py`
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify: `tools/vz-linux-image/README.md`

**Step 1: Write the failing host-gated expectations**

Extend the host-gated tests so that on a prepared Apple silicon host:

- canonical bundle smoke expects real guest `ready`
- real host E2E expects:
  - ephemeral `/bin/echo`
  - session reuse with a second command

**Step 2: Run the host-gated tests to verify current failure or skip**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q
```

Expected: PASS for non-opt-in unit/skip paths; prepared-host paths still skip unless all envs and artifacts are present.

**Step 3: Write the minimal implementation**

- Update the daemon smoke so the canonical-bundle path proves:
  - helper `create_vm` returns success
  - helper or E2E can reach a real guest `ready`
- Update the E2E module so success means:
  - real command execution in the guest
  - session VM reuse still works over the durable transport
- Update operator docs and sandbox README to describe:
  - new envs if any
  - the reconnect-aware durable transport
  - the prepared-host expectations

**Step 4: Run the tests to verify they pass**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q
```

Expected: PASS for default skip paths; PASS on a prepared Apple silicon host with the helper, bundle, and guest agent wired end-to-end.

**Step 5: Commit**

```bash
git add tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py Docs/Sandbox/macos-runtime-operator-notes.md tldw_Server_API/app/core/Sandbox/README.md tools/vz-linux-image/README.md
git commit -m "test(vz_linux): prove real vsock guest transport"
```

### Task 7: Run Verification And Security Checks

**Files:**
- Verify: `tools/tldw-agent/`
- Verify: `tools/macos-vz-helper/`
- Verify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Verify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`
- Verify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py`
- Verify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`

**Step 1: Run Go guest-agent verification**

Run:

```bash
cd tools/tldw-agent && bash ./scripts/verify-local-build.sh
```

Expected: PASS

**Step 2: Run full Swift helper verification**

Run:

```bash
cd tools/macos-vz-helper && swift test
```

Expected: PASS

**Step 3: Run focused Python verification**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q
```

Expected: PASS for unit/contract tests, with host-gated tests skipping cleanly unless all real prerequisites are present

**Step 4: Run Bandit on the changed Python implementation scope**

Run:

```bash
source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py -f json -o /tmp/bandit_vz_linux_vsock_transport.json
```

Expected: `results: 0`, `errors: 0`

**Step 5: Commit**

```bash
git add -A
git commit -m "chore(vz_linux): finalize vsock transport verification"
```
