# VZ Linux First-Party Helper MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate `tldw-agent` into this repo, add a guest-mode data plane plus a Swift macOS helper daemon, and make the existing `vz_linux` real-host E2E tests pass on prepared Apple silicon Macs.

**Architecture:** Keep Python authoritative for sandbox/session semantics, add a Swift Unix-socket helper for `Virtualization.framework` lifecycle, and migrate `tldw-agent` into `tools/tldw-agent/` as the first-party in-guest agent with a new `vz_linux` guest mode over a separate vsock protocol.

**Tech Stack:** Python, Swift Package Manager, `Virtualization.framework`, Go, Unix domain sockets, vsock, `virtiofs`, pytest, XCTest, Go test

---

### Task 1: Migrate `tldw-agent` Into `tools/tldw-agent/` Intact

**Files:**
- Create: `tools/tldw-agent/` (full source import from `../tldw-agent`)
- Create: `tools/tldw-agent/MIGRATION.md`
- Create: `tools/tldw-agent/README_TLDW_SERVER.md`
- Test: `tools/tldw-agent/internal/acp/runner_test.go`
- Test: `tools/tldw-agent/internal/acp/conn_test.go`

**Step 1: Write the failing provenance/build expectations**

Add `MIGRATION.md` with placeholders for:

```markdown
# tldw-agent Migration

- Source repo: `../tldw-agent`
- Upstream commit: `<fill me in during migration>`
- Migration rule: preserve existing behavior before adding guest mode
```

Add `README_TLDW_SERVER.md` with a short statement that `tools/tldw-agent/` is now the in-repo source of truth.

**Step 2: Run baseline checks to verify the repo copy does not exist yet**

Run:

```bash
test -d tools/tldw-agent && echo "exists" || echo "missing"
```

Expected: `missing`

**Step 3: Copy the source tree with minimal changes**

- Copy the full `../tldw-agent` repo into `tools/tldw-agent/`
- Do not rename packages or rewrite module layout yet
- Add provenance docs only

**Step 4: Run the migrated Go test suite**

Run:

```bash
cd tools/tldw-agent && go test ./...
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/tldw-agent
git commit -m "chore(vz_linux): migrate tldw-agent into repo"
```

### Task 2: Preserve Existing `tldw-agent` Host And ACP Builds

**Files:**
- Modify: `tools/tldw-agent/README.md`
- Create: `tools/tldw-agent/scripts/verify-local-build.sh`
- Test: `tools/tldw-agent/internal/native/handler.go`
- Test: `tools/tldw-agent/internal/acp/runner_test.go`

**Step 1: Write the failing local-build verification script**

Create `scripts/verify-local-build.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
go build ./cmd/tldw-agent-host
go build ./cmd/tldw-agent-acp
go test ./...
```

**Step 2: Run it to confirm the script fails before it exists**

Run:

```bash
cd tools/tldw-agent && ./scripts/verify-local-build.sh
```

Expected: shell error about missing file

**Step 3: Add the script and update docs**

- Add the script
- Update `tools/tldw-agent/README.md` to document that host/native-messaging and ACP modes must remain green during the guest-mode work

**Step 4: Run the script**

Run:

```bash
cd tools/tldw-agent && bash ./scripts/verify-local-build.sh
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/tldw-agent/README.md tools/tldw-agent/scripts/verify-local-build.sh
git commit -m "test(vz_linux): preserve migrated tldw-agent host builds"
```

### Task 3: Define The Guest Protocol And Add A Guest Entrypoint

**Files:**
- Create: `tools/tldw-agent/docs/vz-linux-guest-protocol.md`
- Create: `tools/tldw-agent/cmd/tldw-agent-guest/main.go`
- Create: `tools/tldw-agent/internal/guest/types.go`
- Create: `tools/tldw-agent/internal/guest/types_test.go`

**Step 1: Write the failing protocol parser tests**

Create `internal/guest/types_test.go` with tests like:

```go
func TestParseExecRequest(t *testing.T) {
    req := ExecRequest{
        ProtocolVersion: "1",
        RequestID: "req-1",
        Argv: []string{"/bin/echo", "ok"},
        Cwd: "/workspace",
    }
    if req.ProtocolVersion != "1" {
        t.Fatalf("expected protocol version 1")
    }
}
```

**Step 2: Run the new test to verify it fails**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest -run TestParseExecRequest
```

Expected: FAIL because the package/files do not exist yet

**Step 3: Add the minimal guest protocol types and entrypoint**

- Define narrow guest request/response structs for:
  - readiness handshake
  - exec request
  - exec reply
  - error reply
- Add `cmd/tldw-agent-guest/main.go` with a placeholder server bootstrap
- Write `docs/vz-linux-guest-protocol.md` documenting the separate guest protocol version `1`

**Step 4: Run the guest package tests**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/tldw-agent/docs/vz-linux-guest-protocol.md tools/tldw-agent/cmd/tldw-agent-guest/main.go tools/tldw-agent/internal/guest
git commit -m "feat(vz_linux): add tldw-agent guest protocol types"
```

### Task 4: Implement Guest-Mode `tldw-agent` Readiness And Exec Handling

**Files:**
- Create: `tools/tldw-agent/internal/guest/server.go`
- Create: `tools/tldw-agent/internal/guest/exec.go`
- Create: `tools/tldw-agent/internal/guest/server_test.go`
- Create: `tools/tldw-agent/internal/guest/exec_test.go`
- Modify: `tools/tldw-agent/cmd/tldw-agent-guest/main.go`
- Reuse: `tools/tldw-agent/internal/workspace/session.go`

**Step 1: Write the failing guest server tests**

Add tests that expect:

```go
func TestGuestServerReportsReady(t *testing.T) {}
func TestGuestServerExecutesArgvWithoutShell(t *testing.T) {}
func TestGuestServerRejectsEmptyArgv(t *testing.T) {}
```

**Step 2: Run the guest server tests to verify they fail**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest -run 'TestGuestServer|TestGuestExec'
```

Expected: FAIL because readiness/exec behavior is not implemented yet

**Step 3: Write the minimal implementation**

- Add a guest server loop that can:
  - answer readiness requests
  - accept exec requests
  - resolve workspace-root-aware cwd
  - run `exec.CommandContext` with direct `argv`
  - capture stdout/stderr/exit status without shell interpolation
- Keep protocol framing separate from native messaging

**Step 4: Run the guest package tests**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/tldw-agent/internal/guest tools/tldw-agent/cmd/tldw-agent-guest/main.go
git commit -m "feat(vz_linux): add guest-mode tldw-agent exec server"
```

### Task 5: Bootstrap The Swift Helper Daemon Package

**Files:**
- Create: `tools/macos-vz-helper/Package.swift`
- Create: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/main.swift`
- Create: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/Protocol/Request.swift`
- Create: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/Protocol/Response.swift`
- Create: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/Server/UnixSocketServer.swift`
- Create: `tools/macos-vz-helper/Tests/MacOSVZHelperDaemonTests/PingTests.swift`
- Create: `tools/macos-vz-helper/Tests/MacOSVZHelperDaemonTests/ValidateHostTests.swift`

**Step 1: Write the failing Swift protocol tests**

Add XCTest cases that expect:

```swift
func testPingIncludesProtocolAndHelperVersion() throws
func testValidateHostReturnsUnavailableOnUnsupportedConfig() throws
```

**Step 2: Run them to verify they fail**

Run:

```bash
cd tools/macos-vz-helper && swift test
```

Expected: FAIL because the package does not exist yet

**Step 3: Add the helper package scaffold**

- Create the Swift package
- Add a Unix-socket server skeleton
- Implement `ping`
- Implement a narrow `validate_host` stub that returns structured availability/reasons
- Match the frozen Python helper protocol exactly

**Step 4: Run the Swift tests**

Run:

```bash
cd tools/macos-vz-helper && swift test
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/macos-vz-helper
git commit -m "feat(vz_linux): add Swift helper daemon scaffold"
```

### Task 6: Add Template Validation And VM Registry Semantics To The Helper

**Files:**
- Create: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/Templates/TemplateValidator.swift`
- Create: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/VM/VMRegistry.swift`
- Create: `tools/macos-vz-helper/Tests/MacOSVZHelperDaemonTests/TemplateValidatorTests.swift`
- Create: `tools/macos-vz-helper/Tests/MacOSVZHelperDaemonTests/VMRegistryTests.swift`
- Modify: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/main.swift`

**Step 1: Write the failing template/registry tests**

Add tests that expect:

```swift
func testValidateTemplateRejectsMissingImagePath() throws
func testVMRegistryTracksCreatedVMIDs() throws
func testListVMsReturnsKnownVMs() throws
```

**Step 2: Run them to verify they fail**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'TemplateValidatorTests|VMRegistryTests'
```

Expected: FAIL because template validation and VM listing do not exist yet

**Step 3: Write the minimal implementation**

- Validate direct base-image paths first
- Add an in-memory VM registry for:
  - created VM ids
  - health/state
  - list/status lookups
- Implement `validate_template`, `get_vm_status`, and `list_vms`

**Step 4: Run the filtered tests**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'TemplateValidatorTests|VMRegistryTests'
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/macos-vz-helper
git commit -m "feat(vz_linux): add helper template validation and vm registry"
```

### Task 7: Add A Reproducible Reference Image Path

**Files:**
- Create: `tools/vz-linux-image/README.md`
- Create: `tools/vz-linux-image/Makefile`
- Create: `tools/vz-linux-image/scripts/install-agent.sh`
- Create: `tools/vz-linux-image/scripts/smoke-check.sh`
- Create: `tools/vz-linux-image/cloud-init/user-data`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`

**Step 1: Write the failing documentation/test expectations**

Document the required outcome:

```markdown
- one base image path consumable as `RunSpec.base_image`
- guest-mode `tldw-agent` installed in the image
- enough userspace for `/bin/echo`
```

Add `smoke-check.sh` to assert the built image or mounted rootfs contains the guest agent binary at the documented location.

**Step 2: Run the smoke-check to verify it fails before assets exist**

Run:

```bash
bash tools/vz-linux-image/scripts/smoke-check.sh
```

Expected: shell error or failure because the image assets are not there yet

**Step 3: Add the reference image assets**

- Add a minimal README and Makefile
- Add an install script that places `tldw-agent-guest` into the image
- Add minimal cloud-init or equivalent provisioning assets
- Update operator notes with the new local image build path

**Step 4: Run the smoke-check**

Run:

```bash
bash tools/vz-linux-image/scripts/smoke-check.sh
```

Expected: PASS on a prepared local image build workspace

**Step 5: Commit**

```bash
git add tools/vz-linux-image Docs/Sandbox/macos-runtime-operator-notes.md
git commit -m "docs(vz_linux): add reference image build path"
```

### Task 8: Boot A Real VM And Wait For Guest-Agent Readiness

**Files:**
- Create: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/VM/VZLinuxVMManager.swift`
- Create: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/Guest/VSockBridge.swift`
- Create: `tools/macos-vz-helper/Tests/MacOSVZHelperDaemonTests/VMBootTests.swift`
- Modify: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/main.swift`
- Modify: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/VM/VMRegistry.swift`

**Step 1: Write the failing boot/readiness tests**

Add tests that expect helper operations to model:

```swift
func testCreateVMRegistersBootingState() throws
func testGuestReadinessTransitionsVMToRunning() throws
func testTerminateVMRemovesRegistryState() throws
```

**Step 2: Run them to verify they fail**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'VMBootTests'
```

Expected: FAIL because real boot/readiness state handling is missing

**Step 3: Write the minimal implementation**

- Add a `VZLinuxVMManager`
- Configure `virtiofs` and vsock
- Start the VM
- Wait for explicit guest-agent readiness over the guest protocol
- Update registry state to healthy/running only after readiness succeeds
- Implement `create_vm` and `terminate_vm` against that manager

**Step 4: Run the filtered tests**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'VMBootTests'
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/macos-vz-helper
git commit -m "feat(vz_linux): boot vm and wait for guest readiness"
```

### Task 9: Bridge `exec_guest` Through The Helper And Turn Python E2E Green

**Files:**
- Modify: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/Guest/VSockBridge.swift`
- Modify: `tools/macos-vz-helper/Sources/MacOSVZHelperDaemon/main.swift`
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`

**Step 1: Write the failing integrated expectations**

The existing Python E2E already expresses the target behavior:

- ephemeral `/bin/echo vz-linux-e2e`
- session reuse for `/bin/echo first` then `/bin/echo second`
- cleanup on `destroy_session()`

If needed, add one extra assertion that the helper-reported protocol versions are surfaced in docs/setup guidance.

**Step 2: Run the host-gated E2E to verify it still skips or fails**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q
```

Expected: skip on unprepared hosts, or fail until `exec_guest` is really bridged

**Step 3: Write the minimal implementation**

- Bridge helper `exec_guest` to guest-mode `tldw-agent` over vsock
- Return stdout/stderr/exit status in the frozen helper protocol
- Update docs to describe how to start the helper and point pytest at the real socket/image

**Step 4: Run the host-gated E2E**

Run:

```bash
source .venv/bin/activate && TLDW_SANDBOX_VZ_LINUX_E2E=1 TLDW_SANDBOX_MACOS_HELPER_SOCKET=/path/to/helper.sock TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=/path/to/base.img python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q
```

Expected: PASS on a prepared Apple silicon host

**Step 5: Commit**

```bash
git add tools/macos-vz-helper tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py Docs/Sandbox/macos-runtime-operator-notes.md tldw_Server_API/app/core/Sandbox/README.md
git commit -m "feat(vz_linux): wire helper exec path into real host e2e"
```

### Task 10: Run Verification And Security Checks

**Files:**
- Modify: none
- Test: `tools/tldw-agent/internal/guest/...`
- Test: `tools/tldw-agent/internal/acp/...`
- Test: `tools/macos-vz-helper/Tests/MacOSVZHelperDaemonTests/...`
- Test: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py`

**Step 1: Run the migrated `tldw-agent` tests**

Run:

```bash
cd tools/tldw-agent && go test ./...
```

Expected: PASS

**Step 2: Run the Swift helper tests**

Run:

```bash
cd tools/macos-vz-helper && swift test
```

Expected: PASS

**Step 3: Run the focused Python regression suite**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_runner.py \
  tldw_Server_API/tests/sandbox/test_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_feature_discovery_flags.py \
  tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_session_cleanup.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py -q
```

Expected: PASS; host-gated E2E may still skip on unprepared hosts before local helper/image setup is complete

**Step 4: Run Bandit on the touched Python scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Sandbox/macos_virtualization \
  tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py \
  tldw_Server_API/app/core/Sandbox/macos_diagnostics.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/app/core/Sandbox/store.py \
  tldw_Server_API/app/core/Sandbox/orchestrator.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  -f json -o /tmp/bandit_vz_linux_first_party_helper_mvp.json
```

Expected: `results: 0`, `errors: 0`

**Step 5: Commit**

```bash
git add -A
git commit -m "chore(vz_linux): finalize first-party helper mvp verification"
```
