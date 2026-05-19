# VZ Linux Guest Output Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce `vz_linux` output caps inside the Linux guest agent, forward the cap through the Swift helper bridge, and surface guest-side enforcement metadata without changing the public sandbox API.

**Architecture:** Keep the existing Python-to-helper `exec_guest` contract and the existing guest request/response protocol shape. Python continues to choose the cap, Swift validates and forwards it, and the Go guest agent owns bounded capture plus process termination. Swift keeps host-side response capping as defense in depth and Python parses guest-prefixed counters defensively.

**Tech Stack:** Go guest agent (`tools/tldw-agent`), Swift Package helper (`tools/macos-vz-helper`), Python sandbox runner/client (`tldw_Server_API/app/core/Sandbox`), pytest, Go test, Swift test.

---

## Preconditions

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/vz-guest-output-cap`
- Branch: `codex/vz-guest-output-cap`
- Spec: `Docs/superpowers/specs/2026-05-02-vz-guest-output-cap-design.md`
- Python venv for this nested worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv`
- Root checkout has unrelated dirty state in `Docs/Design/Agents.md`; do not touch or revert it from this worktree.

## File Map

- Modify: `tools/tldw-agent/internal/guest/types.go`
  - Add `MaxOutputBytes *int` and `Details map[string]string` to guest exec request/response.
- Modify: `tools/tldw-agent/internal/guest/exec.go`
  - Replace unbounded `bytes.Buffer` capture with pipe-based bounded capture and output-limit cancellation.
- Create: `tools/tldw-agent/internal/guest/process_linux.go`
  - Linux process-group setup and termination helpers.
- Create: `tools/tldw-agent/internal/guest/process_unsupported.go`
  - Portable fallback process termination helpers for non-Linux builds.
- Modify: `tools/tldw-agent/internal/guest/exec_test.go`
  - Add unit coverage for cap enforcement, direct cap validation, UTF-8 safety, timeout distinction, and noisy-process cleanup.
- Modify: `tools/tldw-agent/internal/guest/types_test.go`
  - Add JSON parse/encode coverage for omitted, valid, and invalid `max_output_bytes`.
- Modify: `tools/tldw-agent/docs/vz-linux-guest-protocol.md`
  - Document guest `max_output_bytes` and guest-prefixed response details.
- Modify: `tools/macos-vz-helper/Sources/Guest/VSockBridge.swift`
  - Forward `maxOutputBytes`, decode guest details defensively, and expose details on `GuestExecResult`.
- Modify: `tools/macos-vz-helper/Sources/VM/VZLinuxVMManager.swift`
  - Thread `maxOutputBytes` through to `GuestBridging.exec`.
- Modify: `tools/macos-vz-helper/Sources/Server/HelperService.swift`
  - Pass validated cap to `vmManager.execGuest` and merge only guest-prefixed detail values.
- Modify: `tools/macos-vz-helper/Tests/TestDoubles.swift`
  - Update test doubles for the new `GuestBridging.exec` signature.
- Modify: `tools/macos-vz-helper/Tests/VSockBridgeTests.swift`
  - Add request encoding and detail decoding tests.
- Modify: `tools/macos-vz-helper/Tests/HelperServiceExecTests.swift`
  - Add cap forwarding and guest/host detail merge tests.
- Modify: `tools/macos-vz-helper/PROTOCOL.md`
  - Update helper protocol wording from host-only cap to guest-forwarded cap plus host fallback.
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
  - Add guest-prefixed output counters to `_OUTPUT_COUNTER_KEYS`.
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
  - Add defensive parsing/resource usage tests for guest counters.
- Modify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`
  - Add helper-client detail pass-through tests only if current coverage does not already assert it.
- Modify: `tools/vz-linux-image/README.md`
  - Note that guest kill-on-cap requires rebuilding images with the updated `tldw-agent-guest`.

## Task 1: Go Guest Protocol Shape

**Files:**
- Modify: `tools/tldw-agent/internal/guest/types.go`
- Modify: `tools/tldw-agent/internal/guest/types_test.go`

- [ ] **Step 1: Write failing type tests**

Add tests covering omitted cap, valid cap, and explicit zero:

```go
func TestExecRequestMaxOutputBytesOptional(t *testing.T) {
	var req ExecRequest
	if err := json.Unmarshal([]byte(`{"protocol_version":"1","request_id":"req-1","type":"exec","argv":["/bin/echo","ok"]}`), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.MaxOutputBytes != nil {
		t.Fatalf("expected nil MaxOutputBytes, got %v", *req.MaxOutputBytes)
	}
}

func TestExecRequestMaxOutputBytesPreservesExplicitZero(t *testing.T) {
	var req ExecRequest
	if err := json.Unmarshal([]byte(`{"protocol_version":"1","request_id":"req-1","type":"exec","argv":["/bin/echo","ok"],"max_output_bytes":0}`), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.MaxOutputBytes == nil || *req.MaxOutputBytes != 0 {
		t.Fatalf("expected explicit zero cap, got %#v", req.MaxOutputBytes)
	}
}
```

- [ ] **Step 2: Run tests to confirm failure**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest -run 'TestExecRequestMaxOutputBytes' -count=1
```

Expected: fail because `ExecRequest.MaxOutputBytes` does not exist.

- [ ] **Step 3: Add request/response fields**

Update `ExecRequest` and `ExecResponse`:

```go
type ExecRequest struct {
	ProtocolVersion string            `json:"protocol_version"`
	RequestID       string            `json:"request_id"`
	Type            string            `json:"type,omitempty"`
	Argv            []string          `json:"argv"`
	Cwd             string            `json:"cwd,omitempty"`
	Env             map[string]string `json:"env,omitempty"`
	TimeoutSec      int               `json:"timeout_sec,omitempty"`
	MaxOutputBytes  *int              `json:"max_output_bytes,omitempty"`
}

type ExecResponse struct {
	ProtocolVersion string            `json:"protocol_version"`
	RequestID       string            `json:"request_id"`
	ExitCode        int               `json:"exit_code"`
	Stdout          string            `json:"stdout,omitempty"`
	Stderr          string            `json:"stderr,omitempty"`
	Details         map[string]string `json:"details,omitempty"`
}
```

- [ ] **Step 4: Verify type tests pass**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest -run 'TestExecRequestMaxOutputBytes' -count=1
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tools/tldw-agent/internal/guest/types.go tools/tldw-agent/internal/guest/types_test.go
git commit -m "feat(vz): extend guest exec protocol for output caps"
```

## Task 2: Go Bounded Capture And Process Termination

**Files:**
- Modify: `tools/tldw-agent/internal/guest/exec.go`
- Create: `tools/tldw-agent/internal/guest/process_linux.go`
- Create: `tools/tldw-agent/internal/guest/process_unsupported.go`
- Modify: `tools/tldw-agent/internal/guest/exec_test.go`

- [ ] **Step 1: Write failing tests for direct cap validation**

Add tests that call `Server.Exec` with `MaxOutputBytes` set to `0` and `268435457`.

Expected `ErrorResponse`:

```go
ErrorCode: "invalid_request"
Message:   "max_output_bytes out of range"
```

- [ ] **Step 2: Write failing tests for cap exceeded metadata**

Use a shell command that writes more than a small cap:

```go
capBytes := 16
resp, execErr := server.Exec(ExecRequest{
	ProtocolVersion: ProtocolVersion,
	RequestID:       "req-cap",
	Type:            "exec",
	Argv:            []string{"/bin/sh", "-c", "i=0; while [ $i -lt 4096 ]; do printf x; i=$((i+1)); done"},
	MaxOutputBytes:  &capBytes,
})
if execErr != nil {
	t.Fatalf("Exec() unexpected error = %#v", execErr)
}
if len([]byte(resp.Stdout))+len([]byte(resp.Stderr)) > capBytes {
	t.Fatalf("returned output exceeds cap")
}
if resp.ExitCode != 137 {
	t.Fatalf("expected output-limit exit 137, got %d", resp.ExitCode)
}
if resp.Details["guest_output_limit_exceeded"] != "true" {
	t.Fatalf("expected guest output limit metadata")
}
```

If the shell loop is too slow on a platform, use `/usr/bin/yes` with a short timeout and expect the cap cancellation to win.

- [ ] **Step 3: Write failing UTF-8 truncation test**

Use multibyte output such as `printf 'éééé'`, cap to an odd byte count, and assert `utf8.ValidString(resp.Stdout)` and returned bytes are at or under the cap.

- [ ] **Step 4: Run failing Go tests**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest -run 'TestGuestExec.*Output|TestGuestExec.*MaxOutput|TestGuestExec.*UTF8' -count=1
```

Expected: fail because bounded capture is not implemented.

- [ ] **Step 5: Add platform process helpers**

Create `process_linux.go`:

```go
//go:build linux

package guest

import (
	"os/exec"
	"syscall"
)

func configureCommandProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func terminateCommandProcess(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	pid := cmd.Process.Pid
	if pid > 0 {
		_ = syscall.Kill(-pid, syscall.SIGKILL)
	}
	_ = cmd.Process.Kill()
}
```

Create `process_unsupported.go`:

```go
//go:build !linux

package guest

import "os/exec"

func configureCommandProcessGroup(_ *exec.Cmd) {}

func terminateCommandProcess(cmd *exec.Cmd) {
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
}
```

- [ ] **Step 6: Implement bounded capture helper**

In `exec.go`, add a small focused helper with mutex-protected shared state:

```go
type outputLimitReason string

const (
	outputLimitReasonNone   outputLimitReason = ""
	outputLimitReasonOutput outputLimitReason = "output_limit"
	outputLimitExitCode     int               = 137
	maxGuestOutputBytes     int               = 256 * 1024 * 1024
)

type boundedExecOutput struct {
	mu           sync.Mutex
	limit        int
	stdout       []byte
	stderr       []byte
	stdoutSeen   int
	stderrSeen   int
	exceeded     bool
	cancelReason outputLimitReason
	cancel       context.CancelFunc
	kill         func()
}
```

The writer method should:

- increment observed bytes for the relevant stream
- append only remaining budget
- set `exceeded=true` and `cancelReason=output_limit` once
- call cancel and kill callbacks once
- return `len(p), nil` so `io.Copy` exits because the process is killed, not because a writer error is misinterpreted as command failure

Use `strings.ToValidUTF8` and a byte-safe trim after sanitization so the final returned output remains within the cap.

- [ ] **Step 7: Replace `cmd.Run()` path**

Use:

```go
stdoutPipe, err := cmd.StdoutPipe()
stderrPipe, err := cmd.StderrPipe()
configureCommandProcessGroup(cmd)
err = cmd.Start()
go copy stdout
go copy stderr
waitErr := cmd.Wait()
wait for copy goroutines
```

Classify results in this order:

1. If timeout context won first: return `timeout_exceeded` error.
2. If output cap won first: return normal `ExecResponse` with `ExitCode: 137`.
3. Else preserve existing exit-code behavior for normal command failures.

- [ ] **Step 8: Verify focused Go tests**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest -run 'TestGuestExec.*Output|TestGuestExec.*MaxOutput|TestGuestExec.*UTF8|TestGuestServerExecutesArgvWithoutShell' -count=1
```

Expected: pass.

- [ ] **Step 9: Verify broader Go guest package**

Run:

```bash
cd tools/tldw-agent && go test ./internal/guest -count=1
```

Expected: pass.

- [ ] **Step 10: Commit**

```bash
git add tools/tldw-agent/internal/guest/exec.go tools/tldw-agent/internal/guest/process_linux.go tools/tldw-agent/internal/guest/process_unsupported.go tools/tldw-agent/internal/guest/exec_test.go
git commit -m "feat(vz): enforce guest output caps"
```

## Task 3: Swift Helper Cap Forwarding And Detail Merge

**Files:**
- Modify: `tools/macos-vz-helper/Sources/Guest/VSockBridge.swift`
- Modify: `tools/macos-vz-helper/Sources/VM/VZLinuxVMManager.swift`
- Modify: `tools/macos-vz-helper/Sources/Server/HelperService.swift`
- Modify: `tools/macos-vz-helper/Tests/TestDoubles.swift`
- Modify: `tools/macos-vz-helper/Tests/VSockBridgeTests.swift`
- Modify: `tools/macos-vz-helper/Tests/HelperServiceExecTests.swift`

- [ ] **Step 1: Write failing VSock request encoding test**

In `VSockBridgeTests.swift`, assert the encoded guest request includes `"max_output_bytes":10` when `bridge.exec(..., maxOutputBytes: 10)` is called.

- [ ] **Step 2: Write failing detail decoding test**

Return a guest response with:

```json
"details":{"guest_output_limit_exceeded":"true","ignored_number":1}
```

Assert `GuestExecResult.details["guest_output_limit_exceeded"] == "true"` and `ignored_number` is absent.

- [ ] **Step 3: Write failing HelperService merge test**

Use a guest bridge double that returns large output plus guest-prefixed details. Assert helper response details contain both:

- host keys: `stdout_bytes_original`, `stdout_bytes_returned`, `stdout_truncated`
- guest keys: `guest_output_limit_exceeded`, `guest_stdout_bytes_observed`

Also assert guest keys do not overwrite host keys.

- [ ] **Step 4: Run failing Swift tests**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'VSockBridgeTests|HelperServiceExecTests'
```

Expected: fail because signatures/details are not implemented.

- [ ] **Step 5: Update bridge types and signatures**

Change `GuestExecResult`:

```swift
struct GuestExecResult {
    let exitCode: Int
    let stdout: String
    let stderr: String
    let details: [String: String]
}
```

Add `maxOutputBytes: Int?` to `GuestBridging.exec`, `VZLinuxVMManager.execGuest`, and `VSockBridge.exec`.

Add optional `maxOutputBytes` to `GuestExecRequest` with coding key `max_output_bytes`.

- [ ] **Step 6: Decode guest details defensively**

Implement a custom `GuestExecResponse` decoder that reads details as `[String: JSONValue]` or a permissive dictionary and keeps only string values whose keys start with `guest_`.

Do not throw if `details` is missing, not an object, or contains non-string values.

- [ ] **Step 7: Pass cap from HelperService**

Update:

```swift
let result = try vmManager.execGuest(
    vmID: vmID,
    argv: argv,
    cwd: cwd,
    env: env,
    timeoutSeconds: timeoutSeconds,
    maxOutputBytes: maxOutputBytes
)
```

Merge details:

```swift
var details = ["transport": "vsock", "vm_id": vmID]
for (key, value) in result.details where key.hasPrefix("guest_") {
    details[key] = value
}
for (key, value) in cappedOutput.details {
    details[key] = value
}
```

- [ ] **Step 8: Verify focused Swift tests**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'VSockBridgeTests|HelperServiceExecTests'
```

Expected: pass.

- [ ] **Step 9: Verify full Swift package**

Run:

```bash
cd tools/macos-vz-helper && swift test
```

Expected: pass.

- [ ] **Step 10: Commit**

```bash
git add tools/macos-vz-helper/Sources/Guest/VSockBridge.swift tools/macos-vz-helper/Sources/VM/VZLinuxVMManager.swift tools/macos-vz-helper/Sources/Server/HelperService.swift tools/macos-vz-helper/Tests/TestDoubles.swift tools/macos-vz-helper/Tests/VSockBridgeTests.swift tools/macos-vz-helper/Tests/HelperServiceExecTests.swift
git commit -m "feat(vz): forward output caps to guest agent"
```

## Task 4: Python Guest Counter Parsing

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py` only if needed

- [ ] **Step 1: Write failing runner counter test**

Add a focused unit test for `_output_counters_from_details`:

```python
def test_output_counters_include_guest_enforcement_details():
    counters = VZLinuxRunner._output_counters_from_details(
        {
            "guest_output_limit_bytes": "16",
            "guest_output_limit_exceeded": "true",
            "guest_stdout_bytes_observed": "17",
            "guest_stderr_bytes_observed": "0",
            "guest_stdout_bytes_returned": "16",
            "guest_stderr_bytes_returned": "0",
            "guest_output_kill_reason": "output_limit",
            "ignored": "not-int",
        }
    )
    assert counters["guest_output_limit_bytes"] == 16
    assert counters["guest_output_limit_exceeded"] == 1
    assert counters["guest_stdout_bytes_observed"] == 17
    assert counters["guest_stdout_bytes_returned"] == 16
    assert "guest_output_kill_reason" not in counters
```

- [ ] **Step 2: Run failing pytest**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/sandbox/test_vz_linux_runner.py -k output_counters
```

Expected: fail because guest keys are not in `_OUTPUT_COUNTER_KEYS`.

- [ ] **Step 3: Add guest counter keys**

Update `_OUTPUT_COUNTER_KEYS` with:

```python
"guest_output_limit_bytes",
"guest_output_limit_exceeded",
"guest_stdout_bytes_observed",
"guest_stderr_bytes_observed",
"guest_stdout_bytes_returned",
"guest_stderr_bytes_returned",
```

Do not include string reason fields in `resource_usage`.

- [ ] **Step 4: Verify focused pytest**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/sandbox/test_vz_linux_runner.py -k output_counters
```

Expected: pass.

- [ ] **Step 5: Verify helper-client pass-through if touched**

If `test_macos_virtualization_helper_client.py` is changed, run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py
git commit -m "feat(vz): surface guest output cap counters"
```

If `test_macos_virtualization_helper_client.py` is not touched, omit it from `git add`.

## Task 5: Protocol And Operator Documentation

**Files:**
- Modify: `tools/macos-vz-helper/PROTOCOL.md`
- Modify: `tools/tldw-agent/docs/vz-linux-guest-protocol.md`
- Modify: `tools/vz-linux-image/README.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`

- [ ] **Step 1: Update helper protocol docs**

In `tools/macos-vz-helper/PROTOCOL.md`, replace the host-only warning with:

```markdown
`max_output_bytes` is forwarded to guest agents that support guest-side output
cap enforcement. Rebuilt guests terminate the command when the combined
stdout/stderr observation exceeds the cap and return guest-prefixed detail
metadata. The helper still applies host-side response capping as defense in
depth and as fallback for older guests.
```

- [ ] **Step 2: Update guest protocol docs**

Add `max_output_bytes` to the guest exec request example and add a details block to the response example with guest-prefixed keys.

- [ ] **Step 3: Update image docs**

In `tools/vz-linux-image/README.md`, add a short note:

```markdown
Guest-side kill-on-output-cap requires images rebuilt with the updated
`tldw-agent-guest` binary. Older images still boot and execute, but only the
host helper response cap is guaranteed.
```

- [ ] **Step 4: Update sandbox README**

Update the `vz_linux` output cap bullet to say guest-side enforcement is available when the image contains the updated guest agent, while host-side cap remains fallback.

- [ ] **Step 5: Check markdown diff**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add tools/macos-vz-helper/PROTOCOL.md tools/tldw-agent/docs/vz-linux-guest-protocol.md tools/vz-linux-image/README.md tldw_Server_API/app/core/Sandbox/README.md
git commit -m "docs(vz): document guest output cap enforcement"
```

## Task 6: Full Focused Verification

**Files:**
- No implementation files unless verification reveals defects.

- [ ] **Step 1: Run Go guest tests**

```bash
cd tools/tldw-agent && go test ./...
```

Expected: pass.

- [ ] **Step 2: Run Swift helper tests**

```bash
cd tools/macos-vz-helper && swift test
```

Expected: pass.

- [ ] **Step 3: Run focused Python sandbox tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/sandbox/test_vz_linux_runner.py \
  tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py
```

Expected: pass.

- [ ] **Step 4: Run py_compile on touched Python module**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m py_compile tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py
```

Expected: pass.

- [ ] **Step 5: Run Bandit on touched Python scope**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py -f json -o /tmp/bandit_vz_guest_output_cap.json
```

Expected: no new findings in touched code.

- [ ] **Step 6: Run whitespace check**

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 7: Commit verification-only fixes if needed**

Only commit if verification forced code/doc changes:

```bash
git add <changed-files>
git commit -m "fix(vz): harden guest output cap edge cases"
```

## Task 7: Optional Real Host Smoke

**Files:**
- No implementation files expected.

- [ ] **Step 1: Confirm operator prerequisites**

Confirm an Apple Silicon macOS host with:

- built/signed helper
- rebuilt Debian arm64 image containing updated `tldw-agent-guest`
- owner-only runtime/socket/log directories

- [ ] **Step 2: Run host smoke command**

Use the existing smoke harness after rebuilding the image:

```bash
tools/vz-linux-image/scripts/run-host-e2e-smoke.sh --bundle <rebuilt-bundle-path>
```

Expected: existing ephemeral execution and same-session reuse pass.

- [ ] **Step 3: Add cap-specific manual command if harness lacks it**

If the smoke harness does not yet include a cap-exceeded command, run one manual `vz_linux` sandbox command with a very small `SANDBOX_MAX_LOG_BYTES` and noisy output, then verify:

- exit code is `137`
- `guest_output_limit_exceeded` appears in helper details/resource usage
- a second command in the same session still executes after the capped command

- [ ] **Step 4: Record results in PR**

Do not commit local image artifacts. Put exact command output summary in the PR comment.

## Completion Criteria

- Go guest agent enforces `max_output_bytes` without unbounded output buffers.
- Swift helper forwards `max_output_bytes` and keeps host-side capping as fallback.
- Guest details are guest-prefixed, string-only, and defensively decoded.
- Python runner records guest enforcement counters only as integer `resource_usage` values.
- Protocol/operator docs distinguish rebuilt-image guest enforcement from host fallback.
- Focused Go, Swift, Python, Bandit, py_compile, and `git diff --check` verification complete.
