# VZ Linux Output And Artifact Limit Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound `vz_linux` helper output and artifact capture, then surface truncation/skip facts through existing run metadata and audit paths.

**Architecture:** Add a shared Python limit helper for cap math, artifact collection, and integer-only metadata counters, but wire it only into `VZLinuxRunner` in this PR. Extend the Swift helper `exec_guest` host protocol with an optional `max_output_bytes` field that validates/caps helper-returned stdout/stderr and reports string-encoded details. Emit audit records from `SandboxService` using `RunStatus.resource_usage` counters so runners do not own audit clients.

**Tech Stack:** Python 3.11, pytest, Swift Package tests, macOS helper protocol JSON, existing `UnifiedAuditService`.

---

## Source Spec

- `Docs/superpowers/specs/2026-05-02-vz-output-artifact-audit-design.md`

## File Structure

- Create `tldw_Server_API/app/core/Sandbox/limits.py`
  - Owns shared output/artifact limit helpers.
  - Exposes integer-only metadata counters safe for `RunStatus.resource_usage`.
  - Exposes audit metadata derivation helpers that avoid raw artifact paths.
- Create `tldw_Server_API/tests/sandbox/test_sandbox_limits.py`
  - Unit coverage for cap math and artifact collector behavior.
- Modify `tools/macos-vz-helper/Sources/Server/HelperService.swift`
  - Validate optional `maxOutputBytes`.
  - Cap `GuestExecResult.stdout` / `stderr`.
  - Return string-encoded output counters in helper response details.
- Modify `tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift`
  - Parse optional numeric `max_output_bytes`.
  - Map malformed shape to `invalid_request`.
  - Map semantic invalid cap to `exec_output_limit_invalid`.
- Modify `tools/macos-vz-helper/Sources/Protocol/Response.swift`
  - Keep existing `details: [String: String]`; no response model broadening.
- Modify `tools/macos-vz-helper/Tests/HelperServiceExecTests.swift`
  - Service-level max-output validation and cap tests.
- Modify `tools/macos-vz-helper/Tests/UnixSocketServerTests.swift`
  - Socket-level malformed/invalid `max_output_bytes` tests.
- Modify `tools/macos-vz-helper/PROTOCOL.md`
  - Document `max_output_bytes`, string details, and limits.
- Modify `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
  - TEST_MODE validates `max_output_bytes`.
  - Passes real request through unchanged.
  - Fake response includes helper-style detail counters.
- Modify `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`
  - TEST_MODE and request-shape coverage for `max_output_bytes`.
- Modify `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
  - Pass `SANDBOX_MAX_LOG_BYTES` as helper `max_output_bytes`.
  - Publish output with same cap.
  - Use shared artifact collector and merge integer counters into `resource_usage`.
- Modify `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
  - Runner-level output cap propagation and artifact skip metadata coverage.
- Modify `tldw_Server_API/app/core/Sandbox/policy.py`
  - Add policy-backed artifact cap settings and include them in the canonical policy hash.
- Modify `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
  - Expose artifact cap settings in runtime discovery alongside existing upload/log caps.
- Modify `tldw_Server_API/app/core/Sandbox/service.py`
  - Expose artifact cap settings in feature discovery.
  - Derive completion audit metadata from resource counters.
  - Emit aggregated `output_truncated` and `artifacts_limited` audit events.
- Modify `tldw_Server_API/tests/sandbox/test_policy_hash_determinism.py`
  - Pin new artifact cap settings in deterministic policy-hash setup.
- Modify `tldw_Server_API/tests/sandbox/test_sandbox_api.py`
  - Verify runtime discovery includes artifact cap fields.
- Add or modify focused audit tests under `tldw_Server_API/tests/sandbox/`
  - Verify completion audit metadata includes derived limit fields.
  - Verify separate audit events are aggregate and path-minimized.

## Task 1: Shared Python Limit Helpers

**Files:**
- Create: `tldw_Server_API/app/core/Sandbox/limits.py`
- Create: `tldw_Server_API/tests/sandbox/test_sandbox_limits.py`

- [x] **Step 1: Write failing tests for fair output cap math**

Add tests that describe the desired helper-independent math:

```python
from tldw_Server_API.app.core.Sandbox.limits import cap_output_streams


def test_cap_output_streams_preserves_stderr_when_both_streams_are_large() -> None:
    result = cap_output_streams(b"o" * 100, b"e" * 100, max_output_bytes=10)

    assert len(result.stdout) + len(result.stderr) <= 10
    assert result.stdout
    assert result.stderr
    assert result.counters["stdout_truncated"] == 1
    assert result.counters["stderr_truncated"] == 1
    assert result.counters["stdout_bytes_original"] == 100
    assert result.counters["stderr_bytes_original"] == 100


def test_cap_output_streams_reuses_unused_stream_budget() -> None:
    result = cap_output_streams(b"oo", b"e" * 100, max_output_bytes=10)

    assert result.stdout == b"oo"
    assert result.stderr == b"e" * 8
    assert len(result.stdout) + len(result.stderr) == 10
```

- [x] **Step 2: Run the new test and verify RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/sandbox/test_sandbox_limits.py::test_cap_output_streams_preserves_stderr_when_both_streams_are_large`

Expected: FAIL with `ModuleNotFoundError` or missing `cap_output_streams`.

- [x] **Step 3: Implement `cap_output_streams` minimally**

Create `limits.py` with dataclasses and fair-cap behavior:

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class OutputLimitResult:
    stdout: bytes
    stderr: bytes
    counters: dict[str, int] = field(default_factory=dict)


def cap_output_streams(stdout: bytes, stderr: bytes, *, max_output_bytes: int | None) -> OutputLimitResult:
    stdout_bytes = bytes(stdout or b"")
    stderr_bytes = bytes(stderr or b"")
    original_stdout = len(stdout_bytes)
    original_stderr = len(stderr_bytes)
    if max_output_bytes is None or int(max_output_bytes) <= 0:
        return OutputLimitResult(
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            counters={
                "stdout_bytes_original": original_stdout,
                "stderr_bytes_original": original_stderr,
                "stdout_bytes_returned": original_stdout,
                "stderr_bytes_returned": original_stderr,
                "stdout_truncated": 0,
                "stderr_truncated": 0,
            },
        )

    cap = int(max_output_bytes)
    if original_stdout + original_stderr <= cap:
        return OutputLimitResult(
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            counters={
                "output_limit_bytes": cap,
                "stdout_bytes_original": original_stdout,
                "stderr_bytes_original": original_stderr,
                "stdout_bytes_returned": original_stdout,
                "stderr_bytes_returned": original_stderr,
                "stdout_truncated": 0,
                "stderr_truncated": 0,
            },
        )

    if stdout_bytes and stderr_bytes and cap >= 2:
        stdout_budget = min(original_stdout, max(1, cap // 2))
        stderr_budget = min(original_stderr, max(1, cap - stdout_budget))
        unused = cap - stdout_budget - stderr_budget
        if unused > 0 and original_stdout > stdout_budget:
            extra = min(unused, original_stdout - stdout_budget)
            stdout_budget += extra
            unused -= extra
        if unused > 0 and original_stderr > stderr_budget:
            extra = min(unused, original_stderr - stderr_budget)
            stderr_budget += extra
    elif stdout_bytes:
        stdout_budget = min(original_stdout, cap)
        stderr_budget = 0
    else:
        stdout_budget = 0
        stderr_budget = min(original_stderr, cap)

    returned_stdout = stdout_bytes[:stdout_budget]
    returned_stderr = stderr_bytes[:stderr_budget]
    return OutputLimitResult(
        stdout=returned_stdout,
        stderr=returned_stderr,
        counters={
            "output_limit_bytes": cap,
            "stdout_bytes_original": original_stdout,
            "stderr_bytes_original": original_stderr,
            "stdout_bytes_returned": len(returned_stdout),
            "stderr_bytes_returned": len(returned_stderr),
            "stdout_truncated": int(len(returned_stdout) < original_stdout),
            "stderr_truncated": int(len(returned_stderr) < original_stderr),
        },
    )
```

- [x] **Step 4: Run the output helper tests and verify GREEN**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/sandbox/test_sandbox_limits.py`

Expected: PASS for current tests.

- [x] **Step 5: Write failing tests for artifact skip behavior**

Add tests that create a workspace containing:

- one matching small artifact
- one matching file over per-file cap
- one matching file that would exceed total cap
- one symlink that should be skipped

Expected behavior:

```python
result = collect_limited_artifacts(
    str(workspace),
    ["*.txt"],
    max_file_bytes=5,
    max_total_bytes=8,
)

assert result.artifacts == {"small.txt": b"1234"}
assert result.counters["artifact_files_collected"] == 1
assert result.counters["artifact_files_skipped"] >= 2
assert result.counters["artifact_skip_file_limit"] == 1
assert result.counters["artifact_skip_total_limit"] == 1
assert result.counters["artifact_bytes_collected"] == 4
```

- [x] **Step 6: Run the artifact tests and verify RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/sandbox/test_sandbox_limits.py`

Expected: FAIL with missing `collect_limited_artifacts`.

- [x] **Step 7: Implement `collect_limited_artifacts`**

Add:

- `ArtifactLimitResult(artifacts: dict[str, bytes], counters: dict[str, int])`
- workspace root symlink rejection
- `Path.resolve(strict=False)` root containment checks
- symlink skip
- glob match through `fnmatch.fnmatchcase`
- `Path.stat().st_size` pre-read checks
- read only after cap checks pass
- counters only, no artifact path names in counters

Use defaults supplied by caller; do not read app settings inside this helper.

- [x] **Step 8: Run helper tests and verify GREEN**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/sandbox/test_sandbox_limits.py`

Expected: PASS.

- [x] **Step 9: Commit Task 1**

```bash
git add tldw_Server_API/app/core/Sandbox/limits.py tldw_Server_API/tests/sandbox/test_sandbox_limits.py
git commit -m "feat(sandbox): add shared output artifact limits"
```

## Task 2: Swift Helper `max_output_bytes` Contract

**Files:**
- Modify: `tools/macos-vz-helper/Sources/Server/HelperService.swift`
- Modify: `tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift`
- Modify: `tools/macos-vz-helper/Sources/Protocol/Response.swift`
- Modify: `tools/macos-vz-helper/Tests/HelperServiceExecTests.swift`
- Modify: `tools/macos-vz-helper/Tests/UnixSocketServerTests.swift`
- Modify: `tools/macos-vz-helper/PROTOCOL.md`

- [x] **Step 1: Write failing service-level cap tests**

In `HelperServiceExecTests.swift`, add a guest bridge that returns large stdout
and stderr. Assert:

- combined returned bytes do not exceed cap
- both stdout and stderr remain non-empty when cap >= 2
- multibyte UTF-8 output is truncated to a valid prefix whose re-encoded byte
  count stays within budget
- response details include string counters
- invalid cap throws `HelperServiceError.invalidExecOutputLimit("output_limit_out_of_range")`

- [x] **Step 2: Run service tests and verify RED**

Run: `swift test --package-path tools/macos-vz-helper --filter HelperServiceExecTests`

Expected: FAIL because `execGuest` does not accept `maxOutputBytes` and no error case exists.

- [x] **Step 3: Implement service contract**

In `HelperServiceError`, add:

```swift
case invalidExecOutputLimit(String)
```

Change `execGuest` signature:

```swift
func execGuest(
    vmID: String,
    argv: [String],
    cwd: String,
    env: [String: String],
    timeoutSeconds: TimeInterval,
    maxOutputBytes: Int? = nil
) throws -> HelperExecResponse
```

Add validation:

- `nil` means no helper-side cap.
- `1...268_435_456` is accepted.
- `<=0` or above max throws `invalidExecOutputLimit("output_limit_out_of_range")`.
- Keep this helper protocol ceiling mirrored in Python as `_MAX_EXEC_OUTPUT_BYTES`
  so TEST_MODE and real helper requests fail the same way.

Add a Swift helper that caps UTF-8 output strings by byte budget. Since current
`GuestExecResult` is string-based, cap on `Data(result.stdout.utf8)`, but only
return a valid UTF-8 prefix. Do not use replacement-character decoding for the
truncated prefix because that can re-encode to more bytes than the requested
budget. Count returned bytes from `Data(returnedString.utf8)` after truncation.
Record details as strings:

```swift
[
  "transport": "vsock",
  "vm_id": vmID,
  "output_limit_bytes": "\(cap)",
  "stdout_bytes_original": "\(stdoutOriginal)",
  "stderr_bytes_original": "\(stderrOriginal)",
  "stdout_bytes_returned": "\(stdoutReturned)",
  "stderr_bytes_returned": "\(stderrReturned)",
  "stdout_truncated": stdoutTruncated ? "true" : "false",
  "stderr_truncated": stderrTruncated ? "true" : "false",
]
```

- [x] **Step 4: Wire socket parsing**

In `UnixSocketServer.swift`:

- parse missing `max_output_bytes` as `nil`
- accept JSON int only
- reject strings/arrays/objects with `invalid_request`
- pass parsed value to `service.execGuest`
- map `HelperServiceError.invalidExecOutputLimit` to `exec_output_limit_invalid`
- return associated reason as message

- [x] **Step 5: Add socket tests**

In `UnixSocketServerTests.swift`, add:

- valid `max_output_bytes` returns capped output and details
- `max_output_bytes: "10"` returns `invalid_request`
- `max_output_bytes: 0` returns `exec_output_limit_invalid`

- [x] **Step 6: Run Swift tests and verify GREEN**

Run:

```bash
swift test --package-path tools/macos-vz-helper --filter HelperServiceExecTests
swift test --package-path tools/macos-vz-helper --filter UnixSocketServerTests
```

Expected: PASS.

- [x] **Step 7: Update protocol docs**

In `tools/macos-vz-helper/PROTOCOL.md`, update `exec_guest` request and
response details:

- `max_output_bytes` optional integer
- details counters are strings
- invalid shape/semantic error codes
- note that this is a host response cap, not guest-agent kill-on-cap

- [x] **Step 8: Commit Task 2**

```bash
git add tools/macos-vz-helper/Sources/Server/HelperService.swift tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift tools/macos-vz-helper/Sources/Protocol/Response.swift tools/macos-vz-helper/Tests/HelperServiceExecTests.swift tools/macos-vz-helper/Tests/UnixSocketServerTests.swift tools/macos-vz-helper/PROTOCOL.md
git commit -m "feat(sandbox): cap vz helper exec output"
```

## Task 3: Python Helper Client, Policy Settings, And VZ Runner Wiring

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`
- Modify: `tldw_Server_API/app/core/Sandbox/policy.py`
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- Modify: `tldw_Server_API/tests/sandbox/test_policy_hash_determinism.py`
- Modify: `tldw_Server_API/tests/sandbox/test_sandbox_api.py`
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`

- [x] **Step 1: Write failing Python helper-client tests**

Add tests that:

- TEST_MODE `exec_guest(..., {"argv": ["/bin/echo", "ok"], "max_output_bytes": 2})` returns capped stdout and detail counters.
- TEST_MODE rejects `max_output_bytes=0` with `exec_output_limit_invalid`.
- real transport request includes `max_output_bytes` unchanged.

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`

Expected: FAIL before implementation.

- [x] **Step 2: Implement helper-client validation and fake details**

In `helper_client.py`:

- add max-output validation to `_validate_exec_guest_request`
- reject bool/string/list/dict as `invalid_request`
- reject `<=0` or values above `_MAX_EXEC_OUTPUT_BYTES` as `exec_output_limit_invalid`
- use `cap_output_streams` in TEST_MODE fake replies
- include helper-style detail counters as strings

- [x] **Step 3: Run helper-client tests and verify GREEN**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`

Expected: PASS.

- [x] **Step 4: Write failing policy and runtime-discovery tests**

In `test_policy_hash_determinism.py`, pin:

```python
monkeypatch.setenv("SANDBOX_MAX_ARTIFACT_FILE_BYTES", str(64 * 1024 * 1024))
monkeypatch.setenv("SANDBOX_MAX_ARTIFACT_TOTAL_BYTES", str(256 * 1024 * 1024))
```

In `test_sandbox_api.py`, extend runtime discovery expectations to include:

```python
"max_artifact_file_bytes",
"max_artifact_total_bytes",
```

Add a focused policy unit assertion if needed:

```python
cfg = SandboxPolicyConfig.from_settings()
assert cfg.max_artifact_file_bytes == 64 * 1024 * 1024
assert cfg.max_artifact_total_bytes == 256 * 1024 * 1024
```

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/sandbox/test_sandbox_api.py::test_runtimes_discovery_shape tldw_Server_API/tests/sandbox/test_policy_hash_determinism.py`

Expected: FAIL because the schema/discovery/policy fields do not exist yet.

- [x] **Step 5: Implement policy-backed artifact cap settings**

In `policy.py`:

- add `max_artifact_file_bytes: int = 64 * 1024 * 1024`
- add `max_artifact_total_bytes: int = 256 * 1024 * 1024`
- read `SANDBOX_MAX_ARTIFACT_FILE_BYTES` and `SANDBOX_MAX_ARTIFACT_TOTAL_BYTES` in `from_settings`
- clamp non-positive or malformed values to the documented defaults
- include both values in `_canonical_policy_dict`

In `sandbox_schemas.py`, add optional integer fields to the runtime discovery schema.

In `SandboxService.feature_discovery`, include both fields for each advertised runtime.

- [x] **Step 6: Run policy and runtime-discovery tests and verify GREEN**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/sandbox/test_sandbox_api.py::test_runtimes_discovery_shape tldw_Server_API/tests/sandbox/test_policy_hash_determinism.py`

Expected: PASS.

- [x] **Step 7: Write failing VZ runner tests**

In `test_vz_linux_runner.py`, add tests that:

- monkeypatch `SANDBOX_MAX_LOG_BYTES=5`
- monkeypatch `SANDBOX_MAX_ARTIFACT_FILE_BYTES=5`
- monkeypatch `SANDBOX_MAX_ARTIFACT_TOTAL_BYTES=8`
- fake helper captures `exec_guest` request and asserts `max_output_bytes == 5`
- returned helper details produce integer counters in `status.resource_usage`
- artifact collection skips oversized files and keeps status completed

Expected example assertions:

```python
assert status.resource_usage["output_limit_bytes"] == 5
assert status.resource_usage["stdout_truncated"] == 1
assert status.resource_usage["artifact_files_skipped"] == 1
assert status.phase == RunPhase.completed
```

- [x] **Step 8: Implement VZ runner wiring**

In `vz_linux_runner.py`:

- add `_max_log_bytes()` and artifact cap accessors that reuse `SandboxPolicyConfig.from_settings()` or the same settings defaults:
  - `SANDBOX_MAX_LOG_BYTES`, default `10 * 1024 * 1024`
  - `SANDBOX_MAX_ARTIFACT_FILE_BYTES`, default `64 * 1024 * 1024`
  - `SANDBOX_MAX_ARTIFACT_TOTAL_BYTES`, default `256 * 1024 * 1024`
- sanitize non-positive or malformed configured caps back to the documented
  defaults before passing them to helper or artifact collection
- pass `max_output_bytes=max_log_bytes` in `exec_guest` request
- publish stdout/stderr with `max_log_bytes=max_log_bytes`
- replace `_collect_artifacts` internals or delegate to `collect_limited_artifacts`
- merge output/artifact counters into integer-only `usage`
- keep existing `log_bytes` and `artifact_bytes` keys stable

- [x] **Step 9: Run VZ runner tests and verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/sandbox/test_vz_linux_runner.py \
  tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py \
  tldw_Server_API/tests/sandbox/test_sandbox_limits.py \
  tldw_Server_API/tests/sandbox/test_sandbox_api.py::test_runtimes_discovery_shape \
  tldw_Server_API/tests/sandbox/test_policy_hash_determinism.py
```

Expected: PASS.

- [x] **Step 10: Commit Task 3**

```bash
git add tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py tldw_Server_API/app/core/Sandbox/policy.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/tests/sandbox/test_policy_hash_determinism.py tldw_Server_API/tests/sandbox/test_sandbox_api.py tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py
git commit -m "feat(sandbox): wire vz output artifact limits"
```

## Task 4: Audit Metadata And Aggregated Limit Events

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/limits.py`
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Add or modify: `tldw_Server_API/tests/sandbox/test_sandbox_run_limit_audit.py`

- [x] **Step 1: Write failing audit metadata helper tests**

In `test_sandbox_limits.py` or a new audit test file, verify:

```python
metadata = build_limit_audit_metadata({
    "output_limit_bytes": 5,
    "stdout_truncated": 1,
    "stderr_truncated": 0,
    "artifact_files_skipped": 2,
    "artifact_skip_file_limit": 1,
    "artifact_skip_total_limit": 1,
})

assert metadata["output_truncated"] is True
assert metadata["artifact_skip_reasons"] == ["file_limit", "total_limit"]
assert "artifact_paths" not in metadata
```

Expected: FAIL before helper exists.

- [x] **Step 2: Implement audit metadata derivation**

In `limits.py`, add:

- `build_limit_audit_metadata(resource_usage: Mapping[str, object]) -> dict[str, object]`
- `limit_event_actions(resource_usage: Mapping[str, object]) -> list[str]`

Keep input tolerant and output path-minimized.

- [x] **Step 3: Run helper audit tests and verify GREEN**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/sandbox/test_sandbox_limits.py`

Expected: PASS.

- [x] **Step 4: Write failing `SandboxService` audit tests**

Use monkeypatching to replace `UnifiedAuditService` with a fake collector. Call
`SandboxService._audit_run_completion(...)` with a `RunStatus` containing
resource counters. Assert three events:

- existing completion action `run`
- aggregate action `output_truncated` when output counters indicate truncation
- aggregate action `artifacts_limited` when artifact skip counters are non-zero

Assert metadata does not include artifact path names.

- [x] **Step 5: Implement service audit emission**

In `_audit_run_completion`:

- import `build_limit_audit_metadata` and `limit_event_actions`
- merge derived metadata into existing completion metadata
- after completion event, emit separate events using:
  - `event_type=AuditEventType.API_RESPONSE`
  - `category=AuditEventCategory.API_CALL`
  - `severity=AuditSeverity.WARNING`
  - `resource_type="sandbox.run"`
  - `resource_id=run_id`
  - `action` from `limit_event_actions`
  - `result="limited"`
  - same `AuditContext`
  - same derived metadata

Do not add new audit enum values in this slice.

- [x] **Step 6: Run service audit tests and verify GREEN**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/sandbox/test_sandbox_run_limit_audit.py tldw_Server_API/tests/sandbox/test_sandbox_limits.py`

Expected: PASS.

- [x] **Step 7: Commit Task 4**

```bash
git add tldw_Server_API/app/core/Sandbox/limits.py tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/tests/sandbox/test_sandbox_limits.py tldw_Server_API/tests/sandbox/test_sandbox_run_limit_audit.py
git commit -m "feat(sandbox): audit vz limit outcomes"
```

## Task 5: Docs, Full Verification, And PR Prep

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify: `Docs/superpowers/specs/2026-05-02-vz-output-artifact-audit-design.md` only if implementation needs a justified design clarification.

- [x] **Step 1: Update sandbox docs**

Document:

- `SANDBOX_MAX_LOG_BYTES` now bounds `vz_linux` helper-returned output and WebSocket publication.
- `SANDBOX_MAX_ARTIFACT_FILE_BYTES` and `SANDBOX_MAX_ARTIFACT_TOTAL_BYTES` bound `vz_linux` artifact capture.
- helper-side cap is not guest-agent kill-on-cap.

- [x] **Step 2: Run focused Swift verification**

Run:

```bash
swift test --package-path tools/macos-vz-helper --filter HelperServiceExecTests
swift test --package-path tools/macos-vz-helper --filter UnixSocketServerTests
swift test --package-path tools/macos-vz-helper
```

Expected: PASS.

- [x] **Step 3: Run focused Python verification**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/sandbox/test_sandbox_limits.py \
  tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_runner.py \
  tldw_Server_API/tests/sandbox/test_sandbox_run_limit_audit.py \
  tldw_Server_API/tests/sandbox/test_sandbox_api.py::test_runtimes_discovery_shape \
  tldw_Server_API/tests/sandbox/test_policy_hash_determinism.py \
  tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py \
  tldw_Server_API/tests/sandbox/test_macos_runtime_admission.py
```

Expected: PASS, with any known host-gated skip preserved.

- [x] **Step 4: Run syntax and security checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile \
  tldw_Server_API/app/core/Sandbox/limits.py \
  tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py \
  tldw_Server_API/app/core/Sandbox/policy.py \
  tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py

/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/Sandbox/limits.py \
  tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py \
  tldw_Server_API/app/core/Sandbox/policy.py \
  tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  -f json -o /tmp/bandit_vz_output_artifact_audit.json

git diff --check
```

Expected:

- `py_compile` exits 0.
- Bandit has no new findings in touched files.
- `git diff --check` exits 0.

- [x] **Step 5: Commit docs and final verification notes**

```bash
git add tldw_Server_API/app/core/Sandbox/README.md Docs/superpowers/plans/2026-05-02-vz-output-artifact-audit-implementation-plan.md
git commit -m "docs(sandbox): document vz output artifact limits"
```

- [x] **Step 6: Prepare PR summary**

PR body should include:

- Summary bullets for helper output cap, artifact cap, audit events.
- Human-written `Change summary` placeholder.
- Exact verification commands and outcomes.
- Explicit note that guest-agent streaming/kill-on-cap remains follow-up.

## Final Acceptance Criteria

- `vz_linux` passes `max_output_bytes` to helper `exec_guest`.
- Swift helper rejects malformed/invalid `max_output_bytes`.
- Swift helper caps returned stdout/stderr and reports string detail counters.
- Python TEST_MODE mirrors validation and detail counters.
- Artifact byte caps are policy-backed and visible in runtime discovery.
- `VZLinuxRunner` publishes output with `SANDBOX_MAX_LOG_BYTES`.
- `VZLinuxRunner` skips oversized artifacts without failing successful runs.
- `RunStatus.resource_usage` remains `dict[str, int]` compatible.
- Completion audit metadata includes derived output/artifact limit facts.
- Separate aggregate audit actions are emitted only when limits affect the run.
- No raw artifact paths are added to audit metadata.
- Focused Swift/Python tests, `py_compile`, Bandit, and `git diff --check` pass.
