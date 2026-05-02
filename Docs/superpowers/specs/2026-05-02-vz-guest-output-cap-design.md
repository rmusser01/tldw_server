# VZ Linux Guest Output Cap Design

## Status

Design approved for specification. Implementation is not started in this
document.

## Context

The `vz_linux` runtime now passes `max_output_bytes` through the Python helper
client and the Swift helper caps the helper response before returning it to
Python. That protects the Python boundary and WebSocket publication path, but it
does not protect the guest agent or the Swift bridge if a guest command writes
large stdout/stderr before the helper response cap is applied.

The in-repo guest agent lives under `tools/tldw-agent` and is installed into the
Debian image by `tools/vz-linux-image/scripts/install-agent.sh`. The current
guest exec path uses unbounded `bytes.Buffer` values for stdout and stderr. This
slice should move the output limit enforcement closer to the process that
produces output, while preserving the current request/response protocol shape.

This work follows `Docs/Sandbox/sandbox-architecture-doctrine.md`: Python owns
policy selection and public API behavior, the Swift helper owns the host control
boundary, and the guest agent owns guest-local execution mechanics.

## Goals

- Add guest-side enforcement for the existing `exec_guest.max_output_bytes`
  request policy.
- Forward `max_output_bytes` from the Swift helper to the guest agent over the
  existing vsock request/response protocol.
- Bound guest stdout/stderr retention so the guest agent does not accumulate
  unbounded output in memory.
- Terminate the guest process when the combined stdout/stderr observation
  exceeds the configured cap.
- Preserve Swift helper-side response capping as defense in depth.
- Surface guest enforcement metadata so operators can distinguish guest-side
  limit enforcement from host-side fallback truncation.
- Update protocol documentation and focused tests across Go, Swift, and Python.

## Non-Goals

- Do not implement incremental stdout/stderr streaming in this PR.
- Do not change the public sandbox API shape.
- Do not remove Swift helper-side response capping.
- Do not change `docker`, `seatbelt`, `worktree`, `firecracker`, or `lima`
  output semantics.
- Do not make old VM images magically enforce guest-side caps; old guest agents
  may ignore the new request field.

## Chosen Approach

Extend the existing request/response protocol with an optional
`max_output_bytes` field on guest `exec` requests. The Swift helper already
validates `exec_guest.max_output_bytes`; after validation, it should pass the
same value into `VZLinuxVMManager.execGuest`, `GuestBridging.exec`, and the
encoded `GuestExecRequest`.

The Go guest agent should add `MaxOutputBytes *int` to `ExecRequest` and
replace the current `cmd.Run()` plus unbounded stdout/stderr buffers with
explicit `StdoutPipe` / `StderrPipe` readers and a combined bounded capture
mechanism. A pointer is required so the guest can distinguish an omitted cap
from an explicit invalid `0`. The bounded capture should:

- keep at most `max_output_bytes` returned bytes across stdout and stderr
- track observed byte counts separately for stdout and stderr
- track returned byte counts separately for stdout and stderr
- mark whether the cap was exceeded
- be safe for concurrent stdout and stderr writes
- record output-cap cancellation separately from timeout cancellation
- cancel the command context when the cap is exceeded
- terminate the process group when supported, not just the direct child process
- return UTF-8-safe strings in the existing `stdout` and `stderr` fields

The guest response should remain a single final response. It should add optional
detail fields, not a stream frame protocol. If the cap is exceeded, the guest
agent should return a normal exec response with a stable non-zero exit code when
the process is terminated by the cap, plus metadata indicating
`guest_output_limit_exceeded=true`. The host should avoid turning this into a
helper transport failure; it is a command result constrained by policy.

Swift helper response capping remains active. If a rebuilt guest agent enforces
the cap, Swift details should report both guest-observed metadata and any
host-side truncation metadata. If an old guest agent ignores `max_output_bytes`,
Swift still caps the final response and details should make clear that only
host-side truncation was observed.

Guest-provided output counters must use guest-prefixed detail keys so they do
not overwrite the existing host-side helper cap keys. The existing host keys
such as `stdout_bytes_original`, `stderr_bytes_original`,
`stdout_bytes_returned`, and `stderr_bytes_returned` continue to describe the
bytes Swift received and returned after host-side capping. Guest keys describe
what the guest agent observed before it terminated or completed the process.

## Protocol Contract

Host helper `exec_guest` request semantics remain unchanged externally:

- `max_output_bytes` is optional.
- Valid values are JSON integers in the existing range `1...268435456`.
- Invalid shape returns `invalid_request`.
- Invalid semantic values return `exec_output_limit_invalid`.

Guest `exec` request gains:

```json
{
  "protocol_version": "1",
  "request_id": "req-1",
  "type": "exec",
  "argv": ["/bin/sh", "-c", "yes"],
  "cwd": "/workspace",
  "env": {},
  "timeout_sec": 30,
  "max_output_bytes": 1048576
}
```

Guest `exec` response remains:

```json
{
  "protocol_version": "1",
  "request_id": "req-1",
  "exit_code": 137,
  "stdout": "bounded prefix...",
  "stderr": "",
  "details": {
    "guest_output_limit_bytes": "1048576",
    "guest_output_limit_exceeded": "true",
    "guest_output_kill_reason": "output_limit",
    "guest_stdout_bytes_observed": "1048577",
    "guest_stderr_bytes_observed": "0",
    "guest_stdout_bytes_returned": "1048576",
    "guest_stderr_bytes_returned": "0"
  }
}
```

Detail values should remain strings for consistency with the Swift helper
details map and existing Python parsing behavior. Boolean values should be
encoded as `"true"` or `"false"` and parsed defensively by Python into integer
resource usage counters where needed.

When the guest agent terminates a command because of output cap enforcement, it
should normalize the command result to exit code `137` in the response instead
of leaking platform-specific `os/exec` signal behavior such as `-1`. The
metadata remains the authoritative reason for the termination.

## Output Semantics

When no `max_output_bytes` is provided, the guest agent keeps existing exec
behavior and does not add a new implicit guest kill-on-cap default. This avoids
changing lower-level guest protocol semantics for callers that do not send the
field. The `vz_linux` Python runner is expected to send the sandbox log cap for
normal sandbox execution.

When a cap is provided:

- The cap applies to combined stdout and stderr bytes.
- The guest agent should kill/cancel the command as soon as the combined
  observed bytes exceed the cap.
- The returned stdout/stderr bytes must not exceed the cap.
- Observed byte counts may exceed returned byte counts by at least one byte
  because cap detection happens while reading chunks.
- Stdout and stderr capture must coordinate through shared state because
  `os/exec` may write both streams concurrently.
- The bounded capture helper must store the first cancellation reason so timeout
  and output-cap paths do not misclassify each other.
- The implementation should use `cmd.Start()`, concurrent pipe readers, and
  `cmd.Wait()` rather than relying on `io.Writer` errors from `cmd.Run()` to
  control process lifetime.
- The guest should preserve stderr diagnostics when both streams produce output
  where practical, but exact fair-share truncation can remain host-side in this
  PR if guest-side chunk interleaving makes deterministic fair sharing costly.
- Returned strings must not contain partial UTF-8 sequences. Invalid UTF-8 from
  guest commands should be decoded with replacement rather than crashing the
  agent, but the final UTF-8 byte length after sanitization must still obey the
  cap.

The policy outcome is reported as metadata, not as a transport error. A command
terminated for output cap should complete the `exec_guest` request and let the
runner publish a failed run according to the command exit code and metadata.

## Backward Compatibility

Old guest agents will ignore `max_output_bytes` and may still buffer large
stdout/stderr before responding. The PR must therefore preserve all existing
host-side caps and must not claim full guest-side protection unless the response
contains guest enforcement metadata.

The Swift helper should parse response details defensively. Missing details mean
guest enforcement is unknown or absent. The Python runner should carry enough
resource usage metadata for diagnostics and audit to distinguish:

- guest cap enforced
- host response cap applied
- host response cap applied because guest enforcement metadata was missing

Image-store and operator docs should state that guest kill-on-cap requires
rebuilt images containing the updated `tldw-agent-guest` binary.

## Component Changes

### Go guest agent

- Add `MaxOutputBytes *int` and optional `Details map[string]string` to guest
  exec request/response types.
- Validate direct guest requests defensively with the same effective ceiling as
  the host helper, currently 256 MiB.
- Replace unbounded `bytes.Buffer` usage in guest exec with bounded stream
  capture driven by stdout/stderr pipes.
- Cancel command execution and terminate the process group when combined
  observed bytes exceed the cap.
- Implement process-group setup and termination behind a small platform helper
  so Go packages remain portable where possible. Linux guests should use process
  group termination; unsupported platforms may fall back to direct process
  termination.
- Add focused unit tests for bounded output, process termination, timeout
  precedence, UTF-8-safe returned output, and no-cap compatibility.

### Swift helper

- Extend `GuestExecResult` with optional detail metadata.
- Extend `GuestBridging.exec` and `VZLinuxVMManager.execGuest` to accept
  `maxOutputBytes`.
- Encode `max_output_bytes` in `GuestExecRequest` when provided.
- Decode optional guest response details and merge only string-valued,
  guest-prefixed keys into helper response details.
- Decode guest details defensively so missing, malformed, or non-string detail
  values do not fail an otherwise valid exec response.
- Preserve host-side output counter keys for Swift-received and Swift-returned
  byte counts; do not overwrite them with guest-observed counters.
- Keep existing helper-side `capExecOutput` behavior as fallback.

### Python sandbox runtime

- Keep sending `max_output_bytes` from `VZLinuxRunner` through
  `MacOSVirtualizationHelperClient.exec_guest`.
- Parse new helper detail keys defensively into integer `resource_usage`
  counters, including `guest_output_limit_bytes`,
  `guest_output_limit_exceeded`, `guest_stdout_bytes_observed`,
  `guest_stderr_bytes_observed`, `guest_stdout_bytes_returned`, and
  `guest_stderr_bytes_returned`.
- Preserve existing host-side output and audit metadata.
- Add tests only where Python behavior changes; avoid duplicating Go/Swift
  unit coverage in Python.

### Documentation

- Update `tools/macos-vz-helper/PROTOCOL.md` to remove the statement that
  `max_output_bytes` is host-response-only once guest forwarding is implemented.
- Update `tools/tldw-agent/docs/vz-linux-guest-protocol.md` with the new guest
  request field and response details.
- Update `tools/vz-linux-image/README.md` to note that images must be rebuilt to
  gain guest-side kill-on-cap behavior.

## Error Handling

- Malformed host-side `max_output_bytes` handling remains unchanged.
- Guest-side invalid or non-positive caps should return `invalid_request` if a
  malformed request reaches the guest agent despite host validation.
- Output cap termination should not be represented as a guest protocol error.
- Output cap termination should normalize the response exit code to `137` and
  set `guest_output_limit_exceeded=true`.
- Timeout should remain distinct from output cap termination. If both happen,
  whichever cancellation reason is observed first should determine the primary
  metadata reason, and tests should document that precedence.
- If the guest agent fails to kill a capped process promptly, the existing
  timeout path remains the fallback.

## Testing Strategy

Go guest-agent tests:

- `Exec` with no cap preserves current stdout/stderr behavior.
- `Exec` with cap below command output returns no more than the cap.
- `Exec` with cap below command output cancels a long-running/noisy process.
- `Exec` metadata reports observed, returned, limit, and exceeded fields.
- `Exec` kills or cleans up child processes through the platform process-group
  helper on Linux.
- `Exec` returns UTF-8-safe output when truncating multibyte text.
- `Exec` rejects non-positive or excessive direct guest caps.
- `Exec` timeout still reports `timeout_exceeded` when timeout wins.

Swift helper tests:

- `VSockBridge` encodes `max_output_bytes` into guest exec requests.
- `VSockBridge` decodes optional guest details.
- `VSockBridge` ignores non-string or malformed guest details without failing
  the exec response.
- `HelperService.execGuest` passes the cap to the bridge/manager.
- `HelperService.execGuest` merges guest-prefixed metadata and still applies
  host cap without overwriting host output counters.
- Socket-level `exec_guest` behavior remains backward compatible when guest
  details are absent.

Python tests:

- `MacOSVirtualizationHelperClient` accepts and exposes new detail counters
  defensively.
- `VZLinuxRunner` resource usage includes guest enforcement counters when
  helper details provide them.
- `VZLinuxRunner` preserves existing host-side output counters when guest
  enforcement counters are absent.
- Existing host cap tests still pass when guest enforcement details are absent.

Host-gated smoke follow-up:

- After unit/integration coverage passes, rebuild the Debian arm64 image and run
  a real `vz_linux` command that exceeds a small cap to verify the guest process
  is terminated and session reuse still works afterward.

## Risks And Review Notes

- The guest protocol version remains `1`, so new fields must be optional and
  old-agent tolerant.
- The strongest safety guarantee requires rebuilt images. Host diagnostics
  should not imply guest enforcement for stale images.
- Killing a process on cap can leave child processes if the command spawns its
  own process tree and the platform process-group helper is unavailable. Linux
  guest images should support process-group termination; unsupported platforms
  should be explicitly documented as weaker.
- Pipe readers must not deadlock when one stream hits the cap while the other is
  still being copied. The implementation should cancel once, close/kill the
  process, wait for both reader goroutines, then call `cmd.Wait()`.
- Exact fair sharing between stdout and stderr is harder inside a live bounded
  writer than after both streams are buffered. Preserve host-side fair sharing
  and keep guest-side behavior simple unless tests show stderr starvation is a
  practical issue.
- The details schema should stay integer/string-only so existing helper-client
  parsing and `RunStatus.resource_usage` constraints remain stable.

## Follow-Up Work

- Add incremental guest stdout/stderr streaming over vsock.
- Add guest-agent version/capability reporting to helper diagnostics so stale
  images are obvious before execution.
- Apply similar guest-side cap behavior to future VM runtimes.
