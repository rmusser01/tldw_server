# VZ Helper Lifecycle Hardening Design

**Date:** 2026-04-29
**Status:** Approved for implementation planning
**Scope:** `tools/macos-vz-helper/`, `tools/vz-linux-image/scripts/`, sandbox macOS operator docs, and portable helper lifecycle tests

## Summary

The next sandbox slice should make the `vz_linux` macOS helper lifecycle repeatable for local operators without turning helper management into hidden automation.

The merged runtime path now has real helper-backed `vz_linux` boot, guest command execution, session VM reuse, image-store manifests, host E2E smoke, and admin reconciliation/repair for Python-owned session-control rows. The remaining operational gap is that the helper is still started manually through ad hoc environment variables and temporary paths. This makes failures harder to diagnose and makes future work, especially host-gated CI, orphan VM cleanup, APFS clones, and `vz_macos`, more brittle.

This PR should add a repo-owned helper management command with safe defaults, explicit checks, deterministic paths, signing/entitlements validation, helper protocol compatibility checks, and dry-run launchd plist generation. It should not install or upgrade services automatically.

## Source Documents

- `Docs/Sandbox/sandbox-architecture-doctrine.md`
- `Docs/Design/2026-04-27-vz-linux-operator-image-store-design.md`
- `Docs/superpowers/specs/2026-04-27-vz-linux-lifecycle-recovery-hardening-design.md`
- `Docs/Sandbox/macos-runtime-operator-notes.md`
- `tldw_Server_API/app/core/Sandbox/README.md`
- `tools/macos-vz-helper/README.md`
- `tools/macos-vz-helper/PROTOCOL.md`
- `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`

The active doctrine says the Python sandbox service owns policy, sessions, runs, artifacts, queueing, and API behavior. The native helper owns live VM state, host readiness, template validation, guest transport health, and runtime facts. This design keeps that boundary intact.

## Current State

The Swift helper currently:

- starts from `tools/macos-vz-helper`
- reads `TLDW_SANDBOX_MACOS_HELPER_SOCKET`
- creates a Unix socket server at that path
- unlinks the socket path on start and stop
- exposes `ping`, `validate_host`, `validate_template`, `create_vm`, `exec_guest`, `get_vm_status`, `list_vms`, and `terminate_vm`
- reports `protocol_version` and `helper_version` in successful and failure responses

The operator smoke script currently:

- builds the helper when needed
- optionally ad hoc signs it when entitlements are provided
- starts the helper with a socket and serial-log directory
- runs the helper daemon smoke
- runs the real `vz_linux` host E2E tests for ephemeral execution and same-session reuse
- stops the helper on exit

The gaps are:

- no standalone helper lifecycle command for `check`, `start`, `status`, `stop`, or plist generation
- no stable default socket, pid, or log path contract
- no reusable path-safety checks for sockets, pid files, or log directories
- no operator-facing signing/entitlements status check
- no command that verifies helper protocol compatibility before starting E2E
- no generated launchd plist scaffold
- no clear upgrade/version-check behavior

## Goals

1. Provide one repo-owned operator command for helper lifecycle management.
2. Make helper paths deterministic and overrideable.
3. Validate helper binary, signing state, entitlements inputs, socket path safety, pid ownership, and helper protocol compatibility.
4. Add dry-run/default-safe launchd plist generation and validation.
5. Integrate with the existing host E2E smoke flow instead of duplicating it.
6. Keep the helper lifecycle explicit and operator-controlled.
7. Improve docs so future host-gated CI and runtime work can depend on one workflow.

## Non-Goals

1. Automatic launchd installation or service loading by default.
2. Full launchd uninstall management.
3. Helper auto-upgrade.
4. Orphan VM termination.
5. APFS clone provisioning.
6. `vz_macos` real execution.
7. Changing Python sandbox runtime ownership or making the helper a session store.
8. Replacing the existing host E2E smoke script.

## Recommended Approach

Add `tools/macos-vz-helper/scripts/vz-helperctl.py` as the operator entrypoint.

The command should be portable enough for unit tests on non-macOS hosts, while macOS-specific checks should degrade into explicit unsupported or skipped statuses when the required host tools are unavailable.

Recommended subcommands:

- `check`
- `build`
- `sign`
- `start`
- `status`
- `stop`
- `plist`
- `smoke`

The command should default to dry-run behavior for launchd plist generation and should never call `launchctl bootstrap`, `launchctl bootout`, or write into `~/Library/LaunchAgents` unless an operator explicitly requests that in a future PR.

## Path And Socket Contract

Use deterministic user-owned defaults:

```text
~/Library/Application Support/tldw/sandbox/macos-vz-helper/helper.sock
~/Library/Application Support/tldw/sandbox/macos-vz-helper/helper.pid
~/Library/Logs/tldw/macos-vz-helper/
~/Library/LaunchAgents/org.tldw.macos-vz-helper.plist
```

All paths should be overrideable by flags:

- `--socket`
- `--pid-file`
- `--log-dir`
- `--helper`
- `--entitlements`
- `--plist-output`

Path safety rules:

- refuse empty socket paths
- refuse socket paths whose parent cannot be created safely
- refuse symlink socket paths
- refuse existing non-socket files at the socket path
- allow removing a stale socket only when it is actually a Unix socket
- refuse pid files whose process is alive but not the expected helper binary
- create runtime and log directories with owner-only write expectations where possible
- do not chmod arbitrary existing parent directories

The helper command and the Swift helper must both enforce socket safety. The
wrapper check is not sufficient because operators may launch the helper directly
or through a generated plist. The helper startup path should use `lstat` before
unlinking the configured socket path:

- refuse symlink socket paths
- refuse existing non-socket files
- remove an existing path only when it is a Unix socket
- surface a clear startup error instead of silently unlinking unsafe paths

Directory ownership and permissions are part of the lifecycle contract. The
default runtime directory should be owner-only, preferably `0700`, so the socket
does not become a local multi-user control surface. The implementation may rely
on an owner-only parent directory instead of a `0600` socket mode if macOS socket
permission behavior is not portable, but it must document which guarantee it is
using and test the path-safety decision logic. Log directories should also avoid
world-writable parents, but log readability is less security-critical than the
helper socket.

## Command Behavior

### `check`

Validate:

- host OS and architecture when host facts are available
- helper package path exists
- helper binary exists and is executable, or can be built
- `swift` is available when a build is needed
- `codesign` status is readable on macOS
- entitlements path exists when provided
- signed binary entitlements match the provided entitlements plist when both are available
- socket path is safe
- pid file is absent, stale, or belongs to the expected helper process
- helper protocol version matches the Python client expectation when the helper is reachable

Output should be human-readable by default and optionally JSON with `--json`.

The expected helper protocol version should come from one explicit source of
truth. Prefer importing or reading the Python helper client's expected protocol
constant; otherwise add a clearly named shared constant used by both checks and
tests. The lifecycle command should also allow `--expected-protocol-version` for
diagnostic overrides, but the default must remain the repo-owned expected value.

### `build`

Run:

```bash
swift build --package-path tools/macos-vz-helper -c debug
```

The command should print the command in dry-run mode and should not hide SwiftPM errors.

### `sign`

Ad hoc sign only when an explicit entitlements path is supplied:

```bash
codesign --force --sign - --entitlements <entitlements> <helper>
```

The command should not invent entitlements. It should validate that the helper exists and the entitlements file exists before invoking `codesign`.

Entitlement validation should be concrete. On macOS, `check` should read the
signed helper entitlements with `codesign -d --entitlements :- <helper>` when
possible and compare them against the operator-provided plist. At minimum, a
helper signed without the entitlements required by the provided plist must not
report as fully valid. The PR does not need to define a production certificate
identity, but it must distinguish:

- unsigned helper
- signed helper with unreadable entitlements
- signed helper with matching provided entitlements
- signed helper with mismatched or missing provided entitlements

### `start`

Start the helper with:

- `TLDW_SANDBOX_MACOS_HELPER_SOCKET=<socket>`
- `TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR=<log-dir>/serial`

Redirect stdout and stderr into the log directory and write a pid file only after the process starts. Then wait for the socket and verify `ping`.

If a pid file already points to a live expected helper, `start` should fail with `helper_already_running` unless a future explicit `--restart` option is added.

Failed startup must clean up after itself. If socket creation, helper `ping`, or
protocol validation fails after the command starts a helper process, `start`
should terminate that just-started process, remove the pid file, remove only a
safe stale socket it owns, and preserve stdout/stderr logs for debugging. It
should not kill an already-running helper that existed before this `start`
attempt.

### `status`

Report:

- pid file state
- process liveness
- socket existence
- helper ping status
- protocol version
- helper version
- binary path
- log directory
- launchd plist match status when a plist path exists

This command should not start or stop anything.

### `stop`

Stop only the pid-file-owned helper process after verifying it is the expected helper binary. It should tolerate already-stopped helpers, remove stale pid files, and avoid killing unrelated processes.

### `plist`

Generate a user LaunchAgent plist to stdout by default, or to `--plist-output` when explicitly provided.

The plist should encode:

- helper binary path
- socket path
- serial log directory
- stdout/stderr log paths
- `KeepAlive=false` for the first PR unless there is a strong reason to supervise automatically
- required environment variables

The command should validate generated contents and support `--dry-run`. It should not run `launchctl`.

### `smoke`

Wrap the existing `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh` with the managed defaults. This command should pass through:

- `--bundle`
- `--socket`
- `--serial-log-dir`
- `--helper`
- `--entitlements`
- `--python`
- `--dry-run`

It should not duplicate the E2E logic from the existing script.

## Diagnostics And Reason Strings

Use explicit reason strings suitable for docs and tests:

- `helper_binary_missing`
- `helper_binary_not_executable`
- `helper_swift_unavailable`
- `helper_codesign_unavailable`
- `helper_not_signed`
- `helper_entitlements_missing`
- `helper_socket_unsafe`
- `helper_pid_stale`
- `helper_pid_process_mismatch`
- `helper_already_running`
- `helper_not_running`
- `helper_socket_unavailable`
- `helper_protocol_mismatch`
- `helper_ping_failed`
- `launchd_plist_mismatch`

These strings are operator-command reasons, not a replacement for the existing sandbox API reason strings. Where they overlap, docs should map them to the existing runtime diagnostics vocabulary.

## Testing Strategy

Portable tests should cover:

- argument parsing and defaults
- dry-run output for `build`, `sign`, `start`, `plist`, and `smoke`
- socket path safety checks
- pid file stale/live/mismatch behavior using safe fakes
- plist generation content
- status output with fake helper-client results
- smoke command delegation to `run-host-e2e-smoke.sh`

Host-gated tests should remain opt-in and should not run in normal CI unless a prepared Apple silicon host explicitly enables them.

Suggested files:

- `tools/macos-vz-helper/scripts/vz-helperctl.py`
- `tools/macos-vz-helper/tests/test_vz_helperctl.py`
- `tools/macos-vz-helper/README.md`
- `Docs/Sandbox/macos-runtime-operator-notes.md`
- `tldw_Server_API/app/core/Sandbox/README.md`

## Risks And Mitigations

- **Risk:** The command becomes a second orchestration layer.
  **Mitigation:** Keep it limited to local helper process lifecycle. Python remains sandbox authority.

- **Risk:** Launchd behavior differs across user/system domains.
  **Mitigation:** Generate and validate a user LaunchAgent plist only. Do not install or load it in this PR.

- **Risk:** PID-based stop kills the wrong process.
  **Mitigation:** Verify the process command path before stopping and remove stale pid files conservatively.

- **Risk:** Signing checks become platform-fragile.
  **Mitigation:** Make signing status explicit and best-effort outside macOS. Require explicit entitlements for signing.

- **Risk:** Hidden auto-restarts mask failures.
  **Mitigation:** Do not enable automatic KeepAlive supervision in the first PR.

## Deferred Work

- explicit `install-launchd` and `uninstall-launchd`
- launchd `bootstrap`, `bootout`, and `kickstart` integration
- helper auto-upgrade and version pin policy
- code-signing identity configuration beyond ad hoc signing
- remote Apple silicon CI host integration
- helper-owned VM ownership metadata
- safe orphan VM termination
- APFS clone provisioning

## Success Criteria

The PR is complete when:

- operators can run one command to check helper lifecycle prerequisites
- operators can build/sign/start/status/stop the helper with stable paths
- operators can generate a launchd plist without installing it
- the existing real-host smoke script can be invoked through the managed defaults
- docs describe the new workflow and explicitly call out what is still manual
- portable tests cover command behavior without requiring real VMs
