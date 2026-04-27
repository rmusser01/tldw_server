# vz_linux Real Host E2E Design

Date: 2026-03-10
Status: Approved for planning
Scope: `tldw_Server_API/tests/sandbox/`, sandbox service/store setup, and macOS operator notes

## 1. Summary

This design adds an opt-in, host-gated end-to-end smoke test for real `vz_linux`
execution on Apple silicon macOS hosts.

The test proves two behaviors:

1. a real ephemeral `vz_linux` run can boot a Linux guest and execute one command
2. a real sandbox session can execute two commands through the same persisted
   `vz_linux` VM and then clean up that VM on session destruction

The test is intentionally narrow. It is not a full conformance suite for
`Virtualization.framework`, guest networking, or ACP.

## 2. Current State

The merged `vz_linux` work provides:

- a helper-backed `vz_linux` execution path in the runner
- persisted VZ session control metadata
- session VM reuse logic for repeated sandbox-session commands
- macOS host-gated smoke coverage for preflight and diagnostics

What is still missing is proof that a real operator-installed helper plus a real
Linux template can execute commands end-to-end on host hardware.

The repo now contains a real Unix-socket helper client and a frozen helper
protocol contract, but it still assumes a real native helper daemon exists on the
host where the test is run.

## 3. Review Corrections Applied

Reviewing the proposed design against the current codebase surfaced five changes
that should be explicit in the docs and plan:

1. The test must isolate its store and workspace roots with a temporary SQLite
   sandbox store setup. Using default repo paths would contaminate persistent
   runtime state under `Databases/` and the shared sandbox root.
2. The opt-in env contract should align with the current runner shape. The
   runner stores and passes `spec.base_image`, so the E2E test should use an env
   like `TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE`, not a new vague `template`
   identifier that the current code does not consume directly.
3. macOS gating must include an explicit Apple silicon check. Existing host-gated
   tests only check `sys.platform == "darwin"`, which is too loose for a real VZ
   execution smoke test.
4. Session reuse should be asserted via persisted VZ session control metadata,
   not by counting helper create calls. The stored `vm_id` is the contract the
   service owns today.
5. The session reuse smoke test must explicitly enable foreground sandbox
   execution. `SandboxService.start_run_scaffold()` only executes a run when
   `SANDBOX_ENABLE_EXECUTION=1`, and it may dispatch to background workers if
   `SANDBOX_BACKGROUND_EXECUTION` is enabled.

## 4. Goals and Non-Goals

### Goals

1. Add one opt-in pytest module that can prove real `vz_linux` execution on a
   prepared Apple silicon macOS host.
2. Verify one real ephemeral run and one real session reuse path.
3. Keep failures actionable with clear skip reasons for missing prerequisites.
4. Avoid ACP and other higher-level orchestration so failures point directly at
   the VM-backed sandbox path.
5. Keep test state isolated from shared repo runtime data.

### Non-Goals

1. Making this test part of the default CI path.
2. Validating ACP end-to-end.
3. Testing `vz_macos`.
4. Testing allowlist networking.
5. Building the native helper in this repo if it does not already exist.

## 5. Selected Approach

Add a dedicated host-gated pytest module:

- `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`

This module will:

- skip unless the host is macOS on Apple silicon
- skip unless `TLDW_SANDBOX_VZ_LINUX_E2E=1`
- skip unless a real helper/template-backed preflight says `vz_linux` is ready
- use a temp SQLite sandbox store and temp sandbox root paths
- exercise the real `SandboxService` and `VZLinuxRunner` paths directly

This is better than a manual script because it stays in the normal test harness,
and better than an ACP test because it isolates the VM execution path itself.

## 6. Runtime and Environment Contract

The real-host test should require explicit operator input:

- `TLDW_SANDBOX_VZ_LINUX_E2E=1`
  - hard opt-in gate
- `TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=<value>`
  - the `RunSpec.base_image` / session base image to use for the guest
- `TLDW_SANDBOX_MACOS_HELPER_SOCKET=/path/to/helper.sock`
  - the real helper transport endpoint used for `ping`, `validate_host`, and
    `validate_template`
- `SANDBOX_ENABLE_EXECUTION=1`
  - required for the session-backed smoke test because `SandboxService` only
    runs the queued sandbox job when execution is enabled
- `SANDBOX_BACKGROUND_EXECUTION=0`
  - recommended for the smoke test so `start_run_scaffold()` executes
    synchronously and assertions can inspect the completed run status directly
- whatever real helper config the operator-installed helper requires
  - the test should rely on existing runtime preflight for this, not invent a
    separate parallel config system

The test should not use `TEST_MODE`, fake helper flags, or fake execution flags.
It should first prove helper reachability with `ping` and assert that a
`protocol_version` is present, then prove runnable-template truth through
`validate_template`. If only scaffolding is present, the test must skip or fail
based on those real helper calls rather than silently falling back.

## 7. Test Cases

### 7.1 Ephemeral run smoke test

The first test should:

1. configure an isolated sandbox store and workspace root
2. verify the host/e2e prerequisites
3. execute a real `vz_linux` run with no `session_id`
4. run a minimal command such as `/bin/echo vz-linux-e2e`
5. assert:
   - completed phase
   - exit code `0`
   - stdout contains the expected token

### 7.2 Session reuse smoke test

The second test should:

1. configure an isolated sandbox store and workspace root
2. create a real sandbox session with `runtime=vz_linux`
3. run command 1 in that session
4. read persisted VZ session control metadata from the service orchestrator
5. run command 2 in the same session
6. read the session control again
7. assert:
   - both runs completed
   - both runs exited `0`
   - the stored `vm_id` exists after run 1
   - the stored `vm_id` is unchanged after run 2
8. call `destroy_session()`
9. assert the VZ session control row is gone

The command payloads should stay boring and deterministic:

- ephemeral: `/bin/echo vz-linux-e2e`
- session run 1: `/bin/echo first`
- session run 2: `/bin/echo second`

## 8. Skip and Failure Semantics

### Skip conditions

Skip with explicit reasons when:

- host OS is not macOS
- host architecture is not Apple silicon
- `TLDW_SANDBOX_VZ_LINUX_E2E` is not enabled
- `TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE` is missing
- helper `ping` is unreachable
- helper-backed `validate_template` says the base image is not runnable
- `vz_linux` preflight is unavailable because helper/template readiness is not
  satisfied

### Failure conditions

Fail when:

- preflight reports `vz_linux` available, but the real run fails
- the session-backed first run succeeds but no VZ session control row is stored
- the second session-backed run changes the persisted `vm_id`
- `destroy_session()` returns success but leaves VZ session control behind

## 9. Isolation and Cleanup

The test must use temp-backed store and root config similar to the existing
session durability coverage:

- temp SQLite DB path
- temp sandbox root dir
- temp snapshot dir

That ensures:

- no persistent session metadata leaks into repo defaults
- session workspaces are removed after destruction
- repeated local E2E runs do not pollute each other

Cleanup should rely on both:

- normal `destroy_session()` behavior
- pytest temp directory teardown as a final safety net

## 10. Risks and Constraints

1. The current repo does not contain a real native helper implementation, so this
   test is only executable on operator-prepared hosts.
2. If the helper contract changes, the test env documentation must stay aligned
   with runtime preflight instead of drifting into a separate setup story.
3. Real host execution may be slower or more brittle than fake-helper tests, so
   the module must remain opt-in and outside default CI.
4. Guest command assumptions should stay minimal to avoid coupling the smoke test
   to one specific distro image beyond basic POSIX utilities.

## 11. Success Criteria

The design is successful when:

1. there is a documented, opt-in pytest path for real `vz_linux` host execution
2. the module proves both real ephemeral execution and session VM reuse
3. the test isolates its sandbox store/workspace state
4. operators get actionable skip reasons instead of ambiguous failures when the
   host is not prepared
