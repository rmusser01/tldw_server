# VZ Linux Helper Restart Recovery Drill Design

**Date:** 2026-05-09
**Status:** Approved slice; awaiting implementation-plan review
**Backlog:** TASK-150
**Scope:** Host-gated `vz_linux` failure-drill path, lower-level smoke script lifecycle handoff, real-host pytest coverage, and operator policy docs.

## Summary

PR #1397 added a manual failure-drill path that proves a `vz_linux` session can
recover when its helper-owned VM is terminated behind the sandbox service's
back. The next narrow recovery drill should validate the adjacent failure mode:
the helper process is stopped and restarted after a session VM exists, leaving
the sandbox service with stale session-control metadata that points at helper
state the replacement helper cannot know about.

This slice should stay manual-only and host-gated. It should add one helper
restart drill that creates a real `vz_linux` session, runs a command, restarts
the helper process through an explicit test lease from the smoke script, runs a
second command in the same session, and asserts the runner clears stale control
state and provisions or recovers cleanly.

## Current Baseline

- `run-host-e2e-smoke.sh` owns the lower-level helper lifecycle for real host
  smoke: build/sign, runtime path preparation, direct helper start, socket wait,
  real pytest smoke, optional failure drills, and cleanup on exit.
- `vz-helperctl.py smoke` delegates to the lower-level smoke script with managed
  defaults, but the lower-level script still starts the helper directly.
- Real-host pytest currently validates ephemeral execution, same-session reuse,
  recovery diagnostics/dry-run repair, and stale VM replacement after
  helper-side VM termination.
- The sandbox roadmap still lists helper crash/restart, host reboot, stale
  socket, stuck boot/readiness, and guest-agent mismatch recovery as remaining
  Phase 1/Phase 4 work.

## Goals

- Add one manual helper restart drill under the existing failure-drill path.
- Preserve normal host smoke and scheduled host-gated CI defaults.
- Exercise the real service/runner path, not a fake helper or synthetic control
  row.
- Keep helper lifecycle ownership explicit and local to the smoke harness.
- Add host-independent tests for script/workflow/operator contract changes.
- Document the new accepted host-gated failure mode and remaining gaps.

## Non-Goals

- Do not add host reboot automation.
- Do not add `launchctl bootstrap`, `bootout`, `kickstart`, install, or
  uninstall behavior.
- Do not refactor the entire lower-level smoke script to use `vz-helperctl
  start`/`stop`.
- Do not change helper protocol, VM boot, guest-agent, networking, image-store,
  or sandbox public API behavior.
- Do not run helper restart drills in scheduled smoke by default.
- Do not generalize repair mutation beyond the existing ownership-checked
  `vz_linux` surfaces.

## Approach Options

### Option A: Smoke Script Restart Lease

Keep `run-host-e2e-smoke.sh` as the helper lifecycle owner. When failure drills
are enabled, it writes the current helper PID to a private pid file and passes a
small set of drill-only environment variables to pytest:

- helper PID file path
- helper binary path
- helper socket path
- serial/helper log directory
- an explicit `TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED=1` opt-in

The pytest drill may terminate the current helper PID, wait for the socket to go
away, start the same helper binary with the same socket and serial log
environment, update the pid file with the replacement helper PID, wait for
helper ping/template readiness, then continue with the second same-session run.
The script cleanup reads the pid file on exit so it terminates whichever helper
instance is current.

This is the recommended path for this PR. It is narrow, keeps the lower-level
smoke harness responsible for process cleanup, and avoids making the
implementation depend on `launchd` or a broader helperctl lifecycle refactor.

### Option B: Convert Smoke To `vz-helperctl start`/`stop`

Make the lower-level smoke script start and stop the helper through
`vz-helperctl.py` and use its pid file as the restart handoff. This is cleaner
long term, but it changes the baseline helper lifecycle for every host smoke
run. It also needs a careful compatibility story for custom `--serial-log-dir`
values because `vz-helperctl start` derives serial logs from `--log-dir/serial`.

This should remain a later lifecycle-unification PR unless the restart lease
proves too brittle.

### Option C: Launchd Or Host-Reboot Drill

Use launchd restart or host reboot to prove recovery. This would be closer to a
real operator outage, but it is too destructive and too slow for the next
reviewable slice. It also requires host-level state and runner orchestration
that the current manual host-gated workflow intentionally avoids.

## Chosen Design

Use Option A. Extend the lower-level smoke script with a private helper pid file
and restart lease only when failure drills are enabled. The default smoke path
should still start the helper, wait for the socket, run baseline smoke, and
cleanup exactly as it does now.

The helper restart drill should live in
`tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py` under the existing
`vz_linux_host_failure_drill` marker. It should require the current real-host
opt-in variables plus the helper restart lease variables. If the drill is run
directly without the lease, it should skip with a clear reason instead of trying
to infer process ownership.

## Drill Contract

1. Configure the isolated SQLite sandbox store and temporary sandbox root using
   the existing real-host E2E setup.
2. Create a `vz_linux` sandbox session with the canonical bundle.
3. Run a first command and assert it completes.
4. Capture the persisted VZ session-control VM ID.
5. Confirm helper status reports that VM as healthy before restart.
6. Terminate the helper process from the pid file and wait until the old helper
   socket is unavailable.
7. Start the same helper binary with the same socket path and serial log
   environment.
8. Update the pid file to the replacement helper PID before running the second
   command.
9. Wait for helper ping and template validation to pass.
10. Run a second command in the same sandbox session.
11. Assert the command completes and the session-control VM ID changed, proving
    stale helper-owned VM state was not reused.
12. Destroy the sandbox session in `finally`.

The drill should treat restart setup failures as clear skips or failures based
on ownership:

- missing lease variables: skip, because direct pytest runs cannot safely manage
  the external helper process
- original helper already exited before the drill can stop it: skip with a
  prepared-host/lifecycle reason
- replacement helper cannot create the accepted socket or answer ping: fail the
  drill, because that is the failure mode being tested
- second run aborts instead of provisioning a fresh VM: fail the drill

## Smoke Script Changes

`run-host-e2e-smoke.sh` should gain internal state for a helper pid file under
the private runtime directory. For normal smoke, this can be implementation
detail only. For failure drills, `run_real_vz_linux_failure_drills` should pass:

```text
TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED=1
TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE=<private-runtime-dir>/helper.pid
TLDW_SANDBOX_MACOS_HELPER_BINARY=<helper-path>
TLDW_SANDBOX_MACOS_HELPER_SOCKET=<socket-path>
TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR=<serial-log-dir>
```

Cleanup should read the pid file if present and prefer that PID over the
original shell variable. This lets a pytest-managed replacement helper remain
visible to the script's existing cleanup trap.

Dry-run output should make the restart lease visible only when failure drills
are included. Default dry-run output without failure drills should not advertise
restart behavior.

## Workflow Shape

No new workflow trigger is needed. The existing manual
`include_failure_drills` input remains the gate. Scheduled runs continue to run
baseline smoke only.

The host-gated workflow contract tests should be updated only if script
arguments or documented accepted failure criteria change. Do not add PR or push
triggers.

## Safety Constraints

- Restart management is available only under the real-host E2E opt-in plus
  failure-drill opt-in.
- The drill must use a private runtime directory and owner-only serial/log/pid
  paths.
- The drill must only act on the helper PID supplied by the smoke script's
  private pid file.
- The replacement helper must use the exact accepted socket path; no fallback
  socket path is allowed.
- The test should not terminate arbitrary helper processes discovered through
  `ps`, socket probing, or broad name matching.
- The sandbox session must be destroyed in `finally`.
- The smoke script cleanup must tolerate the original helper already being gone
  and the replacement helper being the current process to stop.

## Host-Independent Test Strategy

Script tests should cover:

- failure-drill dry-run output includes the restart lease environment when
  failure drills are requested
- default dry-run output does not include helper restart lease variables
- cleanup selects a replacement helper PID from the pid file when present
- fake-helper real-run mode can update the helper pid file without leaking the
  runtime directory or logs

Real-host pytest guard tests should cover:

- helper restart drill skips when real-host opt-in is missing
- helper restart drill skips when restart lease opt-in is missing
- restart helper helper functions reject missing or non-private pid/log/socket
  inputs where practical

Real-host verification remains:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py smoke \
  --bundle /path/to/canonical/bundle \
  --entitlements /path/to/helper.entitlements \
  --include-failure-drills
```

If `vz-helperctl.py smoke` does not yet forward `--include-failure-drills`, this
PR should add that pass-through so the documented operator entrypoint can run
the manual drills.

## Documentation Updates

Update:

- `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md` to include helper
  restart replacement as a manual failure-drill criterion.
- `Docs/Sandbox/macos-runtime-operator-notes.md` to describe that manual
  failure drills now include stale VM termination and helper restart recovery,
  while host reboot and launchd restart remain manual/operator-only gaps.
- `Docs/Sandbox/sandbox-runtime-capability-inventory.md` only if the current
  gaps text needs to distinguish helper restart drill coverage from still-manual
  host reboot/destructive repair coverage.

## Design Review Notes

- **Potential problem:** Restarting the helper inside pytest can hide ownership
  from the shell cleanup trap.
  **Mitigation:** Require a script-owned pid file and make cleanup read the
  latest pid from that file.
- **Potential problem:** Refactoring all smoke startup through `vz-helperctl
  start` would be cleaner but broad.
  **Mitigation:** Keep this PR to the restart lease and leave full lifecycle
  unification as a later PR.
- **Potential problem:** Pytest order could matter if one failure drill leaves
  the helper in a different state.
  **Mitigation:** The helper restart drill must leave a healthy replacement
  helper running and update the pid file, so subsequent drills and script
  cleanup see the current helper.
- **Potential problem:** A replacement helper may start but not own the old VM.
  **Mitigation:** That is the intended condition; the second same-session run
  should see missing/unhealthy helper truth and provision a fresh VM.
- **Potential problem:** Direct pytest invocation could kill a developer's
  manually started helper.
  **Mitigation:** The restart drill skips without the explicit restart lease
  variables supplied by the smoke script.

## Open Follow-Ups

- Convert the lower-level smoke script to `vz-helperctl start`/`stop` after the
  restart lease path proves stable.
- Add stale socket and stuck-readiness drills as separate manual failure-drill
  slices.
- Add host reboot recovery documentation or a manual operator playbook before
  attempting automated reboot coverage.
- Promote selected failure drills to scheduled host-gated runs only after
  repeated prepared-host success.
