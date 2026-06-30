# VZ Linux Host Failure Drills Design

**Date:** 2026-05-09
**Status:** Approved for implementation planning
**Backlog:** TASK-145
**Scope:** Host-gated `vz_linux` smoke workflow, operator smoke script, real-host pytest coverage, and host-gated policy docs.

## Summary

The `vz_linux` host-gated path now verifies helper build/sign/start, real
ephemeral execution, same-session VM reuse, recovery diagnostics, and dry-run
repair planning. The next stability slice should add manual opt-in failure
drills that prove the runner can recover from stale session-control metadata
without making normal nightly/manual smoke riskier.

This PR should add a manual-only failure-drill mode. The first drill should
create a real session VM, invalidate only that VM through helper truth, run a
second command in the same sandbox session, and assert the runner detects the
stale or unhealthy reuse candidate, clears stale control state, provisions a
fresh VM, and completes.

## Goals

- Add a failure-drill flag to the host smoke script that is disabled by default.
- Expose a manual `workflow_dispatch` input for failure drills without enabling
  drills for scheduled runs.
- Add one real-host failure drill for stale session VM recovery.
- Keep normal host-gated smoke stable and non-destructive.
- Document that failure drills are manual-only operator coverage.
- Add host-independent tests for script and workflow wiring.

## Non-Goals

- Do not make failure drills part of scheduled nightly smoke by default.
- Do not add PR or push triggers to the host-gated workflow.
- Do not add helper crash, stale socket, launchd restart, or host reboot drills
  in this PR.
- Do not terminate orphaned VMs or repair VMs not created by the current test.
- Do not change public sandbox API contracts.
- Do not relax the prepared-host branch gate.

## Chosen Approach

Add a `--include-failure-drills` option to
`tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`. The existing smoke script
already owns helper build/sign/start, runtime path setup, and helper cleanup, so
reusing it avoids a second lifecycle path. By default it should keep running
only the established `vz_linux_host_smoke` marker.

When `--include-failure-drills` is present, the script should run a second
pytest invocation for a new marker such as `vz_linux_host_failure_drill` after
the baseline smoke passes. This sequencing matters: failure drills should only
run after proving the prepared host and canonical bundle are basically healthy.

Wire the flag through `.github/workflows/vz-linux-host-gated.yml` as a
`workflow_dispatch` boolean input. The scheduled path should never enable it
implicitly. Manual dispatch should allow maintainers to run the drill when
validating recovery behavior or investigating a real-host regression.

## Failure Drill Contract

The first drill should be scoped to stale session VM recovery:

1. Configure an isolated SQLite sandbox store and temporary sandbox root.
2. Create a `vz_linux` sandbox session with the canonical real-host base image.
3. Run a first command in that session and assert completion.
4. Read the persisted VZ session-control row and capture the VM ID.
5. Terminate only that VM through the helper client.
6. Run a second command in the same sandbox session.
7. Assert the second command completes.
8. Read the session-control row again and assert the VM ID changed.
9. Destroy the session in `finally` and tolerate already-gone VM cleanup.

The drill must not use broad helper cleanup, orphan termination, or socket
deletion. It should only act on the VM ID created by the same test and recorded
in the current test's session-control row.

## Safety Constraints

- The drill is opt-in and not part of scheduled runs by default.
- The workflow keeps the existing branch gate for `main` and `dev`.
- The test must use a unique session ID/run context through normal service APIs.
- The test must terminate only the VM ID captured from its own session control.
- The test should verify the helper operation succeeds or skip/fail with a clear
  reason rather than continuing after ambiguous invalidation.
- Cleanup must destroy the sandbox session in `finally`.
- No test should mutate persistent operator configuration or global image-store
  state beyond the existing temporary store setup.

## Workflow Shape

Add manual input:

```yaml
include_failure_drills:
  description: Run manual-only VZ Linux failure recovery drills
  required: false
  default: false
  type: boolean
```

In the managed host smoke step, append `--include-failure-drills` only when the
manual input is truthy. Scheduled runs should resolve this input to false.

## Script Shape

`run-host-e2e-smoke.sh` should gain:

- usage text for `--include-failure-drills`
- an `INCLUDE_FAILURE_DRILLS=0` variable
- argument parsing for `--include-failure-drills`
- a `run_real_vz_linux_failure_drills` helper that calls pytest with the new
  marker
- a conditional call after `run_real_vz_linux_host_smoke`

Dry-run mode should print the failure-drill pytest command only when the flag is
present. Existing dry-run output without the flag should remain unchanged except
for help text.

## Test Strategy

Host-independent tests should cover:

- the smoke script help mentions `--include-failure-drills`
- default dry-run output does not include the failure-drill marker
- dry-run output with `--include-failure-drills` includes the failure-drill
  pytest invocation
- workflow contract tests verify the new input exists, defaults false, and is
  only passed to the script conditionally

Real-host coverage should remain opt-in:

- the new pytest drill is marked `vz_linux_host_failure_drill`
- it is skipped outside macOS Apple Silicon or without the existing real-host
  opt-in variables
- it runs only when the smoke script flag or a direct pytest marker selection
  asks for it

## Risks And Mitigations

- **Risk:** The drill terminates a VM that does not belong to it.
  **Mitigation:** Only terminate the VM ID read from the session-control row
  created by the same service/session flow, then destroy that session in
  `finally`.
- **Risk:** Failure drill makes nightly CI flaky.
  **Mitigation:** Keep drills disabled by default and wired only to manual
  dispatch input.
- **Risk:** The second command fails because helper status probing fails instead
  of falling through to fresh provisioning.
  **Mitigation:** The test intentionally exercises the existing graceful fallback
  contract: absent/unhealthy probe results should clear control state and create
  a fresh VM.
- **Risk:** Helper crash/stale socket behavior remains untested.
  **Mitigation:** Leave those as later drills after the VM-scoped invalidation
  path proves safe and useful.

## Open Follow-Ups

- Promote selected failure drills to scheduled nightly only after repeated
  prepared-host success.
- Add helper crash and stale socket drills once helper restart semantics are
  less manual and less likely to produce false failures.
- Add explicit artifact summaries for failure-drill outcomes if operators need
  richer evidence than pytest logs.
