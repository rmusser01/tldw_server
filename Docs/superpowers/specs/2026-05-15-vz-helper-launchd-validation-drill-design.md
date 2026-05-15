# VZ Helper Launchd Validation Drill Design

**Date:** 2026-05-15
**Status:** Proposed design for implementation planning
**Backlog:** TASK-360
**GitHub:** https://github.com/rmusser01/tldw_server/issues/1442
**Scope:** `tools/macos-vz-helper/`, `tools/vz-linux-image/scripts/`,
host-gated VZ Linux smoke policy, and macOS operator documentation.

## Summary

The repo now has explicit `vz-helperctl.py launchd` operator commands for
`bootstrap`, `bootout`, `kickstart`, and `status`. The remaining gap is proving
that those commands form a repeatable operator drill without changing the
default direct-helper smoke path.

This design adds a dedicated launchd validation drill. The drill should validate
the LaunchAgent plist and private runtime paths, bootstrap the helper through
`launchd`, inspect launchd status, kickstart the helper, verify helper
readiness through the existing socket/protocol checks, optionally run the
existing host-gated `vz_linux` pytest smoke contract against that helper, then
bootout the service and verify cleanup. It stays explicit and operator-owned:
no server startup automation, no implicit install, no scheduled host reboot,
and no hidden repair.

## Source Documents

- `Docs/Sandbox/sandbox-architecture-doctrine.md`
- `Docs/Sandbox/macos-runtime-operator-notes.md`
- `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
- `Docs/superpowers/specs/2026-04-29-vz-helper-lifecycle-hardening-design.md`
- `Docs/superpowers/specs/2026-05-09-vz-linux-helper-crash-host-reboot-recovery-posture-design.md`
- `Docs/superpowers/plans/2026-05-13-vz-helper-launchd-operator.md`
- `tools/macos-vz-helper/PROTOCOL.md`
- `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`

## Current Baseline

Already implemented behavior:

- `vz-helperctl.py launchd` can construct and execute explicit `launchctl`
  `bootstrap`, `bootout`, `kickstart`, and `print` actions.
- Bootstrap can validate or explicitly write a LaunchAgent plist with
  operator-supplied `--write-plist` and `--create-dirs`.
- Launchd plist rendering reuses the helper socket, serial log directory,
  protocol version, stdout/stderr log paths, and label contract.
- `vz-helperctl.py status` can validate helper binary, socket/pid/log paths,
  serial log directory, launchd plist match, entitlements, ping, helper version,
  and protocol version.
- `run-host-e2e-smoke.sh` directly starts the helper, runs helper daemon smoke,
  runs real `vz_linux` ephemeral execution, verifies same-session reuse, runs
  read-only recovery/dry-run repair checks, and optionally runs failure drills.
  That shell script is useful prior art for the test contract, but it cannot be
  invoked unchanged by a launchd-managed drill because it owns helper startup.
- Host-gated CI keeps real VZ execution opt-in to prepared Apple silicon hosts.

Remaining gap:

- There is no single drill that proves the launchd command path can supervise
  the helper and hand off to the same real `vz_linux` smoke guarantees.
- The default smoke intentionally bypasses launchd, so launchd regressions can
  be missed even when direct-helper smoke passes.
- Operator docs describe launchd commands, but do not give a single repeatable
  evidence-producing validation flow.

## Goals

1. Add a dedicated launchd validation drill for prepared macOS hosts.
2. Keep the drill explicit, operator-owned, and separate from default direct
   helper smoke behavior.
3. Reuse the existing `vz-helperctl.py launchd` and `status` primitives instead
   of inventing a workflow-only launchctl path.
4. Preserve strict socket, runtime-directory, log-directory, serial-directory,
   and plist safety checks before any launchd mutation.
5. Preserve logs and report enough launchd/helper evidence to debug partial
   bootstrap, kickstart, readiness, smoke, or bootout failures.
6. Define portable tests for command sequencing and dry-run behavior, plus
   host-gated/manual expectations for real launchd execution.

## Non-Goals

1. No automatic LaunchAgent installation from server startup, diagnostics,
   `status`, `check`, or ordinary smoke.
2. No change to the default `run-host-e2e-smoke.sh` direct-helper lifecycle.
3. No scheduled host reboot drill.
4. No destructive repair behavior tied to launchd status.
5. No cross-runtime launchd abstraction.
6. No replacement of the Swift helper, Python helper client, or guest agent.
7. No vmnet/networking changes.

## Recommended Shape

Add one explicit operator drill entrypoint. The implementation can choose one of
two equivalent surfaces, but the preferred user-facing shape is a helperctl
subcommand:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py launchd-drill \
  --bundle /path/to/canonical/bundle \
  --helper tools/macos-vz-helper/.build/debug/macos-vz-helper \
  --socket "$runtime_dir/helper.sock" \
  --log-dir "$runtime_dir/logs" \
  --plist-output "$runtime_dir/org.tldw.macos-vz-helper.plist" \
  --write-plist \
  --create-dirs
```

The command should support `--dry-run` and `--json`. Real `launchctl` mutation
must require a prepared macOS host and explicit non-dry-run execution. The drill
may optionally expose `--skip-smoke` for launchd-only validation when an operator
wants to prove process supervision without booting a VM.

Do not add launchd as the default path inside `run-host-e2e-smoke.sh`. If shell
integration is useful later, add an opt-in wrapper flag such as `--use-launchd`
that delegates to `vz-helperctl.py launchd-drill`, or add an explicit
`--external-helper-socket` mode that only runs the existing pytest smoke against
an already-managed helper. Do not duplicate launchctl sequencing in the shell
script.

## Drill Lifecycle

The drill should run this sequence:

1. Resolve helper, socket, log directory, serial log directory, plist path,
   launchd label, UID/domain, and optional bundle path.
2. Validate or create private parent directories with the existing helperctl
   path-safety helpers.
3. Validate helper binary and entitlements state using the existing lifecycle
   checks.
4. Render or validate the launchd plist only when explicitly requested.
5. Run `launchd bootstrap`.
6. Run `launchd status` and record the launchctl target.
7. Run `launchd kickstart -k`.
8. Wait for helper socket readiness and protocol-compatible ping through the
   existing helper status path.
9. If bundle smoke is enabled, run the existing host-gated pytest smoke contract
   against the launchd-managed socket without starting a second helper.
10. Run `launchd bootout`.
11. Verify launchd no longer reports a loaded service, helper ping fails or
   helper is absent, and only safe stale sockets are removed.
12. Preserve stdout, stderr, serial logs, plist, and JSON/human drill output.

`bootout` should be attempted during cleanup after any failure that happens
after bootstrap succeeds. Cleanup must not hide the primary failure, but a
cleanup failure should be reported as secondary evidence.

## Safety Contract

The drill should inherit existing helperctl safety behavior:

- refuse symlink socket paths
- refuse existing non-socket files at the socket path
- remove an existing socket only when it is actually a Unix socket and the
  operator-owned parent directory is private
- require owner-only runtime, log, serial, and plist parent directories
- refuse plist writes unless `--write-plist` is present
- refuse creating missing directories unless `--create-dirs` is present
- keep `status` and dry-run checks read-only
- use the explicit launchd label and GUI domain when reporting service targets

The launchd path is a process-supervision drill only. It must not imply VM reuse
or session repair. Python should continue to trust helper protocol and live VM
truth, not launchd state, when deciding whether a session VM can be reused.

## Output And Diagnostics

Human output should be step-oriented and terse:

```text
launchd_plist: ok launchd_plist_written
launchd_bootstrap: ok
launchd_status: ok gui/501/org.tldw.macos-vz-helper
launchd_kickstart: ok
helper_status: ok protocol_version=1 helper_version=...
vz_linux_smoke: ok
launchd_bootout: ok
cleanup: ok
```

JSON output should preserve the existing `CheckResult` style so operator
automation can distinguish setup failures from runtime failures. Include at
least:

- step name
- `ok`
- reason
- message
- launchd label
- launchd service target
- socket path
- plist path
- log directory

Do not include secrets or raw guest command output in drill metadata.

## Failure Handling

Expected failure classes:

- `launchd_launchctl_unavailable`: host is not prepared for launchd execution.
- `launchd_plist_missing` or `launchd_plist_mismatch`: operator plist setup is
  incomplete or stale.
- `helper_directory_not_private` or related path failures: runtime paths are not
  safe enough for a local helper socket.
- `launchd_bootstrap_failed`, `launchd_kickstart_failed`, or
  `launchd_bootout_failed`: launchctl action failed; preserve launchctl target
  and exit code.
- `helper_not_running`, `helper_protocol_mismatch`, or ping failure: launchd may
  have loaded a job but the helper is not usable by Python.
- `vz_linux_smoke_failed`: launchd worked enough to start the helper, but real
  VM execution failed.

If bootstrap succeeds and a later step fails, the drill should still attempt
bootout. If bootout fails, the final result should clearly say that operator
cleanup is required and print the exact `launchctl bootout <target>` command.

## Test Strategy

Portable tests should cover:

- drill command sequencing with injected launchd/status/smoke runners
- dry-run output without filesystem mutation
- no launchctl execution when plist validation fails
- bootout attempted after bootstrap-plus-later-failure
- primary failure preserved when cleanup also fails
- `--skip-smoke` avoids VM smoke while still validating launchd/helper readiness
- JSON result shape and step names
- custom launchd labels and UIDs

Host-gated/manual validation should cover:

- real bootstrap/status/kickstart/bootout on Apple silicon
- helper ping/protocol readiness through the launchd-managed socket
- real `vz_linux` ephemeral execution through the launchd-managed helper
- same-session reuse through the launchd-managed helper
- logs preserved after failure

Scheduled host-gated CI should not enable the launchd drill by default. A manual
workflow input can enable it later once the drill has proven stable on prepared
hosts and the runner is explicitly configured for LaunchAgent validation.

## Documentation Updates

The implementation PR should update:

- `tools/macos-vz-helper/README.md` with local drill commands and cleanup notes
- `Docs/Sandbox/macos-runtime-operator-notes.md` with the launchd validation
  flow and troubleshooting guidance
- `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md` with the drill's
  expected skips and blocking criteria if/when manual workflow support is added
- GitHub tracker #1442 to record that launchd scaffolding exists and the next
  gap is drill validation

## Open Questions For Implementation Planning

1. Should the first implementation expose only `vz-helperctl.py launchd-drill`,
   or also add a thin `run-host-e2e-smoke.sh --external-helper-socket` wrapper
   that runs the existing pytest smoke without owning helper startup?
2. Should the initial drill require a bundle path, or should launchd-only
   validation be the default with `--bundle` enabling VM smoke?
3. Should real launchd host-gated workflow support land in the same PR as the
   local operator command, or in a follow-up after local validation?

Recommended answers for the first implementation:

- implement `launchd-drill` first
- make VM smoke optional through `--bundle`
- defer workflow integration until the local command has portable coverage and
  at least one manual real-host run
