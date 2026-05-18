# VZ Linux Prepared-Host Evidence Tracker

**Status:** Active tracker for prepared Apple silicon `vz_linux` acceptance evidence.
**Scope:** Real `vz_linux` execution evidence from manual operator runs or the host-gated workflow on trusted refs.
**Policy:** `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`.
**Operator entrypoint:** `tools/macos-vz-helper/scripts/vz-helperctl.py smoke` or `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`.

## Purpose

This tracker makes prepared-host acceptance evidence reviewable without making
real VM execution part of normal CI. It records what a prepared Apple silicon
host proved, which expected skips were accepted, which artifacts were preserved,
and what residual gaps remain.

Normal PR checks should continue to use portable unit tests, workflow contract
tests, fake/scaffolded paths, and docs checks. Real `vz_linux` execution remains
manual or host-gated only:

- local operator run on a prepared Apple silicon macOS host
- `workflow_dispatch` on `main` or `dev`
- opted-in scheduled host-gated workflow when
  `TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY=1`

Do not add pull request triggers, push triggers, scheduled destructive drills,
host reboot automation, launchd automation, or network expansion from this
tracker.

## Evidence Packet

Each prepared-host evidence packet should include these fields.

| Field | Required content |
| --- | --- |
| Evidence date | ISO date and local timezone. |
| Evidence source | `local-operator`, `workflow_dispatch`, or `nightly-host-gated`. |
| Git state | repository, branch, commit SHA, PR number if applicable, and dirty/clean status. |
| Host identity | Apple silicon model or runner label summary, macOS version, architecture, runner name if CI, and whether the host is dedicated or shared. |
| Host prep | Xcode command line tools availability, SwiftPM availability, `xcrun codesign` availability, Virtualization.framework availability, and runner labels for CI. |
| Bundle/template | bundle path or registered template id, manifest path if registered, artifact hashes when available, build provenance, and whether validation used canonical bundle or compatibility mode. |
| Helper build/signing | helper binary path, helper version, protocol version, signing mode, entitlements path, entitlement validation result, and skip-sign rationale when signing was skipped. |
| Runtime paths | private runtime directory, socket path, serial-log directory, log directory, and evidence that runtime/log directories were owner-only. |
| Commands | exact smoke, helperctl, pytest, workflow, restart-drill, and optional launchd-drill commands that were run. |
| Results | pass/fail/skip for daemon smoke, ephemeral command execution, same-session VM reuse, recovery diagnostics, dry-run reconciliation repair, helper shutdown, and artifact upload. |
| Failure drills | pass/fail/skip for drill-owned stale VM replacement and helper restart drill; include skip reason when `include_failure_drills` was not requested. |
| Launchd drill | pass/fail/skip for `launchd-drill`; include skip reason unless a maintainer explicitly requested LaunchAgent validation. |
| Artifacts | workflow run URL or local artifact root, helper stdout/stderr files, serial logs, pytest logs, workflow logs, and checksums or sizes for retained artifacts. |
| Expected skips | explicit non-blocking skips from the acceptance policy, including missing nightly opt-in, no launchd request, no failure-drill request, or local unprepared-host checks. |
| Blocking regressions | any failed guarantee from the acceptance policy and the first failing command/log pointer. |
| Residual gaps | known uncovered cases such as host reboot, stuck boot/readiness, guest-agent mismatch, stale socket handling, or launchd validation when skipped. |
| Follow-up owner | issue, task, or PR that will address each residual gap. |

Do not paste secrets, API keys, raw user data, or full runner logs into the
tracker. Prefer artifact links, file names, byte sizes, checksums, and short
redacted excerpts.

## Acceptance Checklist

Use this checklist for a complete prepared-host acceptance entry.

| Check | Evidence requirement | Required for default smoke |
| --- | --- | --- |
| Prepared Apple silicon host validation | Host facts and helper/template preflight passed or skipped with an operator-setup reason. | Yes |
| Helper build/sign/start | Helper built or existing binary validated, signing/entitlements state recorded, daemon smoke passed, and socket/log paths were private. | Yes |
| Real `vz_linux` ephemeral execution | A command executed inside a real VM and returned expected stdout/stderr/exit status. | Yes |
| Same-session VM reuse | A second command in the same sandbox session reused the same healthy VM or recorded a blocking failure. | Yes |
| Recovery diagnostics | macOS diagnostics and dry-run reconciliation repair planning ran without mutating session-control rows or terminating VMs. | Yes |
| Helper shutdown/cleanup | The helper stopped on exit and did not leave the accepted socket path behind. | Yes |
| Artifact upload or retention | Helper logs, serial logs, and pytest/workflow logs were retained or an early setup skip explains why none exist. | Yes |
| Failure drills | Drill-owned stale VM replacement and helper restart drill results recorded. | Manual opt-in only |
| Launchd drill | LaunchAgent bootstrap/kickstart/status/bootout drill results recorded. | Manual opt-in only |
| Host reboot drill | Post-reboot helper/session recovery evidence recorded. | Manual operator procedure only |

## Expected Skip Taxonomy

These states are expected skips or setup gaps, not runtime regressions by
themselves:

- ordinary PR checks do not include the host-gated workflow
- scheduled workflow skipped because
  `TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY` is unset or not `1`
- workflow skipped on a ref other than `main` or `dev`
- hosted CI lacks Apple silicon `Virtualization.framework`
- local machine lacks a prepared bundle, helper, Xcode tools, or entitlements
- failure drills skipped because `include_failure_drills=true` was not requested
- managed helper `restart-drill` skipped because the helper was not started by
  `vz-helperctl.py start`
- `launchd-drill` skipped because no maintainer requested LaunchAgent validation
- host reboot validation skipped because it remains a manual operator procedure

If a prepared host passes preflight and then fails helper startup, real
ephemeral execution, same-session VM reuse, recovery diagnostics, cleanup, or
artifact retention, record it as a potential blocking regression and link the
triage issue.

## Latest Evidence

No prepared-host evidence packet is currently recorded in this tracker. The
next maintainer run should add a dated entry under this section or link a
GitHub Actions run with the packet fields above.

### Template

```markdown
### YYYY-MM-DD: <source> on <branch>@<sha>

- Evidence source:
- Operator or workflow run:
- Host identity:
- Host prep:
- Bundle/template:
- Helper build/signing:
- Runtime paths:
- Commands:
- Results:
- Failure drills:
- Launchd drill:
- Artifacts:
- Expected skips:
- Blocking regressions:
- Residual gaps:
- Follow-up owner:
```

## Current Residual Gaps

| Gap | Current status | Next action |
| --- | --- | --- |
| Prepared-host default smoke evidence | Tracker exists, but no dated evidence packet has been recorded here yet. | Run local or host-gated smoke on a prepared Apple silicon host and add the evidence packet. |
| Failure-drill evidence | Manual opt-in only. | Record results when a maintainer runs with `include_failure_drills=true`. |
| Launchd-drill evidence | Manual opt-in only. | Record results only when a runner is intentionally configured for LaunchAgent validation. |
| Host reboot recovery | Manual operator procedure only and out of scheduled CI. | Add a dedicated operator drill once a prepared host can tolerate disruptive reboot testing and preserve logs. |
| Stuck boot/readiness and guest-agent mismatch | Not covered by the default smoke. | Add narrow manual drills or diagnostics checks before considering automated coverage. |
| Stale socket handling | Covered by helper lifecycle docs and tests, but not yet recorded as prepared-host evidence in this tracker. | Include socket-path cleanup and status output in the next evidence packet. |

## Recording Guidance

For a local prepared-host run, prefer the managed helper wrapper:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py smoke \
  --bundle /path/to/canonical/bundle \
  --entitlements /path/to/helper.entitlements
```

For a lower-level run, use a private runtime directory and cleanup trap:

```bash
runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/tldw-vz-helper-e2e.XXXXXX")"
chmod 700 "$runtime_dir"
trap 'rm -rf "$runtime_dir"' EXIT

tools/vz-linux-image/scripts/run-host-e2e-smoke.sh \
  --bundle /path/to/canonical/bundle \
  --socket "$runtime_dir/helper.sock" \
  --serial-log-dir "$runtime_dir/serial" \
  --entitlements /path/to/helper.entitlements
```

For host-gated CI, record the workflow run URL, runner labels, branch/ref, input
values, artifact names, and any expected skips. The workflow must remain
manual/nightly only and must not be promoted into normal PR-triggered CI.
