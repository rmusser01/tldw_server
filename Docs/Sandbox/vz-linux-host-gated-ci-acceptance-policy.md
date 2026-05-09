# VZ Linux Host-Gated CI Acceptance Policy

**Status:** Active policy for real `vz_linux` Apple silicon CI.
**Date:** 2026-05-03.
**Workflow:** `.github/workflows/vz-linux-host-gated.yml`.

## Purpose

Real `vz_linux` execution depends on Apple silicon macOS,
`Virtualization.framework`, a prepared helper build/signing environment, and a
canonical Linux bundle on the runner. That cannot be part of normal hosted CI.

This policy defines when the host-gated workflow should run, what it proves,
which skips are expected, and what counts as a blocking regression.

## Entry Points

The workflow has only two triggers:

| Trigger | Runs when | Intended use |
| --- | --- | --- |
| `workflow_dispatch` | A maintainer starts it manually on `main` or `dev`. | Validate a candidate `dev` state or investigate a real-host regression. |
| `schedule` | The repository variable `TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY=1` is set and the ref is `main` or `dev`. | Nightly prepared-host acceptance coverage. |

The workflow must not run on pull request events or normal pushes. Normal CI
must remain portable and should continue to exercise fake/scaffolded paths,
workflow contract tests, and host-independent unit tests.

## Host And Repository Requirements

The runner must be a prepared self-hosted Apple silicon macOS runner with these
labels:

- `self-hosted`
- `macOS`
- `ARM64`
- `vz-linux`

The runner must provide:

- SwiftPM and Xcode command line tools
- `codesign` through `xcrun`
- a canonical `vz_linux` bundle containing `kernel` and `rootfs.img`
- permission to create private runtime directories, Unix sockets, serial logs,
  and Virtualization.framework VMs

The workflow needs either the manual input `bundle_path` or repository variable
`TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH`.

Optional configuration:

- `TLDW_SANDBOX_VZ_HELPER_ENTITLEMENTS_PATH` or manual `entitlements_path`
- `TLDW_SANDBOX_VZ_HELPER_SKIP_SIGN`
- `TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY=1` to enable scheduled runs
- manual `include_failure_drills=true` to run VM-invalidation recovery drills
  scoped to VMs created by the drill itself

Manual `skip_sign: false` must override a true repository variable. This keeps
one-off signing validation possible without changing repository configuration.
Failure drills are never enabled by repository variable or schedule in the
current policy.

## Branch And Action Safety

The job is branch-gated to `refs/heads/main` and `refs/heads/dev`. Manual
dispatch must not execute arbitrary feature-branch code on the prepared host.

External actions in this workflow must be pinned to immutable SHAs. The
workflow permissions must remain minimal:

```yaml
permissions:
  contents: read
```

The job should delegate VM work to:

```bash
tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
```

That keeps CI aligned with the operator workflow instead of creating a separate
helper lifecycle path.

## What The Workflow Proves

The host-gated smoke is an acceptance gate for the prepared-host path. It must
cover:

- prepared Apple silicon host validation
- helper build and optional signing
- helper daemon bundle smoke
- helper startup on a private Unix socket
- real `vz_linux` ephemeral execution
- same-session VM reuse with a second command in the same VM
- recovery diagnostics plus dry-run reconciliation repair planning
- helper shutdown/cleanup on exit
- helper stdout/stderr and serial-log artifact upload

The recovery smoke is intentionally non-destructive. It verifies that a prepared
host can compute macOS sandbox reconciliation state and produce a dry-run repair
plan for stale VZ session-control metadata. It does not terminate VMs, delete
session controls, or run image-store cleanup.

Manual failure drills are separate from the default smoke. When a maintainer
starts the workflow with `include_failure_drills=true`, the delegated smoke
script runs additional tests that may invalidate VMs created by those tests. The
current drill terminates only the session VM recorded by the drill's own
session-control row, then verifies the next same-session command provisions a
fresh VM and completes. Scheduled runs must not enable these drills by default.

## Expected Skips And Non-Blocking Conditions

These are expected and should not block ordinary PRs:

- workflow absent from pull request checks
- scheduled workflow skipped because
  `TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY` is unset or not `1`
- workflow skipped on refs other than `main` and `dev`
- normal hosted CI lacking Apple silicon VZ support
- local developer machines without a prepared bundle/helper environment
- failure-drill coverage skipped because manual dispatch did not set
  `include_failure_drills=true`

A manual run that fails before VM execution because the runner is missing the
configured bundle path is an operator setup failure, not a sandbox runtime
regression. Fix the runner or repository variable before treating runtime code
as suspect.

## Blocking Regression Criteria

A host-gated run is a blocking regression for `vz_linux` when it runs on a
prepared host and fails one of the accepted runtime guarantees:

- helper cannot build, sign, start, or answer daemon smoke with the documented
  configuration
- canonical bundle validation fails for a bundle that previously passed on the
  same runner
- real ephemeral command execution fails after helper/template readiness passes
- same-session VM reuse fails after the first command succeeds
- recovery diagnostics or dry-run reconciliation repair planning fails after
  helper/template readiness passes
- a manually requested failure drill cannot replace a drill-owned stale session
  VM after helper-side termination
- cleanup leaves the helper process or accepted socket path behind
- helper protocol mismatch is introduced without a matching compatibility plan
- artifacts/logs are not uploaded when the job fails

Failures caused by missing runner labels, absent bundles, unavailable Xcode
tools, or disabled nightly opt-in are host-preparation failures. They should be
reported clearly but should not block unrelated code unless the change under
review modified the host-gated workflow, smoke script, helper lifecycle, or
bundle contract.

## Artifact And Log Expectations

The workflow must upload helper logs from:

```text
${{ runner.temp }}/tldw-vz-helper-ci/**
```

The upload step must run with `if: always()` and `if-no-files-found: ignore` so
failed early setup still preserves available logs without creating noisy
secondary failures.

Uploaded logs are for operator debugging. They must not be treated as public API
output and should not contain secrets or raw user data.

## Maintenance Rules

- Keep this policy aligned with `.github/workflows/vz-linux-host-gated.yml`.
- Update `tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py`
  when changing workflow triggers, labels, action pins, permissions, artifact
  paths, failure-drill opt-in behavior, or nightly opt-in behavior.
- Do not add PR or push triggers without a new security review.
- Do not remove branch gating without replacing it with an equivalent trusted
  ref policy.
- Prefer improving `run-host-e2e-smoke.sh` over adding ad hoc workflow-only
  helper lifecycle logic.
- Keep failure drills disabled by default until repeated prepared-host runs show
  they are stable enough for scheduled promotion.
