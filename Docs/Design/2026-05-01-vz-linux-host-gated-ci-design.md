# vz_linux Host-Gated CI Design

**Date:** 2026-05-01
**Status:** Implementation slice

## Goal

Add a repeatable CI entrypoint for the real `vz_linux` macOS execution path
without making normal CI depend on Apple Silicon hardware.

The workflow should prove the same path operators run locally:

- prepared Apple silicon macOS host
- canonical `vz_linux` bundle already present on that host
- Swift helper build and optional ad hoc signing
- helper daemon startup over an owner-private Unix socket
- helper bundle smoke
- real ephemeral `vz_linux` execution
- same-session VM reuse
- helper shutdown and log collection

## Approach

Add a dedicated GitHub Actions workflow that runs only on self-hosted runners
with labels:

- `self-hosted`
- `macOS`
- `ARM64`
- `vz-linux`

The workflow supports manual dispatch with a `bundle_path` input. A scheduled
trigger is present but the job is skipped unless repository variable
`TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY=1` is set. That preserves portable
default CI while allowing a prepared private runner to catch regressions.

The workflow delegates to the existing operator smoke script instead of
duplicating helper lifecycle logic. This keeps one source of truth for build,
sign, start, real execution, session reuse, cleanup, and logs.

## Non-Goals

- Provisioning the Apple Silicon runner.
- Building the canonical Debian bundle in CI.
- Installing launchd services automatically.
- Requiring this workflow as a branch-protection gate.
- Adding APFS clone provisioning or network allowlist enforcement.

## Success Criteria

- Normal hosted CI remains portable.
- Operators can run the workflow manually against a prepared host bundle.
- Nightly runs are opt-in through a repository variable.
- Workflow shape is covered by a static infrastructure test.
- Operator docs describe runner labels, required repository variables, and the
  exact smoke path used by CI.
