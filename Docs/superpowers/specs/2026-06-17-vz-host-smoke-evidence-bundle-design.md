# VZ Host Smoke Evidence Bundle Design

**Date:** 2026-06-17
**Status:** Implementation slice
**Task:** `TASK-2368`

## Goal

Make prepared-host VZ Linux smoke evidence repeatable by having the operator
wrapper write a small structured evidence bundle automatically. The bundle
should be useful for local runs and host-gated workflow artifacts without
requiring manual hash/stat collection after every smoke.

## Scope

This slice adds default-on evidence capture to
`tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`.

Default evidence output is:

```text
<runtime-dir>/evidence
```

Operators may override it with:

```bash
--evidence-dir PATH
```

The current host-gated workflow already uploads
`${{ runner.temp }}/tldw-vz-helper-ci/**`, so evidence under the runtime
directory is retained without adding a new action or widening permissions.

## Evidence Files

The wrapper writes these files on real runs:

- `host-smoke-evidence.json`: schema version, timestamps, source bundle,
  image-store root, smoke run id, disposable run bundle, helper path, signing
  flags, failure-drill flag, runtime paths, phase outcomes, cleanup state, and
  log artifact pointers.
- `source-bundle-hashes-before.txt` and `source-bundle-hashes-after.txt`:
  SHA-256 hashes for source bundle files relevant to boot and provenance.
- `run-bundle-hashes.txt`: SHA-256 hashes for the disposable run bundle after
  materialization and smoke execution.
- `runtime-paths.txt`: socket, serial directory, image-store root, evidence
  directory, helper pid file, and owner/mode metadata.
- `cleanup-status.txt`: helper PID status, accepted socket status, cleanup
  result, and final smoke exit code.

The JSON must not include raw helper stdout/stderr or serial log contents. It
may include paths, sizes, and SHA-256 values for those files. Hash and JSON
generation should use a standard-library Python interpreter rather than shell
string assembly, so path quoting remains safe and portable.

## Path Safety

Evidence directory handling follows the same trust-boundary posture as socket
and serial directories:

- refuse symlink paths
- refuse existing non-directory paths
- create missing directories with mode `0700`
- require current-user ownership
- require no group/world permissions

Dry-run validates and prints the resolved evidence directory and planned file
names, but it does not create files or directories.

## Phase Tracking And Exit Semantics

The wrapper records phase state incrementally:

- `validate_inputs`
- `prepare_runtime_paths`
- `prepare_smoke_bundle`
- `helper_daemon_smoke`
- `start_helper`
- `wait_for_helper_socket`
- `real_host_smoke`
- `failure_drills`
- `cleanup`
- `evidence_finalize`

The `EXIT` trap must preserve the original smoke exit code. Cleanup and
evidence finalization run after failure when possible, but they must not hide a
real smoke failure. If the smoke succeeded and evidence finalization fails, the
wrapper should exit non-zero because the host-gated acceptance policy treats
missing artifacts as a regression.

## Tests

Portable tests should cover:

- dry-run mentions the default evidence directory and planned evidence files
  without creating them
- `--evidence-dir PATH` override is propagated
- fake-helper real run creates the expected evidence files
- evidence directory hardening rejects unsafe paths
- a late fake pytest failure preserves the failing exit code while still
  writing cleanup/evidence status
- workflow contract expects the runtime artifact upload to retain evidence

Real VZ execution remains host-gated/manual and is not required for normal CI.

## Non-Goals

- Adding a separate upload-artifact action in this slice.
- Parsing raw serial logs into JSON.
- Changing VM provisioning, helper lifecycle, or image-store GC behavior.
- Making host-gated VZ smoke part of normal PR CI.
