---
id: TASK-12767
title: Record VZ image-store smoke clone evidence
status: Done
labels:
- sandbox
- vz-linux
- operator-evidence
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the current prepared-host evidence gap by running or documenting the image-store-backed VZ Linux smoke path, proving the canonical source bundle remains immutable while execution uses a disposable run bundle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backlog task and implementation notes reference the current prepared-host evidence tracker gap.
- [x] #2 Local prepared-host smoke uses the image-store/disposable-clone path rather than direct bundle mutation, or records a concrete blocker if the host is not ready.
- [x] #3 Evidence records source bundle hash before/after, disposable run bundle hash/path, helper/socket/log paths, command output summary, and explicit skips for manual drills not requested.
- [x] #4 Docs update the prepared-host evidence tracker without weakening host-gated/manual-only policy.
- [x] #5 Verification includes focused docs/script checks, diff check, and Bandit skip/rationale if only docs/evidence files are touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-20: Initial stale-socket target was already implemented on `origin/dev`
  by PR `#1856`; retargeted this task to the open direct-bundle mutability
  evidence gap in `Docs/Sandbox/vz-linux-prepared-host-evidence.md`.
- Real smoke dry-run with default `python3` failed because macOS system Python
  3.9 cannot import the repo's Python 3.10+ dataclass usage; reran dry-run with
  `<repo>/.venv/bin/python`.
- First real smoke attempt built/signed the helper and passed helper-daemon
  smoke, but the long `${TMPDIR}`-style socket path failed helper startup with
  `socketPathTooLong`. Added a regression test and changed the wrapper default
  runtime root to a short `/tmp/tvz-e2e-*` path.
- PR review follow-up hardened the wrapper default further to create a short
  random `mktemp -d /tmp/tvz-e2e.XXXXXX` runtime directory instead of a
  PID-derived directory.
- A first default-path real smoke reached the real host smoke tests but had one
  non-reproduced stdout-buffer assertion failure while the other selected tests
  passed; rerunning the same default-path command passed.
- Successful default-path prepared-host smoke used runtime root
  `/tmp/tvz-e2e-25415`, source bundle `/private/tmp/tldw-vz-bundle`,
  disposable run bundle
  `/private/tmp/tvz-e2e-25415/image-store/runs/host-smoke-25415/bundle`, and
  evidence directory `/tmp/tvz-e2e-25415/evidence`.
- Successful default-path result: helper daemon smoke `2 passed`; real
  `vz_linux` host smoke `3 passed, 11 deselected`; source bundle hashes
  identical before/after; run bundle rootfs hash differed after execution as
  expected.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Recorded a 2026-06-20 prepared-host evidence packet proving the image-store
  smoke path keeps the canonical source bundle immutable while the disposable
  run bundle absorbs VM writes.
- Fixed the lower-level smoke wrapper default runtime path to use short
  random `mktemp -d /tmp/tvz-e2e.XXXXXX` directories, avoiding macOS AF_UNIX
  `socketPathTooLong` failures from long per-user `${TMPDIR}` paths and avoiding
  predictable PID-derived paths in world-writable `/tmp`.
- Updated the smoke README and evidence tracker guidance to recommend short
  runtime paths for helper sockets.
- Addressed PR review follow-up by adding the missing regression-test docstring,
  rejecting relative or filesystem-root default runtime roots, and redacting
  newly added local-machine evidence paths.
- Verification:
  `python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q --tb=short`
  passed with `28 passed, 3 skipped`;
  `bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh` passed;
  smoke dry-run showed default `/tmp/tvz-e2e.XXXXXX/helper.sock`;
  real prepared-host smoke using the new default path passed helper daemon smoke
  `2 passed` and real `vz_linux` host smoke `3 passed, 11 deselected`;
  `git diff --check` passed.
- Bandit on the touched Python test file still reports the existing baseline
  test-harness findings, but current vs `origin/dev` is unchanged:
  `122` findings in both (`B404`: 1, `B101`: 120, `B603`: 1).
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
