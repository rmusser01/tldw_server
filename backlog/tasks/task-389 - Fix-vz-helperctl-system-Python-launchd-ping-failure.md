---
id: TASK-389
title: Fix vz-helperctl system Python launchd ping failure
status: Done
assignee: []
created_date: '2026-05-15 19:56'
updated_date: '2026-05-15 20:07'
labels:
  - sandbox
  - macos-vz-helper
  - launchd
  - cli
dependencies: []
documentation:
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - tools/macos-vz-helper/scripts/vz-helperctl.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The merged launchd-drill operator path works when run with the project Python, but direct script invocation on this macOS host used system Python 3.9 and failed helper ping with dataclass(slots=...) import incompatibility. Harden vz-helperctl so documented direct CLI usage does not fail before helper readiness validation on Python 3.9, or make the failure explicit and operator-safe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Direct vz-helperctl launchd-drill/check paths do not crash or report helper ping failure because helper_client cannot import on Python 3.9.
- [x] #2 Helper ping/status behavior remains protocol-compatible on supported project Python versions.
- [x] #3 Focused helperctl regression coverage captures the Python 3.9 import fallback or explicit unsupported-interpreter behavior.
- [x] #4 Operator docs or CLI guidance are updated if direct script invocation requires the project Python.
- [x] #5 Focused helperctl tests pass and the touched helperctl code passes git diff --check and Bandit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-15: Reproduced the host failure during local launchd-drill validation: direct script invocation used macOS system Python 3.9 and helper_status failed with dataclass(slots=...) import incompatibility. Added a failing regression that blocks helper_client import and proves ping_helper_state can still ping a reachable helper socket through the operator CLI path.

Implemented a narrow direct socket ping in vz-helperctl.py for default helper readiness checks. client_factory remains available for tests. Verified focused regression, launchd/ping helperctl slice, full helperctl tests, direct launchd-drill with the documented script invocation, git diff --check, and Bandit. No docs change was needed because direct script invocation now works without requiring project Python.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the system-Python launchd ping failure discovered during local validation. vz-helperctl now performs helper ping directly over the Unix socket instead of importing the full server helper client for operator readiness checks, avoiding Python 3.9 dataclass(slots=...) import incompatibility while preserving protocol/version validation and test injection through client_factory. Verified with full helperctl tests, direct launchd-drill execution, git diff --check, and Bandit.
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
