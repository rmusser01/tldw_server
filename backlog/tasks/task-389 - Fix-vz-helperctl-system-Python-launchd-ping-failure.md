---
id: TASK-389
title: Fix vz-helperctl system Python launchd ping failure
status: Done
assignee: []
created_date: '2026-05-15 19:56'
updated_date: '2026-05-16 01:03'
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

2026-05-15 review pass: PR #1731 has two actionable Qodo review threads. Scope: add a docstring for _request_helper_ping and normalize direct socket transport/empty/invalid JSON failures to stable macos_virtualization_helper_* messages before resolving review threads.

2026-05-15 review fix: Added _request_helper_ping docstring and stable-message regressions for empty response, invalid JSON, and missing helper socket. Normalized direct socket transport failures to macos_virtualization_helper_unavailable, empty responses to macos_virtualization_helper_empty_response, and decode/JSON failures to macos_virtualization_helper_invalid_json.

Review-fix verification: focused review tests passed 3 passed; full helperctl pytest passed 129 passed, 1 skipped; direct launchd-drill --skip-smoke passed bootstrap/status/kickstart/helper_status/protocol/version/bootout; git diff --check passed; Bandit JSON at /tmp/bandit_vz_helperctl_python39_ping_review_fix.json reported errors=0 and results=0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the system-Python launchd ping failure and the PR #1731 review findings. vz-helperctl now pings the helper directly over the Unix socket without importing server modules, documents that helper, and normalizes transport/protocol parse failures to stable macos_virtualization_helper_* messages. Regression coverage verifies the import-break fallback and stable messages for missing socket, empty response, and invalid JSON. Verified full helperctl tests, direct launchd-drill, git diff --check, and Bandit.
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
