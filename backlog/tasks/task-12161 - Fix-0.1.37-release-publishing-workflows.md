---
id: TASK-12161
title: Fix 0.1.37 release publishing workflows
status: Done
priority: High
references:
- https://github.com/rmusser01/tldw_server/actions/runs/28766551634
- https://github.com/rmusser01/tldw_server/actions/runs/28766485683
modified_files:
- .github/workflows/publish-docker.yml
- .github/workflows/publish-pypi.yml
- tldw_Server_API/tests/CI/test_release_workflow_contracts.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve release publishing failures: publish-docker should publish to GHCR without requiring Docker Hub credentials, and publish-pypi should install PortAudio before installing dev dependencies that build PyAudio.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Release Docker publishing no longer requires Docker Hub credentials.
- [x] PyPI publishing installs PortAudio before installing `.[dev]`.
- [x] Workflow contract tests cover both regressions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root causes verified from failed release logs and workflow files: `publish-docker.yml` unconditionally logged in to Docker Hub and included Docker Hub image metadata; `publish-pypi.yml` installed `.[dev]` without the PortAudio setup used by other CI workflows, causing PyAudio to fail on missing `portaudio.h`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Removed Docker Hub login/image publication from the Docker release workflow so releases publish only to GHCR.
- Added PortAudio setup before PyPI release test-suite dependency installation so PyAudio can build on hosted Linux runners.
- Added workflow contract tests covering both release publishing regressions.
- Verification: `python -m pytest -q tldw_Server_API/tests/CI/test_release_workflow_contracts.py`; `python -m pytest -q tldw_Server_API/tests/CI/test_required_workflow_contracts.py::test_setup_ffmpeg_action_can_skip_ffmpeg_but_keep_portaudio`; `git diff --check`; Bandit on the changed CI test reported only existing pytest `assert` B101 findings.
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
