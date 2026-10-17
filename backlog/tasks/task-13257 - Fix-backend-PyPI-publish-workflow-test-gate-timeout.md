---
id: TASK-13257
title: Fix backend PyPI publish workflow test gate timeout
status: Done
created_date: 2026-09-13 20:17
labels:
- ci
- pypi
- release
priority: High
references:
- https://github.com/rmusser01/tldw_server/actions/runs/29559405511
modified_files:
- .github/workflows/publish-pypi.yml
- tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py
- tldw_Server_API/tests/CI/test_release_workflow_contracts.py
updated_date: 2026-09-13 20:34
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the backend PyPI publish workflow so main-branch version publishes are still gated by meaningful validation but no longer blocked by the flaky/long full pytest suite timeout observed in run 29559405511.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Publish workflow keeps an automated validation gate before build/publish.
- [x] #2 Publish workflow no longer runs the entire backend pytest suite as the release-only gate.
- [x] #3 Workflow contract tests cover the intended publish gate command.
- [x] #4 Changed workflow/test scope validates locally.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented from `origin/dev` in `codex/pypi-publish-gate-timeout`. Root cause evidence: publish workflow run 29559405511 reached the release path, then failed in the full-suite `python -m pytest -q` gate before build/publish. The package build/check path itself validates successfully with `make pypi-check`. Fix narrows the publish workflow gate to PyPI workflow contract tests plus the existing bounded minimal app startup smoke, and keeps `make pypi-check` as the artifact validation gate before publication. Verification: red contract test failed against the original workflow; affected contract tests now pass; `git diff --check` passes; `make pypi-check` passes for 0.1.41; minimal startup smoke passes when localhost binding is allowed. Bandit was run on touched Python test files; the new/changed pytest assertions have targeted `B101` suppressions, and the remaining reported `B101` entries are the existing pytest assertion baseline in those test files.
Known skips/caveats: local `actionlint` is unavailable, so workflow syntax was validated through YAML-loading contract tests and GitHub's actionlint check. The first local minimal startup smoke run failed because sandboxing blocked localhost binding; rerunning the same smoke outside the sandbox passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Release publish workflow now uses a targeted `release-gate` job instead of running the full backend pytest suite. The gate executes the PyPI workflow contract tests and a bounded minimal startup smoke, then the existing build job runs `make pypi-check` before any TestPyPI/PyPI publish job can run.
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
