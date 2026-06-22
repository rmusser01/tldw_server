---
id: TASK-2399
title: Implement MCP Unified internal RC artifact pipeline
status: In Progress
labels:
- mcp
- packaging
- uat
- release
- implementation
documentation:
- Docs/superpowers/specs/2026-06-22-mcp-unified-internal-rc-artifact-pipeline-design.md
- Docs/superpowers/plans/2026-06-22-mcp-unified-internal-rc-artifact-pipeline-implementation-plan.md
modified_files:
- apps/mcp-unified
- Helper_Scripts/mcp_unified_rc.py
- Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
- tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py
- .github/tests/test_mcp_unified_artifact_gate.py
- Makefile
- .github/workflows/pypi-package.yml
- .github/workflows/publish-pypi.yml
- .github/workflows/mcp-unified-rc.yml
- pyproject.toml
- Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved implementation plan for moving the standalone MCP package under apps/mcp-unified, updating artifact boundary tests, adding the private RC harness, Make targets, CI workflow, installed-wheel UAT, and validation/security checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation started using subagent-driven development after the design/spec and implementation plan were approved and re-reviewed. Worktree: codex/mcp-unified-internal-rc-spec.

Task 1 worker completed package relocation in commit `92283dc17f`. Focused relocation tests passed: 3 passed, 4 warnings, using the root project virtualenv. The worker added root pytest `pythonpath` for `apps/mcp-unified/src`, which appears necessary because many MCP tests import `mcp_unified` directly after the root package directory is removed. Duplicate worker-created task `TASK-2400` was removed as redundant with `TASK-2399`.

Task 1 spec-review fixes landed in commit `4bc8507199`: subprocess import environments now include the relocated standalone src path, artifact build/install helpers copy from `apps/mcp-unified`, and artifact-gate workflow paths point at `apps/mcp-unified/pytest-artifact-gate.ini`. Worker reported `test_runtime_package_boundary.py`: 35 passed, 5 warnings. Duplicate worker-created task `TASK-2400` was removed again as redundant with `TASK-2399`.

Task 1 docs/UAT path fixes landed in commit `c74b870faf`: README, user guide, package-resource docs, and the standalone user-guide UAT helper now use `apps/mcp-unified` paths. Worker reported docs boundary test passed, UAT helper `--help` passed, and Bandit on the UAT helper passed.

Task 1 root packaging/tooling cleanup landed in commit `974114f2da`: root `pyproject.toml` no longer advertises standalone MCP console scripts, root package discovery no longer includes `mcp_unified`, Ruff/mypy paths point at the app package source, and admin docs use app package paths. Worker reported `test_runtime_package_boundary.py`: 36 passed, 5 warnings. Task 1 passed spec review and code-quality review after these fixes.
Task 2 completed in this worktree: tightened MCP Unified standalone artifact boundary tests for the apps/mcp-unified project root, added normalized sdist project-root member checks, asserted the built-wheel mcp-unified-smoke console script, and updated the artifact-gate shim wording. Verification: selected pytest nodeids passed (5 passed, 4 warnings); artifact-gate shim pytest passed (4 passed, 2 warnings); git diff --check passed. Bandit was run on the two touched files and exited nonzero due the existing low-severity test-module baseline findings (B101/B404/B603/B105); the new Task 2 assertion lines are nosec-marked and did not add report entries.
Task 2 spec-review hardening: exact standalone setuptools package-list assertion added after reviewer noted the previous package assertion was partial. Verification: `test_mcp_unified_standalone_pyproject_matches_release_metadata` passed.
Task 3 completed in this worktree: added `Helper_Scripts/mcp_unified_rc.py` internal RC harness and `test_mcp_unified_rc_harness.py` coverage for canonical apps/mcp-unified paths, secret redaction, evidence JSON/Markdown summaries, required-failure handling, and artifact SHA256 recording. Verification: focused harness pytest passed (5 passed, 4 warnings); harness plus package-location boundary pytest passed (6 passed, 4 warnings); `python Helper_Scripts/mcp_unified_rc.py evidence --help` passed; `python Helper_Scripts/mcp_unified_rc.py build` passed and produced wheel/sdist hashes under `.artifacts/mcp-unified-rc`; `python Helper_Scripts/mcp_unified_rc.py artifact-gate` passed with twine check and artifact-gate pytest (4 passed); Bandit on touched helper/test files passed with zero findings.
Task 3 quality-review hardening: `run_command()` now clears inherited `PYTHONPATH` unless an individual command supplies an explicit override, so installed-wheel checks cannot accidentally import the checkout. Verification: harness pytest passed (6 passed, 4 warnings); `mcp_unified_rc.py artifact-gate` passed; Bandit on touched helper/test files passed with zero findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
