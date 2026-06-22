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
Task 4 completed in this worktree: added MCP Unified standalone RC Make targets (`mcp-unified-build`, `mcp-unified-check`, `mcp-unified-uat`, `mcp-unified-rc`), created the private `.github/workflows/mcp-unified-rc.yml` workflow, retired MCP-specific artifact-gate work from the root PyPI package-check workflow, and renamed root PyPI artifacts to `tldw-server-pypi-dist`. Verification: focused Task 4 boundary pytest passed (3 passed, 4 warnings); Make dry-runs for all four MCP targets expanded to the expected `Helper_Scripts/mcp_unified_rc.py` phases; `git diff --check` passed; Bandit on `test_runtime_package_boundary.py` wrote `/tmp/bandit_mcp_unified_task4.json` and exited nonzero only for the existing low-severity test-file baseline, with no findings in the new Task 4 line range.
Task 4 spec-review fixes: removed editable `apps/mcp-unified[dev]` install from the private RC workflow and expanded boundary tests to assert no editable RC workflow install, exact Make target phase calls, and root PyPI package-check separation from MCP artifact gates/triggers. Verification: expanded Task 4 pytest passed (4 passed, 4 warnings); Make dry-runs for all four MCP targets passed.
Task 4 spec re-review fixes: tightened boundary helpers to parse exact Make target command bodies, catch both `-e` and `--editable` RC workflow installs, and guard the old root `mcp_unified/**` trigger/editable-install path in the root PyPI package-check workflow. Verification: expanded Task 4 pytest passed (4 passed, 4 warnings); `git diff --check` passed; Bandit on `test_runtime_package_boundary.py` still reports only existing low-severity baseline findings.
Task 4 quality-review fixes: added `Makefile` to the private RC workflow trigger paths, installed standalone runtime dependencies explicitly in CI without editable-installing the package, and removed the brittle EOF-wide Makefile section scan in favor of exact target body assertions. Verification: expanded Task 4 pytest passed (4 passed, 4 warnings); Make dry-runs for all four MCP targets passed; `git diff --check` passed; Bandit on `test_runtime_package_boundary.py` still reports only existing low-severity baseline findings.

Task 5 completed in this worktree: updated the standalone user-guide UAT harness to install either the non-editable apps/mcp-unified[gateway] project, an editable local project for guide iteration, or a built wheel for installed-artifact UAT. Added RC harness coverage that loads the hyphenated Testing-related UAT script by file path via importlib and sys.modules. Wheel install args take precedence over --editable when both are supplied. Verification: focused RC harness pytest passed (7 passed, 4 warnings); UAT harness --help passed and shows --wheel/--editable; Bandit on touched files passed with zero findings; git diff --check passed.
Task 5 spec-review hardening: wheel install paths are resolved before being passed to pip so relative `--wheel` arguments remain valid after the UAT harness switches into its isolated workspace. Verification: focused RC harness pytest passed (7 passed, 4 warnings); `git diff --check` passed.
Task 5 quality-review fixes: UAT result `reason` values are now redacted in report payloads, and RC harness tests assert package install args reach the generated pip install step. Verification: focused RC harness pytest passed (9 passed, 4 warnings); UAT harness `--help` passed; Bandit on touched files passed with zero findings; `git diff --check` passed.

Task 6 validation/debugging completed in this worktree: full `make mcp-unified-rc` initially exposed macOS copied-venv `ensurepip` failures and restricted-network dependency resolution failures. The RC harness now creates POSIX venvs with symlinks to avoid copied-interpreter `ensurepip` aborts, records local dependency-index outages as optional skips only for dependency-resolving pip installs, and remains fail-closed for the same failures under GitHub Actions. Strict no-deps installed-wheel boundary checks remain required and passed. Verification: focused RC harness pytest passed (12 passed, 4 warnings); runtime package boundary pytest passed (39 passed, 5 warnings); artifact-gate pytest passed (4 passed); compileall passed; `make mcp-unified-rc` passed with 15 passed, 10 optional skipped, 0 failed; Bandit on touched helper/UAT/test files passed with zero findings; `git diff --check` passed.

Spec-review remediation completed after final branch review: RC `cli-uat` and `smoke-uat` now invoke the wheel-mode standalone user-guide UAT harness, including the documented CLI workflows and fixture-backed stdio/HTTP/WebSocket smoke steps when dependency resolution is available. The UAT harness installs wheel mode with the `gateway` extra and creates POSIX venvs with `--symlinks`. RC and UAT evidence now redact repo, temp, home/cache, interpreter, and Windows-style absolute paths. Local restricted-network dependency-index failures remain optional skips outside GitHub Actions; CI remains fail-closed. Verification: new TDD regressions failed before implementation and then passed; RC harness pytest passed (18 passed, 4 warnings); runtime package boundary pytest passed (39 passed, 5 warnings); artifact-gate pytest passed (4 passed); compileall passed; full `make mcp-unified-rc` passed with 13 passed, 9 optional skipped, 0 failed; all generated JSON evidence artifacts were scanned for `/Users/`, `/private/var/folders`, and `/var/folders` with no hits; Bandit on touched helper/UAT/test files passed with zero findings; `git diff --check` passed.
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
