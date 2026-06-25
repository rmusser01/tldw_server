---
id: TASK-2402
title: Fix MCP Unified TestPyPI wheel missing policy grants package
status: Done
labels:
- mcp
- packaging
- testpypi
- bug
priority: high
modified_files:
- apps/mcp-unified/pyproject.toml
- apps/mcp-unified/src/mcp_unified/__init__.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
- backlog/tasks/task-2402 - Fix-MCP-Unified-TestPyPI-wheel-missing-policy-grants-package.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TestPyPI smoke for mcp-unified 0.1.0 uploaded successfully, but installed console scripts fail with ModuleNotFoundError for mcp_unified.policy_grants because the standalone package manifest omits that package. Add package-boundary coverage and update the package manifest so built wheels include policy_grants.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Package boundary tests fail before the manifest fix when policy_grants is missing from built artifacts.
- [x] #2 apps/mcp-unified packaging manifest includes mcp_unified.policy_grants.
- [x] #3 Built wheel and sdist include policy_grants package files.
- [x] #4 Fresh local wheel install runs mcp-unified-gateway package-info and mcp-unified-smoke --help successfully.
- [x] #5 TestPyPI smoke outcome is recorded without exposing token values.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TestPyPI smoke found a real packaging issue after uploading `mcp-unified 0.1.0`: installed console scripts failed with `ModuleNotFoundError: No module named 'mcp_unified.policy_grants'`. The root cause was the manual setuptools package list in `apps/mcp-unified/pyproject.toml`, which omitted `mcp_unified.policy_grants` even though gateway bootstrap imports it.

Fix:
- Added package-boundary expectations for `mcp_unified.policy_grants` in the standalone pyproject package list and built wheel/sdist members.
- Added `mcp_unified.policy_grants` to the standalone package manifest.
- Bumped the standalone package version to `0.1.1` because TestPyPI does not allow replacing the already-uploaded broken `0.1.0` artifacts.

Verification:
- Red test before fix: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q -k "pyproject_matches_release_metadata or artifacts_include_package_docs"` failed on the missing package list entry and missing wheel member.
- Green test after fix: same command passed, 2 passed / 44 deselected.
- Full RC: `make PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python mcp-unified-rc` exited 0 with RC status `ok`; evidence reported `mcp-unified 0.1.1`, wheel sha256 `0a8187e94d4931ff38ac01138cca737dbb53d7764eca23f1254a89a19ef76601`, sdist sha256 `ff821cf00926f2bc547875046caca2b49711926cad7e27d762b968ceb97ed9b4`.
- TestPyPI upload: guarded live upload using local token file exited 0 and wrote evidence for `https://test.pypi.org/project/mcp-unified/0.1.1/` without logging token values.
- TestPyPI install smoke: fresh venv installed dependencies from PyPI, then installed `mcp-unified==0.1.1` from TestPyPI with `--no-deps`; import printed `0.1.1`; `mcp-unified-gateway package-info`, `mcp-unified-gateway list-presets`, `mcp-unified-smoke --help`, and `mcp-unified-smoke inprocess --json-report -` exited 0.
- First naive TestPyPI install attempt with TestPyPI as primary index failed because pip selected a broken TestPyPI `FASTAPI-1.0` dependency. The successful smoke used the safer pattern: install dependencies from PyPI, then install the TestPyPI wheel with `--no-deps`.
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r apps/mcp-unified/src/mcp_unified/__init__.py -f json -o /tmp/bandit_mcp_unified_testpypi_packaging.json` exited 0 with zero findings.

Known follow-up:
- Installed CLI commands emit a non-fatal Pydantic warning from `mcp_unified.gateway.snapshots.GatewayConfigSnapshot` because the `schema` field shadows a BaseModel attribute. It does not block the smoke, but should be cleaned up separately if warning-clean CLI output is required.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Tests or verification recorded.
- [x] #2 No TestPyPI token values are logged or committed.
- [x] #3 Bandit run for touched code when applicable or documented skip for package metadata/test-only changes.
- [x] #4 Backlog task updated with final summary.
<!-- DOD:END -->
