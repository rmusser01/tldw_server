---
id: TASK-2401
title: Run MCP Unified standalone release rehearsal after publish-readiness merge
status: Done
labels:
- mcp
- packaging
- release
- uat
priority: medium
modified_files:
- backlog/tasks/task-2401 - Run-MCP-Unified-standalone-release-rehearsal-after-publish-readiness-merge.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the standalone MCP Unified release rehearsal from latest dev after the publish-readiness PR merge. Verify RC build/UAT, dry-run publish plan, and maintainer-owned TestPyPI readiness without exposing secrets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP Unified internal RC runs from latest dev and records evidence.
- [x] #2 Dry-run publish plan runs from freshly built artifacts and records TestPyPI command evidence without upload.
- [x] #3 Maintainer-owned TestPyPI/PyPI workflow path is checked for safe non-secret prerequisites where available.
- [x] #4 Any discovered blockers are documented with concrete next steps.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Release rehearsal completed from latest dev worktree `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-release-rehearsal`.

Verification:
- `make PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python mcp-unified-rc` exited 0 and wrote `.artifacts/mcp-unified-rc/mcp-unified-rc-evidence.json` plus `.artifacts/mcp-unified-rc/mcp-unified-rc-summary.md`; RC status was `ok`.
- `make PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python mcp-unified-publish-dry-run` exited 0. It rebuilt artifacts and ran `publish-plan --target testpypi --dry-run` without upload.
- Evidence summary reports `mcp-unified 0.1.0`, source path `apps/mcp-unified`, layout `src`, wheel `mcp_unified-0.1.0-py3-none-any.whl` sha256 `575828add28ac815fbc2419c640ba112774c7333ac9cf9851171a89efc7f4a9e`, and sdist `mcp_unified-0.1.0.tar.gz` sha256 `569eac5b08851b5223e6a776ab550af81131c43904a1514f57514e06829b7ee6`.

Safe publishing prerequisite check:
- `.github/workflows/mcp-unified-publish.yml` defines `environment: testpypi` and `environment: pypi`, with secret names `MCP_UNIFIED_TESTPYPI_API_TOKEN` and `MCP_UNIFIED_PYPI_API_TOKEN` only.
- `gh api repos/rmusser01/tldw_server/environments --jq '{environments: [.environments[].name]}'` returned only `tldw with API Keys`.
- `gh secret list --env testpypi --repo rmusser01/tldw_server --json name,updatedAt` returned HTTP 404 because the `testpypi` environment is absent or inaccessible.
- `gh secret list --env pypi --repo rmusser01/tldw_server --json name,updatedAt` returned HTTP 404 because the `pypi` environment is absent or inaccessible.

Blocker before live upload:
- A maintainer needs to create protected GitHub Actions environments named `testpypi` and `pypi`, then add environment secrets `MCP_UNIFIED_TESTPYPI_API_TOKEN` and `MCP_UNIFIED_PYPI_API_TOKEN`. This keeps the existing token-based publish workflow intact; no secret values were accessed or logged.

Update 2026-06-24:
- Created the GitHub Actions environment `testpypi` and stored the environment secret name `MCP_UNIFIED_TESTPYPI_API_TOKEN` from the local token file without logging the token value.
- Verified non-secret repo state with `gh secret list --env testpypi --repo rmusser01/tldw_server --json name,updatedAt`, which returned `MCP_UNIFIED_TESTPYPI_API_TOKEN`.
- Verified `gh api repos/rmusser01/tldw_server/environments --jq '{total_count, environments: [.environments[].name]}'` now returns `testpypi` and `tldw with API Keys`.
- The production `pypi` environment and `MCP_UNIFIED_PYPI_API_TOKEN` remain unconfigured until a production PyPI token is available.

Bandit: skipped because no code or workflow files were changed in this rehearsal branch; only the Backlog evidence record changed.
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
