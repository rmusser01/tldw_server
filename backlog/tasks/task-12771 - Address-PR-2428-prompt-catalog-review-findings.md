---
id: TASK-12771
title: Address PR 2428 prompt catalog review findings
status: Done
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up remediation for Qodo/Gemini review findings on merged PR #2428 and PR #2429: keep PromptCatalogError available through core compatibility imports while moving the MCP dependency to a framework-neutral exception type, add missing structured docstrings, preserve prompt capability compatibility, and advertise prompts only when a prompt-capable module is registered.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Centralize `PromptCatalogError` in `tldw_Server_API.app.core.exceptions`.
- [x] Preserve MCP prompt capability compatibility by returning both `available` and `listChanged`.
- [x] Add missing return annotation/docstrings for modified prompt API/module helpers.
- [x] Add unit markers and explicit fixture/helper type annotations to new prompt catalog tests.
- [x] Verify targeted prompt, RBAC, and MCP regression tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- PR #2428 was already merged, so these review remediations were implemented as follow-up branch `codex/pr2428-review-followups` from latest `origin/dev`.
- Qodo's stale Gemini/Docker findings were not included because they refer to files outside the merged prompt-catalog PR diff.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Initial PR #2428 review findings were addressed, and PR #2429 review follow-ups now add the framework-neutral `PromptCatalogError` module, keep the legacy `app.core.exceptions` re-export, update MCP imports away from FastAPI-coupled exceptions, expand `PromptsModule.on_initialize()` with structured side-effect/return documentation, and make initialize advertise prompt availability only when a registered module declares prompt listing hooks.

Final verification after rebasing onto latest `origin/dev`:
- Red checks first failed for missing `exception_types` and imprecise prompt capability advertisement, then passed after implementation.
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py tldw_Server_API/tests/MCP_unified/test_mcp_prompts_http.py -q` passed: 58 passed.
- `python -m pytest -c mcp_unified/pytest-artifact-gate.ini .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_distribution_metadata_matches_extras -q` passed: 1 passed.
- `python -m py_compile` on touched production files passed.
- Bandit touched production scope exited 0 with `results: 0`, `errors: 0` in `/tmp/bandit_pr2429_rebased.json`.
- Latest PR #2429 package/UX check failures were triaged as stale-base CI failures; the package metadata gate is verified green locally after rebase, and the frontend UX failure was outside this backend-only diff and should rerun on the rebased branch.
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
