---
id: TASK-2397
title: Address PR 2428 prompt catalog review findings
status: Done
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up remediation for Qodo review findings on merged PR #2428: centralize PromptCatalogError, add missing annotations/docstrings, restore prompts.available compatibility, and update test markers/type hints.
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
Addressed the actionable Qodo findings from PR #2428 by moving `PromptCatalogError` into the core exception module, restoring the legacy `capabilities.prompts.available` field alongside MCP `listChanged`, adding missing docstrings/type hints, and marking/typing the new tests.

Verification:
- Red checks first failed for missing centralized exception and missing prompt `available` capability, then passed after implementation.
- `python -m pytest ...test_prompts_catalog.py ...test_protocol_prompts_catalog.py ...test_dynamic_module_catalog.py::test_default_mcp_modules_config_declares_prompts_module_with_empty_config_allowlist ...test_mcp_prompts_http.py ...test_rbac_seed_helper.py -v` passed: 62 passed.
- `python -m pytest ...test_gateway_fastapi_package.py::test_gateway_fastapi_app_handles_basic_jsonrpc_flow ...test_smoke_client.py::test_inprocess_gateway_transport_exposes_fixture_tools_resources_and_prompts ...test_basic_functionality.py ...test_registry_iteration_race.py ...test_protocol_catalog_filter.py ...test_dynamic_module_catalog.py ...test_extraction_contracts.py -v` passed: 78 passed.
- `python -m py_compile` on touched production files passed.
- Bandit touched production scope exited 0 with `results: []`.
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
