---
id: TASK-12148
title: Implement first-run MCP tool packs setup
status: In Progress
assignee: []
created_date: '2026-07-04 23:41'
updated_date: '2026-07-05 02:54'
labels:
  - implementation
  - mcp
  - setup
  - first-run
dependencies:
  - TASK-12132
references:
  - >-
    Docs/superpowers/plans/2026-07-04-first-run-mcp-tool-packs-implementation-plan.md
  - Docs/superpowers/specs/2026-07-04-first-run-mcp-tool-packs-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the reviewed first-run MCP tool packs implementation plan using subagent-driven development. Scope includes backend catalog/apply/validate APIs, MCP Hub profile integration, frontend onboarding step, MCP Hub follow-up status, tests, verification, and commits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Backend setup catalog, apply, validate, and admin recovery endpoints are implemented per plan.
- [ ] #2 Frontend onboarding MCP tools step and MCP Hub follow-up/recovery UI are implemented per plan.
- [ ] #3 Focused backend/frontend tests and touched-scope Bandit verification are recorded.
- [ ] #4 Implementation commits are reviewed with spec and code-quality subagent gates.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 backend catalog/policy slice: created tldw_Server_API/app/core/Setup/first_run_mcp_tools.py and tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py. Red check confirmed ModuleNotFoundError before implementation. Verification: python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py -v -> 14 passed; python -m bandit -r tldw_Server_API/app/core/Setup/first_run_mcp_tools.py -f json -o /tmp/bandit_first_run_mcp_tools.json -> 0 findings; git diff --check -> clean.
Task 1 follow-up code-quality fix: kept MCP discovery list tools available with realistic unclassified registry metadata and moved broad grants from allowed_tools into top-level capabilities. Red check before fix: catalog tests failed on missing mcp.catalogs.list/mcp.modules.list/mcp.tools.list and missing capabilities field. Verification: python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py -v -> 15 passed; python -m bandit -r tldw_Server_API/app/core/Setup/first_run_mcp_tools.py -f json -o /tmp/bandit_first_run_mcp_tools_followup.json -> 0 findings; git diff --check -> clean.
Follow-up hash fix: included normalized capabilities in the first-run MCP generated policy hash. Added a public-generator regression where selections and allowed tools stay identical while filesystem.delete capability changes, proving generated_policy_hash changes. Verification: red check failed before helper patch with TypeError on missing capabilities parameter; after fix `python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py -v` -> 16 passed; Bandit on helper -> 0 findings; `git diff --check` -> clean.
Task 2 backend MCP Hub apply service: added service-local apply request/result dataclasses, apply flow for first-run generated permission profiles, global default assignment creation/update, provenance-based profile lookup, manual-edit conflict handling, keep_existing/replace_existing paths, and safe mcp_tools step payload. Red check: service test collection failed before implementation on missing compute_first_run_policy_hash. Verification: python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py -v -> 7 passed; python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py -v -> 23 passed; python -m bandit -r tldw_Server_API/app/core/Setup/first_run_mcp_tools.py tldw_Server_API/app/services/setup_mcp_tools_service.py -f json -o /tmp/bandit_first_run_mcp_tools_service.json -> 0 findings; git diff --check -> clean.
Task 2 self-review follow-up: MCP Hub repo list methods treat owner_scope_id=None and target_id=None as broad filters, so the apply service now filters returned rows to global/null profiles and default/null assignments before mutating. Added a regression that failed before the fix on assignment_id 9 vs expected 10. Verification after fix: python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py::test_apply_filters_broad_hub_lists_to_global_null_scope_and_default_null_target -q -> 1 passed; python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py -v -> 24 passed; Bandit touched-scope report -> 0 findings; git diff --check -> clean.
Task 2 code-quality follow-up: preserved unrelated policy_document fields when refreshing generated policy keys, stopped clearing default assignment path/workspace/inline/approval controls when repointing profile_id, required matching profile_id for keep_existing/replace_existing conflict-resolution requests, filtered confirmed_addon_ids to selected add-ons in the safe step payload, and removed the unused service result serializer. Red check: targeted service tests failed before implementation on dropped denied_tools, cleared assignment fields, missing ValueError, and unrelated confirmed add-on persistence. Verification: targeted red/green subset -> 5 passed after fix; python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py -v -> 28 passed; python -m bandit -r tldw_Server_API/app/core/Setup/first_run_mcp_tools.py tldw_Server_API/app/services/setup_mcp_tools_service.py -f json -o /tmp/bandit_first_run_mcp_tools_service_followup.json -> 0 findings; git diff --check -> clean.
Task 5 frontend API service/hook slice: added TypeScript contracts for first-run MCP tool catalog/apply/validate responses, client path guard entries, setup onboarding service methods, and useSetupOnboarding catalog/apply/validate state/methods. Red check before implementation: `bunx vitest run src/services/tldw/__tests__/setup-onboarding.test.ts src/hooks/__tests__/useSetupOnboarding.test.tsx` from apps/packages/ui failed as expected with 8 missing-method failures (`getMcpToolsCatalog`, `applyMcpTools`, `validateMcpTools`, `loadMcpToolsCatalog`). Green verification after implementation: same command -> 2 files passed, 17 tests passed.
Task 5 code-quality follow-up: fixed applyMcpTools conflict handling to preserve expectedStatuses [409] while catching only thrown 409 responses with a typed detail body; malformed 409 details and non-409 errors rethrow. Tightened McpToolsApplyConflict.profile_id to number. Red check before fix: focused UI tests failed on the thrown Conflict regression. Green verification: `bunx vitest run src/services/tldw/__tests__/setup-onboarding.test.ts src/hooks/__tests__/useSetupOnboarding.test.tsx` from apps/packages/ui -> 2 files passed, 18 tests passed.
Task 5 backend/frontend conflict contract follow-up: changed first-run MCP tools apply 409 detail to serialize the full McpToolsApplyResponse shape so the frontend extractor receives status/conflict/effective fields instead of a bare conflict object. Red check before endpoint fix: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py::test_first_run_mcp_tools_apply_returns_conflict_for_profile_conflict -q` failed with KeyError 'status'. Green verification: same backend test -> 1 passed; UI focused Vitest -> 2 files passed, 18 tests passed; Bandit on setup.py -> 0 findings.
Task 5 frontend conflict guard follow-up: tightened applyMcpTools 409 extraction to accept only full conflict responses with status="conflict" and a non-null conflict containing reason/profile_id. Red check before fix: focused UI tests returned an apply-shaped status="applied" 409 body instead of rejecting. Green verification: `bunx vitest run src/services/tldw/__tests__/setup-onboarding.test.ts src/hooks/__tests__/useSetupOnboarding.test.tsx` from apps/packages/ui -> 2 files passed, 18 tests passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Task 6 frontend MCP tools wizard step: added McpToolsStep and inserted mcp_tools between optional_advanced and first_chat in UnifiedSetupWizard. Step supports default pack selection, collapsed add-ons, inline strong add-on confirmation, apply conflict resolution, sample validation, skip-step persistence, and MCP Hub handoff links. Red check: `bunx vitest run src/components/Option/Onboarding/__tests__/McpToolsStep.test.tsx src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx` from apps/packages/ui failed before implementation on missing McpToolsStep and old wizard route. Green verification: same command -> 2 files passed, 38 tests passed. Bandit not run for this frontend-only slice.

Task 6 spec-review fixes: added a FirstChatStep backLabel prop so the first-run wizard can show Back to MCP tools while standalone/provider flows keep Back to providers; cleared MCP apply/validation results when pack/add-on/confirmation choices change after save; mapped conflict reasons, add-on ids, validation states, and external status values to local user-facing labels. Red check: focused Vitest failed before fixes on the missing backLabel behavior, stale saved selection controls, and raw profile_manually_changed/local_file_read/not_run/not_configured copy. Green verification: `bunx vitest run src/components/Option/Onboarding/__tests__/McpToolsStep.test.tsx src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx` from apps/packages/ui -> 3 files passed, 55 tests passed.

Task 6 code-quality review fixes: passed saved mcp_tools step data into the MCP tools step, restored saved pack/add-on/confirmed add-on selections with an applied local state when returning from first chat, and added a synchronous pending guard for the MCP tools skip action. Added regression coverage for direct step hydration, wizard back-navigation hydration, and fast double-click skip protection. Focused UI verification: bunx vitest run src/components/Option/Onboarding/__tests__/McpToolsStep.test.tsx src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx passed with 58 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 4 safe validation slice: added service validation for exact built-in mcp.tools.list policy checks, external discovery refresh readiness, fail-closed external no-arg read candidate selection, safe fixed validation messages, and setup/admin validate route wiring. Verification: pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -k "mcp_tools or validate" -v -> 41 passed; Bandit touched backend scope -> 0 findings; git diff --check -> clean.

Task 4 follow-up fail-closed fixes: admin MCP status/validate now require explicit SYSTEM_CONFIGURE or * permissions, non-raising refresh results with refreshed_servers=0/errors are treated as external_discovery_incomplete with fixed redacted messaging, and external validation requires a matching mcp.tools.list descriptor with an inputSchema mapping before executing a candidate. Verification: focused regressions -> 3 passed; pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -k "mcp_tools or validate" -v -> 44 passed; Bandit touched backend scope -> 0 findings; git diff --check -> clean.

Task 4 request validation follow-up: McpToolsApplyRequest and McpToolsValidateRequest now forbid unknown fields so raw config/execution payloads are rejected at the API boundary. Red check before fix: focused request-validation tests failed with 200 responses. Verification after fix: focused request-validation tests -> 3 passed; pytest tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -k "mcp_tools" -v -> 22 passed; Bandit on setup_schemas.py -> 0 findings; git diff --check -> clean.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 7 MCP Hub follow-up status/recovery slice: added admin setup-onboarding client methods without noAuth, compact MCP Hub first-run recovery panel with user-facing status labels/recovery/profile routing, backend admin status detection for generated MCP profile hash mismatch, and focused UI/backend tests. Verification: frontend Vitest MCP Hub/status + setup-onboarding service -> 23 passed; backend admin MCP tools integration -> 5 passed; service recovery status target included in combined backend slice -> passed; Bandit touched backend scope -> 0 findings; git diff --check -> clean.
Task 7 spec-review follow-up: fixed MCP Hub Review profile to create a real permission_profile drill target and passed drill targets into PermissionProfilesTab. PermissionProfilesTab now opens the matching profile edit form after profiles load and marks the drill handled. Verification: bunx vitest run src/components/Option/MCPHub/__tests__/McpHubPage.first-run-status.test.tsx src/components/Option/MCPHub/__tests__/PermissionProfilesTab.test.tsx src/services/tldw/__tests__/setup-onboarding.test.ts -> 28 passed; git diff --check -> clean.
Task 7 code-quality follow-up: recovery/manual-change status now requires first-run MCP provenance before running generated profile hash conflict checks, so a stale state pointing at a normal global profile is treated as no generated profile for conflict purposes. Added service regression coverage. Verification: focused regression target -> passed; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -k "mcp_tools" -v -> 45 passed; Bandit touched backend service -> 0 findings; git diff --check -> clean.
Final-review backend fixes: generated first-run MCP profiles now conflict on manual `tool_patterns`/`tool_names`, explicit replace strips those keys, completed catalog access requires SYSTEM_CONFIGURE/*, and validation skips external discovery unless `external_network_read` is selected. Verification: focused pytest 50 passed; git diff --check passed; Bandit results empty at /tmp/bandit_first_run_mcp_tools_final_review.json.
Final-review external validation follow-up: validation now samples only external tools already present in the saved generated profile allowed_tools, preventing a refreshed tool discovered after apply from being executed or reported as passed outside the profile policy. Verification: focused validation regressions -> 3 passed; full focused backend suite -> 180 passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
