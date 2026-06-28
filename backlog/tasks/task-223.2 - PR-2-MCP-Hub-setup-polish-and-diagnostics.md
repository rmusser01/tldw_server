---
id: TASK-223.2
title: 'PR 2: MCP Hub setup polish and diagnostics'
status: In Progress
assignee: []
created_date: 2026-05-10 06:13
labels:
- mcp
- webui
- ux
- diagnostics
dependencies:
- TASK-223.1
parent_task_id: TASK-223
priority: medium
documentation:
- Docs/superpowers/specs/2026-06-26-mcp-hub-setup-recovery-ux-design.md
- Docs/superpowers/plans/2026-06-26-mcp-hub-setup-recovery-ux-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-26-mcp-hub-setup-recovery-ux-design.md
- Docs/superpowers/plans/2026-06-26-mcp-hub-setup-recovery-ux-implementation-plan.md
- backlog/tasks/task-223.2 - PR-2-MCP-Hub-setup-polish-and-diagnostics.md
- tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py
- tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py
- tldw_Server_API/app/core/MCP_unified/mcp_hub_readiness.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py
- apps/packages/ui/src/services/tldw/mcp-hub.ts
- apps/packages/ui/src/services/tldw/__tests__/mcp-hub.test.ts
- apps/packages/ui/src/components/Option/MCPHub/mcpHubReadiness.ts
- apps/packages/ui/src/components/Option/MCPHub/__tests__/mcpHubReadiness.test.ts
- apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
- apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx
- apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx
- apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx
- apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts
- apps/tldw-frontend/e2e/utils/page-objects/MCPHubPage.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the second PR-sized remediation slice from the MCP Hub walkthrough. This phase should make setup states easier to understand after the live-discovery and chat blocker fixes land.
<!-- SECTION:DESCRIPTION:END -->

## Stage 0 Implementation Discovery

<!-- SECTION:STAGE0:BEGIN -->
- Stage 0 decision: Phase 1 requires backend readiness/refresh support before UI fan-out.
- Current `McpHubExternalServer`/`ExternalServerResponse` fields can support only partial readiness inference: credentials from `secret_configured`, `credential_slots`, `auth_template_present`, `auth_template_valid`, and `auth_template_blocked_reason`; runtime availability from `runtime_executable`; and limited capability warnings from tool/module metadata. The current external-server and tool-registry APIs do not expose preflight status, discovery attempts, discovery timestamps, discovery success status, last discovery errors, config fingerprints, or any server-to-tool registry freshness link.
- `credentialState` can be normalized conservatively from existing credential/auth-template fields, but `checking` is currently local UI mutation/loading state only. `runtime_unavailable` can be derived from `runtime_executable === false`. `partial_capability` is only partially available from registry metadata warnings. `preflight_failed`, `discovery_not_run`, `config_changed`, `unreachable`, and `discovery_failed` are not reliably derivable from the current MCP Hub REST payloads. Because the API cannot distinguish successful zero-tool discovery from discovery not run, Phase 1 must not emit `no_tools_returned`; zero tools must map to `discovery_not_run` until backend discovery lifecycle data exists.
- Refresh support: no dedicated MCP Hub management refresh endpoint and no refresh helper in the existing `apps/packages/ui/src/services/tldw/mcp-hub.ts` client exist today. The implemented refresh operation is the MCP tool `external.tools.refresh`, callable through `POST /api/v1/mcp/tools/execute` with descriptive request shape `{ "tool_name": "external.tools.refresh", "arguments": { "server_id": string } }`, where `server_id` is optional. The manager returns `{ "refreshed_servers": number, "total_servers": number, "virtual_tools": number, "errors": Record<string, string> }`; because MCP protocol wraps dict tool results as JSON content and the REST execute endpoint only unwraps text content, the descriptive REST response shape keeps that payload under structured content: `{ "result": { "content": [{ "type": "json", "json": { "refreshed_servers": number, "total_servers": number, "virtual_tools": number, "errors": Record<string, string> } }], "module": string | null, "tool": "external.tools.refresh" }, "execution_time_ms": number, "module": string }`. Task 2 must either add a dedicated MCP Hub readiness/refresh route/client around that real operation or explicitly route the final UI action through this existing execute flow; disabled or placeholder refresh actions are not acceptable.
- Permission decision: new MCP Hub preflight and discovery refresh management routes should use the existing mutation/admin gate from `mcp_hub_management.py`: `_require_mutation_permission`, which accepts admin role, `system.configure`, or `*`. If a UI path calls the generic `/api/v1/mcp/tools/execute` endpoint directly, it must also satisfy the existing tool execution gate for `tools.execute:external.tools.refresh` or `tools.execute:*`.
- Catalog staleness decision: no explicit catalog age threshold exists in the current schemas, endpoints, or UI client, so Phase 1 must not emit `catalog_expired`. Use only explicit `config_changed` once backend support exists, manual invalidation, failed refresh after prior success, and `discovery_not_run`. `no_tools_returned` is not available in Phase 1 unless backend readiness data adds a successful-zero-tools state; otherwise a zero-tool catalog maps to `discovery_not_run`.
- Diagnostics data availability: the frontend can read deployment mode from `getTldwDeploymentMode()`, effective request/API origin from `resolveBrowserRequestTransport()`/browser networking helpers, the stored server URL from `tldw-server.ts`, and health-check state from the shared connection store using `/api/v1/health/live`. Backend health endpoints also expose `/api/v1/health`, `/api/v1/health/live`, and readiness checks. MCP Hub-specific operation timestamps, current preflight/discovery operation metadata, last successful discovery time, and sanitized external-server error category/message are not surfaced by the current MCP Hub REST API; Task 2 should add them with readiness support, or Task 6 must render precise unavailable states with setup-isolation guidance instead of implying live diagnostics exist.
- Permission-gated catalog state: `GET /api/v1/mcp/hub/tool-registry` and `/summary` require authentication but do not expose a response field that distinguishes a permission-gated catalog from an ordinary empty or error state, and `ToolCatalogsTab.tsx` currently collapses errors into a generic load failure. Phase 1 cannot emit a separate permission-gated catalog state unless later backend/client work adds an explicit response shape; do not add the Task 4 permission-boundary recovery test until that shape exists.
- Toy MCP smoke decision: no stable reusable toy MCP server/walkthrough path was found for the MCP Hub UI. Existing toy/stub external servers are test-local fixtures under the backend MCP unified tests, and the frontend has `apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts` for MCP Hub navigation/API coverage. Skip the toy-server smoke for this slice with that reason, and replace it with the closest isolated automated browser path: a route-mocked MCP Hub guided setup flow that exercises the polished setup UI without mutating production data.
<!-- SECTION:STAGE0:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No-auth local stdio servers render a neutral or healthy no-credentials-required state instead of missing-auth warnings.
- [x] #2 Legacy Secret Fallback appears only when the selected managed server actually uses the transitional server-level secret flow.
- [x] #3 Tool Catalog empty and stale states offer clear Add server and Refresh discovery actions with setup, runtime, and permissions distinctions.
- [x] #4 MCP Hub or shared diagnostics expose effective deployment mode, API origin, and health endpoint enough to diagnose quickstart versus advanced split-brain configuration.
- [x] #5 Setup isolation expectations for local walkthrough or E2E runs are documented or verified where practical.
- [x] #6 Focused UI tests and a route-mocked E2E smoke cover the polished setup path. Stage 0 found no stable toy MCP server path, so this slice uses isolated route mocks instead of runtime DB writes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-26-mcp-hub-setup-recovery-ux-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 2 implemented and hardened the backend/client readiness and discovery refresh contract. Added sanitized readiness schemas and MCP Hub routes for GET /api/v1/mcp/hub/readiness, POST /external-servers/{server_id}/validate, and POST /external-servers/{server_id}/refresh-discovery. Refresh delegates to the existing external.tools.refresh federation tool, initializes MCP server modules before lookup, serializes per-server requests, and rejects non-operational visible rows before runtime execution. Added focused backend and frontend service tests, verified red/green TDD, and ran Bandit with zero findings.

Task 3 applied readiness to External Server rows. Rows now load backend readiness plus registry data, show clear credential/readiness tags, wire validate/refresh/edit/credentials/details/tool-catalog actions, and route Tool Catalog from McpHubPage. Verification: focused ExternalServersTab, combined readiness/service/row, and McpHubPage Vitest suites passed. Bandit skipped for Task 3 because only TS/TSX and Backlog markdown changed.

Task 3 review follow-up aligned selected-server detail panels with readiness credential state: no-auth servers now show healthy no-credentials-required auth/secret panels, while legacy fallback remains limited to legacy_fallback readiness. Focused ExternalServersTab and combined readiness/service/row Vitest suites passed.

Task 3 resilience follow-up keeps the external-server inventory visible when readiness or tool-registry metadata fails independently. The UI now degrades with a readiness-warning panel and falls back to local readiness inference instead of clearing rows. Regression test added for readiness metadata failure; focused MCP Hub Vitest suite passed, and the isolated ExternalServersTab design-state guard reports zero blocked findings.

Task 4 added actionable Tool Catalog recovery states for no server, discovery not run, missing credentials, runtime unavailable, preflight failed, unreachable, discovery failed, checking, stale config, and partial-capability states. Recovery actions now route to Add server, focused credentials, server config, details, or real refresh discovery as appropriate. The catalog remains visible when auxiliary readiness/server metadata fails, with a degraded recovery-details panel, and touched MCP Hub product-state alerts were converted to shared StatePanel usage. Verification: focused MCP Hub Vitest suite passed with 85 tests; isolated touched-file design-state guard passed with zero blocked findings. Full design-state verification still fails on existing repo-wide findings outside this slice.

Task 5 added a guided managed-server setup path with Local stdio, HTTP/SSE, Import config, and Advanced/manual choices. Advanced/manual preserves the raw Config JSON path; guided stdio and HTTP/SSE paths build the same create payload shape; managed JSON import previews decoded server details and keeps validation local. Save and discover tools now creates the server, calls real discovery refresh, reloads readiness, and shows a result panel with next actions. Verification: focused MCP Hub Vitest suite passed with 90 tests; isolated touched-file design-state guard passed with zero blocked findings. Bandit skipped because only frontend TSX/tests and plan/backlog markdown changed.

Task 6 added MCP Hub diagnostics to the existing External Server readiness Details modal and a pure sanitized diagnostics formatter. Details now show readiness state, reason codes, credential state, transport, tool count, timestamps, current operation, last error category/message, environment diagnostics with explicit Not available in this client fallbacks, audit-location copy, setup-isolation guidance, and a redacted config block covering nested env, headers, URL query values, args, and raw config objects. Verification: red tests failed for missing helper/diagnostics and then missing health/audit rows; focused MCP Hub frontend suite passed with 92 tests; touched-file design-state guard reported exitCode 0 and blocked 0; backend MCP Hub management API tests passed with 39 tests; git diff --check passed. Bandit skipped because Task 6 touched only frontend TypeScript/tests and plan/backlog markdown.

Task 7 made the MCP Hub top cards truthful by demoting the static status summary to workflow shortcut navigation. The page no longer labels those cards as status, no longer shows workflow/status pills, and avoids readiness-like copy when no page-level readiness data is loaded. Verification: red McpHubPage test failed against the old status summary; McpHubPage tests passed with 11 tests; focused MCP Hub frontend suite passed with 92 tests; touched-file design-state guard reported exitCode 0 and blocked 0; git diff --check passed. Bandit skipped because Task 7 touched only frontend TSX/tests and plan/backlog markdown.

Task 8 added isolated MCP Hub E2E verification and final checks. The tier-2 Playwright spec now route-mocks MCP Hub readiness, external-server inventory, discovery refresh, and tool-registry responses for page load, first-run guided local stdio setup, and desktop/mobile responsive diagnostics/catalog coverage. The first-run smoke verifies empty setup state, setup choices, minimum stdio fields, Save and discover tools, mocked refresh readiness, result panel, and Tool Catalog ready state without runtime DB writes. The responsive test runs at 1440x900 and 390x844, opens guided setup choices and diagnostics, checks sanitized config redaction, switches to Tool Catalog, asserts no horizontal overflow, and attaches `mcp-hub-setup-*` / `mcp-hub-catalog-*` screenshots during Playwright. Verification: focused MCP Hub frontend suite passed with 92 tests; backend MCP Hub management API tests passed with 39 tests; `TLDW_WEB_CMD='bun run dev:webpack -- -p 8080' bun run e2e:pw -- e2e/workflows/tier-2-features/mcp-hub.spec.ts --project=tier-2 --reporter=line` passed with 5 mocked/local tests and 5 live-backend API checks skipped because no backend was available. Initial sandboxed E2E failed on localhost binding and the default Turbopack dev server failed with an internal directory-read panic, so the verified run used the existing webpack dev fallback. Full `bun run verify:design-system-state` still fails on existing repo-wide baseline findings outside this slice; MCP Hub file-scoped guard reported exitCode 0, blocked 0, baselineErrors 0. `rg "rgba\\(0,0,0,0\\.45\\)" apps/packages/ui/src/components/Option/MCPHub` returned no matches, and `git diff --check` passed. Broad Bandit wrote `/tmp/bandit_mcp_hub_setup_recovery.json` and exited 1 for existing unrelated service findings in `admin_e2e_support_service.py`, `admin_guardrails_service.py`, and `quality_eval_scheduler.py`; Task 8 itself changed only frontend E2E TypeScript.

PR rebase follow-up rebased PR #2536 onto `origin/dev`, split the frontend refresh clients into bulk discovery refresh and per-server readiness refresh helpers, preserved Tool Catalog runtime query invalidation after refresh, renamed backend refresh route handlers to distinguish per-server and bulk endpoints, and tightened the MCP Hub E2E page-object Command locator after the global command-palette button made the old label query ambiguous. Verification after rebase: focused MCP Hub Vitest suite passed with 93 tests; backend MCP Hub management API tests passed with 49 tests; Bandit on `mcp_hub_management.py` exited 0 and wrote `/tmp/bandit_mcp_hub_rebase.json`; `TLDW_WEB_CMD='bun run dev:webpack -- -p 8080' bun run e2e:pw -- e2e/workflows/tier-2-features/mcp-hub.spec.ts --project=tier-2 --reporter=line` passed with 5 tests and 5 live-backend checks skipped; `git diff --check` passed.

PR review follow-up addressed Qodo's latest unresolved comments after the rebase. Readiness policy and refresh-result sanitization moved from the FastAPI endpoint into `tldw_Server_API/app/core/MCP_unified/mcp_hub_readiness.py`, the validate route/UI copy now presents a read-only readiness check instead of claiming real validation, SSE auth templates use header targets, Tool Catalog empty-state logic considers visible readiness/inventory rows rather than only active totals, schema/docstring and test-marker issues were corrected, and focused regression tests were added. Verification: Python compile passed on the touched backend files with the main virtualenv; focused MCP Hub Vitest suite passed with 95 tests; backend MCP Hub management API tests passed with 49 tests; Bandit on touched backend scope exited 0 and wrote `/tmp/bandit_mcp_hub_review_followups.json`; `git diff --check` passed.

Second PR review follow-up addressed the remaining PR #2542 comments after confirming the branch was already up to date with origin/dev. Tool Catalog now tolerates readiness metadata without a servers array and falls back to the external-server inventory, successful per-server discovery refresh clears stale discovery_failed readiness state, the readiness helper has a module docstring, diagnostics label the saved timestamp as Last stored validation, and the validate endpoint docstring now documents the mutation-permission gate. Verification: focused ToolCatalogsTab regression passed with 1 test; focused MCP Hub frontend suite passed with 96 tests; backend MCP Hub management API tests passed with 50 tests; Python compile passed on touched backend files; Bandit on touched backend scope exited 0 and wrote /tmp/bandit_mcp_hub_review_followups_round2.json; git diff --check passed.
<!-- SECTION:FINAL_SUMMARY:END -->
