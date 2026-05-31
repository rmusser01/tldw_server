---
id: TASK-580
title: Implement MCP Unified Stage 4M gateway external registry management
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-31 21:07'
labels:
  - mcp-unified
  - stage-4m
  - implementation
  - standalone
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-31-mcp-unified-stage4m-gateway-external-registry-management-design.md
  - >-
    Docs/superpowers/plans/2026-05-31-mcp-unified-stage4m-gateway-external-registry-management-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the reviewed and planned MCP Unified Stage 4M gateway external registry management work. Follow the implementation plan in Docs/superpowers/plans/2026-05-31-mcp-unified-stage4m-gateway-external-registry-management-implementation-plan.md, preserving package boundaries and deferring real external process lifecycle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Storage contract exposes atomic external server create and SQLite rejects duplicate ids without replacing existing definitions.
- [x] #2 GatewayExternalRegistryManager owns external server validation, audit, patch guards, credential-slot relaxation guards, and delete guards.
- [x] #3 Gateway config/bootstrap, FastAPI, and CLI surfaces use the same package-owned manager and storage bundle semantics.
- [x] #4 Implementation does not add real external process lifecycle, credential secret handling, UI, or host-package imports into mcp_unified.
- [x] #5 Focused pytest suite, package-boundary test, Bandit touched-scope scan, and git diff whitespace validation are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 complete: storage contract and SQLite atomic create implemented in commit 7bf79af640. Spec review passed; code-quality review approved. Focused storage tests passed per worker/reviewer evidence.

Task 2 complete: GatewayExternalRegistryManager implemented through commits 3a5bcd1e7a, de11d9e748, and a136f60633. Spec review passed after stale-delete regression; code-quality review requested list normalization, which was fixed and re-approved. Manager tests passed per worker/reviewer evidence.

Task 3 complete: config storage bundle/factory implemented through commits fe20b4d716, d75c36e403, and 882a30cfc5. Spec review passed after error-message clarification; code-quality review requested injected credential-grant reuse, which was fixed and re-approved. Focused config tests passed per worker/reviewer evidence.

Task 4 complete: FastAPI external registry routes implemented through commits 18da5f17 and 7144e989. Spec review requested full error-mapping coverage, which was added; code-quality review approved with no blocking issues. Full FastAPI package test module passed per reviewer evidence.

Task 5 complete: CLI external registry commands implemented through commits 5d4ca8df and a2471e7306. Spec review passed; code-quality review requested memory-config reason-code payload, which was fixed and re-approved. Full CLI test file passed per worker/reviewer evidence.

Final review fixes complete in commit 77741b9ac7: real SQLite profile bootstraps now expose an external registry manager, credential grant lookup failures map to credential_grant_store_unavailable, and PATCH uses update-if-present semantics to avoid stale recreation. Final reviewer found no blocking issues; residual noted that external/custom ExternalRegistryStore implementations need the new update_server method.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented MCP Unified Stage 4M gateway external registry management.

Delivered storage atomic create/update semantics, GatewayExternalRegistryManager validation/audit/guard behavior, config storage bundle and manager factory, FastAPI /external-servers management routes, and CLI list/show/create/patch/delete commands.

Verification: focused Stage 4M suite passed with 218 tests; package boundary was included in that suite; Bandit touched package scope reported 0 results and 0 errors; git diff --check passed. Final code review found no blocking issues after commit 77741b9ac7.

Known residual: the ExternalRegistryStore protocol now requires update_server, so external/custom store implementations outside this repo need to add that method.
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
