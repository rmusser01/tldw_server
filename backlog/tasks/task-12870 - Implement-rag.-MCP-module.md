---
id: TASK-12870
title: Implement rag.* MCP module
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-04 23:21'
labels:
  - mcp
  - rag
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-03-rag-mcp-module-design.md
  - Docs/superpowers/plans/2026-07-03-rag-mcp-module-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved first-slice rag.* MCP module with capabilities, source_health, search, and answer tools. Keep the implementation aligned to the existing RAG pipeline and MCP security/governance controls; do not add a research.* facade or external-provider research loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Expose rag.capabilities, rag.source_health, rag.search, and rag.answer through MCP with strict schemas and no research.* layer.
- [x] #2 Preserve existing RAG behavior through shared transport helpers while adding MCP-specific normalization, metadata, truncation, and citation coverage.
- [x] #3 Enforce MCP module/tool/category/security/quota controls and per-source authorization posture described in the approved plan.
- [x] #4 Register rag module, mcp.search, and rag_generation governance configuration with fail-closed behavior for missing generation policy.
- [x] #5 Cover behavior with targeted unit/integration tests, HTTP tools/execute compatibility, JSON-RPC wrapper compatibility, knowledge.search regression, and Bandit on touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-03-rag-mcp-module-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-07-03 review follow-up: addressed code-review findings for source-module registration checks, all-sources-filtered allow_partial behavior, MCP-to-billing RAG daily quota enforcement, PermissionError propagation for MCP auth/quota failures, bounded/redacted citation payloads, source_health context db_paths, empty sources_used semantics, and MCP top_k capability alignment. Added regression coverage in MCP RAG module tests and RAG transport helper tests. Verification: 67-test MCP/RAG helper/source-health suite passed; focused RAG module + transport helper suite passed; Bandit on touched Python scope completed with zero findings after test fixture cleanup.

2026-07-04 PR follow-up: rebase codex/rag-mcp-module onto latest origin/dev and address fresh PR review comments from CodeRabbit/Qodo/Gemini. Scope: keep PR on rag.* MCP only, fix still-valid comments with minimal changes, validate, then force-push rebased branch and retarget PR to dev.

2026-07-04 PR follow-up completed: rebased codex/rag-mcp-module onto latest origin/dev, tightened rag.* MCP review fixes for explicit-null defaults, strict booleans, include_documents pipeline handling, client-safe RAG error summaries, pipeline exception logging, test marker classification, and verified org metadata hints for RAG billing. Verification: focused RAG/MCP suite passed (73 tests); git diff --check passed; Bandit on touched Python scope completed with zero findings (/tmp/bandit_rag_mcp_pr2587_followup.json).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a proper rag.* MCP module without a research.* layer. Verification: targeted MCP/RAG suite passed (80 tests), and Bandit completed with zero findings for rag_module.py, rag_service/transport.py, and rag_unified.py.

Review follow-up: integrated code-review findings by making default rag.* MCP controls enforce protocol/RBAC/source authorization, forcing MCP-safe RAG pipeline overrides, redacting raw document/response metadata, returning source_health as an ok/warnings contract, and adding regression coverage. Verification: targeted MCP/RAG suite passed (85 tests), and Bandit completed with zero findings for rag_module.py.

Second review follow-up: integrated remaining findings around source module availability, empty authorized source sets, shared RAG daily billing limit enforcement for MCP, auth/quota error propagation, citation bounds/redaction, source_health context db_paths, and MCP top_k limit alignment. Verification: 67-test MCP/RAG helper/source-health suite passed; Bandit on touched Python scope completed with zero findings.
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
