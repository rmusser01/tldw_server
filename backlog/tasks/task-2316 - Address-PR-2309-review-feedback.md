---
id: TASK-2316
title: Address PR 2309 review feedback
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-08 03:38
labels: []
dependencies: []
modified_files:
- Docs/superpowers/plans/2026-06-08-pr-2309-review-remediation-plan.md
- apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.persistence.test.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SourceList.behavior.test.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SourceList.viewer.test.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/SetupDiagnostics.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/SourceList.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/scopeValidation.ts
- apps/packages/ui/src/components/Option/KnowledgeQA/trustState.ts
- apps/extension/tests/e2e/utils/extension-launch-health.spec.ts
- apps/extension/tests/e2e/utils/real-server.test.ts
- apps/extension/tests/e2e/utils/real-server.ts
- tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py
- tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py
- tldw_Server_API/app/core/RAG/rag_service/response_mapping.py
- tldw_Server_API/tests/RAG/test_knowledge_qa_live_regressions.py
- tldw_Server_API/tests/RAG/test_knowledge_qa_uat_fixtures.py
- tldw_Server_API/tests/RAG/test_knowledge_trust_contracts.py
- backlog/tasks/task-2279.10 - Review-and-tighten-Knowledge-QA-follow-on-remediation-spec.md
- backlog/tasks/task-2279.3 - Materialize-Knowledge-QA-evidence-excerpts-and-source-identifiers.md
- backlog/tasks/task-2279.4 - Enforce-Knowledge-QA-citation-validity-and-abstention.md
- backlog/tasks/task-2279.8 - Add-Knowledge-QA-live-UAT-fixtures-and-release-gates.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address reviewer and bot feedback on PR #2309 after rebasing on latest dev, including Knowledge QA sync trust persistence, scoped note metadata and async retrieval, scoped cache/fallback regressions, extension launch gate cleanup, test markers, and malformed task markers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2309 branch is rebased on latest origin/dev and pushed.
- [x] #2 Actionable unresolved review comments are fixed or explicitly answered with technical rationale.
- [x] #3 Focused backend/frontend/extension verification passes for touched areas.
- [x] #4 Bandit runs on touched backend scope or any exclusions are documented.
- [x] #5 PR #2309 is updated with the remediation commit(s).
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-06-08: Rebased local branch `codex/knowledge-qa-follow-on-remediation` onto latest `origin/dev` before remediation. Added red/green coverage and fixes for stale retry-sync trust persistence, selected-note stable source metadata, selected-note SQL allowlist bounding, sync-adapter offload, chat source lookup caching, explicit note selection source fallback/cache bypass, effective retrieval search-mode fallback, classification `skip_search` generation-only behavior, extension launch-health bypass removal, manifest parse diagnostics, shared trust contract usage, requested pytest markers, and malformed Backlog markers.

2026-06-08 follow-up: Addressed remaining current/outside-diff PR comments by fixing SourceList pinned fallback-key cleanup to use `original_result_index` consistently, adding a regression test that fails on the old mapped-index cleanup, making SetupDiagnostics host-access button visibility explicit, typing SourceList viewer fixtures as `RagResult[]`, and marking the dry-run subprocess UAT fixture test as integration while keeping pure fixture tests unit.

Verification: SourceList fallback-key regression red check failed with the old mapped-index cleanup and passed after restoring the fix. Focused SourceList/connection tests passed 47 tests. `tldw_Server_API/tests/RAG/test_knowledge_qa_uat_fixtures.py` passed 3 tests. Backend RAG regression set passed 26 tests. Full Knowledge QA UI suite passed 55 files / 487 tests. TypeScript passed with `NODE_OPTIONS=--max-old-space-size=8192`. Extension compile passed. Extension utility tests passed 3 tests. Extension launch-health built the extension and reported 1 skipped because this host exposes no extension targets; this is the shared conditional launch-unavailable skip, not an expected-failure bypass. Bandit follow-up JSON reported 0 results and 0 errors. `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2309 on latest `origin/dev` and addressed actionable review feedback without adding flashcard behavior. Fixed Knowledge QA retry-sync trust persistence, selected-note evidence identity, explicit scoped-source retrieval/cache behavior, effective-mode fallback, classification skip-search generation, chat source lookup caching, extension launch-health bypass, manifest parse diagnostics, requested test markers, malformed Backlog markers, SourceList pinned fallback-key pruning, explicit setup diagnostics host-access rendering state, and SourceList viewer test typing. Verification covered focused and full Knowledge QA frontend tests, backend RAG regressions, extension compile/utility/launch-health checks, TypeScript, Bandit, and diff hygiene.
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
