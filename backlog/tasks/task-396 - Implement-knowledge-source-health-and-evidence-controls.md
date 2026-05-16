---
id: TASK-396
title: Implement /knowledge source health and evidence controls
status: Done
assignee: []
created_date: '2026-05-16 00:51'
updated_date: '2026-05-16 02:59'
labels:
  - webui
  - knowledge
  - ux
  - feature
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-16-knowledge-source-health-evidence-controls-design.md
  - >-
    Docs/superpowers/plans/2026-05-16-knowledge-source-health-evidence-controls-plan.md
priority: high
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GET /api/v1/rag/source-health returns safe pre-query health for canonical Knowledge QA sources without altering search-response metadata.source_status.
- [x] #2 Knowledge QA shows source health before search and keeps search usable when health loading fails.
- [x] #3 Knowledge QA answer and evidence surfaces show compact trust/evidence controls without adding durable evidence persistence.
- [x] #4 Focused backend, frontend, extension parity, diff-check, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 backend source-health contract implemented. Focused tests passed: pytest -q test_source_health.py test_rag_source_health_endpoint.py test_source_contract.py test_unified_pipeline.py -k 'source_status or source_health or source_contract' (8 passed, 23 deselected). git diff --check passed. Bandit touched backend scope passed with 0 findings at /tmp/bandit_knowledge_source_health_task1.json.

Quality review fix: source-health no longer instantiates MultiDatabaseRetriever or source-specific databases during health polling. Endpoint derives configured source IDs from resolved request handles and existing Kanban DB files only; integration regression now fails if retriever construction is attempted. Re-ran focused backend tests (8 passed, 23 deselected), git diff --check, and Bandit touched backend scope (0 findings at /tmp/bandit_knowledge_source_health_task1_fix.json).

Spec re-review fix: removed source DB dependencies from /api/v1/rag/source-health. The endpoint now depends only on the authenticated user and computes source availability from existing per-user DB files without creating directories, schemas, retrievers, or request-scoped DB handles. Re-ran focused backend tests (8 passed, 23 deselected), git diff --check, and Bandit touched backend scope (0 findings at /tmp/bandit_knowledge_source_health_task1_fix2.json).

Task 2 frontend client/normalization implemented. Added tldw API ragSourceHealth client method, Knowledge QA source-health types/state, normalization helpers, and focused tests. Verification: vitest run sourceHealth.test.ts + tldw-api-client.rag-source-health.test.ts passed (4 tests); provider-focused vitest with KnowledgeQAProvider.feature-flags passed (5 tests total). Full UI tsc --noEmit remains blocked by existing unrelated baseline type errors in audio/chat/flashcards/playground/services tests; no sourceHealth or KnowledgeQAProvider errors appeared in the reported output. Optional ownership guard sanity check also still fails on existing OpenWebUI overlap inventory, not ragSourceHealth.

Task 3 source-picker UI slice complete: provider now loads /api/v1/rag/source-health once after tldw client initialization, KnowledgeQALayout threads sourceHealth/refresh through simple and detailed /knowledge surfaces, KnowledgeContextBar and CompactToolbar render compact health summaries, source picker rows show per-source status chips and retry, and KnowledgeReadyState now surfaces health failure, unavailable selected sources, and empty-source guidance without inline creation/import. Verification: red tests failed before implementation; focused Vitest passed 38 tests across sourceHealth, tldw-api-client.rag-source-health, KnowledgeContextBar.source-health, CompactToolbar, KnowledgeQALayout.behavior, KnowledgeReadyState.activation, and KnowledgeQAProvider.feature-flags. git diff --check passed. Full UI tsc remains red on existing baseline; rg over captured output found no KnowledgeQA/sourceHealth/touched-file errors.

Task 3 quality-review fixes: ready-state source-health notice now prioritizes unavailable/error selected sources over empty-source guidance and only treats sources as empty when available with explicit index_status=empty. CompactToolbar now wires onRefreshSourceHealth to a real one-line source-health retry button instead of carrying an unused prop. Verification: targeted Vitest passed CompactToolbar + KnowledgeReadyState activation (23 tests), broader Task 3 focused Vitest passed 40 tests, git diff --check passed.

Task 4 evidence/trust UI slice implemented. Added deterministic answer trust-summary helper, AnswerPanel trust note outside markdown body, clearer SourceCard Copy citation/Copy excerpt accessible names without duplicate actions, EvidenceRail source-action hint, and SourceList regression expectation for the citation accessible name. Save-to-note remains deferred because a backlink-preserving note handoff contract was not verified in this slice. Verification: red Task 4 tests failed before implementation; updated focused Vitest passed 63 tests across trustSummary, AnswerPanel.states, SourceCard.behavior, SourceList.behavior, and EvidenceRail.motion. git diff --check passed. Full UI tsc --noEmit remains blocked by existing unrelated baseline errors; filtered compiler diagnostics produced no KnowledgeQA/trustSummary/AnswerPanel/SourceCard/SourceList/EvidenceRail matches. Bandit is not applicable to this frontend-only slice.

Task 5 recovery/parity slice implemented. No-results recovery now separates pre-query Source readiness from post-query Search diagnostics, uses concrete handoff labels (Open Quick Ingest, Open source page), and hides Show nearest matches unless evidence exists. Low-quality recovery uses limited-evidence copy and receives selected source-health caveat counts from AnswerPanel without implying automatic web fallback. KnowledgeQALayout passes sourceHealth and selected sources into recovery, and extension /knowledge route parity was rechecked. Verification: red Task 5 tests failed before implementation; focused Vitest passed 23 tests across NoResultsRecovery.source-status, LowQualityRecoveryBanner, and KnowledgeQALayout.behavior. Extension parity passed 4 tests in apps/tldw-frontend. git diff --check passed. Full UI tsc remains blocked by unrelated baseline errors; filtered compiler diagnostics produced no KnowledgeQA/NoResultsRecovery/LowQualityRecoveryBanner/KnowledgeQALayout/AnswerPanel matches. Bandit is not applicable to this frontend-only slice.

Task 6 final verification completed after rebasing onto origin/dev. Backend focused pytest passed: 8 passed, 4 warnings. KnowledgeQA focused Vitest passed: 14 files, 117 tests. Extension parity passed: 1 file, 4 tests. Browser smoke passed after rerunning Playwright outside the macOS sandbox: /knowledge navigation/search-bar test passed against frontend localhost:18001 and backend 127.0.0.1:8000. git diff --check passed. Bandit touched backend files passed with 0 findings at /tmp/bandit_knowledge_source_health.json. Opened PR https://github.com/rmusser01/tldw_server/pull/1745.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented /knowledge source health and evidence controls. Source health is pre-query and read-only, post-query metadata.source_status remains backward compatible, evidence actions reuse existing handoffs, recovery copy separates source readiness from search diagnostics, and /knowledge remains QA-only. PR: https://github.com/rmusser01/tldw_server/pull/1745.
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
