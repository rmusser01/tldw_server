---
id: TASK-2316
title: Address PR 2309 review feedback
status: Done
modified_files:
- Docs/superpowers/plans/2026-06-08-pr-2309-review-remediation-plan.md
- apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.persistence.test.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-08: Rebased local branch `codex/knowledge-qa-follow-on-remediation` onto latest `origin/dev` before remediation. Added focused red/green coverage for stale retry-sync trust persistence, selected-note stable source metadata, selected-note SQL allowlist bounding, sync-adapter offload, chat source lookup caching, explicit note selection source fallback/cache bypass, and effective retrieval search-mode fallback. Fixed the provider retry path to persist recomputed post-sync trust context, added evidence-origin inference for heuristic trust normalization, stabilized notes retriever metadata, bounded explicit-note SQL placeholders, offloaded explicit note retrieval, cached chat source exclusion lookups, and made explicit source selections authoritative while disabling cache reuse. Preserved classification `skip_search` generation-only behavior and switched retrieval-error fallback to `retrieval_search_mode`.

2026-06-08: Addressed extension/test/docs review comments. The extension launch health test now uses the shared conditional launch wrapper instead of unconditional `test.fail(true)` and uses a case-insensitive setup button locator. Manifest parsing now reports invalid JSON with the manifest path and has direct unit coverage. AnswerPanel test fixtures now use narrower Knowledge QA union/result types. Scope validation no longer carries an unreachable web branch in local-source matching. `response_mapping.py` imports the shared evidence text keys from `trust_contracts.py` and documents `_web_fallback_used`. Requested pytest unit markers were added, and malformed Backlog final-summary markers/task-id markdown were repaired.

2026-06-08 verification:
- Red runs failed before implementation for backend live regressions and retry-sync trust persistence.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG/test_knowledge_qa_live_regressions.py tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline.py::TestUnifiedPipeline::test_skip_search_classification_bypasses_retrieval -q` passed 14 tests.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG/test_knowledge_qa_uat_fixtures.py tldw_Server_API/tests/RAG/test_knowledge_trust_contracts.py -q` passed 12 tests.
- `bunx vitest run src/components/Option/KnowledgeQA/__tests__` passed 55 files / 486 tests.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc -p tsconfig.json --noEmit` passed in `apps/packages/ui`; the first run without extra heap hit Node OOM.
- `bun run compile` passed in `apps/extension`.
- `node apps/node_modules/.bun/vitest@4.0.18+0560a95e2bbfec09/node_modules/vitest/vitest.mjs run --config /private/tmp/vitest-extension-utils.config.ts` passed 3 extension utility tests.
- `npx playwright test tests/e2e/utils/extension-launch-health.spec.ts --reporter=line` built the extension and reported 1 skipped because this host still exposes no extension targets; this is now the shared conditional launch-unavailable skip, not an expected-failure bypass.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py tldw_Server_API/app/core/RAG/rag_service/response_mapping.py tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py -f json -o /tmp/bandit_pr2309_review_remediation.json` reported 0 findings, 0 errors, 1 existing skipped nosec.
- `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2309 on latest `origin/dev` and addressed the actionable review feedback without adding flashcard behavior. Fixed Knowledge QA retry-sync trust persistence, selected-note evidence identity, explicit scoped-source retrieval/cache behavior, effective-mode fallback, classification skip-search generation, chat source lookup caching, extension launch-health bypass, manifest parse diagnostics, requested test markers, and malformed Backlog markers. Verification covered focused and full Knowledge QA frontend tests, backend RAG regressions, extension compile/utility/launch-health checks, TypeScript, Bandit, and diff hygiene.
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
