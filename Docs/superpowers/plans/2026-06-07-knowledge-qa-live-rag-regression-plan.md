# Knowledge QA Live RAG Regression Remediation Plan

**Goal:** Fix the Stage 6 live backend regressions where Knowledge QA can return uncited/stale chat answers, miss no-results recovery, or leak Media DB sources into exact-note scoped searches.

**Backlog Task:** TASK-2279.11

## Stage 1: Backend Scope Contract

**Goal:** Ensure retrieval fallbacks cannot broaden a request beyond the selected sources or exact IDs.

**Success Criteria:** A request with `sources=["notes"]` and `include_note_ids=[...]` never returns Media DB documents unless explicit scope-broadening metadata is set.

**Tests:** Add a backend unit regression that forces notes retrieval to return no docs and asserts the direct Media DB fallback is skipped for notes-only scope.

**Status:** Complete

## Stage 2: Knowledge QA Self-History Isolation

**Goal:** Prevent Knowledge QA's own saved question/answer messages from becoming local evidence for future Knowledge QA searches.

**Success Criteria:** Chat retrieval excludes conversations with `source="knowledge_qa"` while preserving ordinary chat-history retrieval.

**Tests:** Add a backend chat retriever regression with one `knowledge_qa` conversation and one normal chat conversation matching the same query.

**Status:** Complete

## Stage 3: WebUI Evidence and Streaming Recovery

**Goal:** Ensure the WebUI presents inspectable evidence for cited answers and does not finalize an empty streaming response when the standard RAG endpoint can return scoped evidence.

**Success Criteria:** Cited answers with fewer than three returned sources still open the evidence rail, and a stream that completes without usable evidence falls back to non-stream RAG search.

**Tests:** Add frontend regressions for cited evidence auto-open and empty-stream fallback, then run the KnowledgeQA provider/layout/answer-panel Vitest suites.

**Status:** Complete

## Stage 4: Live Gate Recheck

**Goal:** Re-run the deterministic live API/browser checks against the seeded fixture.

**Success Criteria:** Scoped note API returns only the seeded note or no scoped evidence; no-match no longer uses Knowledge QA self-history; WebUI live gate improves with any remaining failures documented.

**Tests:** Run focused backend tests, the fixture smoke tests, Bandit on touched backend code, and the WebUI live gate against a launched backend.

**Status:** In Progress

**Verification completed:**
- Backend focused suite: `21 passed, 6 warnings`.
- Frontend KnowledgeQA provider/layout/answer-panel suites: `99 passed`.
- TypeScript UI package check: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit -p tsconfig.json` completed successfully before the rebase.
- `git diff --check` completed successfully.
- Bandit touched backend/test scope wrote `/tmp/bandit_knowledge_qa_live_regressions.json` with zero findings.
- Scope guard over touched Knowledge QA files found no out-of-scope learning-mode terminology.

**Current blocker:** The live Playwright WebUI gate cannot be rerun in this session because the sandbox blocks the local Next server bind (`listen EPERM 0.0.0.0:8080`) and the required escalation was rejected by the approval reviewer due the account usage limit until 4:42 PM. Direct non-health localhost POST probes are also blocked by the sandbox and share the same approval limitation.

**Post-rebase note:** After rebasing on the latest `origin/dev`, the same UI package TypeScript check fails in `src/components/Option/ResearchWorkspace/__tests__/WorkspaceCapabilityRemediation.test.tsx` because the test fixture makes `workspace_profile` optional while `WorkspaceCapabilitiesResponse` now requires it. This is outside the touched Knowledge QA scope.
