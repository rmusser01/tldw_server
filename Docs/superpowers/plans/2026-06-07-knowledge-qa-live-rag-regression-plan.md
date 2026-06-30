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

**Status:** Complete

**Verification completed:**
- Backend focused regression: `1 passed, 7 deselected, 6 warnings` for `test_selected_note_scope_survives_webui_chunk_type_filter`.
- Backend focused suites: `17 passed, 6 warnings` for `test_knowledge_qa_live_regressions.py` and `test_knowledge_trust_contracts.py`.
- Frontend KnowledgeQA provider/layout/answer-panel/search-details suites: `64 passed`.
- Exact WebUI scoped-note POST replay returned the selected note, `chunk_type: "text"`, and `chunk_type_filter_after: 1`.
- WebUI live backend gate: `6 passed` for `e2e/ux-audit/knowledge-qa-live-backend.spec.ts`.
- Research Workspace post-rebase type fixture follow-up tracked separately as `TASK-2315`; its focused test passed (`6 passed`) and the UI package TypeScript check completed successfully.
- `git diff --check` completed successfully before plan/backlog finalization.
- Bandit touched backend/test scope with pytest assert warnings excluded wrote `/tmp/bandit_knowledge_qa_follow_on_retry_skip_b101.json` with zero findings.
- Scope guard over touched Knowledge QA files found no out-of-scope learning-mode terminology.

**Current blocker:** None for the Knowledge QA live gate. The backend and frontend had to be launched with escalation because the sandbox blocks local port binding and browser-driven localhost access.
