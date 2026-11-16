# Political PDF strict Claims implementation plan

**Goal:** Check chapter, overview and QA claims against their actual cited source passages and preserve reviewable findings end to end.
**Architecture:** Reuse Claims_Extraction behind the existing Jobs and workspace APIs; persist versioned findings in existing artifact/message metadata. Preserve generated drafts when checks fail.
**Spec:** [Strict Claims design](POLITICAL_PDF_STRICT_CLAIMS_2026_11_16.md).
**ADR check:** ADR required: yes; ADR-096. Existing ADR-003/015/018/025 remain governing.

## Stage 1: Shared strict verification and owned QA boundary
**Goal:** Add the minimal citation-bound Claims entry point and authenticated QA integration.
**Success Criteria:** Only explicit semantic support from cited owned passages can pass; malformed/uncited/unavailable/overflow inputs remain reviewable.
**Tests:** Claims unit/property cases for wrong evidence, qualifiers, attribution, compound claims, no/unknown IDs, incomplete judgment, limits; workspace endpoint ownership/range/provider tests.
**Status:** Complete

- [ ] Backend worker `pr_backend_verify` owns Claims_Extraction, the QA operation/schema and related tests under TASK-15505.16.
- [x] Write and observe failing tests, implement the shared contract, run connected tests and Bandit. Backend release: 159 connected tests, 13 final API/governor tests; bounded total deadline, Unicode coverage and late-coroutine cleanup qualified. Ruff/Bandit clean.

## Stage 2: Chapter and overview Job integration
**Goal:** Check extracted claims and final chapter/overview prose, checkpoint reports and retain drafts on check failures.
**Success Criteria:** Each completed batch/chapter/overview has a report tied to exact evidence; retry/cancel/source guards hold and old unchecked state is explicit.
**Tests:** Extend `tests/Research_Workspace/test_section_summary_jobs.py` with rejection, bounded failure, source/context binding, checkpoint resume and Markdown findings assertions.
**Status:** Complete

- [ ] Root owns `core/Research_Workspace/section_summaries.py`, a focused section adapter if needed, and its Job tests.
- [x] First assert a completed fake generation produces a persisted strict report; watch failure before implementing. Three expected failures observed.
- [x] Add tests for verifier failure and resume, then implement checkpoint and renderer integration. Connected section suite: 79 passed; touched adapter lint and production Bandit pass. Shared-service integration remains to qualify.

## Stage 3: Final QA, review and portable findings
**Goal:** Show findings and retain them with the exact finalized answer/artifact across review/export/restore.
**Success Criteria:** Pending/unsupported/unchecked states are distinguishable, findings link to evidence, stale verification cannot settle onto a different answer/workspace.
**Tests:** Final QA callback tests; chapter/overview finding UI tests; Markdown/offline/complete-pack round trips, migration and stale-state guards.
**Status:** Complete

- [ ] Frontend worker `pr_frontend_verify` owns UI/services/export changes and tests under TASK-15505.17.
- [x] Confirm the shared report contract, write failing integration tests, then wire the existing review and export paths. Frontend release: 683 connected tests, full types, zero new lint findings; independent review clear. Completed chapter checks retry without regeneration; unavailable QA checks preserve the original answer for review.

## Stage 4: Repeatable eval and fresh Luna evidence
**Goal:** Make strict support checking repeatable across models and run the same PDF against a frozen implementation.
**Success Criteria:** QA and generated artifacts expose real verification receipts; coverage and semantic outcomes stay distinct; regressions exercise known failures without invented human labels.
**Tests:** Runner contract tests and fixture failures; actual Luna chapter/overview/QA run; UI and export/restore readback.
**Status:** In Progress

- [ ] Eval worker `pr_scope_review` owns `Helper_Scripts/Evals/run_political_pdf_eval.py`, fixtures/config/tests under TASK-15505.18.
- [ ] Root coordinates the frozen local runtime and paid provider calls only after connected tests pass.
- [ ] Retain receipts, inspect failures against source and document measured limits.
- [ ] Qualify references-only extraction guidance and reuse of unchanged validated terminal receipts after the live SBA tail failure. Preserve the first response; finish Job 49 before restarting the runtime and explicitly resuming its failed portion. Missing/error/stale checks remain retryable, and no verdict or uncovered count is promoted.

## Stage 5: Review, delivery and CI
**Goal:** Update PR #3480 against dev with reviewed implementation and verifiable results.
**Success Criteria:** Touched tests/types/lint/Bandit/hooks pass; CI failures are repaired or a concrete external blocker is documented; human Change summary remains untouched.
**Tests:** Scoped backend/frontend integration, security/contract ratchets, independent review and current-head hosted CI.
**Status:** In Progress

- [x] Review each worker diff, verify cross-layer integration, fix new findings, and keep Backlog records current. Root independent 274 backend/eval/ratchet tests and 76 frontend contract/export tests pass. Review found and repaired incomplete receipt coverage and verifier identity gaps. Hosted old-head locale mirror failure repaired with the existing generator; parity tests pass.
- [ ] Commit/push without bypassing hooks; update PR description with ADR assessment and honest validation status.
- [ ] Follow current-head checks to a terminal result; do not merge, rebase an armed PR or manually dispatch required gates.
- [ ] Remove this task-owned plan only when every stage is complete; retain design and outcome report.
