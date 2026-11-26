---
id: TASK-13219
title: Implement ACP MCP tool-result context experiment
status: Done
assignee: []
created_date: 2026-09-08 04:46
updated_date: 2026-09-08 06:17
labels: []
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/2932
documentation:
- Docs/Design/ACP_Tool_Result_Context_Experiment.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved ACP/MCP experiment: compare unchanged output, deterministic excerpts, and optional worker-selected exact evidence. Preserve per-run source retrieval, authorization, provider boundaries, and disabled behavior. Separate from TASK-13218 RAG work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Disabled behavior preserves tool output and makes no worker calls.
- [x] #2 Enabled modes bound evidence and provide authorized run-local exact reads.
- [x] #3 Worker extraction verifies source ranges and handles failure, timeout, and cancellation.
- [x] #4 Comparison harness reports context, latency, worker usage, and evidence recovery without invented cost or quality claims.
- [x] #5 Regression and property tests, documentation, review, lint, and Bandit pass.
- [x] #6 Cancellation during result preparation preserves exactly one raw event for an already completed tool, then propagates cancellation.
- [x] #7 Large-result ranking yields for cancellation and bounds query scoring work without losing exact source offsets.
- [x] #8 Benchmark evidence presence excludes generated wrapper text and verifies returned source excerpts.
- [x] #9 Internal result reads do not count as first tools or suppress real typed-tool fallback metrics.
- [x] #10 New ACP result-context tests are assigned to the supported full-suite CI matrices and pass the shard coverage guard.
- [x] #11 Worker self-cancellation falls back without aborting the run, while caller and run cancellation still propagate.
- [x] #12 Worker request construction stops at its encoded byte budget and yields for cancellation.
- [x] #13 New result-context contracts, test helpers, and benchmark helpers have useful docstrings and annotations; missing sources raise a classifiable central exception.
- [x] #14 Superpowers review confirms the worker receives only text from the current user turn and actual MCP text output.
- [x] #15 Superpowers review confirms source rereads use the originally approved call and do not depend on mutable callback arguments.
- [x] #16 Superpowers review confirms bounded UTF-8 accounting, strict worker JSON selection, and cancellation metrics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Original implementation and four remediation stages complete. PR review follow-up completed reproduction, bounded worker encoding/cancellation fixes, contract documentation and annotations, and independent verification. Temporary implementation plans were removed after completion. Persistent design: Docs/Design/ACP_Tool_Result_Context_Experiment.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Isolated worktree on codex/acp-tool-result-experiment. Baseline: 40 ACP runner/adapter/caller tests pass. MCP task tools stalled; using official Backlog CLI. Task ID 13219 avoids the concurrently assigned RAG task 13218.

Implemented result policy, runner/adapter integration, and replay harness. Initial focused suite: 70 passed; harness: 3 passed. Independent review identified malformed worker output, unbounded cancellation cleanup, and retained store after callback failure; all reproduced with failing tests and fixed (6 malformed-output tests and 4 lifecycle tests pass). Bandit on five touched runtime/script files reports zero findings. Final verification in progress.

Final verification: 114 targeted ACP tests passed (4 warnings), covering the new policy, integration and benchmark plus existing LLM/agent runners, adapter, caller, tool presentation, ToolGate and governance. Ruff check passed on all eight touched Python files; new files passed format check. Scoped Bandit on five runtime/script files: zero findings and zero scan errors. git diff --check passed. Independent review findings were reproduced with failing tests and all resolved, including preserving usage on invalid worker text and checking actual returned recovery text. Fixture replay exercised all three modes on three sources; all expected evidence was present or recovered, including one deterministic-excerpt reread. Raw inputs 12029/10852/15421 bytes; deterministic outputs 4096/4095/4096 bytes; fixture-oracle outputs 457/305/265 bytes. These are decoded UTF-8 byte measurements, not token or cost claims.

User approved all four reproduced review findings for remediation. Continue in the existing isolated worktree. Baseline commit c45caf4520; 47 feature tests passed during read-only review. Scope: cancellation result delivery, cooperative ranking, benchmark source evidence scoring, and run-first metrics. Add regression tests before each fix, then run focused ACP, Ruff, Bandit, and independent review.

Review remediation complete. Tests were added before implementation: 6 cancellation/metrics failures (off baseline passed), 3 ranking responsiveness failures, and 2 benchmark header-collision failures. All fixes passed independently: 18 runner integration cases, 53 policy+integration cases, and 6 benchmark cases. Final focused ACP regression suite: 126 passed, 4 warnings in 10.98s. Ruff check on six touched Python files and format check on five formatted files passed. Bandit on three touched runtime/script files: zero findings and zero scan errors. Fixture replay: all 9 rows recovered expected evidence. Original 4 MiB/200-term probe delayed a scheduled 20 ms cancellation for 1.184s; updated excerpt cancellation completed in 23.9ms locally. Independent review reproduced successful cancellation/result preservation, correct real-tool fallback metrics, correct header-collision scoring, and cooperative cancellation (~22ms excerpt/~32ms worker fallback). No actionable issues remain in the reviewed fixes. Only documentation and task tracking changed after final runtime validation. Temporary implementation plan removed on completion per AGENTS.md.

PR #2932 opened against dev: https://github.com/rmusser01/tldw_server/pull/2932. The original local dev checkout diverged from origin/dev, so only the two task commits were rebased onto current origin/dev; git range-diff confirms identical patches. Published implementation commits are 446fdd5fe6 and 317eb8f63a. The resulting PR contains only the 11 task files. Post-rebase validation: 132 targeted ACP tests passed (including current WebSocket broadcaster coverage), Ruff passed, scoped Bandit reported zero findings. PR description includes all four reproduced findings and fixes, validation, rollout limits, and a pending requester-authored Change summary required before merge. Continuing CI/review checks.

PR follow-up: the Shard coverage guard failed because the three new ACP result-context test files were absent from every CI shard. Reproduced locally with check_shard_coverage.py. Plan: add those explicit files to the existing platform-mcp-core shard in all five full-suite matrices, rerun the guard and the new suites with CI plugin settings, then commit and push. Existing coverage baseline and ignore lists remain unchanged.

CI shard fix verified: check_shard_coverage.py now reports new_uncovered=0; parsed YAML confirms all five platform-mcp-core entries include the three new files. The 59 new tests pass with CI plugin-autoload settings (6 warnings). No new runtime code in this commit, so prior scoped Bandit still applies. Qodo review additionally reported request-serialization responsiveness, worker self-cancellation, and contract documentation/typing issues; validating these next.

Confirmed Qodo runtime findings locally: a worker raising CancelledError escapes preparation; a 4 MiB NUL-heavy source rejected at the 32 KiB worker budget peaks at 93,973,055 temporary traced bytes and delays the first event-loop heartbeat by 156 ms. Add public behavior regressions before fixing. Documentation/type-hint feedback matches project guidance; use narrow central source-not-found and worker-cancelled exceptions while retaining existing ValueError handling compatibility.

PR review follow-up complete: five new regression cases failed before implementation (worker-only cancellation at policy/run boundaries, temporary allocations for oversized source/question JSON, cancellation before worker dispatch); two additional cases preserve exact encoded sizes for escapes and Unicode. Incremental JSON construction stops at the byte limit and yields; early rejections keep full request bytes unknown. A cancelled child worker raises a central classifiable error and falls back, while caller/event cancellation propagates. Missing-source handles now raise a central ValueError-compatible exception. All new definitions have docstrings and parameter/return annotations, with explicit casts only for deliberately malformed provider fixtures. Final validation: 143 targeted ACP tests passed (4 warnings), Ruff passed on all nine PR Python files, format checks passed on the five new files, scoped mypy passed those five files, Bandit on four follow-up runtime/script paths found zero findings and zero scan errors, shard guard passed, and all nine fixture replay rows recovered expected evidence. Independent review passed 225 encoded-size boundary probes and cancellation race checks without actionable findings. Only task/docs changed after runtime verification. GitHub shard guard passed; remaining hosted checks continue on the PR. Removed the completed temporary follow-up plan.
Rebased PR #2932 onto dev at 6d31f3d1a7. Resolved the CI workflow conflict by using the new platform-acp shard on Ubuntu 3.12 and retaining explicit coverage for the three experiment test files in the other four full-suite matrices. Preserved the separate MCP in-app shards. Git range-diff confirmed the four remaining patches replayed identically. Canonicalized this task with backlog-py for the current format ratchet. Validation on Python 3.12: 335 focused ACP and CI contract tests passed, shard coverage had zero newly uncovered files, task format and ID checks passed, Ruff and scoped mypy passed, and Bandit reported zero findings or scan errors.
ADR required: no. The experiment uses the existing injected caller and retains source text only for the current run. Provider-backed callers follow Docs/ADR/025-llm-provider-adapter-routing-and-overrides.md. Persisted ACP session ownership follows Docs/ADR/016-acp-session-and-orchestration-persistence.md.
Requester asked for a fresh Superpowers requesting-code-review before merge. Disarmed auto-merge on PR #2932. Independent review of 6d31f3d1a7..902bcd6d87 reproduced six findings. Follow-up plan: Docs/Plans/IMPLEMENTATION_PLAN_acp_result_merge_review.md. ADR required: no. The fixes enforce the existing experiment boundaries and do not change ADR-016 or ADR-025.
Pre-merge Superpowers review complete. Six original findings produced 13 failing cases: latest-turn text selection, non-text MCP passthrough, approval argument isolation, UTF-8 accounting, duplicate worker JSON fields, and the cancelled outcome metric. Follow-up review found a full multipart question allocation. Reproduced it separately, then bounded assembly while preserving the worker budget fallback. The independent reviewer closed all seven findings and approved the fixes after checking 560 multipart prefixes and three runner cases. Final validation: 347 ACP and CI contract tests passed on Python 3.12 (5 warnings), including 12 new cases plus strengthened cancellation assertions. Ruff, scoped mypy, Bandit, the shard guard, Backlog checks, and all nine fixture comparison rows passed. PR #2932 can return to the queue under the requester’s existing change-summary waiver.
PR #2932 backend-required job 138953600337 reported three overly mocked ACP integration tests, an assertion in the recording caller, and eight mypy errors across three PR Python files. Both checks reproduced locally against 0973d22d4b. Auto-merge was disarmed for the deterministic backend corrections and the branch rebased to include the dev workflow update. Preserve behavioral coverage and enforce the existing quality baseline. The E2E Critical Flashcards trace also showed an unexpected Browse cards drawer intercepting the note link after a forced review-mode click; verify any test correction against the real journey.
CI remediation: rebased onto dev at 0e3c3ffbc5 with all seven prior patches unchanged. Fixed the test-quality findings by using real governance and a test-local metrics recorder, and checked that the model receives the expected authorized evidence. Narrowed the ACP source context and reconciled the remaining type annotations on the files checked by backend-required. In the Flashcards page object, skip a selected review mode and use a normal Playwright click plus a checked-state assertion for a mode change. The CI trace confirmed the removed forced click opened the Browse cards drawer. Local validation passed: 194 focused pytest cases, all 84 tenant-isolation ratchet cases, exact changed-file mypy, Ruff, Flashcards ESLint, scoped Bandit with zero findings, and shard coverage with zero new gaps. The full Notes-to-Flashcards journey and the Cram-mode case each passed in Chromium against an isolated real backend. Independent review verified the strengthened adapter and metrics assertions and approved requeue. Hosted checks will run on the rebased head.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the default-off ACP/MCP result context experiment, original review fixes, and cross-platform CI coverage. Rebasing preserved the runtime patches and current CI coverage. Independent pre-merge review closed all seven further findings: latest-turn isolation, non-text MCP passthrough, stable authorization arguments, bounded byte counts, duplicate JSON rejection, cancellation metrics, and bounded multipart input. Final validation was 347 focused passing tests, lint, typing, Bandit, Backlog and shard checks, and fixture replay. Live-provider economics and complete-task quality were outside the experiment’s fixture validation.
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
