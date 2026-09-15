---
id: TASK-13219
title: Implement ACP MCP tool-result context experiment
status: Done
assignee: []
created_date: '2026-09-08 04:46'
updated_date: '2026-09-08 06:17'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2932'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Original implementation and four remediation stages complete. PR review follow-up completed reproduction, bounded worker encoding/cancellation fixes, contract documentation and annotations, and independent verification. Temporary implementation plans were removed after completion. Persistent design: Docs/Design/ACP_Tool_Result_Context_Experiment.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
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
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the default-off ACP/MCP result-context experiment and resolved all four follow-up review findings. Completed tool results are emitted exactly once before optional-selection cancellation propagates. Deterministic ranking uses bounded query scoring, casefolds each segment once, and yields between batches; successful worker selection skips ranking. Benchmark evidence must occur in exact returned source excerpts, excluding generated wrappers. Internal rereads no longer replace real first-tool/fallback metrics. Added 12 regression cases, updated README/design, and passed 126 targeted tests, Ruff, Bandit and independent review. Real-model cost, full-task quality and whole-repository tests remain outside this scoped experiment; fixture replay is explicitly labeled and makes no real-model savings claim. Branch: codex/acp-tool-result-experiment.

PR #2932 targets dev and includes the original four findings plus all five additional Qodo findings and their resolutions. Added seven PR-review regression/boundary cases and CI enrollment for all three new test files. Latest focused validation is 143 passing tests plus Ruff, mypy, Bandit and independent boundary/race review. Requester-authored Change summary remains required before merge.
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
