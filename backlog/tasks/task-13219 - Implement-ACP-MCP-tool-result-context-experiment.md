---
id: TASK-13219
title: Implement ACP MCP tool-result context experiment
status: Done
assignee: []
created_date: '2026-09-08 04:46'
updated_date: '2026-09-08 05:15'
labels: []
dependencies: []
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Completed all four stages: exact result policy; runner/adapter integration; reproducible comparison harness; independent review and delivery. The task-specific temporary implementation plan was removed on completion per AGENTS.md. Persistent design: Docs/Design/ACP_Tool_Result_Context_Experiment.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Isolated worktree on codex/acp-tool-result-experiment. Baseline: 40 ACP runner/adapter/caller tests pass. MCP task tools stalled; using official Backlog CLI. Task ID 13219 avoids the concurrently assigned RAG task 13218.

Implemented result policy, runner/adapter integration, and replay harness. Initial focused suite: 70 passed; harness: 3 passed. Independent review identified malformed worker output, unbounded cancellation cleanup, and retained store after callback failure; all reproduced with failing tests and fixed (6 malformed-output tests and 4 lifecycle tests pass). Bandit on five touched runtime/script files reports zero findings. Final verification in progress.

Final verification: 114 targeted ACP tests passed (4 warnings), covering the new policy, integration and benchmark plus existing LLM/agent runners, adapter, caller, tool presentation, ToolGate and governance. Ruff check passed on all eight touched Python files; new files passed format check. Scoped Bandit on five runtime/script files: zero findings and zero scan errors. git diff --check passed. Independent review findings were reproduced with failing tests and all resolved, including preserving usage on invalid worker text and checking actual returned recovery text. Fixture replay exercised all three modes on three sources; all expected evidence was present or recovered, including one deterministic-excerpt reread. Raw inputs 12029/10852/15421 bytes; deterministic outputs 4096/4095/4096 bytes; fixture-oracle outputs 457/305/265 bytes. These are decoded UTF-8 byte measurements, not token or cost claims.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented opt-in off/excerpt/worker policies for the LLM-driven MCP embedding interface. Exact source segments and authorized run-local reads preserve recoverability; default behavior remains unchanged. Worker selection is bounded and validated, with deterministic fallbacks, cancellation handling, explicit source cleanup and truthful optional usage metadata. Added documented configuration and a labeled fixture replay with an optional real-worker factory. Verified by 114 targeted tests, Ruff, Bandit and independent review. No full-repository test run or paid/real-model run was performed; task success, total cost and real-model quality remain rollout measurements. No concrete production LLMCaller, public REST configuration, or native external ACP interception is introduced. No implementation blockers remain. Branch: codex/acp-tool-result-experiment.
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
