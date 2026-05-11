---
id: TASK-241.1
title: Define Persona Chat judge evaluation contract
status: Done
assignee: []
created_date: '2026-05-11 05:13'
updated_date: '2026-05-11 05:35'
labels:
  - persona
  - chat
  - evaluations
  - stage-2
  - contract
  - tests
dependencies:
  - TASK-245
  - TASK-250
  - TASK-253
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1566'
  - 'https://github.com/rmusser01/tldw_server/issues/1543'
documentation:
  - Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md
  - Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md
  - Docs/superpowers/plans/2026-05-11-persona-chat-judge-contract.md
parent_task_id: TASK-241
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Contract-first slice for #1566. Define the optional calibrated Persona Chat judge input/output/calibration contract and add deterministic tests for contract fixtures without executing a live judge or changing runtime Persona Chat behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Persona Chat judge contract document defines inputs, outputs, calibration labels, privacy limits, and offline-only V1 behavior.
- [x] #2 A redaction-safe fixture artifact includes at least one positive and one negative Persona Chat quality case tied to existing PC-* labels.
- [x] #3 Deterministic tests validate fixture shape and calibration expectations before any judge output is considered usable.
- [x] #4 No runtime Persona Chat path or production evaluation execution behavior changes in this slice.
- [x] #5 Focused tests, Bandit applicability, and diff hygiene are recorded in Backlog.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a contract-first implementation plan at Docs/superpowers/plans/2026-05-11-persona-chat-judge-contract.md.
2. Write and red-verify a pytest contract guard for the judge fixture artifact.
3. Add a Markdown judge contract and minimal calibration fixture JSON with one pass and one fail case.
4. Link the contract from the Stage 2 follow-up review artifact.
5. Run focused pytest, doc placeholder scan, git diff hygiene, Bandit, update Backlog, commit, and open PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Selected approach for #1566: contract-first docs/test slice before any executable judge harness. Created isolated worktree on codex/persona-chat-judge-contract from origin/dev.

Verification recorded: pytest test_persona_chat_judge_contract.py passed with 3 tests; placeholder scan returned no matches; git diff --check passed; Bandit on the touched pytest validator produced zero findings in /tmp/bandit_persona_chat_judge_contract.json. No runtime Persona Chat or production evaluation execution code was changed.

Review-fix pass for PR #1569: addressing Qodo and CodeRabbit findings in the Persona Chat judge contract validator. Actionable items are docstrings, required score keys and numeric types, robust taxonomy parsing, expanded local-path redaction checks, and deterministic_labels validation.

Review-fix verification: added regression coverage for markdown taxonomy variants, local path redaction variants, strict score schema, and deterministic label validation. Focused pytest now passes 10 tests; Bandit review-fix report has zero findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Defined the offline-only Persona Chat judge evaluation contract for #1566, added redaction-safe calibration fixture cases tied to PC-CASE-008 and PC-CASE-015, and added a deterministic pytest guard. Review fixes on PR #1569 added module/function docstrings, robust taxonomy parsing, expanded local path redaction checks, deterministic_labels validation, exact required score-key validation, and strict score value typing.
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
