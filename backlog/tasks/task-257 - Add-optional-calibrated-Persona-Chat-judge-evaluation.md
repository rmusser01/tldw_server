---
id: TASK-257
title: Add optional calibrated Persona Chat judge evaluation
status: Done
assignee: []
created_date: '2026-05-11 05:14'
updated_date: '2026-05-11 05:26'
labels:
  - persona-chat
  - evaluations
  - stage-2
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1566'
  - 'https://github.com/rmusser01/tldw_server/issues/1543'
  - 'https://github.com/rmusser01/tldw_server/pull/1570'
documentation:
  - Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md
  - Docs/Reviews/PERSONA_CHAT_JUDGE_EVALUATION_CONTRACT_2026_05_11.md
  - Docs/superpowers/plans/2026-05-11-persona-chat-judge-evaluation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Persona Chat Stage 2 slice from GitHub issue #1566. Add an optional/offline Persona Chat judge contract tied to the existing deterministic trace/error taxonomy and fixture cases, with calibration-oriented comparison before any judge output is treated as a quality signal. Keep runtime Persona Chat behavior unchanged and avoid creating a parallel evaluation service.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document a calibrated Persona Chat judge contract tied to deterministic fixture cases and prompt-preview diagnostics.
- [x] #2 Compare judge predictions against expected fixture labels before surfacing outputs as quality signals.
- [x] #3 Keep judge execution optional/offline and avoid runtime Persona Chat gating changes.
- [x] #4 Cover at least one positive and one negative Persona Chat quality case with focused tests.
- [x] #5 Record focused tests, Bandit results for touched Python paths, and diff hygiene in the Backlog task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write failing tests for fixture-derived judge input normalization, binary prompt shape, and calibration comparison. 2. Add a minimal offline Evaluations helper with no endpoint or live LLM execution. 3. Document the contract, run focused pytest, Bandit, and diff hygiene, then update Backlog.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red/green: test_persona_chat_judge.py first failed with ModuleNotFoundError for persona_chat_judge, then passed after adding the helper. A second red/green pass tightened calibration so missing predictions are required only for dimensions represented by predictions or fixture labels. Verification: python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge.py tldw_Server_API/tests/Persona/test_persona_chat_quality_fixtures.py -v passed with 11 tests; python -m bandit -r tldw_Server_API/app/core/Evaluations/persona_chat_judge.py reported no issues; git diff --check passed.

Self-review fix: prompt generation now excludes fixture labels so calibration ground truth is not leaked to a judge. The regression test first failed on fixture_labels appearing in the prompt, then passed after removing that field; focused pytest, Bandit, and git diff --check were rerun successfully.

PR opened: https://github.com/rmusser01/tldw_server/pull/1570

PR review fixes: documented known judge failure modes and residual V1 risks, added required identity-field validation for judge inputs, rejected blank/duplicate calibration keys, and rejected invalid judge result values before metric calculation.

PR review verification: python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge.py tldw_Server_API/tests/Persona/test_persona_chat_quality_fixtures.py -v passed with 20 tests; python -m bandit -r tldw_Server_API/app/core/Evaluations/persona_chat_judge.py reported no issues; git diff --check passed.

Docstring follow-up: added concise helper docstrings in persona_chat_judge.py to address the CodeRabbit docstring coverage warning for the new module.

Additional review follow-up: added a focused regression test proving calibrate_persona_chat_judge_predictions rejects invalid result values even when a prediction object bypasses dataclass post-init validation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added an optional/offline Persona Chat judge contract and calibration helper tied to existing deterministic quality fixtures. The implementation defines binary judge dimensions, fixture-derived judge inputs, structured prompt generation, and label-based calibration metrics without changing runtime Persona Chat behavior or adding a live eval service.

PR review follow-up hardened the calibration contract by rejecting missing identity fields, duplicate input/prediction keys, and invalid judge result values before metric calculation. The contract documentation now records known judge failure modes, residual V1 risks, and the offline/runtime boundary.
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
