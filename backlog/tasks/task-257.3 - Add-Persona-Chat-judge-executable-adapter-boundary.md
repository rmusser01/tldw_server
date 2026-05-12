---
id: TASK-257.3
title: Add Persona Chat judge executable adapter boundary
status: Done
assignee: []
created_date: '2026-05-12 03:02'
updated_date: '2026-05-12 03:15'
labels:
  - persona-chat
  - evaluations
  - stage-2
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1591'
  - 'https://github.com/rmusser01/tldw_server/issues/1566'
documentation:
  - Docs/Reviews/PERSONA_CHAT_JUDGE_EVALUATION_CONTRACT_2026_05_11.md
  - Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md
parent_task_id: TASK-257
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1591 as the next optional Persona Chat Stage 2 judge slice under #1566. Add a narrow offline executable adapter boundary that can run explicit Persona Chat judge prompts through an injected or existing LLM completion seam and convert strict JSON responses into bounded PersonaChatJudgePrediction records. Preserve existing V1 boundaries: optional/offline execution only, no runtime Persona Chat gating, no live response mutation, no WebUI state, no DB persistence, no Jobs worker, no moderation gate, and no Persona Live/VN/native companion work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Valid JSON judge responses are converted into PersonaChatJudgePrediction objects for selected Persona Chat judge dimensions.
- [x] #2 Malformed JSON, missing keys, invalid result values, duplicate predictions, unknown dimensions, and provider-call failures fail closed with bounded error/status keys.
- [x] #3 Adapter output and errors do not leak raw prompts, assistant responses, exemplar text, secrets, local paths, or database content.
- [x] #4 Provider execution is explicit and dependency-injected or routed through an existing LLM call seam; tests use fakes and do not call external providers.
- [x] #5 Docs and Backlog record V1 boundaries, failure modes, residual risks, focused tests, Bandit, and diff hygiene.
- [x] #6 Predictions from the adapter can flow through the existing Persona Chat calibration helper without changing runtime Persona Chat behavior; the harness/report/policy path remains separate.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan file: Docs/superpowers/plans/2026-05-12-persona-chat-judge-adapter-boundary.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented offline Persona Chat judge execution boundary in tldw_Server_API/app/core/Evaluations/persona_chat_judge_execution.py with injected completion callable, strict JSON parsing, allowlisted evidence references, duplicate execution-key protection, provider/model metadata bounding, and trace-safe failure keys. Added focused tests in tldw_Server_API/tests/Evaluations/test_persona_chat_judge_execution.py and updated the two Persona Chat judge contract docs plus implementation plan.

Verification: python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_execution.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py -q passed with 47 passed, 5 warnings. Bandit on touched execution/test files wrote /tmp/bandit_persona_chat_judge_execution.json with zero results. git diff --check passed. Marker scan for unfinished-work labels had no matches.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the offline Persona Chat judge executable adapter boundary for #1591. The slice converts strict JSON responses from an explicit injected completion callable into sanitized PersonaChatJudgePrediction objects, records bounded fail-closed error keys, feeds existing calibration, and preserves V1 non-goals: no provider SDK calls, persistence, Jobs, API/WebUI wiring, or runtime Persona Chat gating.
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
