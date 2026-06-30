---
id: TASK-257.4
title: Add Persona Chat judge trace-safe execution artifacts
status: Done
assignee: []
created_date: '2026-05-12 04:30'
updated_date: '2026-05-12 04:35'
labels:
  - persona-chat
  - evaluations
  - stage-2
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1598'
  - 'https://github.com/rmusser01/tldw_server/issues/1566'
documentation:
  - Docs/Reviews/PERSONA_CHAT_JUDGE_EVALUATION_CONTRACT_2026_05_11.md
parent_task_id: TASK-257
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1598 as the next optional Persona Chat judge Stage 2 slice. Add a bounded review artifact for offline executable judge results by combining PersonaChatJudgeExecutionResult data with existing calibration metrics, while preserving V1 boundaries: no raw prompt/response/exemplar/exception leakage, no provider-specific execution, no persistence service, no Jobs/API/WebUI state, and no runtime Persona Chat gating.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A helper converts offline Persona Chat judge execution results plus fixture inputs into a JSON-serializable trace-safe artifact.
- [x] #2 The artifact includes schema version, offline_only=true, runtime_gating_allowed=false, sanitized provider/model metadata, bounded prediction/failure counts, calibration metrics, missing/unknown prediction keys, and warnings without raw prompt or response content.
- [x] #3 Failure-only artifacts preserve bounded error keys and sanitized case/dimension/provider/model metadata only.
- [x] #4 Focused tests cover successful artifacts, failure-only artifacts, calibration warning serialization, and leak resistance.
- [x] #5 Focused pytest, Bandit on touched Python scope, and git diff hygiene are recorded before closeout.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write failing tests for execution artifact success, failure-only output, calibration warnings, and leakage resistance. 2. Implement the minimal trace-safe artifact dataclasses/helper near the execution boundary. 3. Export the helper, run focused tests, Bandit, and diff hygiene. 4. Update Backlog notes and linked GitHub issue with verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED/GREEN: Added failing artifact tests in test_persona_chat_judge_execution.py for successful execution artifact serialization, failure-only artifacts, calibration warning serialization, and raw-content leak resistance. Initial import check failed with ImportError for build_persona_chat_judge_execution_artifact, then passed after adding PersonaChatJudgeExecutionArtifact and the builder in persona_chat_judge_execution.py.

Implementation: build_persona_chat_judge_execution_artifact now combines PersonaChatJudgeExecutionResult with calibrate_persona_chat_judge_predictions, serializes a trace-safe artifact with schema_version, offline_only=true, runtime_gating_allowed=false, sanitized provider/model, bounded input case ids, represented dimension keys, prediction/failure counts, sanitized predictions/failures, and calibration metrics/missing/unknown/warnings. Contract docs now describe the execution artifact boundary and leak constraints.

Verification: baseline focused judge tests passed before implementation with 27 passed. New execution test file passed with 16 passed. Broader focused Persona Chat judge suite passed with 52 passed and 5 warnings. py_compile passed for persona_chat_judge_execution.py. Bandit on touched production/test Python paths wrote /tmp/bandit_persona_chat_judge_artifacts.json with results 0 and errors []. git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added trace-safe Persona Chat judge execution artifacts for issue #1598. The new artifact helper combines bounded offline execution results with existing calibration metrics while preserving offline-only/no-runtime-gating boundaries and avoiding raw prompt, response, exemplar, exception, path, or secret leakage. Tests and docs cover success, failure-only, calibration warning, and leak-resistance behavior.
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
