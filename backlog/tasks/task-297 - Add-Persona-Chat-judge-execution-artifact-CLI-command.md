---
id: TASK-297
title: Add Persona Chat judge execution artifact CLI command
status: Done
assignee: []
created_date: '2026-05-12 05:15'
updated_date: '2026-05-12 05:21'
labels:
  - persona-chat
  - evaluations
  - stage-2
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1601'
  - 'https://github.com/rmusser01/tldw_server/issues/1566'
documentation:
  - Docs/Reviews/PERSONA_CHAT_JUDGE_EVALUATION_CONTRACT_2026_05_11.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1601: expose the trace-safe Persona Chat judge execution artifact helper through the existing offline persona-chat-judge CLI. Keep the command offline-only, review-oriented, and free of provider execution, DB/Jobs/API/WebUI persistence, or runtime Persona Chat gating.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The persona-chat-judge CLI emits a trace-safe execution artifact from fixture inputs plus a bounded execution-result JSON file.
- [x] #2 The command supports optional output-file writing with stdout/file JSON parity.
- [x] #3 Malformed execution-result roots/rows fail with bounded ClickException messages and no raw trace leakage.
- [x] #4 Focused tests cover success, output-file parity, invalid execution input, and raw-content leak resistance.
- [x] #5 Focused pytest, Bandit on touched Python scope, and git diff hygiene are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the offline persona-chat-judge artifact command for issue #1601. The command loads redaction-safe Persona Chat quality inputs, rebuilds bounded PersonaChatJudgeExecutionResult data from JSON, reuses build_persona_chat_judge_execution_artifact(), and emits stdout/output-file artifact JSON without provider execution or persistence.

TDD: new CLI artifact tests first failed with no such command artifact, then passed after implementation. Validation: focused Persona Chat judge suite passed with 58 tests; py_compile for persona_chat_judge_cli.py passed; Bandit on touched Python scope wrote /tmp/bandit_persona_chat_judge_artifact_cli.json with 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added an offline persona-chat-judge artifact CLI command that converts redaction-safe Persona Chat quality inputs plus a bounded execution-result JSON file into the trace-safe execution artifact introduced by PR #1600. The command supports optional output-file writing, bounded malformed-input errors, and preserves the V1 no-provider/no-persistence/no-runtime-gating boundary. Tests and docs were updated, with focused pytest, py_compile, Bandit, and diff hygiene passing.
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
