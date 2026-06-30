---
id: TASK-208
title: Implement ACP structured completion signal gate for issue 1479
status: Done
assignee: []
created_date: '2026-05-10 01:15'
updated_date: '2026-05-10 01:22'
labels:
  - ACP
  - orchestration
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1479'
  - 'https://github.com/rmusser01/tldw_server/issues/1471'
documentation:
  - Docs/Development/ACP_Production_Readiness.md
  - Docs/Product/ACP_Agent_Orchestration_PRD.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first #1479 slice: define and enforce a structured ACP orchestration completion signal so task runs cannot advance to review or complete solely because an ACP prompt returned. Record validation failures in run history, keep intentional manual/non-orchestration behavior compatible, and update issue #1479 with evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A structured ACP task completion contract is defined in code and documented through tests or inline schema behavior.
- [x] #2 Orchestration task state does not advance from inprogress to review or complete without a valid accepted completion signal.
- [x] #3 Missing, malformed, and rejected completion signals fail the run visibly and leave or move the task to triage rather than silently marking it done.
- [x] #4 Focused unit/integration tests cover accepted, missing, malformed, and rejected completion signal cases.
- [x] #5 GitHub issue #1479 is updated with implementation status and verification evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented tldw_Server_API/app/core/Agent_Orchestration/completion_signals.py to validate ACP task completion signals from direct taskCompletion/completionSignal fields or explicit <acp-task-completion>{...}</acp-task-completion> output markers.

Updated dispatch_run in tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py to inject the required completion marker instructions into task prompts, reject returned prompts without a valid accepted signal, fail the run with a visible validation error, and move invalid runs to triage instead of review/complete.

Valid completion now maps the completion summary to run.result_summary. Reviewer tasks move inprogress -> review after a valid signal. Non-review tasks move inprogress -> review -> complete after a valid signal to preserve the existing intended non-review flow while respecting the state machine.

Added focused tests for accepted reviewer flow, accepted non-review flow, missing signal, malformed marker JSON, and rejected completion signal in tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py.

Verification: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Orchestration -q passed with 145 tests. Bandit on touched backend files wrote /tmp/bandit_acp_completion_signal.json with 0 findings. git diff --check passed.

Posted GitHub issue #1479 progress comment: https://github.com/rmusser01/tldw_server/issues/1479#issuecomment-4414134268

Known remaining integration note: GitHub issue #1479 is not closed locally because these changes have not been committed, pushed, or merged yet.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the ACP structured completion signal gate for orchestration dispatch. The endpoint now requires a valid accepted completion signal before advancing tasks, stores accepted completion summaries in run history, fails missing/malformed/rejected signals visibly, and moves invalid runs to triage. Focused and full Agent Orchestration tests passed, Bandit reported zero findings, git diff --check passed, and GitHub issue #1479 was updated with verification evidence.
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
