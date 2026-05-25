---
id: TASK-478.2
title: 'Gate A: harden Research Workspace chat send and failed-response states'
status: To Do
labels:
- research-workspace
- uat
- gate-a
- frontend
- chat
- rag
priority: Critical
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failure: after selecting sources and sending a question, the UI created a user message plus an assistant message labeled `null` with no useful answer, then appeared to reset to General chat/RAG disabled.

User goal: ask a question only when the workspace has a valid model/request configuration, and receive a useful answer or a clear recoverable error.

Scope:
- Disable or block send when no valid provider/model is selected.
- Preserve selected sources/RAG mode through send attempts and failures.
- Replace `null` assistant rendering with a clear error state that does not masquerade as an answer.
- Audit the `/openapi.json` request warning path seen during UAT and remove or route it correctly if it is a WebUI-side accidental request.
- Add tests for missing-model, failed-request, and recovery behavior.

Acceptance criteria:
- Sending without a selected model is impossible or produces an inline actionable validation message before API mutation.
- Failed chat/RAG requests do not create blank assistant messages or reset the workspace mode unexpectedly.
- The user can fix the model/request problem and retry without losing selected sources or the drafted question.
- CDP/Playwright validation confirms the failure path is visible and recoverable.

Depends on: TASK-478.1.
Blocks: grounded RAG Q&A and Studio generation validation.
Parallelization: can be implemented immediately after model catalog behavior is fixed; backend status tasks can proceed independently.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
