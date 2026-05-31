---
id: TASK-478.2
title: 'Gate A: harden Research Workspace chat send and failed-response states'
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-25 07:49'
labels:
  - research-workspace
  - uat
  - gate-a
  - frontend
  - chat
  - rag
milestone: Research Workspace UAT Remediation
dependencies: []
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

<!-- SECTION:NOTES:BEGIN -->
- Historical UAT finding before this task: `Ollama / gemma3:1b` returned HTTP 200 with an empty stream and the UI displayed `No response text was returned`.
- Implemented missing-model preflight blocking, recoverable failed-submit draft restoration, and empty-stream conversion into a friendly chat error payload.
- Live CDP validation against backend `http://127.0.0.1:8000` and WebUI `http://localhost:3000/research-workspace`: missing-model send sent zero chat completion requests and preserved the draft; invalid provider returned 503 and rendered a recoverable error card with draft restored; intercepted empty stream rendered `No response was returned.` and restored the draft.
- OpenAPI warning audit: active client path uses configured-origin absolute `/openapi.json` through request normalization; focused request-core and connection-sync tests cover this path. No `/workspace-playground` alias or redirect added.
- Verification: focused Vitest suite passed, `git diff --check` passed, full UI typecheck remains blocked by pre-existing Watchlists JSX syntax errors in `WatchlistsPlaygroundPage.tsx`.
- Bandit: not applicable because TASK-478.2 touched frontend TypeScript/tests and Backlog metadata only; no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Research Workspace chat send and failure handling. The composer now blocks missing-model sends before API mutation, preserves draft text through recoverable failures, keeps source/RAG context intact in the tested paths, and turns empty model streams into explicit recoverable error cards instead of blank/null assistant responses. Added focused tests for missing model, selected-source preservation, failed submit recovery, optimistic pending behavior, empty stream failure conversion, and OpenAPI path normalization coverage.
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
