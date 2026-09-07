---
id: TASK-13216
title: Expose Writing Predict and Fill prompts through Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-07 21:52'
updated_date: '2026-09-07 22:11'
labels: []
dependencies: []
documentation:
  - Docs/Design/writing-continuation-service-prompts.md
  - Docs/superpowers/plans/2026-09-07-writing-continuation-service-prompts.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the user-approved bounded continuation slice: expose Writing Playground non-chat Predict and Fill system prompts in the existing shared Service Prompts settings. Preserve defaults, explicit chat behavior, context/templates, stopping/provider options, insertion and undo semantics; bind requests to their starting account/server and discard stale output. No new settings system or continuation-engine redesign.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Predict and Fill have independently editable literal system parts in the existing backend registry and shared Settings, with packaged-default compatibility.
- [ ] #2 Both streaming and non-streaming non-chat continuation use one scope-bound prompt snapshot; chat mode keeps its existing explicit system precedence and does not load these prompts.
- [ ] #3 Cancellation, scope changes, session/scene changes and unmount cannot apply stale text, logprobs, history, errors or finalizers to a newer operation.
- [ ] #4 Default payloads, context/template and stop behavior, provider controls, manual-stop partial output and insertion/undo behavior remain compatible.
- [ ] #5 Focused regression tests, lint, Bandit and independent review pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record approved design and concrete implementation plan. 2. Test-first registry/defaults/settings support. 3. Test-first scoped Predict/Fill request integration and lifecycle guards. 4. Focused verification, independent review, tracking update and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Approved design and three-stage implementation plan saved. Isolated baseline: 39 tests passed across WritingPlayground.phase1-baseline.test.tsx and TldwChat.abort.test.ts (2026-09-07); initial invocation from repository root found no tests, corrected to shared UI working directory. Existing transport already supports scope and signal; no transport redesign planned. Product code remains unchanged; awaiting execution-mode selection. Temporary dependency symlinks point to image-prompt-service-prompt worktree and must be removed before commit. git diff --check clean.

Stage 1 registry/defaults/Settings implementation completed test-first: backend registry/API suites 101 passed; shared UI Service Prompt runtime/transport/Settings suites 225 passed. Ruff clean; production Bandit 0 findings; ESLint 0 errors with 10 pre-existing no-explicit-any warnings in tldw-server.ts. Predict and Fill are independent literal system parts with exact packaged defaults and catalog/detail 404 compatibility.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
