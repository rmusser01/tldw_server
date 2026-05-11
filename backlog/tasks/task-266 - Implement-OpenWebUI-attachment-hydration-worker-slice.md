---
id: TASK-266
title: Implement OpenWebUI attachment hydration worker slice
status: Done
assignee: []
created_date: '2026-05-11 16:50'
updated_date: '2026-05-11 16:58'
labels:
  - chatbooks
  - openwebui
  - implementation
dependencies:
  - TASK-265
references:
  - >-
    Docs/superpowers/plans/2026-05-11-openwebui-attachment-hydration-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 6 of the OpenWebUI attachment hydration implementation plan: route openwebui_attachment_hydration Jobs through the Chatbooks worker, revalidate payload/root requirements, call the hydration service, and return a JSON-safe summary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Worker dispatches job_type=openwebui_attachment_hydration without changing import/export routing
- [x] #2 Missing or invalid required payload fields fail non-retryably with descriptive errors
- [x] #3 Worker revalidates data root and user scope before hydration execution
- [x] #4 Worker returns a JSON-safe hydration summary with warnings capped/redacted
- [x] #5 Focused worker pytest, import/export regression checks, diff checks, and Bandit are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 6 worker slice: added worker routing for openwebui_attachment_hydration without requiring legacy chatbooks_job_id, added non-retryable payload/root validation, called ChatbookService.run_openwebui_attachment_hydration, returned capped/redacted JSON-safe Jobs summaries, and preserved existing import/export routing. Added service execution support that revalidates the OpenWebUI data root with uploads required, hydrates image references, registers non-image attachments through Media DB when available, and records summary counts. Verification: initial red worker run failed as expected on missing hydration routing; worker/import tests 11 passed; worker/import/adapter tests 13 passed; broader worker/API/service tests 36 passed; full OpenWebUI hydration/import + worker slice 61 passed; git diff --check clean; Bandit report /tmp/bandit_openwebui_hydration_worker.json has 0 findings and 0 errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 6 routes openwebui_attachment_hydration Jobs through the Chatbooks worker, revalidates payload/root requirements in the worker/service layer, runs actual hydration execution, and stores a capped/redacted summary result. Existing import/export worker paths remain covered by regression tests. Verification recorded: 61 focused OpenWebUI hydration/import and Chatbooks worker tests passed, diff check clean, Bandit 0 findings.
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
