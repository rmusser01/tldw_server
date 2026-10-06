---
id: TASK-13506
title: Show the actual failed-only processing set during resumed import correction
status: To Do
labels:
- media
- ux
- bug
dependencies:
- TASK-13503
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A real mixed import saved one URL and two Markdown files, excluded a duplicate URL, and failed a corrupt PDF. After reload, Resume import -> original-file reattach -> Correct settings -> Review visibly says 2 items and lists the failed PDF plus the already successful URL as ready. Source inspection shows validQueueItems correctly retains retryIds and excludes prior successes, but Configure and Review independently recompute eligibility from the entire queue. The confirmation is misleading even if execution remains properly scoped. See latest-dev live validation report and correction-review evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Configure and Review display the same actual target set used by failed-only processing, while keeping prior outcomes visible with accurate states.
- [ ] #2 Previously saved files do not appear Invalid merely because browser File handles are unavailable for a retry that does not target them.
- [ ] #3 Verify the displayed count and real submitted source set on a mixed import resumed after reload.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Confirmed presentation mismatch before submission. Correction: handleCorrectItems does set retryIds; validQueueItems correctly filters the execution scope. The initial investigation guessed a submission defect too early; this task records the narrower confirmed review/configuration inconsistency. Code: Common/QuickIngestWizardModal.tsx validQueueItems/handleCorrectItems; WizardConfigureStep and ReviewConfirmStep recompute general queue eligibility. ADR required: no. No source fix made during validation.
Real submission checked: review claimed 2 items but POST /media/ingest/jobs accepted exactly one job (id 10), source recoverable.pdf, batch 6cfe3c5e-ef47-4966-8e1a-a5c140d22edf. Prior 3 successes stayed saved and no URL job was enqueued. Execution scope passed; Configure/Review count and status presentation remain wrong.
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
