---
id: TASK-96.6
title: Add Quick Ingest Auto Manual chunking controls
status: Done
assignee:
  - '@codex'
created_date: '2026-05-06 17:43'
updated_date: '2026-05-06 17:50'
labels:
  - frontend
  - chunking
  - quick-ingest
  - auto-chunking
dependencies:
  - TASK-96.5
documentation:
  - Docs/superpowers/specs/2026-05-06-auto-chunking-design.md
  - Docs/superpowers/plans/2026-05-06-auto-chunking-implementation-plan.md
parent_task_id: TASK-96
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Quick Ingest UI control slice from the approved Auto Chunking plan. The Chunking section should default to Auto when enabled, expose goal and AI-assist controls in Auto mode, reveal existing detailed/template controls only in Manual mode, and keep submission behavior aligned with the payload helper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chunking enabled shows Auto selected by default
- [x] #2 Auto mode exposes goal selection and AI-assist toggle
- [x] #3 Manual mode exposes existing detailed/template chunking controls
- [x] #4 Switching back to Auto hides Manual-only settings and does not submit stale Manual fields
- [x] #5 Focused UI tests cover Auto/Manual visibility and Auto submission payload
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Starting Task 5 UI controls after TASK-96.5 payload/state commit c4a0089bd. Following TDD: add focused QuickIngestWizardModal integration coverage first, then wire Auto/Manual controls into existing options panel patterns.

Implemented Quick Ingest Auto/Manual UI controls. IngestOptionsPanel and WizardConfigureStep now show Auto/Manual segmented controls when chunking is enabled, Auto goal and AI-assist controls in Auto mode, and Manual-only chunk method/size/overlap or template controls in Manual mode. The wizard submit payload now carries Auto fields through buildQuickIngestPayload.

Verification: bun run test -- src/services/__tests__/quick-ingest-batch.test.ts src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism passed with 83 tests. Filtered tsc diagnostics for touched files had no matches. git diff --check passed. Bandit not applicable for this TypeScript-only UI slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Quick Ingest Auto/Manual chunking controls and wired the wizard submit payload through the Auto fields. Auto mode exposes goal and AI-assist controls, Manual mode reveals detailed/template chunking controls, and tests cover visibility, state preservation, stale-field hiding, and submitted Auto payloads.
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
