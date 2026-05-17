---
id: TASK-407
title: Add extension playlist quick ingest handoff
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 01:38'
labels:
  - bulk-conference-ingest
  - extension
  - quick-ingest
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 7 from the bulk conference ingest plan: typed Quick Ingest open detail, sidepanel active-tab playlist quick action, and shared preflight seed for extension playlist capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 requestQuickIngestOpen accepts a typed extension_active_tab playlist_preflight detail.
- [x] #2 Sidepanel passes active-tab YouTube playlist context into the shared Quick Ingest open request.
- [x] #3 Quick Ingest consumes the open detail and seeds the playlist preflight path.
- [x] #4 Focused Sidepanel/Quick Ingest tests cover the handoff contract.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented typed Quick Ingest open detail and active-tab YouTube playlist detection in the sidepanel quick action. The shared Quick Ingest event/session path now persists the detail, hydrates it into wizard state, and seeds the existing playlist preflight path. background.ts was reviewed but left unchanged because active-tab resolution happens in ControlRow at click time and existing background runtime metadata handling already covers queued batch processing. Verification: focused quick-ingest-open, sidepanel form contract, QuickIngestWizardModal integration, and sidepanel route registry Vitest suites pass; git diff --check passes; TypeScript still fails only on unrelated baseline files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 7 complete. Extension sidepanel users on a YouTube playlist/watch-with-list tab now see a playlist-aware Quick Ingest action when playlist preflight is supported, and the shared modal starts the existing playlist preflight flow from that tab context. Focused Vitest suites and route registry tests pass; TypeScript still fails only on unrelated baseline files; git diff check passes.
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
