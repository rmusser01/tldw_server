---
id: TASK-2275
title: Add Mermaid artifact panel rendering coverage
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 05:37'
labels:
  - webui
  - chat
  - mermaid
  - tests
dependencies: []
references:
  - Docs/superpowers/specs/2026-06-06-chat-mermaid-card-artifact-rail-design.md
  - 'https://github.com/rmusser01/tldw_server/pull/2276'
modified_files:
  - apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cover the remaining Mermaid chat card follow-up from the design spec: verify ArtifactsPanel renders diagram artifacts and jump-to-source targets Mermaid origins correctly without changing runtime behavior unless tests expose a defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ArtifactsPanel has rendered coverage for diagram artifacts using the shared Mermaid renderer.
- [x] #2 ArtifactsPanel jump-to-source coverage verifies Mermaid artifact origins are targeted before fallback scrolling.
- [x] #3 Related Mermaid artifact tests and UI TypeScript verification pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a focused jsdom component test for ArtifactsPanel diagram artifacts. The test opens a kind: diagram artifact through the real artifact store, verifies the shared Mermaid renderer receives the Mermaid source, and verifies Jump to source scrolls the matching artifact-origin element, closes the panel, and does not dispatch the fallback latest-message event.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added ArtifactsPanel Mermaid coverage for the remaining chat-card design-spec gap. Verification: new ArtifactsPanel Mermaid test passed (2 tests); focused Mermaid/artifact panel regression set passed (4 files, 27 tests); UI TypeScript check passed; git diff --check passed. Bandit is not applicable because only frontend test and Backlog task files were touched.
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
