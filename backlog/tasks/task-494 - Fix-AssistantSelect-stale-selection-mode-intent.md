---
id: TASK-494
title: Fix AssistantSelect stale selection mode intent
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 19:58'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address post-merge review finding where AssistantSelect.handleSelect can use stale closed-over selectionMode state and persist the wrong overlay/tracked intent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AssistantSelect selection uses a ref-backed current intent instead of closed-over React state after reset operations.
- [x] #2 Overlay snapshot persistence uses the same resolved intent as selected-assistant metadata.
- [x] #3 Regression coverage proves rerendered selectionModePreference does not leak stale overlay intent into tracked selections.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED regression failed before implementation with selectedAssistant.metadata.selectionMode=overlay after rerendering selectionModePreference to tracked. Replaced transient selectionMode React state with selectionModeIntentRef. handleSelect now reads one local nextMode and uses it for both selected-assistant metadata and overlay persistence.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed AssistantSelect stale selection intent handling by reading a ref-backed nextMode in handleSelect and using it consistently for metadata and overlay settings writes. Added regression coverage for rerendered selectionModePreference changes before selection.
<!-- SECTION:FINAL_SUMMARY:END -->

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
