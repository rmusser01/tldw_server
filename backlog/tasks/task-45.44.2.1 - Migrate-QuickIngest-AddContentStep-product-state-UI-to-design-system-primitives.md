---
id: TASK-45.44.2.1
title: >-
  Migrate QuickIngest AddContentStep product-state UI to design-system
  primitives
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-16 04:48'
labels:
  - design-system
  - webui
  - product-state
  - quick-ingest
dependencies: []
references:
  - apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx
  - >-
    apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.2
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the QuickIngest AddContentStep product-state warnings and media-type badge from AntD Alert/Tag to shared design-system primitives while preserving large-file, FFmpeg-missing, and queued item behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AddContentStep no longer imports AntD Alert or Tag for product-state UI.
- [x] #2 Large-file and FFmpeg-missing notices render the shared design-system Alert primitive with equivalent warning copy and icon treatment.
- [x] #3 Queued item media type labels render the shared design-system Badge primitive while preserving warning styling when FFmpeg is missing.
- [x] #4 Focused tests assert representative AddContentStep notices and badges render design-system markers.
- [x] #5 The design-system product-state baseline no longer contains AddContentStep exceptions and verification results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented AddContentStep product-state migration: AntD Alert/Tag product-state UI now uses design-system Alert and Badge primitives; removed AddContentStep entries from the design-system product-state baseline.

Verification: RED focused Step 1 tests failed on missing data-ds Alert/Badge markers before implementation; GREEN QuickIngest integration passed 27/27; AddContentStep URL detection passed 2/2; product-state guard unit passed 52/52; verify:design-system-state exited 0; baseline JSON parsed with 0 AddContentStep entries; git diff --check exited 0.

Known verification note: full UI TypeScript check still fails on existing repo-wide type debt outside touched files, starting in audio/composer/flashcards tests. Bandit is not applicable to this TypeScript-only frontend slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated QuickIngest AddContentStep warning notices and media labels to design-system Alert and Badge primitives, added regression coverage, and removed its three legacy product-state baseline entries.
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
