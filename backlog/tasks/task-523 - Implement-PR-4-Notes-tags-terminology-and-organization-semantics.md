---
id: TASK-523
title: Implement PR 4 Notes tags terminology and organization semantics
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 20:20'
labels:
  - notes
  - ux
  - webui
  - extension
  - pr4
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the PR 4 /notes UX remediation slice from the approved plan: present one user-facing concept, Tags, across /notes and directly connected capture flows while preserving the existing keywords API/client/storage contract. Scope is limited to user-facing terminology, search/filter clarity, and tests proving payloads remain keywords. No database/API rename or migration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User-facing labels in /notes and directly connected capture flows say Tags.
- [x] #2 API payloads, TypeScript client fields, and backend schemas continue to use keywords.
- [x] #3 Tests assert user-facing tag labels while preserving keywords payload assertions.
- [x] #4 Filter/search UI makes clear whether a control filters by text, tag, folder, or captured state.
- [x] #5 No database/API rename or migration is introduced.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verification: RED focused Vitest failed on old Filter by keyword and Keywords printable HTML labels; GREEN focused Vitest passed 3 files / 34 tests. git diff --check passed. UI package typecheck with default heap OOMed; rerun with NODE_OPTIONS=--max-old-space-size=8192 completed far enough to report one unrelated baseline error in src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx:35, Type 'comfortable' is not assignable to GalleryCardDensity. Bandit skipped because this slice only touches frontend TypeScript/JSON tests and UI copy.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 4 is complete for /notes tags terminology: in-scope visible copy now uses Tags, printable note export shows Tags metadata, and tests preserve the internal keywords payload/export contract.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document frontend-only skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
