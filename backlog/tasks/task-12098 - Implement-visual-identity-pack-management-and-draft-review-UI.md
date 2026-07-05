---
id: TASK-12098
title: Implement visual identity pack management and draft review UI
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-02 13:57'
labels:
  - visual-identities
  - expression-packs
  - frontend
  - ui
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md
  - Docs/Design/Visual_Identity_Expression_Packs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 9: add reusable VisualIdentityPackPanel, draft review/grid/uploader components, integrate with character and persona visual workflows, and cover activation binding plus slot ordering with Vitest.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 9 UI components and integrations:
- Added VisualIdentityPackPanel, VisualIdentityDraftReview, ExpressionSlotGrid, and ExpressionAssetUploader.
- Integrated expression packs into character metadata and persona visual workflows without removing legacy mood images or Persona Visual Pack management.
- Added slot-map-aware backend activation so replacement/cleared draft slots activate the same assets shown in the review UI.
- Added UI and backend regressions for activation binding, slot ordering, packless drafts, slot-map replacement, slot clearing, and upload selection persistence.

Review follow-up:
- Boyle/Russell found slot-map activation/rendering mismatches; fixed backend activation, UI slot rendering, upload slot-map persistence, packless draft upload behavior, and import polling guard behavior.
- Russell final focused pass reported no remaining important issues. Boyle did not return after the final patch within the wait window; his earlier reported issues were addressed.

Known gap:
- Default-expression selection from valid draft assets is deferred. The current V1 API has no draft default-expression update or activation-time default override, and reviewers agreed there is no safe frontend-only implementation.
- ZIP import polling uses terminal draft polling because there is no current Visual Identity job status endpoint/helper.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Stage 9 expression-pack management UI and corrected draft activation semantics so the review UI and activated pack agree on selected, replaced, and cleared assets. Character edit metadata now surfaces expression packs with legacy mood images collapsed, and Persona Visual Pack management now includes the shared expression-pack panel while preserving existing persona visual workflows.

Verification recorded:
- bunx vitest run VisualIdentityPackPanel/DraftReview/ExpressionSlotGrid/Persona VisualPackEditor tests => 71 passed.
- bun run test:characters-harness => 104 passed.
- pytest visual identity service/db/api/archive import tests => 67 passed.
- Bandit on VisualIdentity_DB.py => 0 findings.
- git diff --check => passed.
- UI package tsc with 8GB heap still fails only on unrelated baseline files; no Common/VisualIdentity errors reported.
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
