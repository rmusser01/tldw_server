---
id: TASK-401
title: Add Persona Visual starter production-readiness metadata
status: Done
assignee: []
created_date: '2026-05-16 02:34'
updated_date: '2026-05-16 04:36'
labels:
  - persona
  - visual-packs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1741'
documentation:
  - Docs/Code_Documentation/Persona_Visual_Packs.md
  - >-
    Docs/superpowers/specs/2026-05-14-persona-buddy-default-catalog-state-catalog-extension-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose bounded production-readiness metadata for the nine Persona Visual starter scaffolds so clients and future asset-generation workers can distinguish scaffold fixtures from final authored default buddy assets. Keep copy-to-draft and explicit activation semantics unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Starter catalog list/detail responses expose production/readiness metadata for each starter
- [x] #2 The nine starters report scaffold status and tier-specific neutral-anchor/animation requirements
- [x] #3 Focused backend/API tests cover the new fields and copy behavior remains unchanged
- [x] #4 Persona Visual docs explain production metadata and the neutral-pose-to-animation pipeline boundary
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented production-readiness metadata for Persona Visual starter catalog summaries/details: complexity_tier, production_status, neutral_anchor_required, expected_asset_groups, and animation_coverage_notes. Added fixture validation for tier/status/neutral-anchor consistency while preserving existing copy-to-draft behavior. PR review follow-up: added canonical metadata validation for complexity_tier/production_status, explicit immutable tuple validation for expected_asset_groups/animation_coverage_notes, strict boolean validation for neutral_anchor_required, direct immutable tuple dataclass defaults, and regression coverage for malformed production metadata. Verification: focused pytest 38 passed; py_compile exit 0; git diff --check exit 0; Bandit JSON results empty at /tmp/bandit_persona_visual_starter_production_1741_review2.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added production-readiness metadata to bundled Persona Visual starter catalog responses and docs so the nine starter entries remain clearly labeled as scaffolds pending neutral-anchor-driven authored assets. Kept copy-to-draft/no-auto-activation behavior unchanged and covered the new contract with focused service/API tests.
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
