---
id: TASK-471
title: Draft Personalization Memory Layer PRD
status: Done
labels:
- persona
- personalization
- memory
- prd
- docs
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/issues/1918
- https://github.com/rmusser01/tldw_server/issues/1902
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Draft a repo-grounded PRD for the Personalization Memory Layer covering cross-app personalization memory, semantic memory tuning, automatic memory create/merge/prune, curation/review, opt-in boundaries, privacy, provenance, and Persona integration boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PRD is grounded in current Persona memory/state and personalization API/storage contracts.
- [x] #2 Scope, non-goals, opt-in boundaries, review/curation model, risks, staged implementation, and validation plan are documented.
- [x] #3 Issue #1918 and tracker #1902 are referenced.
- [x] #4 Docs-only verification is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current Persona memory/state docs, personalization design, APIs, schemas, and storage surfaces. 2. Draft the PRD with scope, non-goals, ownership boundaries, memory lifecycle, review/curation, staged delivery, risks, and validation. 3. Run docs-only verification and update the task status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created `Docs/Product/Personalization_Memory_Layer_PRD.md`. Grounded the plan in the current personalization design, PersonalizationDB storage, personalization endpoints/schemas, Persona memory integration, ChaCha Persona memory entries, companion activity/context/reflection services, and existing Persona future-PRD boundaries.

The PRD keeps `Personalization.db` as the canonical future memory-layer direction, treats ChaCha Persona memory as scoped source/consumer data, preserves explicit Persona `read_only` / `read_write` boundaries, requires review-first handling for inferred sensitive memories, and records docs-only verification. Bandit is not applicable because no executable code changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Drafted the Personalization Memory Layer PRD for issue #1918 and tracker #1902. The PRD defines taxonomy, lifecycle, opt-in/consent, review and curation, Persona integration, RAG precedence, Jobs-backed extraction/merge/prune, API/data model direction, privacy/safety requirements, staged delivery, validation, risks, and acceptance criteria. Docs-only change; Bandit not applicable because no executable code changed.
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
