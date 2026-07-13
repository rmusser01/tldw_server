---
id: TASK-12108
title: Revise and replan research discovery Phase 2 PDF Media handoff
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-12 18:21'
labels:
  - design
  - research
  - media
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Revise the research discovery chokepoint specification and recreate the lost Phase 2 implementation plan after review. Keep Media as the sole ingestion owner, make the first implementation slice PDF-only, remove duplicate eligibility signals and speculative HTML plumbing, and record latest-dev/worktree prerequisites.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The spec separates Phase 2A PDF handoff from later HTML full-text handoff.
- [x] #2 The spec and plan preserve /api/v1/media/add as the existing public synchronous handoff surface and add no Research-owned ingestion endpoint.
- [x] #3 The plan uses the existing ingest_eligible and recommended_candidate_id fields with corrected eligibility semantics instead of duplicate fields.
- [x] #4 The plan resolves the required media_type and normal URL/file validation ordering for discovery selection requests.
- [x] #5 The plan defines a minimal Research selection function and Media-owned ingestion glue without speculative classes or duplicate pipelines.
- [x] #6 The plan defines response mapping, duplicate behavior, security limits, focused tests, Bandit, and latest-dev worktree prerequisites.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-12-research-discovery-phase2a-pdf-media-handoff.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Revised the discovery chokepoint spec and recreated the Phase 2A implementation plan after the old uncommitted plan disappeared and TASK-12083 collided on latest dev. Phase 2A is now PDF-only through /api/v1/media/add; HTML is gated behind a later Phase 2B. The design uses one Research selection-resolution function, corrects existing ingest_eligible/recommended_candidate_id semantics, requires media_type=pdf, branches before normal URL/file validation, keeps Media processing controls, rejects competing sources/credentials, performs Media-owned pre-download duplicate checks, composes streamed byte/MIME limits, preserves the existing results envelope, and adds no handoff-specific idempotency store or unsafe hard parser timeout claim.

Plan review corrections: use the worktree skill with an ignored-directory fallback check; avoid broad git staging; distinguish pre-download identifier/URL dedupe from post-extraction content-hash dedupe; reject duplicate normalized URLs before building trusted metadata maps.

Verification: git diff --check passed; trailing-whitespace awk check passed; the plan has 30 balanced Markdown fence lines and five stages with explicit statuses; stale resolver-class, duplicate eligibility-field, and removed timeout references were absent. Bandit and pytest are not applicable because this task changed Markdown planning documents only. No implementation code was changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the research discovery design and produced a fresh Phase 2A PDF Media handoff plan. The corrected plan keeps /api/v1/media/add as the sole public handoff, limits Research to owner-scoped snapshot selection resolution, reuses existing eligibility fields and the Media PDF pipeline, defines duplicate/security/response behavior, and explicitly defers HTML until its discovery and bounded persistence prerequisites exist.
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
