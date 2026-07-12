---
id: TASK-12110
title: Implement research discovery Phase 2A PDF Media handoff
status: In Progress
labels:
- research
- media
- ingestion
- security
documentation:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Phase 2A PDF-only handoff from persisted Research Discovery snapshots through the existing /api/v1/media/add endpoint. Keep Research resolver-only and Media responsible for validation, duplicate checks, egress/download limits, PDF processing, persistence, and outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Existing ingest_eligible and recommended_candidate_id semantics identify only stable Phase 2A PDF candidates.
- [ ] #2 Owner-scoped discovery selections resolve from server-owned snapshots without Research downloading, parsing, deduplicating, or persisting Media.
- [ ] #3 The existing /api/v1/media/add endpoint accepts discovery selections with media_type=pdf and no Research-owned ingestion endpoint is added.
- [ ] #4 Discovery mode rejects client URLs, files, cookies, duplicate normalized candidate URLs, malformed pairs, and more than five selections.
- [ ] #5 Media performs pre-download URL/identifier duplicate lookup and reuses existing race-safe URL/content persistence duplicate handling.
- [ ] #6 PDF egress, redirect, MIME, and streamed byte limits are enforced through existing Media download and processing paths.
- [ ] #7 Responses retain the existing results envelope with stable per-selection outcomes and input order.
- [ ] #8 Focused tests, compile checks, diff checks, and Bandit pass with no new findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-12-research-discovery-phase2a-pdf-media-handoff.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
