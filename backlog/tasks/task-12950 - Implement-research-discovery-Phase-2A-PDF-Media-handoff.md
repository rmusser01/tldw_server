---
id: TASK-12950
title: Implement research discovery Phase 2A PDF Media handoff
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-12 20:20'
labels:
  - research
  - media
  - ingestion
  - security
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Phase 2A PDF-only handoff from persisted Research Discovery snapshots through the existing /api/v1/media/add endpoint. Keep Research resolver-only and Media responsible for validation, duplicate checks, egress/download limits, PDF processing, persistence, and outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing ingest_eligible and recommended_candidate_id semantics identify only stable Phase 2A PDF candidates.
- [x] #2 Owner-scoped discovery selections resolve from server-owned snapshots without Research downloading, parsing, deduplicating, or persisting Media.
- [x] #3 The existing /api/v1/media/add endpoint accepts discovery selections with media_type=pdf and no Research-owned ingestion endpoint is added.
- [x] #4 Discovery mode rejects client URLs, files, cookies, duplicate normalized candidate URLs, malformed pairs, and more than five selections.
- [x] #5 Media performs pre-download URL/identifier duplicate lookup and reuses existing race-safe URL/content persistence duplicate handling.
- [x] #6 PDF egress, redirect, MIME, and streamed byte limits are enforced through existing Media download and processing paths.
- [x] #7 Responses retain the existing results envelope with stable per-selection outcomes and input order.
- [ ] #8 Focused tests, compile checks, diff checks, and Bandit pass with no new findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-12-research-discovery-phase2a-pdf-media-handoff.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/research-discovery-phase2a-pdf
Branch: codex/research-discovery-phase2a-pdf, based on fetched origin/dev at 30db8bcfd7 plus planning commit 621c6a01db.

Stage 1 contract gate complete: /media/add dependencies remain intact; AddMediaForm.media_type remains required; normal persistence still requires URLs/files; download_url_async owns egress and streamed limits; Media DB exposes URL and safe-metadata lookup.

Baseline verification: 53 passed, 4 warnings in 11.92s for Research discovery identity/service, JSON URL download, and media form dependency tests. The first sandboxed attempt could not create a temp file; the exact suite passed outside the read-only sandbox.

Task-ID correction: the stale planning worktree assigned TASK-12110, which collided with an unrelated latest-dev task. The duplicate record was removed and this implementation is tracked by TASK-12950.

Stage 2 complete: stable-PDF eligibility plus owner-scoped snapshot resolution with five-item bounds, ordered output, fingerprint/candidate identity revalidation, sanitized identifiers/metadata, and no Media side effects. Verification: Black clean; 87 Research discovery tests passed with 5 warnings. Self-review added candidate type, canonical URL, source provenance, and safe metadata to the descriptor and corrected provider-ID-only fingerprint reconstruction.

Stages 3-4 complete. /media/add now accepts paired discovery references, rejects competing sources/cookies, resolves owner-scoped server snapshots, preflights URL and normalized identifier duplicates, blocks known restricted access, and calls existing Media persistence once for remaining PDFs with strict 50 MiB/application-pdf constraints and trusted metadata. Ordered outcomes and stable safe errors remain in the existing results envelope; no Research ingestion route was added. Regression verification: 157 focused Research/Media tests passed with 9 warnings; final integration additions pass 12 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation in progress.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
