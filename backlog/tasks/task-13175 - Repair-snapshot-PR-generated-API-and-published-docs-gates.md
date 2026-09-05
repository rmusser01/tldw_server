---
id: TASK-13175
title: Repair snapshot PR generated API and published docs gates
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 15:39'
updated_date: '2026-09-05 15:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the generated contract and published-documentation omissions reported by PR 2883 CI, without weakening checks or enabling snapshot production support.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Canonical OpenAPI fingerprint and frontend types include the snapshot API and pass drift verification.
- [ ] #2 Published documentation includes ADR-043 with a valid source-design reference and affected docs tests pass.
- [ ] #3 Verified fixes are pushed to draft PR 2883 without enabling production support.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Reproduce existing CI drift/docs failures; regenerate canonical schema/fingerprint and frontend types; correct source-only design link and refresh curated published docs; run affected contract/docs checks, review generated diff, record evidence and push. ADR required: no new ADR. Existing Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md applies; generated artifact and link corrections preserve architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced CI OpenAPI drift locally with exact sha256 53fd934179cff5e5a773a82e7252d63fc608312d47660c81b670a3607d32fb28 (2073 paths/3140 schemas). Existing generator refreshed the fingerprint and ignored schema/types; fresh drift check and TypeScript declaration check pass. Reproduced missing published ADR via tracked-file parity test, corrected source design link to reviewed repository permalink and refreshed curated docs. Docs suite: 206 passed, one strict-build failure from macOS multiprocessing SemLock ENOSPC, reproducible outside sandbox despite 276GiB free; no docs/test configuration weakened. Isolated serial strict-build diagnostic pending, canonical CI verification still required. Bandit not applicable to this docs/JSON-only correction; no Python runtime files changed.

Serial strict MkDocs build passed in 29.40s using a temporary inherited config outside the repository with only plugin parallelism disabled and output redirected to a temporary site directory. Canonical production config remains unchanged. Public/private docs boundary check passed, published ADR matches source byte-for-byte, generated declaration compiles with tsc --noEmit, and diff whitespace checks passed. Original CI failures have direct red/green evidence; canonical strict CI result pending after push.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
