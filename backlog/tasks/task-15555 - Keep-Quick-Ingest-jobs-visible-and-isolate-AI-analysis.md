---
id: TASK-15555
title: Keep Quick Ingest jobs visible and isolate AI analysis
status: In Progress
created_date: 2026-11-17 03:45
labels:
- quick-ingest
- media-ingestion
documentation:
- Docs/ADR/003-jobs-vs-scheduler-default.md
updated_date: 2026-11-17 03:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Public dev drops the watcher for accepted queued jobs after five minutes, and its direct executor waits for one file to finish before submitting the next. With one default media-ingest worker, one analysis blocks subsequent jobs. Reuse the existing Jobs worker and session tracking architecture. ADR-003 governs Jobs ownership. This change does not need a new ADR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keep accepted pending Quick Ingest jobs visible beyond five minutes through the live and restored browser flow.
- [x] #2 Submit subsequent jobs after acknowledgement and independently serve ordinary ingest and a second analysis while one analysis is active.
- [ ] #3 Record the hosted staging evidence and the public branch verification in a PR against dev.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Port the behavior against current dev without depending on unrelated open PRs. Verify queue routing, worker concurrency and the active UI transports before opening the PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Public dev at 61ad5280ca shared the five-minute job observation timeout, direct same-wizard wait and single default-worker bottleneck. Native WebUI and extension polling, acknowledged per-row cancellation, scope changes and restore behavior were verified in fixture-backed tests. The bundled worker configuration serves ordinary jobs independently of a two-worker heavy pool. Existing external worker deployments can opt in with --heavy. Public verification: 611 frontend tests, 196 backend tests, 131 startup and CI ratchet checks after final worker-policy refactor, matching OpenAPI fingerprint and path union, and no new Bandit runtime findings. Hosted staging confirmed real PDF and small document analysis overlap, an ordinary document completing during the same PDF and repeated status polling after a 305-second Chrome clock advance. ADR assessment: existing Jobs ownership under ADR-003, no new ADR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Accepted jobs keep tracking beyond five minutes and additional files can submit while earlier jobs remain active. Ordinary and heavy ingest capacity are independent in the standard public configuration. Public staging evidence is from the hosted implementation, with fixture-backed public verification.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
