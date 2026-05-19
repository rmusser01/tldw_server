---
id: TASK-432
title: Verify setup readiness lanes
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-18 23:41'
labels:
  - implementation
  - setup
  - backend
  - api
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
  - >-
    Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the fifth backend slice from Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md: explicit verification helpers and /api/v1/setup/readiness/verify for chat, embeddings/RAG, and speech lanes without hidden hosted calls or expensive model loads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skipped lanes verify as skipped without contacting hosted providers.
- [x] #2 Speech verification delegates to install_manager.verify_audio_bundle_async and maps partial STT/TTS health to ready_with_warnings.
- [x] #3 Readiness verification endpoint accepts inline selections or stored previews and persists a verification snapshot.
- [x] #4 Verification avoids downloads, expensive embedding loads, and hidden hosted provider calls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 5 implementation from Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md. Verification: pytest setup readiness slice passed with 20 tests; Bandit JSON at /tmp/bandit_first_time_readiness_verify.json has zero findings.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added verify_readiness_lanes with cheap chat and embeddings checks plus delegated speech verification. Added /api/v1/setup/readiness/verify and persisted last_verification in the setup readiness store. Verification results are sanitized before API response and persistence.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Explicit setup readiness verification is implemented for first-run backend readiness lanes. Skipped lanes stay skipped, hosted chat verification does not perform hidden provider calls, embeddings verification remains a cheap readiness warning, and speech verification delegates to install_manager.verify_audio_bundle_async with partial STT/TTS health mapped to ready_with_warnings.
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
