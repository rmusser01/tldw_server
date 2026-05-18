---
id: TASK-428
title: Implement setup readiness profile contract
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-18 23:20'
labels:
  - implementation
  - setup
  - backend
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
  - >-
    Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first backend slice from Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md: canonical setup readiness lane/status/overlay models plus a pure readiness profile builder for chat, embeddings/RAG, and speech readiness. This slice must not add provisioning or config mutation behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Canonical readiness lane/status/overlay constants are defined for chat, embeddings/RAG, and speech setup readiness.
- [x] #2 Pure readiness profile builder returns stable lanes, curated profile IDs, setup access metadata, overlays, and conservative recommendations without provisioning or config mutation.
- [x] #3 Focused tests cover lane semantics, curated profiles, speech TTS secondary metadata, and post-setup admin-required overlay behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added readiness_models.py for canonical lane IDs, supported statuses, overlays, labels, and lane-summary normalization. Added readiness_profiles.py as a pure builder over setup status, config snapshot fields, and audio recommendations. Added focused setup readiness profile tests before implementation, then kept the slice backend-only with no endpoint, schema, WebUI, provisioning, or config mutation changes. Updated the implementation plan Task 1 checklist and file list to match the actual scoped implementation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the backend setup readiness profile contract slice. Verification: initial TDD run failed with the expected missing module error; final `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_profiles.py -q` passed with 4 tests; `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Setup/readiness_models.py tldw_Server_API/app/core/Setup/readiness_profiles.py -f json -o /tmp/bandit_first_time_readiness_profiles.json` completed with zero findings.
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
