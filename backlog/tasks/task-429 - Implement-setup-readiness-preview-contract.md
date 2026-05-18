---
id: TASK-429
title: Implement setup readiness preview contract
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-18 23:25
labels:
- implementation
- setup
- backend
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
- Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md
- tldw_Server_API/app/api/v1/schemas/setup_schemas.py
- tldw_Server_API/app/core/Setup/readiness_service.py
- tldw_Server_API/tests/Setup/test_setup_readiness_preview.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the second backend slice from Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md: read-only setup readiness preview behavior that returns normalized lane previews, config update previews, install-plan summaries, secret handling, overlays, and blockers without writing config or provisioning models.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preview builds normalized lane summaries, config update previews, install-plan summaries, overlays, and operation_required without writing config or starting provisioning.
- [x] #2 Hosted provider secrets are represented only as secret metadata and are never echoed in preview response fields.
- [x] #3 Trusted custom Hugging Face embedding models are blocked until explicitly acknowledged and enter the custom install plan only after acknowledgement.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added setup readiness preview request/response schema models for the future API endpoint. Added readiness_service.py as a read-only preview builder over lane selections, existing install plan schemas, and curated audio bundle expansion. Added focused preview tests for no-write behavior, restart overlay behavior, hosted secret redaction, trusted-model blocking, and acknowledged custom model install-plan routing. Updated the implementation plan Task 2 checklist.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the read-only setup readiness preview contract. Verification: initial TDD run failed at collection because readiness_service was missing; final /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_profiles.py tldw_Server_API/tests/Setup/test_setup_readiness_preview.py -q passed with 9 tests; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Setup/readiness_service.py tldw_Server_API/app/api/v1/schemas/setup_schemas.py -f json -o /tmp/bandit_first_time_readiness_preview.json completed with zero findings.
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
