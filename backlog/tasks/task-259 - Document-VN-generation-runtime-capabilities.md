---
id: TASK-259
title: Document VN generation runtime capabilities
status: Done
assignee: []
created_date: '2026-05-11 05:26'
updated_date: '2026-05-11 05:35'
labels:
  - vn-play
  - api
  - scripted-generation
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md
  - Docs/superpowers/specs/2026-05-10-vn-scripted-model-generation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 9 from the VN scripted generation backend runtime plan: expose capability/setup metadata for scripted generation and update API documentation plus focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capabilities advertise scripted generation support, output schemas, confirmation support, revision activation, history/debug detail, and relevant limits
- [x] #2 Setup options and script-version metadata expose generation profile key, immutable snapshot ID, provider class, max automatic batch count, moderation requirement, estimated cost class, supported output schemas, dynamic choice support, scene update support, and confirmation requirements
- [x] #3 Readiness warnings identify missing, unavailable, or incompatible profile snapshots for scripts with generated output requirements
- [x] #4 VN API docs cover endpoint list, public/debug response boundary, idempotency requirements, and examples for confirmation/cancel/regenerate/activate
- [x] #5 Focused VN tests, compile check, Bandit, and git diff check are run and recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Task 9 capability/setup metadata and docs. VN capabilities now advertise scripted generation support, output schemas, confirmation, revision activation, history, debug detail, reveal support, and batch limits. Scripted-story setup options now include generation profile key/snapshot metadata, provider class, moderation/cost/batch metadata, supported output schemas, derived dynamic choice and scene update support, and confirmation requirements. Setup readiness now warns and blocks when required generation profile snapshots are unavailable or incompatible with generated output requirements.

Verification: .venv/bin/python -m pytest tldw_Server_API/tests/VN_Scripts tldw_Server_API/tests/VN_Play tldw_Server_API/tests/VN_Platform -q --tb=short -> 251 passed, 8 warnings.

Verification: compileall over VN Play, VN Scripts, DB management, VN Play/capabilities endpoints, and VN Play/capabilities schemas -> exit 0; git diff --check -> exit 0; Bandit VN scripted-generation backend scope -> results 0, errors [].
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Made the VN scripted generation runtime discoverable through capabilities, setup metadata, and API documentation. Added typed capability metadata, enriched setup script-version options with backend-owned generation profile/runtime requirements, added readiness warnings for missing generation profile snapshots, documented the public/debug API boundary and idempotent command endpoints, and verified the focused VN surface.
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
