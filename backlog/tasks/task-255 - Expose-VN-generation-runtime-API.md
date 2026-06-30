---
id: TASK-255
title: Expose VN generation runtime API
status: Done
assignee: []
created_date: '2026-05-11 05:00'
updated_date: '2026-05-11 05:25'
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
Expose backend-owned scripted VN generation confirmation, cancellation, regeneration, activation, history, and debug-read surfaces through stable VN Play API endpoints. This implements Task 8 from Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md and must keep raw prompts/model output out of public list and state responses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Request and response schemas cover generation confirm cancel regenerate activate history list revision list/detail and debug detail without exposing raw diagnostics in public responses
- [x] #2 Public script state schemas include waiting generation confirmation and active generated-choice metadata
- [x] #3 Generation lifecycle event literals are accepted by API response validation
- [x] #4 Confirm cancel regenerate and activate endpoints enforce idempotency and stale scene version handling
- [x] #5 Generation history endpoint returns owner-safe offset pagination with canonical pagination metadata and legacy aliases
- [x] #6 Revision debug endpoint verifies session/generation/revision ownership and restricts diagnostics to owner/admin access
- [x] #7 Moderation-blocked raw output is redacted by default and revealed only with explicit confirmation parameters
- [x] #8 Focused VN API tests cover owner access idempotency stale versions pagination debug authorization redaction and public-response redaction
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Task 8 API slice for scripted VN generation runtime: confirm/cancel/regenerate/activate command endpoints, generation history and revision list/detail endpoints, owner/admin debug detail, public state metadata, and raw-output redaction rules.

Verification: .venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py tldw_Server_API/tests/VN_Play/test_vn_play_scripted_generation_runtime.py -q --tb=short -> 65 passed, 8 warnings.

Verification: compileall over touched VN Play endpoint/schema/service/repository files -> exit 0; git diff --check -> exit 0; Bandit touched backend scope -> results 0, errors [].
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exposed the backend-owned scripted VN generation runtime API. Added idempotent confirm/cancel/regenerate/activate command endpoints, owner-safe generation history and revision list/detail surfaces, owner/admin debug diagnostics with moderation-blocked raw-output reveal controls, public script-state generation metadata, and focused API/runtime regression coverage including recovery for completed inner generation actions.
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
