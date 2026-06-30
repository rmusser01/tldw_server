---
id: TASK-246
title: Implement VN script generation profile snapshot maps
status: Done
assignee: []
created_date: '2026-05-10 21:16'
updated_date: '2026-05-10 21:26'
labels:
  - vn
  - scripted-generation
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1535'
documentation:
  - Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md
  - Docs/superpowers/specs/2026-05-10-vn-scripted-model-generation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 1 from Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md: authored generation profile maps for VN scripts, published version snapshot maps, validator rules for generate.profile_key/output_schema/confirmation/cancel/generated-choice semantics, API/schema round-tripping, and focused tests. Scope is backend VN Scripts publish-time behavior only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Authored generation profile map source is persisted and exposed through VN script create/update/list/detail responses while preserving generation_profile_id default compatibility
- [x] #2 Published script versions store generation_profile_snapshots with default plus additional profile keys and no provider secrets in API responses
- [x] #3 Publish idempotency replays the originally stored response for the same request key and rejects reused keys only when the original request payload changes
- [x] #4 Validator rejects invalid/unknown profile_key values unsupported output_schema values invalid confirmation/cancel/generated-choice shapes and raw routing keys while preserving literal generation compatibility
- [x] #5 Focused VN_Scripts tests cover validator publish snapshot map idempotency and API round-trip behavior
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented authored VN script generation profile maps with default fallback compatibility, publish-time profile snapshot maps, stable idempotent replay for stored request payloads, validator profile_key/output_schema/control-flow/routing policy checks, and focused API/service/DB/validator tests.

Verification: source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_validator.py tldw_Server_API/tests/VN_Scripts/test_vn_script_publish_snapshots.py tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py tldw_Server_API/tests/VN_Scripts/test_vn_scripts_db.py -q -> 40 passed, 5 warnings. Bandit touched backend scope -> 0 findings, JSON at /tmp/bandit_vn_script_profile_maps.json. compileall and git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added backend-only publish-time generation profile snapshot maps for VN scripts. Script create/update/list/detail now expose authored profile maps while preserving generation_profile_id as default; publish stores default plus additional generation profile snapshot IDs and replays stored idempotency responses without depending on mutable current profile state; validator now checks profile_key, output_schema, generated choice/cancel/confirmation shapes, raw routing keys, and profile policy incompatibilities while keeping literal generation valid.
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
