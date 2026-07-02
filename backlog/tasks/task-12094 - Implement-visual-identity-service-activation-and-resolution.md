---
id: TASK-12094
title: Implement visual identity service activation and resolution
status: Done
labels:
- visual-identities
- expression-packs
- service
priority: High
references:
- Docs/superpowers/specs/2026-07-01-visual-identity-expression-packs-design.md
- Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md
documentation:
- Implementation started with TDD for Stage 4 service activation/resolution. Focus
  remains limited to the core service
- repository atomic helper if needed
- exports
- and service tests.
modified_files:
- tldw_Server_API/app/core/Visual_Identities/service.py
- tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py
- tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py
- tldw_Server_API/app/core/Visual_Identities/__init__.py
- tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 4: add VisualIdentityService for pack creation, draft activation into immutable versions, optional actor binding, actor ownership validation, and deterministic expression resolution with fallback reasons. Keep API/frontend out of scope for this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- TDD RED captured before production implementation:
  - `test_visual_identity_service.py` failed collection with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Visual_Identities.service'`.
  - Repository helper coverage failed with `AttributeError: 'VisualIdentityRepository' object has no attribute 'activate_draft_as_version'`.
- Added `VisualIdentityService` for active pack shell creation, ready-draft activation, optional actor binding, ownership validation, and expression asset resolution.
- Added `VisualIdentityRepository.activate_draft_as_version(...)` to create/update the pack, create a new immutable version, copy draft assets into version assets, set the active version, mark the draft activated, and optionally upsert a binding in one transaction.
- Changed visual identity binding actor IDs to string-safe storage/signatures so persona UUID bindings work while integer character callers remain supported.
- Resolution now returns explicit dataclass results with fallback reasons for manual override, requested expression, mood, pack default, neutral alias, and placeholder.
- Verification recorded: focused service pytest, full Visual_Identities pytest, Bandit JSON report with no findings, and diff whitespace checks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4 service activation and resolution is implemented. Added service and repository coverage for character bindings, persona UUID bindings, manual override priority, immutable draft-to-version asset copying, deleted-pack placeholder resolution, and active pack creation without an active version until activation. Verification passed: focused service pytest, full Visual_Identities pytest, Bandit JSON report with no findings, and diff whitespace checks.
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
