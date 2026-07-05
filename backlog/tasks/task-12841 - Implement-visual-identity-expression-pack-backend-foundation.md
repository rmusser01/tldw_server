---
id: TASK-12841
title: Implement visual identity expression pack backend foundation
status: Done
references:
- Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md
- Docs/superpowers/specs/2026-07-01-visual-identity-expression-packs-design.md
modified_files:
- tldw_Server_API/app/core/Visual_Identities/__init__.py
- tldw_Server_API/app/core/Visual_Identities/expression_slots.py
- tldw_Server_API/app/core/Visual_Identities/constraints.py
- tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py
- tldw_Server_API/tests/Visual_Identities/test_expression_slots.py
- tldw_Server_API/tests/Visual_Identities/test_visual_identity_capabilities.py
- tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first reviewable backend slice from the Visual Identity Expression Packs plan: expression slot normalization, format capabilities, ChaChaNotes repository schema, and associated tests. This task starts execution of Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md stages 1 and 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented only stages 1 and 2 from `Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md`.

Verification:
- Red TDD run failed during collection because `tldw_Server_API.app.core.Visual_Identities` and `tldw_Server_API.app.core.DB_Management.VisualIdentity_DB` did not exist yet.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_expression_slots.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_capabilities.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py` -> 22 passed.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Visual_Identities tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py -f json -o /tmp/bandit_visual_identity_stage1_2.json` -> passed, zero findings.

Review fix verification:
- Red regression run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py` -> 5 failed, 13 passed before implementation. Failures covered idempotency completion hash conflicts, unsupported pack updates, pack-version ownership, orphan/mismatched asset references, and mismatched binding pack/version resolution.
- Green regression run: same DB command -> 18 passed.
- Full targeted command: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_expression_slots.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_capabilities.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py` -> 27 passed.
- Bandit: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Visual_Identities tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py -f json -o /tmp/bandit_visual_identity_stage1_2.json` -> passed, zero findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented stages 1 and 2 only: expression slot normalization/capability helpers plus a SQLite-only ChaChaNotes VisualIdentityRepository with schema creation, pack/draft/version/asset/binding/idempotency methods. Review fixes added pack/version/binding ownership validation, feasible references/checks, safer asset attachment validation, idempotency hash conflict protection, and strict pack update field handling. Targeted pytest passed: 27 passed. Bandit passed with zero findings for touched backend code.
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
