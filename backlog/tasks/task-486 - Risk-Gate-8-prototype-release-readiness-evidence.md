---
id: TASK-486
title: Risk Gate 8 prototype release readiness evidence
status: Done
labels:
- prototype-workspaces
- risk-gate
- release-readiness
- testing
priority: high
references:
- https://github.com/rmusser01/tldw_server/issues/1461
- https://github.com/rmusser01/tldw_server/issues/1440
documentation:
- Docs/API-related/Prototype_Workspaces_API.md
- Docs/API-related/Prototype_Workspaces_Contract_Matrix.md
- Docs/Operations/Prototype_Workspaces_Runbook.md
- Docs/User_Guides/Prototype_Workspaces.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first Risk Gate 8 release-readiness slice under tracker #1440. Build evidence for the end-to-end prototype workspace path with focused backend/frontend verification matrices and CI-friendly smoke coverage that does not require external runtime services.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend and frontend verification matrices are recorded for Gate 8 release evidence.
- [x] #2 CI-friendly smoke coverage exercises the owner-to-collaborator-to-promotion path with runtime/preview stubs where practical.
- [x] #3 Negative security smoke coverage verifies expired or revoked prototype links fail without enumeration.
- [x] #4 Verification results, docs hygiene, and Bandit when backend production code changes occur are recorded.
- [x] #5 Remaining production-readiness risks are explicitly triaged for #1461.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Docs/Operations/Prototype_Workspaces_Release_Readiness.md` with backend/frontend evidence matrices, CI smoke path, negative security smoke path, and remaining Gate 8 risk triage.
- Cross-linked the release-readiness evidence from the prototype API docs and operator runbook.
- Added `test_release_readiness_smoke.py` using in-memory AuthNZ migrations, prototype/share repos, FastAPI `TestClient`, and stubbed runtime/preview validation.
- Smoke coverage now exercises owner workspace creation, private prototype share exchange, collaborator session creation, candidate persistence, promotion submission, failed owner approval, successful owner approval, and revoked/expired link failures.
- Verified backend/docs with `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_release_readiness_smoke.py tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_docs_contract.py -q`: `5 passed, 5 warnings`.
- Verified focused frontend prototype workspace coverage with local Vitest after `bun install --frozen-lockfile` hydrated ignored dependencies: `5 passed (5)`, `30 passed (30)`.
- Verified docs/whitespace with `git diff --check`: no output.
- Bandit skipped because this slice changes docs, backlog metadata, and tests only; no backend production Python files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Gate 8 release-readiness evidence for prototype workspaces, including backend/frontend verification matrices, cross-links from canonical docs, and CI-friendly smoke coverage for the owner-to-collaborator-to-promotion path. The smoke tests cover failed and successful owner promotion review plus revoked/expired prototype share links returning the same non-enumerating error contract.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched backend production code when applicable or skip documented
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
