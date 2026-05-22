---
id: TASK-481
title: Risk Gate 7 prototype operational visibility and documentation
status: Done
labels:
- prototype-workspaces
- risk-gate
- operations
- documentation
priority: high
references:
- https://github.com/rmusser01/tldw_server/issues/1460
- https://github.com/rmusser01/tldw_server/issues/1440
- https://github.com/rmusser01/tldw_server/pull/1949
documentation:
- Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md
- Docs/API-related/Prototype_Workspaces_Contract_Matrix.md
- Docs/superpowers/plans/2026-05-22-prototype-risk-gate-7-ops-docs-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1460 under tracker #1440. Burn down operational support risk for prototype workspace collaboration by documenting runtime bootstrap, preview health, signing secrets, quotas, job behavior, owner/collaborator workflows, failure examples, and available status/audit fields. Keep implementation scoped to documentation and light status-surface wiring where existing code already exposes the data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Operator docs explain setup, configuration, and failure diagnosis for prototype workspace collaboration.
- [x] #2 Product docs explain owner and collaborator lifecycle without relying on implementation internals.
- [x] #3 Runtime, preview, sharing, promotion, and support-observability fields/events are documented and covered by tests where practical.
- [x] #4 Split Backend/Core and Frontend/Product deliverables are recorded clearly for Risk Gate 8 handoff.
- [x] #5 Focused verification, docs hygiene, and Bandit if backend code changes occur are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Plan: `Docs/superpowers/plans/2026-05-22-prototype-risk-gate-7-ops-docs-plan.md`
- Added `Docs/Operations/Prototype_Workspaces_Runbook.md` for operator setup, signing-secret posture, Jobs behavior, status-field diagnosis, preview/promotion triage, quotas, incident handling, and Gate 8 handoff.
- Added `Docs/User_Guides/Prototype_Workspaces.md` for owner and collaborator lifecycle examples, including password-protected links, single-use/exhausted links, resume cookies, revoked links, archived workspaces, promotion conflicts, and validation failures.
- Cross-linked Gate 7 artifacts from `Docs/API-related/Prototype_Workspaces_API.md` and recorded operational support fields in `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`.
- Added `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_docs_contract.py` as a focused docs-contract guard.
- Verification: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_docs_contract.py -q` passed with 3 tests.
- Verification: `git diff --check` passed.
- Bandit: skipped because this slice changed docs plus a docs-contract test only; no production Python code changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Risk Gate 7 operational visibility documentation is now covered by an operator runbook, a user lifecycle guide, API/contract cross-links, and a focused pytest guard that keeps the required support fields and failure examples present for Gate 8 release review.
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
