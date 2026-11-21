---
id: TASK-15582
title: Let an admin explicitly disallow a provider or model, distinct from 'not configured'
status: To Do
labels:
- llm
- byok
- policy
- admin
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of TASK-15581, asked for by the owner on 2026-11-19. Today the provider override's allowed_models is an allowlist: a model left off it counts as explicitly disallowed (owner, 2026-11-19). The owner wants a distinct explicit "disallowed" state, so an admin can disallow a provider or model on purpose, as opposed to not having configured it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decides the shape of an explicit disallowed state for a provider and for a model, and how it relates to is_enabled and allowed_models
- [ ] #2 The owner decides whether a model left off allowed_models stays refused once the explicit state exists
- [ ] #3 The admin API and WebUI let an admin set the state the owner chose, enforcement follows it, with tests, and ADR-086 and the admin docs say so
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
