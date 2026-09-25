---
id: TASK-13366
title: Align PostgreSQL content startup validator with media RLS policy
status: Done
assignee: []
created_date: '2026-09-25 19:11'
updated_date: '2026-09-25 19:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fresh PostgreSQL content schema v26 creates and forces RLS with media_visibility_access, but validate_postgres_content_backend still requires removed media_scope_* policies and aborts full-app startup. Require the current media policy while retaining sync_log checks and add a regression for startup validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Startup validation accepts the current consolidated media_visibility_access policy on schema v26 and keeps sync_log policy checks
- [x] #2 Missing current media policy still fails closed with a clear error
- [x] #3 Full synthetic PostgreSQL server gets past the RLS startup gate
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Direct isolated PostgreSQL inspection after schema v26 migration: media RLS enabled and forced, only media_visibility_access exists; sync_log retains four scope policies. Runtime validator still checks retired media_scope_* names. Regression will assert the current exact policy contract.

Validator now requires schema v26 media_visibility_access and retains four sync_log policies. Focused tests: 14 passed. Full live PostgreSQL startup and direct forced RLS check passed. Bandit 0 findings; fatal Ruff clean. Skip: no production topology tested.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validator now requires schema v26 media_visibility_access and retains four sync_log policies. Focused tests: 14 passed. Full live PostgreSQL startup and direct forced RLS check passed. Bandit 0 findings; fatal Ruff clean. Skip: no production topology tested.
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
