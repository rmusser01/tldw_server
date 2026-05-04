---
id: TASK-35
title: Phase 2.2 collections reading router conditional cleanup L
status: Done
assignee: []
created_date: '2026-05-04 05:12'
updated_date: '2026-05-04 05:31'
labels:
  - phase-2
  - router-groups
  - refactor
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move only the covered collections and reading content router specs onto the shared lazy ImportedRouterSpec helper. Preserve public route metadata and route ordering for collections feeds, collections WebSub, WebSub callbacks, and reading.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collections feeds, collections WebSub, WebSub callback, and reading RouterSpec metadata is preserved.
- [x] #2 Router module import and router attribute lookup for moved collections/reading specs is lazy through RouterSpec resolution.
- [x] #3 Full router group contract and adjacent main/OpenAPI contract tests pass.
- [x] #4 Bandit touched-source scope and git diff --check pass.
- [x] #5 WebSub callback router skip diagnostics distinguish callback_router from the management router.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression test proving collections/reading router module import and attribute lookup are deferred until RouterSpec resolution.
2. Run the focused selection red on current code.
3. Replace only the collections feeds, collections WebSub, WebSub callback, and reading import blocks with ImportedRouterSpec plus append_imported_router_spec while preserving metadata and ordering.
4. Run focused and full router contract tests plus adjacent main/OpenAPI contract tests.
5. Run Bandit on touched router source and git diff --check.
6. Commit the narrow tranche and update the issue.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Red check: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "collections_reading_router_attr_lookup" -q` failed before implementation because collections feeds, collections WebSub, WebSub callback, and reading router attributes were resolved during spec construction.
- Green focused check: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "collections_reading_router_attr_lookup" -q` passed with `1 passed`.
- Green full/adjacent checks: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q` passed with `53 passed`; `python -m pytest tldw_Server_API/tests/Services/test_main_router_contract.py -q` passed with `6 passed`; `python -m pytest tldw_Server_API/tests/Services/test_openapi_contracts.py -q` passed with `69 passed`.
- Security and hygiene: `python -m bandit -r tldw_Server_API/app/api/v1/router_groups/content.py -f json -o /tmp/bandit_phase2_2_collections_reading_router_conditionals_l.json` reported `0 results` and `0 errors`; `git diff --check` passed.
- Documentation: no user-facing docs required for this internal router registration refactor.
- Known skips or blockers: none.
- Review follow-up: PR #1260 Qodo thread reports that the collections_websub management router and callback_router use the same log_name, making skip logs ambiguous when one fails to resolve. Reopened task to add diagnostic coverage and fix the callback spec.
- Review follow-up verification: focused red test failed on the old callback spec name, then passed after changing the callback ImportedRouterSpec to log_name=collections_websub_callback with skip_context=(callback_router). Verified focused router regression, full router group contracts, main router contract tests, OpenAPI contract tests, Bandit touched-source scope with 0 issues, and git diff --check. Ruff was also attempted on touched files; it reports existing non-blocking baseline style findings in this branch, so no broad unrelated Ruff cleanup was folded into this review fix.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the covered collections feeds, collections WebSub, WebSub callback, and reading registrations to the shared lazy ImportedRouterSpec helper while preserving route prefixes, tags, route keys, and ordering. Added regression coverage proving selected collections/reading router modules and router attributes are not resolved until the selected RouterSpec objects are used, then verified the full router group contract, adjacent main/OpenAPI contract tests, Bandit touched-source scope, and whitespace checks.

PR #1260 review follow-up: made the collections WebSub callback router diagnostics unambiguous by giving the callback ImportedRouterSpec a distinct internal name and callback skip context, with regression coverage that preserves public route metadata while asserting the diagnostic fields.
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
