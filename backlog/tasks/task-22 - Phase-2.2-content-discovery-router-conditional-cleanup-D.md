---
id: TASK-22
title: Phase 2.2 content discovery router conditional cleanup D
status: Done
assignee: []
created_date: '2026-05-04 00:05'
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
Move only the covered content discovery/search router specs onto the shared lazy ImportedRouterSpec helper. Preserve public route prefixes, tags, route keys, order, lazy resolution, and per-router skip behavior. Leave rag_unified unchanged because its current ImportError-only skip semantics differ from the broad optional branches.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected rag_health, research, research_runs, and paper_search RouterSpec metadata is preserved.
- [x] #2 Router attribute lookup for moved content discovery specs is lazy through RouterSpec resolution.
- [x] #3 Full router group contract and adjacent main/OpenAPI contract tests pass.
- [x] #4 Bandit touched-source scope and git diff --check pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression test proving selected content discovery router attribute lookup is deferred until RouterSpec resolution. 2. Run the focused selection red on current code. 3. Replace only the selected covered imports with ImportedRouterSpec plus append_imported_router_spec. 4. Run focused and full router contract tests plus adjacent main/OpenAPI contract tests. 5. Run Bandit on touched router source and git diff --check. 6. Commit the narrow tranche and update the task record.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Red check: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "content_router_specs_defers_discovery" -q` failed before implementation because selected content discovery router attributes were resolved during spec construction.
- Green focused check: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "content_router_specs_defers_discovery or content_router_specs_populates_expected or canonical_rag_key" -q` passed with `3 passed`.
- Review red check: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "append_imported_router_spec" -q` failed on the current branch because imported specs still imported modules during helper append, missing optional routers were pruned before registration, unexpected import errors escaped during append, and lazy attribute misses logged twice.
- Review fix: `append_imported_router_spec` now appends metadata-only `RouterSpec` factories, router imports and attribute lookups happen during registration after `route_enabled`, skip logs are owned by `register_router_specs`, and `skip_context` is preserved through `RouterSpec`.
- Green full/adjacent checks: `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q` passed with `45 passed`; `python -m pytest tldw_Server_API/tests/Services/test_main_router_contract.py -q` passed with `6 passed`; `python -m pytest tldw_Server_API/tests/Services/test_openapi_contracts.py -q` passed with `69 passed`.
- Security and hygiene: `python -m bandit -r tldw_Server_API/app/api/v1/router_groups/conditional.py tldw_Server_API/app/api/v1/router_groups/spec.py tldw_Server_API/app/api/v1/router_registry.py tldw_Server_API/app/api/v1/router_groups/admin.py tldw_Server_API/app/api/v1/router_groups/core.py tldw_Server_API/app/api/v1/router_groups/content.py -f json -o /tmp/bandit_phase2_2_content_router_conditionals_d_review.json` reported `0 results` and `0 errors`; `git diff --check` passed.
- Documentation: no user-facing docs required for this internal router registration refactor.
- Known skips or blockers: `rag_unified` intentionally remains unchanged because its current skip behavior catches only `ImportError`.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the covered `rag_health`, `research`, `research_runs`, and `paper_search` content discovery registrations to the shared lazy `ImportedRouterSpec` helper while preserving prefixes, tags, route keys, order, and per-router skip diagnostics. Review follow-up made the shared helper fully metadata-only so route policy can gate before imports, removed redundant broad wrappers around converted loops, centralized lazy-resolution skip logging in `register_router_specs`, and preserved `skip_context`. Added regression coverage for policy-gated imports, registration-time optional skips, unexpected import failures, and single skip logging, then verified the full router group contract, adjacent main/OpenAPI contract tests, Bandit touched-source scope, and whitespace checks.
<!-- SECTION:FINAL_SUMMARY:END -->
