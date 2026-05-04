---
id: TASK-40
title: Phase 2.2 core LLM router conditional cleanup O
status: Done
assignee: []
created_date: '2026-05-04 06:25'
updated_date: '2026-05-04 06:28'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
documentation:
  - >-
    Docs/superpowers/plans/2026-05-03-phase2-followup-stack-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.2 by deferring core LLM/provider-tail router imports from iter_core_router_specs while preserving existing route metadata and optional-import behavior. Scope is limited to llm_providers, mlx, messages, llamacpp, vlm, and mcp_unified_endpoint in router_groups/core.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 llm_providers, mlx, messages, llamacpp, vlm, and mcp_unified_endpoint core router specs defer router attribute lookup until registration/resolution.
- [x] #2 Existing prefix, tags, route_key, public_router attr_name, and default_stable behavior for the scoped core routers remain unchanged.
- [x] #3 Focused red/green router laziness coverage, full router contract tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red: focused llm_provider_router_attr_lookup test failed before implementation because scoped core LLM/provider router attrs were touched during iter_core_router_specs(). Green: focused test passed after converting scoped routers to lazy ImportedRouterSpec entries. Full verification passed: router_groups_contract 55 passed; main_router_contract 6 passed; openapi_contracts 69 passed; Bandit core.py 0 results/0 errors; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the core LLM/provider-tail router registrations for llm_providers, mlx, messages, llamacpp, vlm, and mcp_unified_endpoint to lazy ImportedRouterSpec entries. Added contract coverage proving iter_core_router_specs no longer resolves those router attributes during spec construction while preserving prefixes, tags, route keys, and public_router attr bindings.
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
