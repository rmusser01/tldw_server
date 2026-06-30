---
id: TASK-2344
title: Implement MCP prompt catalog support
status: Done
labels:
- mcp
- prompts
- implementation
references:
- TASK-2342
- TASK-2343
- TASK-2341
documentation:
- Docs/superpowers/specs/2026-06-22-mcp-prompt-catalog-support-design.md
- Docs/superpowers/plans/2026-06-22-mcp-prompt-catalog-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_catalog.py
- tldw_Server_API/app/core/MCP_unified/modules/base.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_module.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py
- tldw_Server_API/Config_Files/mcp_modules.yaml
- tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py
- tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py
- tldw_Server_API/tests/MCP_unified/test_mcp_prompts_http.py
- Docs/MCP/mcp_prompts.md
- Docs/MCP/mcp_tool_catalogs.md
- Docs/MCP/README.md
- Docs/superpowers/plans/2026-06-22-mcp-prompt-catalog-implementation-plan.md
- tldw_Server_API/app/core/AuthNZ/rbac_seed.py
- tldw_Server_API/app/core/AuthNZ/migrations.py
- tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py
- tldw_Server_API/app/core/AuthNZ/initialize.py
- tldw_Server_API/app/core/MCP_unified/server.py
- tldw_Server_API/tests/AuthNZ/unit/test_rbac_seed_helper.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement MCP protocol-level prompt catalog support for regular user Prompt Library prompts and explicitly allowlisted config prompts. Follow the reviewed implementation plan: tests first, context-aware prompt hooks, namespaced prompt routing, cursor pagination, prompts.read permission behavior, HTTP cursor support, docs, focused pytest, and Bandit. Prompt Studio remains out of scope; broader shared prompt registry remains tracked by TASK-2341.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP initialize advertises prompts capability with `listChanged: false`.
- [x] #2 `prompts/list` returns fresh readable, non-deleted Prompt Library prompts and explicitly allowlisted config prompts, excluding Prompt Studio prompts.
- [x] #3 `prompts/get` renders namespaced `library:` and `config:` prompts with sanitized errors and argument validation.
- [x] #4 Namespaced prompt access uses `prompts.read` and does not require `modules.read`.
- [x] #5 Config prompt allowlist defaults empty in `tldw_Server_API/Config_Files/mcp_modules.yaml`.
- [x] #6 Docs and focused verification cover protocol, HTTP cursor forwarding, AuthNZ seeding, and warning sanitization.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-22-mcp-prompt-catalog-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented MCP prompt discovery/rendering through the existing Prompts MCP module rather than a new shared registry. Regular Prompt Library prompts are exposed as `library:<uuid>` after user/readability and soft-delete filtering; Prompt Studio prompts remain excluded. Explicit config prompt entries are exposed as `config:<id>` only when allowlisted under the prompts MCP module config.

Final review fixes:
- Provisioned `prompts.read` as a baseline MCP read permission in RBAC seed helpers, SQLite migration/backfill paths, Postgres ensure/bootstrap paths, and MCP startup backstop.
- Added regression coverage that `prompts/list` works with `prompts.read` but without `modules.read`.
- Filtered and sanitized prompt-list warning metadata so scoped callers cannot see denied prompt identifiers.

Final privacy review fixes:
- Suppressed identifier-bearing Prompt Library cursors from `prompts/list` responses when restrictive MCP scopes are active, preventing scoped callers from decoding denied prompt names or UUIDs.
- Stripped known prompt identifier warning keys (`_prompt_name`, `prompt_name`, `prompt_uuid`, `prompt_id`, and `id`) unconditionally after warning visibility resolution, including explicit-name and mismatched-source warnings.

Verification recorded:
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py::test_default_mcp_modules_config_declares_prompts_module_with_empty_config_allowlist -v` -> 55 passed.
- `python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_prompts_http.py tldw_Server_API/tests/AuthNZ/unit/test_rbac_seed_helper.py -v` -> 6 passed.
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py tldw_Server_API/app/core/MCP_unified/tests/test_registry_iteration_race.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py -v` -> 34 passed.
- `python -m py_compile` on touched production Python files -> exit 0.
- `python -m bandit -r <touched production Python files> -f json -o /tmp/bandit_mcp_prompt_catalog_final.json` -> exit 0, `results: []`, `errors: []`.
- Docs grep, ASCII check, and focused `git diff --check` passed.
- Red test run: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py::test_prompts_list_does_not_return_denied_identifier_cursor_for_scoped_callers tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py::test_visible_prompt_warning_strips_identifier_fields_after_explicit_name_resolution -v` -> 2 failed as expected.
- Green regression run: same focused command -> 2 passed.
- Green focused suite: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py -v` -> 54 passed.
- Final broader MCP prompt/config suite rerun after privacy fixes -> 55 passed.
- `python -m py_compile tldw_Server_API/app/core/MCP_unified/protocol.py` -> exit 0.
- `python -m bandit -r tldw_Server_API/app/core/MCP_unified/protocol.py -f json -o /tmp/bandit_mcp_prompt_catalog_privacy.json` -> exit 0, `results: 0`, `errors: []`.
- `git diff --check -- tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py` -> exit 0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented MCP prompt catalog support for readable, non-deleted user Prompt Library prompts and explicitly allowlisted config prompts. The MCP initialize response advertises prompts with listChanged false, prompts/list is fresh on each call, and prompts/get renders namespaced library:/config: prompts through the Prompts MCP module. Config prompts default to an empty allowlist in tldw_Server_API/Config_Files/mcp_modules.yaml. AuthNZ now provisions prompts.read across SQLite, Postgres, baseline seeding, initialization, and MCP startup; namespaced prompt access does not require modules.read. Final review fixes filtered/sanitized prompt-list warning metadata and suppress identifier-bearing Prompt Library cursors for restrictive scoped callers so denied prompt names/UUIDs are not returned. Verification: prompt catalog/protocol/config pytest 55 passed; MCP HTTP/AuthNZ pytest 6 passed; broader MCP regression pytest 34 passed; py_compile exited 0; Bandit JSON at /tmp/bandit_mcp_prompt_catalog_continue.json has results: [] and errors: []; docs grep, ASCII check, and git diff --check passed.
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
