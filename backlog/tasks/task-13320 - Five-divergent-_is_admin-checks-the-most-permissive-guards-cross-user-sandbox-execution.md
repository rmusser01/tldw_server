---
id: TASK-13320
title: >-
  Five divergent _is_admin checks; the most permissive guards cross-user sandbox
  execution
status: Done
assignee: []
created_date: '2026-09-22 04:55'
updated_date: '2026-09-23 23:17'
labels:
  - security
  - mcp
  - bug
dependencies: []
references:
  - >-
    tldw_Server_API/app/core/MCP_unified/modules/implementations/sandbox_module.py:142
  - tldw_Server_API/app/core/MCP_unified/protocol_types.py
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`core/MCP_unified/` defines `_is_admin` five times, with different claim sets:

- `modules/implementations/media_module.py:2058` — roles only, requires `isinstance(roles, list)`, no strip
- `modules/implementations/notes_module.py:2200` — byte-identical to media's
- `modules/implementations/kanban_module.py:1585` — media's body plus a `context.is_admin` probe
- `modules/implementations/sandbox_module.py:210` — roles **or** permissions ∩ {`*`, `system.configure`}
- `modules/implementations/mcp_discovery_module.py:283` — roles (no strip) **or** a live SELECT against user_roles

plus the protocol's own `protocol_types.py:_metadata_has_admin_claims`, which accepts roles∋admin or permissions∋`*` — **but not `system.configure`**.

They disagree on three independent axes: whether a bare-string `roles` claim counts, whether permissions count at all, and whether `system.configure` counts.

**Two confirmed consequences:**

1. **The broadest check guards the most dangerous operation.** `sandbox_module.py:142`:
   ```python
   if not self._is_admin(context) and str(owner) != user_id:
       raise PermissionError("sandbox.run session not found")
   ```
   `system.configure` is a real seeded AuthNZ permission (`rbac_seed.py:36`) and is what MCP's own RBAC maps `Action.ADMIN` to. A principal holding it passes this check and **reaches another user's sandbox session** — while the protocol layer that admitted the request does not consider that permission admin at all. The most permissive answer sits on the cross-user boundary.

2. **A bare-string roles claim splits the module.** `server.py` assigns `metadata["roles"]` straight from the MCP JWT with no `list()` coercion, and the protocol accepts the string shape (tested at `test_protocol_scope_enforcement.py:189`). With that same context, sandbox and mcp_discovery grant admin while media, notes and kanban all fail `isinstance(roles, list)` and **deny**. Same principal, same request, admin for three modules and not for three others.

Also verified: the `getattr(context, "is_admin", False)` probe in `sandbox_module.py:212` and `kanban_module.py:1587` is **dead against the real type** — `protocol_types.py` contains zero occurrences of `is_admin`. It fires only for the `SimpleNamespace` fixtures used in tests, so those two copies are partly tested through a branch production cannot reach.

Whichever claim set is intended, four of the five are wrong, and nothing in the code says which.

Found by the comprehensive core-module review; the five definitions, the sandbox call site and the dead probe independently verified by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The intended admin claim set is decided and recorded (the repo-wide convention is frozenset({"*", "system.configure"}); the MCP protocol copy currently says {"*"})
- [x] #2 One admin predicate remains; the other four call sites delegate to it
- [x] #3 A failing test passes a bare-string roles claim and asserts every module reaches the same authorization decision
- [x] #4 A failing test asserts a system.configure principal cannot reach another user's sandbox session unless that is the decided intent
- [x] #5 The dead context.is_admin probe is removed, or RequestContext gains the field so the branch is reachable in production
- [x] #6 Bandit run for touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Duplicate of TASK-13338, fixed in c49b56c281 before this task was picked up. Verified on ea1cbc6941: one predicate, protocol_types.metadata_has_admin_claims, reached via BaseModule.caller_is_admin; media/notes/kanban/sandbox call it, mcp_discovery._is_admin calls it then falls back to user_roles. Decision recorded in Docs/ADR/048-mcp-admin-claims.md: platform admin roles (admin/owner/super_admin, imported from AuthNZ _PLATFORM_ADMIN_ROLES) or the '*' permission; system.configure deliberately NOT admin in MCP (narrower than AuthNZ's frozenset({'*','system.configure'}) named in AC#1). Dead context.is_admin probes removed (test_dead_is_admin_attribute_is_not_consulted). Bare-string roles: matrix row {'roles': 'owner'} -> True for the shared predicate. Red-before: restoring the pre-c49b56c281 sandbox_module fails 3 tests in test_sandbox_module_auth_binding.py incl. test_sandbox_run_denies_system_configure_cross_user_override. Added 4c45412ab9: behavioural parity test for mcp_discovery's claims half (the one module not covered by the structural 'no re-declared _is_admin' test); fails on the pre-c49b56c281 discovery module ([owner] case), 32/32 pass now; admin matrix + sandbox binding 37 -> 42 pass. Bandit -ll clean on the test file. Breaking change carried from 13338: an API key whose only admin-ish claim is system.configure lost cross-user sandbox session access and needs a platform admin role.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Already fixed under TASK-13338 (c49b56c281, ADR-048). Verified every AC against the code and tests, confirmed red-before for the sandbox system.configure denial, and closed the one gap: mcp_discovery's claims half now has a behavioural parity test (4c45412ab9). No production code changed here.
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
