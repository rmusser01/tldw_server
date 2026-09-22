---
id: TASK-13320
title: >-
  Five divergent _is_admin checks; the most permissive guards cross-user sandbox
  execution
status: To Do
assignee: []
created_date: '2026-09-22 04:55'
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
- [ ] #1 The intended admin claim set is decided and recorded (the repo-wide convention is frozenset({"*", "system.configure"}); the MCP protocol copy currently says {"*"})
- [ ] #2 One admin predicate remains; the other four call sites delegate to it
- [ ] #3 A failing test passes a bare-string roles claim and asserts every module reaches the same authorization decision
- [ ] #4 A failing test asserts a system.configure principal cannot reach another user's sandbox session unless that is the decided intent
- [ ] #5 The dead context.is_admin probe is removed, or RequestContext gains the field so the branch is reachable in production
- [ ] #6 Bandit run for touched scope
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
