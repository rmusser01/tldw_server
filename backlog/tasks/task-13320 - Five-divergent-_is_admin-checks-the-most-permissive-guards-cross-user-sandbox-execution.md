---
id: TASK-13320
title: >-
  Five divergent _is_admin checks; the most permissive guards cross-user sandbox
  execution
status: To Do
assignee: []
created_date: '2026-09-22 04:55'
updated_date: '2026-09-22 21:24'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
REFRAMED after verification. Design doc: Docs/Design/2026-09-22-mcp-capability-checks-design.md

The filed framing -- five divergent _is_admin checks, 'four of the five are wrong' -- does not survive looking at what each one gates. The five sites ask FOUR DIFFERENT QUESTIONS:

  may I read another user's data?   sandbox:142 (another user's session), media:2077 (ownership bypass on every media access)
  may I destroy data irreversibly?  media:1847, notes:2010 (permanent hard delete)
  may I control workflow execution? kanban:1753,1763,1773,1802 (pause/resume/drain/force-reassign)
  what scope should I list?         mcp_discovery:157 (arguably not an authz question at all)

There is no claim set that is simultaneously correct for 'may read another tenant's data' and 'may pause a board', so unifying on any single definition necessarily leaves some call site wrong. That is why they drifted and why nothing records which is intended. The defect is that five distinct capability checks were collapsed onto one undifferentiated role name.

Note media:2077 is a SECOND cross-user boundary the description did not mention -- and it sits on the NARROWEST of the five checks, the opposite of the filed 'the broadest check guards the most dangerous operation'.

The vocabulary already exists and is unused: rbac_seed.py:30-53 seeds resource-scoped names, MCP has a Resource x Action enum and AuthNZRBAC.check_permission backed by the AuthNZ DB, failing closed. No module uses any of it for these decisions; all five read raw JWT metadata, bypassing the RBAC layer one directory over.

Adopting that layer as-is would not fix it either: authnz_rbac.py:_map_to_permission resolves Action.ADMIN to 'system.configure' for EVERY resource, so the five would collapse onto one permission again, and Resource has no SANDBOX or board member -- it cannot name two of the four questions.

THREE SUPPORTING CLAIMS DID NOT HOLD UP.

1. The cited test does not exist. tests/.../test_protocol_scope_enforcement.py is not in the repository; it was the only evidence offered for 'the protocol accepts the string shape'. The claim is true -- protocol_types.py:113 returns (value,) for a str -- but it was cited against a file that is not there.

2. The bare-string roles split is latent, not live. All three producers are safe: server.py:1486 and :1592 wrap in list(), and :1615 assigns token_data.roles, a Pydantic list[str] field that rejects a bare string rather than coercing it. A string-shaped roles claim can only arrive from a directly constructed RequestContext.

3. The system.configure cross-user reach is narrow. _build_role_grants (rbac_seed.py:112) gives it to the admin role only -- user, moderator, viewer and reviewer do not receive it. Single-user mode grants it (User_DB_Handling.py:291) but also sets roles=['admin'] and is_admin=True. The set of principals holding system.configure without already being admin-by-role is empty unless granted deliberately as an override.

Severity is therefore lower than filed: a consistency defect on two cross-user boundaries, not an active breach. Confirmed as filed: the five definitions and their three axes of divergence, and that getattr(context,'is_admin',False) is dead against the real type (zero occurrences of is_admin in protocol_types.py).

No code changed. Every permission name in the doc's D2 table needs approval before implementation, because each option moves a cross-user boundary. All 36 file:line citations in the doc were verified to resolve -- deliberately, given claim 1 above.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
