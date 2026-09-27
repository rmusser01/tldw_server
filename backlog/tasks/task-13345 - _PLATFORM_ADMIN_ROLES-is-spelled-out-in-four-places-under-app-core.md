---
id: TASK-13345
title: _PLATFORM_ADMIN_ROLES is spelled out in four places under app/core
status: Done
assignee: []
created_date: '2026-09-22 22:40'
updated_date: '2026-09-23 01:22'
labels:
  - duplication
  - authnz
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The frozenset {"admin", "owner", "super_admin"} is restated identically in:

- tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py:70
- tldw_Server_API/app/core/AuthNZ/byok_helpers.py:55
- tldw_Server_API/app/core/Claims_Extraction/claims_service.py:124

MCP_unified/protocol_types.py now imports the auth_principal_resolver copy rather than making a fourth, but three independent definitions remain. Adding a platform admin role requires finding all three; missing one produces exactly the under-grant that TASK-13338 fixed in MCP, where a platform owner was refused operations they were entitled to everywhere else.

The companion set _ADMIN_CLAIM_PERMISSIONS has the same shape. Note that MCP deliberately diverges on permissions per ADR-048 -- any consolidation must preserve that, not flatten it.

Source: found while fixing TASK-13338 / synthesis F37.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One definition of the platform admin role set under app/core, imported by the other sites
- [x] #2 ADR-048's deliberate MCP permission divergence still holds after consolidation
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done in d8b16dff72.

AC1: the role set now has one definition, core/AuthNZ/platform_admin.PLATFORM_ADMIN_ROLES, aliased by every former copy under its existing local name.

SCOPE CORRECTION: there were SIX copies, not the three this task recorded. The original search covered app/core only. The others are services/admin_budgets_service.py:45, services/admin_profiles_service.py:62, and endpoints/admin/admin_ops.py:149, the last under the local name _INCIDENT_ASSIGNABLE_ROLES -- same value, and used at :203 as an admin check on the current user, so it was rewired too while keeping its name.

The canonical module imports nothing, deliberately. MCP_unified had been importing the private _PLATFORM_ADMIN_ROLES out of auth_principal_resolver, which pulls FastAPI's Request in for the sake of a frozenset; it now imports the dependency-free module, and a test pins that it stays that way.

AC2: ADR-048's MCP divergence is untouched -- MCP still accepts only the '*' permission, and the admin-claims matrix (27 cases) still passes.

A ratchet, tests/lint/test_platform_admin_roles_single_definition.py, fails if a seventh literal copy appears anywhere under app/. Verified by actually adding one: appending _SEVENTH_COPY to admin_budgets_service.py trips it, and it restores clean. It also pins that the canonical module still holds the definition and still imports nothing, so it cannot pass by the definition quietly moving.

FOUND AND FILED SEPARATELY: the companion _ADMIN_CLAIM_PERMISSIONS copies disagree and always have -- the resolver accepts the 'admin' permission, BYOK and Claims do not, all three introduced in the same commit d0654d0cfb. That is a policy decision about authorisation, not a deduplication, so it was documented in the platform_admin docstring and filed rather than resolved by keeping whichever copy I happened to touch first.

Verification: AuthNZ_Unit + lint + MCP admin-claims matrix, 15 failed / 1207 passed with the change vs 15 failed / 1204 passed without (baseline taken by restoring every file from HEAD), identical failure sets; the +3 is the new ratchet. All six sites confirmed to share one object at runtime. app.main still builds its 164 routes.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
