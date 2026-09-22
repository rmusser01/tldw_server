---
id: TASK-13338
title: Six MCP is_admin definitions disagree on what makes a caller an administrator
status: Done
assignee: []
created_date: '2026-09-22 05:00'
updated_date: '2026-09-22 22:52'
labels:
  - bug
  - mcp
  - security
dependencies: []
references:
  - >-
    tldw_Server_API/app/core/MCP_unified/modules/implementations/sandbox_module.py:210
  - 'tldw_Server_API/app/core/MCP_unified/protocol_types.py:134'
  - 'tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py:131'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Six _is_admin definitions with six different claim sets, consuming a cross-user resource gate and two irreversible deletes.

UNDER-GRANT: a principal with AuthNZ role "owner" (AuthPrincipal.is_admin True) connects; server.py:1486 writes metadata roles ["owner"]. Every MCP predicate tests only the literal "admin", so a platform OWNER is refused permanent media delete, permanent note delete and all kanban policy ops while being admin everywhere else in the product. Same for super_admin.

OVER-GRANT: an API key normalised into permissions containing system.configure (no admin role) is admin per sandbox_module.py:219-226 and therefore passes the CROSS-USER check at sandbox_module.py:142, reaching another user sandbox session - while protocol_types._metadata_has_admin_claims, the predicate MCP uses for its own trusted-claims gate, returns False for that same input.

DEAD PROBES: kanban_module.py:1587 and sandbox_module.py:212 read getattr(context, "is_admin", False) on a RequestContext that defines no such attribute, and server.py:1483-1487 drops principal.is_admin when building it - so both quietly run the roles-only logic they appear stricter than.

Canonical: protocol_types._metadata_has_admin_claims, delegating to AuthNZ _claims_mark_admin / _PLATFORM_ADMIN_ROLES. Aligning widens who counts as admin, so it needs a design record and a claim-matrix test.

Source: synthesis F37
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One caller_is_admin on BaseModule used by all six sites
- [x] #2 Dead getattr(context, is_admin) probes removed
- [x] #3 Claim-matrix test covers owner, super_admin, admin and system.configure
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in c49b56c281. Design record: Docs/ADR/048-mcp-admin-claims.md.

AC1: one predicate, protocol_types.metadata_has_admin_claims, reached by modules through BaseModule.caller_is_admin. media, notes, kanban and sandbox lost their own definitions entirely; mcp_discovery keeps its user_roles table lookup as a fallback but its claims half is now the shared predicate -- discovery is the only site that can fall back to a stored role assignment when the request carries no admin claim, so that half is genuinely its own.

AC2: both dead getattr(context, "is_admin", False) probes removed. test_dead_is_admin_attribute_is_not_consulted asserts an object carrying is_admin=True with non-admin roles is still not an admin, so the backdoor cannot be reintroduced.

AC3: test_admin_claims_matrix.py, 27 cases, each row carrying its reason -- owner, super_admin, admin, case and whitespace variants, bare-string and tuple claim shapes, "*", system.configure, the "admin" permission, and junk shapes. Plus test_roles_are_taken_from_authnz (iterates AuthNZ's actual frozenset, so adding a role there is covered automatically) and test_every_module_shares_the_one_predicate.

DECISION (owner, this session): roles widen, permissions do not. caller_is_admin accepts roles in AuthNZ's _PLATFORM_ADMIN_ROLES {admin, owner, super_admin} and the "*" permission, but NOT system.configure and NOT the "admin" permission -- deliberately narrower than AuthNZ's _ADMIN_CLAIM_PERMISSIONS. Full parity was rejected because it would have granted permanent media delete, permanent note delete and kanban policy ops to any system.configure holder. Roles-only was rejected because "*" genuinely means every permission and two sites honour it today. The divergence is load-bearing and must not be tidied into parity without revisiting ADR-048; the matrix test pins both halves.

BREAKING CHANGE: an API key whose only admin-ish claim is system.configure loses cross-user sandbox session access and must be granted a platform admin role instead. This is the over-grant fix. tests/MCP_unified/test_sandbox_module_auth_binding.py::test_sandbox_run_allows_permission_based_admin_override asserted the old behaviour; it is now test_sandbox_run_denies_system_configure_cross_user_override, with a companion parametrised test covering admin/owner/super_admin/"*", both carrying the reason in the docstring rather than changed silently.

Follow-up filed as TASK-13345: _PLATFORM_ADMIN_ROLES is still spelled out identically in AuthNZ/auth_principal_resolver.py, AuthNZ/byok_helpers.py and Claims_Extraction/claims_service.py. MCP imports the first rather than making a fourth copy, but three independent definitions remain -- the same divergence risk one level up.

Verification: MCP_unified in-app 13 failed / 3328 passed vs baseline 13 / 3281, identical failure set. tests/MCP + MCP_Hub + MCP_unified 4 failed, unchanged (the one new failure was the over-grant test above, then updated). tests/AuthNZ_Unit/test_auth_principal_resolver.py and tests/Agent_Client_Protocol/test_acp_endpoints.py both pass, 38 tests. Bandit clean over all seven touched files (run via uvx; bandit is CI-only, not a declared local dependency).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Six divergent admin predicates collapsed to one, backed by AuthNZ's role set. Platform owner and super_admin gain the operations they were entitled to everywhere else; system.configure loses cross-user sandbox access. The permission narrowing is a deliberate divergence from AuthNZ, recorded in ADR-048 and pinned by a 27-case claim matrix.
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
