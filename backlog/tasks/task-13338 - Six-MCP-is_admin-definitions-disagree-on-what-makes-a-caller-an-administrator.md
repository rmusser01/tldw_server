---
id: TASK-13338
title: Six MCP is_admin definitions disagree on what makes a caller an administrator
status: To Do
assignee: []
created_date: '2026-09-22 05:00'
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
- [ ] #1 One caller_is_admin on BaseModule used by all six sites
- [ ] #2 Dead getattr(context, is_admin) probes removed
- [ ] #3 Claim-matrix test covers owner, super_admin, admin and system.configure
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
