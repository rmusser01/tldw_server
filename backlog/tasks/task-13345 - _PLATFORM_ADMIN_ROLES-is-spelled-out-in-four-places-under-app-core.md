---
id: TASK-13345
title: _PLATFORM_ADMIN_ROLES is spelled out in four places under app/core
status: To Do
assignee: []
created_date: '2026-09-22 22:40'
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
- [ ] #1 One definition of the platform admin role set under app/core, imported by the other sites
- [ ] #2 ADR-048's deliberate MCP permission divergence still holds after consolidation
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
