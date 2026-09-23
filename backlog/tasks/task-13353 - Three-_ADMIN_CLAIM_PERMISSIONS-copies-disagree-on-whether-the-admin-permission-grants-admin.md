---
id: TASK-13353
title: >-
  Three _ADMIN_CLAIM_PERMISSIONS copies disagree on whether the 'admin'
  permission grants admin
status: To Do
assignee: []
created_date: '2026-09-23 01:21'
labels:
  - authnz
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while consolidating the role set under TASK-13345. This is a POLICY question, not a deduplication, which is why it was not folded into that change.

The three copies were all introduced in the SAME commit (d0654d0cfb) with one already different, so the divergence has existed from birth and nothing marks it as intentional:

  core/AuthNZ/auth_principal_resolver.py:71   {"*", "system.configure", "admin"}
  core/AuthNZ/byok_helpers.py:56              {"*", "system.configure"}
  core/Claims_Extraction/claims_service.py    {"*", "system.configure"}

Consequence: a principal whose only admin-ish claim is the "admin" PERMISSION (not the admin ROLE) is treated as an administrator during principal resolution, but not by BYOK credential handling and not by Claims. Whether that is correct depends on what the "admin" permission is meant to mean, which the code does not say anywhere.

Resolving it requires a decision, and it moves real authorisation in one direction or the other:
- add "admin" to the other two -> BYOK and Claims newly treat that permission as administrator
- remove it from the resolver -> principals holding only that permission lose administrator status at the front door

MCP_unified diverges further and DELIBERATELY: per ADR-048 it accepts only "*", because a configuration permission should not authorise destroying another user's data. Any consolidation must preserve that rather than flatten it.

Note the role set is now single-sourced at core/AuthNZ/platform_admin.py with a ratchet; only the permission set remains duplicated.

Source: TASK-13345.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single decision is recorded for whether the 'admin' permission confers administrator status
- [ ] #2 The permission sets agree, or each divergence carries a stated reason
- [ ] #3 ADR-048's MCP divergence still holds
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
