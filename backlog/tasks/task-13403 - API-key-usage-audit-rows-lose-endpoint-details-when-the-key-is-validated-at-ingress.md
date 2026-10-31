---
id: TASK-13403
title: >-
  API-key usage audit rows lose endpoint details when the key is validated at
  ingress
status: To Do
assignee: []
created_date: '2026-09-30 09:46'
labels:
  - authnz
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
RG ingress now validates API keys through get_auth_principal before routing (plan 2026-09-29-rg-ingress-safety-net, Task 9), and the route reuses the cached AuthContext. validate_api_key's usage recording therefore runs before require_token_scope sets _auth_endpoint_id, _auth_action and _auth_scope_name. With API_KEY_AUDIT_LOG_USAGE on, the 'used' audit rows lack endpoint details. Usage is also recorded for requests ingress then 429s; the identity cache in 391614cb05 limits that to once per 60 s per credential and IP.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Usage audit rows carry endpoint, action and scope when API_KEY_AUDIT_LOG_USAGE is on
- [ ] #2 A denied (429) request does not record API-key usage
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
