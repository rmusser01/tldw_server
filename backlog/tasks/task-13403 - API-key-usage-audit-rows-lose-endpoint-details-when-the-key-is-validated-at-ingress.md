---
id: TASK-13403
title: >-
  API-key usage audit rows lose endpoint details when the key is validated at
  ingress
status: Done
assignee: []
created_date: '2026-09-30 09:46'
updated_date: '2026-10-02 03:26'
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
- [x] #1 Usage audit rows carry endpoint, action and scope when API_KEY_AUDIT_LOG_USAGE is on
- [x] #2 A denied (429) request does not record API-key usage
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
R-U implemented: API-key usage is recorded at route time, not at ingress.
- RG ingress (middleware_simple._resolve_principal_entity) sets request.state flag User_DB_Handling.API_KEY_USAGE_DEFERRED around get_auth_principal; authenticate_api_key_user then validates with record_usage=False and leaves request.state._api_key_usage_pending = (key_id, user_id, client_ip). A recording (non-deferred) validation clears any pending entry, so usage is never counted twice.
- record_pending_api_key_usage(request) (User_DB_Handling) records the pending usage once, with the endpoint/action/scope require_token_scope left on request state. It is called from the three route-auth fast paths that reuse a cached AuthContext: core get_auth_principal (also behind auth_deps.get_auth_principal / CurrentPrincipal), get_request_user, and auth_deps.get_current_user.
- APIKeyManager.record_key_usage(key_id, user_id, ip, usage_details) extracted from validate_api_key (usage update + optional 'used' audit row); validate_api_key calls it, so non-ingress behavior is unchanged. _api_key_usage_details extracted from authenticate_api_key_user.
- A 429 at ingress never reaches route auth: nothing recorded. Ingress cache hit or RG off: route auth validates and records as before. As with RG off, a route with no auth dependency records no usage.
- ADR-056 identity bullet documents the deferral.
Tests: tests/AuthNZ_Unit/test_api_key_usage_at_route.py (real APIKeyManager recording path, key lookup/storage stubbed): (a) ingress-resolved key + route reached -> one usage update + one 'used' row with endpoint/action/scope, for get_request_user, get_auth_principal and get_current_user; (b) ingress 429 -> zero updates/rows while ingress did validate; (c) RG_ENABLED=false -> one update with details; (d) two requests, second an ingress cache hit -> one update + detailed row each. RED shown for (a) x3, (b), (d) before the fix; (c) is a guard that passes before and after.
Bandit (uvx bandit -ll) on touched AuthNZ/RG files: no findings. Ruff: only pre-existing I001/TRY203 in auth_deps.py and api_key_manager.py (present on origin/dev).

Review follow-up: approved with no Critical or Important issues; minors folded in.
- New test test_api_key_usage_at_route.py::test_route_revalidation_after_ingress_records_usage_once: route auth skips its fast path (non-User _auth_user) and re-validates after ingress deferred the usage; usage_updates == [7]. Mutation-checked: removing the pending-clear in authenticate_api_key_user fails the revalidate-then-reuse case ([7, 7]); removing the flag reset in the middleware finally block fails the revalidate case ([]).
Final summary: ingress validates API keys without recording usage; route auth records it exactly once, with endpoint/action/scope, when it first reuses the cached context; a 429 at ingress records nothing; RG off and ingress cache hits record at route auth as before.
Verification: RG + AuthNZ_Unit (-n 4, TLDW_TEST_NO_DOCKER=1): 1581 passed, 6 skipped, 2 xfailed. Docs tests: 212 passed. Bandit (uvx bandit -ll) on all touched source: no findings.
Known skips: Postgres-backed tests skip locally (no reachable Postgres). Unrelated failures seen in the symbol-hit sweep reproduce on an archive of origin/dev (see TASK-13402 notes).
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
