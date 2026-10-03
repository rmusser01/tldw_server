---
id: TASK-13402
title: RG tenant scoping keys ingress on an unvalidated tenant header
status: Done
assignee: []
created_date: '2026-09-30 09:45'
updated_date: '2026-10-02 03:26'
labels:
  - rate-limit
  - security
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With tenant scoping enabled, deps.derive_entity_key returns tenant:<X-TLDW-Tenant header> (deps.py ~118-122, tenant.py ~38-41) without validating the header against the caller's principal. A client that rotates the header gets a fresh ingress bucket on every request. This predates the RG safety net: R18 kept tenant precedence unchanged, and the auth single-charge skip no longer trusts tenant: entities (commit 832e4a69df). Tenant scoping is opt-in and off by default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The tenant entity comes only from a validated principal's tenant or org, or an unvalidated header can never mint a new bucket
- [x] #2 A test proves that rotating the tenant header on one IP shares one bucket
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
R-T implemented: an RG tenant entity now comes only from a validated principal.
- tenant.get_tenant_id(member_of=...): the tenant header is honored only when it names one of the validated principal's tenants (org ids); otherwise the validated claim is used. An unvalidated header never names a bucket.
- deps.derive_entity_key passes member_of from request state (org_ids, auth.principal.org_ids, tenant/org claims), so endpoint-level reservations (chat, embeddings, workflows) get the same rule.
- middleware_simple: ingress always resolves the principal (same cache + per-IP budget), caching an _Identity(entity, org_ids, active_org_id); _charge_entity picks tenant:<header> if the principal belongs to it, else tenant:<active org>, else the principal's own entity. JWTs use signature-verified org_ids/active_org_id claims; API keys/cookie use AuthPrincipal.org_ids/active_org_id. Anonymous/invalid callers ignore the header and pay their IP. rg_ingress_entity and the auth single-charge exact-match skip are unchanged (comment in endpoints/auth.py updated).
- ADR-056 tenant sentence and bullet rewritten; middleware docstrings updated.
Tests: test_middleware_identity.py (rotating header anonymous -> one ip bucket; invalid credential + header -> ip; foreign header -> principal's tenant / principal entity; member header -> tenant:<id>; cache hit still validates each header; JWT org claims; tenant scoping disabled unchanged; existing tenant-precedence test now uses member principals), test_deps_trusted_proxy.py (route-time derive_entity_key), test_middleware_simple.py (anonymous header -> ip). RED shown for 9 before the fix; all pass after.
Bandit (uvx bandit -ll) on touched RG files: no findings.
Known: AuthPrincipal has no tenant field, so tenant == org id; the TenantScopeConfig.jwt_claim is honored at route time (state claims) but at ingress only org_ids/active_org_id claims are read.

Review follow-up: approved with no Critical or Important issues; minors folded in.
- Ingress now picks a caller's own tenant in the same order as route-level derive_entity_key: deps.tenant_claims_from_state (tenant claim, then active org, then org; for an API key, its scoped org or first org) is the one shared helper. Before, a multi-org API key with no active org and no header was charged user:<id> at ingress but tenant:<first org> at endpoint reservations. jwt_claim stays route-only; a JWT's own tenant at ingress is its active_org_id claim, the only org route JWT auth exposes.
- Test: test_api_key_usage_at_route.py::test_multi_org_key_without_active_org_gets_the_same_tenant_at_ingress_and_route (real authenticate_api_key_user; ingress and route both tenant:1, including on an identity-cache hit). RED before the fix (ingress was user:42).
- ADR-056: a member who sends no header is now pooled under their own tenant; a user removed from an org can keep charging its tenant: bucket for up to 60 s (identity cache) or the access-token lifetime (JWT org claims), only for orgs they once belonged to. Docs/Published copy refreshed; tldw_Server_API/tests/Docs: 212 passed.
Final summary: the RG tenant entity comes only from a validated principal (the header only selects among its orgs; anonymous callers pay their IP), at ingress and at endpoint reservations, with one shared own-tenant order.
Verification: RG + AuthNZ_Unit (-n 4, TLDW_TEST_NO_DOCKER=1): 1581 passed, 6 skipped, 2 xfailed. Bandit (uvx bandit -ll) on all touched source: no findings.
Known skips: Postgres-backed tests skip locally (no reachable Postgres; with Docker auto-start they hang to the timeout identically on origin/dev). Unrelated failures seen in the symbol-hit sweep (workspace_activity_index x4, docling PDF x2, e2e chatbook_sync_v2 x6, Telegram xdist flakes) reproduce on an archive of origin/dev.
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
