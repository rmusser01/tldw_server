---
id: TASK-13402
title: RG tenant scoping keys ingress on an unvalidated tenant header
status: Done
assignee: []
created_date: '2026-09-30 09:45'
updated_date: '2026-10-02 00:53'
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
