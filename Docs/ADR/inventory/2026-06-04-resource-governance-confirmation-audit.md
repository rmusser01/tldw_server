# Resource Governance Confirmation Audit - 2026-06-04

**Related task:** TASK-2233
**Follow-up:** TASK-2234
**Inventory row:** INV-028
**Source candidate:** `tldw_Server_API/app/core/Resource_Governance/README.md`
**Disposition:** Current governing; ready for a bounded accepted ADR backfill.

## Decision Candidate Under Review

INV-028 summarized the Resource Governance convention as:

> New endpoints should use claim-first auth; latency/cost-sensitive endpoints should decide Resource Governor policy and route-map coverage; DB policy store can merge file route maps and fail closed on missing DB policies.

The candidate is current enough for accepted ADR backfill if the ADR is scoped to new-endpoint governance, route-map ownership, DB policy-store route-map merge behavior, and request-ingress missing-policy denial. It should not claim universal coverage for every existing endpoint or blanket fail-closed behavior for every Resource Governor category/outage mode.

## Confirmed Evidence

| Claim | Evidence | Result |
| --- | --- | --- |
| New endpoints should use claim-first auth dependencies. | `tldw_Server_API/app/core/Resource_Governance/README.md:11` says new endpoints should use `get_auth_principal`, `RequirePermission(...)`, `RequireRole(...)`, or `require_service_principal()` and should not gate new behavior on `AUTH_MODE` or mode helpers. `Docs/Published/Code_Documentation/Guides/AuthNZ_Code_Guide.md:281` through `:283` repeats the same guardrail. `tldw_Server_API/tests/AuthNZ_Unit/test_claim_first_single_user_mode_guardrail.py:23` through `:42` scans API v1 endpoint code for non-allowlisted `is_single_user_mode()` authorization branches. Resource Governor admin endpoints import `RequireRole` at `tldw_Server_API/app/api/v1/endpoints/resource_governor.py:11` and gate admin/control routes with `Depends(RequireRole("admin"))` at `:71` through `:75`, `:212` through `:214`, `:275` through `:277`, and later diagnostic routes. | Confirmed as a new-endpoint governance rule and route-level guardrail. |
| Latency/cost-sensitive endpoints should decide Resource Governor applicability and route-map coverage. | `tldw_Server_API/app/core/Resource_Governance/README.md:12` and `:13` require a Resource Governor decision for latency/cost-sensitive/user-facing endpoints and a matching policy-store plus `route_map` entry when applicable. The default YAML contains Resource Governor policies and route maps under `tldw_Server_API/Config_Files/resource_governor_policies.yaml:267` through `:414`, including chat, embeddings, audio, RAG, chatbooks, watchlists, and AuthNZ paths. `tldw_Server_API/tests/Resource_Governance/test_slowapi_decorated_routes_mapped.py:30` through `:57` verifies representative ingress-limited paths resolve to an existing policy, and `tldw_Server_API/tests/Resource_Governance/test_auth_route_map_coverage.py:35` through `:57` verifies AuthNZ routes resolve to `authnz.*` policies. | Confirmed as route-map ownership and representative coverage, not as an all-endpoints guarantee. |
| DB policy store can merge file route maps, with file route-map precedence. | The README documents `RG_POLICY_STORE=file|db` at `tldw_Server_API/app/core/Resource_Governance/README.md:17` through `:18` and says DB mode merges the file `route_map` into the DB policy snapshot at `:114`. `PolicySnapshot` carries `route_map` at `tldw_Server_API/app/core/Resource_Governance/policy_loader.py:22` through `:26`; DB loading reads DB policies and route map at `:63` through `:78`, reads the file route map at `:81` through `:90`, and merges file route maps over DB route maps at `:92` through `:111`. `tldw_Server_API/tests/Resource_Governance/test_policy_loader_route_map_db_store.py:16` through `:30` verifies DB-store snapshots include route-map entries from the file, and `tldw_Server_API/tests/Resource_Governance/test_policy_loader_reload_db_store.py:28` through `:57` verifies file route-map entries survive DB policy reloads. | Confirmed. |
| Route-map resolution is path first, then tag. | The README states this resolution order at `tldw_Server_API/app/core/Resource_Governance/README.md:125`. `RGSimpleMiddleware._derive_policy_id()` initializes route maps at `tldw_Server_API/app/core/Resource_Governance/middleware_simple.py:99` through `:104`, checks `by_path` at `:116` through `:132`, then checks `by_tag` at `:133` through `:145`. `tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py:73` through `:95` verifies denial and headers through tag/path route maps, and `:123` through `:129` verifies newer domain paths resolve through explicit path mappings. | Confirmed. |
| Middleware ingress enforcement is request-category enforcement; other categories need endpoint plumbing. | The README says middleware requests only at `tldw_Server_API/app/core/Resource_Governance/README.md:125` through `:126`. `RGSimpleMiddleware.__call__()` derives the policy id at `tldw_Server_API/app/core/Resource_Governance/middleware_simple.py:210`, stores it on request state at `:217`, creates an `RGRequest` with `categories={"requests": {"units": 1}}` at `:226`, and emits `Retry-After`/`X-RateLimit-*` headers on denial at `:279` through `:296`. | Confirmed. |
| Missing DB policy IDs referenced by route maps fail closed for request ingress. | The README scopes this to DB mode and missing request limits at `tldw_Server_API/app/core/Resource_Governance/README.md:114` through `:115`. The in-memory governor returns `{}` for a missing policy at `tldw_Server_API/app/core/Resource_Governance/governor.py:200` through `:208`; missing/zero request config returns denied headroom at `:303` through `:314`; request checks use the resolved policy at `:390` through `:402`. The Redis governor also resolves missing policies to `{}` at `tldw_Server_API/app/core/Resource_Governance/governor_redis.py:347` through `:354` and reads request rpm as `0` when missing at `:731` through `:737`. | Confirmed for request ingress. Do not broaden this into a blanket claim for tokens, concurrency categories, or Redis outage policy. |

## Caveats For ADR Backfill

- The ADR should describe a new-endpoint governance rule and route-map ownership expectation. It should not claim every existing API endpoint already uses claim-first auth or has Resource Governor route-map coverage.
- Middleware enforcement is request-category ingress only. Token, stream, job, and minute-budget categories still require endpoint-level reserve/commit plumbing.
- The fail-closed claim should be scoped to route-map entries that resolve to missing request policies. Redis backend outage behavior remains configurable through `RG_REDIS_FAIL_MODE` and per-policy/category fail modes.
- I found tests for route-map merge, route-map coverage, path/tag resolution, and claim-first guardrails. I did not find a focused regression named specifically for "route_map references missing DB policy and returns 429"; TASK-2234 can either rely on the existing governor code path or add that narrow test before creating the accepted ADR.

## Recommended Next Action

Create one accepted ADR via TASK-2234, expected as `Docs/ADR/018-resource-governance-endpoint-policy-and-route-map.md`, covering:

1. New endpoints use claim-first auth dependencies for authorization.
2. Latency/cost-sensitive or user-facing endpoints must explicitly decide Resource Governor applicability.
3. Applicable ingress routes need policy-store and route-map ownership.
4. DB policy-store mode merges file `route_map` entries into DB policy snapshots, with file route-map precedence.
5. Request ingress fails closed when the route map resolves to a missing request policy.

Keep Redis outage fail modes, non-request category plumbing, and all-endpoint coverage as consequences or follow-up notes rather than accepted decision claims.
