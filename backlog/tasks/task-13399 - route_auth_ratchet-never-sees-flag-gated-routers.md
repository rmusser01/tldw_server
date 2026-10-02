---
id: TASK-13399
title: route_auth_ratchet never sees flag-gated routers
status: Done
assignee: []
created_date: '2026-09-30 06:33'
updated_date: '2026-10-02 01:05'
labels:
  - ci
  - security
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Helper_Scripts/ci/route_auth_ratchet.py sets ROUTE_POLICY_ENV to force-enable benchmarks, connectors and personalization (comment ~lines 37-38). But load_app() clears PYTEST_CURRENT_TEST, TEST_MODE and TLDW_TEST_MODE before importing the app (~148-149), and config.route_enabled only honors ROUTES_ENABLE / ROUTES_STABLE_ONLY under explicit pytest or test mode (app/core/config.py ~3565-3666). So the force-enable does nothing: the ratchet never inspects routers with default_stable=False, and an unauthenticated route there would pass CI. Found during the RG route-map lint review (plan 2026-09-29-rg-ingress-safety-net, Task 11).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The route-auth ratchet builds the app with benchmarks, connectors and personalization mounted and checks their routes
- [x] #2 A test fails if a force-enabled router is missing from the app the ratchet inspects
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed: Helper_Scripts/ci/route_auth_ratchet.py's load_app() cleared PYTEST_CURRENT_TEST/TEST_MODE/TLDW_TEST_MODE before import, which made config.py's _route_toggle_policy() ignore the ROUTES_ENABLE/ROUTES_STABLE_ONLY env vars it sets (those are only honored under explicit pytest or test-mode runtime). So the force-enable for benchmarks/connectors/personalization (and the other 11 keys in ROUTE_POLICY_ENV) was inert, and those default_stable=False routers were never mounted in the app the ratchet inspects.

Fix: added _ratchet_config_dir() to route_auth_ratchet.py. It copies the real Config_Files directory to a temp dir, adds ROUTE_POLICY_ENV's route keys to the copy's config.txt [API-Routes] enable list (config.txt is read unconditionally by _route_toggle_policy, in every runtime), and points TLDW_CONFIG_DIR at the copy for load_app()'s subprocess only -- the same lever an operator already has via config.txt, never the real repo file, and no change to config.py's production route-gating logic. rg_route_map_lint.py shares load_app(), so it benefits too in principle (see concern below).

Test (AC2): added test_force_enabled_routers_are_mounted to tldw_Server_API/tests/lint/test_route_auth_ratchet.py, building the app out-of-process via load_app() and asserting /api/v1/benchmarks, /api/v1/connectors and /api/v1/personalization paths are present. Verified red against the unfixed load_app() (AssertionError, routes absent) and green after the fix.

Narrow sibling-dependency fixes (every other route in the router already required it):
- tldw_Server_API/app/api/v1/endpoints/connectors.py:445 list_providers() had no Depends at all; every other connectors.py route (except the HMAC-verified webhook) requires get_auth_principal. Added `principal: AuthPrincipal = Depends(get_auth_principal)`.
- tldw_Server_API/app/api/v1/endpoints/personalization.py:308 list_explanations() had no Depends; every other personalization.py route requires get_personalization_db_for_user or get_usage_event_logger (both resolve to get_request_user). Added `log: UsageEventLogger = Depends(get_usage_event_logger)`, unused in the body, matching the existing pattern in add_memory/update_memory/delete_memory in the same file.

Real findings NOT fixed (no safe sibling pattern; reported per instructions, not allowlisted):
- POST /api/v1/benchmarks/{benchmark_name}/run (benchmark_api.py:182) and POST /api/v1/benchmarks/simpleqa/evaluate (benchmark_api.py:345): only Depends(get_rate_limiter_dep), a rate limiter, not an authenticator (matches the exact anti-pattern this ratchet exists to catch). run_benchmark executes model-evaluation workloads; this allows an unauthenticated caller to trigger them.
- GET /api/v1/benchmarks/list (benchmark_api.py:67), GET /api/v1/benchmarks/{benchmark_name}/info (benchmark_api.py:98), GET /api/v1/benchmarks/{benchmark_name}/samples (benchmark_api.py:132): zero auth, zero rate limiting. No consistent sibling pattern in this router to extend (the whole router lacks real auth), so left unfixed per instructions.

Routes newly visible but consistent with existing "public by design" baseline conventions (health checks, OAuth redirect callbacks, signed/secret-verified webhooks, invite-token flows, a deny-only traversal guard) -- reported, not touched or allowlisted: GET /api/v1/discord/oauth/callback, POST /api/v1/discord/interactions (Ed25519-signature verified), GET /api/v1/slack/oauth/callback, POST /api/v1/slack/commands, POST /api/v1/slack/events (Slack-signed), POST /api/v1/telegram/webhook, GET /api/v1/guardian/wizard/invites/preview, POST /api/v1/guardian/wizard/invites/accept/register (invite-token gated, same pattern as /api/v1/invites/preview and /auth/register already in the baseline), GET /api/v1/meetings/health, GET /api/v1/sandbox/health/public, GET /api/v1/self-monitoring/crisis-resources, GET,POST /api/v1/connectors/providers/{provider}/webhook (shared-secret verified), GET /api/v1/sandbox/runs/{run_id}/{rest:path} (sandbox.py:2571, a deny-only 400/404 traversal-guard fallback that never serves data; the real artifact routes require get_request_user + _require_run_owner).

tldw_Server_API/tests/lint/test_route_auth_ratchet.py::test_no_new_unauthenticated_routes now fails (by design -- it reports the above real/undetermined findings) since none were allowlisted per instructions. Full suite: 70 passed, 1 failed, 1 skipped (pre-existing skip, unrelated).

Checked Helper_Scripts/ci/rg_route_map_lint.py / rg_route_map_lint_allowlist.txt per instructions: its three benchmarks/connectors/personalization by_path entries are NOT satisfied despite the fix. Reproduced directly: calling default_policy_loader().load_once() before the first iter_served_routes() call (rg_route_map_lint.py's own call order) leaves ~188 routes (these three families among them) missing from iter_route_contexts()'s result, vs. calling iter_served_routes() first. This is a separate, pre-existing quirk in how rg_route_map_lint.py/fastapi_routes.py interacts with FastAPI's effective-route caching, unrelated to this ratchet fix. Left rg_route_map_lint_allowlist.txt untouched (confirmed via diff against origin/dev) since none of its entries are satisfied; `python Helper_Scripts/ci/rg_route_map_lint.py` exits 0 (clean) with the file as-is.
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
