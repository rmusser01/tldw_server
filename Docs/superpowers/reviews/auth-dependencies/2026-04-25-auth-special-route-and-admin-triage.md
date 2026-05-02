# Phase 3.4 Auth Special-Route And Admin Triage

Date: 2026-04-25

Scope: `tldw_Server_API/app/api/v1/endpoints`

This triage expands the Phase 3.4 auth risk scan with route-level signals for admin checks and special-route categories. It is still static analysis. Treat it as a migration map, not as proof that any route is misconfigured.

## Admin-Check Summary

Route-level admin-signal buckets:

| Bucket | Routes | Migration meaning |
| --- | ---: | --- |
| Explicit route admin guard only | 85 | Usually straightforward to migrate to a standard `RequireRole`/admin alias while preserving behavior. |
| Route admin guard plus manual/service admin check | 19 | Likely intentional defense in depth. Do not remove service/manual checks unless ownership proves they are redundant. |
| Manual or service admin check only | 70 | Highest review priority. Confirm whether admin enforcement comes from router-level dependencies, service-layer checks, or route body logic before changing dependencies. |
| Permission guard plus manual admin check | 25 | Preserve permission checks and admin/ownership fallback semantics. These are not first-wave cleanup candidates. |

## Highest-Risk Admin Buckets

### Manual Or Service Admin Only

Representative high-signal families:

- `admin/admin_ops`: maintenance mode, feature flags, incidents, webhooks, billing analytics, dependency status, compliance schedules.
- `admin/admin_data_ops`: backup schedule list/create/update/pause/resume/delete routes.
- `workflows`: versioning, run, auth-check, and admin-sensitive execution routes.
- `scheduler_workflows`: schedule list/get/update/delete/run-now routes.
- `flashcards`: structured import and package import routes.
- `kanban/kanban_workflow`: workflow state override and recovery controls.
- `prompt_studio/*`: project, prompt, and status routes with admin-body signals.
- `media/ingest_jobs`: stream/cancel routes with admin-body signals.

Migration rule: do not replace these route dependencies until the route include stack and service-layer guards are mapped. If admin is only enforced inside the route/service, moving to a standard route alias may be useful, but it must be backed by denial-path tests.

### Route Guard Plus Manual Or Service Admin

Representative families:

- `orgs`: org/team membership management and invite routes.
- `embeddings_v5_production_enhanced`: cache, circuit-breaker, and metrics admin routes.
- `workflows`: virtual key route.
- `resource_governor`: policy read route.
- `connectors`: org policy route.

Migration rule: keep both layers unless there is a written owner decision to remove one. For service methods callable outside FastAPI, service-layer checks are defense in depth.

### Permission Plus Manual Admin

Representative families:

- `workflows`: run, event, artifact, approval, reject, and retry routes.
- `mcp_unified_endpoint`: request/tool/token routes.
- `audit`: export/count routes.
- `embeddings_v5_production_enhanced`: compactor route.
- `evaluations/evaluations_unified`: history route.

Migration rule: preserve permission semantics first. Admin checks may be ownership fallbacks, tenant boundaries, or service-layer hard stops.

## Special-Route Categories

These categories should not be forced into the generic user-auth cleanup path.

### Setup-Local

The `setup` route family has `23` setup-local route signals, including setup status/config, audio provisioning, Omnivoice setup actions, setup completion/reset, assistant, and self-verify routes.

Migration rule: keep setup-local and shared-audio installer dependencies separate from normal user auth. They are deployment/setup gates, not generic principal gates.

### Webhook, OAuth, And Callback

`38` routes have webhook/OAuth/callback signals. Representative families:

- `slack`, `discord`, `telegram`
- `collections_websub`
- `connectors` OAuth callback
- `auth` federation callback
- `user_keys` OpenAI OAuth callback/status/refresh/disconnect
- `acp_triggers` webhook receiver
- `admin/admin_webhooks` and `admin/admin_ops` webhook management
- `workflows` webhook DLQ/delivery routes

Migration rule: split inbound external callbacks from admin management routes. Inbound callbacks may be public but secret-verified; admin management routes should keep normal admin/permission gates.

### Provider-Compatible

`77` route signals are provider-compatible or provider-adjacent, including:

- OpenAI-compatible chat/messages/vector-store surfaces
- OpenAI-compatible embeddings and vector-store endpoints
- evaluation endpoints with provider-shaped contracts
- media embedding endpoints that may share provider-style payloads

Migration rule: keep public contract compatibility stable. Auth dependency cleanup can happen only when it does not alter status codes, headers, or provider-shaped error bodies.

### Public Health Or Status

`37` route signals are health/status/metrics-like. Representative families:

- `health`
- `metrics`
- `rag_health`
- `audio/audio_health`
- module health routes such as notes, prompts, sandbox, slides, ACP, chatbooks, meetings

Migration rule: distinguish genuinely public health probes from admin metrics or cache-control routes. For example, reset/clear operations are not public just because they live near metrics or health routes.

### Test Support

`test_support/admin_e2e` has `4` test-support route signals.

Migration rule: keep these out of production auth cleanup unless the test-support router is enabled in production builds. If touched, verify the E2E bootstrap flow explicitly.

## First Phase 3.4 Migration Order

1. `skills`: identity-only, already mapped.
2. A second identity-only family with no raw-user, manual-admin, setup, webhook, or provider-compatible signals.
3. Explicit route-admin-only families with focused denial tests.
4. Permission-only families.
5. Route guard plus manual/service admin families.
6. Manual/service-admin-only families after include-stack and service-call maps are complete.
7. Setup, webhook/callback, provider-compatible, and health/status categories only through dedicated route-specific plans.

## Tracker Impact

This closes the scan-level special-route and admin-check triage for Phase 3.4. It does not close endpoint-by-endpoint migration proof. Each implementation slice still needs:

- exact dependency include-stack mapping
- denial-path tests for unauthorized and forbidden users
- confirmation of service-layer defense-in-depth behavior
- Bandit on touched Python paths
