# RateLimiting

## 1. Descriptive of Current Feature Set

- Purpose: Centralized ingress limiting via Resource Governor (RG) policies to protect APIs and background operations.
- Capabilities:
  - RG middleware enforces request caps using route_map (path + tag matching).
  - Ingress rate limiting is enforced exclusively by RG (no legacy ingress decorators).
  - RBAC rate-limit selector (logs strictest user/role limits for a resource; enforcement path stubbed).
  - Token scope dependency (`TokenScopeGuard`) with usage counting hints (`count_as="call"|"run"`).
- Inputs/Outputs:
  - Input: HTTP requests (and contextual user/role data).
  - Output: Allow or HTTP 429 with `Retry-After` header (where applicable).
- Related Endpoints (examples):
  - Audio TTS/STT routes — tldw_Server_API/app/api/v1/endpoints/audio.py:254, 463, 920, 973, 1958, 2143
  - Media ingestion — tldw_Server_API/app/api/v1/endpoints/media.py:2120, 2276; RBAC limiter on create — tldw_Server_API/app/api/v1/endpoints/media.py:4473, 8460
  - RAG search — tldw_Server_API/app/api/v1/endpoints/rag_unified.py:697
  - Chat completions — tldw_Server_API/app/api/v1/endpoints/chat.py:615
  - Embeddings — tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:1630
  - Notes — tldw_Server_API/app/api/v1/endpoints/notes.py:174, 289, 351, 419, 482, 523, 567, 655, 736, 820, 905, 939, 977, 1009
  - Chatbooks — tldw_Server_API/app/api/v1/endpoints/chatbooks.py:120, 304, 514, 897

## 2. Technical Details of Features

- Architecture & Data Flow
  - RG middleware + route_map: tldw_Server_API/app/core/Resource_Governance/middleware_simple.py
  - RBAC selector: `rbac_rate_limit(resource)` returns a dependency that logs selected limits from `rbac_user_rate_limits` and `rbac_role_rate_limits` but does not enforce yet: tldw_Server_API/app/api/v1/API_Deps/auth_deps.py:994
  - Token-scope enforcement: `TokenScopeGuard(scope, ..., endpoint_id=..., count_as=...)` injects scoped virtual-key checks and emits usage hints: tldw_Server_API/app/api/v1/API_Deps/auth_deps.py:1040

- Configuration
  - Testing bypass: `RG_ENABLED` defaults off in tests; set `RG_ENABLED=1` to enforce in CI.
  - Per-route rates are declared in RG policies; RBAC selection reads limits from DB tables.

- Concurrency & Performance
- RG handles request limiting; for multi-instance deployments, prefer the Redis backend.

- Error Handling
- When exceeded, HTTP 429 responses with `Retry-After` are emitted by RG middleware.

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure
  - `app/api/v1/API_Deps/auth_deps.py` — `rbac_rate_limit` and `TokenScopeGuard` dependencies.
- Extension Points
- Use RG policy entries + route_map for new endpoints. For resource-scoped behavior, add `Depends(rbac_rate_limit("<resource>"))`.
  - To enable true RBAC enforcement, extend `enforce_rbac_rate_limit` to check counters and raise 429 based on selected limits.
- Tests
  - Evaluations limiting shape and status: tldw_Server_API/tests/Evaluations/test_evaluations_unified.py:711, 733–736
  - Watchlists optional rate limit headers path: tldw_Server_API/tests/Watchlists/test_rate_limit_headers_optional.py:39
- Local Dev Tips
  - Set `RG_ENABLED=1` to enforce RG limits locally; leave unset for permissive dev runs.
  - Use small per-route limits to validate 429 behavior in dev.
