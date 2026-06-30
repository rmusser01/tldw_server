# RateLimiting

`RateLimiting` is a compatibility package for rate-limit documentation and
imports. Active ingress limiting is implemented through Resource Governance
middleware and AuthNZ API dependencies rather than files in this package.

## Start Here

- Active middleware: `app/core/Resource_Governance/middleware_simple.py`.
- Policy loading and route maps: `app/core/Resource_Governance/policy_loader.py`.
- Endpoint dependencies: `app/api/v1/API_Deps/auth_deps.py` provides
  `rbac_rate_limit(...)` and `TokenScopeGuard`.
- Tests: `tests/Resource_Governance/`, plus rate-limit-specific endpoint tests
  in Watchlists, Evaluations, Usage, and API-dependency suites.

## Responsibilities

- Keep rate-limit contributors pointed at the current implementation.
- Document that route-level request limiting belongs to Resource Governance.
- Document that resource-scoped RBAC limiter dependencies are declared in
  AuthNZ API deps.

## Module Map

- `__init__.py` is intentionally empty.

## How It Connects

- Endpoints declare `Depends(rbac_rate_limit("<resource>"))` for privilege and
  resource metadata.
- `TokenScopeGuard` enforces scoped virtual-key behavior and usage counting hints.
- Resource Governance middleware emits allow/deny decisions and 429 responses
  from configured policies.

## Extension Points

- Add new route limits by changing Resource Governance policies and route maps.
- Add scoped endpoint metadata through AuthNZ dependency helpers.
- Avoid adding a second limiter implementation here; it would split policy and
  observability.

## Testing

- Resource Governance coverage: `tests/Resource_Governance/`.
- Endpoint rate-limit examples: `tests/Watchlists/test_rate_limit_headers_real.py`,
  `tests/Watchlists/test_rate_limit_headers_strict.py`, and
  `tests/Evaluations/test_evaluations_unified.py`.
- Usage interactions: `tests/Usage/test_audio_rg_minutes_and_heartbeat.py`.

## Gotchas

- `RG_ENABLED` is commonly disabled in tests for determinism; set it explicitly
  when validating live 429 behavior.
- RBAC rate-limit dependency metadata and Resource Governance enforcement are
  related but not the same control plane.
