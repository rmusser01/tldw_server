# PrivilegeMaps

PrivilegeMaps inspects FastAPI route dependencies and AuthNZ privilege catalog
data to produce route-to-scope maps, summaries, cached snapshots, and trends.
It is an operations and admin-observability helper for answering which endpoints
use which RBAC scopes, roles, and rate-limit resources.

## Start Here

- Core service: `service.py`.
- Route inspection: `introspection.py`.
- Startup/cache helpers: `startup.py`, `cache.py`, `snapshots.py`, and
  `trends.py`.
- API endpoint: `app/api/v1/endpoints/privileges.py`.
- Schemas: `app/api/v1/schemas/privileges.py`.
- Tests: `tests/Privileges/`.

## Responsibilities

- Collect deterministic route registry data from a FastAPI app instance.
- Join route dependency data with the AuthNZ privilege catalog.
- Cache expensive summaries and expose snapshots/trends for admin views.
- Surface unknown or mismatched scopes so endpoint wiring can be corrected.

## Module Map

- `introspection.py` walks routes and extracts security/rate-limit dependency
  metadata.
- `service.py` builds summaries and high-level map responses.
- `cache.py` stores computed maps with TTL and route-signature invalidation.
- `snapshots.py` and `trends.py` record point-in-time summaries for comparison.
- `startup.py` wires collection into app startup when enabled.

## How It Connects

- AuthNZ owns privilege catalog definitions and role/permission data.
- `privileges.py` exposes admin endpoints and can create Jobs for heavier
  snapshot/export workflows.
- Logging helpers add request IDs to privilege operations.

## Extension Points

- Add derived views in `service.py` when an admin UI needs a new rollup.
- Keep route extraction deterministic; route-order churn makes snapshots noisy.
- Add persistent cache backends behind the existing cache shape rather than
  embedding DB calls in introspection.

## Testing

- Introspection and service behavior: `tests/Privileges/test_privilege_introspection.py`
  and `tests/Privileges/test_privilege_service_sqlite.py`.
- Endpoint and schema aliases: `tests/Privileges/test_privilege_endpoints.py` and
  `tests/Privileges/test_privilege_schema_aliases.py`.
- Cache/snapshot behavior: `tests/Privileges/test_privilege_cache.py` and
  `tests/Privileges/test_privilege_snapshot_retention.py`.

## Gotchas

- Inspecting dependencies should not execute endpoint logic or require provider
  configuration.
- High-cardinality route metadata can make snapshots noisy; normalize labels
  before adding new fields.
