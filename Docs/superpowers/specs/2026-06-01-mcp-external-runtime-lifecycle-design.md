# MCP External Runtime Lifecycle Design

## Context

The standalone gateway can now build a `GatewayExternalRuntimeManager` from config and can start, stop, refresh, reconcile, install, and update external server runtimes through explicit API calls. The remaining integration gap is app lifecycle behavior: configured `auto_start` servers are not reconciled when a gateway starts, and active transports are not stopped when the app shuts down.

## Goals

- Keep existing safe defaults: creating a gateway app must not start external servers unless lifecycle startup is explicitly enabled.
- Add opt-in startup reconciliation that delegates to `GatewayExternalRuntimeManager.reconcile()`, preserving its existing `auto_start=True` semantics and per-server error isolation.
- Add opt-in shutdown cleanup that stops active external transports cleanly.
- Make config bootstrap able to carry lifecycle preferences through `GatewayProfileBootstrap` so config-driven standalone callers do not need ad hoc app wiring.
- Keep errors observable without blocking app startup or shutdown for expected runtime failures.

## Non-Goals

- Adding background retry loops or periodic reconciliation.
- Starting servers whose stored definitions have `auto_start=False`.
- Adding websocket upstream transport support.
- Changing external registry or runtime route contracts.

## Design

Add a small package-owned lifecycle config:

- `GatewayExternalRuntimeLifecycleConfig(reconcile_on_startup=False, stop_on_shutdown=False)`

Extend `GatewayExternalRuntimeBootstrapConfig` with the same two booleans. The config default remains fully inert. When config bootstrap returns a `GatewayProfileBootstrap`, it carries a `GatewayExternalRuntimeLifecycleConfig` alongside the optional runtime manager. `create_gateway_app()` can also accept an explicit lifecycle config, which takes precedence over the bootstrap-carried config.

Startup behavior:

- If no runtime manager is resolved, lifecycle flags are ignored unless explicitly enabled, in which case app creation raises a deterministic `ValueError`.
- If `reconcile_on_startup=True`, the FastAPI lifespan calls `manager.reconcile()` on startup and stores the payload on `app.state.external_runtime_startup`.
- `reconcile()` already starts only enabled servers with `auto_start=True` and records per-server failures in `errors`, so app startup should not fail for individual upstream process failures.
- Unexpected lifecycle exceptions are caught and stored in a compact error payload without logging secrets.

Shutdown behavior:

- Add `GatewayExternalRuntimeManager.stop_all()` to snapshot active server ids, call `stop_server()` for each active server, and return a deterministic summary with `errors` by server id.
- If `stop_on_shutdown=True`, the FastAPI lifespan calls `manager.stop_all()` and stores the payload on `app.state.external_runtime_shutdown`.
- Transport close failures remain best-effort through the existing `stop_server()` cleanup path.

## Testing

- FastAPI app defaults do not call `reconcile()` or `stop_all()`.
- Opt-in startup lifecycle calls `reconcile()` and records the startup payload.
- Startup lifecycle with a failed reconcile payload still serves requests.
- Opt-in shutdown lifecycle calls `stop_all()` and records the shutdown payload.
- Config bootstrap carries lifecycle preferences from config to app creation.
- Runtime manager `stop_all()` stops active transports and clears runtime state.
