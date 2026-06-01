# MCP Stdio Bootstrap Factory Design

## Context

Stage 4N added package-owned external runtime management and Stage 4O added the stdio upstream transport factory. The remaining gap is bootstrap wiring: standalone gateway callers can persist external server definitions, but config bootstrap does not yet create a `GatewayExternalRuntimeManager` that uses the package `create_external_transport` factory.

## Goals

- Allow standalone gateway config bootstrap to opt into external runtime management.
- Use the package-owned stdio transport factory by default when runtime management is enabled.
- Preserve caller-injected runtime manager and transport factory overrides.
- Keep safe defaults: profile config bootstrap must not start or expose external runtime management unless explicitly enabled or directly injected.
- Keep package boundaries clean: `mcp_unified` must not import `tldw_Server_API`.

## Non-Goals

- Auto-starting external servers during bootstrap.
- Adding websocket upstream runtime support.
- Adding install/update adapters beyond the existing null installer default.
- Changing FastAPI route semantics beyond consuming the bootstrap-created runtime manager that already exists on `GatewayProfileBootstrap`.

## Design

Add a nested `GatewayExternalRuntimeBootstrapConfig` with:

- `enabled: bool = False`
- `transport_factory: Literal["stdio"] = "stdio"`

`GatewayProfileBootstrapConfig` owns this nested config as `external_runtime`. The default is disabled so existing memory and SQLite profile bootstrap behavior is unchanged.

`bootstrap_profile_gateway_from_config()` will accept optional dependency overrides:

- `external_runtime_manager`
- `external_transport_factory`
- `credential_broker`
- `external_installer`

Precedence:

1. A directly supplied `external_runtime_manager` is passed through unchanged.
2. If no manager is supplied and `external_runtime.enabled` is true, bootstrap builds `GatewayExternalRuntimeManager`.
3. If runtime management is disabled, no runtime manager is built.

When config builds the manager, it requires an external registry-capable store. SQLite profile storage satisfies this through `SQLiteMCPStore`; memory storage still requires an injected registry store and is rejected by the existing external registry storage configuration path. The transport factory is caller-provided when supplied, otherwise `mcp_unified.federation.create_external_transport`.

Unsupported transport behavior stays in the factory/runtime path. Config only validates the factory selector value; a server definition with `transport="websocket"` can remain persisted, but starting it through the default package factory fails rather than pretending websocket support exists.

## Risks And Mitigations

- Risk: enabling runtime management accidentally starts processes.
  Mitigation: bootstrap only constructs the manager; starts still require explicit runtime API calls or future reconcile/autostart behavior.
- Risk: config accepts unsupported factory names silently.
  Mitigation: `GatewayExternalRuntimeBootstrapConfig` validates selector values and rejects anything except `stdio`.
- Risk: direct injection becomes less useful.
  Mitigation: injected manager takes precedence; injected factory is used only when config builds the manager.
