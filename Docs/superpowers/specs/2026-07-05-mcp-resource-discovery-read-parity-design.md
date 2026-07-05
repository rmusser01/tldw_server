# MCP Resource Discovery and Read Parity Design

## Context

`TASK-2290` covers Claude-style MCP resource discovery, resource reads, and bounded MCP server readiness checks. The current standalone gateway already routes `resources/list` and `resources/read` to a `GatewayRuntime`, and the tldw MCP protocol already supports internal module resources. External runtime support is tool-only.

## Scope

- Keep the existing JSON-RPC gateway paths for internal resources.
- Add resource discovery/read methods to the external transport contract.
- Expose running external server resources through the existing external runtime adapter.
- Redact upstream resource identifiers from public resource metadata.
- Add a bounded readiness helper over existing runtime status rows.

## Design

- External transports return plain MCP resource dictionaries and resource read payloads.
- The runtime manager maps upstream resources to `external://{server_id}/{digest}` virtual URIs and stores the upstream URI in memory only.
- Public descriptors include `external_server_id`, `source`, and safe title/description/mime metadata, but not the raw upstream URI.
- Reads for `external://...` URIs go through the external runtime manager; all other URIs continue to delegate to the base runtime.
- Missing, stopped, or unknown external resources raise `GatewayExternalRuntimeError` with stable reason codes.
- Readiness waiting polls `list_runtime_servers()` until requested servers are healthy or the timeout expires.

## Non-Goals

- No new resource database.
- No new setup UI.
- No package install/update execution changes.
- No local-file default pack changes.
