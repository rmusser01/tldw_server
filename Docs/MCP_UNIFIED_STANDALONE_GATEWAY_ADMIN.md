# MCP Unified Standalone Gateway Admin

This guide covers the standalone gateway admin/config surface for profiles,
external server definitions, credential grants, config snapshots, and remote
runtime lifecycle commands.

## Package Release Status

The `mcp_unified` package boundary is currently internal/experimental and is
not a separately published standalone package. It is still distributed as part
of the broader `tldw-server` source tree while the package release gate is
being hardened.

The current package metadata uses the canonical repository license expression
`GPL-3.0-only`. Downstream projects should treat the boundary as an in-repo
integration surface until a later packaging pass adds a clean minimal install,
independent extras verification, and release CI for the standalone package.

Inspect the current release-readiness metadata with:

```bash
mcp-unified-gateway package-info
```

The dependency groups in that payload intentionally use a `names-only` policy.
They identify the minimal standalone-package surface without duplicating version
floors from `pyproject.toml` or future package build metadata.

## Local Store Commands vs Remote Runtime Commands

The `mcp-unified-gateway` CLI has two different operating modes:

- Local store commands read or mutate the configured gateway store. These
  commands use `--config` or `MCP_UNIFIED_GATEWAY_CONFIG`.
- Remote runtime commands call admin endpoints on an already-running gateway.
  These commands use `--gateway-url` or `MCP_UNIFIED_GATEWAY_URL`.

Local store commands do not start upstream MCP processes. Remote runtime
commands also do not own durable upstream processes from the short-lived CLI;
they ask the running gateway to list, start, stop, restart, refresh, reconcile,
install, or update managed external servers.

## Admin Auth

Management routes can be protected by an admin header. The remote runtime CLI
reads the secret value from the environment so it does not appear in process
arguments:

```bash
export MCP_UNIFIED_GATEWAY_ADMIN_KEY="replace-with-admin-key"
mcp-unified-gateway runtime-list \
  --gateway-url http://127.0.0.1:8000/mcp
```

The default header name is `X-MCP-Gateway-Admin-Key`. Override only the header
name on the command line when the running gateway uses a different header:

```bash
mcp-unified-gateway runtime-list \
  --gateway-url http://127.0.0.1:8000/mcp \
  --admin-header-name X-Admin-Key
```

Do not pass secret values as CLI arguments. The runtime commands intentionally
avoid an `--admin-key` option.

## Gateway URL Semantics

`--gateway-url` is the mounted gateway base path, not only the server origin.
If the router is mounted at `/mcp`, pass `http://127.0.0.1:8000/mcp`. If it is
mounted at the root, pass `http://127.0.0.1:8000`.

The client trims trailing slashes and appends endpoint paths. It does not
auto-add `/mcp`, because gateway prefixes are host-configurable.

## Local Configuration Examples

Validate a JSON or TOML gateway config:

```bash
mcp-unified-gateway validate-config ./gateway.json
```

List bundled profile presets:

```bash
mcp-unified-gateway list-presets
```

Duplicate a preset into a persistent store:

```bash
mcp-unified-gateway duplicate-preset project-researcher \
  --profile-id researcher \
  --name "Project Researcher" \
  --config ./gateway.json
```

Create an external server definition:

```json
{
  "id": "search",
  "name": "Search MCP Server",
  "transport": "stdio",
  "command": ["python", "-m", "search_mcp_server"],
  "env_allowlist": ["PATH", "SEARCH_API_ENDPOINT"],
  "credential_slots": ["search_api_key"],
  "enabled": true,
  "auto_start": false
}
```

```bash
mcp-unified-gateway create-external-server \
  --server-file ./search-server.json \
  --config ./gateway.json
```

Credential grants are broker metadata. Store only the broker id, credential
slot, scopes, and binding metadata; do not embed the secret value:

```json
{
  "id": "researcher-search-api-key",
  "profile_id": "researcher",
  "external_server_id": "search",
  "broker_id": "env",
  "credential_slot": "search_api_key",
  "scopes": ["search:read"],
  "enabled": true
}
```

```bash
mcp-unified-gateway create-credential-grant \
  --grant-file ./researcher-search-grant.json \
  --config ./gateway.json
```

Export a config snapshot:

```bash
mcp-unified-gateway export-config \
  --output ./gateway-snapshot.json \
  --config ./gateway.json
```

Import a config snapshot with validation only:

```bash
mcp-unified-gateway import-config \
  --snapshot-file ./gateway-snapshot.json \
  --dry-run \
  --config ./gateway.json
```

Apply the snapshot after reviewing the dry-run output:

```bash
mcp-unified-gateway import-config \
  --snapshot-file ./gateway-snapshot.json \
  --config ./gateway.json
```

## Remote Runtime Examples

List runtime state:

```bash
mcp-unified-gateway runtime-list \
  --gateway-url http://127.0.0.1:8000/mcp
```

Start one managed external server:

```bash
mcp-unified-gateway runtime-start search \
  --gateway-url http://127.0.0.1:8000/mcp
```

Refresh all runtime metadata:

```bash
mcp-unified-gateway runtime-refresh \
  --gateway-url http://127.0.0.1:8000/mcp
```

Refresh one runtime:

```bash
mcp-unified-gateway runtime-refresh search \
  --gateway-url http://127.0.0.1:8000/mcp
```

Other remote runtime commands follow the same URL/auth rules:

```bash
mcp-unified-gateway runtime-stop search --gateway-url http://127.0.0.1:8000/mcp
mcp-unified-gateway runtime-restart search --gateway-url http://127.0.0.1:8000/mcp
mcp-unified-gateway runtime-reconcile --gateway-url http://127.0.0.1:8000/mcp
mcp-unified-gateway runtime-install search --gateway-url http://127.0.0.1:8000/mcp
mcp-unified-gateway runtime-update search --gateway-url http://127.0.0.1:8000/mcp
```

For repeated use, set the URL in the environment:

```bash
export MCP_UNIFIED_GATEWAY_URL=http://127.0.0.1:8000/mcp
mcp-unified-gateway runtime-list
```

## Error Payloads

The remote runtime CLI preserves public JSON error payloads returned by the
gateway, including `reason_code` values. For example, a missing external server
might produce:

```json
{
  "error": "Gateway rejected request",
  "ok": false,
  "reason_code": "external_server_not_found",
  "server_id": "search",
  "status_code": 404
}
```

Connection failures and malformed gateway responses are sanitized so response
bodies and connection error strings are not echoed back into CLI output.
