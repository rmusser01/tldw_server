# MCP Hub Management

MCP Hub is the shared management surface used by the WebUI and browser extension for MCP-related configuration.

## UI Routes

- `/mcp-hub`
- `/settings/mcp-hub`

Both routes render the same MCP Hub page and tabs.

## Scope

MCP Hub currently covers:

- ACP profile management
- External MCP server registry management
- Secret write/update for external servers (write-only reads)
- Tool catalog management via existing catalog endpoints (see `Docs/MCP/mcp_tool_catalogs.md`)

## Auth and Permissions

- All MCP Hub endpoints require an authenticated principal.
- Read/list endpoints are available to authenticated users.
- Mutation endpoints require admin role, `system.configure`, or wildcard `*` permission.

## API Endpoints

### ACP Profiles

- `GET /api/v1/mcp/hub/acp-profiles` - list profiles
- `POST /api/v1/mcp/hub/acp-profiles` - create profile
- `PUT /api/v1/mcp/hub/acp-profiles/{profile_id}` - update profile
- `DELETE /api/v1/mcp/hub/acp-profiles/{profile_id}` - delete profile

### External Servers

- `GET /api/v1/mcp/hub/external-servers` - list external servers
- `POST /api/v1/mcp/hub/external-servers` - create server
- `PUT /api/v1/mcp/hub/external-servers/{server_id}` - update server
- `DELETE /api/v1/mcp/hub/external-servers/{server_id}` - delete server
- `POST /api/v1/mcp/hub/external-servers/{server_id}/secret` - set or rotate secret

## Local Toy MCP Walkthrough

Use this walkthrough when validating the WebUI or extension MCP Hub screens end to end. It creates a temporary no-auth stdio server that exposes one read-only diagnostic tool named `toy.echo`.

### Isolate Local Runtime State

For a disposable local run, start the API from a shell with explicit temporary paths:

```bash
export TLDW_MCP_WALKTHROUGH_ROOT="$(mktemp -d)"
export AUTH_MODE=single_user
export SINGLE_USER_API_KEY="$(openssl rand -hex 32)"
export USER_DB_BASE_DIR="$TLDW_MCP_WALKTHROUGH_ROOT/user_databases"
export DATABASE_URL="sqlite:///$TLDW_MCP_WALKTHROUGH_ROOT/authnz.db"
export MCP_DATABASE_URL="sqlite+aiosqlite:///$TLDW_MCP_WALKTHROUGH_ROOT/mcp_unified.db"
```

Generate a unique `SINGLE_USER_API_KEY` for each walkthrough run, and use that value in the WebUI or extension test session. If OpenSSL is unavailable, set the variable to another locally generated high-entropy value instead of reusing a committed example key.

`DATABASE_URL` only controls the AuthNZ database. MCP runtime metadata uses `MCP_DATABASE_URL`, while media, notes, prompts, Prompt Studio, vector-store metadata, Chroma storage, Chatbooks, and most per-user evaluation paths derive from `USER_DB_BASE_DIR`. Some legacy or test-only evaluation paths still have dedicated overrides such as `EVALUATIONS_TEST_DB_PATH`; set those separately if the same process will exercise those modules and you need a fully disposable run.

### Create The Toy Server

Create the stdio server script in the same temporary root. The API process must be able to read this absolute file path.

```bash
cat > "$TLDW_MCP_WALKTHROUGH_ROOT/toy-mcp-server.mjs" <<'EOF'
import readline from "node:readline";

const rl = readline.createInterface({ input: process.stdin });

function send(payload) {
  process.stdout.write(JSON.stringify(payload) + "\n");
}

rl.on("line", (line) => {
  if (!line.trim()) return;

  let message;
  try {
    message = JSON.parse(line);
  } catch {
    return;
  }

  const id = message.id;
  const method = message.method;
  const params = message.params || {};

  if (method === "initialize") {
    send({
      jsonrpc: "2.0",
      id,
      result: {
        protocolVersion: "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "toy-mcp", version: "1.0.0" },
      },
    });
    return;
  }

  if (method === "notifications/initialized") {
    return;
  }

  if (method === "tools/list") {
    send({
      jsonrpc: "2.0",
      id,
      result: {
        tools: [
          {
            name: "toy.echo",
            description: "Echoes a short message for MCP Hub validation.",
            inputSchema: {
              type: "object",
              properties: { text: { type: "string" } },
            },
            metadata: { category: "diagnostic", readOnlyHint: true },
          },
        ],
      },
    });
    return;
  }

  if (method === "tools/call") {
    send({
      jsonrpc: "2.0",
      id,
      result: {
        content: [
          { type: "text", text: String((params.arguments || {}).text || "toy-ok") },
        ],
        isError: false,
      },
    });
    return;
  }

  send({
    jsonrpc: "2.0",
    id,
    error: { code: -32601, message: `unknown method: ${method}` },
  });
});
EOF
```

### Add It In MCP Hub

1. Open `/mcp-hub`.
2. In Setup, open Servers & Credentials.
3. Choose New Managed Server.
4. Use:
   - Server ID: `toy-stdio`
   - Name: `Toy Stdio`
   - Transport: `stdio`
   - Config JSON:

```json
{
  "stdio": {
    "command": "node",
    "args": ["/absolute/path/to/toy-mcp-server.mjs"]
  },
  "auth": { "mode": "none" },
  "policy": {
    "allow_tool_patterns": ["toy.*"],
    "allow_writes": false
  },
  "timeouts": {
    "connect_seconds": 2,
    "request_seconds": 5
  }
}
```

Replace `/absolute/path/to/toy-mcp-server.mjs` with `$TLDW_MCP_WALKTHROUGH_ROOT/toy-mcp-server.mjs`.

After saving, the server should read as `No credentials required`; it should not ask for a credential slot, auth template, or legacy secret.

### Validate Discovery And Chat

1. In Setup, open Tool Catalog.
2. Choose Refresh discovery.
3. Confirm `toy.echo` appears in the tool catalog.
4. Open the chat surface, enable tool use for the message if the current model/tool controls expose a Tool Choice setting, and ask for a tool call such as: `Use toy.echo with text "mcp hub smoke test".`
5. If `toy.echo` is listed in Tool Catalog but not available in chat, check profile assignments, disabled tool preferences, and the deployment diagnostics panel in Setup. That state usually means discovery succeeded but the current chat context cannot execute the tool.

## Secret Handling

- Secrets are encrypted before persistence.
- Secret plaintext is not returned by read/list endpoints.
- API responses expose only metadata (`secret_configured`, optional `key_hint`, timestamps).

## Audit Events

MCP Hub mutation flows emit audit events, including:

- `mcp_hub.acp_profile.create|update|delete`
- `mcp_hub.external_server.create|update|delete`
- `mcp_hub.external_secret.update`
