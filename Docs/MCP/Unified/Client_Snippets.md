# MCP Unified - Client Snippets

Quick, copy-paste examples to authenticate, initialize, discover tools, and call tools against the MCP Unified API.

For repeat operator workflows including batch calls, session headers, status, metrics, and failure recovery, see `Operator_Cheatsheet.md`.

## Prerequisites
- Server running with MCP Unified endpoints mounted at `/api/v1/mcp`
- Auth token (preferred) or API key

Primary examples use header or WebSocket subprotocol auth. Query-string tokens are disabled by default and should only be used for explicit legacy/debug workflows.

## JSON-RPC over HTTP (Initialize → Tools List → Tools Call)

```bash
# Status preflight - confirms the embedded TLDW MCP surface is mounted
curl -s -H "Authorization: Bearer <token>" \
  http://127.0.0.1:8000/api/v1/mcp/status

# Initialize (optional) - negotiates an mcp-session-id and can carry a base64 safe config
cfg=$(printf '{"snippet_length": 200}' | base64)
curl -i -H "Authorization: Bearer <token>" \
  "http://127.0.0.1:8000/api/v1/mcp/request?config=$cfg" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"clientInfo":{"name":"demo"}},"id":1}'

# List tools (RBAC-filtered; add &catalog=... or &catalog_id=... if desired)
curl -H "Authorization: Bearer <token>" \
  "http://127.0.0.1:8000/api/v1/mcp/tools"

# Call a tool (HTTP convenience endpoint)
curl -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"media.search","arguments":{"query":"test","limit":5}}' \
  http://127.0.0.1:8000/api/v1/mcp/tools/execute
```

## WebSocket (JavaScript) - Subprotocol Auth

```javascript
const token = "<jwt token>";
// Sends Sec-WebSocket-Protocol: bearer,<token>
const ws = new WebSocket("ws://127.0.0.1:8000/api/v1/mcp/ws?client_id=demo", ["bearer", token]);

ws.onopen = () => {
  ws.send(JSON.stringify({
    jsonrpc: "2.0",
    method: "initialize",
    params: { clientInfo: { name: "demo", version: "1.0.0" } },
    id: 1,
  }));

  // List tools via JSON-RPC
  ws.send(JSON.stringify({ jsonrpc: "2.0", method: "tools/list", params: {}, id: 2 }));

  // Call a tool
  ws.send(JSON.stringify({
    jsonrpc: "2.0",
    method: "tools/call",
    params: { name: "media.search", arguments: { query: "hello", limit: 3 } },
    id: 3,
  }));
};

ws.onmessage = (ev) => console.log("MCP:", JSON.parse(ev.data));
```

## Python (HTTP JSON-RPC helper)

```python
import requests

BASE = "http://127.0.0.1:8000/api/v1/mcp"
HEADERS = {"Authorization": "Bearer <token>", "Content-Type": "application/json"}

def rpc(method, params=None, id=1):
    payload = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": id}
    r = requests.post(f"{BASE}/request", json=payload, headers=HEADERS)
    r.raise_for_status()
    return r.json()["result"]

# Initialize
rpc("initialize", {"clientInfo": {"name": "py-client"}}, id=1)

# Tools list (optionally: params={"catalog": "research"})
tools = rpc("tools/list", {}, id=2)["tools"]
print("tools:", [t["name"] for t in tools])

# Execute tool via convenience endpoint
exec_resp = requests.post(
    f"{BASE}/tools/execute",
    headers=HEADERS,
    json={"tool_name": "media.search", "arguments": {"query": "ai", "limit": 2}},
)
print(exec_resp.json())
```

## Notes
- Tool discovery can be narrowed with catalogs: `GET /api/v1/mcp/tools?catalog=<name>` or `?catalog_id=<id>`.
- Unresolved catalogs return an empty result with `_meta.catalog.status=unresolved`; use `/api/v1/mcp/tool_catalogs` to list visible catalog names.
- Results include `canExecute` for each tool; catalog membership doesn’t grant execute permissions.
- Prefer WS headers/subprotocol for auth; query-param tokens are disabled by default.
