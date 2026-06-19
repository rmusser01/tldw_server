# MCP Unified Smoke Client

The MCP Unified smoke client runs a small JSON-RPC scenario against the
standalone MCP gateway or a compatible MCP server. It is meant for PR checks,
operator validation, and quick manual testing before adding more specialized
tool scenarios.

## Scope

The baseline scenario exercises:

- `initialize`
- `notifications/initialized`
- `ping`
- `tools/list`
- safe `tools/call`
- unknown `tools/call`
- resource list/read when advertised
- prompt list/get when advertised
- JSON-RPC batch correlation
- malformed request handling
- optional policy-denial checks when the transport fixture exposes one

The report is redacted and bounded so it can be attached to PRs, CI artifacts,
or Backlog task notes without including tool arguments, file contents, local
absolute paths, bearer tokens, API keys, or long payloads.

## Commands

Run the deterministic in-process fixture:

```bash
mcp-unified-smoke inprocess --json-report -
```

Run a live HTTP endpoint:

```bash
mcp-unified-smoke http \
  --url http://127.0.0.1:8000/mcp/request \
  --profile-id backend-engineer \
  --api-key-env TLDW_API_KEY \
  --json-report /tmp/mcp-smoke-http.json
```

Run a live WebSocket endpoint:

```bash
mcp-unified-smoke websocket \
  --url ws://127.0.0.1:8000/mcp/ws \
  --profile-id backend-engineer \
  --bearer-token-env TLDW_JWT \
  --json-report /tmp/mcp-smoke-ws.json
```

Run a stdio subprocess:

```bash
mcp-unified-smoke stdio \
  --command python \
  --arg -m \
  --arg my_mcp_server \
  --cwd /path/to/workspace \
  --env PATH \
  --env PYTHONPATH \
  --json-report /tmp/mcp-smoke-stdio.json
```

Common options may be placed before or after the transport name.

## Auth And Profiles

`--profile-id` selects a profile when the transport supports it. HTTP and
WebSocket transports send it as the gateway-supported `x-mcp-profile` selector.
The in-process fixture ignores profile ids.

`--api-key-env NAME` reads an API key from an environment variable and sends it
as `X-API-KEY` for live HTTP/WebSocket transports. `--bearer-token-env NAME`
reads a bearer token and sends it as `Authorization: Bearer ...`. The CLI does
not print these values; report rendering redacts sensitive fields and matching
environment secrets.

For stdio subprocesses, environment inheritance is deny-by-default. Use
`--env NAME` for each variable the subprocess may inherit. If `--command` is a
command name rather than an absolute executable path, include `--env PATH` so it
can be resolved consistently. The CLI uses argv execution and does not invoke a
shell.

## Scenario Tuning

Use strict mode when missing advertised capabilities should fail the run:

```bash
mcp-unified-smoke inprocess --mode strict --json-report -
```

Use safe target overrides when the server does not expose the fixture defaults:

```bash
mcp-unified-smoke http \
  --url http://127.0.0.1:8000/mcp/request \
  --safe-tool-name search.docs \
  --safe-tool-arguments-json '{"query":"smoke"}' \
  --safe-resource-uri resource://docs/index \
  --safe-prompt-name docs.review \
  --safe-prompt-arguments-json '{"topic":"smoke"}'
```

`--timeout SECONDS` controls live transport startup and request timeouts. The
CLI does not retry by default. Treat non-idempotent `tools/call` results as
single-attempt operations unless a future wrapper explicitly scopes retries to
idempotent methods.

## Reports

Use `--json-report -` for stdout or pass a file path. Without `--json-report`,
the CLI prints a compact human-readable summary.

Report fields include:

- top-level `ok`, `transport`, timing, and metadata
- one entry per scenario step
- step result summaries rather than raw tool/resource/prompt payloads
- error reason codes and redacted diagnostics
- optional trace fields when future transports add debug trace details

## Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Scenario passed. |
| `1` | Scenario ran but a required step failed. |
| `2` | Invalid CLI arguments or invalid JSON option payload. |
| `3` | Transport startup/connect failure before a scenario report could be produced. |
| `4` | Strict mode failed because a required capability was unavailable. |

## CI Use

Recommended CI pattern:

```bash
mcp-unified-smoke http \
  --url "$MCP_SMOKE_URL" \
  --profile-id "$MCP_SMOKE_PROFILE" \
  --api-key-env MCP_SMOKE_API_KEY \
  --mode strict \
  --json-report "$RUNNER_TEMP/mcp-smoke.json"
```

Upload the JSON report as an artifact even on failure. Keep secrets in
environment variables and pass only variable names to the CLI.
