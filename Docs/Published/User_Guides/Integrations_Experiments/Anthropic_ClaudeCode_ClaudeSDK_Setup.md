# Anthropic Setup for Claude Code and Claude SDK

This guide is the Anthropic equivalent of first-time provider setup for `tldw_server`, with focus on Claude Code and Claude SDK workflows.

## Support Status (As of June 3, 2026)

Currently supported:

- Anthropic via API key (`ANTHROPIC_API_KEY`) for server-side model calls
- Anthropic BYOK API-key storage in multi-user mode

Documented but not live-certified:

- Claude Code through ACP is supported with caveats through the pinned external `@agentclientprotocol/claude-agent-acp@0.40.0` adapter on the verified macOS host runner profile. Sandbox, artifact-producing workflows, non-empty MCP injection, reviewer-loop behavior, and other host profiles remain unverified.

Not yet supported:

- Anthropic OAuth BYOK account-linking endpoints like OpenAI OAuth (`/users/keys/<provider>/oauth/*`)

If you need OAuth account linking for Anthropic, treat it as a planned feature rather than a currently available setup path.

## Path A: Anthropic in BYOK (Multi-User)

Use this when users should provide their own Anthropic API credentials to `tldw_server`.

### Operator Prerequisites

```bash
AUTH_MODE=multi_user
BYOK_ENABLED=true
BYOK_ENCRYPTION_KEY=<base64-encoded-32-byte-key>
BYOK_ALLOWED_PROVIDERS=anthropic,openai
```

### User First-Time Setup (API)

In multi-user mode, include a valid JWT in `Authorization: Bearer <JWT>` for these endpoints or requests will return `401 Unauthorized`.

Store key:

```http
POST /api/v1/users/keys
Authorization: Bearer <JWT>
Content-Type: application/json

{
  "provider": "anthropic",
  "api_key": "sk-ant-..."
}
```

Verify status:

```http
GET /api/v1/users/keys
Authorization: Bearer <JWT>
```

Optional credential test:

```http
POST /api/v1/users/keys/test
Authorization: Bearer <JWT>
Content-Type: application/json

{
  "provider": "anthropic"
}
```

Expected behavior:

- Key is encrypted at rest.
- Responses return `key_hint`, never plaintext key.
- Runtime resolution still follows BYOK order (user -> team -> org -> server default).

## Path B: Claude Code via ACP

Use this when users operate Claude Code sessions through the ACP UI.

### Prerequisites

- ACP routes enabled in `config.txt` (`enable = tools, jobs, acp`)
- `tldw-agent` configured
- Claude Code installed and executable as `claude`
- pinned `@agentclientprotocol/claude-agent-acp@0.40.0` installed and executable as `claude-agent-acp`
- Claude Code or adapter provider/login state configured in the runner environment

`~/.tldw-agent/config.yaml` example:

```yaml
agents:
  default: claude_code
  agents:
    - type: claude_code
      name: Claude Code
      command: claude
      args: []
      env:
        - TERM=xterm-256color
        - HOME=${TLDW_ACP_HOST_HOME}
      entrypoint_strategy: external_acp_adapter
      acp_command: claude-agent-acp
      acp_args: []
      adapter_source: agentclientprotocol/claude-agent-acp
      adapter_docs_url: https://github.com/agentclientprotocol/claude-agent-acp
      adapter_package: "@agentclientprotocol/claude-agent-acp"
      adapter_version: "0.40.0"
      credential_policy: delegated_to_adapter
      runtime_backend: acp_downstream
```

Validate:

```bash
curl -s http://127.0.0.1:8000/api/v1/acp/agents -H "X-API-KEY: <api-key>"
```

You should see `claude_code` listed with `supported_with_caveats` support on the verified macOS host profile. Runtime status can still block on missing `claude`, missing `claude-agent-acp`, or missing/failed Claude Code or adapter auth in the same environment used by `tldw_server` and the runner.

## Path C: Claude SDK in Custom Agent Mode

If your team uses Claude SDK directly, integrate it as a custom ACP agent process:

```yaml
agent:
  command: "/path/to/your-claude-sdk-wrapper"
  args: ["--stdio"]
  env:
    ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
```

Use this when Claude Code is not your desired runtime and you want SDK-managed behavior in your own wrapper.

## Troubleshooting

### Claude agent shows unconfigured

- Check Claude Code or adapter provider/login state in the same environment used by `tldw_server`/runner.
- Confirm both commands work directly: `claude --version` and `claude-agent-acp --help`.

### `403 Provider not allowed for BYOK`

- Add `anthropic` to `BYOK_ALLOWED_PROVIDERS` and restart.

### `503 missing_provider_credentials`

- No Anthropic key resolved from user/team/org/server-default sources.

### Expecting Anthropic OAuth flow in BYOK UI

- Anthropic OAuth flow is not implemented yet in current `tldw_server`.
- Use API-key BYOK for now.

## Related Docs

- `Docs/User_Guides/Server/BYOK_User_Guide.md`
- `Docs/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md`
- `Docs/User_Guides/Server/OpenAI_OAuth_First_Time_Setup.md`
