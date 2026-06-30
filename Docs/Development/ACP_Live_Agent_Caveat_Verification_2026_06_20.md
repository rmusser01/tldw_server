# ACP Live-Agent Caveat Verification - 2026-06-20

This note records the follow-up evidence for
[GitHub #2402](https://github.com/rmusser01/tldw_server/issues/2402), under
the ACP release tracker
[#2398](https://github.com/rmusser01/tldw_server/issues/2398).

## Environment

- Branch: `codex/acp-live-agent-caveats`
- Commit under test: `ac93d96d9c`
- Backend: `uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 18003`
- Auth mode: `single_user`; API key redacted in command records
- Harness: `Helper_Scripts/Testing-related/acp_certification_smoke.py --profile workspace-live-e2e --run`
- Workspace cwd: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/acp-live-agent-caveats`
- MCP injection: harness default stdio MCP certification server, resulting in `mcp_server_count=1`
- Local provider for OpenCode: llama.cpp OpenAI-compatible server at `127.0.0.1:9099`, model `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`

## Versions

| Agent | Version evidence |
| --- | --- |
| Goose | `goose --version` -> `1.35.0` |
| Hermes | `hermes --version` -> `Hermes Agent v0.13.0 (2026.5.7)` |
| OpenCode | `opencode --version` -> `1.15.7` |

## Commands

The three runs used the same command shape with per-agent values:

```bash
TLDW_E2E_SERVER_URL=127.0.0.1:18003 \
TLDW_E2E_API_KEY=<redacted> \
ACP_AGENT_PROFILE=<goose|hermes|opencode> \
ACP_E2E_WORKSPACE_ID=<agent-specific-workspace-id> \
ACP_E2E_WORKSPACE_CWD=/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/acp-live-agent-caveats \
ACP_BACKEND_E2E_TIMEOUT_SECONDS=<240-or-300> \
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  Helper_Scripts/Testing-related/acp_certification_smoke.py \
  --profile workspace-live-e2e --run
```

## Results

| Agent | Workspace ID | Session ID | Result | Passed capabilities | Skipped or retained caveats |
| --- | --- | --- | --- | --- | --- |
| Goose | `acp-2402-goose-20260620` | `20260620_1` | `PASS workspace_live_backend_acp_e2e`; `stop_reason=end_turn`; `events_total=2`; `artifacts_total=0`; `diagnostics_total=0` | `init`, `session_new`, `prompt`, `structured_completion`, `diagnostics`, `cancel_close`, `workspace_env`, `mcp_injection`, `redacted_support_view`; `mcp_server_count=1` | `artifacts=skip`, `sandbox=skip`, `review_loop=skip`; diagnostics endpoint passed but no failure diagnostic payload was produced. |
| Hermes | `acp-2402-hermes-20260620` | `b56ece70-6f50-41ec-8404-3c898003b08e` | `PASS workspace_live_backend_acp_e2e`; `stop_reason=end_turn`; `events_total=2`; `artifacts_total=0`; `diagnostics_total=0` | `init`, `session_new`, `prompt`, `structured_completion`, `diagnostics`, `cancel_close`, `workspace_env`, `mcp_injection`, `redacted_support_view`; `mcp_server_count=1` | `artifacts=skip`, `sandbox=skip`, `review_loop=skip`; diagnostics endpoint passed but no failure diagnostic payload was produced. |
| OpenCode | `acp-2402-opencode-20260620` | `ses_119b6e209ffejcxauFOWA7H31N` | `PASS workspace_live_backend_acp_e2e`; `stop_reason=end_turn`; `events_total=2`; `artifacts_total=0`; `diagnostics_total=0` | `init`, `session_new`, `prompt`, `structured_completion`, `diagnostics`, `cancel_close`, `workspace_env`, `mcp_injection`, `redacted_support_view`; `mcp_server_count=1` | `artifacts=skip`, `sandbox=skip`, `review_loop=skip`; diagnostics endpoint passed but no failure diagnostic payload was produced. |

## Release Interpretation

- Goose, Hermes, and OpenCode may now claim live host-runner evidence for
  workspace binding and non-empty MCP server injection.
- The support state stays `supported_with_caveats`, not `supported`.
- Artifact-producing workflows remain unverified for these live agents because
  the ACP session artifact drill-through returned `artifacts_total=0`.
- Sandbox behavior remains unverified for these live agents because no sandbox
  session/run IDs were produced by the host-runner workspace-live runs.
- Reviewer-loop behavior remains unverified because the support payloads did
  not include reviewer-loop evidence.
- Failure diagnostic payloads remain unverified because the passing success
  path produced `diagnostics_total=0`; the diagnostics endpoint itself was
  reachable and redacted support surfaces remained available.
