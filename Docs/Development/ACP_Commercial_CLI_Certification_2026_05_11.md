# ACP Commercial CLI Certification Evidence - 2026-05-11

GitHub issue: #1564
Repo commit: `0bbdc96d4`
Branch: `codex/acp-1564-commercial-live-certification`
Host/runtime: macOS 15.6 (`24G84`), host stdio
Runner verification: `tools/tldw-agent/scripts/verify-local-build.sh` passed

This record documents the May 11, 2026 certification attempt for the seeded
commercial CLI profiles. Neither row is upgraded beyond
`documented_unverified` because the configured commands did not complete the
ACP `initialize` handshake in this environment.

## Codex CLI

```text
Agent: OpenAI Codex CLI
Profile key: codex
Support state: documented_unverified
Verification level: documented_only
Commit/branch: 0bbdc96d4 / codex/acp-1564-commercial-live-certification
Host/runtime: macOS 15.6 (24G84), host stdio
Agent binary/version: /opt/homebrew/bin/codex, codex-cli 0.128.0
Config profile: tldw_Server_API/Config_Files/agents.yaml command=codex args=[]
Manifest command: python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --format json
Commands run: codex --version; bounded stdio initialize probe for command=codex; codex mcp-server initialize probe
Capability results: init=skip, session_new=skip, prompt=skip, structured_completion=skip, artifacts=skip, diagnostics=skip, cancel_close=skip, review_loop=skip, workspace_env=skip, mcp_injection=skip, sandbox=skip, redacted_support_view=skip
Caveats: protocol_incompatibility, credentials_missing, sandbox_unverified, mcp_injection_unverified, artifact_capability_unverified, review_loop_unverified, redacted_view_unverified
Follow-up issue: #1564
```

Observed result:

- `codex --version` returned `codex-cli 0.128.0`.
- The configured `codex` command did not answer ACP `initialize`; with
  `TERM=dumb` and no TTY it exited before starting the interactive CLI.
- `codex mcp-server` answered JSON-RPC, but as MCP rather than ACP. It returned
  `method not found: initialize`, so it is not a drop-in ACP runner command.
- `OPENAI_API_KEY` was not present in the local shell, so no provider-backed
  live session could be attempted even if an ACP entrypoint were available.
- `Helper_Scripts/Testing-related/acp_certification_smoke.py --profile
  live-e2e --run` refused to run without `TLDW_E2E_SERVER_URL`,
  `TLDW_E2E_API_KEY`, and `ACP_AGENT_PROFILE`, as intended.

## Claude Code

```text
Agent: Claude Code
Profile key: claude_code
Support state: documented_unverified
Verification level: documented_only
Commit/branch: 0bbdc96d4 / codex/acp-1564-commercial-live-certification
Host/runtime: macOS 15.6 (24G84), host stdio
Agent binary/version: /Users/macbook-dev/.local/bin/claude, 2.1.119 (Claude Code)
Config profile: tldw_Server_API/Config_Files/agents.yaml command=claude args=[]
Manifest command: python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --format json
Commands run: claude --version; bounded stdio initialize probe for command=claude; claude --bare --print ping credential probe
Capability results: init=skip, session_new=skip, prompt=skip, structured_completion=skip, artifacts=skip, diagnostics=skip, cancel_close=skip, review_loop=skip, workspace_env=skip, mcp_injection=skip, sandbox=skip, redacted_support_view=skip
Caveats: credentials_missing, protocol_incompatibility, sandbox_unverified, mcp_injection_unverified, artifact_capability_unverified, review_loop_unverified, redacted_view_unverified
Follow-up issue: #1564
```

Observed result:

- `claude --version` returned `2.1.119 (Claude Code)`.
- The configured `claude` command did not answer ACP `initialize`; it exited
  with `Not logged in - Please run /login`.
- `ANTHROPIC_API_KEY` was not present in the local shell.
- No ACP-specific Claude Code stdio entrypoint was identified from
  `claude --help` during this run.
- `Helper_Scripts/Testing-related/acp_certification_smoke.py --profile
  live-e2e --run` refused to run without `TLDW_E2E_SERVER_URL`,
  `TLDW_E2E_API_KEY`, and `ACP_AGENT_PROFILE`, as intended.

## Conclusion

Keep both commercial CLI profiles at `documented_unverified` /
`documented_only`. The setup and registry surfaces may list them as configured
candidates, but release notes must not claim live ACP support until a future run
records an ACP-compatible command, credentials/login state, and passing
`live-e2e` capability evidence.
