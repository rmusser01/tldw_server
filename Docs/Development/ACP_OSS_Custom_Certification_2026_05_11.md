# ACP OSS and Custom Profile Certification Evidence - 2026-05-11

GitHub issue: #1563
Repo commit: `6884e46c1`
Branch: `codex/acp-1563-oss-custom-live-certification`
Host/runtime: macOS 15.6 (`24G84`), host stdio

This record documents the May 11, 2026 certification attempt for the seeded
OSS/custom ACP registry profiles. No profile is upgraded beyond
`documented_unverified` because this host does not have runnable downstream ACP
commands for these rows, and the custom profile has no concrete command to test.

The live E2E harness also refused to run without `TLDW_E2E_SERVER_URL`,
`TLDW_E2E_API_KEY`, and `ACP_AGENT_PROFILE`, which is the expected safety
behavior for named live-agent support claims.

## Aider

```text
Agent: Aider
Profile key: aider
Support state: documented_unverified
Verification level: documented_only
Commit/branch: 6884e46c1 / codex/acp-1563-oss-custom-live-certification
Host/runtime: macOS 15.6 (24G84), host stdio
Agent binary/version: unavailable; `aider` not found on PATH
Config profile: tldw_Server_API/Config_Files/agents.yaml command=aider
Commands run: whence -va aider; brew list --versions aider goose opencode; live-e2e safety refusal
Capability results: init=skip, session_new=skip, prompt=skip, structured_completion=skip, artifacts=skip, diagnostics=skip, cancel_close=skip, review_loop=skip, workspace_env=skip, mcp_injection=skip, sandbox=skip, redacted_support_view=skip
Caveats: binary_missing, sandbox_unverified, mcp_injection_unverified, artifact_capability_unverified, review_loop_unverified, redacted_view_unverified
Follow-up issue: #1563
```

## Goose

```text
Agent: Goose
Profile key: goose
Support state: documented_unverified
Verification level: documented_only
Commit/branch: 6884e46c1 / codex/acp-1563-oss-custom-live-certification
Host/runtime: macOS 15.6 (24G84), host stdio
Agent binary/version: unavailable; `goose` not found on PATH
Config profile: tldw_Server_API/Config_Files/agents.yaml command=goose
Commands run: whence -va goose; brew list --versions aider goose opencode; live-e2e safety refusal
Capability results: init=skip, session_new=skip, prompt=skip, structured_completion=skip, artifacts=skip, diagnostics=skip, cancel_close=skip, review_loop=skip, workspace_env=skip, mcp_injection=skip, sandbox=skip, redacted_support_view=skip
Caveats: binary_missing, sandbox_unverified, mcp_injection_unverified, artifact_capability_unverified, review_loop_unverified, redacted_view_unverified
Follow-up issue: #1563
```

## Continue

```text
Agent: Continue
Profile key: continue_dev
Support state: documented_unverified
Verification level: documented_only
Commit/branch: 6884e46c1 / codex/acp-1563-oss-custom-live-certification
Host/runtime: macOS 15.6 (24G84), host stdio
Agent binary/version: unavailable; `continue` resolved to a zsh shell builtin, not an installed CLI
Config profile: tldw_Server_API/Config_Files/agents.yaml command=continue
Commands run: whence -va continue; npm list -g --depth=0 @continuedev/cli @sst/opencode @openai/codex @anthropic-ai/claude-code; live-e2e safety refusal
Capability results: init=skip, session_new=skip, prompt=skip, structured_completion=skip, artifacts=skip, diagnostics=skip, cancel_close=skip, review_loop=skip, workspace_env=skip, mcp_injection=skip, sandbox=skip, redacted_support_view=skip
Caveats: binary_missing, protocol_incompatibility, sandbox_unverified, mcp_injection_unverified, artifact_capability_unverified, review_loop_unverified, redacted_view_unverified
Follow-up issue: #1563
```

## OpenCode

```text
Agent: OpenCode
Profile key: opencode
Support state: documented_unverified
Verification level: documented_only
Commit/branch: 6884e46c1 / codex/acp-1563-oss-custom-live-certification
Host/runtime: macOS 15.6 (24G84), host stdio
Agent binary/version: unavailable; `opencode` not found on PATH
Config profile: tldw_Server_API/Config_Files/agents.yaml command=opencode args=[]
Commands run: whence -va opencode; brew list --versions aider goose opencode; npm list -g --depth=0 @continuedev/cli @sst/opencode @openai/codex @anthropic-ai/claude-code; live-e2e safety refusal
Capability results: init=skip, session_new=skip, prompt=skip, structured_completion=skip, artifacts=skip, diagnostics=skip, cancel_close=skip, review_loop=skip, workspace_env=skip, mcp_injection=skip, sandbox=skip, redacted_support_view=skip
Caveats: binary_missing, sandbox_unverified, mcp_injection_unverified, artifact_capability_unverified, review_loop_unverified, redacted_view_unverified
Follow-up issue: #1563
```

## Custom ACP Profile

```text
Agent: Custom ACP-compatible agent
Profile key: custom
Support state: documented_unverified
Verification level: documented_only
Commit/branch: 6884e46c1 / codex/acp-1563-oss-custom-live-certification
Host/runtime: macOS 15.6 (24G84), host stdio
Agent binary/version: operator supplied; no concrete command configured in default registry
Config profile: tldw_Server_API/Config_Files/agents.yaml command="" args=[]
Commands run: registry/config inspection; live-e2e safety refusal
Capability results: init=skip, session_new=skip, prompt=skip, structured_completion=skip, artifacts=skip, diagnostics=skip, cancel_close=skip, review_loop=skip, workspace_env=skip, mcp_injection=skip, sandbox=skip, redacted_support_view=skip
Caveats: workspace_config_missing, binary_missing, sandbox_unverified, mcp_injection_unverified, artifact_capability_unverified, review_loop_unverified, redacted_view_unverified
Follow-up issue: #1563
```

Custom profile support remains a documented template only. A future support
claim needs a named implementation, command, args, env requirements, workspace
policy, host/runtime, binary version, and `live-e2e` capability evidence.

## Conclusion

Keep all #1563 OSS/custom profiles at `documented_unverified` /
`documented_only`. Setup and registry surfaces may show these rows as candidate
profiles, but release notes must not claim live ACP support until real
ACP-compatible commands and passing evidence are recorded.
