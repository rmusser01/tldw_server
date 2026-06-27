# Comprehensive Repository Audit Command Log

Record commands whose output is used as audit evidence. Redact secrets, tokens, sensitive environment values, and sensitive local data.

## Baseline

```text
origin/dev refreshed baseline: 669092178b0ba0fa1e840a37250b0deb55acd5a3
network refreshed: yes
worktree: .worktrees/comprehensive-repo-audit-2026-06-27
branch: codex/comprehensive-repo-audit-2026-06-27
audit branch HEAD after rebase: d33aa41cd6d257e7d9cf46c63083f0f17ba82358
execution task: TASK-12050
```

## Baseline Refresh

```text
previous baseline: superseded by refreshed origin/dev baseline
refreshed origin/dev baseline: 669092178b0ba0fa1e840a37250b0deb55acd5a3
current audit branch HEAD after successful rebase: d33aa41cd6d257e7d9cf46c63083f0f17ba82358
clean status observed before refresh edits: yes
fetch: git fetch origin dev
rebase: git rebase origin/dev
result: audit branch rebased onto refreshed origin/dev with no conflicts
```

## Task 3 Starting State Commands

Observed before Task 3 inventory file generation. This HEAD is the pre-inventory task-start HEAD, not the `origin/dev` baseline SHA or the immediate post-rebase audit branch HEAD recorded above.

```text
$ git rev-parse HEAD
6099dac1d71c9adc0ac9980fa8ac305aa30f938a

$ git status --short --branch
## codex/comprehensive-repo-audit-2026-06-27...origin/dev [ahead 3]
```

## Domain Review Dispatch

```text
Batch 1 dispatched after inventory commit aacb27c4552002e5e15d18c4997a5f89fea58d9a.
Parallelism cap: 4 domain agents.
Domains: AuthNZ and Admin; DB, Migrations, and Data Durability; WebUI, Extension, and API Contracts; CI, Deployment, Operations, and Release Surfaces.

Batch 2 dispatched after domain batch 1 commit 6b2cce0a351429f2d5e46e8e738f38a4bb4fa0c4.
Parallelism cap: 4 domain agents.
Domains: Media, Ingestion, and Storage; Chat, RAG, and LLM; Jobs, Scheduler, and Workflows; Integrations and Providers.

Batch 3 dispatched after domain batch 2 commit 19e41eac0d6b73278e826475d5f923873f489607.
Parallelism cap: 4 domain agents.
Domains: MCP, Sandbox, and Agent Protocol.
```

## Domain Review Evidence: MCP, Sandbox, And Agent Protocol

```text
Inspection commands included:
- find tldw_Server_API/app/core/MCP_unified -maxdepth 3 -type f | sort
- find tldw_Server_API/app/core -maxdepth 2 \( -iname '*Sandbox*' -o -iname '*sandbox*' -o -iname '*Agent*' -o -iname '*agent*' \) -print | sort
- find tldw_Server_API/tests -type f \( -iname '*mcp*' -o -iname '*sandbox*' -o -iname '*agent*' -o -iname '*tool*' -o -iname '*security*' \) | sort
- rg --files apps/mcp-unified | sort
- rg over MCP Unified, ACP, sandbox, and related tests for websocket/auth/scope/reconnect/subscriber patterns.
- targeted nl/sed reads over the source and test files cited in domains/mcp-sandbox-agent-protocol.md.

Focused pytest command:
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_stream_rejects_read_only_api_key_in_multi_user_mode tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_ssh_rejects_read_only_api_key_in_multi_user_mode tldw_Server_API/tests/Agent_Client_Protocol/test_ws_reconnect.py tldw_Server_API/tests/sandbox/test_ws_stdin_caps.py
Result: 7 passed, 51 warnings in 11.18s.

Scoped Bandit command:
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/MCP_unified tldw_Server_API/app/core/Sandbox tldw_Server_API/app/core/Agent_Client_Protocol tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py tldw_Server_API/app/api/v1/endpoints/sandbox.py -f json -o /tmp/bandit_mcp_sandbox_agent_protocol.json
Result: exit code 1; JSON written to /tmp/bandit_mcp_sandbox_agent_protocol.json.

Bandit summary command:
jq '{metrics: .metrics._totals, issue_count: (.results|length), severities: (.results|group_by(.issue_severity)|map({severity: .[0].issue_severity, count: length})), confidences: (.results|group_by(.issue_confidence)|map({confidence: .[0].issue_confidence, count: length}))}' /tmp/bandit_mcp_sandbox_agent_protocol.json
Summary: 4418 results, 0 high severity, 17 medium severity, 4401 low severity. Medium findings were in MCP test files under tldw_Server_API/app/core/MCP_unified/tests, primarily Bandit B108 temp-path findings plus one B103 chmod test case.
```
