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

## Stage 5 Findings Index Normalization

```text
Source reports normalized:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/db-migrations-data-durability.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/webui-extension-api-contracts.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/chat-rag-llm.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/jobs-scheduler-workflows.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/integrations-providers.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/mcp-sandbox-agent-protocol.md

Normalization result:
- findings-index.json retained the existing object root, audit metadata, schema metadata, and date-prefixed ID format.
- 26 raw candidates were normalized into 26 canonical findings.
- Canonical ID ranges used: AUDIT-2026-06-27-AUTH-001..003, DB-001..002, WEBUI-001..002, OPS-001..006, MEDIA-001..004, CHAT-001..002, JOBS-001..002, INTEGRATIONS-001..003, MCP-001..002.
- Each finding includes its original CANDIDATE-* ID in evidence, a source report, owner domain, affected paths, recommendation, status, and validation status.
- Duplicate/overlap review compared candidate titles, affected paths, and recommendations across all nine reports. No candidates were merged; no duplicate titles were found.
- Stable mapping correction: integrations-providers.md now maps to AUDIT-2026-06-27-INTEGRATIONS-NNN, matching the shared index format.

Verification commands run during normalization:
- jq empty Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json
- jq '.findings | length' Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json
- jq -r '.findings[].id' Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json | sort | uniq -d
- jq -e '.schema.allowed_values as $a | all(.findings[]; ((.severity as $v | $a.severity | index($v)) and (.confidence as $v | $a.confidence | index($v)) and (.category as $v | $a.category | index($v)) and (.evidence_tier as $v | $a.evidence_tier | index($v)) and (.evidence_strength as $v | $a.evidence_strength | index($v)) and (.status as $v | $a.status | index($v)) and (.validation_status as $v | $a.validation_status | index($v))))' Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json
- jq -r '.schema.finding_required_fields as $req | .findings[] | .id as $id | ($req - (keys_unsorted))[]? | "missing \($id) \(.)"' Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json
- jq -r '.findings[] | select((.recommendation|length)==0 or (.affected_paths|length)==0 or (.evidence|length)==0) | .id' Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json
- rg -o "CANDIDATE-[A-Za-z0-9_-]+-[0-9]{3}" Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json | sort -u | wc -l | tr -d ' '
- jq -r '.findings[].title' Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json | sort | uniq -d
- placeholder-token scan against Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json
- git diff --check -- Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json Docs/superpowers/reviews/2026-06-27-repo-audit/domains/integrations-providers.md Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/command-log.md "backlog/tasks/task-12050 - Execute-comprehensive-repository-audit.md"
- final-summary marker count check on backlog/tasks/task-12050 - Execute-comprehensive-repository-audit.md
- git status --short

Verification result:
- JSON parse passed.
- Finding count is 26.
- Duplicate ID check returned no output.
- Allowed-value schema check returned true.
- Required-field check returned no output.
- Empty recommendation/evidence/affected path check returned no output.
- Unique original candidate ID count is 26.
- Duplicate title check returned no output.
- Placeholder scan returned no matches.
- Diff whitespace check passed.
- Final-summary markers remain exactly one begin marker and one end marker.
- Git status showed only intended audit files changed plus the two known unrelated untracked watchlist templates.
```
