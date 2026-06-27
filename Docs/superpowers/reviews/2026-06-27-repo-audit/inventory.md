# Comprehensive Repository Audit Inventory

## Baseline

- Audit execution task: `TASK-12050`
- Baseline SHA: `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Audit branch HEAD after rebase: `d33aa41cd6d257e7d9cf46c63083f0f17ba82358`
- Network refreshed: yes
- Worktree: `.worktrees/comprehensive-repo-audit-2026-06-27`
- Branch: `codex/comprehensive-repo-audit-2026-06-27`
- Repeatable audit process: [repeatable-audit-process.md](repeatable-audit-process.md)

## Scope

This audit covers the nine required domain areas and five required specialist passes from `/Users/appledev/Documents/GitHub/tldw_server/Docs/superpowers/specs/2026-06-27-comprehensive-repo-audit-design.md`.

## Domain Coverage Map

| Domain | Report | Status | Notes |
| --- | --- | --- | --- |
| AuthNZ and Admin | `domains/authnz-admin.md` | Not started | |
| Media, Ingestion, and Storage | `domains/media-ingestion-storage.md` | Not started | |
| Chat, RAG, and LLM | `domains/chat-rag-llm.md` | Not started | |
| Jobs, Scheduler, and Workflows | `domains/jobs-scheduler-workflows.md` | Not started | |
| MCP, Sandbox, and Agent Protocol | `domains/mcp-sandbox-agent-protocol.md` | Not started | |
| DB, Migrations, and Data Durability | `domains/db-migrations-data-durability.md` | Not started | |
| WebUI, Extension, and API Contracts | `domains/webui-extension-api-contracts.md` | Not started | |
| Integrations and Providers | `domains/integrations-providers.md` | Not started | |
| CI, Deployment, Operations, and Release Surfaces | `domains/ci-deployment-operations-release.md` | Not started | |

## Specialist Coverage Map

| Specialist Pass | Report | Status | Notes |
| --- | --- | --- | --- |
| Security boundaries | `specialists/security-boundaries.md` | Not started | |
| Reliability and async lifecycle | `specialists/reliability-lifecycle.md` | Not started | |
| API and WebUI contract drift | `specialists/api-webui-contracts.md` | Not started | |
| Test coverage and verification gaps | `specialists/test-coverage-verification.md` | Not started | |
| Dependency and static-analysis risk | `specialists/dependency-static-analysis.md` | Not started | |

## Inventory Summaries

### API Endpoints

Evidence file: `evidence/endpoint-inventory.txt`

Line count: 2,598

Source scope: route decorators matching `@router.(get|post|put|patch|delete|websocket)(...)` under `tldw_Server_API/app/api/v1/endpoints`.

### Backend Tests

Evidence file: `evidence/backend-test-inventory.txt`

Line count: 4,073

Source scope: test file paths under `tldw_Server_API/tests`.

### Frontend API Clients

Evidence file: `evidence/frontend-api-client-inventory.txt`

Line count: 6,583

Source scope: API and streaming client usage matching `fetch(`, `axios`, `apiClient`, `/api/v1`, `WebSocket`, or `EventSource` in TS/TSX/JS/JSX/MJS/CJS files under `apps/tldw-frontend`, `apps/extension`, and `apps/packages`. The evidence records file, line, and matched API token.

Skipped paths: none; all requested frontend scan roots existed.

### Dependency Manifests

Evidence file: `evidence/dependency-manifest-inventory.txt`

Line count: 14

Source scope: tracked dependency manifests and lockfiles across the repository, with `node_modules` excluded.

### DB Migrations

Evidence file: `evidence/db-migration-inventory.txt`

Line count: 240

Source scope: DB-relevant tracked paths under `tldw_Server_API/Databases`, `tldw_Server_API/app/core/DB_Management`, optional scheduler migrations, and SQL/Alembic/migration candidates under `tldw_Server_API` while excluding API and test schema directories.

### CI, Deployment, And Operations

Evidence file: `evidence/ci-deploy-ops-inventory.txt`

Line count: 201

Source scope: tracked files under `.github`, `Dockerfiles`, `Docs/Operations`, `Docs/Deployment`, and `Helper_Scripts/Samples`.

Skipped paths: none; all requested CI, deployment, and operations scan roots existed.

### Static Scan Baseline

Evidence file: `evidence/bandit-app-summary.txt`

Summary: Bandit exited 1 with JSON output at `/tmp/tldw_repo_audit_bandit_app.json`, which is expected for a baseline containing findings. The JSON contains 4,818 results across 1,120,179 scanned LOC: 4,792 low-severity, 26 medium-severity, and 0 high-severity results.

Limitation: this audit worktree did not contain `.venv`, so the scan used the existing parent repository virtual environment at `/Users/appledev/Documents/GitHub/tldw_server/.venv` without installing dependencies.

## Command Log

Evidence file: `evidence/command-log.md`
