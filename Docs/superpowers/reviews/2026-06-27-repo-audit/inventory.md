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
| AuthNZ and Admin | `domains/authnz-admin.md` | Complete | 3 candidate findings |
| Media, Ingestion, and Storage | `domains/media-ingestion-storage.md` | Complete | 4 candidate findings |
| Chat, RAG, and LLM | `domains/chat-rag-llm.md` | Complete | 2 candidate findings |
| Jobs, Scheduler, and Workflows | `domains/jobs-scheduler-workflows.md` | Complete | 2 candidate findings |
| MCP, Sandbox, and Agent Protocol | `domains/mcp-sandbox-agent-protocol.md` | Complete | 2 candidate findings |
| DB, Migrations, and Data Durability | `domains/db-migrations-data-durability.md` | Complete | 2 candidate findings |
| WebUI, Extension, and API Contracts | `domains/webui-extension-api-contracts.md` | Complete | 2 candidate findings |
| Integrations and Providers | `domains/integrations-providers.md` | Complete | 3 candidate findings |
| CI, Deployment, Operations, and Release Surfaces | `domains/ci-deployment-operations-release.md` | Complete | 6 candidate findings |

## Specialist Coverage Map

| Specialist Pass | Report | Status | Notes |
| --- | --- | --- | --- |
| Security boundaries | `specialists/security-boundaries.md` | Complete | Confirmed/cross-linked existing security-boundary findings; no new SEC findings |
| Reliability and async lifecycle | `specialists/reliability-lifecycle.md` | Complete | Added specialist candidate AUDIT-2026-06-27-REL-001; reconcile with AUDIT-2026-06-27-JOBS-001 during index finalization |
| API and WebUI contract drift | `specialists/api-webui-contracts.md` | Complete | Added specialist candidate AUDIT-2026-06-27-APIWEB-001; escalates AUDIT-2026-06-27-WEBUI-002 beyond TTS |
| Test coverage and verification gaps | `specialists/test-coverage-verification.md` | Complete | No new TESTS findings; recorded targeted coverage follow-up and focused pytest result |
| Dependency and static-analysis risk | `specialists/dependency-static-analysis.md` | Complete | Added specialist candidates AUDIT-2026-06-27-DEPS-001 through AUDIT-2026-06-27-DEPS-003 |

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

Specialist follow-up evidence: `evidence/dependency-static-analysis-evidence.txt`

## Command Log

Evidence file: `evidence/command-log.md`
