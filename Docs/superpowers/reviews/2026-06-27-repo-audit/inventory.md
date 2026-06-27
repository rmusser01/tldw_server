# Comprehensive Repository Audit Inventory

## Baseline

- Audit execution task: `TASK-12050`
- Baseline SHA: `59b42819623e35e57208e7928d6c2047d3442a91`
- Network refreshed: yes
- Worktree: `.worktrees/comprehensive-repo-audit-2026-06-27`
- Branch: `codex/comprehensive-repo-audit-2026-06-27`

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

### Backend Tests

Evidence file: `evidence/backend-test-inventory.txt`

### Frontend API Clients

Evidence file: `evidence/frontend-api-client-inventory.txt`

### Dependency Manifests

Evidence file: `evidence/dependency-manifest-inventory.txt`

### DB Migrations

Evidence file: `evidence/db-migration-inventory.txt`

### CI, Deployment, And Operations

Evidence file: `evidence/ci-deploy-ops-inventory.txt`

### Static Scan Baseline

Evidence file: `evidence/bandit-app-summary.txt`

## Command Log

Evidence file: `evidence/command-log.md`
