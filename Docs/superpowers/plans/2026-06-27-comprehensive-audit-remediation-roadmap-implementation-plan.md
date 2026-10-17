# Comprehensive Audit Remediation Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Operationalize the approved comprehensive audit remediation roadmap by creating the umbrella Backlog task, concrete decision-gate tasks, 11 child remediation tasks, dependency wiring, and execution handoff records.

**Architecture:** This plan creates Backlog coordination artifacts only. It does not modify production code, tests, runtime configuration, or CI workflows. Actual remediation work is intentionally split into future track-specific implementation plans and branches.

**Tech Stack:** Backlog.md MCP workflow, Git, Markdown planning artifacts, existing audit artifacts under `Docs/superpowers/reviews/2026-06-27-repo-audit/`.

---

## Scope

This plan implements the coordination layer from `Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md`.

It creates:

- One umbrella remediation task: `TASK-12055`.
- Two decision-gate tasks: `TASK-12055.1` and `TASK-12055.2`.
- Eleven child remediation tasks: `TASK-12055.3` through `TASK-12055.13`.
- Dependency links between decision tasks, remediation tasks, and release/supply-chain tasks.

It does not:

- Implement any remediation code.
- Create implementation plans for individual code tracks.
- Start worktrees for child tracks.
- Push a branch or create a PR.

## File Structure

Backlog MCP should create or edit these task records:

- Create: `backlog/tasks/task-12055 - Remediate-2026-06-27-comprehensive-repository-audit-findings.md`
- Create: `backlog/tasks/task-12055.1 - Decide-WebSocket-auth-contract-for-audit-remediation.md`
- Create: `backlog/tasks/task-12055.2 - Decide-durable-workflow-ownership-contract.md`
- Create: `backlog/tasks/task-12055.3 - Harden-AuthNZ-impersonation-boundary.md`
- Create: `backlog/tasks/task-12055.4 - Enforce-media-authorization-and-tenant-scoped-ingestion-storage.md`
- Create: `backlog/tasks/task-12055.5 - Repair-SQLite-migration-durability.md`
- Create: `backlog/tasks/task-12055.6 - Align-browser-WebSocket-and-OSS-billing-API-contracts.md`
- Create: `backlog/tasks/task-12055.7 - Centralize-Chat-RAG-authorization-and-redact-query-logging.md`
- Create: `backlog/tasks/task-12055.8 - Make-workflow-execution-durable-and-idempotent.md`
- Create: `backlog/tasks/task-12055.9 - Establish-supply-chain-foundations-and-worker-image-hardening.md`
- Create: `backlog/tasks/task-12055.10 - Close-release-verification-gates.md`
- Create: `backlog/tasks/task-12055.11 - Route-integrations-through-central-outbound-HTTP-policy.md`
- Create: `backlog/tasks/task-12055.12 - Enforce-MCP-scoped-WebSocket-auth-and-cleanup-lifecycle.md`
- Create: `backlog/tasks/task-12055.13 - Clean-up-dependency-automation-Bandit-profiles-and-small-test-gaps.md`

If any planned ID already exists at execution time, stop before creating tasks and ask the coordinator for a new ID block.

---

### Task 1: Preflight The Roadmap Task Block

**Files:**
- Read: `Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md`
- Read: `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- Read: `backlog/tasks/`
- Read: `backlog/completed/`

- [ ] **Step 1: Verify the working tree scope**

Run:

```bash
git status --short --branch
```

Expected:

```text
## codex/comprehensive-repo-audit-2026-06-27...origin/dev [ahead N]
?? tldw_Server_API/Config_Files/templates/watchlists/cti_osint_report_markdown.md
?? tldw_Server_API/Config_Files/templates/watchlists/news_briefing_markdown.md
```

If any tracked file is modified before this plan starts, inspect it. Continue only when tracked changes are expected for this plan or are committed.

- [ ] **Step 2: Verify planned task IDs are unused**

Run:

```bash
rg --files backlog/tasks backlog/completed | rg 'task-12055'
```

Expected: no output.

If there is output, stop and choose a new contiguous parent ID block before creating tasks.

- [ ] **Step 3: Verify audit findings still total 31**

Run:

```bash
jq -r '.findings | length' Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json
```

Expected:

```text
31
```

- [ ] **Step 4: Verify roadmap spec still covers every finding**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path("Docs/superpowers/reviews/2026-06-27-repo-audit")
spec = Path("Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md").read_text()
ids = [f["id"] for f in json.loads((root / "findings-index.json").read_text())["findings"]]
missing = [finding_id for finding_id in ids if finding_id not in spec]
print(f"missing audit ids: {len(missing)}")
for finding_id in missing:
    print(finding_id)
raise SystemExit(1 if missing else 0)
PY
```

Expected:

```text
missing audit ids: 0
```

- [ ] **Step 5: Commit preflight only if task metadata changed**

If preflight required no file edits, do not commit. If task metadata changed, run:

```bash
git add "backlog/tasks/task-12054 - Write-comprehensive-audit-remediation-implementation-plan.md"
git commit -m "docs: update audit remediation planning preflight"
```

Expected: commit succeeds.

---

### Task 2: Create The Umbrella Remediation Task

**Files:**
- Create: `backlog/tasks/task-12055 - Remediate-2026-06-27-comprehensive-repository-audit-findings.md`

- [ ] **Step 1: Create umbrella task with Backlog MCP**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055",
  "title": "Remediate 2026-06-27 comprehensive repository audit findings",
  "status": "In Progress",
  "priority": "high",
  "labels": ["audit", "remediation", "parallel-agents"],
  "description": "Umbrella coordination task for addressing all 31 accepted findings from the 2026-06-27 comprehensive repository audit. This task coordinates decision gates, child remediation tasks, wave integration gates, and final closure evidence. Child tasks own implementation.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/plans/2026-06-27-comprehensive-audit-remediation-roadmap-implementation-plan.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/final-report.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md"
  ],
  "acceptanceCriteria": [
    "All two decision-gate tasks and 11 child remediation tasks are created with concrete dependencies.",
    "Each accepted audit finding maps to exactly one child remediation task.",
    "Wave integration gates are recorded after each completed wave.",
    "Findings are marked closed only when closure rules from the roadmap spec are satisfied.",
    "Residual risk and environment-dependent verification skips are recorded before final closure."
  ],
  "definitionOfDone": [
    "All child tasks are Done or explicitly blocked with residual risk.",
    "All 31 audit findings are closed, refuted, or pending external verification with evidence.",
    "Final wave integration verification is recorded.",
    "Final summary links the remediation PRs or commits."
  ],
  "references": [
    "AUDIT-2026-06-27-AUTH-001",
    "AUDIT-2026-06-27-AUTH-002",
    "AUDIT-2026-06-27-AUTH-003",
    "AUDIT-2026-06-27-DB-001",
    "AUDIT-2026-06-27-DB-002",
    "AUDIT-2026-06-27-MEDIA-001",
    "AUDIT-2026-06-27-MEDIA-002",
    "AUDIT-2026-06-27-MEDIA-003",
    "AUDIT-2026-06-27-MEDIA-004",
    "AUDIT-2026-06-27-WEBUI-001",
    "AUDIT-2026-06-27-WEBUI-002",
    "AUDIT-2026-06-27-APIWEB-001",
    "AUDIT-2026-06-27-CHAT-001",
    "AUDIT-2026-06-27-CHAT-002",
    "AUDIT-2026-06-27-JOBS-001",
    "AUDIT-2026-06-27-JOBS-002",
    "AUDIT-2026-06-27-REL-001",
    "AUDIT-2026-06-27-OPS-001",
    "AUDIT-2026-06-27-OPS-002",
    "AUDIT-2026-06-27-OPS-003",
    "AUDIT-2026-06-27-OPS-004",
    "AUDIT-2026-06-27-OPS-005",
    "AUDIT-2026-06-27-OPS-006",
    "AUDIT-2026-06-27-DEPS-001",
    "AUDIT-2026-06-27-DEPS-002",
    "AUDIT-2026-06-27-DEPS-003",
    "AUDIT-2026-06-27-INTEGRATIONS-001",
    "AUDIT-2026-06-27-INTEGRATIONS-002",
    "AUDIT-2026-06-27-INTEGRATIONS-003",
    "AUDIT-2026-06-27-MCP-001",
    "AUDIT-2026-06-27-MCP-002"
  ]
}
```

- [ ] **Step 2: Verify umbrella task exists**

Run:

```bash
rg -n "id: TASK-12055|Remediate 2026-06-27 comprehensive repository audit findings" "backlog/tasks/task-12055 - Remediate-2026-06-27-comprehensive-repository-audit-findings.md"
```

Expected: output contains the task ID and title.

- [ ] **Step 3: Commit umbrella task**

Run:

```bash
git add "backlog/tasks/task-12055 - Remediate-2026-06-27-comprehensive-repository-audit-findings.md"
git commit -m "docs: create audit remediation umbrella task"
```

Expected: commit succeeds.

---

### Task 3: Create Decision-Gate Tasks

**Files:**
- Create: `backlog/tasks/task-12055.1 - Decide-WebSocket-auth-contract-for-audit-remediation.md`
- Create: `backlog/tasks/task-12055.2 - Decide-durable-workflow-ownership-contract.md`

- [ ] **Step 1: Create WebSocket auth decision task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.1",
  "parentTaskId": "TASK-12055",
  "title": "Decide WebSocket auth contract for audit remediation",
  "status": "To Do",
  "priority": "high",
  "labels": ["audit", "remediation", "decision-gate", "websocket"],
  "description": "Record one WebSocket auth contract for browser clients, ACP streams, and sandbox streams before Tracks 4 and 9 implement fixes.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/final-report.md"
  ],
  "acceptanceCriteria": [
    "Contract defines default query-token behavior.",
    "Contract defines first-frame auth semantics for browser clients.",
    "Contract defines scoped JWT enforcement expectations for protocol streams.",
    "Contract defines backend and frontend test expectations.",
    "Tracks 4 and 9 link this decision before implementation starts."
  ],
  "definitionOfDone": [
    "Decision is recorded in the task final summary or linked design note.",
    "Dependent tasks reference the decision task ID.",
    "No conflicting WebSocket auth semantics are left unresolved."
  ],
  "references": [
    "AUDIT-2026-06-27-WEBUI-002",
    "AUDIT-2026-06-27-APIWEB-001",
    "AUDIT-2026-06-27-MCP-001"
  ]
}
```

- [ ] **Step 2: Create durable workflow ownership decision task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.2",
  "parentTaskId": "TASK-12055",
  "title": "Decide durable workflow ownership contract",
  "status": "To Do",
  "priority": "high",
  "labels": ["audit", "remediation", "decision-gate", "workflows"],
  "description": "Decide whether accepted user-visible workflow execution is owned by Jobs or Scheduler before durable workflow remediation starts.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/final-report.md"
  ],
  "acceptanceCriteria": [
    "Decision records whether Jobs or Scheduler owns accepted workflow runs.",
    "Decision defines idempotency key ownership.",
    "Decision defines startup repair behavior.",
    "Decision defines duplicate-fire collapse behavior.",
    "Decision defines shutdown, cancellation, and process-loss verification expectations.",
    "Track 6 links this decision before implementation starts."
  ],
  "definitionOfDone": [
    "Decision is recorded in the task final summary or linked design note.",
    "Track 6 acceptance criteria align with the selected owner.",
    "No conflicting durable execution ownership model is left unresolved."
  ],
  "references": [
    "AUDIT-2026-06-27-JOBS-001",
    "AUDIT-2026-06-27-JOBS-002",
    "AUDIT-2026-06-27-REL-001"
  ]
}
```

- [ ] **Step 3: Verify decision tasks exist**

Run:

```bash
rg -n "id: TASK-12055\\.1|id: TASK-12055\\.2|parentTaskId: TASK-12055|parent_task_id: TASK-12055" backlog/tasks
```

Expected: output includes both decision task IDs and parent links.

- [ ] **Step 4: Commit decision tasks**

Run:

```bash
git add "backlog/tasks/task-12055.1 - Decide-WebSocket-auth-contract-for-audit-remediation.md" "backlog/tasks/task-12055.2 - Decide-durable-workflow-ownership-contract.md"
git commit -m "docs: create audit remediation decision tasks"
```

Expected: commit succeeds.

---

### Task 4: Create Wave 1 High-Risk Remediation Tasks

**Files:**
- Create: `backlog/tasks/task-12055.3 - Harden-AuthNZ-impersonation-boundary.md`
- Create: `backlog/tasks/task-12055.4 - Enforce-media-authorization-and-tenant-scoped-ingestion-storage.md`
- Create: `backlog/tasks/task-12055.5 - Repair-SQLite-migration-durability.md`

- [ ] **Step 1: Create AuthNZ impersonation task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.3",
  "parentTaskId": "TASK-12055",
  "title": "Harden AuthNZ impersonation boundary",
  "status": "To Do",
  "priority": "high",
  "labels": ["audit", "remediation", "authnz", "wave-1"],
  "description": "Fix the AuthNZ impersonation audit boundary: short-lived token semantics, actor-plus-subject propagation, durable audit attribution, and backend-neutral lookup helpers.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md"
  ],
  "modifiedFiles": [
    "tldw_Server_API/app/core/AuthNZ/",
    "tldw_Server_API/app/api/v1/API_Deps/",
    "tldw_Server_API/tests/AuthNZ/"
  ],
  "acceptanceCriteria": [
    "Impersonation token lifetime matches the documented short TTL.",
    "Actor and subject survive from token issuance into downstream request context.",
    "Durable audit events capture impersonation issuance and impersonated actions.",
    "PostgreSQL and SQLite lookup paths use backend-neutral query helpers."
  ],
  "definitionOfDone": [
    "Token decode tests assert exp minus iat.",
    "SQLite and PostgreSQL impersonation tests cover user and role lookup.",
    "Audit attribution tests assert actor and subject fields.",
    "Bandit runs over touched AuthNZ production paths.",
    "Findings AUTH-001, AUTH-002, and AUTH-003 are closed or have residual risk recorded."
  ],
  "references": [
    "AUDIT-2026-06-27-AUTH-001",
    "AUDIT-2026-06-27-AUTH-002",
    "AUDIT-2026-06-27-AUTH-003"
  ]
}
```

- [ ] **Step 2: Create media authorization and storage task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.4",
  "parentTaskId": "TASK-12055",
  "title": "Enforce media authorization and tenant-scoped ingestion storage",
  "status": "To Do",
  "priority": "high",
  "labels": ["audit", "remediation", "media", "wave-1"],
  "description": "Normalize media processing authorization and tenant-scoped MediaWiki storage/vector writes, including compensating cleanup for original-file persistence failures.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md"
  ],
  "modifiedFiles": [
    "tldw_Server_API/app/api/v1/endpoints/",
    "tldw_Server_API/app/core/Ingestion_Media_Processing/",
    "tldw_Server_API/app/core/DB_Management/",
    "tldw_Server_API/tests/"
  ],
  "acceptanceCriteria": [
    "Processing-only media endpoints enforce the chosen permission gate.",
    "MediaWiki ingest writes DB and vector data under the request user in multi-user mode.",
    "Original-file persistence cleans up stored files if DB registration fails."
  ],
  "definitionOfDone": [
    "Unauthorized media processing route tests return HTTP 403.",
    "Multi-user MediaWiki ingest isolation tests pass.",
    "Fake storage test asserts compensating delete on DB failure.",
    "Bandit runs over touched media and ingestion production paths.",
    "Findings MEDIA-001, MEDIA-002, and MEDIA-003 are closed or have residual risk recorded."
  ],
  "references": [
    "AUDIT-2026-06-27-MEDIA-001",
    "AUDIT-2026-06-27-MEDIA-002",
    "AUDIT-2026-06-27-MEDIA-003"
  ]
}
```

- [ ] **Step 3: Create SQLite migration durability task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.5",
  "parentTaskId": "TASK-12055",
  "title": "Repair SQLite migration durability",
  "status": "To Do",
  "priority": "high",
  "labels": ["audit", "remediation", "database", "wave-1"],
  "description": "Repair SQLite Media DB legacy migration support and atomic migration ledger/schema updates.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/domains/db-migrations-data-durability.md"
  ],
  "modifiedFiles": [
    "tldw_Server_API/app/core/DB_Management/",
    "tldw_Server_API/Config_Files/",
    "tldw_Server_API/tests/"
  ],
  "acceptanceCriteria": [
    "Legacy Media DBs below the supported minimum are upgraded through a tested path or rejected with an explicit recovery message.",
    "Multi-statement migration failure does not leave a successful ledger or schema bump.",
    "Migration packaging no longer applies incompatible scripts to the wrong database domain."
  ],
  "definitionOfDone": [
    "File-backed legacy Media DB upgrade tests pass for representative old versions.",
    "Failed multi-statement migration atomicity test passes.",
    "Bandit runs over touched DB production paths when Python changes.",
    "Findings DB-001 and DB-002 are closed or have residual risk recorded."
  ],
  "references": [
    "AUDIT-2026-06-27-DB-001",
    "AUDIT-2026-06-27-DB-002"
  ]
}
```

- [ ] **Step 4: Verify Wave 1 task files**

Run:

```bash
for id in 12055.3 12055.4 12055.5; do
  rg -n "id: TASK-$id|parentTaskId: TASK-12055|parent_task_id: TASK-12055|AUDIT-2026-06-27" backlog/tasks
done
```

Expected: each task ID appears with parent linkage and audit references.

- [ ] **Step 5: Commit Wave 1 tasks**

Run:

```bash
git add "backlog/tasks/task-12055.3 - Harden-AuthNZ-impersonation-boundary.md" "backlog/tasks/task-12055.4 - Enforce-media-authorization-and-tenant-scoped-ingestion-storage.md" "backlog/tasks/task-12055.5 - Repair-SQLite-migration-durability.md"
git commit -m "docs: create wave one audit remediation tasks"
```

Expected: commit succeeds.

---

### Task 5: Create Wave 2 Cross-Cutting Remediation Tasks

**Files:**
- Create: `backlog/tasks/task-12055.6 - Align-browser-WebSocket-and-OSS-billing-API-contracts.md`
- Create: `backlog/tasks/task-12055.7 - Centralize-Chat-RAG-authorization-and-redact-query-logging.md`
- Create: `backlog/tasks/task-12055.8 - Make-workflow-execution-durable-and-idempotent.md`
- Create: `backlog/tasks/task-12055.11 - Route-integrations-through-central-outbound-HTTP-policy.md`
- Create: `backlog/tasks/task-12055.12 - Enforce-MCP-scoped-WebSocket-auth-and-cleanup-lifecycle.md`

- [ ] **Step 1: Create WebUI/API contract task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.6",
  "parentTaskId": "TASK-12055",
  "title": "Align browser WebSocket and OSS billing API contracts",
  "status": "To Do",
  "priority": "medium",
  "labels": ["audit", "remediation", "webui", "api-contract", "wave-2"],
  "dependencies": ["TASK-12055.1"],
  "description": "Align TTS, STT, and voice chat browser WebSocket auth with the shared contract, and guard hosted-only billing routes in the OSS UI.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/domains/webui-extension-api-contracts.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/api-webui-contracts.md"
  ],
  "modifiedFiles": [
    "apps/tldw-frontend/",
    "tldw_Server_API/app/api/v1/endpoints/audio/",
    "tldw_Server_API/tests/"
  ],
  "acceptanceCriteria": [
    "TTS, STT, and voice chat browser flows use the shared first-frame auth contract.",
    "Backend tests reject query-token WebSocket auth when disabled by default.",
    "OSS billing UI routes are hidden, disabled, or guarded by a backend capability signal."
  ],
  "definitionOfDone": [
    "Frontend client tests cover first-frame auth.",
    "Backend WebSocket auth tests pass.",
    "Route-contract check covers unguarded OSS billing calls.",
    "Findings WEBUI-001, WEBUI-002, and APIWEB-001 are closed or have residual risk recorded."
  ],
  "references": [
    "AUDIT-2026-06-27-WEBUI-001",
    "AUDIT-2026-06-27-WEBUI-002",
    "AUDIT-2026-06-27-APIWEB-001"
  ]
}
```

- [ ] **Step 2: Create Chat/RAG authorization and logging task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.7",
  "parentTaskId": "TASK-12055",
  "title": "Centralize Chat RAG authorization and redact query logging",
  "status": "To Do",
  "priority": "medium",
  "labels": ["audit", "remediation", "chat", "rag", "wave-2"],
  "description": "Centralize resource authorization across alternate Chat/RAG generation routes and remove raw user query text from info logs.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/domains/chat-rag-llm.md"
  ],
  "modifiedFiles": [
    "tldw_Server_API/app/api/v1/endpoints/",
    "tldw_Server_API/app/core/RAG/",
    "tldw_Server_API/app/core/Chat/",
    "tldw_Server_API/tests/"
  ],
  "acceptanceCriteria": [
    "Alternate RAG, character completion, document generation, and embedding routes enforce the same virtual-key and max-call rules as primary routes.",
    "Info logs do not contain raw user query text.",
    "Redacted logs preserve non-sensitive debugging context."
  ],
  "definitionOfDone": [
    "Scoped virtual-key HTTP tests cover alternate routes.",
    "caplog tests prove raw query text is absent.",
    "Bandit runs over touched Chat/RAG production paths.",
    "Findings CHAT-001 and CHAT-002 are closed or have residual risk recorded."
  ],
  "references": [
    "AUDIT-2026-06-27-CHAT-001",
    "AUDIT-2026-06-27-CHAT-002"
  ]
}
```

- [ ] **Step 3: Create durable workflow task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.8",
  "parentTaskId": "TASK-12055",
  "title": "Make workflow execution durable and idempotent",
  "status": "To Do",
  "priority": "medium",
  "labels": ["audit", "remediation", "workflows", "wave-2"],
  "dependencies": ["TASK-12055.2"],
  "description": "Implement the selected durable ownership contract for accepted workflow runs, recurring schedules, and continuation resumes.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/domains/jobs-scheduler-workflows.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/reliability-lifecycle.md"
  ],
  "modifiedFiles": [
    "tldw_Server_API/app/core/Chat_Workflows/",
    "tldw_Server_API/app/core/Jobs/",
    "tldw_Server_API/app/core/Scheduler/",
    "tldw_Server_API/tests/"
  ],
  "acceptanceCriteria": [
    "Accepted workflow runs have one durable execution owner.",
    "Startup repair handles accepted but unowned work.",
    "Recurring and continuation work use deterministic idempotency keys.",
    "Duplicate schedule fires collapse to one run or task."
  ],
  "definitionOfDone": [
    "Process-loss and post-acceptance-failure tests pass.",
    "Duplicate schedule-fire tests pass.",
    "Shutdown and startup repair tests pass.",
    "Bandit runs over touched workflow, Jobs, or Scheduler paths.",
    "Findings JOBS-001, JOBS-002, and REL-001 are closed or have residual risk recorded."
  ],
  "references": [
    "AUDIT-2026-06-27-JOBS-001",
    "AUDIT-2026-06-27-JOBS-002",
    "AUDIT-2026-06-27-REL-001"
  ]
}
```

- [ ] **Step 4: Create outbound HTTP policy task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.11",
  "parentTaskId": "TASK-12055",
  "title": "Route integrations through central outbound HTTP policy",
  "status": "To Do",
  "priority": "medium",
  "labels": ["audit", "remediation", "integrations", "http-policy", "wave-2"],
  "description": "Route workflow research adapters, tokenizer resolver, and weather provider calls through central outbound HTTP policy or explicit safe exceptions.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/domains/integrations-providers.md"
  ],
  "modifiedFiles": [
    "tldw_Server_API/app/core/Workflows/",
    "tldw_Server_API/app/core/Integrations/",
    "tldw_Server_API/app/core/http_client.py",
    "tldw_Server_API/tests/"
  ],
  "acceptanceCriteria": [
    "Workflow research adapters use central outbound HTTP policy for fetches and direct pdf_url downloads.",
    "Tokenizer resolver uses central HTTP policy or a documented local-provider exception.",
    "Weather provider uses central HTTP defaults or explicitly safe client configuration."
  ],
  "definitionOfDone": [
    "Private and loopback URL denial tests pass.",
    "Proxy and trust_env behavior tests pass.",
    "Tokenizer URL and provider base URL tests pass.",
    "Bandit runs over touched integration production paths.",
    "Findings INTEGRATIONS-001, INTEGRATIONS-002, and INTEGRATIONS-003 are closed or have residual risk recorded."
  ],
  "references": [
    "AUDIT-2026-06-27-INTEGRATIONS-001",
    "AUDIT-2026-06-27-INTEGRATIONS-002",
    "AUDIT-2026-06-27-INTEGRATIONS-003"
  ]
}
```

- [ ] **Step 5: Create MCP WebSocket auth and lifecycle task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.12",
  "parentTaskId": "TASK-12055",
  "title": "Enforce MCP scoped WebSocket auth and cleanup lifecycle",
  "status": "To Do",
  "priority": "medium",
  "labels": ["audit", "remediation", "mcp", "websocket", "wave-2"],
  "dependencies": ["TASK-12055.1"],
  "description": "Apply scoped AuthNZ JWT restrictions to ACP and sandbox WebSocket streams, and clean up ACP reconnect replay lifecycle.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/domains/mcp-sandbox-agent-protocol.md"
  ],
  "modifiedFiles": [
    "tldw_Server_API/app/core/MCP_unified/",
    "tldw_Server_API/app/core/Agent_Client_Protocol/",
    "tldw_Server_API/app/core/Sandbox/",
    "tldw_Server_API/tests/"
  ],
  "acceptanceCriteria": [
    "Scoped JWT endpoint, method, path, scope, and quota restrictions apply to ACP and sandbox WebSocket handshakes.",
    "ACP reconnect replay stops broadcasters and removes event-bus subscribers on disconnect.",
    "Existing ownership checks remain intact."
  ],
  "definitionOfDone": [
    "Scoped JWT rejection tests pass for ACP stream, ACP SSH, sandbox run stream, and sandbox stdin.",
    "Reconnect-disconnect lifecycle test proves broadcaster task and subscriber counts return to baseline.",
    "Bandit runs over touched MCP and sandbox production paths.",
    "Findings MCP-001 and MCP-002 are closed or have residual risk recorded."
  ],
  "references": [
    "AUDIT-2026-06-27-MCP-001",
    "AUDIT-2026-06-27-MCP-002"
  ]
}
```

- [ ] **Step 6: Verify Wave 2 dependencies**

Run:

```bash
rg -n "TASK-12055\\.1|TASK-12055\\.2|dependencies:" backlog/tasks/task-12055.*
```

Expected: Track 4 and Track 9 tasks reference `TASK-12055.1`; Track 6 references `TASK-12055.2`.

- [ ] **Step 7: Commit Wave 2 tasks**

Run:

```bash
git add "backlog/tasks/task-12055.6 - Align-browser-WebSocket-and-OSS-billing-API-contracts.md" "backlog/tasks/task-12055.7 - Centralize-Chat-RAG-authorization-and-redact-query-logging.md" "backlog/tasks/task-12055.8 - Make-workflow-execution-durable-and-idempotent.md" "backlog/tasks/task-12055.11 - Route-integrations-through-central-outbound-HTTP-policy.md" "backlog/tasks/task-12055.12 - Enforce-MCP-scoped-WebSocket-auth-and-cleanup-lifecycle.md"
git commit -m "docs: create wave two audit remediation tasks"
```

Expected: commit succeeds.

---

### Task 6: Create Wave 3 Supply-Chain And Release Tasks

**Files:**
- Create: `backlog/tasks/task-12055.9 - Establish-supply-chain-foundations-and-worker-image-hardening.md`
- Create: `backlog/tasks/task-12055.10 - Close-release-verification-gates.md`

- [ ] **Step 1: Create supply-chain foundations task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.9",
  "parentTaskId": "TASK-12055",
  "title": "Establish supply-chain foundations and worker image hardening",
  "status": "To Do",
  "priority": "medium",
  "labels": ["audit", "remediation", "supply-chain", "wave-3"],
  "description": "Define and enforce Python lock or constraints strategy, pin static-analysis/tool setup, and harden worker runtime images.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/dependency-static-analysis.md"
  ],
  "modifiedFiles": [
    "pyproject.toml",
    "Dockerfiles/",
    ".github/workflows/",
    "Docs/"
  ],
  "acceptanceCriteria": [
    "Runtime, CI, Docker, and release installs consume a committed lock or constraints profile or documented equivalent.",
    "Static-analysis, action, and tool installers are pinned to releases, checksums, or immutable SHAs.",
    "Worker and audio-worker images run as non-root and minimize build tooling in runtime layers."
  ],
  "definitionOfDone": [
    "Dependency resolution or lock validation command passes.",
    "Docker image build or inspection evidence is recorded, or decisive external verification remains pending.",
    "CI/tool pinning validation is recorded.",
    "Findings OPS-002, DEPS-001, and DEPS-002 are closed or pending decisive external verification with evidence."
  ],
  "references": [
    "AUDIT-2026-06-27-OPS-002",
    "AUDIT-2026-06-27-DEPS-001",
    "AUDIT-2026-06-27-DEPS-002"
  ]
}
```

- [ ] **Step 2: Create release verification gates task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.10",
  "parentTaskId": "TASK-12055",
  "title": "Close release verification gates",
  "status": "To Do",
  "priority": "medium",
  "labels": ["audit", "remediation", "release", "ci", "wave-3"],
  "dependencies": ["TASK-12055.9"],
  "description": "Close release verification gaps for published image build coverage, actionlint coverage, Bun-aware SBOM generation, and Kubernetes sample validation.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md"
  ],
  "modifiedFiles": [
    ".github/workflows/",
    "Dockerfiles/",
    "Docs/",
    "k8s/"
  ],
  "acceptanceCriteria": [
    "PR container build matrix includes every published image, including worker and audio-worker images.",
    "actionlint covers all workflow and composite action files.",
    "SBOM generation includes Python plus Bun-managed frontend and admin dependencies.",
    "Kubernetes sample secrets use safe values and validate correctly."
  ],
  "definitionOfDone": [
    "actionlint coverage evidence is recorded.",
    "SBOM workflow dry run or local equivalent is recorded, or decisive external verification remains pending.",
    "Docker build matrix validation is recorded, or decisive external verification remains pending.",
    "Kubernetes sample validation is recorded.",
    "Findings OPS-001, OPS-003, OPS-004, and OPS-006 are closed or pending decisive external verification with evidence."
  ],
  "references": [
    "AUDIT-2026-06-27-OPS-001",
    "AUDIT-2026-06-27-OPS-003",
    "AUDIT-2026-06-27-OPS-004",
    "AUDIT-2026-06-27-OPS-006"
  ]
}
```

- [ ] **Step 3: Verify Track 7B dependency**

Run:

```bash
rg -n "TASK-12055\\.9|dependencies:" "backlog/tasks/task-12055.10 - Close-release-verification-gates.md"
```

Expected: task `TASK-12055.10` depends on `TASK-12055.9`.

- [ ] **Step 4: Commit Wave 3 tasks**

Run:

```bash
git add "backlog/tasks/task-12055.9 - Establish-supply-chain-foundations-and-worker-image-hardening.md" "backlog/tasks/task-12055.10 - Close-release-verification-gates.md"
git commit -m "docs: create wave three audit remediation tasks"
```

Expected: commit succeeds.

---

### Task 7: Create Wave 4 Cleanup Task

**Files:**
- Create: `backlog/tasks/task-12055.13 - Clean-up-dependency-automation-Bandit-profiles-and-small-test-gaps.md`

- [ ] **Step 1: Create maintenance and test hygiene task**

Call `mcp__backlog.task_create` with:

```json
{
  "project": "/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27",
  "id": "TASK-12055.13",
  "parentTaskId": "TASK-12055",
  "title": "Clean up dependency automation Bandit profiles and small test gaps",
  "status": "To Do",
  "priority": "low",
  "labels": ["audit", "remediation", "maintenance", "wave-4"],
  "description": "Close low-priority maintenance and test hygiene gaps around dependency automation coverage, Bandit profiles, and oversized audio download test coverage.",
  "documentation": [
    "Docs/superpowers/specs/2026-06-27-comprehensive-audit-remediation-roadmap-design.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/dependency-static-analysis.md",
    "Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md"
  ],
  "modifiedFiles": [
    ".github/",
    "pyproject.toml",
    "tldw_Server_API/tests/"
  ],
  "acceptanceCriteria": [
    "Dependency update automation covers intended nested Bun, Python, and Go roots or documents explicit exclusions.",
    "Bandit production profile excludes test directories while a test profile remains available for review.",
    "Oversized audio download regression test invokes the downloader and asserts the expected size error."
  ],
  "definitionOfDone": [
    "Dependency automation config validation is recorded.",
    "Bandit profile validation is recorded.",
    "Focused oversized audio download test passes.",
    "Findings OPS-005, DEPS-003, and MEDIA-004 are closed or have residual risk recorded."
  ],
  "references": [
    "AUDIT-2026-06-27-OPS-005",
    "AUDIT-2026-06-27-DEPS-003",
    "AUDIT-2026-06-27-MEDIA-004"
  ]
}
```

- [ ] **Step 2: Verify Wave 4 task**

Run:

```bash
rg -n "AUDIT-2026-06-27-OPS-005|AUDIT-2026-06-27-DEPS-003|AUDIT-2026-06-27-MEDIA-004" "backlog/tasks/task-12055.13 - Clean-up-dependency-automation-Bandit-profiles-and-small-test-gaps.md"
```

Expected: all three finding IDs appear.

- [ ] **Step 3: Commit Wave 4 task**

Run:

```bash
git add "backlog/tasks/task-12055.13 - Clean-up-dependency-automation-Bandit-profiles-and-small-test-gaps.md"
git commit -m "docs: create wave four audit remediation task"
```

Expected: commit succeeds.

---

### Task 8: Verify Finding Coverage And Dependency Wiring

**Files:**
- Read: `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- Read: `backlog/tasks/task-12055*`
- Modify: `backlog/tasks/task-12055 - Remediate-2026-06-27-comprehensive-repository-audit-findings.md`

- [ ] **Step 1: Verify all 31 findings appear in child tasks**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path("Docs/superpowers/reviews/2026-06-27-repo-audit")
ids = [f["id"] for f in json.loads((root / "findings-index.json").read_text())["findings"]]
task_text = "\n".join(p.read_text() for p in Path("backlog/tasks").glob("task-12055*"))
missing = [finding_id for finding_id in ids if finding_id not in task_text]
print(f"findings: {len(ids)}")
print(f"missing from task map: {len(missing)}")
for finding_id in missing:
    print(finding_id)
raise SystemExit(1 if missing else 0)
PY
```

Expected:

```text
findings: 31
missing from task map: 0
```

- [ ] **Step 2: Verify dependency references**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
checks = {
    "task-12055.6 - Align-browser-WebSocket-and-OSS-billing-API-contracts.md": "TASK-12055.1",
    "task-12055.8 - Make-workflow-execution-durable-and-idempotent.md": "TASK-12055.2",
    "task-12055.10 - Close-release-verification-gates.md": "TASK-12055.9",
    "task-12055.12 - Enforce-MCP-scoped-WebSocket-auth-and-cleanup-lifecycle.md": "TASK-12055.1",
}
failed = []
for name, dependency in checks.items():
    path = Path("backlog/tasks") / name
    text = path.read_text()
    if dependency not in text:
        failed.append(f"{name} missing {dependency}")
print(f"dependency failures: {len(failed)}")
for item in failed:
    print(item)
raise SystemExit(1 if failed else 0)
PY
```

Expected:

```text
dependency failures: 0
```

- [ ] **Step 3: Update umbrella task with creation summary**

Call `mcp__backlog.task_edit` for `TASK-12055` with `appendNotes`:

```text
Operational task map created: two shared decision-gate tasks and 11 child remediation tasks. Finding coverage verification confirmed all 31 accepted audit findings appear in the TASK-12055 task family. Dependency verification confirmed Tracks 4 and 9 depend on the WebSocket auth decision task, Track 6 depends on the durable workflow ownership decision task, and Track 7B depends on Track 7A.
```

- [ ] **Step 4: Run marker and whitespace checks**

Run:

```bash
! rg -in "tbd|todo|fixme|placeholder|recorded at scaffold time|no accepted findings" backlog/tasks/task-12055*
git diff --check -- backlog/tasks/task-12055*
```

Expected: both commands exit 0 with no issue output.

- [ ] **Step 5: Commit verification update**

Run:

```bash
git add backlog/tasks/task-12055*
git commit -m "docs: verify audit remediation task map"
```

Expected: commit succeeds.

---

### Task 9: Final Handoff For Track-Specific Planning

**Files:**
- Modify: `backlog/tasks/task-12054 - Write-comprehensive-audit-remediation-implementation-plan.md`
- Modify: `backlog/tasks/task-12055 - Remediate-2026-06-27-comprehensive-repository-audit-findings.md`

- [ ] **Step 1: Update plan-writing task final summary**

Call `mcp__backlog.task_edit` for `TASK-12054` with:

```json
{
  "status": "Done",
  "checkAc": [1, 2, 3, 4],
  "checkDod": [1, 2, 3],
  "finalSummary": "Implementation plan completed for operationalizing the comprehensive audit remediation roadmap. The plan creates one umbrella task, two concrete decision-gate tasks, 11 child remediation tasks, dependency wiring, coverage verification, and handoff rules. Code remediation remains split into future track-specific implementation plans."
}
```

- [ ] **Step 2: Update umbrella task with next-step handoff**

Call `mcp__backlog.task_edit` for `TASK-12055` with `appendNotes`:

```text
Next step after task-map creation: choose execution mode for the remediation program. Recommended mode is subagent-driven execution with one implementation plan per track, starting with Wave 0 setup and then Wave 1 high-risk tasks.
```

- [ ] **Step 3: Verify task final-summary markers**

Run:

```bash
for file in "backlog/tasks/task-12054 - Write-comprehensive-audit-remediation-implementation-plan.md" "backlog/tasks/task-12055 - Remediate-2026-06-27-comprehensive-repository-audit-findings.md"; do
  printf "%s BEGIN " "$file"
  rg -c "SECTION:FINAL_SUMMARY:BEGIN" "$file"
  printf "%s END " "$file"
  rg -c "SECTION:FINAL_SUMMARY:END" "$file"
done
```

Expected: each file reports one begin marker and one end marker.

- [ ] **Step 4: Run final status check**

Run:

```bash
git status --short --branch
```

Expected: tracked changes are limited to `TASK-12054` and `TASK-12055`, plus the known unrelated untracked watchlist templates if they still exist.

- [ ] **Step 5: Commit final handoff**

Run:

```bash
git add "backlog/tasks/task-12054 - Write-comprehensive-audit-remediation-implementation-plan.md" "backlog/tasks/task-12055 - Remediate-2026-06-27-comprehensive-repository-audit-findings.md"
git commit -m "docs: hand off audit remediation task map"
```

Expected: commit succeeds.

---

## Execution Notes

- This plan creates coordination tasks only. After it is executed, write one future implementation plan per remediation track before code changes.
- Start track-specific code planning with Wave 1 unless the coordinator explicitly prioritizes a different wave.
- Track-specific implementation plans must use test-driven development for code changes and run Bandit on touched Python production paths.
- If Backlog MCP cannot preserve task markers during execution, pause and ask the user before direct task-file edits.
- Keep the two unrelated watchlist template files unstaged unless the user explicitly assigns them.

## Self-Review Checklist

- The plan maps to the approved roadmap spec.
- The plan creates concrete Backlog tasks for Gate 2 and Gate 3.
- The plan creates all 11 child remediation tasks.
- The plan includes dependency verification for Gate 2, Gate 3, and Track 7A before Track 7B.
- The plan verifies all 31 accepted audit IDs appear in the task family.
- The plan stops before code remediation and requires future track-specific implementation plans.
