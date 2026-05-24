# Research Workspace Prerequisite Stack Packaging Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the existing Research Workspace WIP into a clean branch based on `origin/main`, without unrelated chat/sidebar/writing changes.

**Architecture:** Treat the dirty `codex/chat-sidebar-tools-first` checkout as the source tree, but create a clean review branch from `origin/main`. Bring over only Research Workspace route replacement, migration safety/protocol, source status/trust panel/bootstrap, and related tests/docs. Validate with backend tests, frontend Vitest, Bandit for touched Python, route-reference checks, and a real backend plus CDP browser smoke when feasible.

**Tech Stack:** Git worktrees, FastAPI/Pydantic/Pytest/Bandit, React/TypeScript/Vitest, Next.js WebUI, CDP/Playwright.

**Packaging guardrail:** Transfer files by explicit manifest, not broad directory copy. For tracked modifications/deletions, apply path-scoped diffs from the source checkout to the clean worktree. For untracked additions, copy only paths named in this plan. After each packaging task, inspect `git status --short` in the clean worktree and remove any accidental unrelated path before continuing.

---

### Task 1: Create Clean Packaging Branch

**Files:**
- Source checkout: `/Users/macbook-dev/Documents/GitHub/tldw_server2`
- Create worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/research-workspace-prereq-stack`

- [ ] **Step 1: Verify source dirty checkout remains available**

Run:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2 branch --show-current
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2 status --short | wc -l
```

Expected: branch is `codex/chat-sidebar-tools-first`; many dirty files exist.

- [ ] **Step 2: Create clean worktree from `origin/main`**

Run:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2 worktree add \
  .worktrees/research-workspace-prereq-stack \
  -b codex/research-workspace-prereq-stack \
  origin/main
```

Expected: clean worktree created at PR #2018 merge commit.

- [ ] **Step 3: Verify baseline**

Run:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/research-workspace-prereq-stack status --short
```

Expected: no output.

### Task 2: Bring Over Approved Roadmap And Tracking

**Files:**
- Add/modify: `Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md`
- Add/modify: `backlog/tasks/task-463 - Design-Research-Workspace-hard-replacement-roadmap.md`
- Add: `backlog/tasks/task-472 - Stabilize-Research-Workspace-prerequisite-stack.md`
- Add: `Docs/superpowers/plans/2026-05-23-research-workspace-prerequisite-stack-packaging-plan.md`

- [ ] **Step 1: Cherry-pick the roadmap commits**

Run from the clean worktree:

```bash
git cherry-pick 0fbb2424f 4cc0b7ffd
```

Expected: roadmap design and task tracking apply cleanly.

- [ ] **Step 2: Copy the current packaging task and plan**

Copy only:

```text
backlog/tasks/task-472 - Stabilize-Research-Workspace-prerequisite-stack.md
Docs/superpowers/plans/2026-05-23-research-workspace-prerequisite-stack-packaging-plan.md
```

Expected: the clean branch has active task tracking and this plan.

### Task 3: Package Research Workspace Route Replacement

**Files:**
- Add: `apps/packages/ui/src/components/Option/ResearchWorkspace/**`
- Add: `apps/packages/ui/src/routes/option-research-workspace.tsx`
- Add: `apps/tldw-frontend/pages/research-workspace.tsx`
- Add: `apps/tldw-frontend/extension/routes/option-research-workspace.tsx`
- Add: `apps/test-utils/research-workspace/**`
- Add/modify: `apps/packages/ui/src/routes/route-paths.ts`
- Add/modify: `apps/packages/ui/src/routes/route-metadata.ts`
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Delete: `apps/packages/ui/src/routes/option-workspace-playground.tsx`
- Delete: `apps/tldw-frontend/pages/workspace-playground.tsx`
- Delete: `apps/tldw-frontend/extension/routes/option-workspace-playground.tsx`

- [ ] **Step 1: Copy route replacement files from source checkout**

Bring over the Research Workspace route/component/test-utils files and route registry updates.

- [ ] **Step 2: Exclude unrelated route work**

Do not copy or stage:

```text
apps/packages/ui/src/routes/option-chat-workspace.tsx
apps/tldw-frontend/pages/chat-workspace.tsx
apps/tldw-frontend/extension/routes/option-chat-workspace.tsx
apps/packages/ui/src/store/prototype-workspace.ts
apps/packages/ui/src/types/prototype-workspace.ts
tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py
tldw_Server_API/app/api/v1/schemas/prototype_workspace_schemas.py
```

- [ ] **Step 3: Verify old route is not registered**

Run:

```bash
rg -n "workspace-playground|WorkspacePlayground|Workspace Playground" \
  apps/packages/ui/src/routes apps/tldw-frontend/pages apps/tldw-frontend/extension/routes
```

Expected: no current route registrations or user-facing active route files remain.

### Task 4: Package Telemetry, Prefill, Tutorials, And Extension Route Tests

**Files:**
- Add: `apps/packages/ui/src/utils/research-workspace-telemetry.ts`
- Add: `apps/packages/ui/src/utils/research-workspace-prefill.ts`
- Add: `apps/packages/ui/src/utils/__tests__/research-workspace-telemetry.test.ts`
- Add: `apps/packages/ui/src/utils/__tests__/research-workspace-prefill.test.ts`
- Delete: `apps/packages/ui/src/utils/workspace-playground-telemetry.ts`
- Delete: `apps/packages/ui/src/utils/workspace-playground-prefill.ts`
- Add: `apps/packages/ui/src/tutorials/definitions/research-workspace.ts`
- Modify: `apps/packages/ui/src/tutorials/index.ts`
- Modify: `apps/packages/ui/src/tutorials/registry.ts`
- Modify: `apps/packages/ui/src/assets/locale/en/tutorials.json`
- Modify: `apps/packages/ui/src/public/_locales/en/tutorials.json`
- Add/modify: research workspace route/e2e test files under `apps/tldw-frontend/**` and `apps/extension/**`.

- [ ] **Step 1: Copy renamed telemetry/prefill/test files**

Expected: active UI code uses `research-workspace` names and storage keys.

- [ ] **Step 2: Preserve only one-time legacy import behavior**

Expected: old local storage keys are read only as migration/import sources, not active storage.

- [ ] **Step 3: Verify no route alias exists**

Run:

```bash
rg -n "workspace-playground" apps/packages/ui/src apps/tldw-frontend apps/extension/tests
```

Expected: remaining matches are deleted historical files, explicit legacy migration/import tests, or archived/historical docs only.

### Task 5: Package Legacy Inventory And Server Bootstrap

**Files:**
- Add: `Docs/Design/Research_Workspace_Legacy_Storage_Inventory.md`
- Add: `Docs/superpowers/plans/2026-05-23-research-workspace-legacy-storage-inventory-plan.md`
- Add: `Docs/superpowers/plans/2026-05-23-research-workspace-server-bootstrap-sync-plan.md`
- Add: `apps/packages/ui/src/store/research-workspace-legacy-storage-inventory.ts`
- Add: `apps/packages/ui/src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts`
- Modify: `apps/packages/ui/src/store/workspace.ts`
- Modify: `apps/packages/ui/src/store/workspace-bundle.ts`
- Modify: `apps/packages/ui/src/store/__tests__/workspace-bundle.test.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`

- [ ] **Step 1: Copy legacy inventory and bootstrap files**

Expected: `/research-workspace` can create/upsert a backend workspace and mirror visible local source rows before status fetches.

- [ ] **Step 2: Verify deletion safety gate**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts \
  src/store/__tests__/workspace-bundle.test.ts
```

Expected: inventory tests prove unknown/unmapped content is not deletion-eligible.

### Task 6: Package Backend Migration Protocol API

**Files:**
- Add: `Docs/Design/Research_Workspace_Migration_Protocol_API.md`
- Add: `Docs/superpowers/plans/2026-05-23-research-workspace-migration-protocol-api-plan.md`
- Add: `tldw_Server_API/app/api/v1/endpoints/workspace_migrations.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Add: `tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py`
- Add: `backlog/completed/task-469 - Implement-Research-Workspace-migration-protocol-API.md`

- [ ] **Step 1: Copy backend migration protocol files**

Expected: `/api/v1/workspaces/migrations` routes mount before dynamic workspace IDs.

- [ ] **Step 2: Run migration API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py \
  tldw_Server_API/tests/Workspaces/test_workspaces_api.py \
  tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py \
  -q
```

Expected: all pass.

- [ ] **Step 3: Run Bandit**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/workspace_migrations.py \
  tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  tldw_Server_API/app/api/v1/schemas/workspace_schemas.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  -f json -o /tmp/bandit_research_workspace_prereq_stack.json
```

Expected: no new findings in touched code.

### Task 7: Package Source Status, Trust Panel, And Jobs Follow-Ups

**Files:**
- Add/modify: `Docs/superpowers/plans/2026-05-23-research-workspace-source-status-capabilities-implementation-plan.md`
- Add/modify: `Docs/superpowers/plans/2026-05-23-research-workspace-source-jobs-plan.md`
- Add/modify: `Docs/superpowers/plans/2026-05-23-research-workspace-trust-panel-api-wiring-plan.md`
- Add/modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Add: `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts`
- Add: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceTrustPanel.tsx`
- Add: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceTrustPanel.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx`

- [ ] **Step 1: Copy trust panel/API client files**

Expected: trust panel reads authoritative backend source status and capabilities.

- [ ] **Step 2: Run focused Vitest suite**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts \
  src/components/Option/ResearchWorkspace/__tests__/WorkspaceTrustPanel.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx
```

Expected: all pass.

### Task 8: Full Route And Browser Verification

**Files:**
- Modify as needed only inside the included Research Workspace scope.

- [x] **Step 1: Run route and metadata tests**

Run selected route tests covering `/research-workspace` and the old-route absence.

- [x] **Step 2: Start backend and WebUI**

Use a real backend and a WebUI dev server with `NEXT_PUBLIC_API_URL` pointed at that backend.

- [x] **Step 3: Validate with CDP, not Computer Control**

Open `/research-workspace` via CDP. Verify:
- route loads;
- trust panel appears;
- no redirect from `/workspace-playground`;
- old route returns normal 404;
- source status/capability calls reach the backend.

Verification notes:
- Live FastAPI backend on `127.0.0.1:18001` validated `PUT /api/v1/workspaces/{id}`, `POST /api/v1/workspaces/{id}/sources`, `GET /sources/status`, and `GET /capabilities`.
- Next.js WebUI on `127.0.0.1:3000` validated via Playwright/CDP. `/research-workspace` rendered the header and workspace trust panel; source status and capability API calls returned 200; `/workspace-playground` returned 404 and did not redirect.
- CDP diagnostics surfaced and fixed an AntD `destroyOnClose` deprecation in `WorkspaceHeader`, invalid rich-description paragraph nesting in `EmptyState`, and missing package-level Vitest aliasing for `wxt/browser`.

### Task 9: Finalize Branch

**Files:**
- Modify: `backlog/tasks/task-472 - Stabilize-Research-Workspace-prerequisite-stack.md`

- [x] **Step 1: Run final diff checks**

Run:

```bash
git diff --check
git status --short
```

- [x] **Step 2: Update Backlog task**

Record included file boundaries, excluded unrelated paths, verification commands, known skips, and final summary.

- [ ] **Step 3: Commit in reviewable chunks**

Suggested commits:
1. `docs: add research workspace hard replacement roadmap`
2. `feat: replace workspace playground route with research workspace`
3. `feat: add research workspace migration safety and protocol`
4. `feat: wire research workspace trust and status surfaces`

- [ ] **Step 4: Push and open PR**

Expected: PR targets `main`, includes only the Research Workspace prerequisite stack, and explicitly excludes chat-workspace/prototype/writing changes.
