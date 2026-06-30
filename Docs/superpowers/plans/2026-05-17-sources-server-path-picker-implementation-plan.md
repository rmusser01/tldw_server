# Sources Server Path Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe server-side directory picker to the Sources local-directory path field.

**Architecture:** The backend owns filesystem visibility and only lists configured ingestion source allowed roots plus immediate child directories under those roots. The frontend consumes that contract through the shared tldw API client and renders a compact picker modal beside the existing manual path input.

**Tech Stack:** FastAPI, Pydantic, pathlib, React, Ant Design, TanStack Query, Vitest, pytest, Bandit.

---

## Task 1: Backend Browse Contract

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/ingestion_sources.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py`
- Test: `tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_path_browser.py`

- [x] **Step 1: Write failing API tests**

Cover root listing, child directory listing, outside-root rejection, file-path rejection, and permission-error tolerance.

Run:

```bash
.venv/bin/python -m pytest tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_path_browser.py -q
```

Expected: fail because the endpoint and schemas do not exist.

- [x] **Step 2: Add response schemas**

Add `IngestionSourceDirectoryEntryResponse` and `IngestionSourceDirectoryBrowseResponse`.

- [x] **Step 3: Implement browse helpers and endpoint**

Add `GET /api/v1/ingestion-sources/browse-directories?path=...` before `/{source_id}` routes. Use `get_ingestion_source_allowed_roots()`, `resolve_safe_local_path()`, `Path.iterdir()`, and no recursive traversal. Return `roots`, `current_path`, `parent_path`, and `entries`.

- [x] **Step 4: Run backend tests**

Run the focused path-browser tests and relevant ingestion source policy tests.

## Task 2: Shared Client and Hook

**Files:**
- Modify: `apps/packages/ui/src/types/ingestion-sources.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/collections.ts`
- Modify: `apps/packages/ui/src/hooks/use-ingestion-sources.ts`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.ingestion-sources.test.ts`
- Test: `apps/packages/ui/src/hooks/__tests__/use-ingestion-sources.test.tsx`

- [x] **Step 1: Write failing client/hook tests**

Assert `browseIngestionSourceDirectories()` calls `/api/v1/ingestion-sources/browse-directories` and that the hook passes path parameters through.

- [x] **Step 2: Add types, client method, and query hook**

Add browse response types and `useIngestionSourceDirectoryBrowseQuery(path, options)`.

- [x] **Step 3: Run focused frontend tests**

Run the client and hook tests from `apps/packages/ui`.

## Task 3: SourceForm Picker UI

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Sources/SourceForm.tsx`
- Test: `apps/packages/ui/src/components/Option/Sources/__tests__/SourceForm.test.tsx`

- [x] **Step 1: Write failing SourceForm tests**

Assert Browse opens a modal, roots/child directories render, selecting a directory updates `Server directory path`, and manual entry still works.

- [x] **Step 2: Implement the picker modal**

Use a small Ant Design modal/list next to the path input. Keep the existing manual field and help text. Show loading, empty-state, and error states without exposing arbitrary filesystem browsing.

- [x] **Step 3: Run SourceForm tests**

Run the focused SourceForm test file.

## Task 4: Verification and Closeout

**Files:**
- Modify: `backlog/tasks/task-406 - Add-Sources-server-path-picker.md`
- Modify: this plan

- [x] **Step 1: Run focused backend tests**

```bash
.venv/bin/python -m pytest tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_path_browser.py tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_access_policy.py tldw_Server_API/tests/Ingestion_Sources/unit/test_access_policy.py -q
```

Result: `35 passed, 5 warnings`.

- [x] **Step 2: Run focused frontend tests**

```bash
bunx vitest run src/services/__tests__/tldw-api-client.ingestion-sources.test.ts src/hooks/__tests__/use-ingestion-sources.test.tsx src/components/Option/Sources/__tests__/SourceForm.test.tsx
```

Result: `3 passed (3), 27 tests passed`.

- [x] **Step 3: Run Bandit on touched backend endpoint/schema scope**

```bash
.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py tldw_Server_API/app/api/v1/schemas/ingestion_sources.py -f json -o /tmp/bandit_sources_path_picker.json
```

Result: exit 0, no findings.

- [x] **Step 4: Browser QA `/sources/new?preset=notes-folder-sync`**

In-app Browser navigation to the isolated localhost dev server was blocked by the app local-network path, so rendered QA used Playwright from the workspace.

Environment:
- Backend: `http://127.0.0.1:18002`
- Frontend: `http://127.0.0.1:18080/sources/new?preset=notes-folder-sync`
- Allowed root: `/private/tmp/tldw-sources-picker-root`

Checks:
- Desktop viewport: picker opened, root expanded, child directories listed, selecting `notes` wrote `/private/tmp/tldw-sources-picker-root/notes` into the path field.
- Mobile viewport `390x844`: modal fit within viewport bounds (`x=8`, `width=374`) and child folder controls remained visible.
- Console/page errors: none after replacing the deprecated Ant Design `List` component with semantic list markup.

- [x] **Step 5: Update Backlog task and prepare commit**
