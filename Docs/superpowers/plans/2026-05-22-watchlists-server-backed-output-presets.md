# Watchlists Server-Backed Output Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]` / `- [x]`) syntax for tracking.

**Goal:** Add durable per-user Watchlists presets for monitor output, delivery, and audio configuration inside `/watchlists`.

**Architecture:** Persist presets in the Watchlists DB as user-scoped records containing a full `output_prefs` JSON object. Expose CRUD plus an apply endpoint that overlays preset-controlled output fields onto a base `output_prefs` object while preserving unknown advanced keys. The WebUI loads presets in `JobFormModal`, lets users save/update/apply/delete them, and never changes monitor cadence, source scope, filters, or dedupe identity when a preset is applied.

**Tech Stack:** FastAPI, Pydantic, `WatchlistsDatabase`, SQLite/PostgreSQL-compatible SQL, React, Ant Design, Vitest, pytest.

---

## File Structure

- Modify `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
  - Add `watchlist_output_presets` schema for SQLite/PostgreSQL.
  - Add `WatchlistOutputPresetRow` dataclass.
  - Add CRUD helpers and server-side apply/merge helper.
- Modify `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
  - Add output-preset request/response models.
- Modify `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Add row projection and CRUD/apply routes before `/{watchlist_id}` routes.
- Modify `tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py`
  - Add DB coverage for user-scoped presets, default behavior, deletion, validation, and apply merge preservation.
- Modify `tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py`
  - Add API coverage for CRUD/apply behavior.
- Modify `apps/packages/ui/src/types/watchlists.ts`
  - Add preset DTO/request types.
- Modify `apps/packages/ui/src/services/watchlists.ts`
  - Add preset CRUD/apply service calls.
- Modify `apps/packages/ui/src/services/__tests__/watchlists-items-triage.test.ts`
  - Add service path/method/body tests.
- Create `apps/packages/ui/src/components/Option/Watchlists/JobsTab/output-presets.ts`
  - Add frontend helper for applying preset prefs to a base object and preserving unknown fields.
- Create `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/output-presets.test.ts`
  - Add helper tests mirroring backend merge behavior.
- Modify `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx`
  - Load presets, preserve raw output prefs base, and add save/update/apply/delete UI.
- Modify `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx`
  - Add modal interaction tests for loading, save, apply, update, delete, and payload preservation.

---

### Task 1: Backend Preset Persistence

**Files:**
- Modify `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
- Test `tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py`

- [x] **Step 1: Write failing DB tests**

Add tests covering:
- creating a per-user preset with `name`, `description`, `output_prefs`, `is_default`;
- listing only the current user's presets;
- default preset exclusivity for a user;
- update and delete;
- apply merge removes known output/audio/delivery fields from the base, applies preset fields, and preserves unknown fields.

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py::test_output_presets_are_user_scoped_and_validated tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py::test_output_preset_apply_preserves_unknown_fields -q
```

Expected: fail because DB methods do not exist.

- [x] **Step 3: Implement persistence**

Add:
- schema table `watchlist_output_presets`;
- index `idx_output_presets_user_updated`;
- unique index `ux_output_presets_user_name`;
- dataclass `WatchlistOutputPresetRow`;
- `_normalize_output_preset_name`;
- `_normalize_output_prefs_payload`;
- `_check_output_preset_name_available`;
- `create_output_preset`;
- `get_output_preset`;
- `list_output_presets`;
- `update_output_preset`;
- `delete_output_preset`;
- `apply_output_preset`.

- [x] **Step 4: Run DB tests to verify GREEN**

Run the same pytest command. Expected: pass.

---

### Task 2: Backend API

**Files:**
- Modify `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Modify `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Test `tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py`

- [x] **Step 1: Write failing API tests**

Add endpoint tests for:
- `GET /api/v1/watchlists/job-output-presets`;
- `POST /api/v1/watchlists/job-output-presets`;
- `PATCH /api/v1/watchlists/job-output-presets/{preset_id}`;
- `POST /api/v1/watchlists/job-output-presets/{preset_id}/apply`;
- `DELETE /api/v1/watchlists/job-output-presets/{preset_id}`;
- 404 on another user's/nonexistent preset is not exposed.

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py::test_output_presets_endpoint_crud_and_apply -q
```

Expected: fail with 404 or missing schema imports.

- [x] **Step 3: Implement schemas and routes**

Add Pydantic models:
- `WatchlistOutputPresetCreate`;
- `WatchlistOutputPresetUpdate`;
- `WatchlistOutputPreset`;
- `WatchlistOutputPresetsList`;
- `WatchlistOutputPresetApplyRequest`;
- `WatchlistOutputPresetApplyResponse`.

Add API routes under `/watchlists/job-output-presets`.

- [x] **Step 4: Run API tests to verify GREEN**

Run the same pytest command. Expected: pass.

---

### Task 3: Frontend Services and Merge Helper

**Files:**
- Modify `apps/packages/ui/src/types/watchlists.ts`
- Modify `apps/packages/ui/src/services/watchlists.ts`
- Modify `apps/packages/ui/src/services/__tests__/watchlists-items-triage.test.ts`
- Create `apps/packages/ui/src/components/Option/Watchlists/JobsTab/output-presets.ts`
- Create `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/output-presets.test.ts`

- [x] **Step 1: Write failing frontend tests**

Add tests for:
- service paths and methods for preset CRUD/apply;
- helper apply behavior preserving unknown fields and replacing known output/audio/delivery fields.

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
bun run test -- src/services/__tests__/watchlists-items-triage.test.ts src/components/Option/Watchlists/JobsTab/__tests__/output-presets.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: fail because types/services/helper do not exist.

- [x] **Step 3: Implement types/services/helper**

Add type-safe DTOs and service functions. Keep helper semantics aligned with backend apply:
- known output fields are replaced by preset values;
- unknown raw fields remain unless preset explicitly provides the same key.

- [x] **Step 4: Run frontend service/helper tests to verify GREEN**

Run the same Vitest command. Expected: pass.

---

### Task 4: Job Form UI

**Files:**
- Modify `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx`
- Modify `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx`

- [x] **Step 1: Write failing modal tests**

Add tests for:
- preset list loads when modal opens;
- save current setup creates a server preset;
- apply selected preset changes output/audio/delivery state and preserves unknown output prefs on submit;
- update selected preset persists current setup;
- delete selected preset removes it from the list.

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
bun run test -- src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: fail because modal controls do not exist.

- [x] **Step 3: Implement modal controls**

Add a "Saved output presets" panel near guided presets with:
- preset selector;
- preset name input;
- Apply;
- Save current setup;
- Update selected;
- Delete selected.

Implementation requirements:
- keep existing guided presets;
- keep Basic/Advanced behavior;
- do not touch scope, filters, schedule, source rules, or dedupe settings;
- show an inline load/apply error state instead of failing silently;
- preserve advanced/raw output prefs using a raw base state.

- [x] **Step 4: Run modal tests to verify GREEN**

Run the same Vitest command. Expected: pass.

---

### Task 5: Verification and Cleanup

**Files:**
- Update `backlog/tasks/task-488 - Implement-Watchlists-server-backed-output-delivery-audio-presets.md`

- [x] **Step 1: Run backend focused tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py -q
```

- [x] **Step 2: Run frontend focused tests**

```bash
bun run test -- src/services/__tests__/watchlists-items-triage.test.ts src/components/Option/Watchlists/JobsTab/__tests__/output-presets.test.ts src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx --maxWorkers=1 --no-file-parallelism
```

- [x] **Step 3: Run Bandit on touched backend files**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/DB_Management/Watchlists_DB.py tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py -f json -o /tmp/bandit_watchlists_output_presets.json
```

- [x] **Step 4: Update Backlog task**

Record implementation summary, files touched, test results, and any known skips.

- [x] **Step 5: Self-review diff**

Run `git diff --stat` and inspect the diff for route ordering, preservation semantics, and test coverage gaps before final response or PR creation.
