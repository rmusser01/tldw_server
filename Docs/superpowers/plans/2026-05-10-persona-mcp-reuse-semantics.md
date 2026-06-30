# Persona MCP Reuse Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing internal `persona_visuals` MCP module so same-user personal library entries can be listed and reused as reviewable target-persona drafts without bypassing ownership or activation gates.

**Architecture:** Keep the MCP implementation as a thin tool layer over the existing ChaChaNotes persona store and `PersonaVisualLibraryService`. Add read-only library discovery plus one durable reuse action that delegates to the same duplicate-to-persona path used by the REST API. Do not introduce snapshots, shared-library semantics, marketplace behavior, or automatic activation.

**Tech Stack:** FastAPI backend models, ChaChaNotes persona store, `PersonaVisualLibraryService`, MCP Unified module tools, pytest.

---

### Task 1: Add MCP Library Discovery

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py`

- [x] **Step 1: Write the failing test**

Add a test that creates two personas, a visual pack, saves it to the personal library, then calls `persona_visuals.library_items`. Assert the result includes `items`, `count`, reference-backed source ids, live source title/persona fields, `source_available`, and `source_changed`.

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py -q
```

Expected: FAIL because `persona_visuals.library_items` is not registered.

- [x] **Step 3: Implement the minimal tool**

Add `persona_visuals.library_items` to `get_tools`, `execute_tool`, and `validate_tool_arguments`. Implement `_library_items_sync` using `db.list_persona_visual_library_items(user_id=..., include_deleted=False, limit=..., offset=...)` and a small `_library_item_summary` helper.

- [x] **Step 4: Run the test to verify it passes**

Run the same focused pytest command and confirm the new test passes with existing tests.

### Task 2: Add MCP Library Reuse

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py`

- [x] **Step 1: Write the failing success-path test**

Add a test for `persona_visuals.use_library_item` that saves a source pack to the library, calls the tool with `item_id` and `target_persona_id`, and asserts the returned pack is a draft on the target persona with `review_required` true and no active target pack.

- [x] **Step 2: Write the failing rejection-path tests**

Add tests for a missing library item, an unavailable source pack, and missing target persona context. The source-unavailable path should soft-delete the source pack and assert the tool raises a clear failure instead of creating a draft.

- [x] **Step 3: Run the tests to verify they fail**

Run the focused MCP pytest command. Expected failures should be unknown tool or missing behavior, not fixture/setup errors.

- [x] **Step 4: Implement the reuse tool**

Register `persona_visuals.use_library_item`, validate `item_id`, optional `target_persona_id`, and optional `title`, then call `PersonaVisualLibraryService(db).use_item_for_persona(...)`. Return the target `pack` summary plus `library_item_id` and `review_required: true`.

- [x] **Step 5: Run the tests to verify they pass**

Run the focused MCP pytest command. Confirm existing runtime/draft/generation tool tests still pass.

### Task 3: Document Tool Semantics

**Files:**
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
- Modify: `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
- Modify: `backlog/tasks/task-218 - Add-reusable-visual-pack-MCP-tool-semantics.md`

- [x] **Step 1: Update docs**

Document `persona_visuals.library_items` as read-only personal-library discovery and `persona_visuals.use_library_item` as a draft-creating durable action. Clarify that library items remain reference-backed and that live source names are derived from source rows, not snapshots.

- [x] **Step 2: Update task status**

Check completed TASK-218 acceptance criteria and add verification notes.

- [x] **Step 3: Run verification**

Run focused pytest, `git diff --check`, and Bandit on the touched backend module:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py -q
python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py -f json -o /tmp/bandit_persona_mcp_reuse.json
git diff --check
```

- [x] **Step 4: Commit**

Stage the plan, task record, backend module, tests, and docs. Commit with:

```bash
git commit -m "feat: add persona visual library MCP reuse"
```
