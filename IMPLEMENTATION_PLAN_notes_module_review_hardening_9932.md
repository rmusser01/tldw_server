## Stage 1: Regression Tests
**Goal**: Capture the reviewed Notes Studio and search safety issues before implementation.
**Success Criteria**: Tests fail for stale Studio regenerate, invalid diagram section IDs, bounded request inputs, bounded keyword tokens, and workflow cleanup.
**Tests**:
- `tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_service.py`
- `tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py`
- `tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py`
- `tldw_Server_API/tests/Workflows/adapters/test_knowledge_adapters.py`
**Status**: Complete

## Stage 2: Notes Studio Safety Fixes
**Goal**: Prevent scope broadening, stale editor overwrites, and diagram sidecar clobbering.
**Success Criteria**: Studio regenerate requires caller concurrency state, invalid diagram section IDs are rejected, and diagram updates only mutate the manifest after generation.
**Tests**: Targeted Notes Studio unit and API tests pass.
**Status**: Complete

## Stage 3: Input Bounds and Resource Cleanup
**Goal**: Bound request/input fan-out and ensure workflow-created NotesInteropService instances are closed.
**Success Criteria**: Studio schemas enforce size/cardinality limits, keyword search rejects excessive tokens, and workflow notes CRUD closes cached DB handles.
**Tests**: Targeted Notes API and workflow adapter tests pass.
**Status**: Complete

## Stage 4: Verification and Finalization
**Goal**: Verify touched scope and record Backlog results.
**Success Criteria**: Targeted tests and Bandit on touched code complete with results captured in `TASK-9932`.
**Tests**:
- `python -m pytest ...` targeted commands
- `python -m bandit -r <touched_paths> -f json -o /tmp/bandit_notes_module_review_hardening_9932.json`
**Status**: Complete
