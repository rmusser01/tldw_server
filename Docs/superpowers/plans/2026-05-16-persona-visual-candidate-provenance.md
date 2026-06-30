# Persona Visual Candidate Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist and return bounded, trace-safe provenance metadata for Persona Visual generated candidates so review screens can explain where a candidate came from without exposing raw prompts or provider secrets.

**Architecture:** Add one bounded provenance helper in the Persona core layer, store normalized JSON on `persona_visual_candidates`, and return it through the existing candidate response contract. Keep generation review-gated; do not activate packs automatically, add renderer behavior, or introduce MCP provider execution.

**Tech Stack:** Python, FastAPI/Pydantic, SQLite/PostgreSQL schema migrations, existing ChaChaNotesDB Persona state store, pytest.

---

### Task 1: Add Candidate Provenance Storage Contract

**Files:**
- Create: `tldw_Server_API/app/core/Persona/visual_candidate_provenance.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py`

- [x] **Step 1: Write failing DB round-trip tests**

Add a test that creates a candidate with `generation_provenance`, fetches it back through `list_persona_visual_candidates`, and asserts the returned dict includes bounded normalized provenance.

Add a second assertion that unsafe/unbounded text in provenance is redacted or truncated and the raw unsafe string is not returned.

- [x] **Step 2: Run focused DB test to verify RED**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py::test_candidate_generation_provenance_round_trip -q --tb=short --disable-warnings`

Expected: FAIL because `generation_provenance` is not accepted or returned yet.

- [x] **Step 3: Implement schema v47 and store normalization**

Add `generation_provenance_json TEXT NOT NULL DEFAULT '{}'` to `persona_visual_candidates` for fresh SQLite/PostgreSQL schemas, add v46-to-v47 migrations, wire both migration paths, and bump `_CURRENT_SCHEMA_VERSION`.

Add a Persona core helper that normalizes allowed provenance keys and omits or redacts unsafe text. Use it before insert and when decoding candidate rows.

- [x] **Step 4: Run focused DB test to verify GREEN**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py::test_candidate_generation_provenance_round_trip -q --tb=short --disable-warnings`

Expected: PASS.

### Task 2: Capture Generation Job Provenance

**Files:**
- Modify: `tldw_Server_API/app/core/Persona/visual_jobs_worker.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_jobs.py`

- [x] **Step 1: Write failing worker test**

Extend the generated-candidate worker test so a recipe-backed job stores provenance with:
- `schema_version`
- `generation_mode`
- `request_id`
- `job_id`
- `backend`
- `target_state`
- recipe summary fields such as `starter_pack_id`, `recipe_output`, `correlation_id`, and review checks

Assert raw prompt/user prompt text is not copied into `generation_provenance`.

- [x] **Step 2: Run focused worker test to verify RED**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_jobs.py::test_generation_worker_stores_generated_asset_and_candidate -q --tb=short --disable-warnings`

Expected: FAIL because stored candidates have no provenance yet.

- [x] **Step 3: Build provenance in the worker**

Use the new helper to build provenance from safe job payload fields and recipe intent. Pass it to `create_persona_visual_candidate` when `_persist_generated_candidate` stores the review candidate.

- [x] **Step 4: Run focused worker test to verify GREEN**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_jobs.py::test_generation_worker_stores_generated_asset_and_candidate -q --tb=short --disable-warnings`

Expected: PASS.

### Task 3: Return Provenance Through API And Docs

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
- Test: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`

- [x] **Step 1: Write failing API response test**

Extend generated-candidate list/detail coverage to create a candidate with provenance and assert the serialized response includes `generation_provenance` without raw unsafe metadata.

- [x] **Step 2: Run focused API test to verify RED**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py::test_list_generated_candidates_returns_preview_asset_urls -q --tb=short --disable-warnings`

Expected: FAIL because `PersonaVisualCandidateResponse` does not include the field yet.

- [x] **Step 3: Expose the response field and document it**

Add `generation_provenance` to `PersonaVisualCandidateResponse`, populate it in `_persona_visual_candidate_to_response`, and update the Persona Visual Packs documentation with the trace-safe provenance contract.

- [x] **Step 4: Run focused API test to verify GREEN**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py::test_list_generated_candidates_returns_preview_asset_urls -q --tb=short --disable-warnings`

Expected: PASS.

### Task 4: Verification And Closeout

**Files:**
- Modify: `backlog/tasks/task-409 - Add-Persona-Visual-generated-candidate-provenance-review-metadata.md`

- [x] **Step 1: Run focused Persona Visual regression tests**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py tldw_Server_API/tests/Persona/test_persona_visual_jobs.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q --tb=short --disable-warnings`

Expected: PASS.

- [x] **Step 2: Run syntax, whitespace, and Bandit checks**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_candidate_provenance.py tldw_Server_API/app/core/Persona/visual_jobs_worker.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/api/v1/endpoints/persona.py`

Run:
`git diff --check`

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_candidate_provenance.py tldw_Server_API/app/core/Persona/visual_jobs_worker.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/api/v1/endpoints/persona.py -f json -o /tmp/bandit_persona_visual_candidate_provenance.json`

Expected: all commands exit 0 with no new findings in touched code.

- [x] **Step 3: Update task and commit**

Update `TASK-409` with touched files, verification results, and the PR/issue relationship. Commit the implementation with a message that references #1782.
