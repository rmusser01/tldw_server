# Phase 4.3 Prompts DB Decomposition Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. This is a Phase 4 plan. Do not implement it until Phase 2/3 closeout is stable and maintainers accept `Prompts_DB.py` as the first DB decomposition target.

**Goal:** Decompose `tldw_Server_API/app/core/DB_Management/Prompts_DB.py` without changing prompt DB behavior, SQL semantics, public imports, or sync/FTS side effects.

**Architecture:** Preserve `PromptsDatabase` as the compatibility facade. Move only pure helpers first, then low-risk SQL/schema constants, then standalone export helpers. Keep transaction boundaries and CRUD semantics in place until tests prove each slice.

**Tech Stack:** SQLite, pytest, pytest-cov, Bandit

---

## Current Signals

- `Prompts_DB.py` is 2881 lines.
- Public facade: `PromptsDatabase`.
- Public standalone functions:
  - `add_or_update_prompt`
  - `load_prompt_details_for_ui`
  - `export_prompt_keywords_to_csv`
  - `view_prompt_keywords_markdown`
  - `export_prompts_formatted`
- Internal method groups:
  - connection lifecycle and backup
  - query execution and transaction context
  - schema application versions 1 through 5
  - serialization, search text, and keyword normalization
  - FTS update/delete/rebuild helpers
  - sync log and version restore helpers
  - prompt CRUD and keyword CRUD
  - prompt collection CRUD
  - list/search/fetch helpers
  - standalone UI/export compatibility helpers

## Stage 1: Baseline And Guardrails

**Goal:** Record behavior before moving code.
**Success Criteria:** Focused tests pass on the accepted base, and the current public import surface is documented.
**Tests:** Prompt DB focused suite and import checks.
**Status:** Complete

- [ ] Create a clean worktree from the accepted base.
- [ ] Confirm no active dirty work exists in `Prompts_DB.py`.
- [ ] Record current public imports used by tests and app code.
- [ ] Run focused tests:

```bash
source .venv/bin/activate
python3 -m pytest \
  tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py \
  tldw_Server_API/tests/Prompt_Management/test_prompts_interop.py \
  tldw_Server_API/tests/Prompt_Management_NEW/unit/test_prompts_db_deps_lifecycle.py \
  -v
```

- [ ] Run property tests if Hypothesis is available and not too slow:

```bash
source .venv/bin/activate
python3 -m pytest tldw_Server_API/tests/Prompt_Management_NEW/property/test_prompt_properties.py -v
```

- [ ] Do not edit runtime code in this stage.

Status note: clean worktree `codex/phase4-3-prompts-db-decomposition` was created from `origin/dev` at `1016a3b056`. Baseline focused prompt DB tests passed (`28 passed, 2 warnings`), and optional property tests passed (`14 passed, 5 warnings`) before code movement.

Public import surface to preserve:

- `PromptsDatabase`, `DatabaseError`, `InputError`, and `ConflictError` from `Prompts_DB.py`.
- Standalone helpers `add_or_update_prompt`, `load_prompt_details_for_ui`, `export_prompt_keywords_to_csv`, `view_prompt_keywords_markdown`, and `export_prompts_formatted` from `Prompts_DB.py`.
- Compatibility class attributes used by tests, including schema SQL constants and private helper names such as `_serialize_prompt_definition`, `_deserialize_prompt_record`, `build_structured_prompt_searchable_text`, `_normalize_keyword`, and `_normalize_text_for_search`.

## Stage 2: Extract Pure Prompt Helper Functions

**Goal:** Move serialization, deserialization, searchable-text, and normalization helpers out of the large DB class without changing behavior.
**Success Criteria:** `PromptsDatabase` keeps compatibility methods or aliases, and focused tests still pass.
**Tests:** Prompt DB focused suite from Stage 1.
**Status:** Complete

Candidate helper module:

- `tldw_Server_API/app/core/DB_Management/prompts_db_helpers.py`

Candidate helpers:

- `_serialize_prompt_definition`
- `_deserialize_prompt_record`
- `build_structured_prompt_searchable_text`
- `_normalize_keyword`
- `_normalize_text_for_search`

Implementation constraints:

- Keep existing `PromptsDatabase` method names available.
- Do not change JSON output for structured prompt definitions.
- Do not change keyword normalization or FTS searchable text.
- Do not move SQL execution or transaction boundaries in this stage.

Status note: extracted pure helpers to `prompts_db_helpers.py` and kept `PromptsDatabase` compatibility aliases. Added direct helper coverage in `test_prompts_db_helpers.py`; helper tests, the original focused prompt DB suite, optional property tests, and touched-scope Bandit pass after extraction.

## Stage 3: Extract Schema Constants And Migration Helpers

**Goal:** Reduce file size by moving schema SQL constants and schema application helpers without changing schema version behavior.
**Success Criteria:** Schema initialization, v1 migration, v2 collections, v3 prompt definitions, v4 namespaces, and v5 tags still pass focused tests.
**Tests:** Prompt DB focused suite plus migration tests.
**Status:** Not Started

Candidate helper module:

- `tldw_Server_API/app/core/DB_Management/prompts_db_schema.py`

Candidate moved content:

- `_TABLES_SQL_V1`
- `_INDICES_SQL_V1`
- `_TRIGGERS_SQL_V1`
- `_FTS_TABLES_SQL`
- `_COLLECTIONS_SQL_V2`
- schema application helpers, only after tests cover them

Implementation constraints:

- Keep `PromptsDatabase._CURRENT_SCHEMA_VERSION` on the facade unless maintainers approve moving it.
- Preserve direct class attribute access used by tests, or provide compatibility aliases.
- Do not alter SQL text except import boundaries.
- Do not combine schema movement with CRUD movement.

## Stage 4: Extract Standalone UI And Export Helpers

**Goal:** Move bottom-of-file standalone helpers while preserving existing imports from `Prompts_DB.py`.
**Success Criteria:** Existing callers can still import helper names from `Prompts_DB.py`, and interop/export tests pass.
**Tests:** `test_prompts_db_v2.py` standalone tests and `test_prompts_interop.py`.
**Status:** Not Started

Candidate helper module:

- `tldw_Server_API/app/core/DB_Management/prompts_db_exports.py`

Candidate moved functions:

- `add_or_update_prompt`
- `load_prompt_details_for_ui`
- `export_prompt_keywords_to_csv`
- `view_prompt_keywords_markdown`
- `export_prompts_formatted`

Implementation constraints:

- Re-export the same names from `Prompts_DB.py`.
- Preserve status strings and export file formats.
- Do not change filesystem paths used by export tests.

## Stage 5: Optional Row Mapping And FTS Cleanup

**Goal:** Only after earlier stages are stable, consider extracting row conversion and FTS helpers.
**Success Criteria:** Search, keyword update, version restore, and sync log tests remain stable.
**Tests:** Full prompt DB focused suite plus property tests.
**Status:** Not Started

Candidate areas:

- FTS update/delete/rebuild helpers.
- Sync log payload assembly.
- Prompt version payload deserialization.

Implementation constraints:

- Do not move transaction boundaries in the same PR as FTS helper movement.
- Do not change sync payload shape.
- Do not change FTS query syntax or match behavior.

## Verification

Focused tests:

```bash
source .venv/bin/activate
python3 -m pytest \
  tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py \
  tldw_Server_API/tests/Prompt_Management/test_prompts_interop.py \
  tldw_Server_API/tests/Prompt_Management_NEW/unit/test_prompts_db_deps_lifecycle.py \
  -v
```

Optional property tests:

```bash
source .venv/bin/activate
python3 -m pytest tldw_Server_API/tests/Prompt_Management_NEW/property/test_prompt_properties.py -v
```

Touched-scope Bandit:

```bash
source .venv/bin/activate
python3 -m bandit -r tldw_Server_API/app/core/DB_Management/Prompts_DB.py tldw_Server_API/app/core/DB_Management/prompts_db_helpers.py tldw_Server_API/app/core/DB_Management/prompts_db_schema.py tldw_Server_API/app/core/DB_Management/prompts_db_exports.py -f json -o /tmp/bandit_phase4_3_prompts_db.json
```

## Out Of Scope

- Changing prompt schema versions.
- Changing prompt sync log payloads.
- Changing transaction boundaries.
- Replacing `PromptsDatabase` with a new public class.
- Migrating to `Prompts_DB_V2`.
- Raising coverage thresholds.

## Handoff Checklist

- [x] Maintainers accept `Prompts_DB.py` as the first Phase 4.3 DB decomposition target.
- [x] Clean worktree from accepted base exists.
- [x] Stage 1 focused tests pass before code movement.
- [x] Public import compatibility is preserved after each stage.
- [x] Bandit is run on touched source before PR handoff.
