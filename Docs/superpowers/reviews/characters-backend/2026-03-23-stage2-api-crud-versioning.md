# Stage 2 API, CRUD, Versioning, and Restore Integrity

## Scope
- Endpoint contract and persistence review for create, list, query, get, update, delete, restore, version history, version diff, revert, search, and export flows under the Characters backend.
- Focused on lifecycle integrity, optimistic concurrency, soft-delete/restore semantics, snapshot/version behavior, and whether endpoint responses match DB behavior.
- Analysis-only stage. No source code changes were made.

## Code Paths Reviewed
- `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
  - `_get_characters_restore_retention_days()` at `:256`
  - `_build_character_revert_payload()` at `:285`
  - `_build_character_version_entry()` at `:307`
  - `_build_character_version_diff_fields()` at `:335`
  - `list_all_characters()` at `:909`
  - `query_characters()` at `:928`
  - `create_new_character_endpoint()` at `:1009`
  - `get_character_by_id_endpoint()` at `:1547`
  - `get_character_versions_diff_endpoint()` at `:1573`
  - `get_character_versions_endpoint()` at `:1659`
  - `revert_character_to_version_endpoint()` at `:1691`
  - `update_character_endpoint()` at `:1771`
  - `delete_character_endpoint()` at `:1819`
  - `restore_character_endpoint()` at `:1862`
  - `search_characters_endpoint()` at `:1929`
  - `export_character()` at `:2985`
- `tldw_Server_API/app/core/Character_Chat/modules/character_db.py`
  - `_prepare_character_data_for_db_storage()` at `:30`
  - `create_new_character_from_data()` at `:183`
  - `get_character_details()` at `:212`
  - `update_existing_character_details()` at `:225`
  - `delete_character_from_db()` at `:269`
  - `restore_character_from_db()` at `:285`
  - `search_characters_by_query_text()` at `:311`
- `tldw_Server_API/app/core/Character_Chat/Character_Chat_Lib_facade.py`
  - compatibility re-exports for the modular character CRUD helpers
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - `character_cards` schema at `:668` and character sync-log triggers at `:1213-1268`
  - `_get_current_db_version()` at `:17190`
  - `add_character_card()` at `:17327`
  - `get_character_card_by_id()` at `:17451`
  - `get_character_card_by_name()` at `:17478`
  - `query_character_cards()` at `:17548`
  - `_normalize_character_tags_for_operation()` at `:17745`
  - `update_character_card()` at `:17919`
  - `soft_delete_character_card()` at `:18074`
  - `restore_character_card()` at `:18170`
  - `search_character_cards()` at `:18326`
  - `get_character_version_history()` at `:30792`

### Lifecycle map
- Create: endpoint `POST /api/v1/characters/` -> `create_new_character_from_data()` -> `add_character_card()`
- List/query/get/search: endpoint direct DB reads via `list_character_cards()`, `query_character_cards()`, `get_character_card_by_id()`, `search_character_cards()`
- Update: endpoint `PUT /{character_id}` -> `update_existing_character_details()` -> `update_character_card()`
- Delete: endpoint `DELETE /{character_id}` -> `delete_character_from_db()` -> `soft_delete_character_card()`
- Restore: endpoint `POST /{character_id}/restore` -> `restore_character_from_db()` -> `restore_character_card()`
- Versions/diff: endpoints -> `get_character_version_history()` -> sync-log snapshots
- Revert: endpoint `POST /{character_id}/revert` -> `_build_character_revert_payload()` -> `update_existing_character_details()` -> `update_character_card()`
- Export: endpoint `GET /{character_id}/export` reads current active character only via `get_character_details()`

## Tests Reviewed
- `tldw_Server_API/tests/Characters/test_characters_endpoint.py`
  - Main invariants asserted: endpoint CRUD happy paths, query filters, version list/diff, revert happy path, restore happy path/version conflict/retention expiry, and API property tests for create/update payload handling.
  - Lifecycle branches covered: create, get, list, query, update, delete, restore, version history, version diff, revert, search.
  - Assessment: `strong` for endpoint happy paths and several conflict branches; `weak` for empty-update optimistic locking, restore-on-active-row, and image/version-history semantics.
- `tldw_Server_API/tests/Characters/test_character_functionality_db.py`
  - Main invariants asserted: DB create/update/delete/search behavior, uniqueness, version bumps, sync-log emission, deleted-row exclusion, idempotent soft delete.
  - Lifecycle branches covered: create, get, list, update, delete, search.
  - Assessment: `strong` for core DB CRUD and sync logging; `weak` for Stage 2 restore/version-history because `restore_character_card()` and `get_character_version_history()` are not directly exercised here.
- `tldw_Server_API/tests/Characters/test_character_chat_lib.py`
  - Main invariants asserted: payload normalization, import parsing, placeholder behavior, and one direct create helper test for wrapped base64 images.
  - Lifecycle branches covered: create helper normalization only.
  - Assessment: `weak` for Stage 2 lifecycle integrity; it does not materially protect update/delete/restore/versioning behavior.
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py`
  - Main invariants asserted: top-level API CRUD integration, update/delete version conflicts, list/search/filter behavior.
  - Lifecycle branches covered: create, get, list, update, delete, search/filter.
  - Assessment: `medium`; useful integration smoke coverage, but it does not cover query pagination semantics deeply and does not cover versions/diff/revert/restore.
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_cards_fts_bootstrap.py`
  - Main invariants asserted: schema-seeded default character can be updated and is indexed in FTS on a fresh DB.
  - Lifecycle branches covered: DB update + FTS bootstrap regression.
  - Assessment: `strong but narrow`.
- `tldw_Server_API/tests/ChaChaNotesDB/test_character_card_tag_search.py`
  - Main invariants asserted: tag search matches expected rows, is case-insensitive, excludes deleted cards, and validates inputs.
  - Lifecycle branches covered: search plus deleted-row exclusion after soft delete.
  - Assessment: `medium`; useful for search semantics, but mostly outside the versioning/restore surface.

## Validation Commands
- Lifecycle map:
```bash
rg -n "@router\\.(get|post|put|delete)|def _build_character_|def _get_characters_restore_retention_days|def .*character" \
  tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py \
  tldw_Server_API/app/core/Character_Chat/Character_Chat_Lib_facade.py \
  tldw_Server_API/app/core/Character_Chat/modules/character_db.py
```
- DB trace:
```bash
rg -n "add_character_card|get_character_card_by_|query_character_cards|update_character_card|soft_delete_character_card|restore_character_card|get_character_version_history" \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/Character_Chat/modules/character_db.py \
  tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py
```
- Completed targeted pytest runs:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Characters/test_characters_endpoint.py -v
python -m pytest \
  tldw_Server_API/tests/Characters/test_character_functionality_db.py \
  tldw_Server_API/tests/Characters/test_character_chat_lib.py -v
python -m pytest \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_cards_fts_bootstrap.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_character_card_tag_search.py -v
```
- Direct DB repros used to confirm two findings:
```bash
source .venv/bin/activate
python - <<'PY'
import pathlib
import tempfile
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

with tempfile.TemporaryDirectory() as tmpdir:
    db_path = pathlib.Path(tmpdir) / "stage2_characters.db"
    db = CharactersRAGDB(str(db_path), client_id="stage2-review")
    character_id = db.add_character_card({"name": "Stage2 Review Character"})
    print("empty_update_bypass", db.update_character_card(character_id, {}, expected_version=999))
    print("restore_active_row", db.restore_character_card(character_id, expected_version=999))
    db.close_connection()
PY
```
  - Expected review outputs:
    - `empty_update_bypass True`
    - `restore_active_row True`

### Validation results
- `tldw_Server_API/tests/Characters/test_characters_endpoint.py`: `48 passed` in `780.33s`
- `tldw_Server_API/tests/Characters/test_character_functionality_db.py` + `tldw_Server_API/tests/Characters/test_character_chat_lib.py`: `190 passed` in `22.94s`
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py` + `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_cards_fts_bootstrap.py` + `tldw_Server_API/tests/ChaChaNotesDB/test_character_card_tag_search.py`: `48 passed, 3 skipped` in `335.15s`
- The three skipped tests were existing skips in `test_character_api.py` and did not block the targeted lifecycle validation.

## Findings
- High | correctness | `restore_character_card()` accepts an invalid restore request for an already-active row and returns success without checking `expected_version`.
  - The DB restore path checks whether the row exists, then immediately returns success when `deleted == 0` at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:18220-18224`.
  - That early return happens before any optimistic-concurrency check, so `restore_character_card(character_id, expected_version=999)` succeeds on an active row.
  - The endpoint exposes that behavior directly because `restore_character_endpoint()` delegates to the DB restore helper without first verifying the row is deleted or that the supplied version matches an active tombstone at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:1876-1896`.
  - Direct DB repro confirmed the issue: an active row restored with `expected_version=999` returned `True` and remained at the original version.
  - Impact: callers can receive `200 OK` for a logically invalid restore request, which weakens optimistic locking and makes restore semantics inconsistent with the endpoint contract text.

- Medium | correctness | Empty updates bypass optimistic concurrency entirely.
  - `CharacterUpdate` allows an entirely empty payload at `tldw_Server_API/app/api/v1/schemas/character_schemas.py:141-142`.
  - The endpoint serializes with `exclude_unset=True` and passes the resulting dict to `update_existing_character_details()` at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:1790-1792`.
  - The facade forwards empty payloads to `db.update_character_card(character_id, {}, expected_version)` at `tldw_Server_API/app/core/Character_Chat/modules/character_db.py:237-242`.
  - The DB method then returns `True` immediately for empty `card_data` with no version check and no metadata change at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:17924-17928`.
  - Direct DB repro confirmed the issue: `update_character_card(..., {}, expected_version=999)` returns `True` on an active row and leaves the version unchanged.
  - Impact: an API caller can submit `{}` with a stale or fabricated `expected_version` and still receive a success response. This is a concurrency-contract failure, even if it does not mutate the row.

- Medium | correctness | Version history and revert omit avatar state, so image changes are not diffable or reversible.
  - The revert field list excludes `image` / `image_base64` at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:238-253`.
  - Character sync-log snapshots for create/update also omit the image payload even though the update trigger treats image changes as version-relevant at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:1213-1251`.
  - `get_character_version_history()` exposes only those stored payloads at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:30792-30838`, and `revert_character_to_version_endpoint()` rebuilds the revert payload from those snapshots at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:1704-1753`.
  - Net effect: image-only edits can increment the row version, but `/versions`, `/versions/diff`, and `/revert` do not preserve enough state to show or restore the avatar transition.
  - Impact: the versioning API overstates what “revert to a previous version snapshot” can actually restore.

### Critical invariants recorded
- Uniqueness enforcement
  - Primary enforcement is the DB `UNIQUE` constraint on `character_cards.name` at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:668`.
  - Library pre-checks also reject duplicate active names before insert/update at `tldw_Server_API/app/core/Character_Chat/modules/character_db.py:190-194` and `:244-254`.
  - `get_character_card_by_name()` only searches active rows at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:17478`, so uniqueness is effectively “across all rows” at the schema level but “active rows only” at the pre-check level.
- Optimistic concurrency checks
  - Endpoint pre-checks exist for update and delete at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:1779-1788` and `:1826-1835`.
  - DB-level checks enforce version matching for update/delete/restore via `_get_current_db_version()` or explicit row lookups at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:17190`, `:17936-17944`, `:18108-18128`, and `:18208-18231`.
  - Empty update payloads and restore-on-active-row are the two clear exceptions described above.
- Normalization before persistence
  - `_prepare_character_data_for_db_storage()` coerces text-ish fields to strings, parses/normalizes list and dict fields, decodes and validates `image_base64`, and may transcode images to WEBP at `tldw_Server_API/app/core/Character_Chat/modules/character_db.py:30-180`.
  - Tag normalization and single-folder-token behavior are enforced in the DB helper via `_normalize_character_tags_for_operation()` at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:17745`.
  - Pydantic also validates JSON-string fields before the DB layer at `tldw_Server_API/app/api/v1/schemas/character_schemas.py:56-134`.
- Restore and revert semantics
  - Restore toggles `deleted` from `1` to `0`, bumps version, and uses `last_modified` as the deletion timestamp for retention-window enforcement at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:18170-18289`.
  - Revert is implemented as a normal update using the latest active version, not as a separate historical restore primitive at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:1727-1740`.
  - Revert only restores the subset of fields in `_CHARACTER_REVERT_FIELDS`, which excludes image and deletion state.
- Endpoint response semantics vs persistence semantics
  - `get_character_card_by_id()` and `get_character_card_by_name()` hide deleted rows by default at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:17451` and `:17478`.
  - `query_character_cards()` can include or isolate deleted rows via `include_deleted` and `deleted_only` at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:17548`.
  - Restore-not-found is surfaced as `409`, not `404`, because the endpoint maps DB `ConflictError` directly at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:1915-1917`. Current tests encode that behavior.

### Open questions and assumptions
- Assumption: using `last_modified` as the deletion timestamp for restore retention is intentional, but it means retention policy depends on the tombstone update time rather than a dedicated `deleted_at` column.
- Open question: whether the API intends empty `{}` updates to be a valid no-op `200` or whether they should be rejected as `400/422`. Current implementation and tests do not define that clearly.
- Open question: whether restore of an already-active row should be idempotent success at all. The endpoint description reads as if restore is only valid for soft-deleted rows.

## Coverage Gaps
- `tldw_Server_API/tests/Characters/test_character_functionality_db.py` does not directly cover `restore_character_card()` or `get_character_version_history()`, leaving the highest-risk Stage 2 restore/version invariants under-tested.
- `tldw_Server_API/tests/Characters/test_characters_endpoint.py` covers restore/revert happy paths but does not cover:
  - restore against an already-active row
  - empty update payloads with stale `expected_version`
  - image-only mutations across `/versions`, `/versions/diff`, and `/revert`
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py` does not cover `/query`, `/versions`, `/versions/diff`, `/revert`, or `/restore`.
- `tldw_Server_API/tests/Characters/test_character_chat_lib.py` is mostly out of scope for lifecycle integrity and provides little protection for versioning/restore behavior.

## Improvements
- Add a focused DB unit test for `restore_character_card()` that asserts stale-version restore on an active row is rejected.
- Add an endpoint test for `PUT /api/v1/characters/{id}` with `{}` and a stale `expected_version` so the optimistic-locking contract is explicit.
- Add one version-history/revert test that mutates only the avatar and verifies whether `/versions`, `/versions/diff`, and `/revert` are expected to preserve it.
- Consider documenting restore-not-found semantics explicitly if `409` is intentional.

## Exit Note
- Stage 2 review completed with concrete findings and completed targeted validation across the planned test surface.
- No source files were modified. The only intended artifact from this task is this report.
