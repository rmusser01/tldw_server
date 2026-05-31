# VN Play Setup Options Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backend setup-options API so VN Play session setup is usable by the WebUI and custom frontends without duplicating setup rules client-side.

**Architecture:** Keep setup logic inside the API server. Add Pydantic setup response schemas, bounded VN asset pack repository/service helpers, and a focused `core/VN_Play/setup_options.py` composer used by `GET /api/v1/vn-play/setup-options`. PR #1409 remains a frontend reference for UX scenarios, but this implementation centralizes readiness, compatibility, trust, and warning decisions in the backend.

**Tech Stack:** FastAPI, Pydantic, SQLite via `CharactersRAGDB`, existing `VNAssetPackService`, pytest `TestClient`, Bandit.

---

## References

- Spec: `Docs/superpowers/specs/2026-05-09-vn-play-setup-options-design.md`
- Backlog: `TASK-158`
- UX reference only: `https://github.com/rmusser01/tldw_server/pull/1409`

## Task 1: RED Backend API Contract Tests

**Files:**
- Modify: `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`

- [x] Add failing tests for:
  - `GET /api/v1/vn-play/setup-options` returns selector-safe character metadata, selected character, pack warning summary, and pagination.
  - `selected_character` is preserved when outside the current character page.
  - readiness is evaluated only for returned pack rows.
  - untrusted completed import provenance emits `pack_untrusted_import`.
  - per-pack readiness failures degrade to `readiness_unavailable`.
- [x] Run:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q`
- [x] Expected: new tests fail because `/setup-options` does not exist yet.

Status: Complete. RED run failed with seven expected 404s for the missing setup-options route; later GREEN run passed 14 VN Play API tests.

## Task 2: Bounded Pack And Provenance Repository Helpers

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/service.py`
- Modify: `tldw_Server_API/tests/VN_Assets/test_vn_asset_packs_db.py`

- [x] Add failing repository tests for bounded setup pack listing and latest completed import provenance lookup.
- [x] Implement `list_packs_for_setup(owner_user_id, query, limit, offset)` with SQL-level filtering and `LIMIT limit + 1`.
- [x] Implement `latest_completed_import_provenance_by_pack_ids(owner_user_id, pack_ids)` with one query over the returned pack IDs.
- [x] Add a thin `VNAssetPackService.list_packs_for_setup` wrapper.
- [x] Run:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Assets/test_vn_asset_packs_db.py -q`

Status: Complete. RED run failed on the missing setup helpers and exposed an existing multi-hop matrix dependency failure; GREEN run passed 10 VN Assets DB tests after fixing both.

## Task 3: Schemas, Composer, And Endpoint

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
- Create: `tldw_Server_API/app/core/VN_Play/setup_options.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_play.py`

- [x] Add setup response schemas matching the approved spec: top-level `characters`, `selected_character`, `asset_packs`, `defaults`, `pagination`, `empty_states`, and `generated_at`.
- [x] Build selector-safe character options from `query_character_cards`/`get_character_card_by_id` without exposing image bytes.
- [x] Build bounded pack options from returned pack rows only; compute readiness, compatibility, content-rating warnings, and trust provenance warning summaries.
- [x] Add `GET /setup-options` to the VN Play router using the existing per-user `ChaChaNotes.db` dependency.
- [x] Run:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q`

Status: Complete. GREEN run passed 14 VN Play API tests.

## Task 4: Documentation And Closeout

**Files:**
- Modify: `Docs/API-related/VN_PLAY_API.md` if the endpoint is documented there.
- Modify: `backlog/tasks/task-158 - Implement-backend-VN-Play-setup-options-API.md`

- [x] Document the backend endpoint or record why existing docs do not cover VN Play endpoint details.
- [x] Run focused backend tests:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py tldw_Server_API/tests/VN_Assets/test_vn_asset_packs_db.py -q`
- [x] Run Bandit on touched backend files.
- [x] Run `git diff --check`.
- [x] Update `TASK-158` with verification and final summary.

Status: Complete. Final verification:

- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py tldw_Server_API/tests/VN_Assets/test_vn_asset_packs_db.py -q` passed 24 tests.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/VN_Play/setup_options.py tldw_Server_API/app/api/v1/endpoints/vn_play.py tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py tldw_Server_API/app/core/VN_Assets/service.py -f json -o /tmp/bandit_vn_setup_options_prod.json` exited 0.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/tests/VN_Play/test_vn_play_api.py tldw_Server_API/tests/VN_Assets/test_vn_asset_packs_db.py -s B101 -f json -o /tmp/bandit_vn_setup_options_tests.json` exited 0.
- `git diff --check` exited 0.
