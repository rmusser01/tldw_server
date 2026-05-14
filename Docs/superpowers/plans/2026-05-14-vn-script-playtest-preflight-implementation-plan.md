# VN Script Playtest Preflight API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add backend-owned VN script playtest/preflight endpoints that dry-run draft and published VN scripts for custom frontends without creating VN Play sessions, appending events, calling models, or duplicating runtime rules client-side.

**Architecture:** Keep the public API under the existing VN scripts router. Extract the scripted-story interpreter core from `VN_Play.service` into a small shared runtime module, then build a pure `VN_Scripts.playtest` analyzer on top of that module plus existing validation/manifest/profile/audio context. The analyzer explores deterministic paths, records choice and generation boundaries, and emits stable diagnostics.

**Tech Stack:** FastAPI, Pydantic v2, SQLite-backed `CharactersRAGDB`, existing VN script validator, existing VN Play scripted-story interpreter helpers, pytest.

---

## Stage 1: Shared Script Runtime Core

**Goal:** Make VN Play and script playtest use the same deterministic script execution semantics.

**Success Criteria:**
- [x] A new module, `tldw_Server_API/app/core/VN_Play/script_runtime.py`, owns the deterministic script runtime helpers currently embedded in `tldw_Server_API/app/core/VN_Play/service.py`.
- [x] `VN_Play.service` imports and uses the shared helpers without behavior changes.
- [x] Runtime helper API supports an explicit `max_steps` parameter while preserving the existing default of `MAX_SCRIPT_EXECUTION_STEPS = 500`.
- [x] Existing scripted-story generation/runtime tests still pass.

**Tests:**
- [x] `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`
- [x] `tldw_Server_API/tests/VN_Play/test_vn_play_scripted_generation_runtime.py`

**Implementation Notes:**
- Move these helpers and their direct dependencies together: `_initial_script_position`, `_initial_script_variables`, `_execute_script_program`, `_script_selected_choice`, `_script_visible_choices`, `_script_execution_payload`, `_script_random_result`, `_script_generation_result`, `_script_regeneration_result`, `_script_state_payload`, `_script_public_state_payload`, `_script_progress_token`, `_script_public_variables`, `_script_public_choices`, and dependent small normalization/hash helpers as needed.
- Keep the old function names imported into `service.py` to minimize churn and preserve tests that import private helpers such as `_payload_hash`.
- Do not move model-generation persistence, repository mutation, or FastAPI code into the runtime module.

## Stage 2: Pure Playtest Analyzer

**Goal:** Build a backend-only analyzer that explores script paths using validation context and the shared runtime core.

**Success Criteria:**
- [x] New module `tldw_Server_API/app/core/VN_Scripts/playtest.py` exposes a pure `build_script_playtest(...) -> dict[str, Any]`.
- [x] Analyzer accepts source metadata: `source`, `script_id`, `base_revision`, `version_id`, `validation_diagnostics`, and `validation_context_source`.
- [x] Analyzer enforces request limits: `max_steps`, `max_paths`, and truncation reporting.
- [x] Analyzer reports:
  - [x] `runtime_ready`
  - [x] `valid`
  - [x] `summary`
  - [x] `visited_labels`
  - [x] `unvisited_labels`
  - [x] `paths`
  - [x] `choice_boundaries`
  - [x] `generation_boundaries`
  - [x] `endings`
  - [x] `diagnostics.errors`
  - [x] `diagnostics.warnings`
  - [x] `validation_diagnostics`
- [x] Analyzer never creates VN Play sessions, generation rows, action requests, branch nodes, or chat/model calls.

**Tests:**
- [x] Add `tldw_Server_API/tests/VN_Scripts/test_vn_script_playtest.py` for pure analyzer behavior.

**Implementation Notes:**
- Traverse all visible choices breadth-first until `max_paths` or `max_steps` is reached.
- Treat `generate` with no literal output as a boundary, not a model call. Include generation ID, label, op index, profile key, output schema, confirmation flag, and prompt hash only.
- Detect loops using a stable execution-state key based on label, index, waiting choice, ended flag, and public-safe variable state. Emit `playtest_loop_detected`.
- Emit `playtest_truncated` when max path or max step limits cut traversal short.
- Use authoring graph or program labels to identify unvisited labels; do not infer dynamic/generated targets beyond statically declared targets.

## Stage 3: Service and Schema Contract

**Goal:** Expose playtest through `VNScriptService` with Pydantic schemas that are stable for custom frontend consumers.

**Success Criteria:**
- [x] Add request/response schemas in `tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py`.
- [x] Add `VNScriptService.playtest_draft(...)`, `VNScriptService.preview_draft_playtest(...)`, and `VNScriptService.playtest_version(...)`.
- [x] Draft playtest resolves current script metadata, current manifest, current policy/generation profiles, and accessible audio refs before analysis.
- [x] Version playtest uses the immutable published program, stored validation snapshot, and published-version snapshot context.
- [x] Supplied-draft playtest follows the existing graph-preview shape: optional supplied draft plus optional `draft_revision` warning, without mutating stored diagnostics.

**Tests:**
- [x] Extend `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py` for endpoint-level schema and auth/not-found behavior.
- [x] Extend OpenAPI contract tests if VN script path allowlists require updates.

**Implementation Notes:**
- Request schema:
  - `draft: dict[str, Any] | None = None`
  - `draft_revision: int | None = None`
  - `max_steps: int = 500` bounded to `1..5000`
  - `max_paths: int = 100` bounded to `1..1000`
- Public endpoints:
  - `POST /api/v1/vn/vn-scripts/scripts/{script_id}/draft/playtest`
  - `POST /api/v1/vn/vn-scripts/scripts/{script_id}/versions/{version_id}/playtest`
- Draft endpoint source values:
  - `stored_draft` when request omits `draft`
  - `supplied_draft` when request includes `draft`
- Version endpoint source value:
  - `published_version`

## Stage 4: Endpoint Wiring and Capabilities

**Goal:** Wire playtest endpoints into the existing VN API namespace and advertise them through backend capability discovery.

**Success Criteria:**
- [x] Add endpoint handlers in `tldw_Server_API/app/api/v1/endpoints/vn_scripts.py` next to graph/validate/version routes.
- [x] Reuse `_resolve_script_profiles(...)` and `_resolve_accessible_audio_refs(...)` for draft playtests.
- [x] Convert domain `ValueError` cases through existing VN error detail helpers.
- [x] Update `tldw_Server_API/app/core/VN_Platform/capabilities.py` with `features.script_playtest = true` only when both new playtest routes are registered.
- [x] OpenAPI includes both new endpoints.

**Tests:**
- [x] Update `tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py`.
- [x] Update `tldw_Server_API/tests/Services/test_openapi_contracts.py` if it asserts VN path coverage.

**Implementation Notes:**
- Keep endpoints synchronous where the service is pure, but async handlers are fine because profile/audio resolution already uses async dependencies.
- Do not add frontend code in this slice.

## Stage 5: Documentation and Verification

**Goal:** Document the endpoint contract and prove the change is stable.

**Success Criteria:**
- [x] Update `Docs/API/VN.md` with endpoint contract, non-goals, diagnostic shape, and custom frontend usage guidance.
- [x] Update `Docs/API-related/VN_PLATFORM_API.md` capability/path summary if it lists script authoring features.
- [x] Update `TASK-338` notes and final summary with verification evidence.

**Verification Commands:**
- [x] `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_playtest.py tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py tldw_Server_API/tests/VN_Play/test_vn_play_scripted_generation_runtime.py tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py tldw_Server_API/tests/Services/test_openapi_contracts.py` (`139 passed, 27 warnings`)
- [x] `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q` (`48 passed, 8 warnings`)
- [x] `git diff --check`
- [x] `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/VN_Play/script_runtime.py tldw_Server_API/app/core/VN_Play/errors.py tldw_Server_API/app/core/VN_Scripts/playtest.py tldw_Server_API/app/core/VN_Scripts/service.py tldw_Server_API/app/api/v1/endpoints/vn_scripts.py -f json -o /tmp/bandit_vn_script_playtest_preflight.json` (`0 results`)

---

## Risk Review

- **Interpreter drift:** Extract the runtime helpers first and make VN Play import them, so playtest and runtime cannot silently diverge.
- **Accidental mutation:** Keep analyzer pure and assert in tests that no VN Play session rows are created by playtest endpoints.
- **Model-call leakage:** Treat model-backed `generate` opcodes as boundaries and assert the generation adapter is never involved.
- **Path explosion:** Bound both paths and steps. Return partial results with explicit `playtest_truncated` diagnostics.
- **Published-version drift:** Version endpoint must use the stored program and validation snapshot. It must not re-resolve mutable draft context except for ownership/version lookup.
- **Custom frontend ambiguity:** Response envelope should contain stable codes and route source metadata so clients do not need to infer backend state from prose.
