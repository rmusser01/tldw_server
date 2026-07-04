# Test Suite Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the systemic gaps behind the April–July 2026 dev-merged defects (see `audits/2026-07-04-test-suite-audit-round2.md`; profile counts only defects present on latest `dev`). Priority follows severity: (1) singleton/lifecycle test-isolation blindness (~27% of defects, **all High severity**, #2585 still open) → guard plugin + isolation meta-tests; (2) env-matrix blindness (~27%, mostly High) → env-absent tests + auth×DB parametrization; (3) contract/invariant gaps (~36%, mostly Medium) → targeted property tests. Plus mechanized triage of the ~4k-file suite and a frontend API contract-drift gate (#2590 class).

**Architecture:** Test/CI-layer only — no production behavior changes (exception: `Helper_Scripts/` tooling and additive `reset_*()` test hooks). Each task is one small PR branched from and targeting `dev`, passing `run_local_ci.py`, coverage gates (12% global / 35% AuthNZ), and the shard-coverage guard. New backend test files follow the `tests/<Module>/property/` + `@pytest.mark.property` convention and must join a `ci.yml` shard path list where the module's shard enumerates individual files (Character_Chat does; `tests/Config` is covered as a whole directory by the `core-config` shard, so files under it need no ci.yml change).

**Tech Stack:** pytest + hypothesis (already a dep, `pyproject.toml:58`), Python AST for triage tooling, openapi-typescript for frontend codegen, Vitest 4 / Playwright, GitHub Actions.

**Source documents:**
- `audits/2026-07-04-test-suite-audit-round2.md` (this round's findings, RA1–RA7 / RF1–RF3; corrected after adversarial fact-check; dev-merged defects only)
- `audits/2026-07-02-testing-implementation-audit.md` (round 1, F1–F10 — remediated in PR #2579; do not redo)

## Global Constraints

- Python: always `.venv/bin/python`, never system python (system is 3.14, too new).
- Every pytest invocation: `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1` env; local CI via `Helper_Scripts/ci/run_local_ci.py --lane <path>`.
- Frontend: bun workspace root is `apps/`; run bun commands from `apps/tldw-frontend/`.
- New test files must be added to a `ci.yml` shard (or the guard baseline with rationale) — `Helper_Scripts/ci/check_shard_coverage.py` enforces this.
- **Pytest plugins load via `pytest_plugins` in the conftests** (e.g. `tldw_Server_API/tests/conftest.py:8`). The `plugins = [...]` list under `[tool.pytest.ini_options]` in pyproject.toml is NOT a pytest option — pytest silently ignores it. Never register a plugin there.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Verify current state before each task — the suite evolves fast (e.g. round-1 remediation already removed `norecursedirs` and enforced skip reasons; both were re-verified 2026-07-04).

---

### Task 1 (PR 2): Mechanized test-quality triage tooling — RA7

**Files:**
- Create: `Helper_Scripts/ci/test_quality_triage.py`
- Create: `Helper_Scripts/ci/test_quality_baseline.txt` (via `--write-baseline`)
- Modify: `Makefile` (add `test-triage` target)

**Interfaces:** Pattern-follow `Helper_Scripts/ci/check_shard_coverage.py` (ratchet baseline mechanics). Report-only in this PR; ratchet enforcement is a later decision.

- [ ] **Step 1:** AST scanner over `tldw_Server_API/tests/**/test_*.py` emitting ranked JSON + human-readable report with per-file flags:
  - `mock_density`: `patch`/`patch.object`/`MagicMock`/`AsyncMock`/`monkeypatch` count per test function; flag integration-marked files with density ≥ 4
  - `stub_injection`: test-module-defined fake/stub classes (class defs whose instances are assigned to attributes of the object under test) and `dependency_overrides` assignments per test — this is what catches the RAG stub-class and AuthNZ fake-service patterns that raw mock counting misses
  - `status_only`: test functions whose asserts touch only `.status_code`/`response.status`
  - `ambiguous_accept`: asserts of form `status_code in (200, 5xx)` (any 5xx alongside 2xx)
  - `tautology_suspect`: assert target is a literal/collection populated only by a mock/monkeypatch/stub configured in the same function (includes asserts *inside* test-module-defined fake classes)
  - `skip_stale`: skip/xfail whose `reason` lacks an issue/PR reference (informational only — round-1 F9 already enforces reasons exist)
- [ ] **Step 2:** `--write-baseline` mode; deterministic output (sorted paths, no timestamps).
- [ ] **Step 3:** Verify: run twice → identical output; at least 3 of the 4 known offenders flagged (`tests/RAG/test_dual_backend_end_to_end.py` via `stub_injection`, `tests/DB_Management/test_media_db_schema_bootstrap.py` via `tautology_suspect` — note it is unit-marked, so `mock_density`'s integration filter won't fire, `tests/AuthNZ/test_auth_dependency_contract.py` via `stub_injection`/`tautology_suspect`, `tests/Character_Chat/test_complete_v2_streaming_with_mock_openai.py` via `ambiguous_accept`); hand-verify 10 random flagged files for ≥ 70% flag precision.
- [ ] **Step 4:** Commit ranked report summary (top ~50 offenders) into `audits/2026-07-04-test-quality-triage-report.md`.

### Task 2 (PR 3): Fix the four exemplar offenders — RA1/RA2/RA3

**Files:**
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`
- Modify: `tldw_Server_API/tests/AuthNZ/test_auth_dependency_contract.py`
- Modify: `tldw_Server_API/tests/Character_Chat/test_complete_v2_streaming_with_mock_openai.py`
- Modify: `tldw_Server_API/tests/RAG/test_dual_backend_end_to_end.py`

- [ ] **Step 1:** `test_media_db_schema_bootstrap.py` — replace monkeypatch-collection asserts (`assert calls == [db]` at :32,:49,:67; `coordinator_calls` at :131,:155) with tests that run the real schema bootstrap. Note `initialize_sqlite_schema(db)` takes a MediaDatabase-like object, not a path: construct a real `MediaDatabase` against `tmp_path` (precedent: real `:memory:` schema tests around line 1452 *in the same file*) and assert tables/indexes/views exist. Keep the dispatch-routing asserts that are genuinely about `ensure_media_schema` dispatch — not every flagged assert is worthless.
- [ ] **Step 2:** `test_auth_dependency_contract.py:175-189` — replace `_FakeJwtService` self-asserting mock with the real JWT service via the `jwt_service` fixture (`tests/AuthNZ/conftest.py:1878` — pure `JWTService(settings=jwt_settings)`, no DB, already available to this test's location); assert accept/reject behavior on valid/expired/tampered tokens.
- [ ] **Step 3:** `test_complete_v2_streaming_with_mock_openai.py:58` — two defects to fix: the `in (200, 502)` accept AND the fact that the test always skips in CI (`MOCK_OPENAI_BASE_URL` is set in zero workflows despite the file being sharded, `ci.yml:~1141`). Rework on the pattern of `tests/Character_Chat/test_complete_v2_streaming_e2e_mock.py` (in-repo precedent: deterministic in-process multi-chunk SSE, no external env dependency); assert delta accumulation and terminal `[DONE]`, expect exactly 200. Do NOT reach for `tests/_plugins/chat_fixtures.py` — its in-process handler only emits single non-streaming JSON completions and the plugin isn't loaded for `tests/Character_Chat/`.
- [ ] **Step 4:** `test_dual_backend_end_to_end.py` — either back the vector path with a real temp vector store, or de-mark/rename honestly as a unit test of the merge logic. Note a real vector store means ChromaDB + embeddings (heavy); the honest de-mark is the expected outcome unless `deterministic_embeddings` from the conftest makes the real path cheap. Document the choice in the PR.
- [ ] **Step 5:** Verify each rewrite is non-tautological by mutation: break the underlying behavior locally → test must fail; restore → pass.

### Task 3 (PR 4): Singleton guard plugin (warn mode) + inventory — RA5 (top merged-defect class by severity)

**Files:**
- Create: `tldw_Server_API/tests/_plugins/singleton_guard.py`
- Modify: `tldw_Server_API/tests/conftest.py` (add to the `pytest_plugins` tuple at line ~8 — the `http_client_patch_guard` precedent)
- Modify: `pyproject.toml` (annotate or remove the dead `plugins = [...]` key under `[tool.pytest.ini_options]` — it is not a pytest option and silently does nothing; it must not be where this plugin gets "registered")
- Create: `audits/2026-07-04-singleton-inventory.md` (appendix)

**Interfaces:** Precedent: `tests/_plugins/http_client_patch_guard.py` (note its mechanism: it intercepts at patch-*time* via monkeypatch hooks, not post-hoc snapshots — the singleton guard will need boundary snapshots instead, which is a different shape; the precedent is for loading/registration, not mechanism). Warn-only behind `TLDW_SINGLETON_GUARD=warn` initially; `=error` opt-in.

- [ ] **Step 1:** Grep-driven inventory of module-level singletons/`@lru_cache`/`_instance =` under `tldw_Server_API/app/core/` + `app/services/`; rank by defect adjacency (#2580 service caches, #2581 drain state, #2585 app-module identity, WS registries).
- [ ] **Step 2:** Plugin: snapshot registered global state at module boundaries; emit warning on leakage (state present after module that wasn't before it).
- [ ] **Step 3:** Verify: run 2–3 high-risk lanes (Embeddings, Jobs, AuthNZ) with guard on; catalog warnings (expected non-zero — that's the point); zero false positives on a clean lane like `tests/Config`.

### Task 4 (PR 5): Reset fixtures + isolation meta-tests + nightly shuffle — RA5

**Files:**
- Modify: `tldw_Server_API/tests/_plugins/` (reset fixtures for top inventory offenders; add `reset_*()` hooks in app code only where missing and purely additive)
- Create: `tldw_Server_API/tests/infrastructure/test_singleton_isolation.py`
- Create: `.github/workflows/test-order-shuffle-nightly.yml`
- Modify: `.github/workflows/ci.yml` or shard baseline (new `tests/infrastructure/` dir)

- [ ] **Step 1:** Reset fixtures for top-10 inventory offenders; drive guard warnings on those lanes to zero.
- [ ] **Step 2:** Meta-tests reproducing (minimized) #2580 (singleton cache re-registered against second DB), #2581 (drain state across suites), #2585 (`reload_app_main()` sys.modules identity) — each must fail without its fix/reset. #2585 is still open: its meta-test lands xfail-with-issue-link until the fix ships, then flips.
- [ ] **Step 3:** Nightly `pytest-randomly` (new dev-dep; works under `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` with explicit `-p randomly`, consistent with CI's existing `-p pytest_cov -p pytest_asyncio.plugin` usage) on 2–3 high-risk shards; not PR-blocking; promote only after 5 consecutive green runs.

### Task 5 (PR 6): Env-matrix coverage — RA6

**Files:**
- Modify: `tldw_Server_API/tests/_plugins/` (parametrized auth-mode × DB-backend fixture composing `tests/AuthNZ/conftest.py` fixtures + `_plugins/postgres.py`)
- Create: `tldw_Server_API/tests/Config/test_env_absent_defaults.py`
- Create: env-absent tests alongside the workflows/egress policy tests
- Create: `Helper_Scripts/ci/minimal_env_smoke.py` + a CI job wiring it (in `backend-required.yml` or `ci.yml`)

**Constraint that shapes this task:** `tldw_Server_API/tests/conftest.py` (and root `conftest.py`) force-set env at import time — `AUTH_MODE=single_user`, `SINGLE_USER_API_KEY`, `DATABASE_URL`, `WORKFLOWS_EGRESS_BLOCK_PRIVATE=false`, `WORKFLOWS_WEBHOOK_ALLOWLIST=*`, and more. Any in-pytest run is therefore already env-polluted before the first test executes.

- [ ] **Step 1:** Apply auth×DB matrix (single/multi × SQLite/`pg_temp_db`) to riskiest modules only: AuthNZ, Jobs operation contracts, workflows egress policy, Chat.
- [ ] **Step 2:** Env-absent tests: per-test `monkeypatch.delenv(..., raising=False)` of every env var consumed by each `config_sections/*.py` module and the workflows egress policy (conftest force-relaxes `WORKFLOWS_EGRESS_*` — deployment defaults are never tested), **paired with the module's settings-cache reset hook** (delenv alone is defeated by cached settings objects); assert real-deployment defaults. This is the #2590/e88c96500f defect class: hardening passes that broke the no-env-var fallback path.
- [ ] **Step 3:** Minimal-env smoke: **not an in-pytest run** (conftest re-injects env). Boot the server as a subprocess with a scrubbed environment (empty env plus a pinned minimal allowlist), probe startup + auth defaults over HTTP, exit nonzero on drift.
- [ ] **Step 4:** Verify: AuthNZ 35% coverage gate still green.

### Task 6 (PR 7): Property tests — Jobs contracts, character card parser, PNG codec, config parsers — RA4

**Files:**
- Create: `tldw_Server_API/tests/Jobs/property/test_operation_contract_properties.py`
- Create: `tldw_Server_API/tests/Character_Chat/property/test_ccv3_parser_properties.py`
- Create: `tldw_Server_API/tests/Character_Chat/property/test_png_chara_embed_properties.py`
- Create: `tldw_Server_API/tests/Config/property/test_config_section_parser_properties.py`
- Modify: `.github/workflows/ci.yml` (shard entries where the module shard enumerates individual files; `tests/Config/` needs no change — the `core-config` shard covers the directory, `ci.yml:~584`)

- [ ] **Step 1:** Jobs operation-contract invariants — the top merged validation defect (a9b6a2c310, d6319e9e16): hypothesis-generated operation sequences/settings never reach the impossible states those fixes guard against; start from the invariants the fix commits added and generalize.
- [ ] **Step 2:** ccv3 parser invariants — `ccv3_parser.py` exposes `parse_v3_card`/`validate_v3_card` only (**no serializer — do not attempt a serialize→parse round-trip**): hypothesis-generated card dicts → parse idempotence (`parse(parse(x)) == parse(x)`), known-field preservation, deterministic rejection of invalid cards via `validate_v3_card`.
- [ ] **Step 3:** PNG tEXt embed/extract round-trip — this one IS bidirectional: `_encode_png_with_chara_metadata` (`app/api/v1/endpoints/characters_endpoint.py:2961`) → `extract_json_from_image_file` (`app/core/Character_Chat/modules/character_io.py:175`): arbitrary card JSON + generated minimal PNGs survive embed→extract exactly. Precedent tests: `tests/Character_Chat_NEW/unit/test_png_export.py`.
- [ ] **Step 4:** Config parsers (`app/core/config_sections/*.py`, e.g. `chunking.py` `_parse_bool`:41/`_parse_int`:31): never raise on arbitrary str/None/object; default on garbage; idempotent.
- [ ] **Step 5:** Mutation spot-check: weaken a Jobs contract guard and drop a field in ccv3 parsing → property tests fail. Shard guard green.

### Task 7 (PR 8): Property tests — chunking, markdown + triage-ranked picks — RA4

**Files:**
- Modify: `tldw_Server_API/tests/Chunking/test_chunking_overlap_properties.py` (extend; don't duplicate existing property files)
- Create: `tldw_Server_API/tests/Notes_Tasks/property/test_markdown_parser_properties.py`
- Create: 2–3 property files chosen from the Task 1 triage ranking (candidates: pagination invariants, JSON fence extraction, chatbook export/import round-trip)
- Modify: `.github/workflows/ci.yml` (shard entries where the module shard enumerates files)

- [ ] **Step 1:** Chunking invariants for `app/core/Chunking/chunker.py`: reconstruction (concat minus overlap == source), monotone offsets, overlap ≤ chunk size.
- [ ] **Step 2:** Markdown checklist parser invariants — `app/core/Notes_Tasks/markdown_parser.py` is parse-only (`parse_note_checklists`; **no render direction — do not attempt a round-trip**): never raises on arbitrary input, reported spans stay within source bounds, hierarchy/indent monotonicity, re-parse idempotence.
- [ ] **Step 3:** 2–3 triage-ranked picks. (Chat-macro parser bounds is NOT a candidate: `app/core/Chat_Macros/` exists only on unmerged `codex/chat-macros-v1` and its bugs were caught pre-merge — out of scope under the dev-merged-only rule. Revisit only if/when that branch merges.)
- [ ] **Step 4:** Mutation spot-check: off-by-one in chunker overlap → fails.

### Task 8 (PR 9): Backend OpenAPI export + frontend type codegen + drift gate — RF1

**Files:**
- Create: `Helper_Scripts/export_openapi_schema.py`
- Create: `apps/tldw-frontend/lib/api/openapi.json` (checked-in snapshot)
- Create: `apps/tldw-frontend/scripts/generate-api-types.mjs`
- Create: `apps/tldw-frontend/lib/api/generated/schema.d.ts` (generated, checked in)
- Modify: `apps/tldw-frontend/package.json` (dev-dep `openapi-typescript`, script `generate:api-types`)
- Modify: `.github/workflows/frontend-required.yml` (drift gate)

**Decision (recorded):** openapi-typescript codegen, not mirrored zod schemas — backend already emits OpenAPI 3 from FastAPI; codegen-only, no runtime cost, no dual maintenance. zod v4 stays for input validation.

- [ ] **Step 1:** Export script imports `tldw_Server_API.app.main` and dumps `openapi.json` with sorted keys — and **pins a canonical env set internally** (does not inherit ambient env): the app has env-driven route toggles (`MINIMAL_TEST_APP`, `DISABLE_HEAVY_STARTUP` via `services/startup_heavy_policy.py`, `_TEST_MODE` middleware at `main.py:~2252`), so an env-inheriting export produces different schemas on dev machines vs CI. Document the pinned set in the script header.
- [ ] **Step 2:** Codegen wiring + checked-in generated types.
- [ ] **Step 3:** Drift gate placement — **critical:** `frontend-required.yml` gates all real steps on `needs.changes.outputs.frontend_changed == 'true'`, so a backend-only PR (the kind that causes drift) would skip a naively-placed gate. The detect job already exposes `backend_changed` (`frontend-required.yml:22`): condition the regenerate + `git diff --exit-code` steps on `backend_changed == 'true' || frontend_changed == 'true'`. Consider mirroring the check in `backend-required.yml`, which is guaranteed to run on drift-causing PRs.
- [ ] **Step 4:** Verify: rename one Pydantic response field on a **backend-only** scratch branch → gate fails; revert → green. (Testing with a branch that also touches frontend files would mask the placement bug this step exists to catch.)

### Task 9 (PR 10): Typed Playwright mocks + extend real-backend nightly — RF1/RF2

**Files:**
- Create: `apps/tldw-frontend/e2e/utils/typed-route-mock.ts`
- Modify: 5 highest-traffic specs (chat, characters, media) as exemplars — NOT all 53
- Modify: `.github/workflows/ui-research-workspace-nightly.yml` (extend — do NOT create a new nightly; this one already runs `research-workspace.real-backend.spec.ts` against a live server incl. a strict no-skip extension variant)

- [ ] **Step 1:** `typed-route-mock.ts` wraps `page.route()` with generated response types from Task 8.
- [ ] **Step 2:** Migrate 5 exemplar specs; mock payloads must now type-check against the OpenAPI schema.
- [ ] **Step 3:** Add `chat-cockpit.real-server.spec.ts` (currently wired into no workflow) to the existing nightly; note `e2e/real-server-workflows.spec.ts` (~17 tests) and the admin-ui real-backend job in `frontend-required.yml` already exist — the gap to close is chat-cockpit plus, over time, media ingest / knowledge base / settings coverage. Expand only after 5 consecutive green runs.

### Task 10 (PR 11): Vitest global-mock extraction + optional fast-check — RF3

**Files:**
- Modify: `apps/tldw-frontend/vitest.setup.ts` (extract Dexie mock :44–85 and React Query wrapper :92–182 into opt-in modules)
- Create: `apps/tldw-frontend/test/dexie-mock.ts`, `test/react-query-mock.ts` (opt-in; `test/dexie-stub.ts` precedent exists)
- Modify: `apps/tldw-frontend/package.json` (add `fake-indexeddb` dev-dep — not currently a dependency)
- Modify: 2–3 suites to use `fake-indexeddb` instead of the Dexie mock
- Optional/cuttable: fast-check property tests for `lib/chatTransforms.ts`, `lib/urlNormalize.ts`

- [ ] **Step 1:** Extraction with zero behavior change for suites that opt in to the mocks. Note the React Query wrapper is not a blunt stub — it preserves real hooks when a `QueryClientProvider` is mounted; preserve that behavior in the extracted module.
- [ ] **Step 2:** Swap 2–3 suites to `fake-indexeddb`; fix any real-behavior breakage found (that's signal, not noise).
- [ ] **Step 3:** Verify: full `bun run test:run` green; skip count (`assert-playwright-no-skips.mjs`) not increased.

---

## Program-Level Verification

- Triage baseline (`test_quality_baseline.txt`) strictly shrinks PR-over-PR once the ratchet is enabled (post-Task 2 decision).
- Each of #2580/#2581/#2585 has a would-have-caught meta-test (Task 4; #2585's lands xfail-with-issue-link while the issue is open).
- Mutation spot-checks recorded in each property-test PR description.
- Drift gate demonstrated red-then-green on a backend-only scratch branch in the Task 8 PR description.
