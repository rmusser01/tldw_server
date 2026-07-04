# Test Suite Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three systemic gaps behind the April–July 2026 escaped defects (see `audits/2026-07-04-test-suite-audit-round2.md`): (1) contract/invariant gaps → targeted property tests; (2) singleton/lifecycle test-isolation blindness → guard plugin + isolation meta-tests; (3) env-matrix blindness → env-absent tests + auth×DB parametrization. Plus mechanized triage of the 3.9k-file suite and a frontend API contract-drift gate.

**Architecture:** Test/CI-layer only — no production behavior changes (exception: `Helper_Scripts/` tooling and additive `reset_*()` test hooks). Each task is one small PR branched from and targeting `dev`, passing `run_local_ci.py`, coverage gates (12% global / 35% AuthNZ), and the shard-coverage guard. New backend test files follow the `tests/<Module>/property/` + `@pytest.mark.property` convention and must join a `ci.yml` shard path list.

**Tech Stack:** pytest + hypothesis (already a dep, `pyproject.toml:58`), Python AST for triage tooling, openapi-typescript for frontend codegen, Vitest 4 / Playwright, GitHub Actions.

**Source documents:**
- `audits/2026-07-04-test-suite-audit-round2.md` (this round's findings, RA1–RA7 / RF1–RF3)
- `audits/2026-07-02-testing-implementation-audit.md` (round 1, F1–F10 — remediated in PR #2579; do not redo)

## Global Constraints

- Python: always `.venv/bin/python`, never system python (system is 3.14, too new).
- Every pytest invocation: `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1` env; local CI via `Helper_Scripts/ci/run_local_ci.py --lane <path>`.
- Frontend: bun workspace root is `apps/`; run bun commands from `apps/tldw-frontend/`.
- New test files must be added to a `ci.yml` shard (or the guard baseline with rationale) — `Helper_Scripts/ci/check_shard_coverage.py` enforces this.
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
  - `status_only`: test functions whose asserts touch only `.status_code`/`response.status`
  - `ambiguous_accept`: asserts of form `status_code in (200, 5xx)` (any 5xx alongside 2xx)
  - `tautology_suspect`: assert target is a literal/collection populated only by a mock/monkeypatch configured in the same function
  - `skip_stale`: skip/xfail whose `reason` lacks an issue/PR reference (informational only — round-1 F9 already enforces reasons exist)
- [ ] **Step 2:** `--write-baseline` mode; deterministic output (sorted paths, no timestamps).
- [ ] **Step 3:** Verify: run twice → identical output; the four known offenders (`tests/RAG/test_dual_backend_end_to_end.py`, `tests/Media/test_media_navigation.py`, `tests/DB_Management/test_media_db_schema_bootstrap.py`, `tests/Character_Chat/test_complete_v2_streaming_with_mock_openai.py`) all flagged; hand-verify 10 random flagged files for ≥ 70% flag precision.
- [ ] **Step 4:** Commit ranked report summary (top ~50 offenders) into `audits/2026-07-04-test-quality-triage-report.md`.

### Task 2 (PR 3): Fix the four exemplar offenders — RA1/RA2/RA3

**Files:**
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`
- Modify: `tldw_Server_API/tests/AuthNZ/test_auth_dependency_contract.py`
- Modify: `tldw_Server_API/tests/Character_Chat/test_complete_v2_streaming_with_mock_openai.py`
- Modify: `tldw_Server_API/tests/RAG/test_dual_backend_end_to_end.py`

- [ ] **Step 1:** `test_media_db_schema_bootstrap.py` — replace monkeypatch-collection asserts (`assert calls == [db]` at :32,:49,:67,:131,:155) with tests that run the real `initialize_sqlite_schema` against a temp SQLite DB and assert tables/indexes/views exist.
- [ ] **Step 2:** `test_auth_dependency_contract.py:175-189` — replace `_FakeJwtService` self-asserting mock with real JWT service via `tests/_plugins/authnz_fixtures.py`; assert accept/reject behavior on valid/expired/tampered tokens.
- [ ] **Step 3:** `test_complete_v2_streaming_with_mock_openai.py:58` — pin to deterministic 200 using the `tests/_plugins/chat_fixtures.py` mock-OpenAI bootstrap; assert SSE stream content (delta accumulation, terminal `[DONE]`), not just prefix.
- [ ] **Step 4:** `test_dual_backend_end_to_end.py` — either back the vector path with a real (temp) vector store + `pg_temp_db`, or rename/de-mark honestly as a unit test of the merge logic. Decide by effort; document choice in the PR.
- [ ] **Step 5:** Verify each rewrite is non-tautological by mutation: break the underlying behavior locally → test must fail; restore → pass.

### Task 3 (PR 4): Property tests — character card codec + config parsers — RA4

**Files:**
- Create: `tldw_Server_API/tests/Character_Chat/property/test_ccv3_roundtrip_properties.py`
- Create: `tldw_Server_API/tests/Character_Chat/property/test_png_chara_embed_properties.py`
- Create: `tldw_Server_API/tests/Config/property/test_config_section_parser_properties.py`
- Modify: `.github/workflows/ci.yml` (add new dirs to a shard)

- [ ] **Step 1:** ccv3 round-trip: hypothesis-generated card dicts → serialize → parse → field-identical; invalid cards rejected deterministically (`app/core/Character_Chat/ccv3_parser.py`).
- [ ] **Step 2:** PNG tEXt embed/extract round-trip against `_encode_png_with_chara_metadata` (`app/api/v1/endpoints/characters_endpoint.py:~2961`): arbitrary card JSON + generated minimal PNGs survive embed→extract byte-exact.
- [ ] **Step 3:** Config parsers (`app/core/config_sections/*.py`, e.g. `chunking.py` `_parse_bool`/`_parse_int`): never raise on arbitrary str/None/object; default on garbage; idempotent.
- [ ] **Step 4:** Mutation spot-check: drop a field in ccv3 serialization → property test fails. Shard guard green.

### Task 4 (PR 5): Property tests — chunking, markdown, macros + triage-ranked picks — RA4

**Files:**
- Modify: `tldw_Server_API/tests/Chunking/test_chunking_overlap_properties.py` (extend; don't duplicate existing property files)
- Create: `tldw_Server_API/tests/Notes_Tasks/property/test_markdown_parser_roundtrip_properties.py`
- Create: `tldw_Server_API/tests/Chat_Macros/property/` (or the module's existing test dir) — parser bounds properties
- Create: 2 more property files chosen from the Task 1 triage ranking (candidates: pagination invariants, JSON fence extraction, chatbook export/import round-trip)
- Modify: `.github/workflows/ci.yml` (shard entries)

- [ ] **Step 1:** Chunking invariants for `app/core/Chunking/chunker.py`: reconstruction (concat minus overlap == source), monotone offsets, overlap ≤ chunk size.
- [ ] **Step 2:** Markdown task parser round-trip (`app/core/Notes_Tasks/markdown_parser.py`), hierarchy preserved.
- [ ] **Step 3:** Chat-macro parser bounds (named escaped-defect source, commit c1f4e6eb95): arbitrary input never panics; bounds always enforced.
- [ ] **Step 4:** Mutation spot-check: off-by-one in chunker overlap → fails.

### Task 5 (PR 6): Singleton guard plugin (warn mode) + inventory — RA5

**Files:**
- Create: `tldw_Server_API/tests/_plugins/singleton_guard.py`
- Modify: `pyproject.toml` (register plugin)
- Create: `audits/2026-07-04-singleton-inventory.md` (appendix)

**Interfaces:** Precedent: `tests/_plugins/http_client_patch_guard.py`. Warn-only behind `TLDW_SINGLETON_GUARD=warn` initially; `=error` opt-in.

- [ ] **Step 1:** Grep-driven inventory of module-level singletons/`@lru_cache`/`_instance =` under `tldw_Server_API/app/core/` + `app/services/`; rank by escaped-defect adjacency (#2580 service caches, #2581 drain state, #2585 app-module identity, WS registries).
- [ ] **Step 2:** Plugin: snapshot registered global state at module boundaries; emit warning on leakage (state present after module that wasn't before it).
- [ ] **Step 3:** Verify: run 2–3 high-risk lanes (Embeddings, Jobs, AuthNZ) with guard on; catalog warnings (expected non-zero — that's the point); zero false positives on a clean lane like `tests/Config`.

### Task 6 (PR 7): Reset fixtures + isolation meta-tests + nightly shuffle — RA5

**Files:**
- Modify: `tldw_Server_API/tests/_plugins/` (reset fixtures for top inventory offenders; add `reset_*()` hooks in app code only where missing and purely additive)
- Create: `tldw_Server_API/tests/infrastructure/test_singleton_isolation.py`
- Create: `.github/workflows/test-order-shuffle-nightly.yml`
- Modify: `.github/workflows/ci.yml` or shard baseline (new `tests/infrastructure/` dir)

- [ ] **Step 1:** Reset fixtures for top-10 inventory offenders; drive guard warnings on those lanes to zero.
- [ ] **Step 2:** Meta-tests reproducing (minimized) #2580 (singleton cache re-registered against second DB), #2581 (drain state across suites), #2585 (`reload_app_main()` sys.modules identity) — each must fail without its fix/reset.
- [ ] **Step 3:** Nightly `pytest-randomly` (new dev-dep, autoload stays disabled — pass `-p randomly` explicitly) on 2–3 high-risk shards; not PR-blocking; promote only after 5 consecutive green runs.

### Task 7 (PR 8): Env-matrix coverage — RA6

**Files:**
- Modify: `tldw_Server_API/tests/_plugins/` (parametrized auth-mode × DB-backend fixture composing `authnz_fixtures.py` + `postgres.py`)
- Create: `tldw_Server_API/tests/Config/test_env_absent_defaults.py`
- Create: env-absent tests alongside egress policy tests
- Modify: `.github/workflows/backend-required.yml` or `ci.yml` (minimal-env smoke job)

- [ ] **Step 1:** Apply auth×DB matrix (single/multi × SQLite/`pg_temp_db`) to riskiest modules only: AuthNZ, Jobs operation contracts, egress policy, Chat.
- [ ] **Step 2:** Env-absent tests: `monkeypatch.delenv(..., raising=False)` every env var consumed by each `config_sections/*.py` module and the egress policy; assert real-deployment defaults (the exact class of the egress-bypass escape, 883b6c4dbd).
- [ ] **Step 3:** Minimal-env CI smoke job: boot app with scrubbed environment; run a small smoke subset; assert startup + auth defaults.
- [ ] **Step 4:** Verify: AuthNZ 35% coverage gate still green.

### Task 8 (PR 9): Backend OpenAPI export + frontend type codegen + drift gate — RF1

**Files:**
- Create: `Helper_Scripts/export_openapi_schema.py`
- Create: `apps/tldw-frontend/lib/api/openapi.json` (checked-in snapshot)
- Create: `apps/tldw-frontend/scripts/generate-api-types.mjs`
- Create: `apps/tldw-frontend/lib/api/generated/schema.d.ts` (generated, checked in)
- Modify: `apps/tldw-frontend/package.json` (dev-dep `openapi-typescript`, script `generate:api-types`)
- Modify: `.github/workflows/frontend-required.yml` (drift gate: regenerate + `git diff --exit-code`)

**Decision (recorded):** openapi-typescript codegen, not mirrored zod schemas — backend already emits OpenAPI 3 from FastAPI; codegen-only, no runtime cost, no dual maintenance. zod v4 stays for input validation.

- [ ] **Step 1:** Export script imports `tldw_Server_API.app.main`, dumps deterministic `openapi.json` (sorted keys).
- [ ] **Step 2:** Codegen wiring + checked-in generated types.
- [ ] **Step 3:** Drift gate in CI. Verify: rename one Pydantic response field on a scratch branch → gate fails; revert → green.

### Task 9 (PR 10): Typed Playwright mocks + nightly real-backend e2e — RF1/RF2

**Files:**
- Create: `apps/tldw-frontend/e2e/utils/typed-route-mock.ts`
- Modify: 5 highest-traffic specs (chat, characters, media) as exemplars — NOT all 53
- Modify: `.github/workflows/frontend-e2e-tiers.yml` (nightly real-backend tier booting backend single-user + `mock_openai_server/`)

- [ ] **Step 1:** `typed-route-mock.ts` wraps `page.route()` with generated response types from Task 8.
- [ ] **Step 2:** Migrate 5 exemplar specs; mock payloads must now type-check against the OpenAPI schema.
- [ ] **Step 3:** Nightly job runs `research-workspace.real-backend.spec.ts` + `chat-cockpit.real-server.spec.ts`; promote/expand only after 5 consecutive green runs.

### Task 10 (PR 11): Vitest global-mock extraction + optional fast-check — RF3

**Files:**
- Modify: `apps/tldw-frontend/vitest.setup.ts` (extract Dexie mock :44–85 and React Query stub :92–182 into opt-in modules)
- Create: `apps/tldw-frontend/test/dexie-mock.ts`, `test/react-query-mock.ts` (opt-in; `test/dexie-stub.ts` precedent exists)
- Modify: 2–3 suites to use `fake-indexeddb` instead of the Dexie mock
- Optional/cuttable: fast-check property tests for `lib/chatTransforms.ts`, `lib/urlNormalize.ts`

- [ ] **Step 1:** Extraction with zero behavior change for suites that opt in to the mocks.
- [ ] **Step 2:** Swap 2–3 suites to `fake-indexeddb`; fix any real-behavior breakage found (that's signal, not noise).
- [ ] **Step 3:** Verify: full `bun run test:run` green; skip count (`assert-playwright-no-skips.mjs`) not increased.

---

## Program-Level Verification

- Triage baseline (`test_quality_baseline.txt`) strictly shrinks PR-over-PR once the ratchet is enabled (post-Task 2 decision).
- Each of #2580/#2581/#2585 has a would-have-caught meta-test (Task 6).
- Mutation spot-checks recorded in each property-test PR description.
- Drift gate demonstrated red-then-green in the Task 8 PR description.
