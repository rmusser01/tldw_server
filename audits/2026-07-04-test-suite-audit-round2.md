# Test Suite Audit — Round 2 (Defect-Driven)

**Date:** 2026-07-04
**Scope:** Backend pytest suite (`tldw_Server_API/tests/`) + frontend Vitest/Playwright (`apps/tldw-frontend/`)
**Method:** Escaped-defect analysis of git/issue history (April–July 2026) cross-referenced against test-suite design patterns; four parallel exploration passes (infra map, quality sampling across 8 modules, defect history, frontend infra).
**Relationship to prior audit:** Builds on `audits/2026-07-02-testing-implementation-audit.md` (F1–F10, remediated in PR #2579). This round asks a different question: *why did recently shipped bugs escape a suite of ~3,900 test files that was assumed to cover them?* Findings already resolved by round 1 (coverage-gate ratchet F1, `norecursedirs` un-hiding F3, skip-reason hygiene F9) are not re-reported.

---

## 1. Escaped-Defect Profile (April–July 2026)

~20 concrete escaped defects were identified from bug-fix commits and issues. Categorized:

| Category | Share | Representative cases |
|---|---|---|
| **Missing input validation / contract enforcement** | ~45% | Chat macro parser missing bounds validation (c1f4e6eb95); Jobs operation contracts allowing impossible states (a9b6a2c310); integrations HTTP bypassing central egress policy (883b6c4dbd); workspace ingest failure codes not exposed (c725caad5a) |
| **State management / lifecycle** | ~40% | Service-layer singleton caches bypassing test isolation (#2580, 4924719264); `reload_app_main()` permanently swapping `sys.modules` (#2585); Embeddings drain-state corrupting subsequent tests (#2581, 254af77776); ACP WS broadcaster not cleaned up on disconnect (790e3f3264); chat-macro run-state races (7f7820395d, a4feb40ee8, 8f48bd6bc8) |
| **Env/config-dependent behavior** | ~20% | Web shell auth lost on hard reload — no runtime-override fallback (#2590, 626447bd5c); UX smoke gate hidden after credential hardening (e88c96500f) |
| **Cross-module integration breaks** | ~10% | Bit-rotted multiuser load-test helper signature (4924719264) |
| **Serialization / round-trip** | ~10% | Runtime auth credentials not durable across hard reloads (#2590); audio download size regression (4b89ce40a2) |

**Recurring escape mechanism:** a refactor hardens one path (e.g. credential storage) but forgets the fallback/compat layer; tests pass in CI (where env vars are set, singletons are warm, and mocks stand in for the integrated system) but the deployment-shaped world differs.

## 2. Backend Findings

### 2.1 "Integration" tests that don't integrate (RA1)

Tests marked `@pytest.mark.integration` (456 files carry the marker) frequently stub the very layer they claim to integrate:

- `tldw_Server_API/tests/RAG/test_dual_backend_end_to_end.py` — named "end to end", but `_StubVectorStore` (lines ~118–142) hardcodes vector results; the vector-search path is never exercised.
- `tldw_Server_API/tests/Media/test_media_navigation.py` — a real TestClient call surrounded by 8+ `patch.object` calls (`get_storage_backend`, `_extract_pdf_outline`, `get_cached_response`, `cache_response`, …); assertions check field presence and status only.

**Consequence:** cross-module breaks (defect category ~10%, plus contributing to the 45% validation class) sail through because no test wires the real components together.

### 2.2 Tautological / assertion-in-the-mock tests (RA2)

- `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py` — collects monkeypatched calls into a list and asserts the list (`assert calls == [db]` at lines 32, 49, 67, 131, 155). The real `initialize_sqlite_schema` never runs; the test verifies the monkeypatch, not the schema bootstrap.
- `tldw_Server_API/tests/AuthNZ/test_auth_dependency_contract.py:175–189` — `_FakeJwtService` asserts its own hardcoded token (`assert token == "jwt.header.signature"` at lines 180, 186) *inside the mock*; real token validation is never touched. Large stretches (lines ~284–670) assert identity of imported names (`assert RequireRole is auth_deps.RequireRole`), testing wiring, not authorization logic.

### 2.3 Tolerated-failure assertions (RA3)

- `tldw_Server_API/tests/Character_Chat/test_complete_v2_streaming_with_mock_openai.py:58` — `assert response.status_code in (200, 502)`. A 502 is a failure; the test passes either way. Streaming-content assertions (lines 61–63) check only that a line starts with `data: `, then break.

Status-code-only assertions (no body semantics) are common across sampled modules — endpoints can return wrong data and stay green.

### 2.4 Property-based testing is concentrated, not targeted (RA4)

`hypothesis` is a declared dependency (`pyproject.toml:58`) and used in ~55 files across 13+ `tests/<Module>/property/` dirs with a registered `property` marker (`pyproject.toml`, marker list). But coverage does not track the escaped-defect profile: the parse/serialize/bounds functions where the 45% validation class lives mostly lack invariant tests.

**Ranked candidates** (function → invariant):

1. `app/core/Character_Chat/ccv3_parser.py` — card parse/serialize round-trip preserves all fields; invalid cards fail consistently. Plus PNG tEXt chara embed/extract round-trip (`app/api/v1/endpoints/characters_endpoint.py:~2961`, `_encode_png_with_chara_metadata`): arbitrary card JSON + arbitrary base PNG survives byte-exact.
2. `app/core/config_sections/*.py` (e.g. `chunking.py` `_parse_bool`/`_parse_int`) — never raise on arbitrary input; default on garbage; idempotent.
3. `app/core/Chunking/chunker.py` — reconstruction (concat of chunks minus overlap == source), monotone offsets, overlap ≤ chunk size. Property files exist (`tests/Chunking/test_chunking_offsets_property.py`, `test_chunking_overlap_properties.py`, `test_sentence_spans_properties.py`) — extend, don't duplicate.
4. `app/core/Notes_Tasks/markdown_parser.py` — parse/render round-trip, hierarchy preservation.
5. Chat-macro parser bounds (named escaped-defect source), pagination invariants (non-overlap, completeness), JSON fence extraction (output always parseable), chatbook export/import round-trip.

### 2.5 Singleton/lifecycle isolation is undetectable by the suite (RA5)

The 40% state/lifecycle defect class maps to process-global state the test infra neither tracks nor resets:

- Service-layer singleton caches registered against the wrong DB across tests (#2580).
- `reload_app_main()` permanently swaps `sys.modules`, leaking stale drain state (#2585).
- Embeddings suite drain-state corrupting subsequent suites (#2581).

There is precedent for guarding exactly this failure shape: `tldw_Server_API/tests/_plugins/http_client_patch_guard.py` fails tests that leave httpx patched. No equivalent exists for singletons, module identity, or drain state. Test-order dependence is never exercised (no shuffle job).

### 2.6 Env-matrix blindness (RA6)

The 20% env/config defect class escapes because CI always runs with its convenience env vars set. There are no systematic "env-absent" tests (delete all module env vars, assert real-deployment defaults), and auth-mode × DB-backend combinations are exercised ad hoc rather than parametrized for the riskiest modules (AuthNZ, Jobs, egress policy, Chat). The infra to do this cheaply already exists (`tests/_plugins/authnz_fixtures.py`, `tests/_plugins/postgres.py` `pg_temp_db`).

### 2.7 Scale requires mechanized triage (RA7)

~3,924 test files cannot be hand-audited. The patterns in RA1–RA3 are AST-detectable: mock density per test function, status-only assertion sets, `status in (200, 5xx)` accepts, assert-targets configured in the same function. The shard-coverage guard (`Helper_Scripts/ci/check_shard_coverage.py` + ratchet baseline) is the in-repo pattern to copy.

## 3. Frontend Findings

Tooling is healthy: Vitest 4 (108 unit-test files + ~1,880 tests in `packages/ui`), tiered Playwright (52 smoke + 93 workflow specs), no jest remnants, no snapshot abuse, polling instead of sleeps, and a selector-drift guard (`__tests__/e2e-page-object-contracts.guard.test.ts`) worth extending.

### 3.1 API contract drift has no gate (RF1)

53 e2e spec files hand-write `page.route()` mock JSON (e.g. `e2e/workflows/chat-rails-collapse.spec.ts:25–59`) with **no schema link** to the backend's OpenAPI spec. No OpenAPI codegen exists anywhere in the frontend (verified: no `openapi-typescript`/`orval`). This is the direct mechanism behind "frontend and backend each pass their own tests while the integrated system breaks" (#2590 class).

### 3.2 Real-backend coverage is minimal and out of CI (RF2)

Only 2 real-backend specs exist (`e2e/workflows/research-workspace.real-backend.spec.ts`, `chat-cockpit.real-server.spec.ts`); both env-gated (`TLDW_E2E_SERVER_URL`/`TLDW_E2E_API_KEY`) and run manually, not in any workflow. Neither covers media ingest, knowledge base, or settings.

### 3.3 Global mocks mask real behavior (RF3)

`apps/tldw-frontend/vitest.setup.ts:44–85` globally mocks Dexie (IndexedDB never exercised) and `:92–182` globally stubs React Query — every unit test runs against shapes that can drift from the real API/DB. 54 conditional `test.skip()` calls across workflow specs further mask environment-dependent gaps.

## 4. Infrastructure Inventory (reuse, don't rebuild)

- `tldw_Server_API/tests/_plugins/`: `postgres.py` (`pg_temp_db` ephemeral DBs), `chat_fixtures.py` (mock-OpenAI bootstrap), `authnz_fixtures.py`, `http_client_patch_guard.py` (guard-plugin precedent), `media_fixtures.py`, e2e fixtures.
- `mock_openai_server/` — full OpenAI-compatible mock server.
- `Helper_Scripts/ci/run_local_ci.py` (`make ci-local`), `check_shard_coverage.py` + `shard_coverage_baseline.txt` (ratchet mechanics).
- CI: ~25 shards in `ci.yml`; coverage gates 12% global / 35% AuthNZ (`coverage-required.yml`); `tests/CI/test_skip_markers_have_reasons.py` enforces skip reasons (round-1 F9).

## 5. Findings Table

| ID | Finding | Importance (1–10) | Defect class addressed |
|---|---|---|---|
| RA1 | Integration-marked tests stub the integration layer | **8** | validation 45% + integration 10% |
| RA2 | Tautological / assertion-in-the-mock tests | **7** | all (false confidence) |
| RA3 | Tolerated-failure and status-only assertions | **7** | all (false confidence) |
| RA5 | No singleton/lifecycle leak detection; no order-shuffle | **8** | state/lifecycle 40% |
| RA6 | No env-absent tests; no auth×DB matrix for risky modules | **7** | env/config 20% |
| RA4 | Property tests not targeted at validation-bug surfaces | **7** | validation 45% |
| RA7 | No mechanized test-quality triage at 3.9k-file scale | **6** | enabler for RA1–RA3 |
| RF1 | Frontend API mocks unlinked to OpenAPI spec — drift ungated | **8** | env/config + integration |
| RF2 | Real-backend e2e minimal, out of CI | **6** | integration |
| RF3 | Global Dexie/React-Query mocks in vitest.setup.ts | **5** | integration |

## 6. Remediation

See `Docs/superpowers/plans/2026-07-04-test-suite-improvement-implementation-plan.md` (Stages 1–5, one PR per stage slice).
