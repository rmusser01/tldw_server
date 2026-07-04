# Test Suite Audit — Round 2 (Defect-Driven)

**Date:** 2026-07-04 (figures recounted after adversarial fact-check review, same date)
**Scope:** Backend pytest suite (`tldw_Server_API/tests/`) + frontend Vitest/Playwright (`apps/tldw-frontend/`)
**Method:** Escaped-defect analysis of git/issue history (April–July 2026) cross-referenced against test-suite design patterns; four parallel exploration passes (infra map, quality sampling across 8 modules, defect history, frontend infra), followed by an adversarial fact-check pass whose corrections are folded in.
**Relationship to prior audit:** Builds on `audits/2026-07-02-testing-implementation-audit.md` (F1–F10, remediated in PR #2579). This round asks a different question: *why did recently found bugs escape a suite of ~3,976 test files that was assumed to cover them?* Findings already resolved by round 1 (coverage-gate ratchet F1, un-hiding of `norecursedirs`-excluded dirs F3, skip-reason hygiene F9) are not re-reported.

---

## 1. Defect Profile (April–July 2026)

~20 concrete defects that tests did not catch were identified from bug-fix commits and issues.

**Provenance caveat:** 7 of the 13 cited commits (the chat-macro series, egress policy, ACP WS cleanup) live on unmerged `codex/*` feature branches — those bugs were caught by review/UAT *before* merge, not shipped. They still support this audit's premise (tests didn't find them; humans did), but they are not production escapes. The remainder (the #2580/#2581/#2585/#2590 cluster and dev-merged fixes) did land on integration branches or affect users.

| Category | Share | Representative cases |
|---|---|---|
| **Missing input validation / contract enforcement** | ~45% | Chat macro parser missing bounds validation (c1f4e6eb95, pre-merge); Jobs operation contracts allowing impossible states (a9b6a2c310); integrations HTTP bypassing central egress policy (883b6c4dbd, pre-merge); workspace ingest failure codes not exposed (c725caad5a) |
| **State management / lifecycle** | ~40% | Service-layer singleton caches registered against the wrong DB — root cause per fix commit 4924719264 (#2580); `reload_app_main()` permanently swapping `sys.modules` (#2585); Embeddings drain-state corrupting subsequent tests (#2581, 254af77776); ACP WS broadcaster not cleaned up on disconnect (790e3f3264, pre-merge); chat-macro run-state races (7f7820395d, a4feb40ee8, 8f48bd6bc8, pre-merge) |
| **Env/config-dependent behavior** | ~20% | Web shell auth lost on hard reload — no runtime-override fallback (#2590, 626447bd5c); UX smoke gate hidden after credential hardening (e88c96500f) |
| **Cross-module integration breaks** | ~10% | Bit-rotted multiuser load-test helper signature (4924719264) |
| **Serialization / round-trip** | ~10% | Runtime auth credentials not durable across hard reloads (#2590) |

A related find: commit 4b89ce40a2 repaired a *regression test that never invoked the code it claimed to test* (audio download limits, AUDIT-2026-06-27-MEDIA-004) — not a shipped defect, but direct evidence for the tautological-test class (§2.2).

**Recurring escape mechanism:** a refactor hardens one path (e.g. credential storage) but forgets the fallback/compat layer; tests pass in CI (where env vars are set, singletons are warm, and mocks stand in for the integrated system) but the deployment-shaped world differs.

## 2. Backend Findings

### 2.1 "Integration" tests that don't integrate; endpoint tests smothered in mocks (RA1)

458 test files carry `pytest.mark.integration`; sampled integration-marked tests frequently stub the very layer they claim to integrate:

- `tldw_Server_API/tests/RAG/test_dual_backend_end_to_end.py` — named "end to end" and integration-marked (:59, :88, :107), but `_StubVectorStore` (:133–142) hardcodes vector search results; the vector-search path is never exercised.
- `tldw_Server_API/tests/Media/test_media_navigation.py` — endpoint tests (httpx `AsyncClient`, not integration-marked) where single tests stack up to 6 patches (27 `patch`/`patch.object` calls file-wide: `get_storage_backend`, `_extract_pdf_outline`, `get_cached_response`, `cache_response`, …); assertions check field presence and status only.

**Consequence:** cross-module breaks (defect category ~10%, plus contributing to the 45% validation class) sail through because no test wires the real components together.

### 2.2 Tautological / assertion-in-the-mock tests (RA2)

- `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py` — collects monkeypatched calls into a list and asserts the list (`assert calls == [db]` at :32, :49, :67; `assert coordinator_calls == [db]` at :131, :155). The real `initialize_sqlite_schema` never runs — the dispatch logic of `ensure_media_schema` is partially real, but the schema bootstrap itself is only ever the monkeypatch.
- `tldw_Server_API/tests/AuthNZ/test_auth_dependency_contract.py:175–189` — `_FakeJwtService` asserts its own hardcoded token (`assert token == "jwt.header.signature"` at :180, :186) *inside the mock*; real token validation is never touched. Large stretches (:294–660) assert identity of imported names (`assert RequireRole is auth_deps.RequireRole`), testing wiring, not authorization logic.
- The 4b89ce40a2 case above (a regression test that never called the downloader) is the same class caught in an earlier audit.

### 2.3 Tolerated-failure assertions (RA3)

- `tldw_Server_API/tests/Character_Chat/test_complete_v2_streaming_with_mock_openai.py:58` — `assert response.status_code in (200, 502)`. A 502 is a failure; the test passes either way. Streaming-content assertions (:61–64) check only that a line starts with `data: `, then break. (This test also skips unconditionally in CI: `MOCK_OPENAI_BASE_URL` is set in zero workflows despite the file being sharded.)

Status-code-only assertions (no body semantics) are common across sampled modules — endpoints can return wrong data and stay green.

### 2.4 Property-based testing is concentrated, not targeted (RA4)

`hypothesis` is a declared dependency (`pyproject.toml:58`) and used in ~41 files across 11 `tests/<Module>/property/` dirs with a registered `property` marker (`pyproject.toml:573`). But coverage does not track the defect profile: the parse/serialize/bounds functions where the 45% validation class lives mostly lack invariant tests.

**Ranked candidates** (function → invariant; verified against dev — several "round-trips" are one-directional because no serializer exists):

1. `app/core/Character_Chat/ccv3_parser.py` — exposes `parse_v3_card`/`validate_v3_card` only (no serializer): parse idempotence (`parse(parse(x)) == parse(x)`), field preservation, deterministic rejection of invalid cards. Plus PNG tEXt chara embed/extract round-trip — this one is bidirectional: `_encode_png_with_chara_metadata` (`app/api/v1/endpoints/characters_endpoint.py:2961`) vs `extract_json_from_image_file` (`app/core/Character_Chat/modules/character_io.py:175`).
2. `app/core/config_sections/*.py` (e.g. `chunking.py` `_parse_bool`/`_parse_int` at :41/:31) — never raise on arbitrary input; default on garbage; idempotent.
3. `app/core/Chunking/chunker.py` — reconstruction (concat of chunks minus overlap == source), monotone offsets, overlap ≤ chunk size. Property files exist (`tests/Chunking/test_chunking_offsets_property.py`, `test_chunking_overlap_properties.py`, `test_sentence_spans_properties.py`) — extend, don't duplicate.
4. `app/core/Notes_Tasks/markdown_parser.py` — parse-only (`parse_note_checklists`): never raises on arbitrary input, spans stay within source, hierarchy monotonicity, re-parse idempotence.
5. Pagination invariants (non-overlap, completeness), JSON fence extraction (output always parseable), chatbook export/import round-trip. Chat-macro parser bounds is a strong candidate **once `codex/chat-macros-v1` merges** — the module does not exist on dev yet.

### 2.5 Singleton/lifecycle isolation is undetectable by the suite (RA5)

The 40% state/lifecycle defect class maps to process-global state the test infra neither tracks nor resets:

- Service-layer singleton caches registered against the wrong DB across tests (#2580, root cause per 4924719264).
- `reload_app_main()` permanently swaps `sys.modules`, leaking stale drain state (#2585, open).
- Embeddings suite drain-state corrupting subsequent suites (#2581).

There is precedent for guard plugins of this shape: `tldw_Server_API/tests/_plugins/http_client_patch_guard.py` blocks tests at patch-time when they try to monkeypatch `requests`/`httpx`/`aiohttp` directly. No equivalent exists for singletons, module identity, or drain state. Test-order dependence is never exercised (no shuffle job).

### 2.6 Env-matrix blindness (RA6)

The 20% env/config defect class escapes because CI always runs with its convenience env vars set — and the test conftest *forces* many of them at import time (`AUTH_MODE=single_user`, `DATABASE_URL`, `WORKFLOWS_EGRESS_BLOCK_PRIVATE=false`, `WORKFLOWS_WEBHOOK_ALLOWLIST=*`, …), so no in-pytest test ever sees a deployment-shaped environment by default. There are no systematic "env-absent" tests (delete all module env vars + reset cached settings, assert real-deployment defaults), and auth-mode × DB-backend combinations are exercised ad hoc rather than parametrized for the riskiest modules (AuthNZ, Jobs, egress policy, Chat). The infra to do this cheaply already exists (`tests/AuthNZ/conftest.py` fixtures, `tests/_plugins/postgres.py` `pg_temp_db`).

### 2.7 Scale requires mechanized triage (RA7)

~3,976 test files cannot be hand-audited. The patterns in RA1–RA3 are AST-detectable — with the caveat that mock-density alone misses stub-*class* injection (the RAG example uses hand-rolled classes assigned by attribute, and the AuthNZ example injects fakes via `dependency_overrides`), so detectors must also count test-module-defined fake classes and dependency-override density. The shard-coverage guard (`Helper_Scripts/ci/check_shard_coverage.py` + ratchet baseline) is the in-repo pattern to copy.

## 3. Frontend Findings

Tooling is healthy: Vitest 4 (108 unit-test files in `__tests__/` + ~1,889 test files in `apps/packages/ui`), tiered Playwright (53 smoke + 93 workflow specs), no jest remnants, no snapshot abuse, polling instead of sleeps, and a selector-drift guard (`__tests__/e2e-page-object-contracts.guard.test.ts`) worth extending.

### 3.1 API contract drift has no gate (RF1)

53 e2e spec files hand-write `page.route()` mock JSON (e.g. `e2e/workflows/chat-rails-collapse.spec.ts:25–59`) with **no schema link** to the backend's OpenAPI spec. No OpenAPI codegen exists anywhere in the frontend (verified: no `openapi-typescript`/`orval`). This is the direct mechanism behind "frontend and backend each pass their own tests while the integrated system breaks" (#2590 class).

### 3.2 Real-backend e2e exists but coverage is narrow (RF2)

More real-backend coverage exists than a directory listing suggests: `e2e/workflows/research-workspace.real-backend.spec.ts` runs nightly in CI (`.github/workflows/ui-research-workspace-nightly.yml`, including a strict no-skip extension variant against a live server), `e2e/real-server-workflows.spec.ts` (~17 tests, plus an extension mirror) is gated on the same `TLDW_E2E_SERVER_URL`/`TLDW_E2E_API_KEY`, and admin-ui real-backend e2e runs in `frontend-required.yml`. The actual gaps: `chat-cockpit.real-server.spec.ts` is not wired into any workflow, and no real-backend spec covers media ingest, knowledge base, or settings for the main WebUI. Extend the existing nightly rather than build new scaffolding.

### 3.3 Global mocks mask real behavior (RF3)

`apps/tldw-frontend/vitest.setup.ts:44–85` globally mocks Dexie — IndexedDB behavior is never exercised in unit tests. The React Query wrapper (:92–182) is more nuanced: it preserves real hook behavior when a `QueryClientProvider` is mounted and only falls back to inert shapes when none is set — the risk is suites silently running in the inert-fallback mode. 54 conditional `test.skip()` calls across workflow specs further mask environment-dependent gaps.

## 4. Infrastructure Inventory (reuse, don't rebuild)

- `tldw_Server_API/tests/_plugins/`: `postgres.py` (`pg_temp_db` ephemeral DBs), `chat_fixtures.py` (mock-OpenAI bootstrap — non-streaming responses only, as of this audit), `authnz_fixtures.py`, `http_client_patch_guard.py` (guard-plugin precedent), `media_fixtures.py`, e2e fixtures. Real JWT service fixture: `jwt_service` at `tests/AuthNZ/conftest.py:1878`.
- `mock_openai_server/` — full OpenAI-compatible mock server; in-process deterministic SSE precedent at `tests/Character_Chat/test_complete_v2_streaming_e2e_mock.py`.
- `Helper_Scripts/ci/run_local_ci.py` (`make ci-local`), `check_shard_coverage.py` + `shard_coverage_baseline.txt` (ratchet mechanics).
- CI: 27 jobs in `ci.yml` fanning out to ~195 named shards; coverage gates 12% global / 35% AuthNZ (`coverage-required.yml:79,91`); `tests/CI/test_skip_markers_have_reasons.py` enforces skip reasons (round-1 F9).
- Note: the `plugins = [...]` list under `[tool.pytest.ini_options]` in pyproject.toml is **not a pytest option and is silently ignored**; plugins actually load via `pytest_plugins` in the conftests (e.g. `tldw_Server_API/tests/conftest.py:8`).

## 5. Findings Table

| ID | Finding | Importance (1–10) | Defect class addressed |
|---|---|---|---|
| RA1 | Integration-marked tests stub the integration layer; endpoint tests smothered in mocks | **8** | validation 45% + integration 10% |
| RA2 | Tautological / assertion-in-the-mock tests | **7** | all (false confidence) |
| RA3 | Tolerated-failure and status-only assertions | **7** | all (false confidence) |
| RA5 | No singleton/lifecycle leak detection; no order-shuffle | **8** | state/lifecycle 40% |
| RA6 | No env-absent tests; no auth×DB matrix for risky modules; conftest force-sets env | **7** | env/config 20% |
| RA4 | Property tests not targeted at validation-bug surfaces | **7** | validation 45% |
| RA7 | No mechanized test-quality triage at ~4k-file scale | **6** | enabler for RA1–RA3 |
| RF1 | Frontend API mocks unlinked to OpenAPI spec — drift ungated | **8** | env/config + integration |
| RF2 | Real-backend e2e coverage narrow (chat-cockpit spec unwired; no media/KB/settings) | **4** | integration |
| RF3 | Global Dexie mock; React Query inert-fallback mode can mask missing providers | **5** | integration |

## 6. Remediation

See `Docs/superpowers/plans/2026-07-04-test-suite-improvement-implementation-plan.md` (Stages 1–5, one PR per stage slice).
