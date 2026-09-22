# Core-Module Duplication and Correctness Synthesis

**Date:** 2026-09-21
**Scope:** The ten largest `tldw_Server_API/app/core` modules plus `app/api/v1/endpoints/` — ~800,000 LOC of the repo's ~1,686,000. Modules: `DB_Management` (233,293), `MCP_unified` (141,588), `AuthNZ` (74,539), `RAG` (58,804), `Ingestion_Media_Processing` (55,842), `Sync` (43,905), `TTS` (42,067), `Evaluations` (35,805), `Chat` (25,698), `LLM_Calls` (21,841), `api/v1/endpoints` (247,913).
**Method:** One reviewer per module (11 in parallel), each applying `Codeslop-Vibecheck-SKILL.md` extended from three axes to five (duplication, encapsulation, sequential coupling, **correctness**, **efficiency**), each axis carrying an explicit drop rule. Findings were then merged across modules — one finding per piece of shared knowledge, not one per module — and ranked. The orchestrator independently re-verified every High-severity claim and every claim that contradicted another reviewer or the review prompt itself.
**Inclusion rule:** A finding is included only if it cites `file:symbol (line-range)` verified in today's tree, and clears its axis's drop rule — a concrete failure scenario for correctness, a named cost driver and scaling factor for efficiency, and a statement of the shared knowledge that will diverge plus its change-amplification cost for duplication. Grep counts are never findings. Items that failed these tests are listed in [§7](#7-dropped-and-disproved) rather than silently omitted.
**Relationship to prior reviews:** `rag/` and `db-management/` had complete ledgers dated 2026-04-07; both were **extended, not redone**, with explicit still-live vs already-addressed reconciliations ([§6](#6-prior-findings-reconciliation)). `evals-module/`, `auth-dependencies/`, `api-pagination/`, `api-response-envelope/`, `characters-backend/`, `moderation-backend/`, `web-scraping/`, `shared-api-client/` were read and inherited from; findings they already own are not re-reported. One finding (M1) was **already filed** as `task-13287` by a prior smoke run of this same review prompt and is reported as confirmed-still-live, not as a discovery.

---

## Executive summary

The review was commissioned to find functionality reproduced across modules that should live in a shared utility. It found that, and the shape is as predicted: **the canonical helper usually already exists and is bypassed.** But three of the most valuable results invert the premise.

**First, the canonical is often the wrong thing to adopt.** Three designated shared helpers are not merely unadopted — they are *unadoptable*, and a naive "adopt the canonical" campaign would have propagated defects to dozens of sites. `api/v1/utils/datetime_utils.py:coerce_datetime` returns naive datetimes unchanged, makes tz-awareness depend on input format, and silently substitutes `now()` for unparseable input. `core/AuthNZ/repos/datetime_utils.py:_strip_tzinfo` implements the *lossy* one of three timezone semantics. `core/LLM_Calls/tokenizer_resolver.py:resolve_tiktoken_encoding` raises by design to serve strict token counting, which is precisely why eleven sites rewrote it. In each case the fix is to repair or split the canonical first, then migrate.

**Second, the duplication is a bug generator, and the bugs are live.** Nine High-severity correctness defects arose directly from copies drifting apart, four of them reproduced at runtime during this review: a public endpoint that returns HTTP 500 unconditionally on the default database backend; silent permanent per-device data loss in Sync on the default client path; an MCP input sanitizer that strips tab characters and thereby corrupts every file written through it (see the correction in §2.3: the sanitizer had five divergent copies, and the companion claim about its edit tool was wrong); and a timestamp converter that violates a binding ADR by the host's UTC offset.

**Third, the largest systemic finding is not duplication at all — it is that substantial test suites sit outside every contractual CI gate.** `app/core/MCP_unified/tests/` is 149 files, 63.6% of that module's lines, and appears in zero `ci.yml` shards; a test there has been failing on the branch since 2026-06-03 across 28 subsequent commits to the very file it covers. `tests/Sync/` aborts at collection without `psycopg`, so all 2,958 of its tests are interrupted rather than skipped, and a real assertion failure has sat red ~18 days. These are the mechanism by which the other findings survive.

A fourth result concerns the review instrument itself: **three of the prompt's own seed facts did not survive verification** ([§7](#7-dropped-and-disproved)). That is the strongest available argument for its own rule to re-confirm every citation before using it.

---

## 1. Ranked priority table

Weight = severity × blast radius × (1/effort), where blast radius is site count × churn of the files involved, and effort is gated by measured test reachability. Sorted by weight descending.

| ID | Finding | Weight | Class |
|---|---|---|---|
| **F1** | `list_optimizations` exists only on the PostgreSQL class; a public GET 500s unconditionally on the **default** backend | 96 | High / divergent-copies |
| **F2** | Sync pull cursor advances past withheld envelopes on the default non-negotiated client path — silent permanent per-device data loss | 94 | High / divergent-copies |
| **F3** | MCP sanitizer strips `\t`: corrupts every `fs.write`. ~~permanently breaks `fs.edit` on tab-indented files~~ — disproved on fixing, see §2.3 | 92 | High / divergent-copies |
| **F4** | BLE001 remediation **widened** a catch to swallow `asyncio.CancelledError`; also swallows a 413 quota rejection | 90 | High / divergent-copies |
| **F5** | MCP base denylist rejects `--` and `/*` — ordinary content hard-fails on 22 inherited modules | 88 | High / divergent-copies |
| **F6** | Evaluations `created` timestamps violate **ADR-014** by the host's UTC offset; CI is UTC so no test can see it | 86 | High / divergent-copies |
| **F7** | 149-file MCP test tree in **zero** CI shards; a test red since 2026-06-03 across 28 commits to the same file | 85 | High / test-estate |
| **F8** | `tests/Sync` aborts at collection without `psycopg`; behind it, **12 red tests** in a 3-hour suite no blocking gate runs | 84 | High / test-estate |
| **F9** | OCR temp image closed and unlinked before the request that names it — silent total data loss, every page | 82 | High / divergent-copies |
| **F10** | Resampler returns input unchanged on ImportError, caller relabels 48 kHz as 16 kHz — 3×-speed garbage | 80 | High / divergent-copies |
| **F11** | Unkeyed model cache returns the **wrong model** while reporting the requested name | 78 | High / divergent-copies |
| **F12** | Double-escaped regex misclassifies upstream **429 as 502** — *already filed as `task-13287`*; two new compounding defects found | 76 | High / divergent-copies |
| **F13** | RAG alias keys double-count in the overall score — 0.667 where 0.5 is correct, moving pass/fail | 74 | High / correctness |
| **F14** | ElevenLabs cleanup closes the **process-wide shared** pooled HTTP client; one transient failure becomes permanent | 72 | High / divergent-copies |
| **F15** | Scalar/env coercion: ~113 named private re-implementations; three truthy sets gate SSRF and transport decisions | 70 | High / **M3** cluster |
| **F16** | Eleven incompatible relevance-score scales fused by one global sort — wrong documents reach generation | 68 | High / correctness |
| **F17** | `create_session_with_retries` returns a **different class** under pytest than in production | 66 | High / correctness |
| **F18** | Unauthenticated HTTP 500 on the public share-token route (decode outside the `try`) | 64 | High / divergent-copies |
| **F19** | Raw SQL in 20 endpoint files, 5 reaching into core **private** APIs — direct `Docs/Architecture.md` violation | 62 | High / adoption-gap |
| **F20** | `PromptStudioDatabase.py`: ~5,565 LOC implemented twice + a third delegating facade; 1 missing method, 7 signature mismatches | 60 | High / divergent-copies |
| **F21** | Retry/contention policy exists on SQLite only — 28 inline loops, **zero** on PostgreSQL | 58 | High / divergent-copies |
| **F22** | Base64 cursor/token idiom: 23 production sites, 3 strictness postures, 6 error contracts, **2 trust classes** | 56 | Medium / **M2** cluster |
| **F23** | Datetime/timestamp: canonical is **unadoptable**; 39 private helpers, 239 `utcnow()`, 4 wire formats | 54 | Medium / **M5** cluster |
| **F24** | Session-key derivation bypasses the module's own anti-drift helper, diverging on all four sub-decisions | 52 | High / adoption-gap |
| **F25** | Discord/Slack: three source clone pairs (78.6%, **91.2%**, 61.3%) and four test clone pairs | 50 | High / true-duplication |
| **F26** | Reranker model materialized **per request**; 201 blocking queries behind one persona endpoint | 48 | High / efficiency |
| **F27** | Likert normalization re-derived 16 times, 5 answers; the correct one has **zero** callers and the tests pin it | 46 | High / divergent-copies |
| **F28** | Webhook fallback query omits `user_id` under `TEST_MODE` — other users' rows **including secrets** | 44 | Medium / correctness |
| **F29** | Quadratic `bytes +=` accumulation in four adapters — **measured 1,032×** at 8,000 chunks | 42 | High / efficiency |
| **F30** | Retry/backoff: 8 implementations; the best is buried in a 6,600-LOC junk drawer | 40 | Medium / **M6** cluster |
| **F31** | tiktoken fallback rewritten 11× because the canonical **raises by design**; 3 sites crash on `model: null` | 38 | Medium / **M7** cluster |
| **F32** | SSE frame contract (ADR-025) has 33 inline implementations in one module and 9 more in adapters | 36 | Medium / **M8** cluster |
| **F33** | No written rule for which test tree a test belongs in; CI mislabels the more active Chat tree "legacy" | 34 | Medium / **M9** cluster |
| **F34** | `core/` → `api/` layering: **33 production true inversions across 20 files**; existing ratchet guards one direction of a two-way cycle | 32 | Medium / **M10** cluster |
| **F35** | Blob upload sessions leak on any failure; 8 transient failures **permanently** disable attachments per user | 30 | High / correctness |
| **F36** | `_row_to_dict`/JSON coercion: 17 schema-agnostic adapters, 2 incompatible null policies | 28 | Medium / divergent-copies |
| **F37** | Six MCP `_is_admin` definitions; a platform **owner** is refused, a `system.configure` key reaches another user's sandbox | 26 | High / adoption-gap |
| **F38** | `media_metadata.py` ≡ `source_cache.py` — 187 identical lines, 28 identifier-only differences | 24 | Medium / true-duplication |
| **F39** | 614 lines of dead production orchestration code that still takes commits, with a real behavioural divergence | 22 | Medium / true-duplication |
| **F40** | Dead and test-only helpers — **delete, do not migrate** (see [§5](#5-dead-helpers)) | 20 | Low / dead-code |

---

## 2. Correctness findings (verified)

### 2.1 A public endpoint is unconditionally broken on the default backend (F1)
`core/DB_Management/PromptStudioDatabase.py` — `list_optimizations` is defined on `_BackendPromptStudioDatabase` (`:2122-2183`) and **absent** from `_SQLitePromptStudioDatabase` (`:3848-7141`), while the facade `PromptStudioDatabase.list_optimizations` (`:7378`) delegates unconditionally. **ADR-020 makes SQLite the default content backend.**

Reproduced by the orchestrator: MRO is `['_SQLitePromptStudioDatabase','PromptsDatabase','object']` — no inherited fallback — and `hasattr` returns `backend=True, sqlite=False, facade=True`. `GET /api/v1/prompt-studio/projects/{id}/optimizations` therefore raises `AttributeError`, caught by `_OPTIMIZATION_NONCRITICAL_EXCEPTIONS` (`prompt_studio_optimization.py:1019`), and returns HTTP 500 "Failed to list optimizations" for every request on every project.

The tests pass because both `test_api_endpoints.py:369-380` and `test_optimization_endpoint_error_mapping.py:32,37` substitute a stub `list_optimizations`. **Effort:** cheap (port ~60 lines). **Owner-only:** yes.

The same file yields F20: seven of 59 paired methods have incompatible signatures (three reproduced at runtime — `get_prompt(include_deleted=)`, `delete_signature` positional-vs-keyword, `create_bulk_test_cases(client_id=)`; `_format_test_case` differs in **arity**). The facade's `*args/**kwargs` passthroughs erase this from mypy and every IDE.

### 2.2 Sync loses data permanently on the default client path (F2)
`core/Sync/v2/service.py:pull (5252-5260)` — the legacy adapter-v1 branch computes `next_sequence = max(e.server_sequence for e in raw_envelopes)` with **no blocker filter**, while `_scan_pull_page (10658-10700)` deliberately distinguishes `visible` from `raw`. The correct implementation is 5,000 lines away in the same file at `_pull_versioned (10245-10261)`, which derives the boundary from `safe_raw_envelopes` and states the rule in its docstring at `:10172`.

Reproduced end-to-end by the Sync reviewer on SQLite: a device that never negotiated `supported_adapter_versions` (the default) pulls; an envelope at seq 1 is an unresolved ordering blocker per **ADR-034**; seq 2 is deliverable. The pull returns `envelopes=[]`, `next_cursor="2"`, **`has_more=False`**. Seq 2 is never delivered, and the client was told it was caught up. The repo's own `test_versioned_pull_does_not_advance_past_unresolved_conflict` proves the v2 path is correct; no v1 equivalent exists. **Effort:** cheap — one shared `_advance_pull_watermark` helper and a copy of the existing test.

### 2.3 The MCP sanitizer corrupts written files and breaks its own edit tool (F3, F5)
`core/MCP_unified/modules/base.py:sanitize_input (766-807)` is applied to **every** tool call via `tool_execution/security.py:harden_and_sanitize_tool_arguments (614-626)`. Two defects, both executed against the real classes:

- **Tab stripping (F3).** The control-character filter keeps `\n` but drops `\t` and `\r`. `fs.write` of a Makefile writes `all:\ngcc -o x x.c` — the required tab gone — and **reports success**; the `expected_sha256` receipt hashes the on-disk pre-image so the integrity guard structurally cannot catch it. The correct whitespace class is written 15 lines away in the same file at `_sanitize_patch_diff (1480-1485)`.

  > **Correction, recorded on fixing this (commits `475bdfb929`, `6a16f76e48`).** Two claims above were wrong, and one of them hid the actual root cause.
  >
  > 1. ~~`fs.edit` is permanently unusable on tab-indented files.~~ **False.** `old_string` and `new_string` have been exempt from sanitization since `f57da0ef3a`, so the tab always survived. The regression test written for this claim passes against the pre-fix source.
  > 2. ~~The exemption table at `filesystem_module.py:609-626` is dead.~~ **False, and inverted.** It is load-bearing — it is what keeps file content byte-exact — and the defect was that it was *incomplete*: `fs.edit` and `notebook.edit_cell` were exempt while `fs.write`/`fs.write_text` were not, so `fs.write` also stripped form feeds (`\x0c`), the conventional page separator in Python, Lisp and C sources. Now one `_VERBATIM_ARGS` table covering all four tools.
  > 3. **Missed entirely:** fixing `base.py` alone fixes nothing for the reported path. `filesystem_module` carried its own `sanitize_input` override with the identical defect, shadowing the base — as did `run_command`, `sandbox` and `web_tool_base`, each drifted to a different whitespace class. This is the F3/F5 duplication thesis holding more strongly than the finding stated: the sanitizer had *five* copies, not one, and the fix was to delete four of them.
- **Denylist (F5).** `:779-788` rejects `["';", '";', "--", "/*", "*/", "xp_", "sp_"]`. Executed: `NotesModule.sanitize_input({"content": "Heading\n---\nbody"})` raises `ValueError` → JSON-RPC `-32602`. Also rejected: `"run tests in src/*.py"`, `"SELECT 1 -- note"`, git pathspec `"-- src/app.py"`, and the filename `"exp_data.csv"`. These are pure data handed to parameterised queries, so the denylist buys no injection protection. 22 modules inherit it unchanged; `web_tool_base.py:51-70` already fixed it for web tools only, and its docstring diagnoses the exact problem. The base's only sanitizer test uses `os.urandom(4).hex()` as its "safe" fixture — a hex string can never contain a denied substring, so nothing asserts legitimate content survives.

### 2.4 The BLE001 remediation widened a catch to swallow cancellation (F4)
`core/Ingestion_Media_Processing/persistence.py:83-105` defines `_PERSISTENCE_NONCRITICAL_EXCEPTIONS` with **`asyncio.CancelledError` as its first entry**. Verified: `CancelledError.__mro__` is `[CancelledError, BaseException, object]` and `issubclass(CancelledError, Exception)` is **False** since Python 3.8.

The `except Exception:` this tuple replaced would have let cancellation propagate correctly. The explicit tuple written to satisfy BLE001 now swallows it, across 12 `contextlib.suppress` sites (`:872, 1005, 2118, 2236, 3971, 3979, 4172, 5087, 5090, 5093, 5096, 5721`), several wrapping awaits. A client disconnect mid-`POST /media/add` is absorbed; the coroutine keeps writing to the media DB for a gone client and `Task.cancel()` never completes, so graceful shutdown blocks. The same tuple contains `HTTPException` (`:101`) while the file raises it 11 times — the 413 over-quota raise at `:4994` is swallowed at `:5079`, so an over-quota upload returns 200/207.

**This is the inverse of the noise floor.** The review was told not to report grandfathered BLE001 files; here the *remediation* introduced a defect the lint rule cannot see. Sibling tuples with the same defect: `Audio/Audio_Streaming_Unified.py:80`, `Audio/Audio_Transcription_Lib.py:93` (currently inert). **Durable guard:** a `tests/lint/` AST rule rejecting any `BaseException`-derived member in a `*_NONCRITICAL_EXCEPTIONS` tuple.

### 2.5 A binding ADR is violated by the host's UTC offset (F6)
`core/DB_Management/Evaluations_DB.py:_ensure_unix_timestamp (2489-2504)` does `datetime.fromisoformat(s.replace("Z","+00:00"))` then `int(dt.timestamp())`. Verified by the orchestrator under `TZ=America/Los_Angeles` against SQLite's own `CURRENT_TIMESTAMP` format:

- `.replace("Z","+00:00")` is a **no-op** — that format has no `Z`.
- `fromisoformat` returns a **naive** datetime.
- `.timestamp()` interprets it in the **host's local zone**: `1790050015` vs a true UTC `1790024815`. **Delta 25,200 s — exactly 7 h.**

**ADR-014:12** names Unix `created` timestamps as a preserved OpenAI-compatible convention, so this is a binding-contract violation, not merely a bug. The `except` fallback on the same function returns `int(datetime.now().timestamp())` — also naive. Three further copies carry the identical defect (`unified_evaluation_service.py:1503`, `evaluations_datasets.py:51`, `evaluations_rag_pipeline.py:72,116,174`). **On PostgreSQL** the datasets converter matches no `isinstance` branch and falls through to `now()`, so every dataset's `created` becomes the time it was *read* — the same field, two different wrong answers, split by backend.

**No test catches it because CI runs UTC**, where the delta is exactly zero. For a self-hosted product whose target deployment is a user's own machine, UTC-only CI is the wrong validator. **The single cheapest durable fix in this report: run one CI shard under a non-UTC `TZ`.**

### 2.6 Remaining verified correctness defects
- **F9** — `OCR/backends/dots_ocr.py:195-198` and `hunyuan_ocr.py:192-195` use `NamedTemporaryFile(delete=True)`, so the file is unlinked at the `with` exit while the POST at `:225`/`:222` still names its path. Returns `""`; `_ocr_pdf_pages` counts only non-empty pages, so the run reports success with **zero content, every page**. `nemotron_parse.py:350-387` does it correctly next door.
- **F10** — `Audio/Audio_Buffered_Transcription.py:_resample (534-541)` returns the input unchanged on ImportError; callers at `:384-386` and `:752-754` then set `sample_rate = 16000` **unconditionally**. Without librosa, 48 kHz audio is declared 16 kHz: 3×-speed garbage, all timestamps 3× short, HTTP 200. Three of six resamplers fail open; `Audio_Transcription_Lib.py:336-355` is correct.
- **F11** — `Audio_Transcription_Parakeet_MLX.py:_mlx_model_cache (43)` is a bare `Optional[Any]`, not keyed. A request for `parakeet-tdt-1.1b` receives the cached **0.6b** model and logs "Using cached model" while reporting the 1.1b name upstream. `Nemo` and `Parakeet_ONNX` are keyed but **unlocked** (check-then-act), so concurrent Canary loads resident two 1–3 GB models. Six correct examples exist in the same repo.
- **F12** — the double-escaped `r"HTTP\\s+(\\d{3})"` at `LLM_Calls/error_utils.py:145` and `Chat/chat_orchestrator.py:268`. Proven dead by execution; `Local_LLM/http_utils.py:72` is correct. Traced: extraction returns `None` → `ChatProviderError` → default **502** (`core/exceptions.py:865-879`), so an upstream **429 reaches the client as 502** with no `Retry-After`. **Already filed as `task-13287`** (To Do, high, untracked) by a prior smoke run of this same prompt. Two findings are *new*: (a) `NetworkError` is absent from `_CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS (111-116)`, so the `ChatProviderError(504)` branch at `:576` is unreachable — **fixing the regex alone does not fix the Chat path**; (b) a fourth copy at `Embeddings_Create.py:174` has no regex branch and inverts the attribute precedence.
- **F13** — `rag_evaluator.py:381-385` setdefaults alias keys pointing at the same metric dicts; the dedup at `:401-409` runs only `if not explicit_metrics`, and `eval_runner.py:1478` always passes a list. Executed: overall **0.667 with alias dupes vs 0.5 without**, propagating into `avg_score`, the 0.7 pass threshold and `mean_score`.
- **F14** — `TTS/adapters/elevenlabs_adapter.py:_cleanup_resources (676-687)` calls `aclose()` on the **process-wide** pooled client from `tts_resource_manager.ConnectionPool` without evicting it from `_pools`. One transient `_fetch_user_voices` failure at init leaves a dead client cached; every subsequent request raises "client has been closed" until restart, defeating ADR-011's cooldown. `openai_adapter.py:529-541` is correct. The only test that touches it **pins the defect**.
- **F17** — `LLM_Calls/chat_calls.py:create_session_with_retries (108-140)` branches on `PYTEST_CURRENT_TEST` and returns `_RetrySession` under pytest but `_SessionShim` in production. These have different POST paths (dedicated client vs default transport). Since `PYTEST_CURRENT_TEST` is set for every test, **no test ever exercises the production object** for Cohere, Moonshot, Zai or the legacy embeddings path.
- **F18** — `api/v1/endpoints/chat.py:_decode_knowledge_qa_share_token (6496-6519)` performs its base64 decode at `:6507`, **outside** the `try` that begins at `:6511`. `binascii.Error` is a `ValueError` and *is* in the except tuple, but never reaches it. `GET /api/v1/chat/shared/conversations/AAAA.A` → unauthenticated HTTP 500. `notes.py:857-912` is the correct sibling.
- **F28** — `Evaluations/webhook_manager.py:_get_webhooks (601-613)` has a fallback `SELECT id, url, secret, ... WHERE active = ?` with **no `user_id` predicate**, gated only on `core/testing.py:is_test_mode` (env var, no pytest check). With `TEST_MODE=1` in a shared environment, a user receives every other user's active webhook rows including `secret`, and payloads are delivered to those URLs. The module's own auth layer uses a stricter predicate (`evaluations_auth.py:52-53`).
- **F35** — `api/v1/endpoints/notes.py:4883-4954` has no `finally` and no compensating `cancel_blob_upload`. Sessions are capped at 8 (`service.py:844`), counted without an expiry predicate (`Sync_DB.py:12604`), `expires_at` is never set and appears in no `WHERE` clause repo-wide, no reaper exists, and the `upload_id` is never returned so the cancel endpoint is unreachable. **Eight ordinary transient failures permanently disable attachment upload for that user.**
- **F37** — six `_is_admin` definitions across MCP modules with six different claim sets. A principal with AuthNZ role `owner` is **refused** permanent delete (every copy tests the literal `"admin"`), while an API key carrying `system.configure` **passes** `sandbox_module.py:142`'s cross-user check and reaches another user's sandbox session. Two copies probe `context.is_admin`, which `protocol_types.RequestContext (58-95)` does not define and `server.py:1483-1487` drops.

---

## 3. Duplication clusters (merged across modules)

Each is **one** finding with all sites, not one per module.

### 3.1 M3 — Scalar/env coercion (F15)
**Root cause, and the reason ~270 re-implementations exist:** the application's de-facto canonical truthiness parser is `core/testing.py:30 is_truthy`, imported by **140 production files**, living in a module whose docstring opens *"Lightweight helpers for test-mode detection"*. Engineers reasonably do not import production flag semantics from `testing.py`, so they write their own. Measured named private definitions: `_env_bool` 15, `_env_int` 23, `_env_flag` 6, `_env_float` 4, `_coerce_bool` 24, `_as_bool` 19, `_to_bool` 8, `_parse_bool` 14 = **113**.

Consequences that clear the drop rule:
- **Three truthy sets among five sibling OCR backends.** Set A `{1,true,yes,y,on}`, Set B drops `y`, Set C `("1","true","yes")` with **no `.strip()`**. `DOTS_VLLM_USE_DATA_URL=true␣` — one trailing space, the ordinary result of a docker-compose `environment:` list — reads False and sends a server-local filesystem path to a remote vLLM. Set C guards eleven such decisions.
- **An SSRF escape hatch.** `TTS/adapters/audio_cpp_config.py:_as_bool (18-29)` falls through to `bool(value)`, so `allow_remote_base_url = n` evaluates **True** (runtime-verified) and the adapter may point at an arbitrary non-loopback host — the exact thing the flag prevents. ADR-026 governs.
- **Five vocabularies inside `LLM_Calls` alone**, under five names, including a tri-state parser.

`core/MCP_unified/environment.py:17` is a **deliberate** copy for the standalone-package boundary (stated in its docstring) with an identical truthy set → `justified-divergence`, left alone. **Destination:** `core/Utils/coercion.py`, sole responsibility scalar/env coercion, re-exported from `testing.py` for the 140 existing importers.

### 3.2 M2 — Base64 cursor/token codec (F22)
**23 production decode sites across 22 files** (I measured 24 idiom lines, minus one in `core/MCP_unified/tests/`), in **two notational spellings** — `-len(x) % 4` and `(4 - len(x) % 4) % 4` — so a grep-based fix will miss sites. Strictness has already diverged three ways with no owner: **5 strict** (`validate=True` + `altchars`), **2 canonicalization-checked** (`notes.py:878,881` re-encode and compare), **16 lax** (bare decode, which silently *discards* out-of-alphabet characters rather than raising). `api/v1/endpoints/` adds **six different error contracts** for a malformed cursor: 400 in three places, 413-or-400 in one, HTTP 500 in one (F18), and **silently ignore and restart from page 1** in two (`workflows.py:2192-2210, 2593-2602`).

**The destination must be two helpers, not one.** The sites split into opaque pagination cursors and HMAC-signed capability tokens (`AuthNZ/api_key_crypto.py:119`, `notes.py` signature segments, `Sync/v2/service.py:556`). A single flattened codec that grows a `verify=False` default is worse than the duplication. Best models to promote: `mcp_unified_endpoint.py:128-158` for the opaque form; **`Sync/v2/service.py:10488-10530` for the signed form** — it is the reference implementation in the repo, combining a pre-decode size bound, `validate=True`, explicit `altchars`, a post-decode bound, a version check, `hmac.compare_digest` and TTL/skew bounds. By contrast `api_key_crypto.py:_b64decode (118-120)`, same trust class, has neither `validate=True` nor any length bound.

### 3.3 M5 — Datetime and timestamp wire format (F23)
Measured: **239 `datetime.utcnow()`** sites in `app/` (tz-naive, deprecated on the Python 3.12 CI runs), **39** private `_utc_now`/`_now_iso` definitions, and a canonical `api/v1/utils/datetime_utils.py` with **4** importers.

**The canonical is unadoptable** (verified, [§7](#7-dropped-and-disproved) item 5): `coerce_datetime (21-47)` returns naive datetimes unchanged despite promising tz-aware, tries `strptime` before `fromisoformat` so awareness depends on input format, and returns `now()` for unparseable input. It also module-level-imports `api/v1/schemas/chat_dictionary_schemas.TimedEffects` — a "shared datetime util" coupled to one feature's schema, which is why it has one importer.

`AuthNZ` repeats the shape one layer down: `repos/datetime_utils.py:_strip_tzinfo (6-19)` exists with 3 importers and implements the **lossy** one of three semantics, against 11 copies and 40 inline `replace(tzinfo=None)` calls. Adopting it would push wrong semantics onto session-expiry and secret-rotation timestamps. `DB_Management` has **10 helpers producing 4 mutually unsortable renderings**, two sharing a name and differing on precision. The anchor case remains `services/workflows_webhook_dlq_service.py:45` (tz-naive) vs `services/meetings_webhook_dlq_service.py:40` (tz-aware, second-truncated).

**Sequencing is therefore: fix or replace the canonical first, then migrate.** Not the reverse.

### 3.4 M6 — Retry and backoff (F30)
The prompt asked whether `RAG/rag_service/resilience.py:577` should be promoted as canonical. **Verified answer: no.** Ranking the three adopted implementations by reading them:
- `core/http_client.py:_decorrelated_jitter_sleep (2311-2315)` — `min(cap, uniform(base, prev*3))`, genuine **decorrelated jitter**, paired with delta-seconds *and* HTTP-date `Retry-After` parsing (`:2318-2337`) and a classifier treating DNS failures as permanent (`:2340-2357`). **Best.**
- `resilience.py:281-289` — symmetric ±25% jitter, which keeps a retrying fleet clustered; no `Retry-After`, no classifier. Middling.
- `DB_Management/transaction_utils.py:68` — `0.1 * (2 ** retry_count)`, **zero jitter**. Worst.

Plus 28 inline loops in one file with three internal inconsistencies (one missing `.lower()`, one `attempt < 4` inside `range(5)`, five of 28 missing the jitter term) and **zero retry at all on the PostgreSQL class** (F21).

**Destination:** `core/Utils/backoff.py`, seeded by **moving** those three functions *out of* `http_client.py`. This satisfies the constraint against growing the 6,600-LOC junk drawer by actively shrinking it.

### 3.5 M7 — Truncation, token budget, tiktoken (F31)
`core/LLM_Calls/tokenizer_resolver.py:resolve_tiktoken_encoding (938-942)` is `@lru_cache`'d, importable and side-effect-free — and **raises `TokenizerUnavailable` instead of falling back**, deliberately, to serve strict token counting. That is *why* eleven sites in five exception dialects rewrote it. So this is **true-duplication needing a lenient twin**, not an adoption campaign. Three of the eleven (`Workflows/adapters/text/nlp.py:479`, `RAG/rag_service/utils.py:26`, `Workflows/adapters/evaluation/eval.py:459`) catch only `KeyError`, but `encoding_for_model(None)` raises `AttributeError` — so a workflow step with `model: null` crashes where its siblings degrade.

Separately, `_truncate_content_by_tokens` is verbatim-duplicated (`RAG/rag_service/web_fallback.py:50` ≡ `Workflows/adapters/rag/search.py:162`), both binary-searching over *character* prefixes at O(n log n) — ~17 full tokenizations of a 100 KB page — while `RAG/rag_service/utils.py:TokenCounter.truncate (41-48)` does it in one pass. And `api/v1/endpoints/character_chat_sessions.py:5651` re-truncates content the core already truncated, then reports **`truncated: false`** on visibly-cut content.

### 3.6 M8 — SSE frame contract (F32)
**ADR-025** makes single-terminal-`[DONE]` a binding provider contract, so this is contract knowledge, not style. `core/Chat/streaming_utils.py` + `chat_service.py` carry **33 inline** `f"data: {json.dumps(...)}\n\n"` constructions despite `core/LLM_Calls/sse.py` being imported by 19 non-test modules elsewhere; `core/Chat` imports it **zero** times. Nine provider adapters inline the same SSE loop while the shared `streaming.py:iter_sse_lines_requests (73-117)` has exactly **one** production user. Three implementations of `is_done_line`, one of them case-sensitive; and `_SSE_CONTROL_PREFIXES` exists twice **under one name with different values**. The module's most policy-correct helper, `aiter_normalized_sse (280-314)` — the only one enforcing egress policy and retries — has **zero callers**.

### 3.7 M10 — `core/` → `api/` layering (F34)
Measured today: **157 import lines across 106 files** (the prompt said 161/107). Correctly scoped, that decomposes as:
- **23 lines inside `core/MCP_unified/tests/`** — an in-app test tree, not production code.
- **98 schema-only imports** — the mild class; the honest reading is that the *schemas are in the wrong package*.
- **33 production true inversions across 20 distinct files** — `core/` importing `api/v1/endpoints/*` or `api/v1/API_Deps/*`.

The sharpest instance is `Ingestion_Media_Processing/persistence.py`, where **four of five imports exist only so tests can monkeypatch `endpoints.media.*`** — production resolves its file validator, temp-dir manager and template classifier via `getattr` on an API module at request time, and a stray attribute on `endpoints.media` silently reconfigures ingestion. The docstrings say so verbatim (`:1884-1886`, `:2693-2695`).

~~A cheap sub-case: three `core/Chat` files import the **constant** `DEFAULT_CHARACTER_NAME` from `api/v1/API_Deps`; moving one constant clears 3 of the 20 files.~~

**Corrected 2026-09-22 — this "cheap sub-case" is wrong and acting on it would be a defect.** There are *two* `DEFAULT_CHARACTER_NAME` constants with **different values**: `core/Character_Chat/modules/character_utils.py:16` is `"Character"`, while `api/v1/API_Deps/ChaCha_Notes_DB_Deps.py:350` is `"Helpful AI Assistant"`. The three `core/Chat` files import the API_Deps one, and `chat_history.py:112` passes it to `get_character_card_by_name(...)` — a database lookup on that exact string. Repointing them at the core constant would silently fetch the wrong character card. Consolidating the two constants is owner-only work (it must edit `app/api/v1/**`) and is a behaviour decision, not a mechanical move. The same-name/different-value pair across a layer boundary is itself worth a finding.

**And the existing guard protects only one direction of a real cycle:** `tests/lint/test_endpoint_auth_deps_import_boundary.py:14-15` bans endpoints from importing `core.AuthNZ.User_DB_Handling`, while `core/AuthNZ/User_DB_Handling.py:20` imports `oauth2_scheme` **from** `api/v1/API_Deps/v1_endpoint_deps`.

**Recommendation — proportionate, not a mass refactor:** a sibling AST ratchet copying that file's shape, seeded at the current 106 files so the number can only decrease, with a **hard ban** on the 20 production true-inversion files.

---

## 4. Test estate and CI (F7, F8, F33)

This is the systemic finding, and it is the mechanism by which the rest survive.

**F7 — `app/core/MCP_unified/tests/` is in no CI shard.** Verified: `grep -c "app/core/MCP_unified/tests" .github/workflows/ci.yml` → **0**. The `platform-mcp-core` shard (`ci.yml:1758-1760`) runs only `tests/MCP` and `tests/MCP_unified`. The in-app tree is **149 files, 63.6% of the module's lines**, and *is* in `pyproject.toml` `testpaths` — so a bare `pytest` runs it and CI never does. Consequence, reproduced: `test_filesystem_glob_marks_file_size_unavailable` fails at `filesystem_module.py:1778` (`OSError: metadata unavailable` from an unguarded `is_symlink()`, 15 lines above the correctly-guarded `stat(follow_symlinks=False)`). It was added **2026-06-03**, and `filesystem_module.py` has had **28 commits since** — every one merged with this test red.

**F8 — `tests/Sync` aborts at collection, and hides 12 red tests.** Verified: `Interrupted: 2 errors during collection`, `2958 tests collected` and **none executed**, caused by two unguarded `from psycopg import sql` imports at module scope. So the directory does not partially skip — it does not run.

**Revised 2026-09-22 after a full run completed (3h 01m).** The collection abort was concealing **12 failing tests**, not one. Five names were recovered and *all five reproduce in isolation*, so they are deterministic rather than ordering artifacts:

- `test_sync_v2_store.py::test_postgres_personal_context_receipt_locks_binding_before_upsert` — the `link_state` fixture drift, red since 2026-09-03.
- `test_sync_v2_personal_context_exchange_gate.py` × 3 — `test_mixed_selected_conflicts_with_exact_proof_resolve_in_request_order` and both parametrisations of `test_mixed_exact_proof_preserves_native_notes_resolution_actions`. All assert `['mixed-exact-note'] == ['mixed-exact-note', 'mixed-exact-personal']`: a **mixed** notes/personal-context batch returns only the notes item. Reproduced in 7s (3 failed, 95 passed) — and note every failure is a `mixed_*` case while the pure personal-context cases pass, which is the shape of a real defect rather than fixture rot. Root cause now visible (see below) but **not yet classified as defect vs fixture**.
- `test_sync_v2_server_origin_capture.py::test_workspace_chat_api_write_stays_direct_when_sync_active` — `assert 404 == 201`.

The other seven names were lost to the output buffer and were not recovered.

**The runtime is itself the finding: 3h 01m.** That is why nobody runs the directory, and it means the obvious remedy — "add `tests/Sync` to a CI shard" — is wrong as stated. A 3-hour suite cannot sit in a PR gate; it needs a scoped gate-able subset or a nightly.

**A diagnosability defect blocks triage of three of them.** `core/Sync/v2/service.py:resolve_conflicts_batch` caught `except Exception` and appended to `rejected` without recording the cause, so a real regression, a stale fixture and a `KeyError` on dataset metadata were indistinguishable — the endpoint logged only "Sync v2 conflict resolution item failed". That swallow now logs the exception (per-item outcome contract unchanged), which immediately surfaced the real cause: `SyncStoreError: Personal Context conflict candidate is unavailable`, raised at `core/Sync/v2/personal_context_conflicts.py:203` when the source/remote envelope lookup or an identity check fails. **A connection-threading cause was hypothesised (per F-sync-5, where 62 of 125 store forwarders omit `connection=self._connection`) and disproved** — both `get_envelope_by_server_cursor` and `get_envelope_by_client_id` do thread it.

Compounding: neither blocking gate covers the directory (`backend-required.yml:193-195` runs only `tests/unit`; `coverage-required.yml:154-157` runs `tests/unit` + `sanity_tests`). This is **not** the known `--cov-fail-under=12` complaint; it is that the largest behavioural suite protecting a 43,905-LOC module is outside every contractual gate — and 12 tests went red inside it unobserved.

**F33 — no written rule for where a test goes.** The `_NEW` suffix carries no consistent meaning, and the per-module verdicts genuinely differ — this is reported per module rather than generalised:

| Tree pair | Verdict |
|---|---|
| `RAG` 47 / `RAG_NEW` 143 | **Real problem.** The conftests protect different things and pytest scopes each to its own directory. The only sqlite/postgres-parametrized fixture is in `tests/RAG/conftest.py:141` (2 consumers), so the 143-file tree is effectively **SQLite-only**. |
| `Chat` 100 / `Chat_NEW` 58 | **Not duplication.** One shared basename, and that pair is unit-vs-integration with zero overlapping test names; both sharded in CI with a contract test proving each file is covered once. `tests/Chat` is **2× more active** — yet CI labels it "chat-legacy". |
| `TTS` 56 / `TTS_NEW` 93 | **Complementary suites nobody wrote down** (security/sanitization vs public-contract). Coverage *partitions* rather than overlaps. Worse, two **production shim classes exist solely to satisfy `TTS_NEW`** and are registered as the real adapters. |
| `AuthNZ` ×5 (393 files) | `ci.yml` slices them **alphabetically** (`test_[a-l]*` / `test_[m-z]*`) — the axis is shard wall-clock, not domain. `AuthNZ_Postgres` (21 files) imports its fixtures from the parent; the actual Postgres tree is `tests/AuthNZ/integration` (76 files). |
| `Ingestion` 192 files / 21 trees | No discoverable axis. The declared `media_processing` marker has **one** use and `pipeline` has **zero**. |

**The dual-backend coverage asymmetry is now quantified.** The sanctioned Postgres fixture `tests/AuthNZ/conftest.py:isolated_test_environment (631-741)` — which `CLAUDE.md:245-248` designates as the only correct path — is requested by **45 files in `tests/AuthNZ/integration` and zero files in `tests/AuthNZ_Postgres`**. Of 31 test files reaching the five most-branched AuthNZ repos: 6 real-Postgres, 6 **stub**-Postgres (a pool object with `.pool` set, which structurally cannot catch a column-list asymmetry), 19 SQLite-only. `generated_files_repo.py` (35 branches) and `billing_repo.py` (19) have **zero** real-PG coverage. This is the concrete, fixable mechanism behind the 2026-09-21 cross-user isolation finding that leaks fail to converge.

---

## 5. Dead helpers

Deletion is the cheapest win available and is listed separately from consolidation. All verified by symbol grep across `app/` and `tests/`.

| Symbol | Status |
|---|---|
| `core/Utils/Utils.py:truncate_content (383)` | **Zero references.** Note it is a *character*-count truncator, so it is **not** the canonical for the token-budget pair — two separate findings, do not conflate. |
| `core/Utils/Utils.py:generate_unique_identifier (601)` | **Zero references.** (The prompt cited `generate_unique_id`; that symbol does not exist — the two hits are an unrelated e2e data generator.) |
| `core/Utils/Utils.py:is_valid_url (625)` | **Zero importers**, while `Web_Scraping/Article_Extractor_Lib.py:1243` defines a nested `url.startswith("http") and len(url) > 0`. That copy is effectively a **no-op** (`len(url) > 0` is unreachable-false given `startswith`), and the real gate is the adjacent `child_url.startswith(base_url)` at `:1297`. Low severity, **not** an SSRF finding — see [§7](#7-dropped-and-disproved). |
| `core/Utils/Utils.py:save_temp_file (826)` | **Test-only** — sole consumer is its own test at `tests/Utils/test_utils_general.py:147`. A distinct category: the test keeps it green while nothing ships it. |
| `LLM_Calls/streaming.py:aiter_normalized_sse (280-314)` | **Zero callers**, and it is the *only* helper pairing SSE normalization with egress policy and retries. |
| `LLM_Calls/streaming.py:aiter_sse_lines_httpx (120-156)` | Test-only. |
| `RAG/rag_service/batch_utils.py:run_batch_indexed (190-277)` | **Zero production callers**, and already drifted — it omits the `fail_fast` abort log its twin has. |
| `PromptStudioDatabase.py:list_optimization_iterations (2420-2470)` | **51 unreachable lines** — redefined at `:2472`, so Python binds the second. Both carry `# noqa: F811`, a rule on **neither** the global ignore list nor the per-file block. |
| `TTS/tts_validation.py:ProviderLimits.get_max_text_length (219)` | Zero callers; its `max_text_length` table disagrees with the live one by up to 6×. |
| `TTS/tts_config.py:ProviderConfig.max_retries (66)` | **Documented to operators** at `TTS-DEPLOYMENT.md:165` and **read by nothing**. |
| `TTS/adapters/base.py:convert_audio_format` `source_format` param | In the signature and docstring, never in the body; 11 callers pass it. |
| `Sync/v2/service.py:resolve_conflict (6925-6935)` + 2 params | Unreachable **security check** — dead validation reads as coverage. |
| `core/Sync/Sync_Client.py` (1,112 LOC) | Zero production importers, 16 commits/12mo of pure maintenance tax. Deleting is a product decision (still named in two design docs), not a cleanup. |
| `core/Chat/REFACTORING_PLAN.md` | Points contributors at the wrong module and a deleted test file; states "Current Status (May 2025)" against a 2026 codebase. |

---

## 6. Prior-findings reconciliation

**`db-management/` (2026-04-07, 8 findings): 8/8 already addressed** — including #4, the pool-lifecycle item the 2026-04-15 rebaseline itself left live (`content_backend.py:180-201`; `test_content_backend_cache.py` → **22 passed**). The April `## Test Gaps` list is therefore stale and should not be carried forward. **Process note:** the April pass organised stages by subsystem rather than size × churn, so `ChaChaNotes_DB.py` (45,292 LOC / 408 commits) and `PromptStudioDatabase.py` appear **zero times** in its stages 2–5 — and all five High findings in this round sit in those two files.

**`rag/` (2026-04-07, 6 stages): 7 still live, 2 partially addressed, 1 disproved.** Materially worse: `unified_rag_pipeline()` is now a **single 7,026-line function with 49 nested defs** (`unified_pipeline.py:2051-9077`) — the function body alone exceeds the whole file's size when the prior review called it a god module. Notably, the contract-chain abstraction that review asked for **did ship** (`request_resolution.py`, `retrieval_plan.py`, `evidence_models.py`, `retrieval_executor.py`) but was added *additively*, so the file gained 2,609 lines while gaining the seam meant to shrink it. One prior suspicion is **disproved and recorded** so it is not re-investigated: scope filters vs the semantic cache is not a cache-key omission — `unified_pipeline.py:3060-3065` disables caching whenever an explicit include-list is present.

**`evals-module/`: 6 addressed, 1 partial, 5 still live, 4 not re-verified.** One has *worsened*: #9's `pipeline_presets` read-side was fixed to filter by user, but the DDL is still `name TEXT PRIMARY KEY` and the write still upserts on `name` while reassigning `user_id` — so user B saving preset `default` now silently **makes A's preset vanish** rather than visibly sharing it.

---

## 7. Dropped and disproved

Stated rather than silently omitted, so nobody re-derives them.

**Seed facts from the review prompt that failed verification:**
1. **`api/v1/endpoints/files.py:83` "fails on Z-suffixed inputs" — FALSE.** `pyproject.toml:15` sets `requires-python = ">=3.11"`, and `fromisoformat` has accepted trailing `Z` since 3.11 (verified on the CI interpreter, 3.12.11). The real defect at that file is the fail-open expiry gate: a bare `except Exception: return None` makes an unparseable `export_expires_at` mean *never expires*.
2. **`workflows.py` `+ b"=="` padding is not a bug.** `binascii.a2b_base64` tolerates excess padding; verified identical output across all `len%4` classes. Idiom divergence only.
3. **`MCP_unified/environment.py:is_truthy` is not the ignored canonical** — the real one is `core/testing.py:30` with 140 importers. MCP's copy is documented `justified-divergence`.
4. **`LLM_Calls/error_utils.py:407` does exist** — the TTS reviewer reported it absent, having grepped the private spelling `_is_http_status_error`; the symbol is **public** `is_http_status_error`. Its copy handles both `httpx.HTTPStatusError` and `requests.HTTPError` where all three TTS copies are httpx-only, so the adoption-gap framing stands.
5. **`datetime_utils.py` is unadoptable, not merely unadopted** (§3.3). "60 sites bypass the canonical" would have recommended spreading three defects.
6. **`resilience.py` is not the best of the 8 backoff implementations** (§3.4).
7. **`tokenizer_resolver` is not an adoption gap** — it is strict by design (§3.5).

**Dropped under this review's own drop rules:**
- **Streaming SHA-256 chunk sizes.** Measured 3 distinct sizes / 9 sites, not "8× with 4 sizes". Any chunk size yields the same digest, so there is no correctness angle; chunk-size tuning is a micro-optimization excluded by the Axis-5 rule; and no future edit breaks because of the duplication. **Dropped.**
- **`_emit_*_counter` shims.** Measured 8, not 56, with no stated divergence and no nameable breaking edit. **Dropped.**
- **An SSRF theory on `is_valid_url`,** which I raised and then **disproved**: the adjacent `child_url.startswith(base_url)` is the real gate and `:225` documents egress enforcement via `http_client.fetch`. Downgraded to a Low dead-canonical finding.
- **`*_to_dict` (~91 hits)** remains `justified-divergence` as the prompt ruled — *except* for a verified carve-out: AuthNZ's 17 `_row_to_dict` adapters share the byte-identical schema-agnostic signature `(row: Any) -> dict[str, Any]` and contain **zero** column names, unlike the ruling's exemplar `Sync_DB.py:_device_from_row (1578)` which names 13 columns with per-field coercion. They have already split two ways on null handling. Reported as F36.
- **Axis 3 (sequential coupling)** produced nothing clearing its drop rule in the endpoints layer, and is reported **empty there rather than padded**.

---

## 8. `core/Utils` Migration Plan

Sequenced cheapest-well-covered-first. Every destination is a **cohesive module with one stated responsibility**. Nothing here grows `Utils/Utils.py` or `http_client.py`.

### Stage 0 — Free wins, no design doc, no behaviour change
1. **Delete the dead helpers** ([§5](#5-dead-helpers)). Pure subtraction.
2. **Fix F1** — port `list_optimizations` (~60 lines). *Owner-only.*
3. **Fix F4** — remove `asyncio.CancelledError` and `HTTPException` from the three `*_NONCRITICAL_EXCEPTIONS` tuples.
4. **Fix F9, F10, F11, F18** — each is a few lines with a correct sibling to copy.
5. **Add one CI shard under a non-UTC `TZ`.** This alone would have caught F6, a binding-ADR violation.
6. **Add `pytest.importorskip("psycopg")`** to the two files that abort `tests/Sync` collection (F8).
7. **Add `app/core/MCP_unified/tests` to the `platform-mcp-core` shard** (F7), after fixing the red test.

### Stage 1 — Promote-one-and-delete (no new modules)
8. **F12** — promote `Local_LLM/http_utils.py:54-79`; delete the two broken copies; **also** add `NetworkError` to `_CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS`, without which the Chat path stays broken. Extends `task-13287`.
9. **F27, F13** — one Likert normalizer; exclude alias keys from the overall score.
10. **F38** — collapse `media_metadata.py` ≡ `source_cache.py` into one parameterised materializer. The diff proves there is no behaviour to preserve.
11. **F36** — `core/AuthNZ/repos/row_mapping.py` (guard inside, return `{}`), absorbing the five JSON-blob coercers that live in the same files.

### Stage 2 — New cohesive modules
| Destination | Single responsibility | Finding |
|---|---|---|
| `core/Utils/coercion.py` | Normalize scalar and environment values to bool/int/float under **one** documented vocabulary. Re-export from `core/testing.py`. | F15 |
| `core/Utils/backoff.py` | Compute the next retry delay and classify retriability. **Seeded by moving** `_decorrelated_jitter_sleep`, `_parse_retry_after_delay_seconds`, `_should_retry` *out of* `http_client.py`. | F30, F21 |
| `core/Utils/http_status_extraction.py` | Derive an HTTP status from a transport exception of unknown provenance. | F12 |
| `api/v1/utils/iso_datetime.py` | Parse and emit ISO-8601 instants at the API boundary. **Precondition:** split `parse_timed_effects` out of `datetime_utils.py` and fix `coerce_datetime`. | F23 |
| `core/Utils/pagination_cursor.py` **and** a separate signed-token codec | Opaque cursors and HMAC-signed tokens — **two modules, deliberately.** Promote `mcp_unified_endpoint.py:128-158` and `Sync/v2/service.py:10488-10530` respectively. | F22 |
| `core/Evaluations/scoring.py` | Parse a judge's raw score and convert it on a declared scale to 0–1. | F27 |
| `core/MCP_unified` base sanitizer | One sanitizer whose character predicate and preserved-whitespace set are **parameters**. | F3, F5 |

### Stage 3 — Design-first (require `Docs/Design/YYYY-MM-DD-<slug>-design.md`, an ADR, a Backlog task and a staged `IMPLEMENTATION_PLAN_<slug>.md`)
- **F20** — decompose `PromptStudioDatabase.py` into a `prompt_studio_db/` package, following the **already-shipped** `core/DB_Management/media_db/` split of `Media_DB_v2.py`. Same team, same layer, already merged — point at it rather than inventing a shape.
- **`ChaChaNotes_DB.py`** — 45,292 LOC, 408 commits/12mo, one 44,076-line class with **273 methods attached by `setattr` at import time** (invisible to mypy, ruff and every IDE). The `chacha/` extraction is already underway (21 modules, 31,053 LOC); continue it at the **schema/migration layer**, which is what remains.
- **F25** — an `_chatops/` package for the transport-agnostic Discord/Slack shell, injecting the two genuinely protocol-specific pieces (signature algorithm, command parser). Stage 1 is the **91.2%-identical `*_oauth_admin.py` pair** — smallest, highest identity, tested on both sides. *Owner-only.*
- **F16** — route RAG's main path through the existing `retrieve_with_fusion (4949)`; rank-based fusion needs no calibration.
- **F34** — the AST import ratchet, seeded at 106 files with a hard ban on the 20.
- **`unified_pipeline.py`** — note the prior round's lesson: the seam was added without removing the body, and the file grew. Any plan must include deletion, not just extraction.

### Constraints carried into every recommendation
- **Owner-only** (per `CONTRIBUTING.md:13,14,18`): anything touching `tldw_Server_API/app/api/v1/**`. That includes F1's confirming endpoint, F18, F19, F25, and 3 of the 11 tiktoken sites. Flagged per finding.
- **Base branch:** `CONTRIBUTING.md:86,121` says PRs target `dev`; `origin/HEAD` resolves to `main`; both exist. **I assumed `dev`.**
- **Backlog is the ledger of record** (3,112 tasks). Tasks are *proposed* here, not created — this review was read-only. Note `task-13287` and `task-13288` are currently **untracked** in git.
- **Bandit** runs on touched scope (ADR-005); F15's SSRF-adjacent coercion fix and F22's codec work must clear HIGH/CRITICAL.
- **No new CI gate is proposed.** Every CI recommendation fits an existing contractual gate (`backend-required`, `coverage-required`) or an existing shard in `ci.yml`.

---

## 9. What this review did not cover

- **~200 of 269 endpoint files were never opened**, only grepped. Notably unexamined: `embeddings_v5_production_enhanced.py` (6,792 LOC / 127 commits), `audio/audio_streaming.py` (4,642), `mcp_hub_management.py` (4,462), `auth.py` (3,970 / 91), `agent_client_protocol.py`, `setup.py`, `sandbox.py`, all of `evaluations/`, most of `media/`. **`watchlists.py` (8,739 LOC / 103 commits) was swept, not read** — the largest genuinely unreviewed surface.
- **`ChaChaNotes_DB.py`'s ~30,000 lines of business logic** were not line-read; the file was reviewed structurally (census, adapter block, both migration ladders, v67–v70 migrations, delegation tail).
- **PostgreSQL was never executed.** `psycopg` is absent from this environment. Every dual-backend claim is static — AST, grep and SQL reading — and each such finding carries an explicit `probable-risk` split on the half needing a live cluster. Given that the SQLite/Postgres asymmetry is itself a headline finding, this is the most important gap in the review.
- **No coverage was measured.** Every `tests:` field across all 11 ledgers is **import-grep reachability, not coverage**, and is labelled as such. The full suite was not run; only targeted files were executed, with output recorded in the stage files.
- **Nothing was profiled.** Efficiency findings name a cost driver and its scaling factor; wall-clock magnitudes are marked assumptions. The two exceptions are measured and stated as such: the quadratic `bytes +=` (1,032× at 8,000 chunks) and `ssl.create_default_context` (7.29 ms/call).
- **Module-level gaps** each reviewer named: AuthNZ's BYOK/provider-credential subsystem (~5,500 LOC), Evaluations' `user_rate_limiter.py` (1,353 LOC / 32 commits), Sync's `restore_preview` and `_attachment_lifecycle_diagnostics` (646 and 549 lines), TTS's `voice_manager.py` (1,624 LOC / 31 commits, zero `tests/TTS` importers), Ingestion's `Upload_Sink.py` security surface, MCP's `apps/mcp-unified/` gateway.
- **Out of scope by design:** the frontends, `Helper_Scripts/`, `Dockerfiles/`, and every `app/core` module not in the eleven — cited as evidence where relevant, never audited.

## 10. Per-module ledgers

| Module | Ledger |
|---|---|
| DB_Management | [`db-management/`](./db-management/) — extends 2026-04-07 |
| MCP_unified | [`mcp-unified/`](./mcp-unified/) |
| AuthNZ | [`authnz/`](./authnz/) |
| RAG | [`rag/`](./rag/) — extends 2026-04-07 |
| Ingestion_Media_Processing | [`ingestion-media-processing/`](./ingestion-media-processing/) |
| Sync | [`sync/`](./sync/) |
| TTS | [`tts/`](./tts/) |
| Evaluations | [`evaluations/`](./evaluations/) — cross-links `evals-module/` |
| Chat | [`chat/`](./chat/) |
| LLM_Calls | [`llm-calls/`](./llm-calls/) |
| api/v1/endpoints | [`api-endpoints/`](./api-endpoints/) |
