# Stage 3 — Error policy and module boundaries

## Scope

Two cross-cutting questions that no single file answers:

1. **What does this module consider a swallowable error?** Twenty-six per-file
   `_*_NONCRITICAL_EXCEPTIONS` tuples answer it twenty-one different ways, across ~680 catch sites.
2. **Which way do this module's dependencies point?** `persistence.py` — the hottest file in the
   module (6,390 LOC, 98 commits/12mo) — imports from `api/v1/endpoints/` and `api/v1/API_Deps/`,
   and holds its own SQL and its own SQLite driver.

The standard stage-4 "data-source boundaries" stage is folded in here: this module has exactly one
data-source boundary (`core/DB_Management/media_db`) plus one file that bypasses it entirely.

## Code Paths Reviewed

Error policy:
- All 26 tuple definitions and their reduced forms — see
  `2026-09-21-stage3-exception-tuples.txt`. Key ones read in full:
  `persistence.py:_PERSISTENCE_NONCRITICAL_EXCEPTIONS (83-105)`,
  `Upload_Sink.py:_UPLOAD_SINK_NONCRITICAL_EXCEPTIONS (40-56)`,
  `Audio/Audio_Transcription_Lib.py:_AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS (92-114)`,
  `Audio/Audio_Streaming_Unified.py:_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS (79-…, extended at 1139)`,
  `PDF/PDF_Processing_Lib.py:_PDF_NONCRITICAL_EXCEPTIONS (40-56)`.
- `Upload_Sink.py:FileValidator.sanitize_xml_content (1165-1186)`.
- `XML_Ingestion_Lib.py:_ensure_defusedxml (68-78)`, `Plaintext/Plaintext_Files.py:_ensure_defusedxml
  (62-67)`, `Books/Book_Processing_Lib.py:47` — the three fail-closed XML policies.
- `persistence.py:process_document_like_item (4852-5590)` — the `try` at 4912 and its handler at 5079.

Boundaries:
- `persistence.py:_resolve_ingestion_file_validator (1879-1909)`,
  `:schedule_media_add_embeddings (2399-2569)` incl. the `generate_embeddings_task` closure
  (2498-2555), `:add_media_orchestrate (2675-3580)` incl. the helper-resolution block (2693-2738),
  `:process_document_like_item (4852-5590)` incl. the import at 4880.
- `document_upload_preflight.py:8-14`.
- `persistence.py:_source_hash_precheck (4242-4304)`, `:_url_precheck (4328-4343)`,
  `:_build_url_match_clause (1865-1878)`.
- `document_upload_drafts.py:DocumentUploadDraftStore (49-194)` and
  `:get_document_upload_draft_store (197-200)`.
- Contrast case read as the model: `Media_Update_lib.py:process_media_update (17-86)` — every DB
  touch routed through `media_db.api`. No raw SQL anywhere in it.

Clean by inspection (negative results, explicitly): `pipeline.py`, `result_normalization.py`,
`audio_batch.py`, `video_batch.py`, `document_upload_preflight.py`, `Upload_Sink.py` contain no raw
SQL and no `execute(`. All four raw `SELECT`s in `persistence.py` carry `LIMIT 1`; no unbounded
`SELECT` and no unbounded `fetchall()` exists in any of the nine orchestration files.

## Tests Reviewed

- `tests/Media_Ingestion_Modification/` (30 files) and `tests/MediaIngestion_NEW/` (61) carry the
  bulk of `persistence.py` coverage; `tests/Collections/` and `tests/DB_Management/` add the
  dual-write and media-DB paths. Import-grep reachability only — not measured coverage.
- **Nothing tests the exception-tuple policy.** No test asserts that a given exception class is or is
  not swallowed, so all 26 tuples can be edited freely with a green suite. The two defects in
  finding -8 both survive the full suite today.
- **Nothing tests cancellation.** No test in the 192 cancels an in-flight `/media/add` task and
  asserts it unwinds.
- `persistence.py:2703` carries `# pragma: no cover` on its fallback branch — the file itself
  records that the branch is never executed. Finding -8 explains why it *cannot* be.
- `document_upload_drafts.py` is covered by `tests/MediaIngestion_NEW/` draft tests, which is why
  finding -10's second half is a maintainability finding rather than a correctness one.

## Validation Commands

```
$ grep -rn "NONCRITICAL_EXCEPTIONS\s*=" tldw_Server_API/app/core/Ingestion_Media_Processing \
    --include="*.py" | wc -l
      27          # 26 distinct names; Audio_Streaming_Unified rebinds its own at :1139
$ grep -rn "NONCRITICAL_EXCEPTIONS = (" tldw_Server_API/app | wc -l
     287          # repo-wide instances of the same idiom
$ grep -c "except _PERSISTENCE_NONCRITICAL_EXCEPTIONS" \
    tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
     117
$ grep -c "raise HTTPException" .../persistence.py
      11
$ grep -c "except HTTPException:" .../persistence.py
       4

# AST: try-blocks containing an import, guarded by a tuple lacking ImportError
      22 sites across Audio/Audio_Files.py (4), Upload_Sink.py (4), persistence.py (14)
      -- full list in 2026-09-21-stage3-exception-tuples.txt

# AST: try-blocks whose body contains an `await`, guarded by a tuple containing
#      asyncio.CancelledError (a BaseException, never caught by `except Exception`)
  persistence.py:              30 of 117 guarded try-blocks contain an await
  Audio/Audio_Streaming_Unified.py: 18 of  96 guarded try-blocks contain an await
  Audio/Audio_Transcription_Lib.py:  0 of  56   (file has no `async def` at all)

# AST: HTTPException raised inside a try whose tuple handler catches HTTPException,
#      with no earlier `except HTTPException: raise` guard
       1 site — persistence.py try@4912 / handler@5079 swallows the raise at 4994

$ grep -rn "from tldw_Server_API.app.api" .../Ingestion_Media_Processing
       6 hits, 2 files (persistence.py ×5, document_upload_preflight.py ×1)
$ grep -n "sqlite3" .../document_upload_drafts.py | head -3
7:import sqlite3
79:        connection = sqlite3.connect(str(self.db_path), timeout=10)
80:        connection.row_factory = sqlite3.Row
```

## Findings

### FINDING ingestion-media-processing-8 — the per-file "noncritical exception" allowlists widened three catches instead of narrowing them, and killed 22 optional-dependency handlers

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       26 tuple definitions in this module, 21 distinct as-written sets, no two files agreeing.
             Full table with reduced forms in 2026-09-21-stage3-exception-tuples.txt. Definitions:
               persistence.py:83, Upload_Sink.py:40 (+53), chunking_options.py:60,
               download_utils.py:34, Video/Video_DL_Ingestion_Lib.py:154, PDF/PDF_Processing_Lib.py:40,
               Books/Book_Processing_Lib.py:80, Email/Email_Processing_Lib.py:37,
               Plaintext/Plaintext_Files.py:49, VLM/registry.py:5,
               OCR/backends/{chatllm_ocr.py:30, deepseek_ocr.py:31, dolphin_ocr.py:26,
                 hunyuan_ocr.py:25, llamacpp_ocr.py:38, nemotron_parse.py:35},
               Audio/{Audio_Files.py:122, Audio_Streaming_Unified.py:79 and :1139,
                 Audio_Transcription_Lib.py:92, Audio_Transcription_Nemo.py:97,
                 Audio_Transcription_Qwen3ASR.py:56, Diarization_Lib.py:33,
                 stt_provider_adapter.py:174, Parakeet_Core_Streaming/transcriber.py:33,
                 Parakeet_Core_Streaming/ws_server.py:41}
             DEFECT A — BaseException widening. Three tuples include `asyncio.CancelledError`,
             which `except Exception` never caught:
               persistence.py:84            — 30 of its 117 guarded try-blocks contain an `await`,
                                              including all 12 `contextlib.suppress(...)` sites
                                              (872, 1005, 2118, 2236, 3971, 3979, 4172, 5087, 5090,
                                              5093, 5096, 5721)
               Audio/Audio_Streaming_Unified.py:80 — 18 of 96 guarded try-blocks contain an `await`
               Audio/Audio_Transcription_Lib.py:93 — 0 (the file has no `async def`); inert today,
                                              but it is the same mistake waiting for the first
                                              `async def` added to a 5,027-line file
             DEFECT B — dead optional-dependency handlers. 22 `try:` blocks containing an `import`
             are guarded by a tuple that does not include `ImportError`:
               Upload_Sink.py:1167 (import-only body; the `return xml_content` fallback at :1171)
               persistence.py:1887, :2699 (both import-only bodies; :2703 is even marked
                 `# pragma: no cover`), :4879 (import-only body)
               plus 18 mixed-body sites: Audio/Audio_Files.py:925, :986, :1076, :1293;
                 Upload_Sink.py:1098, :1100, :1135; persistence.py:2453, :2508, :3057, :3125,
                 :3282, :3332, :3849, :4157, :4981, :5913, :6163
             DEFECT C — self-swallowed HTTP status. persistence.py:_PERSISTENCE_NONCRITICAL_EXCEPTIONS
             includes `HTTPException` (:101) while the same file raises it 11 times. One of those
             raises is unguarded: try@4912 → `raise HTTPException(HTTP_413_TOO_LARGE, ...)` at :4994
             → caught by `except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as prep_err:` at :5079.
             (The other 10 are protected by an earlier `except HTTPException: raise`, at :2815 and
             three siblings — so the idiom is known and applied 4 times out of 5.)
canonical:   NONE. The repo has 287 of these tuples; there is no shared definition of "noncritical".
destination: `app/core/exceptions.py` already owns the domain exception hierarchy and their default
             status codes (:796-879). It should also own `TRANSIENT_IO_ERRORS`,
             `OPTIONAL_DEPENDENCY_ERRORS` and `MALFORMED_INPUT_ERRORS` as three named, documented
             tuples with ONE stated meaning each — replacing a per-file allowlist with a per-intent
             one. Not `Utils/Utils.py`.
knowledge:   "which exceptions are safe to swallow, and why." Twenty-one answers in one module.
             The reduced forms show most of the variation is noise: every tuple is built from the
             same core six — {AttributeError, OSError, RuntimeError, TypeError, ValueError,
             LookupError} — and the members that actually differ are precisely the dangerous ones
             (ImportError present/absent, CancelledError present/absent, HTTPException
             present/absent). The tuples look like careful policy and behave like `except Exception`
             with three random holes.
scenario:    (A) A client disconnects mid-`POST /api/v1/media/add` while `process_batch_media`
             (4088-4851) is in its pre-check loop. Starlette cancels the request task;
             `CancelledError` is raised at the next `await`; the `contextlib.suppress(
             _PERSISTENCE_NONCRITICAL_EXCEPTIONS)` at :4172 absorbs it. The coroutine does not
             unwind — it keeps iterating the remaining items and keeps writing to the media DB for
             a client that is gone. `Task.cancel()` never completes, so graceful shutdown waits on
             it. Before the tuples were introduced, `except Exception` let `CancelledError` through
             and this could not happen.
             (B) An operator installs without an optional extra (`pip install --no-deps`, a slimmed
             container image). `Upload_Sink.FileValidator.sanitize_xml_content` at :1167 tries
             `from defusedxml import ElementTree as DET`; the `except` at :1169 cannot catch
             `ModuleNotFoundError`, so instead of the intended "log a warning and pass the content
             through" the exception escapes the sanitizer entirely and surfaces as a 500 on upload.
             The three sibling XML entry points — `XML_Ingestion_Lib.py:_ensure_defusedxml (68-78)`,
             `Plaintext/Plaintext_Files.py:_ensure_defusedxml (62-67)`, and
             `Books/Book_Processing_Lib.py:47`'s hard import — all fail closed deliberately and
             correctly. Note this cuts *toward* safety: the fail-open branch that was written
             (`return xml_content`, :1171, returning unsanitised uploaded XML) is exactly the branch
             that cannot run. The bug is that the module has four XML dependency policies and the
             one that would have been a security hole is dead by accident, not by design.
             (C) A user over their storage quota uploads a PDF by URL. `persistence.py:4994` raises
             `413 Content Too Large` with the quota breakdown. It lands in the handler at :5079 —
             which even reads `getattr(prep_err, "detail", prep_err)` at :5080, showing the author
             knew HTTPExceptions arrive there — and is logged as a generic "File preparation/download
             error". The client receives 200/207 with a per-item error string; the quota does not
             stop the request, and the `except HTTPException: raise` at :5025 is dead code because
             the outer handler catches `HTTPException` anyway.
impact:      High. (A) breaks cooperative cancellation on the module's main write path, which is
             also the mechanism by which a disconnected client's work keeps consuming a DB
             connection. (C) turns a quota enforcement decision into a soft error. (B) is currently
             benign because `defusedxml>=0.7.0` is a hard dependency (pyproject.toml:55) — but the
             code at XML_Ingestion_Lib.py:10 and Plaintext_Files.py:23 both comment
             "defusedxml is an optional dependency", so the module's own belief contradicts the
             packaging, and a future move of defusedxml into an extra re-arms all four policies at
             once.
tests:       none. No test asserts swallow/no-swallow for any exception class; no test cancels an
             in-flight ingestion. This is why all three defects are live.
effort:      cheap for the three concrete fixes — drop `asyncio.CancelledError` from the three
             tuples (or add `except asyncio.CancelledError: raise` ahead of each handler, the idiom
             already used at :2665, :3431, :3441, :3449, :3531, :3630), drop `HTTPException` from
             persistence's tuple, add `ImportError` where an import is guarded. Expensive for the
             underlying consolidation across 287 repo-wide tuples — that needs its own design doc.
owner-only:  no
confidence:  confirmed — every count above is from an AST walk over the module, and the three
             defect sites were read individually.
```

### FINDING ingestion-media-processing-9 — core imports the API layer so the test suite can monkeypatch it, and the code says so

```
axis:        encapsulation
class:       n/a
severity:    Medium
sites:       TRUE INVERSIONS (core → api/v1/endpoints/* or api/v1/API_Deps/*):
               persistence.py:_resolve_ingestion_file_validator (1879-1909) — import at :1888-1890
                 of `api.v1.API_Deps.validations_deps.file_validator_instance`; docstring at
                 :1884-1886 states the reason: "Prefer the endpoint-exported validator patchpoint
                 (for tests that monkeypatch `endpoints.media.file_validator_instance`)".
               persistence.py:add_media_orchestrate — import at :2699-2701 of
                 `api.v1.endpoints.media as media_mod`, used ONLY as a monkeypatch lookup table at
                 :2719-2734 (`getattr(media_mod, "file_validator_instance" / "TemplateClassifier" /
                 "TempDirManager", <core default>)`). Docstring at :2693-2695: "selected helpers
                 from modular `media` exports so tests can monkeypatch `endpoints.media.*` patch
                 points."
               persistence.py:add_media_orchestrate — import at :2710-2712 of
                 `api.v1.API_Deps.validations_deps.file_validator_instance`. **Unguarded** — unlike
                 :1888 it has no `try`, so this one is a hard runtime dependency of `/media/add` on
                 the API package, not a best-effort lookup.
               persistence.py:schedule_media_add_embeddings — import at :2509-2512 of
                 `api.v1.endpoints.media_embeddings.{generate_embeddings_for_media, get_media_content}`.
                 This is the worst shape: a core service calling an endpoint handler as a library.
               persistence.py:process_document_like_item — import at :4880-4882 of
                 `api.v1.endpoints.media as _media_mod`, passed to
                 `_resolve_ingestion_file_validator` (i.e. patchpoint only).
             MILD (schema-only — the schema is in the wrong package, core is not wrong to need it):
               document_upload_preflight.py:8-14 — five names from
                 `api.v1.schemas.document_upload_processing`, unconditional top-level import, used
                 as the public in/out types of `preflight_document_upload_files (41-45)`.
canonical:   `Docs/Architecture.md` — "Clients → FastAPI endpoints → Core domain services →
             Databases / Vector stores / External providers"; "keep endpoints thin, push logic into
             core modules".
destination: n/a for the inversions — the fix is deletion of the import, not relocation. For the
             mild case, `api/v1/schemas/document_upload_processing.py` should move to
             `core/Ingestion_Media_Processing/` and be re-exported from the schemas package.
knowledge:   "who owns the file validator / temp-dir manager / embeddings entry point." Today the
             answer is "core, unless a test replaced the endpoint module's copy, in which case
             that one."
impact:      Medium, and the reason is specific rather than architectural taste. Four of the five
             inversions exist **only** so that `monkeypatch.setattr("...endpoints.media.TempDirManager", ...)`
             in a test changes production behaviour. That means (a) production `/media/add` resolves
             three of its collaborators through `getattr` on an API module at request time, so a
             stray attribute on `endpoints.media` silently reconfigures ingestion; (b) the tests
             that rely on it are pinned to the *location* of a symbol rather than to behaviour, so
             the seam cannot be removed without rewriting them; and (c) `:2710` makes the core
             module unimportable without the API package, which is the concrete thing an import
             ratchet would catch.
tests:       the coupling IS the tests — `tests/Media_Ingestion_Modification/` and
             `tests/MediaIngestion_NEW/` patch `endpoints.media.*`. Grep those trees for
             `endpoints.media` before changing anything here.
effort:      moderate. The correct fix is explicit dependency injection: give
             `add_media_orchestrate` optional `file_validator=`, `temp_dir_manager=`,
             `template_classifier=` parameters and have the tests pass them, then delete the
             `getattr` block at :2719-2734 and all four `endpoints`/`API_Deps` imports. The
             embeddings import (:2509) is separable: move the two functions it needs into
             `core/Embeddings/` — note ADR-022 fixes the *ownership* split (Jobs owns the durable
             root record) but says nothing about where the helpers live, so this does not
             contradict it.
owner-only:  yes, if the fix also moves `api/v1/schemas/document_upload_processing.py`; no for the
             core-side DI change alone.
confidence:  confirmed — all six imports read, and the two docstrings quoted verbatim.
```

### FINDING ingestion-media-processing-10 — two separate bypasses of `core/DB_Management/`: hand-built SQL in `persistence.py` and a complete second SQLite storage layer in `document_upload_drafts.py`

```
axis:        encapsulation
class:       adoption-gap
severity:    Medium
sites:       RAW SQL — four f-string-composed SELECTs, executed via `db.execute_query`:
               persistence.py:_source_hash_precheck (4242-4304) — statement 4247-4256
                 (`# nosec B608` at :4256), executed at :4257; statement 4263-4276
                 (`# nosec B608` at :4276), executed at :4278; statement 4284-4297
                 (`# nosec B608` at :4297), executed at :4299
               persistence.py:_url_precheck (4328-4343) — statement 4329-4337
                 (`# nosec B608` at :4337), executed at :4338
               persistence.py:_build_url_match_clause (1865-1878) — the fragment builder
             PARALLEL STORAGE LAYER — `core/DB_Management/` bypassed entirely:
               document_upload_drafts.py:DocumentUploadDraftStore (49-194)
                 own driver:      `import sqlite3` (:7); `sqlite3.connect(..., timeout=10)` (:79);
                                  `PRAGMA busy_timeout = 10000` (:81); `PRAGMA journal_mode = WAL` (:87)
                 own db file:     `Path(get_project_root()) / "Databases" /
                                  "document_upload_drafts.db"` (:74-75), `mkdir` at :69
                 own DDL:         CREATE TABLE (:88-98), two CREATE INDEX (:99-104)
                 own DML:         DELETE expired (:116-119); BEGIN IMMEDIATE (:131);
                                  two COUNT(*) quota checks (:133-139); INSERT (:142-155);
                                  SELECT by draft_id+owner+expiry (:168-175); DELETE (:190-193)
                 own singleton:   `@lru_cache(maxsize=1) get_document_upload_draft_store` (:197-200)
canonical:   `core/DB_Management/media_db/api.py` (`create_media_database`, `managed_media_database`,
             `get_media_repository`) and `core/DB_Management/sqlite_policy.py:47` for pragma setup.
             The in-module model is `Media_Update_lib.py:process_media_update (17-86)` — zero raw
             SQL, every touch through `media_db.api`. `persistence.py:_media_has_source_hash_column
             (1178-1188)` already does the right thing by going through `db.backend.get_table_info`.
destination: `core/DB_Management/media_db/` gains `MediaRepository.find_by_source_hash(...)` and
             `.find_by_url(...)`; the drafts table becomes a table in the per-user content DB or a
             repository under `core/DB_Management/`, not a fifth SQLite file.
knowledge:   the Media schema (`Media`, `DocumentVersions`, `is_trash`, `deleted`, `source_hash`,
             `safe_metadata`) and SQLite connection policy (timeout, busy_timeout, journal_mode),
             both of which `core/DB_Management/` exists to own.
impact:      Medium. **No injection risk** — the only interpolated fragment comes from
             `_build_url_match_clause`, which emits `f"{column} IN ({', '.join(['?'] * n)})"`, i.e.
             placeholders only, with `column` a hardcoded literal (`"url"` at :4233, `"m.url"` at
             :4237); all values are bound. The cost is the schema coupling: a `DocumentVersions`
             migration must now be applied in two packages, and the four `# nosec B608`
             suppressions show the author knew the queries were being flagged. The drafts store is
             the larger of the two — ~150 lines re-implementing connection management, pragma setup,
             migration, quota enforcement and TTL reaping, and it is **SQLite-only**, so the
             PostgreSQL backend path that `Architecture.md` documents does not exist for that table.
             The briefing's known-live consequence — isolation leaks that fail to converge because
             the SQLite/PostgreSQL split is only tested on SQLite — applies directly: this table has
             no Postgres implementation at all, and `owner` is its only tenancy boundary.
tests:       `tests/MediaIngestion_NEW/` covers the drafts store's behaviour (which is why this is
             maintainability, not correctness). Nothing asserts the raw-SQL prechecks against a
             Postgres backend.
effort:      moderate for the four SELECTs (mechanical; they already have a natural home next to
             `_media_has_source_hash_column`). Expensive for the drafts store — it is a storage
             migration and needs `Docs/Design/` + an ADR, because choosing its new home is a
             decision. Note the `media_db/` package split (`api.py`, `constants.py`, `errors.py`,
             `runtime/`, …) that replaced the old `Media_DB_v2.py` monolith is the template to
             follow; it is the same team and the same layer and it already shipped.
owner-only:  no
confidence:  confirmed — every statement and pragma read at the cited line.
```

## Suggested Refactor/Actions

1. **Three one-line safety fixes, no design doc needed** (finding -8): remove
   `asyncio.CancelledError` from `persistence.py:84`, `Audio_Streaming_Unified.py:80` and
   `Audio_Transcription_Lib.py:93`; remove `HTTPException` from `persistence.py:101` and add
   `except HTTPException: raise` ahead of the handler at `persistence.py:5079` (matching the idiom
   already at :2815); add `ImportError` to `_UPLOAD_SINK_NONCRITICAL_EXCEPTIONS` and
   `_PERSISTENCE_NONCRITICAL_EXCEPTIONS`. Each needs one regression test, and those tests are the
   first tests in the module to assert error *policy*.
2. **A `tests/lint/` ratchet, not a mass refactor** (finding -9), seeded at the repo's current
   107 files with core→api imports so the number can only go down — the same AST shape as the
   existing `tests/lint/test_endpoint_auth_deps_import_boundary.py`. Add `persistence.py:2710` to a
   short must-fix list, since it is the one unguarded hard dependency.
3. **DI for the four monkeypatch inversions** (finding -9). Sequence: add the optional constructor
   parameters → migrate the patching tests to pass them → delete the `getattr` block at
   `persistence.py:2719-2734` → delete the imports. Do not attempt this before step 2's ratchet
   exists, or the count will silently drift back.
4. **Exception-policy consolidation** (finding -8, the 287-tuple repo-wide half) needs
   `Docs/Design/2026-MM-DD-noncritical-exception-policy-design.md` and an ADR — it is a decision
   about what the codebase means by a recoverable error, and it interacts with the 1,050-file BLE001
   grandfather list in `pyproject.toml:756`. Propose three named tuples in `core/exceptions.py`
   (`TRANSIENT_IO_ERRORS`, `OPTIONAL_DEPENDENCY_ERRORS`, `MALFORMED_INPUT_ERRORS`) rather than one
   union, so the name states the intent.
5. **Raw SQL** (finding -10) is cheap and self-contained: two repository methods in
   `core/DB_Management/media_db/`. Do it before the drafts-store migration, which is a separate,
   larger task needing its own design note.
6. `persistence.py` at 6,390 LOC / 98 commits is the module's god file. Five seams are visible and
   two are near-zero-risk leaf extractions: the metrics block (`:150-266` + `:563-702`, ~270 LOC,
   no inbound coupling) and the transcript-reuse feature (`:1224-1295` + `:1342-1864`, ~620 LOC,
   its own constant at :113-121 and its own repo accessor). Four functions —
   `add_media_orchestrate (2675-3580)` at 906 lines, `persist_doc_item_and_children (5591-6370)` at
   780, `process_batch_media (4088-4851)` at 764, `process_document_like_item (4852-5590)` at 739 —
   are 50% of the file between them. This is not filed as a numbered finding because "the big file
   is big" fails the Axis 1 drop rule on its own; it is recorded here as the context for findings -8,
   -9 and -10, all three of which live in it.
