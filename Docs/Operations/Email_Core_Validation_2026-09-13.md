# Synthetic Email Core Validation — 2026-09-13

This first section preserves the initial audit at TASK-13250. Its reproduced
identity/cursor defects are addressed by the follow-up recorded below; the audit
counts and characterization describe the earlier revision, not current behavior.

Owner: Project owner / maintainer
Validation task: TASK-13250
Starting revision: `c70387f496` (local `dev`)
Scope: Audit and test existing behavior; update release gates. No production code,
server configuration or personal mailbox changes.

## Outcome

A synthetic, model-free path exists for EML, ZIP-of-EML and MBOX uploads through
`POST /api/v1/media/add`, with SQLite persistence, FTS/operator search and message
detail retrieval. Explicit processing options and intercepted calls establish this
for the tested path. This is not a statement that default uploads or arbitrary
model/chunking configurations are model-free.

**Core identity correctness is not complete.** Two different RFC Message-ID messages
with identical bodies collapse to one stored/searchable message. This reproduced
for separate EML files, a ZIP and an MBOX. TASK-13251 tracks the required fix. Gmail
access is unrelated to resolving this defect.

## Code Trace

| Step | Implementation and observed behavior |
| --- | --- |
| Process only | `app/api/v1/endpoints/media/process_emails.py` expands enabled containers and calls the email library; it returns results without persistence. |
| Save uploads | `app/api/v1/endpoints/media/add.py` delegates to `core/Ingestion_Media_Processing/persistence.py`, which validates/saves temporary files and chooses the email processor by extension and request toggles. |
| Parse | `core/Ingestion_Media_Processing/Email/Email_Processing_Lib.py` uses Python email/MIME parsing. Plain text is preferred; HTML-only bodies are converted to text. Headers, participants, date, Message-ID and attachment metadata are returned. |
| Containers | EML is direct. ZIP needs `accept_archives=true`; MBOX needs `accept_mbox=true`. Both expand to child messages persisted through the normalized graph path. PST/OST requires `accept_pst` and a parser backend; enabled-backend behavior was not validated here. |
| Attachments | Filename, MIME type, byte size, content ID and disposition are parsed. `ingest_attachments` handles nested EML children with a depth limit. Ordinary binary attachments are not extracted/indexed as text. No general attachment download API or original email retention was validated. The current `keep_original_file` storage branch only covers PDF/document/ebook. |
| Legacy persistence | `core/DB_Management/media_db/repositories/media_repository.py` resolves existing Media rows by URL and then content hash. The content-hash lookup can merge distinct email identities before email-native upsert. |
| Normalized persistence | `core/DB_Management/media_db/runtime/email_graph_persistence_ops.py` writes source/message/participant/label/attachment tables and email FTS when native persistence is enabled. It tries source/message identity, then falls back to `media_id`; a legacy merge can therefore overwrite the normalized message identity. Upload source keys derive from filenames/container names; cross-filename/source identity semantics are not certified. |
| Search and detail | `core/DB_Management/media_db/runtime/email_query_ops.py` runs SQL/SQLite FTS; `api/v1/endpoints/email.py` exposes search and detail without an LLM. Current HTTP pagination is offset/limit, despite the PRD cursor requirement. `POST /media/search` provides the compatibility/delegation surface. |
| Analysis | Email processing calls `Summarization_General_Lib.analyze` when analysis and provider are enabled. This can resolve configured credentials, including when no key is supplied in the request. `/media/add` defaults analysis to true. |
| Other model work | Claims extraction can inherit `ENABLE_INGESTION_CLAIMS`; auto chunking has an optional chat boundary assistant; chunking strategies may use models. Embeddings are independently gated by `generate_embeddings`, then dispatched through Jobs or background tasks. |
| Collections | Persistence dual-writes successful results to Collections. Its fallback summary is a text prefix, not an LLM summary. The harness uses temporary storage for this path as well. |

Paths in this table are beneath `tldw_Server_API/app/` except where already prefixed
with `app/`. See the test module for exact import targets.

## Tested Offline Request

```text
media_type=email
perform_analysis=false
perform_claims_extraction=false
perform_chunking=false
auto_chunking_use_llm=false
generate_embeddings=false
keep_original_file=false
```

Use file uploads only; omit URLs/providers. Add the matching archive acceptance
flag for ZIP/MBOX. Native persistence and operator search are enabled in the test;
Gmail is disabled. No server or connector worker is started.

`tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py`
uses real multipart form handling, file validation, parsing, temporary SQLite,
persistence, SQL/FTS search and detail. It substitutes auth/user/usage dependencies
and omits route-level quota/billing/auth infrastructure and application lifespan.
It therefore does not certify the full deployed auth or startup path.

The fixture forbids summarization, claims extraction, LLM boundary assistance,
embedding job creation, background-task scheduling, shared HTTP helpers and socket
connection/DNS calls. It records attempted calls and asserts the list is empty at
teardown, so swallowed guard exceptions still fail validation. The core/Gmail
regression runs additionally used a task-local socket guard during collection and
execution. These are focused Python-call guards, not a general OS sandbox or proof
about separately running workers and later processing of stored data.

## Verification

Environment: macOS, Python 3.11.13, pytest 8.4.1, project `.venv`, temporary SQLite.
Fresh results:

| Check | Result | Meaning |
| --- | --- | --- |
| Focused core slice below | 75 passed; 8 warnings; 0 outbound socket/DNS attempts | Includes six new tests and a passing characterization of the unresolved dedupe defect. |
| Mocked Gmail slice | 13 passed; 9 unrelated tests deselected; 4 warnings; 0 outbound socket/DNS attempts | Simulated provider/worker behavior only. |
| Ruff check and format | Passed | Touched test module only. |
| Bandit | No findings | Touched test module; B101 excluded because pytest assertions are intentional. |
| Metrics fixture checker | Passed | Fixture arithmetic/checker behavior only; no staging SLO claim. |

Activate the environment first:

```bash
source .venv/bin/activate
export TEST_MODE=true AUTO_DOWNLOAD_MODELS=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_search_endpoint.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_search_request_model.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_email_parser_unit.py \
  tldw_Server_API/tests/DB_Management/test_email_native_stage1.py \
  tldw_Server_API/tests/DB_Management/test_media_db_email_query_ops.py \
  tldw_Server_API/tests/DB_Management/test_media_db_email_graph_persistence_ops.py \
  tldw_Server_API/tests/DB_Management/test_media_db_email_retention_ops.py \
  tldw_Server_API/tests/Helper_Scripts/test_email_search_dual_read_parity.py -q
python -m pytest tldw_Server_API/tests/External_Sources/test_policy_and_connectors.py -k gmail -q
```

The committed new module contains six cases: EML/ZIP/MBOX import/search/detail and
repeat imports, HTML-only body conversion, a same-body collision characterization,
and processing-only non-persistence. The characterization asserts the observed
single-row collision; it must become a two-identity correctness regression when
TASK-13251 is fixed. A green characterization is not a passed identity requirement.

The other selected tests cover parser/nested-EML behavior, operator queries,
message detail, normalized graph operations, deleted visibility, retention,
legacy backfill, media-search delegation and parity-checker behavior. Several
endpoint tests use fake DBs; only the new import tests and relevant DB tests establish
real SQLite behavior. Mocked Gmail tests cover body extraction, provider pagination,
backfill/deltas, label/message state, retry budgets and cursor recovery.

Metrics checker reproduction (fixture inputs only):

```bash
python Helper_Scripts/checks/email_m2_gate_validation.py \
  --metrics-file Helper_Scripts/checks/fixtures/email_sync_metrics_before.prom \
  --metrics-file-after Helper_Scripts/checks/fixtures/email_sync_metrics_after.prom \
  --output-json /tmp/email_metrics_fixture_13250.json
```

It passed with fixture deltas of 40 successes and p50 lag 60 seconds. Its printed
“staging lag SLO validated” message does **not** establish real staging performance;
only its processing of these fixture values was validated.

## Remaining Gates and Gaps

- TASK-13251: identical-body distinct messages merge, with the later normalized
  identity replacing the first. Initial strict two-row acceptance probes failed
  for all three tested formats. This remains a core release blocker.
- Cursor pagination promised by FR-SEARCH-004 is not exposed by the current API.
  Existing offset tests do not fulfill that requirement.
- Binary attachment content extraction is outside v1's metadata-first scope;
  no attachment content retrieval, OCR or original email blob retention is claimed.
- Live PST/OST backend behavior, PostgreSQL migration/runtime parity, malformed
  archive completeness and cross-source/renamed-file dedupe were not certified.
- No new 1M-message benchmark, ingestion throughput measurement, production-like
  cutover rehearsal or target-deployment readiness check was performed. Historical
  M3 checkmarks and fixture tests are not refreshed production evidence.
- Gmail OAuth, live API behavior, sync completeness, quota behavior and real lag
  remain explicitly deferred. No personal Gmail/account/email was accessed. Mocked
  provider tests do not prove live provider behavior.
- The release checklist retains separate open core and optional Gmail owner gates.
  Only the owner decides actual rollout after the selected scope's evidence is met.

## Correctness Follow-up — TASK-13251, TASK-13253, TASK-13254

The owner authorized addressing the audit defects. The same-body characterization
has been replaced with strict correctness regressions. The follow-up changes:

- Preserve the parsed email object and attachment descriptors in safe metadata for
  primary and child messages. Distinct IDs with identical bodies receive separate
  Media rows using tenant/provider/source identity; provider ID precedes RFC ID,
  with body hash only when both are missing. Non-email hash dedupe excludes emails.
- Reuse compatible legacy rows when identity evidence is available; reject normalized
  media-ID fallback that would replace another tenant/source/message identity.
  Source names retain their existing filename/container-member convention. Renaming
  the source can produce a separate import. Lost historical content is not repaired.
- Populate email search from accepted persisted content on reimport, including when
  overwrite is disabled and the normalized graph is first being backfilled. Keep
  metadata in new versions on accepted overwrite; preserve richer normalized
  metadata for legacy versions whose allowlist dropped it. Mocked Gmail deliveries
  refresh labels independently. Highlight updates reuse the Media transaction
  connection to avoid SQLite lock contention and remain atomic on rollback.
- Normalize provider ISO timestamps and RFC dates to UTC. Reject unrepresentable
  UTC dates and relative-window overflow without internal server errors.
- Add opt-in cursor pagination ordered by date descending, nulls last, then ID
  descending. Omitted cursor retains the offset contract. Query/tenant/deleted
  scope is bound to the token; relative windows use the first page's clock. This
  remains live traversal, not a snapshot when messages change their sort date.
- Label metrics checker inputs as `offline_fixture` or `live_endpoint`. Fixture
  success explicitly leaves staging unverified.

Design and API contract: `Docs/Design/email-core-correctness-13251.md`,
`Docs/Design/email-search-cursor-pagination.md`, and
`Docs/API-related/Email_Processing_API.md`.

### Follow-up Evidence

| Check | Result | Scope |
| --- | --- | --- |
| Final expanded offline regression run | 218 passed, 2 skipped, 8 warnings; zero outbound attempts | Core import/search/detail, reimports with overwrite on/off, all email DB runtime suites, general Media regressions, parser/process endpoints, checker tests and mocked connector tests. Native PST tests skipped. |
| Review date-boundary regressions | 46 passed, 4 warnings; zero outbound attempts | Identity and cursor suites, including UTC overflow, cursor reference underflow and oversized relative windows. |
| Final mocked Gmail slice | 25 passed, 4 warnings; zero outbound attempts | Existing connector behavior plus saved-content/live-label consistency, empty-body and literal-placeholder regressions. |
| Legacy saved-metadata regression | 1 passed, 4 warnings; zero outbound attempts | Real legacy stripped-version metadata is retained; another tenant cannot supply fallback metadata. |
| Additional Media/Collections compatibility | 22 passed, 4 warnings; zero outbound attempts | Media updates, synced updates, version rollback and schema bootstrap. |
| Bandit | Zero production findings; zero findings in new/expanded focused tests with B101 excluded | Existing large mocked-provider fixture retains its 20 baseline dummy-token findings; no new findings. |
| Ruff | Changed email modules/new tests pass; no new diagnostics in legacy files | Existing checker/persistence/service/test files retain 65 baseline diagnostics in total, verified against HEAD. |
| Independent review | No remaining blockers | Follow-up review confirmed metadata recovery, transaction atomicity and placeholder handling. |
| 1,000-message SQLite benchmark | Warm p50 4.982 ms / p95 6.441 ms over 200 measurements | 10 query shapes, 20 runs each after 3 warmups; all query plans used indexes. Two shapes returned no matches in this small fixture. |
| Cold benchmark observations | p50 5.593 ms / p95 7.989 ms over 10 observations | One observation per query shape. |
| Synthetic fixture writes | 1,000 in 11.38 s, approximately 87.87/second | Direct Media/graph fixture persistence, not full upload parsing/API throughput. |
| PostgreSQL migration | Skipped by official `pg_database_config` fixture | Explicit local test DSN; localhost PostgreSQL unavailable, Docker disabled. No alternate database setup. |
| Native PST | Two selected tests skipped | `pypff` and `readpst` unavailable; no genuine PST fixture supplied. Mocked traversal is not binary-parser certification. |

The offline regression command extends the initial command with
`test_email_identity_dedupe.py`, `test_email_search_cursor.py`,
`test_email_search_cursor_endpoint.py`, `test_email_m2_gate_validation.py`,
`test_process_emails_endpoint.py`, all `test_media_db_email_*.py` files,
`test_media_db_v2_regressions.py`, and the full mocked `test_policy_and_connectors.py`.
The final run also includes `test_persistence_chunk_consistency.py` and
`test_gmail_reimport_consistency.py`. After collection, the literal-placeholder
regression and marker fix were checked in the final mocked Gmail slice; the legacy
helper test ran separately. Counts overlap and should not be summed.
Task-local logs: `/tmp/email_combined_13251_final.log`,
`/tmp/email_gmail_consistency_final.log`, `/tmp/email_persisted_content_final.log`,
`/tmp/email_media_compat_final.log` and `/tmp/email_review_fixes.log`.
Security reports: `/tmp/bandit_email_core_final.json`,
`/tmp/bandit_email_tests_final.json` and `/tmp/bandit_email_gmail_final.json`.

Benchmark reproduction (fresh temporary output/database path):

```bash
source .venv/bin/activate
python Helper_Scripts/benchmarks/email_search_bench.py \
  --ensure-fixture --fixture-messages 1000 --seed 42 \
  --capture-query-plans --runs 20 --warmup-runs 3 \
  --db-path /tmp/email-search-small-4wy1q280/email.sqlite \
  --out /tmp/email-search-small-4wy1q280/report.json
```

The recorded benchmark's report and run log are in
`/tmp/email-search-small-4wy1q280/`; optional backend results are in
`/tmp/email-optional-validation-5up_7fz_/run.log`. These are local diagnostic
artifacts. The small benchmark does not satisfy the 1M-message or full-ingestion
throughput gates. Target deployment, real auth/startup, production-scale parity and
cutover readiness remain unverified. Optional live Gmail remains deferred; no
personal mailbox, OAuth flow or external model was accessed.
