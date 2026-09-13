# Synthetic Email Core Validation — 2026-09-13

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
