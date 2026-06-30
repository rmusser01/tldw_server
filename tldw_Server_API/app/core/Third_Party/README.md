# Third_Party

Provider adapters for external scholarly/metadata services used by the Paper Search API. Each adapter normalizes provider responses to a common shape and exposes simple search/lookup helpers; endpoints are defined in `paper_search.py` and `research.py`.

## 1. Descriptive of Current Feature Set

- Purpose: Unified access to paper/preprint providers (search + DOI/ID lookups) with consistent return signatures and resilient HTTP behavior.
- Capabilities (adapters):
  - arXiv: query + XML convert helpers — `Arxiv.py`
  - BioRxiv/MedRxiv: multiple report feeds, raw passthroughs — `BioRxiv.py`
  - PubMed + PMC (OAI-PMH, OA service): search and ingest helpers — `PubMed.py`, `PMC_OAI.py`, `PMC_OA.py`
  - Semantic Scholar: search + details — `Semantic_Scholar.py`
  - OpenAlex: venue-constrained search + DOI lookup — `OpenAlex.py`
  - Crossref: DOI search/lookup — `Crossref.py`
  - IEEE Xplore (keyed): search + DOI/ID lookup — `IEEE_Xplore.py`
  - Springer Nature (keyed): search + DOI lookup — `Springer_Nature.py`
  - Elsevier Scopus (keyed): search + DOI lookup — `Elsevier_Scopus.py`
  - OSF/EarthArXiv, Figshare, Zenodo, IACR, RePEc, Vixra — dedicated adapters following the same pattern.
- Inputs/Outputs:
  - Inputs: query strings, pagination/filters, optional provider API keys.
  - Outputs: lists of normalized paper dicts or single‑record lookups; on error: `(None, ..., error_message)`.
- Related Endpoints (primary):
  - Paper Search router: tldw_Server_API/app/api/v1/endpoints/paper_search.py:64
  - arXiv search: tldw_Server_API/app/api/v1/endpoints/paper_search.py:67
  - BioRxiv search: tldw_Server_API/app/api/v1/endpoints/paper_search.py:123
  - MedRxiv alias + raw feeds: tldw_Server_API/app/api/v1/endpoints/paper_search.py:186, :224, :260, :295
  - PMC OAI-PMH (identify/list-sets/list-identifiers/list-records): tldw_Server_API/app/api/v1/endpoints/paper_search.py:327, :352, :382, :419
  - arXiv ingest (PDF download + DB persist): tldw_Server_API/app/api/v1/endpoints/paper_search.py:810
  - Deprecated research shims (arXiv/Semantic Scholar): tldw_Server_API/app/api/v1/endpoints/research.py:58, :210
- Related Schemas:
  - arXiv + Semantic Scholar: tldw_Server_API/app/api/v1/schemas/research_schemas.py:21, :64
  - BioRxiv/MedRxiv + PMC/OSF/Generic: tldw_Server_API/app/api/v1/schemas/paper_search_schemas.py:9, :150, :203, :229

## 2. Technical Details of Features

- Architecture & Data Flow:
  - Endpoints (FastAPI) receive validated forms and offload blocking provider calls to a thread pool; adapters use the centralized HTTP client (`core/http_client.py`) with retry/jitter, trust_env=False, and metrics hooks.
  - Adapters return tuples following convention: for search `(items: Optional[List[dict]], total: int, error: Optional[str])`, for lookups `(record: Optional[dict], error: Optional[str])`.
- Key Modules/Functions:
  - arXiv: `search_arxiv_custom_api`, `fetch_arxiv_xml`, `convert_xml_to_markdown` — tldw_Server_API/app/core/Third_Party/Arxiv.py:61, :99, :213
  - Semantic Scholar: `search_papers_semantic_scholar`, `get_paper_details_semantic_scholar` — tldw_Server_API/app/core/Third_Party/Semantic_Scholar.py:39, :152
  - OpenAlex: `search_openalex`, `get_openalex_by_doi` — tldw_Server_API/app/core/Third_Party/OpenAlex.py:58, :102
  - Crossref: `search_crossref`, `get_crossref_by_doi` — tldw_Server_API/app/core/Third_Party/Crossref.py:57, :97
  - IEEE Xplore: `search_ieee`, `get_ieee_by_doi`, `get_ieee_by_id` — tldw_Server_API/app/core/Third_Party/IEEE_Xplore.py:49, :97, :127
  - Springer Nature: `search_springer`, `get_springer_by_doi` — tldw_Server_API/app/core/Third_Party/Springer_Nature.py:63, :110
  - Elsevier Scopus: `search_scopus`, `get_scopus_by_doi` — tldw_Server_API/app/core/Third_Party/Elsevier_Scopus.py:71, :118
- Dependencies:
  - Uses `requests` with `HTTPAdapter` retries; tries `httpx` via `core/http_client.create_client` when available.
  - Optional keys per provider (see Configuration); some offer improved limits when setting contact email (e.g., OpenAlex mailto).
- Data Models & DB:
  - Response shapes normalized to “GenericPaper”-like fields: `id`, `title`, `authors`, `journal`, `pub_date`, `abstract`, `doi`, `url`, `pdf_url`, `provider`.
  - Ingest endpoints optionally persist content to the per‑user Media DB via `MediaDatabase.add_media_with_keywords` (see arXiv/PMC OA ingest paths in `paper_search.py`).
- Configuration (env vars):
  - `IEEE_API_KEY` (IEEE Xplore), `SPRINGER_NATURE_API_KEY` (Springer), `ELSEVIER_API_KEY` (+ optional `ELSEVIER_INST_TOKEN`) (Scopus).
  - `OPENALEX_MAILTO` (recommended to improve reliability/rate limits), `UNPAYWALL_EMAIL` (required for Unpaywall DOI resolution).
  - HTTP client knobs (optional): `HTTP_CONNECT_TIMEOUT`, `HTTP_READ_TIMEOUT`, `HTTP_RETRY_ATTEMPTS`, etc. (see `core/http_client.py`).
- Concurrency & Performance:
  - Endpoints use `run_in_executor` to avoid blocking the event loop; batching handled by providers where supported.
  - Retries/backoff configured per adapter; endpoints accept OK/timeout/gateway errors to avoid flakiness in CI when providers throttle.
- Error Handling:
  - All adapters map network/HTTP errors to informative `error_message` strings; endpoints convert to HTTP 502/504 or 404 when appropriate.
  - Raw passthrough endpoints (BioRxiv) return provider content and media types directly when requested.
- Security:
  - Secrets via env only; headers redacted in logs by the centralized client. Proxies are disabled by default (`trust_env=False`).

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure:
  - One file per provider under `core/Third_Party/` implementing search/lookup helpers; keep shapes aligned with GenericPaper fields.
- Extension Points:
  - New provider adapter should expose `search_<name>(...) -> (items,total,error)` and `get_<name>_by_doi(...) -> (record,error)` when applicable.
  - Normalize outputs to the common fields and use the centralized HTTP client (prefer) or `requests` with retries.
- Coding Patterns:
  - Minimal transformation + explicit normalization; no heavy parsing beyond what the provider returns.
  - Use environment‑gated features (e.g., API keys) and return a helpful error when missing (see IEEE/Springer/Scopus helpers).
- Tests:
  - External/integration tests (skipped by default; set `RUN_EXTERNAL_API_TESTS=1` to enable):
    - tldw_Server_API/tests/PaperSearch/integration/test_paper_search_external.py:1
    - tldw_Server_API/tests/PaperSearch/integration/test_biorxiv_reports_external.py:1
    - tldw_Server_API/tests/PaperSearch/integration/test_figshare_external.py:1
    - tldw_Server_API/tests/PaperSearch/integration/test_earthrxiv_external.py:1
    - tldw_Server_API/tests/PaperSearch/integration/test_vixra_external.py:1
    - tldw_Server_API/tests/PaperSearch/integration/test_iacr_external.py:1
    - tldw_Server_API/tests/PaperSearch/integration/test_medrxiv_external.py:1
    - tldw_Server_API/tests/PaperSearch/integration/test_zenodo_external.py:1
  - Missing‑key behavior is validated (returns 501) for IEEE/Springer/Scopus — see test_paper_search_external.py:61, :69, :77.
- Local Dev Tips:
  - Start with OpenAlex/Crossref (no keys). Configure optional keys as needed (`IEEE_API_KEY`, `SPRINGER_NATURE_API_KEY`, `ELSEVIER_API_KEY`).
  - For Unpaywall DOI OA resolution, set `UNPAYWALL_EMAIL`.
  - Use the `paper_search.py` endpoints for quick manual testing via `/docs`.
- Pitfalls & Gotchas:
  - Providers rate‑limit and occasionally time out; endpoints intentionally allow 200/404/502/504 in tests to reduce flakiness.
  - Some providers don’t return direct `pdf_url` (e.g., Scopus); resolve via Unpaywall when needed.
  - XML/HTML parsing can be brittle (arXiv/PMC); prefer small, defensive transformations.
- Follow-up candidates:
  - Expand provider parity (additional OSF preprint servers), add cache hints, and consider adapter‑level metrics (success rate/latency per provider).
