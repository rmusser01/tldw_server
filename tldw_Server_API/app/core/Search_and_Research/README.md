# Search_and_Research

`Search_and_Research` is an umbrella package for research-facing search flows.
The active implementation is split across the legacy research endpoint,
paper-search endpoints, `WebSearch`, `Web_Scraping`, and `Third_Party`
provider modules. Keep this README focused on orientation so contributors know
where the real code lives.

## Start Here

- Package marker: `__Init__.py`.
- Web search endpoint: `app/api/v1/endpoints/research.py` (`/research/websearch`).
- Paper search endpoint: `app/api/v1/endpoints/paper_search.py`.
- Web search core: `app/core/WebSearch/` and
  `app/core/Web_Scraping/WebSearch_APIs.py`.
- Paper providers: `app/core/Third_Party/`.
- Schemas: `app/api/v1/schemas/websearch_schemas.py` and
  `app/api/v1/schemas/research_schemas.py`.
- Tests: `tests/WebSearch/`, `tests/PaperSearch/`, and relevant
  `tests/WebScraping/` coverage.

## Responsibilities

- Point contributors to the web-search and paper-search implementation split.
- Document that paper-search provider integrations live in `Third_Party`.
- Document that web search aggregation and article extraction use WebSearch and
  Web_Scraping components.

## Module Map

- `__Init__.py` is a legacy package marker. It does not contain the active
  orchestration code.

## How It Connects

- `/api/v1/research/websearch` delegates provider calls, scraping, and optional
  aggregation to web-search modules.
- `/api/v1/paper-search/*` calls provider-specific third-party modules such as
  arXiv, Semantic Scholar, PubMed/PMC, OSF, Zenodo, and OpenAlex helpers.
- Optional ingestion paths can persist paper/search content into Media DB for
  later RAG.
- Outbound calls should honor `Security.egress` before hitting provider URLs.

## Extension Points

- Add paper providers in `Third_Party`, then expose them from `paper_search.py`
  and schemas/tests.
- Add web-search providers in the WebSearch/Web_Scraping provider layer and
  update engine-list tests.
- If moving active code into this package, preserve endpoint compatibility and
  update the inventory row with the new source paths.

## Testing

- Web search endpoint and provider routing: `tests/WebSearch/`.
- Paper provider integration coverage: `tests/PaperSearch/`.
- Scraping and egress behavior: `tests/WebScraping/` and
  `tests/Security/test_websearch_egress_guard.py`.

## Gotchas

- The package name is broader than the code it currently contains. Do not assume
  an empty-looking module means search/research functionality is absent.
- Provider quotas and response schemas vary; tests should mock network calls
  unless they are explicitly external-integration tests.
