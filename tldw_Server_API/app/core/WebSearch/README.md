# WebSearch

## 1. Descriptive of Current Feature Set

- Purpose: Unified web search and aggregation across multiple providers with optional LLM-powered subquery generation, relevance scoring, and final-answer synthesis.
- Providers: Google CSE, DuckDuckGo, Brave (AI/web), Kagi, Searx, Tavily. Bing is present in legacy code but not exposed as a supported engine in the public schema.
- Pipeline (optional stages):
  - Subquery generation via LLM to broaden coverage.
  - Provider search, normalization, and result shaping.
  - Optional user review/selection step.
  - Relevance evaluation via LLM and article scraping for evidence.
  - Aggregation into a concise final answer with citations and a confidence estimate.
- Cancellation-aware: Aggregate stage observes client disconnect and aborts in-flight work.
- Security and egress: Outbound requests respect centralized egress/SSRF policy; provider calls use browser-like headers.
- Inputs/Outputs:
  - Request model: `tldw_Server_API/app/api/v1/schemas/websearch_schemas.py:14` (engine, query, options, aggregation flags)
  - Raw response: `tldw_Server_API/app/api/v1/schemas/websearch_schemas.py:62`
  - Aggregate response + final answer: `tldw_Server_API/app/api/v1/schemas/websearch_schemas.py:52`, `tldw_Server_API/app/api/v1/schemas/websearch_schemas.py:67`
- Related Endpoint:
  - `POST /api/v1/research/websearch` — `tldw_Server_API/app/api/v1/endpoints/research.py:279`

Notes
- The API endpoint today delegates to the Web_Scraping implementation for providers and orchestration. This module hosts the parallel pipeline (and helpers) as part of an ongoing consolidation effort.

## 2. Technical Details of Features

- Architecture & Flow
  - Phase 1 (Generate + Search): `generate_and_search` builds sub-queries (optional), executes provider calls, and normalizes results.
    - Web_Scraping orchestration: `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:154`
  - Phase 2 (Analyze + Aggregate): `analyze_and_aggregate` runs relevance analysis, optional user review, and LLM aggregation.
    - Web_Scraping aggregation: `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:254`
  - Final answer shape: text, evidence (snippets), chunk summaries, confidence.
- Provider Adapters (Web_Scraping)
  - Google CSE: `search_web_google` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:1542`, `parse_google_results` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:1713`
  - Brave: `search_web_brave` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:1199`, `parse_brave_results` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:1269`
  - DuckDuckGo: `search_web_duckduckgo` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:1339`, `parse_duckduckgo_results` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:1459`
  - Kagi: `search_web_kagi` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:1820`, `parse_kagi_results` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:1861`
  - Searx: `search_web_searx` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:1925`, `parse_searx_results` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:2021`
  - Tavily: `search_web_tavily` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:2085`, `parse_tavily_results` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:2134`
  - Article scraping used during relevance/summary: `scrape_article` `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py:335`
  - UA Profiles: `tldw_Server_API/app/core/Web_Scraping/ua_profiles.py:2`, browser-like headers helper `_websearch_browser_headers` `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py:34`
- Endpoint Integration
  - Router: `tldw_Server_API/app/api/v1/endpoints/research.py:279` (offloads phase 1 to a thread pool and observes client disconnect during phase 2).
  - Thread pool configuration: `tldw_Server_API/app/api/v1/endpoints/research.py:321`
- LLM Usage
  - Subquery generation and relevance analysis leverage the unified chat stack and summarization:
    - Chat orchestrator entry: `tldw_Server_API/app/core/Chat/chat_orchestrator.py:77`
    - General summarization: `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py:312`
- Security
  - Egress/SSRF policy: `evaluate_url_policy` `tldw_Server_API/app/core/Security/egress.py:146` is enforced before external calls in providers and scrapers.
- Configuration
  - Provider keys and knobs are read from `search_engines` in `config.txt` (via config loader). Common keys include:
    - Google: `google_search_api_key`, `google_search_engine_id`, `google_search_api_url`, `limit_google_search_to_country`
    - Brave: `brave_search_api_key`, `brave_search_ai_api_key`, `search_engine_country_code_brave`
    - Searx: `searx_search_api_url`
    - Tavily: `tavily_search_api_key`
  - Headers/profile selection via `ua_profiles` helpers.
- Error Handling
  - Provider adapters return structured dicts with `processing_error` when normalization fails; endpoint traps exceptions and returns 500 with error detail.
  - Aggregate stage guards against malformed LLM outputs; returns a safe fallback when summarization is unavailable.

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure
  - This module (`core/WebSearch`) contains the parallel pipeline and helpers. The API endpoint currently delegates to `core/Web_Scraping/WebSearch_APIs.py`, which also holds provider adapters, UA profiles, and the article scraper.
- Adding/Updating a Provider
  - Implement `search_web_<provider>` and a matching `parse_<provider>_results` that appends standardized items into `web_search_results_dict`.
  - Enforce egress policy at the start of any network call using `evaluate_url_policy`.
  - Use `_websearch_browser_headers` for realistic headers where appropriate.
  - Update supported engines if the provider becomes publicly exposed: `tldw_Server_API/app/api/v1/schemas/websearch_schemas.py:9`–`tldw_Server_API/app/api/v1/schemas/websearch_schemas.py:19`.
- Endpoint/Auth
  - The endpoint requires an authenticated user (dependency `get_request_user`) and mounts under `/api/v1/research` in the main app.
- Tests (good starting points)
  - Endpoint happy-path and aggregate path: `tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py:1`
  - Engine-specific routing (tavily, searx, kagi): `tldw_Server_API/tests/WebSearch/integration/test_websearch_engines_endpoint.py:1`
  - Non-blocking generate offload behavior: `tldw_Server_API/tests/WebSearch/unit/test_nonblocking_generate.py:1`
  - Parsers unit tests (Google/Brave/DDG/Kagi): `tldw_Server_API/tests/WebSearch/unit/test_parsers.py:1`
  - Searx/Tavily parsers: `tldw_Server_API/tests/WebSearch/unit/test_parsers_extended.py:1`
  - Egress guard (security): `tldw_Server_API/tests/Security/test_websearch_egress_guard.py:1`
  - Browser-like header shape: `tldw_Server_API/tests/Web_Scraping/test_websearch_headers.py:1`
- Local Dev Tips
  - Start the app and call `POST /api/v1/research/websearch` with a small `result_count` and `aggregate=false` to validate provider wiring before testing aggregation.
  - Configure provider keys in `Config_Files/config.txt` (and/or `.env`) prior to live calls.
  - For aggregation, set `relevance_analysis_llm` and `final_answer_llm` to a configured provider name from the chat stack.
- Pitfalls & Gotchas
  - Provider quotas and per-request limits (e.g., Google CSE `num`, Brave AI token) can constrain `result_count`.
  - Some providers (Searx/Tavily) require self-hosted instance URL or API key.
  - Endpoint-level behavior assumes network access; tests may mock providers or accept 500 in offline environments.
  - Bing is deprecated in the public schema; avoid re-exposing without a clear migration/test plan.
- Follow-up candidates
  - Consolidate Web_Scraping provider adapters into this module and ensure a single pipeline implementation.
  - Expand structured relevance outputs to reduce regex-based parsing.
  - Optional caching layer for provider results to reduce egress and cost.
