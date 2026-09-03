# Integrations And Providers Domain Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: Integrations and Providers
- In scope: outbound providers, web search and scraping, TTS/STT adapters, external API clients, rate limiting, egress, secret handling, and integration tests.
- Out of scope: remediation implementation and new provider additions.
- Review mode: static inspection plus targeted local tests only. No production/source code was edited.
- Candidate finding count: 3.

## Findings Table

| ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CANDIDATE-integrations-providers-001 | likely_risk | static_confirmed | medium | high | security | Workflow research adapters bypass centralized outbound HTTP controls | open | needs_reproduction |
| CANDIDATE-integrations-providers-002 | likely_risk | static_confirmed | medium | high | security | Tokenizer resolver bypasses centralized outbound HTTP controls | open | needs_reproduction |
| CANDIDATE-integrations-providers-003 | improvement_opportunity | static_confirmed | low | high | security | Weather provider uses raw httpx for API-key-bearing request | open | validated |

## Index Mapping

This domain uses the requested candidate ID range `CANDIDATE-integrations-providers-NNN`.
If promoted into `findings-index.json`, map these candidates to the integrations/provider audit range, for example `AUDIT-2026-06-27-INTEGRATIONS-001` through `AUDIT-2026-06-27-INTEGRATIONS-003`.
For each promoted entry:

- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/integrations-providers.md`
- `owner_domain`: `Integrations and Providers`
- Preserve the `affected_paths`, `recommendation`, `status`, and `validation_status` from the detailed candidate entry.

## Confirmed Issues

No issues were promoted to a final confirmed audit ID in this domain pass. The static review did identify three open candidate findings below.

## Likely Risks

### CANDIDATE-integrations-providers-001 - Workflow research adapters bypass centralized outbound HTTP controls

- Evidence tier: `likely_risk`
- Evidence strength: `static_confirmed`
- Severity: `medium`
- Confidence: `high`
- Category: `security`
- Status: `open`
- Validation status: `needs_reproduction`
- Affected paths:
  - `tldw_Server_API/app/core/Workflows/adapters/research/search.py`
  - `tldw_Server_API/app/core/Workflows/adapters/research/bibliography.py`
  - `tldw_Server_API/tests/Workflows/adapters/test_research_adapters.py`
- Evidence:
  - `search.py:194-200` downloads `pdf_url` with `httpx.AsyncClient()` directly when `arxiv_download` is given a URL.
  - `search.py:236` and `search.py:274-298` use raw `httpx.AsyncClient()` for PubMed.
  - `search.py:340` and `search.py:377-388` use raw `httpx.AsyncClient()` for Semantic Scholar.
  - `search.py:507` and `search.py:547-552` use raw `httpx.AsyncClient()` for Google Patents.
  - `bibliography.py:46` and `bibliography.py:94-102` use raw `httpx.AsyncClient()` for DOI resolution.
  - These paths do not call `tldw_Server_API.app.core.http_client.afetch/create_async_client` and do not call `evaluate_url_policy` or the web outbound policy helper before outbound provider calls.
  - The central client provides the behavior these paths miss: `http_client.py:6-13`, `http_client.py:147-149`, `http_client.py:982-1045`, and `http_client.py:1071-1087` document/enforce safe defaults, egress checks, and proxy controls.
- Impact:
  - Workflow research tasks can make external network requests outside the central egress, DNS, redirect, proxy, retry, metrics, and logging policy. The direct `pdf_url` path is the most sensitive because the URL can be supplied through workflow configuration or previous-step context, which makes private-network or metadata-service egress a plausible failure mode if workflows are available to untrusted or lower-trust users.
  - Raw `httpx.AsyncClient()` also defaults to honoring environment proxy configuration unless explicitly disabled, unlike the project central client defaults.
- Existing tests:
  - `test_research_adapters.py` covers test-mode behavior and sanitized backend errors around these adapters, for example `1211-1225`, `1401-1430`, `1521-1550`, `1753-1781`, and `1893-1943`.
  - I did not find tests asserting egress denial, proxy avoidance, or use of the centralized HTTP client for these workflow research calls.
- Recommendation:
  - Route workflow research outbound calls through `tldw_Server_API.app.core.http_client.afetch` or `create_async_client`.
  - Enforce the same outbound URL policy used by web search/scraping before fetching user-supplied or context-derived URLs.
  - Add regression tests for blocked private/loopback URLs, `trust_env=False` behavior, and the direct `pdf_url` download path.

### CANDIDATE-integrations-providers-002 - Tokenizer resolver bypasses centralized outbound HTTP controls

- Evidence tier: `likely_risk`
- Evidence strength: `static_confirmed`
- Severity: `medium`
- Confidence: `high`
- Category: `security`
- Status: `open`
- Validation status: `needs_reproduction`
- Affected paths:
  - `tldw_Server_API/app/core/LLM_Calls/tokenizer_resolver.py`
  - `tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py`
  - `tldw_Server_API/tests/Writing/test_llm_providers_tokenizer_metadata.py`
- Evidence:
  - `tokenizer_resolver.py:190-214` constructs provider-native tokenizer endpoints and calls `_http_post()`.
  - `tokenizer_resolver.py:318-325` sends commercial/token-counting requests via `_http_post()`.
  - `tokenizer_resolver.py:428-459` sends Google countTokens requests via `_http_post()`, including optional query-key fallback behavior.
  - `tokenizer_resolver.py:567-589` sends Bedrock count-only requests via `_http_post()`.
  - `tokenizer_resolver.py:651-656` implements `_http_post()` by importing `requests` and calling `requests.post()` directly.
  - `tokenizer_resolver.py:1119-1190` and `tokenizer_resolver.py:1212-1393` select normalized provider endpoints and commercial base URLs without passing through the central egress policy.
  - By contrast, the primary LLM provider adapters and helper session use the central HTTP layer, such as `LLM_Calls/http_helpers.py:12-20` and `LLM_Calls/http_helpers.py:82-92`.
- Impact:
  - Tokenizer/counting probes can make outbound calls that bypass central egress checks, DNS validation, proxy policy, retry policy, request logging controls, and `trust_env=False` defaults.
  - When runtime exact tokenization is enabled against configured endpoints, a malicious or mistaken provider base URL can cause outbound requests to private or otherwise disallowed hosts. API keys in headers or query parameters can also traverse environment proxies because raw `requests.post()` honors environment proxy settings by default.
- Existing tests:
  - Tokenizer tests cover adapter selection, several request payloads, provider metadata, and some host guard behavior.
  - I did not find tests asserting central `http_client` use, egress denial for tokenizer URLs, or proxy avoidance.
- Recommendation:
  - Replace `_http_post()` with the central HTTP client helpers, or wrap it with equivalent egress/proxy defaults and test coverage.
  - Add regression tests for private/loopback/tokenizer URLs denied by policy, environment proxy avoidance, and commercial provider base URL overrides.
  - If local tokenizer endpoints are intentionally supported, gate them behind an explicit local-provider allow option rather than bypassing the central path.

## Improvement Opportunities

### CANDIDATE-integrations-providers-003 - Weather provider uses raw httpx for API-key-bearing request

- Evidence tier: `improvement_opportunity`
- Evidence strength: `static_confirmed`
- Severity: `low`
- Confidence: `high`
- Category: `security`
- Status: `open`
- Validation status: `validated`
- Affected paths:
  - `tldw_Server_API/app/core/Integrations/weather_providers.py`
  - `tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py`
- Evidence:
  - `weather_providers.py:14` imports `httpx`; `weather_providers.py:29-30` sets `http_client_factory = httpx.Client`.
  - `weather_providers.py:90-107` builds OpenWeather query params including `appid`.
  - `weather_providers.py:219-221` sends the request with raw `http_client_factory(timeout=...)` and `client.get(...)`.
  - The URL is fixed to OpenWeather, so the SSRF surface is much narrower than the workflow research adapter issue. The remaining concern is inconsistent centralized policy and environment proxy exposure for an API-key-bearing request.
- Existing tests:
  - `test_weather_providers.py` covers parsing, configuration, and sanitized error behavior.
  - I did not find tests asserting central HTTP client use, `trust_env=False`, or egress/proxy behavior for weather requests.
- Recommendation:
  - Use `tldw_Server_API.app.core.http_client.fetch` or `create_client` for OpenWeather requests, keeping the existing test injection seam if useful.
  - Add a small regression test that verifies the weather request client inherits central proxy/egress defaults or explicitly sets `trust_env=False`.

## Coverage And Evidence

### Files Inspected

Required context files:

- `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/endpoint-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/dependency-manifest-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`

Primary implementation files sampled or inspected:

- `tldw_Server_API/app/core/http_client.py`
- `tldw_Server_API/app/core/Security/egress.py`
- `tldw_Server_API/app/core/Web_Scraping/outbound_policy.py`
- `tldw_Server_API/app/core/Web_Scraping/url_utils.py`
- `tldw_Server_API/app/core/WebSearch/Web_Search.py`
- `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py`
- `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- `tldw_Server_API/app/core/External_Sources/gmail.py`
- `tldw_Server_API/app/core/External_Sources/google_drive.py`
- `tldw_Server_API/app/core/External_Sources/notion.py`
- `tldw_Server_API/app/core/External_Sources/onedrive.py`
- `tldw_Server_API/app/core/External_Sources/zotero.py`
- `tldw_Server_API/app/core/External_Sources/connectors_service.py`
- `tldw_Server_API/app/core/Integrations/weather_providers.py`
- `tldw_Server_API/app/core/TTS/tts_resource_manager.py`
- `tldw_Server_API/app/core/TTS/adapters/openai_adapter.py`
- `tldw_Server_API/app/core/TTS/adapters/elevenlabs_adapter.py`
- `tldw_Server_API/app/core/TTS/adapters/qwen3_runtime_remote.py`
- `tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_supervisor.py`
- `tldw_Server_API/app/core/LLM_Calls/http_helpers.py`
- `tldw_Server_API/app/core/LLM_Calls/tokenizer_resolver.py`
- `tldw_Server_API/app/core/LLM_Calls/providers/*_adapter.py` by targeted search/read sampling
- `tldw_Server_API/app/core/LLM_Calls/huggingface_api.py`
- `tldw_Server_API/app/core/Workflows/adapters/research/search.py`
- `tldw_Server_API/app/core/Workflows/adapters/research/bibliography.py`
- `tldw_Server_API/app/core/Workflows/adapters/research/_config.py`
- `tldw_Server_API/app/api/v1/endpoints/slack.py`
- `tldw_Server_API/app/api/v1/endpoints/slack_support.py`
- `tldw_Server_API/app/api/v1/endpoints/discord.py`
- `tldw_Server_API/app/api/v1/endpoints/discord_support.py`

Tests inspected by inventory/search or direct reads:

- `tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py`
- `tldw_Server_API/tests/Writing/test_llm_providers_tokenizer_metadata.py`
- `tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py`
- `tldw_Server_API/tests/Workflows/adapters/test_research_adapters.py`
- `tldw_Server_API/tests/WebSearch/test_websearch_core.py`
- `tldw_Server_API/tests/WebSearch/unit/*`
- `tldw_Server_API/tests/TTS_NEW/unit/*`
- `tldw_Server_API/tests/TTS_NEW/integration/*`
- `tldw_Server_API/tests/TTS/*`
- `tldw_Server_API/tests/STT/*`
- `tldw_Server_API/tests/External_Sources/*`
- `tldw_Server_API/tests/Slack/*`
- `tldw_Server_API/tests/Discord/*`
- `tldw_Server_API/tests/Integrations/*`
- `tldw_Server_API/tests/LLM_Calls/*`
- `tldw_Server_API/tests/Research/*`

### Tests Or Scans Run

Local targeted test run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py tldw_Server_API/tests/Workflows/adapters/test_research_adapters.py -q
```

Result: `120 passed, 248 warnings in 12.18s`.

Existing audit scan evidence read:

- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
  - Existing Bandit app summary reported 4,818 total findings, 0 high severity, 26 medium severity. This domain pass did not rerun repository-wide Bandit because no source code was changed and coordinator rules prohibited environment-changing setup.

### Blocked Or Unverified Areas

- No live provider/API calls were made.
- No network access, Docker, service startup, dependency installation, or environment-changing setup was used.
- Full repository tests were not run.
- Third-party libraries that perform their own outbound calls, such as `arxiv` and `scholarly`, were reviewed at call sites but not dynamically instrumented.
- STT provider review was limited to endpoint/test inventory and file discovery; deeper STT runtime behavior was not dynamically exercised.
- Broad grep output was occasionally truncated by the shell-output limit; high-signal hits were followed with targeted reads.

### Evidence Notes

Scoped evidence written:

- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/integrations-providers-static-evidence.txt`

Command log highlights:

```bash
sed -n '1,260p' Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md
sed -n '1,260p' Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json
sed -n '1,260p' Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/endpoint-inventory.txt
sed -n '1,260p' Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt
sed -n '1,260p' Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/dependency-manifest-inventory.txt
sed -n '1,260p' Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt
sed -n '1,260p' Docs/superpowers/reviews/2026-06-27-repo-audit/domains/integrations-providers.md
rg -n "websearch|research|provider|audio|transcription|speech|tts|stt|slack|oauth|Integrations|External|OpenAI|Anthropic|Ollama|llm|search" Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/endpoint-inventory.txt
rg -n "WebSearch|websearch|Research|research|TTS|STT|speech|transcription|provider|adapter|External|OpenAI|Anthropic|Ollama|slack|egress" Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt
find tldw_Server_API/app/core -maxdepth 3 \( -path '*Integrations*' -o -path '*WebSearch*' -o -path '*TTS*' -o -path '*LLM_Calls*' -o -path '*Audio*' -o -path '*STT*' -o -path '*Research*' -o -path '*External*' \) -print | sort
rg --files tldw_Server_API/app/core tldw_Server_API/app/api/v1/endpoints tldw_Server_API/tests | rg '(External_Sources|connectors|slack|discord|research|WebSearch|TTS|STT|LLM_Calls)'
rg -n "from tldw_Server_API\.app\.core\.http_client|import httpx|import requests|requests\.|httpx\.|create_client|create_async_client|fetch\(|afetch\(" tldw_Server_API/app/core tldw_Server_API/app/api/v1/endpoints tldw_Server_API/tests
rg -n "evaluate_url_policy|decide_web_outbound_policy|skip_egress_check|egress|allowlist|block_private|private" tldw_Server_API/app/core tldw_Server_API/app/api/v1/endpoints tldw_Server_API/tests
rg -n "API_KEY|api_key|token|secret|Authorization|Bearer|headers" tldw_Server_API/app/core tldw_Server_API/app/api/v1/endpoints tldw_Server_API/tests
rg -n "CANDIDATE-integrations-providers|AUDIT-2026-06-27-INTEGRATIONS|integrations-providers" Docs/superpowers/reviews/2026-06-27-repo-audit
nl -ba tldw_Server_API/app/core/http_client.py | sed -n '1,360p'
nl -ba tldw_Server_API/app/core/http_client.py | sed -n '760,1180p'
nl -ba tldw_Server_API/app/core/LLM_Calls/tokenizer_resolver.py | sed -n '1,760p'
nl -ba tldw_Server_API/app/core/LLM_Calls/tokenizer_resolver.py | sed -n '1080,1460p'
nl -ba tldw_Server_API/app/core/Integrations/weather_providers.py | sed -n '1,340p'
nl -ba tldw_Server_API/app/core/Workflows/adapters/research/search.py | sed -n '1,620p'
nl -ba tldw_Server_API/app/core/Workflows/adapters/research/bibliography.py | sed -n '1,140p'
nl -ba tldw_Server_API/app/api/v1/endpoints/slack_support.py | sed -n '1,620p'
nl -ba tldw_Server_API/app/api/v1/endpoints/discord_support.py | sed -n '1,660p'
nl -ba tldw_Server_API/tests/Workflows/adapters/test_research_adapters.py | sed -n '1180,1945p'
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py tldw_Server_API/tests/Workflows/adapters/test_research_adapters.py -q
```

Two exploratory path reads failed and were not used as evidence:

```bash
nl -ba tldw_Server_API/app/core/HTTP/client.py | sed -n '1,260p'
nl -ba tldw_Server_API/app/core/External_Sources/connectors.py | sed -n '520,680p'
```
