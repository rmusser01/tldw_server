# Chat, RAG, And LLM Domain Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: Chat, RAG, and LLM
- In scope: chat completions, queueing, streaming, provider routing, prompt handling, RAG retrieval, embeddings, model/provider configuration, and related tests.
- Out of scope: remediation implementation and provider feature expansion.

## Findings Table

| ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CANDIDATE-chat-rag-llm-001 | likely_risk | static_confirmed | medium | high | security | Alternate LLM/RAG generation routes drift from virtual-key endpoint and max-call enforcement | open | needs_reproduction |
| CANDIDATE-chat-rag-llm-002 | confirmed_issue | static_confirmed | medium | high | security | RAG search endpoints log raw user queries at info level | open | validated |

## Index Mapping

Candidate IDs use `CANDIDATE-chat-rag-llm-NNN` per coordinator instruction. If promoted into `findings-index.json`, map them to stable audit IDs like the proposed IDs below. For every promoted finding, set `source_report` to `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/chat-rag-llm.md`, set `owner_domain` to Chat, RAG, and LLM, and preserve the `evidence_tier`, `evidence_strength`, `affected_paths`, `recommendation`, `status`, and `validation_status` fields recorded below.

| Candidate ID | Proposed Audit ID |
| --- | --- |
| CANDIDATE-chat-rag-llm-001 | AUDIT-2026-06-27-CHAT-001 |
| CANDIDATE-chat-rag-llm-002 | AUDIT-2026-06-27-CHAT-002 |

## Confirmed Issues

### CANDIDATE-chat-rag-llm-002: RAG search endpoints log raw user queries at info level

- `evidence_tier`: confirmed_issue
- `evidence_strength`: static_confirmed
- `severity`: medium
- `confidence`: high
- `category`: security
- `status`: open
- `validation_status`: validated
- `affected_paths`:
  - `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- `recommendation`: Replace raw query logging with the safer pattern already used by `/rag/simple`: log a non-security hash, query length, user/request metadata, and feature flags instead of full query text. Keep full query payloads out of info-level logs; if operational debugging needs them, require an explicit debug-only opt-in with redaction.

Evidence:

- `unified_search_endpoint` logs the full request query and username at info level: `logger.info(f"Unified RAG search: query='{request.query}', user=...")` in `rag_unified.py:1238`.
- `advanced_search_endpoint` logs the full query string at info level: `logger.info(f"Advanced search: query='{query}'")` in `rag_unified.py:2062`.
- The same module has a safer precedent in `simple_search_endpoint`, which logs `query_hash` and query length instead of raw text in `rag_unified.py:1671-1675`.

Impact:

RAG queries often include copied private document text, names, credentials, URLs, internal ticket data, or prompts containing confidential instructions. These values can be written to local logs, test logs, support bundles, or operator log aggregation even when the underlying user data remains tenant-scoped. This is directly confirmed by static source inspection; no runtime reproduction was necessary.

## Likely Risks

### CANDIDATE-chat-rag-llm-001: Alternate LLM/RAG generation routes drift from virtual-key endpoint and max-call enforcement

- `evidence_tier`: likely_risk
- `evidence_strength`: static_confirmed
- `severity`: medium
- `confidence`: high
- `category`: security
- `status`: open
- `validation_status`: needs_reproduction
- `affected_paths`:
  - `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
  - `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
  - `tldw_Server_API/app/api/v1/endpoints/chat_documents.py`
  - `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
  - `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
  - `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
  - `tldw_Server_API/app/core/AuthNZ/llm_budget_middleware.py`
- `recommendation`: Give every externally reachable route that spends LLM/RAG/embedding resources a stable endpoint ID and a shared enforcement dependency. At minimum, add `TokenScopeGuard(..., endpoint_id=..., count_as="call")` or an equivalent central guard to RAG batch/simple/stream/advanced routes, character completion routes, document-generation routes, and embeddings create/batch routes. Add HTTP tests with virtual API keys limited by `allowed_endpoints`, `allowed_paths`, `allowed_methods`, and `max_calls` for each alternate path.

Evidence:

- The primary `/api/v1/rag/search` route declares `rbac_rate_limit("rag.search")`, `RequirePermission(MEDIA_READ)`, `TokenScopeGuard("any", endpoint_id="rag.search", count_as="call")`, and the RAG daily limit in `rag_unified.py:1211-1217`. `/api/v1/rag/source-health` uses the same token-scope endpoint ID in `rag_unified.py:1155-1160`.
- Adjacent RAG routes omit `TokenScopeGuard` and the RBAC endpoint-rate dependency:
  - `/api/v1/rag/batch` has only `check_rate_limit` and `RequirePermission(MEDIA_READ)` in `rag_unified.py:1448-1451`.
  - `/api/v1/rag/simple` has `check_rate_limit`, `RequirePermission(MEDIA_READ)`, and the daily RAG limit, but no token-scope guard in `rag_unified.py:1652-1656`.
  - `/api/v1/rag/search/stream` has `check_rate_limit`, `RequirePermission(MEDIA_READ)`, and the daily RAG limit, but no token-scope guard in `rag_unified.py:1945-1949`.
  - `/api/v1/rag/advanced` has only `check_rate_limit` and `RequirePermission(MEDIA_READ)` in `rag_unified.py:2047`.
- Character-chat provider-call routes are externally mounted under `/api/v1/chats` and use per-user auth/ownership checks, but no token-scope endpoint/count dependency:
  - The deprecated `/api/v1/chats/{chat_id}/complete` route has no route-level dependencies and depends on `get_chacha_db_for_user` plus `get_request_user` in `character_chat_sessions.py:4339-4359`.
  - `/api/v1/chats/{chat_id}/complete-v2` says it "calls a provider" and performs streaming/persistence, but its route has no `TokenScopeGuard`; handler dependencies are `get_chacha_db_for_user`, routing decision store, and `get_request_user` in `character_chat_sessions.py:5137-5168`.
- Chat document generation routes are mounted under `/api/v1/chat` and can submit provider-backed work, but do not declare token-scope endpoint/count dependencies:
  - `/api/v1/chat/documents/generate` requires a provider and uses `get_auth_principal`, but no `TokenScopeGuard`, in `chat_documents.py:96-115`.
  - `/api/v1/chat/documents/bulk` submits multiple document generation jobs without a token-scope guard in `chat_documents.py:797-815`.
- Embeddings create/batch routes are covered by the LLM budget middleware default path set, but still lack route-level `TokenScopeGuard(..., count_as="call")`, so the per-key `max_calls` counter enforced by that guard is not obviously applied:
  - `/api/v1/embeddings` has `rbac_rate_limit("embeddings.create")` and the daily API call limit in `embeddings_v5_production_enhanced.py:2250-2258`.
  - `/api/v1/embeddings/batch` has the same dependencies in `embeddings_v5_production_enhanced.py:2935-2941`.
- The auth dependency shows why this inconsistency matters:
  - `TokenScopeGuard` is where JWT `allowed_endpoints` are checked in `auth_deps.py:2194-2200`.
  - For API keys, `TokenScopeGuard` checks `llm_allowed_endpoints`, path metadata, and `max_calls`/`max_runs` quota metadata in `auth_deps.py:2393-2415` and `auth_deps.py:2448-2485`.
  - Generic `RequirePermission` only checks permission claims in `auth_deps.py:1387-1414`; it does not apply endpoint allowlists or per-key call counters.
  - API-key authentication only passes endpoint/path/method `usage_details` into `validate_api_key` when prior middleware/dependencies have attached `_auth_endpoint_id`, `_auth_action`, or `_auth_scope_name` to request state in `User_DB_Handling.py:1049-1081`.
  - Scoped JWTs fail closed on routes that do not declare a token-scope dependency in `User_DB_Handling.py:565-610`; this reduces JWT exposure, but does not cover unscoped or virtual API-key constraints on routes missing the guard.
- The default LLM budget middleware path list covers `/api/v1/chat/completions` and `/api/v1/embeddings` only in `settings.py:651-656`, and the middleware maps only those prefixes to endpoint codes in `llm_budget_middleware.py:108-142`. That helps for embeddings endpoint allowlists/budgets, but it does not cover RAG, `/api/v1/chats`, or `/api/v1/chat/documents` by default, and it does not replace route-level `count_as` max-call enforcement.
- Existing focused quota tests exercise `max_calls` for `/api/v1/chat/completions` and `/api/v1/rag/search`, but not the alternate RAG, character-completion, document-generation, or embeddings max-call paths. The relevant coverage is in `test_quota_enforcement_http_sqlite.py:111-174` for JWT chat/RAG and `test_quota_enforcement_http_sqlite.py:220-283` for API-key RAG/chat.

Impact:

Virtual/API keys meant to be narrowly constrained to a specific endpoint or one call can have inconsistent behavior across adjacent routes that perform similar resource-spending work. The most sensitive drift is on RAG alternate routes and character/document generation because the default LLM budget middleware does not include those paths. Embeddings have endpoint/budget middleware coverage by default, but the route still appears to miss the token-scope `max_calls` counter. Dynamic HTTP reproduction should confirm the exact runtime behavior under both SQLite and Postgres AuthNZ backends before promotion.

## Improvement Opportunities

No improvement-only findings recorded.

## Coverage And Evidence

### Files Inspected

- `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/endpoint-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
- `tldw_Server_API/app/api/v1/endpoints/chat.py`
- `tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py`
- `tldw_Server_API/app/api/v1/endpoints/chat_documents.py`
- `tldw_Server_API/app/api/v1/endpoints/chat_grammars.py`
- `tldw_Server_API/app/api/v1/endpoints/chat_loop.py`
- `tldw_Server_API/app/api/v1/endpoints/chat_workflows.py`
- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- `tldw_Server_API/app/api/v1/endpoints/character_messages.py`
- `tldw_Server_API/app/api/v1/endpoints/character_memory.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`
- `tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py`
- `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- `tldw_Server_API/app/core/AuthNZ/llm_budget_middleware.py`
- `tldw_Server_API/app/core/AuthNZ/settings.py`
- `tldw_Server_API/app/api/v1/router_groups/content.py`
- `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Relevant tests under `tldw_Server_API/tests/AuthNZ_SQLite`, `tldw_Server_API/tests/AuthNZ_Unit`, `tldw_Server_API/tests/RAG_NEW`, `tldw_Server_API/tests/Character_Chat`, `tldw_Server_API/tests/Character_Chat_NEW`, `tldw_Server_API/tests/Chat`, and `tldw_Server_API/tests/Embeddings`.

### Tests Or Scans Run

No runtime tests were run. This was a static domain review under the coordinator constraints: no installs, no services, no Docker, no network, and no environment-changing setup.

Commands and static checks run:

- `sed -n ...` on the audit inventory, findings index, endpoint inventory, backend test inventory, Bandit summary, this report template, and peer domain reports for template conventions.
- `find Docs/superpowers/reviews/2026-06-27-repo-audit -maxdepth 3 -type f | sort`
- `wc -l` on the scoped endpoint files to size the review surface.
- `rg --files ... | rg ...` to enumerate scoped Chat/RAG/LLM/Embeddings source and test files.
- Parent virtualenv-backed Python AST inventory of endpoint routes and dependency declarations. A bare `python` invocation failed because `python` is not on this worktree shell path; the subsequent command used `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python` via activation.
- Targeted `rg` and `nl -ba | sed -n ...` inspections for `TokenScopeGuard`, `RequirePermission`, `llm_allowed_endpoints`, route dependencies, RAG query logging, router prefixes, and existing quota/allowlist tests.
- Existing audit-wide Bandit summary was reviewed from `evidence/bandit-app-summary.txt`; no new Bandit scan was run for this report-only review.

### Blocked Or Unverified Areas

- Dynamic HTTP reproduction was not performed because the coordinator prohibited starting services, Docker, installs, or environment-changing setup.
- The worktree lacks its own `.venv`; the existing parent repository virtualenv was used only for local static Python inspection. No dependency installation was attempted.
- Auth behavior was statically traced for scoped JWTs, API keys, middleware, and route dependencies, but exact runtime results for the candidate virtual-key bypass paths need follow-up tests.
- LLM provider calls were not executed. Provider error handling and streaming were inspected at route/dependency level only, with deeper provider-specific runtime behavior left as residual risk.
- Existing worktree dirty state included `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/command-log.md` before this domain edit; it was not inspected as a finding source and was not modified.

### Evidence Notes

- The highest-risk pattern is inconsistent enforcement around `TokenScopeGuard` and the LLM budget middleware. The primary OpenAI-compatible chat and main RAG search paths have stronger endpoint/count controls than adjacent feature routes that also spend provider or retrieval resources.
- Tenant/user isolation for sampled chat, character, RAG, and embeddings routes generally flows through `get_request_user`, `get_chacha_db_for_user`, `get_media_db_for_user`, or ownership checks. No cross-user data access finding was identified in the sampled static review.
- No separate scoped evidence file was created; the static line evidence is recorded inline above.
