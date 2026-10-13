# Usage Quotas

Usage quotas are per-user budgets: audio minutes, LLM tokens, RAG queries, storage, and so on. They are **off by default**. A stock install, single-user or multi-user, returns no usage-quota 402, 403, 413 or 429, and no number ships as a default.

This page is for operators. Decision record: `Docs/ADR/064-usage-quotas-per-user-limits.md`. Design: `Docs/Design/2026-10-02-usage-quota-posture-design.md`. Request rate limits are a separate system (Resource Governor, ADR-056); `Docs/Operations/Rate_Limits_Troubleshooting.md` helps tell the two apart.

## The master switch

`USAGE_QUOTAS_ENABLED` (`true|1|false|0`) turns quota checks on. It resolves in this order:

1. env `USAGE_QUOTAS_ENABLED`
2. env `LIMIT_ENFORCEMENT_ENABLED`, if set (legacy spelling; logs a deprecation warning once)
3. `config.txt` `[Usage-Quotas] enabled`
4. otherwise `false`

The switch gates the checks, never the counters: usage is recorded whether the switch is on or off, so turning it on mid-day or mid-month counts correctly. The switch does not depend on `RG_ENABLED`.

With the switch on, a quota applies only to a user who has a value. Nothing is set by default, so turning the switch on alone limits no one.

## How a user's limit is resolved

Each `limits.*` value is looked up per user, in this order:

1. the user's own value;
2. otherwise the **most generous** value among the teams the user belongs to that set the key;
3. otherwise the most generous value among the orgs the user belongs to that set the key;
4. otherwise unlimited.

- A team or org value is **each member's own allowance**, not a shared pool.
- `0` is the least generous value and blocks the user. A team that does not set the key does not unlimit anyone, and joining a group never lowers an allowance.
- Results are cached per user for 60 seconds. A write clears the cache in the process that handled it; other workers see the change within 60 seconds.
- A failed lookup is treated as unlimited and logged as a warning.

## Setting values

All values are integers `>= 0`. Setting them requires platform admin; an org admin cannot change them.

| Scope | Route |
|---|---|
| User | `PATCH /api/v1/admin/users/{id}/profile`, body `{"updates": [{"key": "limits.rag_queries_per_day", "value": 50}]}`. A `null` value removes the user's own value. |
| Team | `PUT /api/v1/admin/teams/{team_id}/profile/overrides/{key}`, body `{"value": 100}`; `DELETE` the same path removes it |
| Org | `PUT /api/v1/admin/orgs/{org_id}/profile/overrides/{key}`, body `{"value": 100}`; `DELETE` the same path removes it |

Team and org writes emit an audit event and clear the resolver cache. The bulk profile API (`POST /api/v1/admin/users/profile/bulk`) writes per-user values for the users it selects; it is not a group setting.

Storage also has dedicated routes, which write the same `limits.storage_quota_mb` user value:

- `PUT /api/v1/storage/admin/quotas/user/{id}` and `GET`/`PUT /api/v1/admin/storage-quotas/users/{id}` take `quota_mb` (integer, `0` or more, or `null` to remove). See `Docs/API-related/Storage_API_Documentation.md`.
- `PUT /api/v1/storage/admin/quotas/team/{id}` and `.../org/{id}` set a **shared storage pool** (minimum 100 MB). A pool is separate from a member's own `limits.storage_quota_mb` and runs only when the switch is on. `STORAGE_QUOTA_ENFORCEMENT=0` is an extra off switch for it.

Example:

```bash
# Give every member of org 1 500 RAG queries a day
curl -X PUT "$BASE_URL/api/v1/admin/orgs/1/profile/overrides/limits.rag_queries_per_day" \
  -H "X-API-KEY: $API_KEY" -H "Content-Type: application/json" -d '{"value": 500}'
```

## The `limits.*` keys

Days and months are calendar UTC. With the switch on and the key unset, nothing is refused; a token key refuses when this request would take the counter past the value, and a cost key refuses once spend has reached its value.

| Key | Limits | Counter it reads | Refusal |
|---|---|---|---|
| `limits.audio_daily_minutes` | Audio transcription minutes per day (HTTP, WebSocket streaming, realtime, workers) | Resource ledger `user/minutes` (stored in seconds) | HTTP 402 `{"status":"quota_exceeded","message":"Transcription quota exceeded (daily minutes)"}` |
| `limits.transcription_minutes_per_month` | Transcription minutes per month | The same ledger rows summed over the month | HTTP 402 `{"status":"quota_exceeded","message":"Transcription quota exceeded (monthly minutes)"}` |
| `limits.audio_concurrent_jobs` | Audio jobs processing at once per user | The audio Jobs worker's count of `processing` rows for the owner | The job waits until the owner is under the cap. With `0` the job fails with `audio job quota is 0 for this user (limits.audio_concurrent_jobs)` |
| `limits.llm_tokens_per_month` | LLM tokens per calendar month (UTC), checked before every LLM call (see below) | `llm_usage_log` summed per user over the month | HTTP 402 `{"error":"limit_exceeded","category":"llm_tokens_month",...}` |
| `limits.llm_cost_per_day_usd` | LLM spend per UTC day, in US dollars | `llm_usage_log` costs summed per user over the day | HTTP 402 `{"error":"limit_exceeded","category":"llm_cost_day",...}` |
| `limits.llm_cost_per_month_usd` | LLM spend per calendar month, in US dollars | `llm_usage_log` costs summed per user over the month | HTTP 402 `{"error":"limit_exceeded","category":"llm_cost_month",...}` |
| `limits.rag_queries_per_day` | RAG queries per day (unified RAG, Text2SQL, MCP RAG) | Resource ledger `user/rag_queries` | HTTP 402 `{"error":"limit_exceeded","category":"rag_queries_day",...}` |
| `limits.storage_quota_mb` | Per-user storage in MB (media ingest, audio and video downloads, generated files) | `users.storage_used_mb` | HTTP 413 `Storage quota exceeded. Current: ...MB, New: ...MB, Quota: ...MB, Available: ...MB` |
| `limits.media_ingest_mb_per_day` | Uploaded media megabytes per day | Resource ledger `user/ingestion_bytes` (bytes) | HTTP 429 `Daily ingestion size budget exceeded.`, with `RateLimit-*`, `X-RateLimit-*` and `Retry-After` headers |
| `limits.workflows_runs_per_day` | Workflow runs per day, API-started and scheduled | Resource ledger `workflows_runs` | HTTP 429 `Daily quota exceeded` |
| `limits.evaluations_per_day` | Evaluations per day | Evaluations DB `daily_usage.total_evaluations` | HTTP 429 `Daily evaluation limit exceeded` |
| `limits.evaluation_tokens_per_day` | Evaluation tokens per day | Evaluations DB `daily_usage.total_tokens` | HTTP 429 `Daily evaluation token limit exceeded` |
| `limits.chatbooks_exports_per_day` | Chatbook exports per day | `export_jobs` rows since 00:00 UTC | HTTP 429 `Daily export limit (N) reached. Try again tomorrow.` |
| `limits.chatbooks_imports_per_day` | Chatbook imports per day | `import_jobs` rows since 00:00 UTC | HTTP 429 `Daily import limit (N) reached. Try again tomorrow.` |
| `limits.chatbooks_concurrent_jobs` | Chatbook export and import jobs active at once | Active chatbook job rows | HTTP 429 `Maximum concurrent jobs (N) reached...` |

What `limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd` count:

- The monthly count includes every LLM chat call recorded for the user, not just chat completions: RAG generation, character chat, summarization, evaluations and other calls made during a request. Users reach a token limit sooner than before.
- That includes calls a request triggers without the user asking for them directly, such as query rewriting and classification.
- Provider key validation and first-run chat verification are never counted.
- Embedding tokens do not count toward it.
- Recorded: calls made while serving a request; background jobs, including audio analysis and scheduled agent tasks; workflow runs; watchlist output enrichment; audio and realtime WebSocket sessions.
- Not recorded: MCP tool calls, the in-process claims rebuild, and system jobs with no user, such as the scheduled quality evaluation.
- The count depends on `LLM_USAGE_ENABLED` (default on). With it off, nothing is recorded and the limit never triggers.
- **Where a refusal happens.** These entry points refuse with a 402 before they start any LLM work, streaming requests included: `/chat/completions`; the Anthropic-compatible Messages API (`/messages` and `/v1/messages`); character chat replies; RAG search, batch, batch resume and streaming, when the request generates an answer; advanced search (`/rag/advanced`) when it generates an answer, which is the default; ablation (`/rag/ablate`) with answers on; document generation from a chat; the flashcards and quizzes study assistants; `/evaluations/geval`, `/evaluations/rag` and `/evaluations/response-quality`; `/evaluations/batch` of type `geval`, `rag` or `response_quality`; evaluation runs (`POST /evaluations/{eval_id}/runs`); the `/media/process-*` endpoints for audio, video, documents, PDFs, ebooks and email, and `/media/add`, when analysis is on and a provider is named (`perform_analysis` defaults to true, but without a provider no analysis runs). Simple search (`/rag/simple`) makes no LLM call.
- **Everything else is checked at the call.** Background jobs (`/media/ingest-jobs`, evaluation recipe runs), workflow runs, WebSocket sessions (audio, realtime) and every HTTP endpoint not listed above check each LLM call just before the provider is called, and a call for a user over a limit is refused without reaching the provider. What the caller gets back depends on the caller:
  - **An HTTP endpoint** that passes errors through, as most do, returns the same 402: `/translate`, `/media/ingest-web-content` with analysis, `/media/process-web-scraping` with summarization, `/media/{id}/insights` and the manuscript analysis endpoints among them. The web scraping endpoints discard that request's results. An endpoint that catches errors broadly reports its own error, usually a 500.
  - **A workflow:** an `llm` step fails, and so does the run, with `error` set to `402: {'error': 'limit_exceeded', ...}`; the step is retried once and refused again without a provider call. Other steps that make LLM calls (content generation, NLP, bibliography, image, summarize, translate) report the refusal in their outputs, and the run goes on.
  - **A job** whose worker reads the error's `retryable` flag (WorkerSDK workers and most others) fails at once with the error code `llm_quota_exceeded` and is not retried. A media ingest job marks the item `Error` (`Processing error: LLMQuotaExceededError: ...`); an audio or video batch marks every item in the batch. A media item that reuses an earlier transcript gets no summary.
  - **A WebSocket:** `/audio/chat/stream` sends `{"type":"error","code":"provider_unavailable",...}` for that turn and stays open. The realtime sockets send `{"type":"error","error":{"type":"server_error","code":"internal_error",...}}` and `response.done` with status `failed`, and stay open.
  - **A RAG search with generation off** still returns its results. The optional LLM stages it asked for (query classification, HyDE, gap analysis, LLM reranking and others) are skipped, and their entries in the response metadata say `"failure_code": "provider_unavailable"`.
  - **Shared machinery is not affected.** A refusal never opens the evaluations circuit breaker, is never scored 0.0 by an evaluator, and never marks a chat provider unhealthy, including the auto-router's call on `/chat/completions`, which runs before that endpoint's own check.
  - **Degraded, not refused:** claims verification falls back to heuristic verdicts; Prompt Studio test runs and benchmarks record error outputs or 0.0 scores; character memory extraction finds nothing. TASK-13638 tracks them.
  - **MCP tools** are neither recorded nor checked: they carry no user.
- **Not recorded today:** Messages API calls to Anthropic or llama.cpp take a native path that never reaches a provider adapter, so they write no usage row. The boundary check still refuses them. TASK-13610 covers recording them.
- **How the cost keys compare.** A cost key refuses once today's or this month's spend has reached the limit. The token key refuses when this request's estimated tokens would take the month past it; outside `/chat/completions` the estimate is one token.
- **An admitted request runs to completion,** so a user can pass a limit by one request's worth of usage. A media batch or an evaluation batch can be large.
- **Local providers cost nothing by default:** llama.cpp, kobold, ooba, tabbyapi, vllm, local-llm, ollama, aphrodite and mlx. To give one a price, edit its `default` entry in `tldw_Server_API/Config_Files/model_pricing.json`, or set `PRICING_OVERRIDES` (a JSON string in the environment), which overrides that file. A custom OpenAI-compatible endpoint is priced at the unknown-model rate ($0.01 per 1K prompt tokens, $0.03 per 1K completion tokens) unless you give it a price. A virtual-key USD budget no longer counts local calls.
- **Two settings defeat these limits, and the server warns at startup when quotas are on:** `LLM_USAGE_ENABLED` off (nothing is recorded, so nothing is refused), and `LLM_USAGE_LOG_RETENTION_DAYS` under 31 (the monthly counts lose rows).
- **Failures let requests through.** If a limit cannot be read or a counter fails, the request or call is admitted and a warning is logged. A check at the call that gets no answer within two seconds lets the call through, with a warning. A synchronous LLM call made on the server's event loop thread cannot be checked (waiting there would stall the server); it goes through, recorded, and the first one logs `A synchronous LLM call on the event loop's thread is not quota-checked`. A stream started there is checked when its first item is read.
- **Run workers inside the server.** A job worker started as its own process (`python -m ...`) has no accounting loop, so its synchronous LLM calls are neither checked nor recorded; the first one logs a warning. Workers run inside the server process by default.
- **What the check costs.** With quotas off, nothing. With quotas on and no LLM limit for the user, nothing once the user's limits are cached (for a minute). With a limit set, the counters are read for each call that a request boundary did not already admit, so a job that makes hundreds of calls makes hundreds of small indexed reads. While the user's limits cannot be read (the AuthNZ database is down), every call tries again and logs a warning.
- **The profile reports it.** `GET /api/v1/users/me/profile?sections=quotas` has an `llm` block with tokens used this month, spend today and this month, and each limit that is set.

Two other `limits.*` entries in the catalog are not part of this scheme:

- `limits.evaluations_per_minute` is listed in `user_profile_catalog.yaml` but nothing enforces it.
- The `limits.prompt_studio_*` keys are Prompt Studio's own limits and are unchanged.

`Docs/Operations/Rate_Limits_Troubleshooting.md` maps each refusal to the code that raises it.

## What is not a quota

These are unchanged by the switch and have no `limits.*` key:

- **Per-file caps**: `[Media-Processing] max_*_file_size_mb`, the 100 MB chatbook file cap, character-chat count caps.
- **Per-minute request rates**: the Resource Governor, and module limiters such as evaluations per minute.
- **Synchronous concurrency**: media ingest requests, audio streams and direct transcription have no per-user cap.
- **Other limits that were already off by default**: the `JOBS_QUOTA_*` limits, the embeddings and ingest per-tenant limits, chat token daily caps, and the Prompt Studio limits.
- **Admin storage pools** keep their shared-pool meaning (see above).
- **Billing-plan limits** run only on the hosted product. They need a billing repository wired into the subscription service (nothing in this repository wires one) and the switch on. With a repository wired and the switch off, startup logs a warning once that plan limits are not enforced.

`CHATBOOKS_DISABLE_QUOTAS` and `WORKFLOWS_DISABLE_QUOTAS` remain as extra per-module off switches; see the Usage Quotas section of `Docs/Operations/Env_Vars.md`.

## Upgrade notes

- **Self-hosters have nothing to do.** Quotas you never configured stop applying.
- **A hosted deployment** turns quotas on with `USAGE_QUOTAS_ENABLED=true` (`LIMIT_ENFORCEMENT_ENABLED=true` also works, with a warning) and wires its billing repository.
- **Storage copy.** An AuthNZ migration copies each user's existing `users.storage_quota_mb` into a user-level `limits.storage_quota_mb` value and never replaces a value that already exists. It skips values equal to 5120 and to the `DEFAULT_STORAGE_QUOTA_MB` configured when the migration runs, because those cannot be told apart from "never set". Those users become unlimited until a value is set. If the default was changed over time, users still on an older default keep it as a real per-user value.
- **The column stays.** `users.storage_quota_mb` is kept (`NOT NULL DEFAULT 5120`) but is no longer read for enforcement.
- **Evaluations.** `limits.evaluations_per_day` values set before this change were never enforced; with the switch on, they become enforced daily caps. Evaluation tiers no longer drive daily caps.
- **Tiers are not carried over.** Audio, evaluations and chatbooks tier assignments are dropped.
- **Stored RG policies.** Operators on `RG_POLICY_STORE=db` whose stored `evals.*` policies carry a `daily_cap` keep that cap until they remove it from the stored policy.

## Where users see their limits

`null` means unlimited in all of these.

| Where | What it reports |
|---|---|
| `GET /api/v1/users/storage` | `storage_quota_mb`, used MB, available and percentage |
| `GET /api/v1/audio/stream/limits` | Daily and monthly minute limits, `used_today_minutes`, `used_month_minutes`, `remaining_minutes` and `remaining_month_minutes` (used figures come from the ledger). `active_streams` is `null` because per-user stream concurrency is not tracked |
| `GET /api/v1/evaluations/rate-limits` | The resolved `limits.evaluations_per_day` and `limits.evaluation_tokens_per_day`, with usage and remaining. The daily `X-RateLimit-Daily-*` headers on evaluation responses are omitted when the cap is unlimited |
| `GET /api/v1/audio/jobs/admin/owner/{id}/processing` | An owner's processing count and concurrent-jobs limit (admin) |
| Profile `quotas` section and `limits.*` config | Storage, audio (daily and monthly), evaluations and Prompt Studio quotas, and, when set, the `limits.*` values (unset keys are omitted from `effective_config`) |

A storage quota of `0` shows as full, not as unlimited.

## Deprecated settings and APIs

These no longer set any limit:

- `AUDIO_TIER_LIMITS_JSON` and `[Audio-Quota] {tier}_*`: setting either logs a warning once and changes nothing. Use `limits.audio_daily_minutes`, `limits.transcription_minutes_per_month` and `limits.audio_concurrent_jobs`.
- The audio tier admin API, `GET`/`PUT /api/v1/audio/jobs/admin/tiers/{user_id}`: marked deprecated in the OpenAPI schema; it no longer affects limits.
- `DEFAULT_STORAGE_QUOTA_MB`: setting it logs a deprecation warning and sets no one's quota. Only the storage migration reads it, to skip the old default. Use `limits.storage_quota_mb`.
- `LIMIT_ENFORCEMENT_ENABLED`: the legacy spelling of the master switch; use `USAGE_QUOTAS_ENABLED`.
