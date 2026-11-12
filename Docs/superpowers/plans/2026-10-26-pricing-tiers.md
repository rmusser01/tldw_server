# Price the provider tiers and caches the catalog ignores (TASK-13936) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Review folded in (r2, 2026-10-26).** The independent review (`review_plan_13936.md`: 0 Blocker, 3 Important, 12 Minor; its dry run on 39093fc827 matched every `Expected:` line, and every rate matched the live vendor pages) is folded in as follows. The vendor facts below were re-fetched with WebFetch on 2026-10-26.
> - **I1 (fixed):** OpenAI's Fast-mode guide (`developers.openai.com/api/docs/guides/priority-processing.md`) says "For GPT-5.6 and earlier models, the response returns `priority` whether the request specifies `priority` or `fast`", and a ramp-downgraded Fast call reports `default`. Keying Fast as `priority` is now sourced. Follow-up F3 is dropped (ids are now `<N1>`, `<N2>`). The "not verified" list, the close-out note, Review Focus 1 and both docs say so. One correction to the review: the guide does not say that later models report `fast`, so the TASK-13915 note says to check the guide when gpt-6* are catalogued.
> - **I2 (fixed):** the four Gemini 2.5 Flash and Flash-Lite -preview entries are priced exactly as their GA model, so they get its `cache_read`. `GEMINI_PREVIEW_ALIASES` in the test module drives both the script and a new parametrised test (Task 3). The three 2.0 Flash entries are excluded because Google publishes no cache rate for them. The close-out note (Task 6 Step 3), the PR body and the self-review state that scope for AC #1.
> - **I3 (fixed):** F2 item (1) now states the under-billing as a fact. Sources: Google's thinking guide ("response pricing is the sum of output tokens and thinking tokens") and the repo fixture's `totalTokenCount` arithmetic. F2 is labelled `high` (backlog-py has no priority flag; `high` is the label the backlog uses). `Usage_Quotas.md` says thinking tokens are not costed yet. It names no task id, because Task 5 commits the doc before Task 6 files the follow-ups. The API reference also confirms that `promptTokenCount` includes the cached tokens.
> - **M1 (declined):** the gpt-5.5-pro model page (fetched 2026-10-26) has no long-context sentence. The gpt-5.5 page's "2x input and 1.5x output ... for standard, batch, and flex" names GPT-5.5 only, and the pricing page shows "-". No page gives a gpt-5.5-pro Batch or Flex price above 272K, so Decision 5 keeps the standard long-context rate (a conservative over-charge) and now says why.
> - **M2 (fixed):** every commit step runs `git add` first, then `git diff --cached --check`.
> - **M3 (fixed):** Task 1 Step 9 lists its nine files.
> - **M4 (fixed):** router calls are labelled as real token counts without a usage block, and `Usage_Quotas.md` says they cost the standard rate.
> - **M5 (fixed):** every cited `file:line` was re-checked on origin/dev `2c60650739`. Only `chat_service.py` moved (+34): 6283-6320 became 6317-6354, and 6884 became 6918. F1's description names the function instead of a line.
> - **M6 (fixed):** OpenRouter BYOK is sourced from the usage-accounting page and the FAQ (5% fee) and moved out of "not verified". Decision 1 says both readings are sourced.
> - **M7 (fixed):** the docs no longer say OpenAI reports `batch`; the Batch guide's output example has no `service_tier`.
> - **M8 (fixed):** Task 3 deletes the `gemini-2.5-pro-high` pseudo-entry. It is not a Google model ID, nothing references it, and `gemini-2.5-pro`'s `long_prompt` now carries its price. The model-list tests pass without it.
> - **M9 (fixed):** `Usage_Quotas.md`'s override sentence gains `service_tiers`. The stale first OpenAI example in `Model_Pricing_Catalog.md` is deleted; the refreshed snippet replaces it.
> - **M10 (fixed):** Bandit runs through `uvx`, with output in the scratchpad. Never install into the shared venv.
> - **M11 (fixed):** (a) Decision 1's example is now a hypothetical uncatalogued model. (b) `checked >= 23`. (c) Task 1 Step 7 notes the script's gitignored side effects. (d) Task 4 corrects the normalizer's module docstring.
> - **M12 (fixed):** F2 lists the Google adapter's stream emitting no usage chunk, with its own AC.
> - **Re-measured** in a worktree detached at 39093fc827, with the revised plan applied from its own code blocks:
>   - Task 3 RED: `9 failed, 1 passed` → `13 failed, 1 passed`. GREEN: `10 passed` → `14 passed`.
>   - The Google script: `google: 6 rows` → `google: 10 rows`.
>   - The new file: `123 passed` → `127 passed`.
>   - The focused set: `1658 passed, 1 skipped` → `1662 passed, 1 skipped`.
>   - Unchanged: Task 1 (`openai: 51 rows`, `84 passed`), Task 2 (`26 passed`), Task 4 (`4 failed, 6 passed` → `10 passed`), the adapter suites (`770 passed`) and `tests/Docs` (`212 passed`).
>   - New: the model-list tests after Task 3, `593 passed, 1 skipped`.
>   - The wider set: `976 passed, 14 skipped`, unchanged. The shard and test-quality guards still pass, and `git diff --check` is clean.
>
> **Scoped re-review (r3, 2026-10-26).** Result: 0 Blocker, 0 Important, 4 Minor. It dry-ran Tasks 1-5 and Task 6's checks on origin/dev `2c60650739`, and every `Expected:` line matched. I1-I3 are confirmed fixed, the M1 decline is sound, and deleting `gemini-2.5-pro-high` is safe (no references). The minors:
> - **(1) Task 2 Step 4:** place the batch constant after the US-only comment block, not inside it.
> - **(2) The normalizer docstring:** drop the stale "measurement-only" wording when Task 4 edits it.
> - **(3) Not done:** Google's pricing page lists newer models and price changes from Dec 1, none of them catalogued. This is out of scope; add it to F2's description.
> - **(4) Task 1 Step 8:** now gives `110 passed`.
>
> Executors apply (1)-(3) as they reach those steps.

**Goal:** Cost an LLM call at the service tier its provider reports (OpenAI Batch, Flex and Fast; Anthropic Batch), Gemini cached input at Gemini's context-caching rate, and an OpenRouter call from the cost OpenRouter reports, so the LLM cost limits cover them. A tier the catalog has no rate for costs the standard rate.

**Architecture:** One pipeline, as in the two predecessor plans: a provider usage block goes through `normalize_llm_usage` (`core/Usage/llm_usage_normalizer.py`), then `log_llm_usage`, then `compute_costs` (`core/Usage/usage_tracker.py`), which prices it through `PricingCatalog.get_rate_details` (`core/Usage/pricing_catalog.py`) over `Config_Files/model_pricing.json`. `compute_costs` has one caller, `log_llm_usage`. The changes:
- The normalizer reads `service_tier` for every provider and OpenRouter's `usage.cost`.
- A new `usage_with_service_tier(response)` folds OpenAI's top-level `service_tier` into a copy of the usage block. The three recorders that take `response["usage"]` use it.
- A catalog entry may carry `service_tiers`: per-model rate sets keyed by the reported tier. A script fills them from OpenAI's Batch, Flex and Fast tables.
- Anthropic `batch` is a code multiplier (0.5), like fast mode and US-only.
- The Google adapter keeps `cachedContentTokenCount`. The Gemini entries get `cache_read`, and the two Pro models a 200,000-token `long_prompt`.
- OpenRouter's reported cost replaces the catalog cost.

There is no schema or migration change: the tier is kept in `raw_usage_metadata_json`, as `speed` and `inference_geo` already are.

**Tech Stack:** Python, pytest, loguru, JSON catalog.

**Spec:** `backlog/tasks/task-13936 - Price-the-provider-tiers-and-caches-the-catalog-ignores-Batch-OpenAI-Flex-and-Fast-Gemini-cached-content-OpenRouter.md` (description and ACs #1-#6). Predecessors: `Docs/superpowers/plans/2026-10-23-pricing.md` (this is its follow-up 5) and `Docs/superpowers/plans/2026-10-20-anthropic-cache-token-pricing.md`. Anthropic fast mode is out of scope (TASK-13885 priced it).

**Measured on:** origin/dev `39093fc8273aa9c18aae88ea6c3c5a9cb2276d82`, a detached worktree, 2026-10-26. The prototype is `<scratchpad>/proto_13936_full.patch` plus `<scratchpad>/test_pricing_service_tiers.py`, both in the planning session's scratchpad. The worktree was reverted afterwards. Every count below was measured with the test text in this plan; the r2 additions were measured the same way, with the revised plan applied from its own code blocks. Cited `file:line` references are for origin/dev `2c60650739` (checked 2026-10-26). Between the two commits `chat_service.py` is the only touched or cited file that changed: its cited lines moved by +34, and the plan's text anchors in it are still unique.

---

## What was verified (the paths usage takes to the cost)

| Path | Usage block it records | Where `service_tier` is | Before | After |
|---|---|---|---|---|
| Adapter `chat`/`achat` for a recorded user (RAG, workflows, every non-`/chat/completions` caller): `usage_accounting._record_response` | `response["usage"]` | OpenAI: top level of the response. Anthropic adapter: dropped (`anthropic_adapter.py:385` keeps only `cache_creation`, `speed`, `inference_geo`) | tier lost | OpenAI tier folded in |
| Adapter `stream`/`astream`: `usage_accounting._StreamUsage.feed` | each SSE chunk's `usage`; Anthropic `message_start.message.usage` merged with `message_delta.usage` | OpenAI: top level of each chunk | tier lost | folded in |
| Native Messages API, non-stream: `record_direct_response` to `_record_response` | the Anthropic reply's `usage` | in `usage` (validator allows only `batch`, `priority`, `standard`; `messages.py:430-436`) | read by nobody | priced |
| Native Messages API, stream: `accounted_direct_stream` to `_StreamUsage` | `message_start` usage, delta merged over it (`_MESSAGES_NATIVE_DELTA_USAGE_FIELDS` has no `service_tier`, so the start's survives) | in `usage` | read by nobody | priced |
| `/chat/completions` non-stream: `chat_service._record_dict_response_usage` (the call `records_itself`, so the provider-level recorder skips it) | `response["usage"]` | OpenAI: top level | tier lost | folded in |
| `/chat/completions` stream: `execute_streaming_call`'s `save_callback` after the stream ends (`chat_service.py:6317-6354`) | none: `stream_estimate`, no `usage_metadata` | n/a | estimate | unchanged (follow-up F1) |
| Router calls: `routing/accounting.log_model_router_usage` (`routing/accounting.py:51-89`), called by `chat_route_resolver.log_router_usage` and `character_chat_sessions._log_router_usage` | the provider's real token counts, no usage block | not passed | no tier, cache rate or reported cost | unchanged (a 64-token call) |
| Claims engine, embeddings, notes graph | estimates or their own pricing | n/a | n/a | unchanged |

Other facts the plan relies on:
- **OpenAI.** The adapter returns the raw response JSON, so the top-level `service_tier` reaches the recorders, and `stream` yields the raw SSE lines (`openai_adapter.py:296-375`). The adapter never sets `service_tier`, but `merge_extra_body` lets a caller set it (`payload_utils.py:371`). Nothing sends `stream_options.include_usage` (`git grep` finds none), so an OpenAI stream carries usage only when the caller adds it in `extra_body`.
- **Gemini.** `GoogleAdapter._normalize_to_openai_shape` (`google_adapter.py:441-446`) keeps only prompt, completion and total, so `cachedContentTokenCount` is lost before the normalizer's Google branch can read it. It also sets `completion_tokens` from `candidatesTokenCount` alone and drops `thoughtsTokenCount`, so thinking tokens are not costed (follow-up F2; the repo fixture `test_chat_service_content.py:485-491` has `totalTokenCount` 115 = 80 prompt + 30 candidates + 5 thoughts). The Gemini stream (`GoogleAdapter.stream`, `google_adapter.py:598-678`) emits no usage. `cache_intents._apply_google` only references an existing `cachedContents/...` name; no server code creates a cache.
- **OpenRouter.** The adapter returns the raw JSON. `DEFAULT_PRICING["openrouter"]` is empty, and `model_pricing.json` has 31 OpenRouter entries. Any other model falls to a substring match or the unknown-model rate ($0.01 / $0.03 per 1K).
- **What the limits read.** Quotas sum `total_cost_usd` (`usage_repo.py:147-328`). `raw_usage_metadata_json` keeps `service_tier`: the sanitizer does not treat the key as prompt-like, and a Task 1 test asserts it.
- **CI guards.**
  - No touched source file is listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` or the other `Helper_Scripts/ci/*baseline.txt` files.
  - `test_quality_baseline.txt` lists `test_chat_service_content.py::tautology_suspect=1`; `test_quality_triage.py --enforce` exits 0 after the change.
  - `check_shard_coverage.py` reports `new_uncovered=0`.
  - Neither touched doc is mirrored under `Docs/Published`.
  - No schema changes, so there is no OpenAPI drift.

## Vendor prices (fetched 2026-10-26 with WebFetch; every figure below comes from these pages)

Prices are USD per million tokens (MTok); catalog rates are per 1K tokens, so divide by 1000.

### Anthropic: https://platform.claude.com/docs/en/about-claude/pricing.md

WebFetch returned the raw Markdown (52 KB). Quoted from it:
- "The Batch API allows asynchronous processing of large volumes of requests with a 50 percent discount on both input and output tokens."
- Prompt caching multipliers: "These multipliers stack with other pricing modifiers, including the Batch API discount and data residency."
- "Fast mode is not available with the Batch API."
- Long context: "Prompt caching and batch processing discounts apply at standard rates across the full context window."
- Haiku 5.5: "The model pricing and batch processing pricing sections list both sets of prices."

The batch table, per MTok (input / output). Every row is exactly half its list row, which is why Anthropic Batch is one multiplier:

| Model | Batch input | Batch output |
|---|---|---|
| Fable 5.1, Mythos 5.1, Fable 5, Mythos 5 | 5 | 25 |
| Opus 5.5 | 2 | 10 |
| Opus 5, 4.8, 4.7, 4.6, 4.5 | 2.50 | 12.50 |
| Opus 4.1, Opus 4 | 7.50 | 37.50 |
| Sonnet 5.5, Sonnet 5 | 1 | 5 |
| Sonnet 4.6, 4.5, 4 | 1.50 | 7.50 |
| Haiku 5.5, prompts up to 100,000 tokens | 0.05 | 0.25 |
| Haiku 5.5, prompts over 100,000 tokens | 0.25 | 1.25 |
| Haiku 4.5 | 0.50 | 2.50 |
| Haiku 3.5 | 0.40 | 2 |

The pricing page gives no Priority Tier price. https://platform.claude.com/docs/en/api/service-tiers.md (the service-tier usage doc; fetched to learn how Priority is billed) says:
- "Priority Tier capacity commitments are no longer available for purchase."
- "A Priority Tier commitment consists of: A number of input tokens per minute, A number of output tokens per minute, A commitment duration ..."
- "These burndown rates reflect the relative pricing of each token type."

The response field: "The response `usage` object also includes the service tier assigned to the request" (example `"service_tier": "priority"`). So Priority is paid for by commitment, and a call reported as `priority` costs the standard rates.

### OpenAI: https://developers.openai.com/api/docs/pricing.md

WebFetch summarises this page. The Batch, Flex and Fast sections were fetched as raw section text ("character for character"), and the Fast section twice with matching output. A structured cross-check fetch disagreed on Flex: it reported most models "absent" from Flex. The raw Flex section was then fetched again and matched the first raw fetch. **The executor re-fetches the three sections and checks every figure before Task 1 Step 7** (the 2026-10-22 review read the raw Markdown with curl, at `https://developers.openai.com/api/docs/pricing.md`).

Quoted:
- "Priority processing was renamed Fast mode on July 30, 2026."
- "Regional processing (data residency) endpoints are charged a 10% uplift for models released on or after March 5, 2026." This is out of scope and already in TASK-13915.
- The page has an "Ultrafast" table for gpt-6-astra and gpt-6.1-sol only. Neither model is catalogued (TASK-13915).

Short-context rows, per MTok, as input / output / cached input. "-" is the page's own dash; a blank cell means the model has no row in that table.

| Model | Batch | Flex | Fast |
|---|---|---|---|
| gpt-5.5 | 2.50 / 15.00 / 0.25 | 2.50 / 15.00 / 0.25 | 12.50 / 75.00 / 1.25 |
| gpt-5.5-pro | 15.00 / 90.00 / - | 15.00 / 90.00 / - | |
| gpt-5.4 | 1.25 / 7.50 / 0.13 | 1.25 / 7.50 / 0.13 | 5.00 / 30.00 / 0.50 |
| gpt-5.4-pro | 15.00 / 90.00 / - | 15.00 / 90.00 / - | |
| gpt-5.4-mini | 0.375 / 2.25 / 0.0375 | 0.375 / 2.25 / 0.0375 | 1.50 / 9.00 / 0.15 |
| gpt-5.4-nano | 0.10 / 0.625 / 0.01 | 0.10 / 0.625 / 0.01 | |
| gpt-5.2 | 0.875 / 7.00 / 0.0875 | 0.875 / 7.00 / 0.0875 | 3.50 / 28.00 / 0.35 |
| gpt-5.2-pro | 10.50 / 84.00 / - | | |
| gpt-5.1, gpt-5 | 0.625 / 5.00 / 0.0625 | 0.625 / 5.00 / 0.0625 | 2.50 / 20.00 / 0.25 |
| gpt-5-mini | 0.125 / 1.00 / 0.0125 | 0.125 / 1.00 / 0.0125 | 0.45 / 3.60 / 0.045 |
| gpt-5-nano | 0.025 / 0.20 / 0.0025 | 0.025 / 0.20 / 0.0025 | |
| gpt-5-pro | 7.50 / 60.00 / - | | |
| gpt-4.1 | 1.00 / 4.00 / - | | 3.50 / 14.00 / 0.875 |
| gpt-4.1-mini | 0.20 / 0.80 / - | | 0.70 / 2.80 / 0.175 |
| gpt-4.1-nano | 0.05 / 0.20 / - | | 0.20 / 0.80 / 0.05 |
| gpt-4o | 1.25 / 5.00 / - | | 4.25 / 17.00 / 2.125 |
| gpt-4o-2024-05-13 | 2.50 / 7.50 / - | | 8.75 / 26.25 / - |
| gpt-4o-mini | 0.075 / 0.30 / - | | 0.25 / 1.00 / 0.125 |
| o1 | 7.50 / 30.00 / - | | |
| o1-pro | 75.00 / 300.00 / - | | |
| o3-pro | 10.00 / 40.00 / - | | |
| o3 | 1.00 / 4.00 / - | 1.00 / 4.00 / 0.25 | 3.50 / 14.00 / 0.875 |
| o4-mini | 0.55 / 2.20 / - | 0.55 / 2.20 / 0.138 | 2.00 / 8.00 / 0.50 |
| o3-mini | 0.55 / 2.20 / - | | |
| gpt-4-turbo-2024-04-09 | 5.00 / 15.00 / - | | |
| gpt-4-0613 | 15.00 / 30.00 / - | | |
| gpt-3.5-turbo-0125 | 0.25 / 0.75 / - | | |

Long-context columns (prompts over 272,000 tokens), input / output / cached:

| Model | Batch | Flex | Fast |
|---|---|---|---|
| gpt-5.5 | 5.00 / 22.50 / 0.50 | 5.00 / 22.50 / 0.50 | - / - / - |
| gpt-5.5-pro | - / - / - | - / - / - | |
| gpt-5.4 | 2.50 / 11.25 / 0.25 | 2.50 / 11.25 / 0.25 | - / - / - |
| gpt-5.4-pro | 30.00 / 135.00 / - | 30.00 / 135.00 / - | |

The standard rows the cross-check fetch returned match the catalog's current `openai` rows for every model above.

What the response reports, from OpenAI's tier usage guides (fetched 2026-10-26):
- https://developers.openai.com/api/docs/guides/priority-processing.md (Fast mode):
  - "The `service_tier` field in the Responses or Chat Completions response object identifies the tier used to process the request."
  - "For GPT-5.6 and earlier models, the response returns `priority` whether the request specifies `priority` or `fast`." Every catalogued OpenAI model is GPT-5.6 or earlier, so Fast is keyed `priority`. The guide does not say what a later model (gpt-6*, uncatalogued) reports.
  - "If your traffic ramps too fast, the system may downgrade some Fast mode requests to standard speeds and charge standard rates. When this happens, the response contains `service_tier: "default"`." So `default` costs standard (pinned by the costed test).
- https://developers.openai.com/api/docs/guides/batch.md: the output-file example's response body (`"usage": {"prompt_tokens": 22, "completion_tokens": 2, "total_tokens": 24}`) has no `service_tier`, and the page never mentions the field. No documented OpenAI response reports `batch`; the `batch` rates apply only to a usage block that does (AC #5 still asks for them).
- https://developers.openai.com/api/docs/models/gpt-5.5-pro.md has no long-context sentence. The gpt-5.5 page's "For GPT-5.5, prompts with >272K input tokens are priced at 2x input and 1.5x output for the full session for standard, batch, and flex" names GPT-5.5 only (Decision 5).

### Google Gemini: https://ai.google.dev/gemini-api/docs/pricing

Standard tier, paid, text/image/video, per MTok, as input / output / context caching:

| Model | Prompts up to 200k tokens | Prompts over 200k tokens |
|---|---|---|
| gemini-3.1-pro-preview | 2.00 / 12.00 / 0.20 | 4.00 / 18.00 / 0.40 |
| gemini-3.1-flash-lite | 0.25 / 1.50 / 0.025 | |
| gemini-3-flash-preview | 0.50 / 3.00 / 0.05 | |
| gemini-2.5-pro | 1.25 / 10.00 / 0.125 | 2.50 / 15.00 / 0.25 |
| gemini-2.5-flash | 0.30 / 2.50 / 0.03 | |
| gemini-2.5-flash-lite | 0.10 / 0.40 / 0.01 | |

Two fetches agreed. The input and output figures match the catalog's current `google` rows.
- **Storage.** The context-caching storage price is "$4.50 / 1,000,000 tokens per hour" (the Pro models) or "$1.00 / 1,000,000 tokens per hour" (the others).
- **Audio.** Audio input and cached audio have their own, higher prices; the catalog has text rates only.
- **Tiers.** The page has Batch, Flex and Priority tables. It "contains no explicit sentence describing how to select them or any mention of a service_tier parameter or field."
- **Thinking tokens.** The 2.5 output label is "Output price (including thinking tokens)".
- **Models the page does not list:** `gemini-2.0-flash`, `gemini-2.0-flash-001`, `gemini-2.0-flash-lite`, `gemini-2.5-flash-preview`, `gemini-2.5-flash-preview-09-2025`, `gemini-2.5-flash-lite-preview` and `gemini-2.5-flash-lite-preview-09-2025`.
  - The four 2.5 -preview entries are priced exactly as their GA model (`gemini-2.5-flash`, `gemini-2.5-flash-lite`) in the catalog, so they get its cache rate (Task 3, `GEMINI_PREVIEW_ALIASES`).
  - The three 2.0 entries get no cache rate: Google publishes no price for them, so none can be checked, and their cached tokens cost the input rate (an over-charge).

Token counts, from the Gemini API usage docs (fetched 2026-10-26):
- https://ai.google.dev/api/generate-content, `UsageMetadata`:
  - `promptTokenCount`: "When `cachedContent` is set, this is still the total effective prompt size meaning this includes the number of tokens in the cached content." So billable input is prompt minus cached, as the normalizer already assumes.
  - `candidatesTokenCount`: "Total number of tokens across all the generated response candidates." `thoughtsTokenCount`: "Output only. Number of tokens of thoughts for thinking models."
- https://ai.google.dev/gemini-api/docs/thinking: "When thinking is turned on, response pricing is the sum of output tokens and thinking tokens." Its table lists thinking as on by default for gemini-2.5-pro and gemini-2.5-flash. Thinking tokens are counted apart from the candidates, and the adapter drops them: a known under-billing, out of this task's scope (follow-up F2, labelled `high`).

### OpenRouter

- https://openrouter.ai/docs/use-cases/usage-accounting:
  - "The `usage: { include: true }` and `stream_options: { include_usage: true }` parameters are deprecated and have no effect. Full usage details are now always included automatically in every response."
  - "This information is included in the last SSE message for streaming responses, or in the complete response for non-streaming requests."
  - The example usage block: `"cost": 0.95, "cost_details": {"upstream_inference_cost": 19}`, with the sentences "the `upstream_inference_cost` field is only available for BYOK (Bring Your Own Key) requests. For all other requests it will be 0 or null."
  - `upstream_inference_cost` is described as "The actual cost charged by the upstream AI provider", and `cost` as "The total amount charged to your account".
- https://openrouter.ai/docs/api-reference/overview: `"cost"` is "Cost in credits", and `"is_byok"` is "Whether request used Bring Your Own Key". Both are in `ResponseUsage`, the `usage` object the normalizer reads.
- https://openrouter.ai/docs/faq (the page that says what a credit is):
  - "OpenRouter uses a credit system where the base currency is US dollars. All of the pricing on our site and API is denoted in dollars."
  - BYOK: "Usage above the allowance has a fee of 5% of what the same model and provider would normally cost on OpenRouter." "This fee is deducted from your OpenRouter credits."
- So with BYOK, `cost` is only OpenRouter's fee (the example's 0.95 is 5% of 19), and the provider's charge to the operator's key is `upstream_inference_cost`. The cost is their sum (Decision 1).

### What could not be verified (do not guess; each is handled as stated)

1. **OpenAI long-context prices** for gpt-5.5-pro Batch and Flex, and for gpt-5.5 and gpt-5.4 Fast. The pricing page shows "-", and the gpt-5.5-pro model page gives none. Decision 5 costs these at the standard long-context rate.
2. **Anthropic Priority Tier per-token price.** None is published, so a `priority` call is costed at the standard rates.
3. **The `service_tier` a model after GPT-5.6 reports for Fast.** The Fast-mode guide covers only GPT-5.6 and earlier, and no later model is catalogued. A later model's Fast call reporting `fast` costs the standard rate until TASK-13915 catalogues it (its note says to key those rates by what the guide then says).

The rest of what the first draft could not verify is now sourced above: the `service_tier` Fast reports (`priority`), Gemini's `promptTokenCount` including the cached tokens, thinking tokens counted apart from the candidates, and OpenRouter's BYOK fields.

---

## Global Constraints

- **Git and PR.** Branch from `origin/dev`; the PR targets `dev`. Never `--no-verify`, never force-push, never `git stash`. One PR (see "PR and commits").
- **TDD.** Fix test-first: write the test, run it, see it fail for the stated reason, then fix. The counts below are measured; record the real ones.
- **No network in tests.** Tests make no real network calls; stub transports (the tests here call adapters' pure shaping functions and `log_llm_usage` with a fake repository). PostgreSQL only through the repo's fixtures (`tldw_postgres_test` container); this plan needs none. Never FLUSHALL Redis.
- **Backlog.** Edit backlog tasks only with backlog-py: `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task ...`. Use `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit task files. Any new follow-up task gets an explicit `--id` allocated at execution under the id lock (Task 6 Step 4).
- **Python.** The worktree has no `.venv`. Use `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (written `$PY` below; shell variables do not survive between agent calls, so write it out in each command).
- **Rates.** Rates are per 1K tokens in the catalog. Compare floats with `pytest.approx`. JSON edits keep one entry per line in the existing `{ "k": v }` style, through the script in Task 1 Step 7. Do not hand-type the 51 OpenAI rows.
- **The usage record.** `NormalizedLLMUsage.to_dict()` is not changed. New dataclass fields go at the end, with defaults. `compute_costs` and `get_rate_details` gain only keyword-only parameters with defaults (callers: `log_llm_usage`, `get_rates`, tests).
- **Logging.** Do not log usage payloads or secrets. The new warning logs only the tier name, provider and model.
- **CI traps:**
  - `Helper_Scripts/ci/scope_predicate_baseline.txt`: none of the touched files is listed (checked). Re-check with `git grep -n "pricing_catalog.py\|usage_tracker.py\|llm_usage_normalizer.py\|usage_accounting.py\|google_adapter.py\|chat_service.py" Helper_Scripts/ci/scope_predicate_baseline.txt`. If any is listed and its lines shift, regenerate with `--write-baseline`.
  - This plan is committed in the PR at `Docs/superpowers/plans/2026-10-26-pricing-tiers.md`, so it must have no trailing whitespace, including inside embedded code. Before each commit, `git add` the commit's files, then run `git diff --cached --check` (it must print nothing). A plain `git diff --check` skips staged and untracked files, so it cannot see a new file such as this plan.
  - OpenAPI drift: no schema changes here, so none.
  - `Docs/Published`: `Usage_Quotas.md` and `Model_Pricing_Catalog.md` are not mirrored (checked against `git ls-files Docs/Published`). Editing any other doc that is mirrored needs `bash Helper_Scripts/refresh_docs_published.sh` and the `Docs/Published` diff in the same PR.
  - `Helper_Scripts/ci/test_quality_triage.py --enforce` and `Helper_Scripts/ci/check_shard_coverage.py` must stay green (both green in the prototype).
- **Close-out.** The last task closes TASK-13936 with backlog-py (notes, ACs, DoD, final summary) and files the follow-ups.

## Decisions

Recommendation first. The plan implements each recommendation; each item gives the cost if it is wrong and the alternatives.

1. **OpenRouter: cost a call from the `usage.cost` OpenRouter reports (recommended).**
   - With BYOK (`is_byok: true`), add `cost_details.upstream_inference_cost`, since `cost` is then only OpenRouter's fee.
   - When `cost` is missing or not a finite non-negative number, use the catalogued rate, then the unknown-model rate.
   - **No request flag is needed, and nothing changes on the wire.** OpenRouter always returns the usage block with `cost`, and the old `usage: {include: true}` flag is "deprecated and [has] no effect".
   - The cost is split between `prompt_cost_usd` and `completion_cost_usd` in the catalog's proportions, and the row is `estimated=False`.
   - Both readings are sourced: credits are dollars (FAQ), and with BYOK `cost` is OpenRouter's 5% fee while `upstream_inference_cost` is the provider's charge (usage-accounting page and FAQ).
   - *If wrong* (OpenRouter changes either): OpenRouter rows are off by that factor; BYOK rows would be over-billed by the upstream charge. Switching back is one line: drop the `reported_cost_usd` assignment in the normalizer.
   - *Alternatives:*
     - (b) Catalogued rates only: add OpenRouter models to `model_pricing.json`. There are 31 today and hundreds on OpenRouter, so every uncatalogued model stays at $0.01 / $0.03 per 1K. An uncatalogued model priced like `deepseek/deepseek-v4-flash` ($0.14 per MTok input) would be over-billed about 70x on input; a premium one far under. Provider routing can also change the price per call.
     - (c) Document the gap only. AC #2 allows it, but it leaves the unknown-model rate.
2. **No `service_tier` DB column (recommended).**
   - `raw_usage_metadata_json` already keeps the reported tier: inside the block for Anthropic, and folded in by `usage_with_service_tier` for OpenAI (a Task 1 test asserts it).
   - The cost columns are fixed at write time, already at the tier's price.
   - *If wrong* (per-tier reports become a requirement): until a column exists, query the JSON (`json_extract(raw_usage_metadata_json, '$.service_tier')` on SQLite; a `::jsonb ->> 'service_tier'` cast on PostgreSQL). A column later needs a SQLite migration (`AuthNZ/migrations.py`), a PostgreSQL one (`pg_migrations_extra.py`), the repository insert and both schema tests (`tests/AuthNZ_SQLite/test_authnz_llm_usage_log_router_columns_sqlite.py` and its PostgreSQL twin). It can be backfilled from the JSON.
   - *Alternative:* add the column now, with those migrations, for no current reader.
3. **Per-model `service_tiers` rate sets in the catalog for OpenAI; Anthropic Batch as a code constant (recommended).**
   - OpenAI's Fast is not a flat ratio: 2.5x on gpt-5.5, 2x on gpt-5.4, 1.8x on gpt-5-mini, 1.75x on gpt-4.1, 1.7x on gpt-4o, 1.67x on gpt-4o-mini.
   - OpenAI Batch shows "-" for cached input on the gpt-4.x and o-series rows, so cached tokens there cost the Batch input rate.
   - OpenAI Batch's gpt-5.4 cached price is $0.13, not $0.125.
   - Anthropic's batch table is exactly half of every list row and stacks with every modifier. That is the TASK-13885 precedent for code constants (fast 2x, US-only 1.1x).
   - *If wrong* (Anthropic changes one model's batch ratio): add `service_tiers.batch` support for Anthropic and drop the constant. That is about 5 lines, and `Model_Pricing_Catalog.md` already warns that both together would halve twice.
   - *Alternatives:*
     - (b) Flat per-tier multipliers (batch and flex 0.5, Fast 2.0): Fast is mispriced from 20% under (gpt-5.5) to 20% over (gpt-4o-mini), and cached Batch tokens are under-billed on gpt-4.x and o-series.
     - (c) Per-model tables for Anthropic too: 19 halved rows to keep in sync, with no change in behaviour.
4. **Gemini context-cache storage is not costed; this is documented (recommended).**
   - An explicit cache is created outside the server: `cache_intents` only passes an existing `cachedContents/...` name.
   - It is billed per hour of its TTL ($1.00 to $4.50 per MTok-hour), and is tied to neither a call nor a user.
   - Implicit caching, the common case, has no storage charge, so the per-call cache-read rate covers it.
   - *If wrong:* an operator's explicit-cache storage spend stays outside per-user limits.
   - *Alternatives:*
     - (b) Charge an estimated storage share per referencing call. That needs the cache's size and TTL: a `cachedContents` GET on the billing path, and an arbitrary split of a shared resource.
     - (c) A follow-up for an operator-level overhead line. File it only if the owner wants it.
5. **A prompt over 272,000 tokens on a tier whose table shows no long-context price costs the standard long-context rate (recommended; AC #6's rule applied per prompt length).**
   - *If wrong:*
     - gpt-5.5 and gpt-5.4 Fast above 272K cost $10 / $45 and $5 / $22.50, below their short-context Fast prices ($12.50 / $75, $5 / $30).
     - gpt-5.5-pro Batch and Flex above 272K cost $60 / $270. The gpt-5.5 model page says "For GPT-5.5, prompts with >272K input tokens are priced at 2x input and 1.5x output for the full session for standard, batch, and flex", which would give $30 / $135 if it covered the Pro model, as gpt-5.4-pro's own Batch and Flex rows ($30 / $135) do. It names GPT-5.5 only, the gpt-5.5-pro page (fetched 2026-10-26) has no long-context sentence, and the pricing page shows "-". With no published price, the plan keeps the standard long rate, which over-charges if the 2x / 1.5x rule applies: the conservative direction.
     - Changing this is one condition in `_entry_rate_details`, or one `OPENAI_LONG_TIER_PRICES` row once OpenAI publishes the price.
   - *Alternatives:* the tier's short-context rates at any length (gpt-5.5-pro Batch above 272K at $15 / $90, half the likely price), or the larger of the tier's short rate and the standard long rate.

Settled by the code, the ACs and the vendor pages (not decisions):
- **Anthropic `priority` costs the standard rates.** No per-token price is published; it is paid by commitment.
- **Fast is keyed `priority`**, per AC #5 and OpenAI's Fast-mode guide: "For GPT-5.6 and earlier models, the response returns `priority` whether the request specifies `priority` or `fast`." A ramp-downgraded Fast call reports `default` and is billed standard.
- **Snapshots.** A dated OpenAI snapshot whose standard `prompt` equals its alias's gets the alias's tiers. The predecessor plan verified that those snapshots are priced like their aliases (model pages, 2026-10-22). `gpt-4o-2024-05-13` has its own rows.
- **Gemini 2.5 -preview entries.** The same rule for Gemini: `gemini-2.5-flash-preview`, `-preview-09-2025`, `gemini-2.5-flash-lite-preview` and `-preview-09-2025`, whose rates equal their GA model's, get its `cache_read`. AC #1 is then met for every catalogued Gemini text model with a published cache rate. The three 2.0 Flash entries are excluded because Google publishes no cache rate for them.
- **Gemini Pro long tiers.** `gemini-2.5-pro` and `gemini-3.1-pro-preview` get the 200,000-token `long_prompt` from the same page, so a long prompt's cache reads cost the page's long rate. This also ends an existing under-billing of their input and output above 200K. The `gemini-2.5-pro-high` pseudo-entry (`model_pricing.json:172`) only held the >200K price, is not a Google model ID, and nothing references it (`git grep`), yet model pickers list it. Task 3 deletes it.
- **The Anthropic adapter does not carry `service_tier`.** The Messages API reports `batch` only on Message Batches results, which no server code fetches, and `priority` costs standard, so it would change no cost.
- **The tier comes only from the response**, never from the request.

## Review Focus

Inputs and conditions the spec implies but the tests do not fully cover, most likely first:

1. **A Fast call on a model after GPT-5.6.** The Fast-mode guide says GPT-5.6 and earlier report `priority`, so every catalogued model is covered. A later model is not catalogued (TASK-13915), so it costs the unknown or substring rate anyway; if it reports `fast`, the tier costs standard.
   - Pinned for the unknown case: `test_a_service_tier_the_catalog_has_no_rate_for_costs_the_standard_rate` includes `ultrafast`.
   - No `fast` alias is needed today. TASK-13915's note says to key the later models' Fast rates by what the guide says when they are catalogued.
2. **A streamed `/chat/completions` reply** is costed from an estimate at the standard rate (`chat_service.py:6317-6354`), so a Fast or Batch user who streams is costed at standard, as is a streamed OpenRouter or Gemini reply.
   - Documented in `Usage_Quotas.md` (Task 5); follow-up F1.
   - Not pinned by a test: it is today's behaviour, unchanged.
3. **A user-controlled base URL** can report a cheap tier or `cost: 0`. This is possible only when the operator puts that provider in `BYOK_ALLOWED_BASE_URL_PROVIDERS` (empty by default; `AuthNZ/settings.py:482`). It is the same trust the token counts already get.
   - Not pinned; the reviewer should confirm that the default stays empty.
4. **A long prompt on a tier with no long-context price** (Decision 5). Pinned by `test_a_long_prompt_on_a_tier_with_no_long_context_price_costs_the_standard_long_context_rate`.
5. **OpenRouter edge values.**
   - `cost: 0` (`:free` models) costs $0, which is correct.
   - A string, negative, NaN, infinite or boolean `cost` falls back to the catalog. Pinned by `test_an_openrouter_call_without_a_usable_cost_falls_back_to_the_catalog`.
   - A usage block with `cost` but zero prompt and completion tokens is recorded as an estimate without usage: `_record_response` drops a (0, 0) block, so the reported cost is ignored. Not pinned; unchanged behaviour.
6. **OpenRouter BYOK semantics** (sourced: usage-accounting page and FAQ). Pinned to the documented example: 0.95 + 19 = 19.95.
7. **Gemini.** Audio is costed at text rates. The three 2.0 Flash entries have no cache rate (their cached tokens cost input). Thinking tokens are not costed at all (the adapter drops `thoughtsTokenCount`): an under-billing on Gemini 2.5 Pro and Flash, where thinking is on by default, larger than most of what this plan fixes, and out of its scope. The stream emits no usage. All documented in `Usage_Quotas.md` or in F2; the cache rates are pinned per model, the -preview entries included.
8. **Anthropic `batch` with fast mode** (impossible per the page) stacks to 1.0x. On an unknown Anthropic model it halves the unknown-model rate. It applies only when the provider canonicalises to `anthropic`, so a llama.cpp Messages reply reporting `batch` costs its own (zero) rate. Pinned by `test_a_provider_without_a_batch_price_is_not_discounted` for a non-Anthropic provider.
9. **The no-length estimators** `get_rates` and `compute_token_cost` (ACP admin cost estimates) ignore tiers and price standard. They are estimators, not billing. Unchanged.
10. **An override of an OpenAI entry without `service_tiers`** drops the tiers, so tier calls cost standard. Documented in `Model_Pricing_Catalog.md` and `Usage_Quotas.md` (Task 5), like the existing `long_prompt` override trap.

## PR and commits

One PR, branch `fix/pricing-service-tiers-13936`, from `origin/dev`. Reason: Tasks 1, 2 and 4 all edit `compute_costs` and the normalizer, and Tasks 1 and 3 edit `model_pricing.json`. Separate PRs would conflict on every serial merge. Commits (each ends with the session's attribution trailer, `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`):

0. `docs(plan): add the TASK-13936 pricing tiers plan`
1. `fix(usage): cost OpenAI calls at the service tier the response reports (TASK-13936)`
2. `fix(usage): cost Anthropic Batch usage at half price (TASK-13936)`
3. `fix(usage): cost Gemini cached input at the context-caching rate (TASK-13936)`
4. `fix(usage): cost OpenRouter calls from the cost OpenRouter reports (TASK-13936)`
5. `docs(usage): say which tiers and caches are priced (TASK-13936)`
6. `docs(backlog): close TASK-13936 and file follow-ups`

## Files

- Modify: `tldw_Server_API/app/core/Usage/llm_usage_normalizer.py` (Tasks 1, 4)
- Modify: `tldw_Server_API/app/core/Usage/pricing_catalog.py` (Task 1)
- Modify: `tldw_Server_API/app/core/Usage/usage_tracker.py` (Tasks 1, 2, 4)
- Modify: `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` (Task 1)
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py` (Task 1, two lines)
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/google_adapter.py` (Task 3)
- Modify: `tldw_Server_API/Config_Files/model_pricing.json` (Task 1: 51 OpenAI rows via script; Task 3: 10 Gemini rows via script and the `gemini-2.5-pro-high` line deleted; Task 5: `_catalog_note`)
- Create: `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py` (one section per task)
- Modify tests: `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py`, `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py` (Task 1), `tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py` (Task 2)
- Modify docs: `Docs/Operations/Usage_Quotas.md`, `Docs/Providers/Model_Pricing_Catalog.md` (Task 5)
- Create: `Docs/superpowers/plans/2026-10-26-pricing-tiers.md` (this plan, Task 0)
- Scratch (not committed): `<scratchpad>/edit_pricing_tiers.py` (Task 1 Step 7)

## Test commands

Each task's tests (the four files; `-k` picks the task). Measured counts are under each task:

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_pricing_service_tiers.py tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py -k "<task -k expression>"
```

- Task 1 `-k`: `openai_service_tier or long_prompt_on_a_tier or dated_openai_snapshots or service_tier_the_catalog or tier_a_model_has_no_row or no_cached_input or folded_into or openai_call_is_costed or case_insensitively or reports_is_recorded_with_its_usage or records_the_service_tier_the_response_reports`
- Task 2 `-k`: `anthropic_batch or provider_without_a_batch_price or costed_at_the_service_tier_its_usage_reports or streamed_batch_reply`
- Task 3 `-k`: `gemini`
- Task 4 `-k`: `openrouter or reported_cost`

The focused regression set (Step 0 and Task 6):

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py tldw_Server_API/tests/LLM_Adapters/unit tldw_Server_API/tests/Chat/unit/test_chat_service_content.py tldw_Server_API/tests/LLM_Calls/test_llm_providers.py tldw_Server_API/tests/LLM_Calls/test_provider_response_preservation.py tldw_Server_API/tests/LLM_Calls/test_cache_intents.py tldw_Server_API/tests/LLM_Calls/test_gemini_and_hf_regressions.py tldw_Server_API/tests/Chat/unit/test_messages_endpoint_usage.py tldw_Server_API/tests/Chat/unit/test_messages_endpoint_native_errors.py
```

Measured: dev `1529 passed, 1 skipped`; with the whole plan applied `1662 passed, 1 skipped` (the 133 new tests: 127 in the new file, 1 in `test_usage_accounting.py`, 1 in `test_chat_service_content.py`, 4 in `test_messages_native_usage_recording.py`).

The wider set: every other test file that names `compute_costs`, `get_rate_details`, `normalize_llm_usage`, `NormalizedLLMUsage`, `_StreamUsage`, `record_direct_response`, `accounted_direct_stream`, `model_pricing`, `PricingCatalog`, `_normalize_to_openai_shape`, `usage_accounting` or `log_llm_usage`, found by `git grep -l` over `tldw_Server_API/tests`:

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -rs tldw_Server_API/tests/Chat/integration/test_chat_completions_integration.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback_credential_taxonomy.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py tldw_Server_API/tests/Chat/unit/test_chat_service_queue_estimate.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py tldw_Server_API/tests/Chat/unit/test_model_router_accounting.py tldw_Server_API/tests/Chat/unit/test_prompt_cost_guardrails.py tldw_Server_API/tests/Chat/unit/test_streaming_structured_events.py tldw_Server_API/tests/Claims_Extraction/test_fva_verify_endpoint.py tldw_Server_API/tests/Claims/test_claims_engine_modes.py tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py tldw_Server_API/tests/Guardian/test_chat_integration.py tldw_Server_API/tests/Resource_Governance/test_tokens_daily_cap.py
```

Measured: `976 passed, 14 skipped` both on unchanged dev (the review's run) and with the whole plan applied (about 7 to 9 minutes). Thirteen of the skips need live provider keys or URLs (`test_chat_completions_integration.py:507,565,674,792`). One is a standing skip: `test_chat_endpoint_simplified.py:766`, "Streaming tests hang with TestClient".

---

## Task 0: Branch, plan file, baseline

- [ ] **Step 1:** `git fetch origin dev --quiet && git switch -c fix/pricing-service-tiers-13936 origin/dev`.
- [ ] **Step 2:** Mark the task started: `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13936 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-10-26-pricing-tiers.md (owner-approved). ADR assessment: no new ADR; ADR-075 governs prompt_tokens including cache tokens, ADR-059 the backlog tool."`
- [ ] **Step 3:** Copy the approved plan to `Docs/superpowers/plans/2026-10-26-pricing-tiers.md`. Then `git add Docs/superpowers/plans/2026-10-26-pricing-tiers.md && git add -A backlog/tasks && git diff --cached --check` (it must print nothing), and commit (message 0).
- [ ] **Step 4: Baseline.** Run the focused regression set and the wider set on the unchanged code and record both counts. Measured on 39093fc827: focused `1529 passed, 1 skipped`; wider `976 passed, 14 skipped` (the review's run). Any later failure is compared with this baseline; never dismiss one as environment-only without it.
- [ ] **Step 5: Re-fetch the vendor pages** listed under "Vendor prices" and check every figure in the four tables used by the tests. If a figure differs, use the live figure in the test table (the JSON follows it through the script) and note the change in the commit body.

## Task 1: Carry the reported service tier into the usage record and price OpenAI's Batch, Flex and Fast tiers (AC #5, AC #6)

**Files:**
- Create: `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py`
- Modify: `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py` (append), `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py` (after `test_execute_non_stream_call_normalizes_gemini_usage_for_logging`)
- Modify: `llm_usage_normalizer.py`, `pricing_catalog.py`, `usage_tracker.py`, `usage_accounting.py`, `chat_service.py`, `model_pricing.json`

**Interfaces:**
- Produces:
  - `NormalizedLLMUsage.service_tier: str = ""`, a new last field, lowercased, read for every provider.
  - `usage_with_service_tier(response: Any) -> Any` in `llm_usage_normalizer`, exported in `__all__`. It returns `response["usage"]`, plus the response's top-level `service_tier` when the block lacks one, as a copy; otherwise the block itself; `None` for a non-mapping.
  - `PricingCatalog.get_rate_details(provider, model, *, prompt_tokens: int = 0, service_tier: str = "")`.
  - `PricingCatalog._entry_rate_details(rates, *, estimated, prompt_tokens=0, service_tier="")`.
  - The catalog entry key `service_tiers`: `{tier: {prompt, completion, [cache_read, cache_write, cache_write_1h], [long_prompt]}}`.
  - `compute_costs(..., service_tier: str | None = None)`.
- Consumed by Task 2 (`service_tier` in `compute_costs`) and Task 3 (the edit script).

**RED reason (measured on dev):** `84 failed`.
- 67 fail with `TypeError: PricingCatalog.get_rate_details() got an unexpected keyword argument 'service_tier'`.
- 9 fail with `TypeError: compute_costs() got an unexpected keyword argument 'service_tier'`.
- 5 fail with `AttributeError: module '...llm_usage_normalizer' has no attribute 'usage_with_service_tier'`.
- 1 fails with `AttributeError: 'NormalizedLLMUsage' object has no attribute 'service_tier'`.
- The two recorder tests fail on `usage_metadata`: the tier is missing, e.g. `assert {'prompt_toke...l_tokens': 18} == {'prompt_toke...tier': 'flex'}`.

- [ ] **Step 1: Create the test file** `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py` with exactly this content (Tasks 2-4 append sections to it):

```python
"""Service tiers, Gemini cache reads and OpenRouter's reported cost are priced (TASK-13936)."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from tldw_Server_API.app.core.Usage import llm_usage_normalizer, usage_tracker
from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage
from tldw_Server_API.app.core.Usage.pricing_catalog import PricingCatalog, list_provider_models, reset_pricing_catalog
from tldw_Server_API.app.core.Usage.usage_tracker import compute_costs

pytestmark = pytest.mark.unit

# Vendor pages give prices per million tokens; catalog rates are per thousand.
MTOK = 1000.0

# developers.openai.com/api/docs/pricing, "Batch", "Flex" and "Fast" text-token tables, fetched 2026-10-26.
# (input, output, cached input) per million tokens, short context; None where the page shows "-".
# The page renamed Priority to Fast on July 30, 2026. The Fast-mode guide (guides/priority-processing): "For GPT-5.6
# and earlier models, the response returns `priority` whether the request specifies `priority` or `fast`."
OPENAI_TIER_PRICES: dict[str, dict[str, tuple[float, float, float | None]]] = {
    "gpt-5.5": {"batch": (2.50, 15.00, 0.25), "flex": (2.50, 15.00, 0.25), "priority": (12.50, 75.00, 1.25)},
    "gpt-5.5-pro": {"batch": (15.00, 90.00, None), "flex": (15.00, 90.00, None)},
    "gpt-5.4": {"batch": (1.25, 7.50, 0.13), "flex": (1.25, 7.50, 0.13), "priority": (5.00, 30.00, 0.50)},
    "gpt-5.4-pro": {"batch": (15.00, 90.00, None), "flex": (15.00, 90.00, None)},
    "gpt-5.4-mini": {"batch": (0.375, 2.25, 0.0375), "flex": (0.375, 2.25, 0.0375), "priority": (1.50, 9.00, 0.15)},
    "gpt-5.4-nano": {"batch": (0.10, 0.625, 0.01), "flex": (0.10, 0.625, 0.01)},
    "gpt-5.2": {"batch": (0.875, 7.00, 0.0875), "flex": (0.875, 7.00, 0.0875), "priority": (3.50, 28.00, 0.35)},
    "gpt-5.2-pro": {"batch": (10.50, 84.00, None)},
    "gpt-5.1": {"batch": (0.625, 5.00, 0.0625), "flex": (0.625, 5.00, 0.0625), "priority": (2.50, 20.00, 0.25)},
    "gpt-5": {"batch": (0.625, 5.00, 0.0625), "flex": (0.625, 5.00, 0.0625), "priority": (2.50, 20.00, 0.25)},
    "gpt-5-mini": {"batch": (0.125, 1.00, 0.0125), "flex": (0.125, 1.00, 0.0125), "priority": (0.45, 3.60, 0.045)},
    "gpt-5-nano": {"batch": (0.025, 0.20, 0.0025), "flex": (0.025, 0.20, 0.0025)},
    "gpt-5-pro": {"batch": (7.50, 60.00, None)},
    "gpt-4.1": {"batch": (1.00, 4.00, None), "priority": (3.50, 14.00, 0.875)},
    "gpt-4.1-mini": {"batch": (0.20, 0.80, None), "priority": (0.70, 2.80, 0.175)},
    "gpt-4.1-nano": {"batch": (0.05, 0.20, None), "priority": (0.20, 0.80, 0.05)},
    "gpt-4o": {"batch": (1.25, 5.00, None), "priority": (4.25, 17.00, 2.125)},
    "gpt-4o-2024-05-13": {"batch": (2.50, 7.50, None), "priority": (8.75, 26.25, None)},
    "gpt-4o-mini": {"batch": (0.075, 0.30, None), "priority": (0.25, 1.00, 0.125)},
    "o1": {"batch": (7.50, 30.00, None)},
    "o1-pro": {"batch": (75.00, 300.00, None)},
    "o3-pro": {"batch": (10.00, 40.00, None)},
    "o3": {"batch": (1.00, 4.00, None), "flex": (1.00, 4.00, 0.25), "priority": (3.50, 14.00, 0.875)},
    "o4-mini": {"batch": (0.55, 2.20, None), "flex": (0.55, 2.20, 0.138), "priority": (2.00, 8.00, 0.50)},
    "o3-mini": {"batch": (0.55, 2.20, None)},
    "gpt-4-turbo-2024-04-09": {"batch": (5.00, 15.00, None)},
    "gpt-4-0613": {"batch": (15.00, 30.00, None)},
    "gpt-3.5-turbo-0125": {"batch": (0.25, 0.75, None)},
}
# The long-context columns (prompts over 272,000 tokens) of the same tables. Where the page shows "-" there
# (gpt-5.5 and gpt-5.4 Fast, gpt-5.5-pro Batch and Flex), the tier has no long-context price.
OPENAI_LONG_TIER_PRICES: dict[str, dict[str, tuple[float, float, float | None]]] = {
    "gpt-5.5": {"batch": (5.00, 22.50, 0.50), "flex": (5.00, 22.50, 0.50)},
    "gpt-5.4": {"batch": (2.50, 11.25, 0.25), "flex": (2.50, 11.25, 0.25)},
    "gpt-5.4-pro": {"batch": (30.00, 135.00, None), "flex": (30.00, 135.00, None)},
}


def _rates(prices: tuple[float, float, float | None]) -> dict[str, float]:
    """Catalog rates (per 1K tokens) for a page row (per 1M tokens)."""
    prompt, completion, cached = prices
    rates = {"prompt": prompt / MTOK, "completion": completion / MTOK}
    if cached is not None:
        rates["cache_read"] = cached / MTOK
    return rates


@pytest.fixture(autouse=True)
def _shipped_catalog(monkeypatch: pytest.MonkeyPatch):
    """Price with the shipped catalog only."""
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()
    yield
    reset_pricing_catalog()


@pytest.fixture
def costed_rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The llm_usage_log rows log_llm_usage would insert, costed by the shipped catalog."""
    captured: list[dict[str, Any]] = []

    class _Repo:
        """Stands in for AuthnzUsageRepo and keeps the inserted rows."""

        def __init__(self, _pool: Any) -> None:
            """No database."""

        async def insert_llm_usage_log(self, **kwargs: Any) -> None:
            """Record the row instead of writing it."""
            captured.append(kwargs)

    async def _none() -> None:
        """No pool and no daily ledger."""

    monkeypatch.setenv("LLM_USAGE_ENABLED", "true")
    monkeypatch.setattr(usage_tracker, "get_db_pool", _none)
    monkeypatch.setattr(usage_tracker, "_get_tokens_daily_ledger", _none)
    monkeypatch.setattr(usage_tracker, "AuthnzUsageRepo", _Repo)
    return captured


async def _log(provider: str, model: str, usage: dict[str, Any], prompt_tokens: int, completion_tokens: int) -> None:
    """Record one call the way the provider-level recorder does."""
    await usage_tracker.log_llm_usage(
        user_id=5,
        key_id=None,
        endpoint="internal",
        operation="llm_call",
        provider=provider,
        model=model,
        status=200,
        latency_ms=1,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usage_metadata=usage,
    )


# --- OpenAI service tiers (AC #5) and unpriced tiers (AC #6) ---


@pytest.mark.parametrize(
    ("model", "tier", "prices"),
    sorted((model, tier, prices) for model, tiers in OPENAI_TIER_PRICES.items() for tier, prices in tiers.items()),
)
def test_an_openai_service_tier_is_priced_at_its_published_rate(model: str, tier: str, prices: tuple) -> None:
    got, _estimated = PricingCatalog().get_rate_details("openai", model, service_tier=tier)

    assert got == pytest.approx(_rates(prices))


@pytest.mark.parametrize(
    ("model", "tier", "prices"),
    sorted((model, tier, prices) for model, tiers in OPENAI_LONG_TIER_PRICES.items() for tier, prices in tiers.items()),
)
def test_an_openai_service_tier_prices_a_prompt_over_272k_tokens_at_its_long_context_rate(
    model: str, tier: str, prices: tuple
) -> None:
    cat = PricingCatalog()

    at_limit, _e1 = cat.get_rate_details("openai", model, prompt_tokens=272_000, service_tier=tier)
    over_limit, _e2 = cat.get_rate_details("openai", model, prompt_tokens=272_001, service_tier=tier)

    assert at_limit == pytest.approx(_rates(OPENAI_TIER_PRICES[model][tier]))
    assert over_limit == pytest.approx(_rates(prices))


@pytest.mark.parametrize(
    ("model", "tier"),
    [("gpt-5.5", "priority"), ("gpt-5.4", "priority"), ("gpt-5.5-pro", "batch"), ("gpt-5.5-pro", "flex")],
)
def test_a_long_prompt_on_a_tier_with_no_long_context_price_costs_the_standard_long_context_rate(
    model: str, tier: str
) -> None:
    """The page shows "-" for these tiers' long-context columns: no price, so the standard one (AC #6)."""
    cat = PricingCatalog()

    standard_long, _e1 = cat.get_rate_details("openai", model, prompt_tokens=272_001)
    tier_long, _e2 = cat.get_rate_details("openai", model, prompt_tokens=272_001, service_tier=tier)

    assert tier_long == pytest.approx(standard_long)


def test_dated_openai_snapshots_carry_their_alias_service_tiers() -> None:
    """A snapshot priced like its alias (2026-10-23 pricing plan) takes the alias's tier rates too."""
    cat = PricingCatalog()
    checked = 0

    for model in list_provider_models("openai"):
        dated = re.match(r"^(.*)-(\d{4}-\d{2}-\d{2})$", model)
        if dated is None or dated.group(1) not in OPENAI_TIER_PRICES:
            continue
        alias = dated.group(1)
        if cat.get_rate_details("openai", model)[0] != cat.get_rate_details("openai", alias)[0]:
            continue  # priced on its own row (gpt-4o-2024-05-13)
        for tier in OPENAI_TIER_PRICES[alias]:
            got = cat.get_rate_details("openai", model, service_tier=tier)[0]
            assert got == pytest.approx(cat.get_rate_details("openai", alias, service_tier=tier)[0]), (model, tier)
        checked += 1

    assert checked >= 23  # the 23 snapshots catalogued on 2026-10-26; a new one is checked by the loop above


@pytest.mark.parametrize("tier", ["default", "scale", "auto", "standard", "", "ultrafast", None])
def test_a_service_tier_the_catalog_has_no_rate_for_costs_the_standard_rate(tier: str | None) -> None:
    args = {"cache_read_input_tokens": 400, "billable_input_tokens": 600}

    standard = compute_costs("openai", "gpt-5-mini", 1000, 100, **args)

    assert compute_costs("openai", "gpt-5-mini", 1000, 100, service_tier=tier, **args) == standard


def test_a_tier_a_model_has_no_row_for_costs_the_standard_rate() -> None:
    """gpt-4.1 has Batch and Fast rows but no Flex row; Anthropic publishes no Priority Tier token price."""
    assert compute_costs("openai", "gpt-4.1", 1000, 100, service_tier="flex") == compute_costs(
        "openai", "gpt-4.1", 1000, 100
    )
    assert compute_costs("anthropic", "claude-sonnet-4-6", 1000, 100, service_tier="priority") == compute_costs(
        "anthropic", "claude-sonnet-4-6", 1000, 100
    )


def test_a_tier_with_no_cached_input_price_costs_cached_tokens_at_the_tier_input_rate() -> None:
    """gpt-4.1 Batch shows "-" for cached input: every prompt token pays the Batch input rate."""
    prompt_cost, completion_cost, _t, _e = compute_costs(
        "openai", "gpt-4.1", 1000, 100, cache_read_input_tokens=800, billable_input_tokens=200, service_tier="batch"
    )

    assert prompt_cost == pytest.approx(1000 * 1.00 / 1_000_000)
    assert completion_cost == pytest.approx(100 * 4.00 / 1_000_000)


def test_the_response_service_tier_is_folded_into_a_copy_of_its_usage_block() -> None:
    usage = {"prompt_tokens": 10, "completion_tokens": 2}
    response = {"object": "chat.completion", "service_tier": "flex", "usage": usage}

    assert llm_usage_normalizer.usage_with_service_tier(response) == {**usage, "service_tier": "flex"}
    assert "service_tier" not in usage
    assert llm_usage_normalizer.usage_with_service_tier({"usage": usage}) is usage
    assert llm_usage_normalizer.usage_with_service_tier({"service_tier": "flex"}) is None
    assert llm_usage_normalizer.usage_with_service_tier(
        {"usage": {"service_tier": "batch"}, "service_tier": "flex"}
    ) == {"service_tier": "batch"}
    assert llm_usage_normalizer.usage_with_service_tier("not a response") is None


@pytest.mark.parametrize(
    ("tier", "prices"),
    [
        ("flex", (0.125, 1.00, 0.0125)),
        ("priority", (0.45, 3.60, 0.045)),
        ("batch", (0.125, 1.00, 0.0125)),
        ("default", (0.25, 2.00, 0.025)),
    ],
)
async def test_an_openai_call_is_costed_at_the_service_tier_its_response_reports(
    costed_rows: list[dict[str, Any]], tier: str, prices: tuple[float, float, float]
) -> None:
    response = {
        "object": "chat.completion",
        "service_tier": tier,
        "usage": {
            "prompt_tokens": 10_000,
            "completion_tokens": 1_000,
            "total_tokens": 11_000,
            "prompt_tokens_details": {"cached_tokens": 4_000},
        },
    }

    await _log("openai", "gpt-5-mini", llm_usage_normalizer.usage_with_service_tier(response), 10_000, 1_000)

    (row,) = costed_rows
    in_mtok, out_mtok, cached_mtok = prices
    assert row["prompt_cost_usd"] == pytest.approx((6_000 * in_mtok + 4_000 * cached_mtok) / 1_000_000)
    assert row["completion_cost_usd"] == pytest.approx(1_000 * out_mtok / 1_000_000)
    assert json.loads(row["raw_usage_metadata_json"])["service_tier"] == tier


def test_the_reported_service_tier_is_read_case_insensitively() -> None:
    normalized = normalize_llm_usage(provider="openai", usage={"prompt_tokens": 5, "service_tier": " Flex "})

    assert normalized.service_tier == "flex"
```

- [ ] **Step 2: Append the provider-level recorder test** to the end of `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py` (it uses that file's `_FakeProvider`, `USAGE`, `REQUEST`, `rows` and `_settled`):

```python
class _TieredProvider(_FakeProvider):
    """An OpenAI-shaped provider that reports its service tier beside the usage block, not in it."""

    name = "fake-tiered"

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Return a completion that ran on the flex tier."""
        return {"choices": [{"message": {"content": "hello"}}], "service_tier": "flex", "usage": dict(USAGE)}

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Yield chunks that each name the tier; the last one carries usage."""
        yield 'data: {"service_tier":"flex","choices":[{"delta":{"content":"hel"}}]}\n\n'
        yield 'data: {"service_tier":"flex","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        yield "data: [DONE]\n\n"


async def test_the_service_tier_a_response_reports_is_recorded_with_its_usage(rows) -> None:
    """OpenAI reports the tier beside usage; the row's usage carries it, the response is unchanged (TASK-13936)."""
    with llm_call_context(user_id=7):
        response = _TieredProvider().chat(REQUEST)
        list(_TieredProvider().stream(REQUEST))
    chat_row, stream_row = await _settled(rows)
    assert chat_row["usage_metadata"] == {**USAGE, "service_tier": "flex"}
    assert stream_row["usage_metadata"] == {"prompt_tokens": 5, "completion_tokens": 2, "service_tier": "flex"}
    assert response["usage"] == USAGE
```

- [ ] **Step 3: Add the chat-pipeline recorder test** to `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py`, directly after `test_execute_non_stream_call_normalizes_gemini_usage_for_logging`. It reuses that file's `_CapturingMetrics`, `_RedactingModeration` and `execute_non_stream_call`:

```python
@pytest.mark.asyncio
async def test_execute_non_stream_call_records_the_service_tier_the_response_reports(monkeypatch):
    """OpenAI reports service_tier beside usage; the usage record carries it so the tier is costed (TASK-13936)."""
    logged_usage: dict[str, object] = {}

    async def fake_log_llm_usage(**kwargs):
        logged_usage.update(kwargs)

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    usage = {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110}
    reply = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "service_tier": "priority",
        "usage": usage,
    }

    async def save_message_fn(*_args, **_kwargs):
        return None

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    await execute_non_stream_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "api_key": "test-key",
            "messages_payload": [{"role": "user", "content": "hi"}],
            "model": "gpt-5-mini",
            "streaming": False,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-5-mini",
        request_json="{}",
        request=request,
        metrics=_CapturingMetrics(),
        provider_manager=None,
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        should_persist=False,
        final_conversation_id="conv-123",
        character_card_for_context={"name": "Test"},
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=lambda: reply,
        refresh_provider_params=lambda *_args, **_kwargs: None,
        moderation_getter=lambda: _RedactingModeration(),
    )

    assert logged_usage["usage_metadata"] == {**usage, "service_tier": "priority"}
    assert "service_tier" not in usage
```

- [ ] **Step 4: Run the Task 1 tests; see RED.** Use the "Test commands" command with the Task 1 `-k`. Expected: `84 failed`, with the reasons above.

- [ ] **Step 5: Implement the normalizer and the catalog.**

`tldw_Server_API/app/core/Usage/llm_usage_normalizer.py`:
- In `NormalizedLLMUsage`, after `inference_geo: str = ""`, add:

```python
    # The service tier the provider reports the call ran on (lowercased; empty when not reported).
    service_tier: str = ""
```

- In `normalize_llm_usage`, after `inference_geo = ""` (before `if "anthropic" in provider_key:`), add:

```python
    # Anthropic reports it in the usage block; OpenAI beside it (usage_with_service_tier folds it in).
    service_tier = _usage_str(usage_mapping, "service_tier")
```

- Pass `service_tier=service_tier,` to the returned `NormalizedLLMUsage(...)`, after `inference_geo=inference_geo,`.
- Add this function after `normalize_llm_usage`:

```python
def usage_with_service_tier(response: Any) -> Any:
    """``response["usage"]``, with the response's top-level ``service_tier`` added when the block lacks one.

    OpenAI reports the tier a call ran on beside the usage block; Anthropic reports it inside. The result is a copy,
    so the response a caller passes on is unchanged; without a tier to add it is the usage block itself.
    """
    if not isinstance(response, Mapping):
        return None
    usage = response.get("usage")
    tier = response.get("service_tier")
    if isinstance(usage, Mapping) and isinstance(tier, str) and tier.strip() and "service_tier" not in usage:
        return {**usage, "service_tier": tier}
    return usage
```

- Change `__all__` to `["NormalizedLLMUsage", "normalize_llm_usage", "usage_with_service_tier"]`.

`tldw_Server_API/app/core/Usage/pricing_catalog.py`. Add this function directly above `_has_rate`:

```python
def _parse_service_tier(rates: dict) -> dict | None:
    """One service tier's rate set (``prompt``, ``completion``, cache rates, optional ``long_prompt``), or None.

    None when it would misprice: without ``prompt`` or ``completion`` that side would cost $0, and a malformed
    ``long_prompt`` would drop the tier's long-context price.
    """
    if not _has_rate(rates, "prompt", "in", "input") or not _has_rate(rates, "completion", "out", "output"):
        return None
    tier = _parse_rates(rates)
    long_prompt = rates.get("long_prompt")
    if long_prompt is not None:
        parsed = _parse_long_prompt_tier(long_prompt) if isinstance(long_prompt, dict) else None
        if parsed is None:
            return None
        tier["long_prompt"] = parsed
    return tier
```

In `_merge_overrides`, after the `long_prompt` block (the `else: entry["long_prompt"] = parsed_tier` lines) and before `base[model] = entry`, add:

```python
                service_tiers = rates.get("service_tiers")
                if isinstance(service_tiers, dict):
                    # Keyed by the service_tier the provider reports (OpenAI: batch, flex, priority).
                    parsed_tiers = {}
                    for tier_name, tier_rates in service_tiers.items():
                        parsed = _parse_service_tier(tier_rates) if isinstance(tier_rates, dict) else None
                        if parsed is None:
                            logger.warning(
                                "Ignoring service tier {} of pricing entry {}/{}: it needs prompt and completion "
                                "rates and a valid long_prompt; the standard rates apply",
                                tier_name,
                                provider,
                                model,
                            )
                        else:
                            parsed_tiers[str(tier_name)] = parsed
                    if parsed_tiers:
                        entry["service_tiers"] = parsed_tiers
```

Change `get_rate_details`'s signature to `self, provider: str, model: str, *, prompt_tokens: int = 0, service_tier: str = ""`. Replace the docstring sentence "tier's rates apply." with:

```python
        tier's rates apply. ``service_tier`` is the tier the provider reports;
        an entry's ``service_tiers`` rates for it apply, and a tier the entry
        has no rates for costs the standard rates.
```

Replace its three `return self._entry_rate_details(..., prompt_tokens=prompt_tokens)` calls (exact match, then best substring match, then the provider `default`) with:

```python
            return self._entry_rate_details(
                prov_map[mdl], estimated=False, prompt_tokens=prompt_tokens, service_tier=service_tier
            )
```

```python
            return self._entry_rate_details(
                best_match[1], estimated=True, prompt_tokens=prompt_tokens, service_tier=service_tier
            )
```

```python
            return self._entry_rate_details(
                prov_map["default"], estimated=True, prompt_tokens=prompt_tokens, service_tier=service_tier
            )
```

In `_entry_rate_details`:
- Change the signature to `rates: dict, *, estimated: bool, prompt_tokens: int = 0, service_tier: str = ""`.
- After the `if not isinstance(rates, dict): return {"prompt": 0.0, "completion": 0.0}, True` guard, add:

```python
        estimated = bool(rates.get("estimated", estimated))
        service = (rates.get("service_tiers") or {}).get(service_tier) if service_tier else None
        long_tier = rates.get("long_prompt")
        long_applies = isinstance(long_tier, dict) and prompt_tokens > int(long_tier.get("above_tokens", 0) or 0)
        if isinstance(service, dict) and (not long_applies or "long_prompt" in service):
            # The tier's own prices. A tier with no long-prompt price leaves a long prompt at the standard one.
            rates = service
```

- Change the final `return details, bool(rates.get("estimated", estimated))` to `return details, estimated`. The flag is read from the entry before `rates` may become a tier.

- [ ] **Step 6: Implement the cost and the recorders.**

`tldw_Server_API/app/core/Usage/usage_tracker.py`, `compute_costs`:
- After `inference_geo: str | None = None,`, add `service_tier: str | None = None,`.
- Append to its docstring:

```python
    ``service_tier`` is the tier the provider reports: an entry's
    ``service_tiers`` rates for it apply (OpenAI), Anthropic's ``batch`` halves
    every category, and any other tier costs the standard rates.
```

- Replace the two lines `# The whole prompt, cache tokens included (ADR-075), picks a long-prompt price tier.` and `rates, est = catalog.get_rate_details(provider, model, prompt_tokens=prompt_total)` with:

```python
    tier = (service_tier or "").strip().lower()
    # The whole prompt, cache tokens included (ADR-075), picks a long-prompt price tier.
    rates, est = catalog.get_rate_details(provider, model, prompt_tokens=prompt_total, service_tier=tier)
```

  The docstring's "Anthropic's ``batch`` halves every category" becomes true in Task 2; Task 1's tests do not depend on it.
- In `log_llm_usage`'s `compute_costs(...)` call, after `inference_geo=normalized_usage.inference_geo,`, add `service_tier=normalized_usage.service_tier,`.

`tldw_Server_API/app/core/LLM_Calls/usage_accounting.py`:
- Add the import `from tldw_Server_API.app.core.Usage.llm_usage_normalizer import usage_with_service_tier` after the `llm_quota_gate` import.
- In `_StreamUsage.feed`, replace `usage = data.get("usage")` with:

```python
                usage = usage_with_service_tier(data)  # an OpenAI chunk names its service tier beside the usage
```

- In `_record_response`, replace `usage = response.get("usage") if isinstance(response, dict) else None` with `usage = usage_with_service_tier(response)`.

`tldw_Server_API/app/core/Chat/chat_service.py`:
- Change the import at line 223 to `from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage, usage_with_service_tier`.
- In `_record_dict_response_usage` (line 6918 on origin/dev `2c60650739`), replace `usage = response.get("usage")` with:

```python
        usage = usage_with_service_tier(response)  # OpenAI reports service_tier beside usage
```

- [ ] **Step 7: Write the OpenAI tiers into `model_pricing.json` with a script.**
  - Re-check Task 0 Step 5's figures first.
  - Save this script as `<scratchpad>/edit_pricing_tiers.py`. It reads the rates from the test module, so they are typed once.
  - Importing the test module initialises the app config, which creates gitignored runtime files in the worktree if they are missing (the r2 dry run got `Databases/user_databases/1` and `Databases/circuit_breaker_registry.db`; the review also saw `Logs/` and `Config_Files/session_encryption.key`). `git status` stays clean; do not add them.
  - Then run, from the repo root: `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python <scratchpad>/edit_pricing_tiers.py openai`. Expected last line: `openai: 51 rows`: the 28 models of the table plus the 23 snapshots priced like their alias.

```python
"""Write the OpenAI service tiers or the Gemini cache rates into model_pricing.json, one entry per line (TASK-13936).

Run from the repo root: PYTHONPATH=. .venv/bin/python <this file> openai|google
The rates come from the page transcriptions in tests/Usage/test_pricing_service_tiers.py, so they live in one place.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PATH = Path("tldw_Server_API/Config_Files/model_pricing.json")
ROW = re.compile(r'^(    "(?P<key>[^"]+)": \{ .*) \}(?P<comma>,?)$')
BLOCK = re.compile(r'^  "(?P<provider>[^"]+)": \{$')
TESTS = "tldw_Server_API.tests.Usage.test_pricing_service_tiers"


def num(value: float | int) -> str:
    """A rate in the file's plain decimal style (no exponent)."""
    return str(value) if isinstance(value, int) else format(value, ".10f").rstrip("0").rstrip(".")


def fmt(obj: dict) -> str:
    """An inline object in the file's '{ "k": v }' style."""
    return "{ " + ", ".join(f'"{k}": {fmt(v) if isinstance(v, dict) else num(v)}' for k, v in obj.items()) + " }"


def rate_set(prices: tuple) -> dict:
    """Catalog rates per 1K tokens from a page row per 1M tokens."""
    prompt, completion, cached = prices
    out = {"prompt": round(prompt / 1000, 10), "completion": round(completion / 1000, 10)}
    if cached is not None:
        out["cache_read"] = round(cached / 1000, 10)
    return out


def openai_addition(key: str, rows: dict) -> str | None:
    """The service_tiers member for an OpenAI row: its own, or its alias's for a snapshot priced like the alias."""
    tables = __import__(TESTS, fromlist=["OPENAI_TIER_PRICES"])
    alias = key if key in tables.OPENAI_TIER_PRICES else None
    dated = re.match(r"^(.*)-\d{4}-\d{2}-\d{2}$", key)
    if alias is None and dated and dated.group(1) in tables.OPENAI_TIER_PRICES:
        if rows[key]["prompt"] == rows[dated.group(1)]["prompt"]:
            alias = dated.group(1)
    if alias is None:
        return None
    tiers = {}
    for tier, prices in tables.OPENAI_TIER_PRICES[alias].items():
        tiers[tier] = rate_set(prices)
        long_prices = tables.OPENAI_LONG_TIER_PRICES.get(alias, {}).get(tier)
        if long_prices is not None:
            tiers[tier]["long_prompt"] = {"above_tokens": 272000, **rate_set(long_prices)}
    return f'"service_tiers": {fmt(tiers)}'


def google_addition(key: str, rows: dict) -> str | None:
    """The cache_read (and long_prompt) members for a Gemini row on Google's page, or for a 2.5 -preview alias."""
    tables = __import__(TESTS, fromlist=["GEMINI_PRICES"])
    page_key = tables.GEMINI_PREVIEW_ALIASES.get(key, key)  # off the page; must be priced exactly as its GA model
    if page_key not in tables.GEMINI_PRICES:
        return None
    prompt, completion, cached = tables.GEMINI_PRICES[page_key]
    assert rows[key]["prompt"] == round(prompt / 1000, 10), key
    assert rows[key]["completion"] == round(completion / 1000, 10), key
    text = f'"cache_read": {num(round(cached / 1000, 10))}'
    if page_key in tables.GEMINI_LONG_PRICES:
        text += f', "long_prompt": {fmt({"above_tokens": 200000, **rate_set(tables.GEMINI_LONG_PRICES[page_key])})}'
    return text


def main(which: str) -> None:
    addition, marker = {"openai": (openai_addition, "service_tiers"), "google": (google_addition, "cache_read")}[which]
    text = PATH.read_text()
    data = json.loads(text)
    provider = None
    done: list[str] = []
    out = []
    for line in text.split("\n"):
        block = BLOCK.match(line)
        if block:
            provider = block.group("provider")
        row = ROW.match(line)
        if row and provider == which:
            extra = addition(row.group("key"), data[which])
            if extra is not None:
                assert marker not in line, f"already edited: {line}"
                line = f'{row.group(1)}, {extra} }}{row.group("comma")}'
                done.append(row.group("key"))
        out.append(line)
    new_text = "\n".join(out)
    json.loads(new_text)  # still valid JSON
    PATH.write_text(new_text)
    print(f"{which}: {len(done)} rows")


if __name__ == "__main__":
    main(sys.argv[1])
```

Check two rows by eye with `git diff -U0 -- tldw_Server_API/Config_Files/model_pricing.json | grep '"gpt-4.1":'`. The prototype produced:

```json
    "gpt-4.1": { "prompt": 0.002, "completion": 0.008, "cache_read": 0.0005, "service_tiers": { "batch": { "prompt": 0.001, "completion": 0.004 }, "priority": { "prompt": 0.0035, "completion": 0.014, "cache_read": 0.000875 } } },
```

and for `gpt-5.4`, `"service_tiers": { "batch": { "prompt": 0.00125, "completion": 0.0075, "cache_read": 0.00013, "long_prompt": { "above_tokens": 272000, "prompt": 0.0025, "completion": 0.01125, "cache_read": 0.00025 } }, ...`.

- [ ] **Step 8: Run; see GREEN.**
  - The Task 1 command. Measured: `84 passed`.
  - Then the focused regression set. Every Step 0 test still passes; the count rises by the new tests.
  - Then `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_pricing_catalog.py`. The predecessor's catalog tests compare whole rate dicts for OpenAI models with `pytest.approx`. `service_tiers` never appears in the details dict, so they must stay green. Measured: `110 passed`.
- [ ] **Step 9: Commit.** `git add` the nine files: `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py`, `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py`, `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py`, `tldw_Server_API/app/core/Usage/llm_usage_normalizer.py`, `tldw_Server_API/app/core/Usage/pricing_catalog.py`, `tldw_Server_API/app/core/Usage/usage_tracker.py`, `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py`, `tldw_Server_API/app/core/Chat/chat_service.py` and `tldw_Server_API/Config_Files/model_pricing.json`. Then `git diff --cached --check`, then commit (message 1).

## Task 2: Cost Anthropic Batch usage at half price (AC #4)

**Files:** `tldw_Server_API/app/core/Usage/usage_tracker.py`; tests: `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py` (append), `tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py` (append).

**Interfaces:**
- Consumes Task 1's `compute_costs(..., service_tier=...)` and the normalizer's `service_tier`.
- Produces the constant `ANTHROPIC_BATCH_MULTIPLIER = 0.5` in `usage_tracker`.

**RED reason (measured with Task 1 in place):** `23 failed, 3 passed`.
- The 19 batch-table cases fail at list price, e.g. `assert (5.0, 25.0) == approx((2.5, 12.5))`.
- The Haiku 5.5 and stacking tests fail the same way.
- The native batch reply and stream fail with `assert 0.042515 == 0.0212575`.
- The three that pass are the guard `test_a_provider_without_a_batch_price_is_not_discounted` and the native `standard` and `priority` cases.
- (On unchanged dev the same tests give `24 failed, 2 passed`: the guard fails with the Task 1 `TypeError`.)

- [ ] **Step 1: Append to `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py`** (two blank lines before it):

```python
# --- Anthropic Batch (AC #4) ---

# platform.claude.com/docs/en/about-claude/pricing, "Batch processing", fetched 2026-10-26:
# "a 50 percent discount on both input and output tokens". (batch input, batch output) per million tokens.
ANTHROPIC_BATCH_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (5.00, 25.00),
    "claude-mythos-5-1": (5.00, 25.00),
    "claude-fable-5": (5.00, 25.00),
    "claude-mythos-5": (5.00, 25.00),
    "claude-opus-5-5": (2.00, 10.00),
    "claude-opus-5": (2.50, 12.50),
    "claude-opus-4-8": (2.50, 12.50),
    "claude-opus-4-7": (2.50, 12.50),
    "claude-opus-4-6": (2.50, 12.50),
    "claude-opus-4-5": (2.50, 12.50),
    "claude-opus-4-1": (7.50, 37.50),
    "claude-opus-4": (7.50, 37.50),
    "claude-sonnet-5-5": (1.00, 5.00),
    "claude-sonnet-5": (1.00, 5.00),
    "claude-sonnet-4-6": (1.50, 7.50),
    "claude-sonnet-4-5": (1.50, 7.50),
    "claude-sonnet-4": (1.50, 7.50),
    "claude-haiku-4-5": (0.50, 2.50),
    "claude-3-5-haiku-20241022": (0.40, 2.00),
}


@pytest.mark.parametrize(("model", "prices"), sorted(ANTHROPIC_BATCH_PRICES.items()))
def test_an_anthropic_batch_call_costs_the_published_batch_price(model: str, prices: tuple[float, float]) -> None:
    prompt_cost, completion_cost, _t, _e = compute_costs(
        "anthropic", model, 1_000_000, 1_000_000, billable_input_tokens=1_000_000, service_tier="batch"
    )

    assert (prompt_cost, completion_cost) == pytest.approx(prices)


def test_an_anthropic_batch_call_on_haiku_5_5_halves_both_prompt_length_tiers() -> None:
    """Page rows: Haiku 5.5 batch input $0.05 up to 100,000 prompt tokens, $0.25 over."""
    low = compute_costs(
        "anthropic", "claude-haiku-5-5", 100_000, 0, billable_input_tokens=100_000, service_tier="batch"
    )
    high = compute_costs(
        "anthropic", "claude-haiku-5-5", 100_001, 0, billable_input_tokens=100_001, service_tier="batch"
    )

    assert low[0] == pytest.approx(100_000 * 0.05 / 1_000_000)
    assert high[0] == pytest.approx(100_001 * 0.25 / 1_000_000)


def test_the_anthropic_batch_discount_stacks_with_the_cache_rates_and_us_only_inference() -> None:
    """Cache multipliers "stack with other pricing modifiers, including the Batch API discount and data residency"."""
    args = {
        "cache_read_input_tokens": 40_000,
        "cache_write_input_tokens": 10_000,
        "cache_write_1h_input_tokens": 4_000,
        "billable_input_tokens": 50_000,
        "inference_geo": "us",
    }

    standard = compute_costs("anthropic", "claude-sonnet-4-6", 100_000, 2_000, **args)
    batch = compute_costs("anthropic", "claude-sonnet-4-6", 100_000, 2_000, service_tier="batch", **args)

    assert batch[0] == pytest.approx(standard[0] * 0.5)
    assert batch[1] == pytest.approx(standard[1] * 0.5)


def test_a_provider_without_a_batch_price_is_not_discounted() -> None:
    assert compute_costs("mistral", "mistral-large-2512", 1000, 100, service_tier="batch") == compute_costs(
        "mistral", "mistral-large-2512", 1000, 100
    )
```

- [ ] **Step 2: Append to `tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py`.** It uses that file's `costed_rows`, `CACHED_USAGE`, `_native_reply`, `_native_stream`, `_frames` and `_call`:

```python
@pytest.mark.parametrize(("service_tier", "factor"), [("batch", 0.5), ("standard", 1.0), ("priority", 1.0)])
async def test_a_native_reply_is_costed_at_the_service_tier_its_usage_reports(
    costed_rows, monkeypatch, service_tier, factor
) -> None:
    """Batch is 50% off every token category; Priority Tier has no token price of its own (TASK-13936)."""
    usage = {**CACHED_USAGE, "output_tokens": 100, "service_tier": service_tier}
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply(usage))
    with scoped_context(user_id=5):
        response = await _call("anthropic/claude-opus-5", stream=False)
    assert response.status_code == 200
    await llm_accounting.drain_accounting()
    (row,) = costed_rows
    # Opus 5 per 1K: input $0.005, cache read $0.0005, 5-minute write $0.00625, output $0.025.
    assert row["prompt_cost_usd"] == pytest.approx((3 * 0.005 + 60000 * 0.0005 + 2000 * 0.00625) / 1000 * factor)
    assert row["completion_cost_usd"] == pytest.approx(100 * 0.025 / 1000 * factor)


async def test_a_streamed_batch_reply_is_costed_at_half_price(costed_rows, monkeypatch) -> None:
    """message_start carries service_tier; message_delta does not, and the merge keeps it."""
    start = {**CACHED_USAGE, "output_tokens": 1, "service_tier": "batch"}
    frames = _frames(start, {"output_tokens": 100})
    monkeypatch.setattr(messages_endpoint, "_prepare_native_stream_iterator", _native_stream(frames))
    with scoped_context(user_id=5):
        response = await _call("anthropic/claude-opus-5", stream=True)
        [item async for item in response.body_iterator]
    await llm_accounting.drain_accounting()
    (row,) = costed_rows
    assert row["prompt_cost_usd"] == pytest.approx((3 * 0.005 + 60000 * 0.0005 + 2000 * 0.00625) / 1000 * 0.5)
    assert row["completion_cost_usd"] == pytest.approx(100 * 0.025 / 1000 * 0.5)
```

- [ ] **Step 3: Run the Task 2 command; see RED** (`23 failed, 3 passed`).
- [ ] **Step 4: Implement** in `tldw_Server_API/app/core/Usage/usage_tracker.py`.
  - Add the import `from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name` after the `anthropic_model_ids` import.
  - After `ANTHROPIC_US_ONLY_MULTIPLIER = 1.1`, add:

```python
# Anthropic's Batch API: "a 50 percent discount on both input and output tokens", and the cache multipliers "stack
# with other pricing modifiers, including the Batch API discount and data residency" (same page). Every row of its
# batch table is half the list row, so one multiplier on every category, applied when usage.service_tier is "batch".
ANTHROPIC_BATCH_MULTIPLIER = 0.5
```

  - In `compute_costs`, right after the `multiplier *= ANTHROPIC_US_ONLY_MULTIPLIER` block and before `prompt_cost *= multiplier`, add:

```python
    if tier == "batch" and canonical_provider_name(provider or "") == "anthropic":
        multiplier *= ANTHROPIC_BATCH_MULTIPLIER
```

- [ ] **Step 5: Run; see GREEN.** The Task 2 command measured `26 passed`. Then re-run the Task 1 command (`84 passed`) and `tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py` whole.
- [ ] **Step 6: Commit** (message 2): `git add` the three files, then `git diff --cached --check`.

## Task 3: Cost Gemini cached input at the context-caching rate (AC #1)

**Files:** `tldw_Server_API/app/core/LLM_Calls/providers/google_adapter.py`, `tldw_Server_API/Config_Files/model_pricing.json` (script, plus one deleted line); tests: `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py` (append).

**Interfaces:**
- Produces: `GoogleAdapter._normalize_to_openai_shape(data)["usage"]` gains `cachedContentTokenCount` when the reply's `usageMetadata` has an integer one.
- The normalizer's existing Google branch reads it into `cache_read_input_tokens`.
- `compute_costs` already prices `cache_read`.
- `GEMINI_PREVIEW_ALIASES` in the test module, read by the Task 1 script, maps the four 2.5 -preview entries to their GA model.

**RED reason (measured):** `13 failed, 1 passed`. The pass is the pre-existing `test_execute_non_stream_call_normalizes_gemini_usage_for_logging`, which `-k gemini` also selects.
- The 12 catalog cases (8 page models, 4 -preview entries) fail because the Gemini rows have no `cache_read` and no `long_prompt`, e.g. `assert {'prompt': 0..., 'completion': 0.0025} == approx({... 'cache_read': 3e-05})`.
- The costed reply test fails because the adapter drops `cachedContentTokenCount`, so every prompt token costs input.

- [ ] **Step 1: Append to `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py`** (two blank lines before it):

```python
# --- Gemini cached content (AC #1) ---

# ai.google.dev/gemini-api/docs/pricing, Standard tier, paid, text/image/video, fetched 2026-10-26:
# (input, output, context caching) per million tokens, prompts up to 200k tokens.
GEMINI_PRICES: dict[str, tuple[float, float, float]] = {
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025),
    "gemini-3-flash-preview": (0.50, 3.00, 0.05),
    "gemini-2.5-pro": (1.25, 10.00, 0.125),
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01),
}
# The same rows for prompts over 200k tokens.
GEMINI_LONG_PRICES: dict[str, tuple[float, float, float]] = {
    "gemini-3.1-pro-preview": (4.00, 18.00, 0.40),
    "gemini-2.5-pro": (2.50, 15.00, 0.25),
}
# Catalogued -preview entries the page does not list, priced exactly as their GA model, so they take its cache rate.
# The 2.0 Flash entries are off the page too, and Google publishes no cache rate for them, so they get none.
GEMINI_PREVIEW_ALIASES: dict[str, str] = {
    "gemini-2.5-flash-preview": "gemini-2.5-flash",
    "gemini-2.5-flash-preview-09-2025": "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview": "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025": "gemini-2.5-flash-lite",
}


@pytest.mark.parametrize(("model", "prices"), sorted(GEMINI_PRICES.items()))
def test_a_gemini_model_carries_its_published_context_caching_rate(
    model: str, prices: tuple[float, float, float]
) -> None:
    got, _estimated = PricingCatalog().get_rate_details("google", model, prompt_tokens=200_000)

    assert got == pytest.approx(_rates(prices))


@pytest.mark.parametrize(("model", "ga_model"), sorted(GEMINI_PREVIEW_ALIASES.items()))
def test_a_gemini_2_5_preview_entry_carries_its_ga_model_context_caching_rate(model: str, ga_model: str) -> None:
    got, _estimated = PricingCatalog().get_rate_details("google", model)

    assert got == pytest.approx(_rates(GEMINI_PRICES[ga_model]))


@pytest.mark.parametrize(("model", "prices"), sorted(GEMINI_LONG_PRICES.items()))
def test_a_gemini_prompt_over_200k_tokens_pays_the_long_prompt_rates(
    model: str, prices: tuple[float, float, float]
) -> None:
    got, _estimated = PricingCatalog().get_rate_details("google", model, prompt_tokens=200_001)

    assert got == pytest.approx(_rates(prices))


async def test_a_gemini_reply_costs_its_cached_tokens_at_the_cache_read_rate(costed_rows: list[dict[str, Any]]) -> None:
    """The adapter keeps cachedContentTokenCount, which promptTokenCount includes."""
    from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter

    shaped = GoogleAdapter._normalize_to_openai_shape(
        {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 10_000,
                "candidatesTokenCount": 100,
                "totalTokenCount": 10_100,
                "cachedContentTokenCount": 8_000,
            },
        }
    )

    await _log("google", "gemini-2.5-flash", shaped["usage"], 10_000, 100)

    (row,) = costed_rows
    assert row["cache_read_input_tokens"] == 8_000
    # gemini-2.5-flash per 1M: input $0.30, context caching $0.03, output $2.50.
    assert row["prompt_cost_usd"] == pytest.approx((2_000 * 0.30 + 8_000 * 0.03) / 1_000_000)
    assert row["completion_cost_usd"] == pytest.approx(100 * 2.50 / 1_000_000)
```

- [ ] **Step 2: Run the Task 3 command; see RED** (`13 failed, 1 passed`).
- [ ] **Step 3: Implement.**
  - In `tldw_Server_API/app/core/LLM_Calls/providers/google_adapter.py`, `_normalize_to_openai_shape`, right after the `usage = {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}` dict, add:

```python
            if isinstance(usage_src.get("cachedContentTokenCount"), int):
                # The cached part of prompt_tokens, costed at the cache-read rate (TASK-13936).
                usage["cachedContentTokenCount"] = usage_src["cachedContentTokenCount"]
```

  - Then run the Task 1 script for Google: `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python <scratchpad>/edit_pricing_tiers.py google`. Expected: `google: 10 rows` (the six page models and the four -preview entries). Before writing, the script asserts each row's existing input and output equal the page's (for a -preview entry, its GA model's).
  - It produced, for example:

```json
    "gemini-2.5-pro": { "prompt": 0.00125, "completion": 0.01, "cache_read": 0.000125, "long_prompt": { "above_tokens": 200000, "prompt": 0.0025, "completion": 0.015, "cache_read": 0.00025 } },
    "gemini-2.5-flash": { "prompt": 0.0003, "completion": 0.0025, "cache_read": 0.00003 },
    "gemini-2.5-flash-preview": { "prompt": 0.0003, "completion": 0.0025, "cache_read": 0.00003 },
```

  - Delete the line `    "gemini-2.5-pro-high": { "prompt": 0.0025, "completion": 0.015 },` from `model_pricing.json` (Decisions, "Gemini Pro long tiers"). A saved `gemini-2.5-pro-high` now prices through the substring match to `gemini-2.5-pro` (estimated), whose `long_prompt` carries the same >200K rates (probed: $0.0025 / $0.015 per 1K above 200,000 prompt tokens), and `list_provider_models("google")` no longer offers it.

- [ ] **Step 4: Run; see GREEN.** The Task 3 command measured `14 passed`. Then run `tldw_Server_API/tests/LLM_Adapters/unit`, `tldw_Server_API/tests/LLM_Calls/test_llm_providers.py`, `tldw_Server_API/tests/LLM_Calls/test_gemini_and_hf_regressions.py` and `tldw_Server_API/tests/LLM_Calls/test_provider_response_preservation.py`: measured `770 passed`. A response without `cachedContentTokenCount` keeps its exact old usage shape, so they stay green. Then the other model-list tests, for the deleted entry: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py tldw_Server_API/tests/Setup/test_setup_chat_model_refresh.py tldw_Server_API/tests/Writing/test_llm_providers_tokenizer_metadata.py tldw_Server_API/tests/Usage`: measured `593 passed, 1 skipped` (with Tasks 1-3 applied).
- [ ] **Step 5: Commit** (message 3): `git add` the three files, then `git diff --cached --check`.

## Task 4: Cost OpenRouter calls from the cost OpenRouter reports (AC #2)

**Files:** `tldw_Server_API/app/core/Usage/llm_usage_normalizer.py`, `tldw_Server_API/app/core/Usage/usage_tracker.py`; tests: `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py` (append).

**Interfaces:**
- Produces:
  - `NormalizedLLMUsage.reported_cost_usd: float | None = None`, the new last field. It is set only when the provider name contains `openrouter`.
  - `compute_costs(..., reported_cost_usd: float | None = None)`. When it is not None, it returns `(reported * prompt_share, reported * (1 - prompt_share), reported, False)`, where `prompt_share` is the catalog's prompt share (1.0 when the catalog cost is 0).

**RED reason (measured):** `4 failed, 6 passed`.
- The reported-cost test: `assert 0.016 == 0.0042` (the unknown-model rate).
- BYOK: `assert 0.00103 == 19.95`.
- `AttributeError: 'NormalizedLLMUsage' object has no attribute 'reported_cost_usd'`.
- `TypeError: compute_costs() got an unexpected keyword argument 'reported_cost_usd'`.
- The 6 fallback cases pass already; they are the guard.

- [ ] **Step 1: Append to `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py`** (two blank lines before it):

```python
# --- OpenRouter's reported cost (AC #2) ---


async def test_an_openrouter_call_is_costed_from_the_cost_openrouter_reports(costed_rows: list[dict[str, Any]]) -> None:
    """An uncatalogued model is not costed at the unknown-model rate when OpenRouter reports the call's cost."""
    usage = {"prompt_tokens": 1_000, "completion_tokens": 200, "total_tokens": 1_200, "cost": 0.0042, "is_byok": False}

    await _log("openrouter", "acme/brand-new-model", usage, 1_000, 200)

    (row,) = costed_rows
    assert row["total_cost_usd"] == pytest.approx(0.0042)
    assert row["prompt_cost_usd"] + row["completion_cost_usd"] == pytest.approx(0.0042)
    assert row["estimated"] is False


async def test_an_openrouter_byok_call_adds_the_upstream_provider_charge(costed_rows: list[dict[str, Any]]) -> None:
    """With BYOK, cost is OpenRouter's fee and cost_details.upstream_inference_cost what the provider charged."""
    usage = {
        "prompt_tokens": 194,
        "completion_tokens": 2,
        "total_tokens": 196,
        "cost": 0.95,
        "is_byok": True,
        "cost_details": {"upstream_inference_cost": 19},
    }

    await _log("openrouter", "openai/gpt-5.5", usage, 194, 2)

    (row,) = costed_rows
    assert row["total_cost_usd"] == pytest.approx(19.95)


@pytest.mark.parametrize("cost", [None, "0.5", -1, float("nan"), float("inf"), True])
async def test_an_openrouter_call_without_a_usable_cost_falls_back_to_the_catalog(
    costed_rows: list[dict[str, Any]], cost: Any
) -> None:
    usage = {"prompt_tokens": 1_000, "completion_tokens": 200, "total_tokens": 1_200, "cost": cost}

    await _log("openrouter", "openai/gpt-5.5", usage, 1_000, 200)

    (row,) = costed_rows
    # The catalogued openrouter/openai/gpt-5.5 rate: $0.005 / $0.03 per 1K.
    assert row["total_cost_usd"] == pytest.approx((1_000 * 0.005 + 200 * 0.03) / 1000)


def test_only_openrouter_usage_reports_a_cost() -> None:
    usage = {"prompt_tokens": 1_000, "completion_tokens": 200, "cost": 0.0042}

    assert normalize_llm_usage(provider="openrouter", usage=usage).reported_cost_usd == pytest.approx(0.0042)
    assert normalize_llm_usage(provider="openai", usage=usage).reported_cost_usd is None


def test_a_reported_cost_is_split_by_the_catalog_shares() -> None:
    prompt_cost, completion_cost, total, estimated = compute_costs(
        "openrouter", "openai/gpt-5.5", 1_000, 200, reported_cost_usd=0.022
    )

    # Catalog: 1,000 x $0.005 + 200 x $0.03 per 1K = $0.005 + $0.006.
    assert (prompt_cost, completion_cost, total) == pytest.approx((0.01, 0.012, 0.022))
    assert estimated is False
```

- [ ] **Step 2: Run the Task 4 command; see RED** (`4 failed, 6 passed`).
- [ ] **Step 3: Implement.**

`tldw_Server_API/app/core/Usage/llm_usage_normalizer.py`:
- In the module docstring, the field it now carries is a cost, so replace the two lines `payloads into stable fields without changing billing rates, persistence schema,` and `or provider request behavior.` with:

```text
payloads into stable fields without changing billing rates, persistence schema,
or provider request behavior. It reads the billing facts a provider reports
(cache tokens, service tier, OpenRouter's own cost); compute_costs prices them.
```

- Add `import math` after `import json`.
- After Task 1's `service_tier: str = ""` field, add:

```python
    # OpenRouter's own bill for the call in USD (its usage.cost), or None.
    reported_cost_usd: float | None = None
```

- After Task 1's `service_tier = _usage_str(...)` line, add:

```python
    reported_cost_usd = _openrouter_reported_cost(usage_mapping) if "openrouter" in provider_key else None
```

- Pass `reported_cost_usd=reported_cost_usd,` to the returned `NormalizedLLMUsage(...)`.
- Add these two functions after `usage_with_service_tier`:

```python
def _openrouter_reported_cost(usage: Mapping[str, Any] | None) -> float | None:
    """OpenRouter's ``usage.cost`` in USD (its credits are dollars), or None when absent or not a number.

    With BYOK, ``cost`` is only OpenRouter's fee; ``cost_details.upstream_inference_cost`` is what the upstream
    provider charged the operator's own key, so it is added.
    """
    cost = _usage_float(usage, "cost")
    if cost is None or not usage:
        return None
    if usage.get("is_byok") is True:
        details = usage.get("cost_details")
        cost += _usage_float(details if isinstance(details, Mapping) else None, "upstream_inference_cost") or 0.0
    return cost


def _usage_float(usage: Mapping[str, Any] | None, key: str) -> float | None:
    """A finite non-negative number at ``key``, or None."""
    value = usage.get(key) if usage else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        return None
    return float(value)
```

`tldw_Server_API/app/core/Usage/usage_tracker.py`:
- In `compute_costs`, after `service_tier: str | None = None,`, add `reported_cost_usd: float | None = None,`.
- Append to the docstring:

```python
    ``reported_cost_usd`` (OpenRouter's own bill) is the total cost, split
    between prompt and completion in the catalog's proportions.
```

- Between `completion_cost = (completion_total / 1000.0) * out_per_1k * multiplier` and `total_cost = prompt_cost + completion_cost`, add:

```python
    if reported_cost_usd is not None:
        # The provider's own bill wins; the catalog only splits it between prompt and completion.
        catalog_total = prompt_cost + completion_cost
        prompt_share = prompt_cost / catalog_total if catalog_total > 0 else 1.0
        total = float(reported_cost_usd)
        return total * prompt_share, total * (1.0 - prompt_share), total, False
```

- In `log_llm_usage`'s `compute_costs(...)` call, after `service_tier=normalized_usage.service_tier,`, add `reported_cost_usd=normalized_usage.reported_cost_usd,`.

- [ ] **Step 4: Run; see GREEN.** The Task 4 command measured `10 passed`. Then the whole new file: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_pricing_service_tiers.py`, measured `127 passed`.
- [ ] **Step 5: Commit** (message 4): `git add` the three files, then `git diff --cached --check`.

## Task 5: Say which tiers and caches are priced (AC #3)

Behaviour is final after Task 4. Neither doc is mirrored under `Docs/Published`, so no refresh is needed. `tests/Docs` runs on the PR, so run it (Step 4).

- [ ] **Step 1: `Docs/Operations/Usage_Quotas.md`.** After the "**Prompt-cache tokens.**" bullet, which ends "only the Messages API reads a streamed Anthropic reply's real usage.", insert these four bullets before "- **Two settings defeat these limits,":

```markdown
- **Service tiers.** A call is costed at the service tier its provider reports, and nothing else selects a tier. OpenAI reports it beside the usage block as `service_tier`: `flex`, or `priority` for Fast (GPT-5.6 and earlier report `priority` even when `fast` is requested; a Fast call downgraded during a traffic ramp reports `default` and is billed standard). The `batch` rates apply to a usage block that reports `batch`, which no documented OpenAI response does today. A caller gets a tier by sending `service_tier` in `extra_body`, since the adapter never sets one. Its cost uses the `service_tiers` rates of the model's entry in `model_pricing.json`, copied per model from OpenAI's Batch, Flex and Fast tables: Batch and Flex are about half the standard price, Fast 1.7x to 2.5x. Anthropic reports the tier inside the usage block. `batch` costs half of every token category (the Batch API discount, which stacks with the cache rates, US-only inference and Haiku 5.5's prompt-length tier). `priority` costs the standard rates: Priority Tier is paid for by a capacity commitment, not per token. Any other tier costs the standard rates: `default`, `standard`, `scale`, a model with no row in that tier's table, or a tier the model's entry has no rates for. So does a prompt over 272,000 tokens on a tier whose table shows no long-context price (gpt-5.5 and gpt-5.4 Fast, gpt-5.5-pro Batch and Flex). The tier is kept in the row's `raw_usage_metadata_json`. No server code calls a Batch API, so a batch tier is costed only when a usage block reports it, for example a Messages API reply from a proxy. Fast mode on Anthropic is `speed`, not a tier (see above).
- **Gemini cache reads.** Gemini cached input (`cachedContentTokenCount`, from implicit caching or a `cachedContent` reference) costs the model's context-caching rate on gemini-3.1-pro-preview, gemini-3.1-flash-lite, gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash and gemini-2.5-flash-lite, and on the four -preview entries of 2.5 Flash and Flash-Lite, which are priced like their GA model. On gemini-3.1-pro-preview and gemini-2.5-pro, a prompt over 200,000 tokens pays the higher `long_prompt` rates. The 2.0 Flash entries are not on Google's pricing page and have no cache rate, so their cached tokens cost the input rate. The per-hour storage charge of an explicit cache is not costed: the cache is created outside the server and billed by the hour, not per call. Audio input, cached or not, costs the text rate. Thinking tokens are not costed yet: Google bills them as output, but the usage record counts only the reply's candidate tokens (`candidatesTokenCount`, not `thoughtsTokenCount`), so a Gemini 2.5 reply with thinking on (the default on 2.5 Pro and Flash) is under-billed.
- **OpenRouter.** An OpenRouter call is costed from the cost OpenRouter reports in its usage block (`usage.cost`, in dollars; OpenRouter always sends it, so no request flag is needed). With BYOK, the upstream provider's charge (`cost_details.upstream_inference_cost`) is added to it. A reply without a usable `cost` costs the model's catalogued rate, or the unknown-model rate when the model is not in the catalog. The cost is split between `prompt_cost_usd` and `completion_cost_usd` in the catalog's proportions.
- **Streamed replies and router calls use the standard rate.** A streamed `/chat/completions` reply is costed from an estimate at the standard rate: no tier, cache rate or reported cost applies. So is a streamed Gemini reply through any caller. Other callers' streams are costed from the provider's last usage chunk when it sends one: OpenRouter always does, OpenAI only when the caller sets `stream_options` in `extra_body`. An auto-router call (the model choice made before a `/chat/completions` or character chat reply) is recorded from the provider's token counts without its usage block, so it too costs the standard rate.
```

  Then, in the "**Prompt-cache tokens.**" bullet (`Usage_Quotas.md:108`), replace the text from "so it must repeat" through "lists the model." (both appear once in the file) with:

```markdown
so it must repeat `cache_read`, `cache_write`, `cache_write_1h`, `long_prompt`, `hidden` and `service_tiers` where the shipped entry has them. A missing `cache_write_1h` falls back to the `cache_write` rate, a missing `long_prompt` drops the tier, a missing `hidden` lists the model, and a missing `service_tiers` costs every service tier at the standard rates.
```

- [ ] **Step 2: `Docs/Providers/Model_Pricing_Catalog.md`.**
  - After the "- Format: ..." bullet (it ends "silently loses the tier."), insert:

```markdown
- Service tiers: an optional `service_tiers` object maps a tier name, as the provider reports it in `service_tier` (OpenAI: `flex`, or `priority` for Fast; `batch` for a usage block that reports it, which no documented OpenAI response does), to a full rate set: `prompt`, `completion`, optional cache rates and an optional `long_prompt`. A call that reports that tier costs those rates. A call that reports any other tier costs the entry's standard rates. So does a prompt over the entry's `long_prompt.above_tokens` on a tier with no `long_prompt` of its own. A tier without both `prompt` and `completion`, or with a malformed `long_prompt`, is ignored with a warning. Anthropic entries need no `service_tiers`: Anthropic's `batch` halves every category in code (`ANTHROPIC_BATCH_MULTIPLIER`), and adding a `batch` tier to an Anthropic entry would halve it twice. An override must repeat `service_tiers` too.
```

  - Delete the next bullet, `- Example (OpenAI text models only):`, with its blank line and fenced block (`Model_Pricing_Catalog.md:25-36`). Its four OpenAI rates are stale, and the refreshed OpenAI snippet under "Example Snippets by Provider" below replaces it. The "Tip:" line after it stays.
  - Replace the quick link `- OpenAI (text models): https://developers.openai.com/api/docs/pricing` with these three lines:

```markdown
- OpenAI (text models): https://developers.openai.com/api/docs/pricing (Batch, Flex and Fast tables for `service_tiers`)
- Google Gemini: https://ai.google.dev/gemini-api/docs/pricing (`cache_read` is the "Context caching price")
- OpenRouter: entries are a fallback only; a call is costed from the `usage.cost` OpenRouter reports (https://openrouter.ai/docs/use-cases/usage-accounting)
```

  - In "Example Snippets by Provider":
    - Change the line "OpenAI (text models only – do not include embeddings here):" to "OpenAI (text models only – do not include embeddings here; an override repeats `cache_read` and `service_tiers`):".
    - Replace the four model lines inside that example's fence (`gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `o3-mini`, which carry stale prices) with:

```json
    "gpt-4.1":     { "prompt": 0.002, "completion": 0.008, "cache_read": 0.0005,
                     "service_tiers": { "batch": { "prompt": 0.001, "completion": 0.004 },
                                        "priority": { "prompt": 0.0035, "completion": 0.014, "cache_read": 0.000875 } } },
    "gpt-4o-mini": { "prompt": 0.00015, "completion": 0.0006, "cache_read": 0.000075 },
    "o3-mini":     { "prompt": 0.0011, "completion": 0.0044, "cache_read": 0.00055 }
```

- [ ] **Step 3: The JSON `_catalog_note`** (line 2 of `model_pricing.json`). Append this to the end of its string, after "hidden=true entries are priced but not offered in selectors.":

```text
 OpenAI service_tiers (batch, flex, priority = Fast) are from the Batch, Flex and Fast tables of developers.openai.com/api/docs/pricing, fetched 2026-10-26; a call that reports another tier costs the standard rates. Gemini cache_read (the context caching price) and the 200,000-token long_prompt tiers are from ai.google.dev/gemini-api/docs/pricing, fetched 2026-10-26; the 2.5 -preview entries take their GA model's cache_read, and the 2.0 entries have none (no published price). Anthropic Batch (half price) is applied in code, not here; OpenRouter calls are costed from the usage.cost OpenRouter reports, and these rates are the fallback.
```

Then check that the JSON parses: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -c "import json; json.load(open('tldw_Server_API/Config_Files/model_pricing.json'))"`.

- [ ] **Step 4: Run the docs tests:** `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs`. Measured: `212 passed`.
- [ ] **Step 5: Commit** (message 5): `git add` the three files, then `git diff --cached --check`.

## Task 6: Verify, close TASK-13936, file follow-ups, open the PR

- [ ] **Step 1: Verify.** Run each of these:
  - The focused regression set. Measured `1662 passed, 1 skipped`.
  - The wider set. Measured `976 passed, 14 skipped`; compare with Step 0.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/ci/check_shard_coverage.py`. Expected `new_uncovered=0`.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/ci/test_quality_triage.py --enforce`. Expected exit 0.
  - `git diff origin/dev --check`.
  - The `scope_predicate_baseline.txt` grep from Global Constraints.
- [ ] **Step 2: Bandit** on the touched source, through `uvx` (at `/opt/homebrew/bin/uvx`): `uvx bandit -r tldw_Server_API/app/core/Usage/llm_usage_normalizer.py tldw_Server_API/app/core/Usage/pricing_catalog.py tldw_Server_API/app/core/Usage/usage_tracker.py tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/LLM_Calls/providers/google_adapter.py tldw_Server_API/app/core/Chat/chat_service.py -f json -o <scratchpad>/bandit_13936.json`.
  - The shared venv has no bandit (`No module named bandit`, 2026-10-26). Never install into it: every worktree uses it.
  - If `uvx` cannot fetch bandit, record the environment skip under DoD #4.
  - Fix any new finding in changed lines.
- [ ] **Step 3: Close TASK-13936** (backlog-py, written out):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13936 \
  --append-notes "RED/GREEN (record the real counts): Task 1 84 failed -> 84 passed; Task 2 23 failed, 3 passed -> 26 passed; Task 3 13 failed, 1 passed -> 14 passed; Task 4 4 failed, 6 passed -> 10 passed; new file 127 passed; focused set 1529 passed, 1 skipped on dev -> 1662 passed, 1 skipped; wider set 976 passed, 14 skipped on dev and after (13 need live provider keys, 1 standing TestClient skip); tests/Docs 212 passed." \
  --append-notes "Sources, fetched 2026-10-26: platform.claude.com/docs/en/about-claude/pricing (Batch 50% on input and output, stacks with cache and data residency), platform.claude.com/docs/en/api/service-tiers (Priority Tier is a capacity commitment, no token price), developers.openai.com/api/docs/pricing (Batch, Flex, Fast tables), developers.openai.com/api/docs/guides/priority-processing (GPT-5.6 and earlier report Fast as service_tier priority; a downgraded Fast call reports default), developers.openai.com/api/docs/guides/batch (the output example has no service_tier), ai.google.dev/gemini-api/docs/pricing (context caching prices), ai.google.dev/api/generate-content (promptTokenCount includes the cached content), ai.google.dev/gemini-api/docs/thinking (response pricing is output plus thinking tokens), openrouter.ai/docs/use-cases/usage-accounting and /docs/faq (usage.cost always sent, credits are dollars, BYOK cost is the 5% fee and upstream_inference_cost the provider's charge)." \
  --append-notes "Decisions (owner-approved plan): OpenRouter costed from usage.cost (+ BYOK upstream), catalog fallback; no service_tier DB column (raw_usage_metadata_json keeps it); per-model OpenAI service_tiers, Anthropic batch as a 0.5 code constant; Gemini explicit-cache storage not costed (documented); a long prompt on a tier with no long-context price costs the standard long-context rate." \
  --append-notes "AC #1 scope: every catalogued Gemini text model with a published cache rate carries cache_read: the six on Google's pricing page and the four 2.5 Flash and Flash-Lite -preview entries, which are priced exactly as their GA model. The three 2.0 Flash entries (gemini-2.0-flash, -001, -lite) are deliberately excluded: Google publishes no cache rate for them, so their cached tokens cost the input rate (an over-charge). The gemini-2.5-pro-high pseudo-entry was deleted; gemini-2.5-pro's long_prompt holds its price." \
  --append-notes "Not verified (pages do not say): gpt-5.5-pro Batch/Flex and gpt-5.5/gpt-5.4 Fast long-context prices (the pricing page shows '-', the gpt-5.5-pro model page gives none; costed at the standard long-context rate); the service_tier a model after GPT-5.6 reports for Fast (none is catalogued; TASK-13915 note). Known gap, out of scope: Gemini thinking tokens are not costed (the adapter drops thoughtsTokenCount; follow-up filed in Step 4). The Anthropic adapter does not carry service_tier: no caller gets batch on that path and priority costs standard." \
  --append-notes "Docs updated: Docs/Operations/Usage_Quotas.md (four bullets: service tiers, Gemini cache reads, OpenRouter, streamed replies and router calls; the override sentence names service_tiers), Docs/Providers/Model_Pricing_Catalog.md (service_tiers format, links, OpenAI example; stale format example removed) and the model_pricing.json _catalog_note. Neither doc is mirrored under Docs/Published. Bandit: <result or environment skip>." \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Calls are costed at the service tier the provider reports: OpenAI Batch, Flex and Fast from per-model catalog rates (service_tiers, written from OpenAI's tables), Anthropic Batch at half price on every token category; any other tier costs the standard rate. The tier is folded into the usage record by usage_with_service_tier and kept in raw_usage_metadata_json (no schema change). Gemini cached input costs the context-caching rate (the adapter keeps cachedContentTokenCount; ten Gemini entries carry cache_read, the two Pro models a 200K long_prompt; the 2.0 Flash entries have no published cache rate). OpenRouter calls are costed from the usage.cost OpenRouter reports, plus the upstream charge with BYOK, with the catalog as fallback. Usage_Quotas.md says which tiers and caches are priced and what is not (streamed /chat/completions, router calls, Gemini storage, audio and thinking tokens)." \
  -s Done
```

  Fill in `<result or environment skip>` from Step 2 before running. `--check-dod 4` is for Bandit, run or skipped with a recorded reason.

- [ ] **Step 4: File the follow-ups under the id lock.**
  - Take the lock with `mkdir <scratchpad>/id_lock`. If it exists, wait; never break another session's lock.
  - Create `<scratchpad>/task_next.sh`, verbatim from `Docs/superpowers/plans/2026-10-22-chacha-off-loop-and-transaction-guard.md` (Task 5 Step 4): it prints the highest backlog id on origin/dev and in every open PR.
  - Run `git fetch origin dev --quiet && zsh <scratchpad>/task_next.sh`. Take `<N1>` and `<N2>` as the printed value + 1 and + 2 (and `<N3>` as + 3 if Decision 4 (c) was chosen), and check that no `task-<N>` file exists locally. Write each id literally in the commands below.
  - Release the lock after `gh pr view --json files` lists the new task files (Step 6).

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "A streamed /chat/completions reply is costed from an estimate whatever the provider reports" \
  --id TASK-<N1> -l quotas,backend \
  -d "Found by TASK-13936. chat_service records a streamed /chat/completions reply after the stream ends from an estimate (prompt estimate plus len(reply)/4, estimate_source stream_estimate, no usage_metadata; execute_streaming_call's save_callback in chat_service.py), and the call records itself, so the provider-level recorder that reads the provider's last usage chunk never runs. So a streamed reply is costed at the standard rate with no service tier (OpenAI Flex, Fast, Batch), no cache rates, no Gemini cache reads and no OpenRouter usage.cost, whatever the provider reports. TASK-13916 fixes the Anthropic adapter's stream usage; that alone does not change /chat/completions, which uses this recorder. Usage_Quotas.md documents the gap." \
  --ac "A streamed /chat/completions reply whose provider sends a usage chunk is recorded with that usage (service tier, cache tokens, reported cost), with a test" \
  --ac "A stream without a usage chunk is still recorded from the estimate"
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Gemini pricing gaps: thinking tokens are not costed; Batch/Flex/Priority tiers, audio rates, stream usage" \
  --id TASK-<N2> -l quotas,backend,high \
  -d "Found by TASK-13936 (ai.google.dev, fetched 2026-10-26). High priority because of (1). (1) Thinking tokens are under-billed: GoogleAdapter._normalize_to_openai_shape sets completion_tokens from candidatesTokenCount only and drops thoughtsTokenCount, but thinking tokens are counted apart from the candidates and billed as output. Sources: the thinking guide (ai.google.dev/gemini-api/docs/thinking) says 'When thinking is turned on, response pricing is the sum of output tokens and thinking tokens'; the pricing page's 2.5 output label is 'Output price (including thinking tokens)'; the repo fixture in tests/Chat/unit/test_chat_service_content.py has totalTokenCount 115 = promptTokenCount 80 + candidatesTokenCount 30 + thoughtsTokenCount 5. Thinking is on by default on gemini-2.5-pro and gemini-2.5-flash, so every such reply's thinking is free in the usage record. Fix: add thoughtsTokenCount to completion_tokens. (2) The pricing page has Batch, Flex and Priority tables but says nothing about how a tier is selected or reported; find the response field, then add service_tiers to the Gemini entries. (3) Audio input and cached audio have their own prices; the catalog has text rates only. (4) gemini-2.0-flash, -2.0-flash-001 and -2.0-flash-lite are not on the pricing page and have no cache rate (Google publishes none). (5) The Google adapter's stream (GoogleAdapter.stream) emits no usage chunk, so a streamed Gemini reply through any caller (RAG, workflows, /chat/completions) is costed from an estimate. promptTokenCount including cachedContentTokenCount is confirmed by the API reference (UsageMetadata), as the normalizer assumes." \
  --ac "Gemini thinking tokens (thoughtsTokenCount) are counted and costed as output tokens, with a test" \
  --ac "A streamed Gemini reply is recorded from the provider's usage when the stream reports it, with a test" \
  --ac "Each remaining item is priced with a test or recorded as not priced in Usage_Quotas.md"
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13915 \
  --append-notes "From TASK-13936 (2026-10-26): when the gpt-6* and gpt-5.6* models are catalogued, give them service_tiers too (their Batch, Flex and Fast rows include 30-minute and 12-hour write columns), and price the Ultrafast tier the page lists for gpt-6-astra and gpt-6.1-sol. Key their Fast rates by the service_tier the response reports: OpenAI's Fast-mode guide (developers.openai.com/api/docs/guides/priority-processing) says 'For GPT-5.6 and earlier models, the response returns priority whether the request specifies priority or fast', so gpt-5.6* stay keyed priority; the guide (2026-10-26) does not say what gpt-6* report, likely fast, so check it then and key those rates by the value it gives."
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13916 \
  --append-notes "From TASK-13936 (2026-10-26): fixing the adapter's stream usage does not change /chat/completions, whose streamed replies chat_service records from an estimate; that is TASK-<N1>."
```

  - Then run `git add -A backlog/tasks && git diff --cached --check` and commit (message 6).
  - If the owner chose Decision 4 (c), also file the Gemini explicit-cache storage task the same way (TASK-<N3>).

- [ ] **Step 5: Open the PR.**
  - `git push -u origin fix/pricing-service-tiers-13936`.
  - Then `gh pr create --base dev --title "fix(usage): price service tiers, Gemini cache reads and OpenRouter's reported cost (TASK-13936)"`.
  - The body states:
    - what is priced: OpenAI tiers per model, Anthropic Batch 0.5x, Gemini cache reads and Pro long tiers, OpenRouter reported cost;
    - what is not: a streamed `/chat/completions` reply, router calls, Gemini storage, audio and thinking tokens (the last is an under-billing, filed as TASK-<N2>);
    - the AC #1 scope: every catalogued Gemini model with a published cache rate, so not the three 2.0 Flash entries;
    - the five decisions;
    - that no schema changed.
  - The body ends with the attribution line from the session's reminder.
  - **ADR-004 Change summary:** TASK-13936's PR is not in the owner's recorded waivers (they cover the PR that filed TASK-13936, not this one). Ask the owner for the "## Change summary" section before merge.
  - Editing the PR body re-triggers the required license status, so wait for it before merging.
- [ ] **Step 6: Review and merge.**
  - Run `/superpowers:requesting-code-review` in a clean subagent (the owner's rule while Qodo credits are out) and hold until findings are addressed.
  - Check `gh variable get MERGE_QUEUE`: with `on`, arm auto-merge and leave the PR alone; otherwise rebase only when it is next and merge after the seven statuses are green.
  - Confirm MERGED before deleting the branch.
  - Release `<scratchpad>/id_lock` once the PR's file list shows the new task files.

## Follow-ups to file (Task 6 Step 4)

- **F1:** A streamed `/chat/completions` reply is costed from an estimate whatever the provider reports. New task TASK-<N1>, plus a note on TASK-13916.
- **F2 (label `high`):** Gemini pricing gaps: thinking tokens not costed (a known under-billing), Batch/Flex/Priority, audio, the 2.0 Flash entries, the Google adapter's stream emitting no usage chunk. New task TASK-<N2>.
- **Notes on existing tasks:** TASK-13915 gets the tiers and Ultrafast of the uncatalogued gpt-6 and gpt-5.6 models, and how to key their Fast rates.
- **Only if the owner picks Decision 4 (c):** Gemini explicit-cache storage overhead.

## ADR assessment

No new ADR. The change adds an optional catalog key (`service_tiers`), one code multiplier, two normalizer fields and a usage-block helper inside the existing pipeline. It flips no global default; signatures gain only keyword-only parameters with defaults, and the wider set covers every test file that names them.

Governing decisions: ADR-075 (Anthropic-shaped `prompt_tokens` includes cache tokens, which is why the Gemini and long-prompt tiers use the whole prompt) and ADR-059 (backlog-py). `Docs/ADR` was searched for pricing: only ADR-075 matches, and nothing in it forbids code-constant multipliers.

## Self-review against the spec

- AC #1 (Gemini cache-read rate for the catalogued Gemini models, with a test): Task 3. Every catalogued Gemini text model with a published cache rate carries `cache_read`: the six on Google's page and the four 2.5 -preview entries priced like their GA model. The three 2.0 Flash entries are excluded because Google publishes no cache rate for them (recorded in the close-out note and the PR body). The adapter keeps the count; four tests.
- AC #2 (OpenRouter from reported cost, with a test): Task 4, five tests.
- AC #3 (`Usage_Quotas.md` says which tiers and caches are priced): Task 5 Step 1.
- AC #4 (Anthropic batch, with a test): Task 2, the 19-model table test plus the native reply and stream.
- AC #5 (OpenAI flex, priority, batch, a test per tier): Task 1, the 56-case catalog test plus the costed test per tier through `log_llm_usage`, and the three recorder tests. `priority` is the value OpenAI's Fast-mode guide gives for Fast on GPT-5.6 and earlier.
- AC #6 (unpriced tier costs standard, with a test): Task 1, `test_a_service_tier_the_catalog_has_no_rate_for_costs_the_standard_rate` (7 values) and `test_a_tier_a_model_has_no_row_for_costs_the_standard_rate`.
- The description's "carry `service_tier` into the usage record": Task 1 (normalizer field, `usage_with_service_tier`, the three recorders, `raw_usage_metadata_json` assertion).
