# Anthropic Prompt-Cache Pricing and Adapter Cache Fields Implementation Plan

> **Review folded in (2026-10-20).** An independent review and dry run (`review_plan_fu_anthropic_cache.md`) found 0 Blockers, 3 Important and 6 Minor findings, all folded in:
> - I1: the current Anthropic models get entries in Task 1, priced from Anthropic's pages re-fetched on 2026-10-20, with a catalog rate test.
> - I2: all four consumers of the larger token counts are named.
> - I3: the `Usage_Quotas.md` bullet is corrected, and the override trap is documented in `Model_Pricing_Catalog.md`.
> - M1-M6: the 2x option in Decision 1, the eight extra sweep files, a line number, two expected outputs, the router-usage loggers, and serial execution.
>
> The RED and GREEN counts below were re-measured in a scratch worktree after the fold.
>
> **Re-review folded in (2026-10-20).** A scoped re-review (`rereview_plan_fu_rg_and_anthropic.md`) found it ready, with 3 Minor findings, all folded in:
> - M1: `Model_Pricing_Catalog.md`'s Anthropic example carries the cache keys on API IDs, and links the pricing page;
> - M2: Decision 3 names the provider inference the new entries change;
> - M3: the model-page URLs are written out.
>
> No RED or GREEN count changed; the Docs tests were re-run on the changed doc.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Anthropic prompt-cache reads and writes are costed at Anthropic's cache rates, and the current Anthropic models are costed at their own list prices instead of the unknown-model rate or Opus 4's. A prompt-cached call through the Anthropic adapter (`/chat/completions` and every other adapter caller) counts its cached input as prompt tokens, as the native `/v1/messages` path already does (TASK-13858).

**Architecture:**
- **Data only for pricing.**
  - The pricing catalog already parses `cache_read` and `cache_write` from `model_pricing.json`.
  - The usage normalizer already splits Anthropic usage into cache reads, cache writes and the uncached remainder.
  - `compute_costs` already prices each at its own rate when the catalog has one.
  - So this plan adds the two rates to the 14 Anthropic entries whose model is on Anthropic's price list. It also adds 8 entries for current models that had none, each with all four rates (Decision 3).
- **The adapter's usage block.** `AnthropicAdapter._normalize_to_openai_shape` adds `cache_creation_input_tokens` and `cache_read_input_tokens` to `prompt_tokens` (OpenAI's meaning, as ADR-075 records for the native path) and keeps both fields in the block. Every recorder, the chat pipeline's and the provider base class's, then costs the call correctly with no change of its own.
- **No schema change.** `llm_usage_log` already stores the summed `prompt_tokens` and the cache counts separately, and the cost is computed when the row is written.

**Tech Stack:** Python, `Config_Files/model_pricing.json`, `core/Usage` (`pricing_catalog`, `llm_usage_normalizer`, `usage_tracker`), `LLM_Calls/providers/anthropic_adapter.py`, pytest.

**Spec:** `backlog/tasks/task-13858 - Price-Anthropic-prompt-cache-tokens-and-keep-cache-fields-on-the-Anthropic-adapter-path.md`:
- AC #1: "model_pricing.json gives the Anthropic models cache_read and cache_write rates, and a native /v1/messages row with cache fields is costed with them, with a test".
- AC #2: "The Anthropic adapter keeps the cache fields in its usage block and counts them in prompt_tokens, with a test".

## What was verified

Measured on `origin/dev` `2b925562e5`.

- **`compute_costs` already prices cache tokens** (`core/Usage/usage_tracker.py:235-275`).
  - **Inputs.** It takes `cache_read_input_tokens`, `cache_write_input_tokens` and `billable_input_tokens`.
  - **When the catalog has a `cache_read` or `cache_write` rate:** `prompt_cost = normal × prompt + read × cache_read + write × cache_write`.
    - `normal` is `billable_input_tokens`, capped at `prompt − read − write`.
    - A missing one of the two rates falls back to the prompt rate.
  - **Otherwise** the whole `prompt_tokens` is priced at the prompt rate.
  - `tests/Usage/test_pricing_catalog.py` already covers both branches with an `openai/cached-test` override.
- **The catalog already carries the rates** (`core/Usage/pricing_catalog.py`).
  - `_merge_overrides` (lines 207-257) reads `cache_read` (or `cache-read`, `cached`, `cached_input`, `cache_read_input`, `cached_input_tokens`) and `cache_write` (or `cache-write`, `cache_creation`, `cache_creation_input`, `cache_write_input`, `cache_write_input_tokens`), in USD per 1K tokens like the other rates.
  - `get_rate_details` and `_entry_rate_details` return them (lines 259-303).
  - `model_pricing.json` has no `cache_read` or `cache_write` anywhere, which is the gap.
- **The row stores the cache counts separately, so no migration is needed.**
  - `log_llm_usage` (`usage_tracker.py:278-511`) normalizes the provider usage, costs it, and writes the summed `prompt_tokens` together with `cached_input_tokens`, `cache_write_input_tokens`, `cache_read_input_tokens` and `billable_input_tokens`.
  - Both schemas have those columns:
    - SQLite: `AuthNZ/migrations.py`, the `llm_usage_log` table and migration 088;
    - Postgres: `AuthNZ/pg_migrations_extra.py:1952-1953` and the `ADD COLUMN IF NOT EXISTS` at 1971-1972.
  - `tests/AuthNZ_SQLite/test_authnz_llm_usage_log_router_columns_sqlite.py` and its Postgres twin pin the columns.
  - Costs are fixed when a row is written, so rows already written keep their cost.
- **The native `/v1/messages` row is overcharged today.**
  - A Sonnet 4.6 reply with `input_tokens: 3`, `cache_creation_input_tokens: 2000`, `cache_read_input_tokens: 60000` and `output_tokens: 100` is recorded with `prompt_tokens: 62003` (TASK-13610) and `billable_input_tokens: 3`.
  - `compute_costs` returns a prompt cost of `0.186009` (62003 tokens at the input rate). With Anthropic's rates it is `0.025509`, about 7.3 times less.
- **The Anthropic adapter drops the cache fields** (`anthropic_adapter.py:373-379`).
  - Its usage block is `{prompt_tokens: input_tokens, completion_tokens, total_tokens}`, so the same reply through `/chat/completions` records `prompt_tokens: 3`.
  - The chat pipeline's recorder (`Chat/chat_service.py:6811-6877`, `_record_dict_response_usage`: `normalize_llm_usage(provider, usage)`, then `log_llm_usage(..., usage_metadata=usage)`) and the provider base class's (`LLM_Calls/usage_accounting.py`, `_usage_tokens`) both read `prompt_tokens` from that block.
  - Once the block carries the sum and the two cache fields, both cost the call correctly unchanged. The normalizer reads `cache_creation_input_tokens` and `cache_read_input_tokens` for any provider name containing `anthropic`, and `billable = prompt − read − write`.
- **What pins the current block shape: nothing directly.**
  - `tests/LLM_Calls/test_llm_providers.py:1820` compares against `_normalize_to_openai_shape` itself, on a payload with no cache fields.
  - `test_provider_response_preservation.py` checks only `provider_response`.
  - `test_adapters_chat_endpoint_anthropic_native.py` does not assert on usage.
  - No doc describes the adapter's block. ADR-075's consequence "Anthropic-shaped usage counts cached input" describes the native path, and this plan extends the same rule to the adapter.
  - A response with no cache fields keeps its exact current shape. Anthropic usually sends both fields, often as 0; they then appear in the block as 0 and `prompt_tokens` is unchanged.
- **Anthropic's prices.** Source: `https://platform.claude.com/docs/en/about-claude/pricing.md`, fetched twice on 2026-10-20 (when drafting, and again for the review fold). Model IDs, all fetched 2026-10-20:
  - `https://platform.claude.com/docs/en/about-claude/models/overview.md`;
  - `https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions.md`;
  - `https://platform.claude.com/docs/en/models/fable-5/overview.md`;
  - `https://platform.claude.com/docs/en/models/opus-4-5/overview.md`.
  - The "Prompt caching" table:
    - a 5-minute cache write is **1.25x** base input;
    - a 1-hour cache write is **2x**;
    - a cache read is **0.1x**, except 0.025x on Claude Fable 5.1 and Mythos 5.1, and 0.05x on Claude Opus 5.5 and Sonnet 5.5.
  - The model table gives, in $/MTok, input / 5-minute write / read / output:
    - Fable 5.1: 10 / 12.50 / 0.25 / 50; Fable 5: 10 / 12.50 / 1 / 50;
    - Opus 5.5: 4 / 5 / 0.20 / 20;
    - Opus 5, 4.8, 4.7, 4.6 and 4.5: 5 / 6.25 / 0.50 / 25;
    - Opus 4.1 and 4: 15 / 18.75 / 1.50 / 75;
    - Sonnet 5.5: 2 / 2.50 / 0.10 / 10; Sonnet 5: 2 / 2.50 / 0.20 / 10;
    - Sonnet 4.6, 4.5 and 4: 3 / 3.75 / 0.30 / 15;
    - Haiku 4.5: 1 / 1.25 / 0.10 / 5; Haiku 3.5: 0.80 / 1 / 0.08 / 4.
  - **Flat prices.** The page's long-context section says Claude 4.6 and later models, except Haiku 5.5, bill the whole 1M window at standard prices. Haiku 5.5 alone is priced by prompt length (above or below 100,000 tokens), which a flat catalog entry cannot express.
  - **API IDs.** `claude-fable-5-1`, `claude-opus-5-5` and `claude-sonnet-5-5` (overview); `claude-opus-4-8`, `claude-opus-5` and `claude-sonnet-5` (model IDs page); `claude-fable-5` (its page); Opus 4.5 is `claude-opus-4-5-20251101`, with the alias `claude-opus-4-5` (its page).
  - Every rate in this plan is one of those values divided by 1000 (per 1K tokens).
  - The page does not list Sonnet 3.7, `claude-3-5-sonnet-latest` or the Claude 3 models, so they get no cache rates (see Global Constraints).
- **1-hour writes happen.** The server asks for a 1-hour cache when a cache intent sets `ttl: "1h"` or `ttl_seconds >= 3600` (`LLM_Calls/cache_intents.py:219-225`), and a Messages API client can ask for it directly. Anthropic reports the split in `usage.cache_creation.ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens`. See Decision 1.
- **OpenAI's `cached_tokens` need no adapter change.**
  - OpenAI's `prompt_tokens` already include `prompt_tokens_details.cached_tokens`.
  - The normalizer already maps them to `cache_read_input_tokens` with `billable = prompt − cached` (`tests/Usage/test_llm_usage_normalizer.py::test_normalize_openai_usage_extracts_cache_read_and_reasoning_tokens`).
  - Only the rates are missing, so cached OpenAI input is costed at the full input rate: an overcharge, the conservative direction. See Decision 2.
- **The catalog misprices most current models**, measured with `PricingCatalog().get_rate_details("anthropic", ...)`:
  - **At the unknown-model rate** (0.01 and 0.03 per 1K, no cache rates): `claude-opus-5-5`, `claude-opus-5`, `claude-sonnet-5-5`, `claude-sonnet-5`, `claude-fable-5-1`, `claude-fable-5` and `claude-haiku-5-5`, plus the retired `claude-3-5-haiku-latest` and `claude-3-7-sonnet-20250219`. Fable's output then costs $30 per MTok against Anthropic's $50, an undercharge.
  - **At Opus 4's rate, 3x too high:** `claude-opus-4-8`, the alias `claude-opus-4-5` and `claude-opus-4-5-20251101`. Each matches the `claude-opus-4` key by substring, because the catalog's Opus 4.5 key is the dotted `claude-opus-4.5`, which no API ID contains.
  - **Task 2 makes this worse.** For the example reply (3 uncached, 2,000 written, 60,000 read, 100 output):
    - today a cached `claude-opus-5-5` reply through `/chat/completions` records 3 prompt tokens and costs about $0.003;
    - after Task 2 alone it records 62,003 tokens at the unknown rate: $0.623, against Anthropic's $0.024 (26x);
    - for `claude-sonnet-5-5` it is $0.623 against $0.012 (52x).
    - The native `/v1/messages` path has costed this way since TASK-13610.
  - See Decision 3, whose recommendation Task 1 implements.
- **Four consumers count the larger `prompt_tokens` and `total_tokens` after Task 2.** None double counts, but each now counts cached tokens in full, as ADR-075 defines `prompt_tokens`:
  - `limits.llm_tokens_per_month`, from `llm_usage_log` rows;
  - the org billing plan's `llm_tokens_month`: `api/v1/endpoints/chat.py:5947-5954` calls `_billing_enforcer.record_actual(total)`, and `Billing/enforcement.py:371` (`_get_llm_tokens_month`) sums `llm_usage_log`;
  - the Resource Governor token commit on `/chat/completions`: `chat.py:5931-5945` commits `actuals={"tokens": total_tokens}`. The reservation is already the whole-prompt estimate, so the actual now matches it instead of falling far below. Anthropic itself does not count cache reads toward most of its input rate limits, so cached conversations are throttled harder here than upstream;
  - Prometheus `llm_tokens_used_total{type="prompt"}` (`usage_tracker.py:386-402`) and `metrics.track_tokens` (`chat_service.py:6833`).

## Global Constraints

- **Branch:** `fix/anthropic-cache-token-pricing` from `origin/dev`. The PR targets `dev`. One PR.
- **Rates, verbatim, per 1K tokens** (from the pricing page above), in five model groups:
  - **Opus 4.7, 4.6 and 4.5** (`claude-opus-4-7`, `claude-opus-4-7-20260103`, `claude-opus-4-6`, `claude-opus-4.5`): `"cache_read": 0.0005, "cache_write": 0.00625`.
  - **Opus 4.1 and 4** (`claude-opus-4.1`, `claude-opus-4`): `"cache_read": 0.0015, "cache_write": 0.01875`.
  - **Sonnet 4.6, 4.5 and 4** (`claude-sonnet-4-6`, `claude-sonnet-4-6-20260105`, `claude-sonnet-4.5`, `claude-sonnet-4`): `"cache_read": 0.0003, "cache_write": 0.00375`.
  - **Haiku 4.5** (`claude-haiku-4-5`, `claude-haiku-4-5-20251001`, `claude-haiku-4.5`): `"cache_read": 0.0001, "cache_write": 0.00125`.
  - **Haiku 3.5** (`claude-haiku-3.5`): `"cache_read": 0.00008, "cache_write": 0.001`.
- **New entries, verbatim, per 1K tokens** (Decision 3), each `prompt`, `completion`, `cache_read`, `cache_write`:
  - `claude-fable-5-1`: 0.01, 0.05, 0.00025 (0.025x), 0.0125;
  - `claude-fable-5`: 0.01, 0.05, 0.001, 0.0125;
  - `claude-opus-5-5`: 0.004, 0.02, 0.0002 (0.05x), 0.005;
  - `claude-opus-5`, `claude-opus-4-8` and `claude-opus-4-5`: 0.005, 0.025, 0.0005, 0.00625. The dash key `claude-opus-4-5` is the API alias, and it outranks `claude-opus-4` in the substring match for `claude-opus-4-5-20251101`;
  - `claude-sonnet-5-5`: 0.002, 0.01, 0.0001 (0.05x), 0.0025;
  - `claude-sonnet-5`: 0.002, 0.01, 0.0002, 0.0025.
- **Entries left without cache rates**, because the price list no longer shows them: `claude-sonnet-3.7`, `claude-3-5-sonnet-latest`, `claude-3-opus-20240229`, `claude-3-sonnet-20240229` and `claude-3-haiku-20240307`. The four placeholders are left as well. Do not invent per-model prices.
- **Left to the follow-up (Task 3, Step 5):**
  - Haiku 5.5, priced by prompt length;
  - Mythos 5.1 and 5, which have limited availability;
  - Haiku 3.5's API ID `claude-3-5-haiku-20241022`, which is retired except on Google Cloud;
  - the dotted keys (`claude-opus-4.5`, `claude-opus-4.1`, `claude-sonnet-4.5`, `claude-haiku-4.5`), which match no API ID.
- **`cache_write` is the 5-minute rate (1.25x)** (Decision 1).
- **`_catalog_note`** gains one sentence naming the source and fetch date of the Anthropic prices, and the three read multipliers (0.1x, 0.05x, 0.025x).
- **Adapter:** cache fields are kept only when they are integers, and `prompt_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens`. A block with no cache fields keeps its current shape exactly.
- **Out of scope:**
  - `DEFAULT_PRICING` in `pricing_catalog.py`, whose Anthropic entries `model_pricing.json` all overrides;
  - OpenAI rates (Decision 2);
  - the adapter's streaming path (Review Focus 1);
  - repricing rows already written.
- **Test first:** each task's RED run happens before its source change, and fails for the reason given.
- **Test-quality ratchet:** `Helper_Scripts/ci/test_quality_triage.py --enforce` stays at `OK`.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit ids, `git add -A backlog/tasks`.
- **Docs:** `Docs/Operations/Usage_Quotas.md` and `Docs/Providers/Model_Pricing_Catalog.md`, neither mirrored under `Docs/Published`. ADR-075 is accepted and immutable, and is not edited.
- **Execution order:** run this plan and `plan_fu_keep_artifact_on_refusal.md` one after the other, each opening its PR before the next computes its follow-up ids. Both file follow-ups by the same max-id rule, and both edit `Usage_Quotas.md`, though in different hunks.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never `git stash`.
- **Before merge:**
  - The PR needs the owner's Change summary (ADR-004).
  - A clean-subagent `/superpowers:requesting-code-review` must run.
  - Check `gh variable get MERGE_QUEUE` for the merge procedure.

## Decisions for the owner

1. **1-hour cache writes are costed at the 5-minute rate.**
   - **Recommended:** accept it for this PR, and file the follow-up in Task 3, Step 5. That follow-up reads `usage.cache_creation.ephemeral_1h_input_tokens` and prices it at a new `cache_write_1h` rate (2x).
   - **The cost of waiting:** a 1-hour write is undercounted by 0.75x input per token until then. The default TTL is 5 minutes, so only explicit 1-hour caches are affected.
   - **Alternative A, conservative:** set `cache_write` to the 2x rate until the split lands. That is data only. It overcharges the common 5-minute write by 0.75x input, instead of undercharging the rare 1-hour write by 0.75x.
   - **Alternative B:** do the split here. That needs a normalizer field, a `compute_costs` argument, a catalog key and the adapter keeping the nested `cache_creation` object. It is more than the AC asks.
2. **OpenAI's cached input.**
   - **Recommended:** keep it out of this PR and file the follow-up in Task 3, Step 5.
   - The code path already handles it. Only rates are missing, which must come from OpenAI's published prices. Until then cached input is overcharged at the full input rate.
3. **Price the current Anthropic models in this PR.**
   - **Recommended:** add the 8 flat-priced entries in Task 1, as Global Constraints lists them, with the parametrized catalog test. Leave Haiku 5.5, Mythos, Haiku 3.5's real ID and the dotted keys to the follow-up.
   - **The cost of waiting:** after Task 2, a cached `/chat/completions` reply on Opus 5.5 costs 26x what Anthropic bills, and on Sonnet 5.5 52x. Opus 4.8 and Opus 4.5 cost 3x. With `limits.llm_cost_per_*` set, a user on those models reaches a cost limit that much sooner. Fable output is undercharged.
   - **What users see:** the catalog seeds the model lists of `GET /api/v1/llm/providers` (`list_provider_models`), so the 8 new IDs become selectable in the WebUI's model pickers. All 8 are valid API IDs; `claude-opus-4-5` is the API alias of Opus 4.5.
   - **Routing changes too.** A request that names one of the new models without a provider now goes to Anthropic.
     - `chat_service.resolve_provider_and_model` calls `infer_provider_from_model_catalog`, which switches the provider when the model is in exactly one provider's catalog.
     - Its callers are `/chat/completions` (`api/v1/endpoints/chat.py:3653`), the character chat completion endpoint (`character_chat_sessions.py:7128`), the Messages API (`messages.py:1022`) and `chat_target_resolution.py:203`.
     - Measured: on `origin/dev`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-sonnet-5`, `claude-fable-5-1`, `claude-opus-4-8` and `claude-opus-4-5` with the default provider `openai` or `llama.cpp` stay on that provider (`no_catalog_match`). After Task 1 each goes to `anthropic` (`unique_catalog_match`), as `claude-sonnet-4-6` already does on dev.
     - That is an improvement, since those requests could not succeed at the wrong provider, but clients can see it. The PR body names it.
   - **Alternative:** keep this PR to the AC and leave the entries to the follow-up. The 26-52x overcharge then ships with Task 2.

## ADR assessment

- **No new ADR is needed.**
  - **ADR-067** (accounting in the provider base class) and **ADR-075** (native path and MCP tools use the provider recorder) govern who records. This plan changes neither.
  - The adapter change applies ADR-075's consequence "Anthropic-shaped usage counts cached input" (prompt tokens are `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`) to the adapter path.
  - The pricing change fulfils ADR-075's own follow-up: "The pricing catalog has no Anthropic cache read or write rates ... TASK-13858 adds them".
  - **ADR-064**'s cost keys read `llm_usage_log` costs, which become more accurate.

## Review Focus

1. **A streamed `/chat/completions` reply is still estimated, so its cached input costs the input rate.**
   - The chat pipeline records streams with `estimate_source="stream_estimate"`, from the prompt text, for every provider.
   - The adapter's own stream never yields `message_start`, which carries the input and cache counts. So the provider base class's stream recorder (`_StreamUsage`) finds no usage either, and estimates.
   - That overcharges cached streams, the conservative direction. `Usage_Quotas.md` says so. No test; out of scope.
2. **Every token count includes cache reads in full.** After this PR, a heavily cached Anthropic conversation through the adapter counts every cached token toward:
   - `limits.llm_tokens_per_month`;
   - the org plan's `llm_tokens_month`;
   - the Resource Governor's per-minute token commit;
   - the Prometheus token counters.

   The native path already does this, since TASK-13610. A user can reach a token limit sooner than before while spending less. `Usage_Quotas.md` and the PR body say so. Covered by `test_anthropic_adapter_usage_counts_and_keeps_prompt_cache_tokens` (`prompt_tokens == 62003`).
3. **Clients of `/chat/completions` see a larger `prompt_tokens` and two extra usage fields** for Anthropic models. `prompt_tokens` now matches OpenAI's meaning. Clients that summed `prompt_tokens` with the cache fields from `provider_response` would now double count. No test: a client-side concern; noted in the PR body.
4. **Model IDs that match an entry by substring get that entry's rates.**
   - `claude-sonnet-4-5-20250929` matches `claude-sonnet-4`, and `claude-opus-4-1-20250805` matches `claude-opus-4`. Both are priced like their own model, and both are flagged estimated.
   - `claude-opus-4-5-20251101` now matches the new `claude-opus-4-5` key, the longest that matches, instead of `claude-opus-4`.
   - `test_anthropic_model_ids_get_their_list_prices_and_cache_rates` covers 14 dated and undated IDs.
5. **The month's cost counters do not shrink at deploy.** Rows written before this PR keep their overcharged cost, so a user near `limits.llm_cost_per_month_usd` stays near it until the month rolls over. No test.
6. **The model-router usage loggers pass no `usage_metadata`.** These are `log_model_router_usage` (`LLM_Calls/routing/accounting.py`), called from `chat.py:1702-1716`, `chat_route_resolver.py:202-215` and `character_chat_sessions.py:1195-1206`.
   - A cached router call would have its whole, now larger, `prompt_tokens` costed at the input rate.
   - Router prompts are below Anthropic's minimum cacheable length and carry no cache intent, so this is 0 in practice.
   - No change, no test.

---

### Task 1: Price Anthropic cache reads and writes

**Files:**
- Modify: `tldw_Server_API/Config_Files/model_pricing.json:2` (`_catalog_note`), and `:92-115`: the 14 Anthropic entries gain cache rates, and 8 new entries are added, as listed in Global Constraints.
- Test: `tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py`:
  - factor the endpoint stubs out of the `rows` fixture into `_stub_endpoint(monkeypatch)`;
  - add a `costed_rows` fixture, which runs the real `log_llm_usage` with the shipped catalog and captures the repo insert;
  - add `test_a_cached_native_reply_is_costed_at_the_cache_rates`.
- Test: `tldw_Server_API/tests/Usage/test_pricing_catalog.py`, `test_anthropic_model_ids_get_their_list_prices_and_cache_rates` (14 cases: the new models, their aliases, and dated IDs).
- Add: this plan, as `Docs/superpowers/plans/2026-10-20-anthropic-cache-token-pricing.md`.

**Interfaces:**
- Produces: the `cache_read` and `cache_write` keys on those 14 entries, and 8 new entries.
  - `PricingCatalog.get_rate_details("anthropic", "claude-sonnet-4-6")` then returns `{"prompt": 0.003, "completion": 0.015, "cache_read": 0.0003, "cache_write": 0.00375}`.
  - `PricingCatalog.get_rate_details("anthropic", "claude-opus-5-5")` returns `{"prompt": 0.004, "completion": 0.02, "cache_read": 0.0002, "cache_write": 0.005}`.

Set once for every task: `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`, `export TLDW_TEST_NO_DOCKER=1`, and `SCRATCH=` this session's scratchpad directory (scratch files go there, never in `/tmp`).

- [ ] **Step 1: Branch, and claim the task**

```bash
git fetch -q origin dev
git switch -c fix/anthropic-cache-token-pricing origin/dev
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13858 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-10-20-anthropic-cache-token-pricing.md (branch fix/anthropic-cache-token-pricing)."
```

Copy this plan to `Docs/superpowers/plans/2026-10-20-anthropic-cache-token-pricing.md`.

- [ ] **Step 2: Write the failing tests**

Save this diff as `$SCRATCH/t1_tests.diff` and apply it with `git apply $SCRATCH/t1_tests.diff`. Every later diff is saved and applied the same way.

```diff
diff --git a/tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py b/tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py
index 5022f06534..6a5dcc37e8 100644
--- a/tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py
+++ b/tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py
@@ -12,7 +12,8 @@ import pytest
 from tldw_Server_API.app.api.v1.endpoints import messages as messages_endpoint
 from tldw_Server_API.app.api.v1.schemas.anthropic_messages import AnthropicMessagesRequest
 from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
-from tldw_Server_API.app.core.Usage import llm_accounting
+from tldw_Server_API.app.core.Usage import llm_accounting, usage_tracker
+from tldw_Server_API.app.core.Usage.pricing_catalog import reset_pricing_catalog

 pytestmark = pytest.mark.unit

@@ -79,6 +80,14 @@ class _Runtime:
         """Nothing to close."""


+def _stub_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
+    """Make the test's loop the accounting loop, and stub the endpoint's credentials and base URL."""
+    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
+    monkeypatch.setattr(messages_endpoint, "derive_trusted_credential_scope", lambda _r, user: (user.id_int, [], [], False))
+    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
+    monkeypatch.setattr(messages_endpoint, "_resolve_messages_base_url", lambda _p, _c: "http://provider.test/v1")
+
+
 @pytest.fixture
 async def rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
     """Capture what would be written, and make the test's loop the accounting loop."""
@@ -89,13 +98,39 @@ async def rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
         captured.append(kwargs)

     monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _fake_log)
-    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
-    monkeypatch.setattr(messages_endpoint, "derive_trusted_credential_scope", lambda _r, user: (user.id_int, [], [], False))
-    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
-    monkeypatch.setattr(messages_endpoint, "_resolve_messages_base_url", lambda _p, _c: "http://provider.test/v1")
+    _stub_endpoint(monkeypatch)
     return captured


+@pytest.fixture
+async def costed_rows(monkeypatch: pytest.MonkeyPatch) -> Any:
+    """Capture the llm_usage_log rows the real recorder would insert, costed by the shipped pricing catalog."""
+    captured: list[dict[str, Any]] = []
+
+    class _Repo:
+        """Stands in for AuthnzUsageRepo and keeps the inserted rows."""
+
+        def __init__(self, _pool: Any) -> None:
+            """No database."""
+
+        async def insert_llm_usage_log(self, **kwargs: Any) -> None:
+            """Record the row instead of writing it."""
+            captured.append(kwargs)
+
+    async def _none() -> None:
+        """No pool and no daily ledger."""
+
+    monkeypatch.setenv("LLM_USAGE_ENABLED", "true")
+    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
+    reset_pricing_catalog()
+    monkeypatch.setattr(usage_tracker, "get_db_pool", _none)
+    monkeypatch.setattr(usage_tracker, "_get_tokens_daily_ledger", _none)
+    monkeypatch.setattr(usage_tracker, "AuthnzUsageRepo", _Repo)
+    _stub_endpoint(monkeypatch)
+    yield captured
+    reset_pricing_catalog()
+
+
 async def _call(model: str, *, stream: bool) -> Any:
     """Run the Messages handler as an authenticated request for user 5 would."""
     return await messages_endpoint._handle_messages(
@@ -214,6 +249,22 @@ async def test_a_cached_native_stream_counts_its_cached_input_as_prompt_tokens(r
     assert row["usage_metadata"]["cache_read_input_tokens"] == 60000


+async def test_a_cached_native_reply_is_costed_at_the_cache_rates(costed_rows, monkeypatch) -> None:
+    """Cache reads and writes cost Anthropic's cache rates, not the input rate (TASK-13858)."""
+    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply({**CACHED_USAGE, "output_tokens": 100}))
+    with scoped_context(user_id=5):
+        response = await _call("anthropic/claude-sonnet-4-6", stream=False)
+    assert response.status_code == 200
+    await llm_accounting.drain_accounting()
+    (row,) = costed_rows
+    assert (row["prompt_tokens"], row["cache_read_input_tokens"], row["cache_write_input_tokens"]) == (62003, 60000, 2000)
+    assert row["billable_input_tokens"] == 3
+    # Sonnet 4.6 per 1K tokens: input $0.003, cache read $0.0003 (0.1x), 5-minute cache write $0.00375 (1.25x).
+    assert row["prompt_cost_usd"] == pytest.approx((3 * 0.003 + 60000 * 0.0003 + 2000 * 0.00375) / 1000)
+    assert row["completion_cost_usd"] == pytest.approx(100 * 0.015 / 1000)
+    assert row["estimated"] is False
+
+
 async def test_a_native_stream_closed_early_still_records_one_row(rows, monkeypatch) -> None:
     """A client that disconnects after the first frame still costs the input tokens it was sent."""
     monkeypatch.setattr(messages_endpoint, "_prepare_native_stream_iterator", _native_stream(STREAM_FRAMES))
diff --git a/tldw_Server_API/tests/Usage/test_pricing_catalog.py b/tldw_Server_API/tests/Usage/test_pricing_catalog.py
index 566cd4527d..08fe440970 100644
--- a/tldw_Server_API/tests/Usage/test_pricing_catalog.py
+++ b/tldw_Server_API/tests/Usage/test_pricing_catalog.py
@@ -120,3 +120,45 @@ def test_compute_costs_clamps_billable_input_tokens_to_uncached_remainder(
     assert completion_cost == 0.0
     assert total_cost == pytest.approx(prompt_cost)
     assert estimated is False
+
+
+# Anthropic list prices per 1K tokens: prompt, completion, cache hit, 5-minute cache write.
+# Source: platform.claude.com/docs/en/about-claude/pricing, fetched 2026-10-20 (TASK-13858).
+_FABLE_5_1 = {"prompt": 0.01, "completion": 0.05, "cache_read": 0.00025, "cache_write": 0.0125}
+_FABLE_5 = {"prompt": 0.01, "completion": 0.05, "cache_read": 0.001, "cache_write": 0.0125}
+_OPUS_5_5 = {"prompt": 0.004, "completion": 0.02, "cache_read": 0.0002, "cache_write": 0.005}
+_OPUS_5 = {"prompt": 0.005, "completion": 0.025, "cache_read": 0.0005, "cache_write": 0.00625}
+_OPUS_4_1 = {"prompt": 0.015, "completion": 0.075, "cache_read": 0.0015, "cache_write": 0.01875}
+_SONNET_5_5 = {"prompt": 0.002, "completion": 0.01, "cache_read": 0.0001, "cache_write": 0.0025}
+_SONNET_5 = {"prompt": 0.002, "completion": 0.01, "cache_read": 0.0002, "cache_write": 0.0025}
+_SONNET_4 = {"prompt": 0.003, "completion": 0.015, "cache_read": 0.0003, "cache_write": 0.00375}
+_HAIKU_4_5 = {"prompt": 0.001, "completion": 0.005, "cache_read": 0.0001, "cache_write": 0.00125}
+
+
+@pytest.mark.parametrize(
+    ("model", "rates"),
+    [
+        ("claude-fable-5-1", _FABLE_5_1),
+        ("claude-fable-5", _FABLE_5),
+        ("claude-opus-5-5", _OPUS_5_5),
+        ("claude-opus-5", _OPUS_5),
+        ("claude-opus-4-8", _OPUS_5),
+        ("claude-opus-4-7", _OPUS_5),
+        ("claude-opus-4-5", _OPUS_5),
+        ("claude-opus-4-5-20251101", _OPUS_5),
+        ("claude-opus-4-1-20250805", _OPUS_4_1),
+        ("claude-sonnet-5-5", _SONNET_5_5),
+        ("claude-sonnet-5", _SONNET_5),
+        ("claude-sonnet-4-6", _SONNET_4),
+        ("claude-sonnet-4-5-20250929", _SONNET_4),
+        ("claude-haiku-4-5-20251001", _HAIKU_4_5),
+    ],
+)
+def test_anthropic_model_ids_get_their_list_prices_and_cache_rates(
+    model: str, rates: dict[str, float], monkeypatch: pytest.MonkeyPatch
+) -> None:
+    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
+
+    got, _estimated = PricingCatalog().get_rate_details("anthropic", model)
+
+    assert got == pytest.approx(rates)
```

- [ ] **Step 3: Run them to see them fail**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py tldw_Server_API/tests/Usage/test_pricing_catalog.py
```

Expected:
- **The summary** is `15 failed, 16 passed`.
- **`test_a_cached_native_reply_is_costed_at_the_cache_rates`** fails with `assert 0.186009 == 0.025509 ± 2.6e-08`. All 62003 prompt tokens are priced at the input rate.
- **All 14 cases of `test_anthropic_model_ids_get_their_list_prices_and_cache_rates`** fail:
  - the IDs with no entry return the unknown-model rate, `{'prompt': 0.01, 'completion': 0.03}`;
  - `claude-opus-4-8`, `claude-opus-4-5` and `claude-opus-4-5-20251101` return Opus 4's `{'prompt': 0.015, 'completion': 0.075}`;
  - the rest lack `cache_read` and `cache_write`.

- [ ] **Step 4: Add the cache rates and the current models**

Apply with `git apply`:

```diff
diff --git a/tldw_Server_API/Config_Files/model_pricing.json b/tldw_Server_API/Config_Files/model_pricing.json
index 2f96d4ea07..04bcc5353b 100644
--- a/tldw_Server_API/Config_Files/model_pricing.json
+++ b/tldw_Server_API/Config_Files/model_pricing.json
@@ -1,5 +1,5 @@
 {
-  "_catalog_note": "Commercial model catalog refreshed 2026-05-10 from official provider docs and official model-list endpoints where available. Rates are USD per 1K tokens. Entries with placeholder=true are intentionally hidden from provider selectors; entries with estimated=true are available model IDs with approximate or source-dependent USD rates.",
+  "_catalog_note": "Commercial model catalog refreshed 2026-05-10 from official provider docs and official model-list endpoints where available. Rates are USD per 1K tokens. Entries with placeholder=true are intentionally hidden from provider selectors; entries with estimated=true are available model IDs with approximate or source-dependent USD rates. Anthropic prices, cache_read (the cache-hit rate: 0.1x input, 0.05x on Opus 5.5 and Sonnet 5.5, 0.025x on Fable 5.1) and cache_write (the 5-minute cache-write rate, 1.25x input) are from platform.claude.com/docs/en/about-claude/pricing, fetched 2026-10-20.",
   "openai": {
     "gpt-5.5": { "prompt": 0.005, "completion": 0.03, "estimated": true },
     "gpt-5.5-2026-04-23": { "prompt": 0.005, "completion": 0.03, "estimated": true },
@@ -90,21 +90,29 @@
     "gpt-image-1-mini": { "prompt": 0.0, "completion": 0.0, "placeholder": true, "note": "image model; hidden from general LLM selector" }
   },
   "anthropic": {
-    "claude-opus-4-7": { "prompt": 0.005, "completion": 0.025 },
-    "claude-opus-4-7-20260103": { "prompt": 0.005, "completion": 0.025 },
-    "claude-opus-4-6": { "prompt": 0.005, "completion": 0.025 },
-    "claude-opus-4.5": { "prompt": 0.005, "completion": 0.025 },
-    "claude-opus-4.1": { "prompt": 0.015, "completion": 0.075 },
-    "claude-opus-4": { "prompt": 0.015, "completion": 0.075 },
-    "claude-sonnet-4-6": { "prompt": 0.003, "completion": 0.015 },
-    "claude-sonnet-4-6-20260105": { "prompt": 0.003, "completion": 0.015 },
-    "claude-sonnet-4.5": { "prompt": 0.003, "completion": 0.015 },
-    "claude-sonnet-4": { "prompt": 0.003, "completion": 0.015 },
+    "claude-fable-5-1": { "prompt": 0.01, "completion": 0.05, "cache_read": 0.00025, "cache_write": 0.0125 },
+    "claude-fable-5": { "prompt": 0.01, "completion": 0.05, "cache_read": 0.001, "cache_write": 0.0125 },
+    "claude-opus-5-5": { "prompt": 0.004, "completion": 0.02, "cache_read": 0.0002, "cache_write": 0.005 },
+    "claude-opus-5": { "prompt": 0.005, "completion": 0.025, "cache_read": 0.0005, "cache_write": 0.00625 },
+    "claude-opus-4-8": { "prompt": 0.005, "completion": 0.025, "cache_read": 0.0005, "cache_write": 0.00625 },
+    "claude-opus-4-7": { "prompt": 0.005, "completion": 0.025, "cache_read": 0.0005, "cache_write": 0.00625 },
+    "claude-opus-4-7-20260103": { "prompt": 0.005, "completion": 0.025, "cache_read": 0.0005, "cache_write": 0.00625 },
+    "claude-opus-4-6": { "prompt": 0.005, "completion": 0.025, "cache_read": 0.0005, "cache_write": 0.00625 },
+    "claude-opus-4-5": { "prompt": 0.005, "completion": 0.025, "cache_read": 0.0005, "cache_write": 0.00625 },
+    "claude-opus-4.5": { "prompt": 0.005, "completion": 0.025, "cache_read": 0.0005, "cache_write": 0.00625 },
+    "claude-opus-4.1": { "prompt": 0.015, "completion": 0.075, "cache_read": 0.0015, "cache_write": 0.01875 },
+    "claude-opus-4": { "prompt": 0.015, "completion": 0.075, "cache_read": 0.0015, "cache_write": 0.01875 },
+    "claude-sonnet-5-5": { "prompt": 0.002, "completion": 0.01, "cache_read": 0.0001, "cache_write": 0.0025 },
+    "claude-sonnet-5": { "prompt": 0.002, "completion": 0.01, "cache_read": 0.0002, "cache_write": 0.0025 },
+    "claude-sonnet-4-6": { "prompt": 0.003, "completion": 0.015, "cache_read": 0.0003, "cache_write": 0.00375 },
+    "claude-sonnet-4-6-20260105": { "prompt": 0.003, "completion": 0.015, "cache_read": 0.0003, "cache_write": 0.00375 },
+    "claude-sonnet-4.5": { "prompt": 0.003, "completion": 0.015, "cache_read": 0.0003, "cache_write": 0.00375 },
+    "claude-sonnet-4": { "prompt": 0.003, "completion": 0.015, "cache_read": 0.0003, "cache_write": 0.00375 },
     "claude-sonnet-3.7": { "prompt": 0.003, "completion": 0.015 },
-    "claude-haiku-4-5": { "prompt": 0.001, "completion": 0.005 },
-    "claude-haiku-4-5-20251001": { "prompt": 0.001, "completion": 0.005 },
-    "claude-haiku-4.5": { "prompt": 0.001, "completion": 0.005 },
-    "claude-haiku-3.5": { "prompt": 0.0008, "completion": 0.004 },
+    "claude-haiku-4-5": { "prompt": 0.001, "completion": 0.005, "cache_read": 0.0001, "cache_write": 0.00125 },
+    "claude-haiku-4-5-20251001": { "prompt": 0.001, "completion": 0.005, "cache_read": 0.0001, "cache_write": 0.00125 },
+    "claude-haiku-4.5": { "prompt": 0.001, "completion": 0.005, "cache_read": 0.0001, "cache_write": 0.00125 },
+    "claude-haiku-3.5": { "prompt": 0.0008, "completion": 0.004, "cache_read": 0.00008, "cache_write": 0.001 },
     "claude-3-5-sonnet-latest": { "prompt": 0.003, "completion": 0.015 },
     "claude-3-opus-20240229": { "prompt": 0.015, "completion": 0.075 },
     "claude-3-sonnet-20240229": { "prompt": 0.003, "completion": 0.015 },
```

Check that the file still parses: `$PY -m json.tool tldw_Server_API/Config_Files/model_pricing.json > /dev/null`, which prints nothing.

- [ ] **Step 5: Run the tests to see them pass**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py tldw_Server_API/tests/Usage/test_pricing_catalog.py
```

Expected: `31 passed`.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/Config_Files/model_pricing.json tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py tldw_Server_API/tests/Usage/test_pricing_catalog.py
git commit -m "fix(usage): price Anthropic prompt-cache reads and writes, and the current Anthropic models (TASK-13858)

compute_costs and the pricing catalog already price cache reads and
writes separately, but no Anthropic entry in model_pricing.json had the
rates, so cached tokens cost the full input rate: 10x a cache read.
The 14 Anthropic entries on Anthropic's current price list gain
cache_read (0.1x input) and cache_write (the 5-minute write, 1.25x),
from platform.claude.com/docs/en/about-claude/pricing. A cached Sonnet
4.6 Messages reply (3 uncached, 2000 written, 60000 read) now costs
\$0.025509 of prompt instead of \$0.186009.

Eight current models had no entry (Opus 5.5 and 5, Sonnet 5.5 and 5,
Fable 5.1 and 5) or matched Opus 4's key (Opus 4.8, the claude-opus-4-5
alias and its dated ID). They now have their list prices, including
the 0.05x reads of Opus 5.5 and Sonnet 5.5 and the 0.025x reads of
Fable 5.1, and become selectable through /api/v1/llm/providers.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: The Anthropic adapter keeps the cache fields and counts them in prompt_tokens

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/anthropic_adapter.py:373-379` (`_normalize_to_openai_shape`, the usage block).
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_anthropic_native_http.py`, `test_anthropic_adapter_usage_counts_and_keeps_prompt_cache_tokens`.

**Interfaces:**
- Produces: for an Anthropic reply whose usage has integer cache fields, the adapter's `usage` is:

  ```python
  {
      "prompt_tokens": input + creation + read,
      "completion_tokens": output,
      "total_tokens": prompt + output,
      "cache_creation_input_tokens": creation,
      "cache_read_input_tokens": read,
  }
  ```

  It is unchanged when there are none.

- [ ] **Step 1: Write the failing test**

Apply with `git apply`:

```diff
diff --git a/tldw_Server_API/tests/LLM_Adapters/unit/test_anthropic_native_http.py b/tldw_Server_API/tests/LLM_Adapters/unit/test_anthropic_native_http.py
index 82e4ed6761..6eebcd819d 100644
--- a/tldw_Server_API/tests/LLM_Adapters/unit/test_anthropic_native_http.py
+++ b/tldw_Server_API/tests/LLM_Adapters/unit/test_anthropic_native_http.py
@@ -82,3 +82,37 @@ def test_anthropic_adapter_native_http_streaming(monkeypatch):
     chunks = list(a.stream({"messages": [{"role": "user", "content": "hi"}], "model": "claude-sonnet", "api_key": "k", "max_tokens": 32, "stream": True}))
     assert any(c.startswith("data: ") for c in chunks)
     assert sum(1 for c in chunks if "[DONE]" in c) == 1
+
+
+def test_anthropic_adapter_usage_counts_and_keeps_prompt_cache_tokens(monkeypatch):
+    """input_tokens is only the uncached remainder: prompt_tokens adds the cache fields, which stay in the block (TASK-13858)."""
+    import tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter as anth_mod
+    from tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter import AnthropicAdapter
+    from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage
+
+    message = {
+        "type": "message",
+        "id": "msg-cached",
+        "model": "claude-sonnet-4-6",
+        "content": [{"type": "text", "text": "ok"}],
+        "stop_reason": "end_turn",
+        "usage": {"input_tokens": 3, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 60000, "output_tokens": 100},
+    }
+
+    class _CachedClient(_FakeClient):
+        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
+            return _FakeResponse(200, message)
+
+    monkeypatch.setattr(anth_mod, "http_client_factory", lambda *a, **k: _CachedClient())
+    r = AnthropicAdapter().chat({"messages": [{"role": "user", "content": "hi"}], "model": "claude-sonnet-4-6", "api_key": "k", "max_tokens": 32})
+
+    assert r["usage"] == {
+        "prompt_tokens": 62003,
+        "completion_tokens": 100,
+        "total_tokens": 62103,
+        "cache_creation_input_tokens": 2000,
+        "cache_read_input_tokens": 60000,
+    }
+    # What the chat pipeline and the provider recorder cost: only the 3 uncached tokens at the input rate.
+    normalized = normalize_llm_usage(provider="anthropic", usage=r["usage"])
+    assert (normalized.billable_input_tokens, normalized.cache_read_input_tokens, normalized.cache_write_input_tokens) == (3, 60000, 2000)
```

- [ ] **Step 2: Run it to see it fail**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Adapters/unit/test_anthropic_native_http.py
```

Expected:
- **The summary** is `1 failed, 2 passed`.
- **The failure** is `test_anthropic_adapter_usage_counts_and_keeps_prompt_cache_tokens`:
  - `AssertionError: assert {'prompt_toke..._tokens': 103} == {'prompt_toke...s': 2000, ...}`;
  - `{'prompt_tokens': 3} != {'prompt_tokens': 62003}` and `{'total_tokens': 103} != {'total_tokens': 62103}`;
  - the right side contains 2 more items, the cache fields.

- [ ] **Step 3: Count and keep the cache fields**

Apply with `git apply`:

```diff
diff --git a/tldw_Server_API/app/core/LLM_Calls/providers/anthropic_adapter.py b/tldw_Server_API/app/core/LLM_Calls/providers/anthropic_adapter.py
index 477510543b..be13be678f 100644
--- a/tldw_Server_API/app/core/LLM_Calls/providers/anthropic_adapter.py
+++ b/tldw_Server_API/app/core/LLM_Calls/providers/anthropic_adapter.py
@@ -372,10 +372,20 @@ class AnthropicAdapter(ChatProvider):
         }
         usage = data.get("usage") or {}
         if isinstance(usage, dict):
+            # input_tokens is only the uncached remainder; prompt_tokens is the whole prompt, as in OpenAI's usage (ADR-075).
+            cache = {
+                key: usage[key]
+                for key in ("cache_creation_input_tokens", "cache_read_input_tokens")
+                if isinstance(usage.get(key), int)
+            }
+            prompt_tokens = usage.get("input_tokens")
+            if cache:
+                prompt_tokens = (prompt_tokens or 0) + sum(cache.values())
             shaped["usage"] = {
-                "prompt_tokens": usage.get("input_tokens"),
+                "prompt_tokens": prompt_tokens,
                 "completion_tokens": usage.get("output_tokens"),
-                "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
+                "total_tokens": (prompt_tokens or 0) + (usage.get("output_tokens") or 0),
+                **cache,
             }
         shaped["provider_response"] = data
         return shaped
```

- [ ] **Step 4: Run the test to see it pass**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Adapters/unit/test_anthropic_native_http.py
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/LLM_Calls/providers/anthropic_adapter.py tldw_Server_API/tests/LLM_Adapters/unit/test_anthropic_native_http.py
git commit -m "fix(llm): the Anthropic adapter counts prompt-cache tokens in prompt_tokens and keeps them (TASK-13858)

Anthropic's input_tokens is only the uncached remainder. The adapter
used it as prompt_tokens and dropped cache_creation_input_tokens and
cache_read_input_tokens, so a prompt-cached call through
/chat/completions recorded a fraction of its prompt, as the native
Messages path did before TASK-13610. prompt_tokens now adds the two
cache fields (OpenAI's meaning, ADR-075) and the block keeps them, so
the chat pipeline and provider recorders cost cache reads and writes at
their own rates. A reply without cache fields keeps its usage block
unchanged.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 3: Document, verify, file the follow-ups, and close the task

**Files:**
- Modify: `Docs/Operations/Usage_Quotas.md`: one bullet, before "Two settings defeat these limits".
- Modify: `Docs/Providers/Model_Pricing_Catalog.md`: the "Editing the Catalog" format line, the Anthropic quick link (now the pricing page), and the Anthropic example, now API IDs with `cache_read` and `cache_write`.
- Modify: `backlog/tasks/task-13858*.md`. Create three follow-up task files (backlog-py only).

**Interfaces:**
- Consumes: Tasks 1 and 2.

- [ ] **Step 1: Document cache pricing and the override trap for operators**

Apply with `git apply`:

```diff
diff --git a/Docs/Operations/Usage_Quotas.md b/Docs/Operations/Usage_Quotas.md
index 8188038845..00d16df36d 100644
--- a/Docs/Operations/Usage_Quotas.md
+++ b/Docs/Operations/Usage_Quotas.md
@@ -105,6 +105,7 @@ What `limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd` and `limits.ll
 - **How the cost keys compare.** A cost key refuses once today's or this month's spend has reached the limit. The token key refuses when this request's estimated tokens would take the month past it; outside `/chat/completions` the estimate is one token.
 - **An admitted request runs to completion,** so a user can pass a limit by one request's worth of usage. A media batch or an evaluation batch can be large.
 - **Local providers cost nothing by default:** llama.cpp, kobold, ooba, tabbyapi, vllm, local-llm, ollama, aphrodite and mlx. To give one a price, edit its `default` entry in `tldw_Server_API/Config_Files/model_pricing.json`, or set `PRICING_OVERRIDES` (a JSON string in the environment), which overrides that file. A custom OpenAI-compatible endpoint is priced at the unknown-model rate ($0.01 per 1K prompt tokens, $0.03 per 1K completion tokens) unless you give it a price. A virtual-key USD budget no longer counts local calls.
+- **Prompt-cache tokens.** Anthropic cache reads and cache writes count as prompt tokens on every call recorded for a user: the Messages API and every caller of the Anthropic adapter (`/chat/completions`, RAG, summarization, workflows and the rest). So they count in full toward `limits.llm_tokens_per_month`, the org plan's `llm_tokens_month`, the Resource Governor's token commit on `/chat/completions`, and the `llm_tokens_used_total` metrics. Their cost uses the `cache_read` and `cache_write` rates of the model's entry in `model_pricing.json`. The Anthropic entries that carry them use Anthropic's cache-hit rate (0.1x input; 0.05x on Opus 5.5 and Sonnet 5.5, 0.025x on Fable 5.1) and its 5-minute cache-write rate (1.25x). A 1-hour cache write, which Anthropic bills at 2x, is costed at the 1.25x rate. Claude Haiku 5.5, priced by prompt length, has no entry yet and costs the unknown-model rate. An entry without cache rates costs every prompt token at its input rate. That includes an override in `PRICING_OVERRIDES` or in the file: it replaces the whole entry, so it must repeat `cache_read` and `cache_write`. A streamed `/chat/completions` reply is also costed at the input rate, because its usage is estimated from the prompt.
 - **Two settings defeat these limits, and the server warns at startup when quotas are on:** `LLM_USAGE_ENABLED` off (nothing is recorded, so nothing is refused), and `LLM_USAGE_LOG_RETENTION_DAYS` under 31 (the monthly counts lose rows).
 - **Failures let requests through.** If a limit cannot be read or a counter fails, the request or call is admitted and a warning is logged. A check at the call that gets no answer within two seconds lets the call through, with a warning. A synchronous LLM call made on the server's event loop thread cannot be checked (waiting there would stall the server); it goes through, recorded, and the first one logs `A synchronous LLM call on the event loop's thread is not quota-checked`. A stream started there is checked when its first item is read.
 - **Run workers inside the server.** A job worker started as its own process (`python -m ...`) has no accounting loop, so its synchronous LLM calls are neither checked nor recorded; the first one logs a warning. Workers run inside the server process by default.
diff --git a/Docs/Providers/Model_Pricing_Catalog.md b/Docs/Providers/Model_Pricing_Catalog.md
index 6b939f6f89..7649d56383 100644
--- a/Docs/Providers/Model_Pricing_Catalog.md
+++ b/Docs/Providers/Model_Pricing_Catalog.md
@@ -21,7 +21,7 @@ and available to the Chat API as `provider/model`.

 ## Editing the Catalog

-- Format: JSON object by provider, then model id → `{prompt: number, completion: number}` (USD per 1K tokens).
+- Format: JSON object by provider, then model id → `{prompt: number, completion: number}` (USD per 1K tokens), with optional `cache_read` and `cache_write`: the prompt-cache read and write rates, also per 1K tokens. Without them, cached tokens cost the `prompt` rate. An override replaces the whole entry, so an override of an entry that has cache rates must repeat them.
 - Example (OpenAI text models only):

 ```
@@ -39,7 +39,7 @@ Tip: Keep values conservative if you’re unsure, then update with exact rates f

 ## Provider Quick Links

-- Anthropic: https://docs.claude.com/en/docs/about-claude/models/overview
+- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing (model IDs: https://platform.claude.com/docs/en/about-claude/models/overview)
 - OpenAI (text models): https://platform.openai.com/docs/pricing
 - Z.ai: https://docs.z.ai/guides/overview/pricing
 - Moonshot (Kimi): https://platform.moonshot.ai/docs/pricing/chat#generation-model-kimi-k2
@@ -50,13 +50,13 @@ Tip: Keep values conservative if you’re unsure, then update with exact rates f

 These examples illustrate the expected shape. Replace with current values from the linked pages above.

-Anthropic (Claude 4.5/4.1 family):
+Anthropic (keys are API model IDs; an override repeats the cache rates):
 ```
 {
   "anthropic": {
-    "claude-opus-4.1":   { "prompt": 0.015,  "completion": 0.075 },
-    "claude-sonnet-4.5": { "prompt": 0.003,  "completion": 0.015 },
-    "claude-haiku-4.5":  { "prompt": 0.001,  "completion": 0.005 }
+    "claude-opus-4-7":   { "prompt": 0.005, "completion": 0.025, "cache_read": 0.0005, "cache_write": 0.00625 },
+    "claude-sonnet-4-6": { "prompt": 0.003, "completion": 0.015, "cache_read": 0.0003, "cache_write": 0.00375 },
+    "claude-haiku-4-5":  { "prompt": 0.001, "completion": 0.005, "cache_read": 0.0001, "cache_write": 0.00125 }
   }
 }
 ```
```

- [ ] **Step 2: Run the sweep, the Docs tests and the ratchet**

```bash
FILES=$(git grep -l -E "anthropic_adapter|AnthropicAdapter|pricing_catalog|model_pricing|compute_costs|log_llm_usage|normalize_llm_usage|list_provider_models|compute_token_cost|usage_accounting|record_direct_response|claude-opus-4-7|llm/providers" -- 'tldw_Server_API/tests/*.py')
$PY -m pytest -p no:cacheprovider -q $FILES \
  tldw_Server_API/tests/Admin/test_llm_usage_endpoints.py tldw_Server_API/tests/Admin/test_admin_usage_service.py \
  tldw_Server_API/tests/AuthNZ/unit/test_admin_usage_service_backend_selection.py \
  tldw_Server_API/tests/Chat/integration/test_messages_endpoint.py tldw_Server_API/tests/Chat/integration/test_messages_provider_override_store_boundary.py \
  tldw_Server_API/tests/Chat/unit/test_messages_endpoint_native_errors.py tldw_Server_API/tests/Chat/unit/test_messages_endpoint_usage.py \
  tldw_Server_API/tests/LLM_Adapters/integration/test_adapters_chat_endpoint_anthropic_native.py \
  tldw_Server_API/tests/Usage tldw_Server_API/tests/LLM_Adapters tldw_Server_API/tests/LLM_Calls
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs
$PY Helper_Scripts/ci/test_quality_triage.py --enforce 2>&1 | grep "test-triage\] \(OK\|FAIL\)"
```

The first run covers:
- every Python test file that references the adapter, the pricing catalog, the cost and usage functions, the provider recorder, the Anthropic model list or the providers endpoint (68 files);
- the eight tests that read Anthropic usage or the cache columns without matching that grep;
- the whole Usage, LLM_Adapters and LLM_Calls suites.

It takes about 18 minutes.

Expected, measured on the prototype:
- **The first run:** `3575 passed, 28 skipped`. The skips are Postgres and optional-dependency skips.
- **Docs:** `212 passed`.
- **The ratchet:** `[test-triage] OK — no offenses beyond the baseline.`

If anything fails, run the same test on `origin/dev` before going on.

- [ ] **Step 3: Lint and security**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/app/core/LLM_Calls/providers/anthropic_adapter.py tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py tldw_Server_API/tests/LLM_Adapters/unit/test_anthropic_native_http.py tldw_Server_API/tests/Usage/test_pricing_catalog.py
uvx bandit -q -ll tldw_Server_API/app/core/LLM_Calls/providers/anthropic_adapter.py
```

Expected:
- **Ruff** prints `Found 13 errors.` and exits 1. All 13 exist on `origin/dev` unchanged:
  - 12 in `test_anthropic_native_http.py`, lines 3-78, none in the new test: `UP035` ×2, `UP006` ×6, `SIM300`, `UP028`, `I001` ×2;
  - 1 in `test_pricing_catalog.py`: `F401` for the unused `import os` on line 4.
  - The adapter and the native recording test are clean. Fixing the old findings is out of scope.
- **Bandit** prints nothing and exits 0.

- [ ] **Step 4: Commit the doc**

```bash
git add Docs/Operations/Usage_Quotas.md Docs/Providers/Model_Pricing_Catalog.md
git commit -m "docs(usage): how Anthropic prompt-cache tokens count and cost, and the override trap (TASK-13858)

Cache reads and writes count in full toward limits.llm_tokens_per_month,
the org plan's llm_tokens_month, the Resource Governor token commit and
the token metrics. An override entry replaces the whole catalog entry,
so it must repeat cache_read and cache_write.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: File the follow-ups**

Set `M` to the highest task id across `origin/dev` and open PRs, with the command from AGENTS.md's Backlog section. It was `13866` when this plan was written. The follow-ups take the next three ids. Keep this shell open for Step 6, which reuses `M`.

```bash
git fetch origin dev --quiet
M=$({ git ls-tree -r --name-only origin/dev backlog/; gh pr list --state open --limit 500 --json files --jq '.[].files[].path'; } \
  | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1)
echo "$M"
```

Expected: a number of at least `13866`. Then:

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Cost Anthropic 1-hour prompt-cache writes at 2x, not the 5-minute 1.25x rate" --id TASK-$((M+1)) \
  -d "TASK-13858 gave the Anthropic models one cache_write rate, the 5-minute write price (1.25x input). Anthropic bills a 1-hour cache write at 2x input (platform.claude.com/docs/en/about-claude/pricing) and reports the split in usage.cache_creation.ephemeral_5m_input_tokens and ephemeral_1h_input_tokens. The server asks for 1-hour writes when a cache intent sets ttl 1h or ttl_seconds >= 3600 (LLM_Calls/cache_intents.py), and Messages API clients can ask directly, so those writes are costed 0.75x input too low and cost quotas undercount them." \
  --ac "The usage normalizer reads the 1-hour share of cache writes, and compute_costs prices it at a cache_write_1h rate, with a test" \
  --ac "model_pricing.json gives the Anthropic models a cache_write_1h rate (2x input), and the Anthropic adapter keeps the cache_creation breakdown" \
  -l quotas,backend
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Price OpenAI cached input tokens" --id TASK-$((M+2)) \
  -d "OpenAI's prompt_tokens already include prompt_tokens_details.cached_tokens, and the usage normalizer already splits them out (cache_read_input_tokens, billable_input_tokens), but no OpenAI model in model_pricing.json has a cache_read rate, so compute_costs bills cached input at the full input rate. Found while planning TASK-13858. No adapter change is needed, only rates taken from OpenAI's published pricing." \
  --ac "model_pricing.json gives the OpenAI models that support prompt caching a cache_read rate from OpenAI's published pricing, with a test that a row with cached_tokens is costed with it" \
  -l quotas,backend
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Anthropic pricing catalog: Haiku 5.5's tiered price, Mythos, Haiku 3.5's API ID and the dotted keys" --id TASK-$((M+3)) \
  -d "TASK-13858 priced the flat-priced current Anthropic models. Left over, measured with PricingCatalog().get_rate_details: claude-haiku-5-5 costs the unknown-model rate (0.01 and 0.03 per 1K, about 100x its input price); Anthropic prices it by prompt length (above or below 100,000 tokens, cache tokens included), which a flat catalog entry cannot express. Mythos 5.1 and 5 (limited availability) have no entry. Haiku 3.5's API ID claude-3-5-haiku-20241022 costs the unknown rate, so the claude-haiku-3.5 entry never prices a real request. The dotted keys (claude-opus-4.5, claude-opus-4.1, claude-sonnet-4.5, claude-haiku-4.5) match no API ID and appear in the WebUI model list. Prices are on platform.claude.com/docs/en/about-claude/pricing." \
  --ac "claude-haiku-5-5 is costed at Anthropic's price for its prompt length, with a test on both sides of 100,000 tokens" \
  --ac "Every Anthropic entry's key is an API ID or alias, and Mythos and Haiku 3.5's API ID are priced or deliberately left out" \
  -l quotas,backend
```

Expected: each command prints `TASK-<id> - <title> (To Do)`.

- [ ] **Step 6: Close the task and commit the plan**

The notes quote the measured results. If Steps 3 and 5 of Tasks 1-2, or Steps 2 and 3 here, printed different lines, put the actual ones in. Run it in the shell from Step 5, which set `M`.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13858 --append-notes "Fixed:
- model_pricing.json gives the 14 Anthropic entries on Anthropic's price list cache_read (0.1x input) and cache_write (5-minute write, 1.25x), and adds the 8 flat-priced current models (claude-fable-5-1, claude-fable-5, claude-opus-5-5, claude-opus-5, claude-opus-4-8, claude-opus-4-5, claude-sonnet-5-5, claude-sonnet-5; 0.05x reads on Opus 5.5 and Sonnet 5.5, 0.025x on Fable 5.1), all from platform.claude.com/docs/en/about-claude/pricing (fetched 2026-10-20). The new entries are selectable through /api/v1/llm/providers. compute_costs, the catalog parser, the normalizer and the llm_usage_log cache columns already existed (SQLite migration 088, Postgres pg_migrations_extra), so no code or schema change was needed for pricing.
- AnthropicAdapter._normalize_to_openai_shape adds cache_creation_input_tokens and cache_read_input_tokens to prompt_tokens and keeps both in the usage block, so /chat/completions and provider-recorder rows count and cost cached input like the native path.
- A cached Sonnet 4.6 reply (3 uncached, 2000 written, 60000 read) now costs 0.025509 of prompt, not 0.186009.
Tests:
- T1 RED 15 failed, 16 passed; GREEN 31 passed.
- T2 RED 1 failed, 2 passed; GREEN 3 passed.
- Sweep (68 referencing files, 8 Anthropic-usage tests, plus Usage, LLM_Adapters and LLM_Calls): 3575 passed, 28 skipped. Docs 212 passed. Test-quality ratchet OK.
- Ruff: only the 13 findings dev already has (12 in test_anthropic_native_http.py, an unused import in test_pricing_catalog.py). Bandit clean.
- Token consumers now counting cached input in full: limits.llm_tokens_per_month, the org plan's llm_tokens_month, the RG token commit, the Prometheus token counters.
Follow-ups:
- TASK-$((M+1)): 1-hour cache writes at 2x (Decision 1).
- TASK-$((M+2)): OpenAI cached-input rates (Decision 2).
- TASK-$((M+3)): Haiku 5.5's tiered price, Mythos, Haiku 3.5's API ID, the dotted keys." --check-ac 1 --check-ac 2 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "Anthropic prompt-cache reads and writes are costed at Anthropic's cache rates, and the flat-priced current Anthropic models (Fable 5.1 and 5, Opus 5.5, 5, 4.8 and 4.5, Sonnet 5.5 and 5) have their list prices. The Anthropic adapter counts the cache fields in prompt_tokens and keeps them in its usage block, so /chat/completions rows match the native Messages path. 1-hour writes at 2x, OpenAI cached-input rates, and Haiku 5.5, Mythos and the dotted keys are follow-ups." --status Done
git add -A backlog/tasks
git add Docs/superpowers/plans/2026-10-20-anthropic-cache-token-pricing.md
git commit -m "docs(usage): plan for Anthropic cache-token pricing; close TASK-13858, file its follow-ups

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: `git show --stat HEAD` lists:
- the TASK-13858 file;
- the three new task files;
- the plan.

- [ ] **Step 7: Open the PR**

```bash
git push -u origin fix/anthropic-cache-token-pricing
gh pr create --base dev --title "fix(usage): price Anthropic prompt-cache tokens; adapter counts them (TASK-13858)" --body-file $SCRATCH/pr_body_13858.md
```

Write `$SCRATCH/pr_body_13858.md` from the commit messages. Include:
- the cost example;
- the three owner decisions, including the 8 model IDs that become selectable in the WebUI, and the routing change: a request naming one without a provider now goes to Anthropic;
- Review Focus 3, the client-visible `prompt_tokens` change;
- the four consumers that now count cached tokens in full: `limits.llm_tokens_per_month`, the org plan's `llm_tokens_month`, the Resource Governor token commit, and the Prometheus token counters;
- a `## Change summary` section that the owner writes (ADR-004);
- a last line of `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

Then run `/superpowers:requesting-code-review` in a clean subagent. Follow `gh variable get MERGE_QUEUE` for the merge.
