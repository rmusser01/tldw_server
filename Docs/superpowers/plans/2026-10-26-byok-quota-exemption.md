# BYOK calls do not count against the user's own LLM limits (TASK-<ID>) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Review folded in (r3, 2026-10-29).** The scoped re-review (`rereview_plan_byok.md`: 0 Blocker, 2 Important, 3 Minor; measured on `4581a18e5a`) is folded in as follows. origin/dev has since moved to `03fec47781` (#3363, character chat roleplay resume), so r3 was re-measured there. #3363 touches no prototype file; it moves Decision 3's character-chat lines by 4 and changes one wider-set test file.
> - **I1 (fixed in code):** a user key is the user's own spend only on a billed provider sent to its vendor's own host.
>   - `billed_credential_source(provider, source, *configs)` keeps `user` only for a provider in the new `VENDOR_HOSTS` table (the host each of the 16 billed adapters falls back to) whose base URLs in the given configs are all on that host. Anything else is `user_unmetered`. No new setting.
>   - Every caller routes through it: `_new_handle` passes the handle's `app_config`; the two raw-key callers pass their `app_config` and `load_server_config_snapshot()`, because their adapter falls back to environment base URLs; `users_own_key_serves` checks the server snapshot last.
>   - Tests: the junk-key test is parametrized over the custom slot and probe D's four providers on real storage (the four failed on r2 with `assert 'user' == 'user_unmetered'`); `test_a_user_key_is_the_users_own_spend_only_on_the_vendors_host` (8 cases); `test_every_provider_that_bills_per_key_names_its_vendors_host`; the lookup test's `vendor-base-url` and `operator-base-url` cases (the latter failed on r2 with `assert True is False`). Mutations: `endpoint_ignored` fails 8, `boundary_endpoint_ignored` fails 1.
>   - Decision 7, ADR-083, `Usage_Quotas.md` and the BYOK guide state the real condition.
> - **I2 (fixed):** `_get_llm_tokens_month` binds a naive UTC month start (`tzinfo=None` on a UTC `now`; the SQLite string is unchanged), so the hosted plan's PostgreSQL sum counts. New `test_the_hosted_plans_llm_tokens_leave_out_a_members_own_key_pg` on `test_db_pool`: on r2 it failed with `assert 0 == 345`. The re-review's `billing_pg_predicate` mutation now fails it (`1 failed, 247 passed`), and so does restoring the aware bound. Decision 5's hosted-plan row, Global Constraints, Review Focus 8, ADR-083 and `Usage_Quotas.md` say this limit counted nothing on PostgreSQL and this PR starts it counting.
> - **m1 (docs, the smaller correct fix):** the warning covers the bootstrap's handled errors only. An asyncpg database error such as a missing privilege still stops startup, as today (traced through `startup_auth._ensure_pg_extras` and the lifespan guards). The test's stand-in is now `TimeoutError`, which asyncpg raises when an `ALTER` waits on a lock past the pool's 60 s `command_timeout`. AC #5, Decision 1, Global Constraints, the Task 1 note and ADR-083 say exactly that.
> - **m2 (fixed):** Decision 5's RG row says RG is on by default (300 requests and 1,000,000 tokens a minute per user on chat, no daily cap), and so does Decision 2. The hosted-plan row says the skip covers the whole request, a failover included, and that the lookup runs on every hosted `/chat/completions` request.
> - **m3 (fixed in code):** an unknown provider name is `user_unmetered` by construction (it is not in `VENDOR_HOSTS`). Pinned by the rule test's `unknown-provider` case and the table test. Mutation `unknown_is_billed` fails 6.
> - **Smaller notes:** the wider-set grep now ends `| grep '/test_[^/]*\.py$'`, so its file count matches. Decision 3's resume row cites the line (`:6790`).
> - **Re-measured** on `03fec47781`: the table "Re-measured counts, r2 to r3" under "Test commands" has every old and new count.

> **Review folded in (r2, 2026-10-29).** The independent review (`review_plan_byok.md`: 2 Blocker, 5 Important, 9 Minor; its dry run on `35e711080a` matched every RED/GREEN count) is folded in as follows. origin/dev has since moved to `f8f913fab8` (TASK-13951's streamed usage, TASK-13950 refactors), so the revised plan was re-measured there. Only one anchor moved: the streaming `save_callback` now ends `estimate_source=None if reported else "stream_estimate",`.
> - **B1 (fixed):** a user key counts as the user's own only on a provider that bills per key. New `billed_credential_source(provider, source)` in `llm_call_context.py` keeps `user` only when `provider_requires_api_key` holds for the canonical provider. Otherwise it gives `user_unmetered`, which is checked and counts. That covers all 99 custom OpenAI-compatible slots and the nine local providers. Checked: every registered adapter is in `PROVIDER_REQUIRES_KEY`. The only default-allowlisted providers outside it (elevenlabs, fish_s2, voyage) make no LLM calls. The rule is applied in `_new_handle`, at the two context callers and in `users_own_key_serves`, and it never turns another source into `user`. Tests: the handle test (three operator-endpoint providers), the refused-call test (`user` on `custom-openai-api`), a real stored junk key on the operator's endpoint (resolved, still checked, its row counted), and two boundary cases. The remaining gap is a billed provider that the operator points at a key-injecting gateway: Decision 7 and the docs say to keep such a provider off `BYOK_ALLOWED_PROVIDERS`. No opt-in list of billed custom endpoints (Decision 7).
> - **B2 (fixed):** the limits no longer fail open. `_sum_llm_column_since` retries without the new predicate when the error names `credential_source`, so every row counts, and so does the billing plan's sum (I4). `missing_credential_source` warns once per reader. `ensure_usage_tables_pg` now logs a failed statement as a warning, not debug. AC #5 now says the limits still refuse. Tests: the dropped-column test asserts the full 1,500-token sum and a refusal; a fake-pool test pins the warning; a PostgreSQL test proves the driver's error names the column. That test is read-only, so the shared container is safe.
> - **I1 (fixed):** the three failover blocks re-raise `LLMQuotaExceededError` before their generic handler. The non-stream block still emits its run-first `error` metric. Test: `test_a_refused_failover_does_not_mark_the_fallback_provider_unhealthy` (the review's probe B) records failures `["openai"]` only. It drives the non-stream block; the other two get the same two lines.
> - **I2 (fixed):** the call-layer test is parametrized over `perform_chat_api_call` (in a worker thread, where its check runs) and `perform_chat_api_call_async`. The refused-call test is parametrized over `server_default`, `team`, `org` and an operator-endpoint `user` key. The `sync_layer` and `shared_exempt` mutations now each fail 1 test. The call-layer test also stubs `has_model_listing`, because the sync layer otherwise sent a real model-discovery request.
> - **I3 (fixed):** the ADR is now ADR-083, because #3323 holds 082. Task 0 Step 5 scans every open PR, and lists PRs over 100 files through `gh api .../files --paginate`. Scanned 2026-10-29: no open PR holds 083, and migration 104 is still free.
> - **I4 (owner question, Decision 5):** for each spend limit, the decision says who sets it and whose money it protects. Recommendation, implemented: exempt the hosted billing plan's `llm_tokens_month`, which protects the operator's provider bill (Task 1's sum and Task 3's pre-check). Keep counting: virtual-key budgets (a spend cap the key's creator set), the Resource Governor token caps (capacity), and `limits.evaluation_tokens_per_day` (a per-feature volume allowance). Each row states the diff that flips it.
> - **I5 (owner question, Decision 3):** the decision lists every other boundary with file:line (28 call sites in 19 files, plus the media ingest job start). Full coverage is about 2-3 days, so the recommendation is this PR's two endpoints plus F1. The task's description and AC #4 now name the endpoints covered and the ones that still refuse.
> - **Docs mirror order (fixed, = M1):** Task 4 Step 7 runs `git add` on the docs and the new mirror file before `tests/Docs`.
> - **M1 (fixed):** see above.
> - **M2 (fixed):** Task 1 Step 5 says the pre-088 and pre-054 fallbacks are unchanged (same text, same indentation).
> - **M3 (fixed):** the follow-ups are now filed in Task 0 (M4), so `TASK-<F1>`/`<F2>` exist before the close-out writes the final summary.
> - **M4 (fixed):** Task 0 allocates `<ID>`, `<F1>` and `<F2>` together and files all three. It commits them with the plan, pushes a draft PR, and releases the lock once the PR's file list shows the task files.
> - **M5 (fixed):** the insert fallback calls `missing_credential_source`, which warns once when the column is missing. The retry itself is unchanged, and the row still counts.
> - **M6 (fixed):** the wider-set split puts every `*_postgres.py` file in the PostgreSQL part.
> - **M7 (fixed):** Decision 4 says that raising an org's or team's limits also raises its members' server-key allowance.
> - **M8 (fixed):** `_wrap_chat` and `_wrap_achat` run a `user` call that has no plan inside `accounted_call()`. Test: `test_a_provider_method_nested_in_an_own_key_call_is_not_checked_again`.
> - **M9 (fixed with a comment):** the OAuth retry re-resolves the same user OAuth credential. The recorded source can differ only if that key is deleted mid-request, which under-counts one call. A comment in `chat.py` says so; passing the handle back is not worth the coupling.
> - **Not in the review, found while re-measuring:**
>   - The function-size ratchet (TASK-13950.8) merged after r1. The plan grows four frozen functions, so Tasks 1-3 now bump their ceilings with reasons (Global Constraints).
>   - `test_quality_triage --enforce` flagged the refused-failover test as `tautology_suspect`, so it also asserts the refusal's category.
>   - The owner's ADR-004 waiver now covers this PR (recorded 2026-10-26), so Task 5 records it instead of asking.
>   - Open PR #3361 touches `messages.py` and `usage_accounting.py` (Review Focus).
> - **Re-measured** on `f8f913fab8`. The table "Re-measured counts, r1 to r2" under "Test commands" has every old and new count. The headline:
>   - Task 1: `6 failed` and `2 failed`, then `6 passed` and `2 passed`.
>   - Task 2: `18 failed, 5 passed`, then `23 passed`.
>   - Task 3: `13 failed`, then `13 passed`.
>   - The wider SQLite part: `2 failed, 5826 passed, 6 skipped` on base, `2 failed, 5868 passed, 6 skipped` after (the same two failures). PostgreSQL: `19 passed`, then `21 passed`.

**Goal:** A call made with the user's own stored provider key is still recorded, with its real tokens and cost, but the per-user LLM limits neither count it nor refuse it.

**Architecture:** Credential resolution already knows whose key a call uses (`ResolvedByokCredentials.source`: `user`, `team`, `org`, `server_default`, `none`). The fact is dropped before the usage recorder and the quota checks see the call. The plan carries it to the two places that need it:
- **The usage row.** A new nullable column, `llm_usage_log.credential_source`, stores the source. The per-user LLM counters and the hosted billing plan's token sum leave out `user` rows. On a schema without the column they count every row, so a failed migration never switches a limit off.
- **The check at the call.** `user_to_check` skips a call whose source is `user`.

**What counts as the user's own key.** A user key is the user's own spend only on a provider that bills per key, sent to that vendor's own host (`VENDOR_HOSTS`). A key stored for a custom OpenAI-compatible slot or a local server, or for a billed provider whose base URL the operator pointed at its own model server or a gateway, reaches an endpoint the operator runs, so it is recorded as `user_unmetered`: it is checked and it counts (`billed_credential_source`, Decision 7).

The source travels in two ways:
- **A runtime credential handle** (`ProviderCallCredentials`) gains a read-only `credential_source`. Most callers put that handle in the adapter request under `_provider_call_credentials`.
- **A caller that resolves the key itself** and passes a raw API key marks the call context instead (`LLMCallContext.credential_source`).

At the request boundary, `/chat/completions` and the Messages API name their provider, so an over-limit user whose provider resolves to their own key is let through without the "admitted" mark. Each of that request's calls is then checked where its key is known. On `/chat/completions` the hosted billing plan's token pre-check is skipped for such a request too. Every other boundary still refuses an over-limit user before the key is known (Decision 3, follow-up F1).

**Tech Stack:** Python, FastAPI, SQLite and PostgreSQL (AuthNZ pool), pytest, loguru.

**Spec:** the owner's decision of 2026-10-26, verbatim: "BYOK users own keys shuold not count against their own costs". It reverses owner decision 3 of `Docs/Design/2026-10-09-llm-usage-accounting-design.md` ("BYOK calls are gated too, for now ... exempting them needs new plumbing. That is a follow-up."). It was raised by the final review of PR #3346 (TASK-13936). The backlog task TASK-<ID> (filed in Task 0) holds the ACs. Out of scope: OpenRouter's own `is_byok` flag, which #3346 already prices.

**Measured on:** origin/dev `03fec47781` (r3, after #3362 and #3363), in a detached worktree, 2026-10-29. r2 was measured on `f8f913fab8`. The r1 prototype was measured on PR #3346's head `b94318cce9`, whose tree is the same as `35e711080a`. The r3 prototype is `<scratchpad>/fold_r3_full.patch` (code, tests and the size baseline, with `TASK-99999` standing in for the id in the baseline only); the r2 one, `fold_final_full.patch`, is kept. The docs are in this plan, and `<scratchpad>/fold_apply_docs.py`, run from the repository root, applies them verbatim from Task 4. The worktree was reverted afterwards. Every count below was measured with the text in this plan. `file:line` citations are for `35e711080a`, except where a line says `f8f913fab8`. Since then `chat_service.py`, `usage_accounting.py`, `chat.py` and `messages.py` have moved, so every step in those files anchors on text, not on a line.

---

## What was verified

### 1. How a call's credential is chosen

`resolve_byok_credentials` (`AuthNZ/byok_runtime.py:2425-2879`) resolves one provider for one user, in this order:
1. **The user's own key** (`user_provider_secrets`), when BYOK is on, a user is known, the provider is allowlisted, and the caller did not bind another source (`:2501-2516`, `:2521-2636`). It returns `source="user"` (`:2624`).
2. **A team shared key** (`:2694-2780`, `source="team"`).
3. **An org shared key** (`:2782-2868`, `source="org"`). Both come from `org_provider_secrets`.
4. **The server fallback** (`_fallback_result`, `:755-858`). It returns `source="server_default"` when a key or AWS chain resolves, else `"none"` (`:837-838`).

Related code:
- `ResolvedByokCredentials.uses_byok` is true for `user`, `team` and `org` (`:622-624`).
- Speech gateways have their own resolver (`resolve_gateway_byok_credentials`, `:2330-2422`). It is TTS only, not an LLM quota.
- Two endpoint modules manage keys and never record usage or read a limit:
  - `endpoints/shared_keys_scoped.py`: an org or team manager stores, lists, tests and deletes shared keys (`:164-`, `_require_org_manager`, `:104`);
  - `endpoints/admin/admin_byok.py`: a platform admin lists and revokes user keys, manages shared keys and runs validation (`:97-232`).

**How the key reaches the call:**
- `ProviderCredentialRuntime` (`AuthNZ/provider_credential_runtime.py:372-`) caches one resolution per provider and issues a `ProviderCallCredentials` handle per call (`_new_handle`, `:846-874`). The handle carries the key, `app_config` and `auth_source`, **but not the source** (`:173-309`).
- Callers put the handle in the adapter request under `PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY = "_provider_call_credentials"` (`:53`). 25 app files do so: `chat.py:5073,5153`, `chat_service.py:2470`, `messages.py:1167`, `Summarization_General_Lib.py:262,350`, `ms_g_eval.py:132`, `prompt_executor.py:341`, the RAG pipeline through `attach_runtime_provider_credentials` (`rag_service/runtime_provider_call.py:23-38`), and others.
- Adapters bind the handle in `bind_provider_call_credentials` (`LLM_Calls/adapter_utils.py:57-105`). They copy the request first, so the caller's dict, which the recorder sees, still holds the handle.
- **Three callers resolve the key themselves and pass a raw key:**
  - prompt improvement (`Prompt_Management/prompt_improvement_dispatch.py:178-214`);
  - shared-workspace answers (`Sharing/shared_workspace_chat_service.py:693-723`);
  - the auto-router in `Chat/chat_route_resolver.py:180-200`.

  `endpoints/sharing.py:657` only checks availability. `admin_llm_providers_service.py:465` is an admin key test.

**Where the fact is lost:**
- The adapter recorder records only the user, operation and endpoint (`usage_accounting._plan_call`, `LLM_Calls/usage_accounting.py:47-61`; `_submit_record`, `:167-206`).
- `log_llm_usage` (`Usage/usage_tracker.py:338-576`) and `insert_llm_usage_log` (`AuthNZ/repos/usage_repo.py:493-664`) have no source field.
- The check at the call knows only the user (`user_to_check`, `Usage/llm_quota_gate.py:49-71`).

So nothing tells the usage recorder or the quota checks that a call used the user's own key.

**A user key is not always the user's spend (review B1).**
- `custom-openai-api`, `custom-openai-api-2` and slots 3-99 are on the **default** BYOK allowlist (`AuthNZ/byok_helpers.py:33-56`).
- A user's row for them can carry only `org_id` and `project_id` (`LLM_Calls/provider_metadata.py:39-51`). So the endpoint always comes from the operator's config: `_build_app_config` keeps `api_ip`, `base_url`, `api_base_url` and `endpoint` from the server section (`AuthNZ/byok_runtime.py:135-170`, `:936-957`).
- `resolve_byok_credentials` returns `source="user"` for any stored row with a non-empty key (`:2562-2636`). So a junk key on a slot that fronts the operator's own vLLM resolves as `user` and reaches the operator's GPU. The review measured this (probe C).
- `provider_requires_api_key` (`provider_metadata.py:7-36`, `:56-60`) is False for every custom slot and for llama.cpp, kobold, ooba, tabbyapi, vllm, local-llm, ollama, aphrodite and mlx. It is True for the vendors that bill per key. Measured on `f8f913fab8`: all 124 registered adapters are in that table, and canonical names (`canonical_provider_name`) resolve aliases such as `openai-compatible` and `llamacpp`. The default allowlist's only names outside the table are `elevenlabs`, `fish_s2` and `voyage`, which make no LLM calls.
- A trusted request may let a user set a `base_url` on a provider in `BYOK_ALLOWED_BASE_URL_PROVIDERS` (`byok_runtime.py:512-545`). A custom slot reached that way is still counted, which is the safe direction.

**Nor is a user key on a billed provider always the user's spend (re-review I1; checked on `03fec47781`).**
- Every billed adapter takes its base URL from its `app_config` section before its own default, for a handle call too: `openai_adapter.py:243-258`, `anthropic_adapter.py:93-105`, `groq_adapter.py:45-56`, `mistral_adapter.py:89-100`, `openrouter_adapter.py:70-82`, `google_adapter.py:161-173`, `deepseek_adapter.py:169-178`, `qwen_adapter.py:114-140`, `huggingface_adapter.py:137-189` (also `router_base_url`), `bedrock_adapter.py:280-310` (`runtime_endpoint` first), `moonshot_adapter.py:191`, `zai_adapter.py:110`, `cohere_adapter.py:141`, and `custom_openai_adapter.py:138-162` for Novita, Poe and Together.
- The server config fills those sections. `config.txt` sets `qwen_api_base_url` and `huggingface_api_base_url` (`:982`, `:999`); the loader also reads `moonshot_api_base_url`, `zai_api_base_url` and `bedrock_runtime_endpoint` (`config.py:3957,3968,3980`). `load_server_config_snapshot` (`byok_helpers.py:333-392`) adds environment values: `OPENAI_API_BASE_URL`, `ANTHROPIC_BASE_URL`, `GROQ_BASE_URL`, `MISTRAL_API_BASE` and the rest of `byok_config.PROVIDER_RUNTIME_ENV_CONFIG_KEYS`. `_build_app_config` copies a section's endpoint keys into the resolution's `app_config` (`byok_runtime.py:912-972`).
- So a stored junk key for `qwen`, `moonshot`, `zai` or `huggingface` (all on the default allowlist) whose base URL the operator pointed at its own server resolved as `user`: the re-review's probe D, reproduced here as the RED of the parametrized junk-key test.
- Only a platform admin can set a per-request `base_url` (`chat_service._resolve_base_url_override`, `is_trusted_base_url_request`, `byok_helpers.py:249-266`). It reaches the adapter outside `app_config`, so the rule does not see it; it is the operator's own call.
- Measured: with the shipped `config.txt` and no base-URL environment variable, all 16 billed providers keep `user` (the four base URLs `config.txt` sets are the vendors' own).

### 2. Every place LLM usage counts against a limit

| Limit | What it reads | Own-key call today | After this plan |
|---|---|---|---|
| `limits.llm_tokens_per_month` | `_sum_llm_column_since("total_tokens")` (`usage_repo.py:255-273`), through `quota_checks.llm_tokens_this_month` (`:172-175`) | counted | **not counted** (`user` rows only; a schema without the column counts every row) |
| `limits.llm_cost_per_day_usd`, `limits.llm_cost_per_month_usd` | `_sum_llm_column_since("total_cost_usd")` (`usage_repo.py:275-277`), through `llm_cost_today` / `llm_cost_this_month` (`quota_checks.py:178-187`) | counted | **not counted** |
| The boundary check: `enforce_llm_quota` / `require_llm_quota` (`API_Deps/usage_quota_deps.py:47-57`), on about 30 endpoints, then `mark_llm_quota_prechecked` (`llm_call_context.py:120-127`) | `llm_quota_refusal` (`quota_checks.py:292-322`) over the sums above | refused when over | `/chat/completions` (`chat.py:4380`) and the Messages API (`messages.py:2233,2264`) let the user through when the provider resolves to their own key; the other 28 call sites are unchanged (Decision 3 lists them; F1) |
| The check at the call: `perform_chat_api_call(_async)` (`chat_service.py:2646,2678`), the four adapter wrappers (`usage_accounting.py:335-445`) | `user_to_check` and `check_user_*` | refused when over | **not checked** |
| A media ingest job's check when it starts (`services/media_ingest_jobs_worker.py:363-380`) | `user_to_check` and `check_user_async` (the credential is not known yet) | refused when over | unchanged (F1) |
| The profile's `llm` block, `llm_quota_summary` (`quota_checks.py:361-370`) | the same sums | shown | shown without own-key calls, matching the limits |
| Resource Governor tokens/day cap | the daily ledger's `tokens` category: the shadow write in `log_llm_usage` (`usage_tracker.py:535-571`), the RG reserve on `/chat/completions` (`chat.py:4192`), and the legacy backfill (`usage_tracker.py:82-158`, `chat.py:4173`) | counted | unchanged (Decision 5) |
| Virtual-key budgets (`llm_budget_day_tokens`, `_day_usd`, `_month_tokens`, `_month_usd`) | `is_key_over_budget` (`AuthNZ/virtual_keys.py:109-143`) over `summarize_key_day` and `summarize_key_rolling_window` (`usage_repo.py:113-184,279-352`). Enforced by `llm_budget_guard.py:118` and `llm_budget_middleware.py:362`. Only rows with a `key_id`: `/chat/completions` and router rows | counted | unchanged (Decision 5) |
| Hosted billing plan `llm_tokens_month` (the free plan's default is 300,000, `Billing/plan_limits.py:41-56`) | `BillingEnforcer._get_llm_tokens_month` (`Billing/enforcement.py:371-455`), and the `LimitEnforcer` pre-check on `/chat/completions` (`chat.py:4398-4413` on `f8f913fab8`), which runs before `enforce_llm_quota` and raises its own 402 (`API_Deps/billing_deps.py:446-468`). Active only when `billing_checks_active()` (quotas on and a billing repository wired: the hosted product, `Billing/enforcement.py:1000-1018`) | counted on SQLite; **on PostgreSQL nothing counts**, own key or not: the aware month start meets a `TIMESTAMP` column, asyncpg raises `DataError`, and the sum returns 0 (re-review I2) | **not counted, and not checked** for a request whose provider runs on the user's own key; on PostgreSQL every other call now counts (Decision 5) |
| `limits.evaluation_tokens_per_day` | the Evaluations DB's `daily_usage.total_tokens`, not `llm_usage_log` | counted | unchanged (Decision 5) |

The claims budget guard (`Claims_Extraction/budget_guard.py`) is not a limit on stored usage. It is a per-job budget the request sets, kept in memory.

**Readers that only display usage (not limits):**
- the admin usage reports (`services/admin_usage_service.py`);
- router analytics and its key list (`services/admin_router_analytics_service.py`);
- `services/admin_system_service.py:232,243`;
- the daily rollup (`services/llm_usage_aggregator.py`, `usage_repo.aggregate_llm_usage_daily_for_day`);
- claims provider analytics (`Claims_Extraction/claims_service.py:1325-1400`);
- the debug budget endpoint (`endpoints/authnz_debug.py:149-151`);
- the Prometheus `llm_cost_dollars` and `llm_tokens_used_total` counters (`usage_tracker.py:432-489`).

They keep showing every row; the new column lets a report split own-key usage later.

### 3. Callers that bypass the adapter recorder

`log_llm_usage` is the only writer: `insert_llm_usage_log` has one caller (`usage_tracker.py:500`), and no app code inserts into `llm_usage_log` directly (`git grep "INSERT INTO llm_usage_log"` finds only `usage_repo.py`). These call `log_llm_usage` themselves instead of through the adapter recorder:

| Recorder | Matters here? |
|---|---|
| `/chat/completions`: `chat_service._record_dict_response_usage` (`chat_service.py:6906-7029`, two calls) and the streaming `save_callback` (`:6336-6353`). The call runs under `records_itself` | **Yes.** It is the main chat path. Task 2 passes the source of the handle in `cleaned_args`, which is updated on failover (`:5840`, `:7293`). The queued stream failover (`:5471-5605`) does not update `cleaned_args`, so Task 2 tracks the source in its own variable. |
| The Messages API native path: `record_direct_response` and `accounted_direct_stream` (`messages.py:2069,2093`) | **Yes.** The `credentials` handle is in scope. |
| Router calls: `routing/accounting.log_model_router_usage` (`:51-89`), called from `chat.py:1756`, `chat_route_resolver.py` and `character_chat_sessions.py:1172` | Little: at most 64 output tokens a call. Follow-up F1. `chat.py`'s router call passes the handle, so its check at the call already sees the source after Task 2. `chat_route_resolver`'s router call does not, so an over-limit user's router call there is refused and routing falls back to rules (`chat.py:1707-1710`, today's behaviour). |
| The claims engine: `_log_claims_llm_usage` (`Claims_Extraction/claims_engine.py:660-`), under `records_itself` (`:840,1206`) | No. It calls `analyze` with `api_key=None` and no handle, so it always uses server keys. |
| Embeddings (`embeddings_v5_production_enhanced.py:4117,4799`, `Notes_Graph/semantic_embeddings.py:558`) | No. `operation="embeddings"` rows are already outside the LLM limits (`usage_repo.py:15`). |
| Exempt calls: key validation (`byok_testing.py`) and first-run verification | Never recorded. |

---

## Global Constraints

- **Git and PR.** Branch `fix/byok-quota-exempt-<ID>` from `origin/dev` (`03fec47781` or later; #3346 has merged). The PR targets `dev`. Never `--no-verify`, never force-push, never `git stash`.
- **TDD.** Fix test-first: write the test, run it, see it fail for the stated reason, then fix. The counts below are measured; record the real ones.
- **No network in tests.** Stub transports (the tests use fake providers, a fake registry and monkeypatched recorders). PostgreSQL only through the repo's fixtures (the `test_db_pool` fixture on the `tldw_postgres_test` container; drop only what you created). Never FLUSHALL Redis.
- **WebFetch** only for vendor API, pricing and usage docs. This plan needs none.
- **Backlog.** Use only backlog-py: `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task ...`.
  - `--append-notes`, never `--notes`.
  - Stage with `git add -A backlog/tasks`.
  - Never hand-edit task files.
  - New tasks get an explicit `--id`, allocated under the controller's id lock.
- **Python.** The worktree has no `.venv`; use `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (written `$PY` below; shell variables do not survive between agent calls, so write it out).
- **`TASK-<ID>`** is the id Task 0 allocates. Write the real id wherever the code, tests, docs, branch and commit messages this plan gives say `<ID>`. Before the final commit, `git grep -n "TASK-<ID>" -- ':!Docs/superpowers/plans'` must print nothing.
- **Numbering.** This plan uses AuthNZ SQLite migration **104** and **ADR-083**. On `f8f913fab8` the last migration is 103 and the last ADR is 081, and open PR #3323 holds ADR-082 (scanned 2026-10-29).
  - If `origin/dev` or an open PR has a later migration or ADR when Task 0 runs, use the next free number everywhere this plan says 104 or 083.
  - That includes the legacy-database version in the Task 1 test (103 becomes the new number minus 1) and the pinned `MAX(version)` in Task 1 Step 5.
- **Compatibility.** `log_llm_usage`, `insert_llm_usage_log`, `record_direct_response`, `accounted_direct_stream`, `user_to_check`, `check_llm_call_sync/async`, `enforce_llm_quota` and `ProviderCallCredentials.__init__` gain only optional keyword parameters, with defaults that keep today's behaviour. New: `billed_credential_source`, `USER_KEY_UNMETERED_SOURCE` and `VENDOR_HOSTS` (`llm_call_context.py`), `missing_credential_source` (`usage_repo.py`), `users_own_key_serves` (`usage_quota_deps.py`) and the private `BillingEnforcer._sum_org_llm_tokens`.
- **Logging.** Never log keys or usage payloads. The new warnings log the exception type and a fixed message (`users_own_key_serves`, `missing_credential_source`). The PostgreSQL DDL failure keeps its existing text, `PG ensure usage tables DDL failed: {exc}`, at warning instead of debug, for the errors that handler catches (`_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS`: a timeout, a lost connection and the like). An asyncpg database error such as a missing privilege is not caught there and stops startup, as today (review m1).
- **CI traps:**
  - `Helper_Scripts/ci/scope_predicate_baseline.txt` lists none of the touched files (checked; re-check in Task 5).
  - This plan is committed in the PR, so it must have no trailing whitespace, including inside embedded code. Before each commit, `git add` the commit's files, then run `git diff --cached --check`; it must print nothing.
  - **OpenAPI drift.** Removing `Depends(require_llm_quota)` from the two Messages routes does not change the schema. Measured: `export_openapi_schema.py --check` says "OK — OpenAPI fingerprint matches" and "OK — committed path union matches". Re-run it in Task 5.
  - **`Docs/Published`.** ADR files, `Docs/ADR/README.md` and the BYOK user guide are mirrored. Task 4 runs `bash Helper_Scripts/refresh_docs_published.sh` and commits the mirror diff. `Usage_Quotas.md` and the design doc are not mirrored.
  - **`test_quality_triage.py --enforce`.** `test_chat_llm_tokens_quota_402.py` is at its `mock_density=2` baseline, so its own-key test uses `refuse_llm_work` instead of two more `monkeypatch` calls, and its hosted-plan test sets the plan up through the module helper `_on_a_spent_hosted_plan`. Each new test there makes three mock calls of its own, under the threshold of four. Every collector-list assertion is paired with an assertion on a real output, to avoid `tautology_suspect`.
  - `check_shard_coverage.py` must stay green.
  - **The function-size ratchet** (`tldw_Server_API/tests/lint/test_function_size_ratchet.py`, TASK-13950.8, merged after r1). It freezes every function over 300 lines at its current size in `tldw_Server_API/tests/lint/function_size_baseline.json`. This plan grows four of them, so Tasks 1-3 each raise their ceilings to the measured size, with a reason, in the same commit ("bump with reason": a non-empty new reason without `<...>` placeholders; a raised number must equal the measurement). Splitting `create_chat_completion` or `execute_streaming_call` is out of scope. Measured on `f8f913fab8`: `get_authnz_migrations` 441 to 442 (Task 1), `_execute_non_stream_call_impl` 1045 to 1054 and `execute_streaming_call` 1536 to 1545 (Task 2), `create_chat_completion` 2897 to 2907 (Task 3). If origin/dev has moved, use the numbers the ratchet's failure message prints.
- **Schema.** The change covers SQLite (migration 104) and PostgreSQL (the `ensure_usage_tables_pg` DDL), with a test on each.
- **Global defaults.** The plan flips no setting. It adds a column, keyword parameters, one dependency change on two routes, and a log level (debug to warning) for a failed PostgreSQL usage-table statement. One behaviour changes for every hosted deployment on PostgreSQL: the hosted plan's `llm_tokens_month` has counted nothing there (re-review I2), and after this plan it counts every call except own-key ones, so a plan that has never refused starts refusing at its limit. It applies only where `billing_checks_active()` (Decision 5).
  - The wider set in "Test commands" is every test file that names a touched symbol, and Task 5 runs all of it.
  - One existing test pins the latest migration (`test_native_chat_quota_reservations.py:753`, `MAX(version) == 103`), and Task 1 updates it.
- **Bandit** runs only through `/opt/homebrew/bin/uvx bandit`. Never install into the shared venv.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- **Close-out.** Task 0 files TASK-<ID> and the follow-ups. The last task closes TASK-<ID> with backlog-py (notes, ACs, DoD, final summary).

## Decisions

Recommendation first. The plan implements each recommendation; each item gives the cost if it is wrong and the alternatives. Decisions 3, 4 and 5 are **owner questions**: each says what changes if the owner answers otherwise.

1. **Record the call with its real tokens and cost, and exclude it from the limit sums through a new column (recommended).**
   - `llm_usage_log.credential_source TEXT NULL` holds the source as the quotas read it. That is the resolution's `user`, `team`, `org`, `server_default` or `none`, except that a user key that is not the user's spend (a provider that does not bill per key, or a billed provider sent anywhere but its vendor) is stored as `user_unmetered` (Decision 7).
   - NULL means unknown: rows written before the column, and callers that pass a raw key without marking the context.
   - The three limit sums, and the hosted plan's token sum (Decision 5), add `COALESCE(credential_source, '') <> 'user'` beside their existing predicates.
   - Storing the source, not a boolean, costs nothing more. It lets Decision 4 be reversed in one predicate, without re-recording, and lets a report split usage by key later.
   - No existing column fits:
     - `estimate_source`, `operation` and `currency` mean something else;
     - `raw_usage_metadata_json` exists only when a usage block does, and filtering JSON in SQL differs between SQLite and PostgreSQL.
   - **Insert compatibility.** `insert_llm_usage_log` already falls back to older statements for pre-088 and pre-054 schemas. When a row has a source and the statement fails, it retries the same statement without the column. The row is kept and counts. When the error names the column, `missing_credential_source` logs a warning, once.
   - **Read compatibility (review B2).** On a schema without the column the sums would fail, and the quota checks treat a failed counter as unlimited (`Usage/quota_checks.py:68-74`, `:280-287`). So each sum retries without the predicate when the error names `credential_source`: every row counts, and the same warning fires once. That schema is reachable on PostgreSQL: `ensure_usage_tables_pg` logged a failed statement at debug and still returned True (`AuthNZ/pg_migrations_extra.py:4190-4196`). It now logs a warning, for the errors that handler catches (a timeout, a lost connection). A database error such as a missing privilege or PostgreSQL's own lock timeout is not one of them: it stops startup, as it does today, which is louder still (review m1).
   - *Cost if wrong:* one nullable column and one migration per backend.
   - *Alternatives:*
     - (b) **Record the call at $0.** No schema change, and every cost reader excludes it for free. But the token limit still counts it (see Decision 2), the user and operator lose the real cost, and a $0 row cannot be told from a free local call.
     - (c) **Do not record own-key calls.** Loses the record, against ADR-067.
     - (d) **Check once per pool whether the column exists** (the review's other option). It needs a catalog query per backend and a cache to invalidate, for the same result as the retry.
2. **Exempt both the token limit and the two cost limits (recommended).**
   - The owner wrote "costs". But `limits.llm_tokens_per_month` and the cost keys budget the same thing, the operator's provider bill, in two units: the operator's page groups them ("What `limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd` count").
   - An operator who caps a free tier in tokens would otherwise refuse a user working on their own key. That is the complaint being fixed.
   - Server load from own-key calls stays bounded by the Resource Governor, per minute only (Decision 5): on chat, by default, 300 requests and 1,000,000 tokens a minute per user, with no daily cap.
   - *Cost if wrong* (the owner meant cost only): an own-key user can use unlimited tokens of their own key through the server. The operator's exposure is server load, which RG bounds at those per-minute rates. To reverse, call the token sum without the new predicate: one line in `usage_repo.py`.
   - *Alternative:* cost limits only. The token limit keeps refusing own-key users.
3. **Pre-call refusal: never refuse a call that runs on the user's own key. At the boundary, decide by the provider where the boundary knows it (recommended; OWNER QUESTION on scope).**
   - **At the call** (the shared call layer, the four adapter wrappers, the native Messages path), a call whose source is `user` is not checked. So a job, workflow, WebSocket or unlisted endpoint makes its own-key calls even for a user over a limit, and still refuses that user's server-key calls.
   - **At the boundary** the key is not known yet. `/chat/completions` (`selected_provider`, the `enforce_llm_quota` call at `chat.py:4415` on `f8f913fab8`) and the Messages API (`_resolve_provider_and_model_for_request`, first line of `_handle_messages`) know the provider before any LLM work.
     - For an over-limit user, `enforce_llm_quota(..., provider=...)` asks `users_own_key_serves(user_id, provider)`. That answers True when the provider bills per key and the server's config sends it to the vendor (Decision 7), BYOK is on, the provider is allowlisted, and the user has an active key for it. Apart from the vendor test, that is the first test `resolve_byok_credentials` makes.
     - If true, it returns without `mark_llm_quota_prechecked()`, so every call in that request is checked at the call. Own-key calls pass. A failover to a server-key provider, a server-key tool call, or a native call whose handle turns out not to be the user's is refused there, and is not charged to the provider's health (review I1).
     - The native Messages path has no `ChatProvider`, so Task 3 gives it the same check at the call: a no-op once a request is admitted.
   - **Scope (owner question, review I5).** This PR covers `/chat/completions` and the Messages API (`/messages`, `/v1/messages`). Every other boundary refuses an over-limit user before the key is known. So a user who has spent the allowance and then adds their own key still gets a 402 at these boundaries (call sites on `f8f913fab8`; character chat re-checked on `03fec47781`, where #3363 moved it by 4 lines):

     | Boundary | Provider known before LLM work? | Work to cover it |
     |---|---|---|
     | Character chat: `character_chat_sessions.py:7010` (completions), `:6785` (roleplay resume) | Resume: yes, `effective["provider"]` at `:6790`. Completions: only after the settings, the context and the auto-router (`:7219-7295`), about 280 lines after the check. | Move the check below provider resolution, and confirm the router call before it (checked at the call, unmarked) falls back to rules. |
     | RAG: `rag_unified.py:823`, `:1480`, `:1688`, `:2088`, `:2253`, `:2387` | Generation names its provider in the request. | Pass the generation provider. |
     | Evaluations: `evaluations_unified.py:993`, `:1210`, `:1408`, `:1698`; `evaluations_crud.py:328`; `evaluations_benchmarks.py:109`; `benchmark_api.py:226` | The judge provider is in the request, with a configured default. | Pass it. An own-key batch or benchmark run is then checked per call, not once. |
     | Study tools: `flashcards.py:2466`, `quizzes.py:784` (route dependencies) | In the body. | Turn the dependency into an inline call, as Task 3 does for Messages. |
     | Chat documents `chat_documents.py:366`; watchlists `watchlists.py:7119` | In the request or the stored config. | Pass it. |
     | Media: `media/add.py:85`, `ingest_web_content.py:80`, `process_audios.py:161`, `process_documents.py:152`, `process_ebooks.py:209`, `process_emails.py:98`, `process_pdfs.py:159`, `process_videos.py:139`, `process_web_scraping.py:79` | `api_name` on the form, when analysis is asked for. | Pass `api_name` through one shared helper. |
     | Media ingest job start: `services/media_ingest_jobs_worker.py:363-380` | `form_data.api_name`. | `_admit_or_refuse_analysis` asks `users_own_key_serves` and does not mark the job admitted. |

   - **Recommendation: this PR's two endpoints, plus follow-up F1 for the rest.** Full coverage is 28 call sites in 19 files plus a worker. Each site needs its provider resolved before the check, and each endpoint family needs a test. The routing and batch contracts above need care, too. Estimate: 2-3 days and about 600 lines with tests, which wants its own reviewed plan. It is not a mechanical extension of this PR. The task's description and AC #4 say plainly what is covered and what still refuses. F1 is filed in Task 0, character chat first.
   - *Cost if wrong* (the owner wants every endpoint now): until F1 ships, own-key users over a limit keep getting 402 on character chat, RAG, evaluations, the study tools, chat documents, watchlists and media analysis. Meanwhile the operator can lift a user's limit.
   - *If the owner answers "all in this PR":* F1's ACs become Tasks 3b-3h of this plan, and the plan goes back for review. Tasks 1-4 do not change, because `enforce_llm_quota(..., provider=...)` is the seam every boundary uses.
   - *Alternatives:*
     - (b) **Change no boundary.** Simplest, but the main BYOK use case fails: a free-tier user who adds their own key is still refused on `/chat/completions` and Messages until the month resets.
     - (c) **At every boundary, let through any over-limit user who has an own key for any provider, unmarked.** One change in `enforce_llm_quota`, but it moves the refusal from the boundary to the call on about 30 endpoints. It also breaks the "checked once" guarantees those endpoints document: an admitted benchmark run or media batch completes, with no item refused mid-run.
   - **Known side effect.** `llm_quota_refusals_total` counts the boundary refusal even when the user is then let through on their own key. The check runs before the key lookup, and the counter is shared with the check at the call. It is narrow (over-limit own-key users on two endpoints) and only a metric.
4. **Org and team shared keys still count against the member's limits (recommended; OWNER QUESTION).**
   - The owner's words are "users own keys". A shared key is not the user's own, and not the operator's: the org or team that stored it pays.
   - The `limits.*` values are the operator's, and an org admin cannot set them (ADR-064). A platform admin who caps a member also caps the member's use of the org key.
   - Counting keeps today's behaviour for the case the owner did not name.
   - *Cost if wrong:* members of an org that pays for its own key are refused by the operator's limits. The remedy is configuration: the platform admin raises the org's or team's `limits.*` values. That is not the same as exempting the key (review M7): it also raises those members' allowance for calls on the server's keys.
   - *To reverse:* exempt the two sources in every place:
     - in `llm_call_context.py`, replace `USER_KEY_SOURCE = "user"` with `OWN_KEY_SOURCES = frozenset({"user", "team", "org"})`, test membership in `llm_quota_gate.user_to_check`, and apply `billed_credential_source`'s rule to `team` and `org` as well, so a shared key on an operator endpoint still counts;
     - in `usage_repo.py`, use `COALESCE(credential_source, '') NOT IN ('user', 'team', 'org')`, and the same in `Billing/enforcement.py`'s `_OWN_KEY_EXCLUDED`;
     - in `users_own_key_serves`, add the team and org lookups.

     Nothing is re-recorded, because the column stores the source.
   - **Ask the owner before Task 0.** If they choose exemption, apply the reversal in Tasks 1-3 and do not file F2.
5. **Which other spend limits leave out own-key calls (OWNER QUESTION, review I4).** The owner said own keys "should not count against their own costs". A limit that protects the operator's provider bill should not count them. A spend cap that the user, or an admin for that user, set on themselves can.

   | Limit | Who sets it | Whose money it protects | Recommendation | To change the answer later |
   |---|---|---|---|---|
   | `limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd`, `limits.llm_cost_per_month_usd` | The operator (platform admin; ADR-064). | The operator's provider bill. | **Exempt** (Decisions 1-2). | See Decision 2. |
   | The hosted billing plan's `llm_tokens_month`: `BillingEnforcer._get_llm_tokens_month` and the `LimitEnforcer` pre-check on `/chat/completions`, which runs before `enforce_llm_quota` and raises its own 402 (`API_Deps/billing_deps.py:446-468`). The free plan's default is 300,000 (`Billing/plan_limits.py:41-56`). | The operator's plan catalog, per org (the hosted product: `billing_checks_active()`, `Billing/enforcement.py:1000-1018`). | The operator's provider bill, priced into the plan. | **Exempt; implemented.** Task 1 leaves `user` rows out of the org's sum, with the B2 retry. Task 3 skips the pre-check when the request's provider runs on the user's own key, so the reply is not added to the plan's cached usage either. **On PostgreSQL this limit has counted nothing until now, own key or not** (re-review I2: the month start was bound timezone-aware against a `TIMESTAMP` column, asyncpg raised `DataError`, and the sum returned 0). Task 1 fixes the bound, so this PR starts it counting there: a hosted plan on PostgreSQL that has never refused will refuse server-key and shared-key calls at its limit. The skip covers the whole request: a failover, or any other server-key call inside an own-key `/chat/completions` request, is not checked against the plan; its row counts in the plan's sum when the cache refreshes (60 s). The lookup runs on every hosted `/chat/completions` request: one indexed read, plus a 2 ms config read when the user has a key for the provider. | Two lines: pass `""` instead of `_OWN_KEY_EXCLUDED` in `_get_llm_tokens_month`, and drop the `users_own_key_serves` condition in `chat.py`. |
   | Virtual-key budgets `llm_budget_day_tokens`, `_day_usd`, `_month_tokens`, `_month_usd` (`AuthNZ/virtual_keys.py:109-143`; enforced by `llm_budget_guard.py:118` and `llm_budget_middleware.py:362`). | The user, for a key they issue themselves (`POST /api/v1/users/api-keys/virtual`, `endpoints/users.py:821-842`), or a platform admin for a user (`POST /api/v1/admin/users/{user_id}/virtual-keys`, `admin/admin_api_keys.py:118`). | Whoever pays for calls through that key: for an own-key call, the user. It is a stop on one issued key (a client tool, a leaked key), not an allowance. | **Keep counting.** | About 25 lines and a test (an hour): add `AND {_OPERATOR_PAID_PREDICATE}` to `summarize_key_day` and `summarize_key_rolling_window`, with the same `missing_credential_source` retry. `summarize_key_day` also feeds the RG legacy backfill (`usage_tracker.py:142`), which would then stop counting own-key tokens too. |
   | Resource Governor token caps (per minute, and the tokens/day ledger). | The operator's RG policies (ADR-056). **On by default** (`rg_enabled()`, `[ResourceGovernor] enabled = true`): the shipped `chat.default` policy allows 300 requests and 1,000,000 tokens a minute per user (`resource_governor_policies.yaml:34-39`). No tokens/day cap is set by default. | Server capacity, not the bill. | **Keep counting.** | About a day: a source marker on the RG reserve and ledger write at `/chat/completions`, which run before the key is known. |
   | `limits.evaluation_tokens_per_day` (`Evaluations/user_rate_limiter.py:358`, `:995`). | The operator (platform admin). | Evaluation volume: a per-feature daily allowance like `limits.rag_queries_per_day`, reserved from an estimate before the run (`evaluations_unified.py:1002`). Evaluation LLM spend is capped by `limits.llm_cost_*` (the limiter's module docstring), which this plan exempts. | **Keep counting.** The least clear of the four, because it is counted in tokens. | About half a day, inside F1: each evaluation route names its judge provider and skips the token reservation when `users_own_key_serves` answers True. |

   - *Cost if wrong:*
     - **Billing plan.** If the owner wants it counted, a free-plan user on their own key uses the server's capacity with no plan cap; RG still bounds the load per minute (300 requests and 1,000,000 tokens on chat), with no daily cap. Either way, on PostgreSQL the plan starts counting every other call with this PR.
     - **Virtual keys.** If the owner wants them exempt, an own-key user calling through a budgeted key is refused once the key's budget is spent, until the key's creator raises it.
     - **Resource Governor.** An own-key user can hit a capacity cap, as any user can.
     - **Evaluation tokens.** An own-key user's evaluations stop at the daily evaluation token cap.
6. **Where the source comes from (recommended: the handle, else the call context).**
   - `call_credential_source(request)` (`Usage/llm_call_context.py`) returns the `credential_source` of a runtime-issued handle in the request, else `LLMCallContext.credential_source`. Both are classified by Decision 7 where they are set.
   - A handle is authoritative when present; the context covers the three callers that pass a raw key.
   - A look-alike handle (not issued by the runtime) is ignored (`is_runtime_issued_provider_call_credentials`), so it cannot skip the check.
   - *Cost if wrong:* a call path that passes neither a handle nor the context mark counts as before. It is never under-counted.
   - *Alternative:* look up "does this user have a key for this provider" at record time. It guesses instead of knowing which key was used: wrong under `required_source`, a disallowed provider or a shared-key binding. It also costs a query per row.
7. **A user key is the user's own spend only on a provider that bills per key, sent to that vendor's own host (recommended; review B1, re-review I1).**
   - `billed_credential_source(provider, source, *configs)` keeps `user` only when the canonical provider is in `VENDOR_HOSTS` (the 16 that `PROVIDER_REQUIRES_KEY` marks as billed) and every base URL that `configs` give that provider's section is on one of the vendor's hosts. Otherwise it returns `user_unmetered`, which is checked and counts. Any other source is returned unchanged, so nothing becomes `user`.
   - The vendor's host is the one its adapter falls back to (Task 2 lists them). The base-URL keys are every key a billed adapter reads (`base_url`, `api_base_url`, `api_base`, `api_url`, `api_ip`, `endpoint`, `runtime_endpoint`, and Hugging Face's two router keys).
   - It is applied where a source is set, so every caller routes through it:
     - `ProviderCredentialRuntime._new_handle`, with the handle's own `app_config` (on a handle call the adapter reads nothing else);
     - the prompt-improvement and shared-workspace context marks, with their `app_config` and the server snapshot (their adapter falls back to the environment);
     - `users_own_key_serves`, with the server snapshot, as its last test.
   - An unknown provider name is `user_unmetered` (review m3). A test keeps `VENDOR_HOSTS` equal to the billed providers in `PROVIDER_REQUIRES_KEY`. "What was verified" §1 has the facts.
   - *Cost if wrong (too strict):*
     - An operator who points a custom slot at a vendor that bills the user's key counts those users' own-key calls. An example is Together behind `custom-openai-api-3`. The remedy is the named provider (`together`), which bills per key.
     - An operator who points a billed provider at another host of the same vendor, one its adapter does not default to (`api.moonshot.ai`, `api.cohere.com`, `open.bigmodel.cn`), counts those calls. The remedy is one more host in `VENDOR_HOSTS`.
   - *Cost if wrong (too loose):* a gateway reached under the vendor's own host name (a DNS override or an intercepting proxy for `api.openai.com`) still reads as `user`. The server cannot see that. A per-request `base_url` is accepted only from a platform admin and does not pass through `app_config`, so the rule does not read it; it is the operator's own call.
   - *Alternatives:*
     - (b) **An operator opt-in list of custom slots that bill the user's key** (the review's question for the owner). Not built: no operator has asked, and the named providers cover the real vendors. If one asks, add `BYOK_BILLED_CUSTOM_PROVIDERS`: one set lookup in `billed_credential_source`.
     - (c) **Decide by `endpoint_provenance`** (`byok` when a trusted request let the user set `base_url`). It would exempt a user-supplied URL that points back into the operator's network.
     - (d) **Docs only** (the re-review's option b): tell operators to take a billed provider off `BYOK_ALLOWED_PROVIDERS` when they point its base URL elsewhere. Rejected: one missed provider gives every BYOK user unlimited use of the operator's hardware, and the condition is checkable in code.
8. **The auto-router's usage row is follow-up F1 (recommended).**
   - A router call is at most 64 output tokens.
   - Carrying the source needs plumbing in three recorders (`chat.py:1756`, `chat_route_resolver.py`, `character_chat_sessions.py:1172`) whose resolution and logging are in separate closures.
   - F1 also holds the other boundaries (Decision 3).

## Review Focus

Inputs and conditions most likely to bite a user, most likely first, each with the test that pins it:

1. **An over-limit user who has their own key for one provider asks for another.** Expected: the 402 at the boundary, as today. Pinned by `test_an_over_limit_user_whose_provider_runs_on_their_own_key_is_let_through_unmarked` (`anthropic` and no provider both refuse). Task 3.
2. **A request let through on the user's own key fails over to a server-key provider, or makes a server-key call.** Expected: that call is refused at the call; when it is admitted, it is recorded with its own source.
   - Refused at the call: `test_a_call_on_any_other_key_is_still_refused` (server, team, org, operator endpoint), `test_the_shared_call_layer_does_not_check_a_call_on_the_users_own_key` (sync and async, second half) and the native `test_a_native_call_let_through_at_the_boundary_is_still_checked_when_not_on_the_users_key`.
   - The refusal does not mark the fallback provider unhealthy for everyone: `test_a_refused_failover_does_not_mark_the_fallback_provider_unhealthy`.
   - Recorded with its source: `test_a_failover_reply_is_recorded_with_the_fallback_providers_key`.
   - Tasks 2 and 3.
3. **Anything that is not a runtime handle claims to be one with `credential_source="user"`.** Expected: ignored, so the call is checked. Pinned by `test_a_handle_the_runtime_did_not_issue_is_ignored`. Task 2.
4. **A user stores a junk key for a custom OpenAI-compatible slot that fronts the operator's own model, or for a billed provider whose base URL the operator pointed at its own server** (review B1, re-review I1). Expected: the call is `user_unmetered`, checked and counted, and the boundary does not admit an over-limit user for it. Pinned by `test_a_junk_key_stored_for_the_operators_endpoint_is_still_checked_and_counted` (real storage and resolution; the custom slot, `qwen`, `moonshot`, `zai` and `huggingface`), `test_a_user_key_is_the_users_own_spend_only_on_the_vendors_host` (vendor, regional and operator hosts, and an unknown provider), the handle test, and the `operator-endpoint`, `local-server` and `operator-base-url` cases of the lookup test. Tasks 2 and 3.
5. **A database not yet migrated** (or one whose migration or PostgreSQL bootstrap failed; review B2). Expected: the row is still recorded, without its source; every row counts, so the limits still refuse; a warning is logged when a bootstrap statement fails with a handled error. Pinned by `test_a_table_without_the_column_still_records_counts_and_refuses`, `test_a_failed_postgres_usage_column_is_logged_as_a_warning` (an `ALTER` that times out), the PostgreSQL missing-column test, and the billing file's existing column-less test. Task 1.
6. **Org and team shared keys.** Expected (Decision 4): they still count, and an over-limit member's call on one is refused. Pinned by `test_an_admitted_call_on_a_shared_key_is_recorded_with_its_source` (`team`), the refused-call test (`team`, `org`) and the counter test (`team`, `org`). Tasks 1 and 2.
7. **A hosted free-plan org member over the plan, on their own key** (Decision 5). Expected: `/chat/completions` answers; a server-key request still gets the plan's 402. Pinned by `test_the_hosted_plan_does_not_check_a_request_on_the_users_own_key`, `test_get_llm_tokens_month_leaves_out_calls_on_a_members_own_key` (SQLite) and `test_the_hosted_plans_llm_tokens_leave_out_a_members_own_key_pg` (PostgreSQL). Tasks 1 and 3.
8. **A hosted deployment on PostgreSQL** (re-review I2). Expected: the plan's `llm_tokens_month` counts, where it counted nothing before, so orgs near their plan limit start getting the plan's 402. Pinned by the PostgreSQL hosted-plan test. Task 1.

Also for the reviewer:
- **Streamed `/chat/completions` behind the queue.** The queued stream path's failover now updates `usage_credential_source`. No test drives the queue (the existing queue tests use dummy queues). Read `chat_service.py` around `_queued_processor`.
- **The Messages dependency became an inline call.** `test_messages_refuses_an_over_quota_user_before_any_provider_call` (4 cases, existing) still passes: a 402 before any provider call, streaming or not, adapter or native path. A request with no model now gets its 400 before the 402.
- **A runtime user that passes only the raw key and not the handle** counts as before. Every runtime user found attaches the handle; `git grep -n "credential_runtime.resolve"` lists the callers.
- **Only the non-stream failover block is driven by a test** (I1). The queued and direct stream blocks get the same two lines; the queue has no test harness (above).
- **Open PR #3361** (TASK-13950.13, the LLM_Calls/Chat cut) touches `messages.py` and `usage_accounting.py`. If it merges first, re-anchor Task 2's and Task 3's edits there; every step anchors on text.
- **Open PR #3369** (TASK-13950.13 PR C1, 135 files, checked 2026-10-29) touches `messages.py`, `chat_service.py`, `usage_accounting.py`, `function_size_baseline.json` and `test_chat_llm_tokens_quota_402.py`, and removes `chat_ep.API_KEYS`. If it merges first, Task 3's two new tests in that file replace `monkeypatch.setitem(chat_ep.API_KEYS, "local-llm", "dummy-key")` with that PR's `monkeypatch.setattr(config_llm_providers, "get_api_keys", api_keys_source({"local-llm": "dummy-key"}))` (still one mock call each, so `mock_density` is unchanged), and the size ceilings are re-measured.

## PR and commits

One PR, branch `fix/byok-quota-exempt-<ID>`, from `origin/dev`, opened as a draft in Task 0. Reasons: the tasks share `usage_tracker.py`, `usage_accounting.py` and the test helpers, and Task 3 depends on Task 2's `user_to_check`. Commits, each with the attribution trailer:

0. `docs(plan): add the BYOK quota exemption plan and file its tasks (TASK-<ID>)`
1. `fix(usage): record whose provider key paid for each LLM call (TASK-<ID>)`
2. `fix(usage): do not check or count LLM calls on the user's own key (TASK-<ID>)`
3. `fix(usage): let an over-limit user through on their own key at /chat/completions and Messages (TASK-<ID>)`
4. `docs(usage): BYOK calls do not count against the LLM limits (ADR-083, TASK-<ID>)`
5. `docs(backlog): close TASK-<ID>`

## Files

- Modify (Task 1): `tldw_Server_API/app/core/AuthNZ/migrations.py`, `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`, `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py`, `tldw_Server_API/app/core/Usage/usage_tracker.py`, `tldw_Server_API/app/core/Billing/enforcement.py`, `tldw_Server_API/tests/Storage/test_native_chat_quota_reservations.py` (one number), `tldw_Server_API/tests/Billing/test_billing_enforcer_org_usage.py` (append), `tldw_Server_API/tests/lint/function_size_baseline.json` (one ceiling; Tasks 2 and 3 bump it again)
- Create (Task 1): `tldw_Server_API/tests/Usage/test_llm_quota_byok_rows.py`, `tldw_Server_API/tests/AuthNZ_Postgres/test_llm_usage_log_credential_source_pg.py`
- Modify (Task 2): `tldw_Server_API/app/core/AuthNZ/provider_credential_runtime.py`, `tldw_Server_API/app/core/Usage/llm_call_context.py`, `tldw_Server_API/app/core/Usage/llm_quota_gate.py`, `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py`, `tldw_Server_API/app/core/Chat/chat_service.py`, `tldw_Server_API/app/api/v1/endpoints/messages.py` (recorder arguments), `tldw_Server_API/app/core/Prompt_Management/prompt_improvement_dispatch.py`, `tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py`
- Create (Task 2): `tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py`
- Modify tests (Task 2): `tldw_Server_API/tests/Sharing/test_shared_workspace_chat_generation.py`, `tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py` (append), `tldw_Server_API/tests/lint/function_size_baseline.json` (two ceilings)
- Modify (Task 3): `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py`, `tldw_Server_API/app/api/v1/endpoints/chat.py`, `tldw_Server_API/app/api/v1/endpoints/messages.py`
- Modify tests (Task 3): `tldw_Server_API/tests/Usage/test_llm_quota_decision.py`, `tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py`, `tldw_Server_API/tests/Chat/integration/test_messages_endpoint.py`, `tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py` (append), `tldw_Server_API/tests/lint/function_size_baseline.json` (one ceiling)
- Docs (Task 4): create `Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md`; modify `Docs/ADR/README.md`, `Docs/Operations/Usage_Quotas.md`, `Docs/User_Guides/Server/BYOK_User_Guide.md`, `Docs/Design/2026-10-09-llm-usage-accounting-design.md`; the `Docs/Published` mirror of the first three mirrored files
- Create (Task 0): `Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md` (this plan), and the task files of TASK-<ID>, TASK-<F1> and TASK-<F2> under `backlog/tasks/`

## Test commands

Task tests (measured counts are under each task):

```bash
# Task 1
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_llm_quota_byok_rows.py "tldw_Server_API/tests/Billing/test_billing_enforcer_org_usage.py::test_get_llm_tokens_month_leaves_out_calls_on_a_members_own_key"
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/AuthNZ_Postgres/test_llm_usage_log_credential_source_pg.py
# Task 2
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py "tldw_Server_API/tests/Sharing/test_shared_workspace_chat_generation.py::test_generation_names_whose_key_its_call_runs_on" "tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py::test_a_native_call_on_the_users_own_key_is_recorded_as_such"
# Task 3
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q "tldw_Server_API/tests/Usage/test_llm_quota_decision.py::test_an_over_limit_user_whose_provider_runs_on_their_own_key_is_let_through_unmarked" "tldw_Server_API/tests/Usage/test_llm_quota_decision.py::test_users_own_key_serves_mirrors_the_first_test_of_credential_resolution" "tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py::test_chat_completions_admits_an_over_limit_user_whose_provider_runs_on_their_own_key" "tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py::test_the_hosted_plan_does_not_check_a_request_on_the_users_own_key" "tldw_Server_API/tests/Chat/integration/test_messages_endpoint.py::test_messages_admits_an_over_quota_user_whose_provider_runs_on_their_own_key" "tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py::test_a_native_call_let_through_at_the_boundary_is_still_checked_when_not_on_the_users_key"
```

**The wider set** (Task 0 baseline and Task 5): every test file that names a touched symbol, found by this command from the repository root:

```bash
git grep --untracked -l -E "insert_llm_usage_log|log_llm_usage|llm_usage_log|apply_authnz_migrations|get_authnz_migrations|sum_user_llm|enforce_llm_quota|require_llm_quota|user_to_check|check_llm_call_|_plan_call|record_direct_response|accounted_direct_stream|LLMCallContext|ProviderCallCredentials|ProviderCredentialRuntime|usage_accounting|llm_quota_gate|llm_call_context|quota_checks|usage_quota_deps|prompt_improvement_dispatch|shared_workspace_chat_service|endpoints import messages|/api/v1/messages|ensure_usage_tables_pg|MAX\(version\)|BillingEnforcer|LimitEnforcer|_get_llm_tokens_month|users_own_key_serves|billed_credential_source|missing_credential_source" -- 'tldw_Server_API/tests/*.py' | grep '/test_[^/]*\.py$'
```

The final `grep` drops the seven helper modules the pattern also matches (`conftest.py` files, `helpers/llm_quota.py` and others); they hold no tests. Add `tldw_Server_API/tests/Chat/unit/test_chat_service_*.py`, `tldw_Server_API/tests/Chat/unit/test_messages_*.py` and `tldw_Server_API/tests/Chat/integration/test_messages_*.py`. `--untracked` picks up the plan's new files before they are committed. Split the files in two (review M6): the PostgreSQL part is every file under `AuthNZ_Postgres/`, every file ending `_pg.py`, and every file ending `_postgres.py` (nine files with this plan's new one). Run the SQLite part with `TLDW_TEST_NO_DOCKER=1` and the PostgreSQL part without it. Both were measured with `-n 8 --dist loadfile` (pytest-xdist, in the venv), which keeps each file in one worker; serially the SQLite part takes about an hour, and five `test_summarization_adapter.py[*-kobold]` cases then fail by test order on base and after alike (review answer 8).

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -n 8 --dist loadfile -rf $(cat <scratchpad>/wider_sqlite.txt)
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -rf $(cat <scratchpad>/wider_pg.txt)
```

Measured on `03fec47781` (r3; the machine's load average was about 17-20):
- **SQLite part, base** (200 files): `2 failed, 5826 passed, 6 skipped` in 15:01, as on `f8f913fab8` and `4581a18e5a`.
- **SQLite part, plan applied** (202 files): `2 failed, 5883 passed, 6 skipped` in 15:44. The 57 more passes are the plan's new tests: 5 + 32 in the two new files, and 20 added to existing files.
- **The 2 failures are the same on base and after**, and the review saw both on `35e711080a`:
  - `Chat/integration/test_document_generation_endpoints.py::test_document_factory_starts_without_default_executor_and_drains_cancel`: "Timed out waiting for thread event". It passes when its file runs alone.
  - `prompt_studio/integration/test_api_endpoints.py::TestEvaluationEndpoints::test_list_evaluations`: `assert 401 == 200`. It also fails when run alone.
- **PostgreSQL part:** base (8 files) `19 passed`; plan applied (9 files) `22 passed`. `AuthNZ/integration/test_audio_quota_dates_postgres.py` passed in both. The review saw it error under load on the shared container; if it does, re-run it alone and compare with base.

Also run `tldw_Server_API/tests/Docs` (measured `212 passed`, once the new mirror file is staged) and `tldw_Server_API/tests/lint` (measured `248 passed` on `03fec47781` with the plan applied; `246` on `f8f913fab8`, and dev's #3362 added 2). Under load, `lint/test_rg_route_map_lint.py::test_shipped_route_map_is_clean` once saw its subprocess abort (`assert -6 == 0`) in folder runs with the plan applied. That happened in two of six folder runs. The other four passed, and so did five runs of the file alone. Base passed one folder run and four runs of the file alone. The test imports the app in a subprocess; the plan adds no route and changes no route map. If it fails, re-run the file alone before suspecting the plan.

**Re-measured counts, r1 to r2** (r1 on `b94318cce9`, r2 on `f8f913fab8`):

| What | r1 | r2 |
|---|---|---|
| Task 1 RED | `4 failed`; PostgreSQL `1 failed` | `6 failed` (the billing test joins the command); PostgreSQL `2 failed` |
| Task 1 GREEN | `4 passed`; `1 passed` | `6 passed`; `2 passed` |
| Task 1 regression set | `1977 passed, 1 skipped` (whole plan) | `2334 passed, 1 skipped` (Task 1 only; `tests/Billing` added) |
| Task 2 RED | `14 failed, 2 passed` | `18 failed, 5 passed` |
| Task 2 GREEN | `16 passed` | `23 passed` |
| Task 2 regression set | `660 passed, 1 skipped` (whole plan) | `1036 passed, 1 skipped` (Tasks 1-2; `test_chat_service_fallback.py` added) |
| Task 3 RED | `9 failed` | `13 failed` |
| Task 3 GREEN; touched files | `9 passed`; `60 passed` | `13 passed`; `64 passed` |
| `tests/Docs` | `212 passed` (wrongly before staging) | `1 failed, 211 passed` before staging, `212 passed` after |
| `tests/lint` | `79 passed, 1 skipped` | `246 passed` (the folder grew on dev; it includes the size ratchet) |
| Wider SQLite part | base `7 failed, 5571 passed, 6 skipped`; after `7 failed, 5600 passed, 6 skipped` (serial) | base `2 failed, 5826 passed, 6 skipped`; after `2 failed, 5868 passed, 6 skipped` (`-n 8 --dist loadfile`) |
| Wider PostgreSQL part | `10 passed`; `11 passed` | `19 passed`; `21 passed` (M6 moved three `*_postgres.py` files in) |
| Route-auth ratchet | `190 unauthenticated routes; baseline exact.` | `208 unauthenticated routes; baseline exact.` (dev added routes) |
| Mutations | `sync_layer`, `shared_exempt` survived | both caught; so are the nine new ones (B1 handle and boundary, B2 sum and log level, I1, M8, the billing sum, its fallback and its gate) |

**Re-measured counts, r2 to r3** (r2 on `f8f913fab8`, r3 on `03fec47781`; each stage built as the re-review built it: app files at base for Task 1 RED, then each task's sources, `messages.py` split as Task 2 Step 5 says):

| What | r2 | r3 |
|---|---|---|
| Task 1 RED | `6 failed`; PostgreSQL `2 failed` | `6 failed`; PostgreSQL `3 failed` (the hosted-plan test: `TypeError ... 'credential_source'`) |
| Task 1 GREEN | `6 passed`; `2 passed` | `6 passed`; `3 passed` |
| Task 1 regression set | `2334 passed, 1 skipped` | not re-run: r3 adds no test to that set (its deselected Task 3 cases go from 8 to 10) |
| Task 2 RED | `18 failed, 5 passed` | `31 failed, 5 passed` (+4 junk-key cases, +8 rule cases, +1 table test) |
| Task 2 GREEN | `23 passed` | `36 passed` |
| Task 2 regression set | `1036 passed, 1 skipped` (9 deselected) | `1049 passed, 1 skipped` (11 deselected) |
| Task 3 RED | `13 failed` | `15 failed` (+2 lookup cases) |
| Task 3 GREEN; touched files | `13 passed`; `64 passed` | `15 passed`; `66 passed` |
| `tests/Docs` | `1 failed, 211 passed` before staging, `212 passed` after | same |
| `tests/lint` | `246 passed` | `248 passed` (#3362 added 2; the plan adds none) |
| Wider SQLite part | base `2 failed, 5826 passed, 6 skipped`; after `2 failed, 5868 passed, 6 skipped` | base the same; after `2 failed, 5883 passed, 6 skipped` (same two failures) |
| Wider PostgreSQL part | `19 passed`; `21 passed` | `19 passed`; `22 passed` |
| Wider-set grep | 211 lines, 7 of them helpers | 204 test files (the grep now drops helpers); 202 SQLite and 9 PostgreSQL files after the globs |
| Shard coverage, triage `--enforce`, route-auth, OpenAPI, Bandit, the diff check, the scope grep | as stated | same (`new_uncovered=0`; OK; `208 ... exact`; both OK; no issues; nothing; nothing) |
| Mutations | `billing_pg_predicate` survived (247 passed) | `billing_pg_predicate`: 1 failed. New: `billing_aware_bound` 1, `endpoint_ignored` 8, `boundary_endpoint_ignored` 1, `unknown_is_billed` 6, `billed_rule_off` (every name billed) 8 |

---

## Task 0: Branch, file the tasks, plan file, draft PR, baseline

- [ ] **Step 1: Fetch.** #3346 has merged (it is in `f8f913fab8`). Run `git fetch origin dev --quiet`.
- [ ] **Step 2: Allocate three ids under the id lock** (review M4: the lock is held minutes, not for the whole PR).
  - Take the lock with `mkdir <scratchpad>/id_lock`. If it exists, wait; never break another session's lock.
  - Create `<scratchpad>/task_next.sh` verbatim from `Docs/superpowers/plans/2026-10-22-chacha-off-loop-and-transaction-guard.md` (the `task_next.sh` block in its Task 5 Step 4). It prints the highest backlog id on origin/dev and in every open PR.
  - Run `git fetch origin dev --quiet && zsh <scratchpad>/task_next.sh`. If it prints N, then `<ID>` is N + 1, `<F1>` is N + 2 and `<F2>` is N + 3. Skip `<F2>` if the owner has answered Decision 4. Check that no `task-<ID>`, `task-<F1>` or `task-<F2>` file exists locally.
  - Create the branch: `git switch -c fix/byok-quota-exempt-<ID> origin/dev` (write the id literally).
- [ ] **Step 3: File the task and its follow-ups** (write the ids literally; skip the third command if the owner answered Decision 4):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "BYOK calls do not count against the user's own LLM limits" \
  --id TASK-<ID> -l quotas,backend,byok \
  -d "Owner decision 2026-10-26, verbatim: 'BYOK users own keys shuold not count against their own costs'. Raised by PR #3346's final review (TASK-13936): a call made with the user's own stored provider key (tldw-level BYOK) still counts against that user's per-user LLM limits, although the operator pays nothing for it when the vendor bills the user's key. Credential resolution knows the source (ResolvedByokCredentials.source: user, team, org, server_default, none) but drops it before the usage recorder and the quota checks. Scope: record the call with its real tokens and cost and its credential_source; a user key counts as the user's own only on a provider that bills per key, sent to its vendor's own host (a key stored for a custom OpenAI-compatible slot, a local server or an unknown provider, or for a billed provider whose base URL the operator points at its own model server or a gateway, is user_unmetered and counts); leave credential_source 'user' out of limits.llm_tokens_per_month, limits.llm_cost_per_day_usd, limits.llm_cost_per_month_usd and the hosted billing plan's llm_tokens_month, and make that plan sum count on PostgreSQL at all (it bound a timezone-aware month start against a TIMESTAMP column and returned 0); do not check such a call at the call; at /chat/completions and the Messages API (/messages, /v1/messages), let an over-limit user through when the provider resolves to their own key, and skip the hosted plan's pre-check for such a request. Out of scope, still refusing an over-limit own-key user before the key is known (follow-up TASK-<F1>): character chat, RAG generation, evaluations and benchmarks, flashcards and quizzes, chat document generation, watchlists, the media endpoints and a media ingest job's start. Org and team shared keys still count (owner question). The Resource Governor token caps, virtual-key budgets and evaluation token limits still count own-key calls (owner question). Out of scope: OpenRouter's is_byok, which #3346 prices. Plan: Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md." \
  --ac "A call made with the user's own stored key for a provider that bills per key is recorded in llm_usage_log with its real tokens and cost and credential_source 'user', on SQLite and on PostgreSQL, with a test on each; a key stored for a custom OpenAI-compatible slot, a local server or an unknown provider, or for a billed provider whose configured base URL is not on its vendor's host, is recorded as user_unmetered, with tests" \
  --ac "limits.llm_tokens_per_month, limits.llm_cost_per_day_usd, limits.llm_cost_per_month_usd and the hosted billing plan's llm_tokens_month do not count rows with credential_source 'user', while rows with user_unmetered, team, org, server_default or no source still count, with tests on SQLite and on PostgreSQL; on PostgreSQL the hosted plan's sum counts at all (it returned 0 before), with a test" \
  --ac "For a user over an LLM limit, a call on their own key is not refused at the call (the four adapter methods, perform_chat_api_call and perform_chat_api_call_async, a caller that marks the call context), and a call on a server, team, org or operator-endpoint key still is, with tests; a refused failover call does not mark the fallback provider unhealthy, with a test" \
  --ac "Only /chat/completions and the Messages API (/messages, /v1/messages) admit an over-limit user whose provider resolves to their own key, without marking the request admitted, and still refuse one without their own key for that provider; /chat/completions also skips the hosted billing plan's pre-check for such a request; with tests. Every other LLM quota boundary is unchanged and still refuses (TASK-<F1>)" \
  --ac "A database without the new column still records the row, without its source, and the limits still count every row and refuse, with a test; a PostgreSQL usage-table statement that fails with one of the bootstrap's handled errors (a timeout, a lost connection) is logged as a warning, with a test; a database error such as a missing privilege still stops startup" \
  --ac "Usage_Quotas.md, the BYOK user guide and a new ADR amending ADR-064 say what counts, and the Docs/Published mirror is refreshed"
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Own-key LLM calls: the router's usage row and the other quota boundaries" \
  --id TASK-<F1> -l quotas,backend,byok \
  -d "Follow-up of TASK-<ID>, which made a call on the user's own provider key (on a provider that bills per key) exempt from the per-user LLM limits. Two gaps remain. (1) Boundaries: only /chat/completions and the Messages API pass their provider to enforce_llm_quota. These still refuse an over-limit user before the key is known, even when every call would run on the user's own key: character chat (character_chat_sessions.py, the completion check and the roleplay-resume check; the completion's provider is resolved only after the auto-router, so the check moves below it), RAG generation (rag_unified.py, six sites), evaluations and benchmarks (evaluations_unified.py, evaluations_crud.py, evaluations_benchmarks.py, benchmark_api.py), flashcards and quizzes (route dependencies, to become inline calls), chat document generation (chat_documents.py), watchlists (watchlists.py), the media endpoints with an api_name (media/add.py, ingest_web_content.py and the seven process_* modules), and a media ingest job's start (media_ingest_jobs_worker._admit_or_refuse_analysis). Pass the provider where the endpoint knows it before any LLM work. Character chat first: it is a primary UI surface. If the owner wants evaluation tokens exempt too (TASK-<ID> Decision 5), the evaluation routes also skip the limits.evaluation_tokens_per_day reservation when users_own_key_serves answers True. (2) Router calls: routing/accounting.log_model_router_usage records the auto-router's call without a credential_source, so it counts (at most 64 output tokens a call). Its callers are chat.py's _log_router_usage (the router handle is router_credential_handle), chat_route_resolver.py's log_router_usage (its execute_router_call resolves with resolve_byok_credentials and passes a raw key, so its llm_call_context(records_itself=True) should also carry credential_source=billed_credential_source(provider, byok_resolution.source)) and character_chat_sessions.py's _log_router_usage." \
  --ac "Each LLM quota boundary that knows its provider before any LLM work passes it to enforce_llm_quota, with a test per endpoint family, character chat first" \
  --ac "A media ingest job's start check admits analysis for an over-limit user whose analysis provider resolves to their own key, without marking the job admitted, with a test" \
  --ac "The auto-router's usage row on /chat/completions, character chat and chat_route_resolver records the router call's credential_source, with a test"
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Owner question: should org and team shared keys count against a member's LLM limits?" \
  --id TASK-<F2> -l quotas,byok,question \
  -d "From TASK-<ID> (ADR-083, Decision 4): calls on org or team shared provider keys still count against each member's limits.llm_* values and the hosted plan's llm_tokens_month, because the owner's decision named the user's own keys. The org or team, not the operator, pays for those calls. Raising the org's or team's limits instead also raises its members' server-key allowance. If the owner wants them exempt: in Usage/llm_call_context.py replace USER_KEY_SOURCE with OWN_KEY_SOURCES = frozenset({'user', 'team', 'org'}), test membership in llm_quota_gate.user_to_check, and apply billed_credential_source's rule to team and org too; in AuthNZ/repos/usage_repo.py change _OPERATOR_PAID_PREDICATE, and in Billing/enforcement.py _OWN_KEY_EXCLUDED, to NOT IN ('user', 'team', 'org'); extend usage_quota_deps.users_own_key_serves to the team and org lookups. No row needs re-recording: credential_source stores the source." \
  --ac "The owner's answer is recorded, and if it is 'exempt', shared-key calls are neither counted nor checked, with tests"
```

- [ ] **Step 4:** Mark the task started: `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-<ID> -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md (owner-approved). Owner answers: Decision 3 (scope) <answer>; Decision 4 (shared keys) <answer>; Decision 5 (other spend limits) <answer>. Follow-ups filed: TASK-<F1>, TASK-<F2>."`
- [ ] **Step 5: Check the numbers.**
  - `grep -n "Migration(10[3-9]\|Migration(11" tldw_Server_API/app/core/AuthNZ/migrations.py | tail -2` must show 103 as the last migration.
  - `ls Docs/ADR | sort | tail -3` must show 081 (or 082, if #3323 has merged) as the last ADR.
  - Open PRs (review I3). `gh pr list --repo rmusser01/tldw_server --state open --json number,files,changedFiles --limit 200 --jq '.[] | .number as $n | .files[].path | select(test("Docs/ADR/0[0-9]+-|AuthNZ/migrations.py")) | "\($n) \(.)"'` lists the ADR files and migration edits in open PRs. The `files` list stops at 100 files a PR. So for each PR that `--jq '.[] | select(.changedFiles > 99) | .number'` prints, run `gh api repos/rmusser01/tldw_server/pulls/<n>/files --paginate --jq '.[].filename | select(test("ADR/0[89][0-9]-|AuthNZ/migrations.py"))'`. Measured 2026-10-29 (r3): #3323 holds ADR-082; twelve PRs have over 100 files (#2738, #2821, #2855, #2869, #3019, #3020, #3071, #3159, #3226, #3269, #3361, #3369), and none of them has an ADR from 080 up or touches `migrations.py`. #3187 touches `migrations.py` with migration 100 (stale).
  - If 083 or 104 is taken, renumber (Global Constraints).
- [ ] **Step 6: Commit the plan and the tasks, open a draft PR, release the lock.**
  - Copy the approved plan to `Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md` unchanged. It keeps `<ID>`; the task files, the code, the docs and the commit messages carry the real ids.
  - `git add Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md && git add -A backlog/tasks && git diff --cached --check` (it must print nothing), and commit (message 0).
  - `git push -u origin fix/byok-quota-exempt-<ID>`, then `gh pr create --draft --base dev --title "fix(usage): BYOK calls do not count against the user's own LLM limits (TASK-<ID>)" --body "Draft: plan and tasks only. The body is written in Task 5."`.
  - When `gh pr view <pr> --json files -q '.files[].path'` lists the new task files, `rmdir <scratchpad>/id_lock`.
- [ ] **Step 7: Baseline.** Run the wider set (both parts, see "Test commands") on the branch as it is after commit 0, and record both counts. Compare every later failure with this baseline. Never dismiss one as environment-only without it.

## Task 1: Record whose provider key paid for each call (AC #1, #2, #5)

**Files:** see "Files" (Task 1).

**Interfaces:**
- Produces:
  - `llm_usage_log.credential_source TEXT` (nullable) on both backends.
  - `AuthnzUsageRepo.insert_llm_usage_log(..., credential_source: str | None = None)`.
  - `log_llm_usage(..., credential_source: str | None = None)`.
  - `_OPERATOR_PAID_PREDICATE = "COALESCE(credential_source, '') <> 'user'"` in `usage_repo.py`, used by `_sum_llm_column_since`.
  - `missing_credential_source(exc, where) -> bool` in `usage_repo.py`: True, with a warning once per `where`, when an error names the column.
  - `_OWN_KEY_EXCLUDED` and `BillingEnforcer._sum_org_llm_tokens(pool, org_id, month_start, own_key)` in `Billing/enforcement.py`.
- Consumed by Task 2: `log_llm_usage(credential_source=...)`.

**RED reason (measured on `03fec47781`):** the SQLite command `6 failed`; the PostgreSQL file `3 failed`.
- `test_the_quota_counters_...` fails with `sqlite3.OperationalError: table llm_usage_log has no column named credential_source`.
- The two `log_llm_usage` tests fail:
  - `test_a_call_on_the_users_own_key_...` with `TypeError: log_llm_usage() got an unexpected keyword argument 'credential_source'`;
  - `test_a_table_without_the_column_...` with `sqlite3.OperationalError: no such column: "credential_source"`, at its `DROP COLUMN`.
- `test_a_failed_postgres_usage_column_is_logged_as_a_warning` fails with `assert False`: the failure is logged at debug.
- The migration test fails with `AssertionError: assert 'credential_source' in {...}`.
- The billing test fails with `assert 1345 == 345`: the org's sum counts the own-key row.
- The PostgreSQL tests fail with the `AssertionError` on the column set, with `ImportError: cannot import name 'missing_credential_source'`, and (the hosted-plan test) with `TypeError: AuthnzUsageRepo.insert_llm_usage_log() got an unexpected keyword argument 'credential_source'`.

- [ ] **Step 1: Create `tldw_Server_API/tests/Usage/test_llm_quota_byok_rows.py`:**

```python
"""Calls paid by the user's own provider key are recorded but not counted by the LLM quotas (TASK-<ID>)."""

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import apply_authnz_migrations
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.app.core.Usage.usage_tracker import log_llm_usage

pytestmark = pytest.mark.unit


async def _fresh_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A single-user SQLite AuthNZ pool, migrated to the latest schema."""
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.sqlite'}")
    monkeypatch.setenv("LLM_USAGE_ENABLED", "1")

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()
    return await get_db_pool()


async def test_the_quota_counters_leave_out_calls_paid_by_the_users_own_key(monkeypatch, tmp_path) -> None:
    """Team, org, server, operator-endpoint and unknown-source rows count; the user's own key's row does not."""
    pool = await _fresh_pool(monkeypatch, tmp_path)
    rows = (
        ("user", 1000, 9.0), ("team", 10, 0.1), ("org", 20, 0.2), ("server_default", 30, 0.3),
        ("user_unmetered", 40, 0.4), (None, 5, 0.05),
    )
    for source, tokens, cost in rows:
        await pool.execute(
            "INSERT INTO llm_usage_log (user_id, operation, total_tokens, total_cost_usd, credential_source) "
            "VALUES (?, ?, ?, ?, ?)",
            1, "chat", tokens, cost, source,
        )

    since = datetime.now(timezone.utc) - timedelta(days=1)
    repo = AuthnzUsageRepo(pool)
    assert await repo.sum_user_llm_tokens_since(user_id=1, since=since) == 105.0
    assert await repo.sum_user_llm_cost_since(user_id=1, since=since) == pytest.approx(1.05)


async def test_a_call_on_the_users_own_key_is_recorded_with_its_real_tokens_and_cost(monkeypatch, tmp_path) -> None:
    """The row keeps its tokens and its catalog cost, and says whose key paid."""
    pool = await _fresh_pool(monkeypatch, tmp_path)
    await log_llm_usage(
        user_id=1, key_id=None, endpoint="internal", operation="chat", provider="openai", model="gpt-4o",
        status=200, latency_ms=1, prompt_tokens=1000, completion_tokens=500, request_id="byok-row",
        credential_source="user",
    )
    row = await pool.fetchone(
        "SELECT total_tokens, total_cost_usd, credential_source FROM llm_usage_log WHERE request_id = ?",
        "byok-row",
    )
    assert (row["total_tokens"], row["credential_source"]) == (1500, "user")
    assert row["total_cost_usd"] > 0


async def test_a_table_without_the_column_still_records_counts_and_refuses(monkeypatch, tmp_path) -> None:
    """A schema from before the column keeps the row without its source, counts every row, and still refuses."""
    from tldw_Server_API.app.core.Usage import quota_checks

    pool = await _fresh_pool(monkeypatch, tmp_path)
    await pool.execute("ALTER TABLE llm_usage_log DROP COLUMN credential_source")
    await log_llm_usage(
        user_id=1, key_id=None, endpoint="internal", operation="chat", provider="openai", model="gpt-4o",
        status=200, latency_ms=1, prompt_tokens=1000, completion_tokens=500, request_id="old-schema",
        credential_source="user", usage_metadata={"prompt_tokens": 1000, "completion_tokens": 500},
    )
    row = await pool.fetchone(
        "SELECT total_tokens, estimate_source FROM llm_usage_log WHERE request_id = ?", "old-schema"
    )
    assert (row["total_tokens"], row["estimate_source"]) == (1500, "provider_usage")

    since = datetime.now(timezone.utc) - timedelta(days=1)
    assert await AuthnzUsageRepo(pool).sum_user_llm_tokens_since(user_id=1, since=since) == 1500.0

    async def _limit(_user_id: object, key: str) -> int | None:
        """A 100-token monthly limit and nothing else."""
        return 100 if key == "limits.llm_tokens_per_month" else None

    monkeypatch.setattr(quota_checks, "user_quota", _limit)
    refusal = await quota_checks.llm_quota_refusal(1)
    assert refusal is not None and refusal.category == "llm_tokens_month"


async def test_a_failed_postgres_usage_column_is_logged_as_a_warning() -> None:
    """A usage-table statement that times out on PostgreSQL is a warning, so a missing column is visible."""
    from loguru import logger

    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_usage_tables_pg

    class _Pool:
        """A PostgreSQL pool whose ALTER of llm_usage_log outlives the pool's command_timeout (a lock wait)."""

        pool = object()

        async def execute(self, sql: str, *_params: object) -> None:
            """Time out every ALTER of llm_usage_log, as asyncpg does."""
            if "ALTER TABLE llm_usage_log" in sql:
                raise TimeoutError

    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        assert await ensure_usage_tables_pg(_Pool())
    finally:
        logger.remove(sink)
    assert any("PG ensure usage tables DDL failed" in line for line in warnings)


def _columns(db_path: Path) -> set[str]:
    """The columns of llm_usage_log."""
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(llm_usage_log)").fetchall()}


def test_sqlite_migrations_add_the_column_to_new_and_upgraded_databases(tmp_path: Path) -> None:
    """A fresh database has the column, and so does one upgraded from before it, migrated twice."""
    fresh = tmp_path / "fresh.db"
    apply_authnz_migrations(fresh)
    assert "credential_source" in _columns(fresh)

    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as conn:
        conn.execute("CREATE TABLE llm_usage_log (id INTEGER PRIMARY KEY, ts TIMESTAMP, user_id INTEGER)")
        conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TIMESTAMP NOT NULL)"
        )
        conn.execute("INSERT INTO schema_migrations VALUES (103, 'before credential_source', CURRENT_TIMESTAMP)")
    apply_authnz_migrations(legacy)
    apply_authnz_migrations(legacy)
    assert "credential_source" in _columns(legacy)
```

  Notes:
  - `get_db_pool()` on SQLite runs every AuthNZ migration, so `_fresh_pool` gives the real latest schema.
  - The third test drops the column to stand for an unmigrated database (SQLite 3.35 or later; the venv has 3.50.4). It is the review's probe A: on the r1 plan the sum raised and `llm_quota_refusal` admitted the user ("treating as unlimited").
  - The fourth test stands in for an `ALTER TABLE llm_usage_log` that waits on a lock past the pool's `command_timeout` (60 s, `AuthNZ/database.py:901`). asyncpg then raises `TimeoutError`, one of the bootstrap's handled errors. The statement exists on base too, so the RED reason is the log level alone. A database error such as a missing privilege (`InsufficientPrivilegeError`) or PostgreSQL's own lock timeout (`LockNotAvailableError`) is not a handled error: it leaves `ensure_usage_tables_pg` and stops startup, as it does today (review m1). The plan does not change that.
  - The 088 columns stay, so the asserted `estimate_source` proves the fallback keeps the current statement, not the pre-088 one.

- [ ] **Step 2: Create `tldw_Server_API/tests/AuthNZ_Postgres/test_llm_usage_log_credential_source_pg.py`:**

```python
"""PostgreSQL: llm_usage_log.credential_source exists, is written, and the quota counters skip ``user`` rows (TASK-<ID>)."""

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_usage_tables_pg
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user


@pytest.mark.asyncio
async def test_credential_source_is_stored_and_user_rows_are_not_counted_pg(test_db_pool) -> None:
    """The column is added idempotently; a ``user`` row is stored but not summed."""
    assert await ensure_usage_tables_pg(test_db_pool)
    assert await ensure_usage_tables_pg(test_db_pool)
    columns = await test_db_pool.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'llm_usage_log'"
    )
    assert "credential_source" in {str(row["column_name"]) for row in columns}

    user_id = await ensure_test_user(test_db_pool, "byok-quota-pg")
    repo = AuthnzUsageRepo(test_db_pool)
    for source, tokens, cost in (("user", 1000, 9.0), ("server_default", 30, 0.3), (None, 5, 0.05)):
        await repo.insert_llm_usage_log(
            user_id=user_id, key_id=None, endpoint="internal", operation="chat", provider="openai",
            model="gpt-4o", status=200, latency_ms=1, prompt_tokens=tokens, completion_tokens=0,
            total_tokens=tokens, prompt_cost_usd=cost, completion_cost_usd=0.0, total_cost_usd=cost,
            credential_source=source,
        )

    stored = await test_db_pool.fetch(
        "SELECT credential_source FROM llm_usage_log WHERE user_id = $1 ORDER BY id", user_id
    )
    assert [row["credential_source"] for row in stored] == ["user", "server_default", None]
    since = datetime.now(timezone.utc) - timedelta(days=1)
    assert await repo.sum_user_llm_tokens_since(user_id=user_id, since=since) == 35.0
    assert await repo.sum_user_llm_cost_since(user_id=user_id, since=since) == pytest.approx(0.35)


@pytest.mark.asyncio
async def test_postgres_names_the_missing_column_so_the_counters_fall_back_pg(test_db_pool) -> None:
    """PostgreSQL's undefined-column error names ``credential_source``, which is how the counters recognise it."""
    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import missing_credential_source

    assert await ensure_usage_tables_pg(test_db_pool)
    with pytest.raises(Exception) as missing:  # noqa: PT011 - whatever the driver raises
        await test_db_pool.fetchval("SELECT COUNT(credential_source) FROM usage_log")  # usage_log has no such column
    assert missing_credential_source(missing.value, "a PostgreSQL test")


@pytest.mark.asyncio
async def test_the_hosted_plans_llm_tokens_leave_out_a_members_own_key_pg(test_db_pool, monkeypatch) -> None:
    """The org's monthly LLM tokens count every source but the member's own key on PostgreSQL (TASK-<ID>)."""
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
    from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer

    assert await ensure_usage_tables_pg(test_db_pool)
    user_id = await ensure_test_user(test_db_pool, "byok-plan-pg")
    orgs = AuthnzOrgsTeamsRepo(test_db_pool)
    org_id = int((await orgs.create_organization(name="byok-plan-pg", owner_user_id=user_id))["id"])
    await orgs.add_org_member(org_id=org_id, user_id=user_id)
    repo = AuthnzUsageRepo(test_db_pool)
    for source, tokens in (("user", 1000), ("user_unmetered", 40), ("server_default", 300), (None, 5)):
        await repo.insert_llm_usage_log(
            user_id=user_id, key_id=None, endpoint="internal", operation="chat", provider="openai",
            model="gpt-4o", status=200, latency_ms=1, prompt_tokens=tokens, completion_tokens=0,
            total_tokens=tokens, prompt_cost_usd=0.0, completion_cost_usd=0.0, total_cost_usd=0.0,
            credential_source=source,
        )

    async def _pool():
        return test_db_pool

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.get_db_pool", _pool)
    assert await BillingEnforcer()._get_llm_tokens_month(org_id=org_id) == 345
```

  The second test is read-only: `usage_log` has no `credential_source` column, so the query raises PostgreSQL's own undefined-column error, and the test proves that error names the column. Dropping the column from `llm_usage_log` instead would break other sessions on the shared container.

  The third test is the hosted plan's sum on PostgreSQL (review I2). It writes real rows: `test_db_pool`'s `clean_database` fixture truncates `organizations`, `org_members`, `llm_usage_log` and `users` before and after each test. With the rest of Task 1 in place and only the bound left as it is today, it fails with `assert 0 == 345` (measured on the r2 prototype): `_get_llm_tokens_month` binds a timezone-aware month start against `llm_usage_log.ts`, a `TIMESTAMP` column, so asyncpg raises `DataError` and the method returns 0. It kills the review's `billing_pg_predicate` mutation (the `{own_key}` predicate removed from the PostgreSQL branch: `1 failed`, the sum is 1345).

- [ ] **Step 3: Append to `tldw_Server_API/tests/Billing/test_billing_enforcer_org_usage.py`**, before `class _FakeOrgRepo:`:

```python
@pytest.mark.asyncio
async def test_get_llm_tokens_month_leaves_out_calls_on_a_members_own_key(monkeypatch):
    """A member's own-key row is not plan usage; every other source is (TASK-<ID>, ADR-083)."""

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE org_members (user_id INTEGER NOT NULL, org_id INTEGER NOT NULL, added_at TEXT NOT NULL)")
    conn.execute("CREATE TABLE api_keys (id INTEGER PRIMARY KEY, org_id INTEGER NOT NULL)")
    conn.execute(
        "CREATE TABLE llm_usage_log (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, key_id INTEGER, "
        "total_tokens INTEGER NOT NULL, ts TEXT NOT NULL, credential_source TEXT)"
    )
    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("INSERT INTO org_members (user_id, org_id, added_at) VALUES (101, 7, '2026-01-01 00:00:00')")
    for source, tokens in (("user", 1000), ("user_unmetered", 40), ("server_default", 300), (None, 5)):
        conn.execute(
            "INSERT INTO llm_usage_log (user_id, total_tokens, ts, credential_source) VALUES (101, ?, ?, ?)",
            (tokens, now_ts, source),
        )
    conn.commit()
    fake_pool = _FakePoolSqlite(_LiveSqliteConn(conn))

    async def _fake_get_db_pool():
        return fake_pool

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.get_db_pool", _fake_get_db_pool, raising=False)
    try:
        assert await BillingEnforcer()._get_llm_tokens_month(org_id=7) == 345
    finally:
        conn.close()
```

  The file's existing `test_get_llm_tokens_month_counts_row_with_user_and_key_once` builds `llm_usage_log` without the column, so it now pins the fallback: it keeps passing only because the sum retries without the predicate.

- [ ] **Step 4: Run; see RED.** The two Task 1 commands under "Test commands". Expected: `6 failed` and `3 failed`, for the reasons above.
- [ ] **Step 5: Implement.**

`tldw_Server_API/app/core/AuthNZ/migrations.py`, after `migration_101_add_admin_perf_indexes` (it ends with `logger.info("Migration 101: Added admin perf indexes ...")`) and before `def apply_authnz_migrations(`:

```python
def migration_104_add_llm_usage_log_credential_source(conn: sqlite3.Connection) -> None:
    """Add ``llm_usage_log.credential_source``: whose key paid for the call (TASK-<ID>).

    ``user`` marks a call made with the user's own provider key; the LLM quota
    counters leave those rows out. Fixtures without the table skip (cf. 099).
    """
    if not _sqlite_table_exists(conn, "llm_usage_log"):
        logger.info("Migration 104: llm_usage_log table not present; skipping")
        return
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(llm_usage_log)").fetchall()}
    if "credential_source" not in columns:
        conn.execute("ALTER TABLE llm_usage_log ADD COLUMN credential_source TEXT")
    logger.info("Migration 104: Added llm_usage_log.credential_source")
```

  In `get_authnz_migrations()`, after the `Migration(103, ...)` line:

```python
        Migration(104, "Add llm_usage_log.credential_source", migration_104_add_llm_usage_log_credential_source),
```

`tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py` (`_CREATE_USAGE_TABLES`):
- In the `CREATE TABLE IF NOT EXISTS llm_usage_log` statement, change the last column line `raw_usage_metadata_json TEXT` to `raw_usage_metadata_json TEXT,` and add `credential_source TEXT` after it.
- After `("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS raw_usage_metadata_json TEXT", ()),` add:

```python
    ("ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS credential_source TEXT", ()),
```

`tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`, in `ensure_usage_tables_pg`, the loop's handler becomes a warning (review B2: a failed `ADD COLUMN` used to leave every LLM limit counting nothing, silently):

```python
            except _PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                # Every statement is idempotent, so a failure is a real schema gap (a missing
                # llm_usage_log.credential_source makes the LLM limits count own-key calls).
                logger.warning(f"PG ensure usage tables DDL failed: {exc}")
```

`tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py`:
- After `_LLM_GENERATION_PREDICATE = "COALESCE(operation, '') <> 'embeddings'"` (line 15), add (two blank lines before `_SQLITE_CORRUPTION_SIGNATURES`):

```python
# Nor do they count calls paid by the user's own provider key (TASK-<ID>). NULL (unknown, and every
# row written before the column existed) counts.
_OPERATOR_PAID_PREDICATE = "COALESCE(credential_source, '') <> 'user'"
_MISSING_SOURCE_WARNED: set[str] = set()


def missing_credential_source(exc: BaseException, where: str) -> bool:
    """True when ``exc`` names ``llm_usage_log.credential_source``: the schema predates the column.

    The caller then retries without the column, so every row counts. The first
    time per ``where`` it warns, because the LLM limits then count calls on the
    user's own key.
    """
    if "credential_source" not in str(exc):
        return False
    if where not in _MISSING_SOURCE_WARNED:
        _MISSING_SOURCE_WARNED.add(where)
        logger.bind(exception_type=type(exc).__name__).warning(
            "llm_usage_log has no credential_source column, so calls on users' own keys count ({}); "
            "apply AuthNZ migration 104, or check the PostgreSQL usage-table bootstrap",
            where,
        )
    return True
```

- Replace `_sum_llm_column_since` with:

```python
    async def _sum_llm_column_since(self, column: str, *, user_id: int, since: datetime) -> float:
        """Sum one ``llm_usage_log`` column for a user from ``since`` (inclusive), embeddings excluded.

        Calls made with the user's own provider key are excluded too, except on
        a schema without ``credential_source``, where every row counts.
        ``since`` is bound naive on Postgres: ``llm_usage_log.ts`` is TIMESTAMP
        without time zone there, so an aware bound would never match.
        ``column`` is a constant chosen by the callers below, never user input.
        """
        bound: Any = _strip_tzinfo(since) if self._is_postgres_backend() else since.strftime("%Y-%m-%d %H:%M:%S")
        query = (
            f"SELECT COALESCE(SUM({column}), 0) FROM llm_usage_log "  # nosec B608
            f"WHERE user_id = ? AND ts >= ? AND {_LLM_GENERATION_PREDICATE}"
        )
        try:
            value = await self.db_pool.fetchval(f"{query} AND {_OPERATOR_PAID_PREDICATE}", int(user_id), bound)
        except Exception as exc:
            if not missing_credential_source(exc, "the per-user LLM limits"):
                raise
            value = await self.db_pool.fetchval(query, int(user_id), bound)  # never fail open: count every row
        return float(value or 0)
```

- In `insert_llm_usage_log`:
  - Add the parameter `credential_source: str | None = None,` after `raw_usage_metadata_json: str | None = None,`.
  - Extend the docstring's last sentence to: `do not embed backend-specific SQL. A row with a ``credential_source`` on a schema without that column is written without it.`
  - Replace the first statement (the one that lists `raw_usage_metadata_json`, before `# Backward-compatible fallback for pre-088 schemas.`) with a local function and a two-step call. Put this before the outer `try:`:

```python

        async def _insert_full(with_source: bool) -> None:
            """The current statement, with or without ``credential_source``."""
            source_column = ", credential_source" if with_source else ""
            source_value = ", ?" if with_source else ""
            source_args = (credential_source,) if with_source else ()
            await self.db_pool.execute(
                f"""
                INSERT INTO llm_usage_log (
                    ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
                    prompt_tokens, completion_tokens, total_tokens,
                    prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated, request_id,
                    remote_ip, user_agent, token_name, conversation_id,
                    cached_input_tokens, cache_write_input_tokens, cache_read_input_tokens,
                    billable_input_tokens, reasoning_tokens, choice_count, estimate_source,
                    prompt_fingerprint, prompt_fingerprint_version, world_book_fingerprint,
                    raw_usage_metadata_json{source_column}
                ) VALUES (
                    CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?{source_value}
                )
                """,  # nosec B608 - the interpolated fragments are constants
                user_id,
                key_id,
                endpoint,
                operation,
                provider,
                model,
                int(status),
                int(latency_ms),
                int(prompt_tokens),
                int(completion_tokens),
                int(total_tokens),
                float(prompt_cost_usd),
                float(completion_cost_usd),
                float(total_cost_usd),
                currency,
                bool(estimated),
                request_id,
                remote_ip,
                user_agent,
                token_name,
                conversation_id,
                int(cached_input_tokens) if cached_input_tokens is not None else None,
                int(cache_write_input_tokens) if cache_write_input_tokens is not None else None,
                int(cache_read_input_tokens) if cache_read_input_tokens is not None else None,
                int(billable_input_tokens) if billable_input_tokens is not None else None,
                int(reasoning_tokens) if reasoning_tokens is not None else None,
                int(choice_count) if choice_count is not None else None,
                estimate_source,
                prompt_fingerprint,
                prompt_fingerprint_version,
                world_book_fingerprint,
                raw_usage_metadata_json,
                *source_args,
            )

```

  Then the outer block starts like this:

```python
        try:
            try:
                try:
                    await _insert_full(credential_source is not None)
                except Exception as exc:
                    if credential_source is None:
                        raise
                    missing_credential_source(exc, "usage rows")  # warns once when the column is missing
                    await _insert_full(False)  # keep the row, without its source: it counts
            except Exception:
                try:
                    # Backward-compatible fallback for pre-088 schemas.
```

  The pre-088 and pre-054 fallbacks and the final `except Exception as exc:` are unchanged: same text, same indentation (review M2). The new `try` adds one level only around `_insert_full`. A row with no source takes exactly today's statement, so hand-made test tables without the column keep working. A row with a source is retried without it on any failure, so the row is never lost; `missing_credential_source` only adds the warning when the column is missing (review M5). The `nosec` sits on the string's closing line, which is where Bandit reports a multi-line string (measured: no finding).

`tldw_Server_API/app/core/Usage/usage_tracker.py`, in `log_llm_usage`:
- Add the parameter `credential_source: str | None = None,` after `world_book_fingerprint: str | None = None,`.
- Replace the docstring's last paragraph with:

```python
    ``credential_source`` is whose provider key the call used (``user``,
    ``team``, ``org``, ``server_default``), or None when unknown; the LLM quota
    counters leave ``user`` rows out. This function is best-effort and should
    never raise; errors are logged.
```

- In the `repo.insert_llm_usage_log(...)` call, after `raw_usage_metadata_json=raw_usage_metadata_json,` add `credential_source=credential_source,`.

The daily-ledger shadow write below it is unchanged: RG still counts own-key tokens (Decision 5).

`tldw_Server_API/app/core/Billing/enforcement.py` (Decision 5: the hosted plan's tokens leave own-key calls out):
- Before `_BILLING_ENFORCEMENT_FAILURE_MODE_ENV = "BILLING_ENFORCEMENT_FAILURE_MODE"`, add:

```python
# The plan's LLM tokens leave out calls on a member's own provider key (ADR-083, TASK-<ID>).
_OWN_KEY_EXCLUDED = "AND COALESCE(l.credential_source, '') <> 'user'"
```

- Replace `_get_llm_tokens_month` with the two methods below. The SQL is today's, with `{own_key}` added after the `ts` bound; only its indentation changes. The month start is now naive UTC (review I2). `_sum_llm_column_since` binds `_strip_tzinfo(since)` on PostgreSQL for the same reason; here the bound is built in UTC, so `tzinfo=None` is that conversion, and the SQLite branch's `strftime` string is unchanged.

```python
    async def _get_llm_tokens_month(self, org_id: int) -> int:
        """Get LLM token usage for current month for an organization.

        This aggregates usage from both user-scoped and API-key scoped calls.
        Each llm_usage_log row is counted at most once per org.

        User-scoped calls are attributed to a user's primary org (earliest
        org_members.added_at, tie-broken by lowest org_id) to avoid
        double-counting when a user belongs to multiple orgs.

        Calls on a member's own provider key are left out (ADR-083), except on
        a schema without ``credential_source``, where every row counts.
        """
        try:
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
            from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import missing_credential_source

            pool = await get_db_pool()
            now = datetime.now(timezone.utc)
            # Naive UTC: on PostgreSQL llm_usage_log.ts is TIMESTAMP without time zone, and asyncpg
            # refuses an aware bound for it, so the plan's tokens counted nothing there (TASK-<ID>).
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
            try:
                return await self._sum_org_llm_tokens(pool, org_id, month_start, _OWN_KEY_EXCLUDED)
            except Exception as exc:
                if not missing_credential_source(exc, "the billing plan's llm_tokens_month"):
                    raise
                return await self._sum_org_llm_tokens(pool, org_id, month_start, "")  # every row counts
        except _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS as exc:
            self._log_source_failure("debug", "Failed to get LLM tokens", org_id, exc)
            if self._fail_closed_on_data_error():
                raise
            return 0

    async def _sum_org_llm_tokens(self, pool: Any, org_id: int, month_start: datetime, own_key: str) -> int:
        """Sum the org's ``llm_usage_log`` tokens since ``month_start``; ``own_key`` is "" or ``_OWN_KEY_EXCLUDED``."""
        async with pool.acquire() as conn:
            if self._is_postgres_pool(pool):
                # PostgreSQL: single-pass attribution, no UNION path.
                result = await conn.fetchval(
                    f"""
                    WITH primary_org AS (
                        SELECT user_id, org_id
                        FROM (
                            SELECT user_id, org_id,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY user_id
                                       ORDER BY added_at ASC, org_id ASC
                                   ) AS rn
                            FROM org_members
                        ) ranked
                        WHERE rn = 1
                    )
                    SELECT COALESCE(SUM(l.total_tokens), 0)
                    FROM llm_usage_log AS l
                    LEFT JOIN primary_org AS po ON l.user_id = po.user_id
                    LEFT JOIN api_keys AS ak ON l.key_id = ak.id
                    WHERE l.ts >= $2 {own_key}
                      AND (
                            po.org_id = $1
                            OR ak.org_id = $1
                          )
                    """,  # nosec B608
                    org_id, month_start,
                )
            else:
                # SQLite: single-pass attribution, no UNION path.
                month_start_str = month_start.strftime("%Y-%m-%d %H:%M:%S")
                cur = await conn.execute(
                    f"""
                    WITH primary_org AS (
                        SELECT om.user_id, MIN(om.added_at) AS min_added
                        FROM org_members om
                        GROUP BY om.user_id
                    ),
                    primary_org_resolved AS (
                        SELECT om.user_id, MIN(om.org_id) AS org_id
                        FROM org_members om
                        JOIN primary_org po
                          ON po.user_id = om.user_id AND po.min_added = om.added_at
                        GROUP BY om.user_id
                    )
                    SELECT COALESCE(SUM(l.total_tokens), 0)
                    FROM llm_usage_log AS l
                    LEFT JOIN primary_org_resolved po ON l.user_id = po.user_id
                    LEFT JOIN api_keys AS ak ON l.key_id = ak.id
                    WHERE l.ts >= ? {own_key}
                      AND (
                            po.org_id = ?
                            OR ak.org_id = ?
                          )
                    """,  # nosec B608
                    (month_start_str, org_id, org_id),
                )
                row = await cur.fetchone()
                result = row[0] if row else 0
            return int(result or 0)
```

- [ ] **Step 6: Update the pinned migration version.** In `tldw_Server_API/tests/Storage/test_native_chat_quota_reservations.py:753`, change `== 103` in `SELECT MAX(version) AS version FROM schema_migrations` to `== 104`. That test asserts the migrated database is at the latest version. Its other assertions are unchanged.
- [ ] **Step 7: Run; see GREEN.**
  - The two commands: `6 passed` and `3 passed` (measured).
  - Then `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage tldw_Server_API/tests/AuthNZ/unit tldw_Server_API/tests/Billing tldw_Server_API/tests/Storage/test_native_chat_quota_reservations.py`. Expected: no failures beyond the Task 0 baseline. Measured with Task 1 applied on `f8f913fab8`: `2334 passed, 1 skipped` (12 minutes), with `test_llm_quota_byok_calls.py` ignored and Task 3's appended decision tests deselected (now 10 cases). r3 adds no test to this set, so it was not re-run.
- [ ] **Step 8: Bump the size ceiling.** In `tldw_Server_API/tests/lint/function_size_baseline.json`, the `get_authnz_migrations` entry (`{"lines": 441, "params": 0, "reason": null}`) becomes:

```json
  "tldw_Server_API.app.core.AuthNZ.migrations:get_authnz_migrations": {"lines": 442, "params": 0, "reason": "registers migration 104, llm_usage_log.credential_source (one line, TASK-<ID>)"},
```

  Then `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/lint/test_function_size_ratchet.py` must pass (it fails on this entry before the bump).
- [ ] **Step 9: Commit** (message 1): `git add` the ten files, then `git diff --cached --check`.

## Task 2: Carry the source to the recorder and the check at the call (AC #1, #3)

**Files:** see "Files" (Task 2).

**Interfaces:**
- Consumes: `log_llm_usage(credential_source=...)` (Task 1).
- Produces:
  - `ProviderCallCredentials.credential_source: str | None`: read-only (in `_IMMUTABLE_SLOT_NAMES`), set by `ProviderCredentialRuntime._new_handle` from `resolution.source`. The constructor keyword defaults to None.
  - `LLMCallContext.credential_source: str | None = None`.
  - `USER_KEY_SOURCE = "user"`, `USER_KEY_UNMETERED_SOURCE = "user_unmetered"`, `VENDOR_HOSTS`, `call_credential_source(request: Any = None) -> str | None` and `billed_credential_source(provider, source, *configs) -> str | None` in `Usage/llm_call_context.py`.
  - `user_to_check(credential_source: str | None = None)`, `check_llm_call_sync(credential_source=None)` and `check_llm_call_async(credential_source=None)` in `llm_quota_gate.py`.
  - In `usage_accounting.py`: `_CallPlan.credential_source`, `_plan_call(credential_source=None)`, `record_direct_response(..., *, credential_source=None)` and `accounted_direct_stream(..., *, credential_source=None)`.
- Consumed by Task 3: `user_to_check`, `check_llm_call_async(credential_source)` and `billed_credential_source`.

**RED reason (measured with Task 1 applied, on `03fec47781`):** `31 failed, 5 passed`. The five that pass already are the guards: a call on a server, team, org or operator-endpoint key is refused (four cases), and a look-alike handle is ignored.
- 6 with `KeyError: 'credential_source'`: the recorded row has no such key. These are the shared-key test, both `/chat/completions` cases, the failover test and both native Messages cases.
- 4 with `LLMQuotaExceededError: 402`: own-key calls are refused today. These are the four-method test, both shared call layer cases, and the nested-method test.
- 1 with `AttributeError: 'ProviderCallCredentials' object has no attribute 'credential_source'` (the handle test).
- 1 with `TypeError: LLMCallContext.__init__() got an unexpected keyword argument 'credential_source'` (the context test).
- 6 with `ImportError: cannot import name 'call_credential_source'` (the unreadable-request test and the five junk-key cases).
- 8 with `ImportError: cannot import name 'billed_credential_source'` (the vendor-host rule test), and 1 with `ImportError: cannot import name 'VENDOR_HOSTS'` (the table test).
- 1 with `assert ['openai', 'openai-fallback'] == ['openai']`: the refused failover is charged to the fallback provider (the refused-failover test).
- 1 that surfaces the context's missing attribute as its own sanitized error: the prompt-improvement test (`PromptImprovementDispatchError`).
- 2 with `AttributeError: <module ... shared_workspace_chat_service> has no attribute 'load_server_config_snapshot'`: both shared-workspace cases, whose test pins the server snapshot that Task 2 imports.

- [ ] **Step 1: Create `tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py`:**

```python
"""A call made with the user's own provider key is recorded as such and never checked against the LLM quotas (TASK-<ID>)."""

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.Usage import llm_accounting, llm_quota_gate, quota_resolver
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context
from tldw_Server_API.app.core.Usage.quota_checks import LLMQuotaRefusal

pytestmark = pytest.mark.unit

REQUEST = {"model": "m1", "messages": [{"role": "user", "content": "hello"}]}
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
REFUSAL = LLMQuotaRefusal("llm_tokens_month", 100, 100, "Monthly LLM token limit reached", 60)


class _Provider(ChatProvider):
    """A provider whose four call methods answer with a usage block."""

    name = "openai"

    def capabilities(self) -> dict[str, Any]:
        """Streaming is supported."""
        return {"supports_streaming": True}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Answer with usage."""
        return {"choices": [{"message": {"content": "hi"}}], "usage": dict(USAGE)}

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """One chunk with usage."""
        yield 'data: {"choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Answer with usage."""
        return {"choices": [{"message": {"content": "hi"}}], "usage": dict(USAGE)}

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        """One chunk with usage."""
        yield 'data: {"choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'


async def _handle(source: str, provider: str = "openai") -> ProviderCallCredentials:
    """A runtime-issued credential handle for ``provider`` whose key comes from ``source``."""

    async def _resolver(provider: str, **_kwargs: Any) -> ResolvedByokCredentials:
        """Resolve every provider to a key from ``source``."""
        return ResolvedByokCredentials(
            provider=provider, api_key="sk-test", app_config={}, credential_fields={}, source=source, allowlisted=True
        )

    runtime = ProviderCredentialRuntime(
        user_id=7, team_ids=[], org_ids=[], trusted_base_url_override=False,
        fallback_resolver=lambda _provider: None, server_config_snapshot={}, resolver=_resolver,
    )
    return await runtime.resolve(provider)


@pytest.fixture
async def refused(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Quotas on, user 7 over a limit, the test's loop as the accounting loop; returns the rows written."""
    rows: list[dict[str, Any]] = []

    async def _decide(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Refuse user 7."""
        return REFUSAL if user_id == 7 else None

    async def _fake_log(**kwargs: Any) -> None:
        """Keep the row instead of writing it."""
        rows.append(kwargs)

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _decide)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _fake_log)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    return rows


async def _call_all_four(request: dict[str, Any]) -> None:
    """chat, achat, a read stream and a read astream."""
    _Provider().chat(request)
    await _Provider().achat(request)
    list(_Provider().stream(request))
    [line async for line in _Provider().astream(request)]


async def test_a_runtime_handle_says_whose_key_it_carries() -> None:
    """The handle takes its source from the resolution; a user key is the user's own only where the vendor bills it."""
    assert (await _handle("user")).credential_source == "user"
    assert (await _handle("server_default")).credential_source == "server_default"
    for provider in ("custom-openai-api", "custom-openai-api-7", "llama.cpp"):
        assert (await _handle("user", provider)).credential_source == "user_unmetered"
        assert (await _handle("server_default", provider)).credential_source == "server_default"


@pytest.mark.parametrize(
    ("provider", "section", "expected"),
    [
        ("openai", {}, "user"),
        ("openai", {"api_base_url": "https://api.openai.com/v1"}, "user"),
        ("openai", {"api_base_url": "http://operator-gpu.internal:8000/v1"}, "user_unmetered"),
        ("qwen", {"api_base_url": "https://dashscope-us.aliyuncs.com/compatible-mode/v1"}, "user"),
        ("bedrock", {"runtime_endpoint": "https://bedrock-runtime.eu-west-1.amazonaws.com"}, "user"),
        ("bedrock", {"runtime_endpoint": "https://ec2-3-4-5-6.compute-1.amazonaws.com"}, "user_unmetered"),
        ("huggingface", {"router_base_url": "http://operator-gpu.internal:8000"}, "user_unmetered"),
        ("lmstudio", {}, "user_unmetered"),
    ],
    ids=[
        "vendor-default", "vendor-url", "operator-url", "vendor-region", "bedrock-region", "bedrock-not-bedrock",
        "router-url", "unknown-provider",
    ],
)
def test_a_user_key_is_the_users_own_spend_only_on_the_vendors_host(provider, section, expected) -> None:
    """Any base URL but the vendor's is an endpoint the operator chose, and an unknown provider is never billed."""
    from tldw_Server_API.app.core.Usage.llm_call_context import billed_credential_source

    assert billed_credential_source(provider, "user", {f"{provider}_api": section}) == expected


def test_every_provider_that_bills_per_key_names_its_vendors_host() -> None:
    """A billed provider missing from the host table would count its users' own keys."""
    from tldw_Server_API.app.core.LLM_Calls.provider_metadata import PROVIDER_REQUIRES_KEY
    from tldw_Server_API.app.core.Usage.llm_call_context import VENDOR_HOSTS

    assert set(VENDOR_HOSTS) == {name for name, billed in PROVIDER_REQUIRES_KEY.items() if billed}


async def test_calls_on_the_users_own_key_are_not_checked_and_are_recorded_as_such(refused) -> None:
    """An over-limit user's four calls on their own key all run, and each row says ``user``."""
    request = {**REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user")}
    with llm_call_context(user_id=7):
        await _call_all_four(request)
    await llm_accounting.drain_accounting()
    assert [row["credential_source"] for row in refused] == ["user"] * 4


@pytest.mark.parametrize(
    ("source", "provider"),
    [("server_default", "openai"), ("team", "openai"), ("org", "openai"), ("user", "custom-openai-api")],
)
async def test_a_call_on_any_other_key_is_still_refused(refused, source, provider) -> None:
    """The same user's call on a server, shared or operator-endpoint key is refused before the provider runs."""
    request = {**REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle(source, provider)}
    with llm_call_context(user_id=7), pytest.raises(LLMQuotaExceededError):
        await _Provider().achat(request)
    await llm_accounting.drain_accounting()
    assert refused == []


async def test_an_admitted_call_on_a_shared_key_is_recorded_with_its_source(refused) -> None:
    """A team key's call for an admitted user is recorded with ``team``, so it still counts."""
    request = {**REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("team")}
    with llm_call_context(user_id=8):
        await _Provider().achat(request)
    await llm_accounting.drain_accounting()
    assert [row["credential_source"] for row in refused] == ["team"]


async def test_the_call_context_names_the_key_of_a_call_without_a_handle(refused) -> None:
    """A caller that resolved the key itself marks the context; its call runs and is recorded as ``user``."""
    with llm_call_context(user_id=7, credential_source="user"):
        await _Provider().achat(REQUEST)
    await llm_accounting.drain_accounting()
    assert [row["credential_source"] for row in refused] == ["user"]


async def test_a_handle_the_runtime_did_not_issue_is_ignored(refused) -> None:
    """A look-alike object claiming ``user`` does not skip the check."""
    request = {**REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: SimpleNamespace(credential_source="user")}
    with llm_call_context(user_id=7), pytest.raises(LLMQuotaExceededError):
        await _Provider().achat(request)


def test_a_request_that_cannot_be_read_is_an_unknown_source() -> None:
    """Reading the source never breaks a call: an unreadable request is unknown, so the call is checked."""
    from tldw_Server_API.app.core.Usage.llm_call_context import call_credential_source

    class _Unreadable(dict):
        """A request mapping whose lookups fail."""

        def get(self, *_args: Any) -> Any:
            """Fail."""
            raise RuntimeError("unreadable")

    with llm_call_context(credential_source="user"):
        assert call_credential_source(None) == "user"
        assert call_credential_source(_Unreadable()) is None


@pytest.mark.parametrize("layer", ["sync", "async"])
async def test_the_shared_call_layer_does_not_check_a_call_on_the_users_own_key(refused, monkeypatch, layer) -> None:
    """perform_chat_api_call (what /chat/completions runs) and its async twin read the handle before they check."""
    from tldw_Server_API.app.core.Chat import chat_service

    async def _call(**kwargs: Any) -> Any:
        """The layer under test; the sync one in a worker thread, where its check can run."""
        if layer == "async":
            return await chat_service.perform_chat_api_call_async(**kwargs)
        return await asyncio.to_thread(chat_service.perform_chat_api_call, **kwargs)

    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: SimpleNamespace(get_adapter=lambda _name: _Provider()))
    monkeypatch.setattr(chat_service, "has_model_listing", lambda _provider: False)  # no model discovery request
    args = {
        "api_provider": "openai", "messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o-mini",
        "credentials_resolved": True, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user"),
    }
    with llm_call_context(user_id=7):
        response = await _call(**args)
        with pytest.raises(LLMQuotaExceededError):
            await _call(**{**args, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("server_default")})
    assert response["usage"] == USAGE


@pytest.mark.parametrize(
    ("provider", "section", "field"),
    [
        ("custom-openai-api", "custom_openai_api", "api_ip"),
        ("qwen", "qwen_api", "api_base_url"),
        ("moonshot", "moonshot_api", "api_base_url"),
        ("zai", "zai_api", "api_base_url"),
        ("huggingface", "huggingface_api", "api_base_url"),
    ],
)
async def test_a_junk_key_stored_for_the_operators_endpoint_is_still_checked_and_counted(
    tmp_path, monkeypatch, provider: str, section: str, field: str
) -> None:
    """A key for a custom slot, or for a billed provider the operator pointed at its own server, is not the user's spend."""
    from datetime import datetime, timedelta, timezone

    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
    from tldw_Server_API.app.core.Usage.llm_call_context import call_credential_source
    from tldw_Server_API.app.core.Usage.usage_tracker import log_llm_usage
    from tldw_Server_API.tests.AuthNZ_SQLite.test_byok_endpoints_sqlite import _setup_byok_sqlite
    from tldw_Server_API.tests.AuthNZ_SQLite.test_byok_runtime_sqlite import _upsert_user_key

    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    await _upsert_user_key(AuthnzUserProviderSecretsRepo(state["pool"]), user_id, provider, "anything")
    runtime = ProviderCredentialRuntime(
        user_id=user_id, team_ids=[], org_ids=[], trusted_base_url_override=False,
        server_config_snapshot={section: {field: "http://operator-gpu.internal:8000/v1"}},
    )
    try:
        handle = await runtime.resolve(provider)
    finally:
        await runtime.close()
    source = call_credential_source({PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle})
    assert source == "user_unmetered"

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    monkeypatch.setenv("LLM_USAGE_ENABLED", "1")
    quota_resolver.invalidate_all()
    with llm_call_context(user_id=user_id):
        assert llm_quota_gate.user_to_check(source) == user_id
    await log_llm_usage(
        user_id=user_id, key_id=None, endpoint="internal", operation="chat", provider=provider,
        model="operator-llama", status=200, latency_ms=1, prompt_tokens=30, completion_tokens=10,
        request_id="operator-endpoint", credential_source=source,
    )
    since = datetime.now(timezone.utc) - timedelta(days=1)
    assert await AuthnzUsageRepo(state["pool"]).sum_user_llm_tokens_since(user_id=user_id, since=since) == 40.0


async def test_a_provider_method_nested_in_an_own_key_call_is_not_checked_again(refused) -> None:
    """A self-recording caller's own-key call that delegates without the handle is not refused by the delegate."""

    class _Delegating(_Provider):
        """chat() reads its own stream, as some adapters do."""

        def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            """Delegate to stream() with a request that no longer carries the handle."""
            inner = {key: value for key, value in request.items() if key != PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY}
            return {"chunks": list(self.stream(inner))}

    request = {**REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user")}
    with llm_call_context(user_id=7, records_itself=True):
        response = await asyncio.to_thread(_Delegating().chat, request)  # off the loop, where a check can run
    assert len(response["chunks"]) == 1


def _chat_request() -> SimpleNamespace:
    """A /chat/completions request for user 7."""
    return SimpleNamespace(
        method="POST", url=SimpleNamespace(path="/api/v1/chat/completions"), headers={},
        state=SimpleNamespace(user_id=7, api_key_id=None, team_ids=None, org_ids=None),
    )


class _Metrics:
    """Accepts every chat metrics call."""

    def __getattr__(self, _name: str) -> Any:
        """Any tracker is a no-op."""
        return lambda *_args, **_kwargs: None

    @asynccontextmanager
    async def track_streaming(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        """A stream tracker that ignores everything."""
        yield SimpleNamespace(add_heartbeat=lambda: None, add_chunk=lambda: None)


def _executor_kwargs(cleaned_args: dict[str, Any], llm_call_func: Any) -> dict[str, Any]:
    """The arguments both /chat/completions executors take."""
    from tldw_Server_API.tests.Chat.unit.test_streaming_structured_events import _NoModeration

    async def _save(_db: Any, _conversation_id: Any, _payload: Any, use_transaction: bool = True) -> str:
        """Pretend to save the reply."""
        return "m-1"

    return {
        "current_loop": asyncio.get_running_loop(), "cleaned_args": cleaned_args, "selected_provider": "openai",
        "provider": "openai", "model": "gpt-4o-mini", "request_json": "{}", "request": _chat_request(),
        "metrics": _Metrics(), "provider_manager": None,
        "templated_llm_payload": [{"role": "user", "content": "hi"}], "should_persist": False,
        "final_conversation_id": "conv-byok", "character_card_for_context": {"name": "Test"},
        "chat_db": SimpleNamespace(), "save_message_fn": _save, "audit_service": None, "audit_context": None,
        "client_id": "client", "queue_execution_enabled": False, "enable_provider_fallback": False,
        "llm_call_func": llm_call_func, "refresh_provider_params": lambda *_args, **_kwargs: ({}, None),
        "moderation_getter": lambda: _NoModeration(),
    }


@pytest.mark.parametrize("stream", [False, True])
async def test_chat_completions_records_its_reply_with_the_source_of_its_key(monkeypatch, stream) -> None:
    """/chat/completions records its own reply, streamed or not, with the source of the handle it ran on."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.tests.Chat.unit.test_streaming_structured_events import _collect_sse_chunks

    logged: list[dict[str, Any]] = []

    async def _log(**kwargs: Any) -> None:
        """Keep the row."""
        logged.append(kwargs)

    monkeypatch.setattr(chat_service, "log_llm_usage", _log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: False)
    cleaned_args = {
        "api_endpoint": "openai", "messages_payload": [{"role": "user", "content": "hi"}], "model": "gpt-4o-mini",
        "streaming": stream, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user"),
    }
    if stream:
        response = await chat_service.execute_streaming_call(**_executor_kwargs(cleaned_args, lambda: iter(["hi"])))
        await _collect_sse_chunks(response)
    else:
        reply = {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}], "usage": USAGE}
        await chat_service.execute_non_stream_call(**_executor_kwargs(cleaned_args, lambda: reply))
    assert [row["credential_source"] for row in logged] == ["user"]


async def test_a_failover_reply_is_recorded_with_the_fallback_providers_key(monkeypatch) -> None:
    """The user's own key fails over to a server key: the reply that counts is the server key's."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError

    logged: list[dict[str, Any]] = []
    server_handle = await _handle("server_default")
    primary_error = ChatProviderError(provider="openai", message="primary unavailable", status_code=502)
    primary_error.upstream_dispatched = False
    primary_error.output_emitted = False
    primary_error.allow_non_stream_fallback = True

    async def _log(**kwargs: Any) -> None:
        """Keep the row."""
        logged.append(kwargs)

    def _fail_primary() -> None:
        """The user's provider is down before dispatch."""
        raise primary_error

    async def _fallback_call(**kwargs: Any) -> dict[str, Any]:
        """The fallback provider answers on the server's key."""
        assert kwargs[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is server_handle
        return {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}], "usage": USAGE}

    def _refresh(provider: str) -> tuple[dict[str, Any], str]:
        """The fallback provider's arguments, with the server key."""
        args = {"api_endpoint": provider, "messages_payload": [], "model": "claude-x", "streaming": False}
        return {**args, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: server_handle}, "claude-x"

    monkeypatch.setattr(chat_service, "log_llm_usage", _log)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", _fallback_call)
    cleaned_args = {
        "api_endpoint": "openai", "messages_payload": [], "model": "gpt-4o-mini", "streaming": False,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user"),
    }
    kwargs = _executor_kwargs(cleaned_args, _fail_primary)
    kwargs.update(
        enable_provider_fallback=True,
        refresh_provider_params=_refresh,
        provider_manager=SimpleNamespace(
            get_available_provider=lambda exclude=None: "anthropic",
            record_failure=lambda *_args: None,
            record_success=lambda *_args: None,
        ),
    )
    response = await chat_service.execute_non_stream_call(**kwargs)
    assert response["choices"][0]["message"]["content"] == "hi"
    assert [(row["provider"], row["credential_source"]) for row in logged] == [("anthropic", "server_default")]


async def test_a_refused_failover_does_not_mark_the_fallback_provider_unhealthy(refused, monkeypatch) -> None:
    """An over-limit user's own key fails over to a server key: the 402 reaches the caller, and the fallback is not blamed."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError

    failures: list[str] = []
    server_handle = await _handle("server_default")
    primary_error = ChatProviderError(provider="openai", message="primary unavailable", status_code=502)
    primary_error.upstream_dispatched = False
    primary_error.output_emitted = False
    primary_error.allow_non_stream_fallback = True

    def _fail_primary() -> None:
        """The user's provider is down before dispatch."""
        raise primary_error

    def _refresh(provider: str) -> tuple[dict[str, Any], str]:
        """The fallback provider's arguments, with the server key."""
        args = {"api_endpoint": "openai", "messages_payload": [{"role": "user", "content": "hi"}], "model": "m"}
        return {**args, "streaming": False, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: server_handle}, "m"

    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: SimpleNamespace(get_adapter=lambda _name: _Provider()))
    cleaned_args = {
        "api_endpoint": "openai", "messages_payload": [], "model": "gpt-4o-mini", "streaming": False,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user"),
    }
    kwargs = _executor_kwargs(cleaned_args, _fail_primary)
    kwargs.update(
        enable_provider_fallback=True,
        refresh_provider_params=_refresh,
        provider_manager=SimpleNamespace(
            get_available_provider=lambda exclude=None: "openai-fallback",
            record_failure=lambda name, *_args: failures.append(name),
            record_success=lambda *_args: None,
        ),
    )
    with llm_call_context(user_id=7), pytest.raises(LLMQuotaExceededError) as refusal:
        await chat_service.execute_non_stream_call(**kwargs)
    assert refusal.value.detail["category"] == "llm_tokens_month"
    assert failures == ["openai"]


async def test_prompt_improvement_names_whose_key_its_call_runs_on(monkeypatch) -> None:
    """The dispatcher resolves the key itself, so it puts the key's source in the call context."""
    from tldw_Server_API.app.core.LLM_Calls.routing import InMemoryRoutingDecisionStore
    from tldw_Server_API.app.core.Prompt_Management import prompt_improvement_dispatch as dispatch
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
    from tldw_Server_API.tests.Prompt_Management_NEW.integration.test_prompt_improvement_layered import (
        _provider_listing,
        _request_context,
        _route_user,
    )

    seen: list[str | None] = []

    async def _touch() -> None:
        """Nothing to touch."""

    async def _resolve(_provider: str, **_kwargs: Any) -> SimpleNamespace:
        """The user's own key."""
        return SimpleNamespace(source="user", api_key="sk-user", app_config=None, touch_last_used=_touch)

    async def _perform(**_kwargs: Any) -> dict[str, Any]:
        """Note the call context's source and answer with no change."""
        seen.append(current_llm_call_context().credential_source)
        content = '{"status":"no_change","improved_text":null,"findings":[]}'
        return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}

    monkeypatch.setattr(dispatch, "resolve_byok_credentials", _resolve)
    monkeypatch.setattr(dispatch, "perform_chat_api_call_async", _perform)
    result = await dispatch.dispatch_prompt_improvement(
        request=_request_context(), current_user=_route_user(), routing_decision_store=InMemoryRoutingDecisionStore(),
        selected_model="moonshot/kimi-k2", provider_hint=None, messages=[{"role": "user", "content": "draft"}],
        request_id="byok-dispatch", configured_providers_getter=_provider_listing,
    )
    assert (result.provider, result.model) == ("moonshot", "kimi-k2")
    assert seen == ["user"]
```

  Notes:
  - `_handle` builds real runtime-issued handles with an injected resolver, so the `is_runtime_issued_provider_call_credentials` guard is exercised.
  - The `refused` fixture refuses user 7 by replacing `llm_quota_gate.llm_quota_refusal`, the pattern of `tests/Usage/test_llm_quota_gate.py`.
  - The `/chat/completions` tests drive the real `execute_non_stream_call` and `execute_streaming_call` with a permissive metrics fake.
  - The failover tests certify a pre-dispatch error, as `tests/Chat/unit/test_chat_service_fallback.py::_certified_pre_dispatch` does. The refused-failover test is the review's probe B (I1). It drives the non-stream block; the queued and direct stream blocks get the same two lines.
  - The call-layer test runs `perform_chat_api_call` in a worker thread, because the sync check cannot run on the accounting loop's own thread (`check_user_sync` returns False there). It stubs `has_model_listing`: unstubbed, the sync layer asked `api.openai.com` for its model list.
  - The junk-key test is the first review's probe C and the re-review's probe D on real storage. A key stored for `custom-openai-api`, or for `qwen`, `moonshot`, `zai` or `huggingface` whose base URL the server config points at the operator's own server, resolves to `user_unmetered`; the call is still checked for that user, and its row counts in the sum. On the r2 prototype the four billed cases failed with `assert 'user' == 'user_unmetered'`.
  - The rule test pins the vendor-host table's edges: a vendor URL and a regional vendor host keep `user`; an operator URL, an AWS host that is not Bedrock's, an operator router URL and an unknown provider name (`lmstudio`, review m3) do not. The table test keeps `VENDOR_HOSTS` equal to the providers `PROVIDER_REQUIRES_KEY` marks as billed, so a new billed provider gets its host.
  - The nested-method test is review M8. A self-recording caller's own-key call that delegates to another provider method, without the handle, must not be checked again. It runs in a worker thread, where a check can run.
  - The refused-call test covers review I2's `shared_exempt` mutation (`team`, `org`) and B1's operator endpoint.

- [ ] **Step 2: Append to `tldw_Server_API/tests/Sharing/test_shared_workspace_chat_generation.py`:**

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["user", "team"])
async def test_generation_names_whose_key_its_call_runs_on(monkeypatch, generation_stubs, source: str) -> None:
    """The provider call runs with the credential's source in the call context, so the LLM quota sees it (TASK-<ID>)."""
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context

    byok, _captured = generation_stubs
    byok.source = source
    seen: list[str | None] = []

    async def _perform(**_kwargs):
        seen.append(current_llm_call_context().credential_source)
        return _provider_response('{"answer":"Grounded","citations":["E1"]}')

    monkeypatch.setattr(shared_chat, "perform_chat_api_call_async", _perform)
    monkeypatch.setattr(shared_chat, "load_server_config_snapshot", dict)  # no base URL leaked from the environment
    result = await _service().generate_grounded_answer(
        query="Question?",
        evidence=(_evidence(),),
        target=ResolvedChatTarget("openai", "gpt-model"),
        recipient_user_id=41,
        share_scope_type="team",
        share_scope_id=73,
        request=SimpleNamespace(),
    )
    assert result.answer == "Grounded"
    assert seen == [source]
```

  The test pins `load_server_config_snapshot` to an empty config. The `tests/_plugins/chat_fixtures.py` plugin sets `OPENAI_API_BASE=http://localhost:8080/v1` at import, for the rest of its xdist worker, and the rule then rightly reads `openai` as an operator endpoint. Measured: before the pin, `[user]` failed with `['user_unmetered'] == ['user']` in one wider run and passed alone.

- [ ] **Step 3: Append to `tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py`:**

```python
class _UserKeyRuntime(_Runtime):
    """Resolves the user's own key."""

    async def resolve(self, provider: str, *, model: str | None = None) -> Any:
        """Credentials whose source is the user's own key."""
        del model
        return SimpleNamespace(
            provider=provider, api_key="k", app_config={}, credentials_resolved=True, credential_source="user"
        )


@pytest.mark.parametrize("stream", [False, True])
async def test_a_native_call_on_the_users_own_key_is_recorded_as_such(rows, monkeypatch, stream) -> None:
    """The row says the user's own key paid, so the LLM quota counters leave it out (TASK-<ID>)."""
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _UserKeyRuntime)
    if stream:
        row = await _stream_row(monkeypatch, rows, STREAM_FRAMES)
    else:
        monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply(NATIVE_RESPONSE["usage"]))
        with scoped_context(user_id=5):
            assert (await _call("anthropic/claude-3-5-haiku-latest", stream=False)).status_code == 200
        await llm_accounting.drain_accounting()
        (row,) = rows
    assert row["credential_source"] == "user"
```

- [ ] **Step 4: Run; see RED.** The Task 2 command. Expected: `31 failed, 5 passed`.
- [ ] **Step 5: Implement.**

`tldw_Server_API/app/core/AuthNZ/provider_credential_runtime.py`, class `ProviderCallCredentials`:
- Add `"credential_source",` as the last item of `_IMMUTABLE_SLOT_NAMES` and of `__slots__`.
- In `__init__`, add the keyword `credential_source: str | None = None,` after `trusted_endpoint: TrustedProviderEndpoint | None = None,`. After `self._credential_identity = credential_identity`, add:

```python
        # Whose key this is (``user``, ``team``, ``org``, ``server_default``); usage accounting reads it.
        self.credential_source = credential_source
```

- In `ProviderCredentialRuntime._new_handle`, after the `trusted_endpoint=_trusted_endpoint_from_snapshot(...)` argument, add `credential_source=billed_credential_source(provider, resolution.source, app_config),`. `app_config` is the handle's own copy, and on a handle call it is the only place an adapter takes its base URL from (`bind_provider_call_credentials` sets `credentials_resolved`, and every billed adapter then skips the environment).
- Add `from tldw_Server_API.app.core.Usage.llm_call_context import billed_credential_source` after `from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope`.

`tldw_Server_API/app/core/Usage/llm_call_context.py`:
- Change the import `from collections.abc import Awaitable, Callable, Iterator` to `from collections.abc import Awaitable, Callable, Iterator, Mapping`. Add `from fnmatch import fnmatchcase` after `from dataclasses import dataclass, replace`, and `from urllib.parse import urlsplit` after `from typing import Any`.
- After the `get_scope` import, add:

```python

# The credential source of a call made with the user's own key on a provider that bills that key.
# The operator pays nothing for it, so the LLM quotas neither check nor count it (TASK-<ID>).
USER_KEY_SOURCE = "user"
# A user key on an endpoint the operator runs (a custom OpenAI-compatible slot, a local server, a
# billed provider pointed at a model server or gateway). It answers whatever key is sent, so the call
# counts like a server-key call.
USER_KEY_UNMETERED_SOURCE = "user_unmetered"
# Each provider that bills per key, and the host(s) its adapter calls by default (fnmatch patterns).
# A user key is the user's own spend only there (TASK-<ID>).
VENDOR_HOSTS: dict[str, tuple[str, ...]] = {
    "anthropic": ("api.anthropic.com",),
    "bedrock": ("bedrock-runtime.*.amazonaws.com",),
    "cohere": ("api.cohere.ai",),
    "deepseek": ("api.deepseek.com",),
    "google": ("generativelanguage.googleapis.com",),
    "groq": ("api.groq.com",),
    "huggingface": ("api-inference.huggingface.co", "router.huggingface.co"),
    "mistral": ("api.mistral.ai",),
    "moonshot": ("api.moonshot.cn",),
    "novita": ("api.novita.ai",),
    "openai": ("api.openai.com",),
    "openrouter": ("openrouter.ai",),
    "poe": ("api.poe.com",),
    "qwen": ("dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com", "dashscope.aliyuncs.com"),
    "together": ("api.together.xyz",),
    "zai": ("api.z.ai",),
}
# Every provider-section key an adapter takes its base URL from.
_BASE_URL_KEYS = (
    "base_url", "api_base_url", "api_base", "api_url", "api_ip", "endpoint", "runtime_endpoint",
    "router_base_url", "huggingface_router_base_url",
)
```

  Each host is the one the adapter itself falls back to (re-checked on `03fec47781`): `openai_adapter.py:241`, `anthropic_adapter.py:89`, `bedrock_adapter.py:308` (`bedrock-runtime.{region}.amazonaws.com`), `cohere_adapter.py:141`, `deepseek_adapter.py:169`, `google_adapter.py:173`, `groq_adapter.py:42`, `huggingface_adapter.py:31-32`, `mistral_adapter.py:86`, `moonshot_adapter.py:191`, `openrouter_adapter.py:67`, `qwen_adapter.py:26-28` (its three regions), `zai_adapter.py:110`, and `custom_openai_adapter.py:415,422,429` for Novita, Poe and Together. `_BASE_URL_KEYS` is the union of the keys those adapters read (`byok_config._ENDPOINT_CONFIG_ALIASES` plus Hugging Face's two router keys). The table is code, not configuration: no new setting.

- Add the field `credential_source: str | None = None` after `prechecked: bool = False` in `LLMCallContext`.
- Before `def current_llm_user_id()`, add:

```python
def call_credential_source(request: Any = None) -> str | None:
    """Whose key a call uses: its request's runtime credential handle, else the call context's.

    A request that carries a runtime-issued handle is decided by the handle alone.
    Never raises: a request that cannot be read is an unknown source, so its call is checked and counted.
    """
    try:
        from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
            PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
            is_runtime_issued_provider_call_credentials,
        )

        handle = request.get(PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY) if isinstance(request, Mapping) else None
        if is_runtime_issued_provider_call_credentials(handle):
            return handle.credential_source
        return _CTX.get().credential_source
    except Exception:  # noqa: BLE001 - accounting must never break an LLM call
        return None


```

- After `call_credential_source`, add:

```python
def billed_credential_source(provider: str | None, source: str | None, *configs: Any) -> str | None:
    """``source`` as the LLM quotas read it: a user key is the user's own spend only on its vendor's host.

    ``user`` stays ``user`` only for a provider in ``VENDOR_HOSTS`` whose base
    URLs in ``configs`` (a resolved ``app_config``, a server config snapshot) are
    all on the vendor's host. Otherwise it becomes ``user_unmetered``, which
    counts: a custom OpenAI-compatible slot, a local server, an unknown provider,
    or a billed provider pointed at a model server or gateway. Every other source
    is returned unchanged, so nothing else ever becomes ``user``.
    """
    if source != USER_KEY_SOURCE:
        return source
    from tldw_Server_API.app.core.AuthNZ.byok_config import PROVIDER_APP_CONFIG_KEYS
    from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name

    name = canonical_provider_name(provider) if provider else ""
    hosts = VENDOR_HOSTS.get(name)
    if hosts is None:
        return USER_KEY_UNMETERED_SOURCE
    for config in configs:
        section = config.get(PROVIDER_APP_CONFIG_KEYS[name]) if isinstance(config, Mapping) else None
        for key in _BASE_URL_KEYS if isinstance(section, Mapping) else ():
            url = section.get(key)
            if isinstance(url, str) and url.strip() and not _on_vendor_host(url, hosts):
                return USER_KEY_UNMETERED_SOURCE
    return source


def _on_vendor_host(url: str, hosts: tuple[str, ...]) -> bool:
    """Whether ``url``'s host matches one of the vendor's host patterns."""
    try:
        host = urlsplit(url.strip()).hostname or ""
    except ValueError:  # a malformed URL is not the vendor's
        return False
    return any(fnmatchcase(host, pattern) for pattern in hosts)
```

  `call_credential_source` imports inside the function because `provider_credential_runtime` pulls in the BYOK runtime, which `llm_call_context` must not load at import. `billed_credential_source` imports inside the function too, for the same rule, although its two modules are small. It does not raise: a malformed URL counts.

  Why the vendor's host and not `provider_requires_api_key` alone (re-review I1): every billed adapter takes its base URL from its `app_config` section before its default, and the server config fills that section from `config.txt` (`qwen_api_base_url`, `moonshot_api_base_url`, `zai_api_base_url`, `huggingface_api_base_url`, `bedrock_runtime_endpoint`) and from the environment (`OPENAI_API_BASE_URL`, `ANTHROPIC_BASE_URL`, `GROQ_BASE_URL` and the rest of `byok_config.PROVIDER_RUNTIME_ENV_CONFIG_KEYS`, captured by `load_server_config_snapshot`). So an operator who serves a model behind a billed provider's name made every user key there `user`. Measured on `03fec47781`: with the shipped `config.txt` and no base-URL environment variable, every billed provider keeps `user` (the four configured base URLs are the vendors'). An unknown provider name is `user_unmetered` (review m3), so an adapter registered later without a table entry counts until it gets one.

`tldw_Server_API/app/core/Usage/llm_quota_gate.py`:
- Add `USER_KEY_SOURCE,` first in the `llm_call_context` import list.
- Replace `user_to_check`'s signature, docstring and first test with:

```python
def user_to_check(credential_source: str | None = None) -> int | None:
    """The user this call is checked for, or None when it is not checked.

    Not checked: a call on the user's own key (``credential_source`` ``user``),
    no user, an exempt or admitted (``prechecked``) context, a call nested in an
    accounted call, quotas off, or a user whose cached limits set no LLM key.
    """
    try:
        if credential_source == USER_KEY_SOURCE or in_accounted_call():
            return None
```

- Replace `check_llm_call_sync` and `check_llm_call_async` with:

```python
def check_llm_call_sync(credential_source: str | None = None) -> bool:
    """Check the current call from synchronous code. False when it could not be checked here."""
    user_id = user_to_check(credential_source)
    return True if user_id is None else check_user_sync(user_id)


async def check_llm_call_async(credential_source: str | None = None) -> None:
    """Check the current call from a coroutine."""
    user_id = user_to_check(credential_source)
    if user_id is not None:
        await check_user_async(user_id)
```

`tldw_Server_API/app/core/LLM_Calls/usage_accounting.py`:
- Add `USER_KEY_SOURCE,` before `accounted_call,` and `call_credential_source,` after it, in the `llm_call_context` import.
- `_CallPlan`: add the field `credential_source: str | None = None` after `endpoint: str`.
- `_plan_call`: change the signature to `def _plan_call(credential_source: str | None = None) -> _CallPlan | None:` and the return to:

```python
        return _CallPlan(
            user_id=int(user_id), operation=context.operation, endpoint=context.endpoint,
            credential_source=credential_source,
        )
```

- `_submit_record`: in the `log_llm_usage(...)` call, after `estimate_source=estimate_source,`, add `credential_source=plan.credential_source,`.
- In each of the four wrappers (`chat`, `achat`, `stream`, `astream`), the first lines become (shown for `chat`; `achat` and `stream` keep their own next line, and `astream` has no check line here):

```python
        source = call_credential_source(request)
        user_id = user_to_check(source)
```

  and `plan = _plan_call()` becomes `plan = _plan_call(source)`.
- In `_wrap_chat` and `_wrap_achat`, a `user` call with no plan (a `records_itself` context) also runs inside `accounted_call()`, so a provider method it delegates to is not checked again (review M8). `_wrap_chat` becomes (`_wrap_achat` the same, with `await`):

```python
def _wrap_chat(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Record after a sync chat call."""

    @functools.wraps(fn)
    def chat(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        source = call_credential_source(request)
        user_id = user_to_check(source)
        if user_id is not None:
            check_user_sync(user_id)
        plan = _plan_call(source)
        if plan is None:
            if user_id is None and source != USER_KEY_SOURCE:
                return fn(self, request, *args, **kwargs)
            with accounted_call():  # checked here or on the user's own key: a nested method is not checked again
                return fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():
            response = fn(self, request, *args, **kwargs)
        _record_response(self, plan, request, response, started)
        return response

    return chat
```
- Replace `record_direct_response` and `accounted_direct_stream` with:

```python
def record_direct_response(
    provider: str, request: dict[str, Any], response: Any, started: float, *, credential_source: str | None = None
) -> None:
    """Record a successful provider call made without an adapter (the Messages API native path); never raises."""
    plan = _plan_call(credential_source)
    if plan is not None:
        _record_response(SimpleNamespace(name=provider), plan, request, response, started)


def accounted_direct_stream(
    provider: str,
    request: dict[str, Any],
    stream: AsyncIterator[Any],
    started: float,
    *,
    credential_source: str | None = None,
) -> AsyncIterator[Any]:
    """Return ``stream`` wrapped to record once when it ends, or unchanged when nothing is recorded.

    For a provider stream made without an adapter. Each item must hold whole SSE events.
    """
    plan = _plan_call(credential_source)
    if plan is None:
        return stream
    return _accounted_async_stream(SimpleNamespace(name=provider), plan, request, stream, started)
```

`tldw_Server_API/app/core/Chat/chat_service.py`:
- Import: `from tldw_Server_API.app.core.Usage.llm_call_context import call_credential_source, llm_call_context`.
- `perform_chat_api_call`: `checked = check_llm_call_sync()` becomes `checked = check_llm_call_sync(call_credential_source(kwargs))`.
- `perform_chat_api_call_async`: `await check_llm_call_async()` becomes `await check_llm_call_async(call_credential_source(kwargs))`.
- `execute_streaming_call`:
  - After `llm_start_time = time.time()` (its first statement), add:

```python
    usage_credential_source = call_credential_source(cleaned_args)  # whose key the reply ran on, for the usage row
```

  - In `_queued_processor`, after its `nonlocal selected_provider, model, llm_call_func, stream_failure_recorded, structured_request_context` line, add `                nonlocal usage_credential_source`.
  - In its fallback branch, after `selected_provider = fallback_provider` (the one followed by `# Update run-first metric context to reflect fallback provider/model`), add `usage_credential_source = call_credential_source(refreshed_args)` at the same indent.
  - In the main-body fallback, after `cleaned_args = refreshed_args` (the one followed by `model = refreshed_model or model` and `def llm_call_func_fb(): return perform_chat_api_call(**cleaned_args)`), add `usage_credential_source = call_credential_source(cleaned_args)`.
  - In `save_callback`'s `log_llm_usage(...)`, after `estimate_source=None if reported else "stream_estimate",` (TASK-13951's line on `f8f913fab8`; `estimate_source="stream_estimate",` before it), add `credential_source=usage_credential_source,`.
- `_execute_non_stream_call_impl`, `_record_dict_response_usage`: in both `log_llm_usage(...)` calls, after `estimate_source=normalized_usage.estimate_source,` and after `estimate_source="missing_usage",`, add `credential_source=call_credential_source(cleaned_args),`. `cleaned_args` is the enclosing function's parameter, reassigned on failover (`cleaned_args = refreshed_args`), so the closure reads the active provider's handle.

- The three provider-fallback blocks (review I1). Each tries `fallback_provider` and then catches its failure with `provider_manager.record_failure(fallback_provider, ...)`. Before that generic handler, add a handler that re-raises a quota refusal:
  - in `_queued_processor`, before `except _CHAT_NONCRITICAL_EXCEPTIONS as fallback_error:` (the one followed by `provider_manager.record_failure(` and `raise_detached_error(`), and in the direct streaming fallback, before `except _CHAT_NONCRITICAL_EXCEPTIONS as fallback_error:` (the one followed by `provider_manager.record_failure(` and a bare `raise`):

```python
                    except LLMQuotaExceededError:
                        raise  # the user's limit, not the fallback provider's health (TASK-<ID>)
```

    (at the indentation of the `except` it precedes);
  - in `_execute_non_stream_call_impl`, before `except Exception as fallback_error:  # noqa: BLE001 - adapter trust boundary`, keeping the run-first completion metric that block emits:

```python
                    except LLMQuotaExceededError:
                        _emit_chat_run_first_completion_metric(
                            metrics,
                            context=run_first_metric_context,
                            outcome="error",
                        )
                        raise  # the user's limit, not the fallback provider's health (TASK-<ID>)
```

  `LLMQuotaExceededError` is already imported. The queued processors' own primary calls already re-raise it this way, and the primary paths' `except HTTPException` handlers never record a failure for it.

`tldw_Server_API/app/api/v1/endpoints/messages.py` (native path recorders):
- In `accounted_direct_stream(...)`, after the `started,` argument, add `credential_source=getattr(credentials, "credential_source", None),`.
- Replace `record_direct_response(provider, payload, data, started)` with:

```python
            record_direct_response(
                provider, payload, data, started, credential_source=getattr(credentials, "credential_source", None)
            )
```

  `getattr` because the tests' runtime stand-ins return plain namespaces.

`tldw_Server_API/app/core/Prompt_Management/prompt_improvement_dispatch.py`:
- Add `from tldw_Server_API.app.core.Usage.llm_call_context import billed_credential_source, llm_call_context` after the `prompt_improvement` import block (ruff's isort places it there), and `from tldw_Server_API.app.core.AuthNZ.byok_helpers import load_server_config_snapshot` before the `byok_runtime` import.
- Wrap the call:

```python
        try:
            # Whose key and whose endpoint decide whether the call counts toward the user's LLM quota. The adapter
            # takes a raw key here, so its base URL is the app_config's or else the environment's (in the snapshot).
            credential_source = billed_credential_source(
                provider, getattr(byok, "source", None), getattr(byok, "app_config", None), load_server_config_snapshot()
            )
            with llm_call_context(credential_source=credential_source):
                provider_response = await perform_chat_api_call_async(
                    api_endpoint=provider,
                    messages_payload=messages,
                    api_key=api_key,
                    model=model,
                    max_tokens=PROMPT_IMPROVEMENT_MAX_OUTPUT_TOKENS,
                    streaming=False,
                    tools=None,
                    user_identifier=str(user_id if user_id is not None else "prompt-improvement"),
                    app_config=getattr(byok, "app_config", None),
                    call_policy=PROMPT_IMPROVEMENT_CALL_POLICY,
                )
        except LLMQuotaExceededError:
```

`tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py`:
- Add `from tldw_Server_API.app.core.Usage.llm_call_context import billed_credential_source, llm_call_context` after the `RAG.rag_service.unified_pipeline` import block, and change `from tldw_Server_API.app.core.AuthNZ.byok_helpers import is_trusted_base_url_request` to `... import is_trusted_base_url_request, load_server_config_snapshot`.
- In `generate_grounded_answer`, wrap the call:

```python
        try:
            # Whose key and whose endpoint decide whether the call counts toward the user's LLM quota. The adapter
            # takes a raw key here, so its base URL is the app_config's or else the environment's (in the snapshot).
            credential_source = billed_credential_source(
                target.provider,
                getattr(credentials, "source", None),
                getattr(credentials, "app_config", None),
                load_server_config_snapshot(),
            )
            with llm_call_context(credential_source=credential_source):
                response = await perform_chat_api_call_async(
                    api_endpoint=target.provider,
                    model=target.model,
                    messages_payload=messages,
                    api_key=credentials.api_key,
                    app_config=credentials.app_config,
                    streaming=False,
                    temperature=0,
                    max_tokens=prompt_budget.max_output_tokens,
                    user_identifier=str(recipient_user_id),
                )
            answer, labels = _parse_grounded_response(response, budgeted)
```

  These two callers pass a raw key without `credentials_resolved`, so their adapter takes the `app_config` section's base URL, else the environment's (`OPENAI_API_BASE_URL` and the like), else the vendor's. `byok.app_config` is built from `config.txt` and holds no environment value, so the rule also reads `load_server_config_snapshot()`, which captures both (re-review I1). It costs about 2 ms a call (measured), on two low-volume paths.

- [ ] **Step 6: Run; see GREEN.**
  - The Task 2 command: `36 passed` (measured).
  - Then `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Usage tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py tldw_Server_API/tests/Sharing/test_shared_workspace_chat_generation.py tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py`. Expected: no failures. Measured with Tasks 1 and 2 applied, Task 3's appended tests deselected (11 cases): `1049 passed, 1 skipped`.
  - `$PY -m ruff check` on the changed files. Expected: only `UP035` in `usage_tracker.py:19`, which is pre-existing; the line is not touched.
- [ ] **Step 7: Bump the size ceilings.** In `tldw_Server_API/tests/lint/function_size_baseline.json`, the two `chat_service` entries (`{"lines": 1045, "params": 30, "reason": null}` and `{"lines": 1536, "params": 38, "reason": null}`) become:

```json
  "tldw_Server_API.app.core.Chat.chat_service:_execute_non_stream_call_impl": {"lines": 1054, "params": 30, "reason": "records the reply's credential_source, and a refused failover re-raises without marking the fallback provider unhealthy (TASK-<ID>)"},
  "tldw_Server_API.app.core.Chat.chat_service:execute_streaming_call": {"lines": 1545, "params": 38, "reason": "tracks the active key's credential_source across failover for the usage row, and a refused failover re-raises without marking the fallback provider unhealthy (TASK-<ID>)"},
```

  (Two separate lines of the file; one sits between them.) Then the ratchet test must pass.
- [ ] **Step 8: Commit** (message 2): `git add` the twelve files, then `git diff --cached --check`.

## Task 3: Let an over-limit user through on their own key at /chat/completions and Messages (AC #4)

**Files:** see "Files" (Task 3).

**Interfaces:**
- Consumes: `user_to_check` and `check_llm_call_async(credential_source)` (Task 2).
- Produces:
  - `users_own_key_serves(user_id: Any, provider: str) -> bool` in `API_Deps/usage_quota_deps.py`; `chat.py` imports it for the billing pre-check.
  - `enforce_llm_quota(user_id, estimated_tokens=1, *, provider: str | None = None)`.

**RED reason (measured with Tasks 1 and 2 applied, on `03fec47781`):** `15 failed`, all with `AttributeError: ... has no attribute 'users_own_key_serves'` (13 on `usage_quota_deps`, 2 on `chat` for the hosted-plan cases).
- The unit tests patch it or call it.
- The integration tests patch it to stand in for a stored key, so they fail before the request.

- [ ] **Step 1: Append to `tldw_Server_API/tests/Usage/test_llm_quota_decision.py`:**

```python
async def test_an_over_limit_user_whose_provider_runs_on_their_own_key_is_let_through_unmarked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With their own key for the provider, the boundary admits without the mark; without it, it refuses (TASK-<ID>)."""
    asked: list[tuple[object, str]] = []

    async def _own_key(user_id: object, provider: str) -> bool:
        """The user has their own key for openai only."""
        asked.append((user_id, provider))
        return provider == "openai"

    async def _enforce(provider: str | None) -> bool:
        await usage_quota_deps.enforce_llm_quota(7, provider=provider)
        return current_llm_call_context().prechecked

    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    _set_limits(monkeypatch, tokens=0)
    assert await asyncio.create_task(_enforce("openai")) is False
    for provider in ("anthropic", None):
        with pytest.raises(LLMQuotaExceededError):
            await usage_quota_deps.enforce_llm_quota(7, provider=provider)
    _set_limits(monkeypatch)
    assert await asyncio.create_task(_enforce("openai")) is True
    assert asked == [(7, "openai"), (7, "anthropic")]


@pytest.mark.parametrize(
    ("requested", "byok_on", "allowlisted", "row", "base_url", "expected"),
    [
        ("OpenAI", True, True, {"id": 1}, None, True),
        ("OpenAI", True, True, None, None, False),
        ("OpenAI", False, True, {"id": 1}, None, False),
        ("OpenAI", True, False, {"id": 1}, None, False),
        ("OpenAI", True, True, RuntimeError("store down"), None, False),
        ("custom-openai-api", True, True, {"id": 1}, None, False),
        ("ollama", True, True, {"id": 1}, None, False),
        ("OpenAI", True, True, {"id": 1}, "https://api.openai.com/v1", True),
        ("OpenAI", True, True, {"id": 1}, "http://operator-gpu.internal:8000/v1", False),
    ],
    ids=[
        "own-key", "no-key", "byok-off", "not-allowlisted", "lookup-fails", "operator-endpoint", "local-server",
        "vendor-base-url", "operator-base-url",
    ],
)
async def test_users_own_key_serves_mirrors_the_first_test_of_credential_resolution(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    byok_on: bool,
    allowlisted: bool,
    row: object,
    base_url: str | None,
    expected: bool,
) -> None:
    """A billed provider on its vendor's host, BYOK on, the provider allowlisted and an active key; else False."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo

    async def _pool() -> object:
        return object()

    async def _fetch(_self: object, user_id: int, provider: str, **_kwargs: object) -> object:
        assert (user_id, provider) == (7, requested.lower())
        if isinstance(row, Exception):
            raise row
        return row

    snapshot = {"openai_api": {"api_base_url": base_url}} if base_url else {}
    monkeypatch.setattr(byok_helpers, "is_byok_enabled", lambda: byok_on)
    monkeypatch.setattr(byok_helpers, "is_provider_allowlisted", lambda _provider: allowlisted)
    monkeypatch.setattr(byok_helpers, "load_server_config_snapshot", lambda: snapshot)
    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.get_db_pool", _pool)
    monkeypatch.setattr(AuthnzUserProviderSecretsRepo, "fetch_secret_for_active_user", _fetch)
    assert await usage_quota_deps.users_own_key_serves("7", requested) is expected
```

- [ ] **Step 2: Append to `tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py`:**

```python
def test_chat_completions_admits_an_over_limit_user_whose_provider_runs_on_their_own_key(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch
):
    """Past the monthly token allowance, a user with their own key for the provider still gets a reply (TASK-<ID>)."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    dispatched: list[str] = []

    async def _own_key(_uid, provider):
        """Stands in for the lookup answering True for the request's provider (the boundary is under test)."""
        return provider == "local-llm"

    def _dispatch(**kwargs):
        """Answer the call."""
        dispatched.append(str(kwargs.get("api_endpoint")))
        return {
            "id": "chatcmpl-own-key",
            "object": "chat.completion",
            "created": 0,
            "model": "local-llm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

    refuse_llm_work(monkeypatch)
    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _dispatch)
    monkeypatch.setitem(chat_ep.API_KEYS, "local-llm", "dummy-key")

    request_data = ChatCompletionRequest(
        model="local-llm",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")],
        api_provider="local-llm",
    )
    settings = get_settings()
    headers = {"X-CSRF-Token": client.csrf_token}
    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token
    else:
        headers["X-API-KEY"] = auth_token

    response = client.post("/api/v1/chat/completions", json=request_data.model_dump(), headers=headers)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["choices"][0]["message"]["content"] == "hi"
    assert dispatched == ["local-llm"]
```

  Then append the hosted-plan test (Decision 5), and change the module's `from fastapi import status` to `from fastapi import HTTPException, status`:

```python
class _PlanSpent:
    """The hosted billing plan's LLM token check, with the org's plan already spent."""

    def __init__(self, *_args, **_kwargs):
        """Any org, category and estimate."""

    async def __aenter__(self):
        """Refuse, as LimitEnforcer does past the plan's llm_tokens_month."""
        raise HTTPException(status_code=402, detail={"error": "limit_exceeded", "category": "llm_tokens_month"})


def _on_a_spent_hosted_plan(client, monkeypatch) -> None:
    """Billing checks on for org 7, whose plan refuses every LLM request (the hosted product)."""
    monkeypatch.setattr(chat_ep, "enforcement_enabled", lambda: True)
    monkeypatch.setattr(chat_ep, "LimitEnforcer", _PlanSpent)
    monkeypatch.setitem(client.app.dependency_overrides, chat_ep.get_billing_org_id, lambda: 7)


@pytest.mark.parametrize("own_key", [True, False], ids=["own-key", "server-key"])
def test_the_hosted_plan_does_not_check_a_request_on_the_users_own_key(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch, own_key
):
    """A spent org plan refuses a server-key request but not one whose provider runs on the user's own key (TASK-<ID>)."""
    dispatched: list[str] = []

    async def _own_key(_uid, provider):
        """Stands in for the lookup: the request's provider runs on the user's own key, or not."""
        return own_key and provider == "local-llm"

    def _dispatch(**kwargs):
        """Answer the call."""
        dispatched.append(str(kwargs.get("api_endpoint")))
        return {
            "id": "chatcmpl-plan",
            "object": "chat.completion",
            "created": 0,
            "model": "local-llm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

    _on_a_spent_hosted_plan(client, monkeypatch)
    monkeypatch.setattr(chat_ep, "users_own_key_serves", _own_key)
    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _dispatch)
    monkeypatch.setitem(chat_ep.API_KEYS, "local-llm", "dummy-key")

    request_data = ChatCompletionRequest(
        model="local-llm",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")],
        api_provider="local-llm",
    )
    settings = get_settings()
    headers = {"X-CSRF-Token": client.csrf_token}
    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token
    else:
        headers["X-API-KEY"] = auth_token

    response = client.post("/api/v1/chat/completions", json=request_data.model_dump(), headers=headers)

    if own_key:
        assert response.status_code == status.HTTP_200_OK, response.text
        assert dispatched == ["local-llm"]
    else:
        assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
        assert response.json()["detail"]["category"] == "llm_tokens_month"
        assert dispatched == []
```

  `local-llm` stands in for any provider: the tests stub `users_own_key_serves`, which in production never answers True for a keyless provider (Decision 7). The rule itself is pinned by Step 1's `operator-endpoint`, `local-server` and `operator-base-url` cases. `refuse_llm_work` (an existing helper) sets the monthly token limit to 0. Each test makes three mock calls of its own (two `setattr` and one `setitem`; the hosted-plan test's other three are in `_on_a_spent_hosted_plan`), under the triage's threshold of four, so the file's `mock_density` stays at its baseline of 2. Two more direct `setattr` calls in place of the helper would raise it to 3 and fail `test_quality_triage.py --enforce` (measured).

- [ ] **Step 3: In `tldw_Server_API/tests/Chat/integration/test_messages_endpoint.py`, insert before `def test_messages_count_tokens_unsupported_provider` (and its `@pytest.mark.integration`):**

```python
@pytest.mark.integration
def test_messages_admits_an_over_quota_user_whose_provider_runs_on_their_own_key(client_user_only, monkeypatch):
    """Over the limit, a request whose provider runs on the user's own key still reaches the provider (TASK-<ID>)."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[str] = []

    async def _adapter_call(**kwargs):
        calls.append("adapter")
        return _build_openai_response("hello")

    async def _own_key(_uid, provider):
        return provider == "openai"

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter_call)
    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    refuse_llm_work(monkeypatch)
    payload = {"model": "openai/gpt-4", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    response = client_user_only.post("/api/v1/messages", json=payload)
    assert response.status_code == 200, response.text
    assert calls == ["adapter"]
```

- [ ] **Step 4: Append to `tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py`:**

```python
async def test_a_native_call_let_through_at_the_boundary_is_still_checked_when_not_on_the_users_key(
    monkeypatch,
) -> None:
    """The boundary admits an over-limit user on their own key; a native call on another key is refused (TASK-<ID>)."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_calls

    calls: list[str] = []

    async def _own_key(_uid: Any, _provider: str) -> bool:
        """The boundary believes the provider runs on the user's own key."""
        return True

    async def _post(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """The provider call, which must not happen."""
        calls.append("native")
        return NATIVE_RESPONSE

    _stub_endpoint(monkeypatch)
    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _post)
    refuse_llm_calls(monkeypatch)
    with scoped_context(user_id=5), pytest.raises(LLMQuotaExceededError) as refused:
        await messages_endpoint._handle_messages(
            AnthropicMessagesRequest(model="anthropic/claude-3-5-haiku-latest", messages=[{"role": "user", "content": "hi"}]),
            current_user=SimpleNamespace(id=5, id_int=5),
            request=SimpleNamespace(),
            anthropic_version=None,
            anthropic_beta=None,
        )
    assert refused.value.detail["category"] == "llm_tokens_month"
    assert calls == []
```

  This is a unit test because the native path takes its user from the content scope. `client_user_only` does not set one, so in an integration test the native call would be neither recorded nor checked.

- [ ] **Step 5: Run; see RED.** The Task 3 command. Expected: `15 failed`.
- [ ] **Step 6: Implement.**

`tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py`:
- Add `from loguru import logger` after `from fastapi import Depends, HTTPException, status`.
- Add `as_quota_user_id,` first in the `quota_checks` import list.
- Replace `from tldw_Server_API.app.core.Usage.llm_call_context import mark_llm_quota_prechecked` with:

```python
from tldw_Server_API.app.core.Usage.llm_call_context import (
    USER_KEY_SOURCE,
    billed_credential_source,
    mark_llm_quota_prechecked,
)
```
- Replace `enforce_llm_quota` with:

```python
async def users_own_key_serves(user_id: Any, provider: str) -> bool:
    """True when a call to ``provider`` for this user would use the user's own stored key, billed to that key.

    The same first test credential resolution makes: BYOK on, the provider
    allowlisted, and an active key. A provider whose endpoint the operator runs
    (a custom OpenAI-compatible slot, a local server, a billed provider whose
    base URL is not its vendor's) answers False, as does a failed lookup.
    """
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import (
        is_byok_enabled,
        is_provider_allowlisted,
        load_server_config_snapshot,
    )
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
    from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name

    uid = as_quota_user_id(user_id)
    try:
        name = canonical_provider_name(provider)
        if uid is None or billed_credential_source(name, USER_KEY_SOURCE) != USER_KEY_SOURCE:
            return False
        if not is_byok_enabled() or not is_provider_allowlisted(name):
            return False
        repo = AuthnzUserProviderSecretsRepo(await get_db_pool())
        if await repo.fetch_secret_for_active_user(uid, name) is None:
            return False
        # Last, as the dearest test: the endpoint the server's config gives this provider.
        snapshot = load_server_config_snapshot()
        return billed_credential_source(name, USER_KEY_SOURCE, snapshot) == USER_KEY_SOURCE
    except Exception as error:  # noqa: BLE001 - a failed lookup refuses as before
        logger.bind(exception_type=type(error).__name__).warning("Own-key lookup for the LLM quota failed")
        return False


async def enforce_llm_quota(user_id: Any, estimated_tokens: int = 1, *, provider: str | None = None) -> None:
    """402 limit_exceeded when the user is over an LLM quota; otherwise mark the request admitted (spec §4a).

    An over-limit user whose ``provider`` runs on their own key is let through
    unmarked: each call is then checked where its key is known, and a call on
    the user's own key is not checked (TASK-<ID>).
    """
    refusal = await llm_quota_refusal(user_id, estimated_tokens)
    if refusal is None:
        mark_llm_quota_prechecked()
        return
    if provider and await users_own_key_serves(user_id, provider):
        return
    raise LLMQuotaExceededError(**asdict(refusal))
```

  Why not call `resolve_byok_credentials(required_source="user")`:
  - it decrypts the key;
  - for OpenAI OAuth payloads it may refresh the token over the network;
  - it counts a `byok_resolution_total` metric.

  The repository read is enough, because the call itself still resolves and is checked at the call. `fetch_secret_for_active_user` (`user_provider_secrets_repo.py:310-331`) returns None for a revoked key or an inactive user. The endpoint test comes last because it reads the server config (`load_server_config_snapshot()`, about 2 ms measured): it runs only for a user who has a key for a billed, allowlisted provider. That snapshot is the one the request's credential runtime captures, so the boundary and the call agree. A base URL a platform admin stored with the key is not in it; such a call is let through unmarked and then checked at the call on its handle, which does see it.

`tldw_Server_API/app/api/v1/endpoints/chat.py`:
- Change `from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota` to `... import enforce_llm_quota, users_own_key_serves`.
- The billing pre-check's comment and condition (they open the block at `chat.py:4398` on `f8f913fab8`) become (Decision 5):

```python
            # Billing: LLM token enforcement via context manager (tracks actual usage). A request whose
            # provider runs on the user's own key is neither checked nor counted as plan usage (ADR-083).
            if (
                enforcement_enabled()
                and billing_org_id is not None
                and not _billing_enforcer_entered
                and not await users_own_key_serves(getattr(current_user, "id", None), selected_provider)
            ):
```

  The block's body is unchanged. The lookup runs only when the first three conditions hold, so OSS installs never make it: `get_billing_org_id` returns None unless `billing_checks_active()`. In the hosted product it runs on every `/chat/completions` request: one indexed read, plus one server-config read (about 2 ms) when the user has a key for the provider; none when BYOK is off, the provider is not allowlisted, or it does not bill per key. For a request whose provider runs on the user's own key, `_billing_enforcer` stays None: no plan check, no hot-cache delta, no cost-units ledger write. A failover to a server key in that request is not checked against the plan. Its usage row counts in the plan's sum when the cache refreshes, within `BILLING_CACHE_TTL_SECONDS` (60 s).
- Replace the `enforce_llm_quota` call that follows (`await enforce_llm_quota(getattr(current_user, "id", None), max(1, _estimated_request_tokens))`, `chat.py:4415` on `f8f913fab8`) with:

```python
            await enforce_llm_quota(
                getattr(current_user, "id", None), max(1, _estimated_request_tokens), provider=selected_provider
            )
```

- In `_llm_call_with_openai_oauth_retry`, before `refreshed_response = perform_chat_api_call(**refreshed_args)`, add (review M9):

```python
                                # Checked at the call on its own handle. The usage row keeps the first handle's
                                # source, which differs only if the user's OAuth key was deleted mid-request (TASK-<ID>).
```

`tldw_Server_API/app/api/v1/endpoints/messages.py`:
- Import `enforce_llm_quota` instead of `require_llm_quota` from `usage_quota_deps`. Add `from tldw_Server_API.app.core.Usage.llm_quota_gate import check_llm_call_async` after the `from tldw_Server_API.app.core.testing import ...` line.
- In both route decorators (`/messages` and `/v1/messages`), `dependencies=[Depends(check_rate_limit), Depends(require_llm_quota)]` becomes `dependencies=[Depends(check_rate_limit)]`.
- In `_handle_messages`, after `provider, model = _resolve_provider_and_model_for_request(request_data)`:

```python
    # Usage quotas (LLM usage accounting spec §4a), before any LLM work. Inline, not a dependency, to name the provider.
    await enforce_llm_quota(getattr(current_user, "id", None), provider=provider)
```

- In the native branch, after `stream = _extract_stream_flag(request_data.stream)`:

```python
            # No ChatProvider on this path, so it gets the adapters' check at the call (a no-op once admitted).
            await check_llm_call_async(getattr(credentials, "credential_source", None))
```

  The mark set inline reaches the streaming body as it did from the dependency. Starlette runs the body in a task started after the handler set the mark, and `test_messages_refuses_an_over_quota_user_before_any_provider_call` covers both paths.

- [ ] **Step 7: Run; see GREEN.**
  - The Task 3 command: `15 passed` (measured).
  - Then the files the change touches: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_llm_quota_decision.py tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py tldw_Server_API/tests/Chat/integration/test_messages_endpoint.py tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py`. Measured `66 passed`.
- [ ] **Step 8: Bump the size ceiling.** In `tldw_Server_API/tests/lint/function_size_baseline.json`, the `create_chat_completion` entry (`{"lines": 2897, "params": 14, "reason": null}`) becomes:

```json
  "tldw_Server_API.app.api.v1.endpoints.chat:create_chat_completion": {"lines": 2907, "params": 14, "reason": "a request whose provider runs on the user's own key skips the hosted plan's pre-check and names its provider to enforce_llm_quota (BYOK quota exemption, owner decision 2026-10-26, TASK-<ID>)"},
```

  Then `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/lint/test_function_size_ratchet.py` must pass: measured with the whole plan and the four bumps, the file passes, and `tldw_Server_API/tests/lint` gives `248 passed` (dev's #3362 added 2 lint tests; the plan adds none).
- [ ] **Step 9: Commit** (message 3): `git add` the eight files, then `git diff --cached --check`.

## Task 4: Say what counts (AC #6)

Behaviour is final after Task 3.

- [ ] **Step 1: Create `Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md`:**

```markdown
# ADR-083: The LLM quotas neither count nor check a call on the user's own provider key

**Status:** Accepted
**Date:** 2026-10-26
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-<ID>
**Related spec/plan:** `Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md`
**Amends:** ADR-064 (what the per-user LLM limits count)

## Decision

A call made with the user's own stored key for a provider that bills that key, sent to that vendor's own host, is recorded in `llm_usage_log` with its real tokens and cost and `credential_source = 'user'`. `limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd`, `limits.llm_cost_per_month_usd` and the hosted billing plan's `llm_tokens_month` do not count it, and the per-user limits never refuse it. `/chat/completions` and the Messages API admit an over-limit user whose provider resolves to their own key.

## Context

The owner's decision of 2026-10-26: "BYOK users own keys shuold not count against their own costs". The LLM usage accounting design (`Docs/Design/2026-10-09-llm-usage-accounting-design.md`, owner decision 3) had gated BYOK calls "for now", because most call paths did not say which key they used. The final review of TASK-13936 found that a user's own-key calls still counted against that user's limits. When the vendor bills the user's key, the operator pays nothing for such a call. A key stored for an endpoint the operator runs is different: the operator's model answers whatever key is sent.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Record an own-key call at $0 | No new column, but the token limit still counts the call, the user and the operator lose its real cost, and a $0 row cannot be told from a free local call. |
| Exempt the cost limits only | The token limit would still refuse a user who works on their own key, which is the complaint being fixed. The token and cost limits budget the same thing, the operator's provider bill, in two units. |
| Do not record own-key calls | The user and the operator would lose the record, and ADR-067 records every call made for a user. |
| Treat every stored user key as the user's own | A key stored for a custom OpenAI-compatible slot or a local server reaches the operator's own endpoint, so a junk key would switch the limits off on the operator's hardware. |
| Treat a user key on any provider that bills per key as the user's own, and tell operators to take such a provider off `BYOK_ALLOWED_PROVIDERS` when they point its base URL elsewhere | The operator can point any of those providers at its own model server (`config.txt` or an environment variable). One missed provider would give every BYOK user unlimited use of the operator's hardware. |
| At every request boundary, let through any over-limit user who has an own key for some provider | The refusal would move from the boundary to the call on about 30 endpoints, and the "checked once" guarantees of benchmark runs, evaluation batches and media jobs would no longer hold for those users. |

## Consequences

- **The column.** `llm_usage_log.credential_source` holds `user`, `user_unmetered`, `team`, `org`, `server_default` or `none`. It is NULL when the recorder cannot tell, and on rows written before the column existed. SQLite gets it from an AuthNZ migration, PostgreSQL from the usage-table bootstrap. A bootstrap statement that fails with one of its handled errors (a timeout or a lost connection) is now logged as a warning, not at debug; a database error such as a missing privilege stops startup, as before. On a schema without the column a row is kept without its source, every reader counts every row, and the server logs a warning once.
- **Whose key is the user's own.** Only a key for one of the providers that bill per key, and only while that provider reaches its vendor: every base URL the configuration gives it must be on the host its adapter calls by default. That covers `config.txt` (for example `qwen_api_base_url`, `moonshot_api_base_url`, `zai_api_base_url`, `huggingface_api_base_url`), environment variables such as `OPENAI_API_BASE_URL`, and a base URL a platform admin stored with a key. A key stored for a custom OpenAI-compatible slot, a local provider, a provider the server does not know, or a billed provider pointed at a self-hosted model server or a gateway is recorded as `user_unmetered` and counts.
- **What counts.** The three limits, the profile's `llm` usage figures and the hosted plan's monthly LLM tokens count every row except `user`. So org and team shared keys, server keys, operator endpoints, keyless local providers and unknown rows still count. On PostgreSQL the hosted plan's sum counted nothing before this change: it bound a timezone-aware month start against a `TIMESTAMP` column, and the driver refused it. It now counts.
- **Where the source comes from.** A runtime-issued credential handle carries its source. A caller that resolves the key itself and passes a raw API key marks the call context. A call with neither counts, as the auto-router's call does today.
- **At the call.** A call on the user's own key is not checked, whoever started it: a job, a workflow, a WebSocket or an endpoint. A refused call on a fallback provider is the user's limit, not the provider's failure, so it does not mark the provider unhealthy.
- **At the boundary.** `/chat/completions` and the Messages API know the provider before any LLM work. Suppose the user is over a limit, BYOK is on, and the provider bills per key, reaches its vendor, is allowlisted and has an active key of the user's. Then the request is let through without the admitted mark, so each of its calls is checked where its key is known. `/chat/completions` also skips the hosted plan's check for such a request. The other boundary endpoints and a media ingest job's start still refuse an over-limit user before the key is known.
- **Other limits are unchanged.** The Resource Governor's token caps, virtual-key budgets, `limits.evaluation_tokens_per_day` and the `llm_cost_dollars` metric still count own-key calls.

## Follow-up

TASK-<ID>'s final summary lists the follow-up tasks: the auto-router's usage row and the other boundary endpoints, and whether org and team shared keys should count.
```

- [ ] **Step 2: `Docs/ADR/README.md`.**
  - In the ADR-064 row, replace `Amended in part by [ADR-076](076-per-user-concurrency-quotas.md): synchronous concurrency is now a usage quota. |` with `Amended in part by [ADR-076](076-per-user-concurrency-quotas.md): synchronous concurrency is now a usage quota; and by [ADR-083](083-llm-quotas-skip-calls-on-the-users-own-key.md): the LLM limits do not count a call on the user's own provider key. |`.
  - After the ADR-081 row, add the row below. If #3323 has merged by then, its ADR-082 row sits between them; put this row after it.

```markdown
| [ADR-083](083-llm-quotas-skip-calls-on-the-users-own-key.md) | Accepted | A call on the user's own key for a provider that bills that key is recorded with `credential_source = 'user'` and its real tokens and cost, but the per-user LLM limits and the hosted plan's LLM tokens neither count nor check it; `/chat/completions` and the Messages API admit an over-limit user whose provider resolves to their own key. |
```

- [ ] **Step 3: `Docs/Operations/Usage_Quotas.md`.**
  - After the bullet `- Provider key validation and first-run chat verification are never counted.`, insert:

```markdown
- **A call on the user's own key is not counted or checked** (ADR-083). A call made with the user's own stored provider key (BYOK, `Docs/User_Guides/Server/BYOK_User_Guide.md`) is recorded with its real tokens and cost and `credential_source = user` in `llm_usage_log`. None of the three keys counts it, and it is never refused at the call. That holds only for a provider that bills the user's key, sent to that vendor. A key stored for a custom OpenAI-compatible endpoint or a local server (llama.cpp, Ollama, vLLM and the like) reaches the endpoint you configured, so it is recorded as `user_unmetered` and counts. So does a key for a billed provider (OpenAI, Anthropic and the rest) whose base URL you point anywhere but the vendor's own host, at a self-hosted model server or a gateway: in `config.txt` (for example `qwen_api_base_url`, `moonshot_api_base_url`, `zai_api_base_url`, `huggingface_api_base_url`) or in an environment variable such as `OPENAI_API_BASE_URL`. Every other call counts: org and team shared keys (`team`, `org`), server keys (`server_default`), keyless local providers (`none`), and rows whose key is unknown (`NULL`): rows written before the column existed, and the auto-router's call. If the column is missing (AuthNZ migration 104 did not run, or the PostgreSQL bootstrap could not add it), every row counts and the server logs a warning. The hosted billing plan's `llm_tokens_month` leaves own-key calls out too. On PostgreSQL that plan limit counted no LLM tokens at all before this release (a timestamp bug), so a hosted plan on PostgreSQL now starts refusing at its `llm_tokens_month`. The other limits still count own-key calls: the Resource Governor's token caps, virtual-key budgets and `limits.evaluation_tokens_per_day`.
```

  - Before the bullet that starts `- **How the cost keys compare.**`, insert:

```markdown
- **An over-limit user on their own key.** `/chat/completions` and the Messages API (`/messages`, `/v1/messages`) let an over-limit user through when four things hold: BYOK is on, the request's provider bills per key and its base URL is the vendor's, the provider is on the BYOK allowlist, and the user has an active key for it. The request is not marked admitted, so each of its calls is checked where its key is known. A call on the user's own key runs. A call on any other key, a fallback provider's for example, is refused at the call. On `/chat/completions` the hosted billing plan's check is skipped for such a request. Every other boundary endpoint refuses an over-limit user before the key is known, even when the call would have used the user's own key: character chat, RAG generation, evaluations and benchmarks, flashcards and quizzes, chat document generation, watchlists and media analysis. So does a media ingest job's check when it starts.
```

  - At the end of the bullet `- **The profile reports it.** ...` (it ends "and each limit that is set."), append: ` The figures are the ones the limits count, so they leave out calls on the user's own key.`

- [ ] **Step 4: `Docs/User_Guides/Server/BYOK_User_Guide.md`.** Insert before `## Credential Fields`:

```markdown
## Usage Quotas

A call made with your own key is recorded in your usage, with its tokens and cost. It does not count toward the operator's per-user LLM limits (`limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd`, `limits.llm_cost_per_month_usd`) or a hosted plan's monthly LLM tokens, and those limits never refuse it. That applies to providers that bill your key, such as OpenAI or Anthropic, when the server sends the call to that vendor. A key you store for a custom OpenAI-compatible endpoint reaches the operator's server, so those calls count. So do calls on a provider the operator has pointed at its own server instead of the vendor, and calls on an org or team shared key or on a server key. If you are already over a limit, `/chat/completions` and the Messages API still answer a request whose provider uses your own key. Other endpoints, such as character chat and RAG, still refuse it for now. See `Docs/Operations/Usage_Quotas.md` and ADR-083.
```

- [ ] **Step 5: `Docs/Design/2026-10-09-llm-usage-accounting-design.md`.** At the end of owner decision 3 (it ends "That is a follow-up."), append: ` *Superseded on 2026-10-26 by ADR-083: a call on the user's own key is recorded but neither counted nor checked.*`
- [ ] **Step 6: Refresh the mirror:** `bash Helper_Scripts/refresh_docs_published.sh`. Measured: it changes exactly three files.
  - `Docs/Published/ADR/README.md` (modified);
  - `Docs/Published/User_Guides/Server/BYOK_User_Guide.md` (modified);
  - `Docs/Published/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md` (new).
- [ ] **Step 7: Stage, then run the docs tests.**
  - `git add Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md Docs/ADR/README.md Docs/Operations/Usage_Quotas.md Docs/User_Guides/Server/BYOK_User_Guide.md Docs/Design/2026-10-09-llm-usage-accounting-design.md Docs/Published/ADR/README.md Docs/Published/User_Guides/Server/BYOK_User_Guide.md Docs/Published/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md`. Stage first: `test_docs_published_refresh.py` compares the refresh output with `git ls-files`, so it fails while the new mirror file is untracked (review M1: `1 failed, 211 passed`).
  - `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs`. Expected: `212 passed` (measured; before staging, `1 failed, 211 passed`).
- [ ] **Step 8: Commit** (message 4): run `git diff --cached --check` on the eight staged files.

## Task 5: Verify, close TASK-<ID>, publish the PR

- [ ] **Step 1: Verify.** Run each of these:
  - The wider set, both parts (see "Test commands"), and compare with the Task 0 baseline. Expected (see "Test commands"): the base's failures only, `2 failed, 5883 passed, 6 skipped`, and PostgreSQL `22 passed`. Any other failure is this plan's. If `AuthNZ/integration/test_audio_quota_dates_postgres.py` errors under load, re-run it alone and compare with base before dismissing it.
  - `tldw_Server_API/tests/Docs`: `212 passed`. `tldw_Server_API/tests/lint`: `248 passed`.
  - `$PY Helper_Scripts/ci/check_shard_coverage.py`. Expected `new_uncovered=0` (measured).
  - `$PY Helper_Scripts/ci/test_quality_triage.py --enforce`. Expected exit 0, "OK — no offenses beyond the baseline" (measured).
  - `PYTHONPATH=. $PY Helper_Scripts/ci/route_auth_ratchet.py`. Expected "208 unauthenticated routes; baseline exact." (measured; the Messages routes keep `get_request_user`).
  - `PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts`. Expected exit 0 and both "OK" lines (measured).
  - `git grep -n -E "usage_repo.py|usage_tracker.py|migrations.py|pg_migrations_extra.py|provider_credential_runtime.py|chat_service.py|usage_accounting.py|prompt_improvement_dispatch.py|shared_workspace_chat_service.py|llm_call_context.py|llm_quota_gate.py|usage_quota_deps.py|endpoints/chat.py|endpoints/messages.py|Billing/enforcement.py" Helper_Scripts/ci/scope_predicate_baseline.txt`. Expected: nothing. If a file is listed and its lines shifted, regenerate with `--write-baseline`.
  - `git diff origin/dev --check`. Expected: nothing.
  - `git grep -n "TASK-<ID>" -- ':!Docs/superpowers/plans'`. Expected: nothing.
- [ ] **Step 2: Bandit**, on the touched source, through `uvx`:

```bash
/opt/homebrew/bin/uvx bandit -ll tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/api/v1/endpoints/messages.py tldw_Server_API/app/core/AuthNZ/migrations.py tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py tldw_Server_API/app/core/AuthNZ/provider_credential_runtime.py tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py tldw_Server_API/app/core/Billing/enforcement.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/Prompt_Management/prompt_improvement_dispatch.py tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py tldw_Server_API/app/core/Usage/llm_call_context.py tldw_Server_API/app/core/Usage/llm_quota_gate.py tldw_Server_API/app/core/Usage/usage_tracker.py
```

  Measured: "No issues identified", Medium 0, High 0. Bandit also prints "nosec encountered (B608), but no failed test" for the multi-line SQL strings in `usage_repo.py` and `Billing/enforcement.py`. Those are warnings, not findings: it reports such a string at its first line, and the `nosec` on the closing line still suppresses it. Without the `nosec` the two billing strings are findings (measured). Fix any new finding in changed lines.

- [ ] **Step 3: Close TASK-<ID>** (backlog-py, written out). The follow-ups were filed in Task 0, so their ids are known (review M3):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-<ID> \
  --append-notes "RED/GREEN (record the real counts): Task 1 SQLite 6 failed -> 6 passed, PostgreSQL 3 failed -> 3 passed; Task 2 31 failed, 5 passed -> 36 passed; Task 3 15 failed -> 15 passed. Wider set: SQLite part <before> -> <after>; PostgreSQL part <before> -> <after>. tests/Docs 212 passed; tests/lint <lint>. Shard coverage new_uncovered=0; test-quality ratchet OK; route-auth ratchet exact; OpenAPI fingerprint and path union unchanged." \
  --append-notes "Decisions (owner-approved plan): a credential_source column (the resolution's source; user only on a provider that bills per key whose configured base URLs are all on its vendor's host, else user_unmetered; NULL unknown) excluded from the three limits.llm_* sums and the hosted plan's llm_tokens_month, tokens included; the hosted plan's PostgreSQL sum binds a naive UTC month start, so it counts at all; a schema without the column counts every row and warns; own-key calls never checked at the call; a refused failover does not mark the fallback provider unhealthy; /chat/completions and the Messages API let an over-limit user through unmarked when the provider resolves to their own key (users_own_key_serves), and /chat/completions skips the hosted plan's pre-check for such a request; owner answers: scope <answer>, shared keys <answer>, other spend limits <answer>; RG token caps, virtual-key budgets and evaluation token limits unchanged. New ADR-083 amends ADR-064." \
  --append-notes "Known limits, filed as follow-ups: the other boundary endpoints and a media ingest job's start still refuse an over-limit user before the key is known (TASK-<F1>); the auto-router's usage row carries no source (<=64 output tokens a call) and chat_route_resolver's router check does not see the key (TASK-<F1>); llm_quota_refusals_total counts a boundary refusal that is then let through on the user's own key; a gateway reached under a vendor's own host name (a DNS override) still reads as the user's own key; a vendor host the adapter does not default to (api.moonshot.ai, api.cohere.com) counts until it is added to VENDOR_HOSTS." \
  --append-notes "Docs: Usage_Quotas.md (two bullets, profile sentence), BYOK_User_Guide.md (Usage Quotas section), ADR-083 and the ADR index, the design doc's owner decision 3 marked superseded; Docs/Published refreshed (README, BYOK guide, ADR-083). Bandit: <result>." \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A call made with the user's own stored key for a provider that bills that key is recorded in llm_usage_log with its real tokens and cost and credential_source 'user' (a new nullable column: AuthNZ SQLite migration 104 and the PostgreSQL usage-table DDL). The three per-user LLM limits and the hosted plan's llm_tokens_month leave those rows out; a key stored for a custom OpenAI-compatible slot, a local server or an unknown provider, or for a billed provider whose base URL is not its vendor's, is user_unmetered and counts; a schema without the column counts every row and warns. The hosted plan's PostgreSQL sum, which returned 0 (a timezone-aware bound against a TIMESTAMP column), now counts. The runtime credential handle carries its source, and callers that resolve the key themselves mark the call context; the adapter recorder, the shared call layer, /chat/completions' own recorders and the Messages native path record it, and a call on the user's own key is never checked at the call. /chat/completions and the Messages API let an over-limit user through when the provider resolves to their own key, unmarked, so any other key's call in that request is still refused at the call without blaming the fallback provider. Org and team shared keys, operator endpoints, unknown rows and server keys still count; RG token caps, virtual-key budgets and evaluation token limits are unchanged (ADR-083). The other quota boundaries still refuse: TASK-<F1>. Follow-ups: TASK-<F1><, TASK-<F2>>." \
  -s Done
```

  Fill in every `<...>` from Steps 1-2 and Task 0 before running. `--check-dod 4` is for Bandit. Then `git add -A backlog/tasks && git diff --cached --check` and commit (message 5).

- [ ] **Step 4: Publish the PR** (the draft from Task 0).
  - `git push`, then `gh pr edit <pr> --body-file <scratchpad>/pr_body_<ID>.md` and `gh pr ready <pr>`.
  - The body states:
    - the owner's decision, verbatim;
    - the column and both migrations;
    - what counts and what does not (shared keys, operator endpoints, unknown rows, the hosted plan, RG, virtual keys, evaluation tokens);
    - the two boundaries that admit an over-limit own-key user, and the ones that still refuse (TASK-<F1>);
    - the eight decisions and the owner's answers to Decisions 3-5.
  - The body ends with the attribution line from the session's reminder.
  - **ADR-004 Change summary.** The owner waived it for the BYOK quota-exemption PR (recorded 2026-10-26). Record the waiver as a full "## Change summary" section. Editing the PR body re-triggers the required license status, so wait for it before merging.
- [ ] **Step 5: Review and merge.**
  - Run `/superpowers:requesting-code-review` in a clean subagent (the owner's rule while Qodo credits are out). Hold until its findings are addressed.
  - Check `gh variable get MERGE_QUEUE`. With `on`, arm auto-merge and leave the PR alone. Otherwise rebase only when the PR is next, and merge after the seven statuses are green.
  - Confirm MERGED before deleting the branch.

## Follow-ups (filed in Task 0 Step 3)

- **F1:** the other quota boundaries (character chat first, then RAG, evaluations and benchmarks, flashcards and quizzes, chat documents, watchlists, media endpoints, media ingest job start), and the auto-router's usage row. If the owner wants evaluation tokens exempt (Decision 5), F1 also skips that reservation for own-key runs. New task TASK-<F1>.
- **F2 (only if the owner has not answered Decision 4):** whether org and team shared keys count. New task TASK-<F2>.

## ADR assessment

**A new ADR: ADR-083** (ADR-082 is held by open PR #3323).
- The change creates a durable rule about persistence (a column the limits read), quota semantics and provider integration (whose key a call used, and which providers bill it), which the ADR README requires an ADR for.
- It reverses an owner decision recorded in the design doc.
- It amends ADR-064 in part, as ADR-076 did: the README rule is "do not rewrite accepted ADR rationale", so ADR-064's file is not edited, and its index row gains the amendment note.

Governing ADRs:
- ADR-067: every call made for a user is recorded. This plan keeps recording own-key calls.
- ADR-075: the native Messages path uses the provider recorder's public wrappers, which gain a keyword.
- ADR-056: RG is a safety net, which is why its token caps are unchanged.
- ADR-059: backlog-py.

## Self-review against the spec

- **The owner's decision** ("should not count against their own costs"): Tasks 1-3. Tokens are included (Decision 2). The hosted plan is included and the other spend limits are an owner question (Decision 5). A key on an operator endpoint, including a billed provider the operator pointed at its own server, is not the user's cost (Decision 7).
- **AC #1** (recorded with its real tokens, cost and source, on SQLite and PostgreSQL; operator endpoints are `user_unmetered`): Task 1's recording test and the PostgreSQL test. Task 2 carries the source from every recorder: the adapter wrappers, `/chat/completions` stream and non-stream, the native Messages path, and the context callers. Task 2's handle, junk-key (five providers), vendor-host and table tests cover `user_unmetered`.
- **AC #2** (the limits and the hosted plan leave out `user`; other sources count; the hosted plan's PostgreSQL sum counts): Task 1's counter tests on both backends, the SQLite billing test and the PostgreSQL hosted-plan test.
- **AC #3** (not refused at the call on either call layer; server, shared and operator-endpoint keys still are; a refused failover blames no provider): Task 2's wrapper, call-layer (sync and async), context, guard, refused-call and refused-failover tests.
- **AC #4** (only the two boundaries; the hosted plan's pre-check): Task 3's unit and integration tests, plus the native check at the call. Decision 3 lists every boundary that still refuses.
- **AC #5** (an unmigrated database still records, and the limits still refuse; the PostgreSQL warning for a handled error): Task 1's dropped-column, warning (a timed-out `ALTER`) and PostgreSQL tests.
- **AC #6** (docs and mirror): Task 4.
- **The brief's questions:**
  - credential choice: "What was verified" 1;
  - every limit: section 2, with display-only readers listed apart;
  - bypassing callers: section 3;
  - record versus zero: Decision 1;
  - cost or tokens: Decision 2;
  - pre-call refusal and its scope: Decision 3, an owner question;
  - shared keys: Decision 4, an owner question;
  - other spend limits: Decision 5, an owner question;
  - which keys are the user's own: Decision 7.
