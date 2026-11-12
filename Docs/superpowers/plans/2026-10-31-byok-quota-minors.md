# BYOK quota exemption: the deferred review minors (TASK-<ID>) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the eight minor findings that the final review of PR #3371 (TASK-13955) deferred, so that ADR-083's rule ("a user key is the user's own spend only on its vendor's host; such a call is neither counted nor checked") holds on every path: a trusted caller's per-request `base_url`, the hosts the vendors document, a call nested inside an own-key stream, and the hosted plan's lookup. Plus two test gaps and one long docstring line.

**Architecture:** Each fix goes in the one shared function its callers route through:
- `Usage/llm_call_context.py`:
  - `billed_credential_source` gains a keyword `base_urls=`, and it now reads a provider with no config section as `user_unmetered` instead of raising `KeyError`.
  - `_on_vendor_host` lets a `*` match one DNS label only.
  - `VENDOR_HOSTS` gains the vendors' documented alternate hosts.
  - `call_credential_source` (the one reader used by the adapter wrappers, the gateway check and `/chat/completions`' own recorder) classifies the request's `base_url`/`api_base_url` too.
- `LLM_Calls/usage_accounting.py`: `_wrap_stream`/`_wrap_astream` use the same gate as `_wrap_chat`/`_wrap_achat`, and read every stream item inside `accounted_call()`, also when nothing is recorded.
- `usage_quota_deps.enforce_llm_quota(own_key=)`, plus one helper in `endpoints/chat.py`: `/chat/completions` asks `users_own_key_serves` once per request.
- Tests: one real-storage case for the boundary lookup, and an allowlist stub that honours its argument.
- Docs: an amendment to ADR-083, `Usage_Quotas.md`, and the `Docs/Published` mirror.

**Tech Stack:** Python, FastAPI, SQLite (AuthNZ pool), pytest, loguru.

**Spec:**
- ADR-083 (`Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md`);
- the executed plan `Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md`;
- the whole-branch review `final_review_13955.md` (planning session's plans dir), Minor findings 2-9;
- the owner's request of 2026-10-31: "address the deferred minor findings".

The backlog task TASK-<ID> (filed in Task 0) holds the ACs.

**Measured on:** `origin/dev` at `c251b6f978` (2026-10-30). It contains PR #3371's merge `4d0701bece`. Measured in a detached worktree. The prototype is `byok_minors_proto.patch` in the planning session's plans dir (code, tests, docs and the mirror; 13 files). Every RED/GREEN count below was measured with the text in this plan: first all tests on unchanged code, then each task's code applied in order. The worktree was reverted afterwards. `file:line` citations are for `c251b6f978`.

---

## What was verified (each finding on `c251b6f978`)

All eight findings are still true. None is dropped.

| # | Brief item | Still true? Evidence |
|---|---|---|
| 1 | Trusted per-request `base_url` is not classified | **Yes.** `gateway._resolve_base_url_override` (`LLM_Calls/gateway.py:705-738`) admits a `base_url`/`api_base_url` in the chat args, but only for a trusted caller (`is_trusted_base_url_request`: platform admin, service principal, single user) and an allowlisted provider. `_build_adapter_request_from_chat_args` then sets `request["base_url"]` (`:890-892`). Every billed adapter prefers it to `app_config`: `openai_adapter.py:245`, `anthropic_adapter.py:93`, `bedrock_adapter.py:282`, `deepseek_adapter.py:170`, `google_adapter.py:161`, `groq_adapter.py:45`, `mistral_adapter.py:89`, `openrouter_adapter.py:70`, `qwen_adapter.py:116`, `huggingface_adapter.py:146`; `cohere_adapter.py:500`, `moonshot_adapter.py:304` and `zai_adapter.py:220` forward it. `call_credential_source` (`Usage/llm_call_context.py:80-97`) returns `handle.credential_source` without reading it. That one function serves the adapter wrappers (`usage_accounting.py:354,378,406,431`), the gateway check (`gateway.py:1294,1345`) and `/chat/completions`' usage rows (`chat_service.py:4160,4397,4661,5802,5859`). The review's line (`chat_service.py:2446`) moved to `gateway.py` in TASK-13950.13. The public `/chat/completions` schema rejects `base_url` (`chat_request_schemas.py:88`, `:820`), and the Messages API takes its URL from `app_config` only (`messages.py:2034`), so neither boundary can see a per-request URL. The call layer is the place. |
| 2 | Hard `PROVIDER_APP_CONFIG_KEYS[name]` lookup | **Yes**, at `llm_call_context.py:120` (not in `provider_credential_runtime.py` as the brief says; that file calls it at `:877` inside `_new_handle`). Every `VENDOR_HOSTS` name has a section today (`byok_config.py:12-`). Measured: with a provider added to `VENDOR_HOSTS` only, `billed_credential_source("newvendor", "user", {})` raises `KeyError: 'newvendor'`. |
| 3 | Bedrock `*` spans dots | **Yes.** `fnmatchcase("bedrock-runtime.x.s3.amazonaws.com", "bedrock-runtime.*.amazonaws.com")` is true (`:134`). Measured: `billed_credential_source` answers `user` for that host. |
| 4 | Alternate vendor hosts missing; docs say they count | **Yes.** `VENDOR_HOSTS` (`:24-41`) has only the adapters' defaults. ADR-083 (`:33`) and `Usage_Quotas.md` (`:89`) say the alternates count "until added". The BYOK guide's "Usage Quotas" section (`BYOK_User_Guide.md:97`) names no host. It only says a provider pointed at the operator's server counts, which stays true, so it is not edited. |
| 5 | Own-key streams check a nested call | **Yes.** For an own-key call, `user_to_check` is None. With no plan (a `records_itself` context: `/chat/completions`), `_wrap_stream` (`usage_accounting.py:410`) and `_wrap_astream` (`:434`) return `fn()` unwrapped. A stream body runs only while it is read, outside any `accounted_call()`. So a nested call inside the stream body is checked and refused for an over-limit user. Measured with a delegating test provider: `LLMQuotaExceededError` on both layers. `_wrap_chat`/`_wrap_achat` (`:359-363`, `:383-387`) wrap such a call. No shipped adapter nests that way (the review checked). |
| 6 | Two `users_own_key_serves` lookups | **Yes.** `chat.py:4407-4411` asks once for the hosted plan's skip (when billing enforcement is on and an org is set). Then `enforce_llm_quota` (`usage_quota_deps.py:99`) asks again when the user is over a limit. Each lookup calls `load_server_config_snapshot()` (`:81`). Measured: `asked == ['local-llm', 'local-llm']`. |
| 7 | No real-storage test of `users_own_key_serves`; stub ignores its argument | **Yes.** `test_llm_quota_decision.py:241-269` stubs the repo, the pool and `is_provider_allowlisted` (`lambda _provider: allowlisted`, `:265`). Measured: with `users_own_key_serves` mutated to call `is_provider_allowlisted(provider)` (the raw name) instead of `(name)`, the existing test still passes (`9 passed`). |
| 8 | One docstring line over 120 characters | **Yes**: `AuthNZ/repos/usage_repo.py:564` (132 characters), from #3371 (`d6c76ca9d6e`). Lines 306, 748 and 799 in the same file are also over 120, but predate #3371 and are out of scope. |

### The vendor hosts: what was verified, and how

Each fact below was fetched with WebFetch on 2026-10-30, in the planning session. The tests make no network requests; they only compare strings.

- **OpenAI.** `https://platform.openai.com/docs/guides/your-data` redirects (301) to `https://developers.openai.com/api/docs/guides/your-data`. Its regional-domain table lists **ten** domains: `us.api.openai.com` (United States), `eu.api.openai.com` (Europe, EEA + Switzerland), `au.`, `ca.`, `jp.`, `in.`, `sg.`, `kr.`, `gb.` and `ae.api.openai.com`. The page says: "For requests to projects with data residency configured, add the domain prefix as defined in the table below to each request." The brief named two; see Decision 1.
- **Cohere.** `https://docs.cohere.com/reference/chat`: "POST https://api.cohere.com/v2/chat". The adapter still defaults to `https://api.cohere.ai` (`provider_model_inventory.py:172`), so both hosts stay listed.
- **Moonshot.** `https://platform.moonshot.ai/docs/guide/start-using-kimi-api` redirects (301) to `https://platform.kimi.ai/docs/guide/start-using-kimi-api`, which gives `base_url` `https://api.moonshot.ai/v1`. The adapter's default `api.moonshot.cn` stays listed.
- **Bedrock FIPS.** `https://docs.aws.amazon.com/general/latest/gr/bedrock.html`: `bedrock-runtime-fips.<region>.amazonaws.com` in us-east-1, us-east-2, us-west-2, ca-central-1, us-gov-east-1 and us-gov-west-1. One pattern, `bedrock-runtime-fips.*.amazonaws.com`, matches all six; after Finding 3 its `*` is one label. The adapter already parses this host (`bedrock_adapter.py:105`).

### A consequence of Finding 1 that the plan guards

`call_credential_source` now reads the adapter request's `base_url`. The gateway also writes the resolved discovery base there, for every provider with a model listing (`_validate_provider_model_selection`, `gateway.py:1163`). On a handle call, that base comes from the handle's own `app_config` (already classified in `_new_handle`) or from the adapter's default.

So a default off `VENDOR_HOSTS` would turn every own-key call on that provider into a checked, counted call. Measured with no config: every billed provider's default base is on its vendor's host (14 of 16), and Bedrock and Hugging Face have no default base without config. Task 1 pins this with a test.

## Decisions

1. **OpenAI data-residency hosts: all ten, or only `eu` and `us`?** The brief and the review named two; OpenAI documents ten (above).
   - **Recommendation: all ten** (this plan's default). They are OpenAI's own hosts, under `api.openai.com`, documented for the same API. With two, `Usage_Quotas.md` and the ADR amendment would still have to say that the other eight count.
   - If the owner says two: in Task 1, keep only `"api.openai.com", "us.api.openai.com", "eu.api.openai.com"` in `VENDOR_HOSTS["openai"]` and set `OPENAI_RESIDENCY_REGIONS = ("us", "eu")`. Task 1's counts then lose the eight removed cases (re-measure them). In Task 4, add to the amendment and to `Usage_Quotas.md`: "OpenAI's other data-residency hosts still count."
2. **ADR-004 Change summary.** This PR is AI-authored. The recorded waivers cover the TASK-13955 ("BYOK quota-exemption") PR, but not this follow-up by name.
   - **Recommendation:** ask the owner to extend the waiver to this PR, as for earlier follow-ups. Otherwise the owner writes the `## Change summary` before merge.
3. **Finding 2's direction** follows the brief: a billed provider with no config section reads as `user_unmetered` (counts). The review suggested `.get` (no section means no configured URL, so the call stays `user`). The brief's choice fails safe. Not an owner question; noted for the reviewer.

## Folded after review (2026-10-31)

`TASK-<ID>` is **TASK-13959**. The independent review and dry run (`review_plan_byok_minors.md`, planning session) found 0 Blocker, 0 Important and 6 Minor, and its dry run matched the prototype. The owner approved the plan on 2026-10-31 with these answers: Decision 1, the wildcard below (review M5) rather than ten literal hosts; Decision 2, "Yes, waiver covers it". Each fold overrides the task text it names:

- **M5 (Decision 1), Task 1:** `VENDOR_HOSTS["openai"] = ("api.openai.com", "*.api.openai.com")`. This matches `openai_adapter.py:364` and `provider_readiness.py:94` (`host.endswith(".api.openai.com")`). Keep `OPENAI_RESIDENCY_REGIONS` and its test: each documented region must still read as `user`. Any test that expects some other one-label `*.api.openai.com` host to count changes to expect `user`. A two-label host (`a.b.api.openai.com`) still counts. Task 4's docs say "OpenAI's data-residency hosts (`<region>.api.openai.com`)".
- **M1, Task 1 Step 3(c):** `call_credential_source`'s docstring line saying a request with a handle "is decided by the handle alone" becomes: "A request that carries a runtime-issued handle is decided by the handle and the request's own base URL (``base_url``, ``api_base_url``)."
- **M2, Task 1:** drop the adapter-level `api_base_url` case, since the adapter request never carries that key (`gateway.py:912`). Keep the adapter-level test with `base_url` only. Add the review's gateway-level test, `test_a_trusted_callers_base_url_off_the_vendors_host_is_checked_at_the_gateway`, parametrized over sync and async and over `base_url` and `api_base_url`. It goes through `perform_chat_api_call` and `perform_chat_api_call_async`, whose adapter then runs as already checked. Expected (re-measure): Task 1 Step 2 `20 failed, 15 passed`; focused set at the end `98 passed`.
- **M4, Task 2:** in `_accounted_sync_stream` and `_accounted_async_stream`, feed the chunk parser only when there is a plan (`if plan is not None: seen.feed(item)`), so own-key streams that record nothing do not parse every chunk.
- **M3, Review Focus 4 and the PR body:** a known limit. One label per `*` still matches the legacy S3 look-alikes (`bedrock-runtime.s3-website-us-east-1.amazonaws.com`, `bedrock-runtime.s3-external-1.amazonaws.com`) and IDNA labels. Only the operator or a trusted caller sets this URL, and none of these hosts can serve the Bedrock API.
- **M6:** record the real count at each step. The dates stay 2026-10-31, the day the PR opens.

---

## Global Constraints

Carried verbatim from the brief:

- Tests make no real outbound network requests; stub the provider adapter or its transport. A `/chat/completions` or Messages integration test with a commercial provider calls the real API unless the adapter is stubbed.
- Function-size ratchet `tldw_Server_API/tests/lint/function_size_baseline.json`: frozen functions (`_execute_non_stream_call_impl`, `execute_streaming_call`, `create_chat_completion`, others) must not grow; put new logic in helpers. Check every function you touch against the baseline.
- PostgreSQL tests only through repo fixtures (`tldw_postgres_test`); never FLUSHALL/SCRIPT FLUSH Redis.
- Backlog: file one task for this work with backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task create ...`), with an explicit id from `bash <plans dir>/task_next.sh` (read it; it allocates under `scratchpad/id_lock`), `--append-notes` never `--notes`, stage with `git add -A backlog/tasks`, never hand-edit task files.
- Branch from `origin/dev`; the PR targets `dev`. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- CI checks to run before PR (put them in the last task): `tests/Docs`, `tests/lint`, the shard-coverage check, `test_quality_triage --enforce`, route-auth ratchet, OpenAPI export check, Bandit on touched files, plus every test file that names a touched symbol (git grep). Find the exact commands in the repo (`Docs/Development/CI_REQUIRED_GATES.md`).
- `scope_predicate_baseline.txt` line-number baselines: if a touched file is listed, regenerate with `--write-baseline`.

Plan-specific:

- **`task_next.sh` does not take a lock.** It only prints the highest backlog id on `origin/dev` and in every open PR (read it: the plans dir's `task_next.sh`). Take the lock yourself: `mkdir <scratchpad>/id_lock`. If the directory exists, wait, and never break another session's lock. `<ID>` is the printed value + 1. Keep the lock until the PR's file list shows the new task file.
- **`TASK-<ID>`.** Write the real id wherever this plan's code, docs, branch and commit messages say `<ID>`. Before the last commit, `git grep -n "TASK-<ID>" -- ':!Docs/superpowers/plans'` must print nothing.
- **Python.** The worktree has no `.venv`. Use `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (written `$PY` below; shell variables do not survive between agent calls, so write it out).
- **TDD.** Write each task's tests first, run them, and see them fail for the stated reason. Record the real counts.
- **Git.** Never `--no-verify`, never force-push, never `git stash`. Stage named files; `git diff --cached --check` must print nothing before each commit.
- **Compatibility.** `billed_credential_source` and `enforce_llm_quota` gain only keyword parameters, with defaults that keep today's behaviour. The first's five other call sites and the second's 28 other call sites compile unchanged.
- **Bandit** only through `/opt/homebrew/bin/uvx bandit`. Never install into the shared venv.
- **The CI gate commands** are not in `CI_REQUIRED_GATES.md` (it names the six required statuses only). The commands in Task 4 are the ones `backend-required` runs, as used by the TASK-13955 plan, and each was measured here.

## Review Focus

1. **Finding 1 also classifies the frozen discovery base** (`gateway.py:1163`): see "A consequence of Finding 1". Check the new `test_every_billed_adapters_default_base_is_on_its_vendors_host`, and look for a provider whose discovery base could differ from its handle's `app_config`.
2. **Finding 5 is slightly wider than "own key".** The stream wrappers now use exactly `_wrap_chat`'s gate (`plan is None and user_id is None and source != "user"` returns unwrapped). So a stream that was checked at its start (`user_id` set) but records nothing (`records_itself`) also reads its items inside `accounted_call()`, and a nested call in it is not checked a second time. That is what the non-stream path already does (`usage_accounting.py:362`, "checked here or on the user's own key: a nested method is not checked again").
3. **`own_key` in `enforce_llm_quota`** is trusted as given. Its only passer is `/chat/completions`, which computes it with the same user, the same provider and the same function in the same request. `_billing_enforcer_entered` is always False at that point (it is set True only inside the block, after its init at `chat.py:3813`); the guard is kept unchanged.
4. **One label per `*`** is a dot-count comparison (`host.count(".") == pattern.count(".")`) beside `fnmatchcase`. An empty label (`bedrock-runtime..amazonaws.com`) would still match; no resolver accepts that host, and only the operator sets this URL.
5. **The ADR is amended, not superseded.** The decision is unchanged. The amendment records two corrections to "Whose key is the user's own", following ADR-025's and ADR-063's `## Amendment (date)` sections and ADR-025's README index wording.

## Files

| File | Task | Change |
|---|---|---|
| `tldw_Server_API/app/core/Usage/llm_call_context.py` | 1 | `VENDOR_HOSTS`, `call_credential_source`, `billed_credential_source`, `_on_vendor_host` |
| `tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py` | 1, 2 | 1 parametrize row; 6 new tests (Task 1), 1 (Task 2); 1 assertion added |
| `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` | 2 | `_record_stream`, `_accounted_sync_stream`, `_accounted_async_stream`, `_wrap_stream`, `_wrap_astream` |
| `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py` | 2 | `enforce_llm_quota(own_key=)` |
| `tldw_Server_API/app/api/v1/endpoints/chat.py` | 2 | new `_own_key_before_the_hosted_plan`; `create_chat_completion` (2908 -> 2905 lines, ceiling 2908) |
| `tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py` | 2 | 1 new test |
| `tldw_Server_API/tests/Usage/test_llm_quota_decision.py` | 3 | stub honours its argument; 1 new test (3 cases) |
| `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py` | 3 | wrap one docstring line |
| `Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md`, `Docs/ADR/README.md` | 4 | amendment section, index row |
| `Docs/Operations/Usage_Quotas.md` | 4 | one sentence rewritten in the own-key bullet |
| `Docs/Published/ADR/083-...md`, `Docs/Published/ADR/README.md` | 4 | mirror (generated) |
| `Docs/superpowers/plans/2026-10-31-byok-quota-minors.md`, `backlog/tasks/task-<ID> - ....md` | 0, 4 | this plan; the task |

Function sizes, measured with the ratchet's own counter: of the touched functions, only `endpoints.chat:create_chat_completion` is in the baseline (ceiling 2908, measured 2908 before, 2905 after). Everything else touched is under the 300-line threshold. No touched file is in `Helper_Scripts/ci/scope_predicate_baseline.txt` (checked).

## Test commands

- **Focused set** (the three test files this plan edits):
  `$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py tldw_Server_API/tests/Usage/test_llm_quota_decision.py tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py`
  Measured:
  - `70 passed` on `c251b6f978`;
  - `20 failed, 75 passed` with every new test on unchanged code;
  - `95 passed` at the end.
- **Wider set**: every test file that names a touched symbol, plus the adapter and gateway suites. The finding-1 change reaches every adapter call that carries a handle. Save as `<scratchpad>/wider_byok_minors.sh` and run `zsh <scratchpad>/wider_byok_minors.sh <label>` from the worktree root:

```zsh
#!/bin/zsh
S=<scratchpad>
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
L=${1:-run}
{ git grep --untracked -l -E "call_credential_source|billed_credential_source|VENDOR_HOSTS|_on_vendor_host|users_own_key_serves|enforce_llm_quota|require_llm_quota|usage_quota_deps|usage_accounting|instrument_provider_class|_accounted_sync_stream|_accounted_async_stream|_record_stream|llm_call_context|llm_quota_gate|user_to_check|refuse_llm_calls|refuse_llm_work|insert_llm_usage_log|log_llm_usage|ProviderCallCredentials|ProviderCredentialRuntime|PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY|_build_adapter_request_from_chat_args|base_url_override|LimitEnforcer|enforcement_enabled|get_billing_org_id" -- 'tldw_Server_API/tests/*.py' 'tldw_Server_API/app/*/tests/*.py' | grep '/test_[^/]*\.py$'
  find tldw_Server_API/tests/LLM_Adapters tldw_Server_API/tests/LLM_Calls -name 'test_*.py'
} | sort -u > $S/minors_all_$L.txt
grep -E '/AuthNZ_Postgres/|_pg\.py$|_postgres\.py$' $S/minors_all_$L.txt > $S/minors_pg_$L.txt
grep -vE '/AuthNZ_Postgres/|_pg\.py$|_postgres\.py$' $S/minors_all_$L.txt > $S/minors_sqlite_$L.txt
echo "files: sqlite $(wc -l < $S/minors_sqlite_$L.txt) pg $(wc -l < $S/minors_pg_$L.txt)"
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -n 8 --dist loadfile -rf $(cat $S/minors_sqlite_$L.txt) > $S/minors_sqlite_$L.out 2>&1
echo "sqlite: $(tail -1 $S/minors_sqlite_$L.out)"
grep '^FAILED\|^ERROR' $S/minors_sqlite_$L.out
$PY -m pytest -p no:cacheprovider -q -rf $(cat $S/minors_pg_$L.txt) > $S/minors_pg_$L.out 2>&1
echo "pg: $(tail -1 $S/minors_pg_$L.out)"
grep '^FAILED\|^ERROR' $S/minors_pg_$L.out
```

  The PostgreSQL part runs without `TLDW_TEST_NO_DOCKER`, so the repo fixture provisions `tldw_postgres_test`. Measured: 278 SQLite files and 2 PostgreSQL files.
  - **On `c251b6f978`:** SQLite `4 failed, 6862 passed, 17 skipped, 1 error`; PostgreSQL `6 passed`. All five are pre-existing, and the same ids fail after the change:
    - `Chat/integration/test_document_generation_endpoints.py::test_document_factory_starts_without_default_executor_and_drains_cancel`;
    - `prompt_studio/integration/test_api_endpoints.py::TestEvaluationEndpoints::test_list_evaluations` (401);
    - two tests in `MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py` (401);
    - the error: the teardown of `test_extractor_never_opens_files`.
  - **After this plan:** SQLite `4 failed, 6887 passed, 17 skipped, 1 error` (the same five ids; +25 = the new tests); PostgreSQL `6 passed`.

---

## Task 0: Branch, file the task, plan file, baseline

- [ ] **Step 1: Allocate the id under the lock.**
  - `mkdir <scratchpad>/id_lock` (see Global Constraints).
  - `git fetch origin dev --quiet && zsh <plans dir>/task_next.sh`. `<ID>` is the printed value + 1. Check that no `backlog/tasks/task-<ID> *` file exists locally.
  - Create the branch: `git switch -c fix/byok-quota-minors-<ID> origin/dev` (write the id literally).
- [ ] **Step 2: File the task** (write the id literally):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "BYOK quota exemption: the deferred review minors" \
  --id TASK-<ID> -l quotas,backend,byok \
  -d "Owner request 2026-10-31: 'address the deferred minor findings' of PR #3371's final review (TASK-13955, ADR-083). (1) A trusted caller's per-request base_url (admin, service principal, single user) reaches the billed adapters as request['base_url'] without classification, so a call to an operator host on the user's key reads as 'user'. (2) billed_credential_source does a hard PROVIDER_APP_CONFIG_KEYS[name] lookup that raises KeyError for a billed provider with no config section. (3) The Bedrock host pattern's * spans dots. (4) VENDOR_HOSTS lacks the vendors' documented alternate hosts (OpenAI's data-residency domains, api.cohere.com, api.moonshot.ai, Bedrock FIPS), and the docs say they count. (5) An own-key stream with no recording plan returns the provider's stream unwrapped, so a provider method nested in the stream body is checked and can refuse an over-limit user mid-stream. (6) A hosted over-limit own-key /chat/completions request runs users_own_key_serves twice. (7) No test drives users_own_key_serves against real key storage, and the allowlist stub ignores its argument. (8) usage_repo.insert_llm_usage_log's docstring has a line over 120 characters. Plan: Docs/superpowers/plans/2026-10-31-byok-quota-minors.md." \
  --ac "A per-request base_url or api_base_url off the vendor's hosts makes a user-key call user_unmetered, checked and counted; on the vendor's host it stays user, with tests" \
  --ac "A billed provider without a config section reads as user_unmetered instead of raising, and a test keeps every VENDOR_HOSTS provider in PROVIDER_APP_CONFIG_KEYS" \
  --ac "A * in a VENDOR_HOSTS pattern matches one DNS label, with a test" \
  --ac "VENDOR_HOSTS lists OpenAI's documented data-residency hosts, api.cohere.com, api.moonshot.ai and Bedrock's FIPS endpoints, with a test, and a test keeps every billed adapter's default base on its vendor's host" \
  --ac "A provider method nested in an own-key stream is not checked, sync and async, with a test" \
  --ac "A hosted over-limit own-key /chat/completions request looks up the user's key once, with a test" \
  --ac "users_own_key_serves is tested on real SQLite BYOK storage (stored, revoked, user deactivated), and the allowlist stub honours its argument" \
  --ac "The long docstring line is wrapped; ADR-083 gets an amendment, Usage_Quotas.md is corrected, and the Docs/Published mirror is refreshed"
```

- [ ] **Step 3:** Mark it started: `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-<ID> -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-10-31-byok-quota-minors.md. Owner answers: Decision 1 (OpenAI hosts) <answer>; Decision 2 (Change summary) <answer>."`
- [ ] **Step 4:** Copy this plan to `Docs/superpowers/plans/2026-10-31-byok-quota-minors.md` unchanged (it keeps `<ID>`). Then:
  - `git add Docs/superpowers/plans/2026-10-31-byok-quota-minors.md && git add -A backlog/tasks && git diff --cached --check` (it must print nothing);
  - commit (message 0).
- [ ] **Step 5: Baseline.** On the unchanged branch, run the focused set (expect `70 passed`) and the wider set with label `base` (expect the five pre-existing failures above). Record both. Any later failure is compared with this baseline. Never dismiss one as environment-only without it.

## Task 1: Classify every URL a billed call goes to (findings 1-4)

**Files:** `Usage/llm_call_context.py`; `tests/Usage/test_llm_quota_byok_calls.py`.

**Interfaces:**
- `billed_credential_source(provider, source, *configs, base_urls: Iterable[Any] = ()) -> str | None`. `base_urls` are URLs the call is sent to outside any config section. Its other callers pass none (`provider_credential_runtime.py:877`, `usage_quota_deps.py:73,82`, `prompt_improvement_dispatch.py:207`, `shared_workspace_chat_service.py:716`).
- `call_credential_source(request)` passes `request["base_url"]` and `request["api_base_url"]` for a runtime-issued handle.
- `VENDOR_HOSTS` gains 13 hosts.

- [ ] **Step 1: Write the tests** in `tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py`.

  (a) Below `REFUSAL = ...` (line 27):

```python
# OpenAI's data-residency domains, <region>.api.openai.com (developers.openai.com/api/docs/guides/your-data).
OPENAI_RESIDENCY_REGIONS = ("us", "eu", "au", "ca", "jp", "in", "sg", "kr", "gb", "ae")
```

  (b) In `test_a_user_key_is_the_users_own_spend_only_on_the_vendors_host`'s parametrize list, add this row after the `ec2-3-4-5-6` row:

```python
        ("bedrock", {"runtime_endpoint": "https://bedrock-runtime.x.s3.amazonaws.com"}, "user_unmetered"),
```

  Then add its id after `"bedrock-not-bedrock",`, so the `ids=` list reads:

```python
    ids=[
        "vendor-default", "vendor-url", "operator-url", "vendor-region", "bedrock-region", "bedrock-not-bedrock",
        "bedrock-label-span", "router-url", "unknown-provider", "suffix-lookalike", "prefix-lookalike", "userinfo",
        "case-and-port",
    ],
```

  (c) Replace `test_every_provider_that_bills_per_key_names_its_vendors_host` (lines 138-143) with this block. It holds two new tests before it, the test with one more assertion, and three new tests after it:

```python
@pytest.mark.parametrize(
    ("provider", "url"),
    [
        *[("openai", f"https://{region}.api.openai.com/v1") for region in OPENAI_RESIDENCY_REGIONS],
        ("cohere", "https://api.cohere.com/v2"),
        ("moonshot", "https://api.moonshot.ai/v1"),
        ("bedrock", "https://bedrock-runtime-fips.us-east-1.amazonaws.com"),
    ],
)
def test_a_host_the_vendor_documents_bills_the_users_key(provider: str, url: str) -> None:
    """OpenAI's data-residency hosts, Cohere's and Moonshot's current hosts and Bedrock's FIPS endpoints."""
    from tldw_Server_API.app.core.Usage.llm_call_context import billed_credential_source

    field = "runtime_endpoint" if provider == "bedrock" else "api_base_url"
    assert billed_credential_source(provider, "user", {f"{provider}_api": {field: url}}) == "user"


def test_every_billed_adapters_default_base_is_on_its_vendors_host() -> None:
    """With nothing configured, the base the gateway freezes into the request is on the vendor's host."""
    from tldw_Server_API.app.core.LLM_Calls.provider_model_inventory import resolve_provider_models_base_url
    from tldw_Server_API.app.core.Usage.llm_call_context import VENDOR_HOSTS, billed_credential_source

    # A request's base_url is classified too, so a default off VENDOR_HOSTS would count every own-key call.
    bases = {name: resolve_provider_models_base_url(name, {}, credentials_resolved=True) for name in VENDOR_HOSTS}
    off_host = {
        name: base for name, base in bases.items()
        if base and billed_credential_source(name, "user", {f"{name}_api": {"api_base_url": base}}) != "user"
    }
    assert off_host == {}
    assert {name for name, base in bases.items() if not base} == {"bedrock", "huggingface"}  # no base without config


def test_every_provider_that_bills_per_key_names_its_vendors_host() -> None:
    """A billed provider missing from the host table, or from the config sections, would misread its users' keys."""
    from tldw_Server_API.app.core.AuthNZ.byok_config import PROVIDER_APP_CONFIG_KEYS
    from tldw_Server_API.app.core.LLM_Calls.provider_metadata import PROVIDER_REQUIRES_KEY
    from tldw_Server_API.app.core.Usage.llm_call_context import VENDOR_HOSTS

    assert set(VENDOR_HOSTS) == {name for name, billed in PROVIDER_REQUIRES_KEY.items() if billed}
    assert set(VENDOR_HOSTS) <= set(PROVIDER_APP_CONFIG_KEYS)


def test_a_billed_provider_without_a_config_section_counts(monkeypatch) -> None:
    """A provider added to the host table but not to the config sections counts its users' keys instead of failing."""
    from tldw_Server_API.app.core.Usage import llm_call_context

    monkeypatch.setitem(llm_call_context.VENDOR_HOSTS, "newvendor", ("api.newvendor.example",))
    assert llm_call_context.billed_credential_source("newvendor", "user", {}) == "user_unmetered"


@pytest.mark.parametrize("field", ["base_url", "api_base_url"])
async def test_a_trusted_callers_base_url_off_the_vendors_host_is_checked(refused, field: str) -> None:
    """A per-request base URL (an admin, service or single-user caller's) wins in every adapter, so it is read too."""
    request = {
        **REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user"),
        field: "http://operator-gpu.internal:8000/v1",
    }
    with llm_call_context(user_id=7), pytest.raises(LLMQuotaExceededError):
        await _Provider().achat(request)


async def test_a_trusted_callers_base_url_on_the_vendors_host_is_not_checked(refused) -> None:
    """The same caller sending the call to the vendor's own host is on the user's own key."""
    request = {
        **REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user"),
        "base_url": "https://api.openai.com/v1",
    }
    with llm_call_context(user_id=7):
        await _Provider().achat(request)
    await llm_accounting.drain_accounting()
    assert [row["credential_source"] for row in refused] == ["user"]
```

  The `api_base_url` case stands for the chat args (`cleaned_args`, the gateway's `kwargs`), where a trusted caller's override arrives under either key (`gateway.py:706-708`).

- [ ] **Step 2: Run, and see it fail.**
  `$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py -k "vendors_host or documents or config_section or trusted_callers"`
  Expected: `17 failed, 15 passed` (measured; the deselected count, 28 when every task's tests exist, depends on which later tests are already written). The failures:
  - `bedrock-label-span`: `AssertionError: assert 'user' == 'user_unmetered'`;
  - the 13 `test_a_host_the_vendor_documents_bills_the_users_key` cases: `assert 'user_unmetered' == 'user'`;
  - `test_a_billed_provider_without_a_config_section_counts`: `KeyError: 'newvendor'`;
  - both `test_a_trusted_callers_base_url_off_the_vendors_host_is_checked` cases: `Failed: DID NOT RAISE LLMQuotaExceededError`.

  The 15 that pass guard what must not change: the 12 existing host cases, the two table tests and the on-vendor-host case.

- [ ] **Step 3: Implement** in `tldw_Server_API/app/core/Usage/llm_call_context.py`.

  (a) The import (line 5):

```python
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
```

  (b) The comment above `VENDOR_HOSTS` and four of its entries (lines 22-35). Every other entry is unchanged:

```python
# Each provider that bills per key, and the hosts its vendor documents (fnmatch patterns; ``*`` is one DNS label).
# A user key is the user's own spend only there (TASK-13955).
VENDOR_HOSTS: dict[str, tuple[str, ...]] = {
    "anthropic": ("api.anthropic.com",),
    "bedrock": ("bedrock-runtime.*.amazonaws.com", "bedrock-runtime-fips.*.amazonaws.com"),
    "cohere": ("api.cohere.ai", "api.cohere.com"),
    "deepseek": ("api.deepseek.com",),
    "google": ("generativelanguage.googleapis.com",),
    "groq": ("api.groq.com",),
    "huggingface": ("api-inference.huggingface.co", "router.huggingface.co"),
    "mistral": ("api.mistral.ai",),
    "moonshot": ("api.moonshot.cn", "api.moonshot.ai"),
    "novita": ("api.novita.ai",),
    "openai": (  # and its data-residency domains (developers.openai.com/api/docs/guides/your-data)
        "api.openai.com", "us.api.openai.com", "eu.api.openai.com", "au.api.openai.com", "ca.api.openai.com",
        "jp.api.openai.com", "in.api.openai.com", "sg.api.openai.com", "kr.api.openai.com", "gb.api.openai.com",
        "ae.api.openai.com",
    ),
```

  (c) In `call_credential_source`, replace `return handle.credential_source` (line 94) with:

```python
            # A trusted caller's per-request base URL wins over app_config in every adapter, so it is read too.
            urls = (request.get("base_url"), request.get("api_base_url"))
            return billed_credential_source(handle.provider, handle.credential_source, base_urls=urls)
```

  (d) Replace `billed_credential_source` and `_on_vendor_host` (lines 100-134) with:

```python
def billed_credential_source(
    provider: str | None, source: str | None, *configs: Any, base_urls: Iterable[Any] = ()
) -> str | None:
    """``source`` as the LLM quotas read it: a user key is the user's own spend only on its vendor's host.

    ``user`` stays ``user`` only for a provider in ``VENDOR_HOSTS`` whose base
    URLs in ``configs`` (a resolved ``app_config``, a server config snapshot) and
    in ``base_urls`` (a trusted caller's per-request override) are all on the
    vendor's host. Otherwise it becomes ``user_unmetered``, which counts: a custom
    OpenAI-compatible slot, a local server, an unknown provider, a billed provider
    with no config section, or a billed provider pointed at a model server or
    gateway. Every other source is returned unchanged, so nothing else ever
    becomes ``user``.
    """
    if source != USER_KEY_SOURCE:
        return source
    from tldw_Server_API.app.core.AuthNZ.byok_config import PROVIDER_APP_CONFIG_KEYS
    from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name

    name = canonical_provider_name(provider) if provider else ""
    hosts = VENDOR_HOSTS.get(name)
    section_key = PROVIDER_APP_CONFIG_KEYS.get(name)
    if hosts is None or section_key is None:
        return USER_KEY_UNMETERED_SOURCE
    urls = list(base_urls)
    for config in configs:
        section = config.get(section_key) if isinstance(config, Mapping) else None
        if isinstance(section, Mapping):
            urls.extend(section.get(key) for key in _BASE_URL_KEYS)
    if any(isinstance(url, str) and url.strip() and not _on_vendor_host(url, hosts) for url in urls):
        return USER_KEY_UNMETERED_SOURCE
    return source


def _on_vendor_host(url: str, hosts: tuple[str, ...]) -> bool:
    """Whether ``url``'s host matches one of the vendor's host patterns, a ``*`` matching one DNS label."""
    try:
        host = urlsplit(url.strip()).hostname or ""
    except ValueError:  # a malformed URL is not the vendor's
        return False
    return any(fnmatchcase(host, pattern) and host.count(".") == pattern.count(".") for pattern in hosts)
```

- [ ] **Step 4: Run, and see it pass.** The same command. Expected: `32 passed` (measured).
- [ ] **Step 5:** Run the gateway and adapter suites that build requests with a handle: `$PY -m pytest -p no:cacheprovider -q -n 8 tldw_Server_API/tests/LLM_Calls tldw_Server_API/tests/Usage tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py`. Expected: no failures (measured on the whole prototype: `1590 passed, 3 skipped`). A `libc++abi ... recursive_mutex lock failed` line may print as the xdist workers exit, after the summary; it is not a test result.
- [ ] **Step 6: Commit** (message 1): `git add tldw_Server_API/app/core/Usage/llm_call_context.py tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py && git diff --cached --check`.

## Task 2: Own-key streams, and one own-key lookup per request (findings 5, 6)

**Files:** `LLM_Calls/usage_accounting.py`; `API_Deps/usage_quota_deps.py`; `endpoints/chat.py`; `tests/Usage/test_llm_quota_byok_calls.py`; `tests/Chat/integration/test_chat_llm_tokens_quota_402.py`.

**Interfaces:**
- `enforce_llm_quota(user_id, estimated_tokens=1, *, provider=None, own_key: bool | None = None)`. `own_key` is the caller's `users_own_key_serves` answer for this user and provider; `None` (every other caller) looks it up as before.
- `chat._own_key_before_the_hosted_plan(current_user, provider, billing_org_id) -> bool | None`. It returns `None` when billing enforcement is off or there is no org; otherwise it returns the lookup's answer.
- `_accounted_sync_stream`/`_accounted_async_stream`/`_record_stream` accept `plan=None`: they read the items inside `accounted_call()` and record nothing.

- [ ] **Step 1: Write the tests.**

  (a) In `test_llm_quota_byok_calls.py`, directly after `test_a_provider_method_nested_in_an_own_key_call_is_not_checked_again` (it ends `assert len(response["chunks"]) == 1`):

```python
@pytest.mark.parametrize("layer", ["sync", "async"])
async def test_a_provider_method_nested_in_an_own_key_stream_is_not_checked_again(refused, layer: str) -> None:
    """The same for a stream: its body runs while it is read, after stream() returned, and is not refused mid-stream."""

    def _without_handle(request: dict[str, Any]) -> dict[str, Any]:
        """The request a delegate gets: no handle."""
        return {key: value for key, value in request.items() if key != PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY}

    class _Delegating(_Provider):
        """stream() and astream() read their reply from chat() and achat()."""

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
            """Delegate to chat() while the stream is read."""
            yield self.chat(_without_handle(request))["choices"][0]["message"]["content"]

        async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
            """Delegate to achat() while the stream is read."""
            yield (await self.achat(_without_handle(request)))["choices"][0]["message"]["content"]

    request = {**REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user")}
    with llm_call_context(user_id=7, records_itself=True):
        if layer == "sync":
            chunks = await asyncio.to_thread(lambda: list(_Delegating().stream(request)))
        else:
            chunks = [chunk async for chunk in _Delegating().astream(request)]
    assert chunks == ["hi"]
```

  (b) At the end of `test_chat_llm_tokens_quota_402.py`. It stubs `perform_chat_api_call`, so no provider is called. It makes three direct mock calls (`refuse_llm_work` and `_on_a_spent_hosted_plan` are helpers); `test_quality_triage.py --enforce` passes with it (measured). Do not inline the helpers' patches:

```python
def test_a_hosted_over_limit_request_on_the_users_own_key_looks_the_key_up_once(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch
):
    """The hosted plan's skip and the per-user limits' admission share one own-key lookup (TASK-13955 review)."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    asked: list[str] = []

    async def _own_key(_uid, provider):
        """The request's provider runs on the user's own key; each lookup is noted."""
        asked.append(provider)
        return provider == "local-llm"

    def _dispatch(**_kwargs):
        """Answer the call."""
        return {
            "id": "chatcmpl-once",
            "object": "chat.completion",
            "created": 0,
            "model": "local-llm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

    _on_a_spent_hosted_plan(client, monkeypatch)
    refuse_llm_work(monkeypatch)
    monkeypatch.setattr(chat_ep, "users_own_key_serves", _own_key)
    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _dispatch)

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
    assert asked == ["local-llm"]
```

- [ ] **Step 2: Run, and see it fail.**
  `$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py -k "nested_in_an_own_key_stream or looks_the_key_up_once"`
  Expected: `3 failed, 65 deselected` (measured). The failures:
  - both stream cases: `LLMQuotaExceededError: 402: {... 'category': 'llm_tokens_month' ...}`, raised by the nested `chat()`/`achat()`;
  - the lookup test: `AssertionError` on `['local-llm', 'local-llm'] == ['local-llm']` ("Left contains one more item").

- [ ] **Step 3: Implement the stream fix** in `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py`.

  (a) `_record_stream` (line 244): its signature, docstring, and a first statement:

```python
def _record_stream(provider: Any, plan: _CallPlan | None, request: Any, seen: _StreamUsage, started: float) -> None:
    """Record a stream from its usage chunk, or an estimate of what was received; nothing with no plan; never raises."""
    if plan is None:
        return
```

  (b) `_accounted_sync_stream` (line 263), signature and docstring:

```python
def _accounted_sync_stream(
    provider: Any, plan: _CallPlan | None, request: Any, inner: Any, started: float
) -> Iterator[Any]:
    """Yield ``inner``'s items, each read as an accounted call; record once when the stream ends, however it ends."""
```

  (c) `_accounted_async_stream` (line 327): `plan: _CallPlan` becomes `plan: _CallPlan | None`.

  (d) In `_wrap_stream`'s inner `stream` (lines 410-418), the gate, the comment on the `with`, and `outer`:

```python
        if plan is None and user_id is None and source != USER_KEY_SOURCE:
            return fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():  # checked here or on the user's own key: a nested method is not checked again
            inner = fn(self, request, *args, **kwargs)
        if not hasattr(inner, "__iter__"):
            return inner  # a broken provider's result reaches the caller's own check unchanged
        outer = _accounted_sync_stream(self, plan, request, inner, started)
        return _checked_at_first_item(user_id, outer, inner) if unchecked else outer
```

  (e) In `_wrap_astream`'s inner `astream`:
  - the gate (line 434): `if plan is None and user_id is None and source != USER_KEY_SOURCE:`;
  - the `with accounted_call():` on line 437 gets the same comment as in (d);
  - in `_resolve` (line 451): `if not hasattr(resolved, "__aiter__"):`;
  - line 458: `outer = _accounted_async_stream(self, plan, request, inner, started)`.

- [ ] **Step 4: Implement the single lookup.**

  (a) In `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py`, replace `enforce_llm_quota`'s signature, docstring and own-key line (lines 88-100):

```python
async def enforce_llm_quota(
    user_id: Any, estimated_tokens: int = 1, *, provider: str | None = None, own_key: bool | None = None
) -> None:
    """402 limit_exceeded when the user is over an LLM quota; otherwise mark the request admitted (spec §4a).

    An over-limit user whose ``provider`` runs on their own key is let through
    unmarked: each call is then checked where its key is known, and a call on
    the user's own key is not checked (TASK-13955). ``own_key`` is the caller's
    ``users_own_key_serves`` answer for this user and provider, when it already
    has one; None looks it up.
    """
    refusal = await llm_quota_refusal(user_id, estimated_tokens)
    if refusal is None:
        mark_llm_quota_prechecked()
        return
    if provider and (own_key if own_key is not None else await users_own_key_serves(user_id, provider)):
        return
```

  (b) In `tldw_Server_API/app/api/v1/endpoints/chat.py`, add this helper just above `def _request_has_endpoint_override` (line 417):

```python
async def _own_key_before_the_hosted_plan(current_user: Any, provider: str, billing_org_id: Any) -> bool | None:
    """Whether the user's own key serves ``provider``; None when the hosted plan does not check this request.

    Asked once per request: ``enforce_llm_quota`` reuses the answer (ADR-083).
    """
    if not enforcement_enabled() or billing_org_id is None:
        return None
    return await users_own_key_serves(getattr(current_user, "id", None), provider)
```

  (c) In `create_chat_completion`, replace the six-line condition (lines 4407-4412, from `if (` to `):`) with:

```python
            own_key = await _own_key_before_the_hosted_plan(current_user, selected_provider, billing_org_id)
            if own_key is False and not _billing_enforcer_entered:
```

  The two comment lines above it are unchanged. Then make the `enforce_llm_quota` call (lines 4428-4430) read:

```python
            await enforce_llm_quota(
                getattr(current_user, "id", None), max(1, _estimated_request_tokens), provider=selected_provider,
                own_key=own_key,
            )
```

- [ ] **Step 5: Run, and see it pass.** The Step 2 command. Expected: `3 passed, 65 deselected` (measured).
- [ ] **Step 6:** The usage, stream-accounting and Messages suites:
  `$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py tldw_Server_API/tests/Usage/test_llm_quota_decision.py tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Chat/integration/test_messages_endpoint.py tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py`
  Expected: no failures (measured `168 passed` with Task 3's tests also present).
- [ ] **Step 7: Function size.** `$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/lint/test_function_size_ratchet.py`. Expected: all pass. `create_chat_completion` measures 2905 lines (ceiling 2908). Do not lower the ceiling: `--tighten` is not part of this PR.
- [ ] **Step 8: Commit** (message 2): `git add` the five files, then `git diff --cached --check`.

## Task 3: The boundary lookup on real storage, an honest stub, the docstring (findings 7, 8)

**Files:** `tests/Usage/test_llm_quota_decision.py`; `AuthNZ/repos/usage_repo.py`.

This task adds tests to code that already behaves. So there is no RED run; two mutation probes show that the tests catch what they guard.

- [ ] **Step 1: Make the stub honour its argument.** In `test_users_own_key_serves_mirrors_the_first_test_of_credential_resolution`, replace line 265 with:

```python
    monkeypatch.setattr(byok_helpers, "is_provider_allowlisted", lambda provider: allowlisted and provider == "openai")
```

  Only the `OpenAI` cases reach the allowlist. `custom-openai-api` and `ollama` stop at the host table first.

- [ ] **Step 2: Add the real-storage test** at the end of the file:

```python
@pytest.mark.parametrize("lost", [None, "revoked", "user-deactivated"])
async def test_users_own_key_serves_reads_the_users_stored_key(tmp_path, monkeypatch: pytest.MonkeyPatch, lost) -> None:
    """On real BYOK storage, a stored key serves; once revoked, or once its user is deactivated, it does not."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
    from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import set_authnz_test_user_active
    from tldw_Server_API.tests.AuthNZ_SQLite.test_byok_endpoints_sqlite import _setup_byok_sqlite
    from tldw_Server_API.tests.AuthNZ_SQLite.test_byok_runtime_sqlite import _upsert_user_key

    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    repo = AuthnzUserProviderSecretsRepo(state["pool"])
    await _upsert_user_key(repo, user_id, "openai", "sk-own")
    if lost == "revoked":
        await repo.delete_secret(user_id, "openai")
    elif lost == "user-deactivated":
        await set_authnz_test_user_active(state["pool"], user_id, False)
    # The vendor's default host, whatever base URL this machine's config.txt or environment sets.
    monkeypatch.setattr(byok_helpers, "load_server_config_snapshot", dict)
    assert await usage_quota_deps.users_own_key_serves(user_id, "openai") is (lost is None)
```

  Notes:
  - `_setup_byok_sqlite` and `_upsert_user_key` are the repo's AuthNZ SQLite fixtures; `test_llm_quota_byok_calls.py` already imports them the same way.
  - Do not deactivate the user with raw SQL. The profile write guard rejects it (`ProfileUserWriteRejected`, measured). Use `set_authnz_test_user_active`.
  - Only the server config snapshot is stubbed. BYOK on, the allowlist, the pool and the key are all real.

- [ ] **Step 3: Run.** `$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_llm_quota_decision.py -k users_own_key_serves`. Expected: `12 passed, 15 deselected` (measured on unchanged code and after).
- [ ] **Step 4: Mutation probes.** Run each one, see it fail, and revert it. Record the results in the task notes.
  - In `usage_quota_deps.users_own_key_serves`, change `repo.fetch_secret_for_active_user(uid, name)` to `repo.fetch_secret_for_user(uid, name, include_revoked=True)`. Then `-k reads_the_users_stored_key` gives `2 failed, 1 passed` (`revoked`, `user-deactivated`), measured.
  - Change `is_provider_allowlisted(name)` to `is_provider_allowlisted(provider)`. Then `-k mirrors_the_first_test` gives `2 failed, 7 passed` (`own-key`, `vendor-base-url`), measured. With the old stub, the same mutation gave `9 passed`.
  - `git diff tldw_Server_API/app` must show no change after the reverts.
- [ ] **Step 5: Wrap the docstring** in `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py` (`insert_llm_usage_log`, line 564). Replace the one line with:

```python
        do not embed backend-specific SQL. A row with a ``credential_source`` on a
        schema without that column is written without it.
```

  Then `awk 'length > 120 {print FNR}' tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py` prints `306`, `749` and `800` (measured). Those lines predate #3371; the wrap moved 748 and 799 down by one.
- [ ] **Step 6: Commit** (message 3): `git add tldw_Server_API/tests/Usage/test_llm_quota_decision.py tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py && git diff --cached --check`.

## Task 4: Docs, mirror, verify, close, PR

- [ ] **Step 1: ADR-083 amendment.** Append to `Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md`, after the Follow-up paragraph. Leave a blank line before it, and write the real id:

```markdown
## Amendment (2026-10-31): the vendors' other hosts, and a trusted caller's per-request base URL

TASK-<ID> closes two gaps that TASK-13955's final review found in "Whose key is the user's own".

- **The vendors' other hosts are theirs.** A vendor's own host is no longer only the one its adapter calls by default. `VENDOR_HOSTS` also lists the hosts each vendor documents for the same API: OpenAI's data-residency domains (`us`, `eu`, `au`, `ca`, `jp`, `in`, `sg`, `kr`, `gb` and `ae`, each followed by `.api.openai.com`), `api.cohere.com`, `api.moonshot.ai`, and Bedrock's FIPS endpoints (`bedrock-runtime-fips.<region>.amazonaws.com`). A user key sent to one of them is the user's own, so the sentence in Consequences that such a host counts no longer holds. A `*` in a host pattern matches one DNS label, so `bedrock-runtime.<region>.amazonaws.com` no longer also matches a deeper host.
- **A per-request base URL is read too.** A platform admin, a service principal or the single user may pass a `base_url` with the request (ADR-025), and every billed adapter prefers it to the configured one. Off the vendor's hosts, that call is `user_unmetered` and counts, like a configured base URL. Before, it was recorded as `user`.
```

  The accepted text above it is not edited (ADR-001: "Accepted ADRs are immutable except for supersession metadata"). The repo amends with dated sections (ADR-025, ADR-063).
- [ ] **Step 2: The ADR index.** In `Docs/ADR/README.md`, append this sentence to the end of ADR-083's Decision cell, before the closing ` |`:
  ` Amended (TASK-<ID>): the other hosts a vendor documents are its own, and a trusted caller's per-request `base_url` is classified like a configured one.`
- [ ] **Step 3: `Docs/Operations/Usage_Quotas.md`**, in the "A call on the user's own key is not counted or checked" bullet (line 89). Replace the text from `in \`config.txt\` (for example` through `until it is added to \`VENDOR_HOSTS\` in \`Usage/llm_call_context.py\`.` with:

```markdown
in `config.txt` (for example `qwen_api_base_url`, `moonshot_api_base_url`, `zai_api_base_url`, `huggingface_api_base_url`), in an environment variable such as `OPENAI_API_BASE_URL`, or in the `base_url` a trusted caller (a platform admin, a service principal or the single user) passes with a request. The vendor's own hosts are the one its adapter calls by default and the others the vendor documents for the same API: OpenAI's data-residency hosts (such as `eu.api.openai.com`), `api.cohere.com`, `api.moonshot.ai` and Bedrock's FIPS endpoints. They are listed in `VENDOR_HOSTS` in `Usage/llm_call_context.py`; any other host counts.
```

  The BYOK guide is not edited: its "Usage Quotas" section names no host, and what it says stays true (see "What was verified", row 4).
- [ ] **Step 4: Refresh the mirror:** `bash Helper_Scripts/refresh_docs_published.sh`. Measured: it changes exactly `Docs/Published/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md` and `Docs/Published/ADR/README.md`. `Usage_Quotas.md` is not mirrored.
- [ ] **Step 5: Commit** (message 4): `git add` the five doc files, then `git diff --cached --check`.
- [ ] **Step 6: Verify.** Run each of these (the expected results were measured on the prototype):
  - The focused set: `95 passed`.
  - The wider set with label `final`, compared with Task 0's baseline. Expected: SQLite `4 failed, 6887 passed, 17 skipped, 1 error`, with the same five pre-existing ids; PostgreSQL `6 passed`. Any other failure is this plan's.
  - `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs tldw_Server_API/tests/lint`: `478 passed`.
    - It includes the function-size ratchet. It passes with `TASK-<ID>` still in the docs, so the docs tests do not catch an unreplaced id; the grep below does.
    - Once, `lint/test_rg_route_map_lint.py::test_lint_sees_the_routes_a_fresh_app_serves` failed: its subprocess died with `libc++abi ... recursive_mutex lock failed`. It passed alone (`7 passed`, on the base and on the change) and in the next full run. If it fails, re-run that file alone before judging.
  - `$PY Helper_Scripts/ci/check_shard_coverage.py`: `new_uncovered=0` and "OK — no newly-unshared test files."
  - `$PY Helper_Scripts/ci/test_quality_triage.py --enforce`: exit 0, "OK — no offenses beyond the baseline."
  - `PYTHONPATH=. $PY Helper_Scripts/ci/route_auth_ratchet.py`: "208 unauthenticated routes; baseline exact."
  - `PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts`: exit 0, "OK — OpenAPI fingerprint matches the checked-in snapshot." and "OK — committed path union matches the canonical contract."
  - `git grep -n -E "llm_call_context.py|usage_accounting.py|usage_quota_deps.py|endpoints/chat.py|usage_repo.py" Helper_Scripts/ci/scope_predicate_baseline.txt`: nothing.
  - `git diff origin/dev --check`: nothing.
  - `git grep -n "TASK-<ID>" -- ':!Docs/superpowers/plans'`: nothing.
- [ ] **Step 7: Bandit**, on the touched source:

```bash
/opt/homebrew/bin/uvx bandit -ll tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/Usage/llm_call_context.py
```

  Measured: "No issues identified", Medium 0, High 0. The `nosec encountered (B608), but no failed test` warnings on `usage_repo.py` are pre-existing.
- [ ] **Step 8: Close TASK-<ID>** (backlog-py; fill every `<...>` first):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-<ID> \
  --append-notes "RED/GREEN (record the real counts): Task 1 17 failed, 15 passed -> 32 passed; Task 2 3 failed -> 3 passed; Task 3 12 passed (tests for existing behaviour; mutation probes: fetch_secret_for_user(include_revoked=True) 2 failed, allowlist on the raw name 2 failed, 9 passed with the old stub). Focused set 70 -> 95 passed. Wider set: SQLite <base> -> <after> (the same 4 failed + 1 error before and after); PostgreSQL 6 -> 6 passed. tests/Docs + tests/lint 478 passed. Shard coverage new_uncovered=0; test-quality ratchet OK; route-auth ratchet exact; OpenAPI fingerprint and path union unchanged; create_chat_completion 2908 -> 2905 lines." \
  --append-notes "Decisions: OpenAI data-residency hosts <all ten | eu and us> (Decision 1); a billed provider with no config section counts (user_unmetered), per the owner's brief; streams use _wrap_chat's gate, so a checked stream that records nothing also reads its items as an accounted call; the ADR is amended, not superseded." \
  --append-notes "Docs: ADR-083 amendment and index row, Usage_Quotas.md (own-key bullet); Docs/Published refreshed (ADR-083, ADR README). BYOK guide unchanged (names no host). Bandit: <result>." \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --check-ac 7 --check-ac 8 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Closes PR #3371's deferred minors. A trusted caller's per-request base_url or api_base_url is classified like a configured base URL, so a user-key call off the vendor's hosts is user_unmetered, checked and counted. VENDOR_HOSTS lists the hosts each vendor documents (OpenAI's data-residency domains, api.cohere.com, api.moonshot.ai, Bedrock FIPS), and a * matches one DNS label. A billed provider without a config section counts instead of raising. A provider method nested in an own-key stream is no longer checked. /chat/completions asks users_own_key_serves once per request. users_own_key_serves is tested on real SQLite storage. ADR-083 amended; Usage_Quotas.md corrected; Docs/Published refreshed." \
  -s Done
```

  `--check-dod 4` is Bandit; `--check-dod 6` holds because the notes name the pre-existing failures. Then `git add -A backlog/tasks && git diff --cached --check`, and commit (message 5).
- [ ] **Step 9: Review before the PR.** Run `/superpowers:requesting-code-review` in a clean subagent on `origin/dev..HEAD`. Give it this plan's Review Focus, and address the findings before opening the PR.
- [ ] **Step 10: Open the PR.**
  - `git push -u origin fix/byok-quota-minors-<ID>`, then `gh pr create --base dev --title "fix(usage): BYOK quota exemption review minors (TASK-<ID>)"`.
  - The body states:
    - the owner's request;
    - the eight findings and each fix;
    - the vendor hosts with their documentation URLs;
    - the Review Focus items;
    - the measured counts;
    - the `## Change summary` (or the waiver; Decision 2).
  - It ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - Release `<scratchpad>/id_lock` once the PR's file list shows `backlog/tasks/task-<ID> - ....md`.
  - Check the merge-queue mode before merging (`gh variable get MERGE_QUEUE`; see CLAUDE.md "Merging into dev").

## Commit messages

Each ends with a blank line and `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

- 0: `chore(backlog): file TASK-<ID>, the BYOK quota exemption review minors, and its plan`
- 1: `fix(usage): classify a trusted caller's base_url and the vendors' documented hosts (TASK-<ID>)`. Body: findings 1-4; a provider without a config section counts instead of raising.
- 2: `fix(usage): own-key streams are not checked mid-stream; one own-key lookup per chat request (TASK-<ID>)`
- 3: `test(usage): users_own_key_serves on real BYOK storage; wrap a long docstring (TASK-<ID>)`
- 4: `docs(adr): amend ADR-083 for the vendors' other hosts and per-request base URLs (TASK-<ID>)`
- 5: `chore(backlog): close TASK-<ID>`
