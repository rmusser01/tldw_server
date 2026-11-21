# An explicit model denylist; `allowed_models` governs the server key only (TASK-15582) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An admin can deny a model outright with a new `denied_models` list on the LLM provider override (refused on every key and path), while `allowed_models` becomes the list of models the server's key offers (a user's own key may run others). Task 1 first lands the four deferred minors of #3537's final review.

**Architecture:** The override policy gains one keyword, `allowlist: bool = True`, on `ProviderOverrideCallSnapshot.policy_error`/`enforce`, `validate_provider_override` and `enforce_provider_override`. A disabled provider (`provider_disabled`) and a denied model (`model_denied`, new, 403) are refused whatever `allowlist` says; a model off `allowed_models` (`model_not_allowed`) only when `allowlist` is true. Every check passes `allowlist=False` exactly when the call runs on the user's own key at its vendor (credential source `user`, ADR-083's definition), or when it runs before the key is known and a later check in the same request sees the key. The two shared choke points carry it: the adapter base-class guard reads the key from `call_credential_source(request)`, and the credential runtime checks every-key rules before resolution and the allowlist after it, from `billed_credential_source(provider, resolution.source, resolution.app_config)`. The column ships as SQLite migration 107 and a PostgreSQL `ADD COLUMN IF NOT EXISTS`; the admin API and the admin UI (`admin-ui/`) set it.

**Tech Stack:** Python 3, FastAPI, pydantic, click, pytest (pytest-asyncio auto mode, xdist), loguru, backlog-py; Next.js + vitest (`admin-ui`, bun).

**Spec:** the owner's decision of 2026-11-21 (below), the controller's planning brief, TASK-15582 on `origin/dev`, and Minors 2 to 5 of #3537's final whole-branch review.

**Base:** `origin/dev` at `0d5e9c7228` (#3537 merged). Every line number below is at that commit. Branch `feat/model-denylist-15582`; the PR targets `dev`.

## The owner's decision (2026-11-21, verbatim choice "Server key only")

- A model left off a provider override's `allowed_models` is only "not offered on the server key". It is refused when the call would run on the server's key, and it works with a user's own key, the way a provider with no server key does.
- A new `denied_models` list on the provider override is the explicit model disallow, refused on every key and every path, as `allowed_models` refusals are after #3537.
- Provider level is unchanged: `is_enabled=False` is the explicit provider disallow, refused on every key.
- Standing rules: an explicit admin disable is refused on every key; "no server key or credits" is not a disable, and BYOK works (2026-11-19). A store outage fails closed (2026-11-20).

## Global Constraints

- Tests make no real outbound network requests. Every pytest run loads the network probe plugin (`-p netprobe_15427`, from `<plans dir>`); a run ends with 0 probe lines for the tests this plan adds or changes.
- A test that drives `/chat/completions`, an LLM adapter or a judge stubs the adapter or the provider call; a commercial provider otherwise calls the real API.
- A failed judge or model call, or a reply with no readable score, is unscored (`score: None`, a sample error, out of every average), never 0.0 (owner, 2026-11-10 and 2026-11-12).
- Backlog edits use backlog-py only (`BACKLOG task edit ...`), with `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit a task file. TASK-15582 exists; create no other task.
- Branch `feat/model-denylist-15582` from `origin/dev`; the PR targets `dev`.
  - Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - The ADR-004 Change summary is waived (owner, 2026-11-21). The PR body records it as a full `## Change summary` section reading: "ADR-004: the owner waived a user-written Change summary for this PR on 2026-11-21."
- Never `--no-verify`, never force-push, never `git stash`.
- PostgreSQL only through the repo fixtures, with `TLDW_TEST_NO_DOCKER=1` and a session-unique `TEST_DB_NAME`. Never start Docker for it; never restart a server. Redis: never FLUSHALL or SCRIPT FLUSH.
- A mirrored doc (ADRs, the BYOK guide, everything under `Docs/Published`) needs `bash Helper_Scripts/refresh_docs_published.sh`, with the `Docs/Published` diff in the same PR.
- Scope-predicate ratchet: measured, no file this plan touches is in `Helper_Scripts/ci/scope_predicate_baseline.txt`.
- Function-size ratchet (`tests/lint/function_size_baseline.json`, threshold 300). Measured at `0d5e9c7228`, the capped functions this plan touches and their room: `chat:create_chat_completion` 2902 of 2908 (this plan adds 2 lines), `migrations:get_authnz_migrations` 439 of 443 (adds 1: the 107 entry is one line), `chat_documents:generate_document` 466 of 466 (a set literal swapped in place, 0 lines), `speech_chat_service:run_speech_chat_turn` 563 of 563 (two 5-line set literals become 1 line each, -8). No other changed function is near 300.
- Endpoint-monkeypatch ratchet (`tests/lint/test_endpoint_monkeypatch_ratchet.py`): no new test patches an attribute of a module under `app/api/v1/endpoints/`. Tests reuse existing helpers that already do (`_configure_no_dispatch_boundary`).
- Test-quality triage (`Helper_Scripts/ci/test_quality_triage.py --enforce`): a test whose only asserts compare a list its own nested function fills is a `tautology_suspect`, and a test whose asserts touch no attribute but `.status_code`/`.status` is `status_only` (a bare name such as a call list does not count). New tests assert tuples that include a real output, read through an attribute (`response.json()`, `refusal.value.error_code`).
- Two artefacts of `RUN` itself, both pre-existing and in tests this plan does not add (review round 1, M1):
  - `AuthNZ_Unit/test_llm_provider_overrides.py::test_two_worker_caches_converge_after_shared_store_rotation` fails in every `RUN` that includes its file: its spawn-context `multiprocessing.Manager()` binds an AF_UNIX socket under the long `TMPDIR` (`OSError: AF_UNIX path too long`, then `EOFError`). It fails the same way before any edit.
  - Every `RUN` over `tests/Usage` logs 1 probe line: `test_quota_evals_chatbooks.py::test_evaluations_endpoint_daily_cap_fires_for_the_authenticated_user` resolves `openaipublic.blob.core.windows.net` (tiktoken downloads its encoding into the fresh, empty `TMPDIR` cache).
  - Every `Expected:` below counts them where they occur.
- New test files sit under a CI shard path: `tests/Usage/` (whole directory) and `tests/Admin/test_admin_llm_provider*.py`.
- Disk rules override everything; the owner is angry about the disk filling.
  - Every pytest command runs through `RUN` (below), which is `mkdir -p <T> && TMPDIR=<T> zsh <scratchpad>/disk_guard.zsh <T> -- $PY -m pytest --basetemp=<T>/bt -p no:cacheprovider ...` from a fresh `<T>` under `<scratchpad>`, with at most `-n 2`.
  - Measure RED in place: write the test before the source, or revert a source hunk and restore it with `git restore <file>`. Never `git archive`, never repo copies, never extra worktrees.
  - Leave no log files. The sweep's two id lists are deleted in Task 5.
  - `admin-ui/node_modules` is installed for Task 6 only and deleted at its end. `apps/` dependencies are never installed: its unit tests are measured by CI's `frontend-required` job.
- The plan stays free of trailing whitespace (it is committed under `Docs/superpowers/plans/`), and the committed copy names no private planning file.
- Tests that replace a provider method (`Adapter.chat = ...`, a non-`ChatProvider` from `get_adapter`) bypass the base-class guard (ADR-067). A test of the guard uses a `ChatProvider` subclass (`_Provider` in `tests/Usage/test_llm_quota_byok_calls.py`).
- The guard fails closed while the store is unloaded; the root conftest gives every test a healthy, empty store and restores it after.

**Abbreviations.** `<scratchpad>` = `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad`; `<plans dir>` = `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`; `$PY` = `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Every command runs from the worktree root. `RUN <name> <pytest args>` = `zsh <scratchpad>/run15582.zsh <name> <pytest args>` (Task 0 writes it). `BACKLOG` = `PATH="$(dirname $PY):$PATH" PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd .`. `ADR-086` = `Docs/ADR/086-authoritative-provider-credential-scope.md`.

## Review Focus

The five input classes most likely to bite that the feature's own tests would not exercise, each pinned by a test in the task that owns the code:

1. **A denied model spelled another way.** "GPT-4o" for a denied "gpt-4o", or the dotted and dashed spellings of one Claude model, is denied too: a denial errs wide (the allowlist stays exact: it errs narrow). Task 3, `test_a_denied_model_is_denied_whatever_its_case_or_anthropic_spelling`, and the Slides row `model-denied` (denies "casesensitive-model" for the target "CaseSensitive-Model").
2. **A user's own key sent to an operator's endpoint** (a trusted caller's base URL off the vendor's host, a key stored with a base URL, a custom OpenAI-compatible slot, a local server). The operator's hardware answers it, so it is held to the allowlist, as ADR-083 counts it (`user_unmetered`). Task 5, `test_the_users_key_at_an_operator_endpoint_is_held_to_the_allowlist` (guard) and the `user-at-operator-host` row of `test_a_model_off_the_allowlist_resolves_only_on_the_users_own_key` (runtime). The unchanged MCP every-key test already sends the user's key to `https://fixed.example/v1` and still expects `model_not_allowed`.
3. **OpenRouter fallback models on the user's own key.** An off-list fallback runs; a denied one is refused. Task 5, `test_openrouter_fallbacks_on_the_users_own_key_skip_the_allowlist_but_not_the_denylist`.
4. **A call that names no model.** With only a denylist there is nothing to deny, and `require_model=True` (Data Tables, quizzes, Workflows, the Slides generator) holds a model-less call to the allowlist only where the allowlist applies. Task 3, `test_with_only_a_denylist_a_call_that_names_no_model_runs`.
5. **A stored denylist that does not parse.** Like `allowed_models`, it fails its row, so the store fails closed instead of silently dropping the denial; entries are trimmed and blanks dropped. Task 3, `test_a_stored_denylist_that_does_not_parse_fails_its_row`.

## Rulings

| # | Ruling | Why |
|---|---|---|
| R1 | **A team or org shared key counts as "server" for the allowlist.** The allowlist is skipped only when the call's billed credential source is `user`: `call_credential_source(request) == USER_KEY_SOURCE` at dispatch, `billed_credential_source(provider, resolution.source, resolution.app_config) == USER_KEY_SOURCE` at resolution. | ADR-083 defines "the user's own key" as exactly that, and counts org and team shared keys like the server's for the per-user limits. TASK-15426 runs features on "the user's own key only, never a team or org key". `billed_credential_source` applies its vendor-host test to `user` only, so counting team/org keys as own would let a team key stored for an operator endpoint (a custom slot, a local server) run off-list models on the operator's hardware. The owner's words are "a user's own key". Refusing is the fail-closed side. ADR-086's scopes are unchanged: the allowlist is checked after the scope check, on the credential the scope chose. |
| R2 | **A new code, `model_denied` (403), not a reuse of `model_not_allowed`.** | One code has one message in every table. `model_not_allowed` now means "not on this key; your own key may work", which is wrong for a denial. The code also follows its field (`allowed_models` / `denied_models`). |
| R3 | **The gateway calls the guard through a public seam:** `usage_accounting._refuse_admin_disabled` becomes `refuse_admin_disabled(provider, request)` (Task 1), and Task 5 makes it read the call's key from the request itself. | The review's suggestion; the guard stays the one choke point (ADR-067) and the gateway call honours exempt calls the same way. It sits inside each `_call_adapter` `try`, so a privacy-safe rebuild treats it like the adapter's own refusal. |
| R4 | **Minor 5's CLI test invokes `list-benchmarks` for `tldw-evals`, not `config`.** | The review said `tldw-evals config` alone exits 0; with click 8.5 a group given no arguments raises `NoArgsIsHelpError`, exit 2. `list-benchmarks` is a real non-LLM command on all three CLIs. |
| R5 | **A check that runs before the key is known checks only what every key refuses (`allowlist=False`) when a later check in the same request sees the key**: chat's first check (the credential runtime resolves the key at `chat.py:4663` and checks the allowlist there) and the study assistants' quota dependency (the reply checks again with its handle). | The owner's rule applies to the key the call runs on. |
| R6 | **Paths that never run on a user's own key stay strict** (`allowlist=True`, the default): chat macro runs (durable jobs), scheduled tasks, Data Tables, quiz generation, Workflows, the Slides generator, Slides standalone HTML (server key from config), the admin provider test. **Out-of-scope pre-key checks stay strict:** shared-workspace chat, the persona voice preflight, embeddings, text-to-speech (`/audio/speech` through `_resolve_tts_byok`, and streaming TTS through `tts_provider_credential_scope`, which gains the same check before its runtime resolves: review round 1, I1). | Strict is correct on the server's key. The last four are Known limits. |
| R7 | **A denylist matches case-insensitively and folds Anthropic dotted/dashed IDs; the allowlist stays exact.** | A denial errs wide; an allowance errs narrow. |
| R8 | **The server-key listing (`GET /llm/providers`, auto-routing candidates) hides denied models**, and a denied `config.default_model` falls back to the first listed model. | A model refused on every key must not be offered. The own-key listing is TASK-15583. |
| R9 | **A model on both lists is accepted by the API and refused (`model_denied`).** An empty list clears the column. | The brief: denied wins. Rejecting the write would need a merge with the stored row; enforcement already decides. |
| R10 | **Messages.** The shared table (`PROVIDER_STREAM_ERROR_MESSAGES`, used by chat, the app handler, the guard and stream normalization) and RAG's table say the user's own key may serve an off-list model; the two `apps/` copies of those tables follow. Mappers that answered every policy refusal with one fixed text (character chat, chat documents, chunking, Messages API, Prompt Studio, speech chat) keep it and only learn the code. TTS and embeddings keep their `model_not_allowed` text (their check still refuses every key). | Smallest change that puts the hint where users see it; the fixed-text mappers sit in capped functions. |
| R11 | **The admin page is `admin-ui/app/providers/page.tsx`**, not under `apps/` (the brief's guess). `apps/` has no override editor. | Measured with `git grep`. |
| R12 | **The credential runtime refuses a disabled provider or denied model before resolving any credential, and an off-list model after resolving one** (one credential read, no provider request). | Before resolution the key is unknown. |

## Rulings (review round 1, 2026-11-21)

An independent dry run in an isolated worktree at `0d5e9c7228` returned "Fix first": 4 Blocker, 1 Important, 6 Minor. Ten are folded; M3 is ruled against.

| Finding | Plan edit | Measured evidence |
|---|---|---|
| **B1** (Blocker): Task 1's CLI fix makes a bare LLM group a help request, and all four `tldw-evals` LLM commands are groups, so two existing tests stop loading the store | Task 1 Step 1: the `tldw-evals` row of `_EVALUATION_CLIS` names the leaf `"eval geval"`; the three tests that invoke it split it (`[*llm_command.split()]` at `:2681` and `:2720`; `[*llm_command.split(), "--help"]` and `llm_command.split()` in the help/completion test). Files list updated | Without it, `t1green` had 2 failures (`ByokResolutionError: credential_store_unavailable`; `assert 2 == 1`). With it, `-k "evaluation_cli or llm_group_given"`: 11 passed |
| **B2** (Blocker): Task 4's GREEN run (a) cannot collect under `-n 2`: `test_provider_usage_reservations_repo.py` parametrizes with a fresh `uuid4` (pre-existing); (b) `build_override_response` reads `override.denied_models`, which the concurrency test's `SimpleNamespace` stand-in lacks | Task 4 Step 4 runs the reservations file alone with `-p no:xdist`. Task 4 Step 3 adds `denied_models=None,` to `_override_response()` in `tests/Admin/test_admin_llm_provider_overrides_concurrency.py` (`:145-156`), which joins the Files list | (a) `Different tests were collected between gw0 and gw1`, 0 run; alone, 89 passed. (b) 3 failed (`AttributeError` ×2, `assert not True`); with the field, 4 passed |
| **B3** (Blocker): Task 5's GREEN run had 3 failures from unlisted tests, and the sweep 12 more | Task 5 Step 4: the double at `tests/Chat/test_chat_target_resolution.py:268` takes `**_kwargs`; the judge test is split by key (the own key gets `denied_models=["gpt-denied"]` and `model_denied`, the server key keeps `allowed_models=["x"]` and `model_not_allowed`, because on the server key the judge takes the server config's model, not `DEFAULT_MODEL_OPENAI`); the double at `tests/Sharing/test_shared_workspace_recipient_security_matrix.py:568` takes `**_kwargs` and joins the GREEN run | `TypeError: ... unexpected keyword argument 'allowlist'` (1); `DID NOT RAISE` on `[server-key-eval_runner]` and `[server-key-ms_g_eval]` (`resolve_provider_model("openai", cfg)` is `gpt-4o` there); Sharing: 12 failed, 19 passed with `**_kwargs` |
| **B4** (Blocker): `test_quality_triage.py --enforce` exits 1, `status_only=1 (new)`, on the new chat test | Task 5 Step 1: the chat test also asserts the body's `error_code` through `response.json()`; the Global Constraints name the `status_only` rule | `_check_status_only` flags an assert whose only attribute is `.status_code`; `dispatch_calls` is a bare name |
| **I1** (Important): streaming TTS (`tts_provider_credential_scope`) resolves through the runtime with no check of its own, so after Task 5 a user's own key runs a TTS model off `allowed_models`, contrary to R6, R10, Known limits and the ADR-086 text | Task 5 Step 3: `capture_provider_override_call_snapshot(provider_key).enforce(model)` before the runtime is built, as `_resolve_tts_byok` does (`tts_service.py:508-509`); one test, `test_streaming_tts_holds_the_users_own_key_to_the_allowlist`, measured RED by reverting that hunk | Used at `audio_streaming.py:2793`, `:2915`, `:3672`, `:3943`; embeddings unaffected (their only check is the strict one at `embeddings_v5_production_enhanced.py:823-824`) |
| **M1**: `RUN` artefacts make several counts wrong: (a) the AF_UNIX-path failure of `test_two_worker_caches_converge_after_shared_store_rotation` under the long `TMPDIR`; (b) 1 probe line from every `tests/Usage` run (tiktoken download) | Global Constraints name both; Task 1 and Task 3 RED and GREEN count (a); Task 2, 3 and 5 GREEN count (b); the Task 0 baseline lists (a) | Task 1 RED 8 (not 7), GREEN 1 failed; Task 3 RED 9 (not 8), GREEN 1 failed, 924 passed; Task 2 GREEN 779 passed, 1 skipped, probes 1; Task 5 GREEN probes 1 |
| **M2**: Task 5 RED is 16, not 13: the study test runs 4 own-key cases (flashcards or quizzes, SQLite or PostgreSQL) | Task 5 Step 2 says 16 | Measured 16; every other predicted case matched |
| **M3**: `get_default_model_for_provider` can return a denied model | Ruled against: one Known-limits line | Fails closed (`model_denied`); only an admin who denies the default they set meets it; a fix would copy `_model_denied`'s matching into `chat_target_resolution`, whose injected resolvers return objects without `denied_models` |
| **M4**: on PostgreSQL only the API's start-up adds the column, so a worker or CLI started first fails its store load | One line each in Known limits, the CHANGELOG's and the PR body's Before-upgrading notes: start the API once before the workers and CLIs | `startup_auth.py:252` runs `ensure_llm_provider_overrides_pg`; the refresh calls `list_overrides()` directly (`llm_provider_overrides.py:1190-1193`); SQLite migrates per process (`database.py:1210`) |
| **M5**: `users_own_key_serves` is `async def` | Interfaces for TASK-15583: the signature now reads `async def` | `usage_quota_deps.py:54` |
| **M6**: line references and the CHANGELOG | `cli/evals_cli.py`'s `llm_commands` is `:119`; the CHANGELOG's Fixed bullet names the refusal handler's message (minor 3). The `chat.py` import stays `:337`, measured at the base; Task 2's added import moves it to `:338`, which the review saw, and Task 5 says so | `grep -n` at `0d5e9c7228`: `evals_cli.py:119`, `chat.py:337` |
| **R1** (the review's view) | R1 stays. Known limits and the PR body add one line: a team or org key the team pays for, and a user's own key stored with the user's own base URL (`user_unmetered`), are held to the allowlist; both err closed | The review: R1 is consistent with ADR-083, the 2026-10-29 decision and TASK-15426, and every added check compares against `USER_KEY_SOURCE` |

## How each path knows the call's key

| Path | Where the key comes from | Change |
|---|---|---|
| Adapter base class: the four `usage_accounting` wrappers, and the gateway before model discovery (`refuse_admin_disabled`) | `call_credential_source(request)`: the runtime handle's billed source (and the request's own `base_url`/`api_base_url`), else the call context's | `allowlist = source != "user"` |
| `ProviderCredentialRuntime._capture_override_snapshot` (before resolution) in every runtime built with `override_snapshot_resolver`: chat, character chat, chat documents, Messages, RAG, evaluations, Prompt Studio (three endpoints and the jobs worker), chunking, the auto-boundary assistant, moderation, audio streaming, the realtime pipeline, speech chat, TTS, and `users_own_key_credentials` | Not known yet | `allowlist=False` |
| `ProviderCredentialRuntime._enforce_call_policy` (after resolution, on every `resolve`) | `billed_credential_source(provider, resolution.source, resolution.app_config)`, the source the handle carries | `allowlist = billed != "user"` |
| MCP `model_completion` (authoritative `_resolve_entry`) | the same, after the scope check | `enforce_provider_override(..., allowlist=...)` |
| `_summarize_via_adapter` (analyze) | the handle it runs on, explicit or scoped from the call context, else the context | `call_credential_source({KEY: handle})` |
| `/chat/completions` first check (`chat.py:4469`) | not known; resolved at `:4663` | `allowlist=chat_macro_intent is not None` |
| `/chat/completions` failover check (`chat.py:5295`) | the fallback handle in `refreshed_args` | `call_credential_source(refreshed_args) != "user"` |
| Study assistants: `study_assistant_own_key` dependency | not known; the own-key handle is resolved right after | `resolve_chat_target(..., allowlist=False)` |
| Study assistants: the reply | its `provider_credentials` handle | `allowlist = handle source != "user"` |
| Messages API native path, character chat, chat documents, RAG, Prompt Studio, evaluations | their runtime (row 2 and 3) | none of their own |
| Slides standalone HTML, Data Tables, quizzes, Workflows, Slides generator, scheduled tasks, admin test | always the server's key | unchanged; `denied_models` applies |
| TTS: `/audio/speech` (`_resolve_tts_byok`) and streaming TTS (`tts_provider_credential_scope`, four uses in `audio_streaming.py`) | out of scope (ADR-086): checked before any key is resolved | strict; streaming TTS gains the check `/audio/speech` already has (I1) |
| `POST /llamacpp/inference` | names no model | unchanged |

## Interfaces for TASK-15583

The helpers that answer "is this provider/model usable for this user on this key" (all in `tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py` unless named):

- `validate_provider_override(provider: str, model: str | None, *, allowlist: bool = True) -> dict[str, str] | None`. None means usable. `allowlist=False` for a call on the user's own key at its vendor; `True` for the server's key, a team or org key, or a user key sent to an operator endpoint. Raises `ByokResolutionError("credential_store_unavailable")` when the store cannot be read.
- `capture_provider_override_call_snapshot(provider: str) -> ProviderOverrideCallSnapshot`, then `snapshot.policy_error(model: str | None, *, allowlist: bool = True) -> dict[str, str] | None` for each model of a list, from one snapshot. Codes: `provider_disabled` and `model_denied` on every key, `model_not_allowed` only with `allowlist=True`.
- `async def users_own_key_serves(user_id: Any, provider: str) -> bool` (`tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py`; a coroutine, awaited): whether a call to `provider` for this user would run on the user's own key at its vendor (BYOK on, provider allowlisted, an active key, the vendor's host). True is the `allowlist=False` condition for a listing.
- `apply_llm_provider_overrides_to_listing(payload: dict[str, Any], *, overrides: Mapping[str, LLMProviderOverride] | None = None) -> dict[str, Any]`: the server-key listing (allowlist and denylist applied). An own-key listing must not use it to filter models; filter with `policy_error(model, allowlist=False)`.
- `ADMIN_POLICY_CODES` (`tldw_Server_API/app/core/LLM_Calls/provider_results.py`): `{"provider_disabled", "model_not_allowed", "model_denied"}`, each a 403.

## Decisions for the owner

None. The owner's 2026-11-21 answer settles the product rule. R1 (team and org keys are held to the allowlist) is a ruling the brief asked for; reversing it later is the `USER_KEY_SOURCE` comparison in four places.

## Known limits

- **A denylist names model IDs.** Case and Anthropic dotted/dashed spellings fold (R7), but an alias, a dated snapshot ("gpt-4o-2024-08-06") or the same model through another provider ("openai/gpt-4o" on OpenRouter) is another ID: deny each, per provider.
- **An admin can no longer hold own-key users to a list.** `allowed_models` no longer binds a user's own key, and a denylist cannot say "all but these". This follows from the owner's choice; the CHANGELOG and PR carry a "Before upgrading" note.
- **Team and org keys are held to the allowlist** (R1). So are a team or org key the team itself pays for, and a user's own key stored with the user's own base URL (`user_unmetered`). Both err closed.
- **The default model can be a denied one.** `get_default_model_for_provider` (`chat_target_resolution.py:66-73`) returns the override's `config.default_model`, else the first `allowed_models` entry, without reading `denied_models`; a request that names no model is then refused `model_denied`, while the listing (R8) shows a fallback default. It fails closed, and only an admin who denies the model they made the default meets it (review round 1, M3, ruled against).
- **PostgreSQL upgrade order.** Only the API's start-up (`startup_auth.py:252`) and the admin service add `llm_provider_overrides.denied_models` on PostgreSQL; the store's refresh selects it. A standalone worker or a `tldw-evals` LLM command started after the upgrade but before the API has started once cannot load the store, so it refuses every LLM call (and the CLI names `DATABASE_URL`) until the API has run. It fails closed and heals itself; the Before-upgrading note says to start the API first. SQLite migrates in every process (review round 1, M4).
- **Strict checks before the key is known** (R6): shared-workspace chat (`sharing.py:1061/1071`) and the persona voice preflight call `resolve_chat_target` with the default `allowlist=True`, so a recipient's own key cannot run an off-list model there; embeddings and TTS (`/audio/speech`, and streaming TTS after I1) check `allowed_models` before resolution and keep refusing every key (ADR-086 puts them outside the LLM rule). Upgrade path for sharing: `allowlist=False` there, plus `enforce_provider_override(..., allowlist=billed != "user")` after `resolve_byok_credentials` in `shared_workspace_chat_service.generate_grounded_answer`.
- **Fixed-text mappers** (R10) tell an own-key-capable user only that the model is disabled.
- **An off-list model on the server's key is refused after one credential read** (R12).
- **The guard runs twice on a gateway call** (the gateway, then the adapter's wrapper): two store snapshots, each a deep copy of the override map. Fine at today's size.
- **One unparseable override row fails the whole store** (pre-existing, fail-closed by the owner's 2026-11-20 rule); a corrupt `denied_models` now does too.
- **A call that names no model** is checked for `provider_disabled` only (pre-existing, #3537's M1).
- **TASK-15583:** the WebUI picker and `GET /llm/providers` for own-key users.

---

### Task 0: Branch, run wrapper, backlog, plan copy, draft PR, baseline

**Files:**
- Create: `<scratchpad>/run15582.zsh` (not committed)
- Modify: `backlog/tasks/task-15582 - Let-an-admin-explicitly-disallow-a-provider-or-model-distinct-from-not-configured.md` (backlog-py)
- Create: `Docs/superpowers/plans/2026-11-21-model-denylist.md` (this plan, verbatim)
- Create: `<plans dir>/pr_body_model_denylist.md` (not committed)

- [ ] **Step 1: Branch.**

```bash
git fetch origin dev
git switch -c feat/model-denylist-15582 origin/dev
git log --oneline -1
```

Expected: `0d5e9c7228 Merge pull request #3537 from rmusser01/fix/admin-disabled-provider-refused-15581`. If `origin/dev` moved, note the new sha, re-measure the function-size room in Global Constraints, and continue.

- [ ] **Step 2: The run wrapper.** Write `<scratchpad>/run15582.zsh`:

```zsh
#!/bin/zsh
# run15582.zsh <name> <pytest args...>: one disk-guarded pytest run from the current directory (the worktree root).
# Prints failures and the summary, then the network-probe line count; leaves no file behind.
# With FAILED_TO=<file> it also writes the sorted FAILED/ERROR ids there (the Task 0/5 sweep; delete it after use).
S=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad
PL=/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
n=$1; shift
T=$S/r15582_$n
rm -rf "${T:?}"; mkdir -p "$T"
export PATH="$(dirname $PY):$PATH" PYTHONPATH=$PL:$PL/pyext NETPROBE_LOG=$S/r15582_$n.net
rm -f "$NETPROBE_LOG"
out=$(TMPDIR=$T zsh $S/disk_guard.zsh $T -- $PY -m pytest --basetemp=$T/bt -p no:cacheprovider -p netprobe_15427 -q "$@" 2>&1)
[[ -n $FAILED_TO ]] && print -r -- "$out" | grep -E '^(FAILED|ERROR) ' | sed -E 's/ - .*//' | sort -u > "$FAILED_TO"
print -r -- "$out" | grep -E '^(FAILED|ERROR) |[0-9]+ (passed|failed|skipped|errors?)|DISK_GUARD' | tail -40
print -r -- "netprobe lines: $([[ -f $NETPROBE_LOG ]] && wc -l < $NETPROBE_LOG || print 0)"
rm -f "$NETPROBE_LOG"
```

Expected: `zsh -n <scratchpad>/run15582.zsh` prints nothing.

- [ ] **Step 3: Start TASK-15582 and record the owner's decision against AC #1 and #2.**

```bash
BACKLOG task edit TASK-15582 -s "In Progress" --check-ac 1 --check-ac 2 \
  --append-notes "Owner's decision, 2026-11-21 (chose 'Server key only'). AC #1: the explicit provider disallow stays is_enabled=False, refused on every key; the explicit model disallow is a new denied_models list on the provider override, refused on every key and path with the new code model_denied (403), and it wins over allowed_models. AC #2: a model left off allowed_models is no longer refused on every key; it is only not offered on the server's key. It is refused (model_not_allowed) when the call runs on the server's key, a team or org key, or a user key sent to an operator's endpoint, and it runs on the user's own key at its vendor, as a provider with no server key does. Plan: Docs/superpowers/plans/2026-11-21-model-denylist.md." \
  --append-notes "Also done here, the four minors deferred from PR #3537's final review: the gateway runs the admin-override guard before model discovery, so a disabled provider gets no model-list request; the app's refusal handler answers with the code's own message; an LLM CLI group given no subcommand prints its help without loading the store; and two test gaps (a non-LLM CLI command run without --help, the standalone Slides branch of the app's refusal handler)."
```

Expected: `TASK-15582 - Let an admin explicitly disallow a provider or model, distinct from 'not configured' (In Progress)`.

- [ ] **Step 4: Copy the plan, check it, commit.** Copy this file to `Docs/superpowers/plans/2026-11-21-model-denylist.md`.

```bash
grep -n -E '(brief|plan|final_review|impl_report|design_review)_[0-9a-z_]*\.md' Docs/superpowers/plans/2026-11-21-model-denylist.md
grep -n -E ' +$' Docs/superpowers/plans/2026-11-21-model-denylist.md
git add -A backlog/tasks Docs/superpowers/plans/2026-11-21-model-denylist.md
git commit -m "chore(backlog): start TASK-15582 with its plan (TASK-15582)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: both greps print nothing; the commit lists 2 files.

- [ ] **Step 5: Push and open the draft PR.** Write `<plans dir>/pr_body_model_denylist.md` with a one-paragraph Why (the owner's decision, verbatim choice "Server key only"), "Work in progress: see Docs/superpowers/plans/2026-11-21-model-denylist.md", the `## Change summary` section from Global Constraints, and the closing `🤖 Generated with [Claude Code](https://claude.com/claude-code)` line.

```bash
git push -u origin feat/model-denylist-15582
gh pr create --base dev --draft --title "feat(llm): explicit model denylist; allowed_models governs the server key only (TASK-15582)" \
  --body-file <plans dir>/pr_body_model_denylist.md
```

Expected: a draft PR URL against `dev`.

- [ ] **Step 6: Baseline.** The sweep set is every test file that builds an override or fakes a policy check:

```bash
git grep -l -E 'LLMProviderOverride\(|validate_provider_override|enforce_provider_override|resolve_chat_target|def enforce\(self|model_not_allowed|_refuse_admin_disabled' \
  -- 'tldw_Server_API/tests/*.py' 'tldw_Server_API/app/core/MCP_unified/tests/*.py' | grep -v conftest.py > <scratchpad>/sweep_files_15582.txt
wc -l < <scratchpad>/sweep_files_15582.txt
FAILED_TO=<scratchpad>/sweep0_15582.txt RUN sweep0 -n 2 $(cat <scratchpad>/sweep_files_15582.txt)
wc -l < <scratchpad>/sweep0_15582.txt
```

Expected: 62 files. The run ends with a summary line and `DISK_GUARD: done`. `sweep0_15582.txt` holds roughly 90 to 110 ids: 84 in `Slides/test_standalone_html_provider.py` and 3 in `Slides/test_standalone_html_provider_model_inventory.py` (they resolve live hosts, which the probe blocks; listed in `tests/live_network_baseline.txt`), plus the pre-existing AF_UNIX failure of `test_two_worker_caches_converge_after_shared_store_rotation` (Global Constraints), plus any order-dependent probe leaks of the kind #3537 measured (`test_messages_provider_override_store_boundary.py`, `test_embeddings_orchestrator_endpoint_parity.py`). The probe lines come from the same tests. (The 62-file count was confirmed in review round 1; the sweeps themselves were not dry-run.) Only Task 5's diff against this list matters; keep the two files until then.

---

### Task 1: The four deferred minors of #3537's final review

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` (`_refuse_admin_disabled` `:365` and its calls `:398`, `:423`, `:452`, `:478`)
- Modify: `tldw_Server_API/app/core/LLM_Calls/gateway.py` (imports `:59-65`; `_call_adapter` `:1321`; `_call_adapter_async` `:1371`)
- Modify: `tldw_Server_API/app/api/v1/utils/exception_handlers.py` (`:29`, `:123`)
- Modify: `tldw_Server_API/cli/utils/llm_store.py` (`:24`)
- Test: `tldw_Server_API/tests/Chat/test_provider_authoritative_model_selection.py` (append), `tldw_Server_API/tests/Logging/test_client_disconnect_exception_handler.py` (append two), `tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py` (`:2619`, `:2678`, `:2681`, `:2696-2698`, `:2720`; append one)

**Interfaces:**
- Produces: `refuse_admin_disabled(provider: Any, request: Any) -> None` in `usage_accounting.py` (the renamed guard; same behaviour in this task). Task 5 changes its body.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/Chat/test_provider_authoritative_model_selection.py` (its `pytestmark` is `provider_inventory`, so the real model check runs against the `boundary` fixture's stubbed discovery):

```python
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("async_call", [False, True], ids=["sync", "async"])
def test_a_disabled_provider_gets_no_model_list_request(boundary, async_call, stream):
    """TASK-15582 (minor 2 of #3537's final review): the admin-override guard runs before model discovery, so a
    disabled provider gets no model-list request, and the caller sees provider_disabled."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    set_llm_provider_overrides_cache_for_tests({"openai": LLMProviderOverride(provider="openai", is_enabled=False)})
    fields = args(model="current-model", stream=stream)
    with pytest.raises(ChatConfigurationError) as refused:
        if async_call:
            asyncio.run(chat_service.perform_chat_api_call_async(**fields))
        else:
            chat_service.perform_chat_api_call(**fields)
    assert (refused.value.error_code, boundary.discovery, boundary.adapter.calls) == ("provider_disabled", [], [])
```

Append to `tests/Logging/test_client_disconnect_exception_handler.py`:

```python
@pytest.mark.asyncio
async def test_an_admin_refusal_answers_with_its_codes_own_message():
    """TASK-15582 (minor 3 of #3537's final review): a refusal rebuilt with another message (the privacy-safe rebuild
    says "Provider configuration is unavailable.") still answers with its code's own message."""
    from tldw_Server_API.app.core.exceptions import ChatConfigurationError
    from tldw_Server_API.app.core.LLM_Calls.provider_results import PROVIDER_STREAM_ERROR_MESSAGES

    rebuilt = ChatConfigurationError(
        "Provider configuration is unavailable.", provider="openai", error_code="provider_disabled"
    )
    response = await exception_handlers.admin_refusal_exception_handler(_build_request(request_id="req-msg"), rebuilt)
    assert (response.status_code, json.loads(response.body)["error"]["message"]) == (
        403,
        PROVIDER_STREAM_ERROR_MESSAGES["provider_disabled"],
    )


@pytest.mark.asyncio
async def test_the_app_passes_a_refusal_on_a_standalone_slides_route_to_the_catch_all():
    """TASK-15582 (minor 5 of #3537's final review): a standalone Slides route keeps its own redacted error shape, so
    the app's refusal handler raises the refusal on to the catch-all handler there."""
    from tldw_Server_API.app.core.exceptions import ChatConfigurationError
    from tldw_Server_API.app.main import app

    handler = app.exception_handlers[ChatConfigurationError]
    refusal = ChatConfigurationError("refused", provider="openai", error_code="provider_disabled")
    with pytest.raises(ChatConfigurationError) as passed_on:
        await handler(_build_request(path="/api/v1/slides/presentations/p-1"), refusal)
    assert passed_on.value is refusal
```

In `tests/AuthNZ_Unit/test_llm_provider_overrides.py`:
- the `tldw-evals` row of `_EVALUATION_CLIS` (`:2619`) becomes `("tldw_Server_API.cli.evals_cli", "main", "eval geval", "list-benchmarks"),`. Its LLM command is a leaf (`eval geval`): every `tldw-evals` LLM command is a group, and after this task a bare group is a help request (review round 1, B1). Its other command is `list-benchmarks` (R4);
- the three tests that invoke the LLM command split it: `CliRunner().invoke(group, [*llm_command.split()], obj={})` at `:2681` and `:2720`; in `test_an_evaluation_cli_skips_the_store_for_help_and_completion` (`:2696-2698`), `[*llm_command.split(), "--help"]` for the help call and `llm_command.split()` for `make_context` and `resolve_command`. The core CLIs' one-word commands are unchanged by the split;
- `:2678` becomes `other = CliRunner().invoke(group, [other_command], obj={})`;
- and append:

```python
@pytest.mark.parametrize(
    ("module_name", "group_name", "llm_group"),
    [
        ("tldw_Server_API.cli.evals_cli", "main", "eval"),
        ("tldw_Server_API.app.core.Evaluations.cli.evals_cli_enhanced", "cli", "recipes"),
    ],
    ids=["tldw-evals", "evals_cli_enhanced"],
)
def test_an_llm_group_given_no_subcommand_prints_help_without_the_store(monkeypatch, module_name, group_name, llm_group):
    """TASK-15582 (minor 4 of #3537's final review): an LLM group given no subcommand only prints its help (click's
    no_args_is_help, exit 2), so, like --help, it never touches the AuthNZ database."""
    from click.testing import CliRunner

    group = getattr(importlib.import_module(module_name), group_name)
    _unloaded_store_over_an_empty_database(monkeypatch)

    result = CliRunner().invoke(group, [llm_group], obj={})

    assert (result.exit_code, "Usage:" in result.output) == (2, True)
    with pytest.raises(ByokResolutionError, match="credential_store_unavailable"):
        capture_provider_override_call_snapshot("openai")
```

- [ ] **Step 2: RED.**

```bash
RUN t1red -n 2 tldw_Server_API/tests/Chat/test_provider_authoritative_model_selection.py \
  tldw_Server_API/tests/Logging/test_client_disconnect_exception_handler.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py
```

Expected (measured in review round 1): 8 failed: the 4 `test_a_disabled_provider_gets_no_model_list_request` cases (the error code matches, but `boundary.discovery` holds one discovery call), `test_an_admin_refusal_answers_with_its_codes_own_message` (message "Provider configuration is unavailable."), the 2 `test_an_llm_group_given_no_subcommand...` cases (`DID NOT RAISE`: the store loaded), and the pre-existing AF_UNIX failure of `test_two_worker_caches_converge_after_shared_store_rotation` (Global Constraints; the plan first said 7 without it). `test_the_app_passes_a_refusal_on_a_standalone_slides_route...` and the edited CLI tests pass: they are test gaps. Measure their RED by mutation:

```bash
# delete the two lines "if is_standalone_sensitive_route(...)" / "raise exc" at main.py:2085-2086, and add
# "list-benchmarks" to the llm_commands set of cli/evals_cli.py:119, then:
RUN t1mut "tldw_Server_API/tests/Logging/test_client_disconnect_exception_handler.py::test_the_app_passes_a_refusal_on_a_standalone_slides_route_to_the_catch_all" \
  "tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py::test_each_evaluation_cli_loads_the_store_for_its_llm_commands"
git restore tldw_Server_API/app/main.py tldw_Server_API/cli/evals_cli.py
git diff --stat -- tldw_Server_API/app/main.py tldw_Server_API/cli/evals_cli.py
```

Expected: 2 failed, 2 passed: the standalone-Slides test (`DID NOT RAISE`: the handler answered 403) and `test_each_evaluation_cli_loads_the_store_for_its_llm_commands[tldw-evals]` (`DID NOT RAISE`: `list-benchmarks` loaded the store); the other two CLI ids pass. The last command prints nothing.

- [ ] **Step 3: Implement.**
  - `usage_accounting.py`: rename `_refuse_admin_disabled` to `refuse_admin_disabled` (definition and the four calls). Add to its docstring: "The gateway also calls it before model discovery (TASK-15582)."
  - `gateway.py`: `from tldw_Server_API.app.core.LLM_Calls.usage_accounting import refuse_admin_disabled` with the other `tldw_Server_API.app.core` imports. In `_call_adapter` and in `_call_adapter_async`, the first statement inside the `with provider_error_privacy_scope(...)` block, before `call_started = time.monotonic()`:

```python
            # The admin override first: a disabled provider gets no model-list request (TASK-15582).
            refuse_admin_disabled(adapter, request)
```

  - `exception_handlers.py`: import `PROVIDER_STREAM_ERROR_MESSAGES` beside `ADMIN_REFUSAL_STATUS` (`:29`); `:123` becomes `content={"error": {"code": code, "message": PROVIDER_STREAM_ERROR_MESSAGES[code], "request_id": request_id}},`.
  - `llm_store.py` `resolve_command` (`:24`): load only when the command is an LLM command, not shell completion, not help, and not a group given no subcommand:

```python
        prints_help = not set(ctx.help_option_names).isdisjoint(rest) or (
            not rest and isinstance(cmd, click.Group) and cmd.no_args_is_help
        )
        if cmd_name in self.llm_commands and not ctx.resilient_parsing and not prints_help:
```

  Update the class docstring's last sentence to "A request for help (an option, or an LLM group given no subcommand) and shell completion skip the load, so they never need the AuthNZ database."

- [ ] **Step 4: GREEN, and the import isolation lint.**

```bash
RUN t1green -n 2 tldw_Server_API/tests/Chat/test_provider_authoritative_model_selection.py \
  tldw_Server_API/tests/Logging/test_client_disconnect_exception_handler.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py \
  tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py \
  tldw_Server_API/tests/lint/test_llm_calls_import_isolation.py
```

Expected (measured with the B1 edits): 1 failed, the pre-existing AF_UNIX `test_two_worker_caches_converge_after_shared_store_rotation`; netprobe lines: 0. Without the B1 edits the review measured 3 failed (the two `[tldw-evals]` CLI cases as well).

- [ ] **Step 5: Commit.**

```bash
git add tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/LLM_Calls/gateway.py \
  tldw_Server_API/app/api/v1/utils/exception_handlers.py tldw_Server_API/cli/utils/llm_store.py \
  tldw_Server_API/tests/Chat/test_provider_authoritative_model_selection.py \
  tldw_Server_API/tests/Logging/test_client_disconnect_exception_handler.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py
git commit -m "fix(llm): the four deferred minors of #3537's final review (TASK-15582)

The gateway refuses an admin-disabled provider before model discovery; the app's refusal handler
answers with the code's own message; an LLM CLI group given no subcommand prints help without the
store; and two test gaps are closed.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The `model_denied` code in every table and mapper

No check raises `model_denied` yet; this task teaches every consumer the code, so Task 3 can raise it.

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/provider_results.py` (`:40-69`)
- Modify: `tldw_Server_API/app/core/exceptions.py` (`ChatConfigurationError._ERROR_CODES` `:996-1005`, `ProviderCredentialTerminalError._ERROR_CODES` `:1085-1097`)
- Modify: `tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py` (`ProviderOverridePolicyError` `:72`)
- Modify, one set each (R10): `api/v1/endpoints/chat.py:1147`, `api/v1/endpoints/audio/audio.py:126-147`, `api/v1/endpoints/character_chat_sessions.py:352`, `api/v1/endpoints/chat_documents.py:758`, `api/v1/endpoints/chunking.py:148`, `api/v1/endpoints/embeddings_v5_production_enhanced.py:729-734, 798`, `api/v1/endpoints/messages.py:1052`, `api/v1/endpoints/prompt_studio/prompt_studio_evaluations.py:99`, `api/v1/endpoints/prompt_studio/prompt_studio_prompts.py:145-151`, `api/v1/endpoints/prompt_studio/prompt_studio_test_cases.py:102`, `core/RAG/rag_service/streaming_executor.py:68-79, 102`, `core/Streaming/speech_chat_service.py:860-864, 1005-1009`, `core/TTS/tts_jobs_worker.py:147-154`, `services/scheduled_task_automation_service.py:248, 530` (all under `tldw_Server_API/app/`)
- Modify: `apps/packages/ui/src/services/rag/provider-error-contract.ts:16`, `apps/tldw-frontend/lib/api.ts:57`, `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/errorMessages.test.ts:30-39`
- Test: `tldw_Server_API/tests/Usage/test_model_denied_codes.py` (new), `tldw_Server_API/tests/Notifications/test_scheduled_task_execution_target_bound.py` (append)

**Interfaces:**
- Produces: `ADMIN_POLICY_CODES: frozenset[str] = frozenset({"provider_disabled", "model_not_allowed", "model_denied"})` and `ADMIN_REFUSAL_STATUS` (now with `"model_denied": 403`) in `provider_results.py`; `PROVIDER_STREAM_ERROR_MESSAGES["model_denied"]`; `ProviderOverridePolicyError("model_denied", provider)` constructs.

- [ ] **Step 1: Write the failing tests.** Create `tests/Usage/test_model_denied_codes.py`. New names from `provider_results` are imported inside the tests that need them, so each mapper case fails on its own:

```python
"""Every mapper of an admin provider-override refusal knows model_denied (TASK-15582)."""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import (
    character_chat_sessions,
    chat,
    chunking,
    embeddings_v5_production_enhanced,
    messages,
)
from tldw_Server_API.app.api.v1.endpoints.audio import audio
from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified
from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_evaluations,
    prompt_studio_prompts,
    prompt_studio_test_cases,
)
from tldw_Server_API.app.core.exceptions import ChatConfigurationError, ProviderCredentialTerminalError
from tldw_Server_API.app.core.LLM_Calls.provider_results import normalize_provider_stream_error
from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.RAG.rag_service.streaming_executor import classify_rag_provider_error
from tldw_Server_API.app.core.TTS.tts_jobs_worker import _bounded_tts_credential_failure

pytestmark = pytest.mark.unit


def _denied() -> ByokResolutionError:
    """A denied model at the credential boundary, shaped as ProviderOverridePolicyError carries it."""
    error = ByokResolutionError("invalid_provider_credentials", "openai")
    error.policy_code = "model_denied"
    return error


def _http(exception: Any) -> tuple[int, str]:
    """The status and code of a mapper's HTTPException."""
    return exception.status_code, exception.detail["error_code"]


def _rag(error: BaseException) -> tuple[int, str]:
    """RAG's (code, status, message) as (status, code)."""
    code, status, _message = classify_rag_provider_error(error)
    return status, code


def _stream() -> tuple[int, str]:
    """A refusal raised mid-call, normalized for a stream."""
    normalized = normalize_provider_stream_error(ChatConfigurationError("refused", error_code="model_denied"))
    return normalized.status_code, normalized.code


_MAPPERS = {
    "chat": lambda: _http(chat._provider_credential_http_exception(_denied())),
    "messages": lambda: _http(messages._messages_credential_http_exception(_denied())),
    "character-chat": lambda: _http(character_chat_sessions._character_credential_http_exception(_denied())),
    "chunking": lambda: _http(chunking._chunking_credential_http_exception(_denied())),
    "embeddings": lambda: _http(embeddings_v5_production_enhanced._embeddings_credential_http_exception(_denied())),
    "tts": lambda: _http(audio._tts_credential_http_exception(_denied())),
    "prompt-studio-evaluations": lambda: _http(prompt_studio_evaluations._prompt_studio_credential_http_exception(_denied())),
    "prompt-studio-prompts": lambda: _http(prompt_studio_prompts._credential_http_exception(_denied())),
    "prompt-studio-test-cases": lambda: _http(prompt_studio_test_cases._credential_http_exception(_denied())),
    "rag-resolution": lambda: _rag(_denied()),
    "rag-adapter": lambda: _rag(ChatConfigurationError("refused", error_code="model_denied")),
    "evaluations": lambda: _http(
        evaluations_unified._eval_provider_http_exception(ChatConfigurationError("refused", error_code="model_denied"))
    ),
    "stream": _stream,
}


@pytest.mark.parametrize("mapper", sorted(_MAPPERS))
def test_every_mapper_answers_a_denied_model_with_403(mapper: str) -> None:
    assert _MAPPERS[mapper]() == (403, "model_denied")


def test_a_denied_model_keeps_its_code_through_every_error_type() -> None:
    from tldw_Server_API.app.core.LLM_Calls.provider_results import ADMIN_POLICY_CODES, ADMIN_REFUSAL_STATUS
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import ProviderOverridePolicyError

    assert (
        ProviderOverridePolicyError("model_denied", "openai").policy_code,
        ChatConfigurationError("refused", error_code="model_denied").error_code,
        ProviderCredentialTerminalError("model_denied").code,
        _bounded_tts_credential_failure(_denied()).failure_code,
        sorted(ADMIN_POLICY_CODES),
        ADMIN_REFUSAL_STATUS["model_denied"],
    ) == (
        "model_denied", "model_denied", "model_denied", "model_denied",
        ["model_denied", "model_not_allowed", "provider_disabled"], 403,
    )


def test_the_allowlist_refusal_says_the_users_own_key_may_serve_the_model() -> None:
    """TASK-15582: a model off allowed_models is only not offered on the server's key."""
    from tldw_Server_API.app.core.LLM_Calls.provider_results import PROVIDER_STREAM_ERROR_MESSAGES

    assert (PROVIDER_STREAM_ERROR_MESSAGES["model_not_allowed"], PROVIDER_STREAM_ERROR_MESSAGES["model_denied"]) == (
        "The selected model is not offered on the server's key for this provider. Your own provider key may serve it.",
        "The selected model is disabled by administrator policy.",
    )
```

Append to `tests/Notifications/test_scheduled_task_execution_target_bound.py`:

```python
def test_admin_denied_model_is_a_validation_error_on_the_model(tmp_path, monkeypatch):
    """TASK-15582: a denied model is attributed to input.model, as a model off the allowlist is."""
    _bound(monkeypatch, _LISTING)
    monkeypatch.setattr(
        service_module,
        "validate_provider_override",
        lambda p, m: {"error_code": "model_denied"} if (p == "openai" and m == "gpt-4o") else None,
    )
    service, _repo = _service(tmp_path)

    preview = service.create_preview(owner_id=OWNER_ID, actor=ACTOR, payload=_preview_payload("openai", "gpt-4o"))

    assert (preview.status, [(e["field"], e["code"]) for e in preview.validation_errors]) == (
        "invalid",
        [("input.model", "model_denied")],
    )  # nosec B101
```

- [ ] **Step 2: RED.**

```bash
RUN t2red -n 2 tldw_Server_API/tests/Usage/test_model_denied_codes.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_execution_target_bound.py
```

Expected: 16 failed: the 13 mapper cases (each answers 503, or coerces the code to `provider_configuration_invalid`/`invalid_provider_credentials`), the two table tests (`ImportError` for `ADMIN_POLICY_CODES`; `KeyError: 'model_denied'`), and the scheduler test (`input.provider`).

- [ ] **Step 3: Implement.**
  - `provider_results.py`: in `PROVIDER_STREAM_ERROR_MESSAGES`, `"model_not_allowed": "The selected model is not offered on the server's key for this provider. Your own provider key may serve it."` and, after it, `"model_denied": "The selected model is disabled by administrator policy."`; add `"model_denied": 403` to `_PROVIDER_STREAM_ERROR_STATUS`; replace `ADMIN_REFUSAL_STATUS` (`:68-69`) with:

```python
# The admin provider-override policy refusals (TASK-15581, TASK-15582): each a 403, on whichever key refuses it.
ADMIN_POLICY_CODES = frozenset({"provider_disabled", "model_not_allowed", "model_denied"})
# The admin refusals the provider base class raises, with their HTTP status (an unreadable store is a 503).
ADMIN_REFUSAL_STATUS = {**dict.fromkeys(ADMIN_POLICY_CODES, 403), "credential_store_unavailable": 503}
```

  - `exceptions.py`: add `"model_denied",` after `"model_not_allowed",` in both `_ERROR_CODES`.
  - `llm_provider_overrides.py:72`: `if policy_code not in {"provider_disabled", "model_not_allowed", "model_denied"}:`.
  - In each mapper file, import `ADMIN_POLICY_CODES` from `tldw_Server_API.app.core.LLM_Calls.provider_results` (extend an existing import from that module where there is one), then:
    - `chat.py:1147`, `chunking.py:148`, `messages.py:1052`, `embeddings_v5_production_enhanced.py:798`, `streaming_executor.py:102`: the literal `{"provider_disabled", "model_not_allowed"}` becomes `ADMIN_POLICY_CODES`.
    - `audio.py:145`, `character_chat_sessions.py:352`, `chat_documents.py:758`, `prompt_studio_evaluations.py:99`, `prompt_studio_test_cases.py:102`, `prompt_studio_prompts.py:145-151` and `speech_chat_service.py:860-864` and `:1005-1009`: the literal set of `provider_disabled`, `model_not_allowed` and `credential_scope_revoked` becomes `ADMIN_POLICY_CODES | {"credential_scope_revoked"}` on one line.
    - `audio.py` `_TTS_CREDENTIAL_ERROR_MESSAGES` and `embeddings_v5_production_enhanced.py` `_EMBEDDINGS_CREDENTIAL_ERROR_MESSAGES`: add `"model_denied": "The selected model is disabled by administrator policy.",`.
    - `streaming_executor.py` `_RAG_PROVIDER_ERROR_MESSAGES`: the two texts of `provider_results.py` for `model_not_allowed` and `model_denied`.
  - `tts_jobs_worker.py:147-154`: add `"model_denied",` to the literal set (no import).
  - No edit, they follow `ADMIN_REFUSAL_STATUS` and the shared table: `evaluations_unified._eval_provider_http_exception` (in the test), `media/document_insights.py:328`, `llamacpp.py:1296-1307`, the app's refusal handler, MCP `model_completion` (it publishes `policy_code`), `_summary_error_code` (`configuration`), the Slides standalone provider (`standalone_html_model_denied`, terminal like the other disables) and Data Tables (`DataTablesJobError("model_denied")`, not retried).
  - `scheduled_task_automation_service.py:248`: `if error_code in {"model_not_allowed", "model_denied"} and model is not None:`; `:530`: `hard_codes = {"unusable", "provider_disabled", "model_not_allowed", "model_denied"}`. Its docstring at `:174-176` gains "a model on its `denied_models`".
  - `apps/packages/ui/src/services/rag/provider-error-contract.ts` and `apps/tldw-frontend/lib/api.ts`: `model_not_allowed` gets the new text; add `model_denied: "The selected model is disabled by administrator policy.",` (quote style of each file). In `errorMessages.test.ts`, the `model_not_allowed` row expects the new text, and a third row is `["model_denied", "The selected model is disabled by administrator policy."]`.

- [ ] **Step 4: GREEN.**

```bash
RUN t2green -n 2 tldw_Server_API/tests/Usage tldw_Server_API/tests/Notifications/test_scheduled_task_execution_target_bound.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_provider_override_store_boundary.py \
  tldw_Server_API/tests/Chat/integration/test_chat_provider_override_store_boundary.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_provider_override_store_boundary.py
grep -c "provider_disabled\", \"model_not_allowed\"" tldw_Server_API/app/api/v1/endpoints/chat.py
```

Expected (measured in review round 1): 0 failed, 779 passed, 1 skipped; netprobe lines: 1, the known tiktoken download in `test_quota_evals_chatbooks.py` (Global Constraints); the grep prints `0`. The `apps/` edits are measured by CI (`frontend-required`); do not install `apps/` dependencies.

- [ ] **Step 5: Commit.**

```bash
git add -A tldw_Server_API/app apps/packages/ui/src/services/rag/provider-error-contract.ts apps/tldw-frontend/lib/api.ts \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/errorMessages.test.ts \
  tldw_Server_API/tests/Usage/test_model_denied_codes.py tldw_Server_API/tests/Notifications/test_scheduled_task_execution_target_bound.py
git commit -m "feat(llm): the model_denied refusal code in every table and mapper (TASK-15582)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `denied_models` in the override, and the `allowlist` keyword

Every caller keeps the default `allowlist=True`, so behaviour changes only by the new denylist (and the listing). Task 5 passes the key.

**Files:**
- Modify: `tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py` (dataclass `:40-51`; `_override_allows_model` `:85-89`; `policy_error`/`enforce` `:106-130`; `_parse_override_row` `:691-745`; `apply_llm_provider_overrides_to_listing` `:881-966`; `validate_provider_override`/`enforce_provider_override` `:969-988`)
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py` (append), `tldw_Server_API/tests/Slides/test_standalone_html_provider.py` (`:642-650`), `tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py` (`:593-601`)

**Interfaces:**
- Consumes: Task 2's codes.
- Produces:
  - `LLMProviderOverride.denied_models: list[str] | None = None`, right after `allowed_models`.
  - `ProviderOverrideCallSnapshot.policy_error(model: str | None, *, allowlist: bool = True) -> dict[str, str] | None`, precedence `provider_disabled` > `model_denied` > `model_not_allowed` (the last only with `allowlist`).
  - `ProviderOverrideCallSnapshot.enforce(model: str | None, *, require_model: bool = False, allowlist: bool = True) -> None`; `require_model` refuses a model-less call only with `allowlist` and an allowlist set.
  - `validate_provider_override(provider: str, model: str | None, *, allowlist: bool = True) -> dict[str, str] | None` and `enforce_provider_override(provider: str, model: str | None, *, allowlist: bool = True) -> None`.
  - `_model_denied(provider: str, denied_models: list[str], model: object) -> bool`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/AuthNZ_Unit/test_llm_provider_overrides.py` (its top-level imports already give `LLMProviderOverride`, `ByokResolutionError`, `validate_provider_override`, `apply_llm_provider_overrides_to_listing`, `set_llm_provider_overrides_cache_for_tests`, `importlib`):

```python
def test_a_denied_model_is_refused_on_every_key_and_wins_over_the_allowlist() -> None:
    """TASK-15582 (owner, 2026-11-21): denied_models is the explicit model disallow, refused on every key; a model off
    allowed_models is refused only where the allowlist applies. A model on both lists is denied."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import ProviderOverrideCallSnapshot

    snapshot = ProviderOverrideCallSnapshot(
        "openai",
        LLMProviderOverride(provider="openai", allowed_models=["gpt-4o", "gpt-4o-mini"], denied_models=["gpt-4o"]),
    )
    codes = {
        (model, allowlist): (snapshot.policy_error(model, allowlist=allowlist) or {}).get("error_code")
        for model in ("gpt-4o", "gpt-4o-mini", "o3")
        for allowlist in (True, False)
    }
    assert codes == {
        ("gpt-4o", True): "model_denied", ("gpt-4o", False): "model_denied",
        ("gpt-4o-mini", True): None, ("gpt-4o-mini", False): None,
        ("o3", True): "model_not_allowed", ("o3", False): None,
    }


def test_the_module_checks_read_the_store_and_take_the_key() -> None:
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import enforce_provider_override

    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", allowed_models=["gpt-4o-mini"], denied_models=["gpt-4o"])}
    )
    enforce_provider_override("openai", "o3", allowlist=False)
    with pytest.raises(ByokResolutionError) as refused:
        enforce_provider_override("openai", "gpt-4o", allowlist=False)
    assert (
        validate_provider_override("openai", "o3")["error_code"],
        validate_provider_override("openai", "o3", allowlist=False),
        refused.value.policy_code,
    ) == ("model_not_allowed", None, "model_denied")


def test_a_denied_model_is_denied_whatever_its_case_or_anthropic_spelling() -> None:
    """TASK-15582 (Review Focus 1): a denial errs wide, so another case of a denied ID, or the dotted and dashed
    spellings of one Claude model, is denied too. The allowlist stays exact."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import ProviderOverrideCallSnapshot

    openai = ProviderOverrideCallSnapshot("openai", LLMProviderOverride(provider="openai", denied_models=["gpt-4o"]))
    anthropic = ProviderOverrideCallSnapshot(
        "anthropic", LLMProviderOverride(provider="anthropic", denied_models=["claude-sonnet-4.5"])
    )
    cases = ((openai, "GPT-4o"), (openai, "gpt-4o"), (anthropic, "claude-sonnet-4-5"),
             (anthropic, "Claude-Sonnet-4.5"), (openai, "gpt-4o-mini"))
    codes = [(snapshot.policy_error(model, allowlist=False) or {}).get("error_code") for snapshot, model in cases]
    assert codes == ["model_denied", "model_denied", "model_denied", "model_denied", None]


def test_with_only_a_denylist_a_call_that_names_no_model_runs() -> None:
    """TASK-15582 (Review Focus 4): with no model there is nothing to deny, and require_model holds a model-less call
    to the allowlist only where the allowlist applies."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import ProviderOverrideCallSnapshot

    denying = ProviderOverrideCallSnapshot("openai", LLMProviderOverride(provider="openai", denied_models=["gpt-4o"]))
    allowing = ProviderOverrideCallSnapshot(
        "openai", LLMProviderOverride(provider="openai", allowed_models=["gpt-4o-mini"])
    )
    denying.enforce(None, require_model=True)
    allowing.enforce(None, require_model=True, allowlist=False)
    with pytest.raises(ByokResolutionError) as refused:
        allowing.enforce(None, require_model=True)
    assert refused.value.policy_code == "model_not_allowed"


def test_a_stored_denylist_that_does_not_parse_fails_its_row() -> None:
    """TASK-15582 (Review Focus 5): like allowed_models, a corrupt denied_models fails its row, so the store fails
    closed instead of dropping the denial; entries are trimmed and blanks dropped."""
    module = importlib.import_module("tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides")

    parsed = module._parse_override_row({"provider": "openai", "denied_models": '[" gpt-4o ", ""]'})
    for raw in ('{"gpt-4o": true}', "[not json", '[["nested"]]'):
        with pytest.raises(ValueError, match="denied_models"):
            module._parse_override_row({"provider": "openai", "denied_models": raw})
    assert parsed.denied_models == ["gpt-4o"]


def test_the_server_key_listing_hides_denied_models() -> None:
    """TASK-15582 (R8): a model refused on every key is not offered, and a denied default falls back."""
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", denied_models=["GPT-4o"])}
    )
    entry = apply_llm_provider_overrides_to_listing(
        {
            "providers": [
                {
                    "name": "openai",
                    "models": ["gpt-4o", "gpt-4o-mini"],
                    "models_info": [{"name": "gpt-4o"}, {"name": "gpt-4o-mini"}],
                    "default_model": "gpt-4o",
                }
            ]
        }
    )["providers"][0]
    assert (entry["models"], [m["name"] for m in entry["models_info"]], entry["default_model"]) == (
        ["gpt-4o-mini"],
        ["gpt-4o-mini"],
        "gpt-4o-mini",
    )
```

In `tests/Slides/test_standalone_html_provider.py`, add a row to the parametrization of `test_an_admin_disabled_provider_or_model_fails_before_network` (`:642-650`), after the `model-not-allowed` row, with id `"model-denied"` added to `ids`:

```python
        ({"denied_models": ["casesensitive-model"]}, True, "standalone_html_model_denied"),
```

In `tests/Usage/test_llm_quota_byok_calls.py`, add a row to the parametrization of `test_an_endpoint_that_maps_nothing_answers_with_the_refusals_own_status` (`:593-601`), with id `"model-denied"`:

```python
        ({"denied_models": ["m1"]}, True, 403, "model_denied"),
```

- [ ] **Step 2: RED.**

```bash
RUN t3red -n 2 tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py \
  "tldw_Server_API/tests/Slides/test_standalone_html_provider.py::test_an_admin_disabled_provider_or_model_fails_before_network"
```

Expected (measured in review round 1): 9 failed: the 6 new tests (`TypeError: ... unexpected keyword argument 'denied_models'`, or `AttributeError: ... 'denied_models'` for the parse test), the Slides `model-denied` row and the `model-denied` endpoint row (both `TypeError`), and the pre-existing AF_UNIX failure (Global Constraints; the plan first said 8 without it).

- [ ] **Step 3: Implement** in `llm_provider_overrides.py`.
  - Dataclass: `denied_models: list[str] | None = None` after `allowed_models`.
  - After `_override_allows_model`:

```python
def _model_denied(provider: str, denied_models: list[str], model: object) -> bool:
    """Whether ``model`` is on a denylist: case does not matter (a denial errs wide), and Anthropic's dotted and
    dashed spellings of one ID count as one (TASK-15582). A model that is not a string is not denied."""
    if not isinstance(model, str):
        return False
    return _override_allows_model(provider, [entry.casefold() for entry in denied_models], model.casefold())
```

  - `policy_error` and `enforce`:

```python
    def policy_error(self, model: str | None, *, allowlist: bool = True) -> dict[str, str] | None:
        """Return the bounded policy decision from this captured snapshot (TASK-15581, TASK-15582).

        A disabled provider and a denied model are refused on every key. ``allowed_models`` is what the server's key
        offers: ``allowlist=False`` skips it, for a call on the user's own key at its vendor (credential source
        ``user``), or for a check made before the call's key is known when a later check sees the key.
        """
        override = self._override
        if override is None:
            return None
        if override.is_enabled is False:
            return {
                "error_code": "provider_disabled",
                "message": f"Provider '{self.provider}' is disabled by admin override.",
            }
        if override.denied_models and _model_denied(self.provider, override.denied_models, model):
            return {
                "error_code": "model_denied",
                "message": f"Model '{model}' is disabled for provider '{self.provider}' by admin override.",
            }
        allowed_models = override.allowed_models
        if allowlist and allowed_models and model and not _override_allows_model(self.provider, allowed_models, model):
            return {
                "error_code": "model_not_allowed",
                "message": (
                    f"Model '{model}' is not offered on the server's key for provider '{self.provider}'; "
                    "your own provider key may serve it."
                ),
            }
        return None

    def enforce(self, model: str | None, *, require_model: bool = False, allowlist: bool = True) -> None:
        """Fail closed; final dispatch can require a known model where the allowlist applies."""
        error = self.policy_error(model, allowlist=allowlist)
        if error is not None:
            raise ProviderOverridePolicyError(error["error_code"], self.provider)
        if require_model and allowlist and not model and self._override and self._override.allowed_models:
            raise ProviderOverridePolicyError("model_not_allowed", self.provider)
```

  - `_parse_override_row`: move the `allowed_models` parsing (`:693-702`) into `_parse_models_column(row: dict[str, Any], name: str) -> list[str] | None` (the same three steps, with `field_name=name` and the entry error `f"Invalid stored {name} entry"`), and use it for `allowed_models` and `denied_models`; pass `denied_models=` to the dataclass.
  - `apply_llm_provider_overrides_to_listing`: after the `allowed_models` filter (`:916-921`):

```python
        if override.denied_models:
            models = [m for m in models if not _model_denied(provider_name, override.denied_models, m)]
```

  Widen the `models_info` filter's condition (`:933`) to `(override.allowed_models or override.denied_models or isinstance(config_models, list))`. After the default-model block (`:955-960`):

```python
        if override.denied_models and _model_denied(provider_name, override.denied_models, merged.get("default_model")):
            merged["default_model"] = models[0] if models else None
```

  - `validate_provider_override(provider, model, *, allowlist=True)` passes `allowlist=allowlist` to `policy_error`; `enforce_provider_override(provider, model, *, allowlist=True)` passes it to `enforce`, and its docstring says: "A disabled provider and a denied model are refused on every key; a model off ``allowed_models`` only with ``allowlist`` (TASK-15582)."

- [ ] **Step 4: GREEN.**

```bash
RUN t3green -n 2 tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py tldw_Server_API/tests/Usage \
  tldw_Server_API/tests/Chat/unit/test_anthropic_dotted_model_names.py tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py \
  "tldw_Server_API/tests/Slides/test_standalone_html_provider.py::test_an_admin_disabled_provider_or_model_fails_before_network"
```

Expected (measured in review round 1): 1 failed (the pre-existing AF_UNIX case), 924 passed; netprobe lines: 1 (the known tiktoken download). Both are the Global Constraints' artefacts; the plan first said 0 and 0.

- [ ] **Step 5: Commit.**

```bash
git add tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py tldw_Server_API/tests/Slides/test_standalone_html_provider.py \
  tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py
git commit -m "feat(llm): denied_models on the provider override, refused on every key (TASK-15582)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Persist `denied_models`, and set it through the admin API

**Files:**
- Modify: `tldw_Server_API/app/core/Provider_Credentials/repos/llm_provider_overrides_repo.py` (`_PATCHABLE_OVERRIDE_COLUMNS` `:19-25`, `_OVERRIDE_RESULT_COLUMNS` `:26-29`, the SELECT lists at `:181`, `:405`, `:437`)
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py` (new function before `migration_097_backfill_user_uuid` `:6603`; registry entry after 106, `:6583`)
- Modify: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py` (`_CREATE_LLM_PROVIDER_OVERRIDES` `:1887`)
- Modify: `tldw_Server_API/app/api/v1/schemas/admin_schemas.py` (`:384-408`)
- Modify: `tldw_Server_API/app/services/admin_llm_providers_service.py` (`build_override_response` `:124-135`, `upsert_override` `:209-218` and `:266-270`)
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_mcp_discovery_migration.py` (`:41`, `:97`: the registry now ends at 107)
- Modify: `tldw_Server_API/tests/Admin/test_admin_llm_provider_overrides_concurrency.py` (`_override_response()` `:145-156`: the stand-in gains `denied_models=None`)
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_provider_overrides_repo_sqlite.py` (append two), `tldw_Server_API/tests/AuthNZ_Postgres/test_provider_credentials_storage_pg.py` (append), `tldw_Server_API/tests/Admin/test_admin_llm_provider_denied_models.py` (new)

**Interfaces:**
- Consumes: Task 3's dataclass field.
- Produces: `migration_107_add_override_denied_models(conn: sqlite3.Connection) -> None`; the repo patches and returns `denied_models` (a JSON text column); `LLMProviderOverrideRequest.denied_models` and `LLMProviderOverrideResponse.denied_models`, both `list[str] | None = None`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/AuthNZ_SQLite/test_authnz_llm_provider_overrides_repo_sqlite.py`:

```python
def test_migration_107_adds_denied_models_once(tmp_path) -> None:
    """TASK-15582: an AuthNZ database from before the denylist gains the column, its rows keep their values, and a
    second run changes nothing."""
    import sqlite3

    from tldw_Server_API.app.core.AuthNZ import migrations

    with sqlite3.connect(tmp_path / "users.db") as conn:
        migrations.migration_041_add_llm_provider_overrides(conn)
        conn.execute("INSERT INTO llm_provider_overrides (provider, allowed_models) VALUES ('openai', '[\"gpt-4o\"]')")
        migrations.migration_107_add_override_denied_models(conn)
        migrations.migration_107_add_override_denied_models(conn)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(llm_provider_overrides)")]
        row = conn.execute("SELECT allowed_models, denied_models FROM llm_provider_overrides").fetchone()
    assert (columns.count("denied_models"), row) == (1, ('["gpt-4o"]', None))


@pytest.mark.asyncio
async def test_the_denylist_round_trips_through_the_repo_sqlite(tmp_path, monkeypatch) -> None:
    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)

    stored = await repo.patch_override(
        provider="openai", fields={"denied_models": json.dumps(["gpt-4o"])}, updated_at=datetime.now(timezone.utc)
    )
    listed = await repo.list_overrides("openai")

    assert (stored["denied_models"], listed[0]["denied_models"]) == ('["gpt-4o"]', '["gpt-4o"]')
```

Append to `tests/AuthNZ_Postgres/test_provider_credentials_storage_pg.py`:

```python
@pytest.mark.asyncio
async def test_the_denylist_round_trips_through_the_repo_postgres(test_db_pool) -> None:
    """TASK-15582: the PostgreSQL ensure adds llm_provider_overrides.denied_models, and the repo writes and reads it."""
    from datetime import datetime, timezone

    from tldw_Server_API.app.core.Provider_Credentials.repos.llm_provider_overrides_repo import (
        AuthnzLLMProviderOverridesRepo,
    )

    assert await ensure_llm_provider_overrides_pg(test_db_pool) is True
    repo = AuthnzLLMProviderOverridesRepo(test_db_pool)
    stored = await repo.patch_override(
        provider="openai", fields={"denied_models": '["gpt-4o"]'}, updated_at=datetime.now(timezone.utc)
    )
    listed = await repo.list_overrides("openai")
    assert (stored["denied_models"], listed[0]["denied_models"]) == ('["gpt-4o"]', '["gpt-4o"]')
```

Create `tests/Admin/test_admin_llm_provider_denied_models.py`:

```python
"""The admin API sets, clears and returns a provider override's denied_models (TASK-15582)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.admin_schemas import LLMProviderOverrideRequest
from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import LLMProviderOverride
from tldw_Server_API.app.services import admin_llm_providers_service as service

pytestmark = pytest.mark.unit


class _RecordingRepo:
    """Records the columns each admin write patches."""

    def __init__(self) -> None:
        self.patches: list[dict[str, Any]] = []

    async def fetch_override(self, provider: str) -> None:
        return None

    async def patch_override(
        self, *, provider: str, fields: dict[str, Any], updated_at: datetime,
        compare_secret_blob: bool = False, expected_secret_blob: str | None = None,
    ) -> dict[str, Any]:
        self.patches.append(fields)
        return {"provider": provider, **fields}


@pytest.mark.asyncio
async def test_an_admin_sets_clears_and_reads_back_denied_models(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _RecordingRepo()

    async def get_repo() -> _RecordingRepo:
        return repo

    async def no_refresh(*, force: bool = False) -> None:
        return None

    monkeypatch.setattr(service, "get_llm_provider_overrides_repo", get_repo)
    monkeypatch.setattr(service, "_refresh_overrides_or_503", no_refresh)
    monkeypatch.setattr(
        service,
        "get_llm_provider_override",
        lambda _provider: LLMProviderOverride(provider="openai", denied_models=["gpt-4o", "o1"]),
    )

    response = await service.upsert_override("openai", LLMProviderOverrideRequest(denied_models=[" gpt-4o ", "", "o1"]))
    await service.upsert_override("openai", LLMProviderOverrideRequest(denied_models=[]))

    assert (repo.patches, response.denied_models) == (
        [{"denied_models": '["gpt-4o", "o1"]'}, {"denied_models": None}],
        ["gpt-4o", "o1"],
    )
```

- [ ] **Step 2: RED.**

```bash
RUN t4red -n 2 tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_provider_overrides_repo_sqlite.py \
  tldw_Server_API/tests/Admin/test_admin_llm_provider_denied_models.py
TLDW_TEST_NO_DOCKER=1 TEST_DB_NAME=tldw_test_15582_$$ RUN t4pgred -p no:xdist tldw_Server_API/tests/AuthNZ_Postgres/test_provider_credentials_storage_pg.py
```

Expected: first run 3 failed: the migration test (`AttributeError: ... migration_107_add_override_denied_models`), the SQLite round trip (`ValueError: Provider override patch fields are invalid`) and the admin test (`HTTPException` 400 "No override fields supplied": the request drops the unknown field). Second run: with no local cluster on :5432, 4 skipped; with one, 1 failed (the same `ValueError`) and 3 passed.

- [ ] **Step 3: Implement.**
  - Repo: add `"denied_models"` to `_PATCHABLE_OVERRIDE_COLUMNS` (after `"allowed_models"`) and to `_OVERRIDE_RESULT_COLUMNS` (after `allowed_models`); add `denied_models,` after `allowed_models,` in the SELECT lists at `:181` (`list_overrides`) and `:405`, `:437` (`patch_override`'s SQLite read-backs). `upsert_override` is left as it is (no admin caller; its rows parse with `denied_models=None`).
  - `migrations.py`, before `migration_097_backfill_user_uuid`:

```python
def migration_107_add_override_denied_models(conn: sqlite3.Connection) -> None:
    """Add ``llm_provider_overrides.denied_models``: models the admin refuses on every key (TASK-15582)."""
    if not _sqlite_table_exists(conn, "llm_provider_overrides"):
        logger.info("Migration 107: llm_provider_overrides table not present; skipping")
        return
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(llm_provider_overrides)").fetchall()}
    if "denied_models" not in columns:
        conn.execute("ALTER TABLE llm_provider_overrides ADD COLUMN denied_models TEXT")
    logger.info("Migration 107: Added llm_provider_overrides.denied_models")
```

  and, after the `Migration(106, ...)` entry, one line: `Migration(107, "Add llm_provider_overrides.denied_models", migration_107_add_override_denied_models),`.
  - `test_mcp_discovery_migration.py`: `:41` becomes `assert versions[-7:] == [101, 102, 103, 104, 105, 106, 107]` (same comment); `:97` compares with `107` and its comment reads "Normal registry 105-107 completes without renumbering incoming migrations."
  - `pg_migrations_extra.py`, after the `allowed_models` line (`:1887`): `("ALTER TABLE llm_provider_overrides ADD COLUMN IF NOT EXISTS denied_models TEXT", ()),`.
  - `admin_schemas.py`: `denied_models: list[str] | None = None` after `allowed_models` in both models, with a comment on the request field: "# Refused on every key; wins over allowed_models, which only the server's key follows (TASK-15582)."
  - Service: add `and payload.denied_models is None` to the "No override fields supplied" condition; after the `allowed_models` block (`:266-270`):

```python
        if payload.denied_models is not None:
            denied = normalize_allowed_models(payload.denied_models)
            patch_fields["denied_models"] = json.dumps(denied) if denied else None
```

  and `denied_models=override.denied_models,` in `build_override_response`.
  - `tests/Admin/test_admin_llm_provider_overrides_concurrency.py` `_override_response()` (`:145-156`): add `denied_models=None,` after `allowed_models=None,`. That `SimpleNamespace` stands in for the stored override, and `build_override_response` now reads the field (review round 1, B2).

- [ ] **Step 4: GREEN.** `test_provider_usage_reservations_repo.py` runs alone without xdist: it parametrizes with a fresh `uuid4`, so two xdist workers collect different ids and the run errors before any test (pre-existing; review round 1, B2).

```bash
RUN t4green -n 2 tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_provider_overrides_repo_sqlite.py \
  tldw_Server_API/tests/Admin/test_admin_llm_provider_denied_models.py tldw_Server_API/tests/Admin/test_admin_llm_provider_overrides_concurrency.py \
  tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py tldw_Server_API/tests/AuthNZ/unit/test_mcp_discovery_migration.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_llm_provider_overrides_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_storage_quota_backfill_migration.py tldw_Server_API/tests/lint/test_function_size_ratchet.py
RUN t4resv -p no:xdist tldw_Server_API/tests/AuthNZ_Unit/test_provider_usage_reservations_repo.py
TLDW_TEST_NO_DOCKER=1 TEST_DB_NAME=tldw_test_15582_$$ RUN t4pggreen -p no:xdist tldw_Server_API/tests/AuthNZ_Postgres/test_provider_credentials_storage_pg.py
```

Expected (measured in review round 1): first run 0 failed, netprobe lines 0 (`get_authnz_migrations` is 440 lines, under its 443 cap; without the `_override_response()` field it was 3 failed). Second run: 89 passed. Third run: 4 passed with the cluster on :5432 up (measured), or 4 skipped with none (CI's PostgreSQL jobs are then the measure; the controller checks them before merge).

- [ ] **Step 5: Commit.**

```bash
git add tldw_Server_API/app/core/Provider_Credentials/repos/llm_provider_overrides_repo.py tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py tldw_Server_API/app/api/v1/schemas/admin_schemas.py \
  tldw_Server_API/app/services/admin_llm_providers_service.py tldw_Server_API/tests/AuthNZ/unit/test_mcp_discovery_migration.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_provider_overrides_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ_Postgres/test_provider_credentials_storage_pg.py tldw_Server_API/tests/Admin/test_admin_llm_provider_denied_models.py \
  tldw_Server_API/tests/Admin/test_admin_llm_provider_overrides_concurrency.py
git commit -m "feat(admin): store and set an override's denied_models; SQLite migration 107 (TASK-15582)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `allowed_models` governs the server's key only

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` (`refuse_admin_disabled`)
- Modify: `tldw_Server_API/app/core/Provider_Credentials/provider_credential_runtime.py` (import `:33`; protocol `:42`; `_resolve_entry` `:807-815`; `_capture_override_snapshot` `:1065-1068`; `_enforce_call_policy` `:1084-1090`)
- Modify: `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py` (`:60`, `:318-323`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py` (`:337` at the base, `:338` after Task 2's added import; `:4469`, `:5295`)
- Modify: `tldw_Server_API/app/core/Chat/chat_target_resolution.py` (`resolve_chat_target` `:186-207`), `tldw_Server_API/app/api/v1/API_Deps/study_assistant_deps.py` (`:62`), `tldw_Server_API/app/core/Flashcards/study_assistant.py` (`:465`)
- Modify: `tldw_Server_API/app/core/Audio/tts_service.py` (`tts_provider_credential_scope` `:568-630`: the check before the runtime, review round 1, I1)
- Modify (test doubles that must take the keyword, R5): `tests/Audio/test_audio_tts_oauth_retry_unit.py:51`, `tests/Audio/test_failopen_cap_minutes.py:24`, `tests/Audio/test_tts_credential_runtime_boundary.py:59`, `tests/Audio/test_tts_endpoint_hardening_boundaries.py:950`, `tests/Audio/test_tts_runtime_hardening_boundaries.py:61`, `tests/Chat/integration/test_messages_provider_override_store_boundary.py:199, 303`, `tests/Chunking/test_chunking_endpoint.py:167`, `tests/Chunking/test_chunking_runtime_lifecycle.py:245`, `tests/Embeddings/test_embeddings_provider_override_store_boundary.py:595`, `tests/RAG_NEW/unit/test_rag_provider_credentials.py:287`, `tests/Chat/integration/test_chat_provider_override_store_boundary.py:457-458`, `tests/Chat/test_chat_target_resolution.py:201` and `:268`, `tests/Sharing/test_shared_workspace_recipient_security_matrix.py:568` (all under `tldw_Server_API/`; the last two from review round 1, B3)
- Modify (tests whose own-key allowlist refusal flips, rewritten to the denylist or split by key): `tests/Usage/test_llm_quota_byok_calls.py` (`:479`, `:673-711`), `tests/AuthNZ_Unit/test_provider_credential_runtime.py` (`:545-655`), `tests/LLM_Calls/test_provider_adapter_runtime_boundary.py` (`:539-606`), `tests/LLM_Calls/test_summarization_adapter.py:845-847`, `tests/Usage/test_llm_quota_decision.py:383`
- Test (new): the files above, `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py`, `tests/Chat/integration/test_chat_provider_override_store_boundary.py`, `tests/Flashcards/test_study_assistant_adapter_contract.py`, `tests/Audio/test_tts_credential_runtime_boundary.py`

**Interfaces:**
- Consumes: Task 3's `allowlist` keyword; Task 1's `refuse_admin_disabled`.
- Produces: `resolve_chat_target(*, requested_provider: str | None, requested_model: str | None, allowlist: bool = True) -> ResolvedChatTarget`; the runtime protocol `_ProviderOverrideSnapshot.enforce(self, model: str | None, *, allowlist: bool = True) -> None`.

- [ ] **Step 1: Write the new tests.**

In `tests/Usage/test_llm_quota_byok_calls.py`, after `test_an_admin_disabled_provider_or_model_is_refused_on_every_key`:

```python
@_METHODS
@pytest.mark.parametrize(
    ("source", "expected"),
    [("user", None), ("team", "model_not_allowed"), ("org", "model_not_allowed"),
     ("server_default", "model_not_allowed"), (None, "model_not_allowed")],
    ids=["user", "team", "org", "server", "no-handle"],
)
async def test_a_model_off_the_allowlist_runs_only_on_the_users_own_key(method, source, expected) -> None:
    """TASK-15582 (owner, 2026-11-21, "Server key only"): allowed_models is what the server's key offers. The user's
    own key at its vendor runs a model off it; every other key, team and org included (R1), is refused."""
    _admin_override(allowed_models=["x"])
    request = dict(REQUEST)
    if source is not None:
        request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] = await _handle(source)
    try:
        await _call(method, request)
        outcome = None
    except ChatConfigurationError as refusal:
        outcome = refusal.error_code
    assert outcome == expected


async def test_the_users_key_at_an_operator_endpoint_is_held_to_the_allowlist() -> None:
    """TASK-15582 (Review Focus 2): a user key sent off its vendor's host reaches the operator's endpoint, so it is
    held to the server's allowlist, as ADR-083 counts it (user_unmetered)."""
    _admin_override(allowed_models=["x"])
    request = {
        **REQUEST, "base_url": "https://models.operator.example/v1",
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user"),
    }
    with pytest.raises(ChatConfigurationError) as refusal:
        _Provider().chat(request)
    assert refusal.value.error_code == "model_not_allowed"


async def test_openrouter_fallbacks_on_the_users_own_key_skip_the_allowlist_but_not_the_denylist() -> None:
    """TASK-15582 (Review Focus 3): each OpenRouter fallback model is checked like the primary, by the call's key."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    class _OpenRouter(_Provider):
        name = "openrouter"

    set_llm_provider_overrides_cache_for_tests({"openrouter": LLMProviderOverride(
        provider="openrouter", allowed_models=["openai/gpt-4o-mini"], denied_models=["anthropic/claude-opus-4.1"],
    )})
    request = {**REQUEST, "model": "openai/gpt-4o-mini",
               PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await _handle("user", provider="openrouter")}
    served = _OpenRouter().chat({**request, "extra_body": {"models": ["meta-llama/llama-3.1-8b-instruct"]}})
    with pytest.raises(ChatConfigurationError) as refusal:
        _OpenRouter().chat({**request, "extra_body": {"models": ["anthropic/claude-opus-4.1"]}})
    assert (served["choices"][0]["message"]["content"], refusal.value.error_code) == ("hi", "model_denied")
```

In `tests/AuthNZ_Unit/test_provider_credential_runtime.py`, after `test_implicit_model_policy_uses_numbered_custom_openai_section`:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "app_config", "expected"),
    [
        ("user", None, "user"),
        ("user", {"openai_api": {"api_base_url": "https://models.operator.example/v1"}}, "model_not_allowed"),
        ("team", None, "model_not_allowed"),
        ("org", None, "model_not_allowed"),
        ("server_default", None, "model_not_allowed"),
    ],
    ids=["user", "user-at-operator-host", "team", "org", "server"],
)
async def test_a_model_off_the_allowlist_resolves_only_on_the_users_own_key(source, app_config, expected) -> None:
    """TASK-15582: the runtime checks the allowlist after resolution, against the source the handle will carry."""

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=provider, api_key=SECRET, app_config=app_config, credential_fields={}, source=source,
            allowlisted=True, status=ByokResolutionStatus.RESOLVED,
        )

    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", allowed_models=["gpt-allowed"])}
    )
    runtime = _runtime(resolver, override_snapshot_resolver=capture_provider_override_call_snapshot)
    try:
        outcome = (await runtime.resolve("openai", model="gpt-other")).credential_source
    except ByokResolutionError as error:
        outcome = getattr(error, "policy_code", error.code)
    finally:
        await runtime.close()
        set_llm_provider_overrides_cache_for_tests({})
    assert outcome == expected


@pytest.mark.asyncio
async def test_a_denied_model_is_refused_before_any_key_is_resolved() -> None:
    """TASK-15582 (R12): what every key refuses is refused before a credential is read."""
    calls = 0

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        nonlocal calls
        calls += 1
        return _resolution(provider)

    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", denied_models=["gpt-denied"])}
    )
    runtime = _runtime(resolver, override_snapshot_resolver=capture_provider_override_call_snapshot)
    try:
        with pytest.raises(ByokResolutionError) as refused:
            await runtime.resolve("openai", model="gpt-denied")
    finally:
        await runtime.close()
        set_llm_provider_overrides_cache_for_tests({})
    assert (refused.value.policy_code, calls) == ("model_denied", 0)
```

In `tests/LLM_Calls/test_provider_adapter_runtime_boundary.py`, after `test_an_allowed_model_stays_on_the_users_own_key`:

```python
@pytest.mark.asyncio
@_MODEL_SOURCES
async def test_a_model_off_the_allowlist_runs_on_the_users_own_key(
    monkeypatch: pytest.MonkeyPatch, config_model: str | None, model_override: str | None, expected: str
) -> None:
    """TASK-15582: analyze on the user's own key is not held to the server's allowlist, whatever the model's source."""
    result, calls, warnings = await _summary(
        monkeypatch, overrides=_allowlist("x"), config_model=config_model, model_override=model_override
    )

    assert (result, calls, warnings) == ("summary", [(expected, "handle-key", True)], [])
```

In `tests/Usage/test_llm_quota_decision.py`, after `test_an_admin_disabled_own_key_keeps_no_handle_and_its_limit`:

```python
async def test_an_own_key_keeps_its_handle_for_a_model_off_the_allowlist(monkeypatch) -> None:
    """TASK-15582: the own-key boundary resolves the user's key for a model the server's key does not offer."""
    from tldw_Server_API.app.core.Provider_Credentials import provider_credential_runtime as runtime_module
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )
    from tldw_Server_API.tests.helpers.llm_quota import users_own_key

    users_own_key(monkeypatch, "openai")
    monkeypatch.setattr(runtime_module, "load_server_config_snapshot", lambda: {"openai_api": {"model": "gpt-off-list"}})
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", allowed_models=["gpt-allowed"])}
    )

    handle = await usage_quota_deps.users_own_key_credentials(1, "openai")

    assert getattr(handle, "credential_source", None) == "user"
```

In `app/core/MCP_unified/tests/test_model_completion_integration.py`: add the row `({"denied_models": ["fixed"]}, True, "model_denied", ModelFailureDomain.CREDENTIAL_SCOPE),` with id `"model-denied"` to `test_an_admin_disabled_provider_or_model_is_refused_on_every_key`'s parametrization (`:378-386`). Its existing `model-not-allowed` row stays and keeps passing: the server config sends every key to `https://fixed.example/v1`, an operator endpoint, so the user's key there is `user_unmetered` (Review Focus 2). Then add:

```python
async def test_the_users_own_key_at_its_vendor_runs_a_model_off_the_allowlist(composed):
    """TASK-15582: after the scope check, the user's own key at the vendor's host is not held to the server's
    allowlist; the bounded transport sends it, and nothing else is refused."""
    env = composed
    await seed_keys(env)
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"openai": llm_provider_overrides.LLMProviderOverride(provider="openai", allowed_models=["x"])}
    )
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    port = env.build(server_config={"openai_api": {"api_key": KEYS["server"], "api_base_url": "https://api.openai.com/v1"}})

    result = await port.complete(request(), invocation)

    assert (result.content, env.provider.requests[0].headers["authorization"]) == (
        CONTENT.replace("\r\n", "\n"),
        f"Bearer {KEYS['user']}",
    )
```

In `tests/Chat/integration/test_chat_provider_override_store_boundary.py`, after `test_late_chat_override_policy_change_blocks_provider_dispatch`. It reuses `_configure_no_dispatch_boundary` (no new endpoint patch) and then replaces its dispatch:

```python
@pytest.mark.parametrize(
    ("own_key", "expected"),
    [(True, (200, None, ["openai"])), (False, (403, "model_not_allowed", []))],
    ids=["own-key", "server-key"],
)
def test_chat_runs_a_model_off_the_allowlist_only_on_the_users_own_key(
    authenticated_client, monkeypatch: pytest.MonkeyPatch, own_key: bool, expected: tuple[int, str | None, list[str]]
) -> None:
    """TASK-15582: chat's first check takes only what every key refuses; the credential runtime checks the allowlist
    once it knows the key, so the user's own key runs a model the server's key does not offer."""
    from tldw_Server_API.tests.helpers.llm_quota import users_own_key

    dispatch_calls: list[str] = []

    def dispatch(**kwargs: Any):
        dispatch_calls.append(str(kwargs.get("api_endpoint")))
        return {
            "id": "chatcmpl-own-key", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    _configure_no_dispatch_boundary(monkeypatch, static_calls=[], dispatch_calls=[])
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call", dispatch)
    if own_key:
        users_own_key(monkeypatch, "openai")
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", allowed_models=["gpt-4.1"], api_key="server-key")}
    )

    response = authenticated_client.post("/api/v1/chat/completions", json=_chat_request())

    # The refusal's code is read from the body, so the test is not status-only (review round 1, B4).
    assert (
        response.status_code,
        (response.json().get("detail") or {}).get("error_code"),
        dispatch_calls,
    ) == expected
```

In `tests/Audio/test_tts_credential_runtime_boundary.py`, append (review round 1, I1):

```python
@pytest.mark.asyncio
async def test_streaming_tts_holds_the_users_own_key_to_the_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK-15582 (review round 1, I1): text-to-speech is outside the LLM rule (ADR-086). Streaming TTS checks the
    override before any key is resolved, as /audio/speech does, so a model off allowed_models is refused on the
    user's own key too, and no key is read."""
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Audio import tts_service
    from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError, ResolvedByokCredentials
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    resolved: list[str] = []
    real_runtime = tts_service.ProviderCredentialRuntime

    async def users_key(provider: str, **_kwargs: object) -> ResolvedByokCredentials:
        resolved.append(provider)
        return ResolvedByokCredentials(
            provider=provider, api_key="sk-user", app_config={}, credential_fields={}, source="user", allowlisted=True
        )

    monkeypatch.setattr(tts_service, "ProviderCredentialRuntime", lambda **kwargs: real_runtime(**kwargs, resolver=users_key))
    monkeypatch.setattr(tts_service, "load_server_config_snapshot", lambda: {})
    monkeypatch.setattr(tts_service, "_capture_tts_provider_config", lambda _provider: {"enabled": True})
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", allowed_models=["tts-1"])}
    )

    with pytest.raises(ByokResolutionError) as refused:
        async with tts_service.tts_provider_credential_scope(
            provider="openai",
            model="gpt-4o-mini-tts",
            request=SimpleNamespace(state=SimpleNamespace()),
            current_user=SimpleNamespace(id=101),
        ):
            pass
    assert (getattr(refused.value, "policy_code", None), resolved) == ("model_not_allowed", [])
```

In `tests/Flashcards/test_study_assistant_adapter_contract.py`, after `test_a_reply_runs_on_the_users_stored_key_else_the_servers`:

```python
@pytest.mark.integration
@pytest.mark.parametrize(("stored", "expected"), [(True, (200, 1)), (False, (400, 0))], ids=["own-key", "server-key"])
def test_a_model_off_the_allowlist_runs_only_on_the_users_own_key(
    study_route, provider_boundary, monkeypatch, stored, expected
):
    """TASK-15582: the study assistants hold the server's key to allowed_models and let the user's own key run a
    model off it."""
    from tldw_Server_API.tests.helpers.llm_quota import users_own_key

    if stored:
        users_own_key(monkeypatch, "openai")
    provider_boundary.overrides["openai"] = overrides.LLMProviderOverride(provider="openai", allowed_models=["gpt-allowed"])
    payload = {"action": "follow_up", "message": "Which day?", "provider": "openai", "model": "remote-explicit"}
    response = study_route.client.post(study_route.path + "/respond", json=payload)
    assert (response.status_code, len(provider_boundary.calls)) == expected
```

- [ ] **Step 2: RED.**

```bash
RUN t5red -n 2 tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py \
  tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py tldw_Server_API/tests/Usage/test_llm_quota_decision.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py \
  tldw_Server_API/tests/Chat/integration/test_chat_provider_override_store_boundary.py \
  tldw_Server_API/tests/Flashcards/test_study_assistant_adapter_contract.py \
  "tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py::test_streaming_tts_holds_the_users_own_key_to_the_allowlist"
```

Expected (measured in review round 1): 16 failed, all `own-key`/`user` cases refused with `model_not_allowed`: 4 `test_a_model_off_the_allowlist_runs_only_on_the_users_own_key[*-user]` (one per method), the OpenRouter test, the runtime `[user]` row, the 3 analyze model sources, the quota-decision test (no handle), the MCP vendor test (`ModelCompletionFailure`), the chat `[own-key]` case (403) and the 4 study `[own-key]` cases (400: flashcards or quizzes, on SQLite or PostgreSQL; the plan first counted 1, so 13). The operator-host tests, the denied-before-resolution test, the MCP `model-denied` row, the streaming-TTS test and every `server-key`/team/org/server case already pass: they pin what this task keeps. The streaming-TTS test's RED is measured by mutation in Step 3.

- [ ] **Step 3: Implement the source.**
  - `usage_accounting.refuse_admin_disabled`: first statement after the exempt return, before `request` is replaced: `allowlist = call_credential_source(request) != USER_KEY_SOURCE  # the server-key allowlist (TASK-15582)`; the loop calls `enforce_provider_override(name, model, allowlist=allowlist)`. Its docstring's first line becomes "Refuse what the admin provider override disallows for this call's key (TASK-15581, TASK-15582)." and it says: "A disabled provider and a denied model are refused on any key; a model off ``allowed_models`` only off the user's own key (credential source ``user``)." Update the module docstring's second sentence to match.
  - `provider_credential_runtime.py`: import `USER_KEY_SOURCE` beside `billed_credential_source` (`:33`); the protocol line `:42` becomes `def enforce(self, model: str | None, *, allowlist: bool = True) -> None: ...`. In `_capture_override_snapshot`:

```python
            # The key is not resolved yet: only what every key refuses. _enforce_call_policy checks the server-key
            # allowlist once it is (TASK-15582).
            snapshot.enforce(model, allowlist=False)
```

  In `_enforce_call_policy`, inside its `try`, before `snapshot.enforce(...)`:

```python
            # allowed_models is what the server's key offers; the user's own key at its vendor is not held to it.
            own_key = billed_credential_source(provider, resolution.source, resolution.app_config) == USER_KEY_SOURCE
            snapshot.enforce(effective_model, allowlist=not own_key)
```

  In `_resolve_entry`'s authoritative block (`:807-815`), replace the call and its comment:

```python
                # After the scope check, on whichever key it chose: a disabled provider or a denied model is refused
                # (TASK-15581); a model off the allowlist unless the key is the user's own at its vendor (TASK-15582).
                own_key = billed_credential_source(
                    provider, getattr(resolution, "source", None), getattr(resolution, "app_config", None)
                ) == USER_KEY_SOURCE
                enforce_provider_override(provider, model, allowlist=not own_key)
```

  - `Summarization_General_Lib.py`: extend the `:60` import to `USER_KEY_SOURCE, call_credential_source, scoped_provider_credentials`; the check becomes

```python
        # Refused on every key (TASK-15581); a model off the allowlist only off the user's own key (TASK-15582). The
        # provider base class refuses it too; here it keeps this path's failure codes, and comes before a stream.
        source = call_credential_source({PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: provider_credentials})
        enforce_provider_override(provider, model, allowlist=source != USER_KEY_SOURCE)
```

  - `chat.py`: extend the `llm_call_context` import (`:337` at the base, `:338` once Task 2 has added its `provider_results` import) to `from tldw_Server_API.app.core.Usage.llm_call_context import USER_KEY_SOURCE, call_credential_source, llm_call_context, with_llm_call_context`. At `:4469`:

```python
                # Only what every key refuses: credential_runtime.resolve checks the allowlist below, once it knows the
                # key; a macro runs later as a job on the server's key, so it is held to the allowlist now (TASK-15582).
                override_error = _llm_provider_overrides.validate_provider_override(provider, model, allowlist=chat_macro_intent is not None)
```

  At `:5295` (one line): `override_error = _llm_provider_overrides.validate_provider_override(selected_provider, refreshed_model or model, allowlist=call_credential_source(refreshed_args) != USER_KEY_SOURCE)`.
  - `chat_target_resolution.resolve_chat_target`: add the keyword `allowlist: bool = True`, pass it to `validate_provider_override(provider, model, allowlist=allowlist)`, and add to the docstring: "``allowlist=False`` checks only what every key refuses: for a call on the user's own key, or before the key is known when the credential runtime checks it later (TASK-15582)."
  - `study_assistant_deps.py:62`: `resolve_chat_target(requested_provider=payload.provider, requested_model=payload.model, allowlist=False)`, with the comment "# The key is not known yet: the own-key resolution below checks the allowlist once it is (TASK-15582)."
  - `study_assistant.py:465`: import `USER_KEY_SOURCE` from `tldw_Server_API.app.core.Usage.llm_call_context` with the function's other local imports; `target = resolve_chat_target(requested_provider=provider, requested_model=model, allowlist=getattr(provider_credentials, "credential_source", None) != USER_KEY_SOURCE)`.
  - `tts_service.tts_provider_credential_scope` (review round 1, I1): the first statement inside its first `try`, before `_capture_tts_provider_config`:

```python
        # Text-to-speech is outside the LLM rule (ADR-086): like /audio/speech (_resolve_tts_byok), check the override
        # before any key is resolved, so allowed_models binds every key here (TASK-15582).
        capture_provider_override_call_snapshot(provider_key).enforce(model)
```

  That `try` already re-raises `ByokResolutionError`, so the refusal keeps its `policy_code`. Measure the new test's RED by mutation, then restore:

```bash
# comment out the enforce line just added to tts_service.py, then:
RUN t5ttsmut "tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py::test_streaming_tts_holds_the_users_own_key_to_the_allowlist"
# restore the line
```

  Expected: 1 failed (`DID NOT RAISE`: the runtime resolved the user's key and skipped the allowlist), the review's I1 scenario; with the line, it passes.

- [ ] **Step 4: Update the test doubles and the flipped tests.**
  - Each test-double `enforce` in the Files list takes the keyword: `def enforce(self, _model, **_kwargs):` (typed files: `**_kwargs: object`). The two delegating doubles in `test_messages_provider_override_store_boundary.py` and the one in `test_embeddings_provider_override_store_boundary.py` forward it: `def enforce(self, model: str | None, **kwargs: Any) -> None: captured.enforce(model, **kwargs)`.
  - `test_chat_provider_override_store_boundary.py:457-458`: `def gated_validate(provider: str, model: str | None, **kwargs: Any):` and `decision = original_validate(provider, model, **kwargs)`.
  - The doubles of `validate_provider_override` patched onto `chat_target_resolution`, which now always passes `allowlist=` (review round 1, B3): `test_chat_target_resolution.py:201` becomes `lambda provider, model, **_kwargs: None`; `:268` becomes `lambda provider, model, **_kwargs: {` (body unchanged); `tests/Sharing/test_shared_workspace_recipient_security_matrix.py:568` becomes `lambda _provider, _model, **_kwargs: None`.
  - `test_llm_quota_byok_calls.py`: `_DISABLES` (`:479`) becomes `{"provider_disabled": {"is_enabled": False}, "model_denied": {"denied_models": ["m1"]}}` under the comment "# An admin override for openai that refuses REQUEST's model "m1" on every key, by its reason (TASK-15581, TASK-15582)."
  - The judge test (`:673-692`) is split by key (review round 1, B3): on the server's key the judge takes the server config's model (`resolve_provider_model("openai", cfg)`, `eval_runner.py:136`), not `DEFAULT_MODEL_OPENAI`, so a denylist of `gpt-denied` would not refuse it. It becomes:

```python
@_judge_modules()
@pytest.mark.parametrize(
    ("own_key", "override", "expected"),
    [(True, {"denied_models": ["gpt-denied"]}, "model_denied"), (False, {"allowed_models": ["x"]}, "model_not_allowed")],
    ids=["own-key", "server-key"],
)
async def test_an_evaluation_judge_refuses_a_disallowed_model_by_key(monkeypatch, module, own_key, override, expected) -> None:
    """TASK-15581, TASK-15582: on the run's own key a denied DEFAULT_MODEL_<PROVIDER> is refused (the allowlist no
    longer binds that key); on the server's key, which takes the server config's model, a model off the allowlist is.
    Both with ChatConfigurationError, which G-Eval does not retry."""
    monkeypatch.setattr(module, "get_adapter_or_raise", lambda _provider: _Provider())
    monkeypatch.setattr(module, "resolve_provider_api_key_from_config", lambda *_args, **_kwargs: "sk-server")
    monkeypatch.setenv("DEFAULT_MODEL_OPENAI", "gpt-denied")
    _admin_override(**override)
    handle = await _handle("user") if own_key else None

    def _judge() -> str:
        """One call with no model, so the key's default model applies."""
        return module._call_adapter_text(
            api_endpoint="openai", messages_payload=[{"role": "user", "content": "q"}], provider_credentials=handle
        )

    with pytest.raises(ChatConfigurationError) as refusal:
        await asyncio.to_thread(_judge)
    assert refusal.value.error_code == expected
```

  - The Prompt Studio test (`:695-711`) is renamed `test_prompt_studio_refuses_a_denied_default_model_on_the_users_key`; it calls `_admin_override(denied_models=["gpt-denied"])` and expects `"model_denied"` (its own-key `DEFAULT_MODEL_OPENAI` path passed as measured).
  - `test_provider_credential_runtime.py`: in `test_concurrent_implicit_configured_model_policy_is_enforced_once` the override becomes `LLMProviderOverride(provider="openai", denied_models=["gpt-blocked"])` and the expected code `"model_denied"`; in `test_concurrent_explicit_and_implicit_models_enforce_per_caller_policy` it becomes `denied_models=["gpt-blocked-default"]` and the expected `policy_code` `"model_denied"` (their resolutions are the user's key at its vendor, which no longer meets the allowlist).
  - `test_provider_adapter_runtime_boundary.py` (`:539-606`): the every-key refusals and the other-key ones become two parametrizations:

```python
# Each refusal on every key (TASK-15581, TASK-15582), and a policy that cannot be read.
_EVERY_KEY = [
    ({"denied_models": ["cfg-m", "gpt-env-default", "ovr-m"]}, True, "model_denied", _POLICY_DENIED),
    ({"is_enabled": False}, True, "provider_disabled", _POLICY_DENIED),
    ({}, False, "credential_store_unavailable", _STORE_DOWN),
]
_REFUSAL_FIELDS = ("override_kwargs", "healthy", "reason", "refused")
_REFUSALS = pytest.mark.parametrize(
    _REFUSAL_FIELDS, _EVERY_KEY, ids=["model-denied", "provider-disabled", "override-cache-unhealthy"]
)
# A model off allowed_models is refused on every key but the user's own (TASK-15582).
_OTHER_KEY_REFUSALS = pytest.mark.parametrize(
    _REFUSAL_FIELDS,
    [({"allowed_models": ["x"]}, True, "model_not_allowed", _POLICY_DENIED), *_EVERY_KEY],
    ids=["model-not-allowed", "model-denied", "provider-disabled", "override-cache-unhealthy"],
)
```

  and `test_an_admin_disabled_provider_or_model_is_refused_on_every_other_key` takes `@_OTHER_KEY_REFUSALS` in place of `@_REFUSALS`.
  - `test_summarization_adapter.py:845-847`: the override becomes `LLMProviderOverride(provider="moonshot", denied_models=["blocked-model"])`, and the comment at `:835-836` says the fallback model is refused when the override denies it.
  - `test_llm_quota_decision.py:383`: the row becomes `({"denied_models": ["gpt-denied"]}, True, "model_denied"),` with id `"model-denied"`.

- [ ] **Step 5: GREEN on the changed files.**

```bash
RUN t5green -n 2 tldw_Server_API/tests/Usage tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py \
  tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py \
  tldw_Server_API/tests/Chat/integration/test_chat_provider_override_store_boundary.py \
  tldw_Server_API/tests/Chat/integration/test_messages_provider_override_store_boundary.py \
  tldw_Server_API/tests/Chat/test_chat_target_resolution.py tldw_Server_API/tests/Flashcards/test_study_assistant_adapter_contract.py \
  tldw_Server_API/tests/Audio/test_audio_tts_oauth_retry_unit.py tldw_Server_API/tests/Audio/test_failopen_cap_minutes.py \
  tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py tldw_Server_API/tests/Audio/test_tts_endpoint_hardening_boundaries.py \
  tldw_Server_API/tests/Audio/test_tts_runtime_hardening_boundaries.py tldw_Server_API/tests/Chunking/test_chunking_endpoint.py \
  tldw_Server_API/tests/Chunking/test_chunking_runtime_lifecycle.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_provider_override_store_boundary.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py \
  tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_security_matrix.py
```

Expected: 0 failed, except ids already in `<scratchpad>/sweep0_15582.txt` (order-dependent probe leaks in `test_messages_provider_override_store_boundary.py` and the Embeddings file); netprobe lines: 1 (the known tiktoken download, Global Constraints) plus any from those ids. Review round 1 measured this run before its B3 fixes at 3 failed (the `:268` double, the two `[server-key-*]` judge cases), 1591 passed, probes 1, and the Sharing file alone at 12 failed, then 19 passed with its double fixed.

- [ ] **Step 6: The sweep.** The same 62 files as the baseline:

```bash
FAILED_TO=<scratchpad>/sweep1_15582.txt RUN sweep1 -n 2 $(cat <scratchpad>/sweep_files_15582.txt)
comm -13 <scratchpad>/sweep0_15582.txt <scratchpad>/sweep1_15582.txt
comm -23 <scratchpad>/sweep0_15582.txt <scratchpad>/sweep1_15582.txt
```

Expected: both `comm` commands print nothing. If the first prints ids, classify each, fix, and re-run this step:
- it expects a refusal of a model off `allowed_models` on a key resolved or carried as `user` at the vendor: give the override `denied_models` with the same model(s) and expect `model_denied` (the test's every-key intent), unless the test is about the allowlist itself, where the own-key case now runs and the other keys stay refused;
- a test double of `enforce`, `validate_provider_override`, `enforce_provider_override` or `resolve_chat_target` rejects the `allowlist` keyword: accept `**kwargs`, and forward it where the double delegates;
- it asserts the resolver is not called for an allowlist refusal: the refusal now follows one resolution (R12); keep the code assertion only;
- anything else is a behaviour change this plan did not intend: stop and report it.

Then delete the lists: `rm -f <scratchpad>/sweep0_15582.txt <scratchpad>/sweep1_15582.txt <scratchpad>/sweep_files_15582.txt`.

- [ ] **Step 7: Commit.**

```bash
git add -A tldw_Server_API/app tldw_Server_API/tests
git commit -m "feat(llm): allowed_models governs the server's key only (TASK-15582)

The owner's 2026-11-21 decision: a model off allowed_models is refused on the server's key,
team and org keys and a user key at an operator endpoint, and runs on the user's own key at
its vendor. The adapter guard, the credential runtime (MCP included), analyze, chat and the
study assistants check it by the call's key.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The admin UI sets `denied_models`

**Files:**
- Modify: `admin-ui/types/index.ts` (`:331`)
- Modify: `admin-ui/app/providers/page.tsx` (state `:42-54`, `:306-318`; `openOverrideDialog` `:656-671`; `handleSaveOverride` `:677-704`; the badge `:1209-1213`; the dialog `:1667-1675`)
- Test: `admin-ui/app/providers/__tests__/page.test.tsx` (append)

**Interfaces:**
- Consumes: Task 4's API field `denied_models: string[] | null`.

- [ ] **Step 1: Install, with a disk check.**

```bash
df -g /System/Volumes/Data | awk 'NR==2{print $4" GiB free"}'
cd admin-ui && bun install --frozen-lockfile && cd ..
```

Expected: at least 100 GiB free before; the install ends without error.

- [ ] **Step 2: Write the failing tests.** Append to `admin-ui/app/providers/__tests__/page.test.tsx`:

```tsx
describe('ProvidersPage provider override denylist', () => {
  it('saves denied models from the override dialog', async () => {
    const user = userEvent.setup();
    render(<ProvidersPage />);

    await user.click(await screen.findByTitle('Manage override'));
    await user.type(screen.getByLabelText('Denied models'), 'gpt-4o, gpt-4.1');
    await user.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => {
      expect(apiMock.updateLLMProviderOverride).toHaveBeenCalledWith(
        'openai',
        expect.objectContaining({ allowed_models: [], denied_models: ['gpt-4o', 'gpt-4.1'] }),
      );
    });
  });

  it('shows the stored denylist in the row and the dialog', async () => {
    apiMock.getLLMProviderOverrides.mockResolvedValue({ items: [{ provider: 'openai', denied_models: ['gpt-4o'] }] });
    const user = userEvent.setup();
    render(<ProvidersPage />);

    expect(await screen.findByText('1 models denied')).toBeInTheDocument();
    await user.click(screen.getByTitle('Manage override'));
    expect(screen.getByLabelText('Denied models')).toHaveValue('gpt-4o');
  });
});
```

- [ ] **Step 3: RED.**

```bash
cd admin-ui && bun run test app/providers/__tests__/page.test.tsx; cd ..
```

Expected: 2 failed (no element labelled "Denied models"; no text "1 models denied"), 5 passed.

- [ ] **Step 4: Implement.**
  - `types/index.ts`: `denied_models?: string[];` after `allowed_models`.
  - `page.tsx`: a module-level helper `const parseModelList = (value: string): string[] => value.split(',').map((model) => model.trim()).filter(Boolean);`, used for both lists in `handleSaveOverride`; `deniedModels: string` in `OverrideDialogState`, `''` in the initial state, `override?.denied_models?.join(', ') ?? ''` in `openOverrideDialog`; `denied_models: deniedModels` in the payload beside `allowed_models`; after the allowlisted badge line, `{override.denied_models?.length ? (<div className="text-xs text-muted-foreground">{override.denied_models.length} models denied</div>) : null}`. In the dialog, the allowlist field's label becomes "Allowlisted models (server key)" and its placeholder "Comma-separated; leave empty for all. Users' own keys may run others.", and after it:

```tsx
              <div className="space-y-2">
                <Label htmlFor="overrideDeniedModels">Denied models</Label>
                <Input
                  id="overrideDeniedModels"
                  placeholder="Comma-separated; refused on every key"
                  value={overrideDialog.deniedModels}
                  onChange={(e) => updateOverrideDialog({ deniedModels: e.target.value })}
                />
                <p className="text-xs text-muted-foreground">
                  Refused on every key, users&apos; own keys included. A model on both lists is refused.
                </p>
              </div>
```

- [ ] **Step 5: GREEN, lint, typecheck, then remove the install.**

```bash
cd admin-ui && bun run test app/providers/__tests__/page.test.tsx && bun run lint && bun run typecheck; cd ..
rm -rf admin-ui/node_modules
git status --short admin-ui
```

Expected: 7 passed; lint and typecheck exit 0; `git status` lists only the three changed files.

- [ ] **Step 6: Commit.**

```bash
git add admin-ui/types/index.ts admin-ui/app/providers/page.tsx admin-ui/app/providers/__tests__/page.test.tsx
git commit -m "feat(admin-ui): set a provider override's denied models (TASK-15582)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Docs, the mirror and the CHANGELOG

**Files:**
- Modify: `Docs/ADR/086-authoritative-provider-credential-scope.md` (append an amendment before `## Follow-up`)
- Modify: `Docs/User_Guides/Server/BYOK_User_Guide.md` (`:97`), `Docs/Operations/Usage_Quotas.md` (`:120`), `Docs/Architecture/llm-providers.md` (`:21`, `:33`), `admin-ui/docs/api-reference.md` (`:802-803`), `CHANGELOG.md` (`:13-19`)
- Modify (generated): `Docs/Published/ADR/086-authoritative-provider-credential-scope.md`, `Docs/Published/User_Guides/Server/BYOK_User_Guide.md`

- [ ] **Step 1: ADR-086.** Append, after the TASK-15581 amendment:

> Amendment (TASK-15582, 2026-11-21): the owner chose "Server key only" for the provider override's `allowed_models`, and added `denied_models` as the explicit model disallow. A model on `denied_models` is refused on every credential and every path with `model_denied`, and wins over `allowed_models`. A model left off `allowed_models` is only not offered on the server's credential: it is refused with `model_not_allowed` when the call runs on the server's credential, a team's or an organization's, or a user's credential sent to an endpoint the operator runs, and the user's own credential at its vendor's host (credential source `user`, ADR-083) runs it, as it runs a provider the server has no credential for. `is_enabled = false` stays the explicit provider disallow, refused on every credential. This runtime still checks after the scope check, against the credential the scope chose. Elsewhere, a check made before the call's credential is known refuses only what every credential refuses, and the credential runtime or the adapter base class checks `allowed_models` once the credential is known. Embeddings, text-to-speech, shared-workspace chat and the persona voice preflight check `allowed_models` before they know the credential, so there it still refuses every credential.

- [ ] **Step 2: The guides.**
  - `BYOK_User_Guide.md:97`: replace "A provider or model the operator has turned off in its provider override is refused whatever key the call uses, your own included." with "A provider the operator has turned off, or a model it has denied, in its provider override is refused whatever key the call uses, your own included (`provider_disabled`, `model_denied`). A model the operator only leaves off the models it offers on the server's key (`allowed_models`) runs on your own key when the server sends the call to that provider's own host; on the server's key, an org or team key, or a key you stored for an endpoint the operator runs, it is refused with `model_not_allowed`."
  - `Usage_Quotas.md:120`: replace "A call to a provider the admin provider override turns off, or to a model outside its `allowed_models`, is refused on every key: the user's own, a team's, an org's and the server's, wherever its model came from (the provider's configured model, `DEFAULT_MODEL_<PROVIDER>`, or a model the request names)." with "A call to a provider the admin provider override turns off, or to a model on its `denied_models`, is refused on every key: the user's own, a team's, an org's and the server's, wherever its model came from (the provider's configured model, `DEFAULT_MODEL_<PROVIDER>`, or a model the request names). A model outside its `allowed_models` is refused on every key but the user's own at its vendor (`credential_source = user`): the allowlist is what the server's key offers (TASK-15582)." In the same paragraph, the reason list becomes "(`model_denied`, `model_not_allowed`, `provider_disabled` or `credential_store_unavailable`)".
  - `llm-providers.md:33`: replace "A provider or model the admin override disables is refused on every key (TASK-15581)." with "A provider the admin override disables, or a model on its `denied_models`, is refused on every key; a model off its `allowed_models` on every key but the user's own at its vendor (TASK-15581, TASK-15582)." `:21`'s last cell becomes "provider overrides CRUD (`is_enabled`, `allowed_models` for the server's key, `denied_models` for every key) + provider test + health".
  - `admin-ui/docs/api-reference.md:803`: "Set override for a provider. `is_enabled: false` refuses the provider on every key; `allowed_models` lists the models the server's key offers (a user's own key may run others); `denied_models` lists models refused on every key, and wins when a model is on both lists. An empty list clears it."

- [ ] **Step 3: CHANGELOG.** As the first bullet under `## [Unreleased]` / `### Changed`:

> - **`allowed_models` now governs only the server's key; `denied_models` is the explicit model disallow.** A model off a provider override's `allowed_models` is refused on the server's key, an org or team key, and a user key sent to an operator's endpoint, and it runs on a user's own key at the provider's vendor, as a provider the server has no key for does. A model on the new `denied_models` is refused on every key and path (`model_denied`, 403) and wins when a model is on both lists. This narrows the `allowed_models` part of the bullet below. The admin API (`PUT /api/v1/admin/llm/providers/overrides/{provider}`) and the admin UI's provider override dialog set it, and `GET /api/v1/llm/providers` hides denied models. The AuthNZ database gains `llm_provider_overrides.denied_models` (SQLite migration 107; PostgreSQL adds it at startup) (TASK-15582). **Before upgrading:** an existing `allowed_models` now lets users run off-list models on their own keys. An admin who meant those models to be forbidden must add them to `denied_models`. On PostgreSQL, start the API once before standalone workers and the `tldw-evals` LLM commands: its start-up adds the column, and until then they cannot load the overrides and refuse every LLM call.

and under `### Fixed`:

> - A server-side LLM call to an admin-disabled provider is refused before the gateway asks the provider for its model list, so the provider gets no request and the caller gets `provider_disabled`; an admin refusal that no endpoint maps is answered with its code's own message; and an LLM command group of `tldw-evals` given no subcommand prints its help without reading the AuthNZ database (TASK-15582).

- [ ] **Step 4: Mirror, and the docs tests.**

```bash
bash Helper_Scripts/refresh_docs_published.sh
git status --short Docs/Published
RUN docs -n 2 tldw_Server_API/tests/Docs
```

Expected: `git status` lists exactly ` M Docs/Published/ADR/086-authoritative-provider-credential-scope.md` and ` M Docs/Published/User_Guides/Server/BYOK_User_Guide.md`; the docs tests report 0 failed, netprobe lines 0 (measured in review round 1: 214 passed).

- [ ] **Step 5: Commit.**

```bash
git add Docs/ADR/086-authoritative-provider-credential-scope.md Docs/User_Guides/Server/BYOK_User_Guide.md \
  Docs/Operations/Usage_Quotas.md Docs/Architecture/llm-providers.md admin-ui/docs/api-reference.md CHANGELOG.md Docs/Published
git commit -m "docs: the model denylist and the server-key allowlist (TASK-15582)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Verification, close-out, review, PR

- [ ] **Step 1: tests/lint and tests/CI.**

```bash
RUN lint -n 2 tldw_Server_API/tests/lint --ignore=tldw_Server_API/tests/lint/test_runtime_package_discovery.py
RUN ci -n 2 tldw_Server_API/tests/CI
```

Expected: `tests/lint` 0 failed (#3537 saw an intermittent native abort in `test_rg_route_map_lint.py` under `-n 2` that also hits `origin/dev`; if it appears, re-run that file with `-p no:xdist` and expect it to pass). `tests/CI`: 0 failed apart from the known `test_jobs_postgres_exports_recorded_coverage` cases (no `coverage` module), which fail the same way on `origin/dev`.

- [ ] **Step 2: The gates.**

```bash
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/test_quality_triage.py --enforce
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_shard_coverage.py
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_backlog_task_ids.py
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/scope_predicate_ratchet.py
git diff --name-only --diff-filter=AM origin/dev...HEAD -- '*.py' | xargs $(dirname $PY)/ruff check
uvx bandit -q -r $(git diff --name-only --diff-filter=AM origin/dev...HEAD -- 'tldw_Server_API/app/*.py' 'tldw_Server_API/cli/*.py' ':!*/tests/*')
git diff --check origin/dev...HEAD
```

Expected: the four scripts exit 0 ("no newly-unshared test files", "baseline exact"; the triage would exit 1 with `status_only=1 (new)` had the chat test asserted only its status and call list, review round 1, B4; shard coverage measured `new_uncovered=0`). `ruff` reports nothing that `origin/dev`'s versions of the same files do not already report (for a file it flags, compare with `git show origin/dev:<file> | $(dirname $PY)/ruff check --stdin-filename <file> -`). Bandit runs over the 32 changed source files (review round 1 measured 31, before I1 added `app/core/Audio/tts_service.py`); any finding it prints is on a line `origin/dev` has too. `git diff --check` prints nothing.

- [ ] **Step 3: Close TASK-15582.**

```bash
BACKLOG task edit TASK-15582 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "<measured: each task's RED and GREEN counts, the sweep diff, tests/Docs, tests/lint, tests/CI, the gates, Bandit, the admin-ui run, PostgreSQL run or 'CI measures'>" \
  --final-summary "A provider override's denied_models is the explicit model disallow: refused on every key and path (model_denied, 403), winning over allowed_models; is_enabled=False stays the provider disallow. allowed_models now governs only the server's key: a model off it runs on the user's own key at its vendor (credential source user) and is refused on the server's key, team and org keys, and a user key at an operator endpoint. The adapter base class, the credential runtime (MCP included), analyze, chat and the study assistants apply it by the call's key, and the server-key listing hides denied models. SQLite migration 107 and the PostgreSQL ensure add the column; the admin API and the admin UI set it. ADR-086, the BYOK guide, Usage_Quotas, llm-providers, the admin API reference and the CHANGELOG say so. With it, the four minors deferred from PR #3537's final review. The own-key WebUI picker is TASK-15583." \
  -s Done
git add -A backlog/tasks
git commit -m "chore(backlog): close TASK-15582

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: `TASK-15582 - Let an admin explicitly disallow a provider or model, distinct from 'not configured' (Done)`.

- [ ] **Step 4: Final review.** A clean-session opus review of the whole branch (`/superpowers:requesting-code-review`), given Review Focus and the Rulings. Fold in every Critical and Important finding, re-run the affected files through `RUN`, and record the review in the PR body.

- [ ] **Step 5: The PR.** Push, rewrite `<plans dir>/pr_body_model_denylist.md`, then `gh pr edit <number> --body-file <plans dir>/pr_body_model_denylist.md` and `gh pr ready <number>`. The body has: Why (the owner's decision, verbatim choice "Server key only", and the standing rules); What changes (the `allowlist` keyword and where each path learns its key, the table above; the code; the column, migration 107 and the admin surfaces; the four minors); Measured (the counts); Known limits (from this plan), including the R1 line: "A team or org key the team pays for, and a user's own key stored with the user's own base URL (`user_unmetered`), are held to the allowlist. Both err closed."; a **Before upgrading** line ("an existing `allowed_models` now lets users run off-list models on their own keys; an admin who meant those models to be forbidden must add them to `denied_models`; the AuthNZ database gains a column on start-up; on PostgreSQL, start the API once before standalone workers and the `tldw-evals` LLM commands"); Reviews; the full `## Change summary` section ("ADR-004: the owner waived a user-written Change summary for this PR on 2026-11-21."); and the closing `🤖 Generated with [Claude Code](https://claude.com/claude-code)` line. Merges go through the controller's queue (check `gh variable get MERGE_QUEUE` first).

Expected: the PR is ready for review against `dev`, with the required statuses running.
