# The WebUI offers a provider the user's own key serves (TASK-15583) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A signed-in user with a stored own key for a provider the server has no key for can pick that provider and its models in every WebUI model picker, and no listing ever shows another user's keys or models.

**Architecture:** The four listing routes (`GET /llm/providers`, `/llm/providers/{provider_name}`, `/llm/models`, `/llm/models/metadata`) gain one public dependency, `listing_user_id`, that names the requesting user without ever refusing. For that user, `users_own_key_credentials(..., policy=False)` (the handle chat's own-key paths already use, without the override check the listing makes itself, review round 1 I1) is asked per cloud provider; each handle it issues goes into `get_configured_providers(..., own_keys=...)`, where `_resolve_commercial_listing` lists the provider on the user's key through the same discovery chat runs (`_listing_on_key`, extracted from the server path), so readiness, `models` and `models_info` come from the user's own key. The admin override is then applied per entry: server-key entries through `apply_llm_provider_overrides_to_listing` as before, own-key entries through `policy_error(model, allowlist=False)` (TASK-15582). The WebUI's filters need no change, because the entry now says what they check; the provider keys settings page clears the model catalog after a save or delete. A listing with a user is answered `Cache-Control: private, no-store`, and every listing with `Vary: Authorization, X-API-KEY, Cookie`.

**Tech Stack:** Python 3, FastAPI, pytest (pytest-asyncio auto mode, xdist), httpx, backlog-py; React + vitest (`apps/`, bun).

**Spec:** TASK-15583 (its description and 5 ACs), the owner's rule of 2026-11-19, the owner's decision of 2026-11-21 that TASK-15582 implements, TASK-15582's plan (`Docs/superpowers/plans/2026-11-21-model-denylist.md` on `dev`), and the controller's planning brief.

**Base:** `origin/dev` after TASK-15582's PR merges. Every line number below is at `0d5e9c7228` and holds at #3544's head `ced8df7c15` (TASK-15582 does not edit `llm_providers.py`, `usage_quota_deps.py`, `ProviderKeysSettings.tsx`, the Providers API doc or the function-size entry this plan bumps); `Docs/Architecture/llm-providers.md` is cited at `ced8df7c15`. Branch `feat/webui-byok-provider-15583`; the PR targets `dev`.

## The rules this follows

- **Owner, 2026-11-19:** a provider the admin explicitly disables is refused on every key; a provider that is only "not configured" (no server key or credits) must be usable with a user's own key.
- **Owner, 2026-11-21 (TASK-15582 implements it):** a model left off `allowed_models` is only "not offered on the server's key" and is usable with a user's own key; a model on `denied_models` is refused on every key; team and org shared keys are held to the allowlist, only a user's own key skips it (TASK-15582's R1).
- **AC #2 changes with that decision** (Task 0 edits it): a provider with `is_enabled=False`, or a model on `denied_models`, stays unselectable for every user; a model off `allowed_models` is selectable only for a user whose own key serves that provider.

## Global Constraints

- Tests make no real outbound network requests. Every pytest run loads the network probe plugin (`-p netprobe_15427`, from `<plans dir>`); a run ends with 0 probe lines for the tests this plan adds or changes. The new backend tests replace `provider_model_inventory._http_fetch` with a fake vendor; nothing else in them can reach a host.
- A test that drives `/chat/completions`, an LLM adapter or a judge stubs the adapter or the provider call; a commercial provider otherwise calls the real API. (This plan adds no such test.)
- Backlog edits use backlog-py only (`BACKLOG task edit ...`), with `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit a task file. TASK-15583 exists; create no other task.
- Branch `feat/webui-byok-provider-15583` from `origin/dev`, after TASK-15582 merges; the PR targets `dev`.
  - Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - ADR-004: the Change summary is waived (owner, 2026-11-21). The PR body has a full `## Change summary` section reading: "ADR-004: the owner waived a user-written Change summary for this PR on 2026-11-21." The controller's brief records this waiver; before `gh pr ready`, the controller confirms it is in the recorded waivers (review round 1, M8).
- Never `--no-verify`, never force-push, never `git stash`.
- PostgreSQL only through the repo fixtures, with `TLDW_TEST_NO_DOCKER=1` and a session-unique `TEST_DB_NAME`; never start Docker for it. (This plan runs no PostgreSQL test.) Redis: never FLUSHALL or SCRIPT FLUSH.
- A mirrored doc (the BYOK guide, the Providers API doc, everything under `Docs/Published`) needs `bash Helper_Scripts/refresh_docs_published.sh`, with the `Docs/Published` diff in the same PR.
- The ratchets:
  - Function size (`tests/lint/function_size_baseline.json`, threshold 300): `llm_providers:get_configured_providers` is capped at exactly its size, 662 lines and 2 params. It gains one keyword and one call line (R7): raise its entry to the measured numbers (expected 664 lines, 3 params) with a reason, in Task 1. No other changed function is near 300 lines.
  - Endpoint monkeypatch (`tests/lint/test_endpoint_monkeypatch_ratchet.py`): no new test patches an attribute of a module under `app/api/v1/endpoints/`. The new tests patch core modules by string target and set the user with `app.dependency_overrides`.
  - Endpoint getter calls (`tests/lint/test_service_singleton_ratchet.py`, `llm_providers.py` capped at 9): add no zero-argument `get_*()` call to `llm_providers.py` (`capture_provider_override_call_snapshot(name)` takes an argument; `get_llm_provider_overrides_snapshot()` is not called).
  - Route auth (`Helper_Scripts/ci/route_auth_baseline.txt`): the four listing routes stay listed as public. `listing_user_id` reads credentials inside its body and is not an authenticator, so the baseline does not change.
  - Scope predicate (`Helper_Scripts/ci/scope_predicate_baseline.txt`): measured, no file this plan touches is listed.
  - Test-quality triage (`Helper_Scripts/ci/test_quality_triage.py --enforce`): every new test asserts a real output, not only a list its own fake fills.
  - OpenAPI contract (`apps/packages/api-types/openapi.fingerprint.json`, `paths.d.ts`): `listing_user_id` takes only `Request` and `Response`, so no query, header or body field is added and the fingerprint does not change (the review measured it identical, `4bea48745eb4…`, the value #3544's `ca477f13f3` commits). Task 4 runs the drift check, and regenerates only on a mismatch.
- New test files sit under a CI shard path: `tests/Chat_NEW/unit/test_[a-l]*.py` covers `test_llm_providers_own_key.py`; `frontend-required` runs the new vitest file.
- Disk rules override everything; the owner is angry about the disk filling.
  - Every pytest command runs through `RUN` (below), which is `mkdir -p <T> && TMPDIR=<T> zsh <scratchpad>/disk_guard.zsh <T> -- $PY -m pytest --basetemp=<T>/bt -p no:cacheprovider ...` from a fresh `<T>` under `<scratchpad>`, with at most `-n 2`.
  - Measure RED in place: write the test before the source, or (after a commit) change a source hunk and restore it with `git restore <file>`. No `git archive`, no repo copies, no extra worktrees.
  - Leave no log files. The sweep's id lists are deleted in Task 4.
  - Frontend tests reuse the main checkout's existing `node_modules` by symlink and remove the symlinks after (Task 2). Never install.
- `$PY` = `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`; every command that runs Python puts `$(dirname $PY)` first on `PATH`.
- The plan stays free of trailing whitespace (it is committed under `Docs/superpowers/plans/`), and the committed copy names no private planning file.

**Abbreviations.** `<scratchpad>` = `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad`; `<plans dir>` = `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`; `<main apps>` = `/Users/macbook-dev/Documents/GitHub/tldw_server/apps`. Every command runs from the worktree root. `RUN <name> <pytest args>` = `zsh <scratchpad>/run15583.zsh <name> <pytest args>` (Task 0 writes it). `BACKLOG` = `PATH="$(dirname $PY):$PATH" PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd .`. `LP` = `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`. `OWN` = `tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py`.

## Review Focus

The five inputs the feature's own tests would not exercise that are likeliest to bite, each pinned by a test in the task that owns the code:

1. **A key the user stored for an operator's endpoint** (a base URL off the vendor's host). It is not the user's own spend (ADR-083, `user_unmetered`), so it must not list a provider the server has no key for. Task 1, `test_a_key_stored_for_an_operators_endpoint_does_not_list_the_provider`.
2. **The vendor rejects the user's key.** The provider is listed unusable (`auth_failed`), with no models and no fallback to anything else. Task 1, `test_a_rejected_own_key_lists_the_provider_unusable`.
3. **A user with their own key for a provider the server also has a key for.** The user's listing is their own key's (models off `allowed_models` included), as chat's credential order (user, team, org, server) runs it; the server-key user's listing is held to the allowlist. Task 1, the `allowlist-and-denylist` row of `test_the_admin_override_holds_an_own_key_only_to_what_every_key_refuses`.
4. **No credentials, bad credentials, or any refusal of the auth layer** (401, 403, a 503 in maintenance). The listings stay public: no user, the server's listing, never an error. Task 1, `test_the_listing_names_its_user_and_never_refuses` (the dependency) and `test_bad_credentials_on_a_route_get_the_public_listing` (a real bad `X-API-KEY` through a route).
5. **A key saved or deleted in the WebUI.** The chat model pickers must refetch at once, not after the 5-minute model cache. Task 2, `ProviderKeysSettings.model-cache.test.tsx`.

## Rulings

| # | Ruling | Why |
|---|---|---|
| R1 | **Fix it at the source: the listing says the provider is configured and enabled for this user, with its own-key models; no WebUI filter changes.** | Every picker reads `is_configured`, `provider_is_configured`, `provider_enabled`, `availability` and `catalog_only` from the same two listings (table below). One server change covers all of them, and readiness messages stay true; changing each filter would leave the entry saying "not configured". |
| R2 | **AC #5: the models come from discovery with the user's own key, not a typed model ID.** | It is safe: chat already discovers with the call's key before every call to a listing provider (`gateway._validate_provider_model_selection` calls `discover_provider_models(provider, request["api_key"], ...)`) and refuses a model the list lacks, so a typed ID would still need this list. The listing runs the same function with the same endpoint and egress rule (`inventory_endpoint_scope(..., endpoint_provenance=handle.endpoint_provenance)`), a 5-second budget and bounded responses, only at the vendor's host (R3). Its cache key is `(provider, endpoint, SHA-256(key), scope)` (`provider_model_inventory._cache_key`), so an entry is reachable only by presenting the same key: per key, never across keys. Two users who stored the same key share the entry, which shows each only what that key lists (Known limits). Tested: `test_each_user_lists_a_keyless_provider_on_their_own_key_only` (one fetch per key, no other user's models). |
| R3 | **"Own key" is `users_own_key_credentials(user_id, provider)`**: BYOK on, the provider on the BYOK allowlist, an active stored key, and the server sends that provider to its vendor's host (credential source `user`). The listing asks with `policy=False` and applies the admin override itself (R4; review round 1, I1). | It is the definition chat's own-key paths and the quotas use (ADR-083), so the listing and the call agree. Team and org keys and a user key at an operator endpoint do not count (TASK-15582 R1). |
| R4 | **An own-key entry is held only to what every key refuses**: each model passes `capture_provider_override_call_snapshot(name).policy_error(model, allowlist=False)`; `enabled` is `policy_error(None, allowlist=False) is None`. `apply_llm_provider_overrides_to_listing` is not used for it. | TASK-15582's interface: that function is the server-key listing (allowlist and denylist). The override's `config` (its model list, default and ranking) shapes the server key's offer, so it is not applied either (Known limits). A disabled provider gets no model-list request (`_resolve_commercial_listing` takes the own-key branch only when it is not disabled) and lists nothing (every model is refused). |
| R5 | **The listings stay public.** `listing_user_id` calls `auth_deps.get_auth_principal` inside its body and turns any `HTTPException` into "no user". | On `dev` the routes take no credentials at all; a client with an expired token must not start getting 401 from the model list. The route-auth baseline stays exact. |
| R6 | **All four listing routes go through one helper, `_users_listing`**, not only the two the pickers read. | They share one builder; a provider-details call must agree with the picker. |
| R7 | **`get_configured_providers` takes `own_keys` as an explicit keyword, with a reasoned function-size bump** (expected 662 to 664 lines, 2 to 3 params). | The alternative, a ContextVar read inside the builder, hides the data flow (and `run_in_executor` does not copy context). The growth is two lines. |
| R8 | **One own-key lookup per cloud provider per listing**, sequential, marked `ponytail:`. | 17 cheap checks; each does a store read only when BYOK is on and the provider is allowlisted. Upgrade path: list the user's stored providers once. |
| R9 | **The WebUI clears its model catalog after a key save or delete** (`tldwModels.clearCache()`, which also clears the chat-model cache through its invalidation listener). | Otherwise a new key shows up only after the 5-minute catalog TTL. It refreshes the chat model pickers (the `TldwModels` and chat-model caches and the hooks subscribed to them); the react-query readers of `/llm/providers` refetch on their own staleness (Known limits). |

## Rulings (review round 1, 2026-11-21)

An independent dry run on #3544's head (`08aa1d1c27`) found 0 Blocker, 2 Important and 8 Minor; the base then moved to `ced8df7c15` (an OpenAPI fingerprint and docs wording). Every finding is folded; one sub-point of M8 is ruled against.

| # | Finding | Ruling |
|---|---|---|
| I1 | A provider vanished from every own-key user's picker when the admin denied the model the server's config makes its default: `users_own_key_credentials` resolves with no model, so the runtime checked the config's default and refused the handle (`model_denied`). | **Folded as the review measured it.** `users_own_key_credentials` gains `policy: bool = True`; `policy=False` builds its runtime without `override_snapshot_resolver`, and its docstring says it is for the listing only. `_users_listing` alone passes it. It cannot let a disabled provider or a denied model through: (1) the handle only fetches the provider's model list at its vendor's host (credential source `user` is still required) and never leaves the request; no model call is made with it, and every call path keeps the default `policy=True`; (2) for a disabled provider, `_resolve_commercial_listing` skips the own-key branch, so no model-list request goes out, and `_own_key_entry` lists no model with `enabled: false` (the `provider-disabled` row pins both, `fetched == []`, on all four routes); (3) a denied model is dropped from `models` and `models_info` by `policy_error(m, allowlist=False)` (the `allowlist-and-denylist` row and the I1 test, all four routes; Mutation C); (4) a store outage still fails closed: `get_configured_providers` lists no commercial entry, and `_own_key_entry` reads the store through `capture_provider_override_call_snapshot`, which raises. New test `test_a_denied_config_default_keeps_the_provider_on_an_own_key`; its RED is Mutation D. `usage_quota_deps.py` joins Task 1's files. A disabled provider no longer logs an own-key warning per listing either. |
| I2 | The AC #2 test read only `/llm/providers` `models`; `/llm/models/metadata` (every chat picker's source) flattens `models_info`, and dropping `_own_key_entry`'s `models_info` filter passed. | **Folded.** The test reads each user's models from all four routes and expects the same list on each; it also pins `enabled` and `default_model` (M7) and the model-list requests. Mutation C is added; it fails 2 tests, not the 1 the review predicted, because the new I1 test reads every route too. |
| M1 | Measured values differed: the AC grep also matched the 6 DoD lines; Task 1 GREEN logs 8 pre-existing probe lines; the sweeps log 17; the vitest message reads `"vi.fn()"`; the llm-providers row is `:23`. | **Folded:** `sed -n '/AC:BEGIN/,/AC:END/p'`, and every `Expected:` now carries the measured counts and messages (adjusted for the 2 tests this fold adds). |
| M2 | Docs overstated: only a ready model list is cached (a rejected key is re-sent every listing); the auth bullet dropped `/llm/health` and left "global security"; "refreshes the pickers" is not true of the react-query `/llm/providers` readers. | **Folded:** the Providers doc names `/llm/health`, replaces the "global security" clause and states the cache rule; the BYOK guide and CHANGELOG say "chat model pickers"; Known limits gains the failure-cache and react-query lines. The docs also say which four paths still hold an own key to `allowed_models` (#3544's new wording). |
| M3 | The 128-entry model-list cache is shared with chat's per-call check; own-key listings add one entry per stored key and provider. | **Folded as a Known limit** with the upgrade path (raise the cap, or a separate listing cache); not changed here, as nothing measured it full. |
| M4 | The public routes now resolve credentials on every call (JWT verify and user fetch, API-key validation and usage, a log line per bad key). | **Folded as a Known limit**; it is the cost of R5. |
| M5 | A listing that varies by credential had no `Cache-Control` or `Vary`, so a shared cache could serve one user's listing to another. | **Folded:** `listing_user_id` takes the dependency's `Response`, sets `Vary: Authorization, X-API-KEY, Cookie` on every listing and `Cache-Control: private, no-store` when it names a user; pinned by the dependency test and the route test. No OpenAPI field changes. |
| M6 | Server-wide provider health applies to own-key entries. | **Folded as a Known limit** (pre-existing coupling). |
| M7 | Review Focus 4 was pinned only through a patched `get_auth_principal`; `_own_key_entry`'s `enabled` and `default_model` were unpinned. | **Folded:** `test_bad_credentials_on_a_route_get_the_public_listing` sends a real bad `X-API-KEY` through a route (200, the anonymous body); the AC #2 test pins `enabled` and `default_model`, and the I1 test the default's fallback. |
| M8 | The private-file grep misses `pr_body_webui_byok.md` and `run15583.zsh`; the ADR-004 waiver for this PR is not in the recorded waivers. | **Waiver folded:** the controller's brief states it; Task 4 asks the controller to confirm it is recorded before `gh pr ready`. **Grep ruled against:** those two names are the steps' own working files, named as TASK-15582's committed plan names its own; the grep keeps targeting review and brief artifacts. |

## The WebUI's model pickers (R1: no filter changes)

| Gate | Reads | Pickers behind it |
|---|---|---|
| `TldwModels.isSelectableChatModel` (`apps/packages/ui/src/services/tldw/TldwModels.ts:754`), through `getChatModels` and `getCachedChatModels` | `is_configured`, `provider_is_configured`, `provider_enabled`, `availability` of each `/llm/models/metadata` entry | `fetchChatModels` (`services/tldw-server.ts:348`) and `useChatModelCatalog` (`hooks/useChatModelCatalog.ts`): ModelSelect, ModelSelectOption, Playground, PlaygroundForm, PlaygroundChat, Sidepanel ControlRow and form, QuickChatHelperModal, DocumentGeneratorDrawer, CurrentChatModelSettings, Characters Manager and GenerateCharacterPanel, CompanionHomeShell, ModelPlaygroundChat, ModelClassificationPanel, OnboardingConnectForm, GenerateDocumentDialog, GenerateBoardDialog, the Models settings page, both quick-chat popouts; direct `fetchChatModels` callers QuickIngestModal, ContentReviewPage, useSidepanelChatHandoffImport, ResearchWorkspace ChatPane and SharedWorkspaceChatPane, QuickIngestSettings, `chat-model-validation`, useSelectedModelSamplingSupport; direct `getChatModels` callers AnalysisModal, WorkflowEditor `dynamic-options.ts`, useArtifactGeneration, useModelSelector |
| `isConfiguredUsableModel` / `isCatalogOnlyModel` (`apps/packages/ui/src/hooks/playground/modelSelectorUtils.ts:195-270`) | `catalog_only`, the configured and usable flags, `availability`, `status` | useModelSelector ("not set up" rows of the All models scope), useComposerModelReadiness, GenerateDocumentDialog, GenerateBoardDialog, the Models settings page |
| `isUnavailableModel` in `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx:2944-2953` | `isConfigured`, `providerEnabled`, `availability` | Research Studio |
| `getModelConfiguredSignal` (`apps/packages/ui/src/components/Option/Models/index.tsx:78-95`) | the configured flags | the Models settings page |
| none: they list `/llm/providers` as it comes | the entry's `models` | Flashcards GeneratePanel, QAQuickSettings, AnswerModelMenu, usePromptInteractions, ExecutePlayground and StudioModelTargetFields (`services/prompt-studio.ts`), WorldBookEntryManager, WorkflowEditor `dynamic-options.ts`, LlamaCppAdvancedControls, MlxAdminPage |

An own-key entry has `is_configured: true`, `provider_enabled: true`, `availability: "enabled"` (or the registry's), `catalog_only: false` and its models, so every gate keeps it; the gates' existing tests (`TldwModels.test.ts:1197-1303`) pin that they keep such a model and drop a disabled one. The WebUI's catalog cache is already scoped by a digest of the access token or API key (`TldwModels.buildCacheScope`, `derivePromptAssistAuthorizationRevision`), and cookie-session catalogs are never persisted, so a shared browser never shows one user's own-key catalog to the next (AC #3 on the client).

## Assumed from TASK-15582

The controller updates this plan if TASK-15582's review changes any of these.

1. `ProviderOverrideCallSnapshot.policy_error(model: str | None, *, allowlist: bool = True) -> dict[str, str] | None`; with `allowlist=False` it returns only `provider_disabled` and `model_denied`.
2. `LLMProviderOverride` has `denied_models: list[str] | None = None`, set by keyword.
3. `apply_llm_provider_overrides_to_listing(payload, *, overrides=None)` is the server-key listing: it applies `allowed_models` and `denied_models` to every entry it gets.
4. `capture_provider_override_call_snapshot(provider: str) -> ProviderOverrideCallSnapshot` (unchanged from `dev`).
5. `users_own_key_credentials(user_id, provider, *, model=None)` (`API_Deps/usage_quota_deps.py`) issues a handle for a provider whose override has `allowed_models` (the runtime checks every-key rules before resolution with `allowlist=False`, and skips the allowlist after it for credential source `user`) and returns None for `is_enabled=False`, and also when the admin denies the model the server's config makes that provider's default (the runtime checks `model or configured_provider_model_from_snapshot(...)`; review round 1, I1). Task 1 adds `policy: bool = True` to it; only the listing passes `policy=False`.
6. `set_llm_provider_overrides_cache_for_tests(...)`, and the root conftest's healthy empty store restored after each test.
7. TASK-15582's CHANGELOG bullets (under `### Changed` and `### Fixed`) and its edit of the BYOK guide's Usage Quotas paragraph have merged; that paragraph still ends its own-key sentence with "(the WebUI's model picker does not list such a provider yet, TASK-15583)". #3544's `ced8df7c15` adds, before that sentence, one naming the four paths that still hold an own key to `allowed_models` (shared-workspace chat, the persona voice preflight, embeddings, text-to-speech), and the same list to its CHANGELOG bullet; the anchors this plan edits are unchanged, and `### Fixed` holds TASK-15582's bullet.
8. The function-size entry `llm_providers:get_configured_providers` is still `{"lines": 662, "params": 2, "reason": null}`.

## Decisions for the owner

None. The owner's 2026-11-19 rule and 2026-11-21 decision settle the product rules; R1 to R9 and the review-round rulings are implementation rulings.

## Known limits

- **Team and org shared keys do not list a provider** the server has no key for, though chat would run on them (R3; TASK-15582 R1 counts them as the server's). Upgrade path: build the handle with the request's team and org scope.
- **A key stored for an operator's endpoint** (a custom OpenAI-compatible slot, a local server, a stored base URL) does not list the provider (R3).
- **No `[API]` section in the server config** means no commercial provider is listed at all, own key or not (pre-existing early return in `get_configured_providers`).
- **The override's `config`** (model list, `default_model`, ranking) shapes only server-key entries (R4).
- **Auto-routing (`model=auto`) and the scheduled-task target bound** still read the server's listing; they run on the server's key.
- **Two users who stored the same key share one cached model list** (R2): it is that key's list, so neither sees anything their key does not list.
- **The first listing after a key is saved** makes one model-list request per own key (5-second budget). Only a list the provider returns is cached (5 minutes per key); an unreachable list or a server error is retried after 20 seconds, and a rejected key (`auth_failed`) on every listing, so a rejected own key costs one vendor request per listing, two per WebUI catalog fetch (metadata, then providers). A failing vendor list makes the provider unusable, as it does for the server's key (review round 1, M2).
- **The model-list cache is shared** (`provider_model_inventory._MAX_CACHE_ENTRIES = 128`) by server keys, chat's per-call model check and now each own-key listing (one entry per stored key and provider). Past about 128 active (key, provider) pairs in 5 minutes, LRU eviction makes chat calls fetch the list again (up to 5 s) and sends more vendor requests. Upgrade path: raise the cap, or give listings their own cache (review round 1, M3).
- **The listings now resolve credentials on every call** (a JWT verified and its user fetched; a multi-user API key validated and its use recorded; a bad key logs `Invalid X-API-KEY presented`). Benign; it is the cost of R5 (review round 1, M4).
- **Server-wide provider health applies to own-key entries:** a provider the provider manager marks unhealthy from server-key failures shows `provider_health_unavailable` to own-key users too (pre-existing coupling; review round 1, M6).
- **The react-query readers of `/llm/providers`** (Flashcards GeneratePanel, Prompt Studio, LlamaCppAdvancedControls) refetch on their own staleness (up to 60 s), not when a key is saved (review round 1, M2).
- **Bedrock, Z.AI and MiniMax** have no model listing: on an own key they show the configured and pricing-catalog models, as on the server's key.
- **Connecting OpenAI by OAuth on the Models page** does not clear the catalog by itself; that page's Refresh button does (unchanged).
- **The listings stay public** (R5), as on `dev`. A user's listing carries `Cache-Control: private, no-store` and every listing `Vary: Authorization, X-API-KEY, Cookie`, so a shared cache neither stores a user's listing nor serves one caller's to another (review round 1, M5).

---

### Task 0: Branch, run wrapper, backlog, plan copy, draft PR, baseline

**Files:**
- Create: `<scratchpad>/run15583.zsh` (not committed)
- Modify: `backlog/tasks/task-15583 - The-WebUI-cannot-pick-a-provider-the-server-has-no-key-for-even-with-the-user-s-own-key.md` (backlog-py)
- Create: `Docs/superpowers/plans/2026-11-21-webui-byok-provider.md` (this plan, verbatim)
- Create: `<plans dir>/pr_body_webui_byok.md` (not committed)

- [ ] **Step 1: Branch, and check the base has TASK-15582.**

```bash
git fetch origin dev
git switch -c feat/webui-byok-provider-15583 origin/dev
git log --oneline -1
grep -c "denied_models" tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py
grep -n "def policy_error" tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py
grep -n "llm_providers:get_configured_providers" tldw_Server_API/tests/lint/function_size_baseline.json
```

Expected: the head is TASK-15582's merge (or later); the `grep -c` prints a number above 0; `def policy_error(self, model: str | None, *, allowlist: bool = True) -> dict[str, str] | None:`; the baseline line reads `{"lines": 662, "params": 2, "reason": null}`. If any differs, stop and report: "Assumed from TASK-15582" no longer holds.

- [ ] **Step 2: The run wrapper.** Write `<scratchpad>/run15583.zsh`:

```zsh
#!/bin/zsh
# run15583.zsh <name> <pytest args...>: one disk-guarded pytest run from the current directory (the worktree root).
# Prints failures and the summary, then the network-probe line count; leaves no file behind.
# With FAILED_TO=<file> it also writes the sorted FAILED/ERROR ids there (the Task 0/4 sweep; delete it after use).
S=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad
PL=/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
n=$1; shift
T=$S/r15583_$n
rm -rf "${T:?}"; mkdir -p "$T"
export PATH="$(dirname $PY):$PATH" PYTHONPATH=$PL:$PL/pyext NETPROBE_LOG=$S/r15583_$n.net
rm -f "$NETPROBE_LOG"
out=$(TMPDIR=$T zsh $S/disk_guard.zsh $T -- $PY -m pytest --basetemp=$T/bt -p no:cacheprovider -p netprobe_15427 -q "$@" 2>&1)
[[ -n $FAILED_TO ]] && print -r -- "$out" | grep -E '^(FAILED|ERROR) ' | sed -E 's/ - .*//' | sort -u > "$FAILED_TO"
print -r -- "$out" | grep -E '^(FAILED|ERROR) |[0-9]+ (passed|failed|skipped|errors?)|DISK_GUARD' | tail -40
print -r -- "netprobe lines: $([[ -f $NETPROBE_LOG ]] && wc -l < $NETPROBE_LOG || print 0)"
rm -f "$NETPROBE_LOG"
```

Expected: `zsh -n <scratchpad>/run15583.zsh` prints nothing.

- [ ] **Step 3: Start TASK-15583 and amend AC #2.** backlog-py removes ACs by index and appends new ones, renumbering, so ACs 2 to 5 are removed and re-added in order (3 to 5 verbatim):

```bash
BACKLOG task edit TASK-15583 -s "In Progress" \
  --remove-ac 2 --remove-ac 3 --remove-ac 4 --remove-ac 5 \
  --ac "A provider the admin explicitly disables (is_enabled=False), or a model on its denied_models, stays unselectable for every user, own key or not; a model off its allowed_models is selectable only for a user whose own key serves that provider (owner's decision, 2026-11-21, TASK-15582)" \
  --ac "The listing for one user never reveals another user's keys or models, and a server-key-only user's listing is unchanged" \
  --ac "The docs that describe the provider listing say how a user's own key affects it" \
  --ac "With no server key there is no model inventory, so the picker gets the provider's models from the user's own key (or accepts a typed model ID), never from another user's key" \
  --append-notes "AC #2 amended per the owner's decision of 2026-11-21 (TASK-15582, chose 'Server key only'). It read: 'A provider the admin explicitly disables (is_enabled=False), or a model off its allowed_models, stays unselectable for every user, own key or not'. A model off allowed_models is now only not offered on the server's key and is usable with a user's own key; denied_models is the explicit model disallow, refused on every key. Plan: Docs/superpowers/plans/2026-11-21-webui-byok-provider.md."
sed -n '/AC:BEGIN/,/AC:END/p' "backlog/tasks/task-15583 - The-WebUI-cannot-pick-a-provider-the-server-has-no-key-for-even-with-the-user-s-own-key.md"
```

Expected: `TASK-15583 - The WebUI cannot pick a provider the server has no key for, even with the user's own key (In Progress)`; `sed` prints the AC block: its two markers around 5 lines `- [ ] #1` to `- [ ] #5`, `#2` the new text, `#3` to `#5` the old #3 to #5 word for word (as the review's dry run measured).

- [ ] **Step 4: Copy the plan, check it, commit.** Copy this file to `Docs/superpowers/plans/2026-11-21-webui-byok-provider.md`.

```bash
grep -n -E '(brief|plan|final_review|impl_report|design_review)_[0-9a-z_]*\.md' Docs/superpowers/plans/2026-11-21-webui-byok-provider.md
grep -n -E ' +$' Docs/superpowers/plans/2026-11-21-webui-byok-provider.md
git add -A backlog/tasks Docs/superpowers/plans/2026-11-21-webui-byok-provider.md
git commit -m "chore(backlog): start TASK-15583 with its plan; amend AC #2 (TASK-15583)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: both greps print nothing; the commit lists 2 files.

- [ ] **Step 5: Push and open the draft PR.** Write `<plans dir>/pr_body_webui_byok.md` with a one-paragraph Why (the owner's 2026-11-19 rule and 2026-11-21 decision), "Work in progress: see Docs/superpowers/plans/2026-11-21-webui-byok-provider.md", the `## Change summary` section from Global Constraints, and the closing `🤖 Generated with [Claude Code](https://claude.com/claude-code)` line.

```bash
git push -u origin feat/webui-byok-provider-15583
gh pr create --base dev --draft --title "feat(llm): the WebUI offers a provider the user's own key serves (TASK-15583)" \
  --body-file <plans dir>/pr_body_webui_byok.md
```

Expected: a draft PR URL against `dev`.

- [ ] **Step 6: Baseline.** The sweep set is every test file that reads the listing module or routes:

```bash
git grep -l -E 'endpoints\.llm_providers|endpoints import llm_providers|/api/v1/llm/(providers|models)' -- 'tldw_Server_API/tests/*.py' \
  | grep -v conftest.py | grep -v -E 'tests/(e2e|server_e2e_tests|frontend_e2e)/' > <scratchpad>/sweep_files_15583.txt
wc -l < <scratchpad>/sweep_files_15583.txt
FAILED_TO=<scratchpad>/sweep0_15583.txt RUN sweep0 -n 2 $(cat <scratchpad>/sweep_files_15583.txt)
wc -l < <scratchpad>/sweep0_15583.txt
```

Expected: 20 files. Measured at #3544's head: 488 passed, 0 failed, `netprobe lines: 17` (pre-existing; the same count in Task 4), and `sweep0_15583.txt` empty. Only Task 4's diff against it matters. Keep both files until then.

---

### Task 1: The listings name the requesting user's own keys

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/llm_providers.py` (imports `:1-95`; `_resolve_commercial_listing` `:732-785`; `_resolve_commercial_listings` `:798-837`; `get_configured_providers` `:840-850`, `:1146-1151`; `get_configured_providers_async` `:1504-1517`; new helpers before the `# Endpoints:` banner `:1766-1768`; the four routes `:1770-2085`)
- Modify: `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py` (`users_own_key_credentials` `:91-130`; review round 1, I1)
- Modify: `tldw_Server_API/tests/lint/function_size_baseline.json` (`:25`)
- Test: `tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py` (new)

**Interfaces:**
- Consumes: "Assumed from TASK-15582" items 1 to 6; `users_own_key_credentials(user_id: Any, provider: str | None, *, model: str | None = None, policy: bool = True) -> Any` (this task adds `policy`); on `dev`: `CLOUD_MODEL_PROVIDERS` and `inventory_endpoint_scope(provider, base_url, *, endpoint_provenance=None)` (`core/LLM_Calls/provider_model_inventory.py`), `auth_deps.get_auth_principal(request)`, the handle's `.api_key`, `.app_config`, `.endpoint_provenance`.
- Produces (all in `llm_providers.py`):
  - `async def listing_user_id(request: Request, response: Response) -> int | None` (a FastAPI dependency; tests override it). It sets `Vary` on every listing and `Cache-Control: private, no-store` when it names a user.
  - `get_configured_providers(include_deprecated: bool = False, refresh_openrouter: bool = False, own_keys: Mapping[str, Any] | None = None) -> dict[str, Any]`, and the same keyword on `get_configured_providers_async`.
  - `async def _users_listing(user_id: Any, **options: Any) -> dict[str, Any]`, `def _own_key_entry(entry: dict[str, Any]) -> dict[str, Any]`, `def _listing_on_key(provider_name: str, api_key: str | None, app_config: dict[str, Any] | None, refresh_openrouter: bool, *, endpoint_provenance: str | None = None) -> _CommercialListing`.

- [ ] **Step 1: Write the failing tests.** Create `OWN`:

```python
"""The provider listings for a user with their own provider key (TASK-15583).

A provider the server has no key for is listed on the requesting user's own stored key: usable, with the models
that key lists. Each listing is built from the requesting user's key only; an anonymous or keyless request gets
the server's listing. The admin override (owner's decision, 2026-11-21, TASK-15582) hides a disabled provider and
a denied model for everyone, and a model off ``allowed_models`` only on the server's key.

Only core modules are patched (none under ``app/api/v1/endpoints``): the requesting user comes from a dependency
override, and the vendor's model list from a fake ``_http_fetch`` that answers by the request's key.
"""

from __future__ import annotations

import json
from configparser import ConfigParser
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import llm_providers
from tldw_Server_API.app.core.Provider_Credentials.byok_config import PROVIDER_APP_CONFIG_KEYS
from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError, ResolvedByokCredentials
from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
    LLMProviderOverride,
    set_llm_provider_overrides_cache_for_tests,
)

pytestmark = pytest.mark.unit

_ROUTES = ("/api/v1/llm/providers", "/api/v1/llm/models/metadata", "/api/v1/llm/providers/openai", "/api/v1/llm/models")


def _listing_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stored: dict[tuple[int, str], str],
    server_key: str | None = None,
    stored_base_url: str | None = None,
    stored_model: str | None = None,
) -> tuple[TestClient, SimpleNamespace]:
    """The four listing routes over a config with an [API] section, offline; returns the client and the requester.

    ``stored`` maps (user id, provider) to that user's own key at the vendor; with ``stored_base_url`` each stored
    key is resolved for that endpoint instead, and with ``stored_model`` it carries that default model, as a BYOK
    resolution keeps the server config's ``model``. ``server_key`` is the server's OpenAI key (none by default).
    """
    parser = ConfigParser()
    parser.read_dict({"API": {"default_api": "openai", **({"openai_api_key": server_key} if server_key else {})}})
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "MOCK_OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: parser)
    monkeypatch.setattr("tldw_Server_API.app.core.config_llm_providers.get_api_keys", lambda: {})
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Provider_Credentials.byok_runtime.resolve_static_server_fallback",
        lambda _provider: SimpleNamespace(app_config={}),
    )
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.provider_manager._provider_manager", None)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver.resolve_tokenizer_metadata", lambda *_a, **_k: {}
    )
    monkeypatch.setattr("tldw_Server_API.app.core.Image_Generation.listing.list_image_models_for_catalog", lambda: [])

    async def serves(user_id: Any, provider: str) -> bool:
        return (user_id, provider) in stored

    async def resolve(provider: str, **kwargs: Any) -> ResolvedByokCredentials:
        key = stored.get((kwargs.get("user_id"), provider))
        if key is None:
            raise ByokResolutionError("missing_provider_credentials", provider)
        fields = {"api_base_url": stored_base_url, "model": stored_model}
        section = {name: value for name, value in fields.items() if value}
        return ResolvedByokCredentials(
            provider=provider, api_key=key, app_config={PROVIDER_APP_CONFIG_KEYS[provider]: section},
            credential_fields={}, source="user", allowlisted=True,
        )

    monkeypatch.setattr("tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps.users_own_key_serves", serves)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Provider_Credentials.provider_credential_runtime.resolve_byok_credentials", resolve
    )
    requester = SimpleNamespace(user=None)
    app = FastAPI()
    app.include_router(llm_providers.router, prefix="/api/v1")
    app.dependency_overrides[llm_providers.listing_user_id] = lambda: requester.user
    app.state.llm_manager = SimpleNamespace(llamacpp_supervisor=None)
    return TestClient(app), requester


def _vendor(monkeypatch: pytest.MonkeyPatch, answers: dict[str, list[str] | int]) -> list[str]:
    """The vendor's model list for each key (model IDs, or an HTTP status); returns each request's credential."""
    asked: list[str] = []

    def fetch(**request: Any) -> httpx.Response:
        credential = request["headers"]["Authorization"]
        asked.append(credential)
        answer = answers[credential.removeprefix("Bearer ")]
        if isinstance(answer, int):
            return httpx.Response(answer)
        return httpx.Response(200, json={"data": [{"id": model} for model in answer]})

    monkeypatch.setattr("tldw_Server_API.app.core.LLM_Calls.provider_model_inventory._http_fetch", fetch)
    return asked


def _openai_models(route: str, body: Any) -> list[str]:
    """The OpenAI models one listing route returns."""
    if route.endswith("/models/metadata"):
        return [model["name"] for model in body["models"] if model["provider"] == "openai"]
    if route.endswith("/llm/models"):
        return [model.removeprefix("openai/") for model in body if model.startswith("openai/")]
    return (body if route.endswith("/openai") else _openai_in(body))["models"]


def _openai_in(body: dict[str, Any]) -> dict[str, Any]:
    """The OpenAI entry of a ``/llm/providers`` body."""
    return next(p for p in body["providers"] if p["name"] == "openai")


def _openai_entry(client: TestClient) -> dict[str, Any]:
    return _openai_in(client.get(_ROUTES[0]).json())


def _readiness(entry: dict[str, Any]) -> tuple[Any, Any, Any]:
    return entry["is_configured"], entry["provider_enabled"], entry["readiness_reason_code"]


def test_each_user_lists_a_keyless_provider_on_their_own_key_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK-15583 (AC #1, #3, #5): the server has no OpenAI key; users 7 and 9 stored their own, user 8 did not. Every
    listing route gives each user the models their own key lists, never another user's; user 8's listing is the
    anonymous one; no key appears; each key's model list is fetched once (cached by its digest)."""
    client, requester = _listing_app(monkeypatch, stored={(7, "openai"): "sk-user-7", (9, "openai"): "sk-user-9"})
    asked = _vendor(monkeypatch, {"sk-user-7": ["gpt-7"], "sk-user-9": ["gpt-9"]})

    listings: dict[Any, dict[str, Any]] = {}
    for user in (7, 8, 9, None):
        requester.user = user
        listings[user] = {route: client.get(route).json() for route in _ROUTES}
    requester.user = 7
    again = _openai_entry(client)

    models = {
        user: [_openai_models(route, body) for route, body in by_route.items()] for user, by_route in listings.items()
    }
    assert (models, again["models"], asked) == (
        {7: [["gpt-7"]] * 4, 8: [[]] * 4, 9: [["gpt-9"]] * 4, None: [[]] * 4},
        ["gpt-7"],
        ["Bearer sk-user-7", "Bearer sk-user-9"],
    )
    entries = {user: _openai_in(listings[user][_ROUTES[0]]) for user in (7, 8)}
    assert (
        _readiness(entries[7]),
        _readiness(entries[8]),
        listings[8] == listings[None],
        "sk-user-" in json.dumps(listings),
    ) == ((True, True, None), (False, False, "provider_not_configured"), True, False)


@pytest.mark.parametrize(
    ("override", "expected", "fetched"),
    [
        (
            {"allowed_models": ["gpt-allowed"], "denied_models": ["gpt-denied"]},
            {
                7: (["gpt-allowed", "gpt-off-list", "gpt-7-only"], True, True, "gpt-allowed"),
                8: (["gpt-allowed"], True, True, "gpt-allowed"),
            },
            ["Bearer sk-user-7", "Bearer sk-server"],
        ),
        ({"is_enabled": False}, {7: ([], False, False, None), 8: ([], False, False, None)}, []),
    ],
    ids=["allowlist-and-denylist", "provider-disabled"],
)
def test_the_admin_override_holds_an_own_key_only_to_what_every_key_refuses(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, Any],
    expected: dict[int, tuple[list[str], bool, bool, str | None]],
    fetched: list[str],
) -> None:
    """TASK-15583 (AC #2 as the owner's 2026-11-21 decision amends it; Review Focus 3): the server has a key and user 7
    has their own. A disabled provider and a denied model are unselectable for both on all four routes, and a disabled
    provider gets no model-list request on either key; a model off allowed_models is listed only for user 7, whose own
    key serves the provider and wins over the server's, as in chat. Each user's entry also gives ``provider_enabled``,
    ``enabled`` and ``default_model``."""
    client, requester = _listing_app(monkeypatch, server_key="sk-server", stored={(7, "openai"): "sk-user-7"})
    asked = _vendor(monkeypatch, {
        "sk-server": ["gpt-allowed", "gpt-off-list", "gpt-denied"],
        "sk-user-7": ["gpt-allowed", "gpt-off-list", "gpt-denied", "gpt-7-only"],
    })
    set_llm_provider_overrides_cache_for_tests({"openai": LLMProviderOverride(provider="openai", **override)})

    listed = {}
    for user in (7, 8):
        requester.user = user
        entry = _openai_entry(client)
        routes = [_openai_models(route, client.get(route).json()) for route in _ROUTES]
        listed[user] = (routes, entry["provider_enabled"], entry["enabled"], entry["default_model"])

    assert (listed, asked) == (
        {user: ([models] * 4, *rest) for user, (models, *rest) in expected.items()},
        fetched,
    )


def test_a_denied_config_default_keeps_the_provider_on_an_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review round 1, I1: the admin denies the model the server's config makes OpenAI's default. A model denial is
    not a provider disable, so user 7's own key still lists OpenAI on every route, without that model, with another
    default. (Resolved with the policy check, the runtime would refuse the handle for that default.)"""
    client, requester = _listing_app(monkeypatch, stored={(7, "openai"): "sk-user-7"}, stored_model="gpt-default")
    _vendor(monkeypatch, {"sk-user-7": ["gpt-default", "gpt-other"]})
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", denied_models=["gpt-default"])}
    )
    requester.user = 7

    entry = _openai_entry(client)
    routes = [_openai_models(route, client.get(route).json()) for route in _ROUTES]

    assert (routes, _readiness(entry), entry["default_model"]) == ([["gpt-other"]] * 4, (True, True, None), "gpt-other")


def test_a_key_stored_for_an_operators_endpoint_does_not_list_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK-15583 (Review Focus 1): a user key the server sends off the vendor's host is not the user's own spend
    (ADR-083, user_unmetered), so it does not list a provider the server has no key for, and nothing is fetched."""
    client, requester = _listing_app(
        monkeypatch, stored={(7, "openai"): "sk-user-7"}, stored_base_url="https://models.operator.example/v1"
    )
    asked = _vendor(monkeypatch, {"sk-user-7": ["gpt-7"]})
    requester.user = 7

    entry = _openai_entry(client)

    assert (_readiness(entry), entry["models"], asked) == ((False, False, "provider_not_configured"), [], [])


def test_a_rejected_own_key_lists_the_provider_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK-15583 (Review Focus 2): when the vendor refuses the user's key, the provider is listed on that key as
    unusable (auth_failed), with no models and nothing else to fall back to."""
    client, requester = _listing_app(monkeypatch, stored={(7, "openai"): "sk-user-7"})
    _vendor(monkeypatch, {"sk-user-7": 401})
    requester.user = 7

    entry = _openai_entry(client)

    assert (_readiness(entry), entry["models"]) == ((True, False, "auth_failed"), [])


def test_bad_credentials_on_a_route_get_the_public_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review round 1, M7 (Review Focus 4): through the real dependency, a bad X-API-KEY gets 200 and the anonymous
    body, not a 401; neither is marked private, and both vary by credential (M5)."""
    client, _requester = _listing_app(monkeypatch, stored={})
    client.app.dependency_overrides.clear()

    anonymous = client.get(_ROUTES[0])
    bad = client.get(_ROUTES[0], headers={"X-API-KEY": "not-a-key"})

    assert (
        anonymous.status_code,
        bad.status_code,
        bad.json() == anonymous.json(),
        bad.headers.get("cache-control"),
        anonymous.headers.get("vary"),
    ) == (200, 200, True, None, "Authorization, X-API-KEY, Cookie")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("principal", "expected"),
    [
        (SimpleNamespace(user_id=7), (7, "private, no-store")),
        (SimpleNamespace(user_id=None), (None, None)),
        (401, (None, None)),
        (403, (None, None)),
        (503, (None, None)),
    ],
    ids=["user", "service-principal", "401", "403", "503"],
)
async def test_the_listing_names_its_user_and_never_refuses(
    monkeypatch: pytest.MonkeyPatch, principal: Any, expected: tuple[int | None, str | None]
) -> None:
    """TASK-15583 (Review Focus 4; review round 1, M5): the listings stay public. No credentials, bad ones, or any
    refusal of the auth layer give no user (the server's listing), never an error; a principal without a user is no
    user either. A user's listing is marked private, and every listing varies by credential."""

    async def get_auth_principal(_request: Request) -> Any:
        if isinstance(principal, int):
            raise HTTPException(status_code=principal)
        return principal

    monkeypatch.setattr("tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal", get_auth_principal)
    response = Response()

    user_id = await llm_providers.listing_user_id(Request({"type": "http", "headers": []}), response)

    assert (user_id, response.headers.get("cache-control"), response.headers.get("vary")) == (
        *expected,
        "Authorization, X-API-KEY, Cookie",
    )
```

- [ ] **Step 2: RED.**

```bash
RUN t1red -n 2 tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py
```

Expected: 12 failed, each `AttributeError: module 'tldw_Server_API.app.api.v1.endpoints.llm_providers' has no attribute 'listing_user_id'` (the review measured the same for the 10 tests before this fold). Step 6 measures the behaviour RED by mutation once the code exists.

- [ ] **Step 3: Implement** (in `usage_quota_deps.py`, then `LP`).
  - In `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py`, `users_own_key_credentials` gains a keyword (review round 1, I1; the review measured this fix):

```python
async def users_own_key_credentials(
    user_id: Any, provider: str | None, *, model: str | None = None, policy: bool = True
) -> Any:
```

  its docstring gains the paragraph "``policy=False`` is for the provider listing only (TASK-15583): it skips the admin override check, which the listing applies itself to what it lists. Never pass it for a call."; and the runtime it builds takes `override_snapshot_resolver=capture_provider_override_call_snapshot if policy else None,`. Every existing caller keeps the default.
  - Imports in `LP`: `from collections.abc import Mapping`; `Depends` and `Response` in the `fastapi` import; `from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import users_own_key_credentials` with the other `from tldw_Server_API...` imports; `CLOUD_MODEL_PROVIDERS,` first in the existing `provider_model_inventory` import (`:56-59`); `import tldw_Server_API.app.api.v1.API_Deps.auth_deps as _auth_deps` with the other module aliases (`:84-93`). Call it as `_auth_deps.get_auth_principal(...)` (the test patches it there).
  - New, before `_with_pricing_models` (`:788`), the tail of `_resolve_commercial_listing` moved into its own function:

```python
def _listing_on_key(
    provider_name: str,
    api_key: str | None,
    app_config: dict[str, Any] | None,
    refresh_openrouter: bool,
    *,
    endpoint_provenance: str | None = None,
) -> _CommercialListing:
    """(API key, configured, live inventory) for a commercial provider listed on ``api_key``.

    The key is the server's, or the requesting user's own (TASK-15583); discovery runs as chat's does on it (the
    gateway's ``_provider_inventory_endpoint``). Its cache is keyed by the key's SHA-256 digest, so a listing only
    ever shows a model list fetched with the key it is listed on.
    """
    if not api_key:
        return None, False, None
    if not has_model_listing(provider_name):
        return api_key, True, None  # no listing: the configured models are shown, as before
    base_url = _provider_model_inventory.resolve_provider_models_base_url(
        provider_name, app_config, credentials_resolved=True,
    )
    if base_url is None:
        return api_key, True, ModelDiscoveryResult("unsupported")
    scope = inventory_endpoint_scope(provider_name, base_url, endpoint_provenance=endpoint_provenance)
    scope_option: dict[str, Any] = {"configured_endpoint": scope} if scope is not None else {}
    return api_key, True, _provider_model_inventory.discover_provider_models(
        provider_name, api_key, base_url=base_url,
        force_refresh=refresh_openrouter and provider_name == "openrouter", **scope_option,
    )
```

  - `_resolve_commercial_listing`: a seventh parameter `own_key: Any = None`; its docstring gains "With ``own_key`` (the requesting user's own-key handle, TASK-15583) the provider is listed on that key, unless the admin disabled it."; its first statement:

```python
    if own_key is not None and not (override is not None and override.is_enabled is False):
        return _listing_on_key(
            provider_name, valid_provider_api_key(own_key.api_key), own_key.app_config, refresh_openrouter,
            endpoint_provenance=own_key.endpoint_provenance,
        )
```

  and everything after `if fallback is not None: api_key_value = valid_provider_api_key(fallback.api_key)` (`:770-785`) becomes:

```python
    fallback_config = fallback.app_config if fallback is not None else static.app_config
    return _listing_on_key(
        provider_name, api_key_value, dict(fallback_config) if fallback_config is not None else None,
        refresh_openrouter,
    )
```

  - `_resolve_commercial_listings`: a sixth parameter `own_keys: Mapping[str, Any] | None = None`; its inner `resolve` passes `(own_keys or {}).get(provider_name)` as the seventh argument of `_resolve_commercial_listing`.
  - `get_configured_providers`: the keyword `own_keys: Mapping[str, Any] | None = None,` after `refresh_openrouter`; in the docstring, the two lines `Returns:` / `Dictionary containing provider information` become the two lines ```own_keys`` (provider -> the requesting user's own-key handle) lists each of those commercial providers on` / `that key instead of the server's (TASK-15583). Returns a dictionary containing provider information.`; the call at `:1147-1149` passes it on a line of its own:

```python
            _resolve_commercial_listings(
                provider_mappings, config_parser, api_keys_by_provider, provider_overrides, refresh_openrouter,
                own_keys,
            )
```

  - `get_configured_providers_async`: the same keyword, passed as `own_keys=own_keys` in its `partial(...)`.
  - New, after `_llamacpp_supervisor_from_request` (before the `# Endpoints:` banner):

```python
async def listing_user_id(request: Request, response: Response) -> int | None:
    """The requesting user's id for a provider listing (TASK-15583), or None.

    The listings stay public: a request with no credentials, or one the auth layer refuses (any HTTPException),
    gets the server's listing, never an error. A principal with no user (a service) is no user. Every listing
    varies by credential, and a user's is private, so a shared cache neither stores a user's listing nor serves one
    caller's to another.
    """
    response.headers["Vary"] = "Authorization, X-API-KEY, Cookie"
    try:
        principal = await _auth_deps.get_auth_principal(request)
    except HTTPException:
        return None
    user_id = getattr(principal, "user_id", None)
    if not isinstance(user_id, int):
        return None
    response.headers["Cache-Control"] = "private, no-store"
    return user_id


def _own_key_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """A provider entry listed on the requesting user's own key, held only to what every key refuses (TASK-15583).

    The owner's 2026-11-21 decision (TASK-15582): a disabled provider and a model on ``denied_models`` are refused
    on every key, a model off ``allowed_models`` only on the server's, so it stays. The override's ``config`` (its
    model list, default and ranking) shapes the server key's offer and is not applied.
    """
    snapshot = _llm_provider_overrides.capture_provider_override_call_snapshot(str(entry.get("name") or ""))
    models = [m for m in entry.get("models") or [] if snapshot.policy_error(m, allowlist=False) is None]
    default_model = entry.get("default_model")
    return {
        **entry,
        "enabled": snapshot.policy_error(None, allowlist=False) is None,
        "models": models,
        "models_info": [
            info for info in entry.get("models_info") or [] if isinstance(info, dict) and info.get("name") in models
        ],
        "default_model": default_model if default_model in models else (models[0] if models else None),
    }


async def _users_listing(user_id: Any, **options: Any) -> dict[str, Any]:
    """The provider listing for one requester, with the admin provider override applied (TASK-15583).

    Each commercial provider the user's own stored key serves (``users_own_key_credentials``: BYOK on, the provider
    allowlisted, an active key, the vendor's host) is listed on that key: its readiness and models come from that
    key's own model list. The handle is asked for without the override check (``policy=False``): this listing applies
    the override itself (``_resolve_commercial_listing`` skips a disabled provider; ``_own_key_entry`` refuses what
    every key refuses), so a denied default model cannot hide the provider. Every other entry, and the whole listing
    for a request with no user (anonymous, or a direct call), is the server's, exactly as before.
    """
    own_keys: dict[str, Any] = {}
    if isinstance(user_id, int):
        # ponytail: one own-key lookup per cloud provider (a store read each when BYOK is on); list the user's
        # stored providers once if listings show up in profiles.
        for provider in sorted(CLOUD_MODEL_PROVIDERS):
            handle = await users_own_key_credentials(user_id, provider, policy=False)
            if handle is not None:
                own_keys[provider] = handle
    listing = await get_configured_providers_async(**options, own_keys=own_keys)
    if not own_keys:
        return _llm_provider_overrides.apply_llm_provider_overrides_to_listing(listing)
    providers: list[Any] = []
    for entry in listing.get("providers", []):
        if isinstance(entry, dict) and entry.get("name") in own_keys:
            providers.append(_own_key_entry(entry))
        else:
            applied = _llm_provider_overrides.apply_llm_provider_overrides_to_listing({"providers": [entry]})
            providers += applied["providers"]
    return {**listing, "providers": providers}
```

  - The four routes take `user_id: int | None = Depends(listing_user_id)` as their last parameter, and each pair `result = await get_configured_providers_async(...)` / `result = _llm_provider_overrides.apply_llm_provider_overrides_to_listing(result)` becomes one `result = await _users_listing(user_id, <the same keyword arguments>)`. In `get_llm_providers` the call moves to the first line of the `try` and the later `apply_...` line (`:1814`) is deleted; its docstring gains "A signed-in user's listing puts each provider the user's own key serves on that key (TASK-15583)."

- [ ] **Step 4: The function-size entry.** Measure, then set the entry to exactly what it prints:

```bash
PATH="$(dirname $PY):$PATH" $PY -c "from tldw_Server_API.tests.lint.test_function_size_ratchet import measure; print(measure()[0]['tldw_Server_API.app.api.v1.endpoints.llm_providers:get_configured_providers'])"
```

Expected: `{'lines': 664, 'params': 3}`. Line 25 of `tldw_Server_API/tests/lint/function_size_baseline.json` becomes (with the printed numbers):

```json
  "tldw_Server_API.app.api.v1.endpoints.llm_providers:get_configured_providers": {"lines": 664, "params": 3, "reason": "passes the requesting user's own-key handles through to the commercial listing (TASK-15583)"},
```

- [ ] **Step 5: GREEN, the closest listing tests and the ratchets, then commit.**

```bash
RUN t1green -n 2 tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py \
  tldw_Server_API/tests/LLM_Calls/test_provider_authoritative_catalog.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_llm_providers_error_mapping.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_llm_openrouter_models_refresh.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_llm_providers_capabilities_merge.py \
  tldw_Server_API/tests/lint/test_function_size_ratchet.py tldw_Server_API/tests/lint/test_endpoint_monkeypatch_ratchet.py \
  tldw_Server_API/tests/lint/test_service_singleton_ratchet.py tldw_Server_API/tests/Usage/test_llm_quota_decision.py
git add tldw_Server_API/app/api/v1/endpoints/llm_providers.py tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py \
  tldw_Server_API/tests/lint/function_size_baseline.json tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py
git commit -m "feat(llm): list a provider on the requesting user's own key (TASK-15583)

The provider listings read a signed-in user's credentials without ever refusing a request. Each
commercial provider the user's own stored key serves at its vendor is listed on that key, with
the models that key lists (discovered with it, cached by its digest), held only to what the admin
override refuses on every key. A user's listing is private to caches. Anonymous and keyless
listings are unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: 0 failed. `netprobe lines: 8`, all from `test_llm_providers_capabilities_merge.py` (tiktoken resolving `openaipublic.blob.core.windows.net`, pre-existing), 0 from `OWN`. The review measured 161 passed for the first nine files before this fold; expect 163 from them, plus `test_llm_quota_decision.py`'s own count. The ratchets pass (the getter count stays 9; no new endpoint patch).

- [ ] **Step 6: RED by mutation, then restore.** Each mutation is one edit to `LP`, run, then restored:

```bash
# Mutation A: in _users_listing, change `if isinstance(entry, dict) and entry.get("name") in own_keys:` to `if False:`
RUN t1mutA tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py
git restore tldw_Server_API/app/api/v1/endpoints/llm_providers.py
# Mutation B: delete the `if own_key is not None and not (...): return _listing_on_key(...)` statement in _resolve_commercial_listing
RUN t1mutB tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py
git restore tldw_Server_API/app/api/v1/endpoints/llm_providers.py
# Mutation C: in _own_key_entry, change `if isinstance(info, dict) and info.get("name") in models` to `if isinstance(info, dict)`
RUN t1mutC tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py
git restore tldw_Server_API/app/api/v1/endpoints/llm_providers.py
# Mutation D: in _users_listing, drop `, policy=False` from the users_own_key_credentials call
RUN t1mutD tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py
git restore tldw_Server_API/app/api/v1/endpoints/llm_providers.py
git diff --stat
```

Expected:
- A: 1 failed (`...[allowlist-and-denylist]`: user 7 gets `["gpt-allowed"]`, the server allowlist), 11 passed.
- B: 4 failed (`test_each_user_lists...`: users 7 and 9 get `[]`; `...[allowlist-and-denylist]`: user 7 gets the server key's list less the denied model, `["gpt-allowed", "gpt-off-list"]`; `test_a_denied_config_default...`: `[]` and `provider_not_configured`; `test_a_rejected_own_key...`: `provider_not_configured`), 8 passed.
- C: 2 failed (`...[allowlist-and-denylist]`: user 7's `/llm/models/metadata` also lists `gpt-denied`; `test_a_denied_config_default...`: its metadata also lists `gpt-default`), 10 passed. The review predicted 1 for this mutation; the second is the I1 test, which also reads every route.
- D: 1 failed (`test_a_denied_config_default...`: the runtime refuses user 7's handle with `model_denied`, so OpenAI is `provider_not_configured` with no models), 11 passed.

`git diff --stat` prints nothing.

---

### Task 2: The WebUI refetches the model pickers after a key change

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Settings/ProviderKeysSettings.tsx` (import `:6`; `handleSave` `:102`; the delete `onOk` `:125`)
- Test: `apps/packages/ui/src/components/Option/Settings/__tests__/ProviderKeysSettings.model-cache.test.tsx` (new)

**Interfaces:**
- Consumes: `tldwModels.clearCache(): Promise<void>` (`apps/packages/ui/src/services/tldw/TldwModels.ts:772`); its invalidation listener in `services/tldw-server.ts:260` clears the chat-model cache too.

- [ ] **Step 1: Link the existing install.** The main checkout has `node_modules` under `<main apps>`; link only what this worktree lacks:

```bash
for d in node_modules packages/ui/node_modules tldw-frontend/node_modules; do
  [[ -e apps/$d ]] || ln -s "<main apps>/$d" "apps/$d"
done
ls -ld apps/node_modules apps/packages/ui/node_modules apps/tldw-frontend/node_modules
```

Expected: three entries, each a symlink into `<main apps>` (or a real directory when working in the main checkout).

- [ ] **Step 2: Write the failing tests.** Create `ProviderKeysSettings.model-cache.test.tsx`:

```tsx
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { Modal } from "antd"
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"

import { ProviderKeysSettings } from "../ProviderKeysSettings"

const mocks = vi.hoisted(() => ({
  listUserProviderKeys: vi.fn(),
  upsertUserProviderKey: vi.fn(),
  deleteUserProviderKey: vi.fn(),
  clearCache: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listUserProviderKeys: mocks.listUserProviderKeys,
    upsertUserProviderKey: mocks.upsertUserProviderKey,
    deleteUserProviderKey: mocks.deleteUserProviderKey
  }
}))

vi.mock("@/services/tldw/TldwModels", () => ({
  tldwModels: { clearCache: mocks.clearCache }
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    // A native select, so the test picks a provider without antd's virtual dropdown.
    Select: ({
      value,
      onChange,
      options
    }: {
      value?: string
      onChange: (value: string) => void
      options: Array<{ value: string }>
    }) => (
      <select aria-label="provider" value={value ?? ""} onChange={(event) => onChange(event.target.value)}>
        <option value="" />
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.value}
          </option>
        ))}
      </select>
    ),
    Modal: { ...actual.Modal, confirm: vi.fn() }
  }
})

describe("ProviderKeysSettings refreshes the model pickers (TASK-15583)", () => {
  beforeAll(() => {
    if (typeof window.matchMedia !== "function") {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }
    if (!(globalThis as any).ResizeObserver) {
      ;(globalThis as any).ResizeObserver = class ResizeObserver {
        observe() {}
        unobserve() {}
        disconnect() {}
      }
    }
  })

  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listUserProviderKeys.mockResolvedValue({
      items: [
        {
          provider: "anthropic",
          has_key: true,
          source: "user",
          key_hint: "1234",
          auth_source: "api_key",
          last_used_at: null
        }
      ]
    })
    mocks.upsertUserProviderKey.mockResolvedValue({
      provider: "openai",
      status: "stored",
      key_hint: "abcd",
      updated_at: "2026-11-21T00:00:00Z"
    })
    mocks.deleteUserProviderKey.mockResolvedValue(undefined)
  })

  it("clears the model catalog after a key is saved, so the pickers offer its provider at once", async () => {
    render(<ProviderKeysSettings />)
    fireEvent.click(await screen.findByRole("button", { name: /Add Provider Key/ }))
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "openai" } })
    fireEvent.change(screen.getByPlaceholderText("sk-..."), { target: { value: "sk-users-own" } })
    fireEvent.click(screen.getByRole("button", { name: /Save Key/ }))

    await waitFor(() => expect(mocks.clearCache).toHaveBeenCalledTimes(1))
    expect(mocks.upsertUserProviderKey).toHaveBeenCalledWith("openai", "sk-users-own")
  })

  it("clears the model catalog after a key is deleted", async () => {
    const { container } = render(<ProviderKeysSettings />)
    await screen.findByText("anthropic")
    fireEvent.click(container.querySelector("button.ant-btn-dangerous") as HTMLButtonElement)
    const { onOk } = vi.mocked(Modal.confirm).mock.calls[0][0] as { onOk: () => Promise<void> }
    await act(async () => {
      await onOk()
    })

    expect([mocks.deleteUserProviderKey.mock.calls, mocks.clearCache.mock.calls.length]).toEqual([[["anthropic"]], 1])
  })
})
```

- [ ] **Step 3: RED.**

```bash
(cd apps/tldw-frontend && bunx vitest run \
  ../packages/ui/src/components/Option/Settings/__tests__/ProviderKeysSettings.model-cache.test.tsx --maxWorkers=1)
```

Expected (measured by the review): 2 failed: the save test times out in `waitFor` (`expected "vi.fn()" to be called 1 times, but got 0 times`), the delete test's `toEqual` sees `[ [ [ 'anthropic' ] ], +0 ]`. If vitest cannot start from the linked install (a missing package or version), do not install anything: record "frontend measured by CI (`frontend-required`)", skip to Step 4, and in Step 5 run nothing but the symlink removal.

- [ ] **Step 4: Implement** in `ProviderKeysSettings.tsx`: `import { tldwModels } from "@/services/tldw/TldwModels"` after the `tldwClient` import; in `handleSave`, right after `await tldwClient.upsertUserProviderKey(addProvider, addApiKey)`:

```tsx
      // The model pickers list a provider the user's own key serves (TASK-15583): refetch them now.
      void tldwModels.clearCache()
```

  and in the delete `onOk`, right after `await tldwClient.deleteUserProviderKey(provider)`: `void tldwModels.clearCache()`.

- [ ] **Step 5: GREEN, the neighbours, then remove the links and commit.**

```bash
(cd apps/tldw-frontend && bunx vitest run \
  ../packages/ui/src/components/Option/Settings/__tests__/ProviderKeysSettings.model-cache.test.tsx \
  ../packages/ui/src/components/Option/Settings/__tests__/ProviderKeysSettings.i18n.test.tsx \
  ../packages/ui/src/components/Option/Settings/__tests__/ProviderKeysSettings.design-system-alert.test.tsx \
  ../packages/ui/src/services/tldw/__tests__/TldwModels.test.ts --maxWorkers=1)
for d in node_modules packages/ui/node_modules tldw-frontend/node_modules; do [[ -L apps/$d ]] && rm "apps/$d"; done
git status --short apps
git add apps/packages/ui/src/components/Option/Settings/ProviderKeysSettings.tsx \
  apps/packages/ui/src/components/Option/Settings/__tests__/ProviderKeysSettings.model-cache.test.tsx
git commit -m "feat(webui): refetch the model pickers after a provider key is saved or deleted (TASK-15583)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected (measured by the review): 4 files, 117 tests passed, 0 failed; the existing ProviderKeysSettings files need no `TldwModels` mock. The `rm` removes only symlinks (no trailing slash). `git status --short apps` lists only the two files above before the commit.

---

### Task 3: Docs, the mirror and the CHANGELOG

**Files:**
- Modify: `Docs/API-related/Providers_API_Documentation.md` (`:9`; new subsection before `### 3) GET /api/v1/llm/providers/{provider_name}` `:144`)
- Modify: `Docs/User_Guides/Server/BYOK_User_Guide.md` (the Usage Quotas paragraph `:97`; new section before `## Org/Team Shared Keys (Manager Role)`)
- Modify: `Docs/Architecture/llm-providers.md` (the Discovery routes row `:23` at `ced8df7c15`), `CHANGELOG.md` (`## [Unreleased]` / `### Fixed`)
- Modify (generated): `Docs/Published/API-related/Providers_API_Documentation.md`, `Docs/Published/User_Guides/Server/BYOK_User_Guide.md`

- [ ] **Step 1: The Providers API doc.** Replace the bullet `- These endpoints follow the server’s standard AuthNZ settings and typically require authentication.` (its apostrophe is U+2019) with:

> - `GET /llm/health`, `GET /llm/providers`, `GET /llm/providers/{provider_name}`, `GET /llm/models` and `GET /llm/models/metadata` are public. A request without credentials, or with credentials the server refuses, gets the server's listing; a signed-in user's request also lists the providers that user's own stored key serves (see "Your own provider key" below). Every listing answer carries `Vary: Authorization, X-API-KEY, Cookie`, and a signed-in user's carries `Cache-Control: private, no-store`.

  and, in the bullet `- Standard limits apply; ...` (`:13`), replace `Deployments can choose to relax access via gateway configuration, but the default app configuration applies global security to API routes.` with `Deployments that want these routes private can restrict them at a gateway.`;

  and insert before `### 3) GET /api/v1/llm/providers/{provider_name}`:

```markdown
#### Your own provider key
- A signed-in user's request lists each commercial provider the user's own stored key serves on that key, also
  when the server has no key for it. "Serves" is the test chat uses for the user's own key: BYOK is on, the
  provider is on the BYOK allowlist, the user has an active key for it, and the server sends that provider's calls
  to its vendor's host (not a custom OpenAI-compatible slot, a local server, or a base URL the operator points
  elsewhere). Such an entry has `is_configured: true` and readiness from that key, and its `models` and
  `models_info` are the provider's model list fetched with that key. A list the provider returns is cached for 5
  minutes per key; an unreachable list or a server error is retried after 20 seconds, and a rejected key on every
  listing. When the provider rejects the key, the entry is unavailable, with `readiness_reason_code: "auth_failed"`.
- The admin provider override still applies: a provider with `is_enabled: false`, and a model on `denied_models`,
  are offered to no user. A model off `allowed_models` is listed for a user whose own key serves the provider,
  because `allowed_models` is what the server's key offers; chat runs it on that key, while shared-workspace chat,
  the persona voice preflight, embeddings and text-to-speech still check `allowed_models` on every key.
- A listing never contains a key and never shows another user's keys or models. A team or org shared key does not
  change it. A request without credentials, and a user with no such key, get the server's listing.
- The same applies to `/llm/providers/{provider_name}`, `/llm/models` and `/llm/models/metadata`.
```

- [ ] **Step 2: The BYOK guide.** In the Usage Quotas paragraph, replace `your own key still serves it (the WebUI's model picker does not list such a provider yet, TASK-15583).` with `your own key still serves it, and the WebUI's model pickers offer it (see Model Pickers below).` Insert before `## Org/Team Shared Keys (Manager Role)`:

```markdown
## Model Pickers

The WebUI's model pickers read the provider listing (`GET /api/v1/llm/models/metadata` and
`GET /api/v1/llm/providers`). When you are signed in, a provider your own key serves is listed on your key, also
when the server has no key for it: you can pick it, and its models are the ones your key lists (the server asks
the provider with your key, as chat does). That covers a provider that bills your key at its vendor; a key you
store for an endpoint the operator runs, or an org or team shared key, does not change the listing. A provider the
operator turned off, or a model it denied, is offered to no one; a model it only leaves off the models the
server's key offers (`allowed_models`) is offered to you, because chat runs it on your key (the four paths named
under Usage Quotas still refuse it). Your listing never shows another user's keys or models. Saving or deleting a
key on the LLM Provider Keys settings page refreshes the chat model pickers at once; other pages that list
providers catch up within a minute. See `Docs/API-related/Providers_API_Documentation.md`.
```

- [ ] **Step 3: Architecture and CHANGELOG.** In `Docs/Architecture/llm-providers.md`, the Discovery routes row's last cell `` `/llm/health`, providers, models metadata `` becomes `` `/llm/health`, providers, models metadata; public, and a signed-in user's listing puts each provider the user's own key serves on that key (TASK-15583) ``. In `CHANGELOG.md`, as the first bullet under `## [Unreleased]` / `### Fixed` (before TASK-15582's bullet that starts "A server-side LLM call to an admin-disabled provider"):

> - **The WebUI's model pickers offer a provider your own key serves, also when the server has no key for it.** `GET /api/v1/llm/providers`, `/llm/providers/{provider_name}`, `/llm/models` and `/llm/models/metadata` read a signed-in user's credentials: each commercial provider the user's own stored key serves at its vendor is listed on that key, usable, with the models that key lists (fetched with the user's key and cached per key). The admin override still hides a disabled provider and a denied model from everyone; a model off `allowed_models` is listed for a user whose own key serves the provider (TASK-15582). The listings stay public, and a request without credentials, or a user without such a key, gets the server's listing unchanged; a user's listing is marked `Cache-Control: private, no-store`. Saving or deleting a key on the LLM Provider Keys settings page refreshes the chat model pickers (TASK-15583).

- [ ] **Step 4: Mirror, the docs tests, commit.**

```bash
bash Helper_Scripts/refresh_docs_published.sh
git status --short Docs/Published
RUN docs -n 2 tldw_Server_API/tests/Docs
git add Docs/API-related/Providers_API_Documentation.md Docs/User_Guides/Server/BYOK_User_Guide.md \
  Docs/Architecture/llm-providers.md CHANGELOG.md Docs/Published
git commit -m "docs: how a user's own key affects the provider listing (TASK-15583)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: `git status` lists exactly ` M Docs/Published/API-related/Providers_API_Documentation.md` and ` M Docs/Published/User_Guides/Server/BYOK_User_Guide.md`; the docs tests report 0 failed (the review measured 214 passed), netprobe lines 0.

---

### Task 4: Verification, close-out, review, PR

- [ ] **Step 1: The sweep.** The same files as the baseline, plus the new one:

```bash
FAILED_TO=<scratchpad>/sweep1_15583.txt RUN sweep1 -n 2 $(cat <scratchpad>/sweep_files_15583.txt) tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py
comm -13 <scratchpad>/sweep0_15583.txt <scratchpad>/sweep1_15583.txt
comm -23 <scratchpad>/sweep0_15583.txt <scratchpad>/sweep1_15583.txt
```

Expected: both `comm` commands print nothing; `netprobe lines: 17`, as in the baseline. The review measured 498 passed, 0 failed before this fold; expect 500 (its 2 new tests). If the first `comm` prints ids, classify each and fix:
- a test double of `get_configured_providers_async` or `get_configured_providers` that rejects the `own_keys` keyword: accept `**kwargs`;
- a test that calls a listing route function directly and now sees the `Depends` default for `user_id`: `_users_listing` treats it as no user, so look elsewhere (report it);
- anything else is a behaviour change this plan did not intend: stop and report it.

Then delete the lists: `rm -f <scratchpad>/sweep0_15583.txt <scratchpad>/sweep1_15583.txt <scratchpad>/sweep_files_15583.txt`.

- [ ] **Step 2: tests/lint and the gates.**

```bash
RUN lint -n 2 tldw_Server_API/tests/lint --ignore=tldw_Server_API/tests/lint/test_runtime_package_discovery.py
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/test_quality_triage.py --enforce
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_shard_coverage.py
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_backlog_task_ids.py
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/scope_predicate_ratchet.py
git diff --name-only --diff-filter=AM origin/dev...HEAD -- '*.py' | xargs $(dirname $PY)/ruff check
uvx bandit -q -r tldw_Server_API/app/api/v1/endpoints/llm_providers.py tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py
git diff --check origin/dev...HEAD
PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py \
  --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts
```

Expected: `tests/lint` 0 failed (the review measured 331 passed), the route-auth ratchet included (its baseline unchanged; an intermittent native abort in `test_rg_route_map_lint.py` under `-n 2` also hits `origin/dev`: re-run that file with `-p no:xdist` and expect it to pass). The four scripts exit 0. `ruff` reports nothing that `origin/dev`'s copy of the same file does not (compare with `git show origin/dev:<file> | $(dirname $PY)/ruff check --stdin-filename <file> -`). The review saw `I001` and `F401` on the changed files, the same as on their base copies. Bandit prints no finding on a line this branch added (`uvx` may download bandit; if no download is allowed, record "Bandit not run" in the notes). `git diff --check` prints nothing. The OpenAPI drift check exits 0 (the routes gain no field). If it reports drift, regenerate with `--fingerprint apps/packages/api-types/openapi.fingerprint.json` (and `--paths apps/packages/api-types/paths.d.ts` if the paths drifted), re-run the check, and commit the artifacts as `chore(api-types): regenerate the OpenAPI fingerprint (TASK-15583)`.

- [ ] **Step 3: Close TASK-15583.**

```bash
BACKLOG task edit TASK-15583 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "<measured: Task 1 RED, GREEN and both mutations; Task 2 RED and GREEN or 'frontend measured by CI'; the sweep diff; tests/Docs; tests/lint; the gates; Bandit>" \
  --final-summary "The provider listings (GET /llm/providers, /llm/providers/{provider_name}, /llm/models, /llm/models/metadata) read a signed-in user's credentials through a dependency that never refuses, so they stay public; a user's listing is marked private to caches. Each commercial provider the user's own stored key serves at its vendor (users_own_key_credentials) is listed on that key: configured and enabled, with the models that key lists, discovered as chat discovers them and cached by the key's digest, so a listing never shows another user's key or models. The admin override holds such an entry only to what every key refuses (a disabled provider, a denied model), applied by the listing itself, so a denied default model does not hide the provider; a model off allowed_models stays (owner's decision, 2026-11-21). Anonymous and keyless listings are unchanged. The WebUI's filters needed no change; the provider keys page refetches the pickers after a save or delete. AC #2 was amended for the owner's decision. The Providers API doc, the BYOK guide, llm-providers and the CHANGELOG say so." \
  -s Done
git add -A backlog/tasks
git commit -m "chore(backlog): close TASK-15583

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: `TASK-15583 - The WebUI cannot pick a provider the server has no key for, even with the user's own key (Done)`.

- [ ] **Step 4: Final review.** A clean-session opus review of the whole branch (`/superpowers:requesting-code-review`), given Review Focus, the Rulings and "Assumed from TASK-15582". Fold in every Critical and Important finding, re-run the affected files through `RUN` (or vitest as in Task 2), and record the review in the PR body.

- [ ] **Step 5: The PR.** Push, rewrite `<plans dir>/pr_body_webui_byok.md`, then `gh pr edit <number> --body-file <plans dir>/pr_body_webui_byok.md` and `gh pr ready <number>`. The body has: Why (the owner's 2026-11-19 rule and 2026-11-21 decision; the AC #2 amendment); What changes (the dependency, the own-key listing and how the override applies to it, the WebUI pickers table and the one WebUI change, the function-size bump with its reason); Measured; Known limits (from this plan); Reviews; the full `## Change summary` section ("ADR-004: the owner waived a user-written Change summary for this PR on 2026-11-21."); and the closing `🤖 Generated with [Claude Code](https://claude.com/claude-code)` line. Before `gh pr ready`, ask the controller to confirm the 2026-11-21 ADR-004 waiver for this PR is recorded (review round 1, M8). Before asking for the merge, diff any failing non-required Frontend E2E journey on the PR against a recent `dev` run. Merges go through the controller's queue (check `gh variable get MERGE_QUEUE` first).

Expected: the PR is ready for review against `dev`, with the required statuses running.
