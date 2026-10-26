# Architecture Remediation Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. This is a roadmap of 48 backlog tasks; each task gets its own TDD implementation plan (superpowers:writing-plans) before work starts. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the bugs and security gaps the 2026-10-25 architecture review found, stop the measured structural growth, then reduce the backend's 100-package import cycle and its god objects to a layered, independently testable structure.

**Architecture:** Ratchet first, refactor second. Every finding starts with a ratchet that freezes today's count and can only shrink. Ratchets are AST or lint based, use a committed counter script, and are keyed by module and qualified name, never by line number. Code then moves behind re-export shims, in this order: cut cycle-closing edges, add injection seams, decompose hotspots, consolidate duplicates. The seams come before the moves because more than 5,000 test monkeypatches pin endpoint module structure today.

**Tech Stack:** Python 3.12/3.13, FastAPI, pytest, AST lint tests under `tldw_Server_API/tests/lint/`, SQLite and PostgreSQL (RLS), TypeScript/React (Bun, Vitest, ESLint 9) for `apps/packages/ui`, `apps/tldw-frontend` and `admin-ui` (tldw-admin).

**Spec:** Backlog parent `TASK-13950` and subtasks `TASK-13950.1` to `.48` (`backlog/tasks/task-13950*`). Each subtask's description holds the verified evidence (dev `dca11342de`, file:line), and its acceptance criteria are the requirements. This document adds ordering, dependencies, slicing and exit criteria.

**Review status:** An independent review with a dry run (2026-10-25) found six blockers. All are fixed in the tasks and in this plan:
- the RLS ratchet cannot see `add_tenant_policy` calls;
- annotation sync is not idempotent;
- the `mlx` endpoint test patched the wrong seam;
- the session-race test passed on buggy code;
- the layering baseline counts 311 entries once function-level imports are included;
- moderation overlays (per-user overrides, Guardian) need to survive a global off.

## Global Constraints

- **Branching and PR size.** Every PR targets `dev`. One backlog task per PR. A task with more than about 400 changed lines is split into ordered PRs, each passing CI on its own.
- **Ratchet rules.** Every new ratchet:
  - lives in `tldw_Server_API/tests/lint/` (backend) or runs in `frontend-required` (frontend);
  - seeds its baseline from a counter script committed in the same PR (review numbers are matcher-dependent, so the committed script defines the number);
  - keys entries by module or qualified name, never by line number;
  - fails both on growth and on a stale baseline entry;
  - ships with a self-test that seeds one violation and asserts the ratchet fails.
- **Raising a baseline.** It is allowed only through a documented bump-with-reason entry, reviewed in the PR, so that security fixes inside frozen functions are not blocked.
- **AST caching.** Backend AST ratchets share one cached parse; a full-tree scan takes about 9 s.
- **Line-keyed legacy baseline.** `Helper_Scripts/ci/scope_predicate_baseline.txt` is still keyed by line number until `.23` re-keys it. Any PR that shifts lines in a file it lists regenerates it with `--write-baseline`.
- **Shims.** A move keeps a re-export shim at the old import path until `git grep` shows no importer and no string monkeypatch target still uses it.
- **Postgres proof.** Postgres-dependent acceptance criteria are proven in the Postgres shard with `TLDW_TEST_POSTGRES_REQUIRED=1`. A local skip is not evidence.
- **Flipping a global default** (moderation per-surface flags, production detection, middleware stack, pool backend): `git grep` the symbol across all tests and run every hit.
- **ADR edits** (ADR-003, ADR-025) need `Helper_Scripts/refresh_docs_published.sh` and the `Docs/Published` diff in the same PR.
- **Backlog edits** use `tools/backlog-py` only; stage with `git add -A backlog/tasks`. Cite duplicate ids (TASK-12108, TASK-12013) by title.
- **Review gates.** Each AI-authored PR carries a `## Change summary` (ADR-004) and gets a clean-subagent `/superpowers:requesting-code-review` before merge. Each task's own plan gets an independent review plus a dry run before execution.
- **Owner decisions (2026-10-25), binding on these tasks:**
  - **Moderation** covers every LLM surface but stays optional and policy-configured. Global `[Moderation] enabled = False` disables the default policy on every surface; explicit per-user overrides and Guardian supervised overlays still enable it, as they do today.
  - **Note keywords** are case-insensitive: casefold plus NFC via the existing `normalize_keyword_identity`, with display casing preserved.
  - **Scheduler** folds into Jobs (`.48`; no fold work existed when re-checked on 2026-10-25). `.39` covers the recurring APScheduler wrappers and test collection around it.
  - **Admin** consolidates onto tldw-admin (`/admin-ui`).

## Review Focus

1. **Ratchets that pass for the wrong reason.** A counter that misses function-level imports, aliases, from-imports or loop-built SQL reports green while violations grow. The review found two existing ones: the RLS coverage scanner is regex-based, and the HTTP guard is regex-based. Every ratchet task (`.3`, `.8`, `.9`, `.10`, `.11`, `.19`, `.21`, `.30`, `.34`) carries a seeded-violation self-test.
2. **SQLite-green, Postgres-broken.** `.3`, `.5`, `.23` and `.32` change behaviour that only exists on Postgres. Each adds a Postgres-shard test that fails before the change.
3. **Moderation default drift.** `.27` must make zero moderation calls on any surface when the global flag is off and no overlay applies, and must keep Guardian and per-user overlays working. It must also not moderate `/chat/completions` twice once the gateway hooks exist.
4. **Losing the only admin path.** Single-user WebUI and extension users reach admin only through Option/Admin today. `.42` deletes nothing until tldw-admin starts in the default single-user deployment without a rebuild (AC4 before AC5).
5. **Unload data loss and duplicates.** `.2` requires annotation sync to be idempotent on `client_id` before replay-on-load is enabled. Payloads over 60KB are persisted and deferred, never sent with keepalive, because the Fetch spec caps keepalive bodies at 64 KiB and jsdom will not show the failure.

---

## Phase map

| Phase | Tasks | Purpose | Exit criterion |
|---|---|---|---|
| 0 | `.1`-`.11` | Fix confirmed bugs and security gaps; land growth guards | All 11 Done; every hotspot and import direction is frozen by a ratchet |
| 1 | `.12`-`.17` | Cut the cycle-closing import edges | Layering baseline (function-level imports included; 311 at review) has no foundation-upward, `LLM_Calls -> Chat/AuthNZ`, `core -> api.schemas` or `DB_Management -> domain-logic` entries; both SCC measures recorded (100 module-level / 106 with function-level at review) |
| 2 | `.18`-`.21` | Add injection seams so code can move | Endpoint-module monkeypatch baseline at or below 55% of its committed seed; `reset_db_pool` test callers under 50; `TEST_MODE`-unset CI job required |
| 3 | `.22`-`.30` | Decompose hotspots | `ChaChaNotes_DB.py` under 25,000 lines; `unified_rag_pipeline` at most 800 lines; `chat_service.py` under 4,000 lines; moderation on every LLM surface |
| 4 | `.31`-`.48` | Consolidate duplicated mechanisms | Complete provider tables derived from ProviderSpec; one placeholder rewriter; one admin console; one durable queue (Jobs) |

Phases overlap. A task can start as soon as its `dependencies:` are Done; the phase number orders the default queue, not hard gates. The SCC numbers are reported, not gated: removing every upward edge still leaves a 55-package cycle among domain packages once function-level imports are counted. A module-level-only SCC gate could be met by moving imports into functions, which is Review Focus 1.

## Dependency graph (critical paths)

```
.9 layering contract ──┬─> .12 foundation edges
                       ├─> .13 LLM gateway ──┬─> .14 schema defaults
                       │                     ├─> .17 Provider_Credentials
                       │                     ├─> .25 chat_service split (+ .8)
                       │                     ├─> .27 moderation everywhere   <- owner priority
                       │                     ├─> .31 ProviderSpec ──> .46 adapter bridges
                       ├─> .15 schemas -> core models
                       ├─> .16 DB_Management service logic out
                       └─> .20 endpoint shared code out (+ .18) ──> .26 handler use cases (+ .8)
.4 session race ──> .18 service container ──> .19 test-mode seams (+ .6)
.6 runtime_environment ──> .21 config consolidation
.3 RLS/fail-closed ──┬─> .23 unconditional owner scope ──> .22 CharactersRAGDB (+ .8)
                     ├─> .30 raw-SQL ratchet extension
                     └─> .32 dialect consolidation
.8 size ratchets ──> .24 RAG pipeline
.10 packages/ui lint ──┬─> .29 chat stacks ──> .41 selectedModel
                       ├─> .40 React Query state
                       └─> .45 source-text tests
.33 domain errors ──> .37 services split (+ .9)
```

The chain to the owner's moderation priority is `.9 -> .13 -> .27`. Start `.9` in week one; `.13` begins when `.9` merges.

## Conflict hazards (open PRs on 2026-10-25)

| PR | Files | Sequence |
|---|---|---|
| #3298 (UX remediation wave 3, 471 files) | PlaygroundForm.tsx, Sidepanel/Chat/form.tsx, useChatActions.ts, sidepanel-chat.tsx | `.29` and the `.10` size baseline land after it merges |
| #3299 | ChaChaNotes_DB.py and three chacha stores | `.22` and `.23` start after it merges; `.8` baselines re-measure after |
| #3187, #3191, #3180 | byok_runtime.py, provider_readiness aliases, llm_providers/config_info catalogs, chat_service, providers/base.py | `.13` and `.31` rebase onto them; `.31` re-seeds its drift allowlist after they merge |
| #2855 | Admin_Webhooks/delivery.py, admin_webhooks_repository.py | `.16`'s webhooks slice waits for it |
| #2630 | audio WebSocket auth frames | `.7` adopts its frame protocol in the shared resolver |
| #3071 | owner-fenced chat history | `.23` checks its file list before starting |

---

## Phase 0: bugs, security, growth guards

These eleven tasks have no dependencies on each other and can run in parallel. Each brief is the starting point for that task's own plan.

### Task 0.1 (`TASK-13950.1`): accept `mlx` on `/chat/completions`

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py:203-233`
- Modify: `tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py` (`test_llm_provider_readiness_marks_unsupported_catalog_alias_unavailable` encodes the bug; invert it)
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_request_provider_allowlist.py` (new)

- [ ] **Step 1: Write the failing tests.**
  - `test_chat_request_accepts_mlx`: `ChatCompletionRequest(api_provider="mlx", model="m", messages=[{"role": "user", "content": "hi"}]).api_provider == "mlx"`.
  - `test_every_registered_chat_adapter_is_allowlisted`: registry default keys minus `EXEMPT = frozenset()` is a subset of `ALL_SUPPORTED_PROVIDER_NAMES_LIST`. The dry run shows it fails with exactly `{"mlx"}`.
- [ ] **Step 2: Run them and confirm they fail.** `python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_request_provider_allowlist.py -v`.
- [ ] **Step 3: Implement.** Add `"mlx"` to `_BASE_SUPPORTED_PROVIDER_NAMES_LIST`, and invert the readiness test.
- [ ] **Step 4: Add the endpoint test.** POST `/api/v1/chat/completions` with `api_provider: "mlx"`. Patch the sync seam the non-stream path calls (`perform_chat_api_call` as imported in `endpoints/chat.py`, used via `partial` at `:5406`). Assert HTTP 200 and that the patched call received `api_endpoint == "mlx"`.
- [ ] **Step 5: Run the new file, `test_llm_providers_readiness.py` and the `tests/LLM_Calls/test_mlx_*` files, and confirm PASS. Commit.**

### Task 0.2 (`TASK-13950.2`): authenticated, idempotent annotation unload sync

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/document_annotations.py` (sync endpoint becomes idempotent on `client_id`)
- Modify: the annotations store, so the client id is stored with a uniqueness constraint per user, media and client id
- Modify: `apps/packages/ui/src/hooks/document-workspace/useAnnotationSync.ts:21-71,321-334`
- Modify: `apps/packages/ui/src/services/tldw/request-core.ts:557` (export `applyTldwAuthHeaders` or a wrapper)
- Test: a backend replay test next to the existing document-annotation tests
- Test: `apps/packages/ui/src/hooks/document-workspace/__tests__/useAnnotationSync.test.tsx` (extend)

**Interfaces:**
- Produces: `syncAnnotationsOnUnload(serverUrl: string | null, mediaId: number, pending: Annotation[]): "sent" | "deferred"`, replacing `syncAnnotationsWithBeacon`.
- Headers come from a ref recomputed whenever the pending set or auth state changes. `applyTldwAuthHeaders` is synchronous, but runtime config is async, and `beforeunload` cannot await.

- [ ] **Step 1: Backend first.** Write a failing test: posting the same batch (same `client_ids`) twice creates each annotation once and returns the same server id mapping. Implement upsert-on-`client_id`, run and confirm PASS, then commit.
- [ ] **Step 2: Write the failing frontend tests.**
  - In API-key, bearer and cookie-session mode, the unload request carries `X-API-KEY`, `Authorization` and `X-CSRF-Token` respectively.
  - Pending annotations are in local storage before the send.
  - A payload over 60KB returns `"deferred"` and makes no request.
  - An environment without `sendBeacon` still sends via fetch keepalive.
- [ ] **Step 3: Implement.** Send with `fetch(url, { method: "POST", keepalive: true, headers })`. Persist pending annotations first, and replay them on the next load (safe because of Step 1). Run and confirm PASS, then commit.

### Task 0.3 (`TASK-13950.3`): ChaCha RLS gaps and fail-closed scoping

**Files:**
- Modify: `Helper_Scripts/ci/rls_coverage_ratchet.py` (AST resolution of `add_tenant_policy("t", ...)` and loop tuples), `Helper_Scripts/ci/rls_coverage_baseline.txt`
- Modify: `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py`, `backends/postgresql_backend.py` (scope-failure handlers; delete the fallback pool), `media_db/runtime/connection_lifecycle.py:82-84`, `pyproject.toml` (`psycopg-pool` to core dependencies)
- Test: `tldw_Server_API/tests/CI/test_rls_coverage_ratchet.py` (extend with a self-test)
- Test: `tldw_Server_API/tests/DB_Management/test_chacha_study_tables_rls.py` (new, Postgres), `tldw_Server_API/tests/DB_Management/unit/test_postgres_scope_failure_discards_connection.py` (new)

- [ ] **Step 1: Fix the ratchet before anything else.** Add a self-test: a scratch module calling `add_tenant_policy("t_x", ...)` inside a loop must count `t_x` as covered. Make it pass with AST resolution, then regenerate the baseline with `--write-baseline`. Expect `note_folders`, `writing_*`, `note_attachments` and `note_tasks` to drop out. Commit.
- [ ] **Step 2: Write the failing Postgres test.** On a restricted role, user B reads 0 of user A's rows from `decks`, `flashcards`, `study_packs` and the other study tables listed in the task.
- [ ] **Step 3: Write the failing unit tests.** An injected `set_config` failure makes `get_connection` raise and discard the connection. The no-scope background path still returns a usable connection.
- [ ] **Step 4: Implement.**
  - Add tenant policies on `client_id` for the study tables, and regenerate the baseline.
  - Replace debug-and-continue with discard-and-raise.
  - Make `psycopg-pool` core and delete the fallback pool.
  - Log the active pool class at startup.
- [ ] **Step 5: Run the three test files in the Postgres shard and confirm PASS. Commit.**

### Task 0.4 (`TASK-13950.4`): session-manager initialization race

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/session_manager.py:2076-2093` (`reset_session_manager` recreates the lock), `tldw_Server_API/app/core/MCP_unified/server.py:2317-2331`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_init_race.py` (new)

- [ ] **Step 1: Write the failing tests.**
  - `test_callers_never_receive_uninitialized_manager`: patch `initialize` to `await asyncio.sleep(0.05)` before setting state. Run 50 concurrent `get_session_manager()` calls, each recording `m._initialized` the moment it returns. Assert all True and that `initialize` ran exactly once. This fails 49/50 before the fix.
  - `test_failed_initialize_publishes_nothing`.
  - Each test monkeypatches a fresh `asyncio.Lock`.
- [ ] **Step 2: Implement.** Use a lock and publish only after `initialize()`; `reset_session_manager()` creates a new lock. Run and confirm PASS.
- [ ] **Step 3: Write the MCP thread test.** Run `get_mcp_server` from 8 threads with a slow `MCPServer` constructor and assert one instance. Implement a `threading.Lock`, run and confirm PASS, then commit.

### Task 0.5 (`TASK-13950.5`): case-insensitive keyword identity

**Files:**
- Reuse: `normalize_keyword_identity` (`tldw_Server_API/app/core/DB_Management/organization_sync_store.py:163`). Move it to a neutral module if needed, with a re-export.
- Modify: the call sites listed in the task, plus the Postgres keyword lookup behind `chacha/keyword_store.py:138` (`_get_generic_item_by_unique_text`)
- Test: `tldw_Server_API/tests/Notes/test_keyword_identity.py` (new) and a Postgres case in `tldw_Server_API/tests/ChaChaNotesDB/`

- [ ] **Step 1: Write the failing tests.**
  - Linking `["Foo", "foo"]` to a note that already has `foo` yields one link on SQLite and on Postgres.
  - `get_keyword_by_text("FOO")` finds `foo` on Postgres.
  - The MCP path keeps display casing.
  - Non-ASCII cases (`"Straße"`/`"STRASSE"`, accented capitals) assert the chosen behaviour and document any SQLite `NOCASE` limit.
- [ ] **Step 2: Implement.** Route callers through the helper and make the Postgres lookup case-insensitive. Add a lint guard against new ad hoc keyword normalization, run and confirm PASS, then commit.

### Task 0.6 (`TASK-13950.6`): test-seam guard and one production detector

**Files:**
- Modify: `tldw_Server_API/app/core/testing.py:75-120` (add `runtime_environment() -> RuntimeEnvironment`, with `is_production: bool` and `source_key: str | None`, and no `lru_cache`)
- Modify: every detector copy listed in the task, and `core/AuthNZ/settings.py:73-74,931,943-953,1007`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_runtime_environment.py` (new), `tldw_Server_API/tests/Services/test_startup_test_runtime_guard.py` (extend), `tldw_Server_API/tests/lint/test_production_detector_single_source.py` (new, with a label-only allowlist)

- [ ] **Step 1: Write the failing tests.**
  - A matrix over the keys `tldw_production`, `TLDW_PRODUCTION`, `ENVIRONMENT`, `APP_ENV`, `DEPLOYMENT_ENV`, `FASTAPI_ENV`, `TLDW_ENV` and `ENV`, and the values `prod`, `production` and `live`, shows every caller agrees.
  - `tldw_production=1` plus `PYTEST_CURRENT_TEST=x`, in multi-user mode with no `JWT_SECRET_KEY`, raises `ValueError`.
  - The guard raises when the runtime is production-like and `TLDW_ALLOW_TEST_FALLBACK_KEYS=1`.
- [ ] **Step 2: Implement and migrate the callers.** Add the ratchet, run and confirm PASS, add the release note, then commit.

### Task 0.7 (`TASK-13950.7`): WebSocket auth

**Files:**
- Modify: `Helper_Scripts/ci/route_auth_ratchet.py:245-258`
- Create: `tldw_Server_API/app/core/AuthNZ/websocket_principal.py` with `async def resolve_ws_principal(websocket: WebSocket, *, required_scope: str | None = None, allow_first_frame: bool = False) -> AuthPrincipal` (`AuthPrincipal` is `core/AuthNZ/principal_model.py:34`)
- Test: `tldw_Server_API/tests/lint/test_route_auth_ratchet.py` (extend), `tldw_Server_API/tests/AuthNZ/test_websocket_route_auth_matrix.py` (new)

- [ ] **Step 1: Ratchet first.** `iter_routes` also yields `APIWebSocketRoute`, with today's routes as the baseline. A self-test adds an unauthenticated WebSocket route to a scratch app and asserts the ratchet fails.
- [ ] **Step 2: Write the failing matrix test**, parametrized over every mounted WebSocket path.
  - Header and query routes close before accept when there are no credentials or the JWT is revoked.
  - First-message-frame routes close with a policy code before processing any data frame.
  - The workflows WebSocket accepts a valid AuthNZ access token.
- [ ] **Step 3: Implement `resolve_ws_principal`** from persona's resolver (`persona.py:3676`). Migrate one route per PR, workflows first because it has the token-type bug, and delete each local `_looks_like_jwt` as its route migrates. The persona route is frozen by `.8`, so use its bump-with-reason path.

### Task 0.8 (`TASK-13950.8`): size ratchets

**Files:**
- Create: `tldw_Server_API/tests/lint/_ast_cache.py` (a shared cached parse), `tldw_Server_API/tests/lint/test_function_size_ratchet.py`, `tldw_Server_API/tests/lint/function_size_baseline.json` (keyed `"<module>:<qualname>": {"lines": int, "params": int, "reason": str | null}`, with nested functions under their parent's qualname), `tldw_Server_API/tests/lint/test_charactersragdb_surface_ratchet.py`

- [ ] **Step 1: Write the self-tests.** A scratch 301-line function fails the function ratchet, a seeded `_delegate_store_method` registration fails the surface ratchet, and a raised entry without a `reason` fails.
- [ ] **Step 2: Generate the baselines.** Write them with `--write-baseline`: expect about 209 functions and 34 handlers; re-measure on the PR base. Confirm the ratchets pass on dev and fail when one function grows by one line.
- [ ] **Step 3: Add the stale-entry check. Commit.**

### Task 0.9 (`TASK-13950.9`): layering contract

**Files:**
- Create: `tldw_Server_API/tests/lint/test_layering_contract.py`, `tldw_Server_API/tests/lint/layering_baseline.txt` (lines `"<importer module> -> <target package>"`)
- Modify: `tldw_Server_API/tests/lint/test_core_to_api_import_boundary.py` (pair-based; correct the stale `persistence.py` docstring)

**Interfaces:**
- Produces: `LAYERS: tuple[tuple[str, tuple[str, ...]], ...]`, `scan_upward_edges(root: Path) -> set[tuple[str, str]]`, and `package_scc_sizes(root: Path) -> dict[str, int]` (keys `"module_level"` and `"with_function_level"`). Later tasks lower the baseline by deleting lines.

- [ ] **Step 1: Write the self-tests.**
  - A scratch `core/config_x.py` importing `core.Chat` is reported, at module level and inside a function.
  - A `TYPE_CHECKING` import is not reported.
  - A rename (the same edge moving from module A to module B while A's entry is deleted in the same diff) passes.
- [ ] **Step 2: Implement the scanner over the shared AST cache** and generate the baseline: expect 311 entries on `dca11342de` and re-measure on the PR base.
- [ ] **Step 3: Add the pair ratchets** for endpoint->endpoint, services->api and DB_Management->domain. Run and confirm PASS, then commit.

### Task 0.10 (`TASK-13950.10`): packages/ui lint and boundaries

**Files:**
- Modify: `apps/tldw-frontend/scripts/check-shared-hooks.mjs`, or add a sibling script run from `apps/tldw-frontend`, since eslint does not resolve from `packages/ui`
- Modify: `apps/tldw-frontend/eslint.config.mjs` (a `packages/ui` block: ignore `src/public/**`, disable `@next/next` rules)
- Modify: `.github/workflows/frontend-required.yml`
- Create: a suppressions file and `apps/tldw-frontend/scripts/check-ui-import-cycles.mjs`

- [ ] **Step 1: Run the full rule set on `packages/ui/src` from `apps/tldw-frontend` with `--suppress-all`** and commit the suppressions. The dry run took about 2 minutes and produced 1,367 suppressions, mostly in the vendored PDF worker that the ignore removes.
- [ ] **Step 2: Add the rules.**
  - Core `no-restricted-imports`: `@/services/background-proxy` in components, hooks and routes; utils, services, store and hooks importing components; components importing entries.
  - Core `no-restricted-syntax`: `/api/v1/` literals in components; `as any` on API path literals.
  - `max-lines` and `max-lines-per-function` for the five chat files.
- [ ] **Step 3: Regenerate the suppressions.** Add the cycle-count script (baseline 4), confirm the gate fails on a seeded violation, then commit.

### Task 0.11 (`TASK-13950.11`): enforce the HTTP usage guard

**Files:**
- Modify: `tldw_Server_API/tests/lint/test_http_usage_guard.py` (regex to AST; remove the env gate; add `ALLOWLIST: dict[str, str]` mapping path to reason)

- [ ] **Step 1: Write a self-test** in which a seeded `from httpx import AsyncClient as C` must fail. Convert to AST and remove the gate.
- [ ] **Step 2: Seed the allowlist.** Run and record offenders (12 by regex plus up to 6 aliased ones), add them to the allowlist with reasons, and confirm PASS. Then migrate offenders one PR at a time.

---

## Phase 1: cut the cycle-closing edges (`.12`-`.17`)

| Task | Effort | PR slices | Notes |
|---|---|---|---|
| `.12` foundation edges | S | 1-2 | Removes every foundation-upward entry from the `.9` baseline |
| `.13` LLM gateway | L | 4: (a) import-isolation test plus ratchet; (b) Chat_Deps codemod (36 statements in 22 files) and the other 10 `LLM_Calls -> Chat` statements; (c) gateway module plus re-exports, the API_KEYS hook moved to a fixture, and the 42 + 13 patch targets retargeted; (d) policy hooks plus ADR-025 | `LLM_Calls -> Chat` reaches 0 at (b) |
| `.14` schema defaults | S | 1 | After `.13` is Done (frontmatter dependency) |
| `.15` schemas -> core models | M | 1 per context, Evaluations first | Add an OpenAPI snapshot diff test before the first move |
| `.16` DB_Management logic | M | 4: kanban, resume store, webhooks, Sync contracts | Webhooks slice after #2855 |
| `.17` Provider_Credentials | M | 2 | Security path: run `tests/AuthNZ` in both shards |

**Phase exit check:** run the `.9` contract. Its baseline has no foundation-upward, `LLM_Calls -> Chat/AuthNZ`, `core -> api.schemas` or `DB_Management -> domain-logic` entries. Record both SCC measures and the baseline size in TASK-13950's notes.

## Phase 2: injection seams (`.18`-`.21`)

| Task | Effort | PR slices | Notes |
|---|---|---|---|
| `.18` service container | L | 1 per service: db_pool, session_manager, jwt_service, MCP server, ChaCha runtime | Keep `get_x()` as a deprecated fallback that reads the container; check shutdown order (ADR-021) every slice; `reset_db_pool` callers under 50 |
| `.19` test-mode seams | L | 1 per seam, security seams first | Add the `TEST_MODE`-unset CI job in the first slice (AuthNZ, http_client and Security tests) |
| `.20` endpoint shared code | L | pagination, route-less modules, then each core reach-in | Commit the monkeypatch counter first; lower the baseline every PR |
| `.21` config consolidation | M | ratchets first, then one shared key per PR | After `.6` |

## Phase 3: hotspots (`.22`-`.30`)

| Task | Effort | PR slices | Notes |
|---|---|---|---|
| `.23` unconditional owner scope | M | ratchet plus scope-baseline re-key, then raise-or-derive per store | Before `.22`; restricted-role Postgres test in each slice |
| `.22` CharactersRAGDB | XL | caller ratchet; context object; migrations move; then one domain per PR | After #3299 and `.23` |
| `.24` RAG pipeline | L | golden-output test; config object (no behaviour change); one stage group per PR | |
| `.25` chat_service | L | per executor | Record the AC#8 discrepancy on the chat-pipeline TASK-12013 (by title) first |
| `.26` handler use cases | L | router selector; chat; persona; watchlists | After `.20` |
| `.27` moderation everywhere | L | guard; global-off zero-calls test; gateway hook wiring (moving chat's checks); per-surface config; surfaces in batches | Review Focus 3 |
| `.28` embeddings adapters | M | provider-not-supported 4xx first (any time); then adapters; core-import criterion after `.20` | |
| `.29` chat stacks | XL | deleteMessage; characterChatMode (closes TASK-12108 AC1); submitResponse; store | After #3298 |
| `.30` raw-SQL extension | M | ratchet, then RAG notes/messages through ChaCha | |

**Phase exit check:** the `.8` baselines show `ChaChaNotes_DB.py` under 25,000 lines, `unified_rag_pipeline` at most 800 lines and `chat_service.py` under 4,000 lines. The `.27` guard baseline holds only reasoned exemptions.

## Phase 4: consolidation (`.31`-`.48`)

| Task | Effort | Notes |
|---|---|---|
| `.31` ProviderSpec | L | Drift test first (absorbs `.1`'s parity test); re-seed after #3180/#3187/#3191 merge; complete tables derive from specs, partial tables must be subsets |
| `.32` dialect | L | After `.3` (same pool code); swap AuthNZ's rewriter only after running the AuthNZ Postgres shard |
| `.33` domain errors | M | Status-code contract tests before removing per-module mappers |
| `.34` exception tuples | M | AssertionError removal first, then per-package narrowing |
| `.35` TTS resolver | M | Corpus test with xfails first |
| `.36` STT dispatch | S | Coordinate with TASK-13868 |
| `.37` services split | M | After `.33` |
| `.38` metrics/retry/request-id | M | `/metrics` test for MCP and Evaluations first |
| `.39` Scheduler gaps | S | Collect the Scheduler tests as the safety net for `.48`; the shared rescan base is used by the planned Dreams scheduler |
| `.40` React Query state | M | Watchlists tab by tab |
| `.41` selectedModel | S | After `.29` |
| `.42` tldw-admin | XL | Freeze guard; absorb the shared-UI-only screens; default single-user start (AC4) verified before any deletion (AC5) |
| `.43` notifications client | S | After TASK-13526 stage B2 |
| `.44` extension mirror | S | Correct the TASK-12103 record |
| `.45` source-text tests | S | Baseline, then conversions inside `.29` |
| `.46` adapter bridges | S | After `.13` and `.31` |
| `.47` dead stubs | S | Any time |
| `.48` fold Scheduler into Jobs | L | Supersede ADR-003 first; migrate one submit site per PR (briefing delivery's dependency via `add_job_dependency`); upgrade drain/migration test on SQLite and Postgres; delete `core/Scheduler` last. Start after `.39` collects the Scheduler tests |

## Measuring progress

Seed each number from the counter script committed by the owning task (review figures in parentheses; matcher differences are expected). Record the numbers in TASK-13950's notes at each phase exit.

| Metric | Review figure (dca11342de) | Owning counter |
|---|---|---|
| Upward layering entries (function-level included) | 311 | `.9` |
| Largest package SCC, module-level / with function-level | 100 / 106 | `.9` |
| Endpoint-module monkeypatches | about 5,266-5,672 | `.20` |
| `ChaChaNotes_DB.py` lines / CharactersRAGDB methods | 47,874 / 859 | `.8` |
| `unified_rag_pipeline` lines / params | 7,063 / 233 | `.8` |
| `chat_service.py` lines | 7,847 | `.8` |
| Test-mode reference lines in app/ | about 491 | `.19` |
| Env reads outside config modules | about 2,694 | `.21` |
| Backend dialect branch checks | about 2,150-2,890 | `.32` |
| Provider-table drift entries | seeded by `.31` | `.31` |
