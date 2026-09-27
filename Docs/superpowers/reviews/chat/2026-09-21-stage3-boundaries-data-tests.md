# Stage 3 — Boundaries, data access, tests, and synthesis (`core/Chat`)

Date: 2026-09-21. Read-only audit.

## Scope

The three boundaries `core/Chat` sits between — the API layer above it, the storage layer below it,
and the authorization layer beside it — plus the module's test estate and the audit synthesis.
The briefing's stage-3 (API/schema boundaries) and stage-4 (data-source boundaries) are merged here:
`core/Chat` has exactly one API boundary (`api/v1/endpoints/chat.py`) and one storage boundary
(`CharactersRAGDB` via `core/DB_Management/chacha/`), and splitting them would have produced two thin
files rather than one complete one.

## Code Paths Reviewed

- `core/Chat/chat_service.py:resolve_provider_api_key (2102-2220)`,
  `:_resolve_test_module_provider_api_key (2221-2254)`,
  `:resolve_static_provider_fallback (2256-2267)`, `:merge_api_keys_for_provider (2718-2748)`
- `core/Chat/chat_service.py:34`, `core/Chat/chat_history.py:29`, `core/Chat/chat_helpers.py:13`
  (module-level `api/v1/API_Deps` imports); `core/Chat/chat_service.py:2092, 2138, 2160, 2233, 2241,
  2792`; `core/Chat/command_router.py:658`; `core/Chat/chat_orchestrator.py:25`;
  `core/Chat/chat_target_resolution.py:11`; `core/Chat/chat_loop_store.py:10`;
  `core/Chat/chat_helpers.py:14`
- `core/Chat/document_generator.py:DocumentGeneratorService (80-1503)`, `:_init_tables (180-261)`,
  `:_repair_user_prompts_schema (262-345)`, `:create_generation_job (1082-1126)`,
  `:get_job_status (1127-1171)`, `:update_job_status (1172-1228)`, `:cancel_job (1229-1255)`
- `core/Chat/command_authorization.py (1-122, whole file)`,
  `core/Chat/command_router.py:453-460` (the one `authorize_command` call in core)
- `core/Chat/chat_orchestrator.py:_build_command_context (228-245)` vs
  `api/v1/endpoints/chat.py:2659-2670` vs `api/v1/endpoints/chat.py:3634-3652`
- `core/Chat/chat_service.py:6173-6215` (the `STREAMS_UNIFIED` branch) and
  `core/Streaming/streams.py:_enqueue (337-350)`
- `tldw_Server_API/tests/Chat/**`, `tldw_Server_API/tests/Chat_NEW/**` (tree structure, not contents)
- `.github/workflows/ci.yml` chat shard definitions; `pyproject.toml:756+` per-file-ignores block,
  chat entries at `:959-969` and `:1485-1503`
- `tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py` (the existing AST import
  ratchet, cited as the precedent for `chat-12`)

## Tests Reviewed

Located by import-grep (`grep -rl "core\.Chat" tldw_Server_API/tests` → 288 files). Full inventory:
`2026-09-21-stage3-test-inventory.txt`. **This module is heavily tested; nothing here should be read
as "untested".**

| Test file / group | Protects | Downgrades the risk? |
| --- | --- | --- |
| `tests/Chat/unit/test_api_key_resolution.py`, `test_chat_service_credential_policy.py`, `test_chat_service_fallback_credential_taxonomy.py`, `test_chat_service_base_url_override.py`, `tests/Chat_NEW/unit/test_credential_fixtures.py`, `test_provider_keys_map.py` | provider credential resolution | **Ambiguously.** They exercise the branch at `chat_service.py:2133-2171` that only runs under `PYTEST_CURRENT_TEST`/TEST_MODE. Coverage of `resolve_provider_api_key` is real, but a subset of it is coverage of a path production never takes (`chat-2`). |
| `tests/Chat/integration/test_document_generation_endpoints.py`, `tests/Chat/unit/test_document_generator.py` (7 importers total) | document generation incl. job lifecycle | Partially. Behaviour is covered; the runtime-DDL/schema-repair path (`chat-5`) is exercised only against a fresh database, which is precisely the case where `CREATE TABLE IF NOT EXISTS` cannot fail. |
| `tests/Chat_NEW/unit/test_command_router.py`, `test_chat_command_injection.py`, `tests/Chat_NEW/integration/test_chat_command_{audit,concurrency,perf,replace_mode}.py`, `test_chat_skill_commands_injection.py` (7 files) | slash-command routing and injection | Partially — they cover the router. **Zero test file imports `core.Chat.command_authorization`** (`chat-10`), and none constructs the orchestrator's `CommandContext`. |
| `tests/Chat/unit/test_chat_truthiness_flags.py`, `tests/Chat_NEW/unit/test_rate_limiter.py`, `tests/Chat/unit/test_request_queue.py`, `tests/Chat_NEW/unit/test_request_queue_workers.py` | config flags, limiter, queue | Yes, and they are good. Noted here because they are the counterexample: these subsystems are well covered. |
| 19 files setting `STREAMS_UNIFIED` (incl. `tests/Streaming/test_chat_completions_sse_unified_flag.py`, `tests/Chat/unit/test_chat_service_fallback.py`) | the off-by-default unified stream path | **Yes.** The off-by-default branch IS tested on both settings; the briefing's "isolation mechanisms default to off and are untested" pattern does **not** apply here. The `_enqueue` reach in `chat-12` is an encapsulation complaint, not a coverage one. |

### Modules with ZERO importing test files

Union of both import forms (`core.Chat.<m>` and `from ...core.Chat import <m>`):

| module | LOC | note |
| --- | --- | --- |
| `core/Chat/command_authorization.py` | 122 | **security-relevant** — see `chat-10` |
| `core/Chat/chat_characters.py` | 209 | thin delegator to `Character_Chat` |
| `core/Chat/knowledge_save.py` | 71 | exercised indirectly via the HTTP endpoint by two `test_chat_knowledge_save.py` files |
| `core/Chat/message_utils.py` | 18 | trivial |

## Validation Commands

```
$ grep -rn 'from tldw_Server_API.app.api' tldw_Server_API/app/core/Chat/ | wc -l
14
$ grep -rn 'from tldw_Server_API.app.api.v1.\(endpoints\|API_Deps\)' tldw_Server_API/app/core/Chat/
chat_service.py:34:    ...API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME      [module-level]
chat_service.py:2092:  ...endpoints.llm_providers import (                                 [deferred]
chat_service.py:2160:  ...endpoints import chat as _chat_mod                               [deferred]
chat_service.py:2241:  ...endpoints import chat as _chat_mod                               [deferred]
chat_helpers.py:13:    ...API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME      [module-level]
chat_history.py:29:    ...API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME      [module-level]
command_router.py:658:  ...API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id  [deferred]
(7 true inversions; the other 7 of the 14 are api/v1/schemas/* and are the mild class)
```

```
$ grep -n 'CREATE TABLE\|CREATE INDEX' tldw_Server_API/app/core/Chat/document_generator.py
186:  CREATE TABLE IF NOT EXISTS generation_jobs (
206:  CREATE TABLE IF NOT EXISTS user_prompts (
222:  CREATE TABLE IF NOT EXISTS user_prompt_configs (
235:  CREATE TABLE IF NOT EXISTS generated_documents (
251:  CREATE INDEX IF NOT EXISTS idx_gen_jobs_status ON generation_jobs(status)
252:  CREATE INDEX IF NOT EXISTS idx_gen_jobs_job_id ON generation_jobs(job_id)
253:  CREATE INDEX IF NOT EXISTS idx_gen_docs_conv_id ON generated_documents(conversation_id)
254:  CREATE INDEX IF NOT EXISTS idx_gen_docs_type ON generated_documents(document_type)
279:  CREATE TABLE user_prompts_new (
(9 DDL statements)

$ grep -c 'Jobs\|JobManager\|job_manager' tldw_Server_API/app/core/Chat/document_generator.py
0
```

```
$ grep -rn 'CommandContext(' --include='*.py' tldw_Server_API/app/
core/Chat/chat_orchestrator.py:237
api/v1/endpoints/chat.py:2659
api/v1/endpoints/chat.py:3634
(3 construction sites)

$ { grep -rl "core\.Chat\.command_authorization" tldw_Server_API/tests;
    grep -rlE "core\.Chat import[^#]*\bcommand_authorization\b" tldw_Server_API/tests; } | sort -u | wc -l
0
```

```
$ grep -rl 'core\.Chat' tldw_Server_API/tests | wc -l
288

$ find tldw_Server_API/tests/Chat     -name 'test_*.py' | wc -l   ->  100
$ find tldw_Server_API/tests/Chat     -name 'test_*.py' | xargs wc -l | tail -1  ->  61960 total
$ find tldw_Server_API/tests/Chat_NEW -name 'test_*.py' | wc -l   ->   58
$ find tldw_Server_API/tests/Chat_NEW -name 'test_*.py' | xargs wc -l | tail -1  ->  11468 total

$ comm -12 <(find .../Chat -name 'test_*.py' -exec basename {} \; | sort -u) \
           <(find .../Chat_NEW -name 'test_*.py' -exec basename {} \; | sort -u)
test_chat_knowledge_save.py
(exactly ONE shared basename)

$ git log --diff-filter=A --format='%ad %h' --date=short -- .../tests/Chat_NEW | tail -1  ->  2025-09-04 1e705f784d
$ git log --diff-filter=A --format='%ad %h' --date=short -- .../tests/Chat     | tail -1  ->  2025-04-24 941ce557bb
$ git log --since='6 months ago' --oneline -- .../tests/Chat_NEW | wc -l  ->  49
$ git log --since='6 months ago' --oneline -- .../tests/Chat     | wc -l  ->  108

$ python -m pytest tldw_Server_API/tests/Chat --collect-only -q 2>&1 | tail -1
======================== 2323 tests collected in 7.66s =========================
$ python -m pytest tldw_Server_API/tests/Chat_NEW --collect-only -q 2>&1 | tail -1
==================== 394 tests collected, 1 error in 6.41s =====================
   (the 1 error is `ModuleNotFoundError: No module named 'hypothesis'` on
    tests/Chat_NEW/property/test_message_properties.py — a local environment gap, NOT a repo or CI
    exclusion; that file is in the `chat-new-integration-property` CI shard.)
```

```
$ grep -rn 'Chat_NEW' --include='*.yml' --include='*.toml' --include='*.md' . | grep -v Character_Chat_NEW | wc -l
(hits are: pyproject.toml BLE001 per-file-ignores at :1498-1503; CHANGELOG.md:1235;
 Docs/Design/Pagination_Completion_Matrix.md (18 lines, auto-generated);
 Docs/Plans/2026-03-05-failing-test-clusters-remediation-plan.md:62,105; 10 backlog task files.
 NO README, NO docs file, NO CI comment states why two trees exist.)

$ grep -n 'chat-legacy\|chat-new' .github/workflows/ci.yml | head -6
   (6 shard names, repeated across 5 matrix blocks:)
   chat-legacy-integration, chat-legacy-unit-a-l, chat-legacy-unit-m-z,
   chat-new-integration-property, chat-new-unit-a-l, chat-new-unit-m-z
   → BOTH trees run in CI. No exclusions anywhere (no pytest.ini, setup.cfg, tox.ini, or
     norecursedirs in this repo).
```

## Findings

### FINDING chat-2
```
axis:        encapsulation
class:       n/a
severity:    High
sites:       core/Chat/chat_service.py:resolve_provider_api_key (2102-2220), test branch at
               (2128-2171); the reach-in at :2160
                 `from tldw_Server_API.app.api.v1.endpoints import chat as _chat_mod`
                 `endpoint_keys = getattr(_chat_mod, "API_KEYS", None)`
             core/Chat/chat_service.py:_resolve_test_module_provider_api_key (2221-2254), the same
               reach-in at :2241
             core/Chat/chat_service.py:2138 and :2233 — the sibling reach into
               `api/v1/schemas/chat_request_schemas.API_KEYS`
             consumed by: core/Chat/chat_service.py:resolve_static_provider_fallback (2256-2267)
canonical:   NONE for the seam. `api/v1/schemas/chat_request_schemas.get_api_keys()` is the intended
             env/config accessor and IS used (chat_service.py:2140) — the problem is the second,
             module-global path layered over it.
destination: n/a — this is a seam to remove, not a helper to share. Replace the two `getattr(module,
             "API_KEYS")` reaches with an explicit injected resolver parameter on
             `resolve_provider_api_key` (default: the env/config resolver), so tests inject rather
             than monkeypatch a module global two layers up.
knowledge:   n/a (encapsulation axis)
scenario:    n/a
impact:      High, on three counts.
             (a) **Layering.** `core/Chat` imports `api/v1/endpoints/chat`, which imports
                 `chat_service` back (api/v1/endpoints/chat.py:198 imports `estimate_tokens_from_json`,
                 :212 imports `write_mandatory_moderation_audit`, :312-313 imports the
                 command-authorization helpers). The cycle exists and is broken only by deferring the
                 import inside the function. Operative rule, Docs/Architecture.md: "Clients -> FastAPI
                 endpoints -> Core domain services", one direction.
             (b) **Prod/test path divergence on the credential path.** The branch is gated on
                 `PYTEST_CURRENT_TEST` or `_shared_is_test_mode()` (chat_service.py:2128-2134). The
                 credential resolution ORDER differs between the two, so the six credential test
                 files listed above are, in part, validating an ordering production never executes.
                 This is the module's densest recent defect cluster — `fix(llm): forbid config
                 fallback after credential resolution`, `fix(llm): close explicit credential fallback
                 gaps`, `fix(llm): keep explicit credential policy visible`, `fix(chat): keep
                 credential policy out of provider payload`, all within the last three months — which
                 is exactly what you would expect from a path whose tests do not run what ships.
             (c) **Blast radius if TEST_MODE is ever truthy in a deployment.** A deployed server with
                 TEST_MODE set would prefer API keys found in a mutable module-global dict on an HTTP
                 endpoint module over the configured env/config keys. That is a misconfiguration, not
                 a vulnerability — but the code makes the misconfiguration load-bearing for
                 credentials, which is the wrong thing to make load-bearing.
cost-driver: n/a
tests:       tests/Chat/unit/test_api_key_resolution.py, test_chat_service_credential_policy.py,
             test_chat_service_fallback_credential_taxonomy.py, test_chat_service_base_url_override.py,
             tests/Chat_NEW/unit/test_credential_fixtures.py, test_provider_keys_map.py
             (import-grep reachability, not measured coverage). They are also the reason the seam
             exists, so they must be migrated to the injected resolver in the same change.
effort:      moderate. Mechanically small — one added parameter, two deleted reach-ins, six test files
             re-pointed. Needs `Docs/Design/` because it changes how every credential test is written,
             and an ADR entry because ADR-025 already owns the provider-credential boundary.
owner-only:  YES — the fix touches api/v1/endpoints/chat.py and api/v1/schemas/chat_request_schemas.py.
confidence:  confirmed (the reach-in, the cycle, and the test-only gate — all read directly);
             probable-risk (that this coupling is what is generating the credential defect cluster —
             argued from the commit history, not proven).
```

### FINDING chat-5
```
axis:        duplication
class:       true-duplication
severity:    High
sites:       core/Chat/document_generator.py:DocumentGeneratorService._init_tables (180-261) —
               creates 4 tables and 4 indexes at service construction:
               :186 generation_jobs, :206 user_prompts, :222 user_prompt_configs,
               :235 generated_documents, :251-254 four indexes
             core/Chat/document_generator.py:_repair_user_prompts_schema (262-345) — a hand-rolled
               table-rebuild migration, `CREATE TABLE user_prompts_new` at :279
             core/Chat/document_generator.py — a second job lifecycle:
               :create_generation_job (1082), :get_job_status (1127), :update_job_status (1172),
               :cancel_job (1229), with `GenerationStatus` enum at :71
             core/Chat/document_generator.py:1212-1214 — SQL assembled by string formatting:
               `update_job_sql_template = "UPDATE generation_jobs SET {set_clause} WHERE job_id = ?"`
               `update_job_sql = update_job_sql_template.format_map(locals())  # nosec B608`
             26 raw SQL statements total in this one file; no other core/Chat file contains SQL
               (core/Chat/chat_exceptions.py:292-302 only imports sqlite3 for isinstance checks).
canonical:   core/DB_Management/ owns schema and storage access (Docs/Architecture.md: "keep storage
             access centralized via core/DB_Management/", "no raw SQL in endpoints").
             core/Jobs/manager.py (12,455 LOC) owns job lifecycle. `document_generator.py` imports
             neither — `grep -c 'Jobs\|JobManager\|job_manager' document_generator.py` returns 0.
destination: two moves, not one.
             (1) schema + queries -> a `core/DB_Management/document_generation/` module (follow the
                 already-shipped `core/DB_Management/media_db/` package split as the template), with
                 the DDL expressed as a migration under core/DB_Management/migrations/ rather than
                 runtime `CREATE TABLE IF NOT EXISTS`.
             (2) the job lifecycle -> core/Jobs, or an explicit ADR recording why chat document
                 generation must not use it.
knowledge:   The document-generation schema, and the job state machine. Both are currently owned by a
             1,503-line service module inside core/Chat, on the shared CharactersRAGDB connection.
scenario:    n/a (duplication axis) — but the change-amplification cost is already PROVEN IN THE FILE.
             `CREATE TABLE IF NOT EXISTS` is a no-op against a database where the table already
             exists, so any column added to `user_prompts` silently fails to reach existing
             deployments. That is precisely why `_repair_user_prompts_schema` (262-345) had to be
             written: 84 lines of hand-rolled rebuild-and-copy to undo the consequence of owning
             schema outside the migration system. The next column change needs a second one.
impact:      High. Schema ownership outside the migration system is the failure mode that already
             cost this file an 84-line repair routine; a parallel job lifecycle is 150 lines of
             state machine that will drift from core/Jobs' semantics (leases, retries, cancellation)
             with nothing to hold them together. The `format_map(locals())` at :1213 is not currently
             injectable — `set_clause` is joined from a fixed set of literal column fragments built
             at :1181-1207 — but `format_map(locals())` exposes the entire local namespace to the
             template, so any future `{...}` added to that string silently becomes a substitution.
             A literal f-string over the allowlisted clause would be equally short and not do that.
cost-driver: n/a
tests:       tests/Chat/unit/test_document_generator.py,
             tests/Chat/integration/test_document_generation_endpoints.py,
             tests/Chat/unit/test_chat_document_endpoint_error_mapping.py (7 files import
             document_generator; import-grep reachability, not measured coverage). They run against
             fresh databases, which is the one condition under which `CREATE TABLE IF NOT EXISTS`
             cannot misbehave — so coverage does not downgrade the schema-ownership risk.
effort:      expensive. Needs Docs/Design/, an ADR (job-system exemption or adoption), a Backlog task,
             and a staged IMPLEMENTATION_PLAN with a data migration. Do not attempt as a drive-by.
             The `format_map(locals())` -> f-string change at :1213 is separable and cheap.
owner-only:  no (core/Chat + core/DB_Management), but the endpoint surface must stay stable.
confidence:  confirmed
```

### FINDING chat-10
```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       THREE constructions of command_router.CommandContext:
               api/v1/endpoints/chat.py:2659-2670  (GET /chat/commands listing)
                 request_meta = {permissions, roles, is_admin, auth_mode,
                                 is_single_user_owner: bool(current_user.is_single_user_owner)}
               api/v1/endpoints/chat.py:3634-3652  (POST /chat/completions slash dispatch)
                 same claim set + endpoint/conversation/tool context
               core/Chat/chat_orchestrator.py:_build_command_context (228-245)
                 request_meta = {auth_mode, is_single_user_owner: auth_mode == "single_user"}
                 auth_user_id = int(llm_user_identifier)     # :230-234
                 called at core/Chat/chat_orchestrator.py:935 (_chat_sync_impl) and :1441 (achat)
             consumer: core/Chat/command_router.py:453-460 ->
               core/Chat/command_authorization.py:build_command_authorization_context (46-70),
               :authorize_command (86-122)
canonical:   NONE. `build_command_authorization_context` normalizes the dict, but the dict's CONTENTS
             are assembled independently three times.
destination: give `command_authorization.py` the constructor: one
             `command_context_from_user(user, *, auth_mode, **extra_meta)` that all three call sites
             use, so the claim set cannot be partially populated. That module already owns the
             normalization; owning the construction is the same responsibility.
knowledge:   What a caller must supply for a slash command to be authorized correctly: permissions,
             roles, is_admin, auth_mode, is_single_user_owner, auth_user_id. The two endpoint copies
             supply all six from the authenticated user. The orchestrator copy supplies two, derives
             a third from the process environment, and parses the fourth out of a string.
scenario:    `authorize_command` (command_authorization.py:86-122) grants when
             `context.auth_mode == "single_user" and context.is_single_user_owner`
             (command_authorization.py:101-102). In the orchestrator copy, `is_single_user_owner` is
             `os.getenv("AUTH_MODE").strip().lower() == "single_user"` — a property of the SERVER, not
             of the caller — so on a single-user deployment that condition is unconditionally true and
             every `rbac_required` slash command is auto-granted on that path regardless of who asked.
             Symmetrically, on a multi-user deployment the copy omits `permissions`/`roles`/`is_admin`
             entirely, so `_permission_in_claims` (command_authorization.py:72-82) can never grant and
             the decision falls to `permission_checker(int(context.auth_user_id), ...)` with an
             `auth_user_id` obtained by `int()`-parsing whatever string the caller passed as
             `llm_user_identifier` — i.e. the caller selects the RBAC subject.
             **Reachability is what holds the severity down and must be stated plainly**:
             `_build_command_context` is called only from `_chat_sync_impl` and `achat`, which are
             reachable only through `core/Chat/Workflows.py` (dead — see stage 1 finding `chat-4`).
             This is a latent trap, not a live escalation. It matters because the orchestrator is the
             "core" entrypoint, so any NEW core caller that starts using `achat()` inherits the
             weakened context silently, and nothing would fail.
impact:      Medium. Not currently exploitable; a permanently-armed footgun on an authorization path,
             in the one core/Chat module with zero direct test coverage.
cost-driver: n/a
tests:       command_router: 7 files (tests/Chat_NEW/unit/test_command_router.py,
             test_chat_command_injection.py, tests/Chat_NEW/integration/test_chat_command_audit.py,
             _concurrency, _perf, _replace_mode, test_chat_skill_commands_injection.py).
             `core/Chat/command_authorization.py`: **ZERO importing test files**, verified with both
             import forms. No test constructs the orchestrator's CommandContext.
             Import-grep reachability, not measured coverage.
effort:      cheap for the shared constructor. Cheaper still if `chat-4`'s deletion lands first, which
             removes the divergent copy outright. Add a direct unit test for `authorize_command`
             covering: admin claim, wildcard permission, parent-wildcard, single-user-owner,
             `auth_user_id is None` deny, and `permission_checker` raising -> deny.
owner-only:  YES — two of the three construction sites are in api/v1/endpoints/chat.py.
confidence:  confirmed (the three divergent constructions and the grant conditions); confirmed (the
             zero test coverage of command_authorization.py); confirmed (the reachability limit).
```

### FINDING chat-12
```
axis:        encapsulation
class:       n/a
severity:    Medium
sites:       TRUE INVERSION — core/ importing api/v1/endpoints/* or api/v1/API_Deps/*, 7 sites:
               core/Chat/chat_service.py:34    [module-level] API_Deps.ChaCha_Notes_DB_Deps
                                               -> DEFAULT_CHARACTER_NAME
               core/Chat/chat_helpers.py:13    [module-level] same import
               core/Chat/chat_history.py:29    [module-level] same import
               core/Chat/chat_service.py:2092  [deferred] endpoints.llm_providers
               core/Chat/chat_service.py:2160  [deferred] endpoints.chat            (see chat-2)
               core/Chat/chat_service.py:2241  [deferred] endpoints.chat            (see chat-2)
               core/Chat/command_router.py:658 [deferred] API_Deps.ChaCha_Notes_DB_Deps
                                               -> get_chacha_db_for_user_id
             MILD — schema-only, 7 sites: core/Chat/chat_orchestrator.py:25,
               core/Chat/chat_target_resolution.py:11, core/Chat/chat_loop_store.py:10,
               core/Chat/chat_helpers.py:14, core/Chat/chat_service.py:2138, :2233, :2792
             SEPARATE BOUNDARY, same shape — reaching past a module's public API:
               core/Chat/chat_service.py:6205  `await sse_stream._enqueue(ln)` — a private method of
               core/Streaming/streams.py:SSEStream (:337-350), called because `SSEStream` exposes no
               public "insert this line verbatim, skip normalization" operation and the caller needs
               one for trusted-local frames.
canonical:   Docs/Architecture.md: "Clients -> FastAPI endpoints -> Core domain services -> Databases".
destination: (a) the three module-level `DEFAULT_CHARACTER_NAME` imports: move that constant to
                 `core/Chat/` or `core/Character_Chat/` — the honest reading is that the CONSTANT is
                 in the wrong package, not that core is wrong to want it. One-line move, three
                 re-points, plus a re-export so the API_Deps module keeps working.
             (b) `chat_service.py:6205`: add a public `SSEStream.send_verbatim_line(line)` to
                 `core/Streaming/streams.py` with the same body, and call that.
             (c) the deferred endpoint imports at :2092, :2160, :2241: see `chat-2`; they are that
                 finding's symptom, not a separate fix.
knowledge:   n/a (encapsulation axis)
scenario:    n/a
impact:      Medium. The three module-level imports mean `core/Chat` cannot be imported without
             importing `api/v1/API_Deps`, so the dependency is structural and not merely a lint
             nuisance: it constrains anyone trying to use `core/Chat` outside the FastAPI app (a
             worker, a CLI, a test harness that does not want the API package). The `_enqueue` reach
             is milder — one line — but it is the kind of call that silently breaks when the owning
             module changes its queue policy, and there is nothing at the boundary to notice.
cost-driver: n/a
tests:       tests/lint/test_endpoint_auth_deps_import_boundary.py is the precedent — an AST-based
             import ratchet with a ban list and a required re-export set, already in the suite.
             No equivalent guards core -> api.
effort:      cheap for (a) and (b). The PROPORTIONATE recommendation for the general problem is NOT a
             mass refactor: add a sibling AST ratchet test modelled on
             tests/lint/test_endpoint_auth_deps_import_boundary.py, seeded at the current repo-wide
             count (161 imports / 107 files) so the number can only go down, and let `core/Chat`'s 7
             true inversions be the first entries removed from the allowlist.
owner-only:  no for the core/ edits; the re-export shim touches api/v1/API_Deps -> owner-only.
confidence:  confirmed
```

### FINDING chat-14
```
axis:        duplication
class:       justified-divergence (the split), adoption-gap (the missing contract)
severity:    Low
sites:       tldw_Server_API/tests/Chat/**      — 100 files, 61,960 LOC, 2,323 collected tests
             tldw_Server_API/tests/Chat_NEW/**  —  58 files, 11,468 LOC,   394 collected tests
             .github/workflows/ci.yml — 6 shard names across 5 matrix blocks:
               chat-legacy-integration, chat-legacy-unit-a-l, chat-legacy-unit-m-z,
               chat-new-integration-property, chat-new-unit-a-l, chat-new-unit-m-z
             tldw_Server_API/tests/Chat_NEW/conftest.py:17 and :201 — the only in-repo text that comes
               near a rationale, and it explains a fixture choice, not the split
             core/Chat/README.md:180,329 — points contributors at tests/Chat only
canonical:   n/a
destination: n/a — do NOT merge the trees.
knowledge:   "Which tree does a new Chat test go in." Nothing in the repo answers it.
scenario:    n/a
impact:      Low, and I want to be explicit that this is the OPPOSITE of what the parallel-`_NEW`-tree
             pattern usually indicates, because the evidence says so:
             - **It is not duplication.** Exactly ONE basename appears in both trees
               (`test_chat_knowledge_save.py`), and that pair is a unit test of the router against a
               stubbed DB vs an integration test through the shared fixtures, with zero overlapping
               test names. There is no redundant-test problem to solve.
             - **It is not an abandoned migration.** tests/Chat was born 2025-04-24, Chat_NEW
               2025-09-04. In the last 6 months tests/Chat took 108 commits and Chat_NEW took 49 —
               the "legacy" tree is twice as active. Both were touched in the last week.
             - **Both run in CI**, sharded explicitly, with a contract test that (per
               backlog/task-2234) proves every Chat/Chat_NEW/Chatbooks/Streaming file is covered
               exactly once. There are no exclusions: no pytest.ini, setup.cfg, tox.ini, or
               norecursedirs exist in this repo.
             What IS missing is the written rule. CI is the only place in the repo that labels the
             split, and it labels tests/Chat "legacy" — which the commit counts contradict. Also note
             the trees are 620 vs 198 average LOC per file, so they are not even the same kind of
             test. The cost is Low and steady: every new Chat test is a coin flip, and the module
             README sends contributors to only one of the two.
cost-driver: n/a
tests:       n/a (this finding is about the test estate itself)
effort:      cheap — one README section in core/Chat/README.md or a `tests/Chat_NEW/README.md` stating
             the rule, plus a line in core/Chat/README.md pointing at both trees. Optionally rename
             the CI shards so "legacy" stops describing the more active tree.
owner-only:  no
confidence:  confirmed
```

## Synthesis

Full finding list, most severe first. Each finding's evidence lives in the stage named.

| # | Axis | Class | Sev | One line | Stage |
| --- | --- | --- | --- | --- | --- |
| `chat-1` | correctness | divergent-copies | High | Double-escaped `r"HTTP\\s+(\\d{3})"` never matches, so an upstream 429 is returned as 502; and in this module the branch is doubly dead because `NetworkError` is not in the caught exception tuple | 2 |
| `chat-2` | encapsulation | n/a | High | Production credential resolution reaches into `api/v1/endpoints/chat.API_KEYS` under a test flag — layering inversion plus a prod/test path split on the module's densest defect cluster | 3 |
| `chat-3` | efficiency | adoption-gap | High | One `get_message_metadata` query + one thread hop per history message, serially, when `get_message_metadata_map` exists and has 4 adopters | 2 |
| `chat-4` | duplication | divergent-copies | High | `_chat_sync_impl` / `achat` are 320-line twins that disagree on whether a RAG-assembly failure is a 500 or a silently ungrounded answer — and both are dead code | 2 |
| `chat-5` | duplication | true-duplication | High | `document_generator.py` owns 4 tables, 9 DDL statements, an 84-line hand-rolled migration and a second job lifecycle inside core/Chat | 3 |
| `chat-6` | duplication | adoption-gap | Medium | 33 hand-built SSE frames and a same-named control-prefix constant with a different value, while `core/LLM_Calls/sse.py` has 19 adopters elsewhere | 2 |
| `chat-7` | duplication | divergent-copies | Medium | Three data-URI redactors and two rounding rules behind rate limiting, billing and a blocking guardrail | 2 |
| `chat-8` | duplication | divergent-copies | Medium | `write_mandatory_moderation_audit` twice; one copy redacts the exception, the other logs it raw with a traceback | 2 |
| `chat-9` | encapsulation | n/a | Medium | 3,000 of `chat_service.py`'s 7,285 lines are three functions; five retry-cluster fixes in one quarter is the measured cost | 1 |
| `chat-10` | duplication | divergent-copies | Medium | Three `CommandContext` constructions; the core copy forges single-user-owner from the process env and takes the RBAC subject from a caller string | 3 |
| `chat-11` | duplication | true-duplication | Low | The C1 base64 idiom's one in-module site is an HMAC-signed approval token, not a cursor — record the trust boundary before anyone consolidates | 2 |
| `chat-12` | encapsulation | n/a | Medium | 7 true core→api inversions (3 module-level, for one constant) plus a private `SSEStream._enqueue` reach; fix with a ratchet test, not a mass refactor | 3 |
| `chat-13` | duplication | divergent-copies | Low | `REFACTORING_PLAN.md` names `chat_orchestrator` as the source of truth and `chat_service` as a facade — the exact inverse of the shipped code | 1 |
| `chat-14` | duplication | justified-divergence | Low | tests/Chat vs tests/Chat_NEW is a real, active, non-duplicative partition with no written rule for which tree a test goes in | 3 |

### Observations recorded but NOT raised as findings

- **Blind excepts.** 7 core/Chat files carry blind `except Exception` while *not* being on the BLE001
  grandfather list at `pyproject.toml:959-969` (which covers `Workflows.py`, `chat_characters.py`,
  `chat_helpers.py`, `chat_history.py`, `chat_loop_approval.py`, `chat_metrics.py`,
  `command_router.py`, `conversation_enrichment.py`, `prompt_template_manager.py`,
  `provider_manager.py`, `streaming_utils.py`). The 9 sites carrying no inline `# noqa: BLE001` are
  `chat_service.py:2647, :2712, :3537`; `request_queue.py:742, :953, :1009, :1298`;
  `chat_orchestrator.py:675`; `moderation_pipeline.py:164`. Only one of them causes a concrete
  consequence I can demonstrate (`moderation_pipeline.py:164` — that is `chat-8`), so the rest stay
  out of the findings per the audit's noise floor. Worth a glance, not a ticket.
- **`_attach_internal_http_hooks`** (`chat_service.py:2514-2516`) is a documented no-op
  ("Compatibility no-op: transport hooks are adapter-owned") still called on both hot paths
  (`:2643`, `:2657`). Two function calls per request; delete when convenient.
- **Import-time config freeze.** `chat_service.py:570` and `streaming_utils.py:911` call
  `load_comprehensive_config()` at module scope, and `chat_service.py:1055-1077` reads
  `CHAT_HISTORY_LIMIT` / `CHAT_HISTORY_ORDER` from the environment at import time, while
  `chat_dictionary.py:57,79`, `command_router.py:83` and `prompt_cost_guardrails.py:88` read the same
  config at call time. `load_comprehensive_config` is `@lru_cache(maxsize=1)` (`core/config.py:2343`)
  so there is no repeated I/O cost — this is a lifetime inconsistency, not an efficiency problem, and
  it is too small to be a finding on its own.
- **`STREAMS_UNIFIED` is genuinely tested** (19 test files set it, on both settings). The
  "off-by-default and therefore untested" pattern from the audit briefing does **not** apply here, and
  I checked specifically because it was flagged as a known-real risk elsewhere in the repo.

## Suggested Refactor/Actions

Actions for stage-1 and stage-2 findings are recorded in those stage files. Stage-3 actions:

1. **`chat-2` — remove the endpoint reach-in from credential resolution.** Add an injected resolver
   parameter; migrate the six credential test files to inject. Needs `Docs/Design/`, an ADR entry
   under ADR-025's provider-credential boundary, and a Backlog task. Owner-only.
2. **`chat-10` — move `CommandContext` construction into `command_authorization.py`** and add the
   missing direct unit test for `authorize_command`. If `chat-4`'s deletion lands first, the divergent
   copy disappears and only the test remains to write. Owner-only (two endpoint sites).
3. **`chat-12` — two cheap edits plus a ratchet.** Move `DEFAULT_CHARACTER_NAME` out of
   `api/v1/API_Deps/ChaCha_Notes_DB_Deps.py` (re-export to keep the old path working); add
   `SSEStream.send_verbatim_line`. Then propose a sibling AST import-ratchet test modelled on
   `tests/lint/test_endpoint_auth_deps_import_boundary.py`, seeded at the current 107 files. Do NOT
   propose a mass refactor of the 161 imports.
4. **`chat-5` — stage it.** Split the cheap part off first (`format_map(locals())` ->  a literal
   f-string over the allowlisted `set_clause`, `document_generator.py:1213`). The schema and job-system
   moves need `Docs/Design/`, an ADR, a Backlog task and an `IMPLEMENTATION_PLAN` with a data
   migration; they are not drive-by work and should not be started without owner agreement.
5. **`chat-14` — write the rule down.** One paragraph in `core/Chat/README.md` saying which tree new
   Chat tests go in and why both exist, and a pointer to both trees rather than only `tests/Chat`.
   Optionally rename the CI shards so "legacy" stops describing the more active tree.

## Not covered

Stated explicitly, per the audit's final quality bar.

- **Seed clusters with no material sites in this module**, verified rather than assumed:
  - **C2 (scalar/env coercion)** — 9 private coercer definitions exist in core/Chat
    (`chat_service.py:576,585`; `chat_loop_engine.py:8`; `chat_dictionary.py:45`;
    `validate_dictionary.py:88,95`; `command_router.py:88,102`; `prompt_cost_guardrails.py:431`), but
    **6 of them already delegate to the shared `core/testing.is_truthy`**. Only
    `chat_loop_engine.py:8-18` hand-rolls a truthy set, and it does so to accept the domain tokens
    `"enabled"/"enable"` and `"legacy"` — justified-divergence. Not reported.
  - **C3 (datetime)** — exactly ONE `datetime.utcnow()` in the whole module
    (`chat_exceptions.py:121`), zero private `_utc_now`/`_now_iso`. Below the bar. Not reported.
  - **C4 (inline exponential backoff)** — ZERO sites. Retry in core/Chat is fixed-count with no delay
    (`chat_history.py:308-309`, `chat_helpers.py:315`). Not reported.
  - **C5 (PromptStudioDatabase dual-backend)**, **C9 (Discord/Slack clones)**, **C10 (MCP
    sanitize_input)** — no sites in this module.
  - **C6 (tiktoken fallback)** — ZERO tiktoken usage in core/Chat; all estimation is the `len//4`
    heuristic. The heuristic's own divergence IS reported, as `chat-7`.
  - **C7 (LLM-output JSON / DB-blob coercion)** — no byte-identical pairs found in this module.
- **`core/Chat/request_queue.py` (1,335 LOC), `rate_limiter.py` (644), `chat_metrics.py` (1,073),
  `bounded_daemon.py` (405)** were inventoried and spot-read but not audited line by line. They are
  moderately churned (16 / 23 / 15 / low commits) and well covered (6 / 10 / 10 / 34 importing test
  files). Deprioritized against `chat_service.py`'s 148 commits; a follow-up pass on `request_queue.py`
  is the highest-value remaining target inside this module.
- **`core/Chat/document_generator.py`'s LLM-call and context-formatting halves** (`:346-775`) were not
  audited; only its persistence and job-lifecycle surface, which is where `chat-5` lives.
- **Measured coverage.** Every `tests:` line in this ledger is IMPORT-GREP REACHABILITY. The suite was
  not executed (only `--collect-only`, whose output is recorded above). No claim in this ledger should
  be read as a coverage measurement.
- **`api/v1/endpoints/chat.py` (8,077 LOC / 204 commits)** was read only where a core/Chat finding
  required tracing into it. It is a larger and hotter file than anything in this module and deserves
  its own review; it is owner-only territory.
- **Performance measurement.** `chat-3`'s cost is derived from reading the loop and the configured
  bounds, not from a benchmark. No profiling was run.
