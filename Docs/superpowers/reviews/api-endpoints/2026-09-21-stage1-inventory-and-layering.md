# Stage 1 — Inventory, layering, and god-module shape

## Scope

`tldw_Server_API/app/api/v1/endpoints/` — 269 `.py` files, 247,943 LOC. The largest in-scope target
in the repo-wide audit. 248k LOC cannot be read linearly, so the reading list was derived from
size x churn (sidecars `2026-09-21-stage1-churn-baseline.txt`, `-hotspot-sizes.txt`) and from the
mechanically checkable rules in `Docs/Architecture.md`.

Binding decisions checked before asserting anything: `Docs/ADR/018-resource-governance-endpoint-policy-and-route-map.md`
(claim-first auth, Resource Governor route-map ownership, missing request policy denies) and
`Docs/ADR/019-security-request-edge-middleware.md` (request-edge middleware belongs to
`core/Security/`, wired in `main.py`). Neither ADR blesses raw SQL in endpoints, and no finding in
this ledger contradicts either.

## Code Paths Reviewed

Reading list, in size x churn order (LOC / commits in the last 12 months):

- `persona.py` (11,124 / 132) — 82 route handlers; 3,768 lines of module-level helpers before the
  first route at `:3769`
- `character_chat_sessions.py` (8,888 / 117)
- `watchlists.py` (8,739 / 103)
- `chat.py` (8,077 / 204) — 19 route handlers; `create_chat_completion (3361-6067)` is 2,707 lines,
  33% of the file
- `notes.py` (7,684 / 78)
- `embeddings_v5_production_enhanced.py` (6,792 / 127)
- `paper_search.py` (5,384 / 46)
- `workflows.py` (4,832 / 103)
- `audio/audio_streaming.py` (4,642 / 58)
- `mcp_hub_management.py` (4,462 / 50)
- `auth.py` (3,970 / 91)

Layering-rule sweep across all 269 files (raw SQL, DDL, interpolated SQL, dual-backend branching):
full per-file result in `2026-09-21-stage1-raw-sql-inventory.txt`.

Specific symbols cited below:

- `admin/admin_rbac.py` — 69 `.execute(`/`.fetchall(` sites, all category (a)
- `jobs_admin.py:596,601,681,687,711,1384,1388,1532,1536,1623,1638,1650` — `jm._connect()` /
  `jm._pg_cursor()` private reach-through into `core/Jobs/manager.py`
- `sync.py:2617-2848` — a complete sync write engine (INSERT/UPDATE/soft-delete + optimistic locking)
- `notes.py:3487,3490,3980,3984` — raw keyword SQL beside a call to `db.get_keywords_for_conversation()`
- `vector_stores_openai.py:542` — imports `core/Embeddings/vector_store_batches_db.py`'s private
  `_connect`
- `audio/audio_studio.py:860` — `collections_db.backend.execute` + the owner's private
  `_coerce_bool_flag`
- `prompt_studio/prompt_studio_status.py:236,245` — `warn_seconds` interpolated into
  `INTERVAL '{n} seconds'`
- `users.py:727,738` — `$1`/`$2` placeholders with no backend branch
- `core/AuthNZ/database.py:1966` — `_normalize_sqlite_sql`, the safety net that exists because of
  the above
- `tests/lint/test_endpoint_auth_deps_import_boundary.py` — the AST import-ratchet precedent

## Tests Reviewed

Located by import-grep against endpoint module names, never by path
(`grep -rl "endpoints\.<name>" tldw_Server_API/tests`). Full table in
`2026-09-21-stage3-test-inventory.txt`.

- `tests/lint/test_endpoint_auth_deps_import_boundary.py` — AST-based import ratchet with a ban list
  and a required re-export set. **Downgrades the risk of any new layering violation** if a sibling
  ratchet is added for raw SQL; today it covers auth-dep imports only, so it does not downgrade the
  raw-SQL finding.
- `tests/Admin/test_admin_rate_limits_api.py` — the only test that exercises the dual-backend SQL in
  `admin/admin_rate_limits.py`. It monkeypatches `_get_is_postgres_backend_fn` (`:184-195`,
  `:209-215`, `:245`) and drives a stub DB whose `_is_sqlite` flag is set at `:40` / `:69`. It asserts
  **which SQL string was passed**, not that Postgres executes it. Does not downgrade the
  dual-backend risk.
- `tests/AuthNZ/unit/test_admin_rbac_error_mapping.py`, `tests/Admin/test_admin_integration_smoke.py`,
  `tests/Admin/test_admin_users_role_consistency.py`, `tests/Admin/test_admin_split_compat_shims.py` —
  reference `admin_rbac` but cover error mapping, smoke, role consistency and compat shims, not the
  69 hand-written RBAC SQL statements. Partial downgrade only.

## Validation Commands

```
$ find tldw_Server_API/app/api/v1/endpoints -name '*.py' | wc -l
     269
$ find tldw_Server_API/app/api/v1/endpoints -name '*.py' | xargs wc -l | tail -1
  247943 total

$ git log --since='12 months ago' --name-only --pretty=format: -- tldw_Server_API/app/api/v1/endpoints/ \
    | grep '\.py$' | sort | uniq -c | sort -rn | head -5
 204 tldw_Server_API/app/api/v1/endpoints/chat.py
 132 tldw_Server_API/app/api/v1/endpoints/persona.py
 127 tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
 117 tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py
 103 tldw_Server_API/app/api/v1/endpoints/workflows.py

$ rg -c '\.execute\(|\.executemany\(|\.fetchall\(|\.fetchone\(' tldw_Server_API/app/api/v1/endpoints/ | wc -l
      25
    (top: admin/admin_rbac.py:69, jobs_admin.py:56, sync.py:15,
     prompt_studio/prompt_studio_evaluations.py:12, admin/admin_rate_limits.py:8)

$ rg -i -l '\b(CREATE TABLE|ALTER TABLE|CREATE INDEX|DROP TABLE)\b' tldw_Server_API/app/api/v1/endpoints/
    (no output — zero DDL sites in endpoints)

$ rg -n -i 'raw sql|endpoints thin' Docs/Architecture.md
31:The goal is to keep endpoints thin, push logic into core modules, and keep storage access centralized via `core/DB_Management/` and the vector store adapters.
180:  - Abstractions for SQLite/PostgreSQL; **no raw SQL in endpoints**.
347:- Centralize DB access via `core/DB_Management/`; avoid raw SQL in endpoints.

$ grep -n 'def get_db_transaction' -A5 tldw_Server_API/app/core/AuthNZ/database.py
1908:async def get_db_transaction():
1910:    pool = await get_db_pool()
1911:    async with pool.transaction() as conn:
1912:        yield conn
```

## Findings

### FINDING api-endpoints-2 — raw SQL in 20 endpoint files, 17 of them with an existing core owner

```
axis:        encapsulation
class:       adoption-gap
severity:    High
sites:       admin/admin_rbac.py (69 sites, :246-:1203 — full list in the raw-sql sidecar);
             jobs_admin.py (56 sites, :613-:2216, private reach-through at :596,601,681,687,711,
               1384,1388,1532,1536,1623,1638,1650);
             sync.py:2617,2647,2727,2794,2835,2837,2843,2848;
             prompt_studio/prompt_studio_evaluations.py:368,673,806,815,930,942,1064,1089;
             admin/admin_rate_limits.py:141,145,160,171,190,192,211,222;
             admin/admin_tenant_provisioning.py:87,116,126,134,139;
             media/debug.py:52; users.py:236,242,727,738;
             admin/admin_ops.py:1318,1324,1344,1352;
             prompt_studio/prompt_studio_status.py:60,62,69,97,125-129,165,170,196,203,233,242;
             prompt_studio/prompt_studio_projects.py:231,675;
             vector_stores_openai.py:542; outputs_templates.py:269-271,280;
             notes.py:3487,3490,3980,3984; media/navigation.py:742-751,1046-1054;
             media/listing.py:676-681,703-708; chat_documents.py:1131; watchlists.py:3762;
             media/versions.py:446-454; audio/audio_studio.py:860
canonical:   17 of the 20 already have an owner — core/AuthNZ/repos/rbac_repo.py;
             core/Jobs/manager.py + core/Jobs/operations/{sqlite,postgres}/;
             core/DB_Management/{media_db/,Sync_DB.py,PromptStudioDatabase.py,Users_DB.py,
             UserDatabase_v2.py,ChaChaNotes_DB.py,Collections_DB.py};
             core/Embeddings/vector_store_batches_db.py
destination: For the 3 orphan tables only: a new `core/AuthNZ/repos/rate_limit_repo.py` owning
             rbac_role_rate_limits + rbac_user_rate_limits; password_history folded into the
             existing Users_DB owner; user_prompts into the prompts owner. Everything else is an
             adoption move, not a new module.
knowledge:   The SQL dialect split and the table schemas. Every one of these 20 files must be edited
             again whenever a column is added, an index changes, or the Postgres/SQLite dialect
             diverges — in addition to the core owner that already encodes the same schema. Ten of
             the files hand-roll the dialect branch themselves, so a schema change has to be
             re-derived twice per file.
impact:      High because it is a direct, mechanically checkable violation of an operative rule
             (Architecture.md:180) at 20 sites, and because two of those files reach past a core
             module's *private* API (`jm._connect()`, `jm._pg_cursor()`,
             `vector_store_batches_db._connect`, `PromptStudioDatabase._execute`,
             `Collections_DB._coerce_bool_flag`) — so the owner module cannot refactor its internals
             without breaking endpoints. `core/AuthNZ/database.py:1966 _normalize_sqlite_sql` is a
             production safety net that exists solely because endpoints emit the wrong placeholder
             dialect; its docstring says so.
scenario:    n/a (encapsulation axis)
tests:       Import-grep reachability, not measured coverage. `admin_rbac` — 5 test files reference
             it, none covering its SQL; no direct endpoint test. `jobs_admin` — 7 files. `sync` — 19.
             `notes` — 24. `watchlists` — 47. `admin/admin_rate_limits` — 1
             (`tests/Admin/test_admin_rate_limits_api.py`), and it drives a stub, not a database.
effort:      Expensive overall; cheap per file for the 14 small ones (1-8 sites each with a named
             owner). `admin_rbac.py` (69) and `jobs_admin.py` (56) are genuinely expensive because
             the owner modules expose only reads and the endpoints hand-wrote every write.
owner-only:  yes
confidence:  confirmed (the violation and the site list); confirmed (the private reach-through);
             probable-risk (that any specific site misbehaves today — none was proven to)
```

### FINDING api-endpoints-12 — dual-backend SQL hand-rolled in 10 endpoint files, with the only backend test driving a stub

```
axis:        correctness
class:       divergent-copies
severity:    Medium
sites:       admin/admin_rbac.py (is_pg branch at nearly every one of its 69 sites);
             admin/admin_rate_limits.py:141,145,160,171,190,192,211,222;
             admin/admin_tenant_provisioning.py:87,116,126,134,139 (hardcodes `public.` vs `main.`
               schema prefixes);
             admin/admin_ops.py:1318,1324,1344,1352; users.py:727,738 (NO branch — $1/$2 only);
             jobs_admin.py (jm._pg_cursor branch at :601,687,711,1388,1536,1638,1650, ...);
             prompt_studio/prompt_studio_status.py:236,245 (Postgres-only INTERVAL literal);
             prompt_studio/prompt_studio_evaluations.py:799;
             media/navigation.py:742-751,1046-1054 (branches only to pick False vs 0);
             audio/audio_studio.py:860
canonical:   core/AuthNZ/database.py:1966 `_normalize_sqlite_sql` and its inverse
             `_convert_question_mark_to_dollar` — a runtime safety net, not a design
destination: n/a — the fix is adoption of the existing core owners (finding api-endpoints-2), which
             already own the dialect split
knowledge:   Which placeholder style, which upsert syntax, and which schema prefix each backend
             needs. Today that knowledge is written out 10 more times in the endpoints layer on top
             of the core owners that already encode it.
scenario:    `users.py:727,738` (`change_password`) emits `$1`/`$2` unconditionally. Under SQLite
             that statement only works because `_normalize_sqlite_sql` rewrites it at the pool
             boundary; its own docstring calls itself "a safety net to avoid aiosqlite warnings when
             a $-style query slips through". Any future pool that bypasses or tightens that
             normaliser turns a password change into a SQLite syntax error, on a path with no
             dedicated backend test.
impact:      Medium. The repo-wide audit dated 2026-09-21 established that cross-user isolation
             leaks fail to converge partly because the SQLite/PostgreSQL split is only tested on
             SQLite. This is that pattern inside the endpoints layer: every one of these branches is
             a place where the two backends can silently diverge.
tests:       Import-grep reachability, not measured coverage. `tests/Admin/test_admin_rate_limits_api.py`
             is the only test that deliberately exercises both branches — and it does so by
             monkeypatching `_get_is_postgres_backend_fn` (:184-195, :209-215, :245) against a stub
             DB flagged `_is_sqlite` at :40/:69, asserting the SQL *string*. No Postgres executes it.
             `tests/AuthNZ_Postgres/` exists and is the right home for the real version.
effort:      Moderate — the per-branch tests are cheap once the `isolated_test_environment` Postgres
             fixture in `tldw_Server_API/tests/AuthNZ/conftest.py` is reused; the SQL consolidation
             is the expensive half and is finding api-endpoints-2.
owner-only:  yes (the endpoint SQL); no (adding Postgres-side tests under tests/AuthNZ_Postgres/)
confidence:  confirmed (the branching and the stub-only test); probable-risk (the users.py scenario)
```

### FINDING api-endpoints-14 — two god modules whose shape blocks the repo's own decomposition precedent

```
axis:        encapsulation
class:       n/a
severity:    Medium
sites:       persona.py (11,124 LOC, 82 route handlers, 132 commits/12mo) — module-level helpers
               occupy :1-:3768, the first route is at :3769; the visual-packs group is
               :4030-:5766 (~34 routes, 1,737 lines: renderers, starter packs, library, packs,
               assets, generation jobs, export/import previews); `persona_stream` is one function at
               :7903-:11124 (3,222 lines, 29% of the file);
             chat.py (8,077 LOC, 19 route handlers, 204 commits/12mo) — `create_chat_completion`
               at :3361-:6067 is 2,707 lines, 33% of the file; remaining seams :70-3360 helpers,
               :6072-6448 queue + conversation helpers, :6455-6695 share-link crypto plus inline
               Pydantic models, :6698-7828 conversation CRUD/search/share/analytics,
               :7836-8077 RAG context + citations
canonical:   `core/DB_Management/media_db/` — the already-shipped split of `Media_DB_v2.py`
             (a 121-commit monolith) into `api.py`, `constants.py`, `errors.py`,
             `legacy_content_queries.py`, `runtime/`. Same team, same layer, already merged.
destination: For persona: a `persona/` package mirroring the `media_db/` boundaries, with
             `persona/visual_packs.py` as the first slice (self-contained resource, own router,
             1,737 lines). Not a rewrite — a router extraction.
knowledge:   Route registration, shared helper surface, and the auth/rate-limit dependency set.
             Today any change to a persona helper forces a reader to hold 11k lines in view to know
             which of 82 routes it affects; `chat.py` concentrates 204 commits of churn into one
             2,707-line function.
scenario:    n/a (encapsulation axis)
impact:      Medium rather than High: the shape is a maintenance drag and a review hazard, not a
             defect. It is reportable on taste grounds (god module) per the audit's bounded list,
             and it is the root cause of findings 4 and 5 — a 2,707-line handler and an 82-route
             file are exactly where a missing ownership check (`get_conversation_citations`) and a
             skipped offload helper (`persona_catalog`) hide.
tests:       Import-grep reachability, not measured coverage: `chat` 78 test files, `persona` 27.
             Well covered at the route level, which makes an extraction *cheap* — the router
             surface is pinned.
effort:      Moderate. Gated on coverage, and coverage is good. The `media_db/` migration is the
             template; start with `persona/visual_packs.py`.
owner-only:  yes
confidence:  confirmed
```

## Suggested Refactor/Actions

1. **Seed a raw-SQL import ratchet** modelled on `tests/lint/test_endpoint_auth_deps_import_boundary.py`
   — an AST test that fails when a new `endpoints/` file gains a `.execute(`/`.fetchall(` call,
   seeded at the current 20 files so the number can only go down. This is the proportionate
   recommendation; a mass refactor is not. Fits the `backend-required` gate
   (`Docs/Development/CI_REQUIRED_GATES.md`) since it is a pure-Python lint test. **Not owner-only** —
   the test lives under `tldw_Server_API/tests/lint/`.
2. **Close the 14 small raw-SQL sites first** (1-8 statements each, named core owner, `notes.py`'s
   two pairs being the cheapest — the correct call is already two lines away). Owner-only.
3. **`admin_rbac.py` and `jobs_admin.py` need a design doc**, not a patch: 125 statements between
   them and both reach past a core module's private API. `Docs/Design/2026-MM-DD-endpoint-raw-sql-adoption-design.md`
   + an ADR entry (this is a decision about where RBAC writes live) + `IMPLEMENTATION_PLAN_endpoint-raw-sql.md`
   with the 3 orphan tables as stage 1.
4. **Add real Postgres coverage for `admin/admin_rate_limits.py`** under `tests/AuthNZ_Postgres/`
   using the `isolated_test_environment` fixture from `tldw_Server_API/tests/AuthNZ/conftest.py`.
   Cheap, not owner-only, and it converts finding 12 from probable-risk to measured.
5. **Extract `persona/visual_packs.py`** as the first god-module slice, following the `media_db/`
   package boundaries. Needs a design doc. Owner-only.
6. Note for lint policy: `prompt_studio/prompt_studio_status.py:236,245` is the only endpoint site
   where a request value reaches SQL text rather than a bind parameter. It is clamped and
   `int()`-coerced, so it is not injectable, but it is the one site a Bandit B608 exemption should
   name explicitly rather than blanket.
