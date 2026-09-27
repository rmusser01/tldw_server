# Stage 2 — `repos/` data-source boundary

Date: 2026-09-21. Read-only.

## Scope

The 39 repository classes under `tldw_Server_API/app/core/AuthNZ/repos/`: the private helpers each one
re-declares, and the SQL that differs between its SQLite and Postgres branches beyond placeholder
spelling. Covers seed cluster **C7** (DB JSON-blob coercion) in full, plus the datetime and row-mapper
clusters found alongside it.

## Code Paths Reviewed

Cluster C7 — the five sibling JSON coercers named in the briefing, all confirmed at or within two lines
of the cited positions:

- `repos/shared_workspace_repo.py:_load_json_dict (25-36)`
- `repos/mcp_hub_repo.py:_load_json_dict (152-163)`, `_load_json_list (166-177)`
- `repos/prototype_workspaces_repo.py:_load_json_dict (75-86)`, `_load_json_list (89-100)`
- `repos/managed_secret_refs_repo.py:_parse_json_field (56-66)`
- `repos/data_subject_requests_repo.py:_parse_json_field (44-55)`

Datetime normalisation — canonical plus eleven bypassing copies:

- canonical: `repos/datetime_utils.py:_strip_tzinfo (6-19)`
- adopters (3 files): `repos/token_blacklist_repo.py:10` (uses at :119, :210, :249, :285);
  `repos/byok_oauth_state_repo.py:10` (uses at :88, :89, :191, :246, :356, :429, :470);
  `repos/monitoring_repo.py:13` (uses at :40, :78, :89, :112, :123, :173, :181)
- bypassing copies: `repos/org_provider_secrets_repo.py:66`, `repos/sessions_repo.py:35`,
  `repos/managed_secret_refs_repo.py:51`, `repos/federated_identity_repo.py:67`,
  `repos/identity_provider_repo.py:82`, `repos/user_provider_secrets_repo.py:57`,
  `repos/telegram_runtime_repo.py:178`, `repos/telegram_approvals_repo.py:92`,
  `repos/media_ingest_dedupe_repo.py:92`, `repos/workspace_provider_installations_repo.py:100`,
  `repos/mfa_repo.py:34`

Row mappers — 20 definitions of `_row_to_dict` across `repos/`, of which 17 share the exact signature
`(row: Any) -> dict[str, Any]`:

- `repos/shared_workspace_repo.py:121`, `telegram_runtime_repo.py:184`,
  `llm_provider_overrides_repo.py:90`, `mcp_hub_repo.py:264`, `media_ingest_dedupe_repo.py:98`,
  `telegram_approvals_repo.py:98`, `workspace_provider_installations_repo.py:104`,
  `prototype_workspaces_repo.py:191`
- `repos/org_provider_secrets_repo.py:70`, `byok_oauth_state_repo.py:47`,
  `federated_managed_grant_repo.py:58`, `federated_identity_repo.py:54`,
  `identity_provider_repo.py:69`, `maintenance_rotation_runs_repo.py:77`,
  `byok_validation_runs_repo.py:74`, `user_provider_secrets_repo.py:61`,
  `managed_secret_refs_repo.py:_row_to_dict (68)`

Dual-backend SQL bodies inspected line by line:

- `repos/sessions_repo.py:create_session_record (59-145)`
- `repos/quotas_repo.py:increment_and_check_jwt_quota (68-141)`
- `repos/usage_repo.py:aggregate_usage_daily_for_day (~637-725)`
- `repos/generated_files_repo.py:334-380` and `:441-490`
- `repos/orgs_teams_repo.py:159-210`
- `repos/billing_repo.py:93-101`

## Tests Reviewed

Located by import-grep (`grep -rl "AuthNZ.repos.<name>" tldw_Server_API/tests`), never by path.

| Test file | What it protects | Downgrades risk? |
| --- | --- | --- |
| `tests/AuthNZ/integration/test_authnz_sessions_repo_postgres.py`, `test_authnz_session_schema_postgres.py` | sessions_repo against a real Postgres | Partly — they do not assert `last_activity` is populated on insert |
| `tests/AuthNZ_SQLite/test_authnz_sessions_repo_sqlite.py` | sessions_repo on SQLite | No — single backend |
| `tests/AuthNZ/unit/test_authnz_sessions_repo_backend_selection.py`, `test_sessions_repo_refresh_gate.py` | which branch runs, against a stub pool | No — a stub cannot catch a column-list asymmetry |
| `tests/AuthNZ/integration/test_authnz_orgs_teams_repo_postgres.py`, `test_profile_array_parameters_postgres.py`, `test_auth_principal_api_key_happy_path.py` | orgs_teams_repo on real Postgres | Yes for orgs_teams |
| `tests/AuthNZ/integration/test_authnz_api_keys_repo_postgres.py` | api_keys_repo on real Postgres (marked `integration`, needs a live server) | Partly |
| `tests/AuthNZ_SQLite/test_authnz_{api_keys,users,usage,quotas,rate_limits,monitoring,mfa,sessions,token_blacklist,orgs_teams,org_provider_secrets,user_provider_secrets,llm_provider_overrides,byok_oauth_state}_repo_sqlite.py` | 14 repos on SQLite | No — single backend |
| `tests/AuthNZ/unit/test_authnz_generated_files_repo_backend_selection.py` | the **only** Postgres-branch coverage for `generated_files_repo.py`, and it is a stub | No |
| `tests/Storage/test_storage_quota_service.py`, `tests/Storage/test_storage_helpers.py`, `tests/Storage/test_voice_storage_integration.py`, `tests/VN_Assets/test_storage_cleanup.py` | generated_files_repo behaviour | No — all SQLite |
| `tests/Admin/test_admin_ops_new_endpoints.py`, `tests/Billing/test_billing_package_imports.py`, `tests/Billing/test_billing_repo_limits_defaults.py` | billing_repo | No — SQLite/import-only |

Aggregate for the five most-branched repos (`generated_files` 35 branch sites, `orgs_teams` 25,
`billing` 19, `sessions` 15, `api_keys` 14): of 31 test files reaching them, **6 exercise a real
Postgres, 6 drive the Postgres branch against stubs, and 19 are SQLite-only.** `generated_files_repo.py`
and `billing_repo.py` — 54 branch sites between them — have **zero real-Postgres coverage.**

No test anywhere asserts that two sibling copies of `_row_to_dict`, `_load_json_dict`, or
`_normalize_datetime_for_postgres` behave the same.

## Validation Commands

```
$ ls tldw_Server_API/app/core/AuthNZ/repos/*.py | wc -l
      39

$ grep -rn "def _row_to_dict" tldw_Server_API/app/core/AuthNZ/repos/ | wc -l
      20

$ grep -rn "def _normalize_datetime_for_postgres" tldw_Server_API/app/core/AuthNZ/repos/ | wc -l
      11

$ grep -rln "from tldw_Server_API.app.core.AuthNZ.repos.datetime_utils import" tldw_Server_API/app/ | wc -l
       3

$ grep -rn "replace(tzinfo=None)" tldw_Server_API/app/core/AuthNZ/ | wc -l
      40

$ grep -rc "replace(tzinfo=None)" tldw_Server_API/app/core/AuthNZ/ | grep -v ":0" | sort -t: -k2 -rn | head -5
token_blacklist.py:11
repos/api_keys_repo.py:8
repos/usage_repo.py:5
repos/sessions_repo.py:5
scheduler.py:1

$ grep -rn "def _load_json_dict\|def _load_json_list\|def _parse_json_field" tldw_Server_API/app/core/AuthNZ/repos/
repos/shared_workspace_repo.py:25:def _load_json_dict(raw: Any) -> dict[str, Any]:
repos/mcp_hub_repo.py:152:def _load_json_dict(raw: Any) -> dict[str, Any]:
repos/mcp_hub_repo.py:166:def _load_json_list(raw: Any) -> list[Any]:
repos/prototype_workspaces_repo.py:75:def _load_json_dict(raw: Any) -> dict[str, Any]:
repos/prototype_workspaces_repo.py:89:def _load_json_list(raw: Any) -> list[Any]:
repos/managed_secret_refs_repo.py:57:    def _parse_json_field(value: Any) -> dict[str, Any]:
repos/data_subject_requests_repo.py:45:    def _parse_json_field(value: Any, *, fallback: Any) -> Any:

$ python3 -c "import hashlib; hashlib.pbkdf2_hmac('sha256', b'x', b'salt', 10, dklen=0)"
ValueError: key length must be greater than 0.       # used in stage 3, recorded here for completeness
```

`asyncpg` json/jsonb codec check — `grep -n "set_type_codec" tldw_Server_API/app/core/AuthNZ/database.py`
returned no matches, so asyncpg hands these columns back as `str`, not `bytes` or a decoded object.
That is why the bytes-input divergence between the C7 copies is **not** a live scenario and is not
reported below.

## Findings

### FINDING authnz-5 — A canonical tz-stripping helper exists, eleven siblings bypass it with three incompatible semantics, and the canonical one is the wrong answer

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       canonical (adopted by 3): repos/datetime_utils.py:_strip_tzinfo (6-19)
               adopters: repos/token_blacklist_repo.py:10, repos/byok_oauth_state_repo.py:10,
                         repos/monitoring_repo.py:13
             Variant A — drop the offset, keep the wall clock (6 copies, same behaviour as canonical):
               repos/org_provider_secrets_repo.py:_normalize_datetime_for_postgres (66-67)
               repos/sessions_repo.py:_normalize_datetime_for_postgres (35-39)
               repos/managed_secret_refs_repo.py:_normalize_datetime_for_postgres (51-54)
               repos/federated_identity_repo.py:_normalize_datetime_for_postgres (67-70)
               repos/identity_provider_repo.py:_normalize_datetime_for_postgres (82-85)
               repos/user_provider_secrets_repo.py:_normalize_datetime_for_postgres (57-58)
             Variant B — convert to UTC, then drop the offset (4 copies):
               repos/telegram_runtime_repo.py:_normalize_datetime_for_postgres (178-181)
               repos/telegram_approvals_repo.py:_normalize_datetime_for_postgres (92-95)
               repos/media_ingest_dedupe_repo.py:_normalize_datetime_for_postgres (92-95)
               repos/workspace_provider_installations_repo.py:_normalize_datetime_for_postgres (100-101)
             Variant C — the opposite decision, stay aware in UTC (1 copy):
               repos/mfa_repo.py:_normalize_datetime_for_postgres (34-38)
             plus 40 inline `replace(tzinfo=None)` expressions, concentrated in
               token_blacklist.py (11), repos/api_keys_repo.py (8), repos/usage_repo.py (5),
               repos/sessions_repo.py (5) — including sessions_repo.py:81-86, which re-inlines the
               expression its own helper at :35 already wraps, 46 lines above it.
canonical:   repos/datetime_utils.py:_strip_tzinfo (6) — exists, is correctly placed, has exactly
             three importers, and implements Variant A.
destination: `repos/datetime_utils.py` is already the right home with the right single responsibility
             ("backend-agnostic timestamp normalisation for AuthNZ persistence"). It does not need
             replacing — it needs correcting and then adopting. Explicitly NOT Utils.py.
knowledge:   "how an aware datetime becomes a value for a naive Postgres TIMESTAMP column". There are
             currently three answers to that inside one directory, and the one with a designated
             shared home is the lossy one.
scenario:    `dt.replace(tzinfo=None)` on an aware non-UTC datetime discards the offset and keeps the
             local wall clock. Feed `datetime.now().astimezone()` — a common idiom that yields an
             aware datetime in the host's local zone — on a host set to `America/New_York` at
             23:30 EDT. Variant A and the canonical helper store `23:30`. Variant B stores `03:30` the
             next day, the correct UTC instant. For `repos/sessions_repo.py:990-991` (which normalises
             `access_expires_at` and `refresh_expires_at`) that is a session expiry written four hours
             into the future, read back by everything else in the system as UTC. For
             `repos/managed_secret_refs_repo.py` it is a secret-rotation deadline off by the host's
             UTC offset. The error is silent and scales with the deployment's offset — the same shape
             as the audit's anchor bug #2, except here there are eleven copies and three answers
             instead of two copies and one difference.
impact:      High. Two of the six lossy copies sit on session expiry and managed-secret rotation. The
             fix direction is also inverted from the obvious one: "adopt the canonical helper" would
             propagate the wrong semantics to eleven more call sites. The canonical helper must be
             corrected to Variant B first.
tests:       import-grep reachability, not measured coverage. `tests/AuthNZ_SQLite/test_authnz_sessions_repo_sqlite.py`
             and `tests/AuthNZ/integration/test_authnz_sessions_repo_postgres.py` reach sessions_repo;
             neither passes a non-UTC aware datetime. No test compares any two of the eleven copies.
             The class is invisible to the current suite.
effort:      moderate. The code change is small (one-line fix in `datetime_utils.py`, eleven deletions,
             eleven import lines) but it is a **behaviour change on stored timestamps**, so it needs a
             property test asserting all inputs round-trip to the same UTC instant, and a decision
             recorded about `mfa_repo`'s Variant C — its docstring says "for PostgreSQL TIMESTAMPTZ
             columns", which if true is a legitimately different column type and a
             justified-divergence that should be renamed rather than merged.
owner-only:  no
confidence:  confirmed (the canonical helper, its three importers, the eleven bypassing copies, the
             three semantics, the 40 inline expressions, and the in-file re-inlining at
             sessions_repo.py:81-86); probable-risk (that a non-UTC aware datetime reaches a Variant A
             site in production — I traced the call sites but did not find a caller that provably
             passes one, and everything I read upstream uses `datetime.now(timezone.utc)`).
```

### FINDING authnz-6 — 17 copies of a schema-agnostic row mapper, split on whether the None guard lives in the helper or in every caller

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       Class A — guards `row is None` inside the helper (8 copies):
               repos/shared_workspace_repo.py:121, repos/telegram_runtime_repo.py:184,
               repos/llm_provider_overrides_repo.py:90, repos/mcp_hub_repo.py:264,
               repos/media_ingest_dedupe_repo.py:98, repos/telegram_approvals_repo.py:98,
               repos/workspace_provider_installations_repo.py:104,
               repos/prototype_workspaces_repo.py:191
             Class B — no None guard; `_row_to_dict(None)` raises TypeError (9 copies):
               repos/org_provider_secrets_repo.py:70, repos/byok_oauth_state_repo.py:47,
               repos/federated_managed_grant_repo.py:58, repos/federated_identity_repo.py:54,
               repos/identity_provider_repo.py:69, repos/maintenance_rotation_runs_repo.py:77,
               repos/byok_validation_runs_repo.py:74, repos/user_provider_secrets_repo.py:61,
               repos/managed_secret_refs_repo.py:68
             Class A also inverts the fallback order: it tries `dict(row)` first, then `row.keys()`;
             Class B tries `row.keys()` first, then `dict(row)`.
canonical:   NONE — `repos/datetime_utils.py` is the precedent for where one would go.
destination: `repos/row_mapping.py`, single responsibility: "normalise a backend row object
             (asyncpg Record, sqlite3.Row, aiosqlite Row, plain dict, or None) to a plain dict".
             Explicitly NOT Utils.py and NOT http_client.py.
knowledge:   "what a database row looks like once it leaves the driver". Today that is answered 17
             times, and the two classes disagree about whose job the empty case is.
scenario:    n/a — duplication axis, and I am not claiming a live bug. Every one of the 12 Class-B
             call sites I traced does carry a guard (`repos/org_provider_secrets_repo.py:271, :311`;
             `byok_oauth_state_repo.py:127, :166, :207, :230`;
             `maintenance_rotation_runs_repo.py:199`; `byok_validation_runs_repo.py:183`; and the
             `_normalize_row` wrappers at `identity_provider_repo.py:176, :219, :250, :293, :438` and
             `federated_identity_repo.py:133, :171, :204`). The hazard is that the guard is
             **replicated at every call site instead of living in the helper**, and the call sites do
             not agree on the empty value: `identity_provider_repo.py:176` and `:219` return `{}`
             while `:250`, `:293` and `:438` return `None`, from the same repo. A thirteenth call site
             that forgets the guard gets `TypeError: 'NoneType' object is not iterable` from
             `dict(None)` rather than the empty dict its Class-A siblings would return.
impact:      Medium. Change amplification is the cost: any fix to row normalisation — a new driver, an
             asyncpg `Record` API change, a decision about how to handle duplicate column names —
             must be applied 17 times and will be applied to 12 or 14 of them. The 17 are
             schema-agnostic adapters with an identical signature, which is why the audit's standing
             ruling on `*_to_dict` (per-table typed mappers over distinct schemas, not shareable) does
             **not** apply here: these map no schema at all.
tests:       import-grep reachability, not measured coverage. Each owning repo has SQLite tests under
             `tests/AuthNZ_SQLite/` and, for 5 of the 17, a Postgres counterpart under
             `tests/AuthNZ/integration/`. None tests the helper directly; none tests the None case.
effort:      cheap. One new ~15-line module, 17 deletions, 17 import lines, and one decision recorded
             (guard inside, return `{}`). Well-covered indirectly by existing repo tests, which is
             what makes it cheap.
owner-only:  no
confidence:  confirmed.
```

### FINDING authnz-7 — Cluster C7: five JSON-blob coercers, four clamp the parsed type and one does not

```
axis:        correctness
class:       divergent-copies
severity:    Medium
sites:       repos/shared_workspace_repo.py:_load_json_dict (25-36)
             repos/mcp_hub_repo.py:_load_json_dict (152-163), _load_json_list (166-177)
             repos/prototype_workspaces_repo.py:_load_json_dict (75-86), _load_json_list (89-100)
             repos/managed_secret_refs_repo.py:_parse_json_field (56-66)
             repos/data_subject_requests_repo.py:_parse_json_field (44-55)
canonical:   NONE
destination: the same `repos/row_mapping.py` proposed in authnz-6 — "normalise a backend row and its
             JSON columns to plain Python". These two clusters travel together (every file carrying a
             JSON coercer also carries a `_row_to_dict`) and should not become two modules.
knowledge:   "what a TEXT/JSON column containing JSON becomes in Python, and what happens when it is
             malformed or the wrong shape".
scenario:    The four `_load_json_*` / `managed_secret_refs._parse_json_field` copies all clamp the
             parsed value to the expected container:
             `return dict(parsed) if isinstance(parsed, dict) else {}`.
             `repos/data_subject_requests_repo.py:_parse_json_field (44-55)` does not — it returns
             whatever `json.loads` produced. So a `data_subject_requests` row whose JSON column holds
             `"[]"`, `"null"`, `"123"` or `"\"x\""` — from a migration that changed the shape, a
             hand-written admin fix, or an older writer — yields a list, `None`, an `int` or a `str`
             where the caller's `fallback` promised a dict. `_normalize_record`
             (data_subject_requests_repo.py:63+) then `.get()`s it and raises `AttributeError` on a
             GDPR data-subject-request read path, instead of the empty container its four siblings
             would have produced. The same copy also narrows its exception handling to
             `json.JSONDecodeError` only, where the siblings catch `(TypeError, ValueError,
             json.JSONDecodeError)`; that specific narrowing is harmless because `json.loads` on a
             `str` raises nothing else, and I am not reporting it as a defect.
             Two further divergences I checked and am explicitly **dropping**: (a) all five silently
             return the empty container for `bytes` input — dead, because `database.py` sets no
             asyncpg type codec (`grep -n "set_type_codec"` → no matches) so json/jsonb arrive as
             `str`; (b) whitespace-only strings reach the same result by different routes.
impact:      Medium. One reachable `AttributeError` on an unusual-but-possible stored value, on a
             compliance surface, plus the usual change amplification across five copies. Not High
             because it needs an already-malformed row.
tests:       import-grep reachability, not measured coverage.
             `tests/AuthNZ_SQLite/` covers the SQLite side of the owning repos;
             `tests/AuthNZ/integration/` has no counterpart for `data_subject_requests_repo`, and
             `ensure_data_subject_requests_table_pg` (pg_migrations_extra.py:4155) has no test at all.
             No test feeds any of the five a malformed or wrong-shaped JSON blob.
effort:      cheap. Fold into the authnz-6 module; the correct behaviour is the four-copy majority
             (clamp to the expected container), so promote one of those and delete the other four plus
             the outlier.
owner-only:  no
confidence:  confirmed (the five copies and the missing type clamp); probable-risk (the
             AttributeError, which requires a row whose JSON column holds a non-dict).
```

### FINDING authnz-8 — Dual-backend SQL that differs in what it writes, not just how it spells placeholders

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       repos/sessions_repo.py:create_session_record — Postgres INSERT at :86-105 writes 12
               columns; SQLite INSERT at :114-139 writes 13, the extra one being
               `last_activity = datetime('now')`. Also in the same method: `expires_at` is bound as a
               naive `datetime` on Postgres (:81-86) and as `expires_at.isoformat()`, a string, on
               SQLite (:131); the new id comes from `RETURNING id` vs `cursor.lastrowid` (:140).
             repos/usage_repo.py:aggregate_usage_daily_for_day — the Postgres branch (~:660-685) is
               `ON CONFLICT (user_id, day) DO UPDATE SET requests, errors, bytes_total,
               latency_avg_ms` and **never writes `bytes_in_total`**; the SQLite branch (:688-705) is
               `INSERT OR REPLACE` listing seven columns including `bytes_in_total`; and its
               `except Exception` fallback (:707-724) is the same `INSERT OR REPLACE` with
               `bytes_in_total` **omitted**, which for a row-replacing statement resets that column to
               its default on every aggregation.
             repos/usage_repo.py:~637 — Postgres groups by `date(ts AT TIME ZONE 'UTC')`; SQLite uses
               `DATE(ts)` on stored text. Different day boundaries.
             repos/quotas_repo.py:increment_and_check_jwt_quota — Postgres `updated_at = CURRENT_TIMESTAMP`
               (server clock, timestamp type) at :97/:101; SQLite `datetime.now(timezone.utc).isoformat()`
               (app clock, text) at :108.
             repos/generated_files_repo.py:463-468 — Postgres `updated_at = CURRENT_TIMESTAMP` vs
               SQLite an app-side `datetime.now(timezone.utc).isoformat()`.
             repos/billing_repo.py:93, :95 and repos/generated_files_repo.py:342 — boolean literals
               `TRUE`/`FALSE` vs `1`/`0` inline in the SQL text.
             `INSERT OR IGNORE` standing in for a targeted `ON CONFLICT (...)`:
               repos/token_blacklist_repo.py:125 vs :139/:169;
               repos/orgs_teams_repo.py:865/1233/1297 vs :888/1242/1320;
               repos/quotas_repo.py:110 vs :95; rbac_seed.py:213, :218, :271 vs the PG branch at :156.
             SQL assembled by string interpolation from `locals()`:
               repos/generated_files_repo.py:383 and :471 (`update_sql_template.format_map(locals())  # nosec B608`)
canonical:   NONE
destination: n/a — this is not a helper-consolidation finding. The proportionate fix is coverage
             (below), not an ORM.
knowledge:   the write contract of each table: which columns a row gets on insert, which clock stamps
             it, what type the value is stored as, and which constraint violations are suppressed.
             Each of those is currently stated twice per method, in two SQL dialects, with no
             mechanism that notices when the two statements stop agreeing.
scenario:    Three concrete ones.
             (1) `sessions.last_activity`: a session created on Postgres has it NULL (the column is
             added by `core/DB_Management/authnz_session_schema.py:84` as
             `ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_activity TIMESTAMP` — nullable, no
             DEFAULT — and the PG INSERT omits it). `repos/sessions_repo.py:638-641` then serves the
             user's session list with `ORDER BY last_activity DESC`, where Postgres places NULLs
             first. Same endpoint, same user action: on SQLite the new session shows a timestamp and
             sorts by recency; on Postgres it shows null and sorts to the top. (No idle-timeout sweep
             reads this column today — I grepped for one and there is none — which is what keeps this
             a display bug rather than a session-lifetime bug.)
             (2) `usage_daily.bytes_in_total` has three behaviours: never written (Postgres),
             written correctly (SQLite primary), or reset to default on every run (SQLite fallback,
             because `INSERT OR REPLACE` replaces the whole row and that branch omits the column). The
             fallback is entered from a bare `except Exception`, so a transient error on the primary
             statement silently zeroes an accounting column.
             (3) `INSERT OR IGNORE` suppresses *every* constraint violation, not only the unique
             conflict its `ON CONFLICT (jti)` counterpart targets — so a NOT NULL or FK violation that
             raises on Postgres is a silent no-op on SQLite, at `token_blacklist_repo.py:139/:169`
             (token revocation) and `orgs_teams_repo.py:888/1242/1320` (team membership insertion).
impact:      High. These are not stylistic; they are different persisted state per backend, in
             session management, usage accounting and membership. And the coverage numbers say the
             divergence cannot be caught: `generated_files_repo.py` (35 branch sites, and the file
             that builds SQL with `format_map(locals())`) and `billing_repo.py` (19 branch sites) have
             **no real-Postgres test at all**; their only Postgres-branch test is a stub that asserts
             which method was called. This is the audit's known dual-backend precedent, instantiated.
tests:       import-grep reachability, not measured coverage. Of 31 test files reaching the five
             most-branched repos: 6 real-Postgres
             (`tests/AuthNZ/integration/test_authnz_{sessions,orgs_teams,api_keys}_repo_postgres.py`,
             `test_authnz_session_schema_postgres.py`, `test_profile_array_parameters_postgres.py`,
             `test_auth_principal_api_key_happy_path.py`), 6 stub-Postgres
             (`tests/AuthNZ/unit/test_authnz_*_backend_selection.py` and two others), 19 SQLite-only.
             A stub-pool test structurally cannot catch a column-list asymmetry.
effort:      expensive to unify; **moderate for the proportionate fix**, which is to make the
             Postgres branch reachable in tests for the repos that currently have no real-PG coverage.
             The harness already exists — `tests/AuthNZ/conftest.py:631 isolated_test_environment`
             provisions a per-test Postgres database, auto-starts Docker, and skips cleanly when
             unavailable. It is used by 56 test files, **none of them in `AuthNZ_SQLite`,
             `AuthNZ_Postgres`, `AuthNZ_Unit` or `AuthNZ_Federation`** (see stage 3).
owner-only:  no
confidence:  confirmed for every cited divergence (each read in full at the cited lines) and for the
             coverage counts; confirmed for scenarios (1) and (2); probable-risk for (3), which needs
             a constraint violation to actually occur.
```

## Suggested Refactor/Actions

1. **(cheap, no design doc) Create `repos/row_mapping.py`** owning row-to-dict normalisation and JSON
   column coercion — one module, one responsibility, next to the existing `repos/datetime_utils.py`
   which is the precedent for this shape. Promote the Class-A `_row_to_dict` (guard inside, return
   `{}`) and the majority `_load_json_dict`/`_load_json_list` (clamp to the expected container).
   Delete the 17 row mappers and the 7 JSON coercers. Addresses authnz-6 and authnz-7.

2. **(moderate, needs a design doc) Correct `repos/datetime_utils.py:_strip_tzinfo` to Variant B**
   (`dt.astimezone(timezone.utc).replace(tzinfo=None)`), then delete the eleven private copies and the
   40 inline expressions in favour of it. Decide and record what happens to `repos/mfa_repo.py:34`,
   whose Variant C is plausibly a genuine TIMESTAMPTZ-column difference rather than a bug. This one
   clears the design-first bar because it changes the value written to existing timestamp columns:
   needs `Docs/Design/2026-MM-DD-authnz-timestamp-normalisation-design.md`, an ADR entry, a Backlog
   task linking both, and a staged `IMPLEMENTATION_PLAN_authnz-timestamps.md`. Addresses authnz-5.
   Do the one-line correction to the canonical helper and its property test as stage 1 of that plan,
   before any adoption, so the three current adopters get the fix first.

3. **(moderate) Give the unguarded repos real-Postgres tests before touching their SQL.**
   `generated_files_repo.py` and `billing_repo.py` are the two with zero real-PG coverage and 54
   branch sites between them. Use `isolated_test_environment` from `tests/AuthNZ/conftest.py:631` —
   it already auto-starts Docker Postgres and skips cleanly, per `CLAUDE.md:245-248`. Then fix the
   `sessions.last_activity` omission and the `usage_daily.bytes_in_total` asymmetry. Addresses
   authnz-8; also the coverage half of authnz-2.

4. **(cheap) Delete the `format_map(locals())` SQL assembly** at `generated_files_repo.py:383` and
   `:471` in favour of explicit named substitution. The `# nosec B608` suppressions there are
   per-line, not policy — `B608` appears in neither the global ruff ignore list nor the per-file
   block — so they are worth a look on their own terms, and Bandit runs on touched scope per ADR-005,
   which means any PR editing this file will have to answer for them anyway.
