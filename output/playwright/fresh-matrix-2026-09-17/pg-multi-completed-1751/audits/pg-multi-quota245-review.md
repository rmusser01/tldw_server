# UAT245 / TASK13260.187 — fresh PostgreSQL multi-user quota failure

## Finding

**Fresh PostgreSQL AuthNZ bootstrap omits the `storage_quotas` table required by the active org/team upload guard.** This is a product schema-parity defect, not an exhausted quota, oversized source, invalid native fixture, or a reason to relax quota enforcement.

Retained normal startup completed initialize/bootstrap successfully on frozen `8f8774e6c868b304a96d95ab82e28389c129a78b`. Runtime role has no superuser/BYPASSRLS/role memberships. Alice2's real `/media/ingest/jobs` request returned413, detail.error `storage_quota_exceeded`, message `Quota check unavailable`, used_mb0/quota_mbnull. Backend excerpt at17:07:59UTC identifies `AuthnzStorageQuotasRepo.get_org_quota` and `relation "storage_quotas" does not exist` / `UndefinedTableError`. Thus the displayed413 is a lookup-unavailable failure, not evidence the1914-character source consumed a configured storage allowance.

## Missing schema path (line numbers in frozen source archive)

- `tldw_Server_API/app/core/AuthNZ/migrations.py:2932-2975`: SQLite migration051 defines `storage_quotas` and its indexes; registered around6251. It uses SQLite connection/AUTOINCREMENT and cannot simply be executed as PostgreSQL SQL.
- `.../AuthNZ/initialize.py:509-563`: SQLite setup runs ensure_authnz_tables. The separate PostgreSQL setup initializes users and calls `ensure_authnz_core_tables_pg`; it does not apply SQLite migration051.
- `.../AuthNZ/pg_migrations_extra.py:1220-1271`: canonical PG core DDL creates organizations and teams, the quota table's parents. There is **no `storage_quotas` table DDL** in this bootstrap list. A bounded search of the frozen API Python/SQL files found only the SQLite declaration.
- `.../AuthNZ/pg_migrations_extra.py:3530-3574`: ensure_authnz_core_tables_pg runs its declared DDL transaction and existing profile/session checks, then returns success. Since storage quota schema is absent from that declaration/validation, initialization can legitimately report success while this request dependency is not ready.

## Actual failure propagation

1. `api/v1/endpoints/media/ingest_jobs.py:575` includes guard_storage_quota as an admission dependency.
2. `api/v1/API_Deps/storage_quota_guard.py:65-83` skips the guard in single-user profile mode; multi-user resolves the active organization from authenticated request/user state.
3. `core/Storage/quota_enforcement.py:61-87` invokes the real quota repo and catches lookup errors. Its default is fail closed, returning exactly `Quota check unavailable` with allowed=false and unset allowance. It does not pretend a missing table means no configured quota.
4. `core/AuthNZ/repos/storage_quotas_repo.py:62-84` selects from storage_quotas using PostgreSQL `$1` and reraises the missing-relation exception. This is not a SQLite-placeholder or content-schema query.
5. The guard's result branch `storage_quota_guard.py:118-134` returns the observed sanitized413. No native job-processing or content-RLS success is established by this attempted ingest.

`UndefinedTableError` identifies a missing relation, not InsufficientPrivilege or a row-level-policy denial. Source omission corroborates it without requiring DB catalog queries. This diagnosis does not assert unseen database state beyond retained error evidence.

## Separate issues / prior work

- **UAT238 PG single:** single-user explicitly bypasses this admission guard, so its ingest can reach the distinct Media persistence/RLS failure. This multi-user missing-AuthNZ-table failure happens earlier. Fixing245 does not certify238 or subsequent content writes.
- **TASK12009 (Done):** intentionally hardened Storage failures to sanitized fail-closed behavior. Its relevant guarantee is working here. Do not undo it with STORAGE_QUOTA_FAIL_OPEN, disabled enforcement, missing-table-as-unlimited, or role changes.
- **TASK13233.2 (To Do):** concerns reconciliation of actual stored originals across USER_DATA_BASE_PATH/other storage roots and cache/deduplication. No usage scan caused this failure; a schema lookup fails first. Keep its accounting design separate.

## Smallest future repair / exact constraints

Add idempotent PostgreSQL storage-quota DDL to the canonical AuthNZ bootstrap after organization/team parents, following its existing transaction and ownership conventions. Prefer that established list over request-time DDL or a second independent schema installer. Preserve the SQLite051 contract: identity ID, nullable org/team foreign keys with cascading parent deletion; integer quota/thresholds, fractional used_mb; created/updated timestamps; allowed org-only/team-only/both-null combinations; prohibition of both org and team nonnull. Preserve defaults (table quota10240MB, soft80/hard100; repository team creation explicitly chooses5120).

The two partial unique indexes are functional requirements: PostgreSQL repo upserts use `ON CONFLICT (org_id) WHERE org_id IS NOT NULL` and the corresponding team predicate. Creating a table without those compatible indexes is incomplete. Retain lookup indexes and normal schema-owner/bootstrap versus restricted-runtime grants lifecycle. No global RLS change, BYPASSRLS, special administrator request, manually created held-profile table, or quota policy weakening.

Existing bootstrap already returns false on required-DDL failure; the new declaration must participate in that failure contract. Verify fresh and repeat initialization with actual normal PostgreSQL bootstrap, not only a helper whose test schema creates the table independently.

## Existing tests and gap

- `tests/AuthNZ/unit/test_pg_migrations_authnz_core.py`: canonical DDL emission/readiness/failure controls; extend with the storage table/index declaration and required-DDL failure, but string tests alone are insufficient.
- `tests/AuthNZ/integration/test_authnz_session_schema_postgres.py`: real bootstrap/idempotence pattern to follow with official fixtures.
- `tests/AuthNZ/integration/test_authnz_orgs_teams_repo_postgres.py`: real parent org/team setup useful for quota CRUD controls.
- `tests/AuthNZ/integration/test_authnz_quotas_repo_postgres.py`: **different repository**, AuthnzQuotasRepo; tests vk_jwt_counters/vk_api_key_counters (request limits), not AuthnzStorageQuotasRepo. A pass there cannot close this schema gap.
- `tests/Admin/test_admin_storage_quotas.py:219-242`: preserved mocked-repository fail-closed/error-redaction control; other cases cover limits and warnings. No freshPG schema coverage.
- `tests/Billing/test_storage_quota_guard.py`: mocked check_storage_quota/DB pool admission and single-user-skip/organization resolution controls.
- `tests/Storage/conftest.py`, `tests/Storage/test_storage_quota_service.py`, `tests/Services/test_storage_quota_service.py`, `tests/AuthNZ/unit/test_storage_quota_service_backend_selection.py`: service mocks/fake pools/backend-selection coverage does not establish storage-quota table existence after a clean PostgreSQL initialize.

## Proposed required RED / GREEN (not run)

Using only official disposable pg_temp_db fixture and normal bootstrap:

1. Fresh initialize twice must make actual AuthnzStorageQuotasRepo.get_org_quota/get_team_quota return None for an owned parent without a quota. Before repair this produces UndefinedTableError. Prove no separate test-only CREATE TABLE supplies the missing relation.
2. Real org/team upsert twice (partial-index conflict handling), read, fractional usage update, quota status and deletion; parent-FK cascade and forbidden mixed owner row are enforced. Preserve both-null schema compatibility.
3. Real check_storage_quota + actual guard with authenticated multi-user organization: absent configured quota allows, under limit allows, soft warning remains, exhausted hard limit rejects sanitized413. Restricted runtime role must use bootstrap-created schema without role escalation.
4. A genuine quota lookup failure still rejects with `Quota check unavailable`; no changed fail-open behavior. Single-user guard behavior remains a separate regression control.
5. Repeat native multi-user upload after reviewed migration in the root-controlled new/retained profile, verifying admission and actual downstream result. Any later Media/RLS failure remains independently reported; do not claim full ingest success from fixing this earlier gate.

## Audit / scope

Only this Markdown and sibling `pg-multi-quota245-review.json` were persisted, beneath the packet's existing audits directory. JSON hashes all4 specified native inputs,18 relevant frozen source/test files, and the2 prior task references. All18 archived files exactly match Git blobs at frozen8f8774e6. Workspace HEAD at review was3189bb06; diagnosis uses frozen archive rather than assuming current worktree equivalence.

No production/test/task edits, DB queries/table creation, runtime/browser/config actions, inference or test execution. No native acceptance claimed. Parent owns TASK13260.187 and repair authorization.
