# Usage Quotas PR C (Storage Cut-over) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the per-user storage quota from the `users.storage_quota_mb` column to the UserProfiles `limits.storage_quota_mb` override. The enforced, written and displayed value becomes the one the spec 2 resolver returns (`null` means unlimited), and existing installs keep their custom quotas.

**Architecture:**
- **Service.** `StorageQuotaService` gets three helpers: `resolved_storage_quota_mb`, `quota_view` and `with_resolved_storage_quota`. Every enforcement and service read goes through the resolver, and the service caches hold usage only.
- **Writers.** Every writer funnels into `StorageQuotaService.set_user_quota`, which writes or deletes the user override, or uses the profile path's generic `limits.*` branch.
- **Readers.** Every display reader overlays the resolved value.
- **Migration.** An AuthNZ migration copies non-default column values into overrides: SQLite migration 100, and a one-time Postgres backfill guarded by a marker row. The column stays (`NOT NULL DEFAULT 5120`), but is no longer read for enforcement.

**Tech Stack:** FastAPI, Pydantic v2, SQLite/PostgreSQL (asyncpg) via `DatabasePool`, loguru, pytest. The frontend types are TypeScript.

**Spec:** `Docs/Design/2026-10-02-usage-quota-posture-design.md`, sections 2, 3, 5 and 8, plus Testing items 4, 8 and 9. PR A (#3098) and PR B (#3144, merged at 3700e2d6e7) are on `dev`.

## Global Constraints

- **Branches and PRs:** all PRs target `dev`, never `main`. Never use `git stash`; never pass `--no-verify`. Branch: `fix/usage-quotas-storage`.
- **Commit and PR text:**
  - Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. That exact trailer is the owner's rule.
  - The PR body has a `## Change summary` section recording the owner's ADR-004 waiver: "Waived by the repository owner (@rmusser01) on 2026-09-22 … *"I authorize waiving the change summary for each."* On 2026-10-03 … confirmed that this waiver covers spec 2 PRs B, C and D." It ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Backlog:** use backlog-py only (`PYTHONPATH=tools/backlog-py/src <venv python> -m backlog_py --cwd . task ...`, with ids written as `TASK-N`). Never the Node CLI, never hand-edit.
- **Code conventions:** loguru only; no new dependencies.
  - New tests carry `pytestmark = pytest.mark.unit` (integration tests: `pytest.mark.integration`), plus a one-line docstring per test and helper.
  - No review-process tags such as "(Qodo …)" or "(round N)" in code comments.
- **Running tests:**
  - Use `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (there is no bare `python`), with `TLDW_TEST_NO_DOCKER=1` for suites.
  - A failure counts as environmental only if it also occurs on `origin/dev`, proven on a detached checkout or by an import-graph check.
- **The test session runs with quotas ON:** the root conftest sets `LIMIT_ENFORCEMENT_ENABLED=true`, and the resolver cache is cleared per test. Tests that need quotas off clear both `USAGE_QUOTAS_ENABLED` and `LIMIT_ENFORCEMENT_ENABLED`.
- **Fakes that can't fail.** Never assert inside a fake that raises within a module that swallows `AssertionError`, or within a broad `except Exception`. Use call-recording fakes.
- **Gate the check, never the record.** `users.storage_used_mb` is always updated.
- **Semantics:**
  - The key is `limits.storage_quota_mb`, with catalog `default: null`, `minimum: 0` and `editable_by: [platform_admin]`.
  - Precedence: user, then the most generous team, then the most generous org. `0` blocks; `null` everywhere means unlimited.
  - The 100 MB minimum drops to 0 for per-user values. Team and org shared pools (`storage_quotas` table, `SetQuotaRequest`, `guard_storage_quota`) are out of scope and unchanged.
- **API contract:** schema changes alter the OpenAPI contract. Refresh the fingerprint with `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/export_openapi_schema.py --fingerprint apps/tldw-frontend/lib/api/openapi.fingerprint.json`. `make openapi-fingerprint` picks a too-old Python. Commit the fingerprint in the task that changes the schema.
- **Mirrored docs:** any edit to a doc mirrored under `Docs/Published` needs `bash Helper_Scripts/refresh_docs_published.sh`, committed together.
- **Merging:** merge into `dev` only on the go-ahead of peer session `tldw-server-03`, and message it when the merge lands. No hosted deployments exist, so there are no deploy prerequisites.

## Rulings carried into this plan

- **Group values for storage are allowed.** PR B rejected team/org `limits.storage_quota_mb` overrides only because enforcement still read the column. Once enforcement reads the resolver, a team or org value is each member's own storage allowance, matching owner decision 1. Remove the rejection. The team/org *shared pools* are a separate mechanism and stay as they are.
- **The registration-code branch is dead code and gets deleted.** `registration_service.py:393-394` (`if 'storage_quota_mb' in code_info`) can never fire, because no column or schema field carries it.
- **Admin system stats stop summing the column.** `total_quota_mb` becomes `null`: a sum of mostly-unlimited per-user values has no meaning. The admin page's formatter already renders `null` as "–".
- **Storage-endpoint writes bump no `profile_version`.** `set_user_quota` writes the override repo directly, the same as PR B's group-override routes. The profile path's generic branch still bumps it.
- **Postgres backfill runs once.** It is guarded by the marker table `authnz_data_backfills`, so an override an admin later deletes is never re-created from the stale column. SQLite's numbered migration already runs once.
- **The unused `UserQuotaUpdateRequest`** (`admin_schemas.py:228`, no references) stays untouched.

## Review Focus

1. **A quota an admin cleared stays cleared across restarts.** The Postgres backfill must not re-copy the stale column value. Covered by the Task 5 test `test_pg_backfill_runs_once_and_never_resurrects`.
2. **An admin "edit user" request that omits `storage_quota_mb` leaves the quota alone, and an explicit `null` clears it.** Covered by the Task 2 test `test_admin_update_quota_absent_vs_null`.
3. **Unlimited shows as `null`, never 0 or 5120,** on `GET /users/storage` (including its fallback path), `/users/me`, admin list and detail, and the profile `quotas` section. Covered by the Task 4 tests.
4. **A team `limits.storage_quota_mb` limits members' uploads, and a member's own value overrides it.** Covered by the Task 2 test `test_team_storage_value_enforced_and_user_value_wins`.
5. **Quota `0` blocks any non-empty upload, and quotas off admit 5 GB + 1 MB despite a stale 5120 column.** Covered by the Task 1 tests.

---

### Task 1: StorageQuotaService reads the resolver

**Files:**
- Modify: `tldw_Server_API/app/services/storage_quota_service.py`. This covers:
  - the module helpers;
  - `calculate_user_storage` (91-177), `check_quota` (179-253), `update_usage` (255-374) and `get_storage_breakdown` (395-412);
  - `set_user_quota` (445-501), `get_all_users_storage` (503-545), `check_combined_quota` (686-767) and `get_user_generated_files_usage` (1064-1078).
- Create: `tldw_Server_API/tests/Usage/test_storage_quota_resolver.py`.
- Modify the tests that pin the old contract:
  - `tldw_Server_API/tests/Storage/test_storage_quota_service.py`
  - `tldw_Server_API/tests/Services/test_storage_quota_service.py`
  - `tldw_Server_API/tests/Usage/test_usage_quotas_sites.py`
  - `tldw_Server_API/tests/AuthNZ/unit/test_storage_quota_service_backend_selection.py`
  - `tldw_Server_API/tests/AuthNZ/unit/test_versioned_user_write_gateway.py`
  - `tldw_Server_API/tests/DB_Management/unit/test_users_db_update_backend_detection.py`

**Interfaces:**
- Consumes: `quota_resolver.user_quota(user_id: int, key: str) -> float | None`, `quota_resolver.invalidate_user(user_id: int)`, `quota_checks.as_quota_user_id(value) -> int | None`, and `UserProfileOverridesRepo(db_pool)` with `.ensure_tables()`, `.upsert_override(*, user_id, key, value, updated_by, db_conn=None)` and `.delete_override(*, user_id, key, db_conn=None)`.
- Produces (module level in `storage_quota_service.py`):
  - `STORAGE_QUOTA_KEY = "limits.storage_quota_mb"`
  - `quota_view(used_mb: float, quota_mb: int | None) -> dict[str, Any]`, with the keys `quota_mb`, `available_mb` and `usage_percentage`, all `None` when unlimited
  - `async resolved_storage_quota_mb(user_id) -> int | None`
  - `async with_resolved_storage_quota(user: dict) -> dict`, a copy with `storage_quota_mb` replaced by the resolved value
  - `StorageQuotaService.set_user_quota(user_id: int, quota_mb: int | None, *, updated_by: int | None = None) -> dict`. It returns `user_id`, `storage_quota_mb` (the *effective* value after the write), `storage_used_mb`, `available_mb` and `usage_percentage`. It raises `UserNotFoundError`, raises `ValueError` on negative values, and raises `StorageError` on repo failure.

- [ ] **Step 1: Write the failing tests**

```python
"""Per-user storage quota comes from the spec 2 resolver, not users.storage_quota_mb (spec 2 §5)."""

import pytest

from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.services import storage_quota_service as sqs

pytestmark = pytest.mark.unit
MB = 1024 * 1024


class _Pool:
    """Minimal pool: one users row (used 100 MB, legacy column 5120)."""

    def __init__(self) -> None:
        """Seed one user."""
        self.rows = {7: {"storage_used_mb": 100.0, "storage_quota_mb": 5120}}

    async def fetchone(self, _sql: str, user_id: int):
        """Return the users row."""
        return self.rows.get(int(user_id))


@pytest.fixture()
def service(monkeypatch: pytest.MonkeyPatch) -> sqs.StorageQuotaService:
    """A service whose resolver answers from a dict the test controls."""
    limits: dict[int, float | None] = {}

    async def _user_quota(user_id: int, key: str):
        """The test's limit for the user (storage key only)."""
        return limits.get(user_id) if key == sqs.STORAGE_QUOTA_KEY else None

    monkeypatch.setattr(quota_resolver, "user_quota", _user_quota)
    svc = sqs.StorageQuotaService(db_pool=_Pool())
    svc._initialized = True
    svc.limits = limits  # type: ignore[attr-defined]
    return svc


async def test_unlimited_ignores_stale_column(service) -> None:
    """No limits value: a 5 GB + 1 MB upload is admitted although the column says 5120."""
    ok, info = await service.check_quota(7, 5 * 1024 * MB + MB)
    assert ok is True
    assert info["quota_mb"] is None and info["available_mb"] is None and info["usage_percentage"] is None


async def test_resolved_value_denies_one_mb_over(service) -> None:
    """A limits value of 150 MB with 100 MB used refuses 51 MB and admits 50 MB."""
    service.limits[7] = 150
    assert (await service.check_quota(7, 51 * MB))[0] is False
    service.invalidate_user_cache(7)
    assert (await service.check_quota(7, 50 * MB))[0] is True


async def test_zero_blocks_any_upload(service) -> None:
    """limits.storage_quota_mb = 0 refuses even a 1-byte upload."""
    service.limits[7] = 0
    ok, info = await service.check_quota(7, 1)
    assert ok is False and info["quota_mb"] == 0


async def test_quota_change_applies_without_waiting_for_usage_cache(service) -> None:
    """The 300 s cache holds usage only, so a new limit applies on the next check."""
    assert (await service.check_quota(7, 10 * MB))[0] is True
    service.limits[7] = 105
    assert (await service.check_quota(7, 10 * MB))[0] is False


def test_quota_view_unlimited_and_limited() -> None:
    """quota_view yields None fields when unlimited and arithmetic when limited."""
    assert sqs.quota_view(10.0, None) == {"quota_mb": None, "available_mb": None, "usage_percentage": None}
    assert sqs.quota_view(25.0, 100) == {"quota_mb": 100, "available_mb": 75.0, "usage_percentage": 25.0}
```

Add a test for `set_user_quota` in the same file. It uses a call-recording repo; Task 3's endpoint tests cover the real database round-trip:

```python
async def test_set_user_quota_writes_deletes_and_invalidates(service, monkeypatch: pytest.MonkeyPatch) -> None:
    """set_user_quota upserts the override, deletes it on None, rejects negatives, and clears the resolver entry."""
    calls: list[tuple] = []

    class _Repo:
        """Records override writes."""

        def __init__(self, _pool) -> None:
            """Ignore the pool."""

        async def ensure_tables(self) -> None:
            """Nothing to ensure."""

        async def upsert_override(self, **kw) -> None:
            """Record an upsert."""
            calls.append(("upsert", kw["user_id"], kw["key"], kw["value"]))

        async def delete_override(self, **kw) -> None:
            """Record a delete."""
            calls.append(("delete", kw["user_id"], kw["key"]))

    invalidated: list[int] = []
    monkeypatch.setattr(sqs, "UserProfileOverridesRepo", _Repo)
    monkeypatch.setattr(quota_resolver, "invalidate_user", invalidated.append)
    service.limits[7] = 0
    out = await service.set_user_quota(7, 0)
    assert calls == [("upsert", 7, sqs.STORAGE_QUOTA_KEY, 0)] and invalidated == [7]
    assert out["storage_quota_mb"] == 0 and out["storage_used_mb"] == 100.0
    service.limits.pop(7)
    out = await service.set_user_quota(7, None)
    assert calls[-1] == ("delete", 7, sqs.STORAGE_QUOTA_KEY) and out["storage_quota_mb"] is None
    with pytest.raises(ValueError):
        await service.set_user_quota(7, -1)
```

Import `UserProfileOverridesRepo` at module level in `storage_quota_service.py`, so the test can patch `sqs.UserProfileOverridesRepo`.

- [ ] **Step 2: Run the tests and confirm they FAIL**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_storage_quota_resolver.py -q -p no:xdist`

Expected: FAIL (`AttributeError: STORAGE_QUOTA_KEY` / `quota_view`).

- [ ] **Step 3: Add the module helpers**

Put these at module level, after the imports, in `storage_quota_service.py`:

```python
from tldw_Server_API.app.core.Usage import quota_checks, quota_resolver
from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo

STORAGE_QUOTA_KEY = "limits.storage_quota_mb"


def quota_view(used_mb: float, quota_mb: Optional[int]) -> dict[str, Any]:
    """Quota-derived fields for a usage figure; a None quota means unlimited (spec 2 §5)."""
    if quota_mb is None:
        return {"quota_mb": None, "available_mb": None, "usage_percentage": None}
    return {
        "quota_mb": quota_mb,
        "available_mb": round(max(0.0, quota_mb - used_mb), 2),
        "usage_percentage": round((used_mb / quota_mb * 100) if quota_mb > 0 else 0, 1),
    }


async def resolved_storage_quota_mb(user_id: Any) -> Optional[int]:
    """The user's enforced limits.storage_quota_mb (user > team > org); None is unlimited."""
    uid = quota_checks.as_quota_user_id(user_id)
    if uid is None:
        return None
    value = await quota_resolver.user_quota(uid, STORAGE_QUOTA_KEY)
    return None if value is None else int(value)


async def with_resolved_storage_quota(user: dict[str, Any]) -> dict[str, Any]:
    """A copy of a users row whose storage_quota_mb is the enforced value, not the legacy column."""
    row = dict(user)
    row["storage_quota_mb"] = await resolved_storage_quota_mb(row.get("id"))
    return row
```

Check for an import cycle with `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -c "import tldw_Server_API.app.services.storage_quota_service; import tldw_Server_API.app.main"`. If there is one, import the *module* (`from tldw_Server_API.app.core.UserProfiles import overrides_repo`), use `overrides_repo.UserProfileOverridesRepo`, and have the test patch `overrides_repo.UserProfileOverridesRepo` instead. Record which you chose in the report.

- [ ] **Step 4: Cut the service methods over**

**`check_quota`.** `quota_cache` now holds usage only:

```python
        cache_key = f"quota:{user_id}"
        current_mb = self.quota_cache.get(cache_key)
        if current_mb is None:
            user_info = await self._get_user_storage_info(user_id)
            if not user_info:
                raise UserNotFoundError(f"User {user_id}")
            current_mb = float(user_info["storage_used_mb"])
            self.quota_cache[cache_key] = current_mb
        quota_mb = await resolved_storage_quota_mb(user_id)
```

Then:
- In the gauge block, emit `user_storage_quota_mb` only when `quota_mb is not None`.
- Set `has_quota = quota_mb is None or projected_mb <= quota_mb`. The resolver returns None when quotas are off, so drop the `or not usage_quotas_enabled()`.
- Build `quota_info` as `{"user_id", "current_usage_mb", "new_size_mb", "projected_usage_mb", "has_quota", **quota_view(current_mb, quota_mb)}`, keeping the existing rounding for the first four.

**`check_combined_quota`.** The user level no longer needs the switch, but the shared pools still do:

```python
        has_quota = has_user_quota and (
            (has_team_quota and has_org_quota) or not usage_quotas_enabled()
        )
```

Also pass `user_info.get("quota_mb") or 0` to `QuotaExceededError`, since the user quota is None when a pool blocks.

**`calculate_user_storage`.** `storage_cache` keeps the usage figures. Merge the quota in on every return:
- Keep the `_get_user_storage_info` existence check (it raises `UserNotFoundError`).
- Build `result` without the quota keys and store it in `self.storage_cache`.
- Return `{**result, **quota_view(total_mb, await resolved_storage_quota_mb(user_id))}`.
- On a cache hit, return `{**cached, **quota_view(cached["total_mb"], await resolved_storage_quota_mb(user_id))}`.
- Keep the info log; `{quota_mb}` may print `None`.

**`update_usage`.**
- Keep the transaction that updates and reads `storage_used_mb`.
- Move the quota read AFTER the `async with ... transaction()` block, so no override query runs inside the write transaction: `quota = await resolved_storage_quota_mb(user_id)`.
- Warn only when `quota is not None and new_usage > quota`.
- Emit the quota gauge only when it is not None.
- Return `{"user_id", "storage_used_mb": round(new_usage, 2), "storage_quota_mb": quota, "available_mb": ..., "usage_percentage": ...}`, taking the last two from `quota_view(new_usage, quota)`.

**`get_storage_breakdown`.** Set `"quota_mb": await resolved_storage_quota_mb(user_id)`.

**`get_all_users_storage`.**
- Per row, `quota = await resolved_storage_quota_mb(user.get("id"))`, then `{"user_id", "username", "storage_used_mb", "storage_quota_mb": quota, **{k: v for k, v in quota_view(used, quota).items() if k != "quota_mb"}}`.
- Above the loop, add `# ponytail: one resolver lookup per user (60 s cached); batch by user ids if admin lists grow past a few thousand`.

**`get_user_generated_files_usage`.** Set `usage["quota_mb"] = await resolved_storage_quota_mb(user_id)` and keep `quota_used_mb` from the row.

**`set_user_quota`.** It now writes the user override:

```python
    async def set_user_quota(
        self, user_id: int, quota_mb: Optional[int], *, updated_by: Optional[int] = None
    ) -> dict[str, Any]:
        """Set (or, with None, remove) the user's own limits.storage_quota_mb override (spec 2 §5)."""
        if not self._initialized:
            await self.initialize()
        if quota_mb is not None and int(quota_mb) < 0:
            raise ValueError("quota_mb must be >= 0, or None to remove the user's value")
        user_info = await self._get_user_storage_info(user_id)
        if not user_info:
            raise UserNotFoundError(f"User {user_id}")
        repo = UserProfileOverridesRepo(self.db_pool)
        try:
            await repo.ensure_tables()
            if quota_mb is None:
                await repo.delete_override(user_id=int(user_id), key=STORAGE_QUOTA_KEY)
            else:
                await repo.upsert_override(
                    user_id=int(user_id), key=STORAGE_QUOTA_KEY, value=int(quota_mb), updated_by=updated_by
                )
        except Exception as e:  # noqa: BLE001 - re-raised as the service's StorageError contract
            logger.error(f"Failed to set user quota: {e}")
            raise StorageError(f"Failed to set quota: {e}") from e
        quota_resolver.invalidate_user(int(user_id))
        effective = await resolved_storage_quota_mb(user_id)
        used = float(user_info["storage_used_mb"])
        logger.info(f"Set storage quota for user {user_id}: {quota_mb}MB (effective {effective})")
        view = quota_view(used, effective)
        return {
            "user_id": user_id,
            "storage_quota_mb": effective,
            "storage_used_mb": round(used, 2),
            "available_mb": view["available_mb"],
            "usage_percentage": view["usage_percentage"],
        }
```

If the repo's BLE001 baseline rejects that `noqa` (run `tests/lint`), catch `(RuntimeError, ValueError, OSError, sqlite3.Error)` plus the asyncpg base error the module already handles. Match whatever narrower tuple the file uses elsewhere.

- [ ] **Step 5: Run the new tests and confirm they PASS, then fix the pinning tests**

Run the Step 2 command; expect PASS. Then run:

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 \
  tldw_Server_API/tests/Storage tldw_Server_API/tests/Services/test_storage_quota_service.py \
  tldw_Server_API/tests/Usage tldw_Server_API/tests/AuthNZ/unit/test_storage_quota_service_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_versioned_user_write_gateway.py \
  tldw_Server_API/tests/DB_Management/unit/test_users_db_update_backend_detection.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit tldw_Server_API/tests/VN_Assets
```

Rewrite each failure that pins the OLD contract to the new one. The old contract covers four things:
- the quota read from the column;
- a `(current, quota)` tuple in `quota_cache`;
- `set_user_quota` writing `UPDATE users SET storage_quota_mb`, or clamping to 100;
- `or not usage_quotas_enabled()` on the user level.

Keep every assertion about live behavior. Tests that only stub `check_quota(self, user_id, new_bytes, raise_on_exceed=False)` keep working unchanged. List every rewritten test in the report.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/services/storage_quota_service.py tldw_Server_API/tests/Usage/test_storage_quota_resolver.py <each rewritten test file>
git commit -m "feat(quotas): storage quota comes from limits.storage_quota_mb via the resolver; null is unlimited (spec 2 §5)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Profile, admin-user and registration writers; team and org values allowed

**Files:**
- Modify: `tldw_Server_API/app/core/UserProfiles/update_service.py`. Change line ~150 (null-delete condition), delete the storage special case at ~299-321, and update the comment at ~324.
- Modify: `tldw_Server_API/app/services/admin_users_service.py`. Change `update_user` (the storage block at ~421-425 and the "No fields to update" check).
- Modify: `tldw_Server_API/app/services/registration_service.py`. Change `register_user`: the override at ~378-381, the dead registration-code branch at ~392-394, the INSERT, and the returned dicts at ~429 and ~501.
- Modify: `tldw_Server_API/app/api/v1/schemas/admin_schemas.py:74,148`. Change `UserUpdateRequest` / `AdminUserCreateRequest.storage_quota_mb` from `ge=100` to `ge=0`.
- Modify: `tldw_Server_API/app/services/admin_profiles_service.py`. Remove the storage special cases at ~212, ~239-245, ~306 and ~1086.
- Test: create `tldw_Server_API/tests/UserProfile/_storage_quota_helpers.py` and `tldw_Server_API/tests/UserProfile/test_storage_quota_writers.py`. Update the tests that pin the old write path:
  - `tests/UserProfile/test_user_profile_updates.py` (incl. 129-154), `test_user_profile_bulk.py`, `test_profile_bulk_command_service.py`, `test_profile_command_service.py`
  - `tests/UserProfile/test_user_profile_admin_audit.py`, `test_user_profile_legacy_contract_characterization.py`, `test_stage2_caller_characterization.py`, `test_group_limit_overrides.py:107`
  - `tests/Admin/test_admin_user_api.py`, `tests/AuthNZ/integration/test_registration_role_membership_postgres.py`

**Interfaces:**
- Consumes (Task 1): `StorageQuotaService.set_user_quota(user_id, quota_mb, *, updated_by=None)`, `get_storage_service()`, `resolved_storage_quota_mb(user_id)` and `STORAGE_QUOTA_KEY`.
- Produces: `limits.storage_quota_mb` writes on every path land in `user_config_overrides`. A `null` deletes. Team and org override routes accept the key.

- [ ] **Step 1: Write the failing tests**

These follow `tests/UserProfile/test_group_limit_overrides.py`: the root-conftest `auth_headers` fixture (single-user admin), `TestClient(app)`, and the session-scoped AuthNZ SQLite DB. That DB is shared per xdist worker, so every test removes what it writes in a `finally` block. A leaked override silently changes other tests (PR B lesson).

Create `tldw_Server_API/tests/UserProfile/_storage_quota_helpers.py`. It is a plain module, not a test file, and Task 4 imports it too:

```python
"""Shared helpers for the limits.storage_quota_mb writer/reader tests (spec 2 §5)."""

import asyncio
import uuid

from tldw_Server_API.app.core.AuthNZ.orgs_teams import add_org_member, add_team_member, create_organization, create_team
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.services.storage_quota_service import resolved_storage_quota_mb

KEY = "limits.storage_quota_mb"


def quota(user_id: int) -> object:
    """The user's enforced storage quota, read past the resolver cache."""
    quota_resolver.invalidate_all()
    return asyncio.run(resolved_storage_quota_mb(user_id))


def me(client, headers: dict) -> int:
    """The authenticated user's id."""
    return int(client.get("/api/v1/users/me/profile", headers=headers).json()["user"]["id"])


def patch_quota(client, headers: dict, user_id: int, value: object):
    """Set (or with None clear) the user's own limits.storage_quota_mb through the admin profile route."""
    return client.patch(
        f"/api/v1/admin/users/{user_id}/profile",
        headers=headers,
        json={"updates": [{"key": KEY, "value": value}]},
    )


def org_and_team_with_member(user_id: int) -> tuple[int, int]:
    """A fresh org and a team inside it, with the user a member of both."""
    suffix = uuid.uuid4().hex[:8]

    async def _go() -> tuple[int, int]:
        """Create and join."""
        org = await create_organization(name=f"Storage Org {suffix}", owner_user_id=None)
        team = await create_team(org_id=int(org["id"]), name=f"Storage Team {suffix}")
        await add_org_member(org_id=int(org["id"]), user_id=user_id)
        await add_team_member(team_id=int(team["id"]), user_id=user_id)
        return int(org["id"]), int(team["id"])

    return asyncio.run(_go())
```

`tldw_Server_API/tests/UserProfile/test_storage_quota_writers.py`:

```python
"""Every storage quota writer lands in limits.storage_quota_mb (spec 2 §5)."""

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.main import app
from tldw_Server_API.tests.UserProfile._storage_quota_helpers import (
    KEY,
    me,
    org_and_team_with_member,
    patch_quota,
    quota,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, resolver cache empty."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()


def test_profile_patch_writes_override_and_null_deletes(auth_headers: dict) -> None:
    """PATCH limits.storage_quota_mb=300 enforces 300; null returns the user to unlimited."""
    with TestClient(app) as client:
        user_id = me(client, auth_headers)
        try:
            assert patch_quota(client, auth_headers, user_id, 300).status_code == 200
            assert quota(user_id) == 300
            assert patch_quota(client, auth_headers, user_id, None).status_code == 200
            assert quota(user_id) is None
        finally:
            patch_quota(client, auth_headers, user_id, None)


def test_admin_update_quota_absent_vs_null(auth_headers: dict) -> None:
    """An admin user update without storage_quota_mb keeps the quota; explicit null clears it; 0 is accepted."""
    with TestClient(app) as client:
        user_id = me(client, auth_headers)
        url = f"/api/v1/admin/users/{user_id}"
        try:
            assert client.put(url, headers=auth_headers, json={"storage_quota_mb": 0, "reason": "quota test set"}).status_code == 200
            assert quota(user_id) == 0
            assert client.put(url, headers=auth_headers, json={"is_verified": True, "reason": "unrelated change"}).status_code == 200
            assert quota(user_id) == 0
            assert client.put(url, headers=auth_headers, json={"storage_quota_mb": None, "reason": "quota test clear"}).status_code == 200
            assert quota(user_id) is None
        finally:
            patch_quota(client, auth_headers, user_id, None)


def test_team_storage_value_enforced_and_user_value_wins(auth_headers: dict) -> None:
    """A team limits.storage_quota_mb of 200 applies to a member; their own 500 overrides it."""
    with TestClient(app) as client:
        user_id = me(client, auth_headers)
        _org_id, team_id = org_and_team_with_member(user_id)
        team_url = f"/api/v1/admin/teams/{team_id}/profile/overrides/{KEY}"
        try:
            assert client.put(team_url, headers=auth_headers, json={"value": 200}).status_code == 200
            assert quota(user_id) == 200
            assert patch_quota(client, auth_headers, user_id, 500).status_code == 200
            assert quota(user_id) == 500
        finally:
            patch_quota(client, auth_headers, user_id, None)
            client.delete(team_url, headers=auth_headers)
```

If `PUT /api/v1/admin/users/{id}` refuses an admin editing themselves, or needs extra fields (`reason`, `admin_password`), satisfy the route's own rules. Read `UserUpdateRequest` and the route. If self-edits are refused outright, edit a second user created as in the next test.

Also add `test_admin_create_with_quota_writes_override`, with the docstring "Creating a user with storage_quota_mb=250 yields an enforced 250; without it, unlimited." The admin create route refuses user creation in the local single-user test profile ("User creation is not allowed in local-single-user profile"). So call `registration_service.register_user(...)` directly, exactly as `admin_users_service.create_user` does (same keyword arguments, including `created_by` and `storage_quota_override=250`), then assert `quota(new_id) == 250`. A second call without the override gives `quota(other_id) is None`.

In `tests/UserProfile/test_group_limit_overrides.py`, `test_group_override_rejects_bad_input` asserts that `limits.storage_quota_mb` is refused with 400. Delete that one line, because storage is now accepted; the team test above covers acceptance.

- [ ] **Step 2: Run them and confirm they FAIL**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/UserProfile/test_storage_quota_writers.py -q -p no:xdist`

Expected: FAIL. The profile write lands in the column, so the resolver says None; the team PUT returns 400; and a null update is a no-op.

- [ ] **Step 3: The profile path uses the generic branch**

In `update_service.py`:
- Line ~150: change the null-delete condition to `if key.startswith("preferences.") or key.startswith("limits."):`.
- Delete the whole `if key == "limits.storage_quota_mb":` block (~299-321).
- Change the generic-branch comment to `# Every limits.* key is a plain user override the quota resolver reads (spec 2 §3).`

The generic branch already sets `result.needs_quota_invalidation = True`, which clears the resolver cache after commit.

- [ ] **Step 4: Admin user update writes the override**

In `admin_users_service.update_user`:
- Replace the `if request.storage_quota_mb is not None:` column block with `quota_requested = "storage_quota_mb" in request.model_fields_set`.
- Change the "No fields to update" guard to `if not updates and not quota_requested:`.
- Make the users-table UPDATE (and whatever builds and executes it) run only `if updates:`. Read the function and keep the role-sync, audit and gateway behavior unchanged for the other fields.
- After the transaction commits, add:

```python
        if quota_requested:
            storage_service = await get_storage_service()
            await storage_service.set_user_quota(
                user_id,
                request.storage_quota_mb,
                updated_by=int(principal.user_id) if principal.user_id is not None else None,
            )
```

Import `get_storage_service` from `tldw_Server_API.app.services.storage_quota_service`. If the function's audit record lists the changed fields, keep `storage_quota_mb` in it when `quota_requested`. In `admin_schemas.py`, change `UserUpdateRequest.storage_quota_mb` and `AdminUserCreateRequest.storage_quota_mb` to `int | None = Field(None, ge=0)`.

- [ ] **Step 5: Registration writes the override**

In `registration_service.register_user`:
- The column always gets `self.settings.DEFAULT_STORAGE_QUOTA_MB`, which is no longer read. Keep the variable `storage_quota` for the INSERT, but stop assigning the override to it.
- Keep the `< 0` validation of `storage_quota_override`.
- Delete the dead `if 'storage_quota_mb' in code_info:` branch and its comment.
- After the user INSERT, inside the same transaction, where the new user id is known:

```python
                if privileged_creation and storage_quota_override is not None:
                    await UserProfileOverridesRepo(self.db_pool).upsert_override(
                        user_id=int(new_user_id),
                        key="limits.storage_quota_mb",
                        value=int(storage_quota_override),
                        updated_by=created_by,
                        db_conn=conn,
                    )
```

Use the function's real variable for the new id and its real pool attribute; read the code. Both returned dicts (`"storage_quota_mb": storage_quota`) become `"storage_quota_mb": int(storage_quota_override) if privileged_creation and storage_quota_override is not None else None`.

- [ ] **Step 6: Team and org values allowed; audit diff uses the effective config**

In `admin_profiles_service.py`:
- At ~1086, `set_group_limit_override`: drop `or key == "limits.storage_quota_mb"`.
- At ~212: `needs_user_record = bool(key_set & identity_keys)`.
- Delete the `if user_row and "limits.storage_quota_mb" in key_set:` block (~239-245).
- At ~306, drop `and key != "limits.storage_quota_mb"`, so the "before" value comes from `_build_effective_config` like every other `limits.*` key.

- [ ] **Step 7: Run the tests and confirm they PASS; fix the pinning tests**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 \
  tldw_Server_API/tests/UserProfile tldw_Server_API/tests/Admin tldw_Server_API/tests/AuthNZ/unit \
  tldw_Server_API/tests/AuthNZ_SQLite tldw_Server_API/tests/Usage
```

Rewrite the tests that assert a `limits.storage_quota_mb` write landed on `users.storage_quota_mb`, or that `set_user_quota` was the profile path's "before" baseline, or that a team/org storage PUT returns 400. They must now assert the override (via `resolved_storage_quota_mb`, or `UserProfileOverridesRepo.list_overrides_for_user`). `test_group_limit_overrides.py:107` flips to expect 200 and an enforced value. Also run the Postgres registration test if Postgres is available: `tests/AuthNZ/integration/test_registration_role_membership_postgres.py`. List every rewritten test in the report.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/UserProfiles/update_service.py tldw_Server_API/app/services/admin_users_service.py \
  tldw_Server_API/app/services/registration_service.py tldw_Server_API/app/api/v1/schemas/admin_schemas.py \
  tldw_Server_API/app/services/admin_profiles_service.py tldw_Server_API/tests/UserProfile/_storage_quota_helpers.py tldw_Server_API/tests/UserProfile/test_storage_quota_writers.py \
  <each rewritten test file> apps/tldw-frontend/lib/api/openapi.fingerprint.json
git commit -m "feat(quotas): storage quota writers write limits.storage_quota_mb; null clears; team/org values allowed (spec 2 §5)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Refresh the OpenAPI fingerprint first: the `ge=0` change alters the contract.

---

### Task 3: Storage admin endpoints

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/storage_schemas.py`. Add `SetUserQuotaRequest`; `SetQuotaRequest` stays for team/org pools.
- Modify: `tldw_Server_API/app/api/v1/endpoints/storage_admin_quotas.py:39-66`. Change `set_user_quota`.
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_storage_quotas.py`. Change `get_user_storage_quota` (~109-132) and `update_user_storage_quota` (~135-162), and the local `StorageQuotaResponse` (`usage_pct: float | None = 0.0`).
- Test: create `tldw_Server_API/tests/Storage/test_user_quota_endpoints.py`. Update `tests/Storage/test_storage_endpoints.py:981,1000-1004`, `tests/Admin/test_admin_storage_quotas.py:333-367` and `tests/AuthNZ_Unit/test_storage_admin_claims.py`.

**Interfaces:**
- Consumes (Task 1): `StorageQuotaService.set_user_quota`, `StorageQuotaService.check_quota(user_id, 0)` (for the status read), `get_storage_service()`.
- Produces: `SetUserQuotaRequest(quota_mb: int | None = Field(..., ge=0), soft_limit_pct: int = 80, hard_limit_pct: int = 100)`.

- [ ] **Step 1: Write the failing tests**

These reuse Task 2's helpers and the root-conftest `auth_headers` (single-user admin), and remove what they write in `finally`:

```python
"""Both per-user storage quota admin endpoints write and read limits.storage_quota_mb (spec 2 §5)."""

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import AuthnzStorageQuotasRepo
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.main import app
from tldw_Server_API.tests.UserProfile._storage_quota_helpers import me, patch_quota, quota

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, resolver cache empty."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()


def test_storage_admin_put_accepts_zero_and_null(auth_headers: dict) -> None:
    """PUT /storage/admin/quotas/user/{id} accepts 0 (blocks) and null (unlimited); 404 missing user; 422 negative."""
    with TestClient(app) as client:
        user_id = me(client, auth_headers)
        url = f"/api/v1/storage/admin/quotas/user/{user_id}"
        try:
            resp = client.put(url, headers=auth_headers, json={"quota_mb": 0})
            assert resp.status_code == 200 and resp.json()["quota"]["quota_mb"] == 0
            assert quota(user_id) == 0
            resp = client.put(url, headers=auth_headers, json={"quota_mb": None})
            assert resp.status_code == 200 and resp.json()["quota"]["quota_mb"] is None
            assert quota(user_id) is None
            assert client.put(url, headers=auth_headers, json={"quota_mb": -1}).status_code == 422
            missing = client.put("/api/v1/storage/admin/quotas/user/987654321", headers=auth_headers, json={"quota_mb": 5})
            assert missing.status_code == 404
        finally:
            patch_quota(client, auth_headers, user_id, None)


def test_admin_storage_quotas_user_routes_use_the_user_not_an_org(auth_headers: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """PUT then GET /admin/storage-quotas/users/{id} set and read the user's quota and never touch an org pool."""
    org_writes: list[tuple] = []

    async def _record(self, *args, **kwargs) -> dict:
        """Record any org-pool write."""
        org_writes.append((args, kwargs))
        return {}

    monkeypatch.setattr(AuthnzStorageQuotasRepo, "upsert_org_quota", _record)
    with TestClient(app) as client:
        user_id = me(client, auth_headers)
        url = f"/api/v1/admin/storage-quotas/users/{user_id}"
        try:
            assert client.put(url, headers=auth_headers, json={"quota_mb": 400}).status_code == 200
            assert client.get(url, headers=auth_headers).json()["quota_mb"] == 400
            assert client.put(url, headers=auth_headers, json={"quota_mb": None}).status_code == 200
            assert client.get(url, headers=auth_headers).json()["quota_mb"] is None
            assert org_writes == []
        finally:
            patch_quota(client, auth_headers, user_id, None)
```

- [ ] **Step 2: Run them and confirm they FAIL**

`ge=100` rejects 0 and null (422), and the admin routes call `upsert_org_quota`.

- [ ] **Step 3: Implement**

In `storage_schemas.py`:

```python
class SetUserQuotaRequest(BaseModel):
    """Set a user's own storage quota (MB); null removes it (unlimited unless a team/org value applies)."""
    quota_mb: int | None = Field(..., ge=0, description="Quota in MB; 0 blocks uploads; null removes the user's value")
    soft_limit_pct: int = Field(default=80, ge=0, le=100, description="Soft limit percentage")
    hard_limit_pct: int = Field(default=100, ge=0, le=100, description="Hard limit percentage")
```

In `storage_admin_quotas.set_user_quota`:
- Take `request: SetUserQuotaRequest`.
- Call `result = await service.set_user_quota(user_id, request.quota_mb, updated_by=<principal user id or None>)`. The principal dependency is currently `_principal`; rename it and use it.
- Build the status from `pct = result.get("usage_percentage")`:

```python
        status_data = {
            "quota_mb": result.get("storage_quota_mb"),
            "used_mb": result.get("storage_used_mb", 0.0),
            "remaining_mb": result.get("available_mb"),
            "usage_pct": pct if pct is not None else 0.0,
            "at_soft_limit": pct is not None and pct >= request.soft_limit_pct,
            "at_hard_limit": pct is not None and pct >= request.hard_limit_pct,
            "has_quota": True,
        }
```

- Keep the 404 and 500 mapping. Also map `ValueError` to 422.

In `admin_storage_quotas.py`:
- Add `UpdateUserQuotaRequest(BaseModel)` with `quota_mb: int | None = Field(..., ge=0)`.
- `get_user_storage_quota`: `has_quota, info = await (await get_storage_service()).check_quota(user_id, 0)`, then return `StorageQuotaResponse(quota_mb=info["quota_mb"], used_mb=info["current_usage_mb"], remaining_mb=info["available_mb"], usage_pct=info["usage_percentage"], has_quota=has_quota)`. Map `UserNotFoundError` to 404.
- `update_user_storage_quota(user_id, body: UpdateUserQuotaRequest)`: call `set_user_quota(user_id, body.quota_mb)` and return its dict. Map `UserNotFoundError` to 404 and `ValueError` to 422. Keep the sanitized 500 for the `_NONCRITICAL_EXCEPTIONS`.
- Fix both docstrings (the user is no longer treated as an org), and drop the misleading "Currently delegates to the org-level quota" text.
- Change the local `StorageQuotaResponse.usage_pct` to `float | None = 0.0`.

- [ ] **Step 4: Run the tests and confirm they PASS; fix the pinning tests**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 \
  tldw_Server_API/tests/Storage tldw_Server_API/tests/Admin tldw_Server_API/tests/AuthNZ_Unit
```

The error-mapping tests in `test_admin_storage_quotas.py:333-367` now patch the storage service instead of the repo; keep their sanitized-500 assertions.

- [ ] **Step 5: Refresh the OpenAPI fingerprint, then commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/export_openapi_schema.py --fingerprint apps/tldw-frontend/lib/api/openapi.fingerprint.json
git add tldw_Server_API/app/api/v1/schemas/storage_schemas.py tldw_Server_API/app/api/v1/endpoints/storage_admin_quotas.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_storage_quotas.py tldw_Server_API/tests/Storage/test_user_quota_endpoints.py \
  <each rewritten test file> apps/tldw-frontend/lib/api/openapi.fingerprint.json
git commit -m "fix(storage): per-user quota admin endpoints write limits.storage_quota_mb; /admin/storage-quotas/users/{id} no longer treats the user id as an org id

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Readers show the enforced value; null is unlimited

**Files:**
- Modify `tldw_Server_API/app/api/v1/endpoints/users.py`:
  - `_resolve_user_context` fallback `"storage_quota_mb": 5120` (~267), overlaying the loaded row;
  - `/me` (~573, ~649);
  - `GET /storage` fallback (~1047-1056).
- Modify `tldw_Server_API/app/api/v1/endpoints/auth.py:~3991` (`storage_quota_mb=_current_user_value(..., 1000)`).
- Modify `tldw_Server_API/app/services/admin_users_service.py`: `list_users`, `export_users` and `get_user_details` overlay the resolved value.
- Modify `tldw_Server_API/app/services/admin_system_service.py:~178,~215,~297`: drop `SUM(storage_quota_mb)`, so `total_quota_mb` is None.
- Modify `tldw_Server_API/app/core/UserProfiles/service.py:~439-460` (`_build_quotas`).
- Modify the schemas:
  - `auth_schemas.py:277` `UserResponse.storage_quota_mb: Optional[int]`;
  - `auth_schemas.py:~417-422` `StorageQuotaResponse.storage_quota_mb/available_mb/usage_percentage`: Optional;
  - `admin_schemas.py:179` `UserSummary.storage_quota_mb: int | None`;
  - `admin_schemas.py:~454` `StorageStats.total_quota_mb: float | None`;
  - `user_profile_schemas.py:109` `UserProfileQuotas.storage_quota_mb: Optional[int]`;
  - `storage_schemas.py` `UsageBreakdownResponse.quota_mb/available_mb/usage_percentage`: Optional.
- Modify `tldw_Server_API/app/api/v1/endpoints/storage_usage.py:~60-125` so it is None-safe when building `UsageBreakdownResponse` and `StorageUsageResponse`.
- Modify the frontend types:
  - `apps/packages/ui/src/services/tldw/TldwApiClient.ts:338,1477`: `storage_quota_mb: number | null`;
  - `:1495`: `storage_quota_mb?: number | null`;
  - `apps/tldw-frontend/lib/auth.ts:27`: `storage_quota_mb?: number | null`;
  - the stats type behind `ServerAdminPage.tsx:665` (`total_quota_mb`): `number | null`.
- Test: create `tldw_Server_API/tests/UserProfile/test_storage_quota_readers.py`. Update:
  - `tests/AuthNZ/integration/test_auth_comprehensive.py:411`
  - `tests/AuthNZ/integration/test_user_endpoints_integration.py:298`
  - `tests/AuthNZ/unit/test_user_endpoints.py:350,384`
  - `tests/UserProfile/test_user_profile_service_live_storage_quota.py`
  - `tests/UserProfile/test_user_profile_read.py`
  - any admin-stats test that pins `total_quota_mb`

**Interfaces:**
- Consumes (Task 1): `with_resolved_storage_quota(user: dict) -> dict`, `resolved_storage_quota_mb(user_id) -> int | None`, `quota_view(used_mb, quota_mb)`.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/UserProfile/test_storage_quota_readers.py` reuses Task 2's helpers:

```python
"""Every reader shows the enforced storage quota; unlimited is null (spec 2 §5)."""

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.main import app
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService
from tldw_Server_API.tests.UserProfile._storage_quota_helpers import me, patch_quota

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, resolver cache empty."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()


def _assert_unlimited(body: dict) -> None:
    """The storage response says unlimited (null), not 0 or 5120."""
    assert body["storage_quota_mb"] is None
    assert body["available_mb"] is None
    assert body["usage_percentage"] is None


def test_users_storage_unlimited_is_null(auth_headers: dict) -> None:
    """GET /users/storage reports null quota fields when no limits.storage_quota_mb applies."""
    with TestClient(app) as client:
        _assert_unlimited(client.get("/api/v1/users/storage", headers=auth_headers).json())


def test_users_storage_fallback_unlimited_is_null(auth_headers: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the live calculation fails, the fallback path also reports null, not 5120."""

    async def _fail(self, *args, **kwargs):
        """A calculation failure the endpoint catches."""
        raise RuntimeError("calculation failed")

    monkeypatch.setattr(StorageQuotaService, "calculate_user_storage", _fail)
    with TestClient(app) as client:
        _assert_unlimited(client.get("/api/v1/users/storage", headers=auth_headers).json())


def test_me_admin_list_and_profile_show_resolved_value(auth_headers: dict) -> None:
    """/users/me, GET /admin/users and the profile quotas section show 250 when set, null when cleared."""
    with TestClient(app) as client:
        user_id = me(client, auth_headers)
        try:
            assert patch_quota(client, auth_headers, user_id, 250).status_code == 200
            assert client.get("/api/v1/users/me", headers=auth_headers).json()["storage_quota_mb"] == 250
            listing = client.get("/api/v1/admin/users", headers=auth_headers, params={"limit": 100}).json()
            row = next(u for u in listing["users"] if int(u["id"]) == user_id)
            assert row["storage_quota_mb"] == 250
            profile = client.get("/api/v1/users/me/profile", headers=auth_headers, params={"sections": "quotas"}).json()
            assert profile["quotas"]["storage_quota_mb"] == 250
            assert patch_quota(client, auth_headers, user_id, None).status_code == 200
            assert client.get("/api/v1/users/me", headers=auth_headers).json()["storage_quota_mb"] is None
            profile = client.get("/api/v1/users/me/profile", headers=auth_headers, params={"sections": "quotas"}).json()
            assert profile["quotas"]["storage_quota_mb"] is None
        finally:
            patch_quota(client, auth_headers, user_id, None)
```

Check these against the real code:
- **`RuntimeError`** must be in `users.py`'s `_USERS_ENDPOINT_EXCEPTIONS`; if not, raise a type that is.
- **`listing["users"]`** must be `UserListResponse`'s actual list field name.

Also add `test_admin_stats_total_quota_is_null`, with the docstring "Admin system stats report total_quota_mb null instead of summing the legacy column." Find the route with `git grep -n "StorageStats" -- tldw_Server_API/app/api` and assert `storage.total_quota_mb is None`.

- [ ] **Step 2: Run them and confirm they FAIL**

Expected: the responses show 5120 (or 1000), and Pydantic rejects None.

- [ ] **Step 3: Implement**

**`users.py`.**
- In `_resolve_user_context`, change the fallback dict's `"storage_quota_mb"` to `None`. After the repo loads the user, return `await with_resolved_storage_quota(dict(user))` in place of the raw row.
- At `/me` (~573, ~649), use `storage_quota_mb=user_context.get("storage_quota_mb")` with no default.
- In the `GET /storage` fallback, use `quota = user_context.get("storage_quota_mb")` and `used = float(user_context.get("storage_used_mb", 0.0))`, then return `StorageQuotaResponse(user_id=..., storage_used_mb=used, storage_quota_mb=quota, available_mb=view["available_mb"], usage_percentage=view["usage_percentage"])`, where `view = quota_view(used, quota)`.

**`auth.py:~3991`.** Use `storage_quota_mb=await resolved_storage_quota_mb(user_id)`. The enclosing function is async; check, and use its user id variable.

**`admin_users_service`.**
- In `list_users` and `export_users`, set `users = [await with_resolved_storage_quota(u) for u in users]` before returning or serializing.
- In `get_user_details`, overlay the single row the same way, before the response is built.

**`admin_system_service`.**
- Delete the `SUM(storage_quota_mb) as total_quota_mb,` line from both queries.
- Keep `"total_quota_mb"` in `storage_keys` only if the row-mapping loop needs it; read the code.
- Set `"total_quota_mb": None` at ~297 and in the zeroed default at ~141.

**`UserProfiles/service._build_quotas`.**
- Seed with `"storage_quota_mb": await resolved_storage_quota_mb(user_id)`.
- When the live `storage_info` is available, set `quotas["storage_quota_mb"] = live_quota` (it may be None). Keep `storage_used_mb` handling as it is.

Also apply the schema edits listed under Files, and make `storage_usage.py` None-safe:
- `usage_percentage` stays `round(... ) if quota_mb else None`.
- `at_soft_limit` / `at_hard_limit` are False when the quota is None.

Then apply the four frontend type edits. Run `cd apps/packages/ui && bunx tsc --noEmit -p .`, or the package's existing typecheck script; check `package.json`. If a consumer fails typecheck on `number | null`, make it treat null as "no quota", as `ServerAdminPage.formatMegabytesForAdmin` already does.

- [ ] **Step 4: Run the tests and confirm they PASS; fix the pinning tests**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 \
  tldw_Server_API/tests/Usage tldw_Server_API/tests/AuthNZ/unit tldw_Server_API/tests/AuthNZ/integration \
  tldw_Server_API/tests/UserProfile tldw_Server_API/tests/Admin tldw_Server_API/tests/Storage
```

The `> 0` and `== 1000` assertions on `storage_quota_mb` become assertions on the resolved value: null by default, or the value the test sets.

- [ ] **Step 5: Refresh the OpenAPI fingerprint, then commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/export_openapi_schema.py --fingerprint apps/tldw-frontend/lib/api/openapi.fingerprint.json
git add <every file listed under Files> tldw_Server_API/tests/UserProfile/test_storage_quota_readers.py <each rewritten test file> apps/tldw-frontend/lib/api/openapi.fingerprint.json
git commit -m "feat(quotas): storage quota readers show the enforced limits.storage_quota_mb; null means unlimited (spec 2 §5)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Migrate existing quotas

**Files:**
- Create: `tldw_Server_API/app/core/AuthNZ/storage_quota_backfill.py`, holding the shared skip-values helper.
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`. Add `migration_100_copy_storage_quotas_to_user_overrides` and register `Migration(100, ...)` after 99 in `get_authnz_migrations()`.
- Modify: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`. Add `ensure_storage_quota_overrides_backfill_pg`.
- Modify: `tldw_Server_API/app/services/startup_auth.py:~101-135` (`_ensure_pg_extras`'s `pg_ensures` list, AFTER "AuthNZ core tables") and `tldw_Server_API/app/core/AuthNZ/initialize.py:~555-575` (after `ensure_authnz_core_tables_pg`).
- Test: create `tldw_Server_API/tests/AuthNZ/unit/test_storage_quota_backfill_migration.py` (SQLite) and `tldw_Server_API/tests/AuthNZ/integration/test_storage_quota_backfill_postgres.py` (PG).

**Interfaces:**
- Produces: `storage_quota_backfill.skip_values() -> list[int]`, the sorted unique `{5120, DEFAULT_STORAGE_QUOTA_MB at call time}`, and `ensure_storage_quota_overrides_backfill_pg(pool) -> bool`.

- [ ] **Step 1: Write the failing tests**

**SQLite.** `tldw_Server_API/tests/AuthNZ/unit/test_storage_quota_backfill_migration.py`, modeled on `test_llm_usage_log_index_migration.py`:

```python
"""Migration 100 copies non-default storage quotas into limits.storage_quota_mb overrides (spec 2 §8)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ import settings as authnz_settings
from tldw_Server_API.app.core.AuthNZ import storage_quota_backfill
from tldw_Server_API.app.core.AuthNZ.migrations import apply_authnz_migrations

pytestmark = pytest.mark.unit
KEY = "limits.storage_quota_mb"


def _add_user(conn: sqlite3.Connection, name: str, quota_mb: int) -> int:
    """Insert a user row with the given legacy column value; return its id."""
    cur = conn.execute(
        "INSERT INTO users (username, email, password_hash, storage_quota_mb) VALUES (?, ?, 'x', ?)",
        (name, f"{name}@example.com", quota_mb),
    )
    return int(cur.lastrowid)


def _overrides(db_path: Path) -> dict[int, object]:
    """user_id -> decoded value of every limits.storage_quota_mb override."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT user_id, value_json FROM user_config_overrides WHERE key = ?", (KEY,)).fetchall()
    return {int(uid): json.loads(val) for uid, val in rows}


def test_skip_values_include_the_configured_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """5120 and the DEFAULT_STORAGE_QUOTA_MB configured now are both treated as 'never set'."""
    monkeypatch.setattr(authnz_settings, "get_settings", lambda: SimpleNamespace(DEFAULT_STORAGE_QUOTA_MB=10240))
    assert storage_quota_backfill.skip_values() == [5120, 10240]


def test_copies_only_non_default_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2048 is copied; 5120 and the configured default (10240) are dropped; an existing override is kept."""
    monkeypatch.setattr(storage_quota_backfill, "skip_values", lambda: [5120, 10240])
    db_path = tmp_path / "authnz.db"
    apply_authnz_migrations(db_path, target_version=99)
    with sqlite3.connect(db_path) as conn:
        u_default = _add_user(conn, "dflt", 5120)
        u_custom = _add_user(conn, "custom", 2048)
        u_configured = _add_user(conn, "configured", 10240)
        u_existing = _add_user(conn, "existing", 3000)
        conn.execute(
            "INSERT INTO user_config_overrides (user_id, key, value_json, created_at, updated_at) "
            "VALUES (?, ?, '999', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (u_existing, KEY),
        )
    apply_authnz_migrations(db_path)
    assert _overrides(db_path) == {u_custom: 2048, u_existing: 999}
    assert u_default not in _overrides(db_path) and u_configured not in _overrides(db_path)


def test_skips_when_users_table_absent(tmp_path: Path) -> None:
    """A synthetic version-99 DB without a users table migrates to 100 without error."""
    db_path = tmp_path / "synthetic.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TIMESTAMP NOT NULL)")
        conn.execute("INSERT INTO schema_migrations VALUES (99, 'synthetic', CURRENT_TIMESTAMP)")
    apply_authnz_migrations(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 100
```

If the `users` table has other `NOT NULL` columns without defaults (check migration 001's DDL), add them to `_add_user`'s INSERT. If `apply_authnz_migrations` lacks a `target_version` argument, build the version-99 state the way the 099 test builds version 98.

**Postgres.** `tldw_Server_API/tests/AuthNZ/integration/test_storage_quota_backfill_postgres.py`. Use the `isolated_test_environment` fixture from `tests/AuthNZ/conftest.py`, which skips when Postgres is unavailable. Read how `test_storage_quotas_bootstrap_postgres.py` in the same directory gets a ready `DatabasePool` from that fixture, and do the same.

```python
"""The Postgres storage-quota backfill copies non-default values exactly once (spec 2 §8)."""

import pytest

from tldw_Server_API.app.core.AuthNZ import storage_quota_backfill
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_storage_quota_overrides_backfill_pg

pytestmark = pytest.mark.integration
KEY = "limits.storage_quota_mb"


async def _add_user(pool, name: str, quota_mb: int) -> int:
    """Insert a user with the given legacy column value; return its id."""
    return int(await pool.fetchval(
        "INSERT INTO users (username, email, password_hash, storage_quota_mb) VALUES ($1, $2, 'x', $3) RETURNING id",
        name, f"{name}@example.com", quota_mb,
    ))


async def _override(pool, user_id: int):
    """The user's limits.storage_quota_mb value_json, or None."""
    return await pool.fetchval("SELECT value_json FROM user_config_overrides WHERE user_id = $1 AND key = $2", user_id, KEY)


async def test_pg_backfill_copies_non_default_values(isolated_test_environment, monkeypatch: pytest.MonkeyPatch) -> None:
    """2048 is copied; 5120 and the configured default are not."""
    monkeypatch.setattr(storage_quota_backfill, "skip_values", lambda: [5120, 10240])
    pool = ...  # the DatabasePool from isolated_test_environment, as in test_storage_quotas_bootstrap_postgres.py
    u_custom = await _add_user(pool, "pgcustom", 2048)
    u_default = await _add_user(pool, "pgdefault", 5120)
    u_configured = await _add_user(pool, "pgconfigured", 10240)
    assert await ensure_storage_quota_overrides_backfill_pg(pool) is True
    assert await _override(pool, u_custom) == "2048"
    assert await _override(pool, u_default) is None and await _override(pool, u_configured) is None


async def test_pg_backfill_runs_once_and_never_resurrects(isolated_test_environment, monkeypatch: pytest.MonkeyPatch) -> None:
    """After the first run, a deleted override is not re-created by a second run."""
    monkeypatch.setattr(storage_quota_backfill, "skip_values", lambda: [5120])
    pool = ...  # as above
    user_id = await _add_user(pool, "pgonce", 2048)
    assert await ensure_storage_quota_overrides_backfill_pg(pool) is True
    await pool.execute("DELETE FROM user_config_overrides WHERE user_id = $1 AND key = $2", user_id, KEY)
    assert await ensure_storage_quota_overrides_backfill_pg(pool) is True
    assert await _override(pool, user_id) is None
```

The only thing left open (`pool = ...`) is fixture plumbing that `test_storage_quotas_bootstrap_postgres.py` already shows; copy it exactly. Add other `NOT NULL` `users` columns if the PG DDL requires them, and make sure the AuthNZ core tables are ensured first, as that test does.

- [ ] **Step 2: Run them and confirm they FAIL** (no migration 100, no PG function).

- [ ] **Step 3: Implement the shared helper**

`core/AuthNZ/storage_quota_backfill.py`:

```python
"""Shared values for the users.storage_quota_mb -> limits.storage_quota_mb backfill (spec 2 §8)."""

LEGACY_DEFAULT_STORAGE_QUOTA_MB = 5120
STORAGE_QUOTA_KEY = "limits.storage_quota_mb"
PG_BACKFILL_MARKER = "storage_quota_mb_to_user_overrides_v1"


def skip_values() -> list[int]:
    """Column values that can't be told apart from "never set": 5120 and the configured default."""
    values = {LEGACY_DEFAULT_STORAGE_QUOTA_MB}
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        values.add(int(get_settings().DEFAULT_STORAGE_QUOTA_MB))
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
        pass
    return sorted(values)
```

- [ ] **Step 4: Implement SQLite migration 100**

```python
def migration_100_copy_storage_quotas_to_user_overrides(conn: sqlite3.Connection) -> None:
    """Copy users.storage_quota_mb into limits.storage_quota_mb user overrides (spec 2 §8).

    Values equal to 5120 or to the configured DEFAULT_STORAGE_QUOTA_MB can't be told
    apart from "never set", so they are dropped. An existing override is never replaced.
    """
    from tldw_Server_API.app.core.AuthNZ.storage_quota_backfill import STORAGE_QUOTA_KEY, skip_values

    if not _sqlite_table_exists(conn, "users"):
        logger.info("Migration 100: users table not present; skipping storage quota copy")
        return
    if not _sqlite_table_exists(conn, "user_config_overrides"):
        migration_047_create_user_config_overrides_table(conn)
    skip = skip_values()
    now = datetime.now(timezone.utc).isoformat()
    placeholders = ",".join("?" for _ in skip)
    cur = conn.execute(
        f"""
        INSERT INTO user_config_overrides (user_id, key, value_json, created_at, updated_at, created_by, updated_by)
        SELECT id, ?, CAST(storage_quota_mb AS TEXT), ?, ?, NULL, NULL
        FROM users
        WHERE storage_quota_mb IS NOT NULL AND storage_quota_mb NOT IN ({placeholders})
        ON CONFLICT(user_id, key) DO NOTHING
        """,  # nosec B608 - only "?" placeholders are interpolated
        (STORAGE_QUOTA_KEY, now, now, *skip),
    )
    logger.info("Migration 100: copied {} storage quota(s) to limits.storage_quota_mb", cur.rowcount)
```

Check that `datetime` and `timezone` are imported in `migrations.py`. Also check the column list against migration 047's DDL; if 047 has no `created_by`, drop it. Register `Migration(100, "Copy users.storage_quota_mb into limits.storage_quota_mb overrides", migration_100_copy_storage_quotas_to_user_overrides)`. The runner owns the transaction, as it does for 099, so don't commit here.

- [ ] **Step 5: Implement the one-time Postgres backfill**

In `pg_migrations_extra.py`, follow the error-handling and logging pattern of `ensure_usage_tables_pg`:

```python
async def ensure_storage_quota_overrides_backfill_pg(pool: DatabasePool) -> bool:
    """Copy users.storage_quota_mb into limits.storage_quota_mb overrides once (spec 2 §8).

    A marker row claimed in the same transaction makes it run once, so an override an
    admin later deletes is never re-created from the stale column.
    """
    from tldw_Server_API.app.core.AuthNZ.storage_quota_backfill import (
        PG_BACKFILL_MARKER,
        STORAGE_QUOTA_KEY,
        skip_values,
    )

    try:
        async with pool.transaction() as conn:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS authnz_data_backfills ("
                "name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
            )
            claimed = await conn.fetchval(
                "INSERT INTO authnz_data_backfills (name) VALUES ($1) "
                "ON CONFLICT (name) DO NOTHING RETURNING name",
                PG_BACKFILL_MARKER,
            )
            if claimed is None:
                return True
            await conn.execute(
                """
                INSERT INTO user_config_overrides (user_id, key, value_json, created_at, updated_at)
                SELECT id, $1, storage_quota_mb::text, NOW(), NOW()
                FROM users
                WHERE storage_quota_mb IS NOT NULL AND NOT (storage_quota_mb = ANY($2::int[]))
                ON CONFLICT (user_id, key) DO NOTHING
                """,
                STORAGE_QUOTA_KEY,
                skip_values(),
            )
        return True
    except Exception as exc:  # noqa: BLE001 - mirror the sibling ensure_*_pg contract (log, return False)
        logger.error("Postgres storage quota backfill failed: {}", exc)
        return False
```

Match how the sibling ensure functions get a connection and catch errors. If they narrow the exception type, narrow this one the same way. Wire it into `_ensure_pg_extras`'s `pg_ensures` list right after "AuthNZ core tables", using the same tuple shape: `("storage quota overrides backfill", ensure_storage_quota_overrides_backfill_pg, "AUTHNZ_STORAGE_QUOTA_BACKFILL_NOT_READY")`. In `initialize.py`, call it after `ensure_authnz_core_tables_pg`, with the same failure handling as its neighbors.

- [ ] **Step 6: Run the tests and confirm they PASS**, plus the existing migration suites:

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 \
  tldw_Server_API/tests/AuthNZ/unit tldw_Server_API/tests/AuthNZ_SQLite \
  tldw_Server_API/tests/AuthNZ/integration/test_storage_quota_backfill_postgres.py
```

If Postgres is reachable locally (without `TLDW_TEST_NO_DOCKER`, the fixture auto-starts Docker), run the PG test for real once and record the result; otherwise record the skip.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/storage_quota_backfill.py tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py tldw_Server_API/app/services/startup_auth.py \
  tldw_Server_API/app/core/AuthNZ/initialize.py tldw_Server_API/tests/AuthNZ/unit/test_storage_quota_backfill_migration.py \
  tldw_Server_API/tests/AuthNZ/integration/test_storage_quota_backfill_postgres.py
git commit -m "feat(quotas): migrate existing storage quotas into limits.storage_quota_mb (SQLite migration 100; one-time PG backfill) (spec 2 §8)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Docs, and a sweep of the remaining storage tests

**Files:**
- Modify: `Docs/Operations/Env_Vars.md`. In the Usage Quotas section, replace "`limits.storage_quota_mb` (per user only for now)" with "`limits.storage_quota_mb` (storage MB; team and org values are each member's allowance, separate from team/org shared storage pools)". Add one bullet: "Upgrading copies each user's existing storage quota into `limits.storage_quota_mb`, except 5120 and the `DEFAULT_STORAGE_QUOTA_MB` configured at upgrade time. `DEFAULT_STORAGE_QUOTA_MB` no longer sets anyone's quota." If `DEFAULT_STORAGE_QUOTA_MB` has its own entry elsewhere in the file, change its description to "Legacy; no longer applied as a quota (spec 2). Values equal to it are not migrated."
- Modify: `Docs/Operations/Rate_Limits_Troubleshooting.md`. Add a row: "Storage quota exceeded (413), only when `USAGE_QUOTAS_ENABLED` is on: raise or remove the user's (or their team/org's) `limits.storage_quota_mb`."
- Run `bash Helper_Scripts/refresh_docs_published.sh`, and include the `Docs/Published` diff.

- [ ] **Step 1: Make the doc edits, run the refresh script, and run `tests/Docs`:**

```bash
bash Helper_Scripts/refresh_docs_published.sh
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Docs
```

- [ ] **Step 2: Sweep the tests**

Run every test file that references a changed contract, and fix anything still pinning the old behavior. Keep live-behavior assertions; fixture-only rows that put a value in the `NOT NULL` column need no change.

```bash
git grep -l -e storage_quota_mb -e set_user_quota -e check_quota -e check_combined_quota -e DEFAULT_STORAGE_QUOTA_MB \
  -e calculate_user_storage -e get_all_users_storage -e total_quota_mb -- 'tldw_Server_API/tests/**/test_*.py' | sort \
  > /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad/prc_tests.txt
```

Run that list with `-n 4`. The scratchpad scripts pattern is how to pass the list, because `$(cat ...)` in a raw command is refused by the worktree guard. Record the counts in the report.

- [ ] **Step 3: Commit**

```bash
git add Docs/Operations/Env_Vars.md Docs/Operations/Rate_Limits_Troubleshooting.md Docs/Published <each fixed test file>
git commit -m "docs(quotas): limits.storage_quota_mb is per user/team/org; upgrade copies non-default quotas

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Ship PR C

- [ ] **Step 1: Rebase and normalize the backlog file.** Rebase onto `origin/dev`. Once #3142 (backlog-py cutover) is on dev, run `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task normalize backlog/tasks/task-13434*.md`. The NOTES plus IMPLEMENTATION_NOTES pair fails the new task-format check. Commit the result.
- [ ] **Step 2: Sweep on the rebased branch.** Run the Task 6 list plus `tests/Usage tests/UserProfile tests/Storage tests/Admin tests/AuthNZ/unit tests/AuthNZ_SQLite tests/MediaIngestion_NEW tests/Docs tests/lint`. Prove any failure that also occurs on `origin/dev` on a detached `origin/dev` checkout.
- [ ] **Step 3: Static checks.** Run Bandit `-ll` on the changed app files; it must find nothing at medium or above. Ruff per-rule tallies must be identical to the merge base for the touched files, compared via `--stdin-filename`. Run `Helper_Scripts/export_openapi_schema.py --check` with the venv python; it must pass.
- [ ] **Step 4: Open the PR.** First message `tldw-server-03` for a merge slot, listing the touched areas. Push, then `gh pr create --base dev`. The body covers:
  - the resolver-backed enforcement;
  - the writers (and the deleted dead registration-code branch);
  - the readers and the `null` contract;
  - the fixed `/admin/storage-quotas/users/{id}`;
  - team and org values now allowed;
  - the migration (SQLite 100, one-time PG marker);
  - the frontend type changes;
  - the verification counts;
  - the `## Change summary` waiver section and the Claude Code footer.
- [ ] **Step 5: Qodo and merge.**
  - Address every Qodo finding: fix it, or decline it with a posted rationale. Add the handled titles to the scratchpad `qodo_open.py`.
  - Merge only on the peer's go-ahead, running the merge queue for this PR alone.
  - Then message the peer, and run backlog-py: `task edit TASK-13434 --check-ac 3 --append-notes "PR C merged as #<n> (<sha>) ..."`.
