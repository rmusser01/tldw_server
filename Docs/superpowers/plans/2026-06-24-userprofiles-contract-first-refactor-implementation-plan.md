# UserProfiles Contract-First Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor UserProfiles around typed contracts, planner/executor seams, compatibility adapters, and milestone-gated clean v2 profile behavior.

**Architecture:** Preserve current v1 routes first, add contract and planning seams below them, then migrate self/admin single-update orchestration behind compatibility mappers. Reads and bulk are separate milestones. The clean v2 route surface is gated until existing routes pass on the new internals.

**Tech Stack:** FastAPI, Pydantic, dataclasses, async Python, SQLite/Postgres AuthNZ database pool, pytest, Bandit.

---

## Scope Check

This plan covers one subsystem: UserProfiles. It is broad enough that execution must be milestone-based. The first implementation pass must not add public v2 routes; it must prove the planner, command service, compatibility mapper, and legacy route behavior behind existing endpoints first.

## Source Design

- Design spec: `Docs/superpowers/specs/2026-06-24-userprofiles-contract-first-refactor-design.md`
- Planning task: `TASK-12015`

## File Structure

### New Core Files

- `tldw_Server_API/app/core/UserProfiles/contracts.py`
  Typed request/result/plan models shared by query, command, planner, executor, mapper, and tests.

- `tldw_Server_API/app/core/UserProfiles/error_mapping.py`
  Stable domain error codes and mapping to existing profile API error payloads.

- `tldw_Server_API/app/core/UserProfiles/planner.py`
  Catalog-driven update planner. Produces `UpdatePlan` without mutating state.

- `tldw_Server_API/app/core/UserProfiles/effects.py`
  Effect descriptors and dispatcher split into pre-commit required effects and post-commit best-effort effects.

- `tldw_Server_API/app/core/UserProfiles/command_service.py`
  Self/admin single-update orchestration. Owns dry-run/apply flow and transaction ordering.

- `tldw_Server_API/app/core/UserProfiles/query_service.py`
  Read orchestration wrapper around existing `UserProfileService`, introduced after single-update compatibility is stable.

- `tldw_Server_API/app/core/UserProfiles/bulk_command_service.py`
  Bulk update orchestration wrapper, introduced after single-update compatibility is stable.

- `tldw_Server_API/app/core/UserProfiles/response_mappers.py`
  Legacy v1 and clean-contract response mapping. Initial tasks only use legacy v1 mapping.

### Modified Existing Files

- `tldw_Server_API/app/core/UserProfiles/update_service.py`
  Initially remains the mutation executor. Planning logic moves out incrementally.

- `tldw_Server_API/app/core/UserProfiles/service.py`
  Initially remains the read assembler. Query orchestration moves out after command flow is stable.

- `tldw_Server_API/app/api/v1/endpoints/users.py`
  Self profile update route becomes a thin adapter to `ProfileCommandService`.

- `tldw_Server_API/app/services/admin_profiles_service.py`
  Admin single-update route path delegates to `ProfileCommandService`. Bulk remains unchanged until its own milestone.

- `tldw_Server_API/app/api/v1/utils/profile_errors.py`
  May delegate to new domain error mapping while preserving existing response shape.

### New Tests

- `tldw_Server_API/tests/UserProfile/test_profile_contracts.py`
- `tldw_Server_API/tests/UserProfile/test_profile_error_mapping.py`
- `tldw_Server_API/tests/UserProfile/test_profile_update_planner.py`
- `tldw_Server_API/tests/UserProfile/test_profile_command_service.py`
- `tldw_Server_API/tests/UserProfile/test_profile_response_mappers.py`
- `tldw_Server_API/tests/UserProfile/test_profile_bulk_command_service.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_user_profile_command_backend_selection.py`
- `tldw_Server_API/tests/AuthNZ_Postgres/test_user_profile_version_locking_pg.py`

## Milestone Gate Summary

- **Milestone 1:** Characterize v1 behavior and add typed contracts.
- **Milestone 2:** Add planner and error taxonomy without route changes.
- **Milestone 3:** Route self/admin single updates through command service using legacy response mapping.
- **Milestone 4:** Extract read orchestration behind query service.
- **Milestone 5:** Move bulk into its own command service.
- **Milestone 6:** Decide and add clean v2 route surface after v1 compatibility tests pass on new internals.
- **Milestone 7:** Verification, docs, Bandit, and cleanup.

## Task 1: Characterize Current V1 Profile Contracts

**Files:**
- Create: `tldw_Server_API/tests/UserProfile/test_user_profile_legacy_contract_characterization.py`
- Modify: no production files

- [ ] **Step 1: Add legacy self-update characterization tests**

Create `tldw_Server_API/tests/UserProfile/test_user_profile_legacy_contract_characterization.py`:

```python
from __future__ import annotations

from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def test_legacy_self_update_response_keeps_applied_and_skipped(auth_headers) -> None:
    with TestClient(app) as client:
        response = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": "legacy-contract"}
                ]
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("profile_version"), str)
    assert payload.get("applied") == ["preferences.ui.theme"]
    assert payload.get("skipped") == []


def test_legacy_self_update_unknown_key_is_all_or_reject(auth_headers) -> None:
    with TestClient(app) as client:
        response = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": "not-written"},
                    {"key": "preferences.ui.missing", "value": "bad"},
                ]
            },
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "profile_update_unknown_key"
    assert payload["errors"] == [
        {"key": "preferences.ui.missing", "message": "unknown_key"}
    ]
```

- [ ] **Step 2: Add legacy admin-update characterization tests**

Append to `test_user_profile_legacy_contract_characterization.py`:

```python
def _current_user_id(client: TestClient, auth_headers) -> int:
    response = client.get("/api/v1/users/me/profile", headers=auth_headers)
    assert response.status_code == 200
    return int(response.json()["user"]["id"])


def test_legacy_admin_update_response_keeps_applied_and_skipped(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _current_user_id(client, auth_headers)
        response = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "limits.storage_quota_mb", "value": 3072}
                ]
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("profile_version"), str)
    assert payload.get("applied") == ["limits.storage_quota_mb"]
    assert payload.get("skipped") == []


def test_legacy_admin_update_version_conflict_shape(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _current_user_id(client, auth_headers)
        response = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "profile_version": "2000-01-01T00:00:00Z",
                "updates": [
                    {"key": "limits.storage_quota_mb", "value": 3072}
                ],
            },
        )

    assert response.status_code == 409
    payload = response.json()
    assert payload["error_code"] == "profile_version_mismatch"
    assert payload["errors"] == [
        {"key": "profile_version", "message": "mismatch"}
    ]
```

- [ ] **Step 3: Run characterization tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_user_profile_legacy_contract_characterization.py -q
```

Expected: all tests pass against current code. If a test fails because current behavior differs, update the assertion to match the current v1 behavior and record the observed behavior in the test name.

- [ ] **Step 4: Commit characterization tests**

Run:

```bash
git add tldw_Server_API/tests/UserProfile/test_user_profile_legacy_contract_characterization.py
git commit -m "test: characterize UserProfiles legacy contract"
```

Expected: commit succeeds with only the new characterization test file staged.

## Task 2: Add Typed Contract Models

**Files:**
- Create: `tldw_Server_API/app/core/UserProfiles/contracts.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_contracts.py`

- [ ] **Step 1: Write contract model tests**

Create `tldw_Server_API/tests/UserProfile/test_profile_contracts.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from tldw_Server_API.app.core.UserProfiles.contracts import (
    EffectPolicy,
    EffectTiming,
    ProfileContractMode,
    ProfileUpdateCommand,
    UpdateMutation,
    UpdatePlan,
)


def test_update_plan_separates_pre_commit_and_post_commit_effects() -> None:
    command = ProfileUpdateCommand(
        actor_user_id=5,
        target_user_id=7,
        updates=(("preferences.ui.theme", "paper"),),
        roles=frozenset({"user"}),
        dry_run=False,
        expected_profile_version=datetime(2026, 1, 1, tzinfo=timezone.utc),
        contract_mode=ProfileContractMode.LEGACY_V1,
    )

    plan = UpdatePlan(
        command=command,
        mutations=(
            UpdateMutation(
                key="preferences.ui.theme",
                operation="upsert_override",
                payload={"value": "paper"},
            ),
        ),
        effects=(),
    )

    assert plan.command.target_user_id == 7
    assert plan.mutations[0].operation == "upsert_override"
    assert plan.pre_commit_effects == ()
    assert plan.post_commit_effects == ()


def test_effect_policy_values_are_stable() -> None:
    assert EffectTiming.PRE_COMMIT.value == "pre_commit"
    assert EffectTiming.POST_COMMIT.value == "post_commit"
    assert EffectPolicy.REQUIRED.value == "required"
    assert EffectPolicy.BEST_EFFORT.value == "best_effort"
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_contracts.py -q
```

Expected: fails with `ModuleNotFoundError` for `tldw_Server_API.app.core.UserProfiles.contracts`.

- [ ] **Step 3: Add contract models**

Create `tldw_Server_API/app/core/UserProfiles/contracts.py`:

```python
"""
Typed internal contracts for UserProfiles read/update orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ProfileContractMode(str, Enum):
    LEGACY_V1 = "legacy_v1"
    CLEAN_V2 = "clean_v2"


class EffectTiming(str, Enum):
    PRE_COMMIT = "pre_commit"
    POST_COMMIT = "post_commit"


class EffectPolicy(str, Enum):
    REQUIRED = "required"
    BEST_EFFORT = "best_effort"


@dataclass(frozen=True)
class ProfileReadRequest:
    actor_user_id: int | None
    target_user_id: int
    sections: frozenset[str] | None = None
    include_sources: bool = False
    include_raw: bool = False
    mask_secrets: bool = True
    contract_mode: ProfileContractMode = ProfileContractMode.LEGACY_V1


@dataclass(frozen=True)
class ProfileUpdateCommand:
    actor_user_id: int | None
    target_user_id: int
    updates: tuple[tuple[str, Any], ...]
    roles: frozenset[str]
    dry_run: bool
    expected_profile_version: datetime | None = None
    active_org_id: int | None = None
    active_team_id: int | None = None
    contract_mode: ProfileContractMode = ProfileContractMode.LEGACY_V1


@dataclass(frozen=True)
class UpdateMutation:
    key: str
    operation: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EffectDescriptor:
    name: str
    timing: EffectTiming
    policy: EffectPolicy
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdatePlan:
    command: ProfileUpdateCommand
    mutations: tuple[UpdateMutation, ...] = ()
    effects: tuple[EffectDescriptor, ...] = ()

    @property
    def pre_commit_effects(self) -> tuple[EffectDescriptor, ...]:
        return tuple(effect for effect in self.effects if effect.timing == EffectTiming.PRE_COMMIT)

    @property
    def post_commit_effects(self) -> tuple[EffectDescriptor, ...]:
        return tuple(effect for effect in self.effects if effect.timing == EffectTiming.POST_COMMIT)


@dataclass(frozen=True)
class PlannedUpdateResult:
    profile_version: datetime
    applied: tuple[str, ...] = ()
    rejected: tuple[dict[str, str], ...] = ()
```

- [ ] **Step 4: Run contract tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_contracts.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit contract models**

Run:

```bash
git add tldw_Server_API/app/core/UserProfiles/contracts.py tldw_Server_API/tests/UserProfile/test_profile_contracts.py
git commit -m "feat: add UserProfiles internal contracts"
```

Expected: commit succeeds with only the contract model and test files staged.

## Task 3: Add Domain Error Mapping

**Files:**
- Create: `tldw_Server_API/app/core/UserProfiles/error_mapping.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_error_mapping.py`
- Modify: `tldw_Server_API/app/api/v1/utils/profile_errors.py`

- [ ] **Step 1: Write error mapping tests**

Create `tldw_Server_API/tests/UserProfile/test_profile_error_mapping.py`:

```python
from __future__ import annotations

from fastapi import status

from tldw_Server_API.app.core.UserProfiles.error_mapping import (
    ProfileErrorCode,
    map_profile_error_code,
)


def test_profile_error_code_http_mapping() -> None:
    assert map_profile_error_code(ProfileErrorCode.UNKNOWN_KEY).status_code == status.HTTP_400_BAD_REQUEST
    assert map_profile_error_code(ProfileErrorCode.INVALID_VALUE).status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert map_profile_error_code(ProfileErrorCode.FORBIDDEN_SCOPE).status_code == status.HTTP_403_FORBIDDEN
    assert map_profile_error_code(ProfileErrorCode.TARGET_NOT_FOUND).status_code == status.HTTP_404_NOT_FOUND
    assert map_profile_error_code(ProfileErrorCode.VERSION_MISMATCH).status_code == status.HTTP_409_CONFLICT


def test_forbidden_role_escalation_maps_to_forbidden_profile_update() -> None:
    mapped = map_profile_error_code(ProfileErrorCode.FORBIDDEN_ROLE_ESCALATION)

    assert mapped.status_code == status.HTTP_403_FORBIDDEN
    assert mapped.error_code == "profile_update_forbidden"
    assert mapped.detail == "Caller cannot edit one or more fields"
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_error_mapping.py -q
```

Expected: fails with `ModuleNotFoundError` for `error_mapping`.

- [ ] **Step 3: Add error mapping module**

Create `tldw_Server_API/app/core/UserProfiles/error_mapping.py`:

```python
"""
Stable UserProfiles domain error mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fastapi import status


class ProfileErrorCode(str, Enum):
    UNKNOWN_KEY = "unknown_key"
    UNSUPPORTED_KEY = "unsupported_key"
    INVALID_PAYLOAD = "invalid_payload"
    INVALID_ACTION = "invalid_action"
    INVALID_VALUE = "invalid_value"
    TYPE_MISMATCH = "type_mismatch"
    ENUM_VIOLATION = "enum_violation"
    MIN_VIOLATION = "min_violation"
    MAX_VIOLATION = "max_violation"
    INVALID_ROLE = "invalid_role"
    FORBIDDEN_KEY = "forbidden"
    FORBIDDEN_SCOPE = "forbidden_scope"
    FORBIDDEN_ROLE_ESCALATION = "forbidden_role_escalation"
    TARGET_NOT_FOUND = "user_not_found"
    MEMBERSHIP_NOT_FOUND = "membership_not_found"
    TEAM_NOT_FOUND = "team_not_found"
    ORG_NOT_FOUND = "org_not_found"
    VERSION_MISMATCH = "profile_version_mismatch"


@dataclass(frozen=True)
class ProfileErrorMapping:
    status_code: int
    error_code: str
    detail: str


_BAD_REQUEST = {
    ProfileErrorCode.UNKNOWN_KEY,
    ProfileErrorCode.UNSUPPORTED_KEY,
    ProfileErrorCode.INVALID_PAYLOAD,
    ProfileErrorCode.INVALID_ACTION,
}

_UNPROCESSABLE = {
    ProfileErrorCode.INVALID_VALUE,
    ProfileErrorCode.TYPE_MISMATCH,
    ProfileErrorCode.ENUM_VIOLATION,
    ProfileErrorCode.MIN_VIOLATION,
    ProfileErrorCode.MAX_VIOLATION,
    ProfileErrorCode.INVALID_ROLE,
    ProfileErrorCode.MEMBERSHIP_NOT_FOUND,
}

_FORBIDDEN = {
    ProfileErrorCode.FORBIDDEN_KEY,
    ProfileErrorCode.FORBIDDEN_SCOPE,
    ProfileErrorCode.FORBIDDEN_ROLE_ESCALATION,
}

_NOT_FOUND = {
    ProfileErrorCode.TARGET_NOT_FOUND,
    ProfileErrorCode.TEAM_NOT_FOUND,
    ProfileErrorCode.ORG_NOT_FOUND,
}


def map_profile_error_code(code: ProfileErrorCode | str) -> ProfileErrorMapping:
    normalized = ProfileErrorCode(str(code))
    if normalized in _BAD_REQUEST:
        return ProfileErrorMapping(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="profile_update_unknown_key",
            detail="One or more keys are not recognized",
        )
    if normalized in _FORBIDDEN:
        return ProfileErrorMapping(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="profile_update_forbidden",
            detail="Caller cannot edit one or more fields",
        )
    if normalized in _NOT_FOUND:
        return ProfileErrorMapping(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="profile_update_not_found",
            detail="Target resource not found",
        )
    if normalized == ProfileErrorCode.VERSION_MISMATCH:
        return ProfileErrorMapping(
            status_code=status.HTTP_409_CONFLICT,
            error_code="profile_version_mismatch",
            detail="profile_version_mismatch",
        )
    if normalized in _UNPROCESSABLE:
        return ProfileErrorMapping(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="profile_update_invalid",
            detail="One or more updates failed validation",
        )
    return ProfileErrorMapping(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code="profile_update_invalid",
        detail="One or more updates failed validation",
    )
```

- [ ] **Step 4: Run mapping tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_error_mapping.py -q
```

Expected: tests pass.

- [ ] **Step 5: Keep v1 profile error helper compatible**

Modify `tldw_Server_API/app/api/v1/utils/profile_errors.py` only after tests pass. Keep public behavior unchanged except for delegating known message categories to `map_profile_error_code`.

Use this helper inside `classify_profile_update_skips`:

```python
from tldw_Server_API.app.core.UserProfiles.error_mapping import (
    ProfileErrorCode,
    map_profile_error_code,
)
```

Add this private helper:

```python
def _first_matching_code(messages: set[str]) -> ProfileErrorCode | None:
    for message in messages:
        try:
            return ProfileErrorCode(message)
        except ValueError:
            continue
    return None
```

Keep the existing v1 unknown-key precedence. Do not change the shape returned by current tests.

- [ ] **Step 6: Run existing and new error tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/UserProfile/test_profile_error_mapping.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_updates.py \
  -q
```

Expected: tests pass.

- [ ] **Step 7: Commit error mapping**

Run:

```bash
git add \
  tldw_Server_API/app/core/UserProfiles/error_mapping.py \
  tldw_Server_API/app/api/v1/utils/profile_errors.py \
  tldw_Server_API/tests/UserProfile/test_profile_error_mapping.py
git commit -m "feat: add UserProfiles error taxonomy"
```

Expected: commit succeeds with only error-mapping files staged.

## Task 4: Add Planner Skeleton Behind Existing Update Service

**Files:**
- Create: `tldw_Server_API/app/core/UserProfiles/planner.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_update_planner.py`
- Modify: `tldw_Server_API/app/core/UserProfiles/update_service.py`

- [ ] **Step 1: Write planner dry-run parity tests**

Create `tldw_Server_API/tests/UserProfile/test_profile_update_planner.py`:

```python
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.UserProfiles.contracts import (
    ProfileContractMode,
    ProfileUpdateCommand,
)
from tldw_Server_API.app.core.UserProfiles.planner import ProfileUpdatePlanner
from tldw_Server_API.app.core.UserProfiles.update_service import ProfileUpdateScope


@pytest.mark.asyncio
async def test_planner_rejects_unknown_key_without_mutation() -> None:
    planner = ProfileUpdatePlanner(db_pool=object())
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.missing", "paper"),),
        roles=frozenset({"user"}),
        dry_run=True,
        contract_mode=ProfileContractMode.LEGACY_V1,
    )

    result = await planner.plan(command, db_conn=object(), scope=ProfileUpdateScope(actor_user_id=7))

    assert result.applied == []
    assert result.skipped == [{"key": "preferences.ui.missing", "message": "unknown_key"}]


@pytest.mark.asyncio
async def test_planner_accepts_preference_update_without_executing_write() -> None:
    planner = ProfileUpdatePlanner(db_pool=object())
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.theme", "paper"),),
        roles=frozenset({"user"}),
        dry_run=True,
        contract_mode=ProfileContractMode.LEGACY_V1,
    )

    result = await planner.plan(command, db_conn=object(), scope=ProfileUpdateScope(actor_user_id=7))

    assert result.applied == ["preferences.ui.theme"]
    assert result.skipped == []
```

- [ ] **Step 2: Run planner tests to verify red**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_update_planner.py -q
```

Expected: fails with `ModuleNotFoundError` for `planner`.

- [ ] **Step 3: Add planner that delegates to current dry-run update service**

Create `tldw_Server_API/app/core/UserProfiles/planner.py`:

```python
"""
UserProfiles update planning.
"""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand
from tldw_Server_API.app.core.UserProfiles.update_service import (
    ProfileUpdateScope,
    UpdateResult,
    UserProfileUpdateService,
)


class ProfileUpdatePlanner:
    """Build an update plan using current catalog validation without mutating state."""

    def __init__(self, db_pool: Any) -> None:
        self._db_pool = db_pool

    async def plan(
        self,
        command: ProfileUpdateCommand,
        *,
        db_conn: Any,
        scope: ProfileUpdateScope | None,
    ) -> UpdateResult:
        service = UserProfileUpdateService(self._db_pool)
        return await service.apply_updates(
            user_id=command.target_user_id,
            updates=command.updates,
            roles=set(command.roles),
            dry_run=True,
            db_conn=db_conn,
            updated_by=command.actor_user_id,
            scope=scope,
        )
```

This first planner is deliberately thin. Later tasks move catalog validation and membership planning out of `UserProfileUpdateService` once routes are stable.

- [ ] **Step 4: Run planner tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_update_planner.py -q
```

Expected: tests pass.

- [ ] **Step 5: Run update service tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_user_profile_update_service_backend_selection.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_updates.py \
  -q
```

Expected: tests pass.

- [ ] **Step 6: Commit planner skeleton**

Run:

```bash
git add \
  tldw_Server_API/app/core/UserProfiles/planner.py \
  tldw_Server_API/tests/UserProfile/test_profile_update_planner.py
git commit -m "feat: add UserProfiles update planner seam"
```

Expected: commit succeeds with only planner files staged.

## Task 5: Add Command Service For Single Updates

**Files:**
- Create: `tldw_Server_API/app/core/UserProfiles/command_service.py`
- Create: `tldw_Server_API/app/core/UserProfiles/effects.py`
- Create: `tldw_Server_API/app/core/UserProfiles/response_mappers.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_command_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/users.py`
- Modify: `tldw_Server_API/app/services/admin_profiles_service.py`

- [ ] **Step 1: Write command service stale-version test**

Create `tldw_Server_API/tests/UserProfile/test_profile_command_service.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand
from tldw_Server_API.app.core.UserProfiles.update_service import UpdateResult


class _ProfileService:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, bool]] = []
        self.initial = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.locked = datetime(2026, 1, 2, tzinfo=timezone.utc)

    async def get_profile_version(self, *, user_id: int, db_conn=None, lock_user: bool = False):
        self.calls.append((db_conn, lock_user))
        return self.locked if lock_user else self.initial

    def versions_match(self, current, expected) -> bool:
        return current == expected


class _Planner:
    async def plan(self, command, *, db_conn, scope):
        return UpdateResult(applied=[key for key, _value in command.updates])


class _Executor:
    def __init__(self) -> None:
        self.called = False

    async def apply_updates(self, **_kwargs):
        self.called = True
        return UpdateResult(applied=["preferences.ui.theme"])


@pytest.mark.asyncio
async def test_command_service_rechecks_version_before_apply() -> None:
    profile_service = _ProfileService()
    executor = _Executor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=profile_service,
        planner=_Planner(),
        executor=executor,
    )
    write_conn = object()
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.theme", "paper"),),
        roles=frozenset({"user"}),
        dry_run=False,
        expected_profile_version=profile_service.initial,
    )

    result = await command_service.apply(command, db_conn=write_conn, scope=None)

    assert result.status_code == 409
    assert result.error_code == "profile_version_mismatch"
    assert executor.called is False
```

- [ ] **Step 2: Run command service test to verify red**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_command_service.py -q
```

Expected: fails with `ModuleNotFoundError` for `command_service`.

- [ ] **Step 3: Add minimal response mapper**

Create `tldw_Server_API/app/core/UserProfiles/response_mappers.py`:

```python
"""
Response mapping helpers for UserProfiles command flows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class LegacyProfileCommandResult:
    status_code: int = 200
    profile_version: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    applied: tuple[str, ...] = ()
    skipped: tuple[dict[str, str], ...] = ()
    error_code: str | None = None
    detail: str | None = None
```

- [ ] **Step 4: Add effect dispatcher skeleton**

Create `tldw_Server_API/app/core/UserProfiles/effects.py`:

```python
"""
UserProfiles effect dispatch.
"""

from __future__ import annotations

from collections.abc import Iterable

from loguru import logger

from tldw_Server_API.app.core.UserProfiles.contracts import (
    EffectDescriptor,
    EffectPolicy,
    EffectTiming,
)


class ProfileEffectDispatcher:
    async def run_pre_commit(self, effects: Iterable[EffectDescriptor]) -> None:
        for effect in effects:
            if effect.timing != EffectTiming.PRE_COMMIT:
                continue
            if effect.policy == EffectPolicy.REQUIRED:
                logger.debug("Required profile effect completed: {}", effect.name)

    async def run_post_commit(self, effects: Iterable[EffectDescriptor]) -> None:
        for effect in effects:
            if effect.timing != EffectTiming.POST_COMMIT:
                continue
            try:
                logger.debug("Best-effort profile effect completed: {}", effect.name)
            except Exception as exc:
                logger.debug("Best-effort profile effect failed: {} {}", effect.name, type(exc).__name__)
```

- [ ] **Step 5: Add command service**

Create `tldw_Server_API/app/core/UserProfiles/command_service.py`:

```python
"""
Single-update UserProfiles command orchestration.
"""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand
from tldw_Server_API.app.core.UserProfiles.effects import ProfileEffectDispatcher
from tldw_Server_API.app.core.UserProfiles.planner import ProfileUpdatePlanner
from tldw_Server_API.app.core.UserProfiles.response_mappers import LegacyProfileCommandResult
from tldw_Server_API.app.core.UserProfiles.service import UserProfileService
from tldw_Server_API.app.core.UserProfiles.update_service import (
    ProfileUpdateScope,
    UserProfileUpdateService,
)


class ProfileCommandService:
    def __init__(
        self,
        *,
        db_pool: Any,
        profile_service: Any | None = None,
        planner: Any | None = None,
        executor: Any | None = None,
        effects: ProfileEffectDispatcher | None = None,
    ) -> None:
        self._db_pool = db_pool
        self._profile_service = profile_service or UserProfileService(db_pool)
        self._planner = planner or ProfileUpdatePlanner(db_pool)
        self._executor = executor or UserProfileUpdateService(db_pool)
        self._effects = effects or ProfileEffectDispatcher()

    async def apply(
        self,
        command: ProfileUpdateCommand,
        *,
        db_conn: Any,
        scope: ProfileUpdateScope | None,
    ) -> LegacyProfileCommandResult:
        preflight = await self._planner.plan(command, db_conn=db_conn, scope=scope)
        if preflight.skipped:
            return LegacyProfileCommandResult(
                status_code=422,
                applied=tuple(preflight.applied),
                skipped=tuple(preflight.skipped),
                error_code="profile_update_invalid",
                detail="One or more updates failed validation",
            )
        current_version = await self._profile_service.get_profile_version(
            user_id=command.target_user_id,
        )
        if command.dry_run:
            return LegacyProfileCommandResult(
                profile_version=current_version,
                applied=tuple(preflight.applied),
                skipped=(),
            )
        if command.expected_profile_version is not None:
            locked_version = await self._profile_service.get_profile_version(
                user_id=command.target_user_id,
                db_conn=db_conn,
                lock_user=True,
            )
            if not self._profile_service.versions_match(
                locked_version,
                command.expected_profile_version,
            ):
                return LegacyProfileCommandResult(
                    status_code=409,
                    profile_version=locked_version,
                    error_code="profile_version_mismatch",
                    detail="profile_version_mismatch",
                    skipped=({"key": "profile_version", "message": "mismatch"},),
                )
        result = await self._executor.apply_updates(
            user_id=command.target_user_id,
            updates=command.updates,
            roles=set(command.roles),
            dry_run=False,
            db_conn=db_conn,
            updated_by=command.actor_user_id,
            scope=scope,
        )
        current_version = await self._profile_service.get_profile_version(
            user_id=command.target_user_id,
        )
        return LegacyProfileCommandResult(
            profile_version=current_version,
            applied=tuple(result.applied),
            skipped=tuple(result.skipped),
        )
```

- [ ] **Step 6: Run command service test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_command_service.py -q
```

Expected: tests pass.

- [ ] **Step 7: Add route integration only after command service passes**

Modify `users.py` and `admin_profiles_service.py` in the next task, not in this task. This keeps the command service commit isolated.

- [ ] **Step 8: Commit command service seam**

Run:

```bash
git add \
  tldw_Server_API/app/core/UserProfiles/command_service.py \
  tldw_Server_API/app/core/UserProfiles/effects.py \
  tldw_Server_API/app/core/UserProfiles/response_mappers.py \
  tldw_Server_API/tests/UserProfile/test_profile_command_service.py
git commit -m "feat: add UserProfiles command service seam"
```

Expected: commit succeeds with only command-service seam files staged.

## Task 6: Route Existing Single Updates Through Command Service

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/users.py`
- Modify: `tldw_Server_API/app/services/admin_profiles_service.py`
- Test: `tldw_Server_API/tests/UserProfile/test_user_profile_updates.py`
- Test: `tldw_Server_API/tests/UserProfile/test_user_profile_legacy_contract_characterization.py`

- [ ] **Step 1: Add a mapper helper in `users.py`**

In `tldw_Server_API/app/api/v1/endpoints/users.py`, add imports:

```python
from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import (
    ProfileContractMode,
    ProfileUpdateCommand,
)
```

Add this helper near `_profile_error_response`:

```python
def _command_result_to_legacy_response(result):
    if result.error_code:
        errors = [
            UserProfileErrorDetail(
                key=str(item.get("key") or ""),
                message=str(item.get("message") or ""),
            )
            for item in result.skipped
        ]
        return _profile_error_response(
            status_code=result.status_code,
            error_code=str(result.error_code),
            detail=str(result.detail or result.error_code),
            errors=errors,
        )
    return UserProfileUpdateResponse(
        profile_version=result.profile_version,
        applied=list(result.applied),
        skipped=[UserProfileUpdateError(**item) for item in result.skipped],
    )
```

- [ ] **Step 2: Replace self-update route internals after current preflight block passes tests**

In `update_current_user_profile`, keep the active/verified user check and `updates` empty check. Replace the existing preflight/apply body with:

```python
    db_pool = await get_db_pool()
    command_service = ProfileCommandService(db_pool=db_pool)
    command = ProfileUpdateCommand(
        actor_user_id=user_id,
        target_user_id=user_id,
        updates=tuple((entry.key, entry.value) for entry in payload.updates),
        roles=frozenset({"user"}),
        dry_run=payload.dry_run,
        expected_profile_version=payload.profile_version,
        contract_mode=ProfileContractMode.LEGACY_V1,
    )
    command_result = await command_service.apply(command, db_conn=db, scope=None)
    response = _command_result_to_legacy_response(command_result)
    if not isinstance(response, UserProfileUpdateResponse):
        return response
```

Keep the existing audit emit block after successful non-dry-run response.

- [ ] **Step 3: Run self-update tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/UserProfile/test_user_profile_updates.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_legacy_contract_characterization.py \
  -q
```

Expected: tests pass.

- [ ] **Step 4: Route admin single update through command service**

In `tldw_Server_API/app/services/admin_profiles_service.py`, add imports:

```python
from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import (
    ProfileContractMode,
    ProfileUpdateCommand,
)
```

Inside `update_user_profile`, keep:

- empty update check,
- `admin_scope_service.enforce_admin_user_scope`,
- target user lookup,
- role derivation.

Replace the preflight/apply branch with a `ProfileCommandService` call:

```python
    command_service = ProfileCommandService(db_pool=db_pool)
    command = ProfileUpdateCommand(
        actor_user_id=principal.user_id,
        target_user_id=int(user_id),
        updates=tuple((entry.key, entry.value) for entry in payload.updates),
        roles=frozenset(roles),
        dry_run=payload.dry_run,
        expected_profile_version=payload.profile_version,
        active_org_id=principal.active_org_id,
        active_team_id=principal.active_team_id,
        contract_mode=ProfileContractMode.LEGACY_V1,
    )
    command_result = await command_service.apply(
        command,
        db_conn=db,
        scope=ProfileUpdateScope(
            actor_user_id=principal.user_id,
            active_org_id=principal.active_org_id,
            active_team_id=principal.active_team_id,
        ),
    )
```

Map `command_result.error_code` to the same tuple response shape used today. Reuse `_profile_error_response` and `UserProfileUpdateResponse`.

- [ ] **Step 5: Run admin update tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/UserProfile/test_user_profile_updates.py \
  tldw_Server_API/tests/UserProfile/test_admin_profiles_service_update.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_admin_audit.py \
  -q
```

Expected: tests pass.

- [ ] **Step 6: Run backend-selection tests**

Create `tldw_Server_API/tests/AuthNZ/unit/test_user_profile_command_backend_selection.py` before running this command:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand
from tldw_Server_API.app.core.UserProfiles.update_service import UpdateResult


class _ProfileService:
    async def get_profile_version(
        self,
        *,
        user_id: int,
        db_conn: Any = None,
        lock_user: bool = False,
    ):
        return datetime(2026, 1, 1, tzinfo=timezone.utc)

    def versions_match(self, current, expected) -> bool:
        return current == expected


class _Planner:
    async def plan(self, command, *, db_conn, scope):
        return UpdateResult(applied=[key for key, _value in command.updates])


class _Executor:
    def __init__(self) -> None:
        self.db_conn = None

    async def apply_updates(self, **kwargs):
        self.db_conn = kwargs["db_conn"]
        return UpdateResult(applied=["preferences.ui.theme"])


@pytest.mark.asyncio
async def test_command_service_passes_supplied_transaction_connection_to_executor() -> None:
    executor = _Executor()
    service = ProfileCommandService(
        db_pool=object(),
        profile_service=_ProfileService(),
        planner=_Planner(),
        executor=executor,
    )
    transaction_conn = object()
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.theme", "paper"),),
        roles=frozenset({"user"}),
        dry_run=False,
    )

    result = await service.apply(command, db_conn=transaction_conn, scope=None)

    assert result.applied == ("preferences.ui.theme",)
    assert executor.db_conn is transaction_conn
```

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_user_profile_update_service_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_user_profile_command_backend_selection.py \
  -q
```

Expected: tests pass.

- [ ] **Step 7: Commit route migration**

Run:

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/users.py \
  tldw_Server_API/app/services/admin_profiles_service.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_updates.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_legacy_contract_characterization.py \
  tldw_Server_API/tests/AuthNZ/unit/test_user_profile_command_backend_selection.py
git commit -m "refactor: route profile updates through command service"
```

Expected: commit succeeds with only route migration and related tests staged.

## Task 7: Extract Query Service Behind Existing Reads

**Files:**
- Create: `tldw_Server_API/app/core/UserProfiles/query_service.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_query_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/users.py`
- Modify: `tldw_Server_API/app/services/admin_profiles_service.py`

- [ ] **Step 1: Add query service tests**

Create `tldw_Server_API/tests/UserProfile/test_profile_query_service.py`:

```python
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.UserProfiles.contracts import ProfileReadRequest
from tldw_Server_API.app.core.UserProfiles.query_service import ProfileQueryService


class _ProfileService:
    async def build_profile(self, **kwargs):
        return {
            "profile_version": kwargs["user"]["updated_at"],
            "catalog_version": "1.0.0",
            "user": {"id": kwargs["user"]["id"], "username": kwargs["user"]["username"]},
        }


@pytest.mark.asyncio
async def test_query_service_delegates_to_profile_builder() -> None:
    service = ProfileQueryService(profile_service=_ProfileService())
    request = ProfileReadRequest(actor_user_id=7, target_user_id=7)

    result = await service.build(
        request,
        user={"id": 7, "username": "alice", "updated_at": "2026-01-01T00:00:00Z"},
        security={},
        metrics_scope="self",
    )

    assert result["catalog_version"] == "1.0.0"
    assert result["user"]["id"] == 7
```

- [ ] **Step 2: Run query service test to verify red**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_query_service.py -q
```

Expected: fails with `ModuleNotFoundError` for `query_service`.

- [ ] **Step 3: Add query service**

Create `tldw_Server_API/app/core/UserProfiles/query_service.py`:

```python
"""
UserProfiles read orchestration.
"""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.UserProfiles.contracts import ProfileReadRequest
from tldw_Server_API.app.core.UserProfiles.service import UserProfileService


class ProfileQueryService:
    def __init__(self, profile_service: Any) -> None:
        self._profile_service = profile_service

    async def build(
        self,
        request: ProfileReadRequest,
        *,
        user: dict[str, Any],
        security: dict[str, Any],
        metrics_scope: str,
    ) -> dict[str, Any]:
        return await self._profile_service.build_profile(
            user=user,
            sections=request.sections,
            security=security,
            include_sources=request.include_sources,
            include_raw=request.include_raw,
            mask_secrets=request.mask_secrets,
            metrics_scope=metrics_scope,
        )
```

- [ ] **Step 4: Run query tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_query_service.py -q
```

Expected: tests pass.

- [ ] **Step 5: Route self read through query service**

In `users.py`, keep `_resolve_user_context`, `_require_active_verified_user`, and security building. Replace the direct `service.build_profile` call with:

```python
    query_service = ProfileQueryService(service)
    profile = await query_service.build(
        ProfileReadRequest(
            actor_user_id=user_id,
            target_user_id=user_id,
            sections=requested,
            include_sources=include_sources,
            include_raw=include_raw,
            mask_secrets=mask_secrets,
            contract_mode=ProfileContractMode.LEGACY_V1,
        ),
        user=user_dict,
        security=security,
        metrics_scope="self",
    )
```

- [ ] **Step 6: Run read tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/UserProfile/test_user_profile_read.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_batch.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_effective_layers.py \
  -q
```

Expected: tests pass.

- [ ] **Step 7: Commit query service**

Run:

```bash
git add \
  tldw_Server_API/app/core/UserProfiles/query_service.py \
  tldw_Server_API/app/api/v1/endpoints/users.py \
  tldw_Server_API/app/services/admin_profiles_service.py \
  tldw_Server_API/tests/UserProfile/test_profile_query_service.py
git commit -m "refactor: add UserProfiles query service"
```

Expected: commit succeeds with only query-service changes staged.

## Task 8: Move Bulk Into Its Own Milestone Service

**Files:**
- Create: `tldw_Server_API/app/core/UserProfiles/bulk_command_service.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_bulk_command_service.py`
- Modify: `tldw_Server_API/app/services/admin_profiles_service.py`
- Test: `tldw_Server_API/tests/UserProfile/test_user_profile_bulk.py`

- [ ] **Step 1: Add bulk service authorization-order test**

Create `tldw_Server_API/tests/UserProfile/test_profile_bulk_command_service.py`:

```python
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.UserProfiles.bulk_command_service import ProfileBulkCommandService


@pytest.mark.asyncio
async def test_bulk_service_filters_scope_before_reporting_targets() -> None:
    service = ProfileBulkCommandService()

    visible = await service.filter_visible_targets(
        candidate_user_ids=[1, 2, 3],
        visible_user_ids={1, 3},
    )

    assert visible == [1, 3]
```

- [ ] **Step 2: Run bulk service test to verify red**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_bulk_command_service.py -q
```

Expected: fails with `ModuleNotFoundError` for `bulk_command_service`.

- [ ] **Step 3: Add bulk service skeleton**

Create `tldw_Server_API/app/core/UserProfiles/bulk_command_service.py`:

```python
"""
Bulk UserProfiles command orchestration.
"""

from __future__ import annotations


class ProfileBulkCommandService:
    async def filter_visible_targets(
        self,
        *,
        candidate_user_ids: list[int],
        visible_user_ids: set[int],
    ) -> list[int]:
        return [int(user_id) for user_id in candidate_user_ids if int(user_id) in visible_user_ids]
```

- [ ] **Step 4: Run bulk service unit test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_bulk_command_service.py -q
```

Expected: tests pass.

- [ ] **Step 5: Move candidate filtering and per-user loop gradually**

Move only pure helper behavior first:

- `_load_bulk_user_candidates`
- confirmation threshold enforcement
- per-user `try` block shape
- diff creation defaults

Keep public `bulk_update_user_profiles` response shape unchanged. Use `ProfileCommandService` per target user after Task 6 is merged.

- [ ] **Step 6: Run bulk tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/UserProfile/test_user_profile_bulk.py \
  tldw_Server_API/tests/UserProfile/test_profile_bulk_command_service.py \
  -q
```

Expected: tests pass.

- [ ] **Step 7: Commit bulk milestone**

Run:

```bash
git add \
  tldw_Server_API/app/core/UserProfiles/bulk_command_service.py \
  tldw_Server_API/app/services/admin_profiles_service.py \
  tldw_Server_API/tests/UserProfile/test_profile_bulk_command_service.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_bulk.py
git commit -m "refactor: isolate UserProfiles bulk command flow"
```

Expected: commit succeeds with only bulk milestone files staged.

## Task 9: Targeted Postgres Verification

**Files:**
- Create: `tldw_Server_API/tests/AuthNZ_Postgres/test_user_profile_version_locking_pg.py`
- Modify: no production files unless the test exposes a backend issue

- [ ] **Step 1: Add Postgres version-locking test**

Create `tldw_Server_API/tests/AuthNZ_Postgres/test_user_profile_version_locking_pg.py`:

```python
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.UserProfiles.service import UserProfileService


pytestmark = pytest.mark.postgres


@pytest.mark.asyncio
async def test_profile_version_lock_uses_transaction_connection(isolated_test_environment):
    db_pool = isolated_test_environment["db_pool"]
    user_id = isolated_test_environment["user_id"]
    service = UserProfileService(db_pool)

    async with db_pool.transaction() as conn:
        version = await service.get_profile_version(
            user_id=int(user_id),
            db_conn=conn,
            lock_user=True,
        )

    assert version is not None
```

- [ ] **Step 2: Run targeted Postgres test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ_Postgres/test_user_profile_version_locking_pg.py -q
```

Expected: pass when Postgres fixture is available; skip only when the shared Postgres fixture reports unavailable.

- [ ] **Step 3: Run existing Postgres membership tests when touching membership executors**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ_Postgres/test_orgs_teams_pg.py \
  tldw_Server_API/tests/AuthNZ_Postgres/test_admin_org_members_pg.py \
  -q
```

Expected: pass when Postgres fixture is available; skip only when fixture reports unavailable.

- [ ] **Step 4: Commit Postgres verification**

Run:

```bash
git add tldw_Server_API/tests/AuthNZ_Postgres/test_user_profile_version_locking_pg.py
git commit -m "test: cover UserProfiles postgres version locking"
```

Expected: commit succeeds with only the new Postgres test staged.

## Task 10: Add Clean V2 Surface After Readiness Gate

**Readiness Gate Before This Task:**

All of these commands must pass before creating public v2 routes:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/UserProfile/test_user_profile_legacy_contract_characterization.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_updates.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_read.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_bulk.py \
  tldw_Server_API/tests/UserProfile/test_profile_command_service.py \
  tldw_Server_API/tests/UserProfile/test_profile_update_planner.py \
  -q
```

Expected: all tests pass.

**Files:**
- Create: `tldw_Server_API/app/api/v2/endpoints/user_profiles.py`
- Create: `tldw_Server_API/app/api/v2/router.py`
- Create: `tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py`
- Modify: main API router registration file identified during implementation

- [ ] **Step 1: Add v2 clean update contract test**

Create `tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py`:

```python
from __future__ import annotations

from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def test_v2_single_update_has_no_skipped_field(auth_headers) -> None:
    with TestClient(app) as client:
        response = client.patch(
            "/api/v2/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": "v2-contract"}
                ]
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["applied"] == ["preferences.ui.theme"]
    assert "skipped" not in payload
```

- [ ] **Step 2: Run v2 test to verify red**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py -q
```

Expected: fails with `404` for `/api/v2/users/me/profile`.

- [ ] **Step 3: Add v2 schema response in the endpoint module**

Create `tldw_Server_API/app/api/v2/endpoints/user_profiles.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import UserProfileUpdateRequest
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import (
    ProfileContractMode,
    ProfileUpdateCommand,
)

router = APIRouter()


class UserProfileV2UpdateResponse(BaseModel):
    profile_version: datetime = Field(..., description="Profile version timestamp")
    applied: list[str] = Field(default_factory=list)


@router.patch("/users/me/profile", response_model=UserProfileV2UpdateResponse)
async def update_current_user_profile_v2(
    payload: UserProfileUpdateRequest,
    _request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
) -> UserProfileV2UpdateResponse:
    if principal.user_id is None:
        raise ValueError("authenticated user principal is missing user_id")
    db_pool = await get_db_pool()
    command_service = ProfileCommandService(db_pool=db_pool)
    command = ProfileUpdateCommand(
        actor_user_id=int(principal.user_id),
        target_user_id=int(principal.user_id),
        updates=tuple((entry.key, entry.value) for entry in payload.updates),
        roles=frozenset({"user"}),
        dry_run=payload.dry_run,
        expected_profile_version=payload.profile_version,
        contract_mode=ProfileContractMode.CLEAN_V2,
    )
    result = await command_service.apply(command, db_conn=db, scope=None)
    return UserProfileV2UpdateResponse(
        profile_version=result.profile_version,
        applied=list(result.applied),
    )
```

- [ ] **Step 4: Register v2 router**

Create `tldw_Server_API/app/api/v2/router.py`:

```python
from __future__ import annotations

from fastapi import APIRouter

from tldw_Server_API.app.api.v2.endpoints import user_profiles

api_v2_router = APIRouter(prefix="/api/v2")
api_v2_router.include_router(user_profiles.router, tags=["user-profiles-v2"])
```

Find the main router registration in `tldw_Server_API/app/main.py` or router registry files, then include `api_v2_router` once. Use the same pattern as existing v1 router registration.

Preferred registration in `tldw_Server_API/app/main.py`: after the grouped v1
router registration block and before metrics route registration, add:

```python
try:
    from tldw_Server_API.app.api.v2.router import api_v2_router

    include_router_idempotent(app, api_v2_router)
except _IMPORT_EXCEPTIONS as _api_v2_err:
    logger.warning(f"Failed to include API v2 router: {_api_v2_err}")
```

If the implementation uses router groups instead, add one v2 router spec and
verify `/api/v2/users/me/profile` is available in both normal and minimal test
app modes.

- [ ] **Step 5: Run v2 contract test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py -q
```

Expected: test passes.

- [ ] **Step 6: Commit v2 surface**

Run:

```bash
git add \
  tldw_Server_API/app/api/v2/endpoints/user_profiles.py \
  tldw_Server_API/app/api/v2/router.py \
  tldw_Server_API/app/main.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py
git commit -m "feat: add UserProfiles v2 update contract"
```

Expected: commit succeeds with only v2 route files, router registration, and v2 tests staged.

## Task 11: Final Verification And Documentation

**Files:**
- Modify: `tldw_Server_API/app/core/UserProfiles/README.md`
- Modify: `Docs/Published/API-related/User_Registration_API_Documentation.md`
- Modify: Backlog task for implementation work

- [ ] **Step 1: Update UserProfiles README**

Add a short section to `tldw_Server_API/app/core/UserProfiles/README.md`:

```markdown
## Contract Refactor Notes

The profile API is organized around typed read/update commands, planner output,
and compatibility response mappers. Existing v1 routes preserve legacy response
shape. Clean v2 profile routes use atomic single-update semantics and omit
legacy single-update `skipped` fields. Bulk remains the only public profile
write surface with per-user partial reporting.
```

- [ ] **Step 2: Update public API docs**

In `Docs/Published/API-related/User_Registration_API_Documentation.md`, add a compact note near the profile update endpoint:

```markdown
Profile update compatibility:
- v1 profile update responses include `applied` and `skipped` for compatibility.
- v2 profile single-update responses are atomic all-or-reject and omit `skipped`.
- Bulk profile updates continue to report per-user partial results.
```

- [ ] **Step 3: Run focused UserProfiles tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/UserProfile \
  tldw_Server_API/tests/AuthNZ/unit/test_user_profile_update_service_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_user_profile_command_backend_selection.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Run targeted Postgres tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ_Postgres/test_user_profile_version_locking_pg.py \
  tldw_Server_API/tests/AuthNZ_Postgres/test_orgs_teams_pg.py \
  tldw_Server_API/tests/AuthNZ_Postgres/test_admin_org_members_pg.py \
  -q
```

Expected: pass when Postgres fixture is available; skip only when the fixture reports unavailable.

- [ ] **Step 5: Run Bandit on touched production scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/UserProfiles \
  tldw_Server_API/app/api/v1/endpoints/users.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_profiles.py \
  tldw_Server_API/app/services/admin_profiles_service.py \
  tldw_Server_API/app/api/v2 \
  -f json \
  -o /tmp/bandit_userprofiles_contract_refactor.json
```

Expected: zero new findings in touched production code. Fix any new finding before final commit.

- [ ] **Step 6: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 7: Commit docs and verification notes**

Run:

```bash
git add \
  tldw_Server_API/app/core/UserProfiles/README.md \
  Docs/Published/API-related/User_Registration_API_Documentation.md
git commit -m "docs: document UserProfiles contract refactor"
```

Expected: commit succeeds with only documentation files staged.

## Plan Self-Review Checklist

- Spec coverage:
  - Architecture boundaries: Tasks 2, 4, 5, 7, 8.
  - Read flow: Task 7.
  - Single update flow: Tasks 4, 5, 6.
  - Bulk flow: Task 8.
  - Contracts/errors: Tasks 2, 3, 10.
  - Effects/audit: Task 5, Task 11 verification.
  - Versioning/transactions: Tasks 5, 6, 9.
  - Migration and v2 gate: Task 10 readiness gate.
  - Testing and Bandit: Tasks 1, 3-11.
- Marker scan: no unresolved marker text remains in this plan.
- Type consistency:
  - `ProfileContractMode`, `ProfileUpdateCommand`, `UpdateMutation`, `EffectDescriptor`, and `UpdatePlan` are defined in Task 2 before use.
  - `ProfileUpdatePlanner` is defined in Task 4 before `ProfileCommandService` uses it.
  - `LegacyProfileCommandResult` is defined in Task 5 before route mapping uses it.
  - `ProfileBulkCommandService` is defined in Task 8 before bulk migration.
