"""The Notes suggestion authority fence: what it still refuses after being relaxed.

prepare_notes_suggestion_authority reserves the Notes suggestion canonical scope for
one dataset. Two of its five checks are the security boundary -- the dataset must be
owned by the caller and must be personally scoped -- and those are unconditional. The
other two assert the dataset is the chatbook DEFAULT, and those are now conditional on
``require_default``.

Why: Sync_DB.personal_context_bootstrap_transaction deliberately selects
``bound_rows[0]`` as the authority whatever its markers, and carries an explicit
``require_chatbook_default`` flag that it relaxes for exactly that case (2026-09-04).
The fence call was added to _bind_personal_context_dataset 13 days later
(f0536ee5cf, 2026-09-17) and demanded the default markers unconditionally, refusing a
legitimately bound non-default authority. Six tests failed, all reporting the unhelpful
``personal_context_snapshot_unavailable`` because profile.py maps any other
SyncStoreError to it.

These tests exist so the relaxation cannot quietly become a hole: the owner and scope
checks are pinned separately from the default-marker checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service


@dataclass
class _Dataset:
    dataset_id: str
    owner_user_id: str
    scope_type: str
    metadata: dict[str, Any]


class _Store:
    def __init__(self, dataset: _Dataset | None) -> None:
        self._dataset = dataset

    def get_dataset(self, dataset_id: str) -> _Dataset | None:
        if self._dataset is not None and self._dataset.dataset_id == dataset_id:
            return self._dataset
        return None


def _service(dataset: _Dataset | None) -> SyncV2Service:
    service = SyncV2Service.__new__(SyncV2Service)  # the fence needs no other wiring
    service.store = _Store(dataset)
    service.materializers = {}
    return service


_DEFAULT_MARKERS = {"default_personal": True, "client_family": "chatbook"}
_BOUND_ONLY = {"personal_context": {"profile_id": "p1", "authority_id": "a1"}}


@pytest.mark.parametrize("require_default", [True, False])
def test_another_users_dataset_is_always_refused(require_default: bool) -> None:
    """The cross-user boundary does not depend on the relaxation."""
    dataset = _Dataset("ds-1", "victim", "personal", dict(_DEFAULT_MARKERS))
    with pytest.raises(SyncStoreError, match="owned default dataset"):
        _service(dataset).prepare_notes_suggestion_authority(
            user_id="attacker", dataset=dataset, require_default=require_default
        )


@pytest.mark.parametrize("require_default", [True, False])
def test_a_non_personal_scope_is_always_refused(require_default: bool) -> None:
    dataset = _Dataset("ds-1", "user-1", "workspace", dict(_DEFAULT_MARKERS))
    with pytest.raises(SyncStoreError, match="owned default dataset"):
        _service(dataset).prepare_notes_suggestion_authority(
            user_id="user-1", dataset=dataset, require_default=require_default
        )


@pytest.mark.parametrize("require_default", [True, False])
def test_a_dataset_the_store_cannot_see_is_always_refused(require_default: bool) -> None:
    dataset = _Dataset("ds-1", "user-1", "personal", dict(_DEFAULT_MARKERS))
    with pytest.raises(SyncStoreError, match="owned default dataset"):
        _service(None).prepare_notes_suggestion_authority(
            user_id="user-1", dataset=dataset, require_default=require_default
        )


def test_default_markers_are_required_by_default() -> None:
    """The strict behaviour is still the default, so a new caller gets the fence."""
    dataset = _Dataset("ds-1", "user-1", "personal", dict(_BOUND_ONLY))
    with pytest.raises(SyncStoreError, match="owned default dataset"):
        _service(dataset).prepare_notes_suggestion_authority(
            user_id="user-1", dataset=dataset
        )


def test_a_bound_non_default_authority_is_accepted_when_not_requiring_default() -> None:
    """The case the bootstrap transaction deliberately selects."""
    dataset = _Dataset("ds-1", "user-1", "personal", dict(_BOUND_ONLY))
    _service(dataset).prepare_notes_suggestion_authority(
        user_id="user-1", dataset=dataset, require_default=False
    )


def test_the_chatbook_default_is_accepted_either_way() -> None:
    dataset = _Dataset("ds-1", "user-1", "personal", dict(_DEFAULT_MARKERS))
    for require_default in (True, False):
        _service(dataset).prepare_notes_suggestion_authority(
            user_id="user-1", dataset=dataset, require_default=require_default
        )


def test_the_caller_store_is_preferred_over_the_service_store() -> None:
    """Bootstrap creates the dataset inside a transaction the service store cannot see.

    On PostgreSQL that uncommitted row is invisible to any other connection, so the
    fence read None and refused a dataset that plainly existed. SQLite hid it.
    """
    dataset = _Dataset("ds-1", "user-1", "personal", dict(_DEFAULT_MARKERS))
    service = _service(None)  # the service store cannot see it
    service.prepare_notes_suggestion_authority(
        user_id="user-1", dataset=dataset, store=_Store(dataset)
    )
