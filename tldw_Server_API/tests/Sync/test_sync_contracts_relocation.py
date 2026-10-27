"""The Sync contracts moved to core.Sync_Contracts; the old core.Sync.v2 paths are the same modules (TASK-13950.16)."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit

MOVED = ("errors", "models", "mutation_group_validation", "notes_moodboard_studio_readiness", "notes_task_readiness",
         "notes_provenance_contract", "notes_link", "notes_link_contract", "notes_moodboard_studio_contract",
         "notes_task_contract", "notes_organization", "personal_context_ongoing_contract", "attachment_refs_v2")


@pytest.mark.parametrize("name", MOVED)
def test_the_old_sync_v2_path_is_the_moved_module(name: str) -> None:
    old = importlib.import_module(f"tldw_Server_API.app.core.Sync.v2.{name}")
    assert old is importlib.import_module(f"tldw_Server_API.app.core.Sync_Contracts.{name}")


def test_the_sync_v2_package_reexports_the_moved_classes() -> None:
    from tldw_Server_API.app.core.Sync.v2 import SyncEnvelope, SyncV2Error
    from tldw_Server_API.app.core.Sync_Contracts.errors import SyncV2Error as MovedError
    from tldw_Server_API.app.core.Sync_Contracts.models import SyncEnvelope as MovedEnvelope
    assert (SyncEnvelope, SyncV2Error) == (MovedEnvelope, MovedError)
