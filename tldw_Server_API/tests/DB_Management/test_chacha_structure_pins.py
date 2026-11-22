"""Structure pins for the CharactersRAGDB split (TASK-13950.22). Import-only, so they run in the CI contracts step."""

from __future__ import annotations

import inspect

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import ConversationResumeStore
from tldw_Server_API.app.core.DB_Management.chacha.moodboard_sync_store import MoodboardSyncStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit

# {store class: names retired from CharactersRAGDB, callers use db.<store attr>.<name>}
RETIRED = {
    MoodboardSyncStore: (
        "bind_local_moodboard_graph_to_dataset",
        "bind_local_studio_graph_to_dataset",
        "page_moodboards_for_sync_bootstrap",
        "page_moodboard_placements_for_sync_bootstrap",
        "page_studio_documents_for_sync_bootstrap",
    ),
    ConversationResumeStore: (
        "append_idempotent_user_message",
        "prepare_roleplay_resume_generation",
        "commit_roleplay_resume_generation",
        "put_behavior_snapshot",
        "get_conversation_behavior_snapshot",
        "get_roleplay_resume_state",
    ),
}


def test_retired_names_are_gone() -> None:
    for store, names in RETIRED.items():
        for name in names:
            assert name not in vars(CharactersRAGDB), name  # nosec B101
            assert inspect.isfunction(vars(store)[name]), name  # nosec B101
    # Still delegated on purpose: the module-level assignments keep these two bound.
    assert "resolve_moodboard_compatibility_dataset_id" in vars(CharactersRAGDB)  # nosec B101
