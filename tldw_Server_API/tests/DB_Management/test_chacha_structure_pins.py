"""Structure pins for the CharactersRAGDB split (TASK-13950.22). Import-only, so they run in the CI contracts step."""

from __future__ import annotations

import ast
import inspect
import subprocess  # nosec B404  # standalone-import pin, fixed argv
import sys
from pathlib import Path

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


# --- ChaChaContext (TASK-13950.22 PR C1) --------------------------------------------------------

CONTEXT_SURFACE = frozenset(
    {
        "db", "backend", "backend_type", "client_id", "owner_user_id", "db_path_str", "is_memory_db",
        "code_schema_version", "schema_name",
        "transaction", "get_connection", "execute_query", "execute_many",
        "utc_now_iso", "new_uuid", "normalize_nullable_text", "map_table", "owner_filter",
        "current_db_version", "prepare_statement",
    }
)  # fmt: skip


def test_ctx_exists_on_a_db_built_without_init() -> None:
    """R19: the 55 ``__new__``-built DBs reach the context, and the class steps stay reachable."""
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    assert db.ctx.db is db  # nosec B101
    assert db.ctx is db.ctx  # nosec B101
    assert db._sqlite_linear_migration_steps()[65].__name__ == "_migrate_from_v65_to_v66"  # nosec B101


def test_context_surface_is_pinned() -> None:
    from tldw_Server_API.app.core.DB_Management.chacha.context import ChaChaContext

    members = {n for n in vars(ChaChaContext) if not n.startswith("_")}
    assert members == CONTEXT_SURFACE  # nosec B101
    for name in members:
        member = vars(ChaChaContext)[name]
        target = member.fget if isinstance(member, property) else getattr(member, "__func__", member)
        assert inspect.getdoc(target), f"{name} needs a docstring"  # nosec B101


def test_every_split_rewrite_target_is_a_context_member() -> None:
    from Helper_Scripts.chacha_split import split

    from tldw_Server_API.app.core.DB_Management.chacha.context import ChaChaContext

    for target in (*split.CTX_REWRITES.values(), *split.CTX_PUBLIC):
        assert hasattr(ChaChaContext, target), target  # nosec B101


def test_context_module_privates_are_pinned() -> None:
    from tldw_Server_API.app.core.DB_Management.chacha import context

    tree = ast.parse(Path(context.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("ChaChaNotes_DB")
        for alias in node.names
        if alias.name.startswith("_")
    }
    assert imported == set(context.PINNED_MODULE_PRIVATES)  # nosec B101


@pytest.mark.parametrize("module", ["tldw_Server_API.app.core.DB_Management.chacha.context"])
def test_new_chacha_modules_import_standalone(module: str) -> None:
    """A fresh interpreter imports the module first, which proves the ChaChaNotes_DB import cycle is safe."""
    result = subprocess.run(  # nosec B603  # fixed argv, no shell
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr  # nosec B101
