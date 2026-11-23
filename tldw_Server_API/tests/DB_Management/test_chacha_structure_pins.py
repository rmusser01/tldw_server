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
        "keyword_link_filter", "require_owner_row", "case_insensitive_order_expression", "case_insensitive_order_clause",
        "unique_text_match_sql", "is_unique_violation", "chunk_list", "canonical_json_text_v61", "note_task_v60_hash",
        "deserialize_row_fields", "runtime_schema_version",
        "note_task_v60_json_safe", "build_flashcard_search_shadow_fields", "canonicalize_legacy_task_v60",
        "normalize_postgres_catalog_expression",
    }
)  # fmt: skip


# Renamed names whose old spelling is still a DB attribute: none (a rename removes the private name).
LEGACY_FORWARDERS: frozenset[str] = frozenset()


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


def test_every_split_module_constant_and_db_rename_resolves() -> None:
    """The constants and renames ``rewire`` and ``store`` emit must exist, or the rewritten store breaks at call time."""
    from Helper_Scripts.chacha_split import split

    chacha = Path(CharactersRAGDB.__module__.replace(".", "/") + ".py").parent / "chacha"
    repo = Path(__file__).resolve().parents[3]
    for constant, module in split.CTX_MODULE_CONSTANTS.items():
        tree = ast.parse((repo / chacha / f"{module}.py").read_text(encoding="utf-8"))
        defined = {
            t.id for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)
        }
        assert constant in defined, f"{constant} is not defined in {module}"  # nosec B101
    for old, new in {**split.CTX_DB_RENAMES, **split.all_renames(repo)}.items():
        assert hasattr(CharactersRAGDB, new), f"{old} -> {new}: no such member on CharactersRAGDB"  # nosec B101
        assert old not in vars(CharactersRAGDB) or old in LEGACY_FORWARDERS, f"{old} should be gone"  # nosec B101


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


@pytest.mark.parametrize(
    "module",
    [
        "tldw_Server_API.app.core.DB_Management.chacha.context",
        "tldw_Server_API.app.core.DB_Management.chacha.schema_sql",
        "tldw_Server_API.app.core.DB_Management.chacha.sqlite_schema",
        "tldw_Server_API.app.core.DB_Management.chacha.migrations.support",
        "tldw_Server_API.app.core.DB_Management.chacha.postgres_schema",
    ],
)
def test_new_chacha_modules_import_standalone(module: str) -> None:
    """A fresh interpreter imports the module first, which proves the ChaChaNotes_DB import cycle is safe."""
    result = subprocess.run(  # nosec B603  # fixed argv, no shell
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr  # nosec B101


def test_ladder_helper_reads_callee_names_and_keywords_by_ast() -> None:
    from tldw_Server_API.tests.helpers import chacha_postgres_ladder as ladder

    def ladder_fn(self, conn, current_version):
        self._get_schema_version_postgres(conn, lock=True)
        if current_version < 7:
            postgres_schema._first(self.ctx, conn, flag=SOME_NAME)  # noqa: F821
            self._second(conn)
        helper(conn)  # noqa: F821

    calls = ladder.ladder_calls(ladder_fn)
    assert ladder.names(calls) == ["_get_schema_version_postgres", "_first", "_second", "helper"]  # nosec B101
    assert calls[0].kwargs == {"lock": True}  # nosec B101
    assert calls[1].kwargs == {"flag": "SOME_NAME"}  # nosec B101
    assert ladder.names(ladder.branch_calls(ladder_fn, 7)) == ["_first", "_second"]  # nosec B101
    assert ladder.branch_calls(ladder_fn, 8) == []  # nosec B101


# --- schema SQL constants (TASK-13950.22 PR D, R4/R20/R21) ---------------------------------------------------------------

PLACEHOLDER_CONSTANTS = ("RETIRED_SYNC_PLACEHOLDER_ASSISTANT_KIND", "RETIRED_SYNC_PLACEHOLDER_ASSISTANT_ID")
SCHEMA_SQL_MODULE = "tldw_Server_API.app.core.DB_Management.chacha.schema_sql"


def _schema_manifest() -> dict:
    from Helper_Scripts.chacha_split import split

    repo = Path(__file__).resolve().parents[3]
    return split.load_manifest(repo / "Helper_Scripts/chacha_split/manifests/t05_schema_sql.py")


def test_schema_constants_stay_class_attributes() -> None:
    """Tests and readers still say ``CharactersRAGDB._X``; the class attribute is the one object ``schema_sql`` holds."""
    import importlib

    schema_sql = importlib.import_module(SCHEMA_SQL_MODULE)
    class_names = [n for n in _schema_manifest()["constants"] if n not in PLACEHOLDER_CONSTANTS]
    assert len(class_names) == 108  # nosec B101
    for name in class_names:
        assert getattr(CharactersRAGDB, name) is getattr(schema_sql, name), name  # nosec B101


def test_placeholder_constants_are_reexported() -> None:
    import importlib

    from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB

    schema_sql = importlib.import_module(SCHEMA_SQL_MODULE)
    for name in PLACEHOLDER_CONSTANTS:
        assert getattr(ChaChaNotes_DB, name) is getattr(schema_sql, name), name  # nosec B101


def test_schema_sql_imports_nothing_from_chachanotes_db() -> None:
    import importlib

    schema_sql = importlib.import_module(SCHEMA_SQL_MODULE)
    tree = ast.parse(Path(schema_sql.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "ChaChaNotes_DB" not in (node.module or "") and all("ChaChaNotes_DB" not in a.name for a in node.names)  # nosec B101
        elif isinstance(node, ast.Import):
            assert all("ChaChaNotes_DB" not in a.name for a in node.names)  # nosec B101


def test_versions_stay_literal() -> None:
    """The schema versions are read by the registry pins and the migration ladders; they are not moved or computed."""
    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 85  # nosec B101
    assert CharactersRAGDB._POSTGRES_SCHEMA_VERSION == 91  # nosec B101
    assert CharactersRAGDB._SCHEMA_NAME == "rag_char_chat_schema"  # nosec B101
