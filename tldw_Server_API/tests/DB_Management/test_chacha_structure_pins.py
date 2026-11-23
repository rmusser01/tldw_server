"""Structure pins for the CharactersRAGDB split (TASK-13950.22). Import-only, so they run in the CI contracts step."""

from __future__ import annotations

import ast
import inspect
import subprocess  # nosec B404  # standalone-import pin, fixed argv
import sys
import textwrap
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
        "code_schema_version", "postgres_schema_version", "schema_name",
        "transaction", "get_connection", "execute_query", "execute_many",
        "run_sqlite_migration_step", "utc_now_iso", "new_uuid", "normalize_nullable_text", "map_table", "owner_filter",
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
        "tldw_Server_API.app.core.DB_Management.chacha.schema_bootstrap",
        "tldw_Server_API.app.core.DB_Management.chacha.migrations.sqlite_steps",
        "tldw_Server_API.app.core.DB_Management.chacha.migrations.postgres_steps",
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


def test_ladder_helper_expands_a_registry_loop_into_the_registry_names_in_key_order() -> None:
    from tldw_Server_API.app.core.DB_Management.chacha.migrations import postgres_steps
    from tldw_Server_API.tests.helpers import chacha_postgres_ladder as ladder

    def ladder_fn(self, conn, current_version, target_version):
        first(conn)  # noqa: F821
        for source, step_name in postgres_steps.POSTGRES_STEPS.items():
            if current_version <= source < target_version:
                getattr(postgres_steps, step_name)(self.ctx, conn)
        last(conn)  # noqa: F821

    registry = list(postgres_steps.POSTGRES_STEPS.values())
    assert ladder.names(ladder.ladder_calls(ladder_fn)) == ["first", *registry, "last"]  # nosec B101
    # The real initializer has the fast-path loop and the full-path loop, so the 72 to 91 steps appear twice, in key order.
    real = ladder.names(ladder.ladder_calls(ladder.postgres_init_function()))
    first = real.index(registry[0])
    assert real[first : first + len(registry)] == registry  # nosec B101
    assert real.count(registry[0]) == 2  # nosec B101


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


# --- migration registries (TASK-13950.22 PR G, R3/R12/R21) --------------------------------------------------------------
# Frozen from the unmoved tree by `split.py registry-snapshot` (dev ee0888b51b; the snapshot matched Appendix C). A wrong key, name or
# order here would silently corrupt an upgrade, so a change to either list is a deliberate edit to this file and to the version constant.
FROZEN_SQLITE = [
    (4, '_migrate_from_v4_to_v5'),
    (5, '_migrate_from_v5_to_v6'),
    (6, '_migrate_from_v6_to_v7'),
    (7, '_migrate_from_v7_to_v8'),
    (8, '_migrate_from_v8_to_v9'),
    (9, '_migrate_from_v9_to_v10'),
    (10, '_migrate_from_v10_to_v11'),
    (11, '_migrate_from_v11_to_v12'),
    (12, '_migrate_from_v12_to_v13'),
    (13, '_migrate_from_v13_to_v14'),
    (14, '_migrate_from_v14_to_v15'),
    (15, '_migrate_from_v15_to_v16'),
    (16, '_migrate_from_v16_to_v17'),
    (17, '_migrate_from_v17_to_v18'),
    (18, '_migrate_from_v18_to_v19'),
    (19, '_migrate_from_v19_to_v20'),
    (20, '_migrate_from_v20_to_v21'),
    (21, '_migrate_from_v21_to_v22'),
    (22, '_migrate_from_v22_to_v23'),
    (23, '_migrate_from_v23_to_v24'),
    (24, '_migrate_from_v24_to_v25'),
    (25, '_migrate_from_v25_to_v26'),
    (26, '_migrate_from_v26_to_v27'),
    (27, '_migrate_from_v27_to_v28'),
    (28, '_migrate_from_v28_to_v29'),
    (29, '_migrate_from_v29_to_v30'),
    (30, '_migrate_from_v30_to_v31'),
    (31, '_migrate_from_v31_to_v32'),
    (32, '_migrate_from_v32_to_v33'),
    (33, '_migrate_from_v33_to_v34'),
    (34, '_migrate_from_v34_to_v35'),
    (35, '_migrate_from_v35_to_v36'),
    (36, '_migrate_from_v36_to_v37'),
    (37, '_migrate_from_v37_to_v38'),
    (38, '_migrate_from_v38_to_v39'),
    (39, '_migrate_from_v39_to_v40'),
    (40, '_migrate_from_v40_to_v41'),
    (41, '_migrate_from_v41_to_v42'),
    (42, '_migrate_from_v42_to_v43'),
    (43, '_migrate_from_v43_to_v44'),
    (44, '_migrate_from_v44_to_v45'),
    (45, '_migrate_from_v45_to_v46'),
    (46, '_migrate_from_v46_to_v47'),
    (47, '_migrate_from_v47_to_v48'),
    (48, '_migrate_from_v48_to_v49'),
    (49, '_migrate_from_v49_to_v50'),
    (50, '_migrate_from_v50_to_v51'),
    (51, '_migrate_from_v51_to_v52'),
    (52, '_migrate_from_v52_to_v53'),
    (53, '_migrate_from_v53_to_v54'),
    (54, '_migrate_from_v54_to_v55'),
    (55, '_migrate_from_v55_to_v56'),
    (56, '_migrate_from_v56_to_v57'),
    (57, '_migrate_from_v57_to_v58'),
    (58, '_migrate_from_v58_to_v59'),
    (59, '_migrate_from_v59_to_v60_sqlite'),
    (60, '_migrate_from_v60_to_v61_sqlite'),
    (61, '_migrate_from_v61_to_v62_sqlite'),
    (62, '_migrate_from_v62_to_v63_sqlite'),
    (63, '_migrate_from_v63_to_v64_sqlite'),
    (64, '_migrate_from_v64_to_v65'),
    (65, '_migrate_from_v65_to_v66'),
    (66, '_migrate_from_v66_to_v67'),
    (67, '_migrate_from_v67_to_v68'),
    (68, '_migrate_from_v68_to_v69'),
    (69, '_migrate_from_v69_to_v70'),
    (70, '_migrate_from_v70_to_v71'),
    (71, '_migrate_from_v71_to_v72'),
    (72, '_migrate_from_v72_to_v73_persona_companion'),
    (73, '_migrate_from_v73_to_v74'),
    (74, '_migrate_from_v74_to_v75'),
    (75, '_migrate_from_v75_to_v76'),
    (76, '_migrate_from_v76_to_v77'),
    (77, '_migrate_from_v77_to_v78'),
    (78, '_migrate_from_v78_to_v79'),
    (79, '_migrate_from_v79_to_v80'),
    (80, '_migrate_from_v80_to_v81'),
    (81, '_migrate_from_v81_to_v82_sqlite'),
    (82, '_migrate_from_v82_to_v83_sqlite'),
    (83, '_migrate_from_v83_to_v84_sqlite'),
    (84, 'migrate_sqlite_transcripts'),
]
FROZEN_POSTGRES = [
    (72, '_migrate_from_v72_to_v73_postgres'),
    (73, '_migrate_from_v73_to_v74_postgres'),
    (74, '_migrate_from_v74_to_v75_postgres'),
    (75, '_migrate_from_v75_to_v76_postgres'),
    (76, '_migrate_from_v76_to_v77_persona_companion_postgres'),
    (77, '_migrate_from_v77_to_v78_postgres'),
    (78, '_migrate_from_v78_to_v79_postgres'),
    (79, '_migrate_from_v79_to_v80_postgres'),
    (80, '_migrate_from_v80_to_v81_postgres'),
    (81, '_migrate_from_v81_to_v82_postgres'),
    (82, '_migrate_from_v82_to_v83_postgres'),
    (83, '_migrate_from_v83_to_v84_postgres'),
    (84, '_migrate_from_v84_to_v85_postgres'),
    (85, '_migrate_from_v85_to_v86_postgres'),
    (86, '_migrate_from_v86_to_v87_postgres'),
    (87, '_migrate_from_v87_to_v88_postgres'),
    (88, '_migrate_from_v88_to_v89_postgres'),
    (89, '_migrate_from_v89_to_v90_postgres'),
    (90, '_migrate_from_v90_to_v91_postgres'),
]


def _registries():
    from tldw_Server_API.app.core.DB_Management.chacha.migrations import postgres_steps, sqlite_steps

    return sqlite_steps, postgres_steps


def test_registries_match_dev_before_the_move() -> None:
    sqlite_steps, postgres_steps = _registries()
    assert list(sqlite_steps.SQLITE_STEPS.items()) == FROZEN_SQLITE  # nosec B101
    assert list(postgres_steps.POSTGRES_STEPS.items()) == FROZEN_POSTGRES  # nosec B101


def test_every_registry_name_resolves() -> None:
    sqlite_steps, postgres_steps = _registries()
    for module, registry in ((sqlite_steps, sqlite_steps.SQLITE_STEPS), (postgres_steps, postgres_steps.POSTGRES_STEPS)):
        for source, name in registry.items():
            assert getattr(module, name).__name__ == name, (source, name)  # nosec B101


def test_versions_unchanged() -> None:
    sqlite_steps, postgres_steps = _registries()
    assert max(sqlite_steps.SQLITE_STEPS) + 1 == CharactersRAGDB._CURRENT_SCHEMA_VERSION == 85  # nosec B101
    assert max(postgres_steps.POSTGRES_STEPS) + 1 == CharactersRAGDB._POSTGRES_SCHEMA_VERSION == 91  # nosec B101


def test_class_steps_keep_names() -> None:
    """The class still builds the SQLite step map (R19's ``__new__`` path); each step keeps its dev ``__name__``."""
    steps = CharactersRAGDB.__new__(CharactersRAGDB)._sqlite_linear_migration_steps()
    assert [(version, step.__name__) for version, step in steps.items()] == FROZEN_SQLITE  # nosec B101


def test_no_migrate_name_left_on_the_class() -> None:
    assert [name for name in vars(CharactersRAGDB) if name.startswith("_migrate")] == []  # nosec B101


def test_module_patch_reaches_registry_and_direct_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """One patch point per migration (R12): the module attribute serves the class's step map and a resolved registry name."""
    sqlite_steps, _ = _registries()
    calls: list[object] = []

    def patched_step(ctx: object, conn: object) -> None:
        calls.append(conn)

    monkeypatch.setattr(sqlite_steps, "_migrate_from_v49_to_v50", patched_step)
    sentinel = object()
    class_step = CharactersRAGDB.__new__(CharactersRAGDB)._sqlite_linear_migration_steps()[49]
    assert class_step.__wrapped__ is patched_step  # nosec B101
    class_step(sentinel)
    getattr(sqlite_steps, sqlite_steps.SQLITE_STEPS[49])(None, sentinel)
    assert calls == [sentinel, sentinel]  # nosec B101


# --- schema bootstrap (TASK-13950.22 PR H, R5/R29/R34) -------------------------------------------------------------------


def test_bootstrap_reads_versions_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests patch the version on the class 139 times; the context reads it per call, so a patch reaches the moved bootstrap."""
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    assert db.ctx.postgres_schema_version == 91  # nosec B101
    monkeypatch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 123)
    assert db.ctx.postgres_schema_version == 123  # nosec B101
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 99)
    assert db.ctx.code_schema_version == 99  # nosec B101


def test_runtime_schema_version_is_public_with_a_default() -> None:
    """R29: the instance attribute the bootstrap writes is public, so the context pins no write reach for it."""
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    assert db.runtime_schema_version is None  # nosec B101
    assert db.ctx.runtime_schema_version is None  # nosec B101
    db.runtime_schema_version = 81
    assert db.ctx.runtime_schema_version == 81  # nosec B101
    assert [n for n in dir(db) if n.endswith("runtime_schema_version")] == ["runtime_schema_version"]  # nosec B101  # the private spelling is gone


def test_postgres_init_repairs_legacy_markers_in_order() -> None:
    """The three repairs of a legacy PostgreSQL marker run in this order (R34): the v86 dictionary bridge, then the
    v68 history projections check, then the v73/74 stamp back to 72. AST only, so it needs no database."""
    from tldw_Server_API.tests.helpers import chacha_postgres_ladder as ladder

    tree = ast.parse(textwrap.dedent(inspect.getsource(ladder.postgres_init_function())))

    def _if_at(*needles: str) -> ast.If:
        found = [n for n in ast.walk(tree) if isinstance(n, ast.If) and all(s in ast.unparse(n.test) for s in needles)]
        assert len(found) == 1, (needles, len(found))  # nosec B101
        return found[0]

    def _calls(node: ast.If, callee: str, last_arg: int | None = None) -> list[ast.Call]:
        return [
            c for c in ast.walk(node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == callee
            and (last_arg is None or (isinstance(c.args[-1], ast.Constant) and c.args[-1].value == last_arg))
        ]

    bridge = _if_at("current_version == 86", "target_version >= 89")
    history = _if_at("current_version == 68", "conversation_history_projections")
    stamp = _if_at("current_version in (73, 74)", "assistant_defaults_explicit_none")
    assert _calls(bridge, "_bridge_legacy_dictionary_v86_postgres")  # nosec B101
    assert _calls(history, "_set_schema_version_postgres", 67)  # nosec B101
    assert _calls(stamp, "_set_schema_version_postgres", 72)  # nosec B101
    assert bridge.lineno < history.lineno < stamp.lineno  # nosec B101
