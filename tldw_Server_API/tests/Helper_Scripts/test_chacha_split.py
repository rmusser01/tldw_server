"""Helper_Scripts/chacha_split/split.py ``callers`` on seeded tmp trees (TASK-13950.22)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from Helper_Scripts.chacha_split import split

pytestmark = pytest.mark.unit

DB = "tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py"
APP = "tldw_Server_API/app/api/x.py"
TEST = "tldw_Server_API/tests/test_x.py"
DB_SRC = """
class CharactersRAGDB:
    def __init__(self):
        self.s_store = object()

    def run(self):
        return self.get_x()
"""


CLEAN_DB = DB_SRC.replace("self.get_x()", "self.s_store.get_x()")


def _seed(root: Path, files: dict[str, str] | None = None, db: str = DB_SRC) -> Path:
    for key, text in {DB: db, **(files or {})}.items():
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    return root


def _callers(root: Path, *extra: str) -> int:
    return split.main(["callers", "--store", "s_store", "--names", "get_x", "--root", str(root), *extra])


def _text(root: Path, key: str) -> str:
    return (root / key).read_text(encoding="utf-8")


def test_attribute_call_is_rewritten_through_the_store(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    (root / APP).parent.mkdir(parents=True)
    (root / APP).write_text("def f(db):\n    return db.get_x(1)\n")
    assert _callers(root) == 0
    assert "db.s_store.get_x(1)" in _text(root, APP)


def test_store_receiver_is_left_alone(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    (root / APP).parent.mkdir(parents=True)
    source = "def f(db):\n    return db.s_store.get_x()\n"
    (root / APP).write_text(source)
    assert _callers(root) == 0
    assert _text(root, APP) == source


def test_probe_is_rewritten_through_the_store(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    (root / APP).parent.mkdir(parents=True)
    (root / APP).write_text('a = getattr(db, "get_x", None)\nb = hasattr(db, "get_x")\n')
    assert _callers(root) == 0
    assert _text(root, APP) == (
        'a = getattr(getattr(db, "s_store", None), "get_x", None)\n'
        'b = hasattr(getattr(db, "s_store", None), "get_x")\n'
    )


def test_class_self_callers_are_rewritten(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    assert _callers(root) == 0
    assert "return self.s_store.get_x()" in _text(root, DB)


def test_collision_name_is_listed_not_rewritten(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _seed(tmp_path, {"tldw_Server_API/app/api/other.py": "class Other:\n    def get_x(self): ...\n", APP: "def f(db):\n    return db.get_x()\n"})
    before = _text(root, APP)
    assert _callers(root) == 0
    assert _text(root, APP) == before
    assert "collision get_x" in capsys.readouterr().out


def test_unusual_receiver_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _seed(tmp_path)
    (root / APP).parent.mkdir(parents=True)
    (root / APP).write_text("def f(thing):\n    return thing.get_x()\n")
    assert _callers(root) == 0
    assert "unusual receiver" in capsys.readouterr().out


def test_check_flags_a_fake_without_the_store_attribute(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _seed(tmp_path, {TEST: "class FakeDB:\n    def get_x(self): ...\n"}, db=CLEAN_DB)
    assert _callers(root, "--check") == 1
    assert "FakeDB" in capsys.readouterr().out


def test_check_flags_assignment_and_class_receiver_references(tmp_path: Path) -> None:
    sources = {
        "assign": "def f(db):\n    db.get_x = print\n",
        "class_ref": "ref = CharactersRAGDB.get_x\n",
        "patch_string": 'p = "m.CharactersRAGDB.get_x"\n',
        "probe": 'ok = hasattr(db, "get_x")\nfn = getattr(db, "get_x", None)\n',
    }
    for label, source in sources.items():
        root = _seed(tmp_path / label, {TEST: source}, db=CLEAN_DB)
        assert _callers(root, "--check") == 1, label


def test_check_passes_when_nothing_is_left(tmp_path: Path) -> None:
    root = _seed(
        tmp_path,
        {TEST: "class Fake:\n    s_store = None\n    def get_x(self): ...\n\ndef f(db):\n    db.s_store.get_x()\n"},
    )
    assert _callers(root, "--check") == 1  # DB_SRC still calls self.get_x()
    assert _callers(root) == 0
    assert _callers(root, "--check") == 0


def test_tests_lists_every_referencing_file_even_after_the_rewrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    other = "tldw_Server_API/tests/test_other.py"
    root = _seed(tmp_path, {TEST: "def f(db):\n    db.get_x()\n", other: "def g(db):\n    db.s_store.get_x()\n", "tldw_Server_API/tests/test_none.py": "x = 1\n"})
    assert _callers(root, "--tests") == 0
    first = capsys.readouterr().out.splitlines()
    assert _callers(root, "--tests") == 0
    second = capsys.readouterr().out.splitlines()
    assert {TEST, other} <= set(first) and {TEST, other} <= set(second)
    assert "tldw_Server_API/tests/test_none.py" not in second


PATCH_SOURCE = '''
import inspect
from types import SimpleNamespace
from unittest.mock import patch


def t1(monkeypatch, db):
    monkeypatch.setattr(db, "_x", f)


def t2():
    with patch.object(CharactersRAGDB, "_x"):
        pass


@patch.object(CharactersRAGDB, "_x")
def t3(m):
    pass


def t4(db):
    db._x = f


def t5(db):
    type(db)._x = f


fake = SimpleNamespace(_x=f)
p = patch("m.CharactersRAGDB._x")
src = inspect.getsource(CharactersRAGDB._x)
other = SimpleNamespace(_y=f)


def t6(db):
    db._y = f
    db.ctx._x = f
'''


def _patches(root: Path, *extra: str) -> int:
    return split.main(["patches", "--names", "_x", "--root", str(root), *extra])


def test_patches_finds_each_patch_form(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _seed(tmp_path, {TEST: PATCH_SOURCE})
    assert _patches(root) == 1
    out = capsys.readouterr().out
    for kind in ("setattr", "patch.object", "assign", "kwarg", "string", "getsource"):
        assert kind in out, kind
    assert len([line for line in out.splitlines() if line.startswith(TEST)]) == 8
    assert "_y" not in out


def test_patches_is_empty_on_a_clean_tree(tmp_path: Path) -> None:
    assert _patches(_seed(tmp_path, {TEST: "def t(db):\n    db.ctx.utc_now_iso = f\n"})) == 0


CHECK_PATCHES = {
    "setattr": 'def t(monkeypatch, db):\n    monkeypatch.setattr(db, "get_x", f)\n',
    "patch_object": 'def t(db):\n    with mock.patch.object(db, "get_x"):\n        pass\n',
    "patch_string": 'def t():\n    patch("m.CharactersRAGDB.get_x")\n',
}


@pytest.mark.parametrize("label", sorted(CHECK_PATCHES))
def test_check_flags_a_string_patch_of_a_retired_name(tmp_path: Path, label: str) -> None:
    root = _seed(tmp_path, {TEST: CHECK_PATCHES[label]}, db=CLEAN_DB)
    assert _callers(root, "--check") == 1


def test_check_accepts_a_string_patch_on_the_store(tmp_path: Path) -> None:
    source = (
        'def t(monkeypatch, db):\n    monkeypatch.setattr(db.s_store, "get_x", f)\n'
        '    mock.patch.object(db.s_store, "get_x")\n    mock.patch.object(SStore, "get_x")\n'
        '    store = db.s_store\n    mock.patch.object(store, "get_x")\n'
    )
    assert _callers(_seed(tmp_path, {TEST: source}, db=CLEAN_DB), "--check") == 0


STORE_SRC = '''
class S:
    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def f(self):
        a = self._db._normalize_nullable_text(1)
        b = self._db.execute_query("q")
        c = self._db.native_forks.x()
        d = self._db._ALLOWED_CONVERSATION_STATES
        e = self._db._ensure_message_metadata_table()
        g = self._db
        return self._db.backend_type
'''
STORE = "tldw_Server_API/app/core/DB_Management/chacha/s.py"


def test_rewire_maps_each_kind_of_db_reach(tmp_path: Path) -> None:
    root = _seed(tmp_path, {STORE: STORE_SRC})
    assert split.main(["rewire", "--store", STORE, "--root", str(root)]) == 0
    out = _text(root, STORE)
    assert "self._ctx = ctx" in out  # the constructor
    assert "a = self._ctx.normalize_nullable_text(1)" in out  # CTX_REWRITES
    assert "b = self._ctx.execute_query(" in out  # CTX_PUBLIC
    assert "c = self._ctx.db.native_forks.x()" in out  # another store, public
    assert "d = _ALLOWED_CONVERSATION_STATES" in out  # a constant that became a module constant
    assert "e = self._ctx.db.ensure_message_metadata_table()" in out  # the one rename
    assert "g = self._ctx.db\n" in out  # bare receiver
    assert "return self._ctx.backend_type" in out


def test_rewire_fails_closed_on_an_unmapped_private(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = STORE_SRC + "\n    def h(self):\n        return self._db._mystery()\n"
    root = _seed(tmp_path, {STORE: src})
    assert split.main(["rewire", "--store", STORE, "--root", str(root)]) == 1
    assert "_mystery" in capsys.readouterr().out
    assert _text(root, STORE) == textwrap.dedent(src)


# --- store mode, verify, diffpairs, check, tests (TASK-13950.22 PR C2) ----------------------------

import ast  # noqa: E402
import subprocess  # noqa: E402  # nosec B404  # seeded tmp git repos, fixed argv

from Helper_Scripts.chacha_split import diffpairs  # noqa: E402

MANIFEST_PATH = "Helper_Scripts/chacha_split/manifests/t99_s.py"
S_STORE = "tldw_Server_API/app/core/DB_Management/chacha/s_store.py"
STORE_DB = '''
import time


class CharactersRAGDB:
    def __init__(self):
        self.s_store = object()

    def keep(self):
        return self._moved(1)

    def _moved(self, a):
        """Doc."""
        if self.backend_type == 1:
            return self._map_table_for_backend(a)
        return time.time() + self.client_id + self._renamed(a) + len(str(self))

    def _renamed(self, a):
        return a

    def tick(self):
        return time.time()
'''
STORE_MODULE = """
class SStore:
    def __init__(self, ctx):
        self._ctx = ctx

    def other(self):
        return self._ctx.db._moved(2)
"""
MANIFEST_SRC = (
    'MANIFEST = {"task": "T99", "mode": "store", "source": "%s", "target": "%s", "alias": None,\n'
    '            "store_class": "SStore", "store_attr": "s_store", "names": ["_moved"], "shims": [],\n'
    '            "rewrites": {}, "renames": {"_renamed": "renamed"}, "structural": []}\n' % (DB, S_STORE)
)
GIT = ["git", "-c", "user.name=chacha-split-test", "-c", "user.email=chacha-split-test@example.invalid", "-c", "commit.gpgsign=false"]


def _git(root: Path, *args: str) -> None:
    subprocess.run([*GIT, *args], cwd=root, check=True, capture_output=True)  # nosec B603 B607


def _store_tree(root: Path, manifest: str = MANIFEST_SRC, files: dict[str, str] | None = None) -> Path:
    _seed(root, {S_STORE: STORE_MODULE, MANIFEST_PATH: manifest, **(files or {})}, db=STORE_DB)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _store(root: Path) -> int:
    return split.main(["store", str(root / MANIFEST_PATH), "--root", str(root)])


def _verify(root: Path) -> int:
    return split.main(["verify", str(root / MANIFEST_PATH), "--base", "HEAD", "--root", str(root)])


def test_store_moves_a_method_and_rewrites_its_receivers(tmp_path: Path) -> None:
    root = _store_tree(tmp_path, files={TEST: "def f(db):\n    return db._moved(3) + db._renamed(1)\n"})
    assert _store(root) == 0
    store = _text(root, S_STORE)
    assert "def _moved(self, a):" in store
    assert "self._ctx.backend_type == 1" in store and "self._ctx.map_table(a)" in store
    assert "self._ctx.client_id" in store and "self._ctx.db.renamed(a)" in store  # a rename
    assert "len(str(self._ctx.db))" in store  # a bare self
    assert "import time" in store
    assert "self._moved(2)" in store and "_ctx.db._moved" not in store  # the store calls itself
    db = _text(root, DB)
    assert "def _moved" not in db and "self.s_store._moved(1)" in db
    assert "def renamed(self, a)" in db and "_renamed" not in db
    assert "db.s_store._moved(3) + db.renamed(1)" in _text(root, TEST)


def test_store_fails_closed_on_an_unmapped_private(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _store_tree(tmp_path)
    (root / DB).write_text(STORE_DB.replace("self._renamed(a)", "self._mystery(a)"))
    before = _text(root, S_STORE)
    assert _store(root) == 1
    assert "_mystery" in capsys.readouterr().out
    assert _text(root, S_STORE) == before


def test_verify_passes_on_tool_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _store_tree(tmp_path)
    assert _store(root) == 0
    assert _verify(root) == 0
    assert "moved 1 identical" in capsys.readouterr().out


def test_verify_fails_on_a_hand_edit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _store_tree(tmp_path)
    assert _store(root) == 0
    (root / S_STORE).write_text(_text(root, S_STORE).replace("time.time()", "time.time() + 1"))
    assert _verify(root) == 1
    assert "_moved" in capsys.readouterr().out


def test_verify_fails_on_an_edited_remaining_def_unless_structural(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _store_tree(tmp_path)
    assert _store(root) == 0
    (root / DB).write_text(_text(root, DB).replace("return self.s_store._moved(1)", "return 5"))
    assert _verify(root) == 1
    assert "keep" in capsys.readouterr().out
    (root / MANIFEST_PATH).write_text(MANIFEST_SRC.replace('"structural": []', '"structural": ["keep"]'))
    assert _verify(root) == 0
    assert "structural ['keep']" in capsys.readouterr().out


def test_manifest_is_read_without_import(tmp_path: Path) -> None:
    path = tmp_path / "t98_bad.py"
    path.write_text('MANIFEST = {"names": [__import__("os").getcwd()]}\n')
    with pytest.raises(SystemExit, match="literal"):
        split.load_manifest(path)


def test_diffpairs_lists_receiver_rewrites_and_fails_elsewhere(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _store_tree(tmp_path)
    assert _store(root) == 0
    args = [str(root / MANIFEST_PATH), "--base", "HEAD", "--root", str(root)]
    assert diffpairs.main(args) == 0
    out = capsys.readouterr().out
    assert "self._map_table_for_backend  ->  self._ctx.map_table" in out and "DISALLOWED" not in out
    (root / S_STORE).write_text(_text(root, S_STORE).replace("== 1", "== 2"))
    assert diffpairs.main(args) == 1


def test_diffpairs_shares_no_code_with_split() -> None:
    tree = ast.parse(Path(diffpairs.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "split" not in (node.module or "").split(".")[-1:] and all(a.name != "split" for a in node.names)
        if isinstance(node, ast.Import):
            assert all("split" not in a.name.split(".")[-1:] for a in node.names)


FUNCTIONS_MANIFEST = (
    'MANIFEST = {"task": "T99", "mode": "functions", "source": "%s", "target": "%s", "alias": "mod",\n'
    '            "store_class": None, "store_attr": None, "names": ["_helper"], "shims": [],\n'
    '            "rewrites": {}, "renames": {}, "structural": []}\n' % (DB, "tldw_Server_API/app/core/DB_Management/chacha/mod.py")
)
MOD = "tldw_Server_API/app/core/DB_Management/chacha/mod.py"


def _functions_tree(root: Path, base_def: str, head_def: str) -> list[str]:
    _seed(root, {MANIFEST_PATH: FUNCTIONS_MANIFEST, MOD: head_def}, db="class CharactersRAGDB:\n" + base_def)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return [str(root / MANIFEST_PATH), "--base", "HEAD", "--root", str(root)]


def test_diffpairs_accepts_a_static_move(tmp_path: Path) -> None:
    base = "    @staticmethod\n    def _helper(x):\n        return x + 1\n"
    assert diffpairs.main(_functions_tree(tmp_path / "ok", base, "def _helper(x):\n    return x + 1\n")) == 0
    assert diffpairs.main(_functions_tree(tmp_path / "bad", base, "def _helper(x):\n    return x + 2\n")) == 1


def test_diffpairs_accepts_a_classmethod_move(tmp_path: Path) -> None:
    base = "    @classmethod\n    def _helper(cls, x):\n        return cls._k(x) + 1\n"
    assert diffpairs.main(_functions_tree(tmp_path / "ok", base, "def _helper(x):\n    return mod.k(x) + 1\n")) == 0
    assert diffpairs.main(_functions_tree(tmp_path / "bad", base, "def _helper(x):\n    return mod.k(x) + 2\n")) == 1


def test_diffpairs_accepts_a_renamed_attribute(tmp_path: Path) -> None:
    base = "    def _helper(self, x):\n        return self.y._old(x)\n"
    head = "def _helper(ctx, x):\n    return ctx.y.new(x)\n"
    manifest = FUNCTIONS_MANIFEST.replace('"renames": {}', '"renames": {"_old": "new"}')
    args = _functions_tree(tmp_path, base, head)
    (tmp_path / MANIFEST_PATH).write_text(manifest)
    assert diffpairs.main(args) == 0


def test_patches_watches_module_names_the_moved_bodies_import(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _store_tree(tmp_path)
    assert _store(root) == 0
    source = (
        "from x import ChaChaNotes_DB\n"
        "def t1(monkeypatch):\n    monkeypatch.setattr('m.ChaChaNotes_DB.time', f)\n"
        "def t2():\n    patch('m.ChaChaNotes_DB.time')\n"
        "def t3():\n    setattr(ChaChaNotes_DB, 'time', f)\n"
        "def t4(monkeypatch):\n    monkeypatch.setattr('m.ChaChaNotes_DB.sleep', f)\n"
    )
    (root / TEST).parent.mkdir(parents=True, exist_ok=True)
    (root / TEST).write_text(source)
    assert split.main(["patches", str(root / MANIFEST_PATH), "--root", str(root)]) == 1
    out = capsys.readouterr().out
    hits = [line for line in out.splitlines() if line.startswith(TEST)]
    assert len(hits) == 3 and all("time" in line for line in hits), out
    assert "both sides" in out  # the DB module still reads ``time``


def test_patches_watches_a_module_name_the_move_left_unused_in_the_db_module(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _store_tree(tmp_path)
    assert _store(root) == 0
    db_path = root / DB
    text = db_path.read_text().replace("return time.time()", "return 1")  # the class no longer reads ``time`` ...
    assert "import time\n" in text and "time." not in text
    db_path.write_text(text.replace("import time\n", ""))  # ... so the F401 step dropped the import
    (root / TEST).parent.mkdir(parents=True, exist_ok=True)
    (root / TEST).write_text("from x import ChaChaNotes_DB\ndef t(monkeypatch):\n    monkeypatch.setattr(ChaChaNotes_DB, 'time', f)\n")
    assert split.main(["patches", str(root / MANIFEST_PATH), "--root", str(root)]) == 1
    assert "time" in capsys.readouterr().out


def test_patches_watches_constants_the_moved_bodies_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = STORE_DB.replace("    def keep", "    _LIMIT = 5\n\n    def keep").replace("time.time()", "self._LIMIT")
    manifest = MANIFEST_SRC.replace('"rewrites": {}', '"rewrites": {"_LIMIT": "ctx.limit"}')
    root = _store_tree(tmp_path, manifest, {TEST: 'def t(monkeypatch):\n    monkeypatch.setattr(CharactersRAGDB, "_LIMIT", 9)\n'})
    (root / DB).write_text(db)
    assert _store(root) == 0
    assert "self._ctx.limit" in _text(root, S_STORE)  # the manifest decision
    assert split.main(["patches", str(root / MANIFEST_PATH), "--root", str(root)]) == 1
    assert "_LIMIT" in capsys.readouterr().out


def test_check_lists_an_untouched_reference_to_a_moved_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _store_tree(tmp_path, files={TEST: "def f(db):\n    return db._moved(1)\n"})
    assert _store(root) == 0
    assert split.main(["check", str(root / MANIFEST_PATH), "--root", str(root)]) == 0
    (root / TEST).write_text("def f(db):\n    return db._moved(1)\n    x = db._renamed\n")
    assert split.main(["check", str(root / MANIFEST_PATH), "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "_moved" in out and "_renamed" in out


def test_tests_lists_files_referencing_moved_names(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    other = "tldw_Server_API/tests/test_other.py"
    root = _store_tree(tmp_path, files={TEST: "db._moved(1)\n", other: "db._renamed(1)\n", "tldw_Server_API/tests/test_none.py": "x = 1\n"})
    assert split.main(["tests", str(root / MANIFEST_PATH), "--root", str(root)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert {TEST, other} <= set(out) and "tldw_Server_API/tests/test_none.py" not in out


def test_check_ignores_a_kwarg_named_like_a_retired_name_on_an_unrelated_call(tmp_path: Path) -> None:
    root = _seed(tmp_path, {TEST: "def t():\n    run(get_x=1)\n"}, db=CLEAN_DB)
    assert _callers(root, "--check") == 0


@pytest.mark.parametrize("call", ["SimpleNamespace(get_x=f)", "types.SimpleNamespace(get_x=f)", "MagicMock(get_x=f)", "mock.Mock(get_x=f)"])
def test_check_flags_a_kwarg_fake_of_a_retired_name(tmp_path: Path, call: str) -> None:
    root = _seed(tmp_path, {TEST: f"fake = {call}\n"}, db=CLEAN_DB)
    assert _callers(root, "--check") == 1


# --- constants mode (TASK-13950.22 PR D) ---------------------------------------------------------------------------

SCHEMA_SQL = "tldw_Server_API/app/core/DB_Management/chacha/schema_sql.py"
CONSTANTS_DB = '''
from typing import ClassVar

_MOD = "x"


class CharactersRAGDB:
    _KEEP = 5
    _SQL_A = """
        CREATE TABLE a (
            id INTEGER
        );
    """
    _SQL_F = f"""
        SELECT '{_MOD}'
          FROM t
    """
    _A = "xax"
    _B = _A.replace("x", "y")
    _TABLES: ClassVar[tuple[str, ...]] = ("a", "b")

    def run(self):
        return self._SQL_A + self._B
'''
MOVED_CONSTANTS = ["_MOD", "_SQL_A", "_SQL_F", "_A", "_B", "_TABLES"]


def _constants_manifest(names: list[str]) -> str:
    manifest = {
        "task": "T99", "mode": "constants", "source": DB, "target": SCHEMA_SQL, "alias": "schema_sql",
        "store_class": None, "store_attr": None, "names": [], "shims": [], "constants": names,
        "rewrites": {}, "renames": {}, "structural": [],
    }
    return f"MANIFEST = {manifest!r}\n"


def _constants_tree(root: Path, names: list[str] = MOVED_CONSTANTS) -> Path:
    _seed(root, {MANIFEST_PATH: _constants_manifest(names)}, db=CONSTANTS_DB)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _constants(root: Path) -> int:
    return split.main(["constants", str(root / MANIFEST_PATH), "--root", str(root)])


def _values(text: str) -> dict[str, str]:
    """Constant name -> dump of its value, over the module body and the class body."""
    tree = ast.parse(text)
    bodies = [tree.body, *(n.body for n in tree.body if isinstance(n, ast.ClassDef))]
    return {
        (st.targets[0] if isinstance(st, ast.Assign) else st.target).id: ast.dump(st.value)
        for body in bodies
        for st in body
        if isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None
    }


def test_constants_move_keeps_values_and_class_access(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _constants_tree(tmp_path)
    assert _constants(root) == 0
    original, target = _values(CONSTANTS_DB), _values(_text(root, SCHEMA_SQL))
    assert {n: target[n] for n in MOVED_CONSTANTS} == {n: original[n] for n in MOVED_CONSTANTS}
    db = _text(root, DB)
    assert "    from tldw_Server_API.app.core.DB_Management.chacha.schema_sql import (" in db  # the class body
    assert "\nfrom tldw_Server_API.app.core.DB_Management.chacha.schema_sql import (" in db  # the module
    assert '_SQL_A = """' not in db and "_KEEP = 5" in db and "ClassVar[" not in _text(root, SCHEMA_SQL)
    assert "        CREATE TABLE a (\n            id INTEGER\n        );\n" in _text(root, SCHEMA_SQL)  # a string's own lines stay
    assert _verify(root) == 0
    assert "constants 6 identical" in capsys.readouterr().out
    assert diffpairs.main([str(root / MANIFEST_PATH), "--base", "HEAD", "--root", str(root)]) == 0


def test_verify_fails_on_an_edited_constant(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _constants_tree(tmp_path)
    assert _constants(root) == 0
    (root / SCHEMA_SQL).write_text(_text(root, SCHEMA_SQL).replace('"xax"', '"xaz"'))
    assert _verify(root) == 1
    assert "_A: the moved constant differs" in capsys.readouterr().out
    assert diffpairs.main([str(root / MANIFEST_PATH), "--base", "HEAD", "--root", str(root)]) == 1


def test_constants_fail_closed_on_a_foreign_reference(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _constants_tree(tmp_path, [n for n in MOVED_CONSTANTS if n != "_A"])  # _B reads _A, which would stay behind
    before = _text(root, DB)
    assert _constants(root) == 1
    assert "_B: ['_A']" in capsys.readouterr().out
    assert _text(root, DB) == before and not (root / SCHEMA_SQL).exists()


def test_select_schema_sql_adds_what_the_values_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = (
        CONSTANTS_DB.replace('_SQL_A = """', '_MIGRATION_SQL_Y = """')
        .replace("_SQL_F = f", "_MIGRATION_SQL_F = f")
        .replace("    _A = ", "    _MIGRATION_SQL_X = ")
        .replace("_B = _A.", "_B = _MIGRATION_SQL_X.")
    )
    root = _seed(tmp_path, db=db)
    assert split.main(["select", "schema-sql", "--root", str(root)]) == 0
    assert capsys.readouterr().out.strip() == "['_MOD', '_MIGRATION_SQL_Y', '_MIGRATION_SQL_F', '_MIGRATION_SQL_X']"


# --- ``functions`` and ``select`` (TASK-13950.22 PR E) -------------------------------------------------------------

FN_A = "Helper_Scripts/chacha_split/manifests/t99_a.py"
FN_B = "Helper_Scripts/chacha_split/manifests/t99_b.py"
CHACHA = "tldw_Server_API/app/core/DB_Management/chacha"
MOD_A = f"{CHACHA}/mod_a.py"
MOD_B = f"{CHACHA}/mod_b.py"
CONTEXT = f"{CHACHA}/context.py"
FN_DB = '''
import time
from typing import Any

LIMIT = 3


class CharactersRAGDB:
    def keep(self, conn):
        self._b(conn)
        return self._s(1) + self._k(2)

    def _a(self, conn: Any, x):
        """Doc."""
        self._b(conn)
        self._c()
        value = self._map_table_for_backend(x) + self.client_id + time.time() + LIMIT
        return value + len(str(self)) + self._s(x)

    def _b(self, conn):
        return self._get_current_utc_timestamp_iso()

    def _c(self):
        return 1

    @staticmethod
    def _s(x):
        return x + 1

    @classmethod
    def _k(cls, x):
        return cls._s(x) + 1
'''
CTX_STATIC = """
class ChaChaContext:
    @staticmethod
    def pure(x):
        return x
"""


def _fn_manifest(task: str, alias: str, names: list[str], target: str, **extra: object) -> str:
    fields = {"task": task, "mode": "functions", "source": DB, "target": target, "alias": alias, "store_class": None,
              "store_attr": None, "names": names, "shims": [], "rewrites": {}, "renames": {}, "structural": [], **extra}
    return f"MANIFEST = {fields!r}\n"


def _fn_tree(root: Path, db: str = FN_DB, a: list[str] | None = None, b: list[str] | None = None, files: dict[str, str] | None = None, **extra: object) -> Path:
    manifests = {FN_A: _fn_manifest("T99", "mod_a", a if a is not None else ["_a"], MOD_A, **extra)}
    if b is not None:
        manifests[FN_B] = _fn_manifest("T99", "mod_b", b, MOD_B)
    _seed(root, {**manifests, CONTEXT: CTX_STATIC, **(files or {})}, db=db)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _functions(root: Path) -> int:
    paths = [str(root / p) for p in (FN_A, FN_B) if (root / p).exists()]
    return split.main(["functions", *paths, "--root", str(root)])


def _fn_verify(root: Path) -> int:
    paths = [str(root / p) for p in (FN_A, FN_B) if (root / p).exists()]
    return split.main(["verify", *paths, "--base", "HEAD", "--root", str(root)])


def test_self_becomes_ctx_and_mapped_members_rewrite(tmp_path: Path) -> None:
    root = _fn_tree(tmp_path, b=["_b", "_c", "_s"])
    assert _functions(root) == 0
    mod = _text(root, MOD_A)
    assert "def _a(ctx, conn: Any, x):" in mod
    assert "ctx.map_table(x) + ctx.client_id + time.time() + LIMIT" in mod
    assert "def _a" not in _text(root, DB) and "def _b" not in _text(root, DB)


def test_call_to_moved_function_gets_ctx_inserted(tmp_path: Path) -> None:
    root = _fn_tree(tmp_path, b=["_b", "_c", "_s"])
    assert _functions(root) == 0
    mod = _text(root, MOD_A)
    assert "mod_b._b(ctx, conn)" in mod and "mod_b._c(ctx)" in mod  # an argument, and none
    assert "from tldw_Server_API.app.core.DB_Management.chacha import mod_b" in mod
    assert "ctx.utc_now_iso()" in _text(root, MOD_B)


def test_call_inside_one_module_is_not_qualified(tmp_path: Path) -> None:
    root = _fn_tree(tmp_path, a=["_a", "_b", "_c", "_s"])
    assert _functions(root) == 0
    mod = _text(root, MOD_A)
    assert "    _b(ctx, conn)" in mod and "    _c(ctx)" in mod and "mod_a." not in mod  # a module does not import itself


def test_static_method_moves_without_ctx(tmp_path: Path) -> None:
    root = _fn_tree(tmp_path, a=["_s"])
    assert _functions(root) == 0
    mod = _text(root, MOD_A)
    assert "def _s(x):" in mod and "staticmethod" not in mod and "ctx" not in mod
    assert "mod_a._s(x)" in _text(root, DB) and "mod_a._s(1)" in _text(root, DB)  # called without ctx


def test_classmethod_with_moved_cls_refs_becomes_static(tmp_path: Path) -> None:
    root = _fn_tree(tmp_path, a=["_s", "_k"])
    assert _functions(root) == 0
    mod = _text(root, MOD_A)
    assert "def _k(x):" in mod and "classmethod" not in mod and "return _s(x) + 1" in mod
    assert "mod_a._k(2)" in _text(root, DB)


def test_classmethod_static_context_member_is_reached_through_the_module(tmp_path: Path) -> None:
    """The context imports ChaChaNotes_DB, so a moved body names the context module and never imports the class at top level."""
    db = FN_DB.replace("return cls._s(x) + 1", "return cls._p(x) + 1")
    root = _fn_tree(tmp_path, db=db, a=["_k"], rewrites={"_p": "ctx.pure"})
    assert _functions(root) == 0
    mod = _text(root, MOD_A)
    assert "return chacha_context.ChaChaContext.pure(x) + 1" in mod
    assert "from tldw_Server_API.app.core.DB_Management.chacha import context as chacha_context" in mod
    assert "import ChaChaContext" not in mod
    assert _fn_verify(root) == 0


def test_classmethod_with_other_cls_use_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = FN_DB.replace("return cls._s(x) + 1", "return cls(x)")
    root = _fn_tree(tmp_path, db=db, a=["_k"])
    assert _functions(root) == 1
    assert "cls is used other than as cls.<name>" in capsys.readouterr().out
    assert not (root / MOD_A).exists()


def test_class_receiver_static_call_is_rewritten(tmp_path: Path) -> None:
    root = _fn_tree(tmp_path, a=["_s"], files={TEST: "def f():\n    return CharactersRAGDB._s(1)\n"})
    assert _functions(root) == 0
    out = _text(root, TEST)
    assert "mod_a._s(1)" in out and "import mod_a" in out


def test_class_receiver_unbound_call_is_listed_not_rewritten(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _fn_tree(tmp_path, b=["_b", "_c", "_s"], files={TEST: "def f(db):\n    return CharactersRAGDB._b(db, 1)\n"})
    assert _functions(root) == 0
    assert "CharactersRAGDB._b(db, 1)" in _text(root, TEST)
    capsys.readouterr()
    assert split.main(["check", str(root / FN_A), str(root / FN_B), "--root", str(root)]) == 1
    assert "CharactersRAGDB._b" in capsys.readouterr().out


def test_bare_self_becomes_ctx_db(tmp_path: Path) -> None:
    root = _fn_tree(tmp_path, b=["_b", "_c", "_s"])
    assert _functions(root) == 0
    assert "len(str(ctx.db))" in _text(root, MOD_A)


def test_self_ctx_becomes_ctx(tmp_path: Path) -> None:
    db = FN_DB.replace("self._c()\n", "self._c()\n        self.ctx.thing()\n", 1)
    root = _fn_tree(tmp_path, db=db, b=["_b", "_c", "_s"])
    assert _functions(root) == 0
    assert "    ctx.thing()" in _text(root, MOD_A) and "self.ctx" not in _text(root, MOD_A)


def test_a_body_binding_ctx_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _fn_tree(tmp_path, db=FN_DB.replace("value = self._map", "ctx = self._map"))
    assert _functions(root) == 1
    assert "binds a name ctx" in capsys.readouterr().out


def test_unmapped_private_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = FN_DB.replace("self._c()\n", "self._mystery()\n        self._mystery()\n", 1)
    root = _fn_tree(tmp_path, db=db, b=["_b", "_c", "_s"])
    assert _functions(root) == 1
    out = capsys.readouterr().out
    assert "self._mystery: 2 uses: _a x2" in out
    assert not (root / MOD_A).exists() and "def _a" in _text(root, DB)


def test_multiline_and_fstring_text_is_not_dedented(tmp_path: Path) -> None:
    body = '        sql = """\n    SELECT 1\n  FROM t\n"""\n        text = f"""\n   {x}\n  tail"""\n        return sql, text\n'
    db = FN_DB.replace("        return value + len(str(self)) + self._s(x)\n", body)
    root = _fn_tree(tmp_path, db=db, b=["_b", "_c", "_s"])
    assert _functions(root) == 0
    mod = _text(root, MOD_A)
    assert '    sql = """\n    SELECT 1\n  FROM t\n"""\n' in mod and '    text = f"""\n   {x}\n  tail"""\n' in mod


def test_free_names_import_from_their_source(tmp_path: Path) -> None:
    root = _fn_tree(tmp_path, b=["_b", "_c", "_s"])
    assert _functions(root) == 0
    mod = _text(root, MOD_A)
    assert "import time\n" in mod and "from typing import Any" in mod  # the way ChaChaNotes_DB binds them
    assert f"from {DB[:-3].replace('/', '.')} import LIMIT" in mod  # a module-level name of ChaChaNotes_DB


def test_shim_forwards_and_class_call_sites_rewrite(tmp_path: Path) -> None:
    root = _fn_tree(tmp_path, a=["_b", "_c"], shims=["_b"])
    assert _functions(root) == 0
    db = _text(root, DB)
    assert "def _b(self, conn):\n        return mod_a._b(self.ctx, conn)\n" in db  # the forwarder stays
    assert "def _c" not in db
    assert "mod_a._b(self.ctx, conn)" in db.split("def keep")[1].split("def _a")[0]  # a class call site goes to the module
    assert "def _b(ctx, conn):" in _text(root, MOD_A)


def test_test_call_sites_take_receiver_ctx(tmp_path: Path) -> None:
    root = _fn_tree(tmp_path, a=["_b"], files={TEST: "def f(db, conn):\n    return db._b(conn) + self.db._b(conn)\n"})
    assert _functions(root) == 0
    assert "mod_a._b(db.ctx, conn) + mod_a._b(self.db.ctx, conn)" in _text(root, TEST)


def test_a_call_whose_arguments_start_on_the_next_line_gets_no_trailing_space(tmp_path: Path) -> None:
    root = _fn_tree(tmp_path, a=["_b"], files={TEST: "def f(db, conn):\n    return db._b(\n        conn,\n    )\n"})
    assert _functions(root) == 0
    assert "mod_a._b(db.ctx,\n        conn," in _text(root, TEST)


def test_functions_verify_and_diffpairs_pass_and_fail_on_a_hand_edit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _fn_tree(tmp_path, a=["_a", "_s"], b=["_b", "_c"])
    assert _functions(root) == 0
    args = [str(root / FN_A), str(root / FN_B), "--base", "HEAD", "--root", str(root)]
    assert _fn_verify(root) == 0 and diffpairs.main(args) == 0
    assert "moved 4 identical" in capsys.readouterr().out
    (root / MOD_A).write_text(_text(root, MOD_A).replace("x + 1", "x + 2"))
    assert _fn_verify(root) == 1 and diffpairs.main(args) == 1
    capsys.readouterr()
    (root / MOD_A).write_text(_text(root, MOD_A).replace("x + 2", "x + 1"))
    (root / DB).write_text(_text(root, DB).replace("mod_a._s(1)", "mod_a._s(2)"))
    assert _fn_verify(root) == 1  # a remaining class def is checked too


def test_a_name_in_two_manifests_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _fn_tree(tmp_path, a=["_a", "_s"], b=["_b", "_s"])
    assert _functions(root) == 1
    assert "_s: listed twice" in capsys.readouterr().out and not (root / MOD_A).exists()


def test_patches_lists_a_patch_of_a_moved_name_until_it_targets_the_module(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = "def f(monkeypatch, db):\n    monkeypatch.setattr(db, '_b', print)\n    monkeypatch.setattr(mod_a, '_b', print)\n"
    root = _fn_tree(tmp_path, a=["_b"], files={TEST: source})
    assert _functions(root) == 0
    capsys.readouterr()
    assert split.main(["patches", str(root / FN_A), "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert f"{TEST}:2: setattr _b" in out and f"{TEST}:3" not in out


SELECT_DB = '''
class CharactersRAGDB:
    def _migrate_from_v1_to_v2_sqlite(self, conn): ...
    def _migrate_from_v2_to_v3_postgres(self, conn): ...
    def _ensure_a_schema_sqlite(self, conn):
        self._helper_sq()
        self._shared()
    def _ensure_b_schema_postgres(self, conn):
        self._helper_pg()
        self._shared()
    def _helper_sq(self): ...
    def _helper_pg(self): ...
    def _shared(self): ...
    def _used_by_a_store(self): ...
    def _ensure_json_string(self): ...
    def _initialize_schema_sqlite_legacy(self, conn):
        self._used_by_a_store()
    def _initialize_schema(self): ...
    def other(self): ...
'''


def test_select_matches_research_groups_on_a_seed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = f"{CHACHA}/a_store.py"
    root = _seed(tmp_path, {store: "def f(self):\n    return self._db._used_by_a_store()\n"}, db=SELECT_DB)

    def group(name: str) -> list[str]:
        assert split.main(["select", name, "--root", str(root)]) == 0
        return ast.literal_eval(capsys.readouterr().out)

    assert group("migrate-sqlite") == ["_migrate_from_v1_to_v2_sqlite"]
    assert group("migrate-postgres") == ["_migrate_from_v2_to_v3_postgres"]
    assert group("sqlite-schema") == ["_ensure_a_schema_sqlite", "_helper_sq"]
    assert group("postgres-schema") == ["_ensure_b_schema_postgres", "_helper_pg"]
    assert group("migration-support") == ["_shared"]  # called from both sides
    assert group("init") == ["_initialize_schema_sqlite_legacy"]  # the closure leaves ``_used_by_a_store`` (a store reaches it)


def test_constants_fail_closed_on_a_name_that_is_not_a_constant(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _constants_tree(tmp_path, ["_NOPE"])
    assert _constants(root) == 1
    assert "['_NOPE']" in capsys.readouterr().out and not (root / SCHEMA_SQL).exists()


def test_const_sites_fail_closed_on_a_name_assigned_twice() -> None:
    tree = ast.parse("class CharactersRAGDB:\n    _A = 1\n    _A = 2\n")
    with pytest.raises(SystemExit, match="_A is assigned twice at class level"):
        split._const_sites(tree)
    assert set(split._const_sites(ast.parse("_A = 1\nclass CharactersRAGDB:\n    _A = 2\n"))) == {"_A"}  # one per level is fine


def test_a_name_another_app_class_defines_is_rewritten_only_on_a_db_receiver_in_tests(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    other = "tldw_Server_API/app/core/DB_Management/Other_DB.py"
    app = "tldw_Server_API/app/api/uses.py"
    root = _fn_tree(
        tmp_path,
        a=["_b"],
        files={
            other: "class Other:\n    def _b(self, conn): ...\n",
            app: "def f(db, conn):\n    return db._b(conn)\n",
            TEST: "def f(db, thing, conn):\n    return db._b(conn), thing._b(conn)\n",
        },
    )
    assert _functions(root) == 0
    assert "collision _b" in capsys.readouterr().out
    assert "db._b(conn)" in _text(root, app)  # app code: left to its own class
    assert "mod_a._b(db.ctx, conn), thing._b(conn)" in _text(root, TEST)


def test_a_reference_through_a_moved_module_is_not_rewritten_again(tmp_path: Path) -> None:
    root = _fn_tree(tmp_path, a=["_b"], files={TEST: "def f(db, conn):\n    return db._b(conn)\n"})
    assert _functions(root) == 0
    once = _text(root, TEST)
    tree = ast.parse(once)
    env = split.FnEnv({}, {"_b": split.FnMove("mod_a", MOD_A, False, False)}, {}, {}, frozenset(), frozenset(), frozenset())
    edits, _, problems = split._callsite_edits(once, tree, env, False, set(), True)
    assert not edits and not problems  # ``mod_a._b(db.ctx, conn)`` is left alone
    assert "mod_a._b(db.ctx, conn)" in once and once.count("ctx") == 1


def test_a_body_using_super_fails_closed_naming_the_def(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = FN_DB.replace("        return x + 1\n", "        return x + 1\n\n    def _sup(self):\n        return super().keep()\n", 1)
    root = _fn_tree(tmp_path, db=db, a=["_sup"])
    assert _functions(root) == 1
    assert "_sup: " in capsys.readouterr().out and not (root / MOD_A).exists()


SNAPSHOT_DB = '''
class CharactersRAGDB:
    def _sqlite_linear_migration_steps(self):
        steps = {4: self._migrate_from_v4_to_v5}
        for version, method_name in ((5, "_migrate_from_v5_to_v6_sqlite"),):
            steps[version] = getattr(self, method_name)
        steps[6] = bind_migration(migrate_sqlite_transcripts, schema_name=self._SCHEMA_NAME)
        return steps

    def _initialize_schema_postgres(self):
        if True:
            if current_version in (72, 73):
                if target_version >= 73 and current_version < 73:
                    self._migrate_from_v72_to_v73_postgres(conn)
                if target_version >= 74 and current_version < 74:
                    self._migrate_from_v73_to_v74_persona_postgres(conn)
            if target_version >= 71 and current_version < 71:
                stamp(conn, 71)
            if target_version >= 73 and current_version < 73:
                self._migrate_from_v72_to_v73_postgres(conn)
                self._runtime_schema_version = 73
            if target_version >= 74 and current_version < 74:
                self._migrate_from_v73_to_v74_persona_postgres(conn)
'''


def test_registry_snapshot_reads_each_registry_form_by_ast(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _seed(tmp_path, db=SNAPSHOT_DB)
    assert split.main(["registry-snapshot", "--root", str(root)]) == 0
    ns: dict = {}
    exec(capsys.readouterr().out, ns)  # noqa: S102  # the tool's own literal output, in a test
    assert ns["FROZEN_SQLITE"] == [(4, "_migrate_from_v4_to_v5"), (5, "_migrate_from_v5_to_v6_sqlite"), (6, "migrate_sqlite_transcripts")]
    assert ns["FROZEN_POSTGRES"] == [(72, "_migrate_from_v72_to_v73_postgres"), (73, "_migrate_from_v73_to_v74_persona_postgres")]


def test_registry_snapshot_fails_when_the_two_postgres_chains_differ(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _seed(tmp_path, db=SNAPSHOT_DB.replace("self._migrate_from_v73_to_v74_persona_postgres(conn)\n            if target", "self._migrate_from_v73_to_v74_other_postgres(conn)\n            if target", 1))
    assert split.main(["registry-snapshot", "--root", str(root)]) == 1
    assert "differ" in capsys.readouterr().out


def test_functions_leaves_a_same_named_method_outside_chacha_alone(tmp_path: Path) -> None:
    """M1: T7 and T8 rewrote Media DB files that call their own ``_b``. Neither a path nor a receiver that is not a ChaCha DB is ours."""
    media = "tldw_Server_API/app/core/DB_Management/media_db/schema/helper.py"
    app = "tldw_Server_API/app/api/helper.py"
    source = "def f(db, obj, conn):\n    return db._b(conn), obj._b(conn)\n"
    root = _fn_tree(tmp_path, a=["_b"], files={media: source, app: source})
    assert _functions(root) == 0
    assert _text(root, media) == source  # a path of another DB package: untouched, even on a ``db`` receiver
    assert "obj._b(conn)" in _text(root, app) and "mod_a._b(db.ctx, conn)" in _text(root, app)  # a free function's ``obj`` is not a ChaCha DB


def test_check_flags_a_moved_name_held_in_a_string(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """M2: ``patches`` cannot see a name in a variable, an f-string or a setattr probe; ``check`` must."""
    root = _fn_tree(tmp_path, a=["_a"], b=["_b", "_c", "_s"])
    assert _functions(root) == 0
    clean = "def t(db):\n    return 1\n"
    (root / TEST).parent.mkdir(parents=True, exist_ok=True)
    (root / TEST).write_text(clean)
    manifest = str(root / FN_A)
    assert split.main(["check", manifest, str(root / FN_B), "--root", str(root)]) == 0
    for body in (
        'method = "_a"\n',
        'name = f"_a{suffix}"\n',
        'name = f"_a"\n',
        'setattr(db, "_a", fake)\n',
    ):
        (root / TEST).write_text(f"def t(db, suffix, fake):\n    {body}")
        assert split.main(["check", manifest, str(root / FN_B), "--root", str(root)]) == 1, body
        assert "_a" in capsys.readouterr().out


def test_check_accepts_a_setattr_on_the_moved_module(tmp_path: Path) -> None:
    """The correct retarget of a patch, ``monkeypatch.setattr(mod_a, "_a", fake)``, is not a left reference; the same on a db is."""
    root = _fn_tree(tmp_path, a=["_a"], b=["_b", "_c", "_s"])
    assert _functions(root) == 0
    (root / TEST).parent.mkdir(parents=True, exist_ok=True)
    (root / TEST).write_text('def t(monkeypatch, mod_a, fake):\n    monkeypatch.setattr(mod_a, "_a", fake)\n')
    assert split.main(["check", str(root / FN_A), str(root / FN_B), "--root", str(root)]) == 0
    (root / TEST).write_text('def t(monkeypatch, db, fake):\n    monkeypatch.setattr(db, "_a", fake)\n')
    assert split.main(["check", str(root / FN_A), str(root / FN_B), "--root", str(root)]) == 1
