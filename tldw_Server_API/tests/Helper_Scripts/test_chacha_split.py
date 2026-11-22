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
