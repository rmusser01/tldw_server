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
    for source in ("def f(db):\n    db.get_x = print\n", "ref = CharactersRAGDB.get_x\n", 'p = "m.CharactersRAGDB.get_x"\n'):
        root = _seed(tmp_path / str(abs(hash(source))), {TEST: source}, db=CLEAN_DB)
        assert _callers(root, "--check") == 1, source


def test_check_passes_when_nothing_is_left(tmp_path: Path) -> None:
    root = _seed(
        tmp_path,
        {TEST: "class Fake:\n    s_store = None\n    def get_x(self): ...\n\ndef f(db):\n    db.s_store.get_x()\n"},
    )
    assert _callers(root, "--check") == 1  # DB_SRC still calls self.get_x()
    assert _callers(root) == 0
    assert _callers(root, "--check") == 0
