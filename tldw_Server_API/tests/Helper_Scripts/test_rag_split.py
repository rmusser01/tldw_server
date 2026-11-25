"""Helper_Scripts/rag_split/rag_split.py on mini pipeline modules in tmp trees (TASK-13950.24).

Each test builds ``mini/unified_pipeline.py`` under a base dir, copies it to a head dir, runs ``apply``
there, and checks the result with ``verify --base-dir`` or by reading the text. No git is needed.
"""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from Helper_Scripts.rag_split import rag_split

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FILE = "mini/unified_pipeline.py"
PKG = "mini/pipeline"

BASE = textwrap.dedent(
    '''\
    import os
    from datetime import datetime

    LIMIT = 5


    def helper(x):
        return x


    async def pipe(query, top_k=3, **kwargs):
        """Docstring."""
        result = {}
        flag = kwargs.get("flag", False)
        other = kwargs.get("other", 7)
        total = 0

        def _bump(n):
            nonlocal total
            total += n
            from .cache import thing
            return helper(n) + top_k + thing

        try:
            first = os.getenv("A") or query
            for i in range(top_k):
                total = _bump(i)
            stamp = datetime.now()
            second = await helper(first)
            def _fmt(v):
                return str(v)
            result["x"] = [_fmt(j) for j in range(LIMIT)]
            result["y"] = (second, flag, other, stamp)
        except Exception:
            pass
        return result
    '''
)

REWRITE = {
    "query": "rcfg.request.query",
    "top_k": "rcfg.retrieval.top_k",
    "total": "state.total",
    "first": "state.first",
    "stamp": "state.stamp",
    "second": "state.second",
    "result": "state.result",
    "flag": "state.flag",
    "other": "state.other",
    "_bump": "state.bump",
}


def row(src: str, text: str) -> int:
    hits = [i for i, ln in enumerate(src.split("\n"), 1) if text in ln]
    assert len(hits) == 1, (text, hits)
    return hits[0]


def moves(src: str = BASE) -> list[dict]:
    return [
        {"name": "step_one", "module": "stage_a", "span": [row(src, "first = os"), row(src, "stamp = ")], "doc": "First half."},
        {"name": "step_two", "module": "stage_b", "span": [row(src, "second = await"), row(src, 'result["y"]')]},
    ]


class Tree:
    def __init__(self, tmp: Path, src: str = BASE, **manifest: object) -> None:
        self.base, self.head = tmp / "base", tmp / "head"
        for d in (self.base, self.head):
            (d / "mini").mkdir(parents=True)
            (d / str(manifest.get("file", FILE))).write_text(src)
        self.manifest_path = tmp / "m.py"
        self.set(**manifest)

    def set(self, **manifest: object) -> None:
        full = {"base": "t", "file": FILE, "func": "pipe", "pkg": "pipeline", **manifest}
        self.manifest_path.write_text(f"MANIFEST = {full!r}\n")

    def run(self, cmd: str, *extra: str) -> int:
        args = [cmd, "--root", str(self.head)]
        if cmd == "verify":
            args += ["--base-dir", str(self.base)]
        return rag_split.main([*args, *extra, str(self.manifest_path)] if cmd != "shared" else [*args, "--manifest", str(self.manifest_path), *extra])

    def text(self, rel: str) -> str:
        return (self.head / rel).read_text()


def full(tmp: Path, src: str = BASE) -> Tree:
    return Tree(
        tmp,
        src,
        rewrite=REWRITE,
        hidden={"flag": ["gen", False], "other": ["gen", 7]},
        publish=["_fmt"],
        hoist={"_bump": "bump"},
        moves=moves(src),
    )


def test_apply_then_verify_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = full(tmp_path)
    assert t.run("apply") == 0
    for rel in (FILE, f"{PKG}/stage_a.py", f"{PKG}/stage_b.py", f"{PKG}/run_state.py"):
        compile(t.text(rel), rel, "exec")
    assert t.run("verify") == 0, capsys.readouterr().out
    assert "0 differences" in capsys.readouterr().out
    head = t.text(FILE)
    assert "await _stage_b.step_two(rcfg, state)" in head
    assert "kwargs.get" not in head and "state.flag = rcfg.gen.flag" in head
    assert "def _bump" not in head and "state.bump(i)" in t.text(f"{PKG}/stage_a.py")
    assert "state._fmt = _fmt" in t.text(f"{PKG}/stage_b.py")
    assert "def bump(self, n):" in t.text(f"{PKG}/run_state.py")


def test_rewrite_skips_shadowed_names(tmp_path: Path) -> None:
    src = textwrap.dedent(
        """\
        def pipe(q):
            x = 1
            def inner(x):
                return x
            ys = [x for x in range(3)]
            f = lambda x=x: x
            class C:
                x = 2
                y = x
            return x
        """
    )
    t = Tree(tmp_path, src, rewrite={"x": "state.x"})
    assert t.run("apply") == 0
    head = t.text(FILE)
    assert "state.x = 1" in head and "return state.x" in head and "lambda x=state.x: x" in head
    assert "def inner(x):\n        return x" in head
    assert "[x for x in range(3)]" in head
    assert "        x = 2\n        y = x" in head
    assert t.run("verify") == 0


def test_rewrite_drops_mapped_names_from_nonlocal(tmp_path: Path) -> None:
    src = textwrap.dedent(
        """\
        def pipe():
            a = b = 0
            def f():
                nonlocal a, b
                a, b = 1, 2
            def g():
                nonlocal a
                a = 3
            f()
            g()
        """
    )
    t = Tree(tmp_path, src, rewrite={"a": "state.a"})
    assert t.run("apply") == 0
    head = t.text(FILE)
    assert "nonlocal b" in head and "nonlocal a" not in head
    assert t.run("verify") == 0


def test_hidden_kwarg_reads_are_rewritten(tmp_path: Path) -> None:
    src = textwrap.dedent(
        """\
        def pipe(**kwargs):
            a = kwargs.get("k", 7)
            b = kwargs.get("j", 9)
            c = kwargs.get("m")
            d = kwargs.get("k", 8)
            return a, b, c, d
        """
    )
    t = Tree(tmp_path, src, hidden={"k": ["g", 7], "j": ["g", 7], "m": ["h", None]})
    assert t.run("apply") == 0
    head = t.text(FILE)
    assert "a = rcfg.g.k" in head
    assert 'b = kwargs.get("j", 9)' in head
    assert "c = rcfg.h.m" in head
    assert 'd = kwargs.get("k", 8)' in head
    assert t.run("verify") == 0


def test_prologue_and_imports_round_trip_and_are_verified(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = 'import os\n\n\nasync def pipe(query, **kwargs):\n    """Doc."""\n\n    return query, os, kwargs.get("k")\n'
    t = Tree(
        tmp_path,
        src,
        rewrite={"query": "rcfg.request.query"},
        hidden={"k": ["g", None]},
        prologue="rcfg = Config.build(locals())",
        imports=["from .pipeline.config import Config"],
    )
    assert t.run("apply") == 0
    head = t.text(FILE)
    assert head.index("from .pipeline.config import Config") > head.index("import os")
    assert head.index('"""Doc."""') < head.index("rcfg = Config.build(locals())") < head.index("return rcfg.request.query")
    assert t.run("verify") == 0, capsys.readouterr().out
    _edit(t, FILE, "    rcfg = Config.build(locals())\n\n", "")
    assert t.run("verify") == 1
    t.set(rewrite={"query": "rcfg.request.query"}, hidden={"k": ["g", None]}, prologue="rcfg = Config.build(locals())")
    assert t.run("verify") == 1


def test_state_init_goes_right_after_the_first_statement_and_is_verified(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = 'import os\n\n\nasync def pipe(query, **kwargs):\n    """Doc."""\n\n    cfg = Config()\n\n    return query, os\n'
    t = Tree(tmp_path, src, state_init="state = _st.State(cfg)", imports=["from .pipeline import st as _st"])
    assert t.run("apply") == 0
    head = t.text(FILE)
    assert head.index("cfg = Config()\n    state = _st.State(cfg)\n") < head.index("return query")
    assert t.run("verify") == 0, capsys.readouterr().out
    _edit(t, FILE, "    state = _st.State(cfg)\n", "    state = _st.State(cfg)\n    extra = 1\n")
    assert t.run("verify") == 1
    _edit(t, FILE, "    extra = 1\n", "")
    assert t.run("verify") == 0
    _edit(t, FILE, "_st.State(cfg)", "_st.Other(cfg)")
    assert t.run("verify") == 1
    _edit(t, FILE, "    state = _st.Other(cfg)\n", "")
    assert t.run("verify") == 1


def test_classes_key_moves_a_local_class_to_run_state(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = (
        "async def pipe(query):\n"
        '    """Doc."""\n\n'
        "    class _Stop(Exception):\n"
        '        """Sentinel."""\n\n'
        "    def _h():\n"
        "        return 1\n\n"
        "    try:\n"
        "        raise _Stop()\n"
        "    except _Stop:\n"
        "        return _h()\n"
    )
    t = Tree(tmp_path, src, classes={"_Stop": "Stop"}, rewrite={"_Stop": "_run_state.Stop", "_h": "state.h"}, hoist={"_h": "h"})
    assert t.run("apply") == 0
    head, state = t.text(FILE), t.text(f"{PKG}/run_state.py")
    assert "class _Stop" not in head and "except _run_state.Stop:" in head
    assert state.index("class Stop(Exception):") < state.index("class PipelineState") and '"""Sentinel."""' in state
    assert t.run("verify") == 0, capsys.readouterr().out
    _edit(t, f"{PKG}/run_state.py", '"""Sentinel."""', '"""Other."""')
    assert t.run("verify") == 1


def test_moved_class_reads_module_globals_through_up(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = (
        "LIMIT = 5\n\n\ndef helper2():\n    pass\n\n\n"
        "async def pipe(query):\n"
        '    """Doc."""\n\n'
        "    class _Stop(helper2):\n"
        "        x = LIMIT\n\n"
        "    def _h():\n"
        "        return 1\n\n"
        "    return _Stop, _h()\n"
    )
    t = Tree(tmp_path, src, classes={"_Stop": "Stop"}, rewrite={"_Stop": "_run_state.Stop", "_h": "state.h"}, hoist={"_h": "h"})
    assert t.run("apply") == 0
    state = t.text(f"{PKG}/run_state.py")
    assert "class Stop(up.helper2):" in state and "x = up.LIMIT" in state
    assert t.run("verify") == 0, capsys.readouterr().out
    _edit(t, f"{PKG}/run_state.py", "x = up.LIMIT", "x = LIMIT")
    assert t.run("verify") == 1


STATE_BASE = (
    "async def pipe(query, total=0):\n"
    '    """Doc."""\n\n'
    "    rcfg = Config()\n"
    "    state = _run_state.PipelineState(rcfg)\n"
    "    count = 1\n\n"
    "    def _h():\n"
    "{body}"
    "        return count\n\n"
    "{extra}"
    "    return _h()\n"
)


@pytest.mark.parametrize(
    ("body", "extra", "needle"),
    [
        ("        state = {}\n", "", "binds state itself"),
        ("", "    def other(state):\n        return state.total\n\n", "state is bound outside"),
        ("        rcfg = 1\n", "", "binds rcfg itself"),
        ("        return [self for self in range(2)]\n", "", "binds self itself"),
        ("        return [state for state in range(2)]\n", "", "binds state itself"),
    ],
)
def test_hoist_refuses_a_closure_or_def_that_binds_state_rcfg_or_self(tmp_path: Path, capsys: pytest.CaptureFixture, body: str, extra: str, needle: str) -> None:
    t = Tree(tmp_path, STATE_BASE.format(body=body, extra=extra), rewrite={"count": "state.count", "_h": "state.h"}, hoist={"_h": "h"})
    assert t.run("apply") == 1
    assert needle in capsys.readouterr().out


def test_state_module_constants_must_be_literals(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = Tree(tmp_path, STATE_BASE.format(body="", extra=""), rewrite={"count": "state.count", "_h": "state.h"}, hoist={"_h": "h"})
    assert t.run("apply") == 0
    path = t.head / PKG / "run_state.py"
    path.write_text(path.read_text().replace("class PipelineState", "X: int = 1\n\n\nclass PipelineState", 1))
    capsys.readouterr()
    assert t.run("verify") == 0, capsys.readouterr().out
    path.write_text(path.read_text().replace("X: int = 1", "X: int = print('side effect')"))
    assert t.run("verify") == 1
    assert "not allowed in a stage module" in capsys.readouterr().out


PUB_BASE = (
    "async def pipe(query, flag):\n"
    '    """Doc."""\n\n'
    "    if flag:\n"
    "        async def _pub():\n"
    "            return 1\n\n"
    "        first = await _pub()\n"
    "    second = _pub\n"
    "    return first, second\n"
)
PUB = {"publish": ["_pub"], "rewrite": {"_pub": "state._pub"}}


def test_published_def_is_mapped_and_published_in_its_own_block(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = Tree(tmp_path, PUB_BASE, **PUB)
    assert t.run("apply") == 0
    head = t.text(FILE)
    assert "            return 1\n        state._pub = _pub\n" in head  # inside the `if`, at the def's indentation
    assert "first = await state._pub()" in head and "second = state._pub\n" in head
    assert t.run("verify") == 0, capsys.readouterr().out


def test_published_def_with_a_second_binding_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = PUB_BASE.replace("    second = _pub\n", "    _pub = None\n    second = _pub\n")
    t = Tree(tmp_path, src, **PUB)
    assert t.run("apply") == 1
    assert "only binding of its name" in capsys.readouterr().out


def test_an_unpublished_mapped_def_still_fails_v5(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = Tree(tmp_path, PUB_BASE, rewrite={"_pub": "state._pub"})
    assert t.run("apply") == 1
    assert "a mapped name is bound by a def or class" in capsys.readouterr().out


def test_verify_rejects_a_moved_or_missing_publish_statement(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = Tree(tmp_path, PUB_BASE, **PUB)
    assert t.run("apply") == 0
    _edit(t, FILE, "        state._pub = _pub\n", "")
    assert t.run("verify") == 1
    _edit(t, FILE, "    second = state._pub\n", "    state._pub = _pub\n    second = state._pub\n")
    assert t.run("verify") == 1  # published outside the `if` block
    capsys.readouterr()
    _edit(t, FILE, "    state._pub = _pub\n    second", "    second")
    _edit(t, FILE, "            return 1\n", "            return 1\n        state._pub = _pub\n        state._pub = _pub\n")
    assert t.run("verify") == 1  # an extra publish statement


def test_move_prefixes_module_globals_with_up(tmp_path: Path) -> None:
    t = full(tmp_path)
    assert t.run("apply") == 0
    stage_b = t.text(f"{PKG}/stage_b.py")
    assert "range(up.LIMIT)" in stage_b and "up.helper(" in stage_b
    assert "from mini import unified_pipeline as up" in stage_b
    assert "def helper" not in stage_b


def test_move_keeps_stdlib_names_bare(tmp_path: Path) -> None:
    t = full(tmp_path)
    assert t.run("apply") == 0
    stage_a = t.text(f"{PKG}/stage_a.py")
    assert "os.getenv" in stage_a and "up.os" not in stage_a
    assert "datetime.now()" in stage_a and "up.datetime" not in stage_a
    assert "\nimport os\n" in stage_a and "\nfrom datetime import datetime\n" in stage_a
    assert '"""First half."""' in stage_a


def test_relative_import_level_is_bumped(tmp_path: Path) -> None:
    t = full(tmp_path)
    assert t.run("apply") == 0
    assert "from ..cache import thing" in t.text(f"{PKG}/run_state.py")


def _applied(tmp_path: Path) -> Tree:
    t = full(tmp_path)
    assert t.run("apply") == 0
    assert t.run("verify") == 0
    return t


def _edit(t: Tree, rel: str, old: str, new: str) -> None:
    path = t.head / rel
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new, 1))


def test_verify_rejects_a_changed_statement(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = _applied(tmp_path)
    capsys.readouterr()
    _edit(t, FILE, "result = {}", "result = []")
    assert t.run("verify") == 1
    assert "result" in capsys.readouterr().out


def test_verify_rejects_a_missing_up_prefix(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = _applied(tmp_path)
    capsys.readouterr()
    _edit(t, f"{PKG}/stage_b.py", "range(up.LIMIT)", "range(LIMIT)")
    assert t.run("verify") == 1
    assert "stage_b.py" in capsys.readouterr().out


def test_verify_rejects_a_crossing_local(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = Tree(tmp_path, moves=moves())  # nothing is rewritten, so first/total cross the cells
    assert t.run("verify") == 1
    out = capsys.readouterr().out
    assert "V4" in out and "first" in out


def test_verify_rejects_a_return_in_a_span(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = BASE.replace("        stamp = datetime.now()", "        if first:\n            return 1\n        stamp = datetime.now()")
    t = full(tmp_path, src)
    assert t.run("verify") == 1
    assert "V1" in capsys.readouterr().out


def test_verify_rejects_a_break_outside_its_loop(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = textwrap.dedent(
        """\
        def pipe():
            for i in range(3):
                a = i
                if a:
                    break
                b = a
            return 0
        """
    )
    bad = Tree(tmp_path / "bad", src, moves=[{"name": "m", "module": "mod", "span": [4, 5]}])
    assert bad.run("verify") == 1
    assert "V1" in capsys.readouterr().out
    # the same break inside a loop the span contains is fine
    whole = Tree(tmp_path / "ok", src, moves=[{"name": "m", "module": "mod", "span": [2, 6]}])
    assert whole.run("apply") == 0


def test_verify_rejects_a_name_collision_with_rcfg_or_state(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = "def pipe():\n    state = 1\n    x = 2\n    return state + x\n"
    t = Tree(tmp_path, src, rewrite={"x": "state.x"})
    assert t.run("verify") == 1
    assert "V8" in capsys.readouterr().out


CELLS = textwrap.dedent(
    """\
    def pipe():
        a = 1
        y = 2
        b = y
        return a, b
    """
)


def _shared(tmp_path: Path, src: str, cells: list[list[int]], capsys: pytest.CaptureFixture) -> str:
    t = Tree(tmp_path, src, cells=cells)
    assert t.run("shared", "--base-dir", str(t.base)) == 0
    return capsys.readouterr().out


def test_shared_reports_a_local_read_before_assignment_in_another_cell(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = "def pipe():\n    y = 1\n    z = 2\n    w = y + z\n    return w\n"
    out = _shared(tmp_path, src, [[2, 2], [4, 4]], capsys)
    assert "y: bound in 2-2, live-in at 4-4" in out
    assert "z: bound in <rest>, live-in at 4-4" in out


def test_shared_ignores_a_temp_rebound_in_every_cell(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = "def pipe():\n    tmp = 1\n    a = tmp\n    tmp = 2\n    b = tmp\n    tmp = 3\n    return tmp\n"
    out = _shared(tmp_path, src, [[2, 3], [4, 5]], capsys)
    assert out == ""


def test_shared_keeps_a_closure_local_rebound_only_before_its_def(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # the `sgl` shape: another cell binds the name earlier, the defining cell binds it before the def
    src = textwrap.dedent(
        """\
        def pipe():
            import os as sgl
            sgl.getcwd()
            import os as sgl
            def analyze():
                return sgl.getcwd()
            return analyze()
        """
    )
    assert _shared(tmp_path / "a", src, [[2, 3]], capsys) == ""
    late = src.replace("    return analyze()", "    sgl = None\n    return analyze()")
    out = _shared(tmp_path / "b", late, [[4, 6]], capsys)
    assert "sgl: captured by analyze in 4-6" in out


def test_verify_rejects_an_unbumped_relative_import_in_a_hoist(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = _applied(tmp_path)
    capsys.readouterr()
    _edit(t, f"{PKG}/run_state.py", "from ..cache import thing", "from .cache import thing")
    assert t.run("verify") == 1
    assert "bump" in capsys.readouterr().out


def test_verify_rejects_a_stdlib_name_bound_differently(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = _applied(tmp_path)
    capsys.readouterr()
    _edit(t, f"{PKG}/stage_a.py", "from datetime import datetime", "import datetime")
    assert t.run("verify") == 1
    assert "V9" in capsys.readouterr().out


LEAF_SRC = textwrap.dedent(
    """\
    async def pipe(a):
        try:
            x = 1
            msg = \"\"\"first
      second {x}\"\"\"
            y = await h(x)
            z = y
        finally:
            pass
    """
)


@pytest.mark.parametrize("leaf_module", ["wrap", "other"])
def test_a_wrapper_calls_its_leaf_and_keeps_multiline_strings(tmp_path: Path, leaf_module: str) -> None:
    rw = {"x": "state.x", "y": "state.y", "z": "state.z", "msg": "state.msg"}
    mv = [
        {"name": "wrapper", "module": "wrap", "span": [3, 7]},
        {"name": "leaf", "module": leaf_module, "span": [6, 6]},
    ]
    t = Tree(tmp_path, LEAF_SRC, rewrite=rw, moves=mv)
    assert t.run("apply") == 0
    wrap = t.text(f"{PKG}/wrap.py")
    call = "await leaf(rcfg, state)" if leaf_module == "wrap" else "await _other.leaf(rcfg, state)"
    assert call in wrap
    assert ("from . import other as _other" in wrap) == (leaf_module != "wrap")
    assert '"""first\n  second {x}"""' in wrap
    assert t.run("verify") == 0


def test_entry_split_moves_the_body_behind_a_shim(tmp_path: Path) -> None:
    src = textwrap.dedent(
        """\
        async def pipe(a, b=1):
            \"\"\"Doc.\"\"\"
            rcfg = make(a, b)
            x = await go(rcfg)
            return x
        """
    )
    t = Tree(tmp_path, src, entry_split=True)
    assert t.run("apply") == 0
    head = t.text(FILE)
    assert "return await run_rag_pipeline(make(a, b))" in head
    assert "async def run_rag_pipeline(rcfg: RagRunConfig):" in head
    assert head.index("async def pipe") < head.index("async def run_rag_pipeline")
    assert t.run("verify") == 0
    _edit(t, FILE, "x = await go(rcfg)", "x = await go(None)")
    assert t.run("verify") == 1


ENTRY_SRC = textwrap.dedent(
    """\
    async def pipe(a, b=1):
        \"\"\"Doc.\"\"\"
        rcfg = make(a, b)
        x = await go(rcfg)
        return x
    """
)


def _entry_parts(t: Tree) -> tuple[ast.AsyncFunctionDef, ast.AsyncFunctionDef]:
    fns = [n for n in ast.parse(t.text(FILE)).body if isinstance(n, ast.AsyncFunctionDef)]
    return fns[0], fns[1]


def test_entry_split_keeps_the_docstring_on_the_shim_only(tmp_path: Path) -> None:
    t = Tree(tmp_path, ENTRY_SRC, entry_split=True)
    assert t.run("apply") == 0
    shim, entry = _entry_parts(t)
    assert ast.get_docstring(shim) == "Doc." and ast.get_docstring(entry) is None
    assert isinstance(entry.body[0], ast.Assign)  # `x = ...`: nothing before the old body
    assert t.run("verify") == 0


def test_entry_split_entry_doc_and_shim_doc_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = Tree(tmp_path, ENTRY_SRC, entry_split=True, entry_doc="Run it.", shim_doc="Shim; see run_rag_pipeline.\n\n    More.\n    ")
    assert t.run("apply") == 0
    shim, entry = _entry_parts(t)
    assert ast.get_docstring(entry, clean=False) == "Run it."
    assert ast.get_docstring(shim, clean=False) == "Shim; see run_rag_pipeline.\n\n    More.\n    "
    assert len(shim.body) == 2 and isinstance(entry.body[1], ast.Assign)
    assert t.run("verify") == 0
    capsys.readouterr()
    _edit(t, FILE, "Run it.", "Run it!")
    assert t.run("verify") == 1
    _edit(t, FILE, "Run it!", "Run it.")
    _edit(t, FILE, "Shim; see", "Shim: see")
    assert t.run("verify") == 1


def test_entry_split_verify_rejects_a_stray_or_missing_docstring(tmp_path: Path) -> None:
    t = Tree(tmp_path, ENTRY_SRC, entry_split=True)
    assert t.run("apply") == 0
    _edit(t, FILE, "    x = await go(rcfg)", '    "stray"\n    x = await go(rcfg)')
    assert t.run("verify") == 1  # a docstring where the key is not set
    t2 = Tree(tmp_path / "two", ENTRY_SRC, entry_split=True, entry_doc="Run it.")
    assert t2.run("apply") == 0
    _edit(t2, FILE, '    """Run it."""\n', "")
    assert t2.run("verify") == 1  # a missing docstring where the key is set
    t3 = Tree(tmp_path / "three", ENTRY_SRC, entry_split=True, entry_doc="Run it.")
    assert t3.run("apply") == 0
    _edit(t3, FILE, '    """Run it."""\n', '    """Run it."""\n    1 + 1\n')
    assert t3.run("verify") == 1  # an extra statement after the docstring
    _edit(t3, FILE, '    """Run it."""\n    1 + 1\n', "    f'Run it.'\n")
    assert t3.run("verify") == 1  # not a plain str constant


def test_entry_docs_must_be_strings_and_need_entry_split(tmp_path: Path) -> None:
    t = Tree(tmp_path / "a", entry_split=True, shim_doc=["x"])
    with pytest.raises(rag_split.SplitError, match="shim_doc must be a str"):
        rag_split.load_manifest(t.manifest_path)
    t = Tree(tmp_path / "b", entry_doc="x")
    with pytest.raises(rag_split.SplitError, match="needs entry_split"):
        rag_split.load_manifest(t.manifest_path)


def test_verify_rejects_a_shadowing_duplicate_def(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = _applied(tmp_path)
    capsys.readouterr()
    path = t.head / PKG / "stage_a.py"
    path.write_text(path.read_text() + "\n\ndef step_one(rcfg, state):\n    state.first = 'EVIL'\n")
    assert t.run("verify") == 1
    assert "step_one" in capsys.readouterr().out


def test_verify_rejects_a_stdlib_name_rebound_by_assignment(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = _applied(tmp_path)
    capsys.readouterr()
    path = t.head / PKG / "stage_a.py"
    path.write_text(path.read_text().replace("\n\ndef step_one", "\n\ndatetime = object()\n\n\ndef step_one", 1))
    assert t.run("verify") == 1
    assert "module-level" in capsys.readouterr().out


def test_verify_rejects_a_rebound_up_import(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = _applied(tmp_path)
    capsys.readouterr()
    _edit(t, f"{PKG}/stage_b.py", "from mini import unified_pipeline as up", "import os as up")
    assert t.run("verify") == 1
    assert "up is bound as" in capsys.readouterr().out


def test_verify_rejects_extra_module_level_code(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = _applied(tmp_path)
    capsys.readouterr()
    path = t.head / PKG / "stage_a.py"
    path.write_text(path.read_text() + "\nimport sys\nsys.setrecursionlimit(50)\n")
    assert t.run("verify") == 1
    assert "not allowed in a stage module" in capsys.readouterr().out


IMPORT_SRC = textwrap.dedent(
    """\
    async def pipe(a):
        def critique():
            import json as sgl
            return sgl.dumps(a)

        try:
            x = critique()
            import json as sgl
            y = sgl.dumps(x)
            z = y
        finally:
            pass
        return z
    """
)


def test_a_name_that_is_only_ever_the_same_import_does_not_cross(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The `sgl` shape: a closure in the rest captures an import that a later span repeats."""
    src = IMPORT_SRC.replace("    def critique():\n        import json as sgl\n", "    import json as sgl\n\n    def critique():\n")
    t = Tree(tmp_path, src, rewrite={"x": "state.x", "y": "state.y", "z": "state.z"}, moves=[{"name": "m", "module": "mod", "span": [8, 10]}])
    assert t.run("shared", "--base-dir", str(t.base)) == 0
    assert "sgl" not in capsys.readouterr().out
    # a second, different binding of the name makes it cross again
    diff = src.replace("        y = sgl.dumps(x)", "        sgl = None\n        y = 1")
    t2 = Tree(tmp_path / "d", diff, rewrite={"x": "state.x", "y": "state.y", "z": "state.z"}, moves=[{"name": "m", "module": "mod", "span": [8, 11]}])
    assert t2.run("shared", "--base-dir", str(t2.base)) == 0
    assert "sgl" in capsys.readouterr().out


def test_a_read_before_the_cells_own_import_still_crosses(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = textwrap.dedent(
        """\
        def pipe(a):
            import json as sgl
            try:
                x = a
                y = sgl.dumps(x)
                import json as sgl
                z = y
            finally:
                pass
            return z
        """
    )
    rw = {"a": "rcfg.r.a", "x": "state.x", "y": "state.y", "z": "state.z"}
    t = Tree(tmp_path, src, rewrite=rw, moves=[{"name": "m", "module": "mod", "span": [4, 7]}])
    assert t.run("shared", "--base-dir", str(t.base)) == 0
    assert "sgl: bound in <rest>, live-in at m" in capsys.readouterr().out
    ok = src.replace("        y = sgl.dumps(x)\n        import json as sgl\n", "        import json as sgl\n        y = sgl.dumps(x)\n")
    t2 = Tree(tmp_path / "ok", ok, rewrite=rw, moves=[{"name": "m", "module": "mod", "span": [4, 7]}])
    assert t2.run("shared", "--base-dir", str(t2.base)) == 0
    assert "sgl" not in capsys.readouterr().out


INTRO_SRC = textwrap.dedent(
    """\
    def pipe():
        first = 1
        seen = "first" in locals()
        return seen
    """
)


def test_locals_in_a_span_is_a_v1_error_unless_allowed(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    mv = {"name": "m", "module": "mod", "span": [2, 3]}
    rw = {"first": "state.first", "seen": "state.seen"}
    t = Tree(tmp_path / "a", INTRO_SRC, rewrite=rw, moves=[mv])
    assert t.run("verify") == 1
    assert "locals()" in capsys.readouterr().out
    ok = Tree(tmp_path / "b", INTRO_SRC, rewrite=rw, moves=[{**mv, "allow_locals": True}])
    assert ok.run("apply") == 0
    assert ok.run("verify") == 0
    assert "allow_locals, uses locals()" in capsys.readouterr().out
    name_src = "def pipe():\n    a = __name__\n    return 1\n"
    n = Tree(tmp_path / "c", name_src, rewrite={"a": "state.a"}, moves=[{"name": "m", "module": "mod", "span": [2, 2]}])
    assert n.run("verify") == 1
    assert "__name__" in capsys.readouterr().out


def test_two_lambdas_on_one_line_fail_closed(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(rag_split.SplitError):
        rag_split.Src("def pipe(a):\n    f, g = (lambda: a), (lambda: 1)\n", "pipe")


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout  # noqa: S603, S607  # nosec B603 B607


MANIFEST_GLOB = "Helper_Scripts/rag_split/manifests/*.py"


INTEGRATION_REF = "origin/dev"


def _chain(root: Path, merge_base: str, integration: str) -> list[tuple[str, str, str | None]]:
    """``[(manifest, base_ref, head_ref)]``: each manifest is checked against the tree it was built on.

    Only manifests added since ``merge_base`` whose introducing commit is not already on ``integration``
    are checked: dev proved the others when their PRs merged. That makes a dev -> main release PR (every
    manifest is "new" against main) check nothing, while a PR into dev, stacked or not, checks its own.

    Manifests run oldest first. A manifest's head is the last commit that touched it, not the working tree: a
    later edit to the pipeline file that is not a manifest commit (dead-import cleanup, a fix) cannot break an
    earlier manifest. A manifest edited but not committed yet is checked against the working tree. Manifest
    i's base is manifest i-1's head (for the first, the last skipped manifest's head, else the merge base).
    Two manifests added in one commit are not supported.
    """
    added = _git("diff", "--name-only", "--diff-filter=A", merge_base, "--", MANIFEST_GLOB, cwd=root).split()
    order = {c: i for i, c in enumerate(_git("rev-list", "--reverse", f"{merge_base}..HEAD", cwd=root).split())}

    def introduced(name: str) -> str | None:
        shas = _git("log", "--diff-filter=A", "--format=%H", f"{merge_base}..HEAD", "--", name, cwd=root).split()
        return shas[-1] if shas else None  # None: not committed yet, so newest

    def head_of(name: str) -> str | None:
        if _git("status", "--porcelain", "--", name, cwd=root).strip():
            return None
        return _git("log", "-1", "--format=%H", "HEAD", "--", name, cwd=root).strip()

    def on_integration(commit: str) -> bool:
        return subprocess.run(["git", "merge-base", "--is-ancestor", commit, integration], cwd=root).returncode == 0  # noqa: S603, S607  # nosec B603 B607

    intro = {n: introduced(n) for n in added}
    ordered = sorted(added, key=lambda n: (order.get(intro[n] or "", len(order)), n))
    steps, base = [], merge_base
    for name in ordered:
        head = head_of(name)
        if intro[name] and on_integration(intro[name]):
            base = head or base
            continue
        steps.append((name, base, head))
        base = head or base
    return steps


def _verify_chain(root: Path, merge_base: str, integration: str = INTEGRATION_REF) -> list[str]:
    _git("rev-parse", "--verify", "-q", integration, cwd=root)  # fail closed: no integration ref, no decision
    return [
        name
        for name, base, head in _chain(root, merge_base, integration)
        if rag_split.main(["verify", "--root", str(root), str(root / name), "--base", base, *(["--head", head] if head else [])]) != 0
    ]


def test_new_manifests_verify_against_the_pr_base() -> None:
    """R16's proof, gated: every manifest this change adds verifies against the tree it was built on (a stack of PRs included)."""
    base = os.environ.get("BACKLOG_TASK_FORMAT_BASE") or INTEGRATION_REF
    try:
        merge_base = _git("merge-base", base, "HEAD").strip()
        failed = _verify_chain(REPO_ROOT, merge_base)
    except (OSError, subprocess.CalledProcessError) as exc:
        if os.environ.get("BACKLOG_TASK_FORMAT_BASE"):
            raise
        pytest.skip(f"cannot diff against {base}; fetch it or set BACKLOG_TASK_FORMAT_BASE ({exc})")
    assert failed == []


def _manifest(root: Path, name: str, **manifest: object) -> None:
    path = root / "Helper_Scripts" / "rag_split" / "manifests" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"MANIFEST = {({'base': 't', 'file': FILE, 'func': 'pipe', 'pkg': 'pipeline', **manifest})!r}\n")


def _stacked_repo(tmp: Path) -> tuple[Path, str]:
    """A repo with two stacked changes: a adds ``rcfg``, then b adds ``state`` after it (b's base needs a)."""
    def commit(msg: str) -> str:
        _git("add", "-A", cwd=tmp)
        _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg, cwd=tmp)
        return _git("rev-parse", "HEAD", cwd=tmp).strip()

    _git("init", "-q", cwd=tmp)
    (tmp / "mini").mkdir()
    (tmp / FILE).write_text(BASE)
    root_sha = commit("base")
    _manifest(tmp, "ta", prologue="rcfg = Config()")
    assert rag_split.main(["apply", "--root", str(tmp), str(tmp / "Helper_Scripts/rag_split/manifests/ta.py")]) == 0
    commit("a")
    _manifest(tmp, "tb", state_init="state = St(rcfg)")
    assert rag_split.main(["apply", "--root", str(tmp), str(tmp / "Helper_Scripts/rag_split/manifests/tb.py")]) == 0
    commit("b")
    _git("branch", "integ", root_sha, cwd=tmp)  # a PR into the integration branch: none of the stack is on it yet
    return tmp, root_sha


def test_manifest_chain_verifies_each_manifest_against_the_tree_it_was_built_on(tmp_path: Path) -> None:
    root, merge_base = _stacked_repo(tmp_path)
    assert [n.rsplit("/", 1)[1] for n, _, _ in _chain(root, merge_base, "integ")] == ["ta.py", "tb.py"]
    assert _verify_chain(root, merge_base, "integ") == []
    # a flat verify of the first manifest against the merge base sees the second one's statement as an extra
    first = str(root / "Helper_Scripts/rag_split/manifests/ta.py")
    assert rag_split.main(["verify", "--root", str(root), first, "--base", merge_base]) == 1
    # hand-breaking the second manifest is caught
    _manifest(root, "tb", state_init="state = Other(rcfg)")
    assert _verify_chain(root, merge_base, "integ") == ["Helper_Scripts/rag_split/manifests/tb.py"]


def test_release_pr_verifies_nothing_because_every_manifest_is_already_on_the_integration_branch(tmp_path: Path) -> None:
    root, merge_base = _stacked_repo(tmp_path)  # merge_base is main's old tip: every manifest is "added"
    _git("branch", "-f", "integ", "HEAD", cwd=root)
    # even a manifest broken in history is not re-proven: its own PR did that
    _manifest(root, "tb", state_init="state = Other(rcfg)")
    _git("add", "-A", cwd=root)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x", cwd=root)
    _git("branch", "-f", "integ", "HEAD", cwd=root)
    assert _chain(root, merge_base, "integ") == []
    assert _verify_chain(root, merge_base, "integ") == []


def test_a_later_non_manifest_edit_does_not_break_the_chain(tmp_path: Path) -> None:
    root, merge_base = _stacked_repo(tmp_path)
    (root / FILE).write_text((root / FILE).read_text() + "\n\ndef later():\n    return 1\n")
    _git("add", "-A", cwd=root)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "later", cwd=root)
    assert _verify_chain(root, merge_base, "integ") == []
    # the first manifest on the integration branch: only the second is checked, against the first's tree
    first = _git("log", "--format=%H", "--diff-filter=A", "--", "Helper_Scripts/rag_split/manifests/ta.py", cwd=root).strip()
    _git("branch", "-f", "integ", first, cwd=root)
    assert [n.rsplit("/", 1)[1] for n, _, _ in _chain(root, merge_base, "integ")] == ["tb.py"]
    assert _verify_chain(root, merge_base, "integ") == []


def test_chain_fails_closed_without_the_integration_ref(tmp_path: Path) -> None:
    root, merge_base = _stacked_repo(tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        _verify_chain(root, merge_base, "no-such-ref")


def test_verify_head_reads_the_ref_not_the_working_tree(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root, merge_base = _stacked_repo(tmp_path)
    mf = str(root / "Helper_Scripts/rag_split/manifests/tb.py")
    (root / FILE).write_text((root / FILE).read_text().replace("state = St(rcfg)\n", ""))
    assert rag_split.main(["verify", "--root", str(root), mf, "--base", "HEAD~1"]) == 1
    assert rag_split.main(["verify", "--root", str(root), mf, "--base", "HEAD~1", "--head", "HEAD"]) == 0
    capsys.readouterr()
    assert rag_split.main(["verify", "--root", str(root), mf, "--base", "HEAD~1", "--head", "nope"]) == 1


def test_prologue_and_state_init_must_be_one_statement(tmp_path: Path) -> None:
    for key in ("prologue", "state_init"):
        t = Tree(tmp_path / key, **{key: "a = 1\nb = 2"})
        with pytest.raises(rag_split.SplitError, match="exactly one statement"):
            rag_split.load_manifest(t.manifest_path)


def test_refuses_python_older_than_3_12(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 11, 9, "final", 0))
    assert rag_split.main(["verify", "x.py", "--base", "HEAD"]) == 2


# --- TASK-13950.25 part 1: the tool is parametrised for chat code, and every RAG default is unchanged ---

RAG_MANIFESTS = sorted((REPO_ROOT / "Helper_Scripts/rag_split/manifests").glob("t*.py"))


def test_every_rag_manifest_still_verifies_after_generalisation() -> None:
    """Each RAG manifest verifies at its recorded base against the commit that last touched it, with 0 differences.

    The ``base`` is the manifest's own record, so this needs no integration ref; it does need history
    (CI checks out with ``fetch-depth: 0``).
    """
    if _git("rev-parse", "--is-shallow-repository").strip() == "true":
        pytest.skip("shallow clone: the recorded bases are not reachable")
    assert len(RAG_MANIFESTS) >= 13
    failed = []
    for mf in RAG_MANIFESTS:
        base = rag_split.load_manifest(mf)["base"]
        head = _git("log", "-1", "--format=%H", "HEAD", "--", str(mf.relative_to(REPO_ROOT))).strip()
        assert base and head, mf
        if rag_split.main(["verify", "--root", str(REPO_ROOT), str(mf), "--base", base, "--head", head]) != 0:
            failed.append(mf.name)
    assert failed == []


CHAT_FILE = "mini/chat_service.py"
CHAT_BASE = textwrap.dedent(
    """\
    import os
    import time

    LIMIT = 5


    def audit(*args):
        return args


    async def call(a):
        \"\"\"Doc.\"\"\"
        x = os.getenv("A")
        t0 = time.time()
        audit(LIMIT, x, t0)
        return a
    """
)
CHAT = {
    "file": CHAT_FILE,
    "func": "call",
    "ctx": [["cst", "Call"]],
    "ctx_import": ["from .types import Call"],
    "facade": "cs",
    "facade_stdlib": ["time"],
    "task": "TASK-X",
    "prologue": "cst = Call()",
}


def chat_tree(tmp: Path, src: str = CHAT_BASE, noqa: list[int] | None = None, **extra: object) -> Tree:
    mv = {"name": "step", "module": "stage_a", "span": [row(src, "x = os"), row(src, "audit(LIMIT")], "doc": "Step."}
    if noqa:
        mv["noqa"] = noqa
    return Tree(tmp, src, **{**CHAT, "moves": [mv], **extra})


def test_chat_keys_apply_and_verify(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = chat_tree(tmp_path)
    assert t.run("apply") == 0
    stage = t.text(f"{PKG}/stage_a.py")
    assert stage.startswith('"""stage_a code moved from chat_service.py (TASK-X)."""')
    assert 'def step(cst: "Call") -> None:' in stage
    assert "cs.audit(cs.LIMIT, x, t0)" in stage and "t0 = cs.time.time()" in stage
    assert 'os.getenv("A")' in stage and "cs.os" not in stage
    assert "from mini import chat_service as cs" in stage and "\nimport os\n" in stage and "\nimport time\n" not in stage
    assert "if TYPE_CHECKING:\n    from .types import Call\n" in stage
    assert "from .run_config" not in stage and "PipelineState" not in stage
    assert "cst = Call()" in t.text(CHAT_FILE) and "_stage_a.step(cst)" in t.text(CHAT_FILE)
    assert t.run("verify") == 0, capsys.readouterr().out


def test_chat_noqa_text_uses_the_file_and_task(tmp_path: Path) -> None:
    t = chat_tree(tmp_path, noqa=[row(CHAT_BASE, "x = os")])
    assert t.run("apply") == 0
    assert '  # noqa: BLE001 - moved from chat_service.py (TASK-X)' in t.text(f"{PKG}/stage_a.py")


def test_chat_verify_rejects_a_bare_facade_stdlib_read(tmp_path: Path) -> None:
    t = chat_tree(tmp_path)
    assert t.run("apply") == 0
    _edit(t, f"{PKG}/stage_a.py", "cs.time.time()", "time.time()")
    assert t.run("verify") == 1


def test_chat_verify_rejects_the_rag_facade_name(tmp_path: Path) -> None:
    t = chat_tree(tmp_path)
    assert t.run("apply") == 0
    _edit(t, f"{PKG}/stage_a.py", "cs.audit(", "up.audit(")
    assert t.run("verify") == 1


@pytest.mark.parametrize("name", ["cst", "cs"])
def test_v8_reserves_the_ctx_names_and_the_facade(tmp_path: Path, capsys: pytest.CaptureFixture, name: str) -> None:
    t = chat_tree(tmp_path, CHAT_BASE.replace("return a", f"{name} = 1\n    return a"))
    assert t.run("apply") == 1
    assert f"V8: {name} is already a name" in capsys.readouterr().out


def test_ctx_names_are_not_reported_as_crossing_locals(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = CHAT_BASE.replace('x = os.getenv("A")', 'cst = 1\n    x = os.getenv("A")').replace("audit(LIMIT, x, t0)", "audit(LIMIT, x, t0, cst)")
    mv = {"name": "step", "module": "stage_a", "span": [row(src, "x = os"), row(src, "audit(LIMIT")]}
    t = Tree(tmp_path, src, **{**CHAT, "prologue": "", "moves": [mv]})
    assert t.run("shared") == 0
    assert "cst" not in capsys.readouterr().out


def test_state_class_and_state_module_are_parametrised(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = "def helper2():\n    pass\n\n\nasync def pipe(q):\n    def _h():\n        return helper2()\n\n    return _h()\n"
    t = Tree(tmp_path, src, rewrite={"_h": "state.h"}, hoist={"_h": "h"}, state_class="ChatState", state_module="chat_state")
    assert t.run("apply") == 0
    state = t.text(f"{PKG}/chat_state.py")
    assert "class ChatState:" in state and "def h(self)" in state and "run_state" not in t.text(FILE)
    assert "from .pipeline import chat_state as _chat_state" in t.text(FILE)
    assert t.run("verify") == 0, capsys.readouterr().out


def test_is_pkg_uses_the_manifests_package_name() -> None:
    text = "from .chatpkg import thing\n\n\ndef f():\n    return thing\n"
    assert "thing" in rag_split.Src(text, "f", pkg="chatpkg").exempt
    assert "thing" not in rag_split.Src(text, "f").exempt


# --- fix round 1: refusals ---


@pytest.mark.parametrize("key", ["late_imports", "rewrite_in", "free_imports", "relocate"])
def test_part_2_and_3_manifest_keys_are_refused(tmp_path: Path, key: str) -> None:
    t = Tree(tmp_path, **{key: {}})
    with pytest.raises(rag_split.SplitError, match=rf"{key}.*not supported yet \(TASK-13950.25 part 2/3\)"):
        rag_split.load_manifest(t.manifest_path)


@pytest.mark.parametrize(("key", "hint"), [("rewrite_in", "not supported yet"), ("typo", "unknown")])
def test_unsupported_move_keys_are_refused(tmp_path: Path, key: str, hint: str) -> None:
    mv = {**moves()[0], key: ["e"]}
    t = Tree(tmp_path, moves=[mv])
    with pytest.raises(rag_split.SplitError, match=rf"move step_one: keys \['{key}'\] are {hint}"):
        rag_split.load_manifest(t.manifest_path)
    assert t.run("apply") == 1


def test_publish_is_refused_when_the_facade_is_cs(tmp_path: Path) -> None:
    t = Tree(tmp_path, publish=["_fmt"], facade="cs")
    with pytest.raises(rag_split.SplitError, match="publish is refused when facade is 'cs'"):
        rag_split.load_manifest(t.manifest_path)


# --- fix round 1: apply's RAG output is pinned against the pre-generalisation tool ---

# sha256 of every file each RAG manifest's apply writes or changes, run on the files at the manifest's recorded base.
# The digests were computed with the tool as of b768507f15 (before TASK-13950.25 part 1), never with the current
# one. They must only change with a deliberate, reviewed change to RAG's apply output.
RAG_APPLY_DIGESTS: dict[str, dict[str, str]] = {
    "t03_config": {
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "8f6e43b07719551bac6ea2d9638ca328a27b0a1c86a3aa88e1bb654f03527e1a",
    },
    "t04a_state": {
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "5611c65df704abff3191b270129cdb70284df1bcaf14fcb623e46de99f6dcba4",
    },
    "t04b_hoist": {
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/run_state.py": "d746ccf055842f34c2a1082f03c028d9c258d0837e247b9d86207281a5820dfa",
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "f9fb09e3dbda4e72dee75e7d11ff08e4a0f2ab8d5019feb2d76041b7b48c2513",
    },
    "t04c_shared": {
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "377b25c7e148bf24ef0f634924be791c049d100334facb7ae0d2090d386673c6",
    },
    "t05e_post_processing": {
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/post_processing.py": "faea942e00a6ddd4064ee00c7492432997bca2b4386f13a0ecb4525ed2fc7350",
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "4c973a497375494132b96d338eb2c9a0a46a55a56318cb2a209b940ba7db1cf3",
    },
    "t05f_verification_claims": {
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/claims.py": "32d5e7fe0eb2b510ccddbed5d6ef0c255b5347e88ea248d6acd312368290ab1d",
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/verification.py": "b1f94ca8245d35f7ad36443f4a777af96942521ff989f66d43243b2d43513ded",
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "a6a711dfee8d95074f58c3184d434b2837d4689106225dadbb0c92bac47c79b5",
    },
    "t06g_generation": {
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/generation.py": "e5274ee30c84692cf1ada9660a5c0099708c90735bd9b9cc7493cf5499339293",
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "02a18dc515602db99e8a39a2f3d3496ffe54cb0d98f4c84f09de0f01d83b051c",
    },
    "t06h_rerank_post_retrieval": {
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/post_retrieval.py": "8c74cfd995dd8a6b19f3188bc911227fd5aeb0f12f5f62e67f8dbfe665f75817",
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/rerank.py": "51b3fa65de2fc114e4844fe1d81e39f94a3d428bfd23251609f268ec101915e4",
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "2b7651c819c9254340d6fb36b8c66cd33a91268741c4a701ceeea3009c6b7292",
    },
    "t07i1_documents": {
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/processing.py": "e60bfb2fa114c5be42a979235ddaa6837b5422ccb2450ac461985caa972435bf",
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/self_correct.py": "cc8a45ae8c72c2ec2dcd62c7b1315e4c2067e36520877e4b20e0ce754473a153",
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "8b8853c747dff907693dd0219586bdccf6e73d54a26e6b5b3926fe4e3dc7362e",
    },
    "t07i2_augment_filters": {
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/augment.py": "1fda7c28cd79bde5e67e3416f5efa80996f0bebd116b927cfb48796417d15f45",
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/filters.py": "6959ca8fa382c6327bcdbc2d76ab2d4bda84ba3c571bde8be877c53278af90d6",
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "35923d0acb8d9f561d356eea08e8a5da31a4d80f61e60c1651f83ee81f7477b0",
    },
    "t08j_retrieval": {
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/retrieval.py": "a80ccbc6ddb8891673b33d1fdffb963a0be09beddbf7518ec60d7c86e0839f3c",
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "2e4571d108402b314aa27069531367bd8916c47fbfb85568826a2cdbd699c46c",
    },
    "t09k_scope_query_prep_research": {
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/query_prep.py": "b9ff6fedcc72f94a978c11ac1aad7e07923e786d7ff15c4311346039d497b192",
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/research.py": "4bc3203243b5c1bb29ee94c36c3bce1c8fd8fc143ff3846202657bb05cfbb6ae",
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/scope.py": "75eaeb297698f22dd9762da6952e55262ec5ad63d7932d73743164e47407a947",
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "7f69b9a015aed450ffc7de86789dc0d558ba53d4097ceb6779e431f168c58c77",
    },
    "t10l_entry": {
        "tldw_Server_API/app/core/RAG/rag_service/pipeline/prepare.py": "9816484eea093d1b6683cf1121e382f134a25dd97ea0d8013632abac97da5e97",
        "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": "c7144027eb500cc90065573bde150924f091f556f5eb0dd4a0714b7912b51b73",
    },
}


def _have(ref: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{ref}^{{commit}}"], cwd=REPO_ROOT, capture_output=True).returncode == 0  # noqa: S603, S607  # nosec B603 B607


@pytest.mark.parametrize("mf", RAG_MANIFESTS, ids=lambda p: p.stem)
def test_rag_apply_output_is_unchanged_by_generalisation(mf: Path, tmp_path: Path) -> None:
    m = rag_split.load_manifest(mf)
    if not _have(m["base"]):
        pytest.skip(f"base {m['base']} is not available locally; fetch full history (CI uses fetch-depth: 0)")
    pkg = rag_split.pkg_rel(m).as_posix()
    before: dict[str, str] = {}
    for rel in sorted({m["file"], *_git("ls-tree", "-r", "--name-only", m["base"], "--", pkg).split()}):
        before[rel] = _git("show", f"{m['base']}:{rel}")
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(before[rel], encoding="utf-8")
    assert rag_split.main(["apply", "--root", str(tmp_path), str(mf)]) == 0
    got = {}
    for p in sorted(tmp_path.rglob("*.py")):
        rel = p.relative_to(tmp_path).as_posix()
        text = p.read_text(encoding="utf-8")
        if before.get(rel) != text:
            got[rel] = hashlib.sha256(text.encode()).hexdigest()
    assert got == RAG_APPLY_DIGESTS[mf.stem]


# --- TASK-13950.25 part 2: destinations that already exist (pkg "", late-import block) ---

SIB_FILE = "mini/chat_service.py"
SIB_BASE = textwrap.dedent(
    '''\
    import os
    import time
    from typing import Any

    LIMIT = 5


    def audit(*args):
        return args


    async def call(a):
        """Doc."""
        cst = make(a)
        x = os.getenv("A")
        t0 = time.time()
        audit(LIMIT, x, t0)
        return cst
    '''
)
STAGE_A = textwrap.dedent(
    '''\
    """Stage A."""

    from __future__ import annotations

    import json
    from typing import Any


    def existing(x: Any) -> str:
        return json.dumps(x)


    __all__ = ["existing"]
    '''
)
STAGE_B = textwrap.dedent(
    '''\
    """Stage B."""

    from __future__ import annotations

    from typing import TYPE_CHECKING, Any

    if TYPE_CHECKING:
        from .types import Other


    def other(x: Other) -> Any:
        return x
    '''
)
SIB = {
    "file": SIB_FILE,
    "func": "call",
    "pkg": "",
    "ctx": [["cst", "Call"]],
    "ctx_import": ["from .types import Call"],
    "facade": "cs",
    "facade_stdlib": ["time"],
    "task": "TASK-X",
}


def sib_tree(tmp: Path, src: str = SIB_BASE, dests: dict[str, str] | None = None, **extra: object) -> Tree:
    dests = {"stage_a": STAGE_A, "stage_b": STAGE_B} if dests is None else dests
    t = Tree(tmp, src, **{**SIB, **extra})
    for d in (t.base, t.head):
        for name, text in dests.items():
            (d / "mini" / f"{name}.py").write_text(text)
    return t


def sib_move(src: str = SIB_BASE, **extra: object) -> dict:
    return {"name": "step", "module": "stage_a", "span": [row(src, "x = os"), row(src, "audit(LIMIT")], "doc": "Step.", **extra}


def test_sibling_append_goes_above_a_late_block_and_verifies(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = sib_tree(tmp_path, moves=[sib_move()])
    assert t.run("apply") == 0
    stage = t.text("mini/stage_a.py")
    assert stage.index("def step(cst") > stage.index('__all__ = ["existing"]')
    assert stage.index("def step(cst") < stage.index(LATE) < stage.index("from mini import chat_service as cs")
    assert stage.rstrip().endswith("from mini import chat_service as cs")
    assert stage.index("import os") < stage.index("def existing") and "import time" not in stage
    assert "from typing import TYPE_CHECKING\n" in stage and "if TYPE_CHECKING:\n    from .types import Call\n" in stage
    assert "cs.audit(cs.LIMIT, x, t0)" in stage and "t0 = cs.time.time()" in stage
    assert "await _stage_a.step(cst)" not in t.text(SIB_FILE) and "_stage_a.step(cst)" in t.text(SIB_FILE)
    assert "from mini import stage_a as _stage_a" in t.text(SIB_FILE)
    assert t.run("verify") == 0, capsys.readouterr().out


LATE = "# late imports: moved code reads these at call time (TASK-13950.25 R5)"


def test_sibling_append_reuses_an_existing_type_checking_block(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = sib_tree(tmp_path, moves=[{**sib_move(), "module": "stage_b"}])
    assert t.run("apply") == 0
    stage = t.text("mini/stage_b.py")
    assert stage.count("if TYPE_CHECKING:") == 1 and "    from .types import Call\n" in stage
    assert t.run("verify") == 0, capsys.readouterr().out


SIB_TWO = SIB_BASE.replace("    return cst\n", "    y = os.getenv(\"B\")\n    audit(y)\n    return cst\n")


def test_a_later_manifest_appends_above_the_block_the_first_one_made(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = sib_tree(tmp_path, SIB_TWO, moves=[sib_move(SIB_TWO)])
    assert t.run("apply") == 0
    shutil.copytree(t.head, t.base, dirs_exist_ok=True)  # the head is the next manifest's base
    second = t.text(SIB_FILE)
    t.set(**{**SIB, "moves": [{"name": "again", "module": "stage_a", "span": [row(second, "y = os"), row(second, "audit(y)")]}]})
    assert t.run("apply") == 0
    stage = t.text("mini/stage_a.py")
    assert stage.count(LATE) == 1 and stage.count("from mini import chat_service as cs") == 1
    assert stage.index("def step(") < stage.index("def again(") < stage.index(LATE)
    assert t.run("verify") == 0, capsys.readouterr().out


def test_sibling_verify_rejects_a_changed_late_block_or_a_def_below_it(tmp_path: Path) -> None:
    t = sib_tree(tmp_path, moves=[sib_move()])
    assert t.run("apply") == 0
    _edit(t, "mini/stage_a.py", "from mini import chat_service as cs", "from mini import other as cs")
    assert t.run("verify") == 1


def test_sibling_verify_rejects_a_missing_marker(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = sib_tree(tmp_path, moves=[sib_move()])
    assert t.run("apply") == 0
    _edit(t, "mini/stage_a.py", LATE + "\n", "")
    assert t.run("verify") == 1
    assert "no marker comment" in capsys.readouterr().out


def test_sibling_verify_rejects_module_level_code_that_reads_the_facade(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = sib_tree(tmp_path, moves=[sib_move()])
    assert t.run("apply") == 0
    _edit(t, "mini/stage_a.py", '__all__ = ["existing"]', 'WHERE = cs.LIMIT\n__all__ = ["existing"]')
    assert t.run("verify") == 1
    assert "module-level code reads cs" in capsys.readouterr().out


def test_sibling_verify_rejects_an_import_parity_break(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = sib_tree(tmp_path, moves=[sib_move()])
    assert t.run("apply") == 0
    _edit(t, "mini/stage_a.py", "import os\n", "import os as _o\nos = _o\n")
    assert t.run("verify") == 1


def test_sibling_refuses_a_destination_name_clash(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = sib_tree(tmp_path, dests={"stage_a": STAGE_A.replace("def existing", "def step"), "stage_b": STAGE_B}, moves=[sib_move()])
    assert t.run("apply") == 1
    assert "V10: stage_a.py already binds step" in capsys.readouterr().out


def test_sibling_refuses_a_missing_destination(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = sib_tree(tmp_path, dests={"stage_b": STAGE_B}, moves=[sib_move()])
    assert t.run("apply") == 1
    assert "stage_a.py: not found at the base" in capsys.readouterr().out


@pytest.mark.parametrize("key", ["hoist", "classes", "entry_split", "prologue", "state_init", "hidden", "publish"])
def test_sibling_refuses_the_keys_that_need_a_state_module(tmp_path: Path, key: str) -> None:
    t = sib_tree(tmp_path, moves=[sib_move()], **{key: {"a": 1} if key not in ("prologue", "state_init", "entry_split", "publish") else ("x = 1" if key != "entry_split" else True)})
    with pytest.raises(rag_split.SplitError):
        rag_split.load_manifest(t.manifest_path)


# --- lift (depth 1, R10) ---

LIFT_BASE = textwrap.dedent(
    '''\
    import os
    import time
    from typing import Any

    LIMIT = 5


    def audit(*args):
        return args


    def register(fn):
        return fn


    async def call(a):
        """Doc."""
        cst = make(a)

        def _note(msg: Any) -> int:
            """Note."""
            cst.log.append(audit(msg, LIMIT))
            return cst.count

        async def _outer(msg):
            _note(msg)
            return await audit(_note)

        x = os.getenv("A")
        t0 = time.time()
        _note(x)
        register(_note)
        y = await _outer(t0)
        return cst
    '''
)


def lift_tree(tmp: Path, src: str = LIFT_BASE, lift: dict | None = None, moves_: list[dict] | None = None, **extra: object) -> Tree:
    lift = {"_note": "stage_a", "_outer": "stage_a"} if lift is None else lift
    return sib_tree(tmp, src, lift=lift, moves=moves_ or [], **extra)


def test_lift_makes_a_module_function_and_leaves_a_partial(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = lift_tree(tmp_path)
    assert t.run("apply") == 0, capsys.readouterr().out
    stage = t.text("mini/stage_a.py")
    assert 'def _note(cst: "Call", msg: Any) -> int:\n    """Note."""\n    cst.log.append(cs.audit(msg, cs.LIMIT))' in stage
    assert 'async def _outer(cst: "Call", msg):\n    _note(cst, msg)\n    return await cs.audit(functools.partial(_note, cst))' in stage
    assert stage.index("def _note(") < stage.index("def _outer(") < stage.index(LATE)
    assert "import functools\n" in stage
    head = t.text(SIB_FILE)
    assert "_note = functools.partial(_stage_a._note, cst)" in head and "_outer = functools.partial(_stage_a._outer, cst)" in head
    assert "def _note" not in head and "register(_note)" in head and "await _outer(t0)" in head
    assert "import functools\n" in head and "from mini import stage_a as _stage_a" in head
    assert t.run("verify") == 0, capsys.readouterr().out


def test_a_move_reads_a_lifted_name_through_the_module_that_holds_it(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = LIFT_BASE
    for module, call in (("stage_a", "_note(cst, x)"), ("stage_b", "_stage_a._note(cst, x)")):
        t = lift_tree(tmp_path / module, src, moves_=[{"name": "step", "module": module, "span": [row(src, "x = os"), row(src, "y = await")]}])
        assert t.run("apply") == 0, capsys.readouterr().out
        body = t.text(f"mini/{module}.py")
        assert call in body and "functools.partial(" + ("_note" if module == "stage_a" else "_stage_a._note") + ", cst)" in body
        assert t.run("verify") == 0, capsys.readouterr().out
    other = t.text("mini/stage_b.py")
    assert other.index("from mini import stage_a as _stage_a") < other.index("from mini import chat_service as cs")


def test_a_lift_inside_a_move_is_applied_first(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    span = [row(LIFT_BASE, "def _note"), row(LIFT_BASE, "y = await")]
    # same module as the move: the partial would be an unbound local, so the def goes and every read is rewritten
    t = lift_tree(tmp_path / "same", lift={"_note": "stage_a", "_outer": "stage_a"}, moves_=[{"name": "step", "module": "stage_a", "span": span}])
    assert t.run("apply") == 0, capsys.readouterr().out
    step = t.text("mini/stage_a.py").split("def step(")[1]
    assert "_note = " not in step and "_outer = " not in step and "_note(cst, x)" in step and "await _outer(cst, t0)" in step
    assert t.run("verify") == 0, capsys.readouterr().out
    # another module: the partial stays at the def's place, and the reads in that move keep using it
    t = lift_tree(tmp_path / "other", lift={"_note": "stage_a", "_outer": "stage_a"}, moves_=[{"name": "step", "module": "stage_b", "span": span}])
    assert t.run("apply") == 0, capsys.readouterr().out
    step = t.text("mini/stage_b.py").split("def step(")[1]
    assert "_note = functools.partial(_stage_a._note, cst)" in step and "\n    _note(x)" in step and "await _outer(t0)" in step
    assert t.run("verify") == 0, capsys.readouterr().out


@pytest.mark.parametrize(
    ("old", "new", "needle"),
    [
        ("return cst.count", "return a", "V11: lifted _note reads a"),
        ("def _note(msg: Any) -> int:", "def _note(msg: Any, extra=LIMIT) -> int:", "V12"),
        ("        return cst.count\n", "        nonlocal x\n        return cst.count\n", "V11: lifted _note declares nonlocal"),
        ("cst.log.append(audit(msg, LIMIT))", "cst.log.append(__name__)", "V1: lifted _note reads __name__"),
        ("    x = os.getenv", "    _note = 1\n    x = os.getenv", "V11: _note has a second binding"),
        ("register(_note)", "_note.flush = 1\n    _outer.flush = 2", "V11: line"),
    ],
)
def test_lift_refusals(tmp_path: Path, capsys: pytest.CaptureFixture, old: str, new: str, needle: str) -> None:
    src = LIFT_BASE.replace(old, new, 1)
    t = lift_tree(tmp_path, src, moves_=[{"name": "step", "module": "stage_b", "span": [row(src, "x = os"), row(src, "y = await")]}] if "flush" in new else None)
    got = t.run("apply")
    out = capsys.readouterr().out
    assert got == 1 and needle in out, out


def test_lift_refuses_a_missing_def_and_a_clash(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = lift_tree(tmp_path / "a", lift={"_nope": "stage_a"})
    assert t.run("apply") == 1 and "V11: expected one def _nope" in capsys.readouterr().out
    t = lift_tree(tmp_path / "b", dests={"stage_a": STAGE_A + "\n\ndef _note():\n    pass\n", "stage_b": STAGE_B}, lift={"_note": "stage_a"})
    assert t.run("apply") == 1 and "V10: stage_a.py already binds _note" in capsys.readouterr().out


def test_lift_decorators_and_defaults_must_not_read_the_facade(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = LIFT_BASE.replace("def _note(msg: Any) -> int:", "def _note(msg: Any, n=LIMIT) -> int:")
    t = lift_tree(tmp_path / "d", src, lift={"_note": "stage_a"})
    assert t.run("apply") == 1 and "V12" in capsys.readouterr().out
    src = LIFT_BASE.replace("        def _note(msg: Any) -> int:", "        @register\n        def _note(msg: Any) -> int:")
    src = LIFT_BASE.replace("    def _note(msg: Any) -> int:", "    @audit\n    def _note(msg: Any) -> int:")
    t = lift_tree(tmp_path / "e", src, lift={"_note": "stage_a"})
    assert t.run("apply") == 1 and "V12" in capsys.readouterr().out


def test_lift_annotations_may_read_the_facade_only_under_the_future_import(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = LIFT_BASE.replace("def _note(msg: Any) -> int:", "def _note(msg: Any) -> LIMIT:")
    t = lift_tree(tmp_path / "with", src, lift={"_note": "stage_a"})
    assert t.run("apply") == 0, capsys.readouterr().out
    assert "-> cs.LIMIT:" in t.text("mini/stage_a.py") and t.run("verify") == 0
    no_future = STAGE_A.replace("from __future__ import annotations\n\n", "")
    t = lift_tree(tmp_path / "without", src, dests={"stage_a": no_future, "stage_b": STAGE_B}, lift={"_note": "stage_a"})
    assert t.run("apply") == 1 and "V13: stage_a.py has no `from __future__ import annotations`" in capsys.readouterr().out


def test_lift_verify_rejects_an_unrewritten_read_and_a_missing_partial(tmp_path: Path) -> None:
    t = lift_tree(tmp_path)
    assert t.run("apply") == 0
    _edit(t, "mini/stage_a.py", "_note(cst, msg)", "_note(msg)")
    assert t.run("verify") == 1
    t = lift_tree(tmp_path / "b")
    assert t.run("apply") == 0
    _edit(t, SIB_FILE, "_note = functools.partial(_stage_a._note, cst)", "pass")
    assert t.run("verify") == 1


def test_lift_needs_pkg_empty(tmp_path: Path) -> None:
    t = Tree(tmp_path, lift={"_h": "stage_a"})
    with pytest.raises(rag_split.SplitError, match="lift needs pkg ''"):
        rag_split.load_manifest(t.manifest_path)


# --- args ---

ARGS_BASE = textwrap.dedent(
    '''\
    import os
    import time
    from typing import Any

    LIMIT = 5


    def audit(*args):
        return args


    async def call(a):
        """Doc."""
        cst = make(a)
        try:
            cst.n = int(a)
        except ValueError as e:
            text = str(e)
            audit(text, LIMIT)
            cst.err = os.getenv(text)
        return cst
    '''
)


def args_tree(tmp: Path, src: str = ARGS_BASE, args: list[str] | None = None, **extra: object) -> Tree:
    mv = {"name": "fail", "module": "stage_a", "span": [row(src, "text = str"), row(src, "cst.err = ")], "args": ["e"] if args is None else args}
    return sib_tree(tmp, src, moves=[mv], **extra)


def test_move_args_are_parameters_after_the_ctx(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = args_tree(tmp_path)
    assert t.run("apply") == 0, capsys.readouterr().out
    assert 'def fail(cst: "Call", e) -> None:\n    text = str(e)' in t.text("mini/stage_a.py")
    assert "        _stage_a.fail(cst, e)\n" in t.text(SIB_FILE)
    assert t.run("verify") == 0, capsys.readouterr().out


def test_without_args_the_handler_variable_crosses(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = args_tree(tmp_path, args=[])
    assert t.run("apply") == 1 and "V4: crossing local e" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("src_edit", "args", "needle"),
    [
        (("audit(text, LIMIT)", "e = 1\n        audit(text, LIMIT)"), ["e"], "V3: fail: args e is rebound inside the span"),
        (None, ["zz"], "V3: fail: args zz is not read in the span"),
        (("str(e)", "str(e) + str(LIMIT)"), ["LIMIT"], "V3: fail: args LIMIT is not read in the span"),
        (None, ["cst"], "V3: fail: args cst is a ctx or lifted name"),
    ],
)
def test_move_args_refusals(tmp_path: Path, capsys: pytest.CaptureFixture, src_edit: tuple[str, str] | None, args: list[str], needle: str) -> None:
    src = ARGS_BASE.replace(*src_edit) if src_edit else ARGS_BASE
    t = args_tree(tmp_path, src, args=args)
    assert t.run("apply") == 1 and needle in capsys.readouterr().out


def test_move_args_must_be_distinct_names(tmp_path: Path) -> None:
    t = args_tree(tmp_path, args=["e", "e"])
    with pytest.raises(rag_split.SplitError, match="args must be a list of distinct names"):
        rag_split.load_manifest(t.manifest_path)


def test_verify_rejects_a_changed_args_signature(tmp_path: Path) -> None:
    t = args_tree(tmp_path)
    assert t.run("apply") == 0
    _edit(t, "mini/stage_a.py", "cst: \"Call\", e)", "cst: \"Call\", f)")
    assert t.run("verify") == 1


# --- entry ---

ENTRY_BASE = textwrap.dedent(
    '''\
    import os
    import time
    from typing import Any

    LIMIT = 5


    def audit(*args):
        return args


    async def call(a) -> Any:
        """Doc."""
        cst = make(a)

        def _note(msg):
            cst.log.append(audit(msg, LIMIT))

        x = os.getenv("A")
        t0 = time.time()
        audit(LIMIT, x, t0)
        _note(x)
        if cst.flag:
            return cst
        return await audit(cst)
    '''
)
ENTRY = {"name": "run_call", "module": "stage_a", "doc": "Run it."}


def entry_tree(tmp: Path, src: str = ENTRY_BASE, move_module: str = "stage_b", **extra: object) -> Tree:
    mv = {"name": "step", "module": move_module, "span": [row(src, "x = os"), row(src, "_note(x)")]}
    return sib_tree(tmp, src, lift={"_note": "stage_a"}, moves=[mv], entry=ENTRY, **extra)


def test_entry_moves_the_body_into_another_module_and_leaves_a_shim(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = entry_tree(tmp_path)
    assert t.run("apply") == 0, capsys.readouterr().out
    head = t.text(SIB_FILE)
    assert head.split("async def call(a) -> Any:")[1].strip().startswith('"""Doc."""\n    return await _stage_a.run_call(make(a))')
    assert "_stage_b" not in head and "functools" not in head and "from mini import stage_a as _stage_a" in head
    stage = t.text("mini/stage_a.py")
    entry = stage.split("async def run_call(")[1].split(LATE)[0]
    assert entry.startswith('cst: "Call") -> Any:\n    """Run it."""\n\n    _stage_b.step(cst)\n    if cst.flag:\n')
    assert "_note" not in entry and "_stage_b._note" not in entry
    assert "_stage_a._note(cst, x)" in stage.split("async def step(")[0].split("def step(")[0] + t.text("mini/stage_b.py")
    assert "return await cs.audit(cst)" in entry and "if cst.flag:\n        return cst" in entry
    assert "from mini import stage_b as _stage_b\nfrom mini import chat_service as cs" in stage
    assert stage.index("def _note(") < stage.index("async def run_call(") < stage.index(LATE)
    assert t.run("verify") == 0, capsys.readouterr().out


def test_entry_calls_a_move_in_its_own_module_bare(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = entry_tree(tmp_path, move_module="stage_a")
    assert t.run("apply") == 0, capsys.readouterr().out
    entry = t.text("mini/stage_a.py").split("async def run_call(")[1]
    assert "    step(cst)\n" in entry and "_stage_a" not in entry.split("def step(")[0] and "    _note(cst, x)\n" in t.text("mini/stage_a.py").split("def step(")[1]
    assert t.run("verify") == 0, capsys.readouterr().out


def test_entry_keeps_lift_partials_for_other_modules(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = sib_tree(tmp_path, ENTRY_BASE, lift={"_note": "stage_b"}, moves=[], entry=ENTRY)
    assert t.run("apply") == 0, capsys.readouterr().out
    entry = t.text("mini/stage_a.py").split("async def run_call(")[1]
    assert "_note = functools.partial(_stage_b._note, cst)" in entry and "_note(x)" in entry
    assert t.run("verify") == 0, capsys.readouterr().out


@pytest.mark.parametrize(
    ("old", "new", "needle"),
    [
        ("    cst = make(a)\n", "    cst = make(a)\n    cst = 2\n", "cst is bound outside the prologue statement"),
        ("cst = make(a)", "cst = make(a)\n    audit(a)", "V8: the entry body reads the parameter a"),
        ("return await audit(cst)", "return await audit(locals())", "V1: the entry body calls locals()"),
        ("return await audit(cst)", "return await audit(__name__)", "V1: the entry body reads __name__"),
        ('    cst = make(a)\n', '    audit(a)\n', "V8: entry needs `cst = <expr>`"),
    ],
)
def test_entry_refusals(tmp_path: Path, capsys: pytest.CaptureFixture, old: str, new: str, needle: str) -> None:
    src = ENTRY_BASE.replace(old, new, 1)
    t = sib_tree(tmp_path, src, lift={}, moves=[], entry=ENTRY)
    assert t.run("apply") == 1 and needle in capsys.readouterr().out


def test_entry_name_clash_and_shape(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = sib_tree(tmp_path, ENTRY_BASE, dests={"stage_a": STAGE_A + "\n\ndef run_call():\n    pass\n", "stage_b": STAGE_B}, entry=ENTRY)
    assert t.run("apply") == 1 and "V10: stage_a.py already binds run_call" in capsys.readouterr().out
    t = sib_tree(tmp_path / "x", entry={"name": "r"})
    with pytest.raises(rag_split.SplitError, match="entry must be"):
        rag_split.load_manifest(t.manifest_path)
    t = Tree(tmp_path / "y", entry={"name": "r", "module": "m"})
    with pytest.raises(rag_split.SplitError, match="entry needs pkg ''"):
        rag_split.load_manifest(t.manifest_path)


def test_entry_verify_rejects_an_edited_shim_or_body(tmp_path: Path) -> None:
    t = entry_tree(tmp_path)
    assert t.run("apply") == 0
    _edit(t, SIB_FILE, "run_call(make(a))", "run_call(make(None))")
    assert t.run("verify") == 1
    t = entry_tree(tmp_path / "b")
    assert t.run("apply") == 0
    _edit(t, "mini/stage_a.py", "return await cs.audit(cst)", "return cst")
    assert t.run("verify") == 1


# --- drop ---

DROP_SRC = SIB_BASE.replace(
    "async def call(a):",
    "class Pipe:\n    def go(self, x):\n        return x\n\n\ndef make_pipe():\n    return Pipe()\n\n\ndef run_it(x):\n    return make_pipe().go(x)\n\n\n__all__ = [\"Pipe\", \"make_pipe\", \"run_it\", \"call\"]\n\n\nasync def call(a):",
).replace("from typing import Any\n", "from typing import Any\n\nfrom .holder import Holder, Other\n")
DROP_STAGE = textwrap.dedent(
    '''\
    """Stage A."""

    from __future__ import annotations

    from collections.abc import Awaitable
    from dataclasses import dataclass
    from typing import Any, Callable


    @dataclass
    class Holder:
        run: Callable[..., Awaitable[Any]]


    __all__ = ["Holder"]
    '''
)
DROPS = [
    {"module": "chat_service", "replace": {"in": "run_it", "old": "make_pipe().go(x)", "new": "x"}},
    {"module": "chat_service", "def": "make_pipe"},
    {"module": "chat_service", "def": "Pipe"},
    {"module": "chat_service", "import": "Holder"},
    {"module": "chat_service", "all": ["Pipe", "make_pipe"]},
    {"module": "stage_a", "def": "Holder"},
    {"module": "stage_a", "import": "dataclass"},
    {"module": "stage_a", "import": "Awaitable"},
    {"module": "stage_a", "import": "Callable"},
    {"module": "stage_a", "all": ["Holder"], "all_add": ["step"]},
]


def drop_tree(tmp: Path, drops: list[dict] | None = None, src: str = DROP_SRC, **extra: object) -> Tree:
    return sib_tree(tmp, src, dests={"stage_a": DROP_STAGE, "stage_b": STAGE_B}, drop=DROPS if drops is None else drops, **extra)


def test_drop_deletes_defs_imports_and_all_entries_with_declared_replacements(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = drop_tree(tmp_path, moves=[sib_move(DROP_SRC)])
    assert t.run("apply") == 0, capsys.readouterr().out
    head = t.text(SIB_FILE)
    assert "make_pipe" not in head and "class Pipe" not in head and "Holder" not in head
    assert "def run_it(x):\n    return x\n" in head and '__all__ = ["run_it", "call"]' in head
    stage = t.text("mini/stage_a.py")
    assert "Holder" not in stage and "dataclass" not in stage and "Awaitable" not in stage and "Callable" not in stage
    assert 'from typing import Any' in stage and '__all__ = ["step"]' in stage
    assert t.run("verify") == 0, capsys.readouterr().out


def test_drop_alone_is_enough_for_a_manifest(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    t = drop_tree(tmp_path)
    assert t.run("apply") == 0, capsys.readouterr().out
    assert t.run("verify") == 0, capsys.readouterr().out


@pytest.mark.parametrize(
    ("drop", "needle"),
    [
        ({"module": "chat_service", "def": "make_pipe"}, "V15: chat_service.py: make_pipe is dropped but still read"),
        ({"module": "chat_service", "def": "nope"}, "drop def: expected one top-level nope, found 0"),
        ({"module": "chat_service", "import": "Nope"}, "drop import: no top-level import binds Nope"),
        ({"module": "chat_service", "replace": {"in": "run_it", "old": "zzz", "new": "x"}}, "does not occur in run_it"),
        ({"module": "chat_service", "replace": {"in": "nope", "old": "x", "new": "x"}}, "no top-level function nope"),
        ({"module": "chat_service", "all": ["ghost"]}, "__all__ has no entry ['ghost']"),
        ({"module": "stage_a", "import": "dataclass"}, "V15: stage_a.py: dataclass is dropped but still read"),
        ({"module": "stage_b", "def": "other"}, "ok"),
    ],
)
def test_drop_refusals(tmp_path: Path, capsys: pytest.CaptureFixture, drop: dict, needle: str) -> None:
    t = drop_tree(tmp_path, drops=[drop])
    got = t.run("apply")
    out = capsys.readouterr().out
    if needle == "ok":
        assert got == 0, out
    else:
        assert got == 1 and needle in out, out


def test_drop_refuses_a_dropped_name_the_moved_code_still_reads(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = DROP_SRC.replace('audit(LIMIT, x, t0)', "audit(LIMIT, x, t0, Holder)")
    drops = [{"module": "chat_service", "import": "Holder"}]
    t = drop_tree(tmp_path, drops=drops, src=src, moves=[sib_move(src)])
    assert t.run("apply") == 1 and "V15: stage_a.py: step reads cs.Holder (line" in capsys.readouterr().out  # the move reads it through the facade
    t = drop_tree(tmp_path / "b", drops=drops, src=src)
    assert t.run("apply") == 1 and "V15: chat_service.py: Holder is dropped but still read" in capsys.readouterr().out


@pytest.mark.parametrize(
    "item",
    [{"module": "x"}, {"def": "a"}, {"module": "x", "def": "a", "import": "b"}, {"module": "x", "def": "a", "all_add": ["z"]}, {"module": "x", "all": "a"}, {"module": "x", "replace": {"in": "f"}}, {"module": "x", "replace": {"in": "f", "old": "(", "new": "x"}}, "x"],
)
def test_drop_item_shape_is_checked(tmp_path: Path, item: object) -> None:
    t = drop_tree(tmp_path, drops=[item])  # type: ignore[list-item]
    with pytest.raises(rag_split.SplitError):
        rag_split.load_manifest(t.manifest_path)


def test_drop_verify_rejects_a_kept_def_or_a_stale_all(tmp_path: Path) -> None:
    t = drop_tree(tmp_path)
    assert t.run("apply") == 0
    _edit(t, SIB_FILE, "def run_it(x):", "def make_pipe():\n    return 1\n\n\ndef run_it(x):")
    assert t.run("verify") == 1
    t = drop_tree(tmp_path / "b")
    assert t.run("apply") == 0
    _edit(t, SIB_FILE, '__all__ = ["run_it", "call"]', '__all__ = ["Pipe", "run_it", "call"]')
    assert t.run("verify") == 1
