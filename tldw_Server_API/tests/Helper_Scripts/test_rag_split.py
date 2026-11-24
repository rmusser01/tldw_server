"""Helper_Scripts/rag_split/rag_split.py on mini pipeline modules in tmp trees (TASK-13950.24).

Each test builds ``mini/unified_pipeline.py`` under a base dir, copies it to a head dir, runs ``apply``
there, and checks the result with ``verify --base-dir`` or by reading the text. No git is needed.
"""

from __future__ import annotations

import ast
import os
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
            (d / FILE).write_text(src)
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
