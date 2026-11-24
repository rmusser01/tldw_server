"""Helper_Scripts/rag_split/rag_split.py on mini pipeline modules in tmp trees (TASK-13950.24).

Each test builds ``mini/unified_pipeline.py`` under a base dir, copies it to a head dir, runs ``apply``
there, and checks the result with ``verify --base-dir`` or by reading the text. No git is needed.
"""

from __future__ import annotations

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
    _edit(t, FILE, "    state = _st.State(cfg)\n", "")
    assert t.run("verify") == 1


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


def test_new_manifests_verify_against_the_pr_base() -> None:
    """R16's proof, gated: every manifest this change adds verifies against the merge base."""
    base = os.environ.get("BACKLOG_TASK_FORMAT_BASE") or "origin/dev"
    try:
        merge_base = _git("merge-base", base, "HEAD").strip()
        added = _git("diff", "--name-only", "--diff-filter=A", merge_base, "--", "Helper_Scripts/rag_split/manifests/*.py").split()
    except (OSError, subprocess.CalledProcessError) as exc:
        if os.environ.get("BACKLOG_TASK_FORMAT_BASE"):
            raise
        pytest.skip(f"cannot diff against {base}; fetch it or set BACKLOG_TASK_FORMAT_BASE ({exc})")
    for name in added:
        assert rag_split.main(["verify", str(REPO_ROOT / name), "--base", merge_base]) == 0, name


def test_refuses_python_older_than_3_12(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 11, 9, "final", 0))
    assert rag_split.main(["verify", "x.py", "--base", "HEAD"]) == 2
