"""Move tool for TASK-13950.24: split ``unified_rag_pipeline`` into a typed config and stage modules.

Stdlib only; needs Python 3.12 (exits 2 on anything older).

    python -m Helper_Scripts.rag_split.rag_split shared [--manifest M] [--hoist N1,N2] [--base REV | --base-dir DIR]
    python -m Helper_Scripts.rag_split.rag_split apply M
    python -m Helper_Scripts.rag_split.rag_split verify M (--base REV | --base-dir DIR) [--head REF]

A manifest (``Helper_Scripts/rag_split/manifests/t<NN>_<pr>.py``) holds one ``MANIFEST = {...}`` literal,
read with ``ast.literal_eval`` and never imported. Keys (all optional but ``base``):

    base         the dev sha the manifest was written against
    rewrite      {local name: replacement expression}        e.g. {"top_k": "rcfg.retrieval.top_k"}
    hidden       {kwargs key: [group, default]}              ``kwargs.get("k", d)`` -> ``rcfg.<group>.k``
    publish      [closure name]                              ``state.<n> = <n>`` right after each def
    hoist        {closure name: method name}                 closure -> ``PipelineState`` method
    classes      {local class: module class}                 a class defined in the function becomes a ``run_state`` class
                                                             (its name must also be in ``rewrite``, as ``_run_state.<new>``)
    moves        [{"name", "module", "span": [a, b], "doc", "noqa": [line]}]
    cells        [[a, b]]                                    extra cells for ``shared``
    prologue     "rcfg = ..."                                one statement inserted first in the function, after its docstring
    state_init   "state = ..."                               one statement inserted right after the first statement after the docstring
    imports      ["from .x import Y"]                        module-level imports added after the file's last import
    entry_split  true
    file, func, pkg    the pipeline file, function and package directory (defaults below; tests override)

``apply`` edits the working tree as text edits at AST positions. ``verify`` rebuilds the expected result
from the base *as an AST*, independently of those text edits, and compares it with the head.
Every difference prints one line; any difference exits 1.

Limits:
* V8 only flags names the manifest itself introduces (``rcfg``, ``state``, ``up``, ``_run_state``,
  ``_<module>``); later PRs legitimately use ``rcfg`` and ``state`` already.
* V4 is the crossing definition of A.3: a name live-in at one cell and bound in another, or a closure
  local referenced across cells. A name read before assignment inside one span and bound nowhere
  else is not flagged (it fails the same way before and after the move).
* A leaf in the same module as its wrapper is called bare (``leaf(rcfg, state)``); in another module it is
  ``_module.leaf(rcfg, state)`` with ``from . import module as _module``.
* ``entry_split`` needs ``rcfg = <expr>`` as the first statement after the docstring. The shim becomes
  ``return [await] run_rag_pipeline(<expr>)``; the old body, docstring included, moves into
  ``run_rag_pipeline(rcfg)``.
* Scopes are matched to symtable by (name, line) in visit order; two lambdas on one line can swap.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import io
import re
import subprocess  # nosec B404  # fixed argv, no shell
import symtable
import sys
import tokenize
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

PIPELINE_FILE = "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py"
FUNC = "unified_rag_pipeline"
STATE_CLASS = "PipelineState"
NOQA = "  # noqa: BLE001 - moved from unified_pipeline.py (TASK-13950.24)"
DEFAULTS: dict[str, Any] = {
    "base": "",
    "file": PIPELINE_FILE,
    "func": FUNC,
    "pkg": "pipeline",
    "rewrite": {},
    "hidden": {},
    "publish": [],
    "hoist": {},
    "classes": {},
    "moves": [],
    "cells": [],
    "entry_split": False,
    "prologue": "",
    "state_init": "",
    "imports": [],
}
NEW = object()  # a node the tool created: it carries no scope annotation
SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


class SplitError(Exception):
    """A manifest or source problem that makes the tool refuse to continue."""


# --------------------------------------------------------------------------------------------------
# scope resolution (symtable)
# --------------------------------------------------------------------------------------------------


class Scopes:
    """Annotate every ``Name`` with the scope it resolves to: ``node._r`` is a symtable table, a
    ``("comp", id)`` key, or ``None`` for module globals and builtins.

    symtable (3.12) inlines list/set/dict comprehensions, so their variables are tracked by hand.
    """

    def __init__(self, tree: ast.Module, text: str) -> None:
        self.top = symtable.symtable(text, "<src>", "exec")
        self.tab_of: dict[int, symtable.SymbolTable] = {}
        self._left: dict[int, list[symtable.SymbolTable]] = {}
        self._visit(tree, [("tab", self.top)])

    def _child(self, stack: list, name: str, lineno: int) -> symtable.SymbolTable:
        parent = next(o for k, o in reversed(stack) if k == "tab")
        left = self._left.setdefault(id(parent), list(parent.get_children()))
        if name == "lambda" and sum(1 for t in left if t.get_name() == name and t.get_lineno() == lineno) > 1:
            raise SplitError(f"two lambdas share line {lineno}; symtable cannot tell them apart. Split the line.")
        for pick in (lambda t: t.get_name() == name and t.get_lineno() == lineno, lambda t: t.get_name() == name):
            for i, t in enumerate(left):
                if pick(t):
                    return left.pop(i)
        raise SplitError(f"symtable has no scope for {name} at line {lineno}")

    def _resolve(self, name: str, stack: list) -> Any:
        innermost = True
        for kind, obj in reversed(stack):
            if kind == "comp":
                if name in obj:
                    return ("comp", id(obj))
                continue
            if obj.get_type() == "module":
                return None
            if obj.get_type() == "class" and not innermost:
                continue
            try:
                sym = obj.lookup(name)
            except KeyError:
                innermost = False
                continue
            if sym.is_global():
                return None
            if sym.is_free():
                innermost = False
                continue
            return obj if sym.is_local() else None
        return None

    def _visit_all(self, nodes: Any, stack: list) -> None:
        for n in nodes:
            if n is not None:
                self._visit(n, stack)

    def _visit(self, node: ast.AST, stack: list) -> None:
        if isinstance(node, ast.Name):
            node._r = self._resolve(node.id, stack)  # type: ignore[attr-defined]
        elif isinstance(node, DEFS + (ast.ClassDef,)):
            node._r = self._resolve(node.name, stack)  # type: ignore[attr-defined]
            self._visit_all(node.decorator_list, stack)
            if isinstance(node, ast.ClassDef):
                self._visit_all(node.bases + [k.value for k in node.keywords], stack)
                child = self._child(stack, node.name, node.lineno)
            else:
                a = node.args
                self._visit_all(a.defaults + a.kw_defaults, stack)
                for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
                    if arg is not None and arg.annotation is not None:
                        self._visit(arg.annotation, stack)
                self._visit_all([node.returns], stack)
                child = self._child(stack, node.name, node.lineno)
            self.tab_of[id(node)] = child
            self._visit_all(node.body, stack + [("tab", child)])
        elif isinstance(node, ast.Lambda):
            self._visit_all(node.args.defaults + node.args.kw_defaults, stack)
            child = self._child(stack, "lambda", node.lineno)
            self._visit(node.body, stack + [("tab", child)])
        elif isinstance(node, ast.GeneratorExp):
            self._visit(node.generators[0].iter, stack)
            inner = stack + [("tab", self._child(stack, "genexpr", node.lineno))]
            self._visit_comp(node, inner, skip_first_iter=True)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            self._visit(node.generators[0].iter, stack)
            names = frozenset(n.id for g in node.generators for n in ast.walk(g.target) if isinstance(n, ast.Name))
            self._visit_comp(node, stack + [("comp", names)], skip_first_iter=True)
        else:
            if isinstance(node, ast.ExceptHandler) and node.name:
                node._r = self._resolve(node.name, stack)  # type: ignore[attr-defined]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                node._rs = [self._resolve(_bound(a), stack) for a in node.names]  # type: ignore[attr-defined]
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                node._rs = [self._resolve(n, stack) for n in node.names]  # type: ignore[attr-defined]
            for child in ast.iter_child_nodes(node):
                self._visit(child, stack)

    def _visit_comp(self, node: ast.AST, stack: list, skip_first_iter: bool) -> None:
        for i, g in enumerate(node.generators):  # type: ignore[attr-defined]
            if not (i == 0 and skip_first_iter):
                self._visit(g.iter, stack)
            self._visit(g.target, stack)
            self._visit_all(g.ifs, stack)
        for part in ("elt", "key", "value"):
            if hasattr(node, part):
                self._visit(getattr(node, part), stack)


def _bound(alias: ast.alias) -> str:
    return alias.asname or alias.name.split(".")[0]


def is_f(node: ast.AST, key: Any) -> bool:
    return getattr(node, "_r", NEW) is key


def walk_own(nodes: list[ast.AST]) -> Iterator[ast.AST]:
    """Walk nodes without entering nested defs, lambdas or classes (they are yielded themselves)."""
    todo = list(nodes)
    while todo:
        n = todo.pop()
        yield n
        if not isinstance(n, SCOPE_NODES):
            todo.extend(ast.iter_child_nodes(n))


def stmt_start(s: ast.stmt) -> int:
    return min([s.lineno] + [d.lineno for d in getattr(s, "decorator_list", [])])


def child_blocks(s: ast.stmt) -> list[list[ast.stmt]]:
    if isinstance(s, SCOPE_NODES):
        return []
    out = [v for f in ("body", "orelse", "finalbody") if isinstance(v := getattr(s, f, None), list)]
    out += [h.body for h in getattr(s, "handlers", [])] + [c.body for c in getattr(s, "cases", [])]
    return [b for b in out if b and isinstance(b[0], ast.stmt)]


def own_blocks(fn: ast.AST) -> Iterator[list[ast.stmt]]:
    def walk(block: list[ast.stmt]) -> Iterator[list[ast.stmt]]:
        yield block
        for s in block:
            for sub in child_blocks(s):
                yield from walk(sub)

    yield from walk(fn.body)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------------------
# one parsed source
# --------------------------------------------------------------------------------------------------


class Info(NamedTuple):
    kind: str  # "import" or "from"
    module: str
    name: str
    asname: str | None
    level: int

    @property
    def stdlib(self) -> bool:
        return self.level == 0 and self.module.split(".")[0] in sys.stdlib_module_names and self.module != "__future__"

    def line(self, bump: int = 0) -> str:
        tail = f" as {self.asname}" if self.asname else ""
        if self.kind == "import":
            return f"import {self.name}{tail}"
        return f"from {'.' * (self.level + bump)}{self.module} import {self.name}{tail}"


def walk_module(nodes: list[ast.stmt]) -> Iterator[ast.AST]:
    """Module-level nodes: compound statements are entered, defs and classes are not."""
    for s in nodes:
        yield s
        if not isinstance(s, SCOPE_NODES):
            for b in child_blocks(s):
                yield from walk_module(b)


def import_infos(tree: ast.Module) -> dict[str, Info]:
    out: dict[str, Info] = {}
    for n in walk_module(tree.body):
        if isinstance(n, ast.Import):
            for a in n.names:
                out[_bound(a)] = Info("import", a.name, a.name, a.asname, 0)
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                out[_bound(a)] = Info("from", n.module or "", a.name, a.asname, n.level)
    return out


def module_names(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for n in walk_module(tree.body):
        if isinstance(n, SCOPE_NODES):
            out.add(getattr(n, "name", ""))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update(_bound(a) for a in n.names)
        else:
            for sub in walk_own([n]):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                    out.add(sub.id)
    return out


def _is_pkg(i: Info) -> bool:
    return i.kind == "from" and i.level > 0 and (i.module == "pipeline" or i.module.startswith("pipeline."))


class Src:
    def __init__(self, text: str, func: str | None = None) -> None:
        self.text = text
        self.tree = ast.parse(text)
        self.scopes = Scopes(self.tree, text)
        self.infos = import_infos(self.tree)
        self.names = module_names(self.tree)
        self.fn: Any = None
        self.key: Any = None
        if func:
            self.fn = next((n for n in self.tree.body if isinstance(n, DEFS) and n.name == func), None)
            if self.fn is None:
                raise SplitError(f"no top-level function {func}")
            self.key = self.scopes.tab_of[id(self.fn)]
        self.exempt = {n for n, i in self.infos.items() if i.stdlib or _is_pkg(i)}


def continuation_rows(text: str) -> set[int]:
    """Rows that continue a multi-line string: dedenting them would change the string."""
    rows: set[int] = set()
    fstarts: list[int] = []
    for t in tokenize.generate_tokens(io.StringIO(text).readline):
        if t.type == tokenize.STRING:
            rows.update(range(t.start[0] + 1, t.end[0] + 1))
        elif t.type == tokenize.FSTRING_START:
            fstarts.append(t.start[0])
        elif t.type == tokenize.FSTRING_END:
            rows.update(range(fstarts.pop() + 1, t.end[0] + 1))
    return rows


# --------------------------------------------------------------------------------------------------
# manifest and sources
# --------------------------------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, Any]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MANIFEST" for t in n.targets):
            m = {**DEFAULTS, **ast.literal_eval(n.value)}
            unknown = set(m) - set(DEFAULTS)
            if unknown:
                raise SplitError(f"{path}: unknown manifest keys {sorted(unknown)}")
            for key in ("prologue", "state_init"):
                if m[key] and len(ast.parse(m[key]).body) != 1:
                    raise SplitError(f"{path}: {key} must be exactly one statement")
            return m
    raise SplitError(f"{path}: no MANIFEST literal")


def pkg_rel(m: dict) -> Path:
    return Path(m["file"]).parent / m["pkg"]


def read_base(root: Path, rel: str, base: str | None, base_dir: str | None) -> str | None:
    if base_dir is not None:
        p = Path(base_dir) / rel
        return p.read_text(encoding="utf-8") if p.exists() else None
    if base is None:
        p = root / rel
        return p.read_text(encoding="utf-8") if p.exists() else None
    r = subprocess.run(  # nosec B603 B607  # fixed git argv, no shell
        ["git", "-C", str(root), "show", f"{base}:{rel}"], capture_output=True, text=True
    )
    return r.stdout if r.returncode == 0 else None


# --------------------------------------------------------------------------------------------------
# moves: locating spans
# --------------------------------------------------------------------------------------------------


class Move:
    def __init__(self, spec: dict) -> None:
        self.name: str = spec["name"]
        self.module: str = spec["module"]
        self.a, self.b = spec["span"]
        self.doc: str | None = spec.get("doc")
        self.noqa: list[int] = list(spec.get("noqa", []))
        self.allow_locals: bool = bool(spec.get("allow_locals", False))
        self.block: list[ast.stmt] = []
        self.members: list[ast.stmt] = []
        self.parent: Move | None = None
        self.children: list[Move] = []
        self.is_async = False


def locate(fn: ast.AST, a: int, b: int) -> tuple[list[ast.stmt], list[ast.stmt]]:
    found: list[tuple[list[ast.stmt], list[ast.stmt]]] = []

    def search(block: list[ast.stmt]) -> None:
        inside = []
        for s in block:
            lo, hi = stmt_start(s), s.end_lineno
            if hi < a or lo > b:
                continue
            if lo >= a and hi <= b:
                inside.append(s)
            elif lo < a and hi >= b:
                for sub in child_blocks(s):
                    search(sub)
            else:
                raise SplitError(f"span {a}-{b} cuts through the statement at lines {lo}-{hi}")
        if inside:
            found.append((block, inside))

    search(fn.body)  # type: ignore[attr-defined]
    if len(found) != 1:
        raise SplitError(f"span {a}-{b} is not whole statements of one block")
    block, inside = found[0]
    idx = [next(i for i, x in enumerate(block) if x is s) for s in inside]
    if idx != list(range(idx[0], idx[0] + len(idx))):
        raise SplitError(f"span {a}-{b} is not contiguous")
    return block, inside


def plan_moves(src: Src, specs: list[dict], lines: list[str]) -> list[Move]:
    moves = [Move(s) for s in specs]
    names = [m.name for m in moves]
    if len(set(names)) != len(names):
        raise SplitError("duplicate move names")
    for m in moves:
        m.block, m.members = locate(src.fn, m.a, m.b)
        covered = {r for s in m.members for r in range(stmt_start(s), s.end_lineno + 1)}
        for r in range(m.a, m.b + 1):
            if r not in covered and lines[r - 1].strip() and not lines[r - 1].lstrip().startswith("#"):
                raise SplitError(f"{m.name}: line {r} is in the span but in no statement")
    for m in moves:
        outer = [o for o in moves if o is not m and o.a <= m.a and m.b <= o.b]
        if any((o.a, o.b) == (m.a, m.b) for o in outer):
            raise SplitError(f"{m.name}: same span as another move")
        for o in moves:
            if o is not m and not (o.b < m.a or m.b < o.a) and o not in outer and m not in [x for x in moves if x.a <= o.a and o.b <= x.b]:
                raise SplitError(f"{m.name}: span overlaps {o.name} without containing it")
        m.parent = min(outer, key=lambda o: o.b - o.a, default=None)
        if m.parent:
            m.parent.children.append(m)
    return moves


INTROSPECTION = {"locals", "vars", "globals"}


def check_span(m: Move, notes: list[str] | None = None) -> list[str]:
    """V1: whole statements, no return/yield/global/nonlocal, no break/continue leaving the span,
    and no locals()/vars()/globals()/__name__/__file__ unless the move says ``allow_locals``."""
    errs = []
    mine = lambda n: not any(c.a <= getattr(n, "lineno", m.a) <= c.b for c in m.children)  # noqa: E731
    for n in walk_own(m.members):
        if isinstance(n, (ast.Return, ast.Yield, ast.YieldFrom, ast.Global, ast.Nonlocal)):
            errs.append(f"V1: {m.name}: contains {type(n).__name__.lower()} at line {n.lineno}")
        what = None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in INTROSPECTION and getattr(n.func, "_r", NEW) is None:
            what = f"{n.func.id}()"
        elif isinstance(n, ast.Name) and n.id in ("__name__", "__file__") and isinstance(n.ctx, ast.Load):
            what = n.id
        if what and mine(n):
            if m.allow_locals:
                if notes is not None:
                    notes.append(f"note: {m.name}: allow_locals, uses {what} at line {n.lineno}")
            else:
                errs.append(f"V1: {m.name}: uses {what} at line {n.lineno}; it reads a different scope after the move")

    def scan(block: list[ast.stmt], in_loop: bool) -> None:
        for s in block:
            if isinstance(s, (ast.Break, ast.Continue)) and not in_loop:
                errs.append(f"V1: {m.name}: {type(s).__name__.lower()} at line {s.lineno} leaves the span")
            loop = isinstance(s, (ast.For, ast.AsyncFor, ast.While))
            for f in ("body", "orelse", "finalbody"):
                if isinstance(getattr(s, f, None), list) and not isinstance(s, SCOPE_NODES):
                    scan(getattr(s, f), in_loop or (loop and f == "body"))
            for sub in [h.body for h in getattr(s, "handlers", [])] + [c.body for c in getattr(s, "cases", [])]:
                scan(sub, in_loop)

    scan(m.members, False)
    return errs


# --------------------------------------------------------------------------------------------------
# cells and liveness (A.3)
# --------------------------------------------------------------------------------------------------


def meet(a: set | None, b: set | None) -> set | None:
    if a is None:
        return b
    return a if b is None else a & b


class Cells:
    """Liveness of F's locals across cells: each span, plus the residual function."""

    def __init__(self, fn: Any, key: Any, spans: list[tuple[str, list[ast.stmt]]]) -> None:
        self.fn, self.key = fn, key
        self.labels = [s[0] for s in spans] + ["<rest>"]
        n = len(spans)
        self.owner: dict[int, int] = {}
        self.nodes: dict[int, list[ast.AST]] = defaultdict(list)
        span_of = {id(s): i for i, (_, members) in enumerate(spans) for s in members}
        for s in fn.body:
            self._visit(s, n, span_of)
        a = fn.args
        self.params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg] if x}
        self.def_a: dict[int, frozenset] = {}
        self.live: list[set] = []
        for i, (_, members) in enumerate(spans):
            self.live.append(self._flow(i, members, set()))
        self.live.append(self._flow(n, fn.body, set(self.params)))
        self.bound: list[dict[str, list[int]]] = []
        self.refs: list[set[str]] = []
        self.first_read: list[dict[str, int]] = []
        self.first_import: list[dict[str, int]] = []
        for i in range(n + 1):
            b: dict[str, list[int]] = defaultdict(list)
            r: set[str] = set()
            for x in self.nodes[i]:
                self._collect(x, b, r)
            fr: dict[str, int] = {}
            fi: dict[str, int] = {}
            for x in self.nodes[i]:
                if isinstance(x, ast.Name) and is_f(x, key) and isinstance(x.ctx, ast.Load):
                    fr[x.id] = min(fr.get(x.id, x.lineno), x.lineno)
                elif isinstance(x, (ast.Import, ast.ImportFrom)):
                    for al, res in zip(x.names, x._rs):  # type: ignore[attr-defined]
                        if res is key:
                            fi[_bound(al)] = min(fi.get(_bound(al), x.lineno), x.lineno)
            self.first_read.append(fr)
            self.first_import.append(fi)
            if i == n:
                for p in self.params:
                    b[p].append(0)
                    r.add(p)
            self.bound.append(b)
            self.refs.append(r)
        # A name whose every binding is the same import is idempotent: a moved span that imports it
        # itself sees the same module, however many cells re-import it (the `sgl` / `_re` shape).
        imp: dict[str, set[Info]] = defaultdict(set)
        other: set[str] = set(self.params)
        for i in range(n + 1):
            for x in self.nodes[i]:
                if isinstance(x, (ast.Import, ast.ImportFrom)):
                    for al, res in zip(x.names, x._rs):  # type: ignore[attr-defined]
                        if res is key:
                            info = Info("import", al.name, al.name, al.asname, 0) if isinstance(x, ast.Import) else Info("from", x.module or "", al.name, al.asname, x.level)
                            imp[_bound(al)].add(info)
                elif isinstance(x, ast.Name) and is_f(x, key) and isinstance(x.ctx, ast.Store):
                    other.add(x.id)
                elif (isinstance(x, ast.ExceptHandler) and x.name or isinstance(x, SCOPE_NODES) and getattr(x, "name", None)) and is_f(x, key):
                    other.add(x.name)
        self.idem = {k for k, v in imp.items() if len(v) == 1 and k not in other}

    def _visit(self, s: ast.stmt, cur: int, span_of: dict[int, int]) -> None:
        c = span_of.get(id(s), cur)
        self.owner[id(s)] = c
        acc = self.nodes[c]
        if isinstance(s, SCOPE_NODES):
            acc.extend(ast.walk(s))
            return
        acc.append(s)
        for _, v in ast.iter_fields(s):
            for it in v if isinstance(v, list) else [v]:
                if isinstance(it, ast.stmt):
                    self._visit(it, c, span_of)
                elif isinstance(it, ast.ExceptHandler):
                    acc.append(it)
                    acc.extend(ast.walk(it.type) if it.type else [])
                    for b in it.body:
                        self._visit(b, c, span_of)
                elif isinstance(it, ast.match_case):
                    acc.extend(ast.walk(it.pattern))
                    acc.extend(ast.walk(it.guard) if it.guard else [])
                    for b in it.body:
                        self._visit(b, c, span_of)
                elif isinstance(it, ast.AST):
                    acc.extend(ast.walk(it))

    def _collect(self, x: ast.AST, b: dict, r: set) -> None:
        k = self.key
        if isinstance(x, ast.Name) and is_f(x, k):
            r.add(x.id)
            if isinstance(x.ctx, ast.Store):
                b[x.id].append(x.lineno)
        elif isinstance(x, ast.ExceptHandler) and x.name and is_f(x, k):
            r.add(x.name)
            b[x.name].append(x.lineno)
        elif isinstance(x, (ast.Import, ast.ImportFrom)):
            for al, res in zip(x.names, x._rs):  # type: ignore[attr-defined]
                if res is k:
                    r.add(_bound(al))
                    b[_bound(al)].append(x.lineno)
        elif isinstance(x, SCOPE_NODES) and getattr(x, "name", None) and is_f(x, k):
            r.add(x.name)
            b[x.name].append(x.lineno)

    # -- definite assignment ------------------------------------------------------------------
    def _reads(self, *nodes: ast.AST | None) -> set[str]:
        return {
            n.id
            for root in nodes
            if root is not None
            for n in ast.walk(root)
            if isinstance(n, ast.Name) and is_f(n, self.key) and isinstance(n.ctx, (ast.Load, ast.Del))
        }

    def _stores(self, *nodes: ast.AST | None) -> set[str]:
        return {
            n.id
            for root in nodes
            if root is not None
            for n in ast.walk(root)
            if isinstance(n, ast.Name) and is_f(n, self.key) and isinstance(n.ctx, ast.Store)
        }

    def _flow(self, cell: int, start: list[ast.stmt], a0: set) -> set:
        skip = {sid for sid, o in self.owner.items() if o != cell}

        def block(stmts: list[ast.stmt], A: set) -> tuple[set, set | None]:
            live: set = set()
            for s in stmts:
                if id(s) in skip:
                    continue
                l, A2 = stmt(s, A)
                live |= l
                if A2 is None:
                    return live, None
                A = A2
            return live, A

        def stmt(s: ast.stmt, A: set) -> tuple[set, set | None]:
            rd, st = self._reads, self._stores
            if isinstance(s, ast.If):
                l1, a1 = block(s.body, A | st(s.test))
                l2, a2 = block(s.orelse, A | st(s.test))
                return (rd(s.test) - A) | l1 | l2, meet(a1, a2)
            if isinstance(s, (ast.For, ast.AsyncFor)):
                inner = A | st(s.target)
                lb, _ = block(s.body, inner)
                lo, _ = block(s.orelse, A)
                return (rd(s.iter, s.target) - A) | lb | lo, A
            if isinstance(s, ast.While):
                lb, _ = block(s.body, A)
                lo, _ = block(s.orelse, A)
                return (rd(s.test) - A) | lb | lo, A
            if isinstance(s, (ast.Try, ast.TryStar)):
                live, ab = block(s.body, A)
                outs: list[set | None] = []
                for h in s.handlers:
                    ah = A | ({h.name} if h.name and is_f(h, self.key) else set())
                    lh, oh = block(h.body, ah)
                    live |= lh | (rd(h.type) - A)
                    outs.append(oh)
                lo, ao = block(s.orelse, ab) if ab is not None else (set(), None)
                live |= lo
                merged = ao
                for o in outs:
                    merged = meet(merged, o)
                lf, af = block(s.finalbody, A)
                live |= lf
                if merged is None or (s.finalbody and af is None):
                    return live, None
                return live, merged | (af or set())
            if isinstance(s, (ast.With, ast.AsyncWith)):
                items = [i.context_expr for i in s.items]
                tg = [i.optional_vars for i in s.items]
                lb, ab = block(s.body, A | st(*tg))
                return (rd(*items, *tg) - A) | lb, ab
            if isinstance(s, (ast.Return, ast.Raise)):
                return rd(s) - A, None
            if isinstance(s, (ast.Break, ast.Continue)):
                return set(), None
            if isinstance(s, SCOPE_NODES):
                self.def_a[id(s)] = frozenset(A)
                bound = {s.name} if is_f(s, self.key) else set()
                return rd(s) - A, A | bound
            binds = st(s)
            reads = rd(s)
            if isinstance(s, ast.AugAssign):
                reads |= rd(s.target) | st(s.target)
            if isinstance(s, ast.AnnAssign) and s.value is None:
                binds = set()
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                binds |= {_bound(a) for a, r in zip(s.names, s._rs) if r is self.key}  # type: ignore[attr-defined]
            return reads - A, A | binds

        return block(start, a0)[0]

    def crossing(self) -> dict[str, list[str]]:
        """Name -> reasons. A name is live-in at one cell and bound in another, or a closure local referenced across cells."""
        out: dict[str, list[str]] = defaultdict(list)
        n = len(self.labels)
        for j in range(n):
            for name in self.live[j]:
                for i in range(n):
                    if i != j and name in self.bound[i] and not (name in self.idem and self.first_import[j].get(name, 10**9) < self.first_read[j].get(name, 0)):
                        out[name].append(f"bound in {self.labels[i]}, live-in at {self.labels[j]}")
        for c in range(n):
            for d in (x for x in self.nodes[c] if isinstance(x, DEFS) and is_f(x, self.key)):
                free = {m.id for st in d.body for m in ast.walk(st) if isinstance(m, ast.Name) and is_f(m, self.key)}
                for name in free:
                    others = [o for o in range(n) if o != c and name in self.refs[o]]
                    if not others:
                        continue
                    later = any(ln > d.lineno for o in others for ln in self.bound[o].get(name, []))
                    if name in self.def_a.get(id(d), ()) and (not later or name in self.idem):
                        continue
                    out[name].append(f"captured by {d.name} in {self.labels[c]}, referenced in {self.labels[others[0]]}")
        return {k: v for k, v in out.items() if k not in ("rcfg", "state")}


# --------------------------------------------------------------------------------------------------
# V5 / V6 / V2 as AST transforms (used by verify; apply is checked against them)
# --------------------------------------------------------------------------------------------------


def parse_expr(text: str, ctx: ast.expr_context | None = None) -> ast.expr:
    e = ast.parse(text, mode="eval").body
    if ctx is not None and isinstance(e, (ast.Name, ast.Attribute)):
        e.ctx = ctx
    return e


def hidden_match(node: ast.Call, key: Any, hidden: dict) -> tuple[str, str] | None:
    f = node.func
    if not (
        isinstance(f, ast.Attribute)
        and f.attr == "get"
        and isinstance(f.value, ast.Name)
        and f.value.id == "kwargs"
        and is_f(f.value, key)
        and not node.keywords
        and 1 <= len(node.args) <= 2
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value in hidden
    ):
        return None
    k = node.args[0].value
    group, default = hidden[k]
    if len(node.args) == 1:
        return (group, k) if default is None else None
    same = ast.dump(node.args[1]) == ast.dump(parse_expr(repr(default)))
    return (group, k) if same else None


class Rewriter(ast.NodeTransformer):
    def __init__(self, key: Any, rewrite: dict, hidden: dict) -> None:
        self.key, self.rewrite, self.hidden = key, rewrite, hidden

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if is_f(node, self.key) and node.id in self.rewrite:
            return ast.copy_location(parse_expr(self.rewrite[node.id], node.ctx), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        hit = hidden_match(node, self.key, self.hidden)
        if hit:
            return ast.copy_location(parse_expr(f"rcfg.{hit[0]}.{hit[1]}"), node)
        self.generic_visit(node)
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST | None:
        node.names = [n for n, r in zip(node.names, node._rs) if not (r is self.key and n in self.rewrite)]  # type: ignore[attr-defined]
        return node if node.names else None

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.target, ast.Name):
            node.simple = 0
        return node


class Methodize(ast.NodeTransformer):
    """state -> self, rcfg -> self.rcfg (V6)."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "state":
            return ast.copy_location(ast.Name("self", node.ctx), node)
        if node.id == "rcfg":
            return ast.copy_location(ast.Attribute(ast.Name("self", ast.Load()), "rcfg", node.ctx), node)
        return node


class UpTransform(ast.NodeTransformer):
    """V2: module globals become ``up.<name>`` and relative imports gain a level."""

    def __init__(self, names: set[str], exempt: set[str]) -> None:
        self.names, self.exempt = names, exempt

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if getattr(node, "_r", NEW) is None and node.id in self.names and node.id not in self.exempt:
            return ast.copy_location(ast.Attribute(ast.Name("up", ast.Load()), node.id, node.ctx), node)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        if node.level:
            node.level += 1
        return node


def binding_nodes(root: ast.AST) -> Iterator[tuple[str, ast.AST]]:
    """Every ``(name, node)`` under ``root`` that binds a name: stores, parameters, imports, handlers, defs, matches."""
    for n in ast.walk(root):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            yield n.id, n
        elif isinstance(n, ast.arg):
            yield n.arg, n
        elif isinstance(n, ast.alias):
            yield _bound(n), n
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            for x in n.names:
                yield x, n
        elif isinstance(n, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and n.name:
            yield n.name, n
        elif isinstance(n, ast.MatchMapping) and n.rest:
            yield n.rest, n
        elif isinstance(n, SCOPE_NODES) and getattr(n, "name", None) and n is not root:
            yield n.name, n


def contains_await(stmts: list[ast.stmt]) -> bool:
    return any(isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith)) for n in walk_own(stmts))


def fn_names(fn: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
        elif isinstance(n, ast.alias):
            out.add(_bound(n))
        elif isinstance(n, SCOPE_NODES + (ast.ExceptHandler,)) and getattr(n, "name", None):
            out.add(n.name)
    return out


def call_stmt(m: Move, from_module: str) -> ast.stmt:
    recv = "" if m.module == from_module else f"_{m.module}."
    return ast.parse(f"{'await ' if m.is_async else ''}{recv}{m.name}(rcfg, state)").body[0]


def func_template(name: str, is_async: bool) -> Any:
    sig = 'rcfg: RagRunConfig, state: "PipelineState"'
    return ast.parse(f"{'async ' if is_async else ''}def {name}({sig}) -> None:\n    pass").body[0]


class Built:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.fn: Any = None
        self.methods: dict[str, Any] = {}  # method name -> def
        self.defs: dict[tuple[str, str], Any] = {}  # (module, name) -> def
        self.moves: list[Move] = []
        self.entry: Any = None
        self.classes: dict[str, ast.ClassDef] = {}
        self.imports: list[str] = []  # module names to import into the pipeline file


def spans_for_cells(m: dict, src: Src) -> list[tuple[str, list[ast.stmt]]]:
    spans = []
    for a, b in m["cells"]:
        _, members = locate(src.fn, a, b)
        spans.append((f"{a}-{b}", members))
    return spans


def build(m: dict, src: Src, lines: list[str], notes: list[str] | None = None) -> Built:
    """Apply the manifest to the base *as an AST*. This is the verify side's expectation."""
    out = Built()
    fn, key = src.fn, src.key
    rewrite, hidden, publish, hoist = m["rewrite"], m["hidden"], list(m["publish"]), m["hoist"]
    try:
        moves = plan_moves(src, m["moves"], lines)
    except SplitError as e:
        out.errors.append(f"V1: {e}")
        return out
    out.moves = moves

    # V8: a name the manifest introduces must not already be used in the function.
    used = fn_names(fn)
    intro = {t.split(".")[0] for t in rewrite.values()} | ({"rcfg"} if hidden else set()) | ({"state"} if publish else set())
    intro |= {"rcfg"} if m["prologue"] else set()
    intro |= {"state", "_run_state"} if m["state_init"] else set()
    intro |= ({"up"} if moves or hoist else set()) | ({"_run_state"} if hoist else set()) | {f"_{x.module}" for x in moves}
    init = [x for x in fn.body[:3] if isinstance(x, ast.Assign) and ast.unparse(x).startswith(f"state = _run_state.{STATE_CLASS}(")]
    if init:
        # An earlier PR (D1's state_init) already introduced both names, so reads of them are fine. Only
        # that one statement may bind them; any other binding would capture the rewritten ``state.<x>``.
        intro -= {"state", "_run_state"}
        mine = {id(x) for st in init for x in ast.walk(st)}
        out.errors += [f"V8: line {getattr(x, 'lineno', '?')}: {nm} is bound outside D1's state statement" for nm, x in binding_nodes(fn) if nm in ("state", "_run_state") and id(x) not in mine]
    out.errors += [f"V8: {n} is already a name in {fn.name}" for n in sorted(intro & used)]

    # mapped names that cannot become attributes
    for n in ast.walk(fn):
        bad = None
        if isinstance(n, ast.ExceptHandler) and n.name in rewrite and is_f(n, key):
            bad = "except ... as"
        elif isinstance(n, ast.NamedExpr) and n.target.id in rewrite and is_f(n.target, key):
            bad = "a walrus"
        elif isinstance(n, (ast.Import, ast.ImportFrom)) and any(r is key and _bound(a) in rewrite for a, r in zip(n.names, n._rs)):  # type: ignore[attr-defined]
            bad = "an import"
        elif isinstance(n, ast.Global) and set(n.names) & set(rewrite):
            bad = "a global statement"
        elif isinstance(n, SCOPE_NODES) and getattr(n, "name", None) in rewrite and is_f(n, key) and n.name not in hoist and n.name not in m["classes"]:
            bad = "a def or class"
        if bad:
            out.errors.append(f"V5: line {n.lineno}: a mapped name is bound by {bad}")
    for n in publish:
        if not any(isinstance(s, DEFS) and s.name == n for b in own_blocks(fn) for s in b):
            out.errors.append(f"V5: publish: no closure {n} in {fn.name}")

    # classes: each is one ClassDef of the function, mapped to ``_run_state.<new>``
    class_defs: dict[str, tuple[list, Any]] = {}
    for old, new in m["classes"].items():
        defs = [(b, s) for b in own_blocks(fn) for s in b if isinstance(s, ast.ClassDef) and s.name == old]
        if len(defs) != 1:
            out.errors.append(f"V6: expected one class {old}, found {len(defs)}")
        elif rewrite.get(old) != f"_run_state.{new}":
            out.errors.append(f"V6: class {old} must be in rewrite as _run_state.{new}")
        elif any(is_f(x, key) for x in ast.walk(defs[0][1]) if isinstance(x, ast.Name) and x is not defs[0][1]):
            out.errors.append(f"V6: class {old} reads a function local")
        else:
            class_defs[old] = defs[0]

    # hoist checks on the base
    closures: dict[str, Any] = {}
    for name, method in hoist.items():
        defs = [(b, s) for b in own_blocks(fn) for s in b if isinstance(s, DEFS) and s.name == name]
        if len(defs) != 1:
            out.errors.append(f"V6: expected one closure {name}, found {len(defs)}")
            continue
        block, d = defs[0]
        closures[name] = (block, d, method)
        if name not in rewrite:
            out.errors.append(f"V6: hoisted closure {name} is not in rewrite")
        free = free_locals(d, key)
        out.errors += [f"V6: {name} reads function local {x}, which is not in rewrite" for x in sorted(free - set(rewrite) - {"rcfg", "state"})]  # rcfg and state become self
        out.errors += [f"V6: {name} binds {nm} itself (line {getattr(x, 'lineno', '?')}), which would capture the rewritten name" for nm, x in binding_nodes(d) if nm in ("state", "rcfg", "self")]
        a = d.args
        head_nodes = d.decorator_list + a.defaults + [x for x in a.kw_defaults if x] + [x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs if x.annotation] + ([d.returns] if d.returns else [])
        if any(is_f(x, key) for r in head_nodes for x in ast.walk(r) if isinstance(x, ast.Name)):
            out.errors.append(f"V6: a default or decorator of {name} reads a function local")
        if any(not (mv.b < stmt_start(d) or mv.a > d.end_lineno) for mv in moves):
            out.errors.append(f"V6: {name} overlaps a move span")
        if any(isinstance(x, ast.Nonlocal) and any(n2 not in rewrite for n2 in x.names) for x in ast.walk(d)):
            out.errors.append(f"V6: {name} has a nonlocal outside rewrite")
    for mv in moves:
        out.errors += check_span(mv, notes)
    if out.errors:
        return out

    for s in fn.body:
        Rewriter(key, rewrite, hidden).visit(s)
    names = src.names
    up = UpTransform(names, src.exempt)
    for old, (block, d) in class_defs.items():
        block[:] = [x for x in block if x is not d]
        d.name = m["classes"][old]
        out.classes[d.name] = up.visit(d)  # a module global read by the class becomes up.<name>, as in a def
    # hoist: take the closures out
    for block, d, method in closures.values():
        block[:] = [s for s in block if s is not d]
        d = Methodize().visit(d)
        arg = ast.arg("self")
        (d.args.posonlyargs or d.args.args).insert(0, arg)
        d.name = method
        d = up.visit(d)
        out.methods[method] = d

    # publish
    for block in list(own_blocks(fn)):
        new: list[ast.stmt] = []
        for s in block:
            new.append(s)
            if isinstance(s, DEFS) and s.name in publish:
                p = ast.parse(f"state.{s.name} = {s.name}").body[0]
                p.lineno = p.end_lineno = s.end_lineno
                p.col_offset = p.end_col_offset = s.col_offset
                new.append(p)
                for mv in moves:
                    if any(x is s for x in mv.members):
                        mv.members.insert(next(i for i, x in enumerate(mv.members) if x is s) + 1, p)
        block[:] = new

    # cells and crossings (V4)
    cell_spans = [(mv.name, mv.members) for mv in sorted(moves, key=lambda x: x.a - x.b)]
    if cell_spans:
        for name, why in sorted(Cells(fn, key, cell_spans).crossing().items()):
            out.errors.append(f"V4: crossing local {name}: {'; '.join(why[:2])}")

    # moves, inside out
    for mv in sorted(moves, key=lambda x: x.b - x.a):
        mv.is_async = contains_await(mv.members)
        d = func_template(mv.name, mv.is_async)
        for i, s in enumerate(mv.members):
            mv.members[i] = up.visit(s)
        d.body = ([ast.Expr(ast.Constant(mv.doc))] if mv.doc else []) + mv.members
        out.defs[(mv.module, mv.name)] = d
        call = call_stmt(mv, "")
        i = next(i for i, x in enumerate(mv.block) if x is mv.members[0])
        mv.block[i : i + len(mv.members)] = [call]
        if mv.parent and mv.parent.block is mv.block:
            j = next(j for j, x in enumerate(mv.parent.members) if x is mv.members[0])
            mv.parent.members[j : j + len(mv.members)] = [call]
    # leaf calls inside a wrapper of the same module are bare calls
    for mv in moves:
        if mv.parent and mv.parent.module == mv.module:
            for n in ast.walk(out.defs[(mv.parent.module, mv.parent.name)]):
                if (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == mv.name
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == f"_{mv.module}"
                ):
                    n.func = ast.Name(mv.name, ast.Load())

    if m["prologue"]:
        fn.body.insert(docstring_len(fn), ast.parse(m["prologue"]).body[0])
    if m["state_init"]:
        fn.body.insert(docstring_len(fn) + 1, ast.parse(m["state_init"]).body[0])
    out.fn = fn
    if m["entry_split"]:
        body = fn.body
        i0 = 1 if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str) else 0
        s1 = body[i0] if len(body) > i0 else None
        if not (isinstance(s1, ast.Assign) and len(s1.targets) == 1 and isinstance(s1.targets[0], ast.Name) and s1.targets[0].id == "rcfg"):
            out.errors.append("V8: entry_split needs `rcfg = <expr>` as the first statement after the docstring")
            return out
        aw = isinstance(fn, ast.AsyncFunctionDef)
        new = ast.parse(f"{'async ' if aw else ''}def run_rag_pipeline(rcfg: RagRunConfig):\n    pass").body[0]
        new.body = body[:i0] + body[i0 + 1 :]
        new.returns = fn.returns
        shim = ast.Return(value=ast.Call(ast.Name("run_rag_pipeline", ast.Load()), [s1.value], []))
        if aw:
            shim.value = ast.Await(shim.value)
        fn.body = body[:i0] + [shim]
        out.entry = new
    return out


def docstring_len(fn: Any) -> int:
    b = fn.body
    return 1 if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) and isinstance(b[0].value.value, str) else 0


def free_locals(d: Any, key: Any) -> set[str]:
    """F's locals that a closure reads or writes (its free variables resolved into F)."""
    return {n.id for s in d.body for n in ast.walk(s) if isinstance(n, ast.Name) and is_f(n, key)} | {
        x for s in d.body for n in ast.walk(s) if isinstance(n, ast.Nonlocal) for x, r in zip(n.names, n._rs) if r is key  # type: ignore[attr-defined]
    }


def import_stmt(m: dict, name: str) -> ast.stmt:
    return ast.parse(f"from .{m['pkg']} import {name} as _{name}").body[0]


def expected_module(m: dict, src: Src, b: Built) -> ast.Module:
    body = list(src.tree.body)
    i = next(i for i, s in enumerate(body) if s is src.fn)
    if b.entry is not None:
        body.insert(i + 1, b.entry)
    mods = list(dict.fromkeys([x.module for x in b.moves] + (["run_state"] if m["hoist"] else [])))
    have = {ast.dump(s) for s in body}
    last = max((j for j, s in enumerate(body) if isinstance(s, (ast.Import, ast.ImportFrom))), default=-1)
    wanted = [ast.parse(x).body[0] for x in m["imports"]] + [import_stmt(m, x) for x in mods]
    body[last + 1 : last + 1] = [s for s in wanted if ast.dump(s) not in have]
    return ast.Module(body=body, type_ignores=[])


# --------------------------------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------------------------------


def diff_lines(label: str, want: ast.AST, got: ast.AST, limit: int = 6) -> list[str]:
    a = ast.unparse(want).splitlines()
    b = ast.unparse(got).splitlines()
    ds = [d for d in difflib.unified_diff(a, b, "base-derived", "head", lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")]
    out = [f"{label}: {d[:160]}" for d in ds[:limit]]
    if len(ds) > limit:
        out.append(f"{label}: ... {len(ds) - limit} more differing lines")
    return out or [f"{label}: differs"]


def def_in(tree: ast.Module, name: str) -> Any:
    return next((n for n in tree.body if isinstance(n, DEFS) and n.name == name), None)


def read_head(root: Path, rel: str | Path, head: str | None) -> str | None:
    """The head file: the working tree, or ``git show <head>:<rel>`` when ``--head`` names a ref."""
    if head is None:
        p = root / rel
        return p.read_text(encoding="utf-8") if p.exists() else None
    r = subprocess.run(  # nosec B603 B607  # fixed git argv, no shell
        ["git", "-C", str(root), "show", f"{head}:{Path(rel).as_posix()}"], capture_output=True, text=True
    )
    return r.stdout if r.returncode == 0 else None


def run_verify(root: Path, m: dict, base: str | None, base_dir: str | None, notes: list[str] | None = None, head_ref: str | None = None) -> list[str]:
    rel = m["file"]
    text = read_base(root, rel, base, base_dir)
    if text is None:
        return [f"{rel}: not found at the base"]
    src = Src(text, m["func"])
    b = build(m, src, text.split("\n"), notes)
    if b.errors:
        return b.errors
    errs: list[str] = []
    head_text = read_head(root, rel, head_ref)
    if head_text is None:
        return [f"{rel}: not found at the head"]
    head = ast.parse(head_text)
    want = expected_module(m, src, b)
    if ast.dump(want) != ast.dump(head):  # V7
        if len(want.body) != len(head.body):
            wn = [getattr(s, "name", type(s).__name__) for s in want.body]
            hn = [getattr(s, "name", type(s).__name__) for s in head.body]
            errs.append(f"{rel}: top-level statements differ: {len(want.body)} expected, {len(head.body)} found")
            errs += [f"{rel}: {d[:160]}" for d in difflib.unified_diff(wn, hn, "expected", "found", lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")][:8]
        else:
            for w, h in zip(want.body, head.body):
                if ast.dump(w) != ast.dump(h):
                    errs += diff_lines(f"{rel}: {getattr(w, 'name', type(w).__name__)} (line {getattr(w, 'lineno', '?')})", w, h)
    pkg = pkg_rel(m)
    cache: dict[str, str | None] = {}

    def module_text(name: str) -> str | None:
        if name not in cache:
            cache[name] = read_head(root, pkg / f"{name}.py", head_ref)
        return cache[name]

    def module(name: str) -> ast.Module | None:
        t = module_text(name)
        return ast.parse(t) if t is not None else None

    for (mod, name), want_def in b.defs.items():
        tree = module(mod)
        got = def_in(tree, name) if tree else None
        if got is None:
            errs.append(f"{mod}.py: no function {name}")
            continue
        mv = next(x for x in b.moves if x.name == name)
        errs += [f"{mod}.py: {name}: noqa line {ln} is outside the span {mv.a}-{mv.b}" for ln in mv.noqa if not mv.a <= ln <= mv.b]
        wd = ast.dump(want_def)
        if wd != ast.dump(got):
            errs += diff_lines(f"{mod}.py: {name}", want_def, got)
    for new, want_cls in b.classes.items():
        tree = module("run_state")
        got_cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == new), None) if tree else None
        if got_cls is None:
            errs.append(f"run_state.py: no class {new}")
        elif ast.dump(want_cls) != ast.dump(got_cls):
            errs += diff_lines(f"run_state.py: class {new}", want_cls, got_cls)
    if b.methods:
        tree = module("run_state")
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == STATE_CLASS), None) if tree else None
        for method, want_def in b.methods.items():
            got = next((n for n in cls.body if isinstance(n, DEFS) and n.name == method), None) if cls else None
            if got is None:
                errs.append(f"run_state.py: no method {STATE_CLASS}.{method}")
            elif ast.dump(want_def) != ast.dump(got):
                errs += diff_lines(f"run_state.py: {method}", want_def, got)
    # V9 and the module-shape checks, for every touched pipeline module
    stdlib = {n: i for n, i in src.infos.items() if i.stdlib}
    mods = list(dict.fromkeys([x.module for x in b.moves] + (["run_state"] if m["hoist"] else [])))
    for name in mods:
        t = module_text(name)
        if t is not None:
            errs += check_module(t, name, m, stdlib, {x.module for x in b.moves}, name == "run_state")
    return errs


def _is_literal(node: ast.expr) -> bool:
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


def check_module(text: str, name: str, m: dict, stdlib: dict[str, Info], modules: set[str], is_state: bool) -> list[str]:
    """V7/V9 for a stage module: only declared shapes at module level, one binding per name."""
    errs: list[str] = []
    tree = ast.parse(text)
    psrc = Src(text)
    label = f"{name}.py"
    bound: dict[str, int] = defaultdict(int)

    def bind(n: str) -> None:
        bound[n] += 1

    for i, st in enumerate(tree.body):
        if i == 0 and isinstance(st, ast.Expr) and isinstance(getattr(st, "value", None), ast.Constant) and isinstance(st.value.value, str):
            continue
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            for al in st.names:
                bind(_bound(al))
        elif isinstance(st, ast.If) and isinstance(st.test, ast.Name) and st.test.id == "TYPE_CHECKING" and not st.orelse and all(isinstance(x, (ast.Import, ast.ImportFrom)) for x in st.body):
            for x in st.body:
                for al in x.names:  # type: ignore[attr-defined]
                    bind(_bound(al))
        elif isinstance(st, DEFS) or (is_state and isinstance(st, ast.ClassDef)):
            bind(st.name)
            if isinstance(st, ast.ClassDef):
                seen: dict[str, int] = defaultdict(int)
                for x in st.body:
                    if isinstance(x, DEFS):
                        seen[x.name] += 1
                errs += [f"{label}: {st.name}.{k} is defined {v} times" for k, v in seen.items() if v > 1]
        elif is_state and isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None and _is_literal(st.value):
            bind(st.target.id)  # run_state's own constants, such as MUTATED_PARAMS
        else:
            errs.append(f"{label}: line {st.lineno}: a module-level {type(st).__name__} is not allowed in a stage module")
    errs += [f"{label}: {k} is bound {v} times at module level" for k, v in bound.items() if v > 1]
    for n in stdlib:
        got = psrc.infos.get(n)
        if bound.get(n) and got != stdlib[n]:
            errs.append(f"V9: {label} binds {n} as {got.line() if got else 'a non-import'}; unified_pipeline binds it as {stdlib[n].line()}")
    for n in sorted({x.id for x in ast.walk(psrc.tree) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and getattr(x, "_r", NEW) is None}):
        if n in stdlib and psrc.infos.get(n) != stdlib[n] and not bound.get(n):
            errs.append(f"V9: {label} reads {n}, bound there as nothing; unified_pipeline binds it as {stdlib[n].line()}")
    parts = m["file"][:-3].split("/")
    want = {"up": Info("from", ".".join(parts[:-1]), parts[-1], "up", 0)}
    if not is_state:
        want["RagRunConfig"] = Info("from", "run_config", "RagRunConfig", None, 1)
    for c in modules - {name}:
        want[f"_{c}"] = Info("from", "", c, f"_{c}", 1)
    for n, info in want.items():
        got = psrc.infos.get(n)
        if got is not None and got != info:
            errs.append(f"{label}: {n} is bound as {got.line()}, expected {info.line()}")
    uses_up = any(isinstance(x, ast.Name) and x.id == "up" for x in ast.walk(psrc.tree))
    if uses_up and "up" not in psrc.infos:
        errs.append(f"{label}: reads up but does not import it")
    return errs


# --------------------------------------------------------------------------------------------------
# apply: text edits at AST positions
# --------------------------------------------------------------------------------------------------


class Edit(NamedTuple):
    row: int
    col: int
    erow: int
    ecol: int
    text: str
    conv: bool = False  # inside a hoisted closure: state -> self, rcfg -> self.rcfg


class Cut(NamedTuple):
    lo: int
    hi: int
    text: str | None
    indent: int = 0


def pos(n: ast.AST) -> tuple[int, int, int, int]:
    return n.lineno, n.col_offset, n.end_lineno, n.end_col_offset  # type: ignore[attr-defined]


def conv_text(t: str) -> str:
    return re.sub(r"^rcfg\b", "self.rcfg", re.sub(r"^state\b", "self", t))


def render(lines: list[bytes], cont: set[int], lo: int, hi: int, shift: int, edits: list[Edit], cuts: list[Cut], hoist: bool = False, own: Cut | None = None) -> str:
    inner = [c for c in cuts if c is not own and lo <= c.lo and c.hi <= hi]
    inner = [c for c in inner if not any(o is not c and o.lo <= c.lo and c.hi <= o.hi and (o.lo, o.hi) != (c.lo, c.hi) for o in inner)]
    by_lo = {c.lo: c for c in inner}
    buf = bytearray()
    starts: dict[int, tuple[int, int]] = {}
    row = lo
    while row <= hi:
        if row in by_lo:
            c = by_lo[row]
            if c.text is not None:
                buf += b" " * max(0, c.indent - shift) + c.text.encode() + b"\n"
            row = c.hi + 1
            continue
        ln = lines[row - 1]
        removed = 0
        if row not in cont and ln.strip():
            removed = min(shift, len(ln) - len(ln.lstrip(b" "))) if shift > 0 else shift
        starts[row] = (len(buf), removed)
        buf += ln[removed:] if removed > 0 else b" " * -removed + ln
        row += 1
    todo = []
    for e in edits:
        if e.row in starts and e.erow in starts:
            s, r = starts[e.row], starts[e.erow]
            text = conv_text(e.text) if (hoist and e.conv) else e.text
            text = re.sub(r"\0(\d+)\0", lambda mt: " " * max(0, int(mt.group(1)) - shift), text)
            todo.append((s[0] + e.col - s[1], r[0] + e.ecol - r[1], text.encode()))
    todo.sort(key=lambda t: (t[0], t[1]), reverse=True)
    prev = None
    for a, b, t in todo:
        if prev is not None and b > prev:
            raise SplitError("overlapping edits; the manifest asks for two changes to the same text")
        buf[a:b] = t
        prev = a
    return buf.decode("utf-8")


def row_end(lines: list[bytes], row: int) -> int:
    return len(lines[row - 1].rstrip(b"\r\n"))


def gather(src: Src, m: dict, lines: list[bytes]) -> tuple[list[Edit], list[Cut]]:
    fn, key = src.fn, src.key
    rewrite, hidden, publish = m["rewrite"], m["hidden"], set(m["publish"])
    edits: list[Edit] = []
    cuts: list[Cut] = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and is_f(n, key) and n.id in rewrite:
            edits.append(Edit(*pos(n), rewrite[n.id], True))
        elif isinstance(n, ast.Call) and (hit := hidden_match(n, key, hidden)):
            edits.append(Edit(*pos(n), f"rcfg.{hit[0]}.{hit[1]}", True))
        elif isinstance(n, ast.Nonlocal) and any(r is key and x in rewrite for x, r in zip(n.names, n._rs)):  # type: ignore[attr-defined]
            keep = [x for x, r in zip(n.names, n._rs) if not (r is key and x in rewrite)]  # type: ignore[attr-defined]
            if keep:
                edits.append(Edit(*pos(n), "nonlocal " + ", ".join(keep)))
            else:
                cuts.append(Cut(n.lineno, n.end_lineno, None))
        elif isinstance(n, DEFS) and n is not fn and n.name in publish and any(n in blk for blk in own_blocks(fn)):
            e = row_end(lines, n.end_lineno)
            edits.append(Edit(n.end_lineno, e, n.end_lineno, e, f"\n\0{n.col_offset}\0state.{n.name} = {n.name}"))
    return edits, cuts


def level_edits(lines: list[bytes], nodes: list[ast.ImportFrom]) -> list[Edit]:
    out = []
    for n in nodes:
        mt = re.compile(rb"from\s+(\.+)").match(lines[n.lineno - 1], n.col_offset)
        if mt:
            out.append(Edit(n.lineno, mt.end(1), n.lineno, mt.end(1), "."))
    return out


def region_edits(src: Src, lines: list[bytes], lo: int, hi: int, root_nodes: list[ast.AST], hoist: bool) -> list[Edit]:
    out: list[Edit] = []
    imports = []
    for r in root_nodes:
        for n in ast.walk(r):
            if getattr(n, "lineno", None) is None or not (lo <= n.lineno <= hi):
                continue
            if isinstance(n, ast.Name):
                if getattr(n, "_r", NEW) is None and n.id in src.names and n.id not in src.exempt:
                    out.append(Edit(*pos(n), f"up.{n.id}"))
                elif hoist and n.id in ("state", "rcfg"):
                    out.append(Edit(*pos(n), "self" if n.id == "state" else "self.rcfg"))
            elif isinstance(n, ast.ImportFrom) and n.level:
                imports.append(n)
    return out + level_edits(lines, imports)


def needed_imports(src: Src, m: dict, nodes: list[ast.AST], cross: set[str], need_up: bool, stage: bool) -> dict[str, str]:
    """name -> import line for a pipeline module."""
    out: dict[str, str] = {}
    for r in nodes:
        for n in ast.walk(r):
            if isinstance(n, ast.Name) and getattr(n, "_r", NEW) is None and n.id in src.exempt:
                info = src.infos[n.id]
                out[n.id] = info.line(0 if info.stdlib else 1)
    if stage:
        out.setdefault("TYPE_CHECKING", "from typing import TYPE_CHECKING")
        out["RagRunConfig"] = "from .run_config import RagRunConfig"
    if need_up:
        parts = m["file"][:-3].split("/")
        out["up"] = f"from {'.'.join(parts[:-1])} import {parts[-1]} as up"
    for c in sorted(cross):
        out[f"_{c}"] = f"from . import {c} as _{c}"
    return out


def last_import_row(tree: ast.Module) -> int:
    rows = [s.end_lineno for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))]
    if rows:
        return max(rows)
    first = tree.body[0] if tree.body else None
    return first.end_lineno if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) else 0


def order_imports(wanted: dict[str, str]) -> list[str]:
    std = sorted(v for k, v in wanted.items() if not v.startswith("from .") and k not in ("up", "TYPE_CHECKING"))
    typ = [v for k, v in wanted.items() if k == "TYPE_CHECKING"]
    absolute = [v for k, v in wanted.items() if k == "up"]
    rel = sorted(v for v in wanted.values() if v.startswith("from ."))
    return std + typ + absolute + rel


def add_imports(text: str, wanted: dict[str, str]) -> str:
    have = set(import_infos(ast.parse(text)))
    new = order_imports({k: v for k, v in wanted.items() if k not in have})
    if not new:
        return text
    rows = text.split("\n")
    at = last_import_row(ast.parse(text))
    rows[at:at] = new
    return "\n".join(rows)


def header_text(mod: str, wanted: dict[str, str], stage: bool) -> str:
    parts = [f'"""{mod} code moved from unified_pipeline.py (TASK-13950.24)."""', ""] + order_imports(wanted)
    if stage:
        parts += ["", "if TYPE_CHECKING:", "    from .run_state import PipelineState"]
    return "\n".join(parts) + "\n"


def run_apply(root: Path, m: dict) -> list[str]:
    rel = m["file"]
    path = root / rel
    text = path.read_text(encoding="utf-8")
    rows_text = text.split("\n")
    src = Src(text, m["func"])
    b = build(m, src, rows_text)
    if b.errors:
        return b.errors
    fresh = Src(text, m["func"])  # build() mutated the first tree; positions come from this one
    lines = text.encode("utf-8").split(b"\n")
    lines = [ln + b"\n" for ln in lines[:-1]] + lines[-1:]
    cont = continuation_rows(text)
    fn = fresh.fn
    edits, cuts = gather(fresh, m, lines)
    moves = plan_moves(fresh, m["moves"], rows_text)
    for mv in moves:
        mv.is_async = next(x.is_async for x in b.moves if x.name == mv.name)
    blocks = list(own_blocks(fn))

    hoist_cuts: dict[str, Cut] = {}
    for name in m["hoist"]:
        d = next(s for blk in blocks for s in blk if isinstance(s, DEFS) and s.name == name)
        tail = d.end_lineno + 1 if d.end_lineno < len(rows_text) and not rows_text[d.end_lineno].strip() else d.end_lineno
        hoist_cuts[name] = Cut(stmt_start(d), tail, None, d.col_offset)
    class_text: dict[str, str] = {}
    class_nodes: dict[str, Any] = {}
    for old, new in m["classes"].items():
        d = next(s for blk in blocks for s in blk if isinstance(s, ast.ClassDef) and s.name == old)
        tail = d.end_lineno + 1 if d.end_lineno < len(rows_text) and not rows_text[d.end_lineno].strip() else d.end_lineno
        hoist_cuts[f"class {old}"] = Cut(stmt_start(d), tail, None, d.col_offset)
        class_nodes[new] = d
    move_cuts: dict[str, Cut] = {}
    for mv in moves:
        recv = "" if mv.parent and mv.parent.module == mv.module else f"_{mv.module}."
        call = f"{'await ' if mv.is_async else ''}{recv}{mv.name}(rcfg, state)"
        move_cuts[mv.name] = Cut(mv.a, mv.b, call, mv.members[0].col_offset)
    cuts += list(hoist_cuts.values()) + list(move_cuts.values())

    pkg_dir = root / pkg_rel(m)
    outputs: dict[Path, list[str]] = defaultdict(list)
    wanted: dict[Path, dict[str, str]] = defaultdict(dict)
    for mv in moves:
        shift = mv.members[0].col_offset - 4
        live = lambda row, mv=mv: not any(c.a <= row <= c.b for c in mv.children)  # noqa: E731
        ed = [e for e in edits if mv.a <= e.row <= mv.b] + region_edits(fresh, lines, mv.a, mv.b, list(mv.members), False)
        for ln in mv.noqa:
            if not mv.a <= ln <= mv.b:
                return [f"{mv.name}: noqa line {ln} is outside the span {mv.a}-{mv.b}"]
            ed.append(Edit(ln, row_end(lines, ln), ln, row_end(lines, ln), NOQA))
        body = render(lines, cont, mv.a, mv.b, shift, ed, cuts, own=move_cuts[mv.name])
        if mv.doc is not None:
            if '"""' in mv.doc or "\\" in mv.doc:
                return [f"{mv.name}: doc must not contain triple quotes or backslashes"]
            body = f'    """{mv.doc}"""\n' + body
        head = ("async " if mv.is_async else "") + f'def {mv.name}(rcfg: RagRunConfig, state: "PipelineState") -> None:\n'
        target = pkg_dir / f"{mv.module}.py"
        outputs[target].append(head + body)
        mine = [n for n in (x for s in mv.members for x in ast.walk(s)) if live(getattr(n, "lineno", mv.a))]
        need_up = any(e.text.startswith("up.") and live(e.row) for e in ed)
        cross = {c.module for c in mv.children if c.module != mv.module}
        wanted[target].update(needed_imports(fresh, m, mine, cross, need_up, True))

    methods: list[str] = []
    for name, method in m["hoist"].items():
        d = next(s for blk in blocks for s in blk if isinstance(s, DEFS) and s.name == name)
        lo = stmt_start(d)
        ed = [e for e in edits if lo <= e.row <= d.end_lineno] + region_edits(fresh, lines, lo, d.end_lineno, [d], True)
        mt = re.match(rb"(\s*(?:async\s+)?def\s+)(\w+)(\s*\()", lines[d.lineno - 1])
        if not mt:
            return [f"{name}: cannot find the def line"]
        ed.append(Edit(d.lineno, mt.start(2), d.lineno, mt.end(2), method))
        a = d.args
        has = bool(a.posonlyargs or a.args or a.vararg or a.kwonlyargs or a.kwarg)
        rest = lines[d.lineno - 1][mt.end(3) :]
        sep = "" if not rest.strip() else " "
        ed.append(Edit(d.lineno, mt.end(3), d.lineno, mt.end(3), ("self," + sep) if has else "self"))
        methods.append(render(lines, cont, lo, d.end_lineno, d.col_offset - 4, ed, cuts, hoist=True, own=hoist_cuts[name]))
        wanted[pkg_dir / "run_state.py"].update(needed_imports(fresh, m, [d], set(), any(e.text.startswith("up.") for e in ed), False))

    for new, d in class_nodes.items():
        lo = stmt_start(d)
        ed = region_edits(fresh, lines, lo, d.end_lineno, [d], False)
        mt = re.match(rb"(\s*class\s+)(\w+)", lines[d.lineno - 1])
        if not mt:
            return [f"{new}: cannot find the class line"]
        ed.append(Edit(d.lineno, mt.start(2), d.lineno, mt.end(2), new))
        class_text[new] = render(lines, cont, lo, d.end_lineno, d.col_offset, ed, cuts, own=hoist_cuts[f"class {d.name}"]).rstrip("\n")
        wanted[pkg_dir / "run_state.py"].update(needed_imports(fresh, m, [d], set(), any(e.text.startswith("up.") for e in ed), False))

    # the pipeline function itself
    f_lo, f_hi = stmt_start(fn), fn.end_lineno
    new_f = render(lines, cont, f_lo, f_hi, 0, edits, cuts)
    if m["prologue"]:
        f = ast.parse(new_f).body[0]
        first = f.body[docstring_len(f)]
        frows = new_f.split("\n")
        frows[stmt_start(first) - 1 : stmt_start(first) - 1] = [" " * first.col_offset + m["prologue"], ""]
        new_f = "\n".join(frows)
    if m["state_init"]:
        f = ast.parse(new_f).body[0]
        first = f.body[docstring_len(f)]
        frows = new_f.split("\n")
        frows[first.end_lineno : first.end_lineno] = [" " * first.col_offset + m["state_init"]]
        new_f = "\n".join(frows)
    if m["entry_split"]:
        new_f = split_entry(new_f)
    rows = rows_text[: f_lo - 1] + new_f.rstrip("\n").split("\n") + rows_text[f_hi:]
    new_text = "\n".join(rows)
    mods = list(dict.fromkeys([x.module for x in moves] + (["run_state"] if m["hoist"] else [])))
    tree = ast.parse(new_text)
    have = {ast.dump(s) for s in tree.body}
    imps = [x for x in m["imports"] if ast.dump(ast.parse(x).body[0]) not in have]
    imps += [f"from .{m['pkg']} import {x} as _{x}" for x in mods if ast.dump(import_stmt(m, x)) not in have]
    if imps:
        rows = new_text.split("\n")
        at = last_import_row(tree)
        rows[at:at] = imps
        new_text = "\n".join(rows)

    state_path = pkg_dir / "run_state.py"
    state_text = ""
    if methods:
        cur = state_path.read_text(encoding="utf-8") if state_path.exists() else header_text("run_state", wanted[state_path], False) + f"\n\nclass {STATE_CLASS}:\n    pass\n"
        cur = add_imports(cur, wanted[state_path]) if state_path.exists() else cur
        cls = next((n for n in ast.parse(cur).body if isinstance(n, ast.ClassDef) and n.name == STATE_CLASS), None)
        if cls is None:
            return [f"run_state.py has no {STATE_CLASS} class"]
        crow = cur.split("\n")
        if len(cls.body) == 1 and isinstance(cls.body[0], ast.Pass):
            end = cls.body[0].lineno - 1
            crow[end : cls.body[0].end_lineno] = []
        else:
            end = cls.end_lineno
        crow[end:end] = [x for t in methods for x in [""] + t.rstrip("\n").split("\n")]
        for t in class_text.values():
            crow[cls.lineno - 1 : cls.lineno - 1] = t.split("\n") + ["", ""]
        state_text = "\n".join(crow)
        if not state_text.endswith("\n"):
            state_text += "\n"
    path.write_text(new_text, encoding="utf-8")
    for p, funcs in outputs.items():
        cur = add_imports(p.read_text(encoding="utf-8"), wanted[p]) if p.exists() else header_text(p.stem, wanted[p], True)
        cur = cur.rstrip("\n") + "\n" + "".join("\n\n" + f.rstrip("\n") + "\n" for f in funcs)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(cur, encoding="utf-8")
    if methods:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(state_text, encoding="utf-8")
    return []


def split_entry(f_text: str) -> str:
    tree = ast.parse(f_text)
    f = tree.body[0]
    i0 = 1 if isinstance(f.body[0], ast.Expr) and isinstance(getattr(f.body[0], "value", None), ast.Constant) else 0
    s1 = f.body[i0]
    aw = isinstance(f, ast.AsyncFunctionDef)
    rows = f_text.split("\n")
    value = ast.get_source_segment(f_text, s1.value)
    ret = f" -> {ast.get_source_segment(f_text, f.returns)}" if f.returns else ""
    shim = rows[: stmt_start(s1) - 1] + [f"    return {'await ' if aw else ''}run_rag_pipeline({value})"]
    new = ["", "", f"{'async ' if aw else ''}def run_rag_pipeline(rcfg: RagRunConfig){ret}:"]
    if i0:
        new += rows[f.body[0].lineno - 1 : f.body[0].end_lineno]
    new += rows[s1.end_lineno : f.end_lineno]
    return "\n".join(shim + new) + "\n"


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_shared(root: Path, m: dict | None, hoist: list[str], base: str | None, base_dir: str | None) -> list[str]:
    m = m or {**DEFAULTS}
    text = read_base(root, m["file"], base, base_dir)
    if text is None:
        raise SplitError(f"{m['file']}: not found")
    src = Src(text, m["func"])
    out: list[str] = []
    for name in hoist:
        d = next((s for blk in own_blocks(src.fn) for s in blk if isinstance(s, DEFS) and s.name == name), None)
        if d is None:
            raise SplitError(f"no closure {name}")
        out.append(f"{name}: " + ", ".join(sorted(free_locals(d, src.key))))
    spans = spans_for_cells(m, src) if m["cells"] else []
    if m["moves"]:
        moves = plan_moves(src, m["moves"], text.split("\n"))
        spans += [(mv.name, mv.members) for mv in sorted(moves, key=lambda x: x.a - x.b)]
    if spans:
        for name, why in sorted(Cells(src.fn, src.key, spans).crossing().items()):
            out.append(f"{name}: {'; '.join(why)}")
    return out


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < (3, 12):
        print("rag_split needs Python 3.12 or newer", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(prog="rag_split")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("shared", "apply", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=str(repo_root()))
        if name != "shared":
            p.add_argument("manifest")
        else:
            p.add_argument("--manifest")
            p.add_argument("--hoist", default="")
        if name != "apply":
            g = p.add_mutually_exclusive_group(required=name == "verify")
            g.add_argument("--base")
            g.add_argument("--base-dir")
        if name == "verify":
            p.add_argument("--head", help="read the head files from this git ref instead of the working tree")
    a = ap.parse_args(argv)
    root = Path(a.root)
    try:
        if a.cmd == "apply":
            errs = run_apply(root, load_manifest(Path(a.manifest)))
        elif a.cmd == "verify":
            notes: list[str] = []
            errs = run_verify(root, load_manifest(Path(a.manifest)), a.base, a.base_dir, notes, a.head)
            for n in notes:
                print(n)
            if not errs:
                print("0 differences")
        else:
            m = load_manifest(Path(a.manifest)) if a.manifest else None
            for line in run_shared(root, m, [x for x in a.hoist.split(",") if x], a.base, a.base_dir):
                print(line)
            return 0
    except SplitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for e in errs:
        print(e)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
