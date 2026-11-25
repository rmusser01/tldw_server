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
    publish      [closure name]                              ``state.<n> = <n>`` right after each def, in the def's own block
    hoist        {closure name: method name}                 closure -> ``PipelineState`` method
    classes      {local class: module class}                 a class defined in the function becomes a ``run_state`` class
                                                             (its name must also be in ``rewrite``, as ``_run_state.<new>``)
    moves        [{"name", "module", "span": [a, b], "doc", "noqa": [line]}]
    cells        [[a, b]]                                    extra cells for ``shared``
    prologue     "rcfg = ..."                                one statement inserted first in the function, after its docstring
    state_init   "state = ..."                               one statement inserted right after the first statement after the docstring
    imports      ["from .x import Y"]                        module-level imports added after the file's last import
    entry_split  true
    entry_doc    str: the docstring of ``run_rag_pipeline`` (unset: it has none, and its body starts at the first statement)
    shim_doc     str: replaces the docstring of the shim (unset: the shim keeps the original one)
    file, func, pkg    the pipeline file, function and package directory (defaults below; tests override)

Parametrised for other code (TASK-13950.25; each default is the RAG value, so the RAG manifests verify unchanged):

    ctx          [[name, annotation]]    leading parameters of every moved function; the call passes the names
                                         (default ``[["rcfg","RagRunConfig"],["state","PipelineState"]]``)
    ctx_import   ["from .x import Y"]    ctx annotation imports, all under ``if TYPE_CHECKING`` and quoted in the
                                         signature (unset: RAG's ``RagRunConfig`` runtime import plus the state class)
    facade       alias for the source module in moved code (default ``up``; chat: ``cs``)
    facade_stdlib [name]                 stdlib-bound names still read through the facade (chat: ``time``)
    task         text in the header, ``noqa`` and docstrings (default ``TASK-13950.24``)
    state_class, state_module            the hoist target class and its module (``PipelineState``, ``run_state``)

Chat part 2 (TASK-13950.25; each needs ``pkg: ""``: the destinations are existing siblings of the source file):

    lift         {nested def: module}    a def directly in the function (or, depth 2, in a def that is directly in it) becomes
                                         ``<module>.<def>(cst, ...)``; its site becomes
                                         ``<def> = functools.partial(_<module>.<def>, cst)``, or goes (and every read of it is
                                         rewritten to ``<ref>(cst, ...)``) when the code that held it is itself in that module.
                                         A depth-2 site sits in its parent: the lifted parent's body, or the function while the
                                         parent is still nested (a later manifest lifts it). The partial is bound once, so an
                                         attribute set on it (``_out_transform.flush = _flush``) is set on the object that is read.
    moves[].args [name]                  read-only locals of the function passed after the ctx (``except ... as e``)
    entry        {name, module, doc}     the body after the ``cst = <expr>`` prologue moves whole to ``<module>.<name>(cst)``
    drop         [{module, def|import|all|replace}]   delete a def, an import binding or ``__all__`` entries (``all_add`` adds),
                                         or replace a call-site expression; a dropped name must have no reference left (V15)
    late_imports [{module, lines}]       a top import of a destination moves into its late-import block
    prune_dead_sites true                 a lifted name that no code in the function reads once the manifest is applied gets no
                                         ``functools.partial`` at its def site (the def just goes). Off by default: PR E's committed
                                         output keeps its dead partials, and verify must still accept it

Chat part 3 (TASK-13950.25):

    rewrite_in   {nested def: {name: expr}}   a rewrite, like ``rewrite``, of the names that are locals of that nested def (its own
                                         scope, nested defs included), with their ``nonlocal`` entries deleted. The expressions
                                         must be ``cst.x`` names or attribute chains, distinct from every other rewrite
                                         expression (V16), and the names locals of the def: not parameters, not bound by
                                         ``except as``, a walrus, an import, a def or ``global`` (V5). Works in either layout.
    free_imports {name: import line}     (``pkg: ""``) a lifted body reads ``name`` from a function-local import of the function or
                                         of its lifted parent. The line must be that import, alone (V16); it is added at the top
                                         of each destination whose lifted def reads the name. A lifted parent's own copy of the
                                         import goes when nothing in the parent still reads it (it would be unused, ruff F401).

A destination that exists gets its new defs above a trailing *late-import block* (marker comment, then the ``_<module>`` aliases
the new code calls, then ``from <pkg> import <source> as cs``), so no module reads ``cs`` or an alias at import time (R5).
``verify`` compares the whole module AST with the base module plus those additions.

``apply`` edits the working tree as text edits at AST positions. ``verify`` rebuilds the expected result
from the base *as an AST*, independently of those text edits, and compares it with the head.
Every difference prints one line; any difference exits 1.

Published closures: a name in ``publish`` may also be in ``rewrite``. Its ``def`` keeps its name and is followed
by ``state.<n> = <n>`` (inside the same ``if``/``try`` block, at the same indentation); every other read of the name,
in any cell including the def's own, becomes ``state.<n>``. This is allowed only while the ``def`` is the name's single
binding anywhere in the function (no assignment, import, ``for``/``with``/``except`` target, walrus, parameter or
nested-scope binding of the same name); otherwise V5 refuses it. ``verify`` rebuilds the publish statement and rejects
a missing, extra or misplaced one.

Limits:
* V8 only flags names the manifest itself introduces (``rcfg``, ``state``, ``up``, ``_run_state``,
  ``_<module>``); later PRs legitimately use ``rcfg`` and ``state`` already.
* V4 is the crossing definition of A.3: a name live-in at one cell and bound in another, or a closure
  local referenced across cells. A name read before assignment inside one span and bound nowhere
  else is not flagged (it fails the same way before and after the move).
* A leaf in the same module as its wrapper is called bare (``leaf(rcfg, state)``); in another module it is
  ``_module.leaf(rcfg, state)`` with ``from . import module as _module``.
* ``entry_split`` needs ``rcfg = <expr>`` as the first statement after the docstring. The shim becomes
  ``return [await] run_rag_pipeline(<expr>)``; the old body, minus its docstring, moves into
  ``run_rag_pipeline(rcfg)``. The docstring stays on the shim, or is replaced by ``shim_doc``; ``run_rag_pipeline``
  gets ``entry_doc`` as its docstring or none. ``verify`` rebuilds both docstrings from the manifest and compares
  them exactly, so a missing, extra, non-string or differing docstring is a difference.
* Scopes are matched to symtable by (name, line) in visit order; two lambdas on one line can swap.
"""

from __future__ import annotations

import argparse
import ast
import copy
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
DEFAULT_CTX = [["rcfg", "RagRunConfig"], ["state", "PipelineState"]]
PART2_KEYS = {"relocate"}
MOVE_KEYS = {"name", "module", "span", "doc", "noqa", "allow_locals", "args"}
LATE_MARK = "# late imports: moved code reads these at call time (TASK-13950.25 R5)"
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
    "entry_doc": None,
    "shim_doc": None,
    "prologue": "",
    "state_init": "",
    "imports": [],
    "ctx": DEFAULT_CTX,
    "ctx_import": None,
    "facade": "up",
    "facade_stdlib": [],
    "task": "TASK-13950.24",
    "state_class": "PipelineState",
    "state_module": "run_state",
    "lift": {},
    "entry": None,
    "drop": [],
    "late_imports": [],
    "rewrite_in": {},
    "free_imports": {},
    "prune_dead_sites": False,
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


def _is_pkg(i: Info, pkg: str = "pipeline") -> bool:
    return i.kind == "from" and i.level > 0 and (i.module == pkg or i.module.startswith(pkg + "."))


class Src:
    def __init__(self, text: str, func: str | None = None, pkg: str = "pipeline", facade_stdlib: tuple[str, ...] = ()) -> None:
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
        self.exempt = {n for n, i in self.infos.items() if (i.stdlib and n not in facade_stdlib) or _is_pkg(i, pkg)}


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
            raw = ast.literal_eval(n.value)
            later = sorted(set(raw) & PART2_KEYS)
            if later:
                raise SplitError(f"{path}: manifest keys {later} are not supported yet (TASK-13950.25 part 2/3)")
            m = {**DEFAULTS, **raw}
            unknown = set(m) - set(DEFAULTS)
            if unknown:
                raise SplitError(f"{path}: unknown manifest keys {sorted(unknown)}")
            for key in ("prologue", "state_init"):
                if m[key] and len(ast.parse(m[key]).body) != 1:
                    raise SplitError(f"{path}: {key} must be exactly one statement")
            for spec in m["moves"]:
                args = spec.get("args", [])
                if not (isinstance(args, list) and all(isinstance(a, str) and a.isidentifier() for a in args)) or len(set(args)) != len(args):
                    raise SplitError(f"{path}: move {spec.get('name')}: args must be a list of distinct names")
                extra = sorted(set(spec) - MOVE_KEYS)
                if extra:
                    hint = "not supported yet (TASK-13950.25 part 2/3)" if set(extra) <= PART2_KEYS else "unknown"
                    raise SplitError(f"{path}: move {spec.get('name')}: keys {extra} are {hint}")
            if m["pkg"] == "":
                bad = [k for k in ("publish", "hoist", "classes", "entry_split", "prologue", "state_init", "hidden") if m[k]]
                if bad:
                    raise SplitError(f"{path}: {bad} are not supported when pkg is '' (destinations are siblings of the source)")
            if m["entry"] is not None:
                e = m["entry"]
                if m["pkg"] != "":
                    raise SplitError(f"{path}: entry needs pkg '' (use entry_split for the RAG layout)")
                if not isinstance(e, dict) or set(e) - {"name", "module", "doc"} or not all(isinstance(e.get(k), str) and e.get(k) for k in ("name", "module")) or not isinstance(e.get("doc", ""), str):
                    raise SplitError(f"{path}: entry must be {{'name': str, 'module': str, 'doc': str (optional)}}")
            if m["late_imports"]:
                li = m["late_imports"]
                if m["pkg"] != "":
                    raise SplitError(f"{path}: late_imports needs pkg ''")
                if not (isinstance(li, list) and all(isinstance(x, dict) and set(x) == {"module", "lines"} and isinstance(x["module"], str) and isinstance(x["lines"], list) and len(x["lines"]) == 2 and all(isinstance(n, int) for n in x["lines"]) and 0 < x["lines"][0] <= x["lines"][1] for x in li)):
                    raise SplitError(f"{path}: late_imports is a list of {{'module': name, 'lines': [first, last]}}")
            if m["drop"]:
                if m["pkg"] != "":
                    raise SplitError(f"{path}: drop needs pkg ''")
                check_drop_items(path, m["drop"])
            if m["lift"]:
                if m["pkg"] != "":
                    raise SplitError(f"{path}: lift needs pkg '' (the destination modules are siblings of the source)")
                if not isinstance(m["lift"], dict) or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in m["lift"].items()):
                    raise SplitError(f"{path}: lift must map a def name to a module name")
            ri = m["rewrite_in"]
            if not (isinstance(ri, dict) and all(isinstance(k, str) and isinstance(v, dict) and v and all(isinstance(a, str) and a.isidentifier() and _is_target(e) for a, e in v.items()) for k, v in ri.items())):
                raise SplitError(f"{path}: rewrite_in must map a def name to {{local name: 'cst.x' (a name or an attribute chain)}}")
            if m["prune_dead_sites"] and (m["pkg"] != "" or not isinstance(m["prune_dead_sites"], bool)):
                raise SplitError(f"{path}: prune_dead_sites is a bool and needs pkg ''")
            fi = m["free_imports"]
            if fi:
                if m["pkg"] != "":
                    raise SplitError(f"{path}: free_imports needs pkg ''")
                if not m["lift"]:
                    raise SplitError(f"{path}: free_imports needs a lift that reads the name")
                if not (isinstance(fi, dict) and all(isinstance(k, str) and isinstance(v, str) and _is_import_of(v, k) for k, v in fi.items())):
                    raise SplitError(f"{path}: free_imports must map a name to one import line that binds exactly that name")
            if m["publish"] and m["facade"] == "cs":
                raise SplitError(f"{path}: publish is refused when facade is 'cs' (R6)")
            if not m["ctx"] or any(not (isinstance(c, list) and len(c) == 2 and all(isinstance(x, str) for x in c)) for c in m["ctx"]):
                raise SplitError(f"{path}: ctx must be a non-empty list of [name, annotation]")
            for key in ("entry_doc", "shim_doc"):
                if m[key] is not None and not isinstance(m[key], str):
                    raise SplitError(f"{path}: {key} must be a str")
                if m[key] is not None and not m["entry_split"]:
                    raise SplitError(f"{path}: {key} needs entry_split")
            return m
    raise SplitError(f"{path}: no MANIFEST literal")


def _is_target(text: Any) -> bool:
    try:
        return isinstance(text, str) and isinstance(ast.parse(text, mode="eval").body, (ast.Name, ast.Attribute))
    except SyntaxError:
        return False


def _is_import_of(line: str, name: str) -> bool:
    try:
        body = ast.parse(line).body
    except SyntaxError:
        return False
    return len(body) == 1 and isinstance(body[0], (ast.Import, ast.ImportFrom)) and [_bound(a) for a in body[0].names] == [name]


def is_sibling(m: dict) -> bool:
    """``pkg == ""``: the destination modules sit next to the source file and already exist (chat)."""
    return m["pkg"] == ""


def pkg_dotted(m: dict) -> str:
    return ".".join(Path(m["file"]).parent.parts)


def src_stem(m: dict) -> str:
    return Path(m["file"]).stem


def pkg_rel(m: dict) -> Path:
    return Path(m["file"]).parent / m["pkg"]


def mk_src(text: str, m: dict) -> Src:
    return Src(text, m["func"], m["pkg"], tuple(m["facade_stdlib"]))


def ctx_names(m: dict) -> list[str]:
    return [n for n, _ in m["ctx"]]


def call_args(m: dict) -> str:
    return ", ".join(ctx_names(m))


def src_name(m: dict) -> str:
    return Path(m["file"]).name


def noqa_text(m: dict) -> str:
    return f"  # noqa: BLE001 - moved from {src_name(m)} ({m['task']})"


def ctx_imports(m: dict) -> tuple[dict[str, str], list[str]]:
    """``(runtime imports by name, TYPE_CHECKING import lines)`` for the ctx annotations."""
    if m["ctx_import"] is None:
        return {"RagRunConfig": "from .run_config import RagRunConfig"}, [f"from .{m['state_module']} import {m['state_class']}"]
    return {}, list(m["ctx_import"])


def ctx_sig(m: dict) -> str:
    """The leading parameters; an annotation imported only under ``TYPE_CHECKING`` is quoted."""
    _, tc = ctx_imports(m)
    typed = {_bound(a) for line in tc for a in ast.parse(line).body[0].names}  # type: ignore[attr-defined]
    return ", ".join(f'{n}: "{a}"' if a in typed else f"{n}: {a}" for n, a in m["ctx"])


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
        self.args: list[str] = list(spec.get("args", []))
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


def f_bound(root: ast.AST, key: Any) -> set[str]:
    """Names the function's own scope binds anywhere in ``root`` (stores, handlers, imports, defs, parameters)."""
    out: set[str] = set()
    for n in ast.walk(root):
        if isinstance(n, ast.Name) and is_f(n, key) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name and is_f(n, key):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update(_bound(a) for a, r in zip(n.names, n._rs) if r is key)  # type: ignore[attr-defined]
        elif isinstance(n, SCOPE_NODES) and getattr(n, "name", None) and is_f(n, key):
            out.add(n.name)
    return out


def loop_above(fn: Any, target: ast.stmt) -> ast.stmt | None:
    """The innermost ``for``/``while`` statement that encloses ``target`` inside ``fn``, if any."""

    def walk(block: list[ast.stmt], loop: ast.stmt | None) -> tuple[bool, ast.stmt | None]:
        for st in block:
            if st is target:
                return True, loop
            for sub in child_blocks(st):
                found, lp = walk(sub, st if isinstance(st, (ast.For, ast.AsyncFor, ast.While)) else loop)
                if found:
                    return True, lp
        return False, None

    return walk(fn.body, None)[1]


def check_args(mv: Move, fn: Any, key: Any, ctx: list[str], lifted: set[str]) -> list[str]:
    """V3: a name a span takes as a parameter is a local of the function that the span reads and never rebinds."""
    errs: list[str] = []
    reads = {n.id for s in mv.members for n in ast.walk(s) if isinstance(n, ast.Name) and is_f(n, key) and isinstance(n.ctx, ast.Load)}
    inside = f_bound(ast.Module(body=mv.members, type_ignores=[]), key)
    for a in mv.args:
        if a in ctx or a in lifted:
            errs.append(f"V3: {mv.name}: args {a} is a ctx or lifted name")
        elif a in inside:
            errs.append(f"V3: {mv.name}: args {a} is rebound inside the span; a parameter must be read-only there")
        elif a not in reads:
            errs.append(f"V3: {mv.name}: args {a} is not read in the span")
    return errs


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

    def __init__(self, fn: Any, key: Any, spans: list[tuple[str, list[ast.stmt]]], skip: tuple[str, ...] = ("rcfg", "state"), args: dict[str, list[str]] | None = None, strict: bool = False) -> None:
        self.skip = skip
        self.strict = strict
        self.end = [max(s.end_lineno for s in members) for _, members in spans]
        blocks = list(own_blocks(fn))
        self.span_block = [next(b for b in blocks if any(x is members[0] for x in b)) for _, members in spans]
        self.span_last = [members[-1] for _, members in spans]
        self.span_start = [min(stmt_start(s) for s in members) for _, members in spans]
        self.args = args or {}  # span label -> names it takes as parameters, so they do not cross
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
        self.loads: list[dict[str, list[int]]] = []
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
            ld: dict[str, list[int]] = defaultdict(list)
            for x in self.nodes[i]:
                if isinstance(x, ast.Name) and is_f(x, key) and isinstance(x.ctx, ast.Load):
                    ld[x.id].append(x.lineno)
                elif isinstance(x, ast.AugAssign) and isinstance(x.target, ast.Name) and is_f(x.target, key):
                    ld[x.target.id].append(x.target.lineno)  # x += 1 reads x
            self.loads.append(ld)
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

    def _binds(self, x: ast.AST) -> list[str]:
        if isinstance(x, (ast.Import, ast.ImportFrom)):
            return [_bound(a) for a, r in zip(x.names, x._rs) if r is self.key]  # type: ignore[attr-defined]
        return [x.name] if getattr(x, "name", None) and is_f(x, self.key) else []  # type: ignore[attr-defined]

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
                if name in self.args.get(self.labels[j], ()):
                    continue
                for i in range(n):
                    if i != j and name in self.bound[i] and not (name in self.idem and self.first_import[j].get(name, 10**9) < self.first_read[j].get(name, 0)):
                        out[name].append(f"bound in {self.labels[i]}, live-in at {self.labels[j]}")
        if self.strict:
            # A name bound in a span and read (or augmented) anywhere after it crosses, unless the first such reference
            # follows a plain, unconditional rebinding in the span's own block. Liveness misses the rest (a flag set in the
            # span and tested after it); when unsure the name crosses, which only costs a ctx slot.
            for i in range(n - 1):
                blk = self.span_block[i]
                after = blk[next(k for k, x in enumerate(blk) if x is self.span_last[i]) + 1 :]
                plain_bound = {
                    x.id for x in self.nodes[i] if isinstance(x, ast.Name) and is_f(x, self.key) and isinstance(x.ctx, ast.Store)
                } | {nm for x in self.nodes[i] if isinstance(x, (ast.Import, ast.ImportFrom, *SCOPE_NODES)) for nm in self._binds(x)}
                for name in self.bound[i]:
                    if name in self.idem or name not in plain_bound:  # an `except ... as e` target is scoped to its handler
                        continue
                    for j in range(n):
                        # another span that does not enclose this one binds before it reads, or liveness flagged it above
                        if j < n - 1 and not (self.span_start[j] <= self.span_start[i] and self.end[i] <= self.end[j]):
                            continue
                        late = [ln for ln in self.loads[j].get(name, []) if ln > self.end[i]]
                        if j == i or not late or name in self.args.get(self.labels[j], ()):
                            continue
                        first = min(late)
                        plain = any(
                            self.owner.get(id(x)) == j
                            and isinstance(x, (ast.Assign, ast.AnnAssign))
                            and (x.value is not None)
                            and any(isinstance(t, ast.Name) and t.id == name and is_f(t, self.key) for t in (x.targets if isinstance(x, ast.Assign) else [x.target]))
                            and x.end_lineno < first
                            for x in after
                        )
                        if not plain:
                            out[name].append(f"bound in {self.labels[i]}, read after it in {self.labels[j]}")
        for c in range(n):
            for d in (x for x in self.nodes[c] if isinstance(x, DEFS) and is_f(x, self.key)):
                free = {m.id for st in d.body for m in ast.walk(st) if isinstance(m, ast.Name) and is_f(m, self.key)}
                for name in free:
                    if name in self.args.get(self.labels[c], ()):
                        continue
                    others = [o for o in range(n) if o != c and name in self.refs[o]]
                    if not others:
                        continue
                    later = any(ln > d.lineno for o in others for ln in self.bound[o].get(name, []))
                    if name in self.def_a.get(id(d), ()) and (not later or name in self.idem):
                        continue
                    out[name].append(f"captured by {d.name} in {self.labels[c]}, referenced in {self.labels[others[0]]}")
        return {k: v for k, v in out.items() if k not in self.skip}


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


def plan_rewrite_in(m: dict, src: Src) -> tuple[list[tuple[Any, Any, dict]], list[str]]:
    """``[(def, its scope, {name: expr})]`` for ``rewrite_in``, and the V5/V16 refusals.

    The names are locals of a def nested in the function; the replacement expressions must differ from every other
    replacement, or two variables would share one ``cst`` slot.
    """
    fn = src.fn
    out: list[tuple[Any, Any, dict]] = []
    errs: list[str] = []
    seen = list(m["rewrite"].values())
    for dname, rw in m["rewrite_in"].items():
        hits = [n for n in ast.walk(fn) if isinstance(n, DEFS) and n is not fn and n.name == dname]
        if len(hits) != 1:
            errs.append(f"V16: rewrite_in: expected one def {dname} in {fn.name}, found {len(hits)}")
            continue
        d = hits[0]
        key = src.scopes.tab_of[id(d)]
        a = d.args
        params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg] if x}
        bound = f_bound(d, key)
        for name, expr in rw.items():
            if any(expr == o or expr.startswith(o + ".") or o.startswith(expr + ".") for o in seen):
                errs.append(f"V16: rewrite_in {dname}.{name} -> {expr} collides with another rewritten name (the same slot or a prefix of it); two variables would share one slot")
            seen.append(expr)
            if name in params:
                errs.append(f"V16: rewrite_in {dname}.{name} is a parameter of {dname}; its value never reaches {expr}")
            elif name not in bound:
                errs.append(f"V16: rewrite_in {dname}.{name} is not a local of {dname}")
        for n in ast.walk(d):
            bad = None
            if isinstance(n, ast.ExceptHandler) and n.name in rw and is_f(n, key):
                bad = "except ... as"
            elif isinstance(n, ast.NamedExpr) and n.target.id in rw and is_f(n.target, key):
                bad = "a walrus"
            elif isinstance(n, (ast.Import, ast.ImportFrom)) and any(r is key and _bound(x) in rw for x, r in zip(n.names, n._rs)):  # type: ignore[attr-defined]
                bad = "an import"
            elif isinstance(n, ast.Global) and set(n.names) & set(rw):
                bad = "a global statement"
            elif isinstance(n, SCOPE_NODES) and n is not d and getattr(n, "name", None) in rw and is_f(n, key):
                bad = "a def or class"
            if bad:
                errs.append(f"V5: line {n.lineno}: a name mapped by rewrite_in {dname} is bound by {bad}")
        out.append((d, key, rw))
    return out, errs


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

    def __init__(self, names: set[str], exempt: set[str], facade: str = "up", bump: bool = True) -> None:
        self.names, self.exempt, self.facade, self.bump = names, exempt, facade, bump

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if getattr(node, "_r", NEW) is None and node.id in self.names and node.id not in self.exempt:
            return ast.copy_location(ast.Attribute(ast.Name(self.facade, ast.Load()), node.id, node.ctx), node)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        if node.level and self.bump:
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


def call_stmt(m: Move, from_module: str, args: str) -> ast.stmt:
    recv = "" if m.module == from_module else f"_{m.module}."
    return ast.parse(f"{'await ' if m.is_async else ''}{recv}{m.name}({args})").body[0]


def func_template(name: str, is_async: bool, sig: str) -> Any:
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
        self.order: list[tuple[str, str]] = []  # sibling mode: (module, name) of every appended def, in file order
        self.dests: set[str] = set()  # sibling mode: every destination module the manifest names
        self.src_mods: list[str] = []  # sibling mode: destination aliases the source still reads
        self.src_functools = False
        self.lifted: set[tuple[str, str]] = set()
        self.free_imps: dict[str, set[str]] = {}  # module -> free_imports names its lifted defs read


def spans_for_cells(m: dict, src: Src) -> list[tuple[str, list[ast.stmt]]]:
    spans = []
    for a, b in m["cells"]:
        _, members = locate(src.fn, a, b)
        spans.append((f"{a}-{b}", members))
    return spans


# --------------------------------------------------------------------------------------------------
# lifting nested defs to module functions (R10)
# --------------------------------------------------------------------------------------------------


class Lift:
    """A def nested in the function (or in a def nested directly in it), becoming ``<module>.<name>(cst, ...)``."""

    def __init__(self, name: str, module: str) -> None:
        self.name, self.module = name, module
        self.node: Any = None
        self.block: list[ast.stmt] = []
        self.parent_node: Any = None  # depth 2: the def that holds this one; None at depth 1
        self.key: Any = None  # the scope that binds the name: the function (depth 1) or the parent def (depth 2)
        self.free: set[str] = set()  # free_imports names this body reads
        self.needed = True  # False: nothing in the code that holds the def site reads the name once the manifest is applied, so no partial is bound
        self.dead: list[ast.stmt] = []  # local imports in this body that its lifted children replaced and nothing reads any more
        # where the def site is: the move whose span holds it, the lifted parent (depth 2), or None (the rest of the function)
        self.defregion: Move | Lift | None = None


class Regions:
    """Where a line of the function ends up: a lift body, a move body, or the rest (the entry body, or the source)."""

    def __init__(self, moves: list[Move], lifts: list[Lift], rest_module: str) -> None:
        self.moves, self.lifts, self.rest_module = moves, lifts, rest_module

    def of(self, line: int) -> Any:
        held = [lf for lf in self.lifts if stmt_start(lf.node) <= line <= lf.node.end_lineno]
        if held:
            return min(held, key=lambda lf: lf.node.end_lineno - stmt_start(lf.node))
        inner = [x for x in self.moves if x.a <= line <= x.b]
        return min(inner, key=lambda x: x.b - x.a) if inner else None

    def module(self, region: Any) -> str:
        return self.rest_module if region is None else region.module

    def ref(self, lf: Lift, line: int) -> str | None:
        """The text that replaces a read of the lifted name at ``line``; None when the def site's partial still binds it.

        A partial can bind the name only in code that lives outside the lifted function's module: inside it the
        assignment ``f = partial(f, cst)`` would make ``f`` an unbound local, so there the reads are rewritten instead.
        """
        r = self.of(line)
        here = self.module(r)
        if r is lf.defregion and lf.module != self.module(r):
            return None
        return lf.name if lf.module == here else f"_{lf.module}.{lf.name}"


def lift_defs(fn: Any, name: str) -> list[tuple[list[ast.stmt], Any, Any]]:
    """Every def called ``name`` directly in ``fn`` or directly in a def that is directly in ``fn``: ``(block, def, parent def)``."""
    top = [(blk, s, None) for blk in own_blocks(fn) for s in blk if isinstance(s, DEFS) and s.name == name]
    parents = [s for blk in own_blocks(fn) for s in blk if isinstance(s, DEFS)]
    return top + [(blk, s, p) for p in parents for blk in own_blocks(p) for s in blk if isinstance(s, DEFS) and s.name == name]


def local_import_bindings(root: Any, key: Any, name: str) -> tuple[list[Info], int]:
    """How the scope ``key`` (the function ``root``) binds ``name``: by which imports, and by how many other bindings."""
    infos: list[Info] = []
    other = 0
    a = root.args
    other += sum(1 for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg] if x and x.arg == name)
    for n in ast.walk(root):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for al, r in zip(n.names, n._rs):  # type: ignore[attr-defined]
                if r is key and _bound(al) == name:
                    infos.append(Info("import", al.name, al.name, al.asname, 0) if isinstance(n, ast.Import) else Info("from", n.module or "", al.name, al.asname, n.level))
        else:
            stored = isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id == name
            named = isinstance(n, (ast.ExceptHandler, *SCOPE_NODES)) and n is not root and getattr(n, "name", None) == name
            other += (stored or named) and is_f(n, key)
    return infos, other


def mark_needed(fn: Any, lifts: list[Lift], regions: Regions, prune: bool) -> None:
    """Set ``needed``: some read of the name stays bare (the def site's partial serves it). Run before any read is rewritten."""
    reads = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
    for lf in lifts:
        lf.needed = not prune or any(n.id == lf.name and is_f(n, lf.key) and regions.ref(lf, n.lineno) is None for n in reads)


def plan_lifts(m: dict, src: Src, moves: list[Move]) -> tuple[list[Lift], list[str]]:
    fn, key = src.fn, src.key
    errs: list[str] = []
    lifts: list[Lift] = []
    for name, module in m["lift"].items():
        lf = Lift(name, module)
        hits = lift_defs(fn, name)
        if len(hits) != 1:
            errs.append(f"V11: expected one def {name} directly in {fn.name} (or in a def directly in it), found {len(hits)}")
            continue
        lf.block, lf.node, lf.parent_node = hits[0]
        lf.key = key if lf.parent_node is None else src.scopes.tab_of[id(lf.parent_node)]
        if sum(1 for nm, _ in binding_nodes(fn) if nm == name) != 1:
            errs.append(f"V11: {name} has a second binding in {fn.name}; a lifted name must be bound by its def alone")
        if any(mv.name == name for mv in moves):
            errs.append(f"V10: {name} is both a lift and a move")
        lifts.append(lf)
    for lf in lifts:  # the def site's region: the lifted parent, else the smallest move that holds the def (or its parent)
        host = next((x for x in lifts if lf.parent_node is not None and x.node is lf.parent_node), None)
        site = lf.parent_node if lf.parent_node is not None else lf.node
        lf.defregion = host or next(iter(sorted((x for x in moves if x.a <= stmt_start(site) and site.end_lineno <= x.b), key=lambda x: x.b - x.a)), None)
    allowed = set(ctx_names(m))
    fi = m["free_imports"]
    for lf in lifts:
        outer = [key] + ([lf.key] if lf.parent_node is not None else [])
        scope_node = {id(key): fn, id(lf.key): lf.parent_node or fn}
        for n in ast.walk(lf.node):
            r = getattr(n, "_r", NEW)
            if isinstance(n, ast.Name) and any(r is k for k in outer):
                if (r is key and n.id in allowed) or any(o.name == n.id and o.key is r for o in lifts):
                    continue
                if n.id in fi and isinstance(n.ctx, ast.Load):
                    infos, other = local_import_bindings(scope_node[id(r)], r, n.id)
                    want = import_infos(ast.parse(fi[n.id])).get(n.id)
                    if infos and not other and all(i == want for i in infos):
                        lf.free.add(n.id)
                        continue
                    errs.append(f"V16: free_imports {n.id}: {scope_node[id(r)].name} binds it as {', '.join(i.line() for i in infos) or 'something else'}, not by `{fi[n.id]}` alone (line {n.lineno})")
                    continue
                errs.append(f"V11: lifted {lf.name} reads {n.id} (line {n.lineno}), which is neither {', '.join(ctx_names(m))} nor a lifted name")
            elif isinstance(n, ast.Nonlocal) and any(rr is k for rr in n._rs for k in outer):  # type: ignore[attr-defined]
                errs.append(f"V11: lifted {lf.name} declares nonlocal at line {n.lineno}")
            elif isinstance(n, ast.Name) and n.id in ("__name__", "__file__") and isinstance(n.ctx, ast.Load):
                errs.append(f"V1: lifted {lf.name} reads {n.id} (line {n.lineno}); it reads a different module after the move")
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "globals" and getattr(n.func, "_r", NEW) is None:
                errs.append(f"V1: lifted {lf.name} calls globals() (line {n.lineno}); it reads a different module after the move")
    errs += [f"V16: free_imports {n} is read by no lifted def" for n in fi if lifts and not any(n in lf.free for lf in lifts)]
    skip = {id(x.node) for x in lifts}
    for par in lifts:  # a lifted parent's own `from x import Name` that only its lifted children read is replaced by the module-level import
        kids = [c for c in lifts if c.parent_node is par.node]
        for name in sorted({n for c in kids for n in c.free}):
            stack, readers, stmts = [par.node], 0, []
            while stack:
                n = stack.pop()
                if isinstance(n, (ast.Import, ast.ImportFrom)) and any(r is kids[0].key and _bound(a) == name for a, r in zip(n.names, n._rs)):  # type: ignore[attr-defined]
                    stmts.append(n)
                readers += isinstance(n, ast.Name) and n.id == name and is_f(n, kids[0].key) and isinstance(n.ctx, ast.Load)
                stack.extend(c for c in ast.iter_child_nodes(n) if id(c) not in skip or c is par.node)
            if not readers:
                par.dead += [x for x in stmts if len(x.names) == 1]
    return lifts, errs


class LiftRefs(ast.NodeTransformer):
    """Reads of a lifted name become ``ref(cst, ...)`` when called and ``functools.partial(ref, cst)`` otherwise."""

    def __init__(self, m: dict, lifts: list[Lift], regions: Regions) -> None:
        self.ctx = ctx_names(m)
        self.by_name, self.regions = {lf.name: lf for lf in lifts}, regions
        self.errors: list[str] = []

    def hit(self, n: ast.AST) -> Lift | None:
        lf = self.by_name.get(n.id) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) else None  # type: ignore[attr-defined]
        return lf if lf is not None and is_f(n, lf.key) else None

    def visit_Call(self, node: ast.Call) -> ast.AST:
        lf = self.hit(node.func)
        ref = self.regions.ref(lf, node.func.lineno) if lf else None
        if ref is None:
            self.generic_visit(node)
            return node
        node.args = [self.visit(a) for a in node.args]
        node.keywords = [self.visit(k) for k in node.keywords]
        node.func = ast.copy_location(parse_expr(ref), node.func)
        node.args = [ast.Name(c, ast.Load()) for c in self.ctx] + node.args
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        lf = self.hit(node)
        ref = self.regions.ref(lf, node.lineno) if lf else None
        if ref is None:
            return node
        return ast.copy_location(parse_expr(f"functools.partial({ref}, {', '.join(self.ctx)})"), node)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        ops = (ast.Is, ast.IsNot, ast.Eq, ast.NotEq, ast.In, ast.NotIn)
        for op, a, b in zip(node.ops, [node.left, *node.comparators], node.comparators):
            for x in (a, b):
                lf = self.hit(x)
                if lf and isinstance(op, ops):
                    self.errors.append(f"V11: line {node.lineno}: lifted {lf.name} is compared with {type(op).__name__}; a partial is not the closure it replaces")
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        lf = self.hit(node.value)
        if lf and self.regions.ref(lf, node.value.lineno) is not None:
            self.errors.append(f"V11: line {node.lineno}: {lf.name}.{node.attr}; an attribute of a lifted name is not rewritable (no partial binds it here)")
            return node
        self.generic_visit(node)
        return node


def lift_edits(fn: Any, lines: list[bytes], regions: Regions, lifts: list[Lift], ctx: str) -> list[Edit]:
    """The text counterpart of ``LiftRefs``."""
    by_name = {lf.name: lf for lf in lifts}
    calls = {id(c.func): c for c in ast.walk(fn) if isinstance(c, ast.Call)}
    out: list[Edit] = []
    for n in ast.walk(fn):
        if not (isinstance(n, ast.Name) and n.id in by_name and is_f(n, by_name[n.id].key) and isinstance(n.ctx, ast.Load)):
            continue
        ref = regions.ref(by_name[n.id], n.lineno)
        if ref is None:
            continue
        c = calls.get(id(n))
        if c is None:
            out.append(Edit(*pos(n), f"functools.partial({ref}, {ctx})"))
            continue
        out.append(Edit(*pos(n), ref))
        row, col = n.end_lineno, n.end_col_offset
        if lines[row - 1][col : col + 1] != b"(":
            raise SplitError(f"line {row}: expected '(' right after {n.id}")
        out.append(insert_first(lines, row, col + 1, ctx, [*c.args, *c.keywords][:1]))
    return out


def insert_first(lines: list[bytes], row: int, col: int, text: str, first: list[Any]) -> Edit:
    """Put ``text`` as the first item of a parameter or argument list that opens at ``(row, col)``."""
    rest = lines[row - 1][col:].strip()
    if not first:
        return Edit(row, col, row, col, text)
    if rest and not rest.startswith(b"#"):
        return Edit(row, col, row, col, text + ", ")
    return Edit(row, col, row, col, f"\n\0{first[0].col_offset}\0{text},")


def sig_edit(lines: list[bytes], d: Any, ctx_text: str) -> Edit:
    """Prepend the ctx parameters to a def's signature."""
    mt = re.compile(rb"(?:async\s+)?def\s+" + d.name.encode() + rb"\s*\(").match(lines[d.lineno - 1], d.col_offset)
    if not mt:
        raise SplitError(f"line {d.lineno}: cannot find the signature of {d.name}")
    a = d.args
    has = bool(a.posonlyargs or a.args or a.vararg or a.kwonlyargs or a.kwarg)
    rest = lines[d.lineno - 1][mt.end() :].strip()
    if has and not rest:  # the parameters start on the next line
        nxt = lines[d.lineno]
        return Edit(d.lineno, mt.end(), d.lineno, mt.end(), f"\n\0{len(nxt) - len(nxt.lstrip(b' '))}\0{ctx_text},")
    return Edit(d.lineno, mt.end(), d.lineno, mt.end(), ctx_text + (", " if has else ""))


def ctx_params(m: dict) -> list[ast.arg]:
    return ast.parse(f"def _({ctx_sig(m)}): pass").body[0].args.args  # type: ignore[attr-defined]


def replace_stmt(old: ast.stmt, new: list[ast.stmt], blocks: list[list[ast.stmt]], moves: list[Move]) -> None:
    for blk in blocks:
        for i, x in enumerate(blk):
            if x is old:
                blk[i : i + 1] = new
                break
    for mv in moves:
        for i, x in enumerate(mv.members):
            if x is old:
                mv.members[i : i + 1] = new
                break


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
    lifts, lift_errs = plan_lifts(m, src, moves) if m["lift"] else ([], [])
    out.errors += lift_errs
    entry = m["entry"]
    regions = Regions(moves, lifts, entry["module"] if entry else "")
    if entry:
        pro0 = prologue_stmt(fn, ctx_names(m)[0])
        if pro0 is None:
            out.errors.append(f"V8: entry needs `{ctx_names(m)[0]} = <expr>` as the first statement after the docstring")
        elif any(x is pro0 for mv in moves for x in mv.members):
            out.errors.append("V8: a move span holds the entry's prologue statement")
        if any(x["name"] == entry["name"] for x in m["moves"]) or entry["name"] in m["lift"]:
            out.errors.append(f"V10: entry {entry['name']} is also a move or a lift")

    # V8: a name the manifest introduces must not already be used in the function.
    used = fn_names(fn)
    intro = {t.split(".")[0] for t in rewrite.values()} | ({"rcfg"} if hidden else set()) | ({"state"} if publish else set())
    intro |= {t.split(".")[0] for rw in m["rewrite_in"].values() for t in rw.values()}
    sm, sc = m["state_module"], m["state_class"]
    intro |= {ctx_names(m)[0]} if m["prologue"] else set()
    intro |= {"state", f"_{sm}"} if m["state_init"] else set()
    intro |= ({m["facade"]} if moves or hoist or lifts or entry else set()) | ({f"_{sm}"} if hoist else set()) | {f"_{x.module}" for x in moves}
    intro |= ({"functools"} | {f"_{lf.module}" for lf in lifts}) if lifts else set()
    intro |= {f"_{entry['module']}"} if entry else set()
    pro = prologue_stmt(fn, ctx_names(m)[0])
    kept = pro is not None and not m["prologue"] and is_sibling(m)  # the ctx prologue is already in the source (an earlier PR)
    if (moves or lifts or entry) and m["ctx"] != DEFAULT_CTX:
        intro |= set(ctx_names(m)) - ({ctx_names(m)[0]} if kept else set())
    if kept:
        intro.discard(ctx_names(m)[0])  # a rewrite into the ctx reads the name the source already binds
    if kept and (moves or lifts or entry or rewrite):
        mine = {id(x) for x in ast.walk(pro)}
        out.errors += [f"V8: line {getattr(x, 'lineno', '?')}: {nm} is bound outside the prologue statement" for nm, x in binding_nodes(fn) if nm == ctx_names(m)[0] and id(x) not in mine]
    init = [x for x in fn.body[:3] if isinstance(x, ast.Assign) and ast.unparse(x).startswith(f"state = _{sm}.{sc}(")]
    if init:
        # An earlier PR (D1's state_init) already introduced both names, so reads of them are fine. Only
        # that one statement may bind them; any other binding would capture the rewritten ``state.<x>``.
        intro -= {"state", f"_{sm}"}
        mine = {id(x) for st in init for x in ast.walk(st)}
        out.errors += [f"V8: line {getattr(x, 'lineno', '?')}: {nm} is bound outside D1's state statement" for nm, x in binding_nodes(fn) if nm in ("state", f"_{sm}") and id(x) not in mine]
    if is_sibling(m):
        # an alias an earlier manifest already imported into the source may be read again, never rebound
        reuse = {f"_{x}" for x in {y.module for y in moves} | {lf.module for lf in lifts} | ({entry["module"]} if entry else set()) if src.infos.get(f"_{x}") == Info("from", pkg_dotted(m), x, f"_{x}", 0)}
        if lifts and src.infos.get("functools") == Info("import", "functools", "functools", None, 0):
            reuse.add("functools")  # an earlier manifest's partials already import it
        intro -= reuse
        out.errors += [f"V8: line {getattr(x, 'lineno', '?')}: {nm} is bound inside {fn.name}" for nm, x in binding_nodes(fn) if nm in reuse]
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
            # a published closure may be mapped when its def is the name's only binding in the function
            if isinstance(n, DEFS) and n.name in publish and sum(1 for nm, _ in binding_nodes(fn) if nm == n.name) == 1:
                continue
            bad = "a def or class" + (" (a published def must be the only binding of its name)" if isinstance(n, DEFS) and n.name in publish else "")
        if bad:
            out.errors.append(f"V5: line {n.lineno}: a mapped name is bound by {bad}")
    ri_plan, ri_errs = plan_rewrite_in(m, src)
    out.errors += ri_errs
    for n in publish:
        if not any(isinstance(s, DEFS) and s.name == n for b in own_blocks(fn) for s in b):
            out.errors.append(f"V5: publish: no closure {n} in {fn.name}")

    # classes: each is one ClassDef of the function, mapped to ``_<state_module>.<new>``
    class_defs: dict[str, tuple[list, Any]] = {}
    for old, new in m["classes"].items():
        defs = [(b, s) for b in own_blocks(fn) for s in b if isinstance(s, ast.ClassDef) and s.name == old]
        if len(defs) != 1:
            out.errors.append(f"V6: expected one class {old}, found {len(defs)}")
        elif rewrite.get(old) != f"_{sm}.{new}":
            out.errors.append(f"V6: class {old} must be in rewrite as _{sm}.{new}")
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
    if is_sibling(m):  # a loop could run a moved or lifted statement again after code that reads its result
        out.errors += [f"V1: {mv.name}: the span sits inside the loop at line {lp.lineno}; moves are refused in loops" for mv in moves if (lp := loop_above(fn, mv.members[0]))]
        out.errors += [f"V1: lifted {lf.name} sits inside the loop at line {lp.lineno}; lifts are refused in loops" for lf in lifts if (lp := loop_above(lf.parent_node or fn, lf.node))]
    for mv in moves:
        out.errors += check_span(mv, notes)
        out.errors += check_args(mv, fn, key, ctx_names(m), {lf.name for lf in lifts})
    if out.errors:
        return out
    lifted = tuple(lf.name for lf in lifts)

    for s in fn.body:
        Rewriter(key, rewrite, hidden).visit(s)
    for d, dkey, rw in ri_plan:
        Rewriter(dkey, rw, {}).visit(d)
    names = src.names
    up = UpTransform(names, src.exempt, m["facade"], bump=not is_sibling(m))
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
        for name, why in sorted(Cells(fn, key, cell_spans, tuple(ctx_names(m)) + lifted, {mv.name: mv.args for mv in moves}, strict=is_sibling(m)).crossing().items()):
            out.errors.append(f"V4: crossing local {name}: {'; '.join(why[:2])}")

    # lifts: reads of lifted names first (in every body), then each def becomes a module function and its site a partial
    lifted_defs: dict[tuple[str, str], Any] = {}
    if lifts:
        mark_needed(fn, lifts, regions, m["prune_dead_sites"])
        refs = LiftRefs(m, lifts, regions)
        for st in fn.body:
            refs.visit(st)
        out.errors += refs.errors
        blocks = list(own_blocks(fn)) + [b2 for p in {id(lf.parent_node): lf.parent_node for lf in lifts if lf.parent_node is not None}.values() for b2 in own_blocks(p)]
        bad = {m["facade"]} | {f"_{x}" for x in {y.module for y in moves} | {lf.module for lf in lifts}}
        for lf in lifts:
            for st in lf.dead:
                blk = next(b2 for b2 in own_blocks(lf.node) if any(x is st for x in b2))
                if len(blk) == 1:
                    out.errors.append(f"V11: removing the local import at line {st.lineno} would leave its block empty")
                else:
                    blk[:] = [x for x in blk if x is not st]
        for lf in sorted(lifts, key=lambda x: x.parent_node is None):  # a def inside a lifted def goes first, so its parent carries the site
            d = up.visit(lf.node)
            a = d.args
            (a.posonlyargs if a.posonlyargs else a.args)[0:0] = copy.deepcopy(ctx_params(m))
            heads = d.decorator_list + a.defaults + [x for x in a.kw_defaults if x]
            if any(isinstance(x, ast.Name) and x.id in bad for r in heads for x in ast.walk(r)):
                out.errors.append(f"V12: a decorator or default of lifted {lf.name} reads the facade or a module alias; it runs at import, before the late block")
            site = [] if lf.module == regions.module(lf.defregion) or not lf.needed else [ast.parse(f"{lf.name} = functools.partial(_{lf.module}.{lf.name}, {call_args(m)})").body[0]]
            if not site and (len(lf.block) == 1 or (isinstance(lf.defregion, Move) and len(lf.defregion.members) == 1 and lf.defregion.members[0] is lf.node)):
                out.errors.append(f"V11: lifting {lf.name} would leave its block or move empty")
            replace_stmt(lf.node, site, blocks, moves)
            lifted_defs[(lf.module, lf.name)] = d
        if out.errors:
            return out
        out.defs.update(lifted_defs)
        out.lifted = set(lifted_defs)
        for lf in lifts:
            out.free_imps.setdefault(lf.module, set()).update(lf.free)

    # moves, inside out
    for mv in sorted(moves, key=lambda x: x.b - x.a):
        mv.is_async = contains_await(mv.members)
        d = func_template(mv.name, mv.is_async, ", ".join([ctx_sig(m), *mv.args]))
        for i, s in enumerate(mv.members):
            mv.members[i] = up.visit(s)
        d.body = ([ast.Expr(ast.Constant(mv.doc))] if mv.doc else []) + mv.members
        out.defs[(mv.module, mv.name)] = d
        call = call_stmt(mv, "" if mv.parent else regions.rest_module, ", ".join([call_args(m), *mv.args]))
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
        new.body = ([doc_stmt(m["entry_doc"])] if m["entry_doc"] is not None else []) + body[i0 + 1 :]
        new.returns = fn.returns
        shim = ast.Return(value=ast.Call(ast.Name("run_rag_pipeline", ast.Load()), [s1.value], []))
        if aw:
            shim.value = ast.Await(shim.value)
        head = [doc_stmt(m["shim_doc"])] if m["shim_doc"] is not None else body[:i0]
        fn.body = head + [shim]
        out.entry = new
    if entry and not out.errors:
        split_entry_into_module(m, out, fn, key, up, entry)
    if is_sibling(m):
        out.order = list(lifted_defs) + [(mv.module, mv.name) for mv in moves] + ([(entry["module"], entry["name"])] if entry else [])
        finish_sibling(m, out)
    return out


def split_entry_into_module(m: dict, out: Built, fn: Any, key: Any, up: UpTransform, entry: dict) -> None:
    """The body after the prologue statement moves whole into ``<module>.<name>(cst)``; the source keeps a shim."""
    i0 = docstring_len(fn)
    pro, rest = fn.body[i0], fn.body[i0 + 1 :]
    params = {x.arg for x in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs + [fn.args.vararg, fn.args.kwarg] if x} - set(ctx_names(m))
    stored = {x.id for s in rest for x in ast.walk(s) if isinstance(x, ast.Name) and is_f(x, key) and isinstance(x.ctx, ast.Store)}
    for n in (x for s in rest for x in ast.walk(s)):
        if isinstance(n, ast.Name) and is_f(n, key) and n.id in params - stored and isinstance(n.ctx, ast.Load):
            out.errors.append(f"V8: the entry body reads the parameter {n.id} (line {n.lineno}); map it into {ctx_names(m)[0]} first")
        elif isinstance(n, ast.Name) and n.id in ("__name__", "__file__") and isinstance(n.ctx, ast.Load):
            out.errors.append(f"V1: the entry body reads {n.id} (line {n.lineno}); it reads a different module after the move")
    for n in walk_own(rest):
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            out.errors.append(f"V1: the entry body contains yield at line {n.lineno}; a generator cannot move behind `return await`")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in INTROSPECTION and getattr(n.func, "_r", NEW) is None:
            out.errors.append(f"V1: the entry body calls {n.func.id}() (line {n.lineno}); it reads a different scope after the move")
    aw = isinstance(fn, ast.AsyncFunctionDef)
    d = func_template(entry["name"], aw, ctx_sig(m))
    d.returns = up.visit(clone(fn.returns)) if fn.returns else None
    d.body = ([doc_stmt(entry["doc"])] if entry.get("doc") is not None else []) + [up.visit(x) for x in rest]
    out.defs[(entry["module"], entry["name"])] = d
    call = ast.Call(ast.Attribute(ast.Name(f"_{entry['module']}", ast.Load()), entry["name"], ast.Load()), [pro.value], [])
    fn.body = fn.body[:i0] + [ast.Return(ast.Await(call) if aw else call)]


def finish_sibling(m: dict, out: Built) -> None:
    """Which destination aliases and ``functools`` the source function still reads once everything is moved."""
    out.dests = {mod for mod, _ in out.order}
    names = {n.id for n in ast.walk(out.fn) if isinstance(n, ast.Name)}
    out.src_mods = sorted(x for x in out.dests if f"_{x}" in names)
    out.src_functools = "functools" in names


def clone(n: Any) -> Any:
    """A copy of an AST that keeps its positions and scope annotations (``copy.deepcopy`` cannot copy symtable entries)."""
    if isinstance(n, list):
        return [clone(x) for x in n]
    if not isinstance(n, ast.AST):
        return n
    new = type(n)(**{f: clone(getattr(n, f)) for f in n._fields if hasattr(n, f)})
    for a in ("lineno", "col_offset", "end_lineno", "end_col_offset", "_r", "_rs"):
        if hasattr(n, a):
            setattr(new, a, getattr(n, a))
    return new


def prologue_stmt(fn: Any, ctx0: str) -> ast.stmt | None:
    """``<ctx0> = <expr>``, the first statement after the docstring, if the function has it."""
    i0 = docstring_len(fn)
    s1 = fn.body[i0] if len(fn.body) > i0 else None
    if isinstance(s1, ast.Assign) and len(s1.targets) == 1 and isinstance(s1.targets[0], ast.Name) and s1.targets[0].id == ctx0:
        return s1
    return None


def doc_stmt(text: str) -> ast.stmt:
    return ast.Expr(ast.Constant(text))


def doc_text(text: str, indent: str = "    ") -> list[str]:
    """Source rows of a docstring statement whose value is exactly ``text``."""
    if '"""' in text or "\\" in text or text.endswith('"'):
        return [indent + repr(text)]
    return (indent + '"""' + text + '"""').split("\n")


def docstring_len(fn: Any) -> int:
    b = fn.body
    return 1 if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) and isinstance(b[0].value.value, str) else 0


def free_locals(d: Any, key: Any) -> set[str]:
    """F's locals that a closure reads or writes (its free variables resolved into F)."""
    return {n.id for s in d.body for n in ast.walk(s) if isinstance(n, ast.Name) and is_f(n, key)} | {
        x for s in d.body for n in ast.walk(s) if isinstance(n, ast.Nonlocal) for x, r in zip(n.names, n._rs) if r is key  # type: ignore[attr-defined]
    }


def import_stmt(m: dict, name: str) -> ast.stmt:
    if is_sibling(m):
        return ast.parse(f"from {pkg_dotted(m)} import {name} as _{name}").body[0]
    return ast.parse(f"from .{m['pkg']} import {name} as _{name}").body[0]


def expected_module(m: dict, src: Src, b: Built) -> ast.Module:
    body = list(src.tree.body)
    i = next(i for i, s in enumerate(body) if s is src.fn)
    if b.entry is not None:
        body.insert(i + 1, b.entry)
    drops = [d for d in m["drop"] if d["module"] == src_stem(m)]
    if drops:
        body = drop_apply(body, drops, src_name(m))
    mods = list(dict.fromkeys([x.module for x in b.moves] + ([m["state_module"]] if m["hoist"] else [])))
    extra = []
    if is_sibling(m):
        mods = b.src_mods
        extra = [ast.parse("import functools").body[0]] if b.src_functools else []
    have = {ast.dump(s) for s in body}
    last = max((j for j, s in enumerate(body) if isinstance(s, (ast.Import, ast.ImportFrom))), default=-1)
    wanted = [ast.parse(x).body[0] for x in m["imports"]] + extra + [import_stmt(m, x) for x in mods]
    body[last + 1 : last + 1] = [s for s in wanted if ast.dump(s) not in have]
    if drops:
        drop_refs(body, drops, src_name(m))
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


def cmp_module(rel: str, want: ast.Module, head: ast.Module) -> list[str]:
    """V7: the whole module AST, reported statement by statement."""
    errs: list[str] = []
    if ast.dump(want) != ast.dump(head):
        if len(want.body) != len(head.body):
            wn = [getattr(s, "name", type(s).__name__) for s in want.body]
            hn = [getattr(s, "name", type(s).__name__) for s in head.body]
            errs.append(f"{rel}: top-level statements differ: {len(want.body)} expected, {len(head.body)} found")
            errs += [f"{rel}: {d[:160]}" for d in difflib.unified_diff(wn, hn, "expected", "found", lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")][:8]
        else:
            for w, h in zip(want.body, head.body):
                if ast.dump(w) != ast.dump(h):
                    errs += diff_lines(f"{rel}: {getattr(w, 'name', type(w).__name__)} (line {getattr(w, 'lineno', '?')})", w, h)
    return errs


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
    src = mk_src(text, m)
    b = build(m, src, text.split("\n"), notes)
    if b.errors:
        return b.errors
    errs: list[str] = []
    head_text = read_head(root, rel, head_ref)
    if head_text is None:
        return [f"{rel}: not found at the head"]
    head = ast.parse(head_text)
    try:
        want = expected_module(m, src, b)
    except SplitError as e:
        return [str(e)]
    errs += cmp_module(rel, want, head)  # V7
    if is_sibling(m):
        return errs + verify_siblings(root, m, src, b, base, base_dir, head_ref)
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
        tree = module(m["state_module"])
        got_cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == new), None) if tree else None
        if got_cls is None:
            errs.append(f"{m['state_module']}.py: no class {new}")
        elif ast.dump(want_cls) != ast.dump(got_cls):
            errs += diff_lines(f"{m['state_module']}.py: class {new}", want_cls, got_cls)
    if b.methods:
        tree = module(m["state_module"])
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == m["state_class"]), None) if tree else None
        for method, want_def in b.methods.items():
            got = next((n for n in cls.body if isinstance(n, DEFS) and n.name == method), None) if cls else None
            if got is None:
                errs.append(f"{m['state_module']}.py: no method {m['state_class']}.{method}")
            elif ast.dump(want_def) != ast.dump(got):
                errs += diff_lines(f"{m['state_module']}.py: {method}", want_def, got)
    # V9 and the module-shape checks, for every touched pipeline module
    stdlib = {n: i for n, i in src.infos.items() if i.stdlib}
    mods = list(dict.fromkeys([x.module for x in b.moves] + ([m["state_module"]] if m["hoist"] else [])))
    for name in mods:
        t = module_text(name)
        if t is not None:
            errs += check_module(t, name, m, stdlib, {x.module for x in b.moves}, name == m["state_module"])
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
    psrc = Src(text, pkg=m["pkg"])
    stem = Path(m["file"]).stem
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
            errs.append(f"V9: {label} binds {n} as {got.line() if got else 'a non-import'}; {stem} binds it as {stdlib[n].line()}")
    for n in sorted({x.id for x in ast.walk(psrc.tree) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and getattr(x, "_r", NEW) is None}):
        if n in stdlib and psrc.infos.get(n) != stdlib[n] and not bound.get(n):
            errs.append(f"V9: {label} reads {n}, bound there as nothing; {stem} binds it as {stdlib[n].line()}")
    parts = m["file"][:-3].split("/")
    fa = m["facade"]
    want = {fa: Info("from", ".".join(parts[:-1]), parts[-1], fa, 0)}
    if not is_state:
        for n in ctx_imports(m)[0]:
            want[n] = Info("from", "run_config", n, None, 1)
    for c in modules - {name}:
        want[f"_{c}"] = Info("from", "", c, f"_{c}", 1)
    for n, info in want.items():
        got = psrc.infos.get(n)
        if got is not None and got != info:
            errs.append(f"{label}: {n} is bound as {got.line()}, expected {info.line()}")
    uses_up = any(isinstance(x, ast.Name) and x.id == fa for x in ast.walk(psrc.tree))
    if uses_up and fa not in psrc.infos:
        errs.append(f"{label}: reads {fa} but does not import it")
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
    scopes = [(key, rewrite)] + [(k, rw) for _, k, rw in plan_rewrite_in(m, src)[0]]  # rewrite_in: the nested def's own scope
    gone = lambda x, r: any(r is k and x in rw for k, rw in scopes)  # noqa: E731
    edits: list[Edit] = []
    cuts: list[Cut] = []
    for n in ast.walk(fn):
        hit = next((rw[n.id] for k, rw in scopes if isinstance(n, ast.Name) and is_f(n, k) and n.id in rw), None)
        if hit is not None:
            edits.append(Edit(*pos(n), hit, True))
        elif isinstance(n, ast.Call) and (hit := hidden_match(n, key, hidden)):
            edits.append(Edit(*pos(n), f"rcfg.{hit[0]}.{hit[1]}", True))
        elif isinstance(n, ast.Nonlocal) and any(gone(x, r) for x, r in zip(n.names, n._rs)):  # type: ignore[attr-defined]
            keep = [x for x, r in zip(n.names, n._rs) if not gone(x, r)]  # type: ignore[attr-defined]
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


def region_edits(src: Src, lines: list[bytes], lo: int, hi: int, root_nodes: list[ast.AST], hoist: bool, facade: str = "up", bump: bool = True) -> list[Edit]:
    out: list[Edit] = []
    imports = []
    for r in root_nodes:
        for n in ast.walk(r):
            if getattr(n, "lineno", None) is None or not (lo <= n.lineno <= hi):
                continue
            if isinstance(n, ast.Name):
                if getattr(n, "_r", NEW) is None and n.id in src.names and n.id not in src.exempt:
                    out.append(Edit(*pos(n), f"{facade}.{n.id}"))
                elif hoist and n.id in ("state", "rcfg"):
                    out.append(Edit(*pos(n), "self" if n.id == "state" else "self.rcfg"))
            elif isinstance(n, ast.ImportFrom) and n.level:
                imports.append(n)
    return out + (level_edits(lines, imports) if bump else [])


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
        out.update(ctx_imports(m)[0])
    if need_up:
        parts = m["file"][:-3].split("/")
        out[m["facade"]] = f"from {'.'.join(parts[:-1])} import {parts[-1]} as {m['facade']}"
    for c in sorted(cross):
        out[f"_{c}"] = f"from . import {c} as _{c}"
    return out


def last_import_row(tree: ast.Module) -> int:
    rows = [s.end_lineno for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))]
    if rows:
        return max(rows)
    first = tree.body[0] if tree.body else None
    return first.end_lineno if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) else 0


def order_imports(wanted: dict[str, str], facade: str = "up") -> list[str]:
    std = sorted(v for k, v in wanted.items() if not v.startswith("from .") and k not in (facade, "TYPE_CHECKING"))
    typ = [v for k, v in wanted.items() if k == "TYPE_CHECKING"]
    absolute = [v for k, v in wanted.items() if k == facade]
    rel = sorted(v for v in wanted.values() if v.startswith("from ."))
    return std + typ + absolute + rel


def add_imports(text: str, wanted: dict[str, str], facade: str = "up") -> str:
    have = set(import_infos(ast.parse(text)))
    new = order_imports({k: v for k, v in wanted.items() if k not in have}, facade)
    if not new:
        return text
    rows = text.split("\n")
    at = last_import_row(ast.parse(text))
    rows[at:at] = new
    return "\n".join(rows)


def header_text(mod: str, wanted: dict[str, str], stage: bool, m: dict) -> str:
    parts = [f'"""{mod} code moved from {src_name(m)} ({m["task"]})."""', ""] + order_imports(wanted, m["facade"])
    if stage:
        parts += ["", "if TYPE_CHECKING:"] + [f"    {x}" for x in ctx_imports(m)[1]]
    return "\n".join(parts) + "\n"


def run_apply(root: Path, m: dict) -> list[str]:
    if is_sibling(m):
        return apply_sibling(root, m)
    rel = m["file"]
    path = root / rel
    text = path.read_text(encoding="utf-8")
    rows_text = text.split("\n")
    src = mk_src(text, m)
    b = build(m, src, rows_text)
    if b.errors:
        return b.errors
    fresh = mk_src(text, m)  # build() mutated the first tree; positions come from this one
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
        call = f"{'await ' if mv.is_async else ''}{recv}{mv.name}({', '.join([call_args(m), *mv.args])})"
        move_cuts[mv.name] = Cut(mv.a, mv.b, call, mv.members[0].col_offset)
    cuts += list(hoist_cuts.values()) + list(move_cuts.values())

    pkg_dir = root / pkg_rel(m)
    sm, sc = m["state_module"], m["state_class"]
    state_path = pkg_dir / f"{sm}.py"
    outputs: dict[Path, list[str]] = defaultdict(list)
    wanted: dict[Path, dict[str, str]] = defaultdict(dict)
    for mv in moves:
        shift = mv.members[0].col_offset - 4
        live = lambda row, mv=mv: not any(c.a <= row <= c.b for c in mv.children)  # noqa: E731
        ed = [e for e in edits if mv.a <= e.row <= mv.b] + region_edits(fresh, lines, mv.a, mv.b, list(mv.members), False, m["facade"])
        for ln in mv.noqa:
            if not mv.a <= ln <= mv.b:
                return [f"{mv.name}: noqa line {ln} is outside the span {mv.a}-{mv.b}"]
            ed.append(Edit(ln, row_end(lines, ln), ln, row_end(lines, ln), noqa_text(m)))
        body = render(lines, cont, mv.a, mv.b, shift, ed, cuts, own=move_cuts[mv.name])
        if mv.doc is not None:
            if '"""' in mv.doc or "\\" in mv.doc:
                return [f"{mv.name}: doc must not contain triple quotes or backslashes"]
            body = f'    """{mv.doc}"""\n' + body
        head = ("async " if mv.is_async else "") + f"def {mv.name}({', '.join([ctx_sig(m), *mv.args])}) -> None:\n"
        target = pkg_dir / f"{mv.module}.py"
        outputs[target].append(head + body)
        mine = [n for n in (x for s in mv.members for x in ast.walk(s)) if live(getattr(n, "lineno", mv.a))]
        need_up = any(e.text.startswith(m["facade"] + ".") and live(e.row) for e in ed)
        cross = {c.module for c in mv.children if c.module != mv.module}
        wanted[target].update(needed_imports(fresh, m, mine, cross, need_up, True))

    methods: list[str] = []
    for name, method in m["hoist"].items():
        d = next(s for blk in blocks for s in blk if isinstance(s, DEFS) and s.name == name)
        lo = stmt_start(d)
        ed = [e for e in edits if lo <= e.row <= d.end_lineno] + region_edits(fresh, lines, lo, d.end_lineno, [d], True, m["facade"])
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
        wanted[state_path].update(needed_imports(fresh, m, [d], set(), any(e.text.startswith(m["facade"] + ".") for e in ed), False))

    for new, d in class_nodes.items():
        lo = stmt_start(d)
        ed = region_edits(fresh, lines, lo, d.end_lineno, [d], False, m["facade"])
        mt = re.match(rb"(\s*class\s+)(\w+)", lines[d.lineno - 1])
        if not mt:
            return [f"{new}: cannot find the class line"]
        ed.append(Edit(d.lineno, mt.start(2), d.lineno, mt.end(2), new))
        class_text[new] = render(lines, cont, lo, d.end_lineno, d.col_offset, ed, cuts, own=hoist_cuts[f"class {d.name}"]).rstrip("\n")
        wanted[state_path].update(needed_imports(fresh, m, [d], set(), any(e.text.startswith(m["facade"] + ".") for e in ed), False))

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
        new_f = split_entry(new_f, m)
    rows = rows_text[: f_lo - 1] + new_f.rstrip("\n").split("\n") + rows_text[f_hi:]
    new_text = "\n".join(rows)
    mods = list(dict.fromkeys([x.module for x in moves] + ([sm] if m["hoist"] else [])))
    tree = ast.parse(new_text)
    have = {ast.dump(s) for s in tree.body}
    imps = [x for x in m["imports"] if ast.dump(ast.parse(x).body[0]) not in have]
    imps += [f"from .{m['pkg']} import {x} as _{x}" for x in mods if ast.dump(import_stmt(m, x)) not in have]
    if imps:
        rows = new_text.split("\n")
        at = last_import_row(tree)
        rows[at:at] = imps
        new_text = "\n".join(rows)

    state_text = ""
    if methods:
        cur = state_path.read_text(encoding="utf-8") if state_path.exists() else header_text(sm, wanted[state_path], False, m) + f"\n\nclass {sc}:\n    pass\n"
        cur = add_imports(cur, wanted[state_path], m["facade"]) if state_path.exists() else cur
        cls = next((n for n in ast.parse(cur).body if isinstance(n, ast.ClassDef) and n.name == sc), None)
        if cls is None:
            return [f"{sm}.py has no {sc} class"]
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
        cur = add_imports(p.read_text(encoding="utf-8"), wanted[p], m["facade"]) if p.exists() else header_text(p.stem, wanted[p], True, m)
        cur = cur.rstrip("\n") + "\n" + "".join("\n\n" + f.rstrip("\n") + "\n" for f in funcs)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(cur, encoding="utf-8")
    if methods:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(state_text, encoding="utf-8")
    return []


def split_entry(f_text: str, m: dict) -> str:
    tree = ast.parse(f_text)
    f = tree.body[0]
    i0 = 1 if isinstance(f.body[0], ast.Expr) and isinstance(getattr(f.body[0], "value", None), ast.Constant) else 0
    s1 = f.body[i0]
    aw = isinstance(f, ast.AsyncFunctionDef)
    rows = f_text.split("\n")
    value = ast.get_source_segment(f_text, s1.value)
    ret = f" -> {ast.get_source_segment(f_text, f.returns)}" if f.returns else ""
    head = rows[: stmt_start(s1) - 1]
    if m["shim_doc"] is not None:
        d0 = f.body[0].lineno - 1 if i0 else stmt_start(s1) - 1
        head = head[:d0] + doc_text(m["shim_doc"]) + ([""] if i0 == 0 else head[f.body[0].end_lineno :])
    shim = head + [f"    return {'await ' if aw else ''}run_rag_pipeline({value})"]
    new = ["", "", f"{'async ' if aw else ''}def run_rag_pipeline(rcfg: RagRunConfig){ret}:"]
    if m["entry_doc"] is not None:
        new += doc_text(m["entry_doc"]) + [""]
    new += rows[s1.end_lineno : f.end_lineno]
    return "\n".join(shim + new) + "\n"


# --------------------------------------------------------------------------------------------------
# sibling mode (pkg ""): append into destination modules that already exist, behind a late-import block
# --------------------------------------------------------------------------------------------------


class ModPlan:
    """What one destination module gains: defs above the late block, imports, aliases."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.text = ""
        self.defs: list[ast.stmt] = []
        self.std: dict[str, str] = {}  # bound name -> import line, after the module's last top import
        self.aliases: list[str] = []  # other destination modules the new code calls, imported in the late block
        self.need_cs = False
        self.drops: list[dict] = []
        self.late_ranges: list[list[int]] = []  # rows of top imports that move into the late block
        self.free: set[str] = set()  # free_imports names the appended defs read: module-level imports from the manifest's lines


def split_late(text: str) -> tuple[ast.Module, list[ast.stmt], list[ast.stmt]]:
    """``(tree, top statements, late-block statements)``; the block starts at the marker comment."""
    tree = ast.parse(text)
    k = next((i for i, r in enumerate(text.split("\n"), 1) if r.strip() == LATE_MARK), None)
    if k is None:
        return tree, list(tree.body), []
    late = [s for s in tree.body if s.lineno > k]
    bad = [s for s in late if not isinstance(s, (ast.Import, ast.ImportFrom))]
    if bad:
        raise SplitError(f"line {bad[0].lineno}: only imports may follow the late-import marker")
    return tree, [s for s in tree.body if s.lineno < k], late


def is_top_import(s: ast.stmt) -> bool:
    return isinstance(s, (ast.Import, ast.ImportFrom))


def type_checking_block(top: list[ast.stmt]) -> ast.If | None:
    return next((s for s in top if isinstance(s, ast.If) and isinstance(s.test, ast.Name) and s.test.id == "TYPE_CHECKING" and not s.orelse), None)


def top_additions(m: dict, plan: ModPlan, top: list[ast.stmt]) -> tuple[list[str], list[str]]:
    """``(import lines, TYPE_CHECKING lines)`` the module still lacks; each is added after its last top import."""
    if not plan.defs:  # a module that only loses or reorders imports gets no new ones
        return [], []
    have = set(import_infos(ast.Module(body=top, type_ignores=[])))
    tc = [x for x in ctx_imports(m)[1] if not {_bound(a) for a in ast.parse(x).body[0].names} <= have]  # type: ignore[attr-defined]
    std = dict(plan.std)
    if tc and type_checking_block(top) is None:
        std.setdefault("TYPE_CHECKING", "from typing import TYPE_CHECKING")
    return sorted({line for n, line in std.items() if n not in have}), tc


def late_block(m: dict, plan: ModPlan, late: list[ast.stmt], texts: list[str]) -> list[tuple[ast.stmt, str]]:
    """The late block: what it has, then the new aliases, then ``cs`` last (R5). ``texts`` are the source of ``late``."""
    cs = ast.parse(f"from {pkg_dotted(m)} import {src_stem(m)} as {m['facade']}").body[0]
    out: dict[str, tuple[ast.stmt, str]] = {}
    for s, t in zip(late, texts):
        if ast.dump(s) != ast.dump(cs):
            out.setdefault(ast.dump(s), (s, t))
    for a in plan.aliases:
        s = import_stmt(m, a)
        out.setdefault(ast.dump(s), (s, ast.unparse(s)))
    keep = plan.need_cs or any(ast.dump(s) == ast.dump(cs) for s in late)
    return list(out.values()) + ([(cs, ast.unparse(cs))] if keep else [])


def sib_expected(m: dict, plan: ModPlan) -> tuple[ast.Module, int]:
    """The destination module as an AST: base top, added imports, appended defs, late block. Plus the block's length."""
    _, top, late = split_late(plan.text)
    moved = [x for x in top if is_top_import(x) and [x.lineno, x.end_lineno] in plan.late_ranges]
    if len(moved) != len(plan.late_ranges):
        raise SplitError(f"{plan.name}.py: late_imports {plan.late_ranges} are not whole top-level import statements")
    top = [x for x in top if not any(x is y for y in moved)]
    top = drop_apply(top, plan.drops, f"{plan.name}.py")
    std, tc = top_additions(m, plan, top)
    at = max((i for i, s in enumerate(top) if is_top_import(s)), default=-1) + 1
    if at == 0 and top and isinstance(top[0], ast.Expr) and isinstance(getattr(top[0], "value", None), ast.Constant):
        at = 1
    ins: list[ast.stmt] = [ast.parse(x).body[0] for x in std]
    tcs = [ast.parse(x).body[0] for x in tc]
    blk = type_checking_block(top)
    if tcs and blk is not None:
        blk.body += tcs
    elif tcs:
        ins.append(ast.If(test=ast.Name("TYPE_CHECKING", ast.Load()), body=tcs, orelse=[]))
    block = late_block(m, plan, late + moved, [""] * (len(late) + len(moved)))
    body = top[:at] + ins + top[at:] + plan.defs + [s for s, _ in block]
    if plan.drops:
        drop_refs(body, plan.drops, f"{plan.name}.py")
    return ast.Module(body=body, type_ignores=[]), len(block)


def sib_write(m: dict, plan: ModPlan, def_texts: list[str]) -> str:
    """The text counterpart of ``sib_expected``."""
    rows = plan.text.split("\n")
    moved_txt: list[str] = []
    for a, b in sorted(plan.late_ranges, reverse=True):
        moved_txt.insert(0, "\n".join(rows[a - 1 : b]))
        del rows[a - 1 : b]
    text = drop_text("\n".join(rows), plan.drops, f"{plan.name}.py")
    tree, top, late = split_late(text)
    rows = text.split("\n")
    std, tc = top_additions(m, plan, top)
    blk = type_checking_block(top)
    last = [s.end_lineno for s in top if is_top_import(s)]
    at = max(last) if last else (top[0].end_lineno if top and isinstance(top[0], ast.Expr) and isinstance(getattr(top[0], "value", None), ast.Constant) else 0)
    adds: list[tuple[int, list[str]]] = []
    if tc and blk is not None:
        adds.append((blk.end_lineno, [f"    {x}" for x in tc]))
    new = list(std) + (["", "if TYPE_CHECKING:"] + [f"    {x}" for x in tc] if tc and blk is None else [])
    if new:
        adds.append((at, new))
    for row, lines in sorted(adds, reverse=True):
        rows[row:row] = lines
    text = "\n".join(rows)
    tree, top, late = split_late(text)
    rows = text.split("\n")
    k = next((i for i, r in enumerate(rows, 1) if r.strip() == LATE_MARK), None)
    block = late_block(m, plan, late + [ast.parse(t).body[0] for t in moved_txt], [ast.get_source_segment(text, s) or "" for s in late] + moved_txt)
    head = rows[: k - 1] if k else rows
    out = "\n".join(head).rstrip("\n") + "\n" + "".join("\n\n" + d.rstrip("\n") + "\n" for d in def_texts)
    if block:
        out += "\n\n" + LATE_MARK + "\n" + "\n".join(t for _, t in block) + "\n"
    return out


def dest_names(plan: ModPlan, m: dict, src: Src, dests: set[str]) -> None:
    """Read the imports and the late-block aliases the appended defs need off their (transformed) ASTs."""
    for d in plan.defs:
        for n in ast.walk(d):
            if not isinstance(n, ast.Name):
                continue
            if n.id == m["facade"]:
                plan.need_cs = True
            elif n.id.startswith("_") and n.id[1:] in dests and n.id[1:] != plan.name:
                plan.aliases.append(n.id[1:])
            elif getattr(n, "_r", NEW) is None and n.id in src.exempt:
                plan.std[n.id] = src.infos[n.id].line(0 if src.infos[n.id].stdlib else 1)
            elif n.id == "functools" and getattr(n, "_r", NEW) is NEW:
                plan.std["functools"] = "import functools"
    plan.aliases = sorted(set(plan.aliases))


def make_plans(m: dict, src: Src, b: Built, text_of: Any) -> tuple[dict[str, ModPlan], list[str]]:
    errs: list[str] = []
    plans: dict[str, ModPlan] = {}
    for mod, name in b.order:
        plans.setdefault(mod, ModPlan(mod)).defs.append(b.defs[(mod, name)])
    for it in m["drop"]:
        if it["module"] != src_stem(m):
            plans.setdefault(it["module"], ModPlan(it["module"])).drops.append(it)
    for li in m["late_imports"]:
        plans.setdefault(li["module"], ModPlan(li["module"])).late_ranges.append(li["lines"])
    for mod, p in plans.items():
        t = text_of(mod)
        if t is None:
            errs.append(f"{mod}.py: not found at the base")
            continue
        p.text = t
        dest_names(p, m, src, b.dests)
        p.free = set(b.free_imps.get(mod, ()))
        have = import_infos(ast.parse(t))
        for n in sorted(p.free):
            line = m["free_imports"][n]
            p.std[n] = line
            if n in have and have[n] != import_infos(ast.parse(line))[n]:
                errs.append(f"V9: {mod}.py binds {n} as {have[n].line()}; free_imports wants `{line}`")
        if p.late_ranges:
            tree, top, _ = split_late(t)
            moved = [x for x in top if is_top_import(x) and [x.lineno, x.end_lineno] in p.late_ranges]
            if len(moved) != len(p.late_ranges):
                errs.append(f"{mod}.py: late_imports {p.late_ranges} are not whole top-level import statements")
            else:
                bound = {_bound(a) for x in moved for a in x.names}  # type: ignore[attr-defined]
                rest = [x for x in top if not any(x is y for y in moved)]
                errs += [f"V14: {mod}.py reads {n} when it is imported, but late_imports moves that import to the end" for n in sorted(bound & {x.id for x in eval_names(rest, has_future(tree))})]
        clash = module_names(ast.parse(t))
        errs += [f"V10: {mod}.py already binds {d.name}" for d in p.defs if d.name in clash]  # type: ignore[attr-defined]
        names = [d.name for d in p.defs]  # type: ignore[attr-defined]
        errs += [f"V10: {mod}.py would get {n} twice" for n in sorted({x for x in names if names.count(x) > 1})]
        bad = {m["facade"]} | {f"_{x}" for x in b.dests}
        gone = {it.get("def") or it.get("import") for it in m["drop"] if it["module"] == src_stem(m)} - {None}
        for d in p.defs:
            errs += [
                f"V15: {mod}.py: {d.name} reads {m['facade']}.{n.attr} (line {n.lineno}), but {src_stem(m)} drops {n.attr}"  # type: ignore[attr-defined]
                for n in ast.walk(d)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == m["facade"] and n.attr in gone and getattr(n.value, "_r", NEW) is NEW
            ]
            if (mod, d.name) in b.lifted and not has_future(ast.parse(t)):  # type: ignore[attr-defined]
                a = d.args  # type: ignore[attr-defined]
                heads = [x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg] if x and x.annotation] + [d.returns]  # type: ignore[attr-defined]
                if any(isinstance(x, ast.Name) and x.id in bad for r in heads if r for x in ast.walk(r)):
                    errs.append(f"V13: {mod}.py has no `from __future__ import annotations`, but an annotation of lifted {d.name} reads the facade or a module alias")  # type: ignore[attr-defined]
    return plans, errs


def eval_names(stmts: list[ast.stmt], future: bool) -> Iterator[ast.Name]:
    """Names read when the module is imported: function bodies and (under the future import) annotations do not run."""

    def names(*roots: ast.AST | None) -> Iterator[ast.Name]:
        for r in roots:
            if r is not None:
                yield from (n for n in ast.walk(r) if isinstance(n, ast.Name))

    for s in stmts:
        if isinstance(s, DEFS):
            a = s.args
            heads = [*s.decorator_list, *a.defaults, *a.kw_defaults]
            if not future:
                heads += [x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg] if x] + [s.returns]
            yield from names(*heads)
        elif isinstance(s, ast.ClassDef):
            yield from names(*s.decorator_list, *s.bases, *[k.value for k in s.keywords])
            yield from eval_names(s.body, future)
        elif isinstance(s, ast.AnnAssign):
            yield from names(s.value, None if future else s.annotation, s.target)
        else:
            for c in ast.iter_child_nodes(s):
                if isinstance(c, ast.stmt):
                    yield from eval_names([c], future)
                elif isinstance(c, (ast.ExceptHandler, ast.match_case)):
                    yield from names(getattr(c, "type", None), getattr(c, "pattern", None), getattr(c, "guard", None))
                    yield from eval_names(c.body, future)
                else:
                    yield from names(c)


def has_future(tree: ast.Module) -> bool:
    return any(isinstance(s, ast.ImportFrom) and s.module == "__future__" and any(a.name == "annotations" for a in s.names) for s in tree.body)


def sib_checks(m: dict, src: Src, plan: ModPlan, want_late: int, head_text: str) -> list[str]:
    """What the AST comparison cannot see: the marker, import parity (V9), undefined names, cycle safety (R5)."""
    label = f"{plan.name}.py"
    errs: list[str] = []
    if want_late and not any(r.strip() == LATE_MARK for r in head_text.split("\n")):
        errs.append(f"{label}: the late-import block has no marker comment")
    psrc = Src(head_text, pkg=m["pkg"])
    for n in sorted(plan.std):
        if n == "TYPE_CHECKING":
            continue
        want = src.infos.get(n) if n != "functools" else Info("import", "functools", "functools", None, 0)
        if n in plan.free:
            want = import_infos(ast.parse(m["free_imports"][n])).get(n)
        if psrc.infos.get(n) != want:
            got = psrc.infos.get(n)
            errs.append(f"V9: {label} binds {n} as {got.line() if got else 'nothing'}; {src_stem(m)} binds it as {want.line() if want else '?'}")
    import builtins

    mine = {d.name for d in plan.defs}  # type: ignore[attr-defined]
    for d in psrc.tree.body:
        if isinstance(d, DEFS) and d.name in mine:
            for n in ast.walk(d):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and getattr(n, "_r", NEW) is None and n.id not in psrc.names and not hasattr(builtins, n.id):
                    errs.append(f"{label}: {d.name} reads {n.id} (line {n.lineno}), which the module does not bind")
    bad = {m["facade"]} | {f"_{x}" for x in plan.aliases} | {f"_{plan.name}"}
    errs += [f"R5: {label}: line {n.lineno}: module-level code reads {n.id}, which is bound only in the late block" for n in eval_names(psrc.tree.body, has_future(psrc.tree)) if n.id in bad]
    return errs


def verify_siblings(root: Path, m: dict, src: Src, b: Built, base: str | None, base_dir: str | None, head_ref: str | None) -> list[str]:
    pkg = pkg_rel(m)
    plans, errs = make_plans(m, src, b, lambda mod: read_base(root, (pkg / f"{mod}.py").as_posix(), base, base_dir))
    for mod, p in plans.items():
        if not p.text:
            continue
        ht = read_head(root, pkg / f"{mod}.py", head_ref)
        if ht is None:
            errs.append(f"{mod}.py: not found at the head")
            continue
        try:
            want, nlate = sib_expected(m, p)
        except SplitError as e:
            errs.append(str(e))
            continue
        errs += cmp_module(f"{mod}.py", want, ast.parse(ht))
        errs += sib_checks(m, src, p, nlate, ht)
    return errs


def apply_sibling(root: Path, m: dict) -> list[str]:
    rel = m["file"]
    path = root / rel
    text = path.read_text(encoding="utf-8")
    rows_text = text.split("\n")
    src = mk_src(text, m)
    b = build(m, src, rows_text)
    if b.errors:
        return b.errors
    pkg_dir = root / pkg_rel(m)
    plans, errs = make_plans(m, src, b, lambda mod: (pkg_dir / f"{mod}.py").read_text(encoding="utf-8") if (pkg_dir / f"{mod}.py").exists() else None)
    if errs:
        return errs
    try:  # the drop checks (V15) run on the AST before anything is written
        expected_module(m, src, b)
        for p in plans.values():
            sib_expected(m, p)
    except SplitError as e:
        return [str(e)]
    fresh = mk_src(text, m)  # build() mutated the first tree; positions come from this one
    lines = text.encode("utf-8").split(b"\n")
    lines = [ln + b"\n" for ln in lines[:-1]] + lines[-1:]
    cont = continuation_rows(text)
    fn = fresh.fn
    edits, cuts = gather(fresh, m, lines)
    moves = plan_moves(fresh, m["moves"], rows_text)
    for mv in moves:
        mv.is_async = next(x.is_async for x in b.moves if x.name == mv.name)
    ctx = call_args(m)
    lifts, _ = plan_lifts(m, fresh, moves) if m["lift"] else ([], [])
    entry = m["entry"]
    regions = Regions(moves, lifts, entry["module"] if entry else "")
    mark_needed(fn, lifts, regions, m["prune_dead_sites"])
    edits += lift_edits(fn, lines, regions, lifts, ctx)
    lift_cuts: dict[str, Cut] = {}
    for lf in lifts:
        d = lf.node
        keep = lf.module != regions.module(lf.defregion) and lf.needed
        tail = d.end_lineno + (1 if not keep and d.end_lineno < len(rows_text) and not rows_text[d.end_lineno].strip() else 0)
        lift_cuts[lf.name] = Cut(stmt_start(d), tail, f"{lf.name} = functools.partial(_{lf.module}.{lf.name}, {ctx})" if keep else None, d.col_offset)
    dead_cuts = [Cut(stmt_start(st), st.end_lineno + (1 if st.end_lineno < len(rows_text) and not rows_text[st.end_lineno].strip() else 0), None, st.col_offset) for lf in lifts for st in lf.dead]
    move_cuts: dict[str, Cut] = {}
    for mv in moves:
        recv = "" if (mv.parent.module if mv.parent else regions.rest_module) == mv.module else f"_{mv.module}."
        call = f"{'await ' if mv.is_async else ''}{recv}{mv.name}({', '.join([ctx, *mv.args])})"
        move_cuts[mv.name] = Cut(mv.a, mv.b, call, mv.members[0].col_offset)
    cuts += list(lift_cuts.values()) + list(move_cuts.values()) + dead_cuts
    made: dict[tuple[str, str], str] = {}
    for lf in lifts:
        d = lf.node
        lo = stmt_start(d)
        ed = [e for e in edits if lo <= e.row <= d.end_lineno] + region_edits(fresh, lines, lo, d.end_lineno, [d], False, m["facade"], bump=False)
        ed.append(sig_edit(lines, d, ctx_sig(m)))
        made[(lf.module, lf.name)] = render(lines, cont, lo, d.end_lineno, d.col_offset, ed, cuts, own=lift_cuts[lf.name])
    for mv in moves:
        shift = mv.members[0].col_offset - 4
        ed = [e for e in edits if mv.a <= e.row <= mv.b] + region_edits(fresh, lines, mv.a, mv.b, list(mv.members), False, m["facade"], bump=False)
        for ln in mv.noqa:
            if not mv.a <= ln <= mv.b:
                return [f"{mv.name}: noqa line {ln} is outside the span {mv.a}-{mv.b}"]
            ed.append(Edit(ln, row_end(lines, ln), ln, row_end(lines, ln), noqa_text(m)))
        body = render(lines, cont, mv.a, mv.b, shift, ed, cuts, own=move_cuts[mv.name])
        if mv.doc is not None:
            if '"""' in mv.doc or "\\" in mv.doc:
                return [f"{mv.name}: doc must not contain triple quotes or backslashes"]
            body = f'    """{mv.doc}"""\n' + body
        head = ("async " if mv.is_async else "") + f"def {mv.name}({', '.join([ctx_sig(m), *mv.args])}) -> None:\n"
        made[(mv.module, mv.name)] = head + body
    f_lo, f_hi = stmt_start(fn), fn.end_lineno
    if entry:
        i0 = docstring_len(fn)
        pro, aw = fn.body[i0], isinstance(fn, ast.AsyncFunctionDef)
        lo = pro.end_lineno + 1
        ed = edits + region_edits(fresh, lines, lo, f_hi, fn.body[i0 + 1 :], False, m["facade"], bump=False)
        body = render(lines, cont, lo, f_hi, 0, ed, cuts).lstrip("\n")
        want = b.defs[(entry["module"], entry["name"])]
        sig = f"{'async ' if aw else ''}def {entry['name']}({ctx_sig(m)})" + (f" -> {ast.unparse(want.returns)}" if want.returns else "") + ":\n"
        doc = "\n".join(doc_text(entry["doc"])) + "\n\n" if entry.get("doc") is not None else ""
        made[(entry["module"], entry["name"])] = sig + doc + body
        value = ast.get_source_segment(text, pro.value)
        shim = [f"    return {'await ' if aw else ''}_{entry['module']}.{entry['name']}({value})"]
        new_f = "\n".join(rows_text[f_lo - 1 : pro.lineno - 1] + shim)
    else:
        new_f = render(lines, cont, f_lo, f_hi, 0, edits, cuts)
    rows = rows_text[: f_lo - 1] + new_f.rstrip("\n").split("\n") + rows_text[f_hi:]
    new_text = drop_text("\n".join(rows), [d for d in m["drop"] if d["module"] == src_stem(m)], src_name(m))
    tree = ast.parse(new_text)
    have = {ast.dump(s) for s in tree.body}
    wanted = list(m["imports"]) + (["import functools"] if b.src_functools else []) + [ast.unparse(import_stmt(m, x)) for x in b.src_mods]
    imps = [x for x in wanted if ast.dump(ast.parse(x).body[0]) not in have]
    if imps:
        rows = new_text.split("\n")
        at = last_import_row(tree)
        rows[at:at] = imps
        new_text = "\n".join(rows)
    path.write_text(new_text, encoding="utf-8")
    for mod, p in plans.items():
        (pkg_dir / f"{mod}.py").write_text(sib_write(m, p, [made[k] for k in b.order if k[0] == mod]), encoding="utf-8")
    return []


# --------------------------------------------------------------------------------------------------
# drop: delete defs, imports and __all__ entries, with the call sites that used them replaced
# --------------------------------------------------------------------------------------------------

DROP_KINDS = ("def", "import", "all", "replace")


def check_drop_items(path: Path, items: Any) -> None:
    if not isinstance(items, list):
        raise SplitError(f"{path}: drop must be a list")
    for it in items:
        kinds = [k for k in DROP_KINDS if isinstance(it, dict) and k in it]
        extra = set(it) - {"module", "all_add", *DROP_KINDS} if isinstance(it, dict) else {"?"}
        if not isinstance(it, dict) or len(kinds) != 1 or extra or not isinstance(it.get("module"), str) or not it["module"]:
            raise SplitError(f"{path}: each drop item is {{'module': name, one of {', '.join(DROP_KINDS)}}} (all may add 'all_add'): {it!r}")
        k = kinds[0]
        if "all_add" in it and k != "all":
            raise SplitError(f"{path}: drop all_add goes with all")
        ok = (
            isinstance(it[k], str)
            if k in ("def", "import")
            else all(isinstance(x, str) for x in it[k])
            if k == "all" and isinstance(it[k], list) and all(isinstance(x, str) for x in it.get("all_add", []))
            else k == "replace" and isinstance(it[k], dict) and set(it[k]) == {"in", "old", "new"} and all(isinstance(v, str) for v in it[k].values())
        )
        if not ok:
            raise SplitError(f"{path}: drop {k} item has the wrong shape: {it!r}")
        if k == "replace":
            for v in (it[k]["old"], it[k]["new"]):
                try:
                    ast.parse(v, mode="eval")
                except SyntaxError as e:
                    raise SplitError(f"{path}: drop replace {v!r} is not an expression: {e}") from e


class Swap(ast.NodeTransformer):
    """Replace every outermost expression equal to ``old`` with ``new``."""

    def __init__(self, old: str, new: ast.expr) -> None:
        self.old, self.new, self.hits = old, new, 0

    def visit(self, node: ast.AST) -> Any:
        if isinstance(node, ast.expr) and ast.dump(node) == self.old:
            self.hits += 1
            return clone(self.new)
        return super().visit(node)


def all_stmt(body: list[ast.stmt]) -> int | None:
    return next(
        (i for i, s in enumerate(body) if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) and s.targets[0].id == "__all__" and isinstance(s.value, (ast.List, ast.Tuple))),
        None,
    )


def all_values(s: ast.stmt) -> list[str]:
    return [e.value for e in s.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]  # type: ignore[attr-defined]


def new_all(old: list[str], it: dict) -> list[str]:
    gone = [x for x in it["all"] if x not in old]
    if gone:
        raise SplitError(f"{it['module']}: __all__ has no entry {gone}")
    return [x for x in old if x not in it["all"]] + [x for x in it.get("all_add", []) if x not in old]


def drop_apply(body: list[ast.stmt], items: list[dict], label: str) -> list[ast.stmt]:
    """The module body after the drop items, as an AST (the text side is ``drop_text``)."""
    body = list(body)
    for it in items:
        if "replace" in it:
            r = it["replace"]
            i = next((i for i, s in enumerate(body) if isinstance(s, DEFS) and s.name == r["in"]), None)
            if i is None:
                raise SplitError(f"{label}: replace: no top-level function {r['in']}")
            sw = Swap(ast.dump(parse_expr(r["old"])), parse_expr(r["new"]))
            body[i] = sw.visit(clone(body[i]))
            if not sw.hits:
                raise SplitError(f"{label}: replace: {r['old']!r} does not occur in {r['in']}")
    for it in items:
        if "def" in it:
            hits = [i for i, s in enumerate(body) if isinstance(s, DEFS + (ast.ClassDef,)) and s.name == it["def"]]
            if len(hits) != 1:
                raise SplitError(f"{label}: drop def: expected one top-level {it['def']}, found {len(hits)}")
            del body[hits[0]]
        elif "import" in it:
            hit = next((i for i, s in enumerate(body) if is_top_import(s) and any(_bound(a) == it["import"] for a in s.names)), None)  # type: ignore[attr-defined]
            if hit is None:
                raise SplitError(f"{label}: drop import: no top-level import binds {it['import']}")
            s = body[hit]
            keep = [a for a in s.names if _bound(a) != it["import"]]  # type: ignore[attr-defined]
            if keep:
                s = copy.copy(s)
                s.names = keep  # type: ignore[attr-defined]
                body[hit] = s
            else:
                del body[hit]
        elif "all" in it:
            i = all_stmt(body)
            if i is None:
                raise SplitError(f"{label}: drop all: no literal __all__")
            body[i] = ast.parse(all_text(new_all(all_values(body[i]), it))).body[0]
    return body


def drop_refs(body: list[ast.stmt], items: list[dict], label: str) -> None:
    """V15: a dropped def or import has no reference left, other than the declared replacements."""
    names = {it["def"] for it in items if "def" in it} | {it["import"] for it in items if "import" in it}
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, ast.Name) and n.id in names:
            raise SplitError(f"V15: {label}: {n.id} is dropped but still read at line {getattr(n, 'lineno', '?')}")
    i = all_stmt(body)
    left = sorted(names & set(all_values(body[i]))) if i is not None else []
    if left:
        raise SplitError(f"V15: {label}: __all__ still lists the dropped {left}")


def all_text(values: list[str]) -> str:
    one = "__all__ = [" + ", ".join(f'"{v}"' for v in values) + "]"
    return one if len(one) <= 120 else "__all__ = [\n" + "".join(f'    "{v}",\n' for v in values) + "]"


def stmt_text(s: ast.stmt) -> str:
    one = ast.unparse(s)
    if len(one) <= 120 or not isinstance(s, ast.ImportFrom):
        return one
    names = "".join(f"    {ast.unparse(a)},\n" for a in s.names)
    return f"from {'.' * s.level}{s.module or ''} import (\n{names})"


def drop_text(text: str, items: list[dict], label: str) -> str:
    """The text counterpart of ``drop_apply``."""
    for it in items:
        if "replace" in it:
            text = replace_text(text, it["replace"], label)
    for it in items:
        rows = text.split("\n")
        body = ast.parse(text).body
        if "def" in it:
            s = next(s for s in body if isinstance(s, DEFS + (ast.ClassDef,)) and s.name == it["def"])
            hi = s.end_lineno
            while hi < len(rows) and not rows[hi].strip():
                hi += 1
            del rows[stmt_start(s) - 1 : hi]
        elif "import" in it:
            s = next(s for s in body if is_top_import(s) and any(_bound(a) == it["import"] for a in s.names))  # type: ignore[attr-defined]
            keep = [a for a in s.names if _bound(a) != it["import"]]  # type: ignore[attr-defined]
            new = copy.copy(s)
            new.names = keep  # type: ignore[attr-defined]
            rows[s.lineno - 1 : s.end_lineno] = stmt_text(new).split("\n") if keep else []
        elif "all" in it:
            s = body[all_stmt(body)]  # type: ignore[index]
            rows[s.lineno - 1 : s.end_lineno] = all_text(new_all(all_values(s), it)).split("\n")
        text = "\n".join(rows)
    return text


def replace_text(text: str, r: dict, label: str) -> str:
    old = ast.dump(parse_expr(r["old"]))
    fn = next((s for s in ast.parse(text).body if isinstance(s, DEFS) and s.name == r["in"]), None)
    if fn is None:
        raise SplitError(f"{label}: replace: no top-level function {r['in']}")
    raw = text.encode("utf-8")
    starts = [0]
    for ln in raw.split(b"\n"):
        starts.append(starts[-1] + len(ln) + 1)
    hits: list[tuple[int, int]] = []

    def walk(n: ast.AST) -> None:
        if isinstance(n, ast.expr) and ast.dump(n) == old:
            hits.append((starts[n.lineno - 1] + n.col_offset, starts[n.end_lineno - 1] + n.end_col_offset))
            return
        for c in ast.iter_child_nodes(n):
            walk(c)

    walk(fn)
    if not hits:
        raise SplitError(f"{label}: replace: {r['old']!r} does not occur in {r['in']}")
    for a, b in sorted(hits, reverse=True):
        raw = raw[:a] + r["new"].encode("utf-8") + raw[b:]
    return raw.decode("utf-8")


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
    src = mk_src(text, m)
    out: list[str] = []
    for name in hoist:
        d = next((s for blk in own_blocks(src.fn) for s in blk if isinstance(s, DEFS) and s.name == name), None)
        if d is None:
            raise SplitError(f"no closure {name}")
        out.append(f"{name}: " + ", ".join(sorted(free_locals(d, src.key))))
    spans = spans_for_cells(m, src) if m["cells"] else []
    moves: list[Move] = []
    if m["moves"]:
        moves = plan_moves(src, m["moves"], text.split("\n"))
        spans += [(mv.name, mv.members) for mv in sorted(moves, key=lambda x: x.a - x.b)]
    if spans:
        lifted = tuple(m["lift"])
        args = {mv.name: mv.args for mv in moves} if m["moves"] else {}
        for name, why in sorted(Cells(src.fn, src.key, spans, tuple(ctx_names(m)) + lifted, args, strict=is_sibling(m)).crossing().items()):
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
