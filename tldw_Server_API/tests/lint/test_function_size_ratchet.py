"""Ratchet: no backend function or route handler grows past its size ceiling (TASK-13950.8).

Nothing limited size, and the hotspots kept growing (chat_service.execute_streaming_call
1,528 lines, persona_stream 3,255). This freezes every function in tldw_Server_API/app
over THRESHOLD lines at its current size. function_size_baseline.json holds one ceiling
per function, keyed ``"<module>:<qualname>"`` -- never by line number, so edits elsewhere
in a file cannot fail it.

How a function is measured (the only definition; the counter below implements it):
  * lines  = ``end_lineno - lineno + 1`` of the ``def``: from the ``def`` line to the
    function's last line. Decorators are excluded; the docstring, comments and blank
    lines inside the body, and any nested functions, are included.
  * params = every parameter: positional-only, positional, ``*args``, keyword-only and
    ``**kwargs``, ``self``/``cls`` included.
  * Keys: ``module`` is the dotted path (``tldw_Server_API.app.core.Chat.chat_service``;
    a package's ``__init__.py`` is the package). ``qualname`` joins the enclosing classes
    and functions with dots: ``Class.method``, ``outer.inner`` (no ``<locals>``).
    ``if``/``try``/``with``/loop blocks add nothing to the qualname.
  * One module defining the same qualname more than once (a conditional def, an
    ``@overload``, a property setter) yields one entry holding the max lines and the max
    params over those defs, so a redefinition cannot hide growth past the ceiling.
    Known limit: a duplicate below the max may grow up to the max unnoticed.

A route handler is a function anywhere in tldw_Server_API/app/ (not only app/api/: a
core WebSocket server mounts its own router) decorated with ``@<anything>.<verb>(...)`` for
a verb in ROUTE_DECORATORS, or named as the endpoint of a module-level
``<router>.add_api_route(path, <name>)``-style call. Handlers share the
function baseline (one entry, one reason, one bump): the handler test only re-reports
the handler subset with a handler-specific fix, and the function test is the backstop for
any registration shape the handler matcher does not know.

Failures and fixes:
  * grew past its ceiling, or a new function over THRESHOLD lines: split it. If the
    growth is unavoidable (a security fix in a frozen function), raise the ceiling in the
    baseline with a "reason"; the bump is reviewed in the PR (see _ceilings.py).
  * stale entry (the function is gone, or now at or under THRESHOLD lines): delete it, or
    run ``python -m tldw_Server_API.tests.lint.test_function_size_ratchet --tighten``.
  * Moving, renaming or hoisting a frozen function re-keys it, and so does following the
    handler fix ("move the logic into a core/service function") without first shrinking the
    logic below THRESHOLD. The old entry goes stale and the new key has no ceiling: delete
    the old entry and add the new one with reason ``moved from <old key> (<PR>)``, filled in.
  * A nested function's lines count toward its parent too, so growing a nested function
    fails both its own entry and its parent's; bump both.
Shrinking needs no edit: the numbers are ceilings. ``--tighten`` lowers them;
``--write-baseline`` records everything as measured (keeping reasons).
"""

from __future__ import annotations

import ast
import subprocess
import warnings
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint import _ceilings
from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, REPO_ROOT, parse_tree

pytestmark = pytest.mark.unit

THRESHOLD = 300
ROUTE_DECORATORS = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
    | {"websocket", "api_route", "route", "websocket_route"}
)
ROUTE_REGISTRARS = frozenset({"add_api_route", "add_api_websocket_route", "add_route", "add_websocket_route"})
COMMAND = "python -m tldw_Server_API.tests.lint.test_function_size_ratchet"

FUNCTIONS = _ceilings.Ratchet(
    baseline=Path(__file__).resolve().with_name("function_size_baseline.json"),
    command=COMMAND,
    fix=f"Split it until it is at or under the ceiling (new functions: at or under {THRESHOLD} lines).",
    stale=f"The function is gone or now at or under {THRESHOLD} lines.",
    new=(
        "Moved or renamed an existing function? Delete its old entry and add this one with reason "
        '"moved from <old key> (<PR>)", filled in.'
    ),
)
HANDLERS = FUNCTIONS._replace(
    fix="Route handler: move the logic into a core/service function the handler calls.",
)


# --- The counter --------------------------------------------------------------------------------

_Function = ast.FunctionDef | ast.AsyncFunctionDef


def _defs(tree: ast.Module) -> Iterator[tuple[str, _Function]]:
    """Every function in ``tree`` with its qualname. Only statements can hold a ``def``."""
    stack: list[tuple[ast.AST, str]] = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield prefix + child.name, child
                stack.append((child, f"{prefix}{child.name}."))
            elif isinstance(child, ast.ClassDef):
                stack.append((child, f"{prefix}{child.name}."))
            elif isinstance(child, (ast.stmt, ast.excepthandler, ast.match_case)):
                stack.append((child, prefix))


def _params(fn: _Function) -> int:
    args = fn.args
    return len(args.posonlyargs + args.args + args.kwonlyargs) + (args.vararg is not None) + (args.kwarg is not None)


def _registered_endpoints(tree: ast.Module) -> set[str]:
    """Names passed as the endpoint of a module-level ``<router>.add_api_route(path, <name>)``."""
    names: set[str] = set()
    for statement in tree.body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in ROUTE_REGISTRARS:
            endpoints = call.args[1:2] + [k.value for k in call.keywords if k.arg == "endpoint"]
            names.update(e.id for e in endpoints if isinstance(e, ast.Name))
    return names


def _is_route_decorator(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr in ROUTE_DECORATORS
    )


@cache
def measure(repo_root: Path = REPO_ROOT) -> tuple[dict[str, dict[str, int]], frozenset[str]]:
    """(every function over THRESHOLD lines -> {"lines", "params"}, the key of every route handler).

    Cached per root, like the parse it reads; callers must not mutate the result.
    """
    functions: dict[str, dict[str, int]] = {}
    handlers: set[str] = set()
    for path, tree in parse_tree(repo_root / "tldw_Server_API" / "app").items():
        parts = path.relative_to(repo_root.resolve()).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        endpoints = _registered_endpoints(tree)
        for qualname, fn in _defs(tree):
            key = f"{module}:{qualname}"
            if qualname in endpoints or any(map(_is_route_decorator, fn.decorator_list)):
                handlers.add(key)
            lines = (fn.end_lineno or fn.lineno) - fn.lineno + 1  # end_lineno is always set by ast.parse
            if lines > THRESHOLD:
                seen = functions.setdefault(key, {"lines": 0, "params": 0})
                seen["lines"], seen["params"] = max(seen["lines"], lines), max(seen["params"], _params(fn))
    return functions, frozenset(handlers)


def handler_subset(entries: dict, handlers: frozenset[str]) -> dict:
    return {key: value for key, value in entries.items() if key in handlers}


# --- Self-tests: each seeds a tree under a fresh tmp root and runs the real scan ---------------


def _function(name: str, lines: int, params: str = "", indent: str = "") -> str:
    """Source of a ``def`` exactly ``lines`` long by the counting rule."""
    body = [f"{indent}    x = {i}" for i in range(lines - 1)]
    return "\n".join([f"{indent}def {name}({params}):", *body]) + "\n"


def _seed(root: Path, files: dict[str, str]) -> Path:
    for relative, source in files.items():
        path = root / "tldw_Server_API" / "app" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def _ratchet(tmp_path: Path, entries: dict) -> _ceilings.Ratchet:
    baseline = tmp_path / "function_size_baseline.json"
    _ceilings.dump(baseline, entries)
    return FUNCTIONS._replace(baseline=baseline)


def test_a_301_line_function_without_a_ceiling_fails(tmp_path: Path) -> None:
    root = _seed(tmp_path / "tree", {"core/big.py": _function("at_limit", 300) + _function("over", 301, "a, b")})
    functions, _ = measure(root)
    ratchet = _ratchet(tmp_path, {})

    problems = _ceilings.violations(ratchet, functions, _ceilings.load(ratchet.baseline))

    assert functions == {"tldw_Server_API.app.core.big:over": {"lines": 301, "params": 2}}
    assert len(problems) == 1 and problems[0].startswith("tldw_Server_API.app.core.big:over: lines 301"), problems
    assert 'Moved or renamed an existing function? Delete its old entry and add this one with reason "moved from' in (
        problems[0]
    )


def test_one_line_or_parameter_over_the_ceiling_fails_and_shrinking_passes(tmp_path: Path) -> None:
    entries = {"tldw_Server_API.app.core.big:f": {"lines": 400, "params": 2, "reason": None}}
    ratchet = _ratchet(tmp_path, entries)

    def problems(lines: int, params: str, name: str) -> list[str]:
        functions, _ = measure(_seed(tmp_path / name, {"core/big.py": _function("f", lines, params)}))
        return _ceilings.violations(ratchet, functions, entries)

    assert problems(400, "a, b", "at_ceiling") == []
    assert problems(399, "a", "shrunk") == []  # ceilings: no baseline edit needed
    (grown,) = problems(401, "a, b", "one_more_line")
    assert "lines 401 > ceiling 400" in grown
    # The message names the baseline file and the exact entry to set, reviewed in the PR.
    assert str(ratchet.baseline) in grown and "reviewed in the PR" in grown
    assert '"tldw_Server_API.app.core.big:f": {"lines": 401, "params": 2, "reason": "<why' in grown
    (more_params,) = problems(400, "a, b, *args", "one_more_param")
    assert "params 3 > ceiling 2" in more_params


def test_entries_go_stale_when_the_function_is_gone_or_back_under_the_threshold(tmp_path: Path) -> None:
    entries = {
        "tldw_Server_API.app.core.big:gone": {"lines": 400, "params": 0, "reason": None},
        "tldw_Server_API.app.core.big:small": {"lines": 400, "params": 0, "reason": None},
        "tldw_Server_API.app.core.big:kept": {"lines": 400, "params": 0, "reason": None},
    }
    root = _seed(tmp_path / "tree", {"core/big.py": _function("small", THRESHOLD) + _function("kept", 301)})
    functions, _ = measure(root)

    problems = _ceilings.violations(_ratchet(tmp_path, entries), functions, entries)

    assert [p.split(":")[1] for p in problems] == ["gone", "small"]
    assert all("stale entry" in p and "--tighten" in p for p in problems), problems


def test_nested_functions_and_methods_use_qualnames_and_a_redefinition_keeps_the_max(tmp_path: Path) -> None:
    source = (
        "class Store:\n"
        + _function("method", 301, "self", indent="    ")
        + "def outer():\n"
        + "    if True:\n"
        + _function("inner", 302, indent="        ")
        + "try:\n"
        + _function("twice", 310, "a", indent="    ")
        + "except ImportError:\n"
        + _function("twice", 305, "a, b, c", indent="    ")
    )
    functions, _ = measure(_seed(tmp_path / "tree", {"core/pkg/__init__.py": source}))

    assert functions == {
        "tldw_Server_API.app.core.pkg:Store.method": {"lines": 301, "params": 1},
        "tldw_Server_API.app.core.pkg:outer": {"lines": 304, "params": 0},
        "tldw_Server_API.app.core.pkg:outer.inner": {"lines": 302, "params": 0},
        "tldw_Server_API.app.core.pkg:twice": {"lines": 310, "params": 3},
    }


def test_route_handlers_are_found_by_decorator_or_registration_anywhere_in_the_app(tmp_path: Path) -> None:
    endpoint = (
        "@router.post('/a')\n"
        + _function("decorated", 301)
        + "@rules_router.websocket('/ws')\n"
        + _function("socket", 10)
        + "@functools.cache\n"
        + _function("cached", 301)
        + _function("registered", 5)
        + "router.add_api_route('/r', registered, methods=['POST'])\n"
        + _function("by_keyword", 5)
        + "router.add_api_route('/k', endpoint=by_keyword)\n"
    )
    core_ws = "@app.websocket('/x')\n" + _function("core_ws", 301)  # a core server mounting its own route
    root = _seed(tmp_path / "tree", {"api/v1/endpoints/things.py": endpoint, "core/ws_server.py": core_ws})
    functions, handlers = measure(root)
    module = "tldw_Server_API.app.api.v1.endpoints.things"

    assert handlers == {f"{module}:{name}" for name in ("decorated", "socket", "registered", "by_keyword")} | {
        "tldw_Server_API.app.core.ws_server:core_ws"
    }
    problems = _ceilings.violations(HANDLERS, handler_subset(functions, handlers), {})
    assert [p.split(": lines 301")[0] for p in problems] == [
        f"{module}:decorated",
        "tldw_Server_API.app.core.ws_server:core_ws",
    ]
    assert all("Route handler" in p for p in problems)


def test_a_sub_root_of_the_app_shares_the_app_parse() -> None:
    """Minor: a narrower root must not cache a second copy of the tree."""
    chacha = parse_tree(APP_ROOT / "core" / "DB_Management" / "chacha")
    app = parse_tree(APP_ROOT)

    assert chacha and all(path.is_relative_to(APP_ROOT / "core" / "DB_Management" / "chacha") for path in chacha)
    assert all(tree is app[path] for path, tree in chacha.items())


def test_a_raised_ceiling_needs_a_new_reason(tmp_path: Path) -> None:
    ratchet = _ratchet(tmp_path, {})
    base: _ceilings.Baseline = {
        "m:quiet": {"lines": 400, "params": 2, "reason": None},
        "m:bumped": {"lines": 400, "params": 2, "reason": "TASK-1 fix"},
    }

    def problems(entries: dict) -> list[str]:
        return _ceilings.unreasoned_raises(ratchet, {**base, **entries}, base)

    assert problems({}) == []
    assert problems({"m:quiet": {"lines": 390, "params": 2, "reason": None}}) == []  # tightened
    (no_reason,) = problems({"m:quiet": {"lines": 401, "params": 2, "reason": None}})
    assert no_reason.startswith("m:quiet: raises lines") and "reviewed in the PR" in no_reason
    assert problems({"m:quiet": {"lines": 400, "params": 3, "reason": "  "}})  # blank is no reason
    assert problems({"m:bumped": {"lines": 401, "params": 2, "reason": "TASK-1 fix"}})  # reused reason
    assert problems({"m:bumped": {"lines": 401, "params": 2, "reason": "TASK-2 fix"}}) == []
    assert problems({"m:new": {"lines": 301, "params": 0, "reason": None}})  # a new entry is a raise
    assert problems({"m:new": {"lines": 301, "params": 0, "reason": "TASK-3"}}) == []
    # The failure messages hand out templates; pasting one unfilled is no reason.
    assert problems({"m:quiet": {"lines": 401, "params": 2, "reason": _ceilings.REASON_PLACEHOLDER}})
    assert problems({"m:new": {"lines": 301, "params": 0, "reason": "moved from <old key> (<PR>)"}})
    assert problems({"m:new": {"lines": 301, "params": 0, "reason": "moved from m:old (#3340)"}}) == []
    assert _ceilings.unreasoned_raises(ratchet, {"m:new": {"lines": 301, "params": 0, "reason": None}}, None) == []


def test_a_bump_may_not_reserve_headroom(tmp_path: Path) -> None:
    ratchet = _ratchet(tmp_path, {})
    base: _ceilings.Baseline = {"m:f": {"lines": 400, "params": 2, "reason": None}}
    bumped = {"m:f": {"lines": 450, "params": 2, "reason": "TASK-1 fix"}}

    def problems(lines: int) -> list[str]:
        return _ceilings.unreasoned_raises(ratchet, bumped, base, {"m:f": {"lines": lines, "params": 2}})

    assert problems(450) == []
    (headroom,) = problems(410)
    assert headroom.startswith("m:f: raises lines to 450, above the code's 410") and '"lines": 410' in headroom
    # An unraised ceiling above the code is just a ceiling: shrinking needs no edit.
    assert _ceilings.unreasoned_raises(ratchet, base, base, {"m:f": {"lines": 350, "params": 1}}) == []


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_the_reason_check_reads_the_baseline_at_the_merge_base(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    baseline = repo / "baseline.json"
    _git(repo, "init", "-q", "-b", "dev")
    (repo / "README").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "before the baseline")
    _git(repo, "checkout", "-q", "-b", "adds-baseline")
    _ceilings.dump(baseline, {"m:f": {"lines": 400, "params": 1, "reason": None}})
    _git(repo, "add", "baseline.json")
    _git(repo, "commit", "-q", "-m", "baseline")

    assert _ceilings.base_version(baseline, "dev", repo) is None  # the PR that adds it needs no reasons

    _git(repo, "checkout", "-q", "-b", "bump")
    _ceilings.dump(baseline, {"m:f": {"lines": 401, "params": 1, "reason": None}})
    _git(repo, "commit", "-q", "-am", "bump without a reason")
    base = _ceilings.base_version(baseline, "adds-baseline", repo)

    assert base == {"m:f": {"lines": 400, "params": 1, "reason": None}}
    assert _ceilings.unreasoned_raises(FUNCTIONS._replace(baseline=baseline), _ceilings.load(baseline), base)


def test_tighten_lowers_and_drops_but_never_raises_and_write_baseline_keeps_reasons(tmp_path: Path) -> None:
    ratchet = _ratchet(
        tmp_path,
        {
            "m:shrunk": {"lines": 400, "params": 3, "reason": "TASK-1"},
            "m:grew": {"lines": 400, "params": 3, "reason": None},
            "m:gone": {"lines": 400, "params": 3, "reason": None},
        },
    )
    measured = {
        "m:shrunk": {"lines": 350, "params": 2},
        "m:grew": {"lines": 410, "params": 3},
        "m:new": {"lines": 320, "params": 1},
    }

    seeded = _ceilings.load(ratchet.baseline)
    with pytest.warns(UserWarning, match="--tighten"):  # shrinking prompts a tighten, never fails
        _ceilings.warn_if_tightenable(ratchet, {"m:shrunk": measured["m:shrunk"]}, seeded)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ceilings.warn_if_tightenable(ratchet, {"m:grew": measured["m:grew"]}, seeded)
    assert _ceilings.main(ratchet, lambda: measured, ["--tighten"]) == 0
    assert _ceilings.load(ratchet.baseline) == {
        "m:shrunk": {"lines": 350, "params": 2, "reason": "TASK-1"},
        "m:grew": {"lines": 400, "params": 3, "reason": None},
    }
    assert _ceilings.main(ratchet, lambda: measured, ["--write-baseline"]) == 0
    assert _ceilings.load(ratchet.baseline) == {
        "m:shrunk": {"lines": 350, "params": 2, "reason": "TASK-1"},
        "m:grew": {"lines": 410, "params": 3, "reason": None},
        "m:new": {"lines": 320, "params": 1, "reason": None},
    }
    assert ratchet.baseline.read_text(encoding="utf-8").count("\n") == 5  # one entry per line


# --- The ratchets -------------------------------------------------------------------------------


def test_functions_stay_under_their_ceilings() -> None:
    functions, _ = measure()
    baseline = _ceilings.load(FUNCTIONS.baseline)

    problems = _ceilings.violations(FUNCTIONS, functions, baseline)

    assert not problems, "\n".join(problems)
    _ceilings.warn_if_tightenable(FUNCTIONS, functions, baseline)


def test_route_handlers_stay_under_their_ceilings() -> None:
    functions, handlers = measure()
    baseline = _ceilings.load(FUNCTIONS.baseline)

    problems = _ceilings.violations(HANDLERS, handler_subset(functions, handlers), handler_subset(baseline, handlers))

    assert not problems, "\n".join(problems)


def test_raised_function_ceilings_carry_a_new_reason() -> None:
    problems = _ceilings.reasonless_bumps(FUNCTIONS, measure()[0])
    assert not problems, "\n".join(problems)


if __name__ == "__main__":
    raise SystemExit(_ceilings.main(FUNCTIONS, lambda: measure()[0]))
