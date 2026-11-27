"""Guard: LLM call sites that do not name their moderation surface do not grow (TASK-13950.27).

Moderation (TASK-13950.27) applies at core/LLM_Calls/gateway.py, keyed by the ``llm_surface`` keyword.
A call that omits it, and a gateway function passed around as a plain reference, can reach a provider
unmoderated. This freezes, per module, the count of:
  1. calls to a gateway function (GATEWAY_NAMES or an import alias of one), bare or attribute
     (``cs.perform_chat_api_call(...)``), without a surfaced ``llm_surface=`` keyword;
  2. ``partial``/``functools.partial`` of one without a surfaced ``llm_surface=``;
  3. plain Load references to one (an injected default, a function passed as an argument), except operands
     of ``is``/``is not``/``isinstance`` and a module-level ``Name = <gateway name>`` alias;
plus the .13 direct-adapter references (``_references``).

Surfaced means the keyword's value is a string literal, or a Name that is a parameter of an enclosing
function (forwarding). A module-level constant is not surfaced and a ``**splat`` is not a surface.
chat_orchestrator.py and gateway.py (the definitions) and in-app ``tests`` packages are skipped.

Lock gains in with ``python -m tldw_Server_API.tests.lint.test_unmoderated_llm_calls_guard --tighten``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint import _ceilings
from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, parse_tree
from tldw_Server_API.tests.lint.test_direct_llm_adapter_calls_ratchet import _references

pytestmark = pytest.mark.unit

GATEWAY_NAMES = frozenset(
    {"perform_chat_api_call", "perform_chat_api_call_async", "chat_api_call", "chat_api_call_async"}
)
EXEMPT: frozenset[str] = frozenset({"core/Chat/chat_orchestrator.py", "core/LLM_Calls/gateway.py"})
_SURFACE = "llm_surface"

RATCHET = _ceilings.Ratchet(
    baseline=Path(__file__).resolve().with_name("unmoderated_llm_calls_baseline.json"),
    command="python -m tldw_Server_API.tests.lint.test_unmoderated_llm_calls_guard",
    fix=(
        'Pass llm_surface="<surface>" (a literal, or a parameter you forward) to the gateway call, '
        "or call it through a moderated surface."
    ),
    stale="The module no longer has an unmoderated LLM call site.",
)


# --- The counter --------------------------------------------------------------------------------


def _gateway_names(tree: ast.Module) -> set[str]:
    names = set(GATEWAY_NAMES)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname and alias.name in GATEWAY_NAMES:
                    names.add(alias.asname)
    return names


def _is_partial(call: ast.Call) -> bool:
    func = call.func
    return (isinstance(func, ast.Name) and func.id == "partial") or (
        isinstance(func, ast.Attribute) and func.attr == "partial"
    )


def _params(func: ast.AST) -> set[str]:
    args = func.args  # type: ignore[attr-defined]
    return {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}


def _surfaced(call: ast.Call, params: set[str]) -> bool:
    for kw in call.keywords:
        if kw.arg != _SURFACE:
            continue
        value = kw.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return True
        if isinstance(value, ast.Name) and value.id in params:
            return True
    return False


def unsurfaced_gateway_references(tree: ast.Module) -> int:
    """Unsurfaced gateway calls and partials, plus plain gateway references, in one module."""
    names = _gateway_names(tree)
    count = 0

    def counts(node: ast.AST, parent: ast.AST | None, params: set[str]) -> int:
        if isinstance(parent, ast.Call):
            if parent.func is node:
                return 0 if _surfaced(parent, params) else 1
            if _is_partial(parent) and parent.args and parent.args[0] is node:
                return 0 if _surfaced(parent, params) else 1
            if isinstance(parent.func, ast.Name) and parent.func.id == "isinstance" and node in parent.args:
                return 0
        elif isinstance(parent, ast.Compare):
            if all(isinstance(op, (ast.Is, ast.IsNot)) for op in parent.ops):
                return 0
        elif isinstance(parent, ast.Assign) and parent in tree.body and parent.value is node:
            return 0  # module-level identity alias
        return 1

    def visit(node: ast.AST, parent: ast.AST | None, params: set[str]) -> None:
        nonlocal count
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            params = params | _params(node)
        is_name = isinstance(node, ast.Name) and node.id in names
        is_attr = isinstance(node, ast.Attribute) and node.attr in names
        is_ref = (is_name or is_attr) and isinstance(node.ctx, ast.Load)  # type: ignore[attr-defined]
        if is_ref:
            count += counts(node, parent, params)
        for child in ast.iter_child_nodes(node):
            if is_ref and is_attr and child is node.value:  # type: ignore[attr-defined]
                continue  # ``cs`` in ``cs.perform_chat_api_call`` is not a reference
            visit(child, node, params)

    visit(tree, None, set())
    return count


def count_module(tree: ast.Module) -> int:
    return unsurfaced_gateway_references(tree) + _references(tree)


def measure(root: Path = APP_ROOT) -> dict[str, dict[str, int]]:
    """``{app-relative module: {"references": n}}`` for every module with an unmoderated site."""
    root = root.resolve()
    found = {}
    for path, tree in parse_tree(root).items():
        relative = path.relative_to(root)
        key = relative.as_posix()
        if "tests" in relative.parts[:-1] or key in EXEMPT:
            continue
        if count := count_module(tree):
            found[key] = {"references": count}
    return found


# --- Self-tests ---------------------------------------------------------------------------------


def _n(source: str) -> int:
    return unsurfaced_gateway_references(ast.parse(source))


def test_bare_call_without_surface_counts() -> None:
    assert _n('perform_chat_api_call(api_endpoint="x")') == 1


def test_attribute_call_without_surface_counts() -> None:
    assert _n("cs.perform_chat_api_call_async(x)") == 1


def test_call_with_literal_surface_does_not_count() -> None:
    assert _n('perform_chat_api_call(x, llm_surface="chat.completions")') == 0


def test_forwarded_parameter_surface_does_not_count() -> None:
    assert _n("def w(*, llm_surface):\n    perform_chat_api_call(llm_surface=llm_surface)") == 0


def test_non_parameter_name_surface_counts() -> None:
    assert _n("perform_chat_api_call(llm_surface=SOME_GLOBAL)") == 1


def test_splat_alone_counts() -> None:
    assert _n("perform_chat_api_call(**args)") == 1


def test_partial_without_and_with_surface() -> None:
    assert _n("partial(perform_chat_api_call, **a)") == 1
    assert _n('functools.partial(cs.perform_chat_api_call, **a, llm_surface="x")') == 0


def test_import_alias_call_counts() -> None:
    assert _n("from m import perform_chat_api_call as call_llm\ncall_llm(x)") == 1


def test_injected_default_reference_counts() -> None:
    assert _n("def f(fn=perform_chat_api_call_async): ...") == 1


def test_identity_comparison_does_not_count() -> None:
    assert _n("x is perform_chat_api_call") == 0
    assert _n("ORIG = perform_chat_api_call\nx is not cs.perform_chat_api_call") == 0


def test_direct_adapter_reference_counts() -> None:
    assert count_module(ast.parse("get_registry().get_adapter(p).chat(r)")) == 1


def test_every_surface_literal_is_registered() -> None:
    from tldw_Server_API.app.core.Moderation.surfaces import KNOWN_SURFACES

    problems = []
    for path, tree in parse_tree(APP_ROOT.resolve()).items():
        relative = path.relative_to(APP_ROOT.resolve())
        if "tests" in relative.parts[:-1]:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != _SURFACE:
                    continue
                v = kw.value
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    if v.value not in KNOWN_SURFACES:
                        problems.append(f"{relative.as_posix()}:{node.lineno}: unknown surface {v.value!r}")
                elif not isinstance(v, ast.Name):
                    problems.append(f"{relative.as_posix()}:{node.lineno}: llm_surface must be a literal or a forwarded parameter")
    assert problems == [], "\n".join(problems)


# --- The ratchet --------------------------------------------------------------------------------


def test_unmoderated_llm_calls_do_not_grow() -> None:
    measured, baseline = measure(), _ceilings.load(RATCHET.baseline)

    problems = _ceilings.violations(RATCHET, measured, baseline)

    assert problems == [], "\n".join(problems)
    _ceilings.warn_if_tightenable(RATCHET, measured, baseline)


def test_raised_unmoderated_call_ceilings_carry_a_new_reason() -> None:
    problems = _ceilings.reasonless_bumps(RATCHET, measure())
    assert problems == [], "\n".join(problems)


if __name__ == "__main__":
    raise SystemExit(_ceilings.main(RATCHET, measure))
