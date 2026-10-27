"""Ratchet: direct LLM adapter calls do not grow (TASK-13950.13).

Every LLM provider call is meant to go through one module, core/LLM_Calls/gateway.py, so
that moderation (TASK-13950.27) has one pre-call and one post-call hook. A module that
fetches a chat adapter and calls it skips that gateway. This freezes those calls per
module in direct_llm_adapter_calls_baseline.json, each a ceiling ``{"references": int,
"reason": ...}``.

Counting. In each module under tldw_Server_API/app (modules in an in-app ``tests``
package directory are skipped, and so is every module in EXEMPT), count every attribute
reference ``<x>.<m>`` with ``m`` in METHODS where ``<x>`` is one of:
  * a call to a GETTERS name: ``get_adapter_or_raise(p).chat(...)``;
  * a local or module name bound from such a call (``adapter = registry.get_adapter(p)``);
  * ``self.<attr>`` bound from such a call.
References count, not only calls, so ``asyncio.to_thread(adapter.chat, payload)`` and
``partial(adapter.chat, ...)`` count. Names are matched per module, not per scope.

Known limits: an adapter passed in as a parameter is not seen, and neither is one bound
from ``await <getter>(...)`` or from a conditional expression around the call.

Growth fails: call the provider through the gateway (perform_chat_api_call or
perform_chat_api_call_async), not the adapter. If growth is unavoidable, raise the count
with a "reason" (reviewed in the PR; see _ceilings.py). A count below its ceiling passes;
lock the gain in with
``python -m tldw_Server_API.tests.lint.test_direct_llm_adapter_calls_ratchet --tighten``.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint import _ceilings
from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, parse_tree

pytestmark = pytest.mark.unit

EXEMPT: frozenset[str] = frozenset({"core/LLM_Calls/gateway.py"})
METHODS = frozenset({"chat", "achat", "stream", "astream"})
GETTERS = frozenset({"get_adapter", "get_adapter_or_raise", "_get_adapter"})

RATCHET = _ceilings.Ratchet(
    baseline=Path(__file__).resolve().with_name("direct_llm_adapter_calls_baseline.json"),
    command="python -m tldw_Server_API.tests.lint.test_direct_llm_adapter_calls_ratchet",
    fix=(
        "Call the provider through core/LLM_Calls/gateway.py (perform_chat_api_call or "
        "perform_chat_api_call_async), not the adapter."
    ),
    stale="The module no longer calls an adapter directly.",
)


# --- The counter --------------------------------------------------------------------------------


def _is_getter_call(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id in GETTERS) or (
        isinstance(func, ast.Attribute) and func.attr in GETTERS
    )


def _self_attr(node: ast.AST) -> str | None:
    """``x`` for ``self.x``, else None."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _references(tree: ast.Module) -> int:
    """Direct adapter references in one module; see the module docstring."""
    names: set[str] = set()
    attrs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_getter_call(node.value):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and _is_getter_call(node.value):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif (attr := _self_attr(target)) is not None:
                attrs.add(attr)

    def holds_adapter(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in names
        return _is_getter_call(node) or _self_attr(node) in attrs

    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in METHODS and holds_adapter(node.value)
    )


def measure(root: Path = APP_ROOT) -> dict[str, dict[str, int]]:
    """``{app-relative module: {"references": n}}`` for every module with a direct adapter reference."""
    root = root.resolve()
    found = {}
    for path, tree in parse_tree(root).items():
        relative = path.relative_to(root)
        key = relative.as_posix()
        if "tests" in relative.parts[:-1] or key in EXEMPT:
            continue  # an in-app test package, or the gateway itself
        if b"get_adapter" not in path.read_bytes():
            continue  # every GETTERS name contains it; skips walking modules that cannot match
        if count := _references(tree):
            found[key] = {"references": count}
    return found


# --- Self-tests: each seeds a tree under a fresh tmp root and runs the real scan ---------------


def _seed(root: Path, files: dict[str, str]) -> Path:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return root


def test_the_counter_counts_each_direct_adapter_shape(tmp_path: Path) -> None:
    root = _seed(
        tmp_path / "app",
        {
            "core/X/a.py": "def f(p, r):\n    return get_adapter_or_raise(p).chat(r)\n",
            "core/X/b.py": (
                "import asyncio\n"
                "async def f(registry, p, r):\n"
                "    adapter = registry.get_adapter(p)\n"
                "    await adapter.achat(r)\n"
                "    return await asyncio.to_thread(adapter.chat, r)\n"
            ),
            "core/X/c.py": (
                "class C:\n"
                "    def __init__(self, reg, p):\n"
                "        self._adapter = reg.get_adapter(p)\n"
                "    def go(self, r):\n"
                "        return self._adapter.stream(r)\n"
            ),
            "core/X/d.py": "def f(client, url):\n    return client.stream('GET', url)\n",
            "core/LLM_Calls/gateway.py": "def f(registry, p, r):\n    adapter = registry.get_adapter(p)\n    return adapter.chat(r)\n",
            "core/X/tests/test_x.py": "def f(p, r):\n    return get_adapter_or_raise(p).chat(r)\n",
        },
    )

    assert measure(root) == {
        "core/X/a.py": {"references": 1},
        "core/X/b.py": {"references": 2},
        "core/X/c.py": {"references": 1},
    }


def test_growth_a_new_module_and_a_stale_entry_fail_and_shrinking_passes(tmp_path: Path) -> None:
    root = _seed(
        tmp_path / "app",
        {
            "core/X/b.py": "def f(registry, p, r):\n    adapter = registry.get_adapter(p)\n    return adapter.chat(r), adapter.stream(r)\n",
            "core/X/new.py": "def f(p, r):\n    return get_adapter_or_raise(p).achat(r)\n",
        },
    )
    baseline = {
        "core/X/b.py": {"references": 1, "reason": None},
        "core/X/gone.py": {"references": 1, "reason": None},
    }

    grown, gone, new = _ceilings.violations(RATCHET, measure(root), baseline)

    assert grown.startswith("core/X/b.py: references 2 > ceiling 1")
    assert gone.startswith("core/X/gone.py: stale entry")
    assert new.startswith("core/X/new.py: references 1, and no ceiling")
    assert (
        _ceilings.violations(
            RATCHET,
            measure(root),
            {
                "core/X/b.py": {"references": 3, "reason": None},
                "core/X/new.py": {"references": 1, "reason": None},
            },
        )
        == []
    )


# --- The ratchet --------------------------------------------------------------------------------


def test_direct_llm_adapter_calls_do_not_grow() -> None:
    measured, baseline = measure(), _ceilings.load(RATCHET.baseline)

    problems = _ceilings.violations(RATCHET, measured, baseline)

    assert problems == [], "\n".join(problems)
    _ceilings.warn_if_tightenable(RATCHET, measured, baseline)


def test_raised_adapter_call_ceilings_carry_a_new_reason() -> None:
    problems = _ceilings.reasonless_bumps(RATCHET, measure())
    assert problems == [], "\n".join(problems)


if __name__ == "__main__":
    raise SystemExit(_ceilings.main(RATCHET, measure))
