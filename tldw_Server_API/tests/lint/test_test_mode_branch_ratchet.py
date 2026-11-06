"""Ratchet: test-mode branches in app code do not grow (TASK-13950.19).

App code must not behave differently under test. Every read of a test signal (TEST_MODE,
PYTEST_CURRENT_TEST, a TLDW_TEST_* variable, "pytest" in sys.modules, core.testing's
predicates) is a branch that production never runs. This freezes those references per
module in test_mode_branch_baseline.json, each a ceiling ``{"lines": int, "predicates":
int, "reason": ...}``.

Counting. In each module under tldw_Server_API/app (modules in an in-app ``tests``
package directory and EXEMPT are skipped), ``lines`` is the number of distinct lines
holding a reference: a string constant that is a signal (SIGNAL_KEYS, ``TLDW_TEST_*``,
a ``tldw_Server_API.tests...`` path), a name or attribute that reads like a test-mode
signal or is one of core.testing's predicates (also under an import alias), or the
``def`` of a function so named. Imports, comments and docstrings never count.
``predicates`` counts the private "is this a test" functions: the name has a test word,
it returns ``bool`` and its body holds a reference.

PREDICATE_EXEMPT modules implement ``MCPServer.dependencies.environment_flags_provider``,
which is an injected seam: their lines count and their predicates are reported as 0.

SECURITY_FUNCTIONS is a function-level zero pin. The module ratchet allows a bump with a
reason; this pin has no bump path. Removing an entry needs a reason in the PR description.
The pin sees only the function's own body; a bypass moved into a callee is still counted
by the module ratchet.

Growth fails: read test mode through core.testing and drive the behaviour from a fixture,
a dependency override or a constructor argument. Lock a gain in with
``python -m tldw_Server_API.tests.lint.test_test_mode_branch_ratchet --tighten``.
"""

from __future__ import annotations

import ast
import re
from functools import cache
import textwrap
from collections.abc import Iterable
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint import _ceilings
from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, parse_tree

pytestmark = pytest.mark.unit

SIGNAL_KEYS: frozenset[str] = frozenset({"TEST_MODE", "TESTING", "TLDW_TEST_MODE", "PYTEST_CURRENT_TEST", "pytest"})
CORE_TESTING_SIGNALS: frozenset[str] = frozenset(
    {"is_test_mode", "is_explicit_pytest_runtime", "is_test_context", "audio_imports_enabled_for_runtime"}
)
EXEMPT: frozenset[str] = frozenset({"core/testing.py"})
PREDICATE_EXEMPT: frozenset[str] = frozenset(
    {"core/MCP_unified/environment.py", "core/MCP_unified/adapters/tldw_runtime.py"}
)
SECURITY_FUNCTIONS: frozenset[str] = frozenset(
    {
        "api/v1/endpoints/evaluations/evaluations_auth.py:get_eval_request_user",
        "api/v1/endpoints/evaluations/evaluations_auth.py:verify_api_key",
        "core/AuthNZ/User_DB_Handling.py:get_request_user",
        "core/AuthNZ/csrf_protection.py:resolve_effective_csrf_enabled",
        "core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py:_raise_if_url_blocked",
        "core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py:process_single_video",
        "core/Ingestion_Media_Processing/Audio/Audio_Files.py:_validate_outbound_url",
        "core/Evaluations/webhook_manager.py:WebhookManager._deliver_webhook",
        "core/Utils/egress.py:evaluate_url_policy",
        "core/http_client.py:_validate_egress_or_raise",
        "services/workflows_webhook_dlq_service.py:_host_allowed",
        "api/v1/API_Deps/auth_deps.py:_authenticate_api_key_from_request",
        "api/v1/API_Deps/auth_deps.py:get_session_manager_dep",
        "api/v1/API_Deps/auth_deps.py:get_db_transaction",
        "api/v1/API_Deps/auth_deps.py:_enforce_auth_deps_ingress_guard",
        "api/v1/endpoints/mcp_unified_endpoint.py:_resolve_token_data_compat",
        "core/AuthNZ/auth_principal_resolver.py:get_auth_principal",
        "core/AuthNZ/initialize.py:bootstrap_single_user_profile",
        "core/Audit/unified_audit_service.py:UnifiedAuditService.__init__",
        "core/MCP_unified/server.py:MCPServer._single_user_compat_auth_via",
    }
)

_TEST_WORDS = frozenset({"test", "tests", "testing", "pytest"})
_MODE_WORDS = frozenset({"mode", "context", "runtime", "pytest"})
_TLDW_TEST = re.compile(r"TLDW_TEST_[A-Z0-9_]+")
# The pin's stricter matcher: the seam guard's constant pattern, and any test word in a name.
_STRICT_CONSTANT = re.compile(r"(?:^|_)(?:PY)?TEST(?:S|ING)?(?:_|$)")

RATCHET = _ceilings.Ratchet(
    baseline=Path(__file__).resolve().with_name("test_mode_branch_baseline.json"),
    command="python -m tldw_Server_API.tests.lint.test_test_mode_branch_ratchet",
    fix=(
        "Read test mode through core.testing and drive the behaviour from a fixture, a dependency "
        "override or a constructor argument instead of a new branch; private 'is this a test' "
        "predicates are deleted, not added (TASK-13950.19)."
    ),
    stale="The module no longer reads a test-mode signal.",
)


# --- The counter --------------------------------------------------------------------------------


def _words(name: str) -> set[str]:
    return set(name.lower().strip("_").split("_"))


# ponytail: counts a reference by its own line only, so a test-named local
# (``_is_test = is_test_mode()`` then ``if _is_test:``) counts the assignment, not the branch;
# upgrade to name-flow tracking if such locals appear.
def is_signal_name(name: str) -> bool:
    words = _words(name)
    return bool(words & _TEST_WORDS) and bool(words & _MODE_WORDS)


def is_signal_constant(value: object) -> bool:
    return isinstance(value, str) and (
        value in SIGNAL_KEYS or _TLDW_TEST.fullmatch(value) is not None or value.startswith("tldw_Server_API.tests")
    )


def _core_testing_aliases(tree: ast.Module) -> frozenset[str]:
    # Any module: absolute, relative (``from ..testing import``) or a re-export (MCP_unified.environment).
    return frozenset(
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name in CORE_TESTING_SIGNALS
    )


def _reference_lines(nodes: Iterable[ast.AST], aliases: frozenset[str], *, strict: bool = False) -> set[int]:
    lines: set[int] = set()
    for root in nodes:
        docstrings = {id(n.value) for n in ast.walk(root) if isinstance(n, ast.Expr)}
        for node in ast.walk(root):
            if isinstance(node, ast.Constant):
                hit = id(node) not in docstrings and (
                    is_signal_constant(node.value)
                    or (strict and isinstance(node.value, str) and _STRICT_CONSTANT.search(node.value) is not None)
                )
            elif isinstance(node, ast.Name):
                hit = is_signal_name(node.id) or node.id in CORE_TESTING_SIGNALS or node.id in aliases
                hit = hit or (strict and bool(_words(node.id) & _TEST_WORDS))
            elif isinstance(node, ast.Attribute):
                hit = is_signal_name(node.attr) or node.attr in CORE_TESTING_SIGNALS
                hit = hit or (strict and bool(_words(node.attr) & _TEST_WORDS))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                hit = is_signal_name(node.name)
            else:
                hit = False
            if hit:
                lines.add(node.lineno)
    return lines


def _is_predicate(node: ast.FunctionDef | ast.AsyncFunctionDef, aliases: frozenset[str]) -> bool:
    return (
        bool(_words(node.name) & _TEST_WORDS)
        # ponytail: exactly ``-> bool``; ``bool | None``, ``Optional[bool]`` and string annotations evade it.
        and isinstance(node.returns, ast.Name)
        and node.returns.id == "bool"
        and bool(_reference_lines(node.body, aliases))
    )


def count_module(tree: ast.Module, *, count_predicates: bool = True) -> dict[str, int]:
    aliases = _core_testing_aliases(tree)
    predicates = (
        sum(
            _is_predicate(node, aliases)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        if count_predicates
        else 0
    )
    return {"lines": len(_reference_lines([tree], aliases)), "predicates": predicates}


def measure(root: Path = APP_ROOT) -> dict[str, dict[str, int]]:
    """``{app-relative module: {"lines": n, "predicates": k}}`` for every module that reads a test signal."""
    root = root.resolve()
    found = {}
    for path, tree in parse_tree(root).items():
        relative = path.relative_to(root)
        key = relative.as_posix()
        if "tests" in relative.parts[:-1] or key in EXEMPT:
            continue
        counts = count_module(tree, count_predicates=key not in PREDICATE_EXEMPT)
        if counts["lines"] or counts["predicates"]:
            found[key] = counts
    return found


def function_reference_lines(tree: ast.Module, qualname: str) -> list[int] | None:
    """Reference lines inside the function ``qualname`` (its own ``def`` line excluded), or None if absent.

    Stricter than the module counter: any name with a test word and any ``*TEST*`` string counts, so
    a helper such as ``_single_user_test_key_compat_allowed`` or ``SINGLE_USER_TEST_API_KEY`` is found.
    """
    scope: ast.AST = tree
    for part in qualname.split("."):
        scope = next(
            (
                n
                for n in getattr(scope, "body", [])
                if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == part
            ),
            None,
        )
        if scope is None:
            return None
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    args = scope.args
    parts = [*scope.body, *scope.decorator_list, *args.defaults, *(d for d in args.kw_defaults if d is not None)]
    return sorted(_reference_lines(parts, _core_testing_aliases(tree), strict=True))


# --- Self-tests: each seeds a tree under a fresh tmp root and runs the real scan ---------------


def _seed(root: Path, files: dict[str, str]) -> Path:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return root


def test_the_counter_counts_each_reference_shape(tmp_path: Path) -> None:
    root = _seed(tmp_path / "app", {
        "core/X/env_reads.py": 'import os\nA = os.getenv("TEST_MODE")\nB = os.environ.get("PYTEST_CURRENT_TEST")\n'
                               'KEYS = ("TESTING", "TLDW_TEST_MODE")\nC = "TLDW_TEST_DB_DIR"\n',
        "core/X/aliased.py": "from tldw_Server_API.app.core.testing import is_test_mode as _tm\n\n\ndef f():\n"
                             "    return 1 if _tm() else 0\n",
        "core/X/predicate.py": 'import os\n\n\ndef _is_test_context() -> bool:\n    return bool(os.getenv("PYTEST_CURRENT_TEST"))\n',
        "core/X/method.py": "class S:\n    def _is_test_mode(self) -> bool:\n        return self.flags.is_test_mode()\n",
        "core/X/nested.py": 'import sys\n\n\ndef outer():\n    def _in_tests() -> bool:\n        return "pytest" in sys.modules\n'
                            "    return _in_tests()\n",
        "core/X/sniff.py": 'X = "tldw_Server_API.tests.helpers"\n',
        "core/X/relative.py": "from ..testing import is_test_mode as _x\n\n\ndef f():\n    return _x()\n",
        "core/X/reexport.py": "from ...MCP_unified.environment import is_test_mode as _y\n\n\ndef g():\n    return _y()\n",
        "core/X/domain.py": 'def test_connection() -> bool:\n    return True\n\n\nLATEST = "ABTEST"\n',
        "core/X/docs.py": '"""Reads TEST_MODE."""\n# os.getenv("TEST_MODE")\n',
        "core/testing.py": 'import os\nX = os.getenv("TEST_MODE")\n',
        "core/X/tests/test_x.py": 'X = "TEST_MODE"\n',
        "core/MCP_unified/environment.py": 'def is_test_mode() -> bool:\n    return env_flag_enabled("TEST_MODE")\n',
    })

    assert measure(root) == {
        "core/MCP_unified/environment.py": {"lines": 2, "predicates": 0},
        "core/X/aliased.py": {"lines": 1, "predicates": 0},
        "core/X/env_reads.py": {"lines": 4, "predicates": 0},
        "core/X/method.py": {"lines": 2, "predicates": 1},
        "core/X/nested.py": {"lines": 1, "predicates": 1},
        "core/X/predicate.py": {"lines": 2, "predicates": 1},
        "core/X/reexport.py": {"lines": 1, "predicates": 0},
        "core/X/relative.py": {"lines": 1, "predicates": 0},
        "core/X/sniff.py": {"lines": 1, "predicates": 0},
    }


def test_a_new_predicate_growth_and_a_stale_entry_fail(tmp_path: Path) -> None:
    root = _seed(tmp_path / "app", {
        "core/X/a.py": 'import os\n\n\ndef _is_test_mode() -> bool:\n    return os.getenv("TEST_MODE") == "1"\n',
        "core/X/new.py": 'import os\nX = os.getenv("PYTEST_CURRENT_TEST")\n',
    })
    baseline = {
        "core/X/a.py": {"lines": 2, "predicates": 0, "reason": None},
        "core/X/gone.py": {"lines": 1, "predicates": 0, "reason": None},
    }

    problems = _ceilings.violations(RATCHET, measure(root), baseline)

    assert [p.split(":")[0] for p in problems] == ["core/X/a.py", "core/X/gone.py", "core/X/new.py"]
    assert "predicates 1 > ceiling 0" in problems[0]
    assert "stale entry" in problems[1]
    assert problems[2].startswith("core/X/new.py: lines 1, predicates 0, and no ceiling")


def test_the_function_pin_finds_a_signal_and_a_missing_function() -> None:
    tree = ast.parse(
        'import os\n\ndef guarded(url):\n    if os.getenv("PYTEST_CURRENT_TEST"):\n        return True\n    return False\n'
        "\nclass C:\n    def clean(self):\n        return 1\n"
    )
    assert function_reference_lines(tree, "guarded") == [4]
    assert function_reference_lines(tree, "C.clean") == []
    assert function_reference_lines(tree, "C.gone") is None
    defaulted = ast.parse('import os\n\n@os.getenv("TEST_MODE")\ndef g(x=os.getenv("TEST_MODE")):\n    return 1\n')
    assert function_reference_lines(defaulted, "g") == [3, 4]
    compat = ast.parse(
        "import os\n\nclass S:\n    def auth(self, k):\n"
        '        key = os.getenv("SINGLE_USER_TEST_API_KEY")\n'
        "        return key and self._test_key_compat_allowed()\n"
    )
    assert function_reference_lines(compat, "S.auth") == [5, 6]


# --- The ratchet --------------------------------------------------------------------------------


@cache
def _app_measurement() -> dict[str, dict[str, int]]:
    """The real tree, measured once per session for the tests below (~15 s)."""
    return measure()


def test_test_mode_branches_do_not_grow() -> None:
    measured, baseline = _app_measurement(), _ceilings.load(RATCHET.baseline)

    problems = _ceilings.violations(RATCHET, measured, baseline)

    assert problems == [], "\n".join(problems)
    _ceilings.warn_if_tightenable(RATCHET, measured, baseline)


def test_raised_test_mode_ceilings_carry_a_new_reason() -> None:
    problems = _ceilings.reasonless_bumps(RATCHET, _app_measurement())
    assert problems == [], "\n".join(problems)


def test_core_testing_helpers_are_all_counted_by_name() -> None:
    """core/testing.py is exempt from the scan, so a signal helper it adds must be countable at its call sites."""
    tree = parse_tree(APP_ROOT)[APP_ROOT / "core/testing.py"]
    unnamed = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _reference_lines(node.body, frozenset())
        and node.name not in CORE_TESTING_SIGNALS
        and not is_signal_name(node.name)
    ]
    assert unnamed == [], f"add {unnamed} to CORE_TESTING_SIGNALS or give them a test-mode name"


def test_security_functions_read_no_test_signal() -> None:
    trees = parse_tree(APP_ROOT)
    problems = []
    for key in sorted(SECURITY_FUNCTIONS):
        module, qualname = key.split(":")
        lines = function_reference_lines(trees[APP_ROOT / module], qualname)
        if lines is None:
            problems.append(f"{key}: not found; update SECURITY_FUNCTIONS with the new name")
        elif lines:
            problems.append(f"{key}: reads a test-mode signal at lines {lines}; drive it from a fixture instead")
    assert problems == [], "\n".join(problems)


if __name__ == "__main__":
    raise SystemExit(_ceilings.main(RATCHET, measure))
