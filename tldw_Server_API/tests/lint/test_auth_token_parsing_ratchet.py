"""Ratchet: token-shape guesses and hand-parsed bearer headers may only decrease (TASK-13950.7).

Every WebSocket route used to resolve its own caller. Seven modules define a private
``_looks_like_jwt``, and endpoints split ``Authorization: Bearer ...`` themselves, so
each copy drifts on its own: the workflows WebSocket verified AuthNZ tokens with the
MCP JWT manager, rejected all of them and never checked revocation.
core/AuthNZ/websocket_principal.resolve_ws_principal is the one WebSocket resolver.
Routes migrate to it one per PR, each deleting its copies and lowering the counts here.

Counted, AST-derived so docstrings and comments never count:

- ``_looks_like_jwt``: function definitions with exactly that name, anywhere under
  tldw_Server_API/app. The goal is one.
- Hand-parsed bearer headers, in tldw_Server_API/app/api/v1/endpoints only:
  ``x.startswith("Bearer ")`` with any casing or trailing space, and ``==``/``!=``
  between a ``.lower()``/``.casefold()`` call and ``"bearer"``. Comparisons against a
  plain name or attribute (``req.auth_type == "bearer"``) are configuration, not
  parsing, and are not counted.

Entries are keyed by module and must match exactly: a module above its count fails, and
so does one below it, so a migration lowers the entry in the same PR. Seed it with:

    python tldw_Server_API/tests/lint/test_auth_token_parsing_ratchet.py
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "tldw_Server_API" / "app"
ENDPOINTS_ROOT = APP_ROOT / "api" / "v1" / "endpoints"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "backend-required.yml"

# Per-module counts; lower or remove an entry when a module migrates.
LOOKS_LIKE_JWT_BASELINE: dict[str, int] = {
    "tldw_Server_API/app/api/v1/API_Deps/auth_deps.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/persona.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/sandbox.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/voice_assistant.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/watchlists.py": 1,
    "tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py": 1,
    "tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py": 1,
}

BEARER_PARSE_BASELINE: dict[str, int] = {
    "tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/auth.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/authnz_debug.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/persona.py": 3,
    "tldw_Server_API/app/api/v1/endpoints/prompts.py": 3,
    "tldw_Server_API/app/api/v1/endpoints/rpg.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/sandbox.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/watchlists.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/workflows.py": 1,
}

_FIX = (
    "Resolve WebSocket callers with core/AuthNZ/websocket_principal.resolve_ws_principal "
    "and HTTP callers with the AuthNZ dependencies instead of parsing tokens by hand."
)


def _is_bearer(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip().lower() == "bearer"


def _is_case_fold(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"lower", "casefold"}
        and not node.args
    )


def count_looks_like_jwt(tree: ast.AST) -> int:
    """Return how many functions in ``tree`` are named ``_looks_like_jwt``."""
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_looks_like_jwt"
    )


def count_bearer_parses(tree: ast.AST) -> int:
    """Return how many places in ``tree`` strip or match a bearer scheme by hand."""
    found = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "startswith"
            and node.args
            and _is_bearer(node.args[0])
        ):
            found += 1
        elif isinstance(node, ast.Compare) and all(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
            operands = [node.left, *node.comparators]
            if any(map(_is_bearer, operands)) and any(map(_is_case_fold, operands)):
                found += 1
    return found


def _counts(root: Path, needle: str, counter: Callable[[ast.AST], int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if needle not in source.lower():  # cheap prefilter; the AST decides
            continue
        found = counter(ast.parse(source, filename=str(path)))
        if found:
            counts[path.relative_to(REPO_ROOT).as_posix()] = found
    return counts


def current_looks_like_jwt() -> dict[str, int]:
    return _counts(APP_ROOT, "_looks_like_jwt", count_looks_like_jwt)


def current_bearer_parses() -> dict[str, int]:
    return _counts(ENDPOINTS_ROOT, "bearer", count_bearer_parses)


def ratchet_violations(what: str, counts: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """Describe every module whose count differs from its baseline entry."""
    problems: list[str] = []
    for module in sorted(set(counts) | set(baseline)):
        now, seeded = counts.get(module, 0), baseline.get(module, 0)
        if now > seeded:
            problems.append(f"{module}: {now} {what}, baseline {seeded}. {_FIX}")
        elif now < seeded:
            problems.append(
                f"{module}: {now} {what}, baseline {seeded}. Migrated one? Lower its entry in "
                "this file so the ratchet stays tight."
            )
    return problems


def test_looks_like_jwt_copies_only_decrease() -> None:
    problems = ratchet_violations("_looks_like_jwt definition(s)", current_looks_like_jwt(), LOOKS_LIKE_JWT_BASELINE)
    assert not problems, "\n".join(problems)


def test_hand_parsed_bearer_headers_only_decrease() -> None:
    problems = ratchet_violations("hand-parsed bearer header(s)", current_bearer_parses(), BEARER_PARSE_BASELINE)
    assert not problems, "\n".join(problems)


def test_the_ratchet_rejects_a_new_copy() -> None:
    """Self-test: a seeded _looks_like_jwt and a seeded bearer split must both fail."""
    module = "tldw_Server_API/app/api/v1/endpoints/new_feature.py"
    tree = ast.parse(
        "def _looks_like_jwt(token):\n"
        "    return token.count('.') == 2\n"
        "def token_from(header):\n"
        "    if header.lower().startswith('bearer '):\n"
        "        return header[7:]\n"
        "    scheme, _, value = header.partition(' ')\n"
        "    return value if scheme.casefold() == 'Bearer' else None\n"
    )

    jwt_problems = ratchet_violations(
        "x", {**LOOKS_LIKE_JWT_BASELINE, module: count_looks_like_jwt(tree)}, LOOKS_LIKE_JWT_BASELINE
    )
    bearer_problems = ratchet_violations(
        "x", {**BEARER_PARSE_BASELINE, module: count_bearer_parses(tree)}, BEARER_PARSE_BASELINE
    )

    assert len(jwt_problems) == 1 and jwt_problems[0].startswith(module), jwt_problems
    assert bearer_problems == [f"{module}: 2 x, baseline 0. {_FIX}"], bearer_problems


def test_the_ratchet_asks_for_a_lower_seed_after_a_migration() -> None:
    """Self-test: deleting a copy without lowering the seed must also fail."""
    module, seeded = next(iter(BEARER_PARSE_BASELINE.items()))
    problems = ratchet_violations("x", {**BEARER_PARSE_BASELINE, module: seeded - 1}, BEARER_PARSE_BASELINE)
    assert len(problems) == 1 and "Lower its entry" in problems[0], problems


def test_configuration_comparisons_docstrings_and_comments_are_not_counted() -> None:
    tree = ast.parse(
        '"""if h.lower().startswith("bearer "): ..."""\n'
        "# token.startswith('Bearer ')\n"
        "if req.auth_type == 'bearer' or auth_type == 'bearer':\n"
        "    headers = {'Authorization': f'Bearer {secret}'}\n"
        "def looks_like_jwt(token):\n"
        "    return False\n"
    )
    assert count_bearer_parses(tree) == 0
    assert count_looks_like_jwt(tree) == 0


def test_backend_required_runs_the_websocket_auth_guards() -> None:
    """Workflow contract: these guards enforce nothing unless a required gate runs them."""
    steps = [
        step
        for job in yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Enforce CI contracts and code ratchets"
    ]
    assert len(steps) == 1
    assert not steps[0].get("continue-on-error", False)
    for suite in ("tldw_Server_API/tests/lint/test_auth_token_parsing_ratchet.py",):
        assert suite in steps[0]["run"], suite


if __name__ == "__main__":
    for name, counts in (("LOOKS_LIKE_JWT_BASELINE", current_looks_like_jwt()), ("BEARER_PARSE_BASELINE", current_bearer_parses())):
        print(f"{name} = {counts!r}  # {sum(counts.values())} in {len(counts)} modules")
