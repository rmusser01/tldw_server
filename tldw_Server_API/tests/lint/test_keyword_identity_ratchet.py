"""Ratchet: keyword and tag identity goes through normalize_keyword_identity (TASK-13950.5).

Keywords and tags are case-insensitive product-wide with display casing preserved.
The one identity rule is ``core/Utils/keyword_identity.normalize_keyword_identity``
(NFC plus casefold). Ad hoc ``.lower()`` or ``.casefold()`` keys drift from it: they
miss NFC, ``lower()`` keeps the German sharp s, and lower-casing on write loses the
display casing. This ratchet freezes today's ad hoc sites per module so they can
only go down.

Counted, per module in SCANNED_MODULES: every zero-argument ``.lower()`` or
``.casefold()`` call that
  * has a receiver naming a keyword or tag (a name, attribute or identifier-like
    string key whose snake_case words include one of KEYWORD_WORDS, such as
    ``tag.lower()``, ``row["keyword"].casefold()`` or ``kw_text.lower()``), or
  * sits in a function whose name includes one of those words (such as
    ``text.lower()`` inside ``_sync_note_keywords``).
Receivers that mention an exception name (EXCEPTION_NAMES: ``str(exc).lower()``
error-message checks) are not counted. AST-derived, so comments and docstrings
never count. Modules outside SCANNED_MODULES are not scanned; add a module there
when it starts storing or comparing keywords or tags.

Each entry is a ceiling with no slack: growth fails, and so does a module that
drops below its entry until the entry is lowered. Raising an entry needs a
trailing ``# bump: <reason>`` comment on it, reviewed in the PR.

backend-required's "Enforce CI contracts and code ratchets" step runs this file;
tests/CI/test_license_first_workflow_contracts.py, which that step also runs, fails
if it stops.

Recount (the counter that seeded the baseline):
    python tldw_Server_API/tests/lint/test_keyword_identity_ratchet.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]

KEYWORD_WORDS = frozenset({"kw", "kws", "keyword", "keywords", "tag", "tags"})
EXCEPTION_NAMES = frozenset({"e", "err", "error", "exc"})
NORMALIZING_METHODS = frozenset({"lower", "casefold"})

# Per-module ad hoc normalization counts; modules at 0 stay listed so they are scanned.
KEYWORD_NORMALIZATION_BASELINE: dict[str, int] = {
    # Slice 1 (ChaCha keywords): routed through the helper.
    "tldw_Server_API/app/api/v1/endpoints/chat.py": 0,
    "tldw_Server_API/app/api/v1/endpoints/notes.py": 1,  # YAML front-matter key match, not identity
    "tldw_Server_API/app/api/v1/schemas/notes_schemas.py": 0,
    "tldw_Server_API/app/core/DB_Management/chacha/keyword_store.py": 0,
    "tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py": 0,
    "tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py": 0,
    "tldw_Server_API/app/core/Notes/organization_capture.py": 0,
    "tldw_Server_API/app/core/WebClipper/service.py": 0,
    # Other ChaCha keyword code, not yet migrated.
    "tldw_Server_API/app/core/Chat/conversation_enrichment.py": 4,
    "tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py": 1,
    "tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py": 5,
    "tldw_Server_API/app/core/DB_Management/chacha/note_store.py": 3,
    "tldw_Server_API/app/core/Sync/v2/notes_organization_coordinator.py": 3,
    # Slice 2 (Media and Prompts keywords).
    "tldw_Server_API/app/core/DB_Management/media_db/repositories/keywords_repository.py": 3,
    "tldw_Server_API/app/core/DB_Management/media_db/runtime/keyword_access_ops.py": 1,
    "tldw_Server_API/app/core/DB_Management/Prompts_DB.py": 2,
    # Slice 3 (Collections and Watchlists tags).
    "tldw_Server_API/app/core/DB_Management/Collections_DB.py": 1,
    "tldw_Server_API/app/core/DB_Management/Watchlists_DB.py": 1,
}
SCANNED_MODULES = tuple(KEYWORD_NORMALIZATION_BASELINE)


def _words(name: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", name.lower()))


def _receiver_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value.isidentifier():
            names.add(child.value)
    return names


def _is_ad_hoc_normalization(call: ast.Call, function_name: str | None) -> bool:
    if not (isinstance(call.func, ast.Attribute) and call.func.attr in NORMALIZING_METHODS):
        return False
    if call.args or call.keywords:
        return False
    receiver = _receiver_names(call.func.value)
    if receiver & EXCEPTION_NAMES:
        return False
    named_keyword = any(_words(name) & KEYWORD_WORDS for name in receiver)
    in_keyword_function = function_name is not None and bool(_words(function_name) & KEYWORD_WORDS)
    return named_keyword or in_keyword_function


def count_ad_hoc_keyword_normalization(source: str, filename: str = "<memory>") -> int:
    """Return how many ad hoc keyword or tag normalizations ``source`` contains."""
    count = 0
    stack: list[tuple[ast.AST, str | None]] = [(ast.parse(source, filename=filename), None)]
    while stack:
        node, function_name = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_name = node.name
        if isinstance(node, ast.Call) and _is_ad_hoc_normalization(node, function_name):
            count += 1
        stack.extend((child, function_name) for child in ast.iter_child_nodes(node))
    return count


def current_counts() -> dict[str, int]:
    """Count every scanned module; a missing module is reported as -1."""
    counts: dict[str, int] = {}
    for module in SCANNED_MODULES:
        path = REPO_ROOT / module
        counts[module] = (
            count_ad_hoc_keyword_normalization(path.read_text(encoding="utf-8"), module) if path.exists() else -1
        )
    return counts


def ratchet_violations(counts: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """Describe every module whose count differs from its baseline entry."""
    problems: list[str] = []
    for module in sorted(set(counts) | set(baseline)):
        now, seeded = counts.get(module, 0), baseline.get(module, 0)
        if now < 0:
            problems.append(f"{module}: listed in KEYWORD_NORMALIZATION_BASELINE but missing. Drop or move its entry.")
        elif now > seeded:
            problems.append(
                f"{module}: {now} ad hoc keyword/tag .lower()/.casefold() call(s), baseline {seeded}. Compare "
                "normalize_keyword_identity(...) values (core/Utils/keyword_identity) and keep the display casing."
            )
        elif now < seeded:
            problems.append(
                f"{module}: {now} ad hoc keyword/tag normalization(s), baseline {seeded}. Migrated one? "
                "Lower its entry in KEYWORD_NORMALIZATION_BASELINE so the ratchet stays tight."
            )
    return problems


def test_keyword_normalization_only_decreases() -> None:
    problems = ratchet_violations(current_counts(), KEYWORD_NORMALIZATION_BASELINE)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "source",
    [
        "def link(rows):\n    return {row['keyword'].casefold() for row in rows}\n",
        "def link(tag):\n    return tag.strip().lower()\n",
        "def link(kw_text):\n    return str(kw_text).lower()\n",
        "def _sync_note_keywords(values):\n    return {text.lower() for text in values}\n",
        "class Store:\n    def _normalize_tags(self, values):\n        return [lambda: v.lower() for v in values]\n",
    ],
)
def test_counter_sees_each_ad_hoc_shape(source: str) -> None:
    assert count_ad_hoc_keyword_normalization(source) == 1


@pytest.mark.parametrize(
    "source",
    [
        '"""tag.lower()"""\n# keyword.casefold()\nx = "keyword.lower()"\n',
        "def add_keyword(text):\n    try:\n        pass\n    except Exception as exc:\n        return 'unique' in str(exc).lower()\n",
        "def add_keyword(text):\n    return normalize_keyword_identity(text)\n",
        "def title(name):\n    return name.lower()\n",
    ],
)
def test_counter_ignores_non_violations(source: str) -> None:
    assert count_ad_hoc_keyword_normalization(source) == 0


def test_the_ratchet_rejects_a_seeded_violation() -> None:
    """Self-test: one new ad hoc lower() in a scanned module must fail, not pass silently."""
    module = "tldw_Server_API/app/api/v1/endpoints/chat.py"
    seeded = (REPO_ROOT / module).read_text(encoding="utf-8") + (
        "\n\ndef _seeded(tags):\n    return [tag.lower() for tag in tags]\n"
    )
    counts = {**current_counts(), module: count_ad_hoc_keyword_normalization(seeded, module)}

    problems = ratchet_violations(counts, KEYWORD_NORMALIZATION_BASELINE)

    assert len(problems) == 1 and problems[0].startswith(module), problems


def test_the_ratchet_asks_for_a_lower_entry_after_a_migration() -> None:
    module = "tldw_Server_API/app/core/DB_Management/chacha/note_store.py"
    counts = {**KEYWORD_NORMALIZATION_BASELINE, module: KEYWORD_NORMALIZATION_BASELINE[module] - 1}

    problems = ratchet_violations(counts, KEYWORD_NORMALIZATION_BASELINE)

    assert len(problems) == 1 and "Lower its entry" in problems[0], problems


def test_the_ratchet_reports_a_missing_scanned_module() -> None:
    problems = ratchet_violations({"gone.py": -1}, {"gone.py": 0})

    assert problems == ["gone.py: listed in KEYWORD_NORMALIZATION_BASELINE but missing. Drop or move its entry."]


if __name__ == "__main__":
    for module, count in current_counts().items():
        print(f'    "{module}": {count},')
