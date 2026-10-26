"""Ratchet: keyword and tag identity goes through normalize_keyword_identity (TASK-13950.5).

Keywords and tags are case-insensitive product-wide with display casing preserved.
The one identity rule is ``core/Utils/keyword_identity.normalize_keyword_identity``
(NFC plus casefold; ``dedupe_keywords`` for lists). Ad hoc ``.lower()`` or
``.casefold()`` keys drift from it: they miss NFC, ``lower()`` keeps the German
sharp s, and lower-casing on write loses the display casing. This ratchet freezes
today's ad hoc sites per module so they can only go down.

Scanned: every module under ``tldw_Server_API/app``, so a new module is covered by
default with an implicit baseline of 0. Counted: each lowercasing of a keyword or
tag value, in any of these shapes:
  * ``x.lower()`` / ``x.casefold()``,
  * ``str.lower(x)``,
  * ``map(str.lower, xs)``,
  * a call taking ``key=str.lower`` (``sorted(tags, key=str.lower)``).
It counts only when the lowered value is a keyword or tag. That means one of:
  * the operand names a keyword or tag: a name, attribute or identifier-like string
    key whose snake_case words include one of KEYWORD_WORDS, such as ``tag``,
    ``row["keyword"]`` or ``kw_text``;
  * the operand is a loop or comprehension variable drawn from such an iterable,
    such as ``{k.lower() for k in keywords}``;
  * the call sits in a function whose name includes one of those words, such as
    ``text.lower()`` inside ``_sync_note_keywords``.
Not counted:
  * operands that mention an exception name (EXCEPTION_NAMES), as in
    ``str(exc).lower()`` error-message checks;
  * HTML-parser ``handle_*tag`` callbacks, whose tags are markup.
AST-derived, so comments and docstrings never count.

Each entry is a ceiling with no slack. Growth fails, including any hit in a module
that has no entry yet. A module that drops below its entry also fails, until the
entry is lowered (or dropped when it reaches 0). Raising an entry needs a trailing
``# bump: <reason>`` comment on it, reviewed in the PR.

backend-required's "Enforce CI contracts and code ratchets" step runs this file;
tests/CI/test_license_first_workflow_contracts.py, which that step also runs, fails
if it stops.

Recount (the counter that seeded the baseline):
    python tldw_Server_API/tests/lint/test_keyword_identity_ratchet.py
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = Path("tldw_Server_API") / "app"

KEYWORD_WORDS = frozenset({"kw", "kws", "keyword", "keywords", "tag", "tags"})
EXCEPTION_NAMES = frozenset({"e", "err", "error", "exc"})
NORMALIZING_METHODS = frozenset({"lower", "casefold"})
HTML_TAG_HANDLER = re.compile(r"handle_\w*tag")
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)

# Per-module ad hoc keyword/tag normalization counts; modules not listed must stay at 0.
KEYWORD_NORMALIZATION_BASELINE: dict[str, int] = {
    "tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/items.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/media/item.py": 4,
    "tldw_Server_API/app/api/v1/endpoints/media/listing.py": 2,
    "tldw_Server_API/app/api/v1/endpoints/notes.py": 1,  # YAML front-matter key match, not identity
    "tldw_Server_API/app/api/v1/endpoints/paper_search.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/prompts.py": 3,
    "tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/watchlists.py": 2,
    "tldw_Server_API/app/api/v1/schemas/chat_dictionary_schemas.py": 1,
    "tldw_Server_API/app/api/v1/schemas/persona.py": 1,
    "tldw_Server_API/app/core/AuthNZ/email_service.py": 2,
    "tldw_Server_API/app/core/Character_Chat/chat_dictionary.py": 1,
    "tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_selector.py": 1,
    "tldw_Server_API/app/core/Character_Chat/world_book_manager.py": 2,
    "tldw_Server_API/app/core/Chat/conversation_enrichment.py": 4,
    "tldw_Server_API/app/core/Collections/reading_digest_jobs.py": 4,
    "tldw_Server_API/app/core/Collections/reading_importers.py": 3,
    "tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py": 2,
    "tldw_Server_API/app/core/DB_Management/Collections_DB.py": 1,
    "tldw_Server_API/app/core/DB_Management/Prompts_DB.py": 2,
    "tldw_Server_API/app/core/DB_Management/Watchlists_DB.py": 1,
    "tldw_Server_API/app/core/DB_Management/chacha/character_store.py": 4,
    "tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py": 1,
    "tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py": 5,
    "tldw_Server_API/app/core/DB_Management/chacha/note_semantic_store.py": 1,
    "tldw_Server_API/app/core/DB_Management/chacha/note_store.py": 3,
    "tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py": 2,
    "tldw_Server_API/app/core/DB_Management/media_db/repositories/clone_snapshot_repository.py": 1,
    "tldw_Server_API/app/core/DB_Management/media_db/repositories/keywords_repository.py": 3,
    "tldw_Server_API/app/core/DB_Management/media_db/repositories/media_repository.py": 2,
    "tldw_Server_API/app/core/DB_Management/media_db/repositories/media_search_repository.py": 2,
    "tldw_Server_API/app/core/DB_Management/media_db/runtime/keyword_access_ops.py": 1,
    "tldw_Server_API/app/core/DB_Management/media_db/runtime/safe_metadata_search_ops.py": 2,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/Books/Book_Processing_Lib.py": 1,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/Upload_Sink.py": 1,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py": 5,
    "tldw_Server_API/app/core/LLM_Calls/structured_output.py": 1,
    "tldw_Server_API/app/core/Local_LLM/Llamafile_Handler.py": 1,
    "tldw_Server_API/app/core/MCP_unified/tests/test_notes_crud_tags.py": 2,  # fake DB in a test
    "tldw_Server_API/app/core/Notes/attachment_policy.py": 2,
    "tldw_Server_API/app/core/Notes_Graph/suggestion_generation.py": 1,
    "tldw_Server_API/app/core/Notes_Graph/suggestion_service.py": 1,
    "tldw_Server_API/app/core/Persona/exemplar_ingestion.py": 1,
    "tldw_Server_API/app/core/Persona/visual_library_service.py": 1,
    "tldw_Server_API/app/core/PrivilegeMaps/service.py": 1,
    "tldw_Server_API/app/core/Prompt_Management/Prompts_Interop.py": 1,
    "tldw_Server_API/app/core/Prompt_Management/prompt_studio/prompt_improver.py": 1,
    "tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py": 1,
    "tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py": 1,
    "tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py": 3,
    "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py": 1,
    "tldw_Server_API/app/core/RAG/rag_service/utils.py": 1,
    "tldw_Server_API/app/core/Research/planner.py": 1,
    "tldw_Server_API/app/core/Sync/v2/notes_organization_coordinator.py": 3,
    "tldw_Server_API/app/core/Sync/v2/notes_task_contract.py": 2,
    "tldw_Server_API/app/core/Third_Party/RePEc.py": 1,
    "tldw_Server_API/app/core/Utils/keyword_identity.py": 1,  # the identity rule itself
    "tldw_Server_API/app/core/Watchlists/content_alerts.py": 2,
    "tldw_Server_API/app/core/Watchlists/fetchers.py": 2,
    "tldw_Server_API/app/core/Watchlists/filters.py": 3,
    "tldw_Server_API/app/core/Web_Scraping/extraction/strategies/cluster.py": 2,
    "tldw_Server_API/app/core/Web_Scraping/scoring.py": 1,
    "tldw_Server_API/app/core/Workflows/adapters/text/nlp.py": 1,
    "tldw_Server_API/app/main.py": 1,
    "tldw_Server_API/app/services/admin_system_ops_service.py": 2,
    "tldw_Server_API/app/services/quiz_generator.py": 2,
}


def _words(name: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", name.lower()))


def _names(nodes: list[ast.AST]) -> set[str]:
    names: set[str] = set()
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
            elif isinstance(child, ast.Attribute):
                names.add(child.attr)
            elif isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value.isidentifier():
                names.add(child.value)
    return names


def _names_keyword(names: set[str], bound: frozenset[str]) -> bool:
    return bool(names & bound) or any(_words(name) & KEYWORD_WORDS for name in names)


def _is_str_method(node: ast.AST) -> bool:
    """``str.lower`` / ``str.casefold`` used as a function value."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr in NORMALIZING_METHODS
        and isinstance(node.value, ast.Name)
        and node.value.id == "str"
    )


def _lowered_operands(call: ast.Call) -> list[ast.AST] | None:
    """Return what a lowercasing call lowers, or None when ``call`` is not one."""
    func = call.func
    if _is_str_method(func):
        return list(call.args)  # str.lower(tag)
    if isinstance(func, ast.Attribute) and func.attr in NORMALIZING_METHODS and not call.args and not call.keywords:
        return [func.value]  # tag.lower()
    if isinstance(func, ast.Name) and func.id == "map" and call.args and _is_str_method(call.args[0]):
        return list(call.args[1:])  # map(str.lower, tags)
    if any(keyword.arg == "key" and _is_str_method(keyword.value) for keyword in call.keywords):
        return [func, *call.args]  # sorted(tags, key=str.lower), tags.sort(key=str.lower)
    return None


def _is_ad_hoc_normalization(call: ast.Call, function_name: str | None, bound: frozenset[str]) -> bool:
    operands = _lowered_operands(call)
    if operands is None:
        return False
    if function_name is not None and HTML_TAG_HANDLER.fullmatch(function_name):
        return False
    names = _names(operands)
    if names & EXCEPTION_NAMES:
        return False
    in_keyword_function = function_name is not None and bool(_words(function_name) & KEYWORD_WORDS)
    return in_keyword_function or _names_keyword(names, bound)


def _keyword_loop_targets(node: ast.AST, bound: frozenset[str]) -> frozenset[str]:
    """Names bound by a ``for`` or comprehension that iterates over keywords or tags."""
    if isinstance(node, (ast.For, ast.AsyncFor)):
        loops = [(node.target, node.iter)]
    elif isinstance(node, COMPREHENSIONS):
        loops = [(generator.target, generator.iter) for generator in node.generators]
    else:
        return bound
    added = {name for target, iterable in loops if _names_keyword(_names([iterable]), bound) for name in _names([target])}
    return bound | added


def count_ad_hoc_keyword_normalization(source: str, filename: str = "<memory>") -> int:
    """Return how many ad hoc keyword or tag normalizations ``source`` contains."""
    count = 0
    stack: list[tuple[ast.AST, str | None, frozenset[str]]] = [(ast.parse(source, filename=filename), None, frozenset())]
    while stack:
        node, function_name, bound = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_name = node.name
        bound = _keyword_loop_targets(node, bound)
        if isinstance(node, ast.Call) and _is_ad_hoc_normalization(node, function_name, bound):
            count += 1
        stack.extend((child, function_name, bound) for child in ast.iter_child_nodes(node))
    return count


def scan_counts(repo_root: Path = REPO_ROOT) -> dict[str, int]:
    """Count every module under ``tldw_Server_API/app`` that has at least one hit."""
    counts: dict[str, int] = {}
    for path in sorted((repo_root / APP_DIR).rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "lower" not in source and "casefold" not in source:
            continue
        module = path.relative_to(repo_root).as_posix()
        found = count_ad_hoc_keyword_normalization(source, module)
        if found:
            counts[module] = found
    return counts


@lru_cache(maxsize=1)
def _repo_counts() -> tuple[tuple[str, int], ...]:
    return tuple(scan_counts().items())


def ratchet_violations(counts: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """Describe every module whose count differs from its baseline entry."""
    problems: list[str] = []
    for module in sorted(set(counts) | set(baseline)):
        now, seeded = counts.get(module, 0), baseline.get(module, 0)
        if now > seeded:
            problems.append(
                f"{module}: {now} ad hoc keyword/tag lowercasing call(s), baseline {seeded}. Compare "
                "normalize_keyword_identity(...) values or use dedupe_keywords (core/Utils/keyword_identity) "
                "and keep the display casing."
            )
        elif now < seeded:
            problems.append(
                f"{module}: {now} ad hoc keyword/tag normalization(s), baseline {seeded}. Migrated or removed "
                "one? Lower its entry in KEYWORD_NORMALIZATION_BASELINE (drop it at 0) so the ratchet stays tight."
            )
    return problems


def test_keyword_normalization_only_decreases() -> None:
    problems = ratchet_violations(dict(_repo_counts()), KEYWORD_NORMALIZATION_BASELINE)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "source",
    [
        "def link(rows):\n    return {row['keyword'].casefold() for row in rows}\n",
        "def link(tag):\n    return tag.strip().lower()\n",
        "def link(kw_text):\n    return str(kw_text).lower()\n",
        "def _sync_note_keywords(values):\n    return {text.lower() for text in values}\n",
        "class Store:\n    def _normalize_tags(self, values):\n        return [lambda: v.lower() for v in values]\n",
        "def link(values):\n    return {k.lower() for k in values.keywords}\n",
        "def link(rows):\n    for i, label in enumerate(rows.tags):\n        yield label.casefold()\n",
        "def link(value):\n    return str.lower(value.tag)\n",
        "def link(values):\n    return set(map(str.lower, values.tags))\n",
        "def link(values):\n    return sorted(values.tags, key=str.casefold)\n",
        "def link(values):\n    values.keywords.sort(key=str.lower)\n",
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
        "def title(names):\n    return {n.lower() for n in names}\n",
        "def title(names):\n    return sorted(names, key=str.lower)\n",
        "class P:\n    def handle_starttag(self, tag, attrs):\n        return tag.lower()\n",
    ],
)
def test_counter_ignores_non_violations(source: str) -> None:
    assert count_ad_hoc_keyword_normalization(source) == 0


def test_the_ratchet_rejects_a_seeded_violation() -> None:
    """Self-test: one new ad hoc lower() in an existing module must fail, not pass silently."""
    module = "tldw_Server_API/app/api/v1/endpoints/chat.py"
    seeded = (REPO_ROOT / module).read_text(encoding="utf-8") + (
        "\n\ndef _seeded(values):\n    return [k.lower() for k in values.keywords]\n"
    )
    counts = {**dict(_repo_counts()), module: count_ad_hoc_keyword_normalization(seeded, module)}

    problems = ratchet_violations(counts, KEYWORD_NORMALIZATION_BASELINE)

    assert len(problems) == 1 and problems[0].startswith(module), problems


def test_a_new_module_is_scanned_without_a_baseline_entry(tmp_path: Path) -> None:
    """Self-test: a violation in a module nobody listed is found and fails."""
    module = APP_DIR / "core" / "Example" / "new_feature.py"
    (tmp_path / module).parent.mkdir(parents=True)
    (tmp_path / module).write_text("def link(values):\n    return set(map(str.lower, values.tags))\n", encoding="utf-8")

    counts = scan_counts(tmp_path)
    problems = ratchet_violations(counts, {})

    assert counts == {module.as_posix(): 1}
    assert len(problems) == 1 and problems[0].startswith(module.as_posix()), problems


def test_the_ratchet_asks_for_a_lower_entry_after_a_migration() -> None:
    module = "tldw_Server_API/app/core/DB_Management/chacha/note_store.py"
    counts = {**KEYWORD_NORMALIZATION_BASELINE, module: KEYWORD_NORMALIZATION_BASELINE[module] - 1}

    problems = ratchet_violations(counts, KEYWORD_NORMALIZATION_BASELINE)

    assert len(problems) == 1 and "Lower its entry" in problems[0], problems


if __name__ == "__main__":
    for module, count in scan_counts().items():
        print(f'    "{module}": {count},')
