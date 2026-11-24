"""Structure pins for the TASK-13950.24 split of ``unified_rag_pipeline`` (AST only).

Task 5 adds subprocess import-order tests to this file; nothing here imports the pipeline.

* Patch scan: every name a test patches on ``unified_pipeline`` (string targets, ``setattr`` and
  ``patch.object`` on an alias of the module, and ``<alias>.<Name> = ...``).
* ``pipeline_module_violations``: a stage module must read such a name through ``up.<name>``
  (R4, R11). Binding or importing it locally would leave the test's patch inert. Vacuous until
  Task 3 creates ``rag_service/pipeline``.
* The adaptive rerun's frozen argument set (R6).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, REPO_ROOT, parse_tree

pytestmark = pytest.mark.unit

RAG_SERVICE = APP_ROOT / "core" / "RAG" / "rag_service"
PIPELINE_FILE = RAG_SERVICE / "unified_pipeline.py"
PIPELINE_PKG = RAG_SERVICE / "pipeline"
SCAN_ROOTS = (
    REPO_ROOT / "tldw_Server_API" / "tests",
    APP_ROOT / "core" / "MCP_unified" / "tests",
    APP_ROOT / "api" / "v1" / "endpoints" / "rag_unified.py",
)
KNOWN_TARGETS = {
    "MultiDatabaseRetriever",
    "AnswerGenerator",
    "get_shared_cache",
    "create_reranker",
    "logger",
    "unified_rag_pipeline",
    "RetrievalConfig",
}
MODULE = "unified_pipeline"
STRING_TARGET = re.compile(r"(?:^|\.)unified_pipeline\.(\w+)")
PATCHERS = {"patch", "setattr"}
RERUN_CALLERS = {"unified_batch_pipeline", "simple_search", "advanced_search"}

# The 79 keywords of the adaptive rerun call in unified_rag_pipeline, in order (dev 5033a7689b).
RERUN_KEYWORDS: tuple[str, ...] = (
    "query",
    "sources",
    "media_db_path",
    "notes_db_path",
    "character_db_path",
    "kanban_db_path",
    "media_db",
    "chacha_db",
    "search_mode",
    "fts_level",
    "hybrid_alpha",
    "top_k",
    "min_score",
    "expand_query",
    "expansion_strategies",
    "spell_check",
    "enable_cache",
    "cache_threshold",
    "adaptive_cache",
    "keyword_filter",
    "include_media_ids",
    "include_note_ids",
    "enable_security_filter",
    "detect_pii",
    "redact_pii",
    "sensitivity_level",
    "content_filter",
    "enable_table_processing",
    "table_method",
    "enable_vlm_late_chunking",
    "vlm_backend",
    "vlm_detect_tables_only",
    "vlm_max_pages",
    "vlm_late_chunk_top_k_docs",
    "enable_enhanced_chunking",
    "chunk_type_filter",
    "enable_parent_expansion",
    "parent_context_size",
    "include_sibling_chunks",
    "sibling_window",
    "include_parent_document",
    "parent_max_tokens",
    "enable_reranking",
    "reranking_strategy",
    "rerank_top_k",
    "reranking_model",
    "rerank_min_relevance_prob",
    "rerank_sentinel_margin",
    "enable_citations",
    "citation_style",
    "include_page_numbers",
    "enable_chunk_citations",
    "enable_generation",
    "generation_model",
    "generation_provider",
    "generation_prompt",
    "max_generation_tokens",
    "enable_post_verification",
    "_adaptive_rerun",
    "enable_injection_filter",
    "injection_filter_strength",
    "require_hard_citations",
    "enable_numeric_fidelity",
    "numeric_fidelity_behavior",
    "enable_claims",
    "index_namespace",
    "user_id",
    "session_id",
    "enable_monitoring",
    "enable_observability",
    "trace_id",
    "credential_runtime",
    "enable_performance_analysis",
    "timeout_seconds",
    "highlight_results",
    "highlight_query_terms",
    "track_cost",
    "debug_mode",
    "include_rerank_debug_documents",
)
RERUN_COMPUTED: dict[str, str] = {
    "expand_query": "rerun_expand",
    "enable_cache": "False if adaptive_rerun_bypass_cache else enable_cache",
    "enable_generation": "bool(adaptive_rerun_include_generation)",
    "enable_post_verification": "False",
    "_adaptive_rerun": "True",
    "enable_claims": "False",
}


def _files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from sorted(root.rglob("*.py"))


def _alias_exprs(tree: ast.AST) -> set[str]:
    """Expressions that name ``unified_pipeline`` here: any import of it, in any scope."""
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            out.update(a.asname or a.name for a in n.names if a.name == MODULE)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[-1] == MODULE:
                    out.add(a.asname or a.name)
    return out


def _call_name(func: ast.expr) -> str | None:
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None


def _names_in(tree: ast.AST) -> set[str]:
    aliases = _alias_exprs(tree)
    found: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and (_call_name(n.func) in PATCHERS or _call_name(n.func) == "object") and n.args:
            first = n.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                m = STRING_TARGET.search(first.value)
                if m:
                    found.add(m.group(1))
                elif first.value.endswith(MODULE) and len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                    found.add(str(n.args[1].value))
            elif ast.unparse(first) in aliases and len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                if isinstance(n.args[1].value, str):
                    found.add(n.args[1].value)
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                if isinstance(t, ast.Attribute) and ast.unparse(t.value) in aliases:
                    found.add(t.attr)
    return found


def patched_pipeline_names(test_roots: Iterable[Path]) -> set[str]:
    """Names tests patch on ``unified_pipeline``, in every form (prefiltered by the text ``unified_pipeline``)."""
    names: set[str] = set()
    for path in _files(test_roots):
        text = path.read_text(encoding="utf-8", errors="replace")
        if MODULE in text:
            names |= _names_in(ast.parse(text, filename=str(path)))
    return names


def pipeline_module_violations(pkg_dir: Path, patched: set[str]) -> list[str]:
    """Stage modules that bind or read a patched name locally instead of through ``up.<name>``."""
    out: list[str] = []
    for path in sorted(pkg_dir.rglob("*.py")) if pkg_dir.is_dir() else []:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    bound = a.asname or a.name.split(".")[0]
                    if a.name != MODULE and not a.name.endswith("." + MODULE):
                        modules.add(bound)
                    if bound in patched:
                        out.append(f"{path.name}:{n.lineno}: imports {bound}")
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in patched:
                out.append(f"{path.name}:{n.lineno}: defines {n.name}")
            elif isinstance(n, ast.Name) and n.id in patched:
                kind = "reads" if isinstance(n.ctx, ast.Load) else "assigns"
                out.append(f"{path.name}:{n.lineno}: {kind} {n.id}")
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and n.attr in patched and isinstance(n.value, ast.Name) and n.value.id in modules:
                out.append(f"{path.name}:{n.lineno}: reads {n.value.id}.{n.attr}")
    return out


# ---- the patch scan ---------------------------------------------------------------------------


def test_patch_scan_finds_the_known_targets() -> None:
    assert patched_pipeline_names(SCAN_ROOTS) >= KNOWN_TARGETS


SEEDED = '''
from unittest.mock import patch
import pkg.rag_service.unified_pipeline as up


def test_forms(monkeypatch, mocker):
    from pkg.rag_service import unified_pipeline as upm
    with patch("pkg.rag_service.unified_pipeline.NameA"):
        pass
    mocker.patch("pkg.rag_service.unified_pipeline.NameB")
    monkeypatch.setattr("pkg.rag_service.unified_pipeline.NameC", 1)
    monkeypatch.setattr(up, "NameD", 1)
    setattr(upm, "NameE", 1)
    patch.object(up, "NameF")
    up.NameG = 1
    monkeypatch.setattr("pkg.rag_service.unified_pipeline", "NameH", 1)
    patch("pkg.other_module.NotCounted")
'''


def test_patch_scan_sees_each_form(tmp_path: Path) -> None:
    (tmp_path / "test_seeded.py").write_text(SEEDED)
    (tmp_path / "test_unrelated.py").write_text("patch('pkg.other.X')\n")
    assert patched_pipeline_names([tmp_path]) == {f"Name{c}" for c in "ABCDEFGH"}


# ---- stage modules read patched names through up ---------------------------------------------


def test_pipeline_modules_read_patched_names_through_the_pipeline_module() -> None:
    assert pipeline_module_violations(PIPELINE_PKG, patched_pipeline_names(SCAN_ROOTS)) == []


def test_patched_name_import_in_a_pipeline_module_is_reported(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("from ..database_retrievers import MultiDatabaseRetriever\n")
    assert len(pipeline_module_violations(tmp_path, {"MultiDatabaseRetriever"})) == 1
    (tmp_path / "a.py").unlink()
    (tmp_path / "b.py").write_text("from .. import database_retrievers as dr\n\n\ndef f():\n    return dr.MultiDatabaseRetriever()\n")
    assert len(pipeline_module_violations(tmp_path, {"MultiDatabaseRetriever"})) == 1
    (tmp_path / "b.py").write_text("from .. import unified_pipeline as up\n\n\ndef f():\n    return up.MultiDatabaseRetriever()\n")
    assert pipeline_module_violations(tmp_path, {"MultiDatabaseRetriever"}) == []


# ---- the adaptive rerun (R6) ------------------------------------------------------------------


class _Chain(ast.NodeTransformer):
    """``rcfg.<group>.<x>`` and ``state.<x>`` become the bare ``x`` the call passes today."""

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        inner = node.value
        if isinstance(inner, ast.Name) and inner.id == "state":
            return ast.copy_location(ast.Name(node.attr, ast.Load()), node)
        if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name) and inner.value.id == "rcfg":
            return ast.copy_location(ast.Name(node.attr, ast.Load()), node)
        self.generic_visit(node)
        return node


def _rerun_calls() -> list[ast.Call]:
    trees = parse_tree(RAG_SERVICE)
    files = [PIPELINE_FILE] + [p for p in trees if p.is_relative_to(PIPELINE_PKG)]
    calls: list[ast.Call] = []
    for path in files:
        for top in trees[path].body:
            if getattr(top, "name", None) in RERUN_CALLERS:
                continue
            calls += [
                n
                for n in ast.walk(top)
                if isinstance(n, ast.Call) and ast.unparse(n.func) in ("unified_rag_pipeline", "up.unified_rag_pipeline")
            ]
    return calls


def test_rerun_passes_the_frozen_argument_set() -> None:
    calls = _rerun_calls()
    assert len(calls) == 1, [c.lineno for c in calls]
    call = calls[0]
    assert not call.args and all(k.arg for k in call.keywords)
    assert tuple(k.arg for k in call.keywords) == RERUN_KEYWORDS
    values = {k.arg: _Chain().visit(k.value) for k in call.keywords}
    for name, value in values.items():
        if name in RERUN_COMPUTED:
            assert ast.unparse(value) == RERUN_COMPUTED[name], name
        else:
            assert isinstance(value, ast.Name) and value.id == name, name


def test_rerun_chain_rewrite_reads_through_config_and_state() -> None:
    src = ast.parse("f(a=rcfg.retrieval.a, b=state.b, c=False if rcfg.cache.d else state.c)").body[0].value  # type: ignore[attr-defined]
    got = {k.arg: ast.unparse(_Chain().visit(k.value)) for k in src.keywords}
    assert got == {"a": "a", "b": "b", "c": "False if d else c"}
