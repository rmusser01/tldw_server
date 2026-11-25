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
import symtable
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
PATCHERS = {"patch", "setattr", "delattr"}
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
                if isinstance(n, ast.Call) and _call_name(n.func) == "unified_rag_pipeline"
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


# ---- the run configuration (Task 3) -----------------------------------------------------------

FUNC = "unified_rag_pipeline"
CONFIG_FILE = PIPELINE_PKG / "run_config.py"
SHIM_RETURN = "return await run_rag_pipeline(RagRunConfig.from_call_locals(locals()))"
ENTRY = "run_rag_pipeline"
# The 35 parameters the body assigns; ``PipelineState`` holds them (Task 4).
MUTATED_PARAMS = frozenset(
    {
        "adaptive_time_budget_sec", "date_range", "enable_cache", "enable_date_filter", "enable_discussion_search",
        "enable_hyde", "enable_multi_vector_passages", "enable_numeric_fidelity", "enable_parent_expansion",
        "enable_reranking", "enable_web_fallback", "expand_query", "fts_level", "generation_prompt", "hybrid_alpha",
        "hyde_model", "hyde_provider", "include_parent_document", "low_confidence_behavior", "media_db_path",
        "mv_max_spans", "mv_span_chars", "mv_stride", "nli_model", "numeric_fidelity_behavior",
        "parent_context_size", "parent_max_tokens", "query", "require_hard_citations", "reranking_strategy",
        "resolved_request", "retrieval_plan", "sql_retriever", "strict_extractive", "top_k",
    }
)  # fmt: skip


def _pipeline_fn() -> ast.AsyncFunctionDef:
    tree = parse_tree(RAG_SERVICE)[PIPELINE_FILE]
    return next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == FUNC)


def _entry_fn() -> ast.AsyncFunctionDef:
    """``run_rag_pipeline``: the orchestrator, defined right after the kwargs shim."""
    body = parse_tree(RAG_SERVICE)[PIPELINE_FILE].body
    shim = next(i for i, n in enumerate(body) if isinstance(n, ast.AsyncFunctionDef) and n.name == FUNC)
    entry = body[shim + 1]
    assert isinstance(entry, ast.AsyncFunctionDef) and entry.name == ENTRY
    return entry


def _config_fields() -> dict[str, set[str]]:
    """``{group: {field, ...}}`` read from ``run_config.py`` without importing it."""
    classes = {n.name: n for n in parse_tree(RAG_SERVICE)[CONFIG_FILE].body if isinstance(n, ast.ClassDef)}

    def names(cls: ast.ClassDef) -> list[tuple[str, str]]:
        return [(s.target.id, ast.unparse(s.annotation)) for s in cls.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]

    return {group: {f for f, _ in names(classes[cls])} for group, cls in names(classes["RagRunConfig"])}


def test_config_is_built_from_the_call_locals_and_handed_to_the_entry_point() -> None:
    body = _pipeline_fn().body
    assert isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)  # the docstring
    assert ast.dump(body[1]) == ast.dump(ast.parse(SHIM_RETURN, mode="exec").body[0])


def test_shim_only_builds_the_config() -> None:
    fn = _pipeline_fn()
    assert len(fn.body) == 2 and isinstance(fn.body[1], ast.Return)
    assert ENTRY in (ast.get_docstring(fn) or "")
    entry = _entry_fn()
    assert [a.arg for a in entry.args.args] == ["rcfg"] and not (entry.args.vararg or entry.args.kwonlyargs or entry.args.kwarg)
    assert ast.unparse(entry.returns) == ast.unparse(fn.returns)


def test_orchestrator_has_no_nested_defs() -> None:
    nested = [
        f"{n.lineno}: {type(n).__name__}"
        for stmt in _entry_fn().body
        for n in ast.walk(stmt)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))
    ]
    assert nested == []


def _bare_parameter_reads(source: str, func: str, allowed: frozenset[str], params_of: str | None = None) -> set[str]:
    """Parameters of top-level ``params_of`` (default ``func``) read by name, in the scope of top-level ``func`` or any scope
    inside it, that are not in ``allowed``."""
    top = symtable.symtable(source, "<pipeline>", "exec")
    table = next(t for t in top.get_children() if t.get_name() == func)
    owner = next(t for t in top.get_children() if t.get_name() == (params_of or func))
    params = set(owner.get_parameters()) - allowed
    out: set[str] = set()

    def walk(t: symtable.SymbolTable, shadowed: frozenset[str]) -> None:
        own: set[str] = set()
        for name in params - shadowed:
            try:
                sym = t.lookup(name)
            except KeyError:
                continue
            if t is table:
                if sym.is_referenced() or sym.is_assigned():
                    out.add(name)
            elif sym.is_free():
                out.add(name)
            elif sym.is_local():
                own.add(name)  # a nested scope's own binding hides the parameter from its children
        for child in t.get_children():
            walk(child, shadowed | own)

    walk(table, frozenset())
    return out


def test_body_reads_read_only_parameters_through_rcfg() -> None:
    fn = _pipeline_fn()
    assert {a.arg for a in fn.args.args} >= MUTATED_PARAMS and len(MUTATED_PARAMS) == 35
    assert _bare_parameter_reads(PIPELINE_FILE.read_text(encoding="utf-8"), ENTRY, MUTATED_PARAMS, params_of=FUNC) == set()


# ---- PipelineState (Task 4) -----------------------------------------------------------------------
STATE_FILE = PIPELINE_PKG / "run_state.py"
STATE_INIT = "state = _run_state.PipelineState(rcfg)"


def _state_class() -> ast.ClassDef:
    return next(n for n in parse_tree(RAG_SERVICE)[STATE_FILE].body if isinstance(n, ast.ClassDef) and n.name == "PipelineState")


def _slots_and_methods(cls: ast.ClassDef) -> set[str]:
    slots: set[str] = set()
    for s in cls.body:
        targets = s.targets if isinstance(s, ast.Assign) else [s.target] if isinstance(s, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == "__slots__" for t in targets):
            slots |= set(ast.literal_eval(s.value))  # type: ignore[arg-type]
    return slots | {s.name for s in cls.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _attrs_of(tree: ast.AST, owner: str) -> dict[str, int]:
    """``{attr: first line}`` for every ``<owner>.<attr>`` (Load, Store or Del) in ``tree``."""
    out: dict[str, int] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == owner:
            out.setdefault(n.attr, n.lineno)
    return out


def test_state_slots_cover_every_state_attribute() -> None:
    known = _slots_and_methods(_state_class())
    assert {"rcfg", "result"} <= known
    bad: list[str] = []
    for path in [PIPELINE_FILE, *sorted(PIPELINE_PKG.rglob("*.py"))]:
        tree = parse_tree(RAG_SERVICE)[path]
        bad += [f"{path.name}:{ln}: state.{a}" for a, ln in _attrs_of(tree, "state").items() if a not in known]
    bad += [f"run_state.py:{ln}: self.{a}" for a, ln in _attrs_of(_state_class(), "self").items() if a not in known]
    assert bad == []


def test_every_state_slot_is_read_or_written() -> None:
    cls = _state_class()
    slots = {
        n
        for s in cls.body
        if isinstance(s, (ast.Assign, ast.AnnAssign))
        for n in (ast.literal_eval(s.value) if any(isinstance(t, ast.Name) and t.id == "__slots__" for t in (s.targets if isinstance(s, ast.Assign) else [s.target])) else ())
    }
    used = set(_attrs_of(cls, "self"))
    for path in [PIPELINE_FILE, *sorted(PIPELINE_PKG.rglob("*.py"))]:
        used |= set(_attrs_of(parse_tree(RAG_SERVICE)[path], "state"))
    # the 35 mutated parameters are set by ``__init__`` through ``setattr``
    assert slots - used - MUTATED_PARAMS == set()


def test_state_attribute_scan_sees_a_seeded_attribute() -> None:
    tree = ast.parse("state.a = 1\nprint(state.b)\ndel state.c\nother.d")
    assert set(_attrs_of(tree, "state")) == {"a", "b", "c"}


def test_state_is_the_orchestrators_first_statement() -> None:
    assert ast.dump(_entry_fn().body[0]) == ast.dump(ast.parse(STATE_INIT).body[0])


def test_orchestrator_reads_no_signature_parameter() -> None:
    """``run_rag_pipeline`` reads only ``rcfg`` and ``state``; none of the shim's 234 parameter names."""
    assert _bare_parameter_reads(PIPELINE_FILE.read_text(encoding="utf-8"), ENTRY, frozenset(), params_of=FUNC) == set()


def test_mutated_parameters_are_never_read_from_rcfg() -> None:
    bad: list[str] = []
    for path in [PIPELINE_FILE, *sorted(PIPELINE_PKG.rglob("*.py"))]:
        tree = parse_tree(RAG_SERVICE)[path]
        skip: set[int] = set()
        if path == STATE_FILE:
            init = next(n for n in _state_class().body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
            skip = {id(n) for n in ast.walk(init)}
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Attribute)
                and n.attr in MUTATED_PARAMS
                and isinstance(n.value, ast.Attribute)
                and isinstance(n.value.value, ast.Name)
                and n.value.value.id == "rcfg"
                and id(n) not in skip
            ):
                bad.append(f"{path.name}:{n.lineno}: rcfg.{n.value.attr}.{n.attr}")
    assert bad == []


def test_bare_read_scan_sees_a_seeded_read() -> None:
    src = "async def f(a, b, c, **kwargs):\n    rcfg = g(locals())\n    return [a, lambda: b, kwargs]\n"
    assert _bare_parameter_reads(src, "f", frozenset({"c"})) == {"a", "b", "kwargs"}
    shadowed = "async def f(a, **kwargs):\n    def g(**kwargs):\n        return lambda: kwargs\n    return g\n"
    assert _bare_parameter_reads(shadowed, "f", frozenset({"a"})) == set()


def test_every_rcfg_read_names_a_config_field() -> None:
    fields = _config_fields()
    assert len(fields) == 17 and sum(len(v) for v in fields.values()) == 246
    bad: list[str] = []
    files = [PIPELINE_FILE, *sorted(p for p in PIPELINE_PKG.rglob("*.py") if p != CONFIG_FILE)]
    for path in files:
        tree = parse_tree(RAG_SERVICE)[path]
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "rcfg"):
                continue
            outer = parents.get(n)
            if n.attr not in fields:
                bad.append(f"{path.name}:{n.lineno}: rcfg.{n.attr} is not a group")
            elif not (isinstance(outer, ast.Attribute) and outer.value is n):
                bad.append(f"{path.name}:{n.lineno}: rcfg.{n.attr} is not followed by a field")
            elif outer.attr not in fields[n.attr]:
                bad.append(f"{path.name}:{n.lineno}: rcfg.{n.attr}.{outer.attr} is not a field")
    assert bad == []


HOISTED = (
    "resolve_effective_enable_generation", "planned_index_namespace", "refresh_retrieval_snapshot_from_current_knobs",
    "retrieval_plan_for", "sync_effective_query", "latch_optional_provider_failure",
    "latched_optional_provider_failure", "ensure_profile_resolution_metadata", "record_profile_degradation",
    "build_cache_namespace", "get_cache_instance", "with_timeout", "resilient_call", "apply_generation_gate",
    "build_multi_retriever", "build_pipeline_db_paths", "apply_workspace_filtering_to_result",
    "decorate_calibration_metadata",
)  # fmt: skip


def test_hoisted_helpers_are_methods_not_closures() -> None:
    defined = {n.name for n in ast.walk(_entry_fn()) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    gone = {f"_{h}" for h in HOISTED} | {"_EarlyReturn"}
    assert len(HOISTED) == 18 and defined & gone == set()
    cls = _state_class()
    methods = {s.name for s in cls.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert set(HOISTED) <= methods
    module = parse_tree(RAG_SERVICE)[STATE_FILE]
    assert any(isinstance(n, ast.ClassDef) and n.name == "EarlyReturn" for n in module.body)


# ---- import order (Task 5) ----------------------------------------------------------------------
PKG_NAME = "tldw_Server_API.app.core.RAG.rag_service"
UNIFIED_MODULE = f"{PKG_NAME}.unified_pipeline"


def test_pipeline_modules_import_up_only_as_a_module() -> None:
    """No pipeline module takes a name from ``unified_pipeline``; ``up.<name>`` is the only way (R4)."""
    bad: list[str] = []
    for path in sorted(PIPELINE_PKG.rglob("*.py")):
        for n in ast.walk(parse_tree(RAG_SERVICE)[path]):
            if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[-1] == MODULE:
                bad.append(f"{path.name}:{n.lineno}: from {'.' * n.level}{n.module} import ...")
    assert bad == []


def test_unified_pipeline_imports_pipeline_modules_as_modules() -> None:
    """``from .pipeline import x`` in ``unified_pipeline.py`` binds modules, apart from ``RagRunConfig``."""
    bad: list[str] = []
    modules = {p.stem for p in PIPELINE_PKG.glob("*.py")}
    for n in ast.walk(parse_tree(RAG_SERVICE)[PIPELINE_FILE]):
        if not (isinstance(n, ast.ImportFrom) and n.level == 1 and (n.module or "").split(".")[0] == "pipeline"):
            continue
        for a in n.names:
            if n.module == "pipeline":
                ok = a.name in modules
            else:
                ok = n.module == "pipeline.run_config" and a.name == "RagRunConfig"
            if not ok:
                bad.append(f"{n.lineno}: from .{n.module} import {a.name}")
    assert bad == []


def test_pipeline_modules_import_in_either_order() -> None:
    import subprocess
    import sys

    bad: list[str] = []
    for first in ("run_state", "post_processing", "prepare", "unified_pipeline"):
        target = UNIFIED_MODULE if first == "unified_pipeline" else f"{PKG_NAME}.pipeline.{first}"
        code = f"import importlib; importlib.import_module({target!r}); import {UNIFIED_MODULE} as up; up.unified_rag_pipeline, up.run_rag_pipeline"
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=REPO_ROOT)  # nosec B603
        if done.returncode:
            bad.append(f"{first}: {done.stderr[-800:]}")
    assert bad == []


# ---- entry-point patches (PR M: callers pass a RagRunConfig to run_rag_pipeline) ---------------

ENTRY_NAMES = frozenset({"unified_rag_pipeline", "run_rag_pipeline"})
UNIFIED_DOTTED = UNIFIED_MODULE

# Tests that still patch ``unified_pipeline.unified_rag_pipeline``: each one intercepts a path that
# keeps calling the kwargs shim (R17), not a migrated caller. Keyed (test file, enclosing function).
PIPELINE_ENTRY_PATCH_ALLOWLIST: dict[tuple[str, str], str] = {
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_checkpoint_api.py", "test_rag_batch_checkpoint_records_errors"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_checkpoint_api.py", "test_rag_batch_creates_checkpoint"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_checkpoint_api.py", "test_rag_batch_ownerless_checkpoint_persists_no_credential_scope"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_checkpoint_api.py", "test_rag_batch_with_generation_refuses_an_over_quota_user"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_resume_api.py", "test_rag_batch_legacy_checkpoint_uses_server_runtime_scope"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_resume_api.py", "test_rag_batch_resume_records_errors"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_resume_api.py", "test_rag_batch_resume_respects_query_indices"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_resume_api.py", "test_rag_batch_resume_with_generation_refuses_an_over_quota_owner"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_convenience_endpoint_credentials.py", "test_simple_search_does_not_generate_an_answer"): "simple_search",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py", "test_rag_streaming_agentic_path_uses_bundle_contracts_without_re_resolving"): "streaming",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py", "test_rag_streaming_emits_explicit_clean_empty_completion"): "streaming",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py", "test_rag_streaming_emits_research_progress_before_generation"): "streaming",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py", "test_rag_streaming_generation_uses_shared_resolved_payload"): "streaming",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py", "test_rag_streaming_partial_provider_error_is_terminal_and_sanitized"): "streaming",
    ("tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py", "test_rag_streaming_preserves_delta_and_claim_events"): "streaming",
    ("tldw_Server_API/tests/RAG_NEW/unit/test_batch_round2_flags.py", "test_unified_batch_pipeline_forwards_round2_flags"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/unit/test_hyde.py", "test_batch_clustering_runtime_remote_local_uses_selected_deployment"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/unit/test_hyde.py", "test_batch_clustering_uses_runtime_credentials_for_hosted_embeddings"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py", "test_checkpoint_batch_cancellation_waits_for_started_queries"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py", "test_simple_wrapper_forwards_runtime_without_changing_legacy_default"): "simple_search",
    ("tldw_Server_API/tests/RAG_NEW/unit/test_retrieval_plan_usage.py", "test_unified_batch_pipeline_fallback_rewrites_plan_query_per_item"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/unit/test_retrieval_plan_usage.py", "test_unified_batch_pipeline_rewrites_plan_query_per_item"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/unit/test_retrieval_plan_usage.py", "test_unified_batch_pipeline_rewrites_resolved_request_per_item"): "batch",
    ("tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_golden.py", "_g10"): "rerun",
}


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> dotted module path, for ``from a import b``, ``import a.b as c`` and ``c = import_module("a.b")``."""
    out: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and not n.level:
            out.update({a.asname or a.name: f"{n.module}.{a.name}" for a in n.names})
        elif isinstance(n, ast.Import):
            out.update({a.asname: a.name for a in n.names if a.asname})
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            v = n.value
            if isinstance(v, ast.Call) and _call_name(v.func) == "import_module" and v.args:
                if isinstance(v.args[0], ast.Constant) and isinstance(v.args[0].value, str):
                    out[n.targets[0].id] = v.args[0].value
    return out


def _entry_patches(tree: ast.AST) -> list[tuple[str, str | None, str]]:
    """``(innermost enclosing function, dotted module or None, name)`` for each patch of an entry name."""
    aliases = _import_aliases(tree)
    out: list[tuple[str, str | None, str]] = []

    def visit(node: ast.AST, fn: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = node.name
        if isinstance(node, ast.Call) and (_call_name(node.func) in PATCHERS or _call_name(node.func) == "object") and node.args:
            first = node.args[0]
            second = node.args[1] if len(node.args) > 1 else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                mod, _, name = first.value.rpartition(".")
                if name in ENTRY_NAMES:
                    out.append((fn, mod, name))
                elif isinstance(second, ast.Constant) and second.value in ENTRY_NAMES:
                    out.append((fn, first.value, str(second.value)))
            elif isinstance(second, ast.Constant) and second.value in ENTRY_NAMES:
                out.append((fn, aliases.get(ast.unparse(first)), str(second.value)))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr in ENTRY_NAMES:
                    out.append((fn, aliases.get(ast.unparse(t.value)), t.attr))
        for child in ast.iter_child_nodes(node):
            visit(child, fn)

    visit(tree, "<module>")
    return out


def _module_file(repo_root: Path, dotted: str) -> Path | None:
    base = repo_root.joinpath(*dotted.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _loads_name(path: Path, name: str) -> bool:
    """Whether the module reads ``name`` (a Name load or attribute access); an import binding alone does not count."""
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load):
            return True
        if isinstance(n, ast.Attribute) and n.attr == name and isinstance(n.ctx, ast.Load):
            return True
    return False


def entry_patch_violations(test_roots: Iterable[Path], repo_root: Path) -> list[str]:
    """Patches of an entry name on a module that no longer loads it (the patch would be inert).

    ``unified_pipeline`` itself is skipped: it defines both names, and its shim patches are pinned by
    ``PIPELINE_ENTRY_PATCH_ALLOWLIST``.
    """
    out: list[str] = []
    for path in _files(test_roots):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not any(name in text for name in ENTRY_NAMES):
            continue
        for fn, mod, name in _entry_patches(ast.parse(text, filename=str(path))):
            if mod == UNIFIED_DOTTED:
                continue
            target = _module_file(repo_root, mod) if mod else None
            if target is None:
                out.append(f"{path.name}:{fn}: cannot resolve the module patched for {name} ({mod})")
            elif not _loads_name(target, name):
                out.append(f"{path.name}:{fn}: {mod} no longer loads {name}")
    return out


def pipeline_entry_patch_sites(test_roots: Iterable[Path], repo_root: Path) -> set[tuple[str, str]]:
    """``(test file, function)`` of every patch of ``unified_pipeline.unified_rag_pipeline``."""
    out: set[tuple[str, str]] = set()
    for path in _files(test_roots):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "unified_rag_pipeline" not in text:
            continue
        for fn, mod, name in _entry_patches(ast.parse(text, filename=str(path))):
            if mod == UNIFIED_DOTTED and name == "unified_rag_pipeline":
                out.add((path.relative_to(repo_root).as_posix(), fn))
    return out


ENTRY_TEST_ROOTS = SCAN_ROOTS[:2]


def test_entry_point_patches_target_a_module_that_calls_it() -> None:
    assert entry_patch_violations(ENTRY_TEST_ROOTS, REPO_ROOT) == []


def test_entry_point_patch_on_a_module_that_dropped_the_call_is_reported(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    mod = tmp_path / "pkg" / "mod.py"
    mod.write_text("def f():\n    return 1\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_a.py").write_text(
        "import pkg.mod as m\n\n\ndef test_a(monkeypatch):\n    monkeypatch.setattr('pkg.mod.run_rag_pipeline', 1)\n"
        "\n\ndef test_b(monkeypatch):\n    monkeypatch.setattr(m, 'unified_rag_pipeline', 1)\n"
    )
    assert len(entry_patch_violations([tests], tmp_path)) == 2
    mod.write_text("from x import run_rag_pipeline, unified_rag_pipeline\n")  # imported, never used
    assert len(entry_patch_violations([tests], tmp_path)) == 2
    mod.write_text("from x import run_rag_pipeline, unified_rag_pipeline\n\n\nasync def f():\n    await run_rag_pipeline(1)\n    return x.unified_rag_pipeline\n")
    assert entry_patch_violations([tests], tmp_path) == []


def test_pipeline_entry_patches_are_allow_listed() -> None:
    sites = pipeline_entry_patch_sites(ENTRY_TEST_ROOTS, REPO_ROOT)
    assert sites == set(PIPELINE_ENTRY_PATCH_ALLOWLIST)
    assert set(PIPELINE_ENTRY_PATCH_ALLOWLIST.values()) <= {"batch", "streaming", "sharing", "rerun", "simple_search"}
