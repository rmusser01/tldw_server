from __future__ import annotations

import ast
import asyncio
import dataclasses
import inspect
import json
import math
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SELECTORS_ROOT = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "selectors"
FETCHERS_PATH = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Watchlists" / "fetchers.py"
ENDPOINT_PATH = REPO_ROOT / "tldw_Server_API" / "app" / "api" / "v1" / "endpoints" / "watchlists.py"
ARTICLE_PATH = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "Article_Extractor_Lib.py"
HANDLERS_PATH = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "handlers.py"
ENHANCED_SCRAPER_PATH = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "enhanced_web_scraping.py"
EXTRACTION_ROOT = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "extraction"
EXTRACTION_DEPENDENCIES_PATH = (
    REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "extraction" / "dependencies.py"
)
METRICS_PATH = EXTRACTION_ROOT / "metrics.py"

EXPECTED_IMPORTS = {
    "__init__.py": {".caches", ".schema"},
    "caches.py": {"__future__", "collections", "os", "threading", "typing"},
    "engine.py": {
        ".caches",
        "__future__",
        "collections.abc",
        "cssselect",
        "functools",
        "loguru",
        "lxml.cssselect",
        "lxml.etree",
        "lxml.html",
        "os",
        "re",
        "typing",
    },
    "schema.py": {
        "..safe_regex",
        ".engine",
        "__future__",
        "codecs",
        "collections.abc",
        "contextlib",
        "contextvars",
        "dataclasses",
        "dateutil",
        "datetime",
        "email.utils",
        "lxml",
        "lxml.html",
        "re",
        "string",
        "typing",
        "urllib.parse",
    },
}

BANNED_UPWARD_IMPORT_PARTS = {
    "Article_Extractor_Lib",
    "Enhanced_Web_Scraping",
    "Watchlists",
    "WebSearch",
    "enhanced_web_scraping",
    "extraction",
    "handlers",
    "orchestration",
    "policy",
    "preflight",
    "routing",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    return imported


def _defined_functions(path: Path) -> set[str]:
    return {node.name for node in _tree(path).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _imported_names(path: Path, module: str) -> set[str]:
    names: set[str] = set()
    for node in _tree(path).body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def _imported_names_at_any_scope(path: Path, module: str) -> set[str]:
    return {
        alias.name
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def _function(path: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return next(
        node
        for node in _tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _called_names(node: ast.AST) -> set[str]:
    return {
        child.func.id for child in ast.walk(node) if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


_FORBIDDEN_CANCELLATION_NAMES = {"BaseException", "CancelledError"}
_TYPE_PARAMETER_NODES = tuple(
    getattr(ast, name) for name in ("ParamSpec", "TypeVar", "TypeVarTuple") if hasattr(ast, name)
)


def _forbidden_cancellation_reference(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and node.id in _FORBIDDEN_CANCELLATION_NAMES:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_CANCELLATION_NAMES:
        return ast.unparse(node)
    return None


def _forbidden_bound_identifier(node: ast.AST) -> str | None:
    candidate: str | None = None
    if isinstance(node, ast.alias):
        candidate = node.asname
    elif isinstance(node, ast.arg):
        candidate = node.arg
    elif isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.ExceptHandler,
            ast.MatchAs,
            ast.MatchStar,
        ),
    ):
        candidate = node.name
    elif isinstance(node, ast.MatchMapping):
        candidate = node.rest
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        candidate = next((name for name in node.names if name in _FORBIDDEN_CANCELLATION_NAMES), None)
    elif _TYPE_PARAMETER_NODES and isinstance(node, _TYPE_PARAMETER_NODES):
        candidate = node.name
    return candidate if candidate in _FORBIDDEN_CANCELLATION_NAMES else None


def _forbidden_cancellation_violations(tree: ast.Module) -> list[str]:
    """Reject bare handlers and explicit cancellation references."""
    violations: list[str] = []
    for node in ast.walk(tree):
        violation: str | None = None
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            violation = "bare except"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for imported in node.names:
                if imported.name.rsplit(".", 1)[-1] in _FORBIDDEN_CANCELLATION_NAMES:
                    violation = f"import {imported.name.rsplit('.', 1)[-1]}"
                    break
        if violation is None:
            bound_identifier = _forbidden_bound_identifier(node)
            if bound_identifier is not None:
                violation = f"bound identifier {bound_identifier}"
        if violation is None:
            reference = _forbidden_cancellation_reference(node)
            if reference is not None:
                violation = f"reference {reference}"
        if violation is not None and violation not in violations:
            violations.append(violation)
    return violations


_RAW_METRIC_SINK_NAMES = {"increment_counter", "log_counter", "observe_histogram"}
_RAW_METRICS_MODULE_PREFIX = "tldw_Server_API.app.core.Metrics"


def _metric_boundary_bypasses(paths: list[Path]) -> list[str]:
    """Find extraction modules that emit metrics without the canonical boundary."""
    violations: list[str] = []
    for path in paths:
        if path == METRICS_PATH:
            continue
        tree = _tree(path)
        raw_aliases: set[str] = set()
        callable_aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(_RAW_METRICS_MODULE_PREFIX):
                raw_aliases.update(alias.asname or alias.name for alias in node.names)
                violations.append(f"{path.name}: raw metrics import")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(_RAW_METRICS_MODULE_PREFIX):
                        raw_aliases.add(alias.asname or alias.name.split(".")[0])
                        violations.append(f"{path.name}: raw metrics import")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if isinstance(value, ast.Attribute) and value.attr in _RAW_METRIC_SINK_NAMES:
                    callable_aliases.update(target.id for target in targets if isinstance(target, ast.Name))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in _RAW_METRIC_SINK_NAMES:
                    violations.append(f"{path.name}: direct {node.func.attr} call")
                elif isinstance(node.func, ast.Name) and node.func.id in raw_aliases | callable_aliases:
                    violations.append(f"{path.name}: aliased metric sink call")
    return violations


def test_selector_package_has_the_approved_files() -> None:
    assert {path.name for path in SELECTORS_ROOT.glob("*.py")} == set(EXPECTED_IMPORTS)


def test_selector_files_have_an_explicit_dependency_inventory() -> None:
    actual = {path.name: _imports(path) for path in sorted(SELECTORS_ROOT.glob("*.py"))}

    assert actual == EXPECTED_IMPORTS


def test_selector_package_has_no_upward_application_dependencies() -> None:
    violations: list[str] = []
    for path in sorted(SELECTORS_ROOT.glob("*.py")):
        for imported in sorted(_imports(path)):
            if any(part in imported.split(".") for part in BANNED_UPWARD_IMPORT_PARTS):
                violations.append(f"{path.name}: {imported}")

    assert violations == []


def test_cache_state_has_one_canonical_owner() -> None:
    cache_tokens = {
        "_CSS_SELECTOR_CACHE",
        "_SELECTOR_CACHE_LOCK",
        "_XPATH_SELECTOR_CACHE",
    }
    owners: dict[str, list[str]] = {token: [] for token in cache_tokens}
    inspected = [*SELECTORS_ROOT.glob("*.py"), FETCHERS_PATH, ARTICLE_PATH]
    for path in inspected:
        text = path.read_text(encoding="utf-8")
        for token in cache_tokens:
            if token in text:
                owners[token].append(path.name)

    assert owners == {token: ["caches.py"] for token in cache_tokens}


def test_watchlists_has_direct_canonical_imports_without_selector_bodies() -> None:
    forbidden_definitions = {
        "clear_selector_caches",
        "extract_schema_fields",
        "get_selector_cache_stats",
        "reload_selector_guardrails_from_env",
        "validate_selector_rules",
    }
    assert _defined_functions(FETCHERS_PATH).isdisjoint(forbidden_definitions)

    facade_module = "tldw_Server_API.app.core.Web_Scraping.selectors"
    engine_module = f"{facade_module}.engine"
    schema_module = f"{facade_module}.schema"
    assert _imported_names(FETCHERS_PATH, facade_module) == {
        "clear_selector_caches",
        "extract_schema_fields",
        "get_selector_cache_stats",
        "validate_selector_rules",
    }
    assert _imported_names(FETCHERS_PATH, engine_module) == {
        "coerce_value",
        "ensure_sequence",
        "extract_value",
        "reload_selector_guardrails_from_env",
        "select_nodes",
    }
    assert _imported_names(FETCHERS_PATH, schema_module) == {"normalize_datetime"}

    private_imports = {
        alias.name
        for node in ast.walk(_tree(FETCHERS_PATH))
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(facade_module)
        for alias in node.names
        if alias.name.startswith("_")
    }
    assert private_imports == set()


def test_article_selector_responsibilities_import_the_canonical_facade() -> None:
    facade_module = "tldw_Server_API.app.core.Web_Scraping.selectors"
    article_selector_names = {
        "clear_selector_caches",
        "get_selector_cache_stats",
    }
    extraction_selector_names = {
        "extract_schema_fields",
        "validate_selector_rules",
    }

    assert _imported_names(ARTICLE_PATH, facade_module) == article_selector_names
    assert _imported_names_at_any_scope(EXTRACTION_DEPENDENCIES_PATH, facade_module) == extraction_selector_names

    violations = []
    for path in (ARTICLE_PATH, EXTRACTION_DEPENDENCIES_PATH):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            imported = {alias.name for alias in node.names}
            if ".Watchlists" in module and imported & (article_selector_names | extraction_selector_names):
                violations.append({"module": module, "names": sorted(imported)})
    assert violations == []


def test_phase4b_moved_consumers_import_canonical_content_and_extraction_facades() -> None:
    content_module = "tldw_Server_API.app.core.Web_Scraping.content"
    extraction_module = "tldw_Server_API.app.core.Web_Scraping.extraction"
    orchestration_module = "tldw_Server_API.app.core.Web_Scraping.orchestration"
    legacy_module = "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib"

    assert _imported_names_at_any_scope(HANDLERS_PATH, content_module) == {"convert_html_to_markdown"}
    assert _imported_names_at_any_scope(HANDLERS_PATH, extraction_module) == {"extract_article_data_from_html"}
    assert _imported_names_at_any_scope(ENHANCED_SCRAPER_PATH, content_module) == {"convert_html_to_markdown"}
    assert _imported_names_at_any_scope(ENHANCED_SCRAPER_PATH, extraction_module) == {"extract_article_with_pipeline"}
    assert _imported_names_at_any_scope(HANDLERS_PATH, legacy_module) == set()
    generic_handler = _function(HANDLERS_PATH, "handle_generic_html")
    handler_imports = {
        node.module: {alias.name for alias in node.names}
        for node in generic_handler.body
        if isinstance(node, ast.ImportFrom) and node.module in {content_module, extraction_module}
    }
    assert handler_imports == {
        content_module: {"convert_html_to_markdown"},
        extraction_module: {"extract_article_data_from_html"},
    }
    canonical_handler_import_locations = {
        (node.lineno, node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(_tree(HANDLERS_PATH))
        if isinstance(node, ast.ImportFrom) and node.module in {content_module, extraction_module}
    }
    assert len(canonical_handler_import_locations) == 2
    assert canonical_handler_import_locations == {
        (node.lineno, node.module, tuple(alias.name for alias in node.names))
        for node in generic_handler.body
        if isinstance(node, ast.ImportFrom) and node.module in {content_module, extraction_module}
    }
    enhanced_tree = _tree(ENHANCED_SCRAPER_PATH)
    assert _imported_names_at_any_scope(ENHANCED_SCRAPER_PATH, orchestration_module) == {"scrape_article"}
    legacy_imports = [
        node for node in ast.walk(enhanced_tree) if isinstance(node, ast.ImportFrom) and node.module == legacy_module
    ]
    assert legacy_imports == []
    manager = next(
        node for node in enhanced_tree.body if isinstance(node, ast.ClassDef) and node.name == "ScrapingJobQueue"
    )
    execute_job = next(
        node for node in manager.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_execute_job"
    )
    parent_scraper_branch = next(
        node
        for node in execute_job.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "self"
        and node.test.attr == "parent_scraper"
    )
    fallback_imports = [node for node in parent_scraper_branch.orelse if isinstance(node, ast.ImportFrom)]
    assert len(fallback_imports) == 1
    assert fallback_imports[0].module == orchestration_module
    assert [alias.name for alias in fallback_imports[0].names] == ["scrape_article"]
    fallback_calls = [
        statement.value
        for statement in parent_scraper_branch.orelse
        if isinstance(statement, ast.Return)
        and isinstance(statement.value, ast.Await)
        and isinstance(statement.value.value, ast.Call)
        and isinstance(statement.value.value.func, ast.Name)
        and statement.value.value.func.id == "scrape_article"
    ]
    assert len(fallback_calls) == 1
    fallback_call = fallback_calls[0].value
    assert len(fallback_call.args) == 1
    assert isinstance(fallback_call.args[0], ast.Attribute)
    assert isinstance(fallback_call.args[0].value, ast.Name)
    assert fallback_call.args[0].value.id == "job"
    assert fallback_call.args[0].attr == "url"
    assert [keyword.arg for keyword in fallback_call.keywords] == ["custom_cookies", "allow_llm_extraction"]
    expected_metadata_gets = {
        "custom_cookies": ("custom_cookies", None),
        "allow_llm_extraction": ("allow_llm_extraction", True),
    }
    for keyword in fallback_call.keywords:
        assert keyword.arg is not None
        assert isinstance(keyword.value, ast.Call)
        assert isinstance(keyword.value.func, ast.Attribute)
        assert keyword.value.func.attr == "get"
        assert isinstance(keyword.value.func.value, ast.Attribute)
        assert keyword.value.func.value.attr == "metadata"
        assert isinstance(keyword.value.func.value.value, ast.Name)
        assert keyword.value.func.value.value.id == "job"
        expected_key, expected_default = expected_metadata_gets[keyword.arg]
        assert len(keyword.value.args) == 1 + (expected_default is not None)
        assert isinstance(keyword.value.args[0], ast.Constant)
        assert keyword.value.args[0].value == expected_key
        if expected_default is not None:
            assert isinstance(keyword.value.args[1], ast.Constant)
            assert keyword.value.args[1].value is expected_default


def test_phase4b_enhanced_no_parent_fallback_forwards_canonical_job_arguments(monkeypatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping import orchestration
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ScrapingJob, ScrapingJobQueue

    calls: list[tuple[str, dict[str, Any]]] = []

    async def fallback(url: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((url, kwargs))
        return {"url": url, "extraction_successful": True}

    monkeypatch.setattr(orchestration, "scrape_article", fallback)
    job = ScrapingJob(
        job_id="fallback-job",
        url="https://example.com/fallback",
        method="auto",
        metadata={"custom_cookies": {"session": "value"}, "allow_llm_extraction": False},
    )

    result = asyncio.run(ScrapingJobQueue(parent_scraper=None)._execute_job(job))

    assert calls == [
        (
            "https://example.com/fallback",
            {"custom_cookies": {"session": "value"}, "allow_llm_extraction": False},
        )
    ]
    assert result == {"url": "https://example.com/fallback", "extraction_successful": True}


def test_phase4b_crawl_bound_article_helper_keeps_its_async_surface_and_canonical_dependency() -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article

    extraction_module = "tldw_Server_API.app.core.Web_Scraping.extraction"
    helper = _function(ARTICLE_PATH, "scrape_article_async")

    assert isinstance(helper, ast.AsyncFunctionDef)
    assert inspect.signature(article.scrape_article_async) == inspect.Signature(
        parameters=[
            inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("url", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter(
                "allow_llm_extraction",
                inspect.Parameter.KEYWORD_ONLY,
                default=True,
                annotation=bool,
            ),
        ],
        return_annotation=dict[str, Any],
    )
    assert inspect.get_annotations(article.scrape_article_async, eval_str=True) == {
        "url": str,
        "allow_llm_extraction": bool,
        "return": dict[str, Any],
    }
    assert "extract_article_with_pipeline" in _imported_names(ARTICLE_PATH, extraction_module)
    assert "run_extraction_in_thread" in _imported_names(
        ARTICLE_PATH,
        "tldw_Server_API.app.core.Web_Scraping.extraction_async",
    )
    assert any(isinstance(node, ast.Name) and node.id == "extract_article_with_pipeline" for node in ast.walk(helper))
    assert "run_extraction_in_thread" in _called_names(helper)


def test_phase4b_crawl_bound_article_helper_forwards_guarded_html_to_canonical_extraction(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article

    acquire = AsyncMock(
        return_value="<html><head><title>Browser title</title></head><article>Body</article></html>"
    )

    class GuardedBrowser:
        """Provide deterministic rendered HTML without a native browser."""

        def __init__(self, **_kwargs: object) -> None:
            self.acquire = acquire

    calls: list[tuple[str, str, dict[str, bool]]] = []
    event_loop_thread = threading.get_ident()
    extraction_threads: list[int] = []

    def extract(html: str, url: str, **kwargs: bool) -> dict[str, Any]:
        extraction_threads.append(threading.get_ident())
        calls.append((html, url, kwargs))
        return {"title": "N/A", "content": "Body", "extraction_successful": True}

    monkeypatch.setattr(article, "extract_article_with_pipeline", extract)
    monkeypatch.setattr(article, "convert_html_to_markdown", lambda content: f"markdown:{content}")
    monkeypatch.setattr(article, "GuardedArticleBrowser", GuardedBrowser)

    result = asyncio.run(
        article.scrape_article_async(None, "https://example.com/article", allow_llm_extraction=False)
    )

    assert calls == [
        (
            "<html><head><title>Browser title</title></head><article>Body</article></html>",
            "https://example.com/article",
            {"allow_llm_extraction": False},
        )
    ]
    assert extraction_threads and extraction_threads[0] != event_loop_thread
    assert result == {"title": "Browser title", "content": "markdown:Body", "extraction_successful": True}
    acquire.assert_awaited_once()


def test_phase4b_crawl_bound_article_helper_returns_sanitized_guarded_failure(monkeypatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article

    class GuardedBrowser:
        """Raise the canonical bounded browser failure."""

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def acquire(self, _url: str, _profile: object) -> str:
            raise article.ArticleFailure("browser_error", "navigation")

    monkeypatch.setattr(article, "GuardedArticleBrowser", GuardedBrowser)
    result = asyncio.run(article.scrape_article_async(None, "https://example.com/article"))

    assert result == {
        "url": "https://example.com/article",
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": "browser_error",
    }


def test_phase4b_handler_import_is_lazy_and_cycle_free() -> None:
    script = """
import json
import sys

import tldw_Server_API.app.core.Web_Scraping.handlers

blocked = (
    "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
    "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping",
    "tldw_Server_API.app.core.Web_Scraping.extraction",
    "tldw_Server_API.app.core.Metrics",
    "tldw_Server_API.app.core.Metrics.metrics_logger",
)
print("HANDLER_IMPORTS=" + json.dumps([name for name in blocked if name in sys.modules]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    marker = next(line for line in result.stdout.splitlines() if line.startswith("HANDLER_IMPORTS="))

    assert json.loads(marker.removeprefix("HANDLER_IMPORTS=")) == []


def test_phase4b_metric_contract_rejects_sensitive_or_high_cardinality_values() -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction import metrics

    invalid_events = [
        ("extraction_strategy_total", {"strategy": "https://example.com", "status": "success"}),
        ("extraction_retry_total", {"strategy": "regex", "attempt": "secret-token"}),
        ("extraction_cluster_total", {"status": "RuntimeError: raw error"}),
        ("extraction_cluster_cache_total", {"cache": "embedding", "url": "https://example.com"}),
        ("llm_tokens_used_total", {"provider": "api-key-secret", "model": "payload-hash", "type": "prompt"}),
        (
            "llm_tokens_used_total_by_operation",
            {"provider": "openai", "model": "configured", "type": "prompt", "operation": "https://example.com"},
        ),
    ]
    for name, labels in invalid_events:
        with pytest.raises(ValueError):
            metrics.validate_metric(name, labels=labels)


def test_phase4b_llm_provider_labels_have_one_frozen_contract() -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction import metrics
    from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import llm

    contract = metrics.METRIC_LABEL_CONTRACT
    provider_labels = metrics.LLM_PROVIDER_LABEL_VALUES

    assert contract["llm_tokens_used_total"]["provider"] is provider_labels
    assert contract["llm_tokens_used_total_by_operation"]["provider"] is provider_labels
    assert llm.LLM_PROVIDER_LABEL_VALUES is provider_labels
    assert {llm._metric_provider(provider) for provider in provider_labels} == provider_labels
    assert llm._metric_provider("future-provider") == "other"

    with pytest.raises(TypeError):
        contract["future_metric"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        contract["llm_tokens_used_total"]["provider"] = frozenset({"other"})  # type: ignore[index]


def test_extraction_executor_lifecycle_metric_has_bounded_outcomes() -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction import metrics

    assert metrics.METRIC_LABEL_CONTRACT["extraction_executor_total"] == {
        "outcome": frozenset({"queued", "running", "saturated", "cancelled", "discarded"})
    }


def test_phase4b_only_the_canonical_metric_boundary_owns_metric_sinks() -> None:
    assert METRICS_PATH.is_file()
    assert _metric_boundary_bypasses(sorted(EXTRACTION_ROOT.rglob("*.py"))) == []


def test_phase4b_metric_boundary_rejects_unlisted_emitters_and_bypasses(tmp_path: Path) -> None:
    cases = {
        "new_emitter.py": """
from tldw_Server_API.app.core.Metrics import increment_counter
increment_counter("future_metric", labels={"url": "https://example.com"})
""",
        "alias.py": """
def emit(dependencies):
    sink = dependencies.increment_counter
    sink("article_extracted", labels={"success": "true"})
""",
        "late_alias.py": """
sink = None

def emit():
    sink("article_extracted", labels={"success": "true"})

def configure(dependencies):
    global sink
    sink = dependencies.increment_counter
""",
        "wrapper.py": """
def emit(dependencies):
    dependencies.observe_histogram(
        "extraction_strategy_duration_seconds", 1.0, labels={"strategy": "jsonld", "status": "success"}
    )
""",
    }
    for filename, source in cases.items():
        path = tmp_path / filename
        path.write_text(source, encoding="utf-8")
        assert _metric_boundary_bypasses([path])


def test_phase4b_metric_contract_covers_every_production_emission_and_allowed_value(monkeypatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction import caches, metrics, pipeline
    from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies
    from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import cluster, llm, trafilatura

    events: list[tuple[str, float | None, dict[str, str]]] = []

    def record(name: str, value: float | None = None, labels: dict[str, str] | None = None) -> None:
        events.append((name, value, dict(labels or {})))

    dependencies = dataclasses.replace(
        build_default_dependencies(),
        increment_counter=record,
        log_counter=record,
        observe_histogram=record,
    )

    contract = metrics.METRIC_LABEL_CONTRACT
    strategies = contract["extraction_strategy_total"]["strategy"]
    statuses = contract["extraction_strategy_total"]["status"]
    for strategy in strategies:
        for status in statuses:
            pipeline._trace_entry(dependencies, strategy, status, "test")
            pipeline._record_strategy_metrics(
                dependencies,
                strategy,
                status,
                0.0,
                {"content": "metric coverage"},
            )

    def always_fails() -> dict[str, Any]:
        raise RuntimeError("retry")

    monkeypatch.setattr(pipeline, "_extractor_retry_settings", lambda: (4, 0.0, 0.0))
    pipeline._run_with_retries(always_fails, strategy="unrecognized", dependencies=dependencies)
    monkeypatch.setattr(pipeline, "_extractor_retry_settings", lambda: (1, 0.0, 0.0))
    for strategy in strategies - {"unknown"}:
        pipeline._run_with_retries(always_fails, strategy=strategy, dependencies=dependencies)

    def failing_provider(**_kwargs: Any) -> None:
        raise RuntimeError("provider retry")

    retrying_dependencies = dataclasses.replace(dependencies, perform_chat_api_call=failing_provider)
    monkeypatch.setattr(llm, "_retry_settings", lambda: (4, 0.0, 0.0))
    llm.call_llm_provider(
        provider="openai",
        settings={},
        messages=[],
        app_config=None,
        dependencies=retrying_dependencies,
        stage="extraction",
        url="https://example.com",
    )

    caches.clear_extraction_caches()
    assert caches._cluster_cache_get("metric-cache", increment_counter=record) is None
    caches._cluster_cache_put("metric-cache", [1.0])
    assert caches._cluster_cache_get("metric-cache", increment_counter=record) == [1.0]
    for status in contract["extraction_cluster_total"]["status"]:
        cluster._increment_counter(dependencies, "extraction_cluster_total", labels={"status": status})

    monkeypatch.setattr(trafilatura.trafilatura, "extract_metadata", lambda _html: None)
    monkeypatch.setattr(trafilatura, "log_counter", record)
    monkeypatch.setattr(trafilatura.trafilatura, "extract", lambda _html, **_kwargs: "article body")
    trafilatura.extract_with_trafilatura("<html></html>", "https://example.com/success")
    monkeypatch.setattr(trafilatura.trafilatura, "extract", lambda _html, **_kwargs: None)
    trafilatura.extract_with_trafilatura("<html></html>", "https://example.com/failure")

    providers = contract["llm_tokens_used_total"]["provider"]
    for provider in providers:
        input_provider = provider if provider != "other" else "https://user:secret@example.com"
        llm.record_llm_usage_metrics(
            {"prompt_tokens": 1, "completion_tokens": 1},
            provider=input_provider,
            model="unbounded-model-payload",
            dependencies=dependencies,
        )

    for outcome in contract["extraction_executor_total"]["outcome"]:
        metrics.emit_callback_counter(
            record,
            "extraction_executor_total",
            labels={"outcome": outcome},
        )

    assert {name for name, _value, _labels in events} == set(contract)
    for name, value, labels in events:
        metrics.validate_metric(name, value=value, labels=labels)
        if value is not None:
            assert math.isfinite(value)
    for metric_name, labels_contract in contract.items():
        metric_events = [labels for name, _value, labels in events if name == metric_name]
        for label_name, allowed_values in labels_contract.items():
            assert {labels[label_name] for labels in metric_events} == allowed_values


def test_phase4b_metric_boundary_rejects_uncontracted_and_nonfinite_values() -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction import metrics

    with pytest.raises(ValueError):
        metrics.validate_metric("future_uncontracted_metric", labels={})
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            metrics.validate_metric(
                "extraction_strategy_duration_seconds",
                value=value,
                labels={"strategy": "jsonld", "status": "success"},
            )


def test_phase4b_metric_boundary_blocks_an_unsafe_real_pipeline_branch() -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction import metrics, pipeline
    from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies

    events: list[tuple[str, float | None, dict[str, str]]] = []

    def record(name: str, value: float | None = None, labels: dict[str, str] | None = None) -> None:
        events.append((name, value, dict(labels or {})))

    dependencies = dataclasses.replace(build_default_dependencies(), log_counter=record)
    pipeline._trace_entry(dependencies, "jsonld", "https://example.com/raw-error", "test")

    assert events == []
    with pytest.raises(ValueError):
        metrics.validate_metric(
            "extraction_strategy_total",
            labels={"strategy": "jsonld", "status": "https://example.com/raw-error"},
        )


def test_phase4b_extraction_never_catches_or_aliases_cancellation_signals() -> None:
    violations: list[str] = []
    for path in EXTRACTION_ROOT.rglob("*.py"):
        violations.extend(
            f"{path.relative_to(EXTRACTION_ROOT)}: {violation}"
            for violation in _forbidden_cancellation_violations(_tree(path))
        )

    assert violations == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            """
try:
    operation()
except Exception:
    recover()
""",
            [],
        ),
        (
            """
try:
    operation()
except (ValueError, RuntimeError):
    recover()
""",
            [],
        ),
        (
            """
try:
    operation()
except:
    raise
""",
            ["bare except"],
        ),
        (
            """
try:
    operation()
except BaseException:
    raise
""",
            ["reference BaseException"],
        ),
        (
            """
import builtins
try:
    operation()
except builtins.BaseException:
    raise
""",
            ["reference builtins.BaseException"],
        ),
        (
            """
import asyncio
try:
    operation()
except asyncio.CancelledError:
    raise
""",
            ["reference asyncio.CancelledError"],
        ),
        (
            """
from asyncio import CancelledError as CE
try:
    operation()
except CE:
    raise
""",
            ["import CancelledError"],
        ),
        (
            """
from builtins import BaseException as RootError
try:
    operation()
except RootError:
    raise
""",
            ["import BaseException"],
        ),
        (
            """
import asyncio
CE = asyncio.CancelledError
""",
            ["reference asyncio.CancelledError"],
        ),
        (
            """
import asyncio
CE = Other = asyncio.CancelledError
""",
            ["reference asyncio.CancelledError"],
        ),
        (
            """
import asyncio
RECOVERABLE = (ValueError, asyncio.CancelledError)
""",
            ["reference asyncio.CancelledError"],
        ),
        (
            """
import asyncio
ae = asyncio.exceptions
try:
    operation()
except ae.CancelledError:
    raise
""",
            ["reference ae.CancelledError"],
        ),
        (
            """
CancelledError = ValueError
try:
    operation()
except CancelledError:
    recover()
""",
            ["reference CancelledError"],
        ),
        (
            """
import asyncio
def extract(error_type=asyncio.CancelledError):
    return error_type
""",
            ["reference asyncio.CancelledError"],
        ),
        (
            """
import asyncio as CancelledError
""",
            ["bound identifier CancelledError"],
        ),
        (
            """
def BaseException():
    pass
""",
            ["bound identifier BaseException"],
        ),
        (
            """
class CancelledError:
    pass
""",
            ["bound identifier CancelledError"],
        ),
        (
            """
def extract(BaseException):
    return BaseException
""",
            ["bound identifier BaseException", "reference BaseException"],
        ),
        (
            """
try:
    operation()
except ValueError as BaseException:
    recover()
""",
            ["bound identifier BaseException"],
        ),
        (
            """
match value:
    case BaseException:
        pass
""",
            ["bound identifier BaseException"],
        ),
        (
            """
match value:
    case [*CancelledError]:
        pass
""",
            ["bound identifier CancelledError"],
        ),
        (
            """
match value:
    case {"error": _, **BaseException}:
        pass
""",
            ["bound identifier BaseException"],
        ),
        (
            """
def extract():
    global CancelledError
""",
            ["bound identifier CancelledError"],
        ),
        pytest.param(
            """
def extract[CancelledError]():
    pass
""",
            ["bound identifier CancelledError"],
            marks=pytest.mark.skipif(sys.version_info < (3, 12), reason="type-parameter syntax requires Python 3.12"),
        ),
    ],
    ids=(
        "ordinary-exception",
        "ordinary-exception-tuple",
        "bare-handler",
        "base-exception-handler",
        "qualified-base-exception-handler",
        "cancelled-error-handler",
        "cancelled-error-import",
        "base-exception-import",
        "direct-alias",
        "chained-alias",
        "tuple-alias",
        "module-alias-handler",
        "shadowed-cancelled-error-handler",
        "default-argument-alias",
        "import-bound-name",
        "function-bound-name",
        "class-bound-name",
        "parameter-bound-name",
        "exception-target-bound-name",
        "match-capture-bound-name",
        "match-star-bound-name",
        "match-mapping-bound-name",
        "global-bound-name",
        "type-parameter-bound-name",
    ),
)
def test_phase4b_cancellation_guard_enforces_strict_package_rule(
    source: str,
    expected: list[str],
) -> None:
    assert _forbidden_cancellation_violations(ast.parse(source)) == expected


def test_article_cache_stats_do_not_load_watchlists_in_a_fresh_process() -> None:
    script = """
import json
import sys

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article

article.get_extraction_cache_stats()
loaded = sorted(
    name
    for name in sys.modules
    if name == "tldw_Server_API.app.core.Watchlists"
    or name.startswith("tldw_Server_API.app.core.Watchlists.")
)
print("SELECTOR_ARCH=" + json.dumps(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    marker = next(line for line in result.stdout.splitlines() if line.startswith("SELECTOR_ARCH="))

    assert json.loads(marker.removeprefix("SELECTOR_ARCH=")) == []


def test_watchlists_endpoint_imports_validation_from_canonical_facade() -> None:
    facade_module = "tldw_Server_API.app.core.Web_Scraping.selectors"
    fetchers_module = "tldw_Server_API.app.core.Watchlists.fetchers"

    assert _imported_names(ENDPOINT_PATH, facade_module) == {"validate_selector_rules"}
    assert _imported_names(ENDPOINT_PATH, fetchers_module) == {
        "fetch_rss_feed",
        "fetch_site_items_with_rules",
    }
