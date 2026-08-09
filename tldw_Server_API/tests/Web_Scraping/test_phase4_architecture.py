from __future__ import annotations

import ast
import asyncio
import dataclasses
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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
METRIC_SOURCE_PATHS = (
    EXTRACTION_ROOT / "caches.py",
    EXTRACTION_ROOT / "pipeline.py",
    EXTRACTION_ROOT / "strategies" / "cluster.py",
    EXTRACTION_ROOT / "strategies" / "llm.py",
    EXTRACTION_ROOT / "strategies" / "trafilatura.py",
)
_METRIC_EMISSION_INVENTORY = {
    "caches.py": {76: "extraction_cluster_cache_total", 84: "extraction_cluster_cache_total"},
    "pipeline.py": {
        132: "extraction_strategy_total",
        152: "extraction_strategy_duration_seconds",
        164: "extraction_content_length_bytes",
        274: "extraction_retry_total",
    },
    "strategies/cluster.py": {
        75: "<forwarded_metric_name>",
        353: "extraction_cluster_total",
        363: "extraction_cluster_total",
        404: "extraction_cluster_total",
        416: "extraction_cluster_total",
        438: "extraction_cluster_total",
    },
    "strategies/llm.py": {
        183: "extraction_retry_total",
        267: "llm_tokens_used_total",
        268: "llm_tokens_used_total_by_operation",
    },
    "strategies/trafilatura.py": {71: "article_extracted", 83: "article_extracted"},
}
_METRIC_FORWARDING_ONLY_SITES = {("strategies/cluster.py", 367)}

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


def _module_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    assignments: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            assignments[node.target.id] = node.value
    return assignments


def _cancelled_error_aliases(tree: ast.Module) -> set[str]:
    aliases = {"asyncio"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "asyncio":
                    aliases.add(imported.asname or imported.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "asyncio":
            for imported in node.names:
                if imported.name == "CancelledError":
                    aliases.add(imported.asname or imported.name)
    return aliases


def _is_cancelled_error_expression(expression: ast.expr, aliases: set[str]) -> bool:
    if isinstance(expression, ast.Name):
        return expression.id in aliases and expression.id != "asyncio"
    return (
        isinstance(expression, ast.Attribute)
        and expression.attr == "CancelledError"
        and isinstance(expression.value, ast.Name)
        and expression.value.id in aliases
    )


def _resolve_exception_expression(
    expression: ast.expr,
    assignments: dict[str, ast.expr],
    aliases: set[str],
    seen: set[str],
) -> tuple[bool, bool]:
    """Return whether an exception expression contains cancellation and is tuple-shaped."""
    if _is_cancelled_error_expression(expression, aliases):
        return True, False
    if isinstance(expression, ast.Name) and expression.id in assignments and expression.id not in seen:
        return _resolve_exception_expression(
            assignments[expression.id],
            assignments,
            aliases,
            seen | {expression.id},
        )
    if isinstance(expression, ast.Starred):
        contains_cancelled, _ = _resolve_exception_expression(expression.value, assignments, aliases, seen)
        return contains_cancelled, True
    if isinstance(expression, ast.Tuple):
        return (
            any(_resolve_exception_expression(item, assignments, aliases, seen)[0] for item in expression.elts),
            True,
        )
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left_contains, _ = _resolve_exception_expression(expression.left, assignments, aliases, seen)
        right_contains, _ = _resolve_exception_expression(expression.right, assignments, aliases, seen)
        return left_contains or right_contains, True
    return False, False


def _is_unconditional_bare_reraise(handler: ast.ExceptHandler) -> bool:
    return (
        len(handler.body) == 1
        and isinstance(handler.body[0], ast.Raise)
        and handler.body[0].exc is None
        and handler.body[0].cause is None
    )


def _recoverable_cancelled_error_violations(tree: ast.Module) -> list[str]:
    assignments = _module_assignments(tree)
    aliases = _cancelled_error_aliases(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        contains_cancelled, is_tuple = _resolve_exception_expression(node.type, assignments, aliases, set())
        if contains_cancelled and (is_tuple or not _is_unconditional_bare_reraise(node)):
            violations.append(ast.unparse(node.type))
    return violations


_METRIC_LABEL_CONTRACT: dict[str, dict[str, set[str]]] = {
    "article_extracted": {"success": {"true", "false"}},
    "extraction_cluster_cache_total": {"cache": {"embedding"}, "result": {"hit", "miss"}},
    "extraction_cluster_total": {"status": {"started", "no_blocks", "no_clusters", "empty", "success"}},
    "extraction_content_length_bytes": {
        "strategy": {"jsonld", "schema", "regex", "llm", "cluster", "trafilatura", "unknown"}
    },
    "extraction_retry_total": {
        "strategy": {"jsonld", "schema", "regex", "llm", "cluster", "trafilatura", "unknown"},
        "attempt": {"1", "2", "3", "4_plus"},
    },
    "extraction_strategy_duration_seconds": {
        "strategy": {"jsonld", "schema", "regex", "llm", "cluster", "trafilatura", "unknown"},
        "status": {"skipped", "failed", "success", "enriched"},
    },
    "extraction_strategy_total": {
        "strategy": {"jsonld", "schema", "regex", "llm", "cluster", "trafilatura", "unknown"},
        "status": {"skipped", "failed", "success", "enriched"},
    },
    "llm_tokens_used_total": {
        "provider": {
            "openai",
            "anthropic",
            "cohere",
            "deepseek",
            "google",
            "groq",
            "huggingface",
            "mistral",
            "openrouter",
            "qwen",
            "moonshot",
            "zai",
            "other",
        },
        "model": {"configured"},
        "type": {"prompt", "completion"},
    },
    "llm_tokens_used_total_by_operation": {
        "provider": {
            "openai",
            "anthropic",
            "cohere",
            "deepseek",
            "google",
            "groq",
            "huggingface",
            "mistral",
            "openrouter",
            "qwen",
            "moonshot",
            "zai",
            "other",
        },
        "model": {"configured"},
        "type": {"prompt", "completion"},
        "operation": {"extraction"},
    },
}


def _assert_metric_contract(events: list[tuple[str, dict[str, str]]]) -> None:
    for name, labels in events:
        assert name in _METRIC_LABEL_CONTRACT
        expected = _METRIC_LABEL_CONTRACT[name]
        assert set(labels) == set(expected)
        for key, allowed_values in expected.items():
            assert labels[key] in allowed_values


def _metric_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _metric_emission_call(path: Path, lineno: int) -> ast.Call:
    return next(
        node
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Call)
        and node.lineno == lineno
        and _metric_call_name(node) in {"log_counter", "increment_counter", "observe_histogram", "_increment_counter"}
    )


def _production_metric_names() -> set[str]:
    names: set[str] = set()
    for path in METRIC_SOURCE_PATHS:
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call) or _metric_call_name(node) not in {
                "log_counter",
                "increment_counter",
                "observe_histogram",
                "_increment_counter",
            }:
                continue
            name_index = 1 if _metric_call_name(node) == "_increment_counter" else 0
            if len(node.args) > name_index and isinstance(node.args[name_index], ast.Constant):
                if isinstance(node.args[name_index].value, str):
                    names.add(node.args[name_index].value)
    return names


def _production_metric_call_sites() -> set[tuple[str, int]]:
    sites: set[tuple[str, int]] = set()
    for path in METRIC_SOURCE_PATHS:
        relative_path = str(path.relative_to(EXTRACTION_ROOT))
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call) and _metric_call_name(node) in {
                "log_counter",
                "increment_counter",
                "observe_histogram",
                "_increment_counter",
            }:
                sites.add((relative_path, node.lineno))
    return sites


def _assert_metric_name_contract(metric_names: set[str]) -> None:
    assert metric_names == set(_METRIC_LABEL_CONTRACT)


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
    legacy_imports = [
        node for node in ast.walk(enhanced_tree) if isinstance(node, ast.ImportFrom) and node.module == legacy_module
    ]
    assert len(legacy_imports) == 1
    assert [alias.name for alias in legacy_imports[0].names] == ["scrape_article"]
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
    assert fallback_imports == legacy_imports
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


def test_phase4b_enhanced_no_parent_fallback_forwards_legacy_job_arguments(monkeypatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ScrapingJob, ScrapingJobQueue

    calls: list[tuple[str, dict[str, Any]]] = []

    async def fallback(url: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((url, kwargs))
        return {"url": url, "extraction_successful": True}

    monkeypatch.setattr(article, "scrape_article", fallback)
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
    assert "extract_article_with_pipeline" in _called_names(helper)


def test_phase4b_crawl_bound_article_helper_forwards_to_canonical_extraction_and_closes_page(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article

    class Page:
        closed = False

        async def goto(self, url: str) -> None:
            assert url == "https://example.com/article"

        async def wait_for_load_state(self, state: str) -> None:
            assert state == "networkidle"

        async def title(self) -> str:
            return "Browser title"

        async def content(self) -> str:
            return "<article>Body</article>"

        async def close(self) -> None:
            self.closed = True

    class Context:
        def __init__(self) -> None:
            self.page = Page()

        async def new_page(self) -> Page:
            return self.page

    calls: list[tuple[str, str, dict[str, bool]]] = []

    def extract(html: str, url: str, **kwargs: bool) -> dict[str, Any]:
        calls.append((html, url, kwargs))
        return {"title": "N/A", "content": "Body", "extraction_successful": True}

    monkeypatch.setattr(article, "extract_article_with_pipeline", extract)
    monkeypatch.setattr(article, "convert_html_to_markdown", lambda content: f"markdown:{content}")
    context = Context()

    result = asyncio.run(
        article.scrape_article_async(context, "https://example.com/article", allow_llm_extraction=False)
    )

    assert calls == [("<article>Body</article>", "https://example.com/article", {"allow_llm_extraction": False})]
    assert result == {"title": "Browser title", "content": "markdown:Body", "extraction_successful": True}
    assert context.page.closed is True


def test_phase4b_crawl_bound_article_helper_returns_legacy_failure_shape_and_closes_page(monkeypatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article

    class Page:
        closed = False

        async def goto(self, _url: str) -> None:
            raise RuntimeError("legacy failure")

        async def close(self) -> None:
            self.closed = True

    class Context:
        def __init__(self) -> None:
            self.page = Page()

        async def new_page(self) -> Page:
            return self.page

    context = Context()
    result = asyncio.run(article.scrape_article_async(context, "https://example.com/article"))

    assert result == {
        "url": "https://example.com/article",
        "extraction_successful": False,
        "error": "legacy failure",
    }
    assert context.page.closed is True


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
    for event in invalid_events:
        try:
            _assert_metric_contract([event])
        except AssertionError:
            continue
        raise AssertionError(f"accepted invalid metric event: {event}")


def test_phase4b_metric_inventory_is_exhaustive_and_bidirectionally_contracted() -> None:
    expected_emission_sites = {
        (relative_path, lineno) for relative_path, sites in _METRIC_EMISSION_INVENTORY.items() for lineno in sites
    }
    assert len(expected_emission_sites) == 17
    assert _production_metric_call_sites() == expected_emission_sites | _METRIC_FORWARDING_ONLY_SITES
    for relative_path, sites in _METRIC_EMISSION_INVENTORY.items():
        path = EXTRACTION_ROOT / relative_path
        for lineno, expected_name in sites.items():
            call = _metric_emission_call(path, lineno)
            name_argument = call.args[1] if _metric_call_name(call) == "_increment_counter" else call.args[0]
            if expected_name == "<forwarded_metric_name>":
                assert isinstance(name_argument, ast.Name)
                assert name_argument.id == "name"
            else:
                assert isinstance(name_argument, ast.Constant)
                assert name_argument.value == expected_name

    _assert_metric_name_contract(_production_metric_names())
    with pytest.raises(AssertionError):
        _assert_metric_name_contract(_production_metric_names() | {"future_uncontracted_metric"})


def test_phase4b_metric_contract_covers_every_production_emission_and_allowed_value(monkeypatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction import caches, pipeline
    from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies
    from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import cluster, llm, trafilatura

    events: list[tuple[str, dict[str, str]]] = []

    def record(name: str, _value: float | None = None, labels: dict[str, str] | None = None) -> None:
        events.append((name, dict(labels or {})))

    dependencies = dataclasses.replace(
        build_default_dependencies(),
        increment_counter=record,
        log_counter=record,
        observe_histogram=record,
    )

    strategies = _METRIC_LABEL_CONTRACT["extraction_strategy_total"]["strategy"]
    statuses = _METRIC_LABEL_CONTRACT["extraction_strategy_total"]["status"]
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
    for status in _METRIC_LABEL_CONTRACT["extraction_cluster_total"]["status"]:
        cluster._increment_counter(dependencies, "extraction_cluster_total", labels={"status": status})

    monkeypatch.setattr(trafilatura.trafilatura, "extract_metadata", lambda _html: None)
    monkeypatch.setattr(trafilatura, "log_counter", record)
    monkeypatch.setattr(trafilatura.trafilatura, "extract", lambda _html, **_kwargs: "article body")
    trafilatura.extract_with_trafilatura("<html></html>", "https://example.com/success")
    monkeypatch.setattr(trafilatura.trafilatura, "extract", lambda _html, **_kwargs: None)
    trafilatura.extract_with_trafilatura("<html></html>", "https://example.com/failure")

    providers = _METRIC_LABEL_CONTRACT["llm_tokens_used_total"]["provider"]
    for provider in providers:
        input_provider = provider if provider != "other" else "https://user:secret@example.com"
        llm.record_llm_usage_metrics(
            {"prompt_tokens": 1, "completion_tokens": 1},
            provider=input_provider,
            model="unbounded-model-payload",
            dependencies=dependencies,
        )

    _assert_metric_contract(events)
    assert {name for name, _labels in events} == set(_METRIC_LABEL_CONTRACT)
    for metric_name, labels_contract in _METRIC_LABEL_CONTRACT.items():
        metric_events = [labels for name, labels in events if name == metric_name]
        for label_name, allowed_values in labels_contract.items():
            assert {labels[label_name] for labels in metric_events} == allowed_values


def test_phase4b_extraction_never_recovers_cancelled_error_in_exception_tuples() -> None:
    violations: list[str] = []
    for path in EXTRACTION_ROOT.rglob("*.py"):
        violations.extend(
            f"{path.relative_to(EXTRACTION_ROOT)}: {violation}"
            for violation in _recoverable_cancelled_error_violations(_tree(path))
        )

    assert violations == []


def test_phase4b_cancellation_guard_rejects_inline_and_named_recoverable_tuples() -> None:
    cases = {
        "qualified_bare_reraise": (
            """
import asyncio
try:
    pass
except asyncio.CancelledError:
    raise
""",
            [],
        ),
        "imported_alias_bare_reraise": (
            """
from asyncio import CancelledError as Cancelled
try:
    pass
except Cancelled:
    raise
""",
            [],
        ),
        "qualified_swallow": (
            """
import asyncio
try:
    pass
except asyncio.CancelledError:
    pass
""",
            ["asyncio.CancelledError"],
        ),
        "alias_swallow": (
            """
from asyncio import CancelledError as Cancelled
try:
    pass
except Cancelled:
    return None
""",
            ["Cancelled"],
        ),
        "explicit_exception": (
            """
import asyncio
try:
    pass
except asyncio.CancelledError:
    raise RuntimeError("cancelled")
""",
            ["asyncio.CancelledError"],
        ),
        "failure_construction": (
            """
import asyncio
try:
    pass
except asyncio.CancelledError:
    result = {"error": "cancelled"}
    return result
""",
            ["asyncio.CancelledError"],
        ),
        "fallthrough": (
            """
import asyncio
try:
    pass
except asyncio.CancelledError:
    cleanup()
""",
            ["asyncio.CancelledError"],
        ),
        "retry": (
            """
from asyncio import CancelledError
try:
    pass
except CancelledError:
    retry_request()
""",
            ["CancelledError"],
        ),
        "named_starred_tuple": (
            """
import asyncio
_BASE = (asyncio.CancelledError,)
_RECOVERABLE = (*_BASE, ValueError)
try:
    pass
except _RECOVERABLE:
    pass
""",
            ["_RECOVERABLE"],
        ),
        "composed_tuple": (
            """
from asyncio import CancelledError as Cancelled
_BASE = (ValueError,)
_RECOVERABLE = _BASE + (Cancelled,)
try:
    pass
except _RECOVERABLE:
    pass
""",
            ["_RECOVERABLE"],
        ),
        "inline_tuple": (
            """
import asyncio as aio
try:
    pass
except (ValueError, aio.CancelledError):
    pass
""",
            ["(ValueError, aio.CancelledError)"],
        ),
    }

    for source, expected in cases.values():
        assert _recoverable_cancelled_error_violations(ast.parse(source)) == expected


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
