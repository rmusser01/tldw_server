"""Executable Phase 5 ownership boundaries, including hostile fixture modules."""

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PREFIX = "tldw_Server_API.app.core.Web_Scraping"
PACKAGE = ROOT / "tldw_Server_API/app/core/Web_Scraping"
LOWER = (f"{PREFIX}.crawl", f"{PREFIX}.jobs")
OWNERS = (
    f"{PREFIX}.orchestration",
    f"{PREFIX}.Article_Extractor_Lib",
    f"{PREFIX}.enhanced_web_scraping",
    f"{PREFIX}.WebSearch_APIs",
    "tldw_Server_API.app.services",
)
pytestmark = pytest.mark.unit


def _imports(source: str, module: str, *, package: bool = False) -> set[str]:
    tree = ast.parse(source)
    current = module if package else module.rsplit(".", 1)[0]
    modules, loaders, result = {"importlib"}, set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
            modules.update(alias.asname or alias.name for alias in node.names if alias.name == "importlib")
        elif isinstance(node, ast.ImportFrom):
            name = resolve_name("." * node.level + (node.module or ""), current) if node.level else node.module
            if name:
                result.add(name)
                result.update(f"{name}.{alias.name}" for alias in node.names)
            if name == "importlib":
                loaders.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        loader = (
            isinstance(node.func, ast.Name)
            and node.func.id in loaders
            or isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in modules
            and node.func.attr == "import_module"
        )
        if not loader or not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = node.args[0].value
        if not isinstance(name, str):
            continue
        arguments = node.args[1:2] + [item.value for item in node.keywords if item.arg == "package"]
        if name.startswith("."):
            if not arguments or not isinstance(arguments[0], ast.Constant):
                continue
            name = resolve_name(name, arguments[0].value)
        result.add(name)
    return result


def _forbidden(graph: dict[str, set[str]]) -> list[tuple[str, str]]:
    return sorted(
        (module, target)
        for module, targets in graph.items()
        if any(module == lower or module.startswith(lower + ".") for lower in LOWER)
        for target in targets
        if any(target == owner or target.startswith(owner + ".") for owner in OWNERS)
    )


def _cycles(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    found, finished = [], set()

    def visit(module, trail):
        if module in trail:
            found.append((*trail[trail.index(module) :], module))
            return
        if module in finished:
            return
        for target in sorted(graph.get(module, set()) & graph.keys()):
            visit(target, (*trail, module))
        finished.add(module)

    for module in sorted(graph):
        visit(module, ())
    return found


def _alternate_page_routes(source: str) -> list[str]:
    alternatives = {"scrape_article", "scrape_article_blocking", "_run_article", "_run_raw_browser_article"}
    return [
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in alternatives
            or isinstance(node.func, ast.Name)
            and node.func.id in alternatives
        )
    ]


@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize("form", ["direct", "from", "dynamic", "alias", "relative"])
def test_guard_detects_forbidden_lower_imports(owner, form):
    source = {
        "direct": f"import {owner} as forbidden",
        "from": f"from {owner} import forbidden",
        "dynamic": f"import importlib; importlib.import_module('{owner}')",
        "alias": f"from importlib import import_module as load; load('{owner}')",
        "relative": (
            f"import importlib as loader; loader.import_module('{owner.removeprefix(PREFIX)}', '{PREFIX}')"
            if owner.startswith(PREFIX)
            else f"from {owner.rsplit('.', 1)[0]} import {owner.rsplit('.', 1)[1]}"
        ),
    }[form]
    module = f"{PREFIX}.crawl.fixture"
    assert _forbidden({module: _imports(source, module)}), source


def test_relative_from_import_and_package_reexports_are_resolved():
    module = f"{PREFIX}.jobs.fixture"
    assert _forbidden({module: _imports("from .. import enhanced_web_scraping", module)})
    assert _forbidden(
        {f"{PREFIX}.jobs": _imports("from ..orchestration import article", f"{PREFIX}.jobs", package=True)}
    )


def test_guard_detects_cycles_but_not_diamonds():
    assert _cycles({"a": {"b"}, "b": {"a"}})
    assert _cycles({"a": {"b", "c"}, "b": {"d"}, "c": {"d"}, "d": set()}) == []


@pytest.mark.parametrize(
    "name", ["scrape_article", "scrape_article_blocking", "_run_article", "_run_raw_browser_article"]
)
def test_guard_detects_alternate_page_execution(name):
    assert _alternate_page_routes(f"async def execute():\n    return await article.{name}('https://example.com')")


@pytest.fixture
def import_graph():
    graph = {}
    for path in PACKAGE.rglob("*.py"):
        parts = path.relative_to(ROOT).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        graph[module] = _imports(path.read_text(), module, package=path.name == "__init__.py")
    return graph


def test_lower_modules_do_not_import_execution_owners(import_graph):
    assert _forbidden(import_graph) == []


def test_new_execution_packages_have_no_dependency_cycles(import_graph):
    scoped = {
        module: dependencies
        for module, dependencies in import_graph.items()
        if module.startswith((*LOWER, f"{PREFIX}.orchestration"))
    }
    assert _cycles(scoped) == []


def test_crawl_has_one_canonical_page_execution_route():
    source = (PACKAGE / "orchestration/crawl.py").read_text()
    assert _alternate_page_routes(source) == []
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    routes = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "start_page"]
    assert len(routes) == 1
    assert ast.unparse(routes[0].func) == "article.start_page"
    definitions = [
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "start_page"
    ]
    assert definitions == ["orchestration/article.py"]


def test_projection_redaction_imports_remain_canonical_after_cycle_removal():
    from tldw_Server_API.app.core.Web_Scraping.jobs import admission, projections

    for name in (
        "PUBLIC_CONTENT_KEYS",
        "_PRIVATE_STATE_KEYS",
        "_URL_IN_TEXT",
        "private_state_key",
        "public_state_string",
    ):
        assert getattr(projections, name) is getattr(admission, name)
