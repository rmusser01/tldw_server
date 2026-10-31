from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_MODULE = "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib"
CONTENT_MODULE = "tldw_Server_API.app.core.Web_Scraping.content"
EXTRACTION_MODULE = "tldw_Server_API.app.core.Web_Scraping.extraction"
ORCHESTRATION_MODULE = "tldw_Server_API.app.core.Web_Scraping.orchestration"

CONSUMER_IMPORTS = {
    "tldw_Server_API/app/core/Collections/reading_service.py": {
        CONTENT_MODULE: {"ContentMetadataHandler", "convert_html_to_markdown"},
        ORCHESTRATION_MODULE: {"scrape_article"},
        LEGACY_MODULE: set(),
    },
    "tldw_Server_API/app/core/Evaluations/article_extraction_benchmark.py": {
        CONTENT_MODULE: {"ContentMetadataHandler"},
        EXTRACTION_MODULE: {"extract_article_data_from_html"},
        LEGACY_MODULE: set(),
    },
    "tldw_Server_API/app/core/RAG/rag_service/research_agent.py": {
        ORCHESTRATION_MODULE: {"scrape_article"},
        LEGACY_MODULE: set(),
    },
    "tldw_Server_API/app/core/Watchlists/fetchers.py": {
        CONTENT_MODULE: {"ContentMetadataHandler"},
        ORCHESTRATION_MODULE: {"scrape_article_blocking"},
        LEGACY_MODULE: {"is_content_page"},
    },
    "tldw_Server_API/app/core/Workflows/adapters/rag/search.py": {
        ORCHESTRATION_MODULE: {"scrape_article"},
        LEGACY_MODULE: set(),
    },
    "tldw_Server_API/app/core/WebSearch/Web_Search.py": {
        ORCHESTRATION_MODULE: {"scrape_article"},
        LEGACY_MODULE: set(),
    },
    "tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py": {
        ORCHESTRATION_MODULE: {"scrape_article"},
        LEGACY_MODULE: set(),
    },
    "tldw_Server_API/app/core/Web_Scraping/handlers.py": {
        CONTENT_MODULE: {"convert_html_to_markdown"},
        EXTRACTION_MODULE: {"extract_article_data_from_html"},
        LEGACY_MODULE: set(),
    },
    "tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py": {
        CONTENT_MODULE: {"convert_html_to_markdown"},
        EXTRACTION_MODULE: {"extract_article_with_pipeline"},
        ORCHESTRATION_MODULE: {"scrape_article"},
        LEGACY_MODULE: set(),
    },
    "tldw_Server_API/app/services/enhanced_web_scraping_service.py": {
        CONTENT_MODULE: {"ContentMetadataHandler"},
        LEGACY_MODULE: {"is_content_page"},
    },
    "tldw_Server_API/app/services/web_scraping_service.py": {
        ORCHESTRATION_MODULE: {"scrape_article"},
        LEGACY_MODULE: {"scrape_from_sitemap"},
    },
}

REQUIRED_LEGACY_IMPORTS = {
    "tldw_Server_API/app/core/Watchlists/fetchers.py": {"is_content_page"},
    "tldw_Server_API/app/services/enhanced_web_scraping_service.py": {"is_content_page"},
    "tldw_Server_API/app/services/web_scraping_service.py": {
        "scrape_from_sitemap",
    },
}


def _imported_names(path: Path) -> dict[str, set[str]]:
    imported: dict[str, set[str]] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.setdefault(node.module, set()).update(alias.name for alias in node.names)
    return imported


def _legacy_module_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name == "Article_Extractor_Lib" or alias.name.endswith(".Article_Extractor_Lib")
            )
        elif isinstance(node, ast.ImportFrom):
            imports.update(alias.name for alias in node.names if alias.name == "Article_Extractor_Lib")
    return imports


@pytest.mark.parametrize(
    "source",
    [
        f"import {LEGACY_MODULE} as legacy\n",
        ("from tldw_Server_API.app.core.Web_Scraping " "import Article_Extractor_Lib as legacy\n"),
    ],
)
def test_legacy_module_alias_imports_are_detected(tmp_path: Path, source: str) -> None:
    consumer = tmp_path / "consumer.py"
    consumer.write_text(source, encoding="utf-8")

    assert _legacy_module_imports(consumer)


def test_phase4_consumers_import_only_canonical_article_owners() -> None:
    for relative_path, expected_imports in CONSUMER_IMPORTS.items():
        consumer_path = REPO_ROOT / relative_path
        actual_imports = _imported_names(consumer_path)
        assert not _legacy_module_imports(consumer_path), relative_path
        for module, expected_names in expected_imports.items():
            actual_names = actual_imports.get(module, set())
            if module == LEGACY_MODULE:
                assert REQUIRED_LEGACY_IMPORTS.get(relative_path, set()) <= actual_names, relative_path
                assert actual_names <= expected_names, relative_path
            else:
                assert expected_names <= actual_names, relative_path
