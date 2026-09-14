from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import (
    scrape_and_summarize_multiple as legacy_scrape_and_summarize_multiple,
)
from tldw_Server_API.app.services import web_scraping_service as ws_service

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
        CONTENT_MODULE: {"ContentMetadataHandler"},
        ORCHESTRATION_MODULE: {"scrape_article"},
        LEGACY_MODULE: {
            "recursive_scrape",
            "scrape_and_summarize_multiple",
            "scrape_by_url_level",
            "scrape_from_sitemap",
        },
    },
}

REQUIRED_LEGACY_IMPORTS = {
    "tldw_Server_API/app/core/Watchlists/fetchers.py": {"is_content_page"},
    "tldw_Server_API/app/services/enhanced_web_scraping_service.py": {"is_content_page"},
    "tldw_Server_API/app/services/web_scraping_service.py": {
        "recursive_scrape",
        "scrape_and_summarize_multiple",
        "scrape_by_url_level",
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


@pytest.mark.asyncio
async def test_legacy_fallback_forwards_system_message_with_real_helper_signature(monkeypatch) -> None:
    received: dict[str, Any] = {}

    async def strict_scrape_and_summarize_multiple(
        urls: str,
        custom_prompt_arg: str | None,
        api_name: str,
        api_key: str | None,
        keywords: str,
        custom_article_titles: str | None,
        system_message: str | None = None,
        summarize_checkbox: bool = False,
        custom_cookies: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        allow_llm_extraction: bool = True,
        summary_prompt_overrides: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        received.update(
            {
                "urls": urls,
                "custom_prompt_arg": custom_prompt_arg,
                "api_name": api_name,
                "api_key": api_key,
                "keywords": keywords,
                "custom_article_titles": custom_article_titles,
                "system_message": system_message,
                "summarize_checkbox": summarize_checkbox,
                "custom_cookies": custom_cookies,
                "temperature": temperature,
                "allow_llm_extraction": allow_llm_extraction,
                "summary_prompt_overrides": summary_prompt_overrides,
            }
        )
        return [
            {
                "url": "https://example.com/article",
                "title": "Article",
                "content": "Body",
                "extraction_successful": True,
            }
        ]

    real_parameters = inspect.signature(legacy_scrape_and_summarize_multiple).parameters
    strict_parameters = inspect.signature(strict_scrape_and_summarize_multiple).parameters
    assert [(parameter.name, parameter.kind, parameter.default) for parameter in strict_parameters.values()] == [
        (parameter.name, parameter.kind, parameter.default) for parameter in real_parameters.values()
    ]

    monkeypatch.setenv("TLDW_ENABLE_LEGACY_WEB_SCRAPING_FALLBACK", "1")
    monkeypatch.setattr(
        ws_service,
        "get_web_scraping_service",
        lambda: (_ for _ in ()).throw(RuntimeError("enhanced service unavailable")),
        raising=True,
    )
    monkeypatch.setattr(
        ws_service,
        "scrape_and_summarize_multiple",
        strict_scrape_and_summarize_multiple,
        raising=True,
    )
    monkeypatch.setattr(ws_service.ephemeral_storage, "store_data", lambda _data: "fallback-result")

    result = await ws_service.process_web_scraping_task(
        scrape_method="Individual URLs",
        url_input="https://example.com/article",
        url_level=None,
        max_pages=2,
        max_depth=1,
        summarize_checkbox=True,
        custom_prompt="Summarize the source.",
        api_name="test-provider",
        api_key="test-key",
        keywords="web scraping",
        custom_titles="Custom title",
        system_prompt="Use a neutral editorial voice.",
        temperature=0.2,
        custom_cookies=[{"name": "session", "value": "cookie-value"}],
        mode="ephemeral",
        user_id=1,
        perform_chunking=False,
        summary_prompt_overrides={"system": "Saved system", "user": "Saved summary"},
    )

    assert result["status"] == "ephemeral-ok"
    assert received == {
        "urls": "https://example.com/article",
        "custom_prompt_arg": "Summarize the source.",
        "api_name": "test-provider",
        "api_key": "test-key",
        "keywords": "web scraping",
        "custom_article_titles": "Custom title",
        "system_message": "Use a neutral editorial voice.",
        "summarize_checkbox": True,
        "custom_cookies": [{"name": "session", "value": "cookie-value"}],
        "temperature": 0.2,
        "allow_llm_extraction": True,
        "summary_prompt_overrides": {"system": "Saved system", "user": "Saved summary"},
    }
