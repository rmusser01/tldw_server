"""Shared fixtures for legacy WebScraping extraction tests."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from tldw_Server_API.app.core.Web_Scraping.extraction import pipeline
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import (
    ExtractionDependencies,
    build_default_dependencies,
)


@pytest.fixture
def install_extraction_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., ExtractionDependencies]:
    """Install a provider and optional dependency overrides for one test."""

    def install(provider: Callable[..., Any], **overrides: Any) -> ExtractionDependencies:
        dependencies = dataclasses.replace(
            build_default_dependencies(),
            perform_chat_api_call=provider,
            **overrides,
        )
        monkeypatch.setattr(pipeline, "build_default_dependencies", lambda: dependencies)
        return dependencies

    return install
