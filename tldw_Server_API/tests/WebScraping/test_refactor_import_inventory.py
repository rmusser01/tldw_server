from __future__ import annotations

import json
from pathlib import Path

import pytest

from Helper_Scripts.web_scraping_refactor_inventory import (
    LEGACY_WRAPPER_MODULES,
    NEW_INTERNAL_PACKAGES,
    TARGET_MODULES,
    ImportRecord,
    inventory_for_roots,
    scan_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
INVENTORY_JSON = REPO_ROOT / "Docs/Design/web_scraping_refactor_import_inventory.json"
NEW_PACKAGE_ROOT = REPO_ROOT / "tldw_Server_API/app/core/Web_Scraping"


@pytest.mark.unit
def test_scan_file_finds_direct_imports(tmp_path: Path) -> None:
    source = tmp_path / "caller.py"
    source.write_text(
        "\n".join(
            [
                "from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article",
                "from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws",
                "import tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping as enhanced",
            ]
        ),
        encoding="utf-8",
    )

    records = scan_file(source, project_root=tmp_path, targets=TARGET_MODULES)

    assert [
        (record.target_name, record.module, record.imported_name, record.line)
        for record in records
    ] == [
        (
            "Article_Extractor_Lib",
            "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
            "scrape_article",
            1,
        ),
        (
            "WebSearch_APIs",
            "tldw_Server_API.app.core.Web_Scraping",
            "WebSearch_APIs",
            2,
        ),
        (
            "enhanced_web_scraping",
            "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping",
            None,
            3,
        ),
    ]


@pytest.mark.unit
def test_inventory_for_roots_groups_records_by_target(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    tests_dir = tmp_path / "tests"
    app_dir.mkdir()
    tests_dir.mkdir()
    (app_dir / "service.py").write_text(
        "from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import perform_websearch\n",
        encoding="utf-8",
    )
    (tests_dir / "test_service.py").write_text(
        "from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article\n",
        encoding="utf-8",
    )

    inventory = inventory_for_roots(tmp_path, [app_dir, tests_dir], targets=TARGET_MODULES)

    assert inventory["WebSearch_APIs"][0].path == "app/service.py"
    assert inventory["Article_Extractor_Lib"][0].path == "tests/test_service.py"


@pytest.mark.unit
def test_new_internal_packages_constant_excludes_legacy_wrappers() -> None:
    assert "Article_Extractor_Lib" not in NEW_INTERNAL_PACKAGES
    assert "enhanced_web_scraping" not in NEW_INTERNAL_PACKAGES
    assert "WebSearch_APIs" not in NEW_INTERNAL_PACKAGES
    assert "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib" in LEGACY_WRAPPER_MODULES
