from __future__ import annotations

from pathlib import Path

import pytest
from Helper_Scripts.web_scraping_refactor_inventory import (
    LEGACY_WRAPPER_MODULES,
    NEW_INTERNAL_PACKAGES,
    TARGET_MODULES,
    inventory_for_roots,
    main,
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
def test_scan_file_sorts_mixed_import_forms_without_type_error(tmp_path: Path) -> None:
    source = tmp_path / "caller.py"
    source.write_text(
        "\n".join(
            [
                "import tldw_Server_API.app.core.Web_Scraping.ua_profiles as ua_profiles",
                "from tldw_Server_API.app.core.Web_Scraping.ua_profiles import pick_ua_profile",
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
            "ua_profiles",
            "tldw_Server_API.app.core.Web_Scraping.ua_profiles",
            None,
            1,
        ),
        (
            "ua_profiles",
            "tldw_Server_API.app.core.Web_Scraping.ua_profiles",
            "pick_ua_profile",
            2,
        ),
    ]


@pytest.mark.unit
def test_scan_file_resolves_relative_imports_for_web_scraping_modules(tmp_path: Path) -> None:
    module_path = tmp_path / "tldw_Server_API/app/core/Web_Scraping/scraper_router.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("from .ua_profiles import pick_ua_profile\n", encoding="utf-8")

    records = scan_file(module_path, project_root=tmp_path, targets=TARGET_MODULES)

    assert [
        (record.target_name, record.module, record.imported_name, record.line)
        for record in records
    ] == [
        (
            "ua_profiles",
            "tldw_Server_API.app.core.Web_Scraping.ua_profiles",
            "pick_ua_profile",
            1,
        )
    ]


@pytest.mark.unit
def test_main_resolves_relative_output_paths_against_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "repo"
    external_cwd = tmp_path / "cwd"
    project_root.mkdir()
    external_cwd.mkdir()
    monkeypatch.chdir(external_cwd)

    assert (
        main(
            [
                "--root",
                str(project_root),
                "--json",
                "out/inventory.json",
                "--markdown",
                "out/inventory.md",
            ]
        )
        == 0
    )

    assert (project_root / "out/inventory.json").exists()
    assert (project_root / "out/inventory.md").exists()
    assert not (external_cwd / "out/inventory.json").exists()
    assert not (external_cwd / "out/inventory.md").exists()


@pytest.mark.unit
def test_main_rejects_output_paths_outside_project_root_before_writing(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    outside_root = tmp_path / "outside"
    project_root.mkdir()

    with pytest.raises(ValueError, match="Output path must be under project root"):
        main(
            [
                "--root",
                str(project_root),
                "--json",
                str(outside_root / "inventory.json"),
                "--markdown",
                "out/inventory.md",
            ]
        )

    assert not (outside_root / "inventory.json").exists()
    assert not (project_root / "out/inventory.md").exists()


@pytest.mark.unit
def test_new_internal_packages_constant_excludes_legacy_wrappers() -> None:
    assert "Article_Extractor_Lib" not in NEW_INTERNAL_PACKAGES
    assert "enhanced_web_scraping" not in NEW_INTERNAL_PACKAGES
    assert "WebSearch_APIs" not in NEW_INTERNAL_PACKAGES
    assert "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib" in LEGACY_WRAPPER_MODULES
