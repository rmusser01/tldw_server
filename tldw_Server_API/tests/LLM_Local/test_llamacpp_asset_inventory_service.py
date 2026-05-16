from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path

import pytest


def _llamacpp_parser(models_dir: Path, **overrides: str) -> ConfigParser:
    parser = ConfigParser()
    parser.add_section("LlamaCpp")
    values = {
        "enabled": "true",
        "models_dir": str(models_dir),
        "allowed_paths": "",
        "registered_model_paths": "",
        "imported_asset_folders": "",
    }
    values.update(overrides)
    parser["LlamaCpp"] = values
    return parser


@pytest.mark.unit
def test_config_state_reads_imported_asset_folders(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    models_dir = tmp_path / "models"
    imported = tmp_path / "external"
    models_dir.mkdir()
    imported.mkdir()
    monkeypatch.setattr(
        llamacpp_config_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, imported_asset_folders=str(imported)),
    )

    state = llamacpp_config_service.get_config_state(llm_manager=object())

    assert state["saved_config"]["imported_asset_folders"] == [str(imported)]


@pytest.mark.unit
def test_scan_assets_discovers_gguf_mmproj_and_folder(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    imported = tmp_path / "imported"
    models_dir.mkdir()
    imported.mkdir()
    (models_dir / "Llama-3-8B-Q4_K_M.gguf").write_text("base")
    (models_dir / "mmproj-Llama-3-vision-f16.gguf").write_text("projector")
    (imported / "notes.txt").write_text("not a model")

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, allowed_paths=str(imported), imported_asset_folders=str(imported)),
    )

    result = llamacpp_inventory_service.scan_assets(limit=500)

    by_kind = {asset.kind: asset for asset in result.assets}
    assert by_kind["gguf"].asset_id.startswith("gguf:")
    assert by_kind["mmproj"].asset_id.startswith("mmproj:")
    assert by_kind["folder"].asset_id.startswith("folder:")
    assert "vision_projector" in by_kind["mmproj"].capabilities
    assert by_kind["folder"].source == "imported_folder"


@pytest.mark.unit
def test_scan_assets_reports_stale_imported_folder_without_failing(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    missing = tmp_path / "missing"
    models_dir.mkdir()
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, imported_asset_folders=str(missing)),
    )

    result = llamacpp_inventory_service.scan_assets(limit=500)

    folder = next(asset for asset in result.assets if asset.kind == "folder")
    assert folder.resolved_path == str(missing)
    assert any("missing" in warning.lower() for warning in folder.warnings)
