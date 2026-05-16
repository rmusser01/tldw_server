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
