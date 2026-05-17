from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path

import pytest


def _llamacpp_parser(default_models_dir: Path, **overrides: str) -> ConfigParser:
    parser = ConfigParser()
    parser.add_section("LlamaCpp")
    values = {
        "enabled": "true",
        "models_dir": str(default_models_dir),
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
        lambda: _llamacpp_parser(
            models_dir,
            allowed_paths=str(imported),
            imported_asset_folders=str(imported),
        ),
    )

    result = llamacpp_inventory_service.scan_assets(limit=500)

    by_kind = {asset.kind: asset for asset in result.assets}
    assert by_kind["gguf"].asset_id.startswith("gguf:")
    assert by_kind["mmproj"].asset_id.startswith("mmproj:")
    assert by_kind["folder"].asset_id.startswith("folder:")
    assert "vision_projector" in by_kind["mmproj"].capabilities
    assert by_kind["folder"].source == "imported_folder"


@pytest.mark.unit
def test_preview_import_asset_folder_summarizes_assets_without_persisting(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    imported = tmp_path / "imported"
    models_dir.mkdir()
    imported.mkdir()
    (imported / "chat.Q4_K_M.gguf").write_text("base")
    (imported / "mmproj-chat-f16.gguf").write_text("projector")
    updates: list[dict[str, dict[str, str]]] = []
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(
            models_dir,
            allowed_paths=str(imported),
            imported_asset_folders="",
        ),
    )
    monkeypatch.setattr(llamacpp_inventory_service.setup_manager, "update_config", updates.append)

    preview = llamacpp_inventory_service.preview_import_asset_folder(imported, limit=500)

    assert preview.folder.kind == "folder"
    assert preview.folder.display_name == "imported"
    assert preview.asset_counts == {"gguf": 1, "mmproj": 1}
    assert {asset.kind for asset in preview.assets} == {"gguf", "mmproj"}
    assert preview.assets[0].source == "imported_folder"
    assert preview.scan_limited is False
    assert preview.will_persist is False
    assert updates == []


@pytest.mark.unit
def test_preview_import_asset_folder_deduplicates_resolved_asset_ids(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    imported = tmp_path / "imported"
    models_dir.mkdir()
    imported.mkdir()
    model_path = imported / "chat.Q4_K_M.gguf"
    model_path.write_text("base")
    symlink_path = imported / "chat-link.Q4_K_M.gguf"
    try:
        symlink_path.symlink_to(model_path)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable in this environment: {exc}")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, allowed_paths=str(imported)),
    )

    preview = llamacpp_inventory_service.preview_import_asset_folder(imported, limit=500)

    assert [asset.kind for asset in preview.assets] == ["gguf"]
    assert preview.asset_counts == {"gguf": 1}


@pytest.mark.unit
def test_preview_import_asset_folder_fails_for_file_path(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError

    models_dir = tmp_path / "models"
    imported = tmp_path / "imported"
    models_dir.mkdir()
    imported.mkdir()
    file_path = imported / "chat.gguf"
    file_path.write_text("base")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, allowed_paths=str(imported)),
    )

    with pytest.raises(ServerError, match="not a folder"):
        llamacpp_inventory_service.preview_import_asset_folder(file_path)


@pytest.mark.unit
def test_preview_import_asset_folder_fails_closed_outside_allowed_paths(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError

    models_dir = tmp_path / "models"
    imported = tmp_path / "imported"
    models_dir.mkdir()
    imported.mkdir()
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, allowed_paths=""),
    )

    with pytest.raises(ServerError, match="outside allowed"):
        llamacpp_inventory_service.preview_import_asset_folder(imported)


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


@pytest.mark.unit
def test_scan_assets_adds_inferred_mmproj_candidates_with_warnings(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "llava-7b-Q4_K_M.gguf"
    projector = models_dir / "mmproj-llava-7b-f16.gguf"
    base.write_text("base")
    projector.write_text("projector")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )

    result = llamacpp_inventory_service.scan_assets(limit=500)

    base_asset = next(asset for asset in result.assets if asset.kind == "gguf")
    projector_asset = next(asset for asset in result.assets if asset.kind == "mmproj")
    assert projector_asset.asset_id in base_asset.mmproj_asset_ids
    assert base_asset.asset_id in projector_asset.base_model_asset_ids
    assert any("inferred" in warning.lower() for warning in base_asset.warnings)


@pytest.mark.unit
def test_resolve_asset_id_rejects_wrong_kind(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelNotFoundError

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "chat.gguf"
    projector = models_dir / "mmproj-chat.gguf"
    base.write_text("base")
    projector.write_text("projector")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )

    base_id = llamacpp_inventory_service.asset_id_for_path(base, "gguf")

    with pytest.raises(ModelNotFoundError, match="mmproj"):
        llamacpp_inventory_service.resolve_asset_id(base_id, expected_kind="mmproj")


@pytest.mark.unit
def test_resolve_asset_id_rejects_missing_asset_id(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelNotFoundError

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )

    with pytest.raises(ModelNotFoundError, match="was not found"):
        llamacpp_inventory_service.resolve_asset_id("gguf:missing")


@pytest.mark.unit
def test_resolve_asset_id_rejects_stale_registered_asset(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelNotFoundError

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    stale = models_dir / "stale.gguf"
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, registered_model_paths=str(stale)),
    )

    stale_id = llamacpp_inventory_service.asset_id_for_path(stale, "gguf")

    with pytest.raises(ModelNotFoundError, match="available local asset"):
        llamacpp_inventory_service.resolve_asset_id(stale_id, expected_kind="gguf")


@pytest.mark.unit
def test_resolve_asset_id_accepts_pre_scanned_assets(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "chat.gguf"
    base.write_text("base")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )

    assets = llamacpp_inventory_service.scan_assets(limit=500).assets

    def fail_scan_assets(*_args, **_kwargs):
        raise AssertionError("resolve_asset_id should use the provided pre-scanned assets")

    monkeypatch.setattr(llamacpp_inventory_service, "scan_assets", fail_scan_assets)

    resolved = llamacpp_inventory_service.resolve_asset_id(
        llamacpp_inventory_service.asset_id_for_path(base, "gguf"),
        expected_kind="gguf",
        assets=assets,
    )

    assert resolved.resolved_path == str(base.resolve())


@pytest.mark.unit
def test_resolve_asset_id_accepts_folder_expected_kind(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    imported = tmp_path / "imported"
    models_dir.mkdir()
    imported.mkdir()
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(
            models_dir,
            allowed_paths=str(imported),
            imported_asset_folders=str(imported),
        ),
    )
    folder_id = llamacpp_inventory_service.asset_id_for_path(imported, "folder")

    resolved = llamacpp_inventory_service.resolve_asset_id(folder_id, expected_kind="folder")

    assert resolved.kind == "folder"
    assert resolved.resolved_path == str(imported.resolve())


@pytest.mark.unit
def test_resolve_asset_id_fails_closed_without_allowed_paths(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError

    base = tmp_path / "chat.gguf"
    base.write_text("base")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(
            Path(""),
            models_dir="",
            registered_model_paths=str(base),
        ),
    )

    with pytest.raises(ServerError, match="outside allowed"):
        llamacpp_inventory_service.resolve_asset_id(
            llamacpp_inventory_service.asset_id_for_path(base, "gguf"),
            expected_kind="gguf",
        )
