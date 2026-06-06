from pathlib import Path

import pytest

from tldw_Server_API.app.core.Setup import setup_manager


def _write_config(tmp_path: Path, base_dir: str) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.txt"
    config_path.write_text(
        "[TTS-Settings]\n"
        f"USER_DB_BASE_DIR = {base_dir}\n",
        encoding="utf-8",
    )
    return config_dir


def _patch_project_root(monkeypatch, project_root: Path) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "Databases").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(setup_manager, "get_project_root", lambda: str(project_root))


def test_user_db_base_dir_update_rejects_outside_allowed_roots(monkeypatch, tmp_path):
    config_dir = _write_config(tmp_path, "Databases/user_databases")
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_dir))
    _patch_project_root(monkeypatch, tmp_path / "project")

    outside = tmp_path / "project" / "Other" / "user_dbs"
    with pytest.raises(ValueError, match="USER_DB_BASE_DIR"):
        setup_manager.update_config(
            {"TTS-Settings": {"USER_DB_BASE_DIR": str(outside)}},
            create_backup=False,
        )


def test_user_db_base_dir_update_allows_default_root(monkeypatch, tmp_path):
    config_dir = _write_config(tmp_path, "Databases/user_databases")
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_dir))
    _patch_project_root(monkeypatch, tmp_path / "project")

    allowed = tmp_path / "project" / "Databases" / "user_dbs_alt"
    setup_manager.update_config(
        {"TTS-Settings": {"USER_DB_BASE_DIR": str(allowed)}},
        create_backup=False,
    )

    updated = (config_dir / "config.txt").read_text(encoding="utf-8")
    assert f"USER_DB_BASE_DIR = {allowed}" in updated


def test_user_db_base_dir_update_allows_env_allowlist(monkeypatch, tmp_path):
    config_dir = _write_config(tmp_path, "Databases/user_databases")
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_dir))
    _patch_project_root(monkeypatch, tmp_path / "project")

    allow_root = tmp_path / "external"
    monkeypatch.setenv("USER_DB_BASE_DIR_ALLOWED_ROOTS", str(allow_root))
    allowed = allow_root / "user_dbs"
    setup_manager.update_config(
        {"TTS-Settings": {"USER_DB_BASE_DIR": str(allowed)}},
        create_backup=False,
    )

    updated = (config_dir / "config.txt").read_text(encoding="utf-8")
    assert f"USER_DB_BASE_DIR = {allowed}" in updated


@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_update_config_preserves_line_breaks_for_adjacent_updates(monkeypatch, tmp_path, line_ending):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.txt"
    original_lines = [
        "[Embeddings]",
        "embedding_provider = old-provider",
        "embedding_model = old-model",
        "onnx_model_path = ./models",
    ]
    config_path.write_text(
        line_ending.join(original_lines) + line_ending,
        encoding="utf-8",
        newline="",
    )
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_dir))
    _patch_project_root(monkeypatch, tmp_path / "project")

    setup_manager.update_config(
        {
            "Embeddings": {
                "embedding_provider": "huggingface",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            }
        },
        create_backup=False,
    )

    expected_lines = [
        "[Embeddings]",
        "embedding_provider = huggingface",
        "embedding_model = Qwen/Qwen3-Embedding-0.6B",
        "onnx_model_path = ./models",
    ]
    with config_path.open("r", encoding="utf-8", newline="") as handle:
        updated = handle.read()

    assert updated == line_ending.join(expected_lines) + line_ending


def test_user_db_base_dir_update_rejects_relative_escape(monkeypatch, tmp_path):
    config_dir = _write_config(tmp_path, "Databases/user_databases")
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_dir))
    _patch_project_root(monkeypatch, tmp_path / "project")

    with pytest.raises(ValueError, match="Relative USER_DB_BASE_DIR"):
        setup_manager.update_config(
            {"TTS-Settings": {"USER_DB_BASE_DIR": "../outside"}},
            create_backup=False,
        )


def test_normalize_root_path_does_not_resolve_absolute_input(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    absolute_root = tmp_path / "external" / "allowed"
    original_resolve = Path.resolve

    def _guarded_resolve(self, strict=False):
        if self == absolute_root:
            raise AssertionError("absolute allowlist roots should not be resolved before validation")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", _guarded_resolve)

    normalized = setup_manager._normalize_root_path(str(absolute_root), project_root=project_root)

    assert normalized == absolute_root


def test_normalize_user_db_base_dir_candidate_does_not_resolve_absolute_input(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    absolute_root = tmp_path / "external" / "user_dbs"
    original_resolve = Path.resolve

    def _guarded_resolve(self, strict=False):
        if self == absolute_root:
            raise AssertionError("absolute USER_DB_BASE_DIR should not be resolved before allowlist checks")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", _guarded_resolve)

    normalized = setup_manager._normalize_user_db_base_dir_candidate(
        str(absolute_root),
        project_root=project_root,
    )

    assert normalized == absolute_root
