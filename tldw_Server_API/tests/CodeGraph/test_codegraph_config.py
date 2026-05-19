from __future__ import annotations

from tldw_Server_API.app.core.CodeGraph.config import CodeGraphSettings
from tldw_Server_API.app.core.CodeGraph.dependencies import probe_codegraph_dependencies


def test_settings_use_safe_defaults() -> None:
    settings = CodeGraphSettings.from_mapping({})

    assert str(settings.index_base_dir).endswith("Databases/codegraph")
    assert settings.max_file_size_bytes == 1_048_576
    assert settings.foreground_max_files == 500
    assert ".git" in settings.exclude_dirs
    assert "node_modules" in settings.exclude_dirs


def test_settings_coerce_positive_integer_limits() -> None:
    settings = CodeGraphSettings.from_mapping(
        {"max_file_size_bytes": "2048", "foreground_max_files": "12"}
    )

    assert settings.max_file_size_bytes == 2048
    assert settings.foreground_max_files == 12


def test_settings_clamp_unsafe_integer_limits() -> None:
    settings = CodeGraphSettings.from_mapping(
        {"max_file_size_bytes": "0", "foreground_max_files": "-5"}
    )

    assert settings.max_file_size_bytes == 1
    assert settings.foreground_max_files == 1


def test_settings_use_default_index_base_dir_when_config_value_is_blank() -> None:
    settings = CodeGraphSettings.from_mapping({"index_base_dir": "  "})

    assert settings.index_base_dir == CodeGraphSettings().index_base_dir


def test_dependency_probe_reports_missing_without_importing_tree_sitter(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    health = probe_codegraph_dependencies()

    assert health.available is False
    assert "tree_sitter" in health.missing


def test_dependency_probe_keeps_core_available_when_optional_jvm_parsers_are_missing(monkeypatch) -> None:
    present_modules = {
        "tree_sitter",
        "tree_sitter_javascript",
        "tree_sitter_typescript",
    }

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name in present_modules else None)

    health = probe_codegraph_dependencies()

    assert health.available is True
    assert health.all_optional_available is False
    assert "tree_sitter_java" in health.missing
    assert "tree_sitter_kotlin" in health.missing
