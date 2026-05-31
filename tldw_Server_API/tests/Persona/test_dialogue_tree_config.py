from __future__ import annotations

import pytest

from tldw_Server_API.app.core import config


_DIALOGUE_TREE_ENV_KEYS = (
    "PERSONA_DIALOGUE_TREE_EVAL_ENABLED",
    "PERSONA_RUNTIME_EXPLORER_ENABLED",
    "PERSONA_RUNTIME_EXPLORER_MAX_DEPTH",
    "PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING",
    "PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS",
    "PERSONA_RUNTIME_EXPLORER_TIMEOUT_MS",
    "PERSONA_RUNTIME_EXPLORER_MAX_TOKENS",
    "PERSONA_RUNTIME_EXPLORER_P95_ADDED_LATENCY_MS",
    "PERSONA_RUNTIME_EXPLORER_LLM_JUDGES_ENABLED",
    "PERSONA_DIALOGUE_TREE_TRACE_RETENTION_DAYS",
)


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch):
    monkeypatch.setattr(config, "_load_env_files_early", lambda: None)
    for key in (*_DIALOGUE_TREE_ENV_KEYS, "TLDW_CONFIG_FILE", "TLDW_CONFIG_PATH", "TLDW_CONFIG_DIR"):
        monkeypatch.delenv(key, raising=False)
    config.clear_config_cache()
    yield
    config.clear_config_cache()


def _load_settings_for_test() -> dict:
    config.clear_config_cache()
    return dict(config.load_settings())


def test_persona_runtime_explorer_defaults_are_safe():
    settings = _load_settings_for_test()

    assert settings["PERSONA_DIALOGUE_TREE_EVAL_ENABLED"] is False
    assert settings["PERSONA_RUNTIME_EXPLORER_ENABLED"] is False
    assert settings["PERSONA_RUNTIME_EXPLORER_MAX_DEPTH"] == 1
    assert settings["PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING"] == 2
    assert settings["PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS"] == 1
    assert settings["PERSONA_RUNTIME_EXPLORER_TIMEOUT_MS"] == 750
    assert settings["PERSONA_RUNTIME_EXPLORER_MAX_TOKENS"] == 256
    assert settings["PERSONA_RUNTIME_EXPLORER_P95_ADDED_LATENCY_MS"] == 1000
    assert settings["PERSONA_RUNTIME_EXPLORER_LLM_JUDGES_ENABLED"] is False
    assert settings["PERSONA_DIALOGUE_TREE_TRACE_RETENTION_DAYS"] == 7


def test_persona_runtime_explorer_config_file_overrides_load(tmp_path, monkeypatch):
    cfg = tmp_path / "config.txt"
    cfg.write_text(
        "\n".join(
            [
                "[persona]",
                "dialogue_tree_eval_enabled = true",
                "runtime_explorer_enabled = true",
                "runtime_explorer_max_depth = 2",
                "runtime_explorer_max_branching = 3",
                "runtime_explorer_max_provider_calls = 4",
                "runtime_explorer_timeout_ms = 1500",
                "runtime_explorer_max_tokens = 512",
                "runtime_explorer_p95_added_latency_ms = 2500",
                "runtime_explorer_llm_judges_enabled = true",
                "dialogue_tree_trace_retention_days = 30",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_FILE", str(cfg))

    settings = _load_settings_for_test()

    assert settings["PERSONA_DIALOGUE_TREE_EVAL_ENABLED"] is True
    assert settings["PERSONA_RUNTIME_EXPLORER_ENABLED"] is True
    assert settings["PERSONA_RUNTIME_EXPLORER_MAX_DEPTH"] == 2
    assert settings["PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING"] == 3
    assert settings["PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS"] == 4
    assert settings["PERSONA_RUNTIME_EXPLORER_TIMEOUT_MS"] == 1500
    assert settings["PERSONA_RUNTIME_EXPLORER_MAX_TOKENS"] == 512
    assert settings["PERSONA_RUNTIME_EXPLORER_P95_ADDED_LATENCY_MS"] == 2500
    assert settings["PERSONA_RUNTIME_EXPLORER_LLM_JUDGES_ENABLED"] is True
    assert settings["PERSONA_DIALOGUE_TREE_TRACE_RETENTION_DAYS"] == 30


def test_persona_runtime_explorer_env_overrides_are_typed(monkeypatch):
    monkeypatch.setenv("PERSONA_DIALOGUE_TREE_EVAL_ENABLED", "true")
    monkeypatch.setenv("PERSONA_RUNTIME_EXPLORER_ENABLED", "true")
    monkeypatch.setenv("PERSONA_RUNTIME_EXPLORER_MAX_DEPTH", "2")
    monkeypatch.setenv("PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING", "3")
    monkeypatch.setenv("PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS", "4")
    monkeypatch.setenv("PERSONA_RUNTIME_EXPLORER_TIMEOUT_MS", "1500")
    monkeypatch.setenv("PERSONA_RUNTIME_EXPLORER_MAX_TOKENS", "512")
    monkeypatch.setenv("PERSONA_RUNTIME_EXPLORER_P95_ADDED_LATENCY_MS", "2500")
    monkeypatch.setenv("PERSONA_RUNTIME_EXPLORER_LLM_JUDGES_ENABLED", "yes")
    monkeypatch.setenv("PERSONA_DIALOGUE_TREE_TRACE_RETENTION_DAYS", "30")

    settings = _load_settings_for_test()

    assert settings["PERSONA_DIALOGUE_TREE_EVAL_ENABLED"] is True
    assert settings["PERSONA_RUNTIME_EXPLORER_ENABLED"] is True
    assert settings["PERSONA_RUNTIME_EXPLORER_MAX_DEPTH"] == 2
    assert settings["PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING"] == 3
    assert settings["PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS"] == 4
    assert settings["PERSONA_RUNTIME_EXPLORER_TIMEOUT_MS"] == 1500
    assert settings["PERSONA_RUNTIME_EXPLORER_MAX_TOKENS"] == 512
    assert settings["PERSONA_RUNTIME_EXPLORER_P95_ADDED_LATENCY_MS"] == 2500
    assert settings["PERSONA_RUNTIME_EXPLORER_LLM_JUDGES_ENABLED"] is True
    assert settings["PERSONA_DIALOGUE_TREE_TRACE_RETENTION_DAYS"] == 30


def test_blank_persona_boolean_env_vars_fall_back_to_config_file(tmp_path, monkeypatch):
    cfg = tmp_path / "config.txt"
    cfg.write_text(
        "\n".join(
            [
                "[persona]",
                "dialogue_tree_eval_enabled = true",
                "runtime_explorer_enabled = true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("PERSONA_DIALOGUE_TREE_EVAL_ENABLED", "")
    monkeypatch.setenv("PERSONA_RUNTIME_EXPLORER_ENABLED", "null")

    settings = _load_settings_for_test()

    assert settings["PERSONA_DIALOGUE_TREE_EVAL_ENABLED"] is True
    assert settings["PERSONA_RUNTIME_EXPLORER_ENABLED"] is True


def test_persona_runtime_explorer_invalid_env_int_raises(monkeypatch):
    monkeypatch.setenv("PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS", "many")

    with pytest.raises(ValueError, match="PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS"):
        _load_settings_for_test()
