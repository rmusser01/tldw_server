from __future__ import annotations

import configparser
from pathlib import Path

import pytest

from tldw_Server_API.tests.run_first_constants import (
    PHASE2C_RUN_FIRST_COHORT,
    PHASE2C_RUN_FIRST_CSV,
)

pytestmark = pytest.mark.unit


def test_resolve_chat_run_first_rollout_mode_defaults_off_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config

    monkeypatch.delenv("CHAT_RUN_FIRST_ROLLOUT_MODE", raising=False)
    monkeypatch.delenv("CHAT_RUN_FIRST_PROVIDER_ALLOWLIST", raising=False)
    monkeypatch.delenv("CHAT_RUN_FIRST_PRESENTATION_VARIANT", raising=False)
    monkeypatch.delenv("ACP_RUN_FIRST_ROLLOUT_MODE", raising=False)
    monkeypatch.delenv("ACP_RUN_FIRST_PROVIDER_ALLOWLIST", raising=False)
    monkeypatch.delenv("ACP_RUN_FIRST_PRESENTATION_VARIANT", raising=False)
    monkeypatch.setattr(config, "load_comprehensive_config", lambda: configparser.ConfigParser())

    assert config.resolve_chat_run_first_rollout_mode() == "off"
    assert config.resolve_chat_run_first_provider_allowlist() == []
    assert config.resolve_chat_run_first_presentation_variant() == "chat_phase2b_v1"


def test_resolve_chat_run_first_provider_allowlist_parses_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config

    monkeypatch.setenv(
        "CHAT_RUN_FIRST_PROVIDER_ALLOWLIST",
        PHASE2C_RUN_FIRST_CSV,
    )

    assert config.resolve_chat_run_first_provider_allowlist() == PHASE2C_RUN_FIRST_COHORT


def test_resolve_chat_run_first_rollout_mode_uses_chat_module_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config

    monkeypatch.delenv("CHAT_RUN_FIRST_ROLLOUT_MODE", raising=False)

    parser = configparser.ConfigParser()
    parser.add_section("Chat-Module")
    parser.set("Chat-Module", "run_first_rollout_mode", "gated")
    monkeypatch.setattr(config, "load_comprehensive_config", lambda: parser)

    assert config.resolve_chat_run_first_rollout_mode() == "gated"


def test_resolve_chat_run_first_rollout_mode_accepts_default_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config

    monkeypatch.setenv("CHAT_RUN_FIRST_ROLLOUT_MODE", "default_on")

    assert config.resolve_chat_run_first_rollout_mode() == "default_on"


def test_resolve_chat_run_first_provider_allowlist_uses_chat_module_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config

    monkeypatch.delenv("CHAT_RUN_FIRST_PROVIDER_ALLOWLIST", raising=False)

    parser = configparser.ConfigParser()
    parser.add_section("Chat-Module")
    parser.set(
        "Chat-Module",
        "run_first_provider_allowlist",
        PHASE2C_RUN_FIRST_CSV,
    )
    monkeypatch.setattr(config, "load_comprehensive_config", lambda: parser)

    assert config.resolve_chat_run_first_provider_allowlist() == PHASE2C_RUN_FIRST_COHORT


def test_phase2c_shipped_run_first_defaults_match_config_and_env_examples() -> None:
    from tldw_Server_API.app.core import config

    repo_root = Path(__file__).resolve().parents[4]
    config_path = repo_root / "tldw_Server_API" / "Config_Files" / "config.txt"
    env_example_path = repo_root / "tldw_Server_API" / "Config_Files" / ".env.example"

    parser = configparser.ConfigParser()
    assert parser.read(config_path) == [str(config_path)]

    assert config._split_run_first_provider_allowlist(
        parser.get("Chat-Module", "run_first_provider_allowlist")
    ) == PHASE2C_RUN_FIRST_COHORT
    assert config._split_run_first_provider_allowlist(
        parser.get("ACP", "run_first_provider_allowlist")
    ) == PHASE2C_RUN_FIRST_COHORT

    env_lines = env_example_path.read_text(encoding="utf-8").splitlines()
    assert f"#CHAT_RUN_FIRST_PROVIDER_ALLOWLIST={PHASE2C_RUN_FIRST_CSV}" in env_lines
    assert f"#ACP_RUN_FIRST_PROVIDER_ALLOWLIST={PHASE2C_RUN_FIRST_CSV}" in env_lines


def test_resolve_chat_run_first_presentation_variant_uses_chat_module_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config

    monkeypatch.delenv("CHAT_RUN_FIRST_PRESENTATION_VARIANT", raising=False)

    parser = configparser.ConfigParser()
    parser.add_section("Chat-Module")
    parser.set("Chat-Module", "run_first_presentation_variant", "chat_phase2a_v2")
    monkeypatch.setattr(config, "load_comprehensive_config", lambda: parser)

    assert config.resolve_chat_run_first_presentation_variant() == "chat_phase2a_v2"


def test_resolve_run_first_cohort_label_maps_default_on_out_of_cohort() -> None:
    from tldw_Server_API.app.core import config

    assert config.resolve_run_first_cohort_label(
        "default_on",
        eligible=False,
        ineligible_reason="provider_not_in_rollout_allowlist",
    ) == "out_of_cohort"


def test_resolve_run_first_cohort_label_maps_gated_allowlist_miss_to_out_of_cohort() -> None:
    from tldw_Server_API.app.core import config

    assert config.resolve_run_first_cohort_label(
        "gated",
        eligible=False,
        ineligible_reason="provider_not_in_rollout_allowlist",
    ) == "out_of_cohort"


def test_run_first_resolvers_have_docstrings() -> None:
    from tldw_Server_API.app.core import config

    assert config.resolve_chat_run_first_rollout_mode.__doc__
    assert config.resolve_chat_run_first_provider_allowlist.__doc__
    assert config.resolve_chat_run_first_presentation_variant.__doc__
    assert config.resolve_acp_run_first_rollout_mode.__doc__
    assert config.resolve_acp_run_first_provider_allowlist.__doc__
    assert config.resolve_acp_run_first_presentation_variant.__doc__
