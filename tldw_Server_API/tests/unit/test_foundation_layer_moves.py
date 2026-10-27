"""The moves that cut the foundation layer's upward imports (TASK-13950.12)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_database_error_is_defined_in_core_exceptions_and_reexported_by_authnz() -> None:
    from tldw_Server_API.app.core import exceptions as core_exc
    from tldw_Server_API.app.core.AuthNZ import exceptions as authnz_exc

    assert core_exc.DatabaseError.__module__ == "tldw_Server_API.app.core.exceptions"
    assert authnz_exc.DatabaseError is core_exc.DatabaseError
    assert authnz_exc.UserRegistrationException is core_exc.UserRegistrationException
    assert issubclass(core_exc.DatabaseError, core_exc.UserRegistrationException)
    assert issubclass(authnz_exc.TransactionError, core_exc.DatabaseError)
    assert issubclass(core_exc.AudioQuotaStoreUnavailable, authnz_exc.DatabaseError)


def test_provider_identity_lives_in_utils_and_the_llm_calls_path_reexports_it() -> None:
    from tldw_Server_API.app.core.LLM_Calls import provider_identity as old
    from tldw_Server_API.app.core.Utils import provider_identity as new

    assert new.__name__ == "tldw_Server_API.app.core.Utils.provider_identity"
    assert old.canonical_provider_name is new.canonical_provider_name
    assert old.provider_lookup_names is new.provider_lookup_names
    assert old.PROVIDER_ALIASES is new.PROVIDER_ALIASES
    assert new.canonical_provider_name("OpenAI-Compatible") == "custom-openai-api"


def test_llamacpp_handler_settings_are_plain_data(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from tldw_Server_API.app.core import config

    monkeypatch.setenv("LLAMACPP_ENABLED", "1")
    monkeypatch.setenv("LLAMACPP_PORT", "9001")
    monkeypatch.setenv("LLAMACPP_ALLOWED_PATHS", "/models/a, /models/b")

    settings = config.get_llamacpp_handler_settings()

    assert settings is not None
    assert settings["enabled"] is True and settings["default_port"] == 9001
    assert settings["allowed_paths"] == [Path("/models/a"), Path("/models/b")]


def test_llamacpp_handler_settings_are_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core import config

    monkeypatch.setenv("LLAMACPP_ENABLED", "0")
    assert config.get_llamacpp_handler_settings() is None
