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


def test_request_id_helpers_live_in_logging() -> None:
    from tldw_Server_API.app.core.Logging import log_context
    from tldw_Server_API.app.core.Security import request_id_middleware

    assert log_context._clean_request_id.__module__ == "tldw_Server_API.app.core.Logging.log_context"
    assert request_id_middleware._clean_request_id is log_context._clean_request_id
    assert log_context._clean_request_id("req-1") == "req-1"
    assert log_context._clean_request_id("bad id!") != "bad id!"


def test_stt_observability_context_is_one_contextvar() -> None:
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core import stt_observability_context as old
    from tldw_Server_API.app.core.Logging import stt_observability_context as new

    assert old.OPAQUE_STT_ENDPOINT_ID is new.OPAQUE_STT_ENDPOINT_ID
    endpoint_id = "sha256:" + "a" * 64
    with http_client.opaque_stt_http_observability(endpoint_id):
        assert old.get_opaque_stt_endpoint_id() == endpoint_id
    assert new.get_opaque_stt_endpoint_id() is None


def test_prompt_loader_lives_in_context_integrity_with_no_utils_shim() -> None:
    import importlib

    from tldw_Server_API.app.core.Context_Integrity import prompt_loader

    assert callable(prompt_loader.load_prompt)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tldw_Server_API.app.core.Utils.prompt_loader")


def test_utils_no_longer_owns_db_code() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import safe_metadata_write_ops
    from tldw_Server_API.app.core.Utils import Utils, metadata_utils

    assert callable(safe_metadata_write_ops.update_version_safe_metadata_in_transaction)
    assert not hasattr(metadata_utils, "update_version_safe_metadata_in_transaction")
    assert not hasattr(Utils, "get_db_config")
