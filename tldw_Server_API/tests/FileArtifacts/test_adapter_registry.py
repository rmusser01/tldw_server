from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.File_Artifacts import adapter_registry as registry_mod
from tldw_Server_API.app.core.File_Artifacts.adapter_registry import FileAdapterRegistry
from tldw_Server_API.app.core.exceptions import AdapterInitializationError


class BoomAdapter:
    file_type = "boom"
    export_formats: ClassVar[set[str]] = set()

    def __init__(self) -> None:
        raise RuntimeError("boom")


def test_get_adapter_missing_returns_none():
    registry = FileAdapterRegistry()
    assert registry.get_adapter("missing_adapter") is None


def test_get_adapter_init_failure_raises():
    registry = FileAdapterRegistry()
    registry.register_adapter("boom", BoomAdapter)
    with pytest.raises(AdapterInitializationError) as excinfo:
        registry.get_adapter("boom")
    assert excinfo.value.adapter_name == "boom"
    with pytest.raises(AdapterInitializationError):
        registry.get_adapter("boom")


def test_get_adapter_init_failure_log_omits_raw_spec_and_error(monkeypatch):
    logger_mock = MagicMock()
    monkeypatch.setattr(registry_mod, "logger", logger_mock)
    registry = FileAdapterRegistry()
    raw_spec = "/private/raw-adapter-spec-token"
    registry.register_adapter("leaky", raw_spec)

    with pytest.raises(AdapterInitializationError) as excinfo:
        registry.get_adapter("leaky")

    logger_mock.error.assert_called_once_with(
        "Failed to initialize file adapter error_type={}",
        "ImportError",
    )
    rendered_log_call = str(logger_mock.error.call_args)
    assert raw_spec not in rendered_log_call
    assert "raw-adapter-spec-token" in excinfo.value.detail
    assert "Invalid adapter spec" in excinfo.value.detail
