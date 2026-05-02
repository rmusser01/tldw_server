import pytest

from tldw_Server_API.app.core import config_utils

pytestmark = pytest.mark.unit


def test_load_module_yaml_failure_log_omits_exception_details(monkeypatch, tmp_path):
    leaked_path = "/private/config/module.yaml"
    leaked_token = "token=secret-config"
    module_file = tmp_path / "module.yaml"
    module_file.write_text("invalid: [", encoding="utf-8")
    messages: list[str] = []

    def fail_safe_load(_file_obj):
        raise RuntimeError(f"yaml parser failed at {leaked_path} {leaked_token}")

    monkeypatch.setattr(config_utils, "resolve_module_yaml", lambda *_args, **_kwargs: module_file)
    monkeypatch.setattr(config_utils.yaml, "safe_load", fail_safe_load)
    monkeypatch.setattr(config_utils.logger, "warning", messages.append)

    data, path = config_utils.load_module_yaml("search-agent")

    assert data == {}
    assert path == module_file
    assert messages == ["Failed to load module YAML for search-agent"]
    joined = "\n".join(messages)
    assert leaked_path not in joined
    assert leaked_token not in joined
    assert "yaml parser failed" not in joined
