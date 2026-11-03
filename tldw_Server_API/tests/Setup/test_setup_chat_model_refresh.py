"""Provider setup changes must reach strict Chat validation without a restart."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import setup as setup_endpoint
from tldw_Server_API.app.api.v1.schemas.setup_schemas import SetupProviderSaveRequest
from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.custom_openai_providers import custom_openai_model_env_keys
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.app.core.Setup import setup_manager


@pytest.fixture
def isolated_model_config(tmp_path, monkeypatch):
    """Use actual config writes/loaders, isolated from the developer's settings."""
    path = tmp_path / "config.txt"
    path.write_text(
        "[API]\ncustom_openai_api_model = startup-custom\n"
        "custom_openai2_api_model = startup-second\n"
        "[Local-API]\nollama_model = startup-local\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "resolve_config_file", lambda: path)
    monkeypatch.setattr(config, "_candidate_env_paths", lambda *_args: [])
    monkeypatch.setattr(setup_manager, "get_config_file_path", lambda: path)
    # Catalog transport is unrelated; retain a catalog-only positive control.
    monkeypatch.setattr(chat_service, "list_provider_models", lambda _provider: ["catalog-model"])
    monkeypatch.setattr(gateway, "list_provider_models", chat_service.list_provider_models)
    for number in (1, 2):
        for name in custom_openai_model_env_keys(number):
            monkeypatch.delenv(name, raising=False)
    config.clear_config_cache()
    chat_service.invalidate_model_alias_caches()
    # Reproduce a worker that imported Chat before first-run setup saved a model.
    monkeypatch.setattr(chat_service, "_config", config.load_comprehensive_config())
    yield path
    config.clear_config_cache()
    chat_service.invalidate_model_alias_caches()


@pytest.mark.asyncio
@pytest.mark.parametrize("warm_inventory", [False, True])
@pytest.mark.parametrize(
    ("setup_provider", "chat_provider", "startup_model"),
    [
        ("custom_openai", "custom-openai-api", "startup-custom"),
        ("ollama", "ollama", "startup-local"),
    ],
)
async def test_setup_save_refreshes_cold_and_warm_chat_inventory_without_restart(
    isolated_model_config, warm_inventory, setup_provider, chat_provider, startup_model
):
    if warm_inventory:
        assert chat_service.is_model_known_for_provider(chat_provider, startup_model) is True

    for model in ("first-saved-model", "second-saved-model"):
        result = await setup_endpoint.save_first_run_provider(
            SetupProviderSaveRequest(provider_key=setup_provider, model=model), _guard=None
        )
        assert result.status.value == "saved"
        assert result.requires_restart is False
        request = SimpleNamespace(api_provider=chat_provider, model=model)
        provider = chat_service.normalize_request_provider_and_model(request, default_provider=chat_provider)
        assert provider == chat_provider
        assert chat_service.is_model_known_for_provider(provider, request.model) is True
        assert chat_service.is_model_known_for_provider(provider, startup_model) is False
        assert chat_service.is_model_known_for_provider(provider, "unrelated-model") is False
        assert chat_service.is_model_known_for_provider(provider, "catalog-model") is True


def test_numbered_custom_provider_refresh_remains_scoped(isolated_model_config):
    assert chat_service.is_model_known_for_provider("custom-openai-api-2", "startup-second") is True
    setup_manager.update_config({"API": {"custom_openai2_api_model": "new-second"}})
    config.clear_config_cache()

    assert chat_service.is_model_known_for_provider("custom-openai-api-2", "new-second") is True
    assert chat_service.is_model_known_for_provider("custom-openai-api", "new-second") is False
    assert chat_service.is_model_known_for_provider("custom-openai-api-2", "startup-second") is False


@pytest.mark.asyncio
async def test_setup_save_keeps_effective_environment_model_precedence(isolated_model_config, monkeypatch):
    monkeypatch.setenv("CUSTOM_OPENAI_API_MODEL", "environment-model")
    result = await setup_endpoint.save_first_run_provider(
        SetupProviderSaveRequest(provider_key="custom_openai", model="saved-file-model"),
        _guard=None,
    )

    assert result.status.value == "saved"
    assert chat_service.is_model_known_for_provider("custom-openai-api", "environment-model") is True
    assert chat_service.is_model_known_for_provider("custom-openai-api", "saved-file-model") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint_keys",
    [
        ("ollama_api_ip",),
        ("ollama_api_IP",),
        ("ollama_api_ip", "ollama_api_IP"),
        ("ollama_api_IP", "ollama_api_ip"),
    ],
)
async def test_a01_provider_save_keeps_case_aliases_readable(isolated_model_config, endpoint_keys):
    endpoint_lines = "".join(
        (
            f"# endpoint-{index}\n{key} = http://127.0.0.1:11434/old\n"
            if index == 0
            else f"{key} = http://127.0.0.1:11434/old # endpoint-{index}\n"
        )
        for index, key in enumerate(endpoint_keys)
    )
    unrelated = "[Other]\nKeepMe = keep # unrelated\n"
    isolated_model_config.write_text(
        "[Local-API]\n" + endpoint_lines + "ollama_model = old-model\n" + unrelated,
        encoding="utf-8",
    )

    result = await setup_endpoint.save_first_run_provider(
        SetupProviderSaveRequest(
            provider_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
            model="qwen2.5:0.5b",
        ),
        _guard=None,
    )

    assert result.status.value == "saved"  # nosec B101 # pytest verifies the real provider save.
    assert result.requires_restart is False  # nosec B101 # pytest verifies live cache refresh.
    section = config.get_config_section("Local-API", reload=True)
    assert section["ollama_api_ip"] == "http://127.0.0.1:11434/v1"  # nosec B101 # pytest verifies runtime parsing.
    assert section["ollama_model"] == "qwen2.5:0.5b"  # nosec B101 # pytest verifies the selected model.
    written = isolated_model_config.read_text(encoding="utf-8")
    assert unrelated in written  # nosec B101 # pytest protects unrelated config and comments.
    assert all(f"# endpoint-{index}" in written for index in range(len(endpoint_keys)))  # nosec B101 # pytest protects alias comments.
    parser = setup_manager._load_config_parser()
    assert tuple(parser["Local-API"]) == (endpoint_keys[0], "ollama_model")  # nosec B101 # pytest verifies one option with its original spelling.


def test_a01_config_update_rejects_conflicting_case_aliases(isolated_model_config):
    original = "[Local-API]\nollama_api_ip = http://127.0.0.1:11434/v1\n"
    isolated_model_config.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate config option"):
        setup_manager.update_config(
            {
                "Local-API": {
                    "ollama_api_IP": "http://127.0.0.1:11434/v1",
                    "ollama_api_ip": "http://127.0.0.1:11435/v1",
                }
            },
            create_backup=False,
        )

    assert isolated_model_config.read_text(encoding="utf-8") == original  # nosec B101 # pytest verifies validation precedes the write.


def test_a01_existing_config_key_recovers_in_a_cold_process(isolated_model_config):
    import os
    import subprocess  # nosec B404 # Fixed supported interpreter is required to test a genuinely cold config process.
    import sys
    from pathlib import Path

    isolated_model_config.write_text(
        "[Local-API]\nollama_api_ip = old\nollama_api_IP = old\n",
        encoding="utf-8",
    )
    root = Path(setup_manager.__file__).resolve().parents[4]
    child_env = dict(os.environ)
    child_env["TLDW_CONFIG_FILE"] = str(isolated_model_config)
    child_env["PYTHONPATH"] = str(root)
    result = subprocess.run(  # nosec B603 # Fixed interpreter and source exercise the real cold config writer.
        [
            sys.executable,
            "-c",
            (
                "from tldw_Server_API.app.core.Setup import setup_manager; "
                "setup_manager.update_config({'Local-API': {'ollama_api_IP': "
                "'http://127.0.0.1:11434/v1'}}, create_backup=False)"
            ),
        ],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr  # nosec B101 # pytest verifies real cold recovery.
    parser = config.configparser.ConfigParser()
    parser.read(isolated_model_config)
    assert parser.get("Local-API", "ollama_api_ip") == "http://127.0.0.1:11434/v1"  # nosec B101 # pytest verifies strict runtime parsing after recovery.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "   ",
        "invalid://not-a-provider",
        "http://operator:private@127.0.0.1:8010/v1",
        "http://127.0.0.1:8010/v1?token=private",
        "http://127.0.0.1:8010/v1#private",
        "http://127.0.0.1:invalid/v1",
        "http:///v1",
        "http://bad host:8010/v1",
        "http://bad\thost:8010/v1",
        "http://bad\u00a0host:8010/v1",
        "http:\t//bad host:8010/v1",
        "http://bad%20host:8010/v1",
    ],
    ids=[
        "blank",
        "scheme",
        "credentials",
        "query",
        "fragment",
        "port",
        "host",
        "host-space",
        "host-tab",
        "host-nonbreaking-space",
        "delimiter-tab",
        "host-encoded-space",
    ],
)
async def test_a01_provider_save_rejects_invalid_url_before_config_write(isolated_model_config, base_url):
    original = (
        "[API]\ndefault_api = llama.cpp\n"
        "[Local-API]\nllama_api_IP = http://127.0.0.1:8010/v1\n"
        "llama_model = working-model\n[Other]\nKeepMe = keep # unrelated\n"
    )
    isolated_model_config.write_text(original, encoding="utf-8")

    with pytest.raises(HTTPException) as error:
        await setup_endpoint.save_first_run_provider(
            SetupProviderSaveRequest(
                provider_key="llamacpp",
                base_url=base_url,
                model="replacement-model",
                make_default=True,
            ),
            _guard=None,
        )

    assert error.value.status_code == 400  # nosec B101 - pytest verifies endpoint rejection.
    assert isolated_model_config.read_text(encoding="utf-8") == original  # nosec B101 - pytest preserves the real working config.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "field", "base_url"),
    [
        ("llamacpp", "llama_api_IP", "http://127.0.0.1:8010/v1"),
        ("koboldcpp", "kobold_api_IP", "https://localhost:5001/api/v1/generate"),
        ("llamacpp", "llama_api_IP", "http://[::1]:8010/v1"),
        ("llamacpp", "llama_api_IP", "http://llama.local:8010/native/path"),
    ],
)
async def test_a01_provider_save_normalizes_valid_http_endpoint(isolated_model_config, provider, field, base_url):
    result = await setup_endpoint.save_first_run_provider(
        SetupProviderSaveRequest(
            provider_key=provider,
            base_url=f"  {base_url}  ",
            model="working-model",
        ),
        _guard=None,
    )

    assert result.status.value == "saved"  # nosec B101 - pytest retains valid endpoint saves.
    assert setup_manager._load_config_parser().get("Local-API", field) == base_url  # nosec B101 - pytest verifies the normal real writer.
