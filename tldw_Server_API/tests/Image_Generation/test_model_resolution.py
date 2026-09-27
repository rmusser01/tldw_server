"""Shared, provider-free effective model selection."""

from dataclasses import replace

import pytest

from tldw_Server_API.app.core.Image_Generation import config as image_config


@pytest.mark.parametrize("backend", ["openrouter", "novita", "together", "modelstudio"])
def test_cloud_model_precedence(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve request, environment, configuration, and built-in precedence."""
    monkeypatch.setattr(image_config, "_config_cache", None)
    monkeypatch.setattr(image_config, "get_config_section", lambda *_args, **_kwargs: {})
    env_name = f"{backend.upper()}_IMAGE_MODEL"
    monkeypatch.delenv(env_name, raising=False)
    config = image_config.get_image_generation_config()
    default = getattr(image_config, f"DEFAULT_{backend.upper()}_IMAGE_MODEL")
    config = replace(config, **{f"{backend}_image_default_model": None})
    assert image_config.resolve_image_generation_model(backend, None, config) == default
    config = replace(config, **{f"{backend}_image_default_model": "configured"})
    assert image_config.resolve_image_generation_model(backend, None, config) == "configured"
    monkeypatch.setenv(env_name, "environment")
    assert image_config.resolve_image_generation_model(backend, None, config) == "environment"
    assert image_config.resolve_image_generation_model(backend, "requested", config) == "requested"


def test_local_defaults_do_not_expose_filesystem_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only SwarmUI has a public local default model identifier."""
    monkeypatch.setattr(image_config, "_config_cache", None)
    monkeypatch.setattr(image_config, "get_config_section", lambda *_args, **_kwargs: {})
    config = replace(
        image_config.get_image_generation_config(),
        swarmui_default_model="swarm-model",
        sd_cpp_model_path="/private/model.gguf",
    )
    assert image_config.resolve_image_generation_model("swarmui", None, config) == "swarm-model"
    assert image_config.resolve_image_generation_model("swarmui", "requested", config) == "requested"
    for backend in ("stable_diffusion_cpp", "custom", None):
        assert image_config.resolve_image_generation_model(backend, None, config) is None
