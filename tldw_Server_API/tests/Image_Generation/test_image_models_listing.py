from pathlib import Path

import pytest

from tldw_Server_API.app.core.Image_Generation.config import ImageGenerationConfig
from tldw_Server_API.app.core.Image_Generation import listing


class _FakeAdapter:
    supported_formats = {"png", "jpg", "webp"}


class _FakeRegistry:
    def __init__(self, names):
        self._names = list(names)

    def list_backend_names(self, *, include_disabled: bool = False):
        return list(self._names)

    def get_adapter_class(self, name):
        if name in self._names:
            return _FakeAdapter
        return None


def _make_config(**overrides) -> ImageGenerationConfig:
    base = dict(
        default_backend="stable_diffusion_cpp",
        enabled_backends=["stable_diffusion_cpp"],
        max_width=1024,
        max_height=1024,
        max_pixels=1024 * 1024,
        max_steps=50,
        max_prompt_length=1000,
        inline_max_bytes=4000000,
        sd_cpp_diffusion_model_path=None,
        sd_cpp_llm_path=None,
        sd_cpp_binary_path=None,
        sd_cpp_model_path=None,
        sd_cpp_vae_path=None,
        sd_cpp_lora_paths=[],
        sd_cpp_allowed_extra_params=[],
        sd_cpp_default_steps=25,
        sd_cpp_default_cfg_scale=7.5,
        sd_cpp_default_sampler="euler_a",
        sd_cpp_device="auto",
        sd_cpp_timeout_seconds=120,
        swarmui_base_url=None,
        swarmui_default_model=None,
        swarmui_swarm_token=None,
        swarmui_allowed_extra_params=[],
        swarmui_timeout_seconds=120,
        openrouter_image_base_url=None,
        openrouter_image_api_key=None,
        openrouter_image_default_model=None,
        openrouter_image_allowed_extra_params=[],
        openrouter_image_timeout_seconds=120,
        novita_image_base_url=None,
        novita_image_api_key=None,
        novita_image_default_model=None,
        novita_image_allowed_extra_params=[],
        novita_image_timeout_seconds=180,
        novita_image_poll_interval_seconds=2,
        together_image_base_url=None,
        together_image_api_key=None,
        together_image_default_model=None,
        together_image_allowed_extra_params=[],
        together_image_timeout_seconds=120,
        modelstudio_image_base_url=None,
        modelstudio_image_api_key=None,
        modelstudio_image_default_model=None,
        modelstudio_image_region="sg",
        modelstudio_image_mode="auto",
        modelstudio_image_poll_interval_seconds=2,
        modelstudio_image_timeout_seconds=180,
        modelstudio_image_allowed_extra_params=[],
    )
    base.update(overrides)
    return ImageGenerationConfig(**base)


def _touch(path: Path) -> Path:
    path.write_text("x", encoding="utf-8")
    return path


def test_list_image_models_configured(monkeypatch, tmp_path):
    binary = _touch(tmp_path / "sd-cli")
    model = _touch(tmp_path / "model.gguf")
    cfg = _make_config(sd_cpp_binary_path=str(binary), sd_cpp_model_path=str(model))

    monkeypatch.setattr(listing, "get_image_generation_config", lambda: cfg)
    monkeypatch.setattr(listing, "get_registry", lambda: _FakeRegistry(["stable_diffusion_cpp"]))

    models = listing.list_image_models_for_catalog()
    assert len(models) == 1
    entry = models[0]
    assert entry["id"] == "image/stable_diffusion_cpp"
    assert entry["type"] == "image"
    assert entry["is_configured"] is True
    assert entry["capabilities"]["image_reference_input"] is False
    assert "supported_formats" in entry
    assert "png" in entry["supported_formats"]


def test_list_image_models_unconfigured_missing_model(monkeypatch, tmp_path):
    binary = _touch(tmp_path / "sd-cli")
    cfg = _make_config(sd_cpp_binary_path=str(binary))

    monkeypatch.setattr(listing, "get_image_generation_config", lambda: cfg)
    monkeypatch.setattr(listing, "get_registry", lambda: _FakeRegistry(["stable_diffusion_cpp"]))

    models = listing.list_image_models_for_catalog()
    assert len(models) == 1
    entry = models[0]
    assert entry["is_configured"] is False


@pytest.mark.parametrize(
    ("preferred_exists", "legacy_exists", "configured"),
    [(False, True, False), (True, False, True), (True, True, True)],
)
def test_list_image_models_validates_preferred_diffusion_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preferred_exists: bool,
    legacy_exists: bool,
    configured: bool,
) -> None:
    """Do not hide an invalid preferred model behind an unused legacy model."""
    binary = _touch(tmp_path / "sd-cli")
    preferred = tmp_path / "diffusion.gguf"
    legacy = tmp_path / "legacy.gguf"
    if preferred_exists:
        _touch(preferred)
    if legacy_exists:
        _touch(legacy)
    cfg = _make_config(
        sd_cpp_binary_path=str(binary),
        sd_cpp_diffusion_model_path=str(preferred),
        sd_cpp_model_path=str(legacy),
    )
    monkeypatch.setattr(listing, "get_image_generation_config", lambda: cfg)
    monkeypatch.setattr(listing, "get_registry", lambda: _FakeRegistry(["stable_diffusion_cpp"]))

    models = listing.list_image_models_for_catalog()

    assert models[0]["is_configured"] is configured


def test_list_image_models_swarmui_configured(monkeypatch):
    cfg = _make_config(enabled_backends=["swarmui"], swarmui_base_url="http://localhost:7801")

    monkeypatch.setattr(listing, "get_image_generation_config", lambda: cfg)
    monkeypatch.setattr(listing, "get_registry", lambda: _FakeRegistry(["swarmui"]))

    models = listing.list_image_models_for_catalog()
    assert len(models) == 1
    entry = models[0]
    assert entry["id"] == "image/swarmui"
    assert entry["is_configured"] is True
    assert entry["capabilities"]["image_reference_input"] is False


def test_list_image_models_swarmui_unconfigured(monkeypatch):
    cfg = _make_config(enabled_backends=["swarmui"], swarmui_base_url=None)

    monkeypatch.setattr(listing, "get_image_generation_config", lambda: cfg)
    monkeypatch.setattr(listing, "get_registry", lambda: _FakeRegistry(["swarmui"]))

    models = listing.list_image_models_for_catalog()
    assert len(models) == 1
    entry = models[0]
    assert entry["is_configured"] is False


@pytest.mark.parametrize(
    ("backend", "key_field", "env_var", "fallback_env_var"),
    [
        ("openrouter", "openrouter_image_api_key", "OPENROUTER_API_KEY", None),
        ("novita", "novita_image_api_key", "NOVITA_API_KEY", None),
        ("together", "together_image_api_key", "TOGETHER_API_KEY", None),
        ("modelstudio", "modelstudio_image_api_key", "DASHSCOPE_API_KEY", "QWEN_API_KEY"),
    ],
)
def test_list_image_models_remote_backend_configured_via_api_key(
    monkeypatch,
    backend: str,
    key_field: str,
    env_var: str,
    fallback_env_var: str | None,
):
    monkeypatch.delenv(env_var, raising=False)
    if fallback_env_var:
        monkeypatch.delenv(fallback_env_var, raising=False)
    cfg = _make_config(enabled_backends=[backend], **{key_field: "sk-test"})

    monkeypatch.setattr(listing, "get_image_generation_config", lambda: cfg)
    monkeypatch.setattr(listing, "get_registry", lambda: _FakeRegistry([backend]))

    models = listing.list_image_models_for_catalog()
    assert len(models) == 1
    entry = models[0]
    assert entry["id"] == f"image/{backend}"
    assert entry["is_configured"] is True


@pytest.mark.parametrize(
    ("backend", "key_field", "env_var", "fallback_env_var"),
    [
        ("openrouter", "openrouter_image_api_key", "OPENROUTER_API_KEY", None),
        ("novita", "novita_image_api_key", "NOVITA_API_KEY", None),
        ("together", "together_image_api_key", "TOGETHER_API_KEY", None),
        ("modelstudio", "modelstudio_image_api_key", "DASHSCOPE_API_KEY", "QWEN_API_KEY"),
    ],
)
def test_list_image_models_remote_backend_unconfigured_without_api_key(
    monkeypatch,
    backend: str,
    key_field: str,
    env_var: str,
    fallback_env_var: str | None,
):
    monkeypatch.delenv(env_var, raising=False)
    if fallback_env_var:
        monkeypatch.delenv(fallback_env_var, raising=False)
    cfg = _make_config(enabled_backends=[backend], **{key_field: None})

    monkeypatch.setattr(listing, "get_image_generation_config", lambda: cfg)
    monkeypatch.setattr(listing, "get_registry", lambda: _FakeRegistry([backend]))

    models = listing.list_image_models_for_catalog()
    assert len(models) == 1
    entry = models[0]
    assert entry["id"] == f"image/{backend}"
    assert entry["is_configured"] is False


def test_list_image_models_modelstudio_configured_via_qwen_env(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("QWEN_API_KEY", "sk-qwen")
    cfg = _make_config(
        enabled_backends=["modelstudio"],
        modelstudio_image_api_key=None,
        modelstudio_image_default_model="foo-model",
    )

    monkeypatch.setattr(listing, "get_image_generation_config", lambda: cfg)
    monkeypatch.setattr(listing, "get_registry", lambda: _FakeRegistry(["modelstudio"]))

    models = listing.list_image_models_for_catalog()
    assert len(models) == 1
    entry = models[0]
    assert entry["id"] == "image/modelstudio"
    assert entry["is_configured"] is True
    assert entry["capabilities"]["image_reference_input"] is True


def test_list_image_models_reference_capability_follows_config_override(monkeypatch):
    cfg = _make_config(
        enabled_backends=["modelstudio"],
        modelstudio_image_default_model="foo-model",
        reference_image_supported_models={"modelstudio": []},
    )

    monkeypatch.setattr(listing, "get_image_generation_config", lambda: cfg)
    monkeypatch.setattr(listing, "get_registry", lambda: _FakeRegistry(["modelstudio"]))

    models = listing.list_image_models_for_catalog()
    assert len(models) == 1
    entry = models[0]
    assert entry["id"] == "image/modelstudio"
    assert entry["capabilities"]["image_reference_input"] is False
