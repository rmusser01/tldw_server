from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Image_Generation.adapters import stable_diffusion_cpp_adapter as stable_module
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest
from tldw_Server_API.app.core.Image_Generation.config import ImageGenerationConfig
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageGenerationError


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


def test_stable_diffusion_failure_redacts_prompt_paths_and_tokens(monkeypatch, tmp_path):
    binary_path = tmp_path / "sd"
    model_path = tmp_path / "secret-model.gguf"
    binary_path.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path.write_text("model", encoding="utf-8")
    secret_prompt = "confidential prompt token-123"

    cfg = _make_config(
        sd_cpp_binary_path=str(binary_path),
        sd_cpp_model_path=str(model_path),
        sd_cpp_allowed_extra_params=["cli_args"],
    )
    monkeypatch.setattr(stable_module, "get_image_generation_config", lambda: cfg)

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=1,
            stderr=f"failed for {secret_prompt} using {model_path} and token-123",
        )

    monkeypatch.setattr(stable_module.subprocess, "run", fake_run)

    logs: list[str] = []
    sink_id = stable_module.logger.add(logs.append, format="{message}")
    try:
        adapter = stable_module.StableDiffusionCppAdapter()
        with pytest.raises(ImageGenerationError) as exc_info:
            adapter.generate(
                ImageGenRequest(
                    backend="stable_diffusion_cpp",
                    prompt=secret_prompt,
                    negative_prompt=None,
                    width=512,
                    height=512,
                    steps=20,
                    cfg_scale=7.5,
                    seed=None,
                    sampler=None,
                    model=None,
                    format="png",
                    extra_params={"cli_args": ["--api-key", "token-123"]},
                    request_id=None,
                )
            )
    finally:
        stable_module.logger.remove(sink_id)

    error_text = str(exc_info.value)
    log_text = "\n".join(logs)
    assert secret_prompt not in error_text
    assert str(model_path) not in error_text
    assert "token-123" not in error_text
    assert secret_prompt not in log_text
    assert str(model_path) not in log_text
    assert "token-123" not in log_text
