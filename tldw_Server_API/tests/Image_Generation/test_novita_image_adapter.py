import pytest

from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest
from tldw_Server_API.app.core.Image_Generation.adapters import novita_image_adapter as novita_module
from tldw_Server_API.app.core.Image_Generation.config import ImageGenerationConfig
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageGenerationError

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="


def _make_config(**overrides) -> ImageGenerationConfig:
    base = dict(
        default_backend="novita",
        enabled_backends=["novita"],
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
        novita_image_base_url="https://api.novita.ai",
        novita_image_api_key="sk-novita",
        novita_image_default_model="sdxl",
        novita_image_allowed_extra_params=[],
        novita_image_timeout_seconds=30,
        novita_image_poll_interval_seconds=1,
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


def _make_request(**overrides) -> ImageGenRequest:
    base = dict(
        backend="novita",
        prompt="portrait at golden hour",
        negative_prompt="blurry",
        width=768,
        height=1024,
        steps=30,
        cfg_scale=6.5,
        seed=12345,
        sampler="euler",
        model=None,
        format="png",
        extra_params={},
        request_id=None,
    )
    base.update(overrides)
    return ImageGenRequest(**base)


def test_novita_generate_async_polling_success(monkeypatch):
    cfg = _make_config()
    monkeypatch.setattr(novita_module, "get_image_generation_config", lambda: cfg)

    calls = []
    poll_counter = {"count": 0}

    def fake_fetch_json(method, url, headers, timeout, **kwargs):
        calls.append((method, url, kwargs))
        if method == "POST" and url.endswith("/v3/async/txt2img"):
            return {"task_id": "task-123"}
        if method == "GET" and url.endswith("/v3/async/task-result"):
            poll_counter["count"] += 1
            if poll_counter["count"] == 1:
                return {"status": "pending"}
            return {"status": "success", "images": [{"image_base64": PNG_B64}]}
        raise AssertionError("unexpected request")

    monkeypatch.setattr(novita_module, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(novita_module.time, "sleep", lambda *_args, **_kwargs: None)

    adapter = novita_module.NovitaImageAdapter()
    result = adapter.generate(_make_request())
    assert result.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert result.content_type == "image/png"
    assert result.bytes_len == len(result.content)
    assert poll_counter["count"] == 2
    assert any(call[1].endswith("/v3/async/txt2img") for call in calls)
    assert any(call[1].endswith("/v3/async/task-result") for call in calls)


def test_novita_generate_async_polling_failure(monkeypatch):
    cfg = _make_config()
    monkeypatch.setattr(novita_module, "get_image_generation_config", lambda: cfg)

    def fake_fetch_json(method, url, headers, timeout, **kwargs):
        if method == "POST" and url.endswith("/v3/async/txt2img"):
            return {"task_id": "task-456"}
        if method == "GET" and url.endswith("/v3/async/task-result"):
            return {"status": "failed", "message": "safety check failed"}
        raise AssertionError("unexpected request")

    monkeypatch.setattr(novita_module, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(novita_module.time, "sleep", lambda *_args, **_kwargs: None)

    adapter = novita_module.NovitaImageAdapter()
    with pytest.raises(ImageGenerationError, match="Novita task failed"):
        adapter.generate(_make_request())
