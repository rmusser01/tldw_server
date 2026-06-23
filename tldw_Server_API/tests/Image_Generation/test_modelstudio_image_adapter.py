import base64

import pytest

from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest
from tldw_Server_API.app.core.Image_Generation.adapters import modelstudio_image_adapter as modelstudio_module
from tldw_Server_API.app.core.Image_Generation.adapters.image_format_utils import reference_image_data_url
from tldw_Server_API.app.core.Image_Generation.capabilities import ResolvedReferenceImage
from tldw_Server_API.app.core.Image_Generation.config import ImageGenerationConfig
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageGenerationError

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="


def _make_config(**overrides) -> ImageGenerationConfig:
    base = dict(
        default_backend="modelstudio",
        enabled_backends=["modelstudio"],
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
        modelstudio_image_base_url="https://dashscope-intl.aliyuncs.com/api/v1",
        modelstudio_image_api_key="sk-modelstudio",
        modelstudio_image_default_model="qwen-image",
        modelstudio_image_region="sg",
        modelstudio_image_mode="sync",
        modelstudio_image_poll_interval_seconds=2,
        modelstudio_image_timeout_seconds=180,
        modelstudio_image_allowed_extra_params=[],
    )
    base.update(overrides)
    return ImageGenerationConfig(**base)


def _make_request(**overrides) -> ImageGenRequest:
    base = dict(
        backend="modelstudio",
        prompt="draw a cat",
        negative_prompt=None,
        width=1024,
        height=1024,
        steps=None,
        cfg_scale=None,
        seed=None,
        sampler=None,
        model=None,
        format="png",
        extra_params={},
        request_id=None,
    )
    base.update(overrides)
    return ImageGenRequest(**base)


def _make_reference_image(**overrides) -> ResolvedReferenceImage:
    base = dict(
        file_id=123,
        filename="reference.png",
        mime_type="image/png",
        width=64,
        height=64,
        bytes_len=10,
        content=b"reference",
        temp_path=None,
    )
    base.update(overrides)
    return ResolvedReferenceImage(**base)


def test_reference_image_data_url_reads_temp_path(tmp_path):
    path = tmp_path / "reference.png"
    path.write_bytes(b"reference-bytes")

    reference_image = _make_reference_image(content=None, temp_path=str(path))

    assert reference_image_data_url(reference_image) == (
        "data:image/png;base64," + base64.b64encode(b"reference-bytes").decode("ascii")
    )


def test_modelstudio_sync_payload_includes_reference_image_content(monkeypatch):
    cfg = _make_config()
    monkeypatch.setattr(modelstudio_module, "get_image_generation_config", lambda: cfg)

    adapter = modelstudio_module.ModelStudioImageAdapter()
    payload = adapter._build_sync_payload(
        _make_request(model="qwen-image-edit-v1", reference_image=_make_reference_image())
    )

    assert payload["input"]["messages"][0]["content"] == [
        {"image": "data:image/png;base64," + base64.b64encode(b"reference").decode("ascii")},
        {"text": "draw a cat"},
    ]


def test_modelstudio_rejects_reference_image_for_unsupported_model(monkeypatch):
    cfg = _make_config()
    monkeypatch.setattr(modelstudio_module, "get_image_generation_config", lambda: cfg)
    monkeypatch.setattr(
        modelstudio_module,
        "fetch_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("upstream should not be called")),
    )

    adapter = modelstudio_module.ModelStudioImageAdapter()
    with pytest.raises(ImageGenerationError, match="unsupported model"):
        adapter.generate(_make_request(model="qwen-image", reference_image=_make_reference_image()))


def test_modelstudio_generate_sync_data_url(monkeypatch):
    cfg = _make_config()
    monkeypatch.setattr(modelstudio_module, "get_image_generation_config", lambda: cfg)

    captured = {}

    def fake_fetch_json(method, url, headers, json, timeout, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"image_url": f"data:image/png;base64,{PNG_B64}"},
                            ]
                        }
                    }
                ]
            }
        }

    monkeypatch.setattr(modelstudio_module, "fetch_json", fake_fetch_json)

    adapter = modelstudio_module.ModelStudioImageAdapter()
    result = adapter.generate(_make_request())
    assert result.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert result.content_type == "image/png"
    assert result.bytes_len == len(result.content)
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/services/aigc/multimodal-generation/generation")


def test_modelstudio_generate_async_submit_and_poll_success(monkeypatch):
    cfg = _make_config(modelstudio_image_mode="async")
    monkeypatch.setattr(modelstudio_module, "get_image_generation_config", lambda: cfg)

    calls = []

    def fake_fetch_json(method, url, headers, timeout, **kwargs):
        calls.append((method, url, kwargs))
        if method == "POST" and url.endswith("/services/aigc/text2image/image-synthesis"):
            return {"output": {"task_id": "task-123", "task_status": "PENDING"}}
        if method == "GET" and url.endswith("/tasks/task-123"):
            return {
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [{"url": "https://dashscope-result-us.oss-us.aliyuncs.com/out.png"}],
                }
            }
        raise AssertionError("unexpected request")

    monkeypatch.setattr(modelstudio_module, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(
        modelstudio_module,
        "fetch_image_bytes",
        lambda *_args, **_kwargs: (b"\x89PNG\r\n\x1a\nabc", "image/png"),
    )
    monkeypatch.setattr(modelstudio_module.time, "sleep", lambda *_args, **_kwargs: None)

    adapter = modelstudio_module.ModelStudioImageAdapter()
    result = adapter.generate(_make_request())
    assert result.content.startswith(b"\x89PNG")
    assert result.content_type == "image/png"
    assert any(url.endswith("/services/aigc/text2image/image-synthesis") for _, url, _ in calls)
    assert any(url.endswith("/tasks/task-123") for _, url, _ in calls)


def test_modelstudio_generate_async_terminal_failure(monkeypatch):
    cfg = _make_config(modelstudio_image_mode="async")
    monkeypatch.setattr(modelstudio_module, "get_image_generation_config", lambda: cfg)

    def fake_fetch_json(method, url, headers, timeout, **kwargs):
        if method == "POST" and url.endswith("/services/aigc/text2image/image-synthesis"):
            return {"output": {"task_id": "task-456", "task_status": "PENDING"}}
        if method == "GET" and url.endswith("/tasks/task-456"):
            return {"output": {"task_status": "FAILED", "message": "safety check failed"}}
        raise AssertionError("unexpected request")

    monkeypatch.setattr(modelstudio_module, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(modelstudio_module.time, "sleep", lambda *_args, **_kwargs: None)

    adapter = modelstudio_module.ModelStudioImageAdapter()
    with pytest.raises(ImageGenerationError, match="Model Studio task failed"):
        adapter.generate(_make_request())


def test_modelstudio_generate_auto_falls_back_to_async(monkeypatch):
    cfg = _make_config(modelstudio_image_mode="auto")
    monkeypatch.setattr(modelstudio_module, "get_image_generation_config", lambda: cfg)

    calls = []

    def fake_fetch_json(method, url, headers, timeout, **kwargs):
        calls.append((method, url))
        if method == "POST" and url.endswith("/services/aigc/multimodal-generation/generation"):
            raise RuntimeError("sync endpoint unavailable")
        if method == "POST" and url.endswith("/services/aigc/text2image/image-synthesis"):
            return {"output": {"task_id": "task-789", "task_status": "PENDING"}}
        if method == "GET" and url.endswith("/tasks/task-789"):
            return {
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [{"url": "https://dashscope-result-us.oss-us.aliyuncs.com/out.png"}],
                }
            }
        raise AssertionError("unexpected request")

    monkeypatch.setattr(modelstudio_module, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(
        modelstudio_module,
        "fetch_image_bytes",
        lambda *_args, **_kwargs: (b"\x89PNG\r\n\x1a\nabc", "image/png"),
    )
    monkeypatch.setattr(modelstudio_module.time, "sleep", lambda *_args, **_kwargs: None)

    adapter = modelstudio_module.ModelStudioImageAdapter()
    result = adapter.generate(_make_request())
    assert result.content.startswith(b"\x89PNG")
    assert any(url.endswith("/services/aigc/multimodal-generation/generation") for _, url in calls)
    assert any(url.endswith("/services/aigc/text2image/image-synthesis") for _, url in calls)


def test_modelstudio_generate_sync_sanitizes_transport_error(monkeypatch):
    cfg = _make_config(modelstudio_image_mode="sync")
    monkeypatch.setattr(modelstudio_module, "get_image_generation_config", lambda: cfg)

    def fake_fetch_json(method, url, headers, json, timeout, **kwargs):
        raise RuntimeError("socket exploded")

    monkeypatch.setattr(modelstudio_module, "fetch_json", fake_fetch_json)

    adapter = modelstudio_module.ModelStudioImageAdapter()
    with pytest.raises(ImageGenerationError) as exc_info:
        adapter.generate(_make_request())
    assert str(exc_info.value) == "Model Studio sync request failed"
    assert "exploded" not in str(exc_info.value)


def test_modelstudio_resolve_base_url_uses_region_preset_when_base_unset(monkeypatch):
    cfg = _make_config(modelstudio_image_base_url=None, modelstudio_image_region="us")
    monkeypatch.setattr(modelstudio_module, "get_image_generation_config", lambda: cfg)
    monkeypatch.delenv("MODELSTUDIO_IMAGE_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)

    adapter = modelstudio_module.ModelStudioImageAdapter()
    assert adapter._resolve_base_url() == "https://dashscope-us.aliyuncs.com/api/v1"


def test_modelstudio_resolve_base_url_env_override_precedes_region(monkeypatch):
    cfg = _make_config(modelstudio_image_base_url=None, modelstudio_image_region="cn")
    monkeypatch.setattr(modelstudio_module, "get_image_generation_config", lambda: cfg)
    monkeypatch.setenv("MODELSTUDIO_IMAGE_BASE_URL", "https://custom.example.com/v1")

    adapter = modelstudio_module.ModelStudioImageAdapter()
    assert adapter._resolve_base_url() == "https://custom.example.com/v1"


def test_modelstudio_rejects_non_dashscope_image_url(monkeypatch):
    cfg = _make_config()
    monkeypatch.setattr(modelstudio_module, "get_image_generation_config", lambda: cfg)
    monkeypatch.setattr(
        modelstudio_module,
        "fetch_image_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fetch should not be called")),
    )

    adapter = modelstudio_module.ModelStudioImageAdapter()
    with pytest.raises(ImageGenerationError, match="unsupported image URL host"):
        adapter._extract_from_link_value("https://example.com/malicious.png")
