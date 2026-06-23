import pytest

import tldw_Server_API.app.core.Image_Generation.adapters.swarmui_adapter as swarmui_module
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest
from tldw_Server_API.app.core.Image_Generation.config import ImageGenerationConfig
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageGenerationError

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="


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
        swarmui_base_url="http://localhost:7801",
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


def _make_request(**overrides) -> ImageGenRequest:
    base = dict(
        backend="swarmui",
        prompt="hello",
        negative_prompt="nope",
        width=512,
        height=512,
        steps=20,
        cfg_scale=7.5,
        seed=123,
        sampler="euler",
        model=None,
        format="png",
        extra_params={},
        request_id=None,
    )
    base.update(overrides)
    return ImageGenRequest(**base)


def test_swarmui_generate_data_url(monkeypatch):
    cfg = _make_config()
    monkeypatch.setattr(swarmui_module, "get_image_generation_config", lambda: cfg)

    calls = []

    def fake_fetch_json(method, url, json, **kwargs):
        calls.append((url, json))
        if url.endswith("/API/GetNewSession"):
            return {"session_id": "sess"}
        if url.endswith("/API/GenerateText2Image"):
            assert json["prompt"] == "hello"
            assert json["negativeprompt"] == "nope"
            return {"images": [f"data:image/png;base64,{PNG_B64}"]}
        raise AssertionError("unexpected URL")

    monkeypatch.setattr(swarmui_module, "fetch_json", fake_fetch_json)

    adapter = swarmui_module.SwarmUIAdapter()
    result = adapter.generate(_make_request())
    assert result.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert result.content_type == "image/png"
    assert result.bytes_len == len(result.content)
    assert len(calls) == 2


def test_swarmui_rejects_off_origin_image_url_with_token(monkeypatch):
    cfg = _make_config(swarmui_swarm_token="secret-token")
    monkeypatch.setattr(swarmui_module, "get_image_generation_config", lambda: cfg)

    def fake_fetch_json(method, url, json, **kwargs):
        if url.endswith("/API/GetNewSession"):
            return {"session_id": "sess"}
        if url.endswith("/API/GenerateText2Image"):
            return {"images": ["https://attacker.example/out.png"]}
        raise AssertionError("unexpected URL")

    monkeypatch.setattr(swarmui_module, "fetch_json", fake_fetch_json)

    adapter = swarmui_module.SwarmUIAdapter()
    with pytest.raises(ImageGenerationError, match="off-origin"):
        adapter.generate(_make_request())


def test_swarmui_invalid_session_refresh(monkeypatch):
    cfg = _make_config()
    monkeypatch.setattr(swarmui_module, "get_image_generation_config", lambda: cfg)

    session_calls = []
    generate_calls = []

    def fake_fetch_json(method, url, json, **kwargs):
        if url.endswith("/API/GetNewSession"):
            session_calls.append(url)
            return {"session_id": f"sess-{len(session_calls)}"}
        if url.endswith("/API/GenerateText2Image"):
            generate_calls.append(json)
            if len(generate_calls) == 1:
                return {"error_id": "invalid_session_id"}
            return {"images": ["View/local/raw/test.png"]}
        raise AssertionError("unexpected URL")

    def fake_fetch_image_bytes(url, **kwargs):
        assert url.endswith("/View/local/raw/test.png")
        return b"\x89PNG\r\n\x1a\nabc", "image/png"

    monkeypatch.setattr(swarmui_module, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(swarmui_module, "fetch_shared_image_bytes", fake_fetch_image_bytes)

    adapter = swarmui_module.SwarmUIAdapter()
    result = adapter.generate(_make_request())
    assert result.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert result.content_type == "image/png"
    assert len(session_calls) == 2
    assert len(generate_calls) == 2


def test_swarmui_converts_to_jpg(monkeypatch):
    cfg = _make_config()
    monkeypatch.setattr(swarmui_module, "get_image_generation_config", lambda: cfg)

    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    )
    data_url = f"data:image/png;base64,{png_b64}"

    def fake_fetch_json(method, url, json, **kwargs):
        if url.endswith("/API/GetNewSession"):
            return {"session_id": "sess"}
        if url.endswith("/API/GenerateText2Image"):
            return {"images": [data_url]}
        raise AssertionError("unexpected URL")

    monkeypatch.setattr(swarmui_module, "fetch_json", fake_fetch_json)

    adapter = swarmui_module.SwarmUIAdapter()
    result = adapter.generate(_make_request(format="jpg"))
    assert result.content_type == "image/jpeg"
    assert result.content[:2] == b"\xff\xd8"
