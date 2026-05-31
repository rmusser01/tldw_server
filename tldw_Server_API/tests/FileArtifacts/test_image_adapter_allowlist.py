from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.File_Artifacts.adapters import image_adapter as image_adapter_module
from tldw_Server_API.app.core.File_Artifacts.adapters.image_adapter import ImageAdapter
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenResult
from tldw_Server_API.app.core.Image_Generation.capabilities import ResolvedReferenceImage
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageGenerationError


class _StubBackendAdapter:
    def __init__(self, result=None, exc=None) -> None:
        self.result = result
        self.exc = exc
        self.seen_requests = []

    def generate(self, request):
        self.seen_requests.append(request)
        if self.exc is not None:
            raise self.exc
        if self.result is not None:
            return self.result
        return ImageGenResult(content=b"image", content_type="image/png", bytes_len=5)


class _StubRegistry:
    def __init__(self, adapter=None) -> None:
        self.adapter = adapter or _StubBackendAdapter()

    def resolve_backend(self, requested):
        return requested

    def get_adapter(self, name):
        return self.adapter


def test_modelstudio_allowlist_uses_config_field():
    cfg = SimpleNamespace(modelstudio_image_allowed_extra_params=["watermark", "seed_offset"])
    allowlist = ImageAdapter._allowed_extra_params("modelstudio", cfg)
    assert allowlist == {"watermark", "seed_offset"}


def test_modelstudio_mode_control_param_allowed_without_allowlist():
    cfg = SimpleNamespace(modelstudio_image_allowed_extra_params=[])
    adapter = ImageAdapter()
    issues = []
    adapter._validate_extra_params(  # noqa: SLF001 - direct unit test coverage
        {"backend": "modelstudio", "extra_params": {"mode": "async"}},
        cfg,
        issues,
    )
    assert issues == []


def test_modelstudio_mode_control_does_not_bypass_other_keys():
    cfg = SimpleNamespace(modelstudio_image_allowed_extra_params=[])
    adapter = ImageAdapter()
    issues = []
    adapter._validate_extra_params(  # noqa: SLF001 - direct unit test coverage
        {"backend": "modelstudio", "extra_params": {"mode": "async", "foo": "bar"}},
        cfg,
        issues,
    )
    assert len(issues) == 1
    assert issues[0].path == "extra_params.foo"


def test_image_adapter_normalize_preserves_reference_file_provenance(monkeypatch):
    cfg = SimpleNamespace(
        max_prompt_length=1000,
        sd_cpp_allowed_extra_params=[],
        swarmui_allowed_extra_params=[],
        openrouter_image_allowed_extra_params=[],
        novita_image_allowed_extra_params=[],
        together_image_allowed_extra_params=[],
        modelstudio_image_allowed_extra_params=[],
    )
    monkeypatch.setattr(image_adapter_module, "get_registry", lambda: _StubRegistry())
    monkeypatch.setattr(image_adapter_module, "get_image_generation_config", lambda: cfg)

    adapter = ImageAdapter()
    structured = adapter.normalize(
        {
            "backend": "modelstudio",
            "prompt": "draw a fox",
            "reference_file_id": "17",
        }
    )

    assert structured["reference_file_id"] == 17
    assert structured["reference_image_provenance"] == {
        "source": "managed_reference_image",
        "reference_file_id": 17,
    }
    assert "content" not in structured["reference_image_provenance"]
    assert "temp_path" not in structured["reference_image_provenance"]


def test_image_adapter_export_attaches_reference_image_when_supported(monkeypatch):
    cfg = SimpleNamespace(
        reference_image_supported_models={"modelstudio": ["qwen-image-edit"]},
        modelstudio_image_allowed_extra_params=[],
        sd_cpp_allowed_extra_params=[],
        swarmui_allowed_extra_params=[],
        openrouter_image_allowed_extra_params=[],
        novita_image_allowed_extra_params=[],
        together_image_allowed_extra_params=[],
    )
    backend = _StubBackendAdapter()
    monkeypatch.setattr(image_adapter_module, "get_registry", lambda: _StubRegistry(adapter=backend))
    monkeypatch.setattr(image_adapter_module, "get_image_generation_config", lambda: cfg)

    reference = ResolvedReferenceImage(
        file_id=17,
        filename="reference.png",
        mime_type="image/png",
        width=64,
        height=64,
        bytes_len=4,
        content=b"data",
        temp_path=None,
    )
    monkeypatch.setattr(
        image_adapter_module.ImageAdapter,
        "_resolve_reference_image",
        lambda self, structured, backend: reference,
    )

    token = image_adapter_module.set_image_adapter_request_context(collections_db=SimpleNamespace(), user_id=321)
    try:
        adapter = ImageAdapter()
        structured = {
            "backend": "modelstudio",
            "prompt": "draw a fox",
            "model": "qwen-image-edit-v1",
            "reference_file_id": 17,
            "reference_image_provenance": {"source": "managed_reference_image", "reference_file_id": 17},
            "extra_params": {},
        }
        result = adapter.export(
            structured,
            format="png",
        )
    finally:
        image_adapter_module.reset_image_adapter_request_context(token)

    assert result.content == b"image"
    assert backend.seen_requests
    assert backend.seen_requests[0].reference_image is reference
    assert structured["reference_image_provenance"]["snapshot"] == {
        "filename": "reference.png",
        "mime_type": "image/png",
        "width": 64,
        "height": 64,
    }


def test_image_adapter_export_rejects_unsupported_reference_image_backend(monkeypatch):
    cfg = SimpleNamespace(
        reference_image_supported_models={"modelstudio": ["qwen-image-edit"]},
        modelstudio_image_allowed_extra_params=[],
        sd_cpp_allowed_extra_params=[],
        swarmui_allowed_extra_params=[],
        openrouter_image_allowed_extra_params=[],
        novita_image_allowed_extra_params=[],
        together_image_allowed_extra_params=[],
    )
    monkeypatch.setattr(image_adapter_module, "get_registry", lambda: _StubRegistry())
    monkeypatch.setattr(image_adapter_module, "get_image_generation_config", lambda: cfg)

    adapter = ImageAdapter()

    with pytest.raises(
        image_adapter_module.FileArtifactsValidationError,
        match="reference_image_unsupported_by_backend",
    ):
        adapter.export(
            {
                "backend": "swarmui",
                "prompt": "draw a fox",
                "model": "any-model",
                "reference_file_id": 17,
                "reference_image_provenance": {"source": "managed_reference_image", "reference_file_id": 17},
                "extra_params": {},
            },
            format="png",
        )


def test_image_adapter_export_rejects_unsupported_reference_image_model(monkeypatch):
    cfg = SimpleNamespace(
        reference_image_supported_models={"modelstudio": ["qwen-image-edit"]},
        modelstudio_image_allowed_extra_params=[],
        sd_cpp_allowed_extra_params=[],
        swarmui_allowed_extra_params=[],
        openrouter_image_allowed_extra_params=[],
        novita_image_allowed_extra_params=[],
        together_image_allowed_extra_params=[],
    )
    monkeypatch.setattr(image_adapter_module, "get_registry", lambda: _StubRegistry())
    monkeypatch.setattr(image_adapter_module, "get_image_generation_config", lambda: cfg)

    adapter = ImageAdapter()

    with pytest.raises(
        image_adapter_module.FileArtifactsValidationError,
        match="reference_image_unsupported_by_model",
    ):
        adapter.export(
            {
                "backend": "modelstudio",
                "prompt": "draw a fox",
                "model": "other-model",
                "reference_file_id": 17,
                "reference_image_provenance": {"source": "managed_reference_image", "reference_file_id": 17},
                "extra_params": {},
            },
            format="png",
        )


@pytest.mark.parametrize(
    "backend_exc",
    [
        ImageGenerationError("backend failed at /tmp/private/image.png using sk-secret"),
        RuntimeError("unexpected backend traceback /var/private/token=abc123"),
    ],
)
def test_image_adapter_export_sanitizes_backend_generation_failures(monkeypatch, backend_exc):
    backend = _StubBackendAdapter(exc=backend_exc)
    monkeypatch.setattr(image_adapter_module, "get_registry", lambda: _StubRegistry(adapter=backend))

    adapter = ImageAdapter()

    with pytest.raises(image_adapter_module.FileArtifactsError) as exc_info:
        adapter.export(
            {
                "backend": "modelstudio",
                "prompt": "draw a fox",
                "extra_params": {},
            },
            format="png",
        )

    assert exc_info.value.code == "image_generation_failed"
    assert exc_info.value.detail == "image_generation_failed"
    assert "/tmp/private" not in str(exc_info.value.detail)
    assert "/var/private" not in str(exc_info.value.detail)
    assert "sk-secret" not in str(exc_info.value.detail)
    assert "token=abc123" not in str(exc_info.value.detail)


def test_image_adapter_export_sanitizes_unexpected_backend_failure_log(monkeypatch):
    leaked_path = "/var/private/image-generation.db"
    leaked_token = "token=abc123"
    backend = _StubBackendAdapter(
        exc=RuntimeError(f"unexpected backend traceback {leaked_path} {leaked_token}")
    )
    messages: list[str] = []
    monkeypatch.setattr(image_adapter_module, "get_registry", lambda: _StubRegistry(adapter=backend))
    sink_id = image_adapter_module.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level="WARNING",
        format="{message}",
    )

    try:
        with pytest.raises(image_adapter_module.FileArtifactsError):
            ImageAdapter().export(
                {
                    "backend": "modelstudio",
                    "prompt": "draw a fox",
                    "extra_params": {},
                },
                format="png",
            )
    finally:
        image_adapter_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "image adapter: backend generate failed" in joined
    assert leaked_path not in joined
    assert leaked_token not in joined
    assert "unexpected backend traceback" not in joined


def test_image_adapter_reference_resolution_fallback_log_omits_raw_exception_details(monkeypatch):
    leaked_path = "/Users/private/reference-images/source.png"
    leaked_token = "token=ref-secret-123"
    cfg = SimpleNamespace(reference_image_supported_models={"modelstudio": ["qwen-image-edit"]})
    messages: list[str] = []

    async def fail_resolution(*args, **kwargs):
        raise RuntimeError(f"reference image failed from {leaked_path} with {leaked_token}")

    monkeypatch.setattr(image_adapter_module, "get_image_generation_config", lambda: cfg)
    monkeypatch.setattr(image_adapter_module, "resolve_reference_image", fail_resolution)
    sink_id = image_adapter_module.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level="WARNING",
        format="{message}",
    )
    token = image_adapter_module.set_image_adapter_request_context(collections_db=SimpleNamespace(), user_id=321)

    try:
        with pytest.raises(
            image_adapter_module.FileArtifactsValidationError,
            match="reference_image_invalid",
        ):
            ImageAdapter()._resolve_reference_image(  # noqa: SLF001 - direct fallback branch coverage
                {
                    "backend": "modelstudio",
                    "model": "qwen-image-edit-v1",
                    "reference_file_id": 17,
                },
                backend="modelstudio",
            )
    finally:
        image_adapter_module.reset_image_adapter_request_context(token)
        image_adapter_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "image adapter: reference image resolution failed" in joined
    assert leaked_path not in joined
    assert leaked_token not in joined
    assert "reference image failed from" not in joined
