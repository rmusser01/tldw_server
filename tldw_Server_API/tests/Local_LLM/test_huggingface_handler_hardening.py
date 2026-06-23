from pathlib import Path

import pytest

from tldw_Server_API.app.core.Local_LLM.Huggingface_Handler import HuggingFaceHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import HuggingFaceConfig


@pytest.mark.asyncio
async def test_hf_is_model_available_rejects_outside_local_path(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    outside_dir = tmp_path / "outside"
    models_dir.mkdir()
    outside_dir.mkdir()
    (outside_dir / "config.json").write_text("{}", encoding="utf-8")

    handler = HuggingFaceHandler(HuggingFaceConfig(models_dir=models_dir), global_app_config={})

    assert await handler.is_model_available(str(outside_dir)) is False


@pytest.mark.asyncio
async def test_hf_loaded_model_cache_respects_configured_limit(monkeypatch, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    for name in ("first", "second"):
        model_dir = models_dir / name
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

    cfg = HuggingFaceConfig(models_dir=models_dir, max_loaded_models=1)
    handler = HuggingFaceHandler(cfg, global_app_config={})

    class FakeTorch:
        bfloat16 = object()
        float16 = object()
        float32 = object()

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(_path: str):
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(_path: str, **_kwargs):
            return object()

    monkeypatch.setattr(
        handler,
        "_ensure_hf_dependencies",
        lambda: (FakeTorch, FakeModel, FakeTokenizer, lambda **kwargs: kwargs, None),
    )

    first_model, _first_tokenizer = await handler._load_model_and_tokenizer("first")
    second_model, _second_tokenizer = await handler._load_model_and_tokenizer("second")

    assert first_model is not second_model
    assert len(handler.loaded_models) == 1
    assert next(iter(handler.loaded_models))[0] == "second"
