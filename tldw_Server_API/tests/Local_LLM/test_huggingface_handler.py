from pathlib import Path

import pytest

from tldw_Server_API.app.core.Local_LLM.Huggingface_Handler import HuggingFaceHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import HuggingFaceConfig
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelDownloadError


class _FakeTorch:
    bfloat16 = object()
    float16 = object()
    float32 = object()


def _stub_hf_dependencies(monkeypatch, *, tokenizer_factory, model_factory):
    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return model_factory(*args, **kwargs)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return tokenizer_factory(*args, **kwargs)

    def fake_bits_and_bytes_config(**kwargs):
        return {"bnb": kwargs}

    monkeypatch.setattr(
        HuggingFaceHandler,
        "_ensure_hf_dependencies",
        staticmethod(
            lambda: (
                _FakeTorch,
                FakeAutoModelForCausalLM,
                FakeAutoTokenizer,
                fake_bits_and_bytes_config,
                object(),
            )
        ),
    )


@pytest.mark.asyncio
async def test_hf_download_rejects_traversal(monkeypatch, tmp_path: Path):
    models_dir = tmp_path / "models"
    cfg = HuggingFaceConfig(models_dir=models_dir)
    handler = HuggingFaceHandler(cfg, global_app_config={})

    def _should_not_load_dependencies():
        raise AssertionError("HF dependencies should not be loaded for unsafe paths")

    monkeypatch.setattr(
        HuggingFaceHandler,
        "_ensure_hf_dependencies",
        staticmethod(_should_not_load_dependencies),
    )

    with pytest.raises(ModelDownloadError, match="allowed directories"):
        await handler.download_model("gpt2", save_directory="../evil")


@pytest.mark.asyncio
async def test_hf_cache_key_includes_quantization(monkeypatch, tmp_path: Path):
    cfg = HuggingFaceConfig(models_dir=tmp_path)
    handler = HuggingFaceHandler(cfg, global_app_config={})
    model_dir = tmp_path / "toy"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}")

    tok_calls = []
    model_calls = []

    def fake_tokenizer(*args, **kwargs):
        obj = object()
        tok_calls.append(obj)
        return obj

    def fake_model(*args, **kwargs):
        obj = object()
        model_calls.append(obj)
        return obj

    _stub_hf_dependencies(
        monkeypatch,
        tokenizer_factory=fake_tokenizer,
        model_factory=fake_model,
    )

    model_a, tok_a = await handler._load_model_and_tokenizer("toy", {"load_in_8bit": True})
    model_b, tok_b = await handler._load_model_and_tokenizer("toy", {"load_in_8bit": True})
    model_c, tok_c = await handler._load_model_and_tokenizer("toy", {"load_in_4bit": True})

    assert model_a is model_b
    assert tok_a is tok_b
    assert model_a is not model_c
    assert tok_a is not tok_c
    assert len(model_calls) == 2
    assert len(tok_calls) == 2
