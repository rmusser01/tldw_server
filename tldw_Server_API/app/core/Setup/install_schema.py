"""Shared Pydantic models for installer plans.

The API layer and background installer both depend on these models, so they live in a
dedicated module to avoid import-time side effects or heavyweight dependencies when
only the schema is required.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

DEFAULT_WHISPER_MODELS = ['small']

STT_ENGINES = {
    'faster_whisper',
    'qwen2_audio',
    'nemo_parakeet_standard',
    'nemo_parakeet_onnx',
    'nemo_parakeet_mlx',
    'nemo_canary',
}

TTS_ENGINES = {'kokoro', 'kitten_tts', 'dia', 'higgs', 'vibevoice', 'echo_tts', 'pocket_tts', 'neutts', 'omnivoice'}


class STTInstall(BaseModel):
    engine: str
    models: list[str] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def _normalise_models(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(values, dict):
            return values
        engine = values.get('engine')
        models = values.get('models') or []
        if engine == 'faster_whisper' and not models:
            values['models'] = DEFAULT_WHISPER_MODELS.copy()
        return values

    @model_validator(mode='after')
    def _validate(self) -> STTInstall:
        if self.engine not in STT_ENGINES:
            raise ValueError(f"Unsupported STT engine '{self.engine}'")
        self.models = list(dict.fromkeys(self.models))
        return self


class TTSInstall(BaseModel):
    engine: str
    variants: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate(self) -> TTSInstall:
        if self.engine not in TTS_ENGINES:
            raise ValueError(f"Unsupported TTS engine '{self.engine}'")
        self.variants = list(dict.fromkeys(self.variants))
        return self


class EmbeddingsInstall(BaseModel):
    huggingface: list[str] = Field(default_factory=list)
    custom: list[str] = Field(default_factory=list)
    onnx: list[str] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def _normalise(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(values, dict):
            return values
        normalised = dict(values)
        for key in ('huggingface', 'custom', 'onnx'):
            entries = normalised.get(key) or []
            normalised[key] = [item.strip() for item in entries if item and item.strip()]
        return normalised


class InstallPlan(BaseModel):
    stt: list[STTInstall] = Field(default_factory=list)
    tts: list[TTSInstall] = Field(default_factory=list)
    embeddings: EmbeddingsInstall = Field(default_factory=EmbeddingsInstall)

    def is_empty(self) -> bool:
        if self.stt or self.tts:
            return False
        embeddings = self.embeddings
        return not (embeddings.huggingface or embeddings.custom or embeddings.onnx)


__all__ = [
    'DEFAULT_WHISPER_MODELS',
    'EmbeddingsInstall',
    'InstallPlan',
    'STTInstall',
    'STT_ENGINES',
    'TTSInstall',
    'TTS_ENGINES',
]
