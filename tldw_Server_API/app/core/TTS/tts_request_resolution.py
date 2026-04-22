from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.core.TTS.tts_config import get_tts_config

DEFAULT_KITTEN_TTS_PROVIDER = "kitten_tts"
DEFAULT_KITTEN_TTS_MODEL = "KittenML/kitten-tts-nano-0.8"
DEFAULT_KITTEN_TTS_VOICE = "Bella"
DEFAULT_OPENAI_TTS_MODEL = "tts-1"
DEFAULT_OPENAI_TTS_VOICE = "alloy"
DEFAULT_KOKORO_TTS_MODEL = "kokoro"
DEFAULT_KOKORO_TTS_VOICE = "af_heart"
DEFAULT_ELEVENLABS_TTS_MODEL = "eleven_monolingual_v1"
DEFAULT_ELEVENLABS_TTS_VOICE = "Rachel"
DEFAULT_OMNIVOICE_TTS_MODEL = "omnivoice"
DEFAULT_OMNIVOICE_TTS_VOICE = "auto"

_PROVIDER_ALIASES = {
    "kitten": DEFAULT_KITTEN_TTS_PROVIDER,
    "kittentts": DEFAULT_KITTEN_TTS_PROVIDER,
    "tldw": DEFAULT_KITTEN_TTS_PROVIDER,
    "omnivoice": "omnivoice",
    "omni-voice": "omnivoice",
    "omni_voice": "omnivoice",
}

_DEFAULT_MODELS_BY_PROVIDER = {
    DEFAULT_KITTEN_TTS_PROVIDER: DEFAULT_KITTEN_TTS_MODEL,
    "openai": DEFAULT_OPENAI_TTS_MODEL,
    "kokoro": DEFAULT_KOKORO_TTS_MODEL,
    "pocket_tts_cpp": "pocket_tts_cpp",
    "pocket_tts": "pocket_tts",
    "elevenlabs": DEFAULT_ELEVENLABS_TTS_MODEL,
    "omnivoice": DEFAULT_OMNIVOICE_TTS_MODEL,
}

_DEFAULT_VOICES_BY_PROVIDER = {
    DEFAULT_KITTEN_TTS_PROVIDER: DEFAULT_KITTEN_TTS_VOICE,
    "openai": DEFAULT_OPENAI_TTS_VOICE,
    "kokoro": DEFAULT_KOKORO_TTS_VOICE,
    "elevenlabs": DEFAULT_ELEVENLABS_TTS_VOICE,
    "omnivoice": DEFAULT_OMNIVOICE_TTS_VOICE,
}

_MODEL_ALIASES = {
    "omnivoice": DEFAULT_OMNIVOICE_TTS_MODEL,
    "omni-voice": DEFAULT_OMNIVOICE_TTS_MODEL,
    "omni_voice": DEFAULT_OMNIVOICE_TTS_MODEL,
}


@dataclass(frozen=True)
class ResolvedTTSRequestDefaults:
    provider: str
    model: str
    voice: str


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_tts_provider(provider: str | None) -> str | None:
    cleaned = _clean_optional_text(provider)
    if cleaned is None:
        return None
    lowered = cleaned.lower()
    return _PROVIDER_ALIASES.get(lowered, lowered)


def infer_tts_provider_from_model(model: str | None) -> str | None:
    cleaned = _clean_optional_text(model)
    if cleaned is None:
        return None
    lowered = cleaned.lower()
    if lowered in {"tts-1", "tts-1-hd"}:
        return "openai"
    if lowered.startswith("kokoro"):
        return "kokoro"
    if lowered.startswith("pocket_tts_cpp") or lowered.startswith("pocket-tts-cpp"):
        return "pocket_tts_cpp"
    if lowered.startswith("pocket_tts") or lowered.startswith("pocket-tts"):
        return "pocket_tts"
    if (
        lowered.startswith("kitten_tts")
        or lowered.startswith("kitten-tts")
        or lowered.startswith("kittentts")
        or lowered.startswith("kittenml/kitten-tts")
    ):
        return DEFAULT_KITTEN_TTS_PROVIDER
    if lowered.startswith("eleven"):
        return "elevenlabs"
    if lowered.startswith("omnivoice") or lowered.startswith("omni-voice") or lowered.startswith("omni_voice"):
        return "omnivoice"
    return None


def _default_model_for_provider(provider: str) -> str:
    return _DEFAULT_MODELS_BY_PROVIDER.get(provider, DEFAULT_KITTEN_TTS_MODEL)


def _default_voice_for_provider(provider: str) -> str:
    return _DEFAULT_VOICES_BY_PROVIDER.get(provider, DEFAULT_KITTEN_TTS_VOICE)


def _normalize_model_for_provider(provider: str, model: str | None) -> str | None:
    if model is None:
        return None
    lowered = model.lower()
    if provider == "omnivoice":
        return _MODEL_ALIASES.get(lowered, model)
    return model


def resolve_tts_request_defaults(
    *,
    provider: str | None,
    model: str | None,
    voice: str | None,
) -> ResolvedTTSRequestDefaults:
    cleaned_model = _clean_optional_text(model)
    cleaned_voice = _clean_optional_text(voice)
    explicit_provider = normalize_tts_provider(provider)
    inferred_provider = infer_tts_provider_from_model(cleaned_model)

    configured_provider: str | None = None
    configured_voice: str | None = None
    try:
        cfg = get_tts_config()
        configured_provider = normalize_tts_provider(getattr(cfg, "default_provider", None))
        configured_voice = _clean_optional_text(getattr(cfg, "default_voice", None))
    except Exception:
        configured_provider = None
        configured_voice = None

    resolved_provider = (
        explicit_provider
        or inferred_provider
        or configured_provider
        or DEFAULT_KITTEN_TTS_PROVIDER
    )
    normalized_model = _normalize_model_for_provider(resolved_provider, cleaned_model)
    resolved_model = normalized_model or _default_model_for_provider(resolved_provider)
    resolved_voice = (
        cleaned_voice
        or (configured_voice if configured_voice and resolved_provider == configured_provider else None)
        or _default_voice_for_provider(resolved_provider)
    )

    return ResolvedTTSRequestDefaults(
        provider=resolved_provider,
        model=resolved_model,
        voice=resolved_voice,
    )
