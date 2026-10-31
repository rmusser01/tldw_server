"""Compatibility runtime for KittenTTS model assets.

This module intentionally follows the upstream asset contract (`config.json`,
ONNX model file, voices archive) while keeping the runtime behavior under this
repo's control. The main reason is upstream PR #25: we need the explicit
`espeakng_loader` + `EspeakWrapper` initialization path even before that change
is reliably available in packaged releases.
"""

from __future__ import annotations

import json
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


def _log_optional_dependency_fallback(module_name: str, exc: BaseException) -> None:
    logger.warning(
        "KittenTTS optional dependency '{}' failed to import; falling back to lazy error stubs: {}",
        module_name,
        exc,
    )


def _missing_dependency_callable(module_name: str, feature: str, exc: BaseException):
    def _missing(*_args: Any, **_kwargs: Any):
        raise ImportError(
            f"{module_name} is required for KittenTTS {feature}: {exc}"
        ) from exc

    return _missing


def _missing_dependency_type(module_name: str, feature: str, exc: BaseException):
    class _MissingDependency:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError(
                f"{module_name} is required for KittenTTS {feature}: {exc}"
            ) from exc

    return _MissingDependency

try:
    import espeakng_loader  # type: ignore
except (ImportError, OSError) as exc:  # pragma: no cover - exercised through missing-dependency paths
    _log_optional_dependency_fallback("espeakng_loader", exc)
    espeakng_loader = types.SimpleNamespace(
        get_library_path=_missing_dependency_callable("espeakng_loader", "eSpeak library discovery", exc),
        get_data_path=_missing_dependency_callable("espeakng_loader", "eSpeak data discovery", exc),
    )

try:
    import onnxruntime as ort  # type: ignore
except (ImportError, OSError) as exc:  # pragma: no cover
    _log_optional_dependency_fallback("onnxruntime", exc)
    ort = types.SimpleNamespace(
        InferenceSession=_missing_dependency_callable("onnxruntime", "runtime sessions", exc)
    )

try:
    import phonemizer  # type: ignore
    from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore
except (ImportError, OSError) as exc:  # pragma: no cover
    _log_optional_dependency_fallback("phonemizer", exc)
    # `exc` is unbound once this block exits; the wrapper methods run later.
    _phonemizer_error = exc
    phonemizer = types.SimpleNamespace(
        backend=types.SimpleNamespace(
            EspeakBackend=_missing_dependency_type("phonemizer", "phonemizer backend", exc)
        )
    )

    class EspeakWrapper:  # type: ignore[override]
        @staticmethod
        def set_library(_path: str) -> None:
            raise ImportError(
                f"phonemizer is required for KittenTTS eSpeak wrapper setup: {_phonemizer_error}"
            ) from _phonemizer_error

        @staticmethod
        def set_data_path(_path: str) -> None:
            raise ImportError(
                f"phonemizer is required for KittenTTS eSpeak wrapper setup: {_phonemizer_error}"
            ) from _phonemizer_error


try:
    from huggingface_hub import hf_hub_download
except (ImportError, OSError) as exc:  # pragma: no cover
    _log_optional_dependency_fallback("huggingface_hub", exc)
    hf_hub_download = None


DEFAULT_REPO_ID = "KittenML/kitten-tts-nano-0.8"
DEFAULT_CANONICAL_REPO_ID = "KittenML/kitten-tts-nano-0.8-fp32"
DEFAULT_INTERNAL_VOICE = "expr-voice-5-m"
TAIL_TRIM_SAMPLES = 5000
REVISION_RE = re.compile(r"^[0-9a-fA-F]{7,}$")
MODEL_REPO_ALIASES: dict[str, str] = {
    "KittenML/kitten-tts-nano-0.8": DEFAULT_CANONICAL_REPO_ID,
}
PINNED_MODEL_REVISIONS: dict[str, str] = {
    DEFAULT_CANONICAL_REPO_ID: "8d6d5a1851ffd13c894c40227c888302c2a86ef7",
    "KittenML/kitten-tts-nano-0.8-int8": "b5c9e5ce0faf0b025c0aa5afa02637c087f813ee",
    "KittenML/kitten-tts-micro-0.8": "ff7d47695d85548bcb0ac9378e063682e1cf0548",
    "KittenML/kitten-tts-mini-0.8": "5ae6da49a9401ca30334aa303a9a93246cf6ebb6",
}
DEFAULT_VOICE_ALIASES: dict[str, str] = {
    "Bella": "expr-voice-2-f",
    "Jasper": "expr-voice-2-m",
    "Luna": "expr-voice-3-f",
    "Bruno": "expr-voice-3-m",
    "Rosie": "expr-voice-4-f",
    "Hugo": "expr-voice-4-m",
    "Kiki": "expr-voice-5-f",
    "Leo": "expr-voice-5-m",
}


class TextPreprocessor:
    """Small placeholder preprocessor used when clean_text is enabled.

    The adapter defaults `clean_text` to false, so a conservative normalizer is
    enough here. This keeps the runtime lightweight and avoids coupling the
    integration to the upstream package import path.
    """

    def __init__(self, **_kwargs: Any) -> None:
        return

    def __call__(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()


@dataclass(frozen=True)
class KittenModelAssets:
    repo_id: str
    revision: str
    config_path: Path
    model_path: Path
    voices_path: Path
    speed_priors: dict[str, float]
    voice_aliases: dict[str, str]


def normalize_repo_id(model_name: str | None) -> str:
    raw = str(model_name or "").strip()
    if not raw:
        return DEFAULT_CANONICAL_REPO_ID
    if raw.lower() in {"kitten_tts", "kitten-tts", "kittentts"}:
        return DEFAULT_CANONICAL_REPO_ID
    if "/" not in raw:
        raw = f"KittenML/{raw}"
    else:
        namespace, repo_name = raw.split("/", 1)
        if namespace.lower() == "kittenml":
            raw = f"KittenML/{repo_name}"
    return MODEL_REPO_ALIASES.get(raw, raw)


def resolve_model_revision(model_name: str | None, revision: str | None = None) -> str:
    repo_id = normalize_repo_id(model_name)
    requested = str(revision or "").strip()
    if requested:
        if not REVISION_RE.fullmatch(requested):
            raise ValueError(
                "KittenTTS model_revision must be an immutable Hugging Face commit hash"
            )
        return requested

    pinned_revision = PINNED_MODEL_REVISIONS.get(repo_id)
    if pinned_revision:
        return pinned_revision

    raise ValueError(
        f"No pinned KittenTTS revision configured for model '{repo_id}'. "
        "Provide model_revision with an immutable commit hash."
    )


def initialize_espeak_paths() -> None:
    library_path = espeakng_loader.get_library_path()
    data_path = espeakng_loader.get_data_path()
    EspeakWrapper.set_library(library_path)
    EspeakWrapper.set_data_path(data_path)


def download_model_assets(
    model_name: str | None = None,
    *,
    cache_dir: str | None = None,
    auto_download: bool = True,
    revision: str | None = None,
) -> KittenModelAssets:
    if hf_hub_download is None:
        raise ImportError("huggingface_hub is required for KittenTTS")

    repo_id = normalize_repo_id(model_name)
    resolved_revision = resolve_model_revision(repo_id, revision)
    local_only = not auto_download
    config_path = Path(
        hf_hub_download(  # nosec B615
            repo_id=repo_id,
            filename="config.json",
            cache_dir=cache_dir,
            revision=resolved_revision,
            local_files_only=local_only,
        )
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("type") not in {"ONNX1", "ONNX2"}:
        raise ValueError("Unsupported KittenTTS model type")

    model_path = Path(
        hf_hub_download(  # nosec B615
            repo_id=repo_id,
            filename=config["model_file"],
            cache_dir=cache_dir,
            revision=resolved_revision,
            local_files_only=local_only,
        )
    )
    voices_path = Path(
        hf_hub_download(  # nosec B615
            repo_id=repo_id,
            filename=config["voices"],
            cache_dir=cache_dir,
            revision=resolved_revision,
            local_files_only=local_only,
        )
    )
    voice_aliases = dict(config.get("voice_aliases") or DEFAULT_VOICE_ALIASES)
    if not voice_aliases:
        voice_aliases = DEFAULT_VOICE_ALIASES.copy()

    return KittenModelAssets(
        repo_id=repo_id,
        revision=resolved_revision,
        config_path=config_path,
        model_path=model_path,
        voices_path=voices_path,
        speed_priors=dict(config.get("speed_priors") or {}),
        voice_aliases=voice_aliases,
    )


def basic_english_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text)


def ensure_punctuation(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    if value[-1] not in ".!?,;:":
        return value + ","
    return value


def chunk_text(text: str, max_len: int = 400) -> list[str]:
    sentences = re.split(r"[.!?]+", str(text or ""))
    chunks: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_len:
            chunks.append(ensure_punctuation(sentence))
            continue
        words = sentence.split()
        current = ""
        for word in words:
            addition = word if not current else f"{current} {word}"
            if len(addition) <= max_len:
                current = addition
                continue
            if current:
                chunks.append(ensure_punctuation(current))
            current = word
        if current:
            chunks.append(ensure_punctuation(current))
    return chunks


class _TextCleaner:
    def __init__(self) -> None:
        symbols = (
            ["$"]
            + list(';:,.!?¡¿—…"«»"" ')
            + list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
            + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
        )
        self._indices = {symbol: idx for idx, symbol in enumerate(symbols)}

    def __call__(self, text: str) -> list[int]:
        return [self._indices[ch] for ch in text if ch in self._indices]


class KittenRuntime:
    """Small ONNX runtime wrapper for KittenTTS assets."""

    def __init__(self, assets: KittenModelAssets):
        if not hasattr(ort, "InferenceSession"):
            raise ImportError("onnxruntime is required for KittenTTS")
        if not hasattr(phonemizer, "backend") or not hasattr(phonemizer.backend, "EspeakBackend"):
            raise ImportError("phonemizer is required for KittenTTS")

        initialize_espeak_paths()
        self.assets = assets
        self.speed_priors = dict(assets.speed_priors)
        self.voice_aliases = dict(assets.voice_aliases or DEFAULT_VOICE_ALIASES)
        self.available_voices = list(self.voice_aliases.keys())
        self._lower_aliases = {name.lower(): value for name, value in self.voice_aliases.items()}
        self.voices = np.load(str(assets.voices_path))
        self.session = ort.InferenceSession(str(assets.model_path))
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us",
            preserve_punctuation=True,
            with_stress=True,
        )
        self.text_cleaner = _TextCleaner()
        self.preprocessor = TextPreprocessor(remove_punctuation=False)

    def resolve_voice(self, voice: str | None) -> str:
        requested = str(voice or "").strip()
        if not requested:
            return self.voice_aliases.get("Leo", DEFAULT_INTERNAL_VOICE)
        if requested in self.voice_aliases:
            return self.voice_aliases[requested]
        lowered = requested.lower()
        if lowered in self._lower_aliases:
            return self._lower_aliases[lowered]
        if requested in self.voices:
            return requested
        raise ValueError(f"Voice '{voice}' not available")

    def _prepare_inputs(self, text: str, voice: str, speed: float = 1.0) -> dict[str, np.ndarray]:
        resolved_voice = self.resolve_voice(voice)
        if resolved_voice in self.speed_priors:
            speed = speed * float(self.speed_priors[resolved_voice])

        phonemes = self.phonemizer.phonemize([text])[0]
        tokenized = " ".join(basic_english_tokenize(phonemes))
        tokens = self.text_cleaner(tokenized)
        tokens.insert(0, 0)
        tokens.append(10)
        tokens.append(0)

        ref_id = min(len(text), self.voices[resolved_voice].shape[0] - 1)
        ref_style = self.voices[resolved_voice][ref_id : ref_id + 1]

        return {
            "input_ids": np.asarray([tokens], dtype=np.int64),
            "style": ref_style,
            "speed": np.asarray([speed], dtype=np.float32),
        }

    def generate_single_chunk(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> np.ndarray:
        outputs = self.session.run(None, self._prepare_inputs(text, voice or "", speed))
        audio = np.asarray(outputs[0], dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        if audio.shape[-1] > TAIL_TRIM_SAMPLES:
            # Upstream ONNX outputs carry a short padded tail we do not want in the rendered clip.
            audio = audio[:-TAIL_TRIM_SAMPLES]
        return audio

    def generate(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        clean_text: bool = False,
    ) -> np.ndarray:
        value = self.preprocessor(text) if clean_text else str(text or "")
        chunks = chunk_text(value)
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        rendered = [
            self.generate_single_chunk(chunk, voice=voice, speed=speed)
            for chunk in chunks
        ]
        return np.concatenate(rendered, axis=-1)
