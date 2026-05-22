from __future__ import annotations

import asyncio
import importlib
import io
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .omnivoice_sidecar_protocol import OmniVoiceSynthesizeRequest


class OmniVoiceRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class OmniVoiceSynthesizeResult:
    audio_bytes: bytes
    audio_format: str
    sample_rate: int
    channels: int
    cold_start: bool
    model: str


class OmniVoiceRuntime:
    NATIVE_SAMPLE_RATE = 24000
    _SAFE_LOAD_KWARGS = ("device_map", "dtype", "load_asr", "asr_model_name")
    _REFERENCE_DIR_KEYS = (
        "scratch_dir",
        "managed_reference_dir",
        "reference_dir",
    )
    _REFERENCE_DIR_LIST_KEYS = (
        "managed_reference_dirs",
        "reference_dirs",
    )

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config or {})
        self.model = None
        self.status = "idle_stopped"
        self._load_lock = asyncio.Lock()
        self._generate_lock = asyncio.Lock()
        self._model_path: Path | None = None
        self._model_id: str | None = None
        self.last_error_code: str | None = None

    async def load(self) -> object:
        if self.model is not None:
            return self.model

        async with self._load_lock:
            if self.model is not None:
                return self.model

            model_path = self._resolve_model_path()
            self.status = "loading"
            try:
                omnivoice_module = importlib.import_module("omnivoice")
                omnivoice_cls = getattr(omnivoice_module, "OmniVoice")
            except Exception as exc:
                self._record_error("RUNTIME_IMPORT_FAILED")
                raise OmniVoiceRuntimeError(
                    "RUNTIME_IMPORT_FAILED",
                    "OmniVoice runtime package could not be imported",
                    retryable=False,
                ) from exc

            load_kwargs = {
                key: self.config[key]
                for key in self._SAFE_LOAD_KWARGS
                if key in self.config and self.config[key] is not None
            }
            try:
                self.model = await asyncio.to_thread(
                    omnivoice_cls.from_pretrained,
                    str(model_path),
                    **load_kwargs,
                )
            except Exception as exc:
                self._record_error("MODEL_LOAD_FAILED")
                raise OmniVoiceRuntimeError(
                    "MODEL_LOAD_FAILED",
                    "OmniVoice model failed to load from the configured local model directory",
                    retryable=False,
                ) from exc

            self._model_path = model_path
            self._model_id = self._resolve_model_id(model_path)
            self.status = "ready"
            self.last_error_code = None
            return self.model

    async def synthesize(self, request: OmniVoiceSynthesizeRequest) -> OmniVoiceSynthesizeResult:
        generate_kwargs = self._build_generation_kwargs(request)
        cold_start = self.model is None
        model = await self.load()
        if not hasattr(model, "generate"):
            self._record_error("MODEL_GENERATE_FAILED")
            raise OmniVoiceRuntimeError(
                "MODEL_GENERATE_FAILED",
                "OmniVoice model does not expose generate()",
                retryable=False,
            )

        async with self._generate_lock:
            try:
                generated = await asyncio.to_thread(model.generate, **generate_kwargs)
            except OmniVoiceRuntimeError:
                raise
            except Exception as exc:
                self._record_error("MODEL_GENERATE_FAILED")
                raise OmniVoiceRuntimeError(
                    "MODEL_GENERATE_FAILED",
                    "OmniVoice generation failed",
                    retryable=True,
                ) from exc

        audio_bytes = self._to_wav_bytes(generated)
        self.status = "ready"
        self.last_error_code = None
        return OmniVoiceSynthesizeResult(
            audio_bytes=audio_bytes,
            audio_format="wav",
            sample_rate=self.NATIVE_SAMPLE_RATE,
            channels=1,
            cold_start=cold_start,
            model=self._model_id or self._resolve_model_id(self._resolve_model_path()),
        )

    def _resolve_model_path(self) -> Path:
        configured_model_path = str(self.config.get("model_path") or "").strip()
        if configured_model_path:
            candidate = Path(configured_model_path).expanduser()
        else:
            configured_model = str(self.config.get("model") or "").strip()
            candidate = Path(configured_model).expanduser() if configured_model else None
            if candidate is not None and not candidate.is_dir():
                candidate = None

        if candidate is None or not candidate.is_dir():
            self._record_error("MODEL_NOT_AVAILABLE")
            raise OmniVoiceRuntimeError(
                "MODEL_NOT_AVAILABLE",
                "OmniVoice requires a configured local model directory",
                retryable=False,
            )
        return candidate.resolve()

    def _resolve_model_id(self, model_path: Path) -> str:
        configured_model = str(self.config.get("model") or "").strip()
        if configured_model and not Path(configured_model).expanduser().is_dir():
            return configured_model
        return str(model_path)

    def _build_generation_kwargs(self, request: OmniVoiceSynthesizeRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"text": request.text}
        if request.language_id:
            kwargs["language"] = request.language_id
        if request.mode == "design":
            kwargs["instruct"] = request.instruct
        if request.mode == "clone":
            reference_audio_path = self._validate_reference_audio_path(request.reference_audio_path)
            kwargs["ref_audio"] = str(reference_audio_path)
            kwargs["ref_text"] = request.reference_text
        kwargs.update(request.generation.compact())
        return kwargs

    def _validate_reference_audio_path(self, reference_audio_path: str | None) -> Path:
        if not reference_audio_path:
            self._record_error("REFERENCE_PATH_NOT_ALLOWED")
            raise OmniVoiceRuntimeError(
                "REFERENCE_PATH_NOT_ALLOWED",
                "OmniVoice clone reference audio path is not configured",
                retryable=False,
            )

        reference_path = Path(reference_audio_path).expanduser().resolve(strict=False)
        allowed_dirs = self._managed_reference_dirs()
        if any(self._is_relative_to(reference_path, allowed_dir) for allowed_dir in allowed_dirs):
            if not reference_path.exists() or not reference_path.is_file():
                self._record_error("INVALID_REFERENCE_AUDIO")
                raise OmniVoiceRuntimeError(
                    "INVALID_REFERENCE_AUDIO",
                    "OmniVoice clone reference audio must be an existing regular file",
                    retryable=False,
                )
            return reference_path

        self._record_error("REFERENCE_PATH_NOT_ALLOWED")
        raise OmniVoiceRuntimeError(
            "REFERENCE_PATH_NOT_ALLOWED",
            "OmniVoice clone reference audio path is outside managed directories",
            retryable=False,
        )

    def _managed_reference_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        for key in self._REFERENCE_DIR_KEYS:
            value = self.config.get(key)
            if isinstance(value, str) and value.strip():
                dirs.append(Path(value).expanduser().resolve(strict=False))
        for key in self._REFERENCE_DIR_LIST_KEYS:
            value = self.config.get(key)
            if isinstance(value, str) and value.strip():
                dirs.append(Path(value).expanduser().resolve(strict=False))
            elif isinstance(value, Iterable):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        dirs.append(Path(item).expanduser().resolve(strict=False))
        return dirs

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
        except ValueError:
            return False
        return True

    def _to_wav_bytes(self, generated: object) -> bytes:
        audio = self._coerce_audio_array(generated)
        try:
            soundfile = importlib.import_module("soundfile")
            buffer = io.BytesIO()
            soundfile.write(buffer, audio, self.NATIVE_SAMPLE_RATE, format="WAV")
            return buffer.getvalue()
        except ImportError:
            return self._to_wav_bytes_stdlib(audio)
        except Exception as exc:
            self._record_error("AUDIO_ENCODING_FAILED")
            raise OmniVoiceRuntimeError(
                "AUDIO_ENCODING_FAILED",
                "OmniVoice audio output could not be encoded as WAV",
                retryable=False,
            ) from exc

    def _coerce_audio_array(self, generated: object) -> object:
        np = importlib.import_module("numpy")
        chunks: list[object] = []

        if isinstance(generated, np.ndarray):
            candidates = [generated]
        elif isinstance(generated, Iterable) and not isinstance(generated, (bytes, bytearray, str)):
            candidates = list(generated)
        else:
            candidates = [generated]

        for candidate in candidates:
            chunk = np.asarray(candidate)
            if chunk.ndim > 1:
                chunk = chunk.reshape(-1)
            if chunk.size:
                chunks.append(chunk)

        if not chunks:
            self._record_error("EMPTY_AUDIO_OUTPUT")
            raise OmniVoiceRuntimeError(
                "EMPTY_AUDIO_OUTPUT",
                "OmniVoice returned no audio samples",
                retryable=True,
            )

        audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        if audio.size == 0:
            self._record_error("EMPTY_AUDIO_OUTPUT")
            raise OmniVoiceRuntimeError(
                "EMPTY_AUDIO_OUTPUT",
                "OmniVoice returned no audio samples",
                retryable=True,
            )
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        return audio.astype("float32", copy=False)

    def _to_wav_bytes_stdlib(self, audio: object) -> bytes:
        np = importlib.import_module("numpy")
        pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2", copy=False)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.NATIVE_SAMPLE_RATE)
            wav_file.writeframes(pcm.tobytes())
        return buffer.getvalue()

    def _record_error(self, code: str) -> None:
        self.status = "error"
        self.last_error_code = code


__all__ = [
    "OmniVoiceRuntime",
    "OmniVoiceRuntimeError",
    "OmniVoiceSynthesizeResult",
]
