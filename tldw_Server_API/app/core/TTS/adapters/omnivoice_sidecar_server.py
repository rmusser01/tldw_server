from __future__ import annotations

import asyncio
import gc
import ipaddress
import os
import threading
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from loguru import logger

from .omnivoice_sidecar_protocol import (
    X_TLDW_SIDECAR_TOKEN_HEADER,
    OmniVoiceHealthResponse,
    OmniVoiceSynthesizeRequest,
    OmniVoiceSynthesizeResponse,
)


class OmniVoiceRuntimeError(RuntimeError):
    """Raised for expected OmniVoice runtime load or synthesis failures."""


class OmniVoiceRuntime(Protocol):
    """Runtime contract used by the internal sidecar app."""

    runtime_mode: str

    def health(self) -> OmniVoiceHealthResponse:
        """Return sidecar process and model readiness state."""

    def warmup(self) -> OmniVoiceHealthResponse:
        """Warm the runtime if supported and return current readiness."""

    def reload(self) -> OmniVoiceHealthResponse:
        """Reload the runtime if supported and return current readiness."""

    def shutdown(self) -> OmniVoiceHealthResponse:
        """Prepare the runtime for shutdown and return current readiness."""

    def synthesize(
        self,
        request: OmniVoiceSynthesizeRequest,
    ) -> tuple[bytes, OmniVoiceSynthesizeResponse]:
        """Generate audio bytes and response metadata for a request."""


def validate_loopback_host(host: str | None) -> str:
    """Normalize accepted loopback bind hosts and reject everything else."""
    candidate = str(host or "").strip()
    if not candidate or candidate.lower() == "localhost":
        return "127.0.0.1"

    try:
        parsed = ipaddress.ip_address(candidate)
    except ValueError as exc:
        raise ValueError("OmniVoice sidecar host must be a loopback address") from exc

    if not parsed.is_loopback:
        raise ValueError("OmniVoice sidecar host must be a loopback address")
    return "127.0.0.1" if parsed.version == 4 else "::1"


def _build_silent_wav(*, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Create a minimal valid WAV payload for the sidecar boundary contract."""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * sample_width * channels * 32)
    return buffer.getvalue()


class StubOmniVoiceRuntime:
    """Explicit test/development runtime that returns valid silent WAV bytes."""

    runtime_mode = "stub"

    def health(self) -> OmniVoiceHealthResponse:
        return OmniVoiceHealthResponse(
            runtime_mode=self.runtime_mode,
            model_loaded=False,
            model_ready=True,
        )

    def warmup(self) -> OmniVoiceHealthResponse:
        return self.health()

    def reload(self) -> OmniVoiceHealthResponse:
        return self.health()

    def shutdown(self) -> OmniVoiceHealthResponse:
        return self.health().model_copy(update={"status": "shutting-down", "ready": False})

    def synthesize(
        self,
        request: OmniVoiceSynthesizeRequest,
    ) -> tuple[bytes, OmniVoiceSynthesizeResponse]:
        metadata = OmniVoiceSynthesizeResponse(sample_rate=request.sample_rate, mode=request.mode)
        audio_bytes = _build_silent_wav(sample_rate=request.sample_rate, channels=metadata.channels)
        return audio_bytes, metadata


class RealOmniVoiceRuntime:
    """Lazy OmniVoice runtime isolated inside the sidecar process."""

    DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"
    runtime_mode = "real"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        dtype: str | None = None,
        model_loader: Callable[..., Any] | None = None,
        wav_writer: Callable[[BytesIO, Any, int], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._model_loader = model_loader or self._default_model_loader
        self._wav_writer = wav_writer or self._default_wav_writer
        self._model: Any | None = None
        self._last_error: str | None = None
        self._lock = threading.Lock()

    def health(self) -> OmniVoiceHealthResponse:
        model_loaded = self._model is not None
        return OmniVoiceHealthResponse(
            runtime_mode=self.runtime_mode,
            model_loaded=model_loaded,
            model_ready=model_loaded and self._last_error is None,
            last_error=self._last_error,
        )

    def warmup(self) -> OmniVoiceHealthResponse:
        with self._lock:
            self._load_model_locked()
            return self._health_locked()

    def reload(self) -> OmniVoiceHealthResponse:
        with self._lock:
            self._model = None
            self._last_error = None
            self._load_model_locked()
            return self._health_locked()

    def shutdown(self) -> OmniVoiceHealthResponse:
        with self._lock:
            self._model = None
            gc.collect()
            return OmniVoiceHealthResponse(
                status="shutting-down",
                ready=False,
                runtime_mode=self.runtime_mode,
                model_loaded=False,
                model_ready=False,
                last_error=self._last_error,
            )

    def synthesize(
        self,
        request: OmniVoiceSynthesizeRequest,
    ) -> tuple[bytes, OmniVoiceSynthesizeResponse]:
        with self._lock:
            model = self._load_model_locked()
            generate_kwargs = self._build_generate_kwargs(request)
            try:
                generated_audio = model.generate(**generate_kwargs)
                audio = self._prepare_audio_for_wav_writer(generated_audio)
                sample_rate = int(getattr(model, "sampling_rate", request.sample_rate) or request.sample_rate)
                buffer = BytesIO()
                self._wav_writer(buffer, audio, sample_rate)
            except OmniVoiceRuntimeError:
                raise
            except Exception as exc:
                self._last_error = f"OmniVoice generation failed: {exc}"
                raise OmniVoiceRuntimeError(self._last_error) from exc

            self._last_error = None
            metadata = OmniVoiceSynthesizeResponse(sample_rate=sample_rate, mode=request.mode)
            return buffer.getvalue(), metadata

    def _health_locked(self) -> OmniVoiceHealthResponse:
        model_loaded = self._model is not None
        return OmniVoiceHealthResponse(
            runtime_mode=self.runtime_mode,
            model_loaded=model_loaded,
            model_ready=model_loaded and self._last_error is None,
            last_error=self._last_error,
        )

    def _load_model_locked(self) -> Any:
        if self._model is not None:
            return self._model

        resolved_device = self.device or self._detect_device()
        resolved_dtype = self.dtype or self._default_dtype_for_device(resolved_device)
        try:
            self._model = self._model_loader(
                model_id=self.model_id,
                device=resolved_device,
                dtype=resolved_dtype,
            )
        except OmniVoiceRuntimeError as exc:
            self._last_error = str(exc)
            raise
        except Exception as exc:
            self._last_error = f"OmniVoice model load failed: {exc}"
            raise OmniVoiceRuntimeError(self._last_error) from exc
        self._last_error = None
        return self._model

    @staticmethod
    def _build_generate_kwargs(request: OmniVoiceSynthesizeRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"text": request.text}
        if request.language:
            kwargs["language"] = request.language
        if request.instruct:
            kwargs["instruct"] = request.instruct
        if request.duration is not None:
            kwargs["duration"] = request.duration
        if request.speed is not None:
            kwargs["speed"] = request.speed
        if request.mode == "clone":
            kwargs["ref_audio"] = request.reference_audio_path
            kwargs["ref_text"] = request.reference_text
        kwargs.update(request.generation_params)
        return kwargs

    @staticmethod
    def _prepare_audio_for_wav_writer(generated_audio: Any) -> Any:
        audio = generated_audio[0] if isinstance(generated_audio, (list, tuple)) else generated_audio
        detach = getattr(audio, "detach", None)
        if callable(detach):
            audio = detach()
        cpu = getattr(audio, "cpu", None)
        if callable(cpu):
            audio = cpu()
        numpy = getattr(audio, "numpy", None)
        if callable(numpy):
            audio = numpy()
        squeeze = getattr(audio, "squeeze", None)
        if callable(squeeze):
            audio = squeeze()
        return audio

    @staticmethod
    def _default_dtype_for_device(device: str) -> str:
        normalized = device.lower()
        if normalized.startswith("cuda") or normalized == "mps":
            return "float16"
        return "float32"

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
        except Exception as exc:
            raise OmniVoiceRuntimeError(f"OmniVoice runtime dependency missing: torch ({exc})") from exc

        if torch.cuda.is_available():
            return "cuda:0"
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_torch_dtype(torch_module: Any, dtype: str) -> Any:
        dtype_name = dtype.strip().lower()
        dtype_value = getattr(torch_module, dtype_name, None)
        if dtype_value is None:
            raise OmniVoiceRuntimeError(f"Unsupported OmniVoice dtype: {dtype}")
        return dtype_value

    @classmethod
    def _default_model_loader(cls, *, model_id: str, device: str, dtype: str) -> Any:
        try:
            import torch
            from omnivoice import OmniVoice
        except Exception as exc:
            raise OmniVoiceRuntimeError(f"OmniVoice runtime dependency missing: {exc}") from exc

        torch_dtype = cls._resolve_torch_dtype(torch, dtype)
        try:
            return OmniVoice.from_pretrained(model_id, device_map=device, dtype=torch_dtype)
        except Exception as exc:
            raise OmniVoiceRuntimeError(f"OmniVoice model load failed: {exc}") from exc

    @staticmethod
    def _default_wav_writer(buffer: BytesIO, audio: Any, sample_rate: int) -> None:
        try:
            import soundfile as sf
        except Exception as exc:
            raise OmniVoiceRuntimeError(f"OmniVoice runtime dependency missing: soundfile ({exc})") from exc

        try:
            sf.write(buffer, audio, sample_rate, format="WAV")
        except Exception as exc:
            raise OmniVoiceRuntimeError(f"OmniVoice WAV encoding failed: {exc}") from exc


def create_app(*, sidecar_token: str, runtime: OmniVoiceRuntime | None = None) -> FastAPI:
    """Create the narrow internal OmniVoice sidecar app."""
    app = FastAPI(title="OmniVoice Sidecar", version="0.1.0")
    sidecar_runtime = runtime or StubOmniVoiceRuntime()
    app.state.omnivoice_runtime = sidecar_runtime
    app.state.uvicorn_server = None

    def _request_shutdown() -> None:
        server = getattr(app.state, "uvicorn_server", None)
        if server is not None:
            server.should_exit = True

    app.state.request_shutdown = _request_shutdown

    async def require_sidecar_token(
        supplied_token: str | None = Header(default=None, alias=X_TLDW_SIDECAR_TOKEN_HEADER),
    ) -> None:
        if not supplied_token or supplied_token != sidecar_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid sidecar token",
            )

    @app.get("/health", response_model=OmniVoiceHealthResponse)
    async def health(_: None = Depends(require_sidecar_token)) -> OmniVoiceHealthResponse:
        return sidecar_runtime.health()

    @app.post("/control/warmup", response_model=OmniVoiceHealthResponse)
    async def warmup(_: None = Depends(require_sidecar_token)) -> OmniVoiceHealthResponse:
        return await asyncio.to_thread(sidecar_runtime.warmup)

    @app.post("/control/reload", response_model=OmniVoiceHealthResponse)
    async def reload_runtime(_: None = Depends(require_sidecar_token)) -> OmniVoiceHealthResponse:
        return await asyncio.to_thread(sidecar_runtime.reload)

    @app.post("/control/shutdown", response_model=OmniVoiceHealthResponse)
    async def shutdown(_: None = Depends(require_sidecar_token)) -> OmniVoiceHealthResponse:
        result = await asyncio.to_thread(sidecar_runtime.shutdown)
        app.state.request_shutdown()
        return result

    @app.post("/v1/synthesize")
    async def synthesize(
        request: OmniVoiceSynthesizeRequest,
        _: None = Depends(require_sidecar_token),
    ) -> Response:
        if request.mode == "clone":
            if not request.reference_audio_path:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Clone reference audio path does not exist",
                )
            reference_path = Path(request.reference_audio_path)
            if not reference_path.is_file():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Clone reference audio path does not exist",
                )
        try:
            audio_bytes, metadata = await asyncio.to_thread(sidecar_runtime.synthesize, request)
        except OmniVoiceRuntimeError as exc:
            logger.opt(exception=True).warning("OmniVoice sidecar synthesis failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OmniVoice service unavailable",
            ) from exc
        return Response(
            content=audio_bytes,
            media_type=metadata.content_type,
            headers={
                "X-OmniVoice-Audio-Format": metadata.audio_format,
                "X-OmniVoice-Sample-Rate": str(metadata.sample_rate),
                "X-OmniVoice-Channels": str(metadata.channels),
                "X-OmniVoice-Provider": metadata.provider,
                "X-OmniVoice-Mode": metadata.mode,
            },
        )

    return app


def _load_app_from_env() -> FastAPI:
    token = os.environ["OMNIVOICE_SIDECAR_TOKEN"]
    runtime_mode = os.environ.get("OMNIVOICE_RUNTIME_MODE", "stub").strip().lower()
    if runtime_mode == "stub":
        runtime: OmniVoiceRuntime = StubOmniVoiceRuntime()
    elif runtime_mode == "real":
        runtime = RealOmniVoiceRuntime(
            model_id=(os.environ.get("OMNIVOICE_MODEL") or RealOmniVoiceRuntime.DEFAULT_MODEL_ID).strip(),
            device=(os.environ.get("OMNIVOICE_DEVICE") or "").strip() or None,
            dtype=(os.environ.get("OMNIVOICE_DTYPE") or "").strip() or None,
        )
    else:
        raise RuntimeError("OMNIVOICE_RUNTIME_MODE must be 'stub' or 'real'")
    return create_app(sidecar_token=token, runtime=runtime)


app = _load_app_from_env() if os.environ.get("OMNIVOICE_SIDECAR_TOKEN") else None


if __name__ == "__main__":  # pragma: no cover - runtime entrypoint
    import uvicorn

    host = validate_loopback_host(os.environ.get("OMNIVOICE_SIDECAR_HOST", "127.0.0.1"))
    port = int(os.environ.get("OMNIVOICE_SIDECAR_PORT", "8039"))
    if app is None:
        raise RuntimeError("OMNIVOICE_SIDECAR_TOKEN is required")
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    app.state.uvicorn_server = server
    server.run()
