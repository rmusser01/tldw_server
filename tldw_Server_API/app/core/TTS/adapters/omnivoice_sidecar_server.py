from __future__ import annotations

import ipaddress
import os
import wave
from io import BytesIO
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status

from .omnivoice_sidecar_protocol import (
    OmniVoiceHealthResponse,
    OmniVoiceSynthesizeRequest,
    OmniVoiceSynthesizeResponse,
    X_TLDW_SIDECAR_TOKEN_HEADER,
)


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


def create_app(*, sidecar_token: str) -> FastAPI:
    """Create the narrow internal OmniVoice sidecar app."""
    app = FastAPI(title="OmniVoice Sidecar", version="0.1.0")

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
        return OmniVoiceHealthResponse()

    @app.post("/control/warmup", response_model=OmniVoiceHealthResponse)
    async def warmup(_: None = Depends(require_sidecar_token)) -> OmniVoiceHealthResponse:
        return OmniVoiceHealthResponse()

    @app.post("/control/reload", response_model=OmniVoiceHealthResponse)
    async def reload_runtime(_: None = Depends(require_sidecar_token)) -> OmniVoiceHealthResponse:
        return OmniVoiceHealthResponse()

    @app.post("/control/shutdown", response_model=OmniVoiceHealthResponse)
    async def shutdown(_: None = Depends(require_sidecar_token)) -> OmniVoiceHealthResponse:
        return OmniVoiceHealthResponse(status="shutting-down", ready=False)

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
        metadata = OmniVoiceSynthesizeResponse(sample_rate=request.sample_rate, mode=request.mode)
        audio_bytes = _build_silent_wav(sample_rate=request.sample_rate, channels=metadata.channels)
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
    return create_app(sidecar_token=token)


app = _load_app_from_env() if os.environ.get("OMNIVOICE_SIDECAR_TOKEN") else None


if __name__ == "__main__":  # pragma: no cover - runtime entrypoint
    import uvicorn

    host = validate_loopback_host(os.environ.get("OMNIVOICE_SIDECAR_HOST", "127.0.0.1"))
    port = int(os.environ.get("OMNIVOICE_SIDECAR_PORT", "8039"))
    if app is None:
        raise RuntimeError("OMNIVOICE_SIDECAR_TOKEN is required")
    uvicorn.run(app, host=host, port=port, log_level="warning")
