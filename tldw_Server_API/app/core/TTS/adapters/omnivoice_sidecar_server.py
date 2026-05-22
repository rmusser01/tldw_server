from __future__ import annotations

import ipaddress
import inspect
import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import JSONResponse

from .omnivoice_runtime import OmniVoiceRuntime, OmniVoiceRuntimeError
from .omnivoice_sidecar_protocol import (
    OmniVoiceHealthResponse,
    OmniVoiceRuntimeStatus,
    OmniVoiceSynthesizeRequest,
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


def load_runtime_config_from_env() -> dict[str, Any]:
    """Load OmniVoice runtime configuration from sidecar environment variables."""
    return {
        "model": os.environ.get("OMNIVOICE_MODEL", "omnivoice"),
        "model_path": os.environ.get("OMNIVOICE_MODEL_PATH"),
        "runtime_path": os.environ.get("OMNIVOICE_RUNTIME_PATH"),
        "scratch_dir": os.environ.get("OMNIVOICE_SCRATCH_DIR"),
        "device_map": os.environ.get("OMNIVOICE_DEVICE_MAP"),
        "dtype": os.environ.get("OMNIVOICE_DTYPE"),
    }


def _runtime_error_status_code(exc: OmniVoiceRuntimeError) -> int:
    if exc.code == "RUNTIME_RELOAD_UNSUPPORTED":
        return status.HTTP_501_NOT_IMPLEMENTED
    if exc.code in {"MODEL_NOT_AVAILABLE", "MODEL_LOAD_FAILED", "RUNTIME_IMPORT_FAILED"}:
        return status.HTTP_503_SERVICE_UNAVAILABLE
    if exc.code in {
        "INVALID_REFERENCE_AUDIO",
        "INVALID_GENERATION_PARAMETER",
        "REFERENCE_PATH_NOT_ALLOWED",
    }:
        return status.HTTP_422_UNPROCESSABLE_ENTITY
    if exc.retryable:
        return status.HTTP_503_SERVICE_UNAVAILABLE
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _runtime_error_response(exc: OmniVoiceRuntimeError) -> JSONResponse:
    return JSONResponse(
        status_code=_runtime_error_status_code(exc),
        content={
            "error": {
                "code": exc.code,
                "message": str(exc),
                "retryable": exc.retryable,
            }
        },
    )


def _runtime_model_id(runtime: OmniVoiceRuntime) -> str | None:
    model_id = getattr(runtime, "_model_id", None)
    if model_id:
        return str(model_id)
    model = getattr(runtime, "model", None)
    if isinstance(model, str) and model:
        return model
    configured_model = getattr(runtime, "config", {}).get("model") if hasattr(runtime, "config") else None
    return str(configured_model) if configured_model else None


def _runtime_model_path(runtime: OmniVoiceRuntime) -> str | None:
    return None


async def _runtime_status(runtime: OmniVoiceRuntime) -> OmniVoiceRuntimeStatus:
    status_value = getattr(runtime, "status", "idle_stopped")
    if callable(status_value):
        status_value = status_value()
        if inspect.isawaitable(status_value):
            status_value = await status_value
        if isinstance(status_value, OmniVoiceRuntimeStatus):
            return status_value
        if isinstance(status_value, dict):
            return OmniVoiceRuntimeStatus(**status_value)

    status_text = str(status_value or "idle_stopped")
    ready = status_text == "ready" and getattr(runtime, "model", None) is not None
    return OmniVoiceRuntimeStatus(
        status=status_text,
        ready=ready,
        model=_runtime_model_id(runtime),
        model_path=_runtime_model_path(runtime),
        last_error_code=getattr(runtime, "last_error_code", None),
    )


def create_app(*, sidecar_token: str, runtime: OmniVoiceRuntime | None = None) -> FastAPI:
    """Create the narrow internal OmniVoice sidecar app."""
    app = FastAPI(title="OmniVoice Sidecar", version="0.1.0")
    runtime = runtime or OmniVoiceRuntime(load_runtime_config_from_env())

    async def require_sidecar_token(
        supplied_token: str | None = Header(default=None, alias=X_TLDW_SIDECAR_TOKEN_HEADER),
    ) -> None:
        if not supplied_token or supplied_token != sidecar_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid sidecar token",
            )

    @app.get("/status", response_model=OmniVoiceRuntimeStatus)
    async def get_status(_: None = Depends(require_sidecar_token)) -> OmniVoiceRuntimeStatus:
        return await _runtime_status(runtime)

    @app.get("/health", response_model=OmniVoiceHealthResponse)
    async def health(_: None = Depends(require_sidecar_token)) -> OmniVoiceHealthResponse:
        return OmniVoiceHealthResponse(status="ok", ready=True)

    @app.post("/control/warmup", response_model=OmniVoiceHealthResponse)
    async def warmup(_: None = Depends(require_sidecar_token)) -> OmniVoiceRuntimeStatus | JSONResponse:
        try:
            await runtime.load()
        except OmniVoiceRuntimeError as exc:
            return _runtime_error_response(exc)
        return await _runtime_status(runtime)

    @app.post("/control/reload", response_model=OmniVoiceHealthResponse)
    async def reload_runtime(_: None = Depends(require_sidecar_token)) -> OmniVoiceRuntimeStatus | JSONResponse:
        try:
            reload_method = getattr(runtime, "reload", None)
            if not callable(reload_method):
                return _runtime_error_response(
                    OmniVoiceRuntimeError(
                        "RUNTIME_RELOAD_UNSUPPORTED",
                        "OmniVoice runtime reload is not supported",
                        retryable=False,
                    )
                )
            result = reload_method()
            if inspect.isawaitable(result):
                await result
        except OmniVoiceRuntimeError as exc:
            return _runtime_error_response(exc)
        return await _runtime_status(runtime)

    @app.post("/control/shutdown", response_model=OmniVoiceHealthResponse)
    async def shutdown(_: None = Depends(require_sidecar_token)) -> OmniVoiceRuntimeStatus | JSONResponse:
        try:
            shutdown_method = getattr(runtime, "shutdown", None)
            if callable(shutdown_method):
                result = shutdown_method()
                if inspect.isawaitable(result):
                    await result
        except OmniVoiceRuntimeError as exc:
            return _runtime_error_response(exc)

        current_status = await _runtime_status(runtime)
        if callable(getattr(runtime, "shutdown", None)):
            return OmniVoiceRuntimeStatus(
                status=current_status.status,
                ready=False,
                model=current_status.model,
                model_path=current_status.model_path,
                last_error_code=current_status.last_error_code,
            )
        return OmniVoiceRuntimeStatus(
            status="shutting-down",
            ready=False,
            model=current_status.model,
            model_path=current_status.model_path,
            last_error_code=current_status.last_error_code,
        )

    @app.post("/v1/synthesize", response_model=None)
    async def synthesize(
        request: OmniVoiceSynthesizeRequest,
        _: None = Depends(require_sidecar_token),
    ) -> Response | JSONResponse:
        try:
            result = await runtime.synthesize(request)
        except OmniVoiceRuntimeError as exc:
            return _runtime_error_response(exc)

        content_type = f"audio/{result.audio_format}"
        return Response(
            content=result.audio_bytes,
            media_type=content_type,
            headers={
                "X-OmniVoice-Audio-Format": result.audio_format,
                "X-OmniVoice-Sample-Rate": str(result.sample_rate),
                "X-OmniVoice-Channels": str(result.channels),
                "X-OmniVoice-Provider": "omnivoice",
                "X-OmniVoice-Mode": request.mode,
                "X-OmniVoice-Model": result.model,
                "X-OmniVoice-Cold-Start": str(result.cold_start).lower(),
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
