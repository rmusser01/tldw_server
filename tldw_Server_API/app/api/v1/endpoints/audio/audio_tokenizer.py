# audio_tokenizer.py
# Description: Audio tokenizer encode/decode endpoints.
import io
from typing import Any, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user, TokenScopeGuard, User

from tldw_Server_API.app.api.v1.schemas.audio_schemas import (
    AudioTokenizerDecodeRequest,
    AudioTokenizerEncodeRequest,
    AudioTokenizerEncodeResponse,
)
from tldw_Server_API.app.core.Audio.error_payloads import _http_error_detail
from tldw_Server_API.app.core.Audio.tokenizer_service import (
    _decode_base64_payload,
    _enforce_payload_limit,
    _enforce_payload_size,
    _get_qwen3_tokenizer_settings,
    _load_qwen3_tokenizer,
    _normalize_tokens,
    _read_audio_from_bytes,
    _resolve_tokenizer_frame_rate,
    _resolve_tokenizer_sample_rate,
    _serialize_audio_output,
    _serialize_tokens,
)
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id

router = APIRouter(
    tags=["Audio"],
    responses={
        404: {"description": "Not found"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)


def _audio_shim_attr(name: str):
    defaults: dict[str, Any] = {
        "_get_qwen3_tokenizer_settings": _get_qwen3_tokenizer_settings,
        "_load_qwen3_tokenizer": _load_qwen3_tokenizer,
    }
    try:
        from tldw_Server_API.app.api.v1.endpoints import audio as audio_shim

        if hasattr(audio_shim, name):
            return getattr(audio_shim, name)
    except Exception as resolve_error:
        _ = resolve_error  # best-effort shim resolution; default map below
    if name in defaults:
        return defaults[name]
    raise NameError(name)


@router.post(
    "/tokenizer/encode",
    summary="Encode audio into Qwen3-TTS tokens",
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            TokenScopeGuard(
                "audio.tokenizer",
                require_if_present=True,
                endpoint_id="audio.tokenizer.encode",
                count_as="call",
            )
        ),
    ],
)
async def encode_audio_tokenizer(
    request: Request,
    current_user: User = Depends(get_request_user),
):
    request_id = ensure_request_id(request)
    settings = _audio_shim_attr("_get_qwen3_tokenizer_settings")()
    tokenizer_model = settings["tokenizer_model"]
    # Response format label, not a credential.
    token_format = "list"  # nosec B105
    sample_rate_hint: Optional[int] = None

    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file") or form.get("audio")
        if not isinstance(upload, UploadFile):
            raise HTTPException(
                status_code=400,
                detail=_http_error_detail("Missing audio file in multipart form", request_id),
            )
        audio_bytes = await upload.read()
        tokenizer_model = str(form.get("tokenizer_model") or tokenizer_model)
        token_format = str(form.get("token_format") or token_format)
        try:
            if form.get("sample_rate") is not None:
                sample_rate_hint = int(form.get("sample_rate"))
        except Exception:
            sample_rate_hint = None
    else:
        try:
            payload = AudioTokenizerEncodeRequest(**(await request.json()))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=_http_error_detail("Invalid JSON payload", request_id, exc=exc),
            ) from exc
        audio_bytes = _decode_base64_payload(payload.audio_base64)
        tokenizer_model = payload.tokenizer_model or tokenizer_model
        token_format = payload.token_format or token_format
        sample_rate_hint = payload.sample_rate

    if token_format not in {"list", "base64"}:
        raise HTTPException(
            status_code=400,
            detail=_http_error_detail("token_format must be 'list' or 'base64'", request_id),
        )

    _enforce_payload_limit(audio_bytes, settings["tokenizer_max_payload_mb"], request_id)
    audio_data, sample_rate, duration_seconds = _read_audio_from_bytes(
        audio_bytes, sample_rate_hint, request_id
    )

    max_audio_seconds = settings["tokenizer_max_audio_seconds"]
    if max_audio_seconds > 0 and duration_seconds > max_audio_seconds:
        raise HTTPException(
            status_code=413,
            detail=_http_error_detail(
                f"Audio too long ({duration_seconds:.2f}s, max {max_audio_seconds}s)",
                request_id,
            ),
        )

    tokenizer = _audio_shim_attr("_load_qwen3_tokenizer")(tokenizer_model, settings["auto_download"])
    encode_fn = getattr(tokenizer, "encode", None)
    if not callable(encode_fn):
        raise HTTPException(
            status_code=501,
            detail="Tokenizer backend does not expose encode()",
        )

    try:
        tokens_raw = encode_fn(audio_data, sample_rate=sample_rate)
    except TypeError:
        try:
            tokens_raw = encode_fn(audio_data, sample_rate)
        except TypeError:
            tokens_raw = encode_fn(audio_data)

    tokens, frame_rate = _normalize_tokens(tokens_raw)
    max_tokens = settings["tokenizer_max_tokens"]
    if max_tokens > 0 and len(tokens) > max_tokens:
        raise HTTPException(
            status_code=413,
            detail=_http_error_detail(
                f"Token payload too large ({len(tokens)} tokens, max {max_tokens})",
                request_id,
            ),
        )

    if frame_rate is None:
        frame_rate = _resolve_tokenizer_frame_rate(tokenizer)

    payload_tokens = _serialize_tokens(tokens, token_format)
    response = AudioTokenizerEncodeResponse(
        tokens=payload_tokens,
        token_format=token_format,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        tokenizer_model=tokenizer_model,
        duration_seconds=duration_seconds,
    )
    return JSONResponse(content=response.model_dump(), headers={"X-Request-Id": request_id})


@router.post(
    "/tokenizer/decode",
    summary="Decode Qwen3-TTS tokens into audio",
    response_class=Response,
    responses={
        200: {
            "description": "Decoded audio bytes from tokenizer tokens.",
            "content": {
                "audio/wav": {},
                "application/octet-stream": {},
            },
        },
    },
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            TokenScopeGuard(
                "audio.tokenizer",
                require_if_present=True,
                endpoint_id="audio.tokenizer.decode",
                count_as="call",
            )
        ),
    ],
)
async def decode_audio_tokenizer(
    payload: AudioTokenizerDecodeRequest,
    request: Request,
    current_user: User = Depends(get_request_user),
):
    request_id = ensure_request_id(request)
    settings = _audio_shim_attr("_get_qwen3_tokenizer_settings")()
    tokenizer_model = payload.tokenizer_model or settings["tokenizer_model"]

    if isinstance(payload.tokens, str):
        token_bytes = _decode_base64_payload(payload.tokens)
        _enforce_payload_limit(token_bytes, settings["tokenizer_max_payload_mb"], request_id)
        tokens = np.frombuffer(token_bytes, dtype=np.int32).tolist()
    elif isinstance(payload.tokens, list):
        token_bytes_len = len(payload.tokens) * 4
        _enforce_payload_size(token_bytes_len, settings["tokenizer_max_payload_mb"], request_id)
        tokens = [int(tok) for tok in payload.tokens]
    else:
        raise HTTPException(
            status_code=400,
            detail=_http_error_detail("tokens must be list[int] or base64 string", request_id),
        )

    max_tokens = settings["tokenizer_max_tokens"]
    if max_tokens > 0 and len(tokens) > max_tokens:
        raise HTTPException(
            status_code=413,
            detail=_http_error_detail(
                f"Token payload too large ({len(tokens)} tokens, max {max_tokens})",
                request_id,
            ),
        )

    tokenizer = _audio_shim_attr("_load_qwen3_tokenizer")(tokenizer_model, settings["auto_download"])
    decode_fn = getattr(tokenizer, "decode", None)
    if not callable(decode_fn):
        raise HTTPException(
            status_code=501,
            detail="Tokenizer backend does not expose decode()",
        )

    try:
        decoded = decode_fn(tokens)
    except TypeError:
        decoded = decode_fn(tokens=tokens)

    sample_rate = _resolve_tokenizer_sample_rate(tokenizer, 24000)
    audio = decoded
    if isinstance(decoded, tuple) and len(decoded) == 2:
        audio, sample_rate = decoded
    elif isinstance(decoded, dict):
        audio = decoded.get("audio") or decoded.get("samples") or decoded.get("pcm")
        sample_rate = int(decoded.get("sample_rate") or sample_rate)

    audio_bytes = _serialize_audio_output(audio, sample_rate, payload.response_format)
    duration_seconds = 0.0
    try:
        if payload.response_format == "pcm":
            duration_seconds = len(audio_bytes) / 2.0 / float(sample_rate or 24000)
        else:
            with sf.SoundFile(io.BytesIO(audio_bytes)) as info:
                duration_seconds = float(len(info)) / float(info.samplerate or sample_rate)
    except Exception:
        duration_seconds = 0.0

    media_type = "audio/wav" if payload.response_format == "wav" else "application/octet-stream"
    headers = {
        "X-Request-Id": request_id,
        "X-Audio-Sample-Rate": str(sample_rate),
        "X-Audio-Duration-Seconds": str(duration_seconds),
        "X-Tokenizer-Model": tokenizer_model,
    }
    return Response(content=audio_bytes, media_type=media_type, headers=headers)
