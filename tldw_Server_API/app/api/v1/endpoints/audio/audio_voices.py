# audio_voices.py
# Description: Custom voice management endpoints.
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Request, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from starlette import status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user, TokenScopeGuard, User

from tldw_Server_API.app.api.v1.endpoints.audio.audio_tts import get_tts_service
from tldw_Server_API.app.api.v1.schemas.audio_schemas import (
    OpenAISpeechRequest,
    VoiceEncodeRequest,
    VoiceEncodeResponse,
)
from tldw_Server_API.app.core.Audio.error_payloads import _http_error_detail
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2

router = APIRouter(
    tags=["Audio"],
    responses={
        404: {"description": "Not found"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)

VOICE_SCOPE_UPLOAD = "audio.voices.upload"
VOICE_SCOPE_ENCODE = "audio.voices.encode"
VOICE_SCOPE_LIST = "audio.voices.list"
VOICE_SCOPE_GET = "audio.voices.get"
VOICE_SCOPE_DELETE = "audio.voices.delete"
VOICE_SCOPE_PREVIEW = "audio.voices.preview"
VOICE_COUNTER_TYPE = "voice_call"


@router.post(
    "/voices/upload",
    summary="Upload a custom voice sample",
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            TokenScopeGuard(
                "any",
                require_if_present=True,
                endpoint_id=VOICE_SCOPE_UPLOAD,
                count_as=VOICE_COUNTER_TYPE,
            )
        ),
    ],
)
async def upload_voice(
    request: Request,
    file: UploadFile = File(..., description="Voice sample audio file (WAV, MP3, FLAC, OGG)"),
    name: str = Form(..., description="Name for the voice"),
    description: Optional[str] = Form(None, description="Description of the voice"),
    provider: str = Form(default="vibevoice", description="Target TTS provider"),
    reference_text: Optional[str] = Form(
        default=None,
        description="Optional transcript of the reference audio for cloning providers",
    ),
    current_user: User = Depends(get_request_user),
):
    """
    Upload a custom voice sample for use with TTS.
    """
    request_id = ensure_request_id(request)
    try:
        from tldw_Server_API.app.core.TTS.voice_manager import (
            VoiceProcessingError,
            VoiceQuotaExceededError,
            VoiceUploadRequest,
            get_voice_manager,
        )

        voice_manager = get_voice_manager()
        file_content = await file.read()

        upload_request = VoiceUploadRequest(
            name=name,
            description=description,
            provider=provider,
            reference_text=reference_text,
        )

        result = await voice_manager.upload_voice(
            user_id=current_user.id, file_content=file_content, filename=file.filename, request=upload_request
        )

        return result.model_dump()

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Custom voice upload is not available in this build"
        ) from None
    except VoiceQuotaExceededError as e:
        logger.warning("Voice quota exceeded")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=_http_error_detail("Voice quota exceeded", request_id, exc=e),
        ) from e
    except VoiceProcessingError as e:
        logger.warning("Voice processing failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_http_error_detail("Voice processing failed", request_id, exc=e),
        ) from e
    except Exception as e:
        logger.error("Voice upload error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload voice sample") from e


@router.post(
    "/voices/encode",
    summary="Encode stored voice reference for a provider",
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            TokenScopeGuard(
                "any",
                require_if_present=True,
                endpoint_id=VOICE_SCOPE_ENCODE,
                count_as=VOICE_COUNTER_TYPE,
            )
        ),
    ],
)
async def encode_voice_reference(
    payload: VoiceEncodeRequest,
    current_user: User = Depends(get_request_user),
):
    """
    Encode provider-specific artifacts for a stored voice reference.
    """
    try:
        from tldw_Server_API.app.core.TTS.voice_manager import (
            VoiceProcessingError,
            get_voice_manager,
        )

        voice_manager = get_voice_manager()
        result = await voice_manager.encode_voice_reference(
            user_id=current_user.id,
            voice_id=payload.voice_id,
            provider=payload.provider,
            reference_text=payload.reference_text,
            force=payload.force,
        )
        return VoiceEncodeResponse(**result.model_dump())
    except VoiceProcessingError as e:
        logger.warning("Voice encoding failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Voice encode error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to encode voice reference") from e


@router.get(
    "/voices",
    summary="List user's custom voices",
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            TokenScopeGuard(
                "any",
                require_if_present=True,
                endpoint_id=VOICE_SCOPE_LIST,
                count_as=VOICE_COUNTER_TYPE,
            )
        ),
    ],
)
async def list_voices(request: Request, current_user: User = Depends(get_request_user)):
    """
    List all custom voice samples uploaded by the user.
    """
    try:
        from tldw_Server_API.app.core.TTS.voice_manager import get_voice_manager

        voice_manager = get_voice_manager()
        voices = await voice_manager.list_user_voices(current_user.id, refresh=True)

        return {"voices": [voice.model_dump() for voice in voices], "count": len(voices)}

    except ImportError:
        return {"voices": [], "count": 0}
    except Exception as e:
        logger.error("Error listing voices")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list voices") from e


@router.get(
    "/voices/{voice_id}",
    summary="Get voice details",
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            TokenScopeGuard(
                "any",
                require_if_present=True,
                endpoint_id=VOICE_SCOPE_GET,
                count_as=VOICE_COUNTER_TYPE,
            )
        ),
    ],
)
async def get_voice_details(
    request: Request, voice_id: str = Path(..., description="Voice ID"), current_user: User = Depends(get_request_user)
):
    """
    Get detailed information about a specific voice.
    """
    try:
        from tldw_Server_API.app.core.TTS.voice_manager import get_voice_manager

        voice_manager = get_voice_manager()
        voice = await voice_manager.get_voice(current_user.id, voice_id, refresh=True)

        if not voice:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found")

        return voice.model_dump()

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Custom voice management not available") from None
    except Exception as e:
        logger.error("Error getting voice details")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get voice details") from e


@router.delete(
    "/voices/{voice_id}",
    summary="Delete a custom voice",
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            TokenScopeGuard(
                "any",
                require_if_present=True,
                endpoint_id=VOICE_SCOPE_DELETE,
                count_as=VOICE_COUNTER_TYPE,
            )
        ),
    ],
)
async def delete_voice(
    request: Request,
    voice_id: str = Path(..., description="Voice ID to delete"),
    current_user: User = Depends(get_request_user),
):
    """
    Delete a custom voice sample.
    """
    try:
        from tldw_Server_API.app.core.TTS.voice_manager import get_voice_manager

        voice_manager = get_voice_manager()
        deleted = await voice_manager.delete_voice(current_user.id, voice_id)

        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found")

        return {"message": "Voice deleted successfully", "voice_id": voice_id}

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Custom voice management not available") from None
    except Exception as e:
        logger.error("Error deleting voice")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete voice") from e


@router.post(
    "/voices/{voice_id}/preview",
    summary="Generate voice preview",
    response_class=StreamingResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "MP3 audio stream for a custom voice preview",
            "content": {"audio/mpeg": {}},
        },
    },
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            TokenScopeGuard(
                "any",
                require_if_present=True,
                endpoint_id=VOICE_SCOPE_PREVIEW,
                count_as=VOICE_COUNTER_TYPE,
            )
        ),
    ],
)
async def preview_voice(
    request: Request,
    voice_id: str = Path(..., description="Voice ID to preview"),
    text: str = Form(default="Hello, this is a preview of your custom voice.", description="Text to speak"),
    current_user: User = Depends(get_request_user),
    tts_service: TTSServiceV2 = Depends(get_tts_service),
):
    """
    Generate a short preview of a custom voice.
    """
    request_id = ensure_request_id(request)
    try:
        from tldw_Server_API.app.core.TTS.voice_manager import get_voice_manager

        voice_manager = get_voice_manager()
        voice = await voice_manager.get_voice(current_user.id, voice_id, refresh=True)

        if not voice:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found")

        if len(text) > 100:
            text = text[:100]

        preview_request = OpenAISpeechRequest(
            model=voice.provider, input=text, voice=f"custom:{voice_id}", response_format="mp3", stream=True
        )

        audio_stream = tts_service.generate_speech(
            preview_request,
            provider=None,
            fallback=True,
            user_id=current_user.id,
            request_id=request_id,
        )

        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename=preview_{voice_id}.mp3",
                "X-Voice-Name": voice.name,
                "X-Request-Id": request_id,
            },
        )

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Custom voice preview not available") from None
    except Exception as e:
        logger.error("Voice preview error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate voice preview"
        ) from e
