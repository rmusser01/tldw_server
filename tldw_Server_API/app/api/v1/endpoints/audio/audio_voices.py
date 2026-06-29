# audio_voices.py
# Description: Custom voice management endpoints.
import base64
import binascii
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Request, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from starlette import status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    TokenScopeGuard,
    User,
    check_rate_limit,
    get_request_user,
    require_token_scope,
)

from tldw_Server_API.app.api.v1.endpoints.audio.audio_tts import get_tts_service
from tldw_Server_API.app.api.v1.schemas.audio_schemas import (
    FishS2ReferenceDeleteResponse,
    FishS2ReferenceImportResponse,
    FishS2ReferenceListResponse,
    FishS2ReferenceResponse,
    OpenAISpeechRequest,
    VoiceEncodeRequest,
    VoiceEncodeResponse,
)
from tldw_Server_API.app.core.Audio.error_payloads import _http_error_detail
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id
from tldw_Server_API.app.core.TTS.fish_s2_reference_imports import (
    FISH_S2_REFERENCE_IMPORT_MAX_BYTES,
    FISH_S2_REFERENCE_IMPORT_MAX_DECODED_AUDIO_BYTES,
    FISH_S2_REFERENCE_IMPORT_MAX_ITEMS,
    FishS2ReferenceImportError,
    parse_fish_s2_reference_import_result,
)
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSError
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


def _fish_s2_import_error(index: int, message: str, code: Optional[str] = None) -> dict[str, object]:
    payload: dict[str, object] = {"index": index, "message": message}
    if code:
        payload["code"] = code
    return payload


def _estimated_base64_decoded_size(value: str) -> int:
    encoded = value.strip()
    padding = len(encoded) - len(encoded.rstrip("="))
    return max(0, (len(encoded) * 3 // 4) - padding)


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


@router.post(
    "/providers/fish_s2/references",
    response_model=FishS2ReferenceResponse,
    response_model_exclude_none=True,
    summary="Create or sync a managed Fish S2 reference",
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            require_token_scope(
                "any",
                require_if_present=True,
                endpoint_id=VOICE_SCOPE_UPLOAD,
                count_as=VOICE_COUNTER_TYPE,
            )
        ),
    ],
)
async def create_fish_s2_reference(
    request: Request,
    voice_id: Optional[str] = Form(default=None, description="Existing stored voice ID to sync"),
    reference_text: Optional[str] = Form(default=None, description="Transcript of the reference audio"),
    name: Optional[str] = Form(default=None, description="Name when creating from a new upload"),
    description: Optional[str] = Form(default=None, description="Description when creating from a new upload"),
    force: bool = Form(default=False, description="Recreate the remote Fish reference even if already cached"),
    file: Optional[UploadFile] = File(default=None, description="Optional audio upload when creating a new managed reference"),
    current_user: User = Depends(get_request_user),
    tts_service: TTSServiceV2 = Depends(get_tts_service),
):
    """Create a managed Fish S2 reference from an existing stored voice or a new upload."""
    request_id = ensure_request_id(request)
    if voice_id:
        if file is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_http_error_detail("Provide either voice_id or file, not both", request_id),
            )
        file_content = None
        filename = None
    else:
        if file is None or not name or not reference_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_http_error_detail(
                    "file, name, and reference_text are required when voice_id is not provided",
                    request_id,
                ),
            )
        file_content = await file.read()
        filename = file.filename

    try:
        return await tts_service.create_fish_s2_reference(
            user_id=current_user.id,
            voice_id=voice_id,
            file_content=file_content,
            filename=filename,
            name=name,
            description=description,
            reference_text=reference_text,
            force=force,
        )
    except HTTPException:
        raise
    except Exception as e:
        if e.__class__.__name__ == "VoiceProcessingError":
            logger.warning(f"Fish S2 reference creation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_http_error_detail("Fish S2 reference creation failed", request_id, exc=e),
            ) from e
        logger.error(f"Fish S2 reference creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create Fish S2 reference",
        ) from e


@router.get(
    "/providers/fish_s2/references",
    response_model=FishS2ReferenceListResponse,
    response_model_exclude_none=True,
    summary="List managed Fish S2 references for the current user",
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            require_token_scope(
                "any",
                require_if_present=True,
                endpoint_id=VOICE_SCOPE_LIST,
                count_as=VOICE_COUNTER_TYPE,
            )
        ),
    ],
)
async def list_fish_s2_references(
    current_user: User = Depends(get_request_user),
    tts_service: TTSServiceV2 = Depends(get_tts_service),
):
    """List Fish S2 managed references from local user-scoped metadata."""
    try:
        references = await tts_service.list_fish_s2_references(user_id=current_user.id)
        return {"references": references, "count": len(references)}
    except Exception as e:
        logger.error(f"Fish S2 reference listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list Fish S2 references",
        ) from e


@router.post(
    "/providers/fish_s2/references/import",
    response_model=FishS2ReferenceImportResponse,
    response_model_exclude_none=True,
    summary="Import managed Fish S2 references from JSON or Markdown",
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            require_token_scope(
                "any",
                require_if_present=True,
                endpoint_id=VOICE_SCOPE_UPLOAD,
                count_as=VOICE_COUNTER_TYPE,
            )
        ),
    ],
)
async def import_fish_s2_references(
    request: Request,
    file: UploadFile = File(..., description="Fish S2 reference import file (.json, .md, .markdown)"),
    force: bool = Form(default=False, description="Recreate remote Fish references unless an item overrides force"),
    current_user: User = Depends(get_request_user),
    tts_service: TTSServiceV2 = Depends(get_tts_service),
):
    """Import one or more Fish S2 managed references from JSON or Markdown."""
    request_id = ensure_request_id(request)
    try:
        content = await file.read(FISH_S2_REFERENCE_IMPORT_MAX_BYTES + 1)
        if len(content) > FISH_S2_REFERENCE_IMPORT_MAX_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_http_error_detail("Fish S2 reference import file is too large", request_id),
            )
        parsed = parse_fish_s2_reference_import_result(
            filename=file.filename or "",
            content=content,
            max_items=FISH_S2_REFERENCE_IMPORT_MAX_ITEMS,
            max_bytes=FISH_S2_REFERENCE_IMPORT_MAX_BYTES,
        )
    except FishS2ReferenceImportError as e:
        logger.warning(f"Fish S2 reference import parse failed: {e}")
        detail = _http_error_detail("Fish S2 reference import file is invalid", request_id, exc=e)
        detail["details"] = str(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        ) from e

    results = []
    errors = [_fish_s2_import_error(error.index, error.message) for error in parsed.errors]
    for item in parsed.items:
        file_content = None
        filename = None
        if item.audio_base64:
            if _estimated_base64_decoded_size(item.audio_base64) > FISH_S2_REFERENCE_IMPORT_MAX_DECODED_AUDIO_BYTES:
                errors.append(
                    _fish_s2_import_error(
                        item.source_index,
                        "audio_base64 decoded audio exceeds the maximum size",
                    )
                )
                continue
            try:
                file_content = base64.b64decode(item.audio_base64, validate=True)
            except (binascii.Error, ValueError) as e:
                errors.append(_fish_s2_import_error(item.source_index, "audio_base64 must be valid base64"))
                logger.warning(f"Fish S2 reference import item {item.source_index} has invalid audio_base64: {e}")
                continue
            if len(file_content) > FISH_S2_REFERENCE_IMPORT_MAX_DECODED_AUDIO_BYTES:
                errors.append(
                    _fish_s2_import_error(
                        item.source_index,
                        "audio_base64 decoded audio exceeds the maximum size",
                    )
                )
                continue
            filename = item.filename

        try:
            item_result = await tts_service.create_fish_s2_reference(
                user_id=current_user.id,
                voice_id=item.voice_id,
                file_content=file_content,
                filename=filename,
                name=item.name,
                description=item.description,
                reference_text=item.reference_text,
                force=item.force if item.force is not None else force,
            )
            result_payload = dict(item_result) if isinstance(item_result, dict) else {"result": item_result}
            results.append({"index": item.source_index, **result_payload})
        except TTSError as e:
            logger.warning(f"Fish S2 reference import item {item.source_index} failed: {e}", exc_info=True)
            errors.append(_fish_s2_import_error(item.source_index, str(e), getattr(e, "error_code", None)))
            continue
        except Exception as e:
            if e.__class__.__name__ == "VoiceProcessingError":
                logger.warning(f"Fish S2 reference import item {item.source_index} failed: {e}", exc_info=True)
                errors.append(_fish_s2_import_error(item.source_index, str(e), getattr(e, "error_code", None)))
                continue
            logger.error(f"Fish S2 reference import error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to import Fish S2 references",
            ) from e

    response = {
        "results": results,
        "errors": errors,
        "imported": len(results),
        "failed": len(errors),
    }
    if not results and errors:
        detail = _http_error_detail("Fish S2 reference import failed", request_id)
        detail["errors"] = errors
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    return response


@router.delete(
    "/providers/fish_s2/references/{reference_id}",
    response_model=FishS2ReferenceDeleteResponse,
    response_model_exclude_none=True,
    summary="Delete a managed Fish S2 reference",
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            require_token_scope(
                "any",
                require_if_present=True,
                endpoint_id=VOICE_SCOPE_DELETE,
                count_as=VOICE_COUNTER_TYPE,
            )
        ),
    ],
)
async def delete_fish_s2_reference(
    request: Request,
    reference_id: str = Path(..., description="Local Fish S2 reference ID"),
    current_user: User = Depends(get_request_user),
    tts_service: TTSServiceV2 = Depends(get_tts_service),
):
    """Delete the remote Fish S2 reference while preserving the local voice asset."""
    request_id = ensure_request_id(request)
    try:
        return await tts_service.delete_fish_s2_reference(
            user_id=current_user.id,
            reference_id=reference_id,
        )
    except Exception as e:
        if e.__class__.__name__ == "VoiceProcessingError":
            logger.warning(f"Fish S2 reference deletion failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=_http_error_detail("Fish S2 reference not found", request_id, exc=e),
            ) from e
        logger.error(f"Fish S2 reference deletion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete Fish S2 reference",
        ) from e
