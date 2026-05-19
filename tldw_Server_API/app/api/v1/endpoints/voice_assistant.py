# voice_assistant.py
# Voice Assistant API endpoints (WebSocket + REST)
#
# Provides:
# - WebSocket endpoint for real-time voice assistant sessions
# - REST endpoints for text-based commands and configuration
#
#######################################################################################################################
import asyncio
import base64
import contextlib
import json
import os
import time
import uuid
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user,
    get_chacha_db_for_user_id,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.voice_assistant_schemas import (
    VoiceActionType,
    VoiceAnalytics,
    VoiceAnalyticsSummary,
    VoiceAssistantState,
    VoiceCommandDefinition,
    VoiceCommandInfo,
    VoiceCommandListResponse,
    VoiceCommandRequest,
    VoiceCommandResponse,
    VoiceCommandToggleRequest,
    VoiceCommandUsage,
    VoiceCommandValidationResponse,
    VoiceCommandValidationStep,
    VoiceSessionInfo,
    VoiceSessionListResponse,
    VoiceWorkflowTemplateInfo,
    VoiceWorkflowTemplateListResponse,
    WorkflowCancelResponse,
    WorkflowStatusResponse,
    WSActionResultMessage,
    WSActionStartMessage,
    WSAuthErrorMessage,
    WSAuthOKMessage,
    WSConfigAckMessage,
    WSErrorMessage,
    WSIntentMessage,
    WSMessageType,
    WSStateChangeMessage,
    WSTranscriptionMessage,
    WSTTSChunkMessage,
    WSTTSEndMessage,
    WSWorkflowCompleteMessage,
    WSWorkflowProgressMessage,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSError
from tldw_Server_API.app.core.TTS.tts_request_resolution import (
    resolve_tts_request_defaults,
)
from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime
from tldw_Server_API.app.core.VoiceAssistant import (
    ActionType,
    VoiceCommand,
    VoiceCommandRouter,
    VoiceSessionManager,
    count_user_voice_sessions,
    get_active_voice_session_count,
    get_user_voice_sessions,
    get_voice_analytics_summary_stats,
    get_voice_command_counts,
    get_voice_command_registry,
    get_voice_command_router,
    get_voice_command_usage_stats,
    get_voice_session_manager,
    get_voice_top_commands,
    get_voice_usage_by_day,
    get_voice_workflow_handler,
    save_voice_command,
    save_voice_session,
)
from tldw_Server_API.app.core.VoiceAssistant import (
    delete_voice_command as delete_voice_command_db,
)
from tldw_Server_API.app.core.VoiceAssistant import (
    delete_voice_session as delete_voice_session_db,
)
from tldw_Server_API.app.core.VoiceAssistant import (
    get_voice_command as get_voice_command_db,
)
from tldw_Server_API.app.core.VoiceAssistant import (
    get_voice_session as get_voice_session_db,
)

# REST router
router = APIRouter(
    tags=["voice-assistant"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        429: {"description": "Rate limit exceeded"},
    },
)

# WebSocket router (separate to avoid auth middleware conflicts)
ws_router = APIRouter()

_AUTH_ACCESS_KIND = "access"


# Helper functions

def _action_type_to_voice(action_type: ActionType) -> VoiceActionType:
    """Convert internal ActionType to API VoiceActionType."""
    return VoiceActionType(action_type.value)


def _voice_to_action_type(voice_type: VoiceActionType) -> ActionType:
    """Convert API VoiceActionType to internal ActionType."""
    return ActionType(voice_type.value)


def _voice_command_to_info(command: VoiceCommand) -> VoiceCommandInfo:
    """Serialize an internal voice command into the API response model."""
    return VoiceCommandInfo(
        id=command.id,
        user_id=command.user_id,
        persona_id=command.persona_id,
        connection_id=command.connection_id,
        name=command.name,
        phrases=command.phrases,
        action_type=_action_type_to_voice(command.action_type),
        action_config=command.action_config,
        priority=command.priority,
        enabled=command.enabled,
        requires_confirmation=command.requires_confirmation,
        description=command.description,
        created_at=command.created_at,
    )


def _looks_like_jwt(token: Optional[str]) -> bool:
    return isinstance(token, str) and token.count(".") == 2


async def _authenticate_websocket(
    websocket: WebSocket,
    token: Optional[str] = None,
) -> tuple[bool, Optional[int]]:
    """
    Authenticate WebSocket connection.

    Args:
        websocket: The WebSocket connection
        token: JWT token or API key

    Returns:
        Tuple of (authenticated, user_id)
    """
    if not token:
        return False, None

    if _looks_like_jwt(token):
        try:
            from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError, TokenExpiredError
            from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
            from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager

            jwt_service = get_jwt_service()
            payload = await jwt_service.verify_token_async(token, token_type=_AUTH_ACCESS_KIND)
            session_manager = await get_session_manager()
            if await session_manager.is_token_blacklisted(token, payload.get("jti")):
                return False, None

            user_id = payload.get("user_id") or payload.get("sub")
            if isinstance(user_id, str):
                try:
                    user_id = int(user_id)
                except (TypeError, ValueError):
                    user_id = None
            if isinstance(user_id, int):
                return True, user_id
        except (InvalidTokenError, TokenExpiredError):
            pass
        except (AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError):
            pass

    try:
        from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
        from tldw_Server_API.app.core.AuthNZ.ip_allowlist import is_single_user_ip_allowed, resolve_client_ip
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        settings = get_settings()
        client_ip = resolve_client_ip(websocket, settings)
        if getattr(settings, "AUTH_MODE", None) == "single_user":
            allowed_keys: set[str] = set()
            primary_key = getattr(settings, "SINGLE_USER_API_KEY", None)
            if primary_key:
                allowed_keys.add(primary_key)
            if is_explicit_pytest_runtime():
                test_key = os.getenv("SINGLE_USER_TEST_API_KEY")
                if test_key:
                    allowed_keys.add(test_key)
            if token in allowed_keys and is_single_user_ip_allowed(client_ip, settings):
                return True, int(getattr(settings, "SINGLE_USER_FIXED_ID", 1))
            return False, None

        api_mgr = await get_api_key_manager()
        info = await api_mgr.validate_api_key(api_key=token, required_scope="read", ip_address=client_ip)
        if info and info.get("user_id") is not None:
            return True, int(info["user_id"])
    except (AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        pass

    return False, None


async def _generate_tts_audio(
    text: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    voice: Optional[str] = None,
    response_format: str = "mp3",
) -> tuple[bytes, str]:
    """
    Generate TTS audio for response text.

    Returns:
        Tuple of (audio_bytes, mime_type)
    """
    try:
        from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
        from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2

        tts_service = await get_tts_service_v2()
        resolved = resolve_tts_request_defaults(
            provider=provider,
            model=model,
            voice=voice,
        )

        request = OpenAISpeechRequest(
            model=resolved.model,
            input=text,
            voice=resolved.voice,
            response_format=response_format,
            stream=False,
        )

        # Collect all audio chunks
        audio_chunks = []
        async for chunk in tts_service.generate_speech(
            request=request,
            provider=resolved.provider,
            fallback=True,
        ):
            if chunk:
                audio_chunks.append(chunk)

        audio_bytes = b"".join(audio_chunks)

        # Determine MIME type
        mime_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }
        mime_type = mime_types.get(response_format, "audio/mpeg")

        return audio_bytes, mime_type

    except (AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        logger.error("TTS generation failed")
        return b"", "audio/mpeg"


# REST Endpoints

@router.post(
    "/command",
    response_model=VoiceCommandResponse,
    summary="Process voice command",
    description="Process a text command as if spoken (bypasses STT).",
)
async def process_voice_command(
    request: VoiceCommandRequest,
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandResponse:
    """Process a voice command from text input."""
    start_time = time.time()

    router_instance = get_voice_command_router()

    # Process command
    result, session_id = await router_instance.process_command(
        text=request.text,
        user_id=current_user.id,
        session_id=request.session_id,
        db=db,
        persona_id=request.persona_id,
    )

    # Build intent message
    intent_msg = WSIntentMessage(
        type=WSMessageType.INTENT,
        action_type=_action_type_to_voice(result.action_type),
        command_name=None,  # TODO: Get from parsed intent
        entities={},
        confidence=1.0,
        requires_confirmation=False,
    )

    # Build action result message
    action_result_msg = WSActionResultMessage(
        type=WSMessageType.ACTION_RESULT,
        success=result.success,
        action_type=_action_type_to_voice(result.action_type),
        result_data=result.result_data,
        response_text=result.response_text,
        execution_time_ms=result.execution_time_ms,
    )

    # Generate TTS if requested
    output_audio = None
    output_format = None
    if request.include_tts and result.response_text:
        audio_bytes, mime_type = await _generate_tts_audio(
            text=result.response_text,
            provider=request.tts_provider,
            model=request.tts_model,
            voice=request.tts_voice,
            response_format=request.tts_format,
        )
        if audio_bytes:
            output_audio = base64.b64encode(audio_bytes).decode("ascii")
            output_format = request.tts_format

    return VoiceCommandResponse(
        session_id=session_id,
        success=result.success,
        transcription=request.text,
        intent=intent_msg,
        action_result=action_result_msg,
        output_audio=output_audio,
        output_audio_format=output_format,
        processing_time_ms=(time.time() - start_time) * 1000,
    )


@router.get(
    "/commands",
    response_model=VoiceCommandListResponse,
    summary="List voice commands",
    description="List all available voice commands for the current user.",
)
async def list_voice_commands(
    include_system: bool = Query(True, description="Include system commands"),
    include_disabled: bool = Query(False, description="Include disabled commands"),
    persona_id: Optional[str] = Query(None, description="Optional persona scope"),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandListResponse:
    """List available voice commands."""
    registry = get_voice_command_registry()
    registry.load_defaults()
    # Refresh user commands from DB to ensure persistence
    registry.refresh_user_commands(
        db,
        user_id=current_user.id,
        include_disabled=include_disabled,
        persona_id=persona_id,
    )

    commands = registry.get_all_commands(
        current_user.id,
        include_system=include_system,
        include_disabled=include_disabled,
        persona_id=persona_id,
    )

    command_infos = [_voice_command_to_info(cmd) for cmd in commands]

    return VoiceCommandListResponse(
        commands=command_infos,
        total=len(command_infos),
    )


@router.post(
    "/commands",
    response_model=VoiceCommandInfo,
    status_code=status.HTTP_201_CREATED,
    summary="Create voice command",
    description="Create a new user-specific voice command.",
)
async def create_voice_command(
    definition: VoiceCommandDefinition,
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandInfo:
    """Create a new voice command."""
    command = VoiceCommand(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        persona_id=definition.persona_id,
        connection_id=definition.connection_id,
        name=definition.name,
        phrases=definition.phrases,
        action_type=_voice_to_action_type(definition.action_type),
        action_config=definition.action_config,
        priority=definition.priority,
        enabled=definition.enabled,
        requires_confirmation=definition.requires_confirmation,
        description=definition.description,
    )

    save_voice_command(db, command)

    registry = get_voice_command_registry()
    registry.load_defaults()
    registry.register_command(command)

    saved = get_voice_command_db(
        db,
        command.id,
        current_user.id,
        persona_id=definition.persona_id,
    )
    if not saved:
        saved = command

    return _voice_command_to_info(saved)


@router.get(
    "/commands/{command_id}",
    response_model=VoiceCommandInfo,
    summary="Get voice command",
    description="Get a specific voice command by ID.",
)
async def get_voice_command_endpoint(
    command_id: str,
    persona_id: Optional[str] = Query(None, description="Optional persona scope"),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandInfo:
    """Get a specific voice command."""
    registry = get_voice_command_registry()
    registry.load_defaults()

    command = get_voice_command_db(db, command_id, current_user.id, persona_id=persona_id)
    if not command:
        # Fallback to system registry commands
        command = registry.get_command(command_id, current_user.id, persona_id=persona_id)

    if not command:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Command not found",
        )

    if command.user_id not in (0, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this command",
        )

    return _voice_command_to_info(command)


@router.put(
    "/commands/{command_id}",
    response_model=VoiceCommandInfo,
    summary="Update voice command",
    description="Update a user-specific voice command.",
)
async def update_voice_command(
    command_id: str,
    definition: VoiceCommandDefinition,
    persona_id: Optional[str] = Query(None, description="Optional persona scope"),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandInfo:
    """Update a voice command."""
    effective_persona_id = definition.persona_id if definition.persona_id is not None else persona_id
    existing = get_voice_command_db(db, command_id, current_user.id, persona_id=effective_persona_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Command not found",
        )

    if existing.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot modify system commands",
        )

    # Preserve enabled if the field wasn't provided
    enabled_value = (
        definition.enabled
        if "enabled" in definition.model_fields_set
        else existing.enabled
    )

    updated = VoiceCommand(
        id=command_id,
        user_id=current_user.id,
        persona_id=effective_persona_id if effective_persona_id is not None else existing.persona_id,
        connection_id=definition.connection_id if definition.connection_id is not None else existing.connection_id,
        name=definition.name,
        phrases=definition.phrases,
        action_type=_voice_to_action_type(definition.action_type),
        action_config=definition.action_config,
        priority=definition.priority,
        enabled=enabled_value,
        requires_confirmation=definition.requires_confirmation,
        description=definition.description,
        created_at=existing.created_at,
    )

    save_voice_command(db, updated)

    registry = get_voice_command_registry()
    registry.load_defaults()
    registry.register_command(updated)

    saved = get_voice_command_db(
        db,
        command_id,
        current_user.id,
        persona_id=updated.persona_id,
    ) or updated

    return _voice_command_to_info(saved)


@router.post(
    "/commands/{command_id}/toggle",
    response_model=VoiceCommandInfo,
    summary="Toggle voice command",
    description="Enable or disable a voice command.",
)
async def toggle_voice_command(
    command_id: str,
    payload: VoiceCommandToggleRequest,
    persona_id: Optional[str] = Query(None, description="Optional persona scope"),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandInfo:
    """Toggle a voice command enabled state."""
    existing = get_voice_command_db(db, command_id, current_user.id, persona_id=persona_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Command not found",
        )

    if existing.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot modify system commands",
        )

    updated = VoiceCommand(
        id=command_id,
        user_id=existing.user_id,
        persona_id=existing.persona_id,
        connection_id=existing.connection_id,
        name=existing.name,
        phrases=existing.phrases,
        action_type=existing.action_type,
        action_config=existing.action_config,
        priority=existing.priority,
        enabled=payload.enabled,
        requires_confirmation=existing.requires_confirmation,
        description=existing.description,
        created_at=existing.created_at,
    )

    save_voice_command(db, updated)

    registry = get_voice_command_registry()
    registry.load_defaults()
    registry.register_command(updated)

    saved = get_voice_command_db(
        db,
        command_id,
        current_user.id,
        persona_id=existing.persona_id,
    ) or updated

    return _voice_command_to_info(saved)


@router.post(
    "/commands/{command_id}/validate",
    response_model=VoiceCommandValidationResponse,
    summary="Validate voice command (dry run)",
    description=(
        "Validate a voice command's action configuration without executing it. "
        "Checks config schema, target existence (MCP tool, workflow template, "
        "custom handler), and trigger phrases."
    ),
)
async def validate_voice_command(
    command_id: str,
    persona_id: Optional[str] = Query(None, description="Optional persona scope"),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandValidationResponse:
    """Dry-run validation: check config without executing the command."""
    registry = get_voice_command_registry()
    registry.load_defaults()

    command = get_voice_command_db(db, command_id, current_user.id, persona_id=persona_id)
    if not command:
        command = registry.get_command(command_id, current_user.id, persona_id=persona_id)

    if not command:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Command not found",
        )

    if command.user_id not in (0, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to validate this command",
        )

    router_instance = get_voice_command_router()
    raw_steps = await router_instance.validate_command_config(
        command,
        db=db,
        persona_id=persona_id,
    )

    steps = [VoiceCommandValidationStep(**s) for s in raw_steps]
    all_passed = all(s.passed for s in steps)

    return VoiceCommandValidationResponse(
        command_id=command.id,
        command_name=command.name,
        action_type=_action_type_to_voice(command.action_type),
        valid=all_passed,
        steps=steps,
    )


@router.get(
    "/commands/{command_id}/usage",
    response_model=VoiceCommandUsage,
    summary="Get voice command usage",
    description="Get usage statistics for a voice command.",
)
async def get_voice_command_usage(
    command_id: str,
    days: int = Query(30, ge=1, le=365, description="Lookback window in days"),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandUsage:
    """Get usage stats for a voice command."""
    stats = get_voice_command_usage_stats(
        db,
        command_id=command_id,
        user_id=current_user.id,
        days=days,
    )
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No usage data found for this command",
        )

    return VoiceCommandUsage(**stats)


@router.delete(
    "/commands/{command_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete voice command",
    description="Delete a user voice command.",
)
async def delete_voice_command(
    command_id: str,
    persona_id: Optional[str] = Query(None, description="Optional persona scope"),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> None:
    """Delete a voice command."""
    existing = get_voice_command_db(db, command_id, current_user.id, persona_id=persona_id)
    scoped_persona_id = existing.persona_id if existing else persona_id
    deleted = delete_voice_command_db(db, command_id, current_user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Command not found or not authorized",
        )

    registry = get_voice_command_registry()
    registry.unregister_command(command_id, current_user.id, persona_id=scoped_persona_id)


@router.get(
    "/sessions",
    response_model=VoiceSessionListResponse,
    summary="List voice sessions",
    description="List active voice sessions for the current user.",
)
async def list_voice_sessions(
    active_only: bool = Query(True, description="Only return active sessions"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum sessions to return"),
    offset: int = Query(0, ge=0, description="Zero-based pagination offset"),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceSessionListResponse:
    """List active voice sessions."""
    active_after = None
    if active_only:
        from datetime import datetime, timedelta

        active_after = datetime.utcnow() - timedelta(seconds=VoiceSessionManager.SESSION_TIMEOUT)

    sessions = get_user_voice_sessions(
        db,
        user_id=current_user.id,
        limit=limit,
        offset=offset,
        active_after=active_after,
    )
    total = count_user_voice_sessions(
        db,
        user_id=current_user.id,
        active_after=active_after,
    )

    session_infos = [
        VoiceSessionInfo(
            session_id=session.session_id,
            user_id=session.user_id,
            state=VoiceAssistantState(session.state.value),
            created_at=session.created_at,
            last_activity=session.last_activity,
            turn_count=len(session.conversation_history),
        )
        for session in sessions
    ]
    pagination = build_offset_pagination_meta(
        limit=limit,
        offset=offset,
        total=total,
        count=len(session_infos),
    )

    return VoiceSessionListResponse(
        sessions=session_infos,
        total=total,
        pagination=pagination,
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="End voice session",
    description="End an active voice session.",
)
async def end_voice_session(
    session_id: str,
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> None:
    """End a voice session."""
    session = get_voice_session_db(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to end this session",
        )

    delete_voice_session_db(db, session_id)

    # Best-effort cleanup in memory
    session_manager = get_voice_session_manager()
    await session_manager.end_session(session_id)


@router.get(
    "/sessions/{session_id}",
    response_model=VoiceSessionInfo,
    summary="Get voice session",
    description="Get details for a voice session.",
)
async def get_voice_session_endpoint(
    session_id: str,
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceSessionInfo:
    """Get a voice session by ID."""
    session = get_voice_session_db(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session",
        )

    return VoiceSessionInfo(
        session_id=session.session_id,
        user_id=session.user_id,
        state=VoiceAssistantState(session.state.value),
        created_at=session.created_at,
        last_activity=session.last_activity,
        turn_count=len(session.conversation_history),
    )


@router.get(
    "/analytics",
    response_model=VoiceAnalyticsSummary,
    summary="Get voice analytics summary",
    description="Get aggregated voice assistant usage analytics.",
)
async def get_voice_analytics(
    days: int = Query(7, ge=1, le=365, description="Lookback window in days"),
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceAnalyticsSummary:
    """Get voice analytics summary for the current user."""
    summary_stats = get_voice_analytics_summary_stats(
        db,
        user_id=current_user.id,
        days=days,
    )
    top_commands_raw = get_voice_top_commands(
        db,
        user_id=current_user.id,
        days=days,
        limit=10,
    )
    usage_by_day_raw = get_voice_usage_by_day(
        db,
        user_id=current_user.id,
        days=days,
    )

    registry = get_voice_command_registry()
    registry.load_defaults()
    system_commands = registry.get_all_commands(
        user_id=0,
        include_system=True,
        include_disabled=True,
    )
    system_total = len(system_commands)
    system_enabled = len([c for c in system_commands if c.enabled])

    user_counts = get_voice_command_counts(db, user_id=current_user.id)

    active_sessions = get_active_voice_session_count(
        db,
        user_id=current_user.id,
        activity_window_seconds=VoiceSessionManager.SESSION_TIMEOUT,
    )

    usage_by_day = [
        VoiceAnalytics(
            date=row["date"],
            total_commands=row["total_commands"],
            unique_users=row["unique_users"],
            success_rate=row["success_rate"],
            avg_response_time_ms=row["avg_response_time_ms"],
            top_commands=[],
        )
        for row in usage_by_day_raw
    ]

    top_commands = [
        VoiceCommandUsage(
            command_id=cmd["command_id"],
            command_name=cmd.get("command_name"),
            total_invocations=cmd["total_invocations"],
            success_count=cmd["success_count"],
            error_count=cmd["error_count"],
            avg_response_time_ms=cmd["avg_response_time_ms"],
            last_used=cmd.get("last_used"),
        )
        for cmd in top_commands_raw
    ]

    return VoiceAnalyticsSummary(
        total_commands_processed=summary_stats["total_commands"],
        active_sessions=active_sessions,
        total_voice_commands=user_counts["total"] + system_total,
        enabled_commands=user_counts["enabled"] + system_enabled,
        success_rate=summary_stats["success_rate"],
        avg_response_time_ms=summary_stats["avg_response_time_ms"],
        top_commands=top_commands,
        usage_by_day=usage_by_day,
    )


# Workflow REST Endpoints

@router.get(
    "/workflows/templates",
    response_model=VoiceWorkflowTemplateListResponse,
    summary="List workflow templates",
    description="List available voice workflow templates.",
)
async def list_workflow_templates(
    current_user: User = Depends(get_request_user),
) -> VoiceWorkflowTemplateListResponse:
    """List available voice workflow templates."""
    workflow_handler = get_voice_workflow_handler()
    templates = workflow_handler.get_voice_workflow_templates()

    template_infos = []
    for template_id, definition in templates.items():
        template_infos.append(VoiceWorkflowTemplateInfo(
            template_id=template_id,
            name=definition.get("name", template_id),
            description=definition.get("metadata", {}).get("description"),
            voice_trigger=definition.get("metadata", {}).get("voice_trigger", True),
            steps_count=len(definition.get("steps", [])),
        ))

    return VoiceWorkflowTemplateListResponse(
        templates=template_infos,
        total=len(template_infos),
    )


@router.get(
    "/workflows/{run_id}/status",
    response_model=WorkflowStatusResponse,
    summary="Get workflow status",
    description="Get the status of a workflow run.",
)
async def get_workflow_status(
    run_id: str,
    current_user: User = Depends(get_request_user),
) -> WorkflowStatusResponse:
    """Get workflow run status."""
    router_instance = get_voice_command_router()
    status_data = await router_instance.get_workflow_status(run_id, current_user.id)

    if not status_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow run not found or not authorized",
        )

    return WorkflowStatusResponse(
        run_id=status_data["run_id"],
        status=status_data["status"],
        status_reason=status_data.get("status_reason"),
        started_at=status_data.get("started_at"),
        ended_at=status_data.get("ended_at"),
        duration_ms=status_data.get("duration_ms"),
        outputs=status_data.get("outputs"),
        error=status_data.get("error"),
    )


@router.post(
    "/workflows/{run_id}/cancel",
    response_model=WorkflowCancelResponse,
    summary="Cancel workflow",
    description="Cancel a running workflow.",
)
async def cancel_workflow(
    run_id: str,
    current_user: User = Depends(get_request_user),
) -> WorkflowCancelResponse:
    """Cancel a running workflow."""
    router_instance = get_voice_command_router()
    cancelled = await router_instance.cancel_workflow(run_id, current_user.id)

    if cancelled:
        return WorkflowCancelResponse(
            run_id=run_id,
            cancelled=True,
            message="Workflow cancellation requested",
        )
    else:
        return WorkflowCancelResponse(
            run_id=run_id,
            cancelled=False,
            message="Could not cancel workflow (not found, not authorized, or already completed)",
        )


# WebSocket Endpoint

@ws_router.websocket("/assistant")
async def websocket_voice_assistant(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time voice assistant sessions.

    Protocol:
    1. Client sends AUTH message with token
    2. Server responds with AUTH_OK or AUTH_ERROR
    3. Client sends CONFIG message with preferences
    4. Server responds with CONFIG_ACK
    5. Client streams AUDIO messages
    6. Client sends COMMIT when utterance ends
    7. Server sends TRANSCRIPTION, INTENT, ACTION_START, ACTION_RESULT, TTS_CHUNK*, TTS_END
    8. Repeat from step 5
    """
    await websocket.accept()

    user_id: Optional[int] = None
    session_id: Optional[str] = None
    db: Optional[CharactersRAGDB] = None
    config: dict[str, Any] = {}
    transcriber = None

    try:
        # Wait for authentication
        auth_timeout = 10.0  # seconds
        try:
            auth_data = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=auth_timeout,
            )
        except asyncio.TimeoutError:
            await websocket.send_json(
                WSAuthErrorMessage(error="Authentication timeout").model_dump()
            )
            await websocket.close(code=4401)
            return

        # Process auth message
        if auth_data.get("type") != WSMessageType.AUTH.value:
            await websocket.send_json(
                WSAuthErrorMessage(error="Expected AUTH message").model_dump()
            )
            await websocket.close(code=4400)
            return

        auth_token = auth_data.get("token") or token
        authenticated, user_id = await _authenticate_websocket(websocket, auth_token)

        if not authenticated or user_id is None:
            await websocket.send_json(
                WSAuthErrorMessage(error="Invalid credentials").model_dump()
            )
            await websocket.close(code=4401)
            return

        # Create session
        session_manager = get_voice_session_manager()
        await session_manager.start()
        session, _ = await session_manager.get_or_create_session(
            session_id=None,
            user_id=user_id,
        )
        session_id = session.session_id

        # Initialize DB + load user commands for persistence
        try:
            db = await get_chacha_db_for_user_id(user_id, client_id="voice_assistant")
            save_voice_session(db, session)
            registry = get_voice_command_registry()
            registry.load_defaults()
            registry.refresh_user_commands(db, user_id, include_disabled=True)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.warning("Voice assistant DB init failed")

        await websocket.send_json(
            WSAuthOKMessage(
                user_id=user_id,
                session_id=session_id,
            ).model_dump()
        )

        logger.info(f"Voice assistant WebSocket authenticated: user={user_id}, session={session_id}")

        # Main message loop
        router_instance = get_voice_command_router()
        audio_buffer: list[bytes] = []

        while True:
            try:
                message = await websocket.receive_json()
                msg_type = message.get("type")

                if msg_type == WSMessageType.CONFIG.value:
                    # Store configuration
                    config = {
                        "stt_model": message.get("stt_model", "parakeet"),
                        "stt_language": message.get("stt_language"),
                        "tts_provider": None,
                        "tts_model": None,
                        "tts_voice": None,
                        "tts_format": message.get("tts_format", "mp3"),
                        "sample_rate": message.get("sample_rate", 16000),
                    }
                    resolved_tts = resolve_tts_request_defaults(
                        provider=message.get("tts_provider"),
                        model=message.get("tts_model"),
                        voice=message.get("tts_voice"),
                    )
                    config["tts_provider"] = resolved_tts.provider
                    config["tts_model"] = resolved_tts.model
                    config["tts_voice"] = resolved_tts.voice

                    # Resume existing session if provided
                    if message.get("session_id"):
                        existing = await session_manager.get_session(message["session_id"])
                        if existing and existing.user_id == user_id:
                            session_id = existing.session_id
                            session = existing

                    await websocket.send_json(
                        WSConfigAckMessage(
                            session_id=session_id,
                            stt_model=config["stt_model"],
                            tts_provider=config["tts_provider"],
                            tts_model=config["tts_model"],
                        ).model_dump()
                    )

                elif msg_type == WSMessageType.AUDIO.value:
                    # Accumulate audio data
                    audio_b64 = message.get("data", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_buffer.append(audio_bytes)

                    # Update state
                    await websocket.send_json(
                        WSStateChangeMessage(
                            state=VoiceAssistantState.LISTENING,
                        ).model_dump()
                    )

                elif msg_type == WSMessageType.TEXT.value:
                    # Text input (skip STT)
                    text = message.get("text", "").strip()
                    if text:
                        await _process_text_command(
                            websocket=websocket,
                            text=text,
                            user_id=user_id,
                            session_id=session_id,
                            config=config,
                            router_instance=router_instance,
                            db=db,
                        )

                elif msg_type == WSMessageType.COMMIT.value:
                    # Process accumulated audio
                    if audio_buffer:
                        await _process_audio_command(
                            websocket=websocket,
                            audio_buffer=audio_buffer,
                            user_id=user_id,
                            session_id=session_id,
                            config=config,
                            router_instance=router_instance,
                            db=db,
                        )
                        audio_buffer.clear()
                    else:
                        await websocket.send_json(
                            WSErrorMessage(
                                error="No audio data to process",
                                recoverable=True,
                            ).model_dump()
                        )

                elif msg_type == WSMessageType.CANCEL.value:
                    # Cancel current operation
                    audio_buffer.clear()
                    await session_manager.set_pending_intent(session_id, None)
                    await websocket.send_json(
                        WSStateChangeMessage(
                            state=VoiceAssistantState.IDLE,
                        ).model_dump()
                    )

                elif msg_type == WSMessageType.WORKFLOW_SUBSCRIBE.value:
                    # Subscribe to workflow progress updates
                    workflow_run_id = message.get("run_id")
                    if workflow_run_id:
                        await _stream_workflow_progress(
                            websocket=websocket,
                            run_id=workflow_run_id,
                            user_id=user_id,
                            router_instance=router_instance,
                        )

                elif msg_type == WSMessageType.WORKFLOW_CANCEL.value:
                    # Cancel a running workflow
                    workflow_run_id = message.get("run_id")
                    if workflow_run_id:
                        cancelled = await router_instance.cancel_workflow(
                            workflow_run_id, user_id
                        )
                        if cancelled:
                            await websocket.send_json(
                                WSWorkflowCompleteMessage(
                                    run_id=workflow_run_id,
                                    status="cancelled",
                                    response_text="Workflow cancelled.",
                                ).model_dump()
                            )
                        else:
                            await websocket.send_json(
                                WSErrorMessage(
                                    error="Could not cancel workflow",
                                    recoverable=True,
                                ).model_dump()
                            )

                else:
                    await websocket.send_json(
                        WSErrorMessage(
                            error=f"Unknown message type: {msg_type}",
                            recoverable=True,
                        ).model_dump()
                    )

            except json.JSONDecodeError:
                await websocket.send_json(
                    WSErrorMessage(
                        error="Invalid JSON message",
                        recoverable=True,
                    ).model_dump()
                )

    except WebSocketDisconnect:
        logger.info(f"Voice assistant WebSocket disconnected: session={session_id}")

    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, asyncio.TimeoutError) as e:
        logger.error("Voice assistant WebSocket error")
        try:
            await websocket.send_json(
                WSErrorMessage(
                    error=str(e),
                    recoverable=False,
                ).model_dump()
            )
            await websocket.close(code=1011)
        except (OSError, RuntimeError, TypeError, ValueError):
            pass

    finally:
        # Cleanup
        if transcriber:
            with contextlib.suppress(AttributeError, OSError, RuntimeError, TypeError, ValueError):
                await transcriber.finalize()


async def _process_audio_command(
    websocket: WebSocket,
    audio_buffer: list[bytes],
    user_id: int,
    session_id: str,
    config: dict[str, Any],
    router_instance: VoiceCommandRouter,
    db: Optional[CharactersRAGDB] = None,
) -> None:
    """Process audio buffer through STT -> Intent -> Action -> TTS pipeline."""
    await websocket.send_json(
        WSStateChangeMessage(state=VoiceAssistantState.PROCESSING).model_dump()
    )

    # Combine audio chunks
    audio_bytes = b"".join(audio_buffer)

    # Transcribe audio
    try:
        text = await _transcribe_audio(audio_bytes, config)

        if not text:
            await websocket.send_json(
                WSTranscriptionMessage(
                    text="",
                    is_final=True,
                    confidence=0.0,
                ).model_dump()
            )
            await websocket.send_json(
                WSErrorMessage(
                    error="Could not transcribe audio",
                    recoverable=True,
                ).model_dump()
            )
            await websocket.send_json(
                WSStateChangeMessage(state=VoiceAssistantState.IDLE).model_dump()
            )
            return

        await websocket.send_json(
            WSTranscriptionMessage(
                text=text,
                is_final=True,
            ).model_dump()
        )

    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
        logger.error("Transcription failed")
        await websocket.send_json(
            WSErrorMessage(
                error=f"Transcription failed: {e}",
                recoverable=True,
            ).model_dump()
        )
        await websocket.send_json(
            WSStateChangeMessage(state=VoiceAssistantState.IDLE).model_dump()
        )
        return

    # Process the transcribed text
    await _process_text_command(
        websocket=websocket,
        text=text,
        user_id=user_id,
        session_id=session_id,
        config=config,
        router_instance=router_instance,
        db=db,
    )


async def _transcribe_audio(
    audio_bytes: bytes,
    config: dict[str, Any],
) -> str:
    """Transcribe audio bytes to text."""
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import transcribe_audio

        # Convert bytes to numpy array (assuming PCM float32)
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: transcribe_audio(
                audio_np,
                sample_rate=config.get("sample_rate", 16000),
                model=config.get("stt_model", "parakeet"),
                language=config.get("stt_language"),
            ),
        )

        if isinstance(result, dict):
            return result.get("text", "")
        return str(result) if result else ""

    except Exception:
        logger.error("Audio transcription failed")
        raise


async def _process_text_command(
    websocket: WebSocket,
    text: str,
    user_id: int,
    session_id: str,
    config: dict[str, Any],
    router_instance: VoiceCommandRouter,
    db: Optional[CharactersRAGDB] = None,
) -> None:
    """Process text through Intent -> Action -> TTS pipeline."""
    # Process command
    result, _ = await router_instance.process_command(
        text=text,
        user_id=user_id,
        session_id=session_id,
        db=db,
    )

    # Send intent
    await websocket.send_json(
        WSIntentMessage(
            action_type=_action_type_to_voice(result.action_type),
            entities={},
            confidence=1.0,
            requires_confirmation=False,
        ).model_dump()
    )

    # Send action start
    await websocket.send_json(
        WSActionStartMessage(
            action_type=_action_type_to_voice(result.action_type),
        ).model_dump()
    )

    # Send action result
    await websocket.send_json(
        WSActionResultMessage(
            success=result.success,
            action_type=_action_type_to_voice(result.action_type),
            result_data=result.result_data,
            response_text=result.response_text,
            execution_time_ms=result.execution_time_ms,
        ).model_dump()
    )

    # Generate and stream TTS
    if result.response_text:
        await websocket.send_json(
            WSStateChangeMessage(state=VoiceAssistantState.SPEAKING).model_dump()
        )

        await _stream_tts_response(
            websocket=websocket,
            text=result.response_text,
            config=config,
        )

    # Return to idle
    await websocket.send_json(
        WSStateChangeMessage(state=VoiceAssistantState.IDLE).model_dump()
    )


async def _stream_tts_response(
    websocket: WebSocket,
    text: str,
    config: dict[str, Any],
) -> None:
    """Stream TTS audio to WebSocket."""
    try:
        from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
        from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2

        tts_service = await get_tts_service_v2()
        tts_format = config.get("tts_format", "mp3")
        resolved = resolve_tts_request_defaults(
            provider=config.get("tts_provider"),
            model=config.get("tts_model"),
            voice=config.get("tts_voice"),
        )

        request = OpenAISpeechRequest(
            model=resolved.model,
            input=text,
            voice=resolved.voice,
            response_format=tts_format,
            stream=True,
        )

        chunk_count = 0
        total_bytes = 0

        async for chunk in tts_service.generate_speech(
            request=request,
            provider=resolved.provider,
            fallback=True,
        ):
            if chunk:
                chunk_count += 1
                total_bytes += len(chunk)

                await websocket.send_json(
                    WSTTSChunkMessage(
                        data=base64.b64encode(chunk).decode("ascii"),
                        sequence=chunk_count,
                        format=tts_format,
                    ).model_dump()
                )

        await websocket.send_json(
            WSTTSEndMessage(
                total_chunks=chunk_count,
                total_bytes=total_bytes,
            ).model_dump()
        )

    except (TTSError, AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError) as e:
        logger.error("TTS streaming failed")
        await websocket.send_json(
            WSErrorMessage(
                error=f"TTS failed: {e}",
                recoverable=True,
            ).model_dump()
        )


async def _stream_workflow_progress(
    websocket: WebSocket,
    run_id: str,
    user_id: int,
    router_instance: VoiceCommandRouter,
) -> None:
    """Stream workflow progress events to WebSocket."""
    try:
        async for event in router_instance.stream_workflow_progress(
            run_id=run_id,
            user_id=user_id,
            poll_interval=0.5,
            timeout_seconds=300.0,
        ):
            if event.is_terminal:
                # Send completion message
                await websocket.send_json(
                    WSWorkflowCompleteMessage(
                        run_id=event.run_id,
                        status=event.data.get("status", event.event_type.replace("workflow_", "")),
                        outputs=event.data.get("outputs"),
                        error=event.data.get("error"),
                        duration_ms=event.data.get("duration_ms"),
                        response_text=event.message or "Workflow completed.",
                    ).model_dump()
                )
                break
            else:
                # Send progress update
                await websocket.send_json(
                    WSWorkflowProgressMessage(
                        run_id=event.run_id,
                        event_type=event.event_type,
                        message=event.message,
                        data=event.data,
                        timestamp=event.timestamp,
                    ).model_dump()
                )

    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, asyncio.TimeoutError) as e:
        logger.error("Workflow progress streaming failed")
        await websocket.send_json(
            WSErrorMessage(
                error=f"Workflow progress failed: {e}",
                recoverable=True,
            ).model_dump()
        )


#######################################################################################################################
#
# Voice Command Dry-Run
#

class VoiceCommandDryRunRequest(BaseModel):
    """Request payload for voice command dry-run."""
    phrase: str = Field(..., min_length=1, max_length=500)
    command_id: str | None = None


class VoiceCommandDryRunAlternative(BaseModel):
    action_type: str
    confidence: float | None = None
    raw_text: str | None = None


class VoiceCommandDryRunResponse(BaseModel):
    dry_run: bool = True
    phrase: str
    matched: bool
    match_method: str
    matched_phrase: str | None = None
    confidence: float | None = None
    action_type: str
    action_config: dict[str, Any] = {}
    processing_time_ms: float
    alternatives: list[VoiceCommandDryRunAlternative] = []


@router.post(
    "/voice/commands/dry-run",
    response_model=VoiceCommandDryRunResponse,
    summary="Dry-run a voice command phrase",
)
async def voice_command_dry_run(
    payload: VoiceCommandDryRunRequest,
    request: Request,
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandDryRunResponse:
    """Parse a phrase through the voice pipeline without executing.

    Returns the matched command, confidence score, extracted entities,
    and what action would be taken — without actually running it.
    """
    try:
        from tldw_Server_API.app.core.VoiceAssistant.intent_parser import IntentParser

        registry = get_voice_command_registry()
        registry.load_defaults()
        registry.refresh_user_commands(
            db,
            user_id=current_user.id,
            include_disabled=True,
        )
        parser = IntentParser(registry=registry)

        result = await parser.parse(
            text=payload.phrase,
            user_id=current_user.id,
            context=None,
        )

        intent = result.intent
        alternatives = []
        if hasattr(result, "alternatives") and result.alternatives:
            for alt in result.alternatives[:3]:
                alternatives.append(VoiceCommandDryRunAlternative(
                    action_type=alt.action_type.value if hasattr(alt.action_type, "value") else str(alt.action_type),
                    confidence=getattr(alt, "confidence", None),
                    raw_text=getattr(alt, "raw_text", None),
                ))

        return VoiceCommandDryRunResponse(
            phrase=payload.phrase,
            matched=result.match_method != "default",
            match_method=result.match_method,
            matched_phrase=getattr(result, "matched_phrase", None),
            confidence=getattr(intent, "confidence", None),
            action_type=intent.action_type.value if hasattr(intent.action_type, "value") else str(intent.action_type),
            action_config=intent.action_config,
            processing_time_ms=getattr(result, "processing_time_ms", 0.0),
            alternatives=alternatives,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="Voice assistant module not available",
        ) from exc
    except Exception as exc:
        logger.error("Voice command dry-run failed")
        raise HTTPException(
            status_code=500,
            detail="Dry-run failed",
        ) from exc


#
# End of voice_assistant.py
#######################################################################################################################
