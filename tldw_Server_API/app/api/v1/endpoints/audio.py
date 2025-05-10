# audio.py
# Description: This file contains the API endpoints for audio processing.
#
# Imports
import asyncio
import json
import os
from typing import AsyncGenerator # Add this import
#
# Third-party libraries
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.responses import StreamingResponse, Response # Add Response
from starlette import status # For status codes
#
# Local imports
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
# from your_project.services.tts_service import TTSService, get_tts_service

# For logging (if you use the same logger as in your PDF endpoint)
import logging # or from your_project.utils import logger
logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/v1/audio", # Standard OpenAI prefix
    tags=["TTS (OpenAI Compatible)"],
    responses={404: {"description": "Not found"}},
)



# --- Placeholder for TTSService and Mappings ---
# We will define these in subsequent steps.
# This is just to get the endpoint structure.

_openai_mappings = { # Load this from a JSON file later
    "models": {
        "tts-1": "openai_official_tts-1", # Maps to your backend identifier
        "tts-1-hd": "openai_official_tts-1-hd",
        "eleven_english_v1": "elevenlabs_english_v1",
        "kokoro_local": "local_kokoro_default_onnx"
    },
    "voices": { # This part is more complex and often backend-specific
        # OpenAI voices
        "alloy": "alloy", "echo": "echo", "fable": "fable",
        "onyx": "onyx", "nova": "nova", "shimmer": "shimmer",
        # ElevenLabs (IDs or names)
        "Rachel": "21m00Tcm4TlvDq8ikWAM", # Example ID
        # Kokoro local voices
        "bella": "af_bella", # Example mapping
    }
}

# Dummy TTSService for now
class DummyTTSService:
    async def generate_audio_stream(self, request: OpenAISpeechRequest, internal_model_id: str) -> AsyncGenerator[bytes, None]:
        logger.info(f"TTSService (dummy): Generating for model '{request.model}' (internal: {internal_model_id}), voice '{request.voice}'")
        logger.info(f"Input: {request.input[:50]}...")
        # Simulate audio chunks
        for i in range(3):
            await asyncio.sleep(0.1) # Simulate work
            yield f"audio_chunk_{i}_for_{request.input[:10]}.{request.response_format}".encode()
        # Simulate closing/finalizing stream if needed by some formats
        if request.response_format in ["wav", "mp3"] and not request.stream: # Non-streamed needs a full file
            yield b"--final_boundary_for_non_streamed--" # Placeholder

async def get_tts_service() -> DummyTTSService: # Later, this will return the real TTSService
    return DummyTTSService()

# --- End of Placeholder ---


@router.post("/speech", summary="Generates audio from text input.")
async def create_speech(
    request_data: OpenAISpeechRequest, # FastAPI will parse JSON body into this
    client_request: Request, # To check for client disconnects
    tts_service: DummyTTSService = Depends(get_tts_service),
    # FIXME - add check for user
):
    """
    Generates audio from the input text.
    """
    logger.info(f"Received speech request: model={request_data.model}, voice={request_data.voice}, format={request_data.response_format}")

    # 1. Map OpenAI model name to your internal backend identifier
    internal_model_id = _openai_mappings["models"].get(request_data.model)
    if not internal_model_id:
        logger.warning(f"Unsupported model requested: {request_data.model}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_model",
                "message": f"The model `{request_data.model}` does not exist or you do not have access to it.",
                "type": "invalid_request_error",
            },
        )

    # 2. Basic voice validation (can be enhanced per backend later)
    # For now, we assume the voice name is passed as is to the backend.
    # More sophisticated mapping might be needed if OpenAI voice names
    # differ significantly from how your backends expect them.
    # internal_voice_id = _openai_mappings["voices"].get(request_data.voice, request_data.voice)
    # request_data.voice = internal_voice_id # Update request_data if you map voices

    # 3. Determine Content-Type
    content_type_map = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/L16; rate=24000; channels=1", # Example for raw PCM
    }
    content_type = content_type_map.get(request_data.response_format)
    if not content_type:
        logger.warning(f"Unsupported response format: {request_data.response_format}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported response_format: {request_data.response_format}. Supported formats are: {', '.join(content_type_map.keys())}",
        )

    # 4. Streaming Logic (simplified from target app)
    async def audio_chunk_generator():
        try:
            async for audio_chunk_bytes in tts_service.generate_audio_stream(request_data, internal_model_id):
                if await client_request.is_disconnected():
                    logger.info("Client disconnected, stopping audio generation.")
                    break
                yield audio_chunk_bytes
        except HTTPException: # Re-raise HTTPExceptions directly
            raise
        except Exception as e:
            logger.error(f"Error during audio streaming: {e}", exc_info=True)
            # Important: Don't yield anything here if an error occurs,
            # let FastAPI handle the error response.
            # For a production system, you might want to yield a specific error chunk
            # if the protocol requires it, but for simple streaming, just raising is often enough.
            # If you raise HTTPException here, it should be caught by FastAPI.
            # If you raise a standard Python exception, it will result in a 500.
            # Consider how to signal errors in the stream if the client expects it.
            # For now, we'll let it become a 500 or be handled by the `tts_service` itself.
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


    if request_data.stream:
        return StreamingResponse(
            audio_chunk_generator(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request_data.response_format}",
                "X-Accel-Buffering": "no", # Useful for Nginx
                "Cache-Control": "no-cache",
            },
        )
    else:
        # Non-streaming: Collect all chunks and send as a single response
        all_audio_bytes = b""
        try:
            async for chunk in audio_chunk_generator():
                all_audio_bytes += chunk
            # Remove any "final boundary" placeholder if it was added by the dummy service
            all_audio_bytes = all_audio_bytes.replace(b"--final_boundary_for_non_streamed--", b"")

            if not all_audio_bytes: # Handle case where generation yielded nothing
                 logger.error("Non-streaming generation resulted in empty audio data.")
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Audio generation failed to produce data.")

            return Response(
                content=all_audio_bytes,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request_data.response_format}",
                    "Cache-Control": "no-cache",
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during non-streaming audio generation: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Add other OpenAI compatible endpoints like /models, /voices later
# For now, this is the core.

#
# End of audio.py
#######################################################################################################################
