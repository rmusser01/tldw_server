# tts_generation.py
# Description: This module handles the text-to-speech (TTS) generation process.
#
# Imports
from typing import AsyncGenerator, Optional, Dict, Any
#
# Third-party Imports
import asyncio # For semaphore
#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS.tts_backends import TTSBackendManager, TTSBackendBase
#
#######################################################################################################################
#
# Functions:


# FIXME _ placeholder
# Potentially borrow from target app for text processing and audio writing
# from target_app_path.services.text_processing import smart_split
# from target_app_path.services.audio import AudioService
# from target_app_path.services.streaming_audio_writer import StreamingAudioWriter

# For logging
import logging
logger = logging.getLogger(__name__)

class TTSService:
    # Limit concurrent calls to a single backend's generate method if needed
    _backend_semaphore = asyncio.Semaphore(4) # Example: 4 concurrent generations per backend instance

    def __init__(self, backend_manager: TTSBackendManager):
        self.backend_manager = backend_manager
        # If you adopt the target app's text/audio processing:
        # self.smart_split = smart_split_function_from_target_app
        # self.audio_service = AudioService_class_from_target_app
        # self.streaming_audio_writer_class = StreamingAudioWriter_class_from_target_app

    async def generate_audio_stream(
        self, request: OpenAISpeechRequest, internal_model_id: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Orchestrates fetching the backend and calling its stream generation.
        Handles text processing (chunking, normalization) if the backend expects it.
        Handles audio format conversion if the backend produces raw audio.
        """
        backend: Optional[TTSBackendBase] = await self.backend_manager.get_backend(internal_model_id)
        if not backend:
            logger.error(f"TTSService: No backend found for internal_model_id: {internal_model_id}")
            # This case should ideally be caught by the router's mapping check,
            # but as a safeguard:
            yield f"ERROR: Backend for model '{request.model}' not configured.".encode()
            return

        logger.info(f"TTSService: Using backend {type(backend).__name__} for model '{request.model}' (internal: {internal_model_id})")

        # --- Sophisticated Text Processing & Audio Conversion (Ideal, like target app) ---
        # IF you adopt the target app's text processing and audio writers:
        #
        # 1. Initialize StreamingAudioWriter for the target format (only if backend produces raw PCM)
        #    saw = self.streaming_audio_writer_class(format=request.response_format, sample_rate=24000) # Adjust SR
        #
        # 2. Process text using smart_split (if backend doesn't handle long texts or needs phonemes)
        #    async for text_chunk, phoneme_tokens in self.smart_split(
        #        request.input,
        #        lang_code=request.lang_code, # if you add lang_code to OpenAISpeechRequest
        #        normalization_options=request.normalization_options # if you add this
        #    ):
        #        prepared_input_for_backend = text_chunk # or phoneme_tokens if backend needs them
        #
        #        async with self._backend_semaphore: # Control concurrency
        #            # Backend yields raw audio (e.g., numpy array)
        #            raw_audio_chunks_from_backend = backend.generate_raw_audio_stream(prepared_input_for_backend, request)
        #
        #            async for raw_audio_np in raw_audio_chunks_from_backend:
        #                # Convert raw audio to target format using AudioService/StreamingAudioWriter
        #                # This is a conceptual adaptation of target app's _process_chunk
        #                # audio_output_bytes = await self.audio_service.convert_audio(
        #                #     AudioChunk(audio=raw_audio_np), # Adapt to your AudioChunk or just pass np array
        #                #     request.response_format,
        #                #     saw, # The streaming audio writer instance
        #                #     is_last_chunk=False # Manage this flag
        #                # )
        #                # yield audio_output_bytes
        #
        # 3. Finalize the stream with the StreamingAudioWriter
        #    final_bytes = await self.audio_service.convert_audio(..., is_last_chunk=True)
        #    if final_bytes: yield final_bytes
        #
        # --- Simpler Approach (Backends handle their own formatting or you simplify) ---
        # If backends are expected to yield bytes in the correct request.response_format directly:
        try:
            async with self._backend_semaphore:
                 async for audio_bytes_chunk in backend.generate_speech_stream(request):
                    yield audio_bytes_chunk
        except Exception as e:
            logger.error(f"TTSService: Error streaming from backend {type(backend).__name__}: {e}", exc_info=True)
            # Decide how to propagate: re-raise, or yield an error marker if the protocol supports it.
            # Raising here will likely lead to the StreamingResponse stopping and client getting an error.
            raise # Re-raise to be caught by the main endpoint handler

# --- Singleton pattern for TTSService and its manager ---
_tts_service_instance: Optional[TTSService] = None
_tts_backend_manager_instance: Optional[TTSBackendManager] = None
_init_lock = asyncio.Lock()

async def get_tts_service(app_config: Optional[Dict[str, Any]] = None) -> TTSService: # app_config for initialization
    global _tts_service_instance, _tts_backend_manager_instance
    if not _tts_service_instance:
        async with _init_lock:
            if not _tts_service_instance:
                if app_config is None:
                    # Load your default app config here if not provided
                    # from your_project.config import APP_SETTINGS_LOADED
                    # app_config = APP_SETTINGS_LOADED
                    # For now, let's assume it must be passed on first call
                    raise ValueError("TTSService requires app_config on first initialization.")

                if not _tts_backend_manager_instance:
                    _tts_backend_manager_instance = TTSBackendManager(app_config=app_config)
                _tts_service_instance = TTSService(backend_manager=_tts_backend_manager_instance)
                logger.info("TTSService initialized.")
    return _tts_service_instance

async def close_tts_resources():
    """Call this during application shutdown (e.g., FastAPI lifespan event)."""
    global _tts_backend_manager_instance, _tts_service_instance
    async with _init_lock:
        if _tts_backend_manager_instance:
            logger.info("Closing TTS backend resources...")
            await _tts_backend_manager_instance.close_all_backends()
            _tts_backend_manager_instance = None
            _tts_service_instance = None
            logger.info("TTS backend resources closed.")

#
# End of tts_generation.py
#######################################################################################################################
