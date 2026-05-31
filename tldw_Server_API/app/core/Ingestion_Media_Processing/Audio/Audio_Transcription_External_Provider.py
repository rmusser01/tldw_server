# Audio_Transcription_External_Provider.py
#########################################
# External Transcription Provider Support
# Allows forwarding transcription requests to external OpenAI-compatible Audio APIs
#
####################
# Function List
#
# 1. transcribe_with_external_provider() - Main transcription function
# 2. validate_external_provider_config() - Validate provider configuration
# 3. format_external_request() - Format request for external API
# 4. parse_external_response() - Parse response from external API
# 5. ExternalProviderConfig - Configuration dataclass
#
####################

import asyncio
import contextlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urljoin, urlparse

import numpy as np
import soundfile as sf
from loguru import logger

from tldw_Server_API.app.core.exceptions import NetworkError, RetryExhaustedError
from tldw_Server_API.app.core.http_client import RetryPolicy, afetch
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import open_safe_local_path

logger = logger


@dataclass
class ExternalProviderConfig:
    """Configuration for external transcription provider."""
    base_url: str
    api_key: Optional[str] = None
    model: str = "whisper-1"
    timeout: float = 300.0  # 5 minutes default
    max_retries: int = 3
    verify_ssl: bool = True
    custom_headers: Optional[dict[str, str]] = None
    response_format: str = "json"
    temperature: float = 0.0
    language: Optional[str] = None
    prompt: Optional[str] = None


# Global cache for provider configurations
_provider_configs: dict[str, ExternalProviderConfig] = {}
EXTERNAL_PROVIDER_RUNTIME_EXCEPTIONS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)


async def _close_response(resp: Any) -> None:
    if resp is None:
        return
    close = getattr(resp, "aclose", None)
    if callable(close):
        await close()
        return
    close = getattr(resp, "close", None)
    if callable(close):
        close()


def load_external_provider_config(provider_name: str = "default") -> Optional[ExternalProviderConfig]:
    """
    Load external provider configuration from config file or environment.

    Args:
        provider_name: Name of the provider configuration to load

    Returns:
        ExternalProviderConfig or None if not configured
    """
    if provider_name in _provider_configs:
        return _provider_configs[provider_name]

    # Try to load from environment variables
    env_prefix = f"EXTERNAL_TRANSCRIPTION_{provider_name.upper()}_"

    # Initialize external_config
    external_config = {}

    base_url = os.getenv(f"{env_prefix}BASE_URL")
    if not base_url:
        # Try loading from config file
        try:
            from tldw_Server_API.app.core.Config_Management.config_utils import load_comprehensive_config
            config_data = load_comprehensive_config()

            external_config = config_data.get('STT-Settings', {}).get('external_providers', {}).get(provider_name, {})
            if external_config:
                base_url = external_config.get('base_url')
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            logger.debug(f"Could not load config for external provider {provider_name}: {e}")
            return None

    if not base_url:
        return None

    config = ExternalProviderConfig(
        base_url=base_url,
        api_key=os.getenv(f"{env_prefix}API_KEY") or external_config.get('api_key'),
        model=os.getenv(f"{env_prefix}MODEL") or external_config.get('model', 'whisper-1'),
        timeout=float(os.getenv(f"{env_prefix}TIMEOUT", "300")) or external_config.get('timeout', 300),
        max_retries=int(os.getenv(f"{env_prefix}MAX_RETRIES", "3")) or external_config.get('max_retries', 3),
        verify_ssl=os.getenv(f"{env_prefix}VERIFY_SSL", "true").lower() == "true",
        response_format=os.getenv(f"{env_prefix}RESPONSE_FORMAT") or external_config.get('response_format', 'json'),
        temperature=float(os.getenv(f"{env_prefix}TEMPERATURE", "0")) or external_config.get('temperature', 0),
        language=os.getenv(f"{env_prefix}LANGUAGE") or external_config.get('language'),
        prompt=os.getenv(f"{env_prefix}PROMPT") or external_config.get('prompt')
    )

    # Cache the configuration
    _provider_configs[provider_name] = config

    return config


def validate_external_provider_config(config: ExternalProviderConfig) -> tuple[bool, Optional[str]]:
    """
    Validate external provider configuration.

    Args:
        config: Provider configuration to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate base URL
    try:
        parsed = urlparse(config.base_url)
        if not parsed.scheme or not parsed.netloc:
            return False, "Invalid base URL format"
    except (ValueError, TypeError, AttributeError) as e:
        return False, f"Invalid base URL: {e}"

    # Validate timeout
    if config.timeout <= 0:
        return False, "Timeout must be positive"

    # Validate temperature
    if not 0 <= config.temperature <= 2:
        return False, "Temperature must be between 0 and 2"

    # Validate response format
    valid_formats = ['json', 'text', 'srt', 'verbose_json', 'vtt']
    if config.response_format not in valid_formats:
        return False, f"Invalid response format. Must be one of: {valid_formats}"

    return True, None


async def transcribe_with_external_provider_async(
    audio_data: Union[np.ndarray, str, Path],
    sample_rate: int = 16000,
    provider_name: str = "default",
    config: Optional[ExternalProviderConfig] = None,
    base_dir: Optional[Path] = None,
    **kwargs
) -> str:
    """
    Asynchronously transcribe audio using an external OpenAI-compatible API.

    Args:
        audio_data: Audio data as numpy array or path to audio file
        sample_rate: Sample rate of the audio
        provider_name: Name of the provider configuration to use
        config: Optional ExternalProviderConfig to use instead of loading from config
        base_dir: Optional base directory to enforce when reading local file paths
        **kwargs: Additional parameters to pass to the API

    Returns:
        Transcribed text or error message.

    Notes:
        - ``config.base_url`` is interpreted as either:
          * a bare API base such as ``https://api.openai.com`` (the standard
            ``/v1/audio/transcriptions`` path will be appended), or
          * a full audio transcription endpoint such as
            ``https://api.example.com/v1/audio/transcriptions`` (used as-is).
        - Intermediate paths (for example a proxy prefix without the final
          ``/v1/audio/transcriptions`` suffix) are not automatically supported;
          in those cases, configure ``base_url`` with the full endpoint URL.
    """
    # Load configuration
    if config is None:
        config = load_external_provider_config(provider_name)
        if config is None:
            return f"[Error: External provider '{provider_name}' not configured]"

    # Validate configuration
    is_valid, error_msg = validate_external_provider_config(config)
    if not is_valid:
        logger.warning(f"Invalid external provider configuration: {error_msg}")
        return "[Error: Invalid external provider configuration]"

    # Prepare audio file/stream
    file_handle = None

    try:
        if isinstance(audio_data, (str, Path)):
            audio_path = Path(audio_data)
            base_dir_for_open = base_dir or audio_path.parent
            file_handle = open_safe_local_path(audio_path, base_dir_for_open, mode="rb")
            if file_handle is None:
                return "[Error: Audio file path rejected outside allowed base directory]"
        else:
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format="WAV")
            buffer.seek(0)
            file_handle = buffer

        # Prepare the request endpoint:
        # - If base_url already points at the full audio endpoint
        #   (.../v1/audio/transcriptions), use it as-is.
        # - Otherwise, treat base_url as an API root and append the standard
        #   OpenAI-compatible audio transcription path.
        parsed = urlparse(config.base_url)
        if parsed.path.rstrip("/").endswith("/v1/audio/transcriptions"):
            endpoint = config.base_url
        else:
            endpoint = urljoin(config.base_url.rstrip("/") + "/", "v1/audio/transcriptions")

        # Prepare headers
        headers = {}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        if config.custom_headers:
            headers.update(config.custom_headers)

        # Prepare form data
        if file_handle is None:
            return "[Error: No audio input provided]"

        with file_handle:
            files = {
                'file': ('audio.wav', file_handle, 'audio/wav')
            }

            data = {
                'model': config.model,
                'response_format': config.response_format,
                'temperature': str(config.temperature)
            }

            # Add optional parameters
            if config.language:
                data['language'] = config.language
            if config.prompt:
                data['prompt'] = config.prompt

            # Add any additional kwargs
            for key, value in kwargs.items():
                if key not in data:
                    data[key] = str(value)

            retry_policy = RetryPolicy(
                attempts=max(1, int(config.max_retries)),
                retry_on_status=(429,),
                retry_on_unsafe=True,
            )

            resp = None
            try:
                with contextlib.suppress(OSError, ValueError):
                    file_handle.seek(0)

                resp = await afetch(
                    method="POST",
                    url=endpoint,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=config.timeout,
                    allow_redirects=False,
                    retry=retry_policy,
                    verify=config.verify_ssl,
                )

                status_code = int(resp.status_code)
                if status_code == 200:
                    # Parse response based on format
                    if config.response_format == 'text':
                        return resp.text
                    if config.response_format in ['json', 'verbose_json']:
                        result = resp.json()
                        return result.get('text', '')
                    if config.response_format in ['srt', 'vtt']:
                        return resp.text
                    return resp.text

                if status_code == 429:
                    return f"[Error: Rate limit exceeded after {config.max_retries} attempts]"

                logger.warning(f"External transcription provider returned status {status_code}")
                return f"[Error: API returned {status_code}]"

            except RetryExhaustedError as e:
                logger.warning(f"External provider retries exhausted: {e}")
                return "[Error: External transcription request failed]"
            except NetworkError as e:
                logger.warning(f"External provider network error: {e}")
                return "[Error: External transcription request failed]"
            except EXTERNAL_PROVIDER_RUNTIME_EXCEPTIONS as e:
                logger.exception(f"External provider request failed: {e}")
                return "[Error: External transcription request failed]"
            finally:
                await _close_response(resp)

        return "[Error: Failed to transcribe after all retries]"

    except EXTERNAL_PROVIDER_RUNTIME_EXCEPTIONS as e:
        logger.exception(f"Error in external provider transcription: {e}")
        return "[Error: External transcription failed]"

    finally:
        pass


def transcribe_with_external_provider(
    audio_data: Union[np.ndarray, str, Path],
    sample_rate: int = 16000,
    provider_name: str = "default",
    config: Optional[ExternalProviderConfig] = None,
    base_dir: Optional[Path] = None,
    **kwargs
) -> str:
    """
    Synchronous wrapper for external provider transcription.

    Args:
        audio_data: Audio data as numpy array or path to audio file
        sample_rate: Sample rate of the audio
        provider_name: Name of the provider configuration to use
        config: Optional ExternalProviderConfig to use instead of loading from config
        base_dir: Optional base directory to enforce when reading local file paths
        **kwargs: Additional parameters to pass to the API

    Returns:
        Transcribed text or error message
    """
    try:
        # Run async function in sync context, handling various loop states robustly
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            # We are in a running loop (e.g., notebook, async context). Use a worker thread
            # to run a fresh event loop and avoid cross-loop issues.
            import concurrent.futures
            def _run_in_fresh_loop():
                return asyncio.run(
                    transcribe_with_external_provider_async(
                        audio_data,
                        sample_rate,
                        provider_name,
                        config,
                        base_dir=base_dir,
                        **kwargs,
                    )
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_in_fresh_loop)
                return future.result()
        else:
            # Normal case - no loop running in this thread
            return asyncio.run(
                transcribe_with_external_provider_async(
                    audio_data,
                    sample_rate,
                    provider_name,
                    config,
                    base_dir=base_dir,
                    **kwargs,
                )
            )
    except EXTERNAL_PROVIDER_RUNTIME_EXCEPTIONS as e:
        logger.exception(f"Error in external provider transcription: {e}")
        return "[Error: External transcription failed]"


def add_external_provider(name: str, config: ExternalProviderConfig) -> bool:
    """
    Add or update an external provider configuration.

    Args:
        name: Name of the provider
        config: Provider configuration

    Returns:
        True if successful
    """
    is_valid, error_msg = validate_external_provider_config(config)
    if not is_valid:
        logger.error(f"Invalid configuration for provider {name}: {error_msg}")
        return False

    _provider_configs[name] = config
    logger.info(f"Added external provider: {name}")
    return True


def list_external_providers() -> list[str]:
    """
    List configured external providers.

    Returns:
        List of provider names
    """
    return list(_provider_configs.keys())


def remove_external_provider(name: str) -> bool:
    """
    Remove an external provider configuration.

    Args:
        name: Name of the provider to remove

    Returns:
        True if removed, False if not found
    """
    if name in _provider_configs:
        del _provider_configs[name]
        logger.info(f"Removed external provider: {name}")
        return True
    return False


async def test_external_provider(
    provider_name: str = "default",
    config: Optional[ExternalProviderConfig] = None
) -> dict[str, Any]:
    """
    Test an external provider with a simple audio sample.

    Args:
        provider_name: Name of the provider to test
        config: Optional configuration to test

    Returns:
        Dictionary with test results
    """
    import time

    # Generate test audio (1 second of sine wave)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(440 * 2 * np.pi * t).astype(np.float32) * 0.5

    start_time = time.time()

    try:
        result = await transcribe_with_external_provider_async(
            audio,
            sample_rate,
            provider_name,
            config
        )

        elapsed = time.time() - start_time

        return {
            'success': not result.startswith('[Error:'),
            'result': result,
            'elapsed_time': elapsed,
            'provider': provider_name
        }

    except EXTERNAL_PROVIDER_RUNTIME_EXCEPTIONS:
        logger.warning("External provider test failed")
        return {
            'success': False,
            'error': "External provider test failed",
            'elapsed_time': time.time() - start_time,
            'provider': provider_name
        }


# Integration with main transcription library
def register_external_provider_with_library():
    """
    Register external provider support with the main transcription library.

    This should be called during application initialization.
    """
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
            register_transcription_provider,
        )

        def external_provider_wrapper(audio_data, sample_rate=16000, **kwargs):
            provider_name = kwargs.get('provider_name', 'default')
            return transcribe_with_external_provider(
                audio_data,
                sample_rate,
                provider_name=provider_name,
                **kwargs
            )

        register_transcription_provider('external', external_provider_wrapper)
        logger.info("External provider support registered with transcription library")

    except ImportError:
        logger.warning("Could not register external provider with transcription library")


#######################################################################################################################
# End of Audio_Transcription_External_Provider.py
#######################################################################################################################
