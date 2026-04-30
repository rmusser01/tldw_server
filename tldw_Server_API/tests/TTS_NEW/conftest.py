"""
TTS Module Test Configuration and Fixtures

Provides fixtures for testing TTS functionality including audio generation,
adapter management, and provider switching.
"""

import os
import io
import tempfile
import wave
import json
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional
from unittest.mock import MagicMock, AsyncMock, Mock
from datetime import datetime
import uuid

import pytest
import numpy as np
from fastapi.testclient import TestClient
from tldw_Server_API.app.api.v1.endpoints import audio as audio_endpoints

# TTS integration tests hit /api/v1/audio routes. Opt this suite into mounting
# audio and audio-jobs routers even when pytest uses the minimal test app.
os.environ.setdefault("MINIMAL_TEST_INCLUDE_AUDIO", "1")
os.environ.setdefault("MINIMAL_TEST_INCLUDE_AUDIO_JOBS", "1")

# Import actual TTS components for integration tests
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    VoiceSettings
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSError,
    TTSProviderNotConfiguredError,
    TTSGenerationError,
    TTSRateLimitError
)

# =====================================================================
# Test Markers
# =====================================================================

def pytest_configure(config):

    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests with minimal mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real components")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1 second")
    config.addinivalue_line("markers", "requires_api_key: Tests requiring provider API keys")
    config.addinivalue_line("markers", "streaming: Tests for streaming audio")
    config.addinivalue_line("markers", "adapter: Adapter-specific tests")

# =====================================================================
# Environment Configuration
# =====================================================================

@pytest.fixture
def test_env_vars(monkeypatch):
    """Set up test environment variables without polluting global state."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("TTS_DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("TTS_DEFAULT_MODEL", "tts-1")
    monkeypatch.setenv("TTS_DEFAULT_VOICE", "alloy")
    # Ensure TTS providers considered configured in tests
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test-123")
    yield

# =====================================================================
# Audio Generation Fixtures
# =====================================================================

@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Generate sample audio bytes for testing."""
    # Create a simple sine wave audio
    sample_rate = 24000
    duration = 1.0  # 1 second
    frequency = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = np.sin(2 * np.pi * frequency * t)

    # Convert to 16-bit PCM
    samples = (samples * 32767).astype(np.int16)

    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())

    return buffer.getvalue()

@pytest.fixture
def sample_mp3_bytes() -> bytes:
    """Generate sample MP3 bytes (mock)."""
    # For testing, return a minimal MP3 header
    # Real MP3 generation would require an encoder
    return b'ID3\x04\x00\x00\x00\x00\x00\x00' + b'\x00' * 1024

@pytest.fixture
def streaming_audio_generator():
    """Factory for streaming audio chunks."""

    class _StreamingFactory:
        def __init__(self, default_total: int = 4096, default_chunk: int = 1024) -> None:
            self._default_total = default_total
            self._default_chunk = default_chunk

        def __call__(self, *, total_size: Optional[int] = None, chunk_size: Optional[int] = None):
            total = total_size if total_size is not None else self._default_total
            chunk = chunk_size if chunk_size is not None else self._default_chunk

            async def _generator():
                for _ in range(0, total, chunk):
                    yield os.urandom(chunk)

            return _generator()

        def __aiter__(self):
            return self().__call__()

    return _StreamingFactory()

# =====================================================================
# TTS Request/Response Fixtures
# =====================================================================

@pytest.fixture
def basic_tts_request() -> TTSRequest:
    """Basic TTS request object."""
    return TTSRequest(
        text="Hello, this is a test.",
        voice="alloy",
        model="tts-1",
        speed=1.0,
        format=AudioFormat.MP3
    )

@pytest.fixture
def advanced_tts_request() -> TTSRequest:
    """Advanced TTS request with all settings."""
    return TTSRequest(
        text="This is a more complex test with various settings.",
        voice="nova",
        model="tts-1-hd",
        speed=1.2,
        format=AudioFormat.WAV,
        voice_settings=VoiceSettings(
            pitch=1.1,
            rate=1.0,
            volume=0.9
        ),
        language="en-US"
    )

@pytest.fixture
def long_text_request() -> TTSRequest:
    """TTS request with long text for chunking tests."""
    long_text = " ".join([f"Sentence number {i}." for i in range(100)])
    return TTSRequest(
        text=long_text,
        voice="echo",
        model="tts-1",
        speed=1.0,
        format=AudioFormat.MP3
    )

@pytest.fixture
def mock_tts_response(sample_audio_bytes) -> TTSResponse:
    """Mock TTS response object."""
    return TTSResponse(
        audio_content=sample_audio_bytes,
        format=AudioFormat.WAV,
        sample_rate=24000,
        duration=1.0,
        provider="openai",
        model="tts-1",
        metadata={
            "characters": 24,
            "request_id": str(uuid.uuid4())
        }
    )

# =====================================================================
# Provider Configuration Fixtures
# =====================================================================

@pytest.fixture
def provider_configs():
    """Configuration for different TTS providers."""
    return {
        "openai": {
            "api_key": "test-openai-key",
            "models": ["tts-1", "tts-1-hd"],
            "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            "max_characters": 4096
        },
        "elevenlabs": {
            "api_key": "test-elevenlabs-key",
            "models": ["eleven_monolingual_v1", "eleven_multilingual_v2"],
            "voices": ["rachel", "clyde", "paul"],
            "max_characters": 5000
        },
        "kokoro": {
            "api_key": "test-kokoro-key",
            "models": ["kokoro-v1"],
            "voices": ["sarah", "john"],
            "max_characters": 3000
        }
    }

@pytest.fixture
def openai_config():
    """OpenAI TTS provider configuration."""
    return {
        "api_key": "test-openai-key",
        "base_url": "https://api.openai.com/v1",
        "models": {
            "tts-1": {"max_chars": 4096, "voices": ["alloy", "echo", "fable"]},
            "tts-1-hd": {"max_chars": 4096, "voices": ["alloy", "echo", "fable"]}
        }
    }

# =====================================================================
# Mock Adapter Fixtures
# =====================================================================

@pytest.fixture
def mock_openai_adapter():
    """Mock OpenAI TTS adapter."""
    adapter = MagicMock()
    adapter.provider = "openai"
    adapter.is_available = True
    adapter.supported_models = ["tts-1", "tts-1-hd"]
    adapter.supported_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    async def mock_generate(request):
        return TTSResponse(
            audio_content=b"mock_audio_data",
            format=request.format,
            sample_rate=24000,
            duration=1.0,
            provider="openai",
            model=request.model
        )

    adapter.generate = AsyncMock(side_effect=mock_generate)
    adapter.validate_request = AsyncMock(return_value=True)

    return adapter

@pytest.fixture
def mock_adapter_factory():
    """Mock adapter factory."""
    factory = MagicMock()

    adapters = {}

    def get_adapter(provider):
        if provider not in adapters:
            adapter = MagicMock()
            adapter.provider = provider
            adapter.is_available = True
            adapter.generate = AsyncMock(return_value=MagicMock(audio_content=b"mock_audio"))
            adapters[provider] = adapter
        return adapters[provider]

    factory.get_adapter = Mock(side_effect=get_adapter)
    factory.list_available_providers = Mock(return_value=["openai", "elevenlabs"])
    factory.is_provider_configured = Mock(return_value=True)

    return factory

# =====================================================================
# Service Fixtures
# =====================================================================

import pytest_asyncio
@pytest_asyncio.fixture
async def tts_service(mock_adapter_factory):
    """Create a TTS service instance with mocked adapters."""
    # Pass mock factory into constructor to match current TTSServiceV2 API
    service = TTSServiceV2(factory=mock_adapter_factory)
    # For unit tests that consume generate_speech() and expect errors to be
    # surfaced as "ERROR:" chunks instead of exceptions, enable the
    # stream_errors_as_audio mode explicitly on the fixture instance. Tests
    # that need strict HTTP-style error propagation construct their own
    # TTSServiceV2 instances with the appropriate config.
    service._stream_errors_as_audio = True
    # Keep backward-compat for tests that reference _factory explicitly
    service._factory = mock_adapter_factory
    yield service
    await service.shutdown()

@pytest_asyncio.fixture
async def real_tts_service():
    """Create a real TTS service for integration tests."""
    # Use a minimal mock factory to satisfy constructor requirements in tests
    minimal_factory = MagicMock()
    minimal_factory.get_adapter = Mock(side_effect=lambda prov: MagicMock(generate=AsyncMock(return_value=TTSResponse(audio_content=b"mock_audio"))))
    minimal_factory.registry = MagicMock()
    service = TTSServiceV2(factory=minimal_factory)
    yield service
    await service.shutdown()

# =====================================================================
# Circuit Breaker Fixtures
# =====================================================================

@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker for testing."""
    breaker = MagicMock()
    breaker.is_open = False
    breaker.call = AsyncMock(side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))
    breaker.record_success = Mock()
    breaker.record_failure = Mock()
    breaker.reset = Mock()

    return breaker

# =====================================================================
# Resource Manager Fixtures
# =====================================================================

@pytest.fixture
def mock_resource_manager():
    """Mock resource manager for testing."""
    manager = MagicMock()
    manager.check_resources = AsyncMock(return_value=True)
    manager.allocate_resources = AsyncMock(return_value=True)
    manager.release_resources = AsyncMock()
    manager.get_memory_usage = Mock(return_value={"used": 1024, "available": 4096})
    manager.get_gpu_usage = Mock(return_value={"used": 0, "available": 0})

    return manager

# =====================================================================
# Validation Fixtures
# =====================================================================

@pytest.fixture
def valid_requests():
    """Collection of valid TTS requests."""
    return [
        {"text": "Simple text", "voice": "alloy"},
        {"text": "Text with settings", "voice": "nova", "speed": 1.2},
        {"text": "Different format", "voice": "echo", "format": "wav"},
        {"text": "HD model", "voice": "fable", "model": "tts-1-hd"}
    ]

@pytest.fixture
def invalid_requests():
    """Collection of invalid TTS requests."""
    return [
        {},  # Missing required fields
        {"text": ""},  # Empty text
        {"text": "a" * 5000},  # Text too long
        {"text": "Test", "voice": "invalid_voice"},  # Invalid voice
        {"text": "Test", "speed": 5.0},  # Speed out of range
        {"text": "Test", "format": "invalid_format"}  # Invalid format
    ]

# =====================================================================
# API Client Fixtures
# =====================================================================

@pytest.fixture
def test_client(test_env_vars, bypass_api_limits):
    """Create a test client for the FastAPI app with auth override."""
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

    async def _override_user():
        return User(id=1, username="tester", email="t@example.com", is_active=True)

    app.dependency_overrides[get_request_user] = _override_user
    try:
        with bypass_api_limits(app), TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_request_user, None)

@pytest.fixture
def auth_headers():
    """Authentication headers for API requests."""
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()
    return {
        "X-API-KEY": settings.SINGLE_USER_API_KEY,
        "Content-Type": "application/json"
    }

# =====================================================================
# Error Fixtures
# =====================================================================

@pytest.fixture
def tts_errors():
    """Collection of TTS errors for testing."""
    return {
        "rate_limit": TTSRateLimitError("Rate limit exceeded", retry_after=60),
        "generation": TTSGenerationError("Failed to generate audio"),
        "provider_not_configured": TTSProviderNotConfiguredError("Provider not configured"),
        "general": TTSError("General TTS error")
    }

# =====================================================================
# Test Audio Files
# =====================================================================

@pytest.fixture
def test_audio_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test audio files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_dir = Path(temp_dir)
        yield audio_dir

@pytest.fixture
def test_audio_file(test_audio_dir, sample_audio_bytes) -> Path:
    """Create a test audio file."""
    audio_path = test_audio_dir / "test_audio.wav"
    audio_path.write_bytes(sample_audio_bytes)
    return audio_path

# =====================================================================
# Cleanup Fixtures
# =====================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Any cleanup code here
    import gc
    gc.collect()
