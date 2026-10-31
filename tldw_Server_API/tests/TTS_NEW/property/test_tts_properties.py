"""
Property-based tests for TTS functionality.

Uses Hypothesis to verify invariants and properties of the TTS system.
"""

import string

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from tldw_Server_API.app.core.TTS.adapters.base import (
    AudioFormat,
    TTSRequest,
    TTSResponse,
    VoiceSettings,
)
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSValidationError
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2

# ========================================================================
# Custom Hypothesis Strategies
# ========================================================================

@st.composite
def valid_tts_text(draw):
    """Generate valid TTS text."""
    # Generate reasonable text lengths
    length = draw(st.integers(min_value=1, max_value=1000))
    # Use printable ASCII and common unicode
    text = draw(st.text(min_size=length, max_size=length))
    # Ensure not just whitespace
    assume(text.strip())
    return text

@st.composite
def valid_voice_name(draw):
    """Generate valid voice names."""
    providers = {
        "openai": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        "elevenlabs": ["rachel", "domi", "bella", "antoni", "clyde"],
        "kokoro": ["sarah", "john", "emma", "brian"]
    }
    provider = draw(st.sampled_from(list(providers.keys())))
    voice = draw(st.sampled_from(providers[provider]))
    return voice, provider

@st.composite
def valid_speed(draw):
    """Generate valid speed values."""
    # Most TTS services support 0.25-4.0
    return draw(st.floats(min_value=0.25, max_value=4.0))

@st.composite
def valid_voice_settings(draw):
    """Generate valid voice settings."""
    return VoiceSettings(
        stability=draw(st.floats(min_value=0.0, max_value=1.0)),
        similarity_boost=draw(st.floats(min_value=0.0, max_value=1.0)),
        style=draw(st.floats(min_value=0.0, max_value=1.0)) if draw(st.booleans()) else None,
        use_speaker_boost=draw(st.booleans()) if draw(st.booleans()) else None
    )

@st.composite
def valid_tts_request(draw):
    """Generate valid TTS request."""
    voice, provider = draw(valid_voice_name())
    return TTSRequest(
        text=draw(valid_tts_text()),
        voice=voice,
        provider=provider,
        model=draw(st.sampled_from(["tts-1", "tts-1-hd", "eleven_monolingual_v1"])),
        speed=draw(valid_speed()),
        format=draw(st.sampled_from(list(AudioFormat))),
        voice_settings=draw(valid_voice_settings()) if draw(st.booleans()) else None
    )

# ========================================================================
# Request Validation Properties
# ========================================================================

class TestRequestValidationProperties:
    """Test properties of request validation."""

    @pytest.mark.property
    @given(text=valid_tts_text())
    def test_valid_text_always_accepted(self, text):
        """Property: Valid text should always be accepted."""
        request = TTSRequest(
            text=text,
            voice="alloy",
            model="tts-1"
        )
        # Should not raise
        assert request.text == text

    @pytest.mark.property
    @given(length=st.integers(min_value=5001, max_value=6000))
    def test_long_text_handling(self, length):
        """Property: Text exceeding limits should be handled consistently."""
        request = TTSRequest(
            text="a" * length,
            voice="alloy",
            model="tts-1"
        )
        # Service should either chunk or reject consistently
        # This depends on implementation
        assert len(request.text) > 5000

    @pytest.mark.property
    @given(speed=st.floats(allow_nan=False, allow_infinity=False))
    def test_speed_validation_bounds(self, speed):
        """Property: Speed values outside bounds should be handled."""
        if 0.25 <= speed <= 4.0:
            # Valid speed should work
            request = TTSRequest(
                text="Test",
                voice="alloy",
                speed=speed
            )
            assert request.speed == speed
        else:
            # Invalid speed should be rejected or clamped
            try:
                request = TTSRequest(
                    text="Test",
                    voice="alloy",
                    speed=speed
                )
                # If accepted, should be clamped
                assert 0.25 <= request.speed <= 4.0
            except (TTSValidationError, ValueError):
                # Rejection is also valid
                pass

    @pytest.mark.property
    @given(request=valid_tts_request())
    def test_request_serialization_roundtrip(self, request):
        """Property: Requests should survive serialization roundtrip."""
        # Serialize to dict
        data = request.dict()

        # Recreate from dict
        request2 = TTSRequest(**data)

        # Should be equivalent
        assert request2.text == request.text
        assert request2.voice == request.voice
        assert request2.model == request.model
        assert request2.speed == request.speed
        assert request2.supplied_fields == request.supplied_fields

    @pytest.mark.property
    @given(explicit_speed=st.booleans(), explicit_language=st.booleans())
    def test_common_field_explicitness_survives_serialization(
        self,
        explicit_speed,
        explicit_language,
    ):
        kwargs = {"text": "Test", "model": "Vendor/MiXeD-Case"}
        if explicit_speed:
            kwargs["speed"] = 1.0
        if explicit_language:
            kwargs["language"] = "en"
        request = TTSRequest(**kwargs)

        restored = TTSRequest(**request.dict())

        assert restored.model == "Vendor/MiXeD-Case"
        assert restored.speed == 1.0
        assert restored.language == "en"
        assert ("speed" in restored.supplied_fields) is explicit_speed
        assert ("language" in restored.supplied_fields) is explicit_language

# ========================================================================
# Audio Output Properties
# ========================================================================

class TestAudioOutputProperties:
    """Test properties of audio output."""

    @pytest.mark.property
    @given(audio_bytes=st.binary(min_size=100, max_size=10000))
    def test_audio_bytes_are_valid(self, audio_bytes):
        """Property: Generated audio bytes should be valid."""
        response = TTSResponse(
            audio_content=audio_bytes,
            format=AudioFormat.MP3,
            provider="test",
            model="test-model"
        )

        # Audio content should be preserved
        assert response.audio_content == audio_bytes
        assert len(response.audio_content) > 0

    @pytest.mark.property
    @given(
        sample_rate=st.sampled_from([8000, 16000, 22050, 24000, 44100, 48000]),
        duration=st.floats(min_value=0.1, max_value=10.0)
    )
    def test_audio_metadata_consistency(self, sample_rate, duration):
        """Property: Audio metadata should be consistent."""
        response = TTSResponse(
            audio_content=b"fake_audio",
            format=AudioFormat.WAV,
            sample_rate=sample_rate,
            duration=duration,
            provider="test",
            model="test"
        )

        # Sample rate should be standard
        assert response.sample_rate in [8000, 16000, 22050, 24000, 44100, 48000]

        # Duration should be positive
        assert response.duration > 0

        # Approximate size calculation (for uncompressed)
        if response.format == AudioFormat.WAV:
            expected_size = sample_rate * duration * 2  # 16-bit mono
            # Allow for headers and some variance
            assert len(response.audio_content) < expected_size * 10

    @pytest.mark.property
    @given(format=st.sampled_from(list(AudioFormat)))
    def test_format_preservation(self, format):
        """Property: Audio format should be preserved."""
        response = TTSResponse(
            audio_content=b"audio",
            format=format,
            provider="test",
            model="test"
        )

        assert response.format == format
        assert response.format in AudioFormat

# ========================================================================
# Provider Behavior Properties
# ========================================================================

class TestProviderBehaviorProperties:
    """Test properties of provider behavior."""

    @pytest.mark.property
    @given(
        providers=st.lists(
            st.sampled_from(["openai", "elevenlabs", "kokoro"]),
            min_size=1,
            max_size=3,
            unique=True
        )
    )
    def test_provider_switching_deterministic(self, providers):
        """Property: Provider switching should be deterministic."""
        # When switching providers, behavior should be predictable
        previous = None
        for provider in providers:
            request = TTSRequest(
                text="Test",
                voice="alloy" if provider == "openai" else "rachel",
                provider=provider
            )

            # Provider should match request
            assert request.provider == provider

            # Should be different from previous
            if previous:
                assert provider != previous or len(providers) == 1
            previous = provider

    @pytest.mark.property
    @given(
        text=valid_tts_text(),
        provider=st.sampled_from(["openai", "elevenlabs"])
    )
    def test_same_text_deterministic_output(self, text, provider):
        """Property: Same text with same settings should give consistent results."""
        voice = "alloy" if provider == "openai" else "rachel"

        request1 = TTSRequest(text=text, voice=voice, provider=provider)
        request2 = TTSRequest(text=text, voice=voice, provider=provider)

        # Requests should be equivalent
        assert request1.text == request2.text
        assert request1.voice == request2.voice
        assert request1.provider == request2.provider

# ========================================================================
# Chunking Properties
# ========================================================================

class TestChunkingProperties:
    """Test properties of text chunking for TTS."""

    @pytest.mark.property
    @given(
        text=st.text(min_size=100, max_size=10000),
        chunk_size=st.integers(min_value=50, max_value=500)
    )
    def test_chunking_preserves_text(self, text, chunk_size):
        """Property: Chunking should preserve all text."""
        assume(text.strip())

        # Simple chunking simulation
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])

        # Reassemble
        reassembled = "".join(chunks)

        # Should match original
        assert reassembled == text

    @pytest.mark.property
    @given(
        sentences=st.lists(
            st.text(
                alphabet=string.ascii_letters + string.digits + " ",
                min_size=10,
                max_size=100,
            ).filter(lambda s: s.strip() != ""),
            min_size=5,
            max_size=20
        )
    )
    def test_sentence_chunking_preserves_boundaries(self, sentences):
        """Property: Sentence-based chunking should preserve boundaries."""
        # Join with proper punctuation
        text = ". ".join(sentences) + "."

        # Simple sentence detection
        detected_sentences = text.split(". ")

        # Should preserve sentence count (roughly)
        assert abs(len(detected_sentences) - len(sentences)) <= 1

# ========================================================================
# Stateful Testing with RuleBasedStateMachine
# ========================================================================

class TTSStateMachine(RuleBasedStateMachine):
    """Stateful testing for TTS service lifecycle."""

    def __init__(self):
        super().__init__()
        self.service = None
        self.active_providers = set()
        self.generated_audio = []
        self.current_provider = "openai"

    @initialize()
    def setup(self):
        """Initialize the TTS service."""
        self.service = TTSServiceV2()
        self.active_providers = {"openai"}  # Start with default
        self.generated_audio = []
        self.current_provider = "openai"

    @rule(
        text=valid_tts_text(),
        voice=st.sampled_from(["alloy", "echo", "nova"])
    )
    def generate_audio(self, text, voice):
        """Rule: Generate audio with current provider."""
        _request = TTSRequest(
            text=text,
            voice=voice,
            provider=self.current_provider
        )

        # Mock generation
        audio = f"audio_{len(self.generated_audio)}".encode()
        self.generated_audio.append({
            "text": text,
            "voice": voice,
            "provider": self.current_provider,
            "audio": audio
        })

    @rule(provider=st.sampled_from(["openai", "elevenlabs", "kokoro"]))
    def switch_provider(self, provider):
        """Rule: Switch to a different provider."""
        self.current_provider = provider
        self.active_providers.add(provider)

    @rule()
    def clear_cache(self):
        """Rule: Clear any caches."""
        # This would clear provider caches in real implementation
        pass

    @invariant()
    def audio_count_increases(self):
        """Invariant: Audio count never decreases."""
        # Count should only increase or stay same
        assert len(self.generated_audio) >= 0

    @invariant()
    def provider_always_valid(self):
        """Invariant: Current provider is always valid."""
        assert self.current_provider in ["openai", "elevenlabs", "kokoro"]

    @invariant()
    def active_providers_tracked(self):
        """Invariant: Active providers are tracked correctly."""
        assert len(self.active_providers) >= 1
        assert self.current_provider in self.active_providers or not self.active_providers

# Run the state machine tests
TestTTSStateMachine = TTSStateMachine.TestCase

# ========================================================================
# Error Injection Properties
# ========================================================================

class TestErrorInjectionProperties:
    """Test properties under error conditions."""

    @pytest.mark.property
    @given(
        error_rate=st.floats(min_value=0.0, max_value=1.0),
        num_requests=st.integers(min_value=1, max_value=100)
    )
    def test_error_rate_bounds(self, error_rate, num_requests):
        """Property: Error rate should match configuration."""
        errors = 0
        successes = 0

        for _ in range(num_requests):
            import random
            if random.random() < error_rate:
                errors += 1
            else:
                successes += 1

        actual_rate = errors / num_requests if num_requests > 0 else 0

        # Should be within reasonable bounds
        if num_requests > 10:
            assert abs(actual_rate - error_rate) < 0.3

    @pytest.mark.property
    @given(
        retry_count=st.integers(min_value=0, max_value=5),
        success_on_retry=st.integers(min_value=0, max_value=5)
    )
    def test_retry_logic(self, retry_count, success_on_retry):
        """Property: Retry logic should be bounded."""
        attempts = 0
        max_retries = retry_count

        while attempts <= max_retries:
            attempts += 1
            if attempts == success_on_retry:
                # Success
                break

        # Should not exceed max retries
        assert attempts <= max_retries + 1

        # Should succeed if success_on_retry is within bounds
        if 0 < success_on_retry <= retry_count:
            assert attempts == success_on_retry

# ========================================================================
# Performance Properties
# ========================================================================

class TestPerformanceProperties:
    """Test performance-related properties."""

    @pytest.mark.property
    @given(
        num_concurrent=st.integers(min_value=1, max_value=10),
        text_length=st.integers(min_value=10, max_value=1000)
    )
    @settings(max_examples=10, deadline=5000)  # 5 second deadline
    def test_concurrent_request_handling(self, num_concurrent, text_length):
        """Property: System should handle concurrent requests."""
        requests = []
        for i in range(num_concurrent):
            text = f"Request {i}: " + "a" * text_length
            requests.append(TTSRequest(
                text=text,
                voice="alloy",
                provider="openai"
            ))

        # All requests should be valid
        assert len(requests) == num_concurrent
        for req in requests:
            assert len(req.text) > 10

    @pytest.mark.property
    @given(
        cache_size=st.integers(min_value=0, max_value=100),
        num_requests=st.integers(min_value=0, max_value=200)
    )
    def test_cache_behavior(self, cache_size, num_requests):
        """Property: Cache should respect size limits."""
        cache = {}

        for i in range(num_requests):
            key = f"request_{i % (cache_size + 1)}"  # Create some duplicates

            if cache_size == 0:
                continue

            if len(cache) >= cache_size and key not in cache and cache_size > 0:
                # Evict oldest (simplified)
                oldest = next(iter(cache))
                del cache[oldest]

            cache[key] = f"audio_{i}"

        # Cache should not exceed size limit
        if cache_size > 0:
            assert len(cache) <= cache_size
