# test_chatterbox_adapter_mock.py
# Description: Mock/Unit tests for Chatterbox TTS adapter
#
# Imports
import pytest
pytestmark = pytest.mark.unit
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters import chatterbox_adapter as chatterbox_mod
from tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter import ChatterboxAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    ProviderStatus
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSModelNotFoundError,
    TTSModelLoadError
)
from tldw_Server_API.app.core.TTS.chatterbox_catalog import (
    CHATTERBOX_LANGUAGE_CODES,
    CHATTERBOX_MODEL_PROVIDER_ALIASES,
    ChatterboxModelFamily,
    resolve_chatterbox_model_family,
)
#
#######################################################################################################################
#
# Mock Tests for Chatterbox Adapter

class _LogCapture:
    def __init__(self, level: str = "ERROR"):
        self.messages: list[str] = []
        self._level = level
        self._sink_id: int | None = None

    def __enter__(self):
        self._sink_id = chatterbox_mod.logger.add(
            lambda message: self.messages.append(message.record["message"]),
            level=self._level,
        )
        return self.messages

    def __exit__(self, exc_type, exc, tb):
        if self._sink_id is not None:
            chatterbox_mod.logger.remove(self._sink_id)


@pytest.mark.asyncio
class TestChatterboxAdapterMock:
    """Mock/Unit tests for Chatterbox adapter"""

    async def test_initialization_configuration(self):
        """Test initialization with configuration"""
        adapter = ChatterboxAdapter({
            "chatterbox_model": "large-v2",
            "chatterbox_api_key": "test-key",
            "chatterbox_model_path": "./models/chatterbox",
            "chatterbox_device": "cuda"
        })

        assert adapter.config.get("chatterbox_model") == "large-v2"
        assert adapter.config.get("chatterbox_api_key") == "test-key"
        assert adapter.config.get("chatterbox_model_path") == "./models/chatterbox"
        assert adapter.device in {"cpu", "cuda", "mps"}

    async def test_initialization_honors_target_latency_configuration(self):
        """Target latency config should control progressive stream chunking."""
        adapter = ChatterboxAdapter({"target_latency_ms": 350})
        prefixed_adapter = ChatterboxAdapter({
            "target_latency_ms": 350,
            "chatterbox_target_latency_ms": 125,
        })
        invalid_adapter = ChatterboxAdapter({"chatterbox_target_latency_ms": "not-an-int"})

        assert adapter.target_latency_ms == 350
        assert prefixed_adapter.target_latency_ms == 125
        assert invalid_adapter.target_latency_ms == 200

    async def test_initialization_honors_bf16_configuration_and_env(self, monkeypatch):
        """BF16 should stay off by default and honor config/env opt-ins."""
        monkeypatch.delenv("TTS_BF16", raising=False)
        default_adapter = ChatterboxAdapter({"chatterbox_device": "cpu"})

        monkeypatch.setenv("TTS_BF16", "auto")
        env_adapter = ChatterboxAdapter({"chatterbox_device": "cpu"})
        generic_adapter = ChatterboxAdapter({"chatterbox_device": "cpu", "use_bf16": True})
        prefixed_adapter = ChatterboxAdapter({
            "chatterbox_device": "cpu",
            "use_bf16": True,
            "chatterbox_use_bf16": "off",
        })
        invalid_adapter = ChatterboxAdapter({"chatterbox_device": "cpu", "chatterbox_use_bf16": "maybe"})

        assert default_adapter.bf16_mode == "off"
        assert env_adapter.bf16_mode == "auto"
        assert generic_adapter.bf16_mode == "on"
        assert prefixed_adapter.bf16_mode == "off"
        assert invalid_adapter.bf16_mode == "off"

    async def test_capabilities_reporting(self):
        """Test capabilities are correctly reported"""
        adapter = ChatterboxAdapter({})
        caps = await adapter.get_capabilities()

        assert caps.provider_name == "Chatterbox"
        assert caps.supports_streaming is True
        assert caps.supports_voice_cloning is True
        assert caps.supports_emotion_control is True
        assert caps.supports_speech_rate is False
        assert caps.max_text_length == 10000
        assert caps.sample_rate == 24000
        assert AudioFormat.WAV in caps.supported_formats
        assert AudioFormat.MP3 in caps.supported_formats
        assert AudioFormat.OPUS in caps.supported_formats

    async def test_capabilities_expose_family_metadata(self):
        """Capabilities should let clients discover Chatterbox family-specific support."""
        adapter = ChatterboxAdapter({})
        caps = await adapter.get_capabilities()

        model_families = caps.metadata["model_families"]
        assert model_families["standard"]["languages"] == ["en"]
        assert model_families["standard"]["model_ids"] == ["chatterbox", "chatterbox-emotion"]
        assert model_families["multilingual"]["languages"] == sorted(CHATTERBOX_LANGUAGE_CODES)
        assert model_families["multilingual"]["model_ids"] == ["chatterbox-multilingual"]
        assert model_families["turbo"]["languages"] == ["en"]
        assert model_families["turbo"]["supports_paralinguistic_tags"] is True
        assert caps.metadata["voice_conversion"]["endpoint"] == "/api/v1/audio/voice-conversion"
        assert caps.metadata["voice_conversion"]["model_id"] == "chatterbox-vc"
        generation_parameters = caps.metadata["generation_parameters"]
        assert generation_parameters["standard"] == [
            "exaggeration",
            "cfg_weight",
            "temperature",
            "repetition_penalty",
            "min_p",
            "top_p",
            "top_k",
            "seed",
            "speed_factor",
        ]
        assert generation_parameters["multilingual"] == generation_parameters["standard"]
        assert generation_parameters["turbo"] == [
            "temperature",
            "repetition_penalty",
            "top_p",
            "top_k",
            "norm_loudness",
            "speed_factor",
        ]
        assert caps.metadata["speed_factor"] == {
            "request_fields": ["extra_params.speed_factor", "speed"],
            "requires_runtime_support": True,
        }
        assert caps.metadata["chunking"] == {
            "request_fields": ["extra_params.split_text", "extra_params.chunk_size"],
            "service_modes": ["non_streaming"],
        }
        assert caps.metadata["bf16"] == {
            "config_keys": ["chatterbox_use_bf16", "use_bf16"],
            "environment_variable": "TTS_BF16",
            "modes": ["off", "on", "auto"],
            "default": "off",
        }
        assert caps.supports_speech_rate is False

    async def test_character_voice_presets(self):
        """Test character voice presets"""
        adapter = ChatterboxAdapter({})

        # Check character voices exist
        assert "narrator" in adapter.CHARACTER_VOICES
        assert "hero" in adapter.CHARACTER_VOICES
        assert "villain" in adapter.CHARACTER_VOICES
        assert "sidekick" in adapter.CHARACTER_VOICES
        assert "sage" in adapter.CHARACTER_VOICES
        assert "comic_relief" in adapter.CHARACTER_VOICES

    async def test_voice_mapping(self):
        """Test voice mapping functionality"""
        adapter = ChatterboxAdapter({})

        # Test character voice mapping
        assert adapter.map_voice("narrator") == "narrator"
        assert adapter.map_voice("hero") == "hero"
        assert adapter.map_voice("villain") == "villain"

        # Test generic mappings
        assert adapter.map_voice("default") == "narrator"
        assert adapter.map_voice("assistant") == "sidekick"
        assert adapter.map_voice("evil") == "villain"
        assert adapter.map_voice("wise") == "sage"
        assert adapter.map_voice("funny") == "comic_relief"

    async def test_style_parameters(self):
        """Test speech style parameters"""
        adapter = ChatterboxAdapter({})

        request = TTSRequest(
            text="Dramatic speech",
            voice="narrator",
            style="dramatic",
            extra_params={
                "emphasis_level": 0.8,
                "tone": "serious",
                "pacing": "slow"
            }
        )

        assert request.style == "dramatic"
        assert request.extra_params.get("emphasis_level") == 0.8
        assert request.extra_params.get("tone") == "serious"
        assert request.extra_params.get("pacing") == "slow"

    async def test_device_selection(self):
        """Test device selection for inference"""
        # Test CUDA selection
        with patch('tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter._torch_cuda_available', return_value=True):
            adapter = ChatterboxAdapter({"chatterbox_device": "cuda"})
            assert adapter.device == "cuda"

        # Test CPU fallback
        with patch('tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter._torch_cuda_available', return_value=False):
            adapter = ChatterboxAdapter({"chatterbox_device": "cuda"})
            assert adapter.device == "cpu"

    async def test_model_not_installed_error(self):
        """Test error when Chatterbox library not installed"""
        adapter = ChatterboxAdapter({})

        # Mock import error for chatterbox-tts
        with patch('builtins.__import__', side_effect=ImportError("chatterbox-tts not found")):
            with patch(
                'tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter._get_torch',
                return_value=MagicMock(),
            ):
                success = await adapter.initialize()
                assert not success
                assert adapter._status == ProviderStatus.ERROR

    async def test_generation_without_initialization(self):
        """Test generation fails without initialization"""
        adapter = ChatterboxAdapter({})

        request = TTSRequest(
            text="Test",
            voice="narrator",
            format=AudioFormat.WAV
        )

        with patch.object(adapter, "ensure_initialized", new=AsyncMock(return_value=False)):
            with pytest.raises(Exception):  # Should raise provider not configured
                await adapter.generate(request)

    async def test_character_dialogue_preparation(self):
        """Test preparation of character dialogue"""
        adapter = ChatterboxAdapter({})

        # Test dialogue with character voices
        text = "Narrator: Once upon a time... Hero: I must save the day! Villain: You'll never stop me!"
        dialogues = adapter.parse_dialogue(text)

        assert len(dialogues) == 3
        assert dialogues[0] == ("Narrator", "Once upon a time...")
        assert dialogues[1] == ("Hero", "I must save the day!")
        assert dialogues[2] == ("Villain", "You'll never stop me!")

    async def test_emotion_intensity_mapping(self):
        """Test emotion intensity mapping"""
        adapter = ChatterboxAdapter({})

        # Test different emotion intensities
        emotions = [
            ("happy", 0.5, "slightly happy"),
            ("sad", 1.0, "moderately sad"),
            ("angry", 2.0, "very angry"),
            ("excited", 0.3, "mildly excited")
        ]

        for emotion, intensity, expected in emotions:
            request = TTSRequest(
                text="Test",
                emotion=emotion,
                emotion_intensity=intensity
            )

            # Verify emotion parameters
            assert request.emotion == emotion
            assert request.emotion_intensity == intensity

    async def test_cleanup_on_close(self):
        """Test resource cleanup on close"""
        adapter = ChatterboxAdapter({"chatterbox_device": "cuda"})

        # Mock resources
        adapter.model = MagicMock()
        adapter.vocoder = MagicMock()
        adapter._initialized = True
        adapter._status = ProviderStatus.AVAILABLE

        fake_torch = MagicMock()
        with patch('tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter._torch_cuda_available', return_value=True):
            with patch('tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter._get_torch', return_value=fake_torch):
                await adapter.close()

                assert adapter.model is None
                assert adapter.vocoder is None
                assert adapter._initialized is False
                assert adapter._status == ProviderStatus.DISABLED
                fake_torch.cuda.empty_cache.assert_called_once()

    async def test_model_variant_selection(self):
        """Test model variant selection"""
        # Test small model
        adapter_small = ChatterboxAdapter({"chatterbox_model": "small"})
        assert adapter_small.config.get("chatterbox_model") == "small"

        # Test medium model
        adapter_medium = ChatterboxAdapter({"chatterbox_model": "medium"})
        assert adapter_medium.config.get("chatterbox_model") == "medium"

        # Test large model
        adapter_large = ChatterboxAdapter({"chatterbox_model": "large-v2"})
        assert adapter_large.config.get("chatterbox_model") == "large-v2"

    async def test_speech_rate_control(self):
        """Test speech rate control"""
        adapter = ChatterboxAdapter({})

        speeds = [0.5, 1.0, 1.5, 2.0]

        for speed in speeds:
            request = TTSRequest(
                text="Speed test",
                voice="narrator",
                speed=speed
            )

            assert request.speed == speed

    async def test_catalog_exposes_upstream_family_aliases(self):
        """Chatterbox model aliases should cover original, emotion, multilingual, and turbo."""
        assert CHATTERBOX_MODEL_PROVIDER_ALIASES["chatterbox"] == "chatterbox"
        assert CHATTERBOX_MODEL_PROVIDER_ALIASES["chatterbox-emotion"] == "chatterbox"
        assert CHATTERBOX_MODEL_PROVIDER_ALIASES["chatterbox-multilingual"] == "chatterbox"
        assert CHATTERBOX_MODEL_PROVIDER_ALIASES["chatterbox-turbo"] == "chatterbox"
        assert "fr" in CHATTERBOX_LANGUAGE_CODES
        assert "zh" in CHATTERBOX_LANGUAGE_CODES

    async def test_registry_routes_chatterbox_family_aliases_to_adapter_provider(self):
        """Registry model lookup should treat all Chatterbox family IDs as Chatterbox."""
        from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterFactory, TTSProvider

        assert TTSAdapterFactory.MODEL_PROVIDER_MAP["chatterbox"] is TTSProvider.CHATTERBOX
        assert TTSAdapterFactory.MODEL_PROVIDER_MAP["chatterbox-emotion"] is TTSProvider.CHATTERBOX
        assert TTSAdapterFactory.MODEL_PROVIDER_MAP["chatterbox-multilingual"] is TTSProvider.CHATTERBOX
        assert TTSAdapterFactory.MODEL_PROVIDER_MAP["chatterbox-turbo"] is TTSProvider.CHATTERBOX

    async def test_model_family_resolution_uses_model_language_and_config(self):
        """Explicit family IDs and legacy multilingual config should resolve the runtime family."""
        assert resolve_chatterbox_model_family("chatterbox-turbo") is ChatterboxModelFamily.TURBO
        assert resolve_chatterbox_model_family("turbo") is ChatterboxModelFamily.TURBO
        assert resolve_chatterbox_model_family("chatterbox-multilingual") is ChatterboxModelFamily.MULTILINGUAL
        assert resolve_chatterbox_model_family("chatterbox", language="fr") is ChatterboxModelFamily.STANDARD
        assert (
            resolve_chatterbox_model_family("chatterbox", language="fr", use_multilingual=True)
            is ChatterboxModelFamily.MULTILINGUAL
        )
        assert (
            resolve_chatterbox_model_family(
                "chatterbox",
                language="en",
                config_variant="multilingual",
            )
            is ChatterboxModelFamily.MULTILINGUAL
        )

    async def test_turbo_model_selection_loads_turbo_runtime(self):
        """Turbo requests should load the upstream Turbo runtime import path."""
        adapter = ChatterboxAdapter({})
        fake_turbo = MagicMock()
        fake_turbo.sr = 24000
        fake_turbo.watermarker = MagicMock()
        fake_cls = MagicMock()
        fake_cls.from_pretrained.return_value = fake_turbo

        with patch.dict("sys.modules", {"chatterbox.tts_turbo": MagicMock(ChatterboxTurboTTS=fake_cls)}):
            model = await adapter._get_model("en", family=ChatterboxModelFamily.TURBO)

        assert model is fake_turbo
        fake_cls.from_pretrained.assert_called_once_with(device=adapter.device)
        assert adapter.model_turbo is fake_turbo

    async def test_standard_model_loads_from_local_configured_path(self, tmp_path):
        """A local Chatterbox model path should use upstream from_local()."""
        model_path = tmp_path / "chatterbox"
        model_path.mkdir()
        adapter = ChatterboxAdapter({"chatterbox_model_path": str(model_path)})
        fake_model = MagicMock()
        fake_model.sr = 24000
        fake_model.watermarker = MagicMock()
        fake_cls = MagicMock()
        fake_cls.from_local.return_value = fake_model

        with patch.dict("sys.modules", {"chatterbox.tts": MagicMock(ChatterboxTTS=fake_cls)}):
            model = await adapter._get_model("en", family=ChatterboxModelFamily.STANDARD)

        assert model is fake_model
        fake_cls.from_local.assert_called_once_with(str(model_path), device=adapter.device)
        fake_cls.from_pretrained.assert_not_called()

    async def test_standard_model_repo_id_stays_on_pretrained_loader(self):
        """Repo-style configured model paths should preserve from_pretrained behavior."""
        adapter = ChatterboxAdapter({"chatterbox_model_path": "ResembleAI/chatterbox"})
        fake_model = MagicMock()
        fake_model.sr = 24000
        fake_model.watermarker = MagicMock()
        fake_cls = MagicMock()
        fake_cls.from_pretrained.return_value = fake_model

        with patch.dict("sys.modules", {"chatterbox.tts": MagicMock(ChatterboxTTS=fake_cls)}):
            model = await adapter._get_model("en", family=ChatterboxModelFamily.STANDARD)

        assert model is fake_model
        fake_cls.from_pretrained.assert_called_once_with(device=adapter.device)
        fake_cls.from_local.assert_not_called()

    async def test_voice_conversion_model_selection_loads_vc_runtime(self):
        """Voice conversion should lazy-load the upstream ChatterboxVC runtime."""
        adapter = ChatterboxAdapter({})
        fake_vc = MagicMock()
        fake_vc.sr = 24000
        fake_vc.watermarker = MagicMock()
        fake_cls = MagicMock()
        fake_cls.from_pretrained.return_value = fake_vc

        with patch.dict("sys.modules", {"chatterbox.vc": MagicMock(ChatterboxVC=fake_cls)}):
            model = await adapter._get_vc_model()

        assert model is fake_vc
        fake_cls.from_pretrained.assert_called_once_with(device=adapter.device)
        assert adapter.model_vc is fake_vc

    async def test_voice_conversion_model_loads_from_local_configured_path(self, tmp_path):
        """A local Chatterbox VC model path should use upstream from_local()."""
        model_path = tmp_path / "chatterbox-vc"
        model_path.mkdir()
        adapter = ChatterboxAdapter({"chatterbox_vc_model_path": str(model_path)})
        fake_vc = MagicMock()
        fake_vc.sr = 24000
        fake_vc.watermarker = MagicMock()
        fake_cls = MagicMock()
        fake_cls.from_local.return_value = fake_vc

        with patch.dict("sys.modules", {"chatterbox.vc": MagicMock(ChatterboxVC=fake_cls)}):
            model = await adapter._get_vc_model()

        assert model is fake_vc
        fake_cls.from_local.assert_called_once_with(str(model_path), device=adapter.device)
        fake_cls.from_pretrained.assert_not_called()

    async def test_voice_conversion_uses_upstream_generate_signature(self):
        """VC generation should call ChatterboxVC.generate with source and target paths."""
        adapter = ChatterboxAdapter({"target_latency_ms": 125})
        fake_vc = MagicMock()
        fake_vc.sr = 24000
        fake_waveform = object()
        fake_vc.generate.return_value = fake_waveform
        seen: dict[str, object] = {}

        async def fake_stream(waveform, format, sample_rate, channels=1, chunk_duration_sec=0.2):
            seen["waveform"] = waveform
            seen["format"] = format
            seen["sample_rate"] = sample_rate
            seen["channels"] = channels
            seen["chunk_duration_sec"] = chunk_duration_sec
            yield b"encoded-vc"

        with patch.object(adapter, "_get_vc_model", new=AsyncMock(return_value=fake_vc)):
            with patch(
                "tldw_Server_API.app.core.TTS.waveform_streamer.stream_encoded_waveform",
                new=fake_stream,
            ):
                response = await adapter.convert_voice(
                    source_audio_path="/tmp/source.wav",
                    target_voice_path="/tmp/target.wav",
                    format=AudioFormat.WAV,
                    stream=False,
                )

        fake_vc.generate.assert_called_once_with(
            audio="/tmp/source.wav",
            target_voice_path="/tmp/target.wav",
        )
        assert seen == {
            "waveform": fake_waveform,
            "format": "wav",
            "sample_rate": 24000,
            "channels": 1,
            "chunk_duration_sec": 0.125,
        }
        assert response.audio_data == b"encoded-vc"
        assert response.metadata == {
            "mode": "voice_conversion",
            "model_family": "voice_conversion",
            "target_voice_path_provided": True,
            "watermarked": False,
        }

    async def test_generation_kwargs_forward_seed_and_omit_unsupported_none_values(self):
        """Generation kwargs should include reproducibility seed only when requested."""
        adapter = ChatterboxAdapter({})
        request = TTSRequest(
            text="Seeded generation",
            format=AudioFormat.WAV,
            seed=1234,
            extra_params={"cfg_weight": 0.3, "temperature": 0.7, "top_k": 250},
        )

        kwargs = adapter._build_generation_kwargs(
            request,
            voice_reference_path=None,
            exaggeration=0.6,
            family=ChatterboxModelFamily.STANDARD,
        )

        assert kwargs["seed"] == 1234
        assert kwargs["cfg_weight"] == 0.3
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_k"] == 250
        assert "audio_prompt_path" not in kwargs

    async def test_generation_kwargs_forward_speed_factor_when_requested(self):
        """Chatterbox should pass speed_factor only when callers request non-default speed."""
        adapter = ChatterboxAdapter({})
        default_request = TTSRequest(text="Default speed", format=AudioFormat.WAV)
        speed_request = TTSRequest(text="Faster generation", speed=1.35, format=AudioFormat.WAV)
        explicit_request = TTSRequest(
            text="Explicit speed factor",
            speed=1.1,
            format=AudioFormat.WAV,
            extra_params={"speed_factor": 0.85},
        )

        default_kwargs = adapter._build_generation_kwargs(
            default_request,
            voice_reference_path=None,
            exaggeration=0.5,
            family=ChatterboxModelFamily.STANDARD,
        )
        speed_kwargs = adapter._build_generation_kwargs(
            speed_request,
            voice_reference_path=None,
            exaggeration=0.5,
            family=ChatterboxModelFamily.STANDARD,
        )
        explicit_kwargs = adapter._build_generation_kwargs(
            explicit_request,
            voice_reference_path=None,
            exaggeration=0.5,
            family=ChatterboxModelFamily.STANDARD,
        )

        assert "speed_factor" not in default_kwargs
        assert speed_kwargs["speed_factor"] == 1.35
        assert explicit_kwargs["speed_factor"] == 0.85

    async def test_turbo_generation_kwargs_forward_speed_factor_when_requested(self):
        """Turbo should offer speed_factor to runtimes that support it."""
        adapter = ChatterboxAdapter({})
        request = TTSRequest(
            text="Hello [laugh]",
            model="chatterbox-turbo",
            speed=1.2,
            format=AudioFormat.WAV,
        )

        kwargs = adapter._build_generation_kwargs(
            request,
            voice_reference_path=None,
            exaggeration=0.5,
            family=ChatterboxModelFamily.TURBO,
        )

        assert kwargs["speed_factor"] == 1.2

    async def test_generation_defaults_parse_unprefixed_aliases_and_fallbacks(self):
        """Generation default config should accept aliases and ignore malformed numerics."""
        adapter = ChatterboxAdapter({
            "default_exaggeration": 0.7,
            "cfg_weight": 0.4,
            "temperature": 0.9,
            "repetition_penalty": 1.1,
            "min_p": 0.02,
            "top_p": 0.95,
        })
        prefixed_adapter = ChatterboxAdapter({
            "temperature": 0.1,
            "chatterbox_temperature": 0.85,
        })
        invalid_adapter = ChatterboxAdapter({
            "chatterbox_default_exaggeration": "bad",
            "chatterbox_cfg_weight": "bad",
            "chatterbox_temperature": "bad",
            "chatterbox_repetition_penalty": "bad",
            "chatterbox_min_p": "bad",
            "chatterbox_top_p": "bad",
        })

        assert adapter.default_exaggeration == 0.7
        assert adapter.default_cfg_weight == 0.4
        assert adapter.default_temperature == 0.9
        assert adapter.default_repetition_penalty == 1.1
        assert adapter.default_min_p == 0.02
        assert adapter.default_top_p == 0.95
        assert prefixed_adapter.default_temperature == 0.85
        assert invalid_adapter.default_exaggeration == 0.5
        assert invalid_adapter.default_cfg_weight == 0.5
        assert invalid_adapter.default_temperature == 0.8
        assert invalid_adapter.default_repetition_penalty == 1.2
        assert invalid_adapter.default_min_p == 0.05
        assert invalid_adapter.default_top_p == 1.0

    async def test_streaming_generation_uses_target_latency_chunk_duration(self):
        """Streaming TTS should pass configured target latency to the waveform encoder."""
        adapter = ChatterboxAdapter({"chatterbox_target_latency_ms": 125})
        fake_model = MagicMock()
        fake_model.sr = 24000
        fake_waveform = object()
        fake_model.generate.return_value = fake_waveform
        seen: dict[str, object] = {}

        async def fake_stream(waveform, format, sample_rate, channels=1, chunk_duration_sec=0.2):
            seen["waveform"] = waveform
            seen["format"] = format
            seen["sample_rate"] = sample_rate
            seen["channels"] = channels
            seen["chunk_duration_sec"] = chunk_duration_sec
            yield b"encoded"

        request = TTSRequest(text="Hello", format=AudioFormat.WAV)

        with patch.object(adapter, "_get_model", new=AsyncMock(return_value=fake_model)):
            with patch(
                "tldw_Server_API.app.core.TTS.waveform_streamer.stream_encoded_waveform",
                new=fake_stream,
            ):
                chunks = [
                    chunk
                    async for chunk in adapter._stream_audio_chatterbox(
                        request,
                        "en",
                        None,
                        adapter.default_exaggeration,
                        ChatterboxModelFamily.STANDARD,
                    )
                ]

        assert chunks == [b"encoded"]
        assert seen == {
            "waveform": fake_waveform,
            "format": "wav",
            "sample_rate": 24000,
            "channels": 1,
            "chunk_duration_sec": 0.125,
        }

    async def test_streaming_generation_uses_bf16_autocast_when_enabled(self):
        """Opt-in BF16 should prepare T3 and wrap generation in torch autocast."""
        adapter = ChatterboxAdapter({
            "chatterbox_device": "cpu",
            "chatterbox_use_bf16": "on",
        })
        fake_t3 = MagicMock()
        converted_t3 = MagicMock()
        fake_t3.to.return_value = converted_t3
        fake_model = MagicMock()
        fake_model.sr = 24000
        fake_model.t3 = fake_t3
        fake_waveform = object()
        seen: dict[str, object] = {"autocast_active": False}

        class FakeAutocast:
            def __init__(self, device_type: str, dtype: object):
                seen["autocast_device_type"] = device_type
                seen["autocast_dtype"] = dtype

            def __enter__(self):
                seen["autocast_active"] = True
                return self

            def __exit__(self, exc_type, exc, tb):
                seen["autocast_active"] = False
                seen["autocast_exited"] = True
                return False

        class FakeTorch:
            bfloat16 = object()

            @staticmethod
            def autocast(device_type: str, dtype: object):
                return FakeAutocast(device_type, dtype)

        def fake_generate(*args, **kwargs):
            seen["generate_inside_autocast"] = seen["autocast_active"]
            return fake_waveform

        async def fake_stream(waveform, format, sample_rate, channels=1, chunk_duration_sec=0.2):
            seen["waveform"] = waveform
            yield b"encoded-bf16"

        fake_model.generate.side_effect = fake_generate
        request = TTSRequest(text="Hello", format=AudioFormat.WAV)

        with patch.object(adapter, "_get_model", new=AsyncMock(return_value=fake_model)):
            with patch(
                "tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter._get_torch",
                return_value=FakeTorch(),
            ):
                with patch(
                    "tldw_Server_API.app.core.TTS.waveform_streamer.stream_encoded_waveform",
                    new=fake_stream,
                ):
                    chunks = [
                        chunk
                        async for chunk in adapter._stream_audio_chatterbox(
                            request,
                            "en",
                            None,
                            adapter.default_exaggeration,
                            ChatterboxModelFamily.STANDARD,
                        )
                    ]

        assert chunks == [b"encoded-bf16"]
        fake_t3.to.assert_called_once_with(dtype=FakeTorch.bfloat16)
        assert fake_model.t3 is converted_t3
        assert seen["autocast_device_type"] == "cpu"
        assert seen["autocast_dtype"] is FakeTorch.bfloat16
        assert seen["generate_inside_autocast"] is True
        assert seen["autocast_exited"] is True
        assert seen["waveform"] is fake_waveform

    async def test_streaming_generation_defers_voice_reference_cleanup(self, tmp_path):
        """Streaming TTS should keep the prepared reference file until stream consumption ends."""
        adapter = ChatterboxAdapter({})
        reference_path = tmp_path / "reference.wav"
        reference_path.write_bytes(b"reference-audio")
        model = MagicMock()
        model.sr = 24000

        async def fake_stream(request, language_id, voice_reference_path, exaggeration, family):
            assert voice_reference_path == str(reference_path)
            assert Path(voice_reference_path).exists()
            yield b"audio"

        request = TTSRequest(
            text="Hello with a cloned voice",
            voice_reference=b"raw-reference",
            format=AudioFormat.WAV,
            stream=True,
        )

        with patch.object(adapter, "ensure_initialized", new=AsyncMock(return_value=True)):
            with patch.object(adapter, "validate_request", new=AsyncMock(return_value=(True, None))):
                with patch.object(adapter, "_get_model", new=AsyncMock(return_value=model)):
                    with patch.object(adapter, "_prepare_voice_reference", new=AsyncMock(return_value=str(reference_path))):
                        with patch.object(adapter, "_stream_audio_chatterbox", fake_stream):
                            response = await adapter.generate(request)

        assert reference_path.exists()
        assert response.audio_stream is not None
        chunks = [chunk async for chunk in response.audio_stream]
        assert chunks == [b"audio"]
        assert not reference_path.exists()

    async def test_turbo_generation_kwargs_drop_controls_ignored_upstream(self):
        """Turbo should not pass no-op controls that upstream only warns about."""
        adapter = ChatterboxAdapter({})
        request = TTSRequest(
            text="Hello [laugh]",
            model="chatterbox-turbo",
            emotion="happy",
            format=AudioFormat.WAV,
            extra_params={
                "cfg_weight": 0.5,
                "min_p": 0.2,
                "temperature": 0.7,
                "top_k": 250,
            },
        )

        kwargs = adapter._build_generation_kwargs(
            request,
            voice_reference_path="/tmp/reference.wav",
            exaggeration=0.7,
            family=ChatterboxModelFamily.TURBO,
        )

        assert kwargs["audio_prompt_path"] == "/tmp/reference.wav"
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_k"] == 250
        assert "cfg_weight" not in kwargs
        assert "exaggeration" not in kwargs
        assert "min_p" not in kwargs

    async def test_generation_reuses_cached_voice_conditionals(self, tmp_path):
        """Repeated reference audio should use prepared Chatterbox conditionals once."""
        adapter = ChatterboxAdapter({})
        reference_path = tmp_path / "reference.wav"
        reference_path.write_bytes(b"RIFF" + b"\x00" * 128)

        class FakeModel:
            sr = 24000

            def __init__(self):
                self.conds = None
                self.prepare_calls: list[tuple[str, float]] = []
                self.generate_kwargs: list[dict[str, object]] = []

            def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5) -> None:
                self.prepare_calls.append((wav_fpath, exaggeration))
                self.conds = {"wav_fpath": wav_fpath, "exaggeration": exaggeration}

            def generate(self, text: str, **kwargs):  # noqa: ANN001
                self.generate_kwargs.append(dict(kwargs))
                return object()

        fake_model = FakeModel()
        request = TTSRequest(text="Cached voice", format=AudioFormat.WAV)

        async def fake_stream(waveform, format, sample_rate, channels=1, chunk_duration_sec=0.2):  # noqa: ARG001
            yield b"encoded"

        with patch.object(adapter, "_get_model", new=AsyncMock(return_value=fake_model)):
            with patch(
                "tldw_Server_API.app.core.TTS.waveform_streamer.stream_encoded_waveform",
                new=fake_stream,
            ):
                first = [
                    chunk
                    async for chunk in adapter._stream_audio_chatterbox(
                        request,
                        "en",
                        str(reference_path),
                        0.6,
                        ChatterboxModelFamily.STANDARD,
                    )
                ]
                second = [
                    chunk
                    async for chunk in adapter._stream_audio_chatterbox(
                        request,
                        "en",
                        str(reference_path),
                        0.6,
                        ChatterboxModelFamily.STANDARD,
                    )
                ]

        assert first == [b"encoded"]
        assert second == [b"encoded"]
        assert fake_model.prepare_calls == [(str(reference_path), 0.6)]
        assert len(fake_model.generate_kwargs) == 2
        assert all("audio_prompt_path" not in kwargs for kwargs in fake_model.generate_kwargs)

    async def test_conditionals_cache_key_hashes_reference_in_worker_thread(self, tmp_path):
        """Reference hashing should not perform blocking file I/O on the async request path."""
        adapter = ChatterboxAdapter({})
        reference_path = tmp_path / "reference.wav"
        reference_path.write_bytes(b"RIFF-threaded-hash")

        async def fake_to_thread(func, *args, **kwargs):
            fake_to_thread.called = True
            return func(*args, **kwargs)

        fake_to_thread.called = False

        with patch.object(chatterbox_mod.asyncio, "to_thread", new=fake_to_thread):
            cache_key = await adapter._voice_conditionals_cache_key(
                str(reference_path),
                family=ChatterboxModelFamily.STANDARD,
                exaggeration=0.5,
            )

        assert fake_to_thread.called is True
        assert cache_key is not None
        assert cache_key[0] == ChatterboxModelFamily.STANDARD.value

    async def test_conditionals_cache_stores_cpu_conditionals(self, tmp_path):
        """Cached Chatterbox conditionals should be moved off accelerator memory when possible."""
        adapter = ChatterboxAdapter({"chatterbox_conditionals_cache_size": 2})
        adapter.device = "cuda"
        reference_path = tmp_path / "reference.wav"
        reference_path.write_bytes(b"RIFF-cpu-cache")

        class FakeConditionals:
            def __init__(self, device: str):
                self.device = device

            def detach(self):
                return self

            def cpu(self):
                return FakeConditionals("cpu")

            def to(self, device: str):
                return FakeConditionals(device)

        class FakeModel:
            def __init__(self):
                self.conds = None
                self.prepare_calls = 0

            def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5) -> None:  # noqa: ARG002
                self.prepare_calls += 1
                self.conds = FakeConditionals("cuda")

        fake_model = FakeModel()

        first = await adapter._prepare_cached_conditionals(
            fake_model,
            voice_reference_path=str(reference_path),
            family=ChatterboxModelFamily.STANDARD,
            exaggeration=0.5,
        )
        cached_conditionals = next(iter(adapter._conditionals_cache.values()))
        second = await adapter._prepare_cached_conditionals(
            fake_model,
            voice_reference_path=str(reference_path),
            family=ChatterboxModelFamily.STANDARD,
            exaggeration=0.5,
        )

        assert first is True
        assert second is True
        assert fake_model.prepare_calls == 1
        assert cached_conditionals.device == "cpu"
        assert fake_model.conds.device == "cuda"

    async def test_conditionals_cache_evicts_least_recently_used_reference(self, tmp_path):
        """Conditionals cache should stay bounded and refresh recency on hits."""
        adapter = ChatterboxAdapter({"chatterbox_conditionals_cache_size": 2})
        references = {}
        for name in ("a", "b", "c"):
            reference_path = tmp_path / f"{name}.wav"
            reference_path.write_bytes(f"RIFF-{name}".encode("ascii") + b"\x00" * 128)
            references[name] = reference_path

        class FakeModel:
            def __init__(self):
                self.conds = None
                self.prepare_calls: list[str] = []

            def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5) -> None:
                self.prepare_calls.append(wav_fpath)
                self.conds = {"wav_fpath": wav_fpath, "exaggeration": exaggeration}

        fake_model = FakeModel()

        async def prepare(name: str) -> None:
            prepared = await adapter._prepare_cached_conditionals(
                fake_model,
                voice_reference_path=str(references[name]),
                family=ChatterboxModelFamily.STANDARD,
                exaggeration=0.5,
            )
            assert prepared is True

        await prepare("a")
        await prepare("b")
        await prepare("a")
        await prepare("c")
        await prepare("b")

        assert fake_model.prepare_calls == [
            str(references["a"]),
            str(references["b"]),
            str(references["c"]),
            str(references["b"]),
        ]
        assert len(adapter._conditionals_cache) == 2
        remaining_keys = set(adapter._conditionals_cache)
        assert await adapter._voice_conditionals_cache_key(
            str(references["a"]),
            family=ChatterboxModelFamily.STANDARD,
            exaggeration=0.5,
        ) not in remaining_keys
        assert await adapter._voice_conditionals_cache_key(
            str(references["b"]),
            family=ChatterboxModelFamily.STANDARD,
            exaggeration=0.5,
        ) in remaining_keys
        assert await adapter._voice_conditionals_cache_key(
            str(references["c"]),
            family=ChatterboxModelFamily.STANDARD,
            exaggeration=0.5,
        ) in remaining_keys

    async def test_turbo_metadata_reports_ignored_controls(self):
        """Turbo metadata should make intentionally ignored controls explicit."""
        adapter = ChatterboxAdapter({})
        request = TTSRequest(
            text="Hello [laugh]",
            model="chatterbox-turbo",
            emotion="happy",
            emotion_intensity=1.5,
            seed=1234,
            extra_params={"cfg_weight": 0.5},
        )

        metadata = adapter._build_generation_metadata(
            request,
            language_id="en",
            family=ChatterboxModelFamily.TURBO,
            exaggeration=0.7,
        )

        assert metadata["model_family"] == "turbo"
        assert metadata["seed"] == 1234
        assert metadata["ignored_controls"] == [
            "cfg_weight",
            "emotion",
            "emotion_intensity",
            "exaggeration",
        ]

    async def test_cleanup_clears_all_lazy_chatterbox_models(self):
        """Close should clear standard, multilingual, and turbo model handles."""
        adapter = ChatterboxAdapter({})
        adapter.model_en = MagicMock()
        adapter.model_multi = MagicMock()
        adapter.model_turbo = MagicMock()
        adapter.model_vc = MagicMock()

        await adapter._cleanup_resources()

        assert adapter.model_en is None
        assert adapter.model_multi is None
        assert adapter.model_turbo is None
        assert adapter.model_vc is None


@pytest.mark.asyncio
class TestChatterboxAdapterSanitizedLogs:
    async def test_initialization_failure_log_sanitizes_exception_text(self):
        raw_marker = "RAW_CHATTERBOX_INIT_SECRET_MARKER token=init-secret /tmp/chatterbox-init.wav"
        adapter = ChatterboxAdapter({})

        with _LogCapture() as messages:
            with patch(
                "tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter._get_torch",
                return_value=MagicMock(),
            ):
                with patch("builtins.__import__", side_effect=RuntimeError(raw_marker)):
                    success = await adapter.initialize()

        assert success is False
        assert adapter._status == ProviderStatus.ERROR
        assert any("Initialization failed" in message for message in messages)
        assert all(raw_marker not in message for message in messages)
        assert all("init-secret" not in message for message in messages)

    async def test_voice_reference_processing_log_sanitizes_error_text(self):
        raw_marker = "RAW_CHATTERBOX_REFERENCE_PROCESSING_SECRET_MARKER token=voice-secret"
        adapter = ChatterboxAdapter({})

        with _LogCapture() as messages:
            with patch(
                "tldw_Server_API.app.core.TTS.audio_utils.process_voice_reference_async",
                new=AsyncMock(return_value=(b"", raw_marker)),
            ):
                result = await adapter._prepare_voice_reference(b"audio")

        assert result is None
        assert any("Voice reference processing failed" in message for message in messages)
        assert all(raw_marker not in message for message in messages)
        assert all("voice-secret" not in message for message in messages)

    async def test_voice_reference_prepared_log_sanitizes_temp_path(self):
        raw_path = "/tmp/RAW_CHATTERBOX_REFERENCE_PATH_MARKER/token=path-secret/reference.wav"
        adapter = ChatterboxAdapter({})

        class _TempFile:
            name = raw_path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def write(self, data):
                return len(data)

        with _LogCapture("INFO") as messages:
            with patch(
                "tldw_Server_API.app.core.TTS.audio_utils.process_voice_reference_async",
                new=AsyncMock(return_value=(b"wav-bytes", None)),
            ):
                with patch("tempfile.NamedTemporaryFile", return_value=_TempFile()):
                    result = await adapter._prepare_voice_reference(b"audio")

        assert result == raw_path
        assert any("Voice reference prepared" in message for message in messages)
        assert all(raw_path not in message for message in messages)
        assert all("path-secret" not in message for message in messages)

    async def test_prepare_voice_reference_fallback_log_sanitizes_exception_text(self):
        raw_marker = "RAW_CHATTERBOX_PREPARE_REFERENCE_SECRET_MARKER token=prepare-secret"
        adapter = ChatterboxAdapter({})

        with _LogCapture() as messages:
            with patch(
                "tldw_Server_API.app.core.TTS.audio_utils.process_voice_reference_async",
                new=AsyncMock(side_effect=RuntimeError(raw_marker)),
            ):
                result = await adapter._prepare_voice_reference(b"audio")

        assert result is None
        assert any("Failed to prepare voice reference" in message for message in messages)
        assert all(raw_marker not in message for message in messages)
        assert all("prepare-secret" not in message for message in messages)

#######################################################################################################################
#
# End of test_chatterbox_adapter_mock.py
