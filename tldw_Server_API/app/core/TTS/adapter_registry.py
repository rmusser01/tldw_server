# adapter_registry.py
# Description: Registry and factory for TTS adapters
#
import asyncio
import importlib
import os
from enum import Enum
from typing import Any, Optional, Union

#
# Third-party Imports
from loguru import logger

from tldw_Server_API.app.core.Infrastructure.provider_registry import (
    ProviderRegistryBase,
    ProviderRegistryConfig,
    ProviderStatus as RegistryProviderStatus,
)
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

#
# Local Imports
from .adapters.base import AudioFormat, ProviderStatus, TTSAdapter, TTSCapabilities
from .tts_config import get_tts_config_manager
from .tts_exceptions import (
    TTSError,
    TTSProviderNotConfiguredError,
)
from .tts_resource_manager import get_existing_resource_manager, get_resource_manager
from .utils import parse_bool

#
#######################################################################################################################
#
# TTS Adapter Registry and Factory

_TTS_REGISTRY_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_TTS_REGISTRY_ADAPTER_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TTSError,
) + _TTS_REGISTRY_NONCRITICAL_EXCEPTIONS


def _safe_exception_label(exc: BaseException) -> str:
    """Return a non-sensitive exception identifier for logs."""
    return type(exc).__name__


class TTSProvider(Enum):
    """
    Enumeration of TTS providers known to the service.

    Note: some members (for example, ALLTALK and MOCK) are placeholders
    without concrete adapters registered in DEFAULT_ADAPTERS. Requests
    targeting those providers will surface as "provider not configured"
    at runtime until an adapter is implemented and enabled.
    """
    OPENAI = "openai"
    KOKORO = "kokoro"
    HIGGS = "higgs"
    DIA = "dia"
    CHATTERBOX = "chatterbox"
    ELEVENLABS = "elevenlabs"
    VIBEVOICE = "vibevoice"
    VIBEVOICE_REALTIME = "vibevoice_realtime"
    NEUTTS = "neutts"
    INDEX_TTS = "index_tts"
    SUPERTONIC = "supertonic"
    SUPERTONIC2 = "supertonic2"
    POCKET_TTS = "pocket_tts"
    POCKET_TTS_CPP = "pocket_tts_cpp"
    ECHO_TTS = "echo_tts"
    QWEN3_TTS = "qwen3_tts"
    OMNIVOICE = "omnivoice"
    LUX_TTS = "lux_tts"
    KITTEN_TTS = "kitten_tts"
    # Additional providers
    ALLTALK = "alltalk"  # TODO: Implement AllTalk adapter
    MOCK = "mock"  # Mock provider for testing


def _provider_alias_tokens(value: str) -> set[str]:
    token = str(value or "").strip().lower()
    if not token:
        return set()
    return {
        token,
        token.replace("_", "-"),
        token.replace("-", "_"),
        token.replace("_", "").replace("-", ""),
    }


def _build_tts_provider_aliases() -> dict[str, TTSProvider]:
    aliases: dict[str, TTSProvider] = {}

    for provider in TTSProvider:
        for token in _provider_alias_tokens(provider.value):
            aliases[token] = provider
        for token in _provider_alias_tokens(provider.name):
            aliases[token] = provider

    # Common user-facing synonyms
    explicit_aliases: dict[str, TTSProvider] = {
        "open-ai": TTSProvider.OPENAI,
        "oai": TTSProvider.OPENAI,
        "eleven-labs": TTSProvider.ELEVENLABS,
        "elevenlabs-tts": TTSProvider.ELEVENLABS,
        "qwen3tts": TTSProvider.QWEN3_TTS,
        "echotts": TTSProvider.ECHO_TTS,
        "kittentts": TTSProvider.KITTEN_TTS,
        "vibevoice-asr": TTSProvider.VIBEVOICE,
    }
    for alias, provider in explicit_aliases.items():
        for token in _provider_alias_tokens(alias):
            aliases[token] = provider

    return aliases


_TTS_PROVIDER_ALIASES: dict[str, TTSProvider] = _build_tts_provider_aliases()


def _apply_provider_aliases(provider_key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Duplicate generic provider config keys into legacy adapter-prefixed aliases."""
    normalized_provider = str(provider_key or "").strip().lower()
    normalized_cfg = dict(cfg or {})

    def alias(src: str, dst: str) -> None:
        if src in normalized_cfg and dst not in normalized_cfg and normalized_cfg[src] is not None:
            normalized_cfg[dst] = normalized_cfg[src]

    if normalized_provider == "openai":
        alias("api_key", "openai_api_key")
        alias("base_url", "openai_base_url")
        alias("model", "openai_model")
    elif normalized_provider == "kokoro":
        alias("use_onnx", "kokoro_use_onnx")
        alias("model_path", "kokoro_model_path")
        alias("voices_json", "kokoro_voices_json")
        alias("voice_dir", "kokoro_voice_dir")
        alias("device", "kokoro_device")
        alias("target_latency_ms", "kokoro_target_latency_ms")
    elif normalized_provider == "higgs":
        alias("model_path", "higgs_model_path")
        alias("tokenizer_path", "higgs_tokenizer_path")
        alias("device", "higgs_device")
        alias("use_fp16", "higgs_use_fp16")
        alias("batch_size", "higgs_batch_size")
        alias("target_latency_ms", "higgs_target_latency_ms")
    elif normalized_provider == "dia":
        alias("model_path", "dia_model_path")
        alias("device", "dia_device")
        alias("use_safetensors", "dia_use_safetensors")
        alias("use_bf16", "dia_use_bf16")
        alias("sample_rate", "dia_sample_rate")
        alias("auto_detect_speakers", "dia_auto_detect_speakers")
        alias("max_speakers", "dia_max_speakers")
        alias("target_latency_ms", "dia_target_latency_ms")
    elif normalized_provider == "chatterbox":
        alias("device", "chatterbox_device")
        alias("use_multilingual", "chatterbox_use_multilingual")
        alias("disable_watermark", "chatterbox_disable_watermark")
        alias("target_latency_ms", "chatterbox_target_latency_ms")
    elif normalized_provider == "elevenlabs":
        alias("api_key", "elevenlabs_api_key")
        alias("base_url", "elevenlabs_base_url")
        alias("model", "elevenlabs_model")
        alias("stability", "elevenlabs_stability")
        alias("similarity_boost", "elevenlabs_similarity_boost")
        alias("style", "elevenlabs_style")
        alias("speaker_boost", "elevenlabs_speaker_boost")
    elif normalized_provider == "vibevoice":
        alias("device", "vibevoice_device")
        alias("sample_rate", "vibevoice_sample_rate")
        alias("variant", "vibevoice_variant")
        alias("model_path", "vibevoice_model_path")
        alias("model_dir", "vibevoice_model_dir")
        alias("cache_dir", "vibevoice_cache_dir")
        alias("voices_dir", "vibevoice_voices_dir")
        alias("background_music", "vibevoice_background_music")
        alias("enable_singing", "vibevoice_enable_singing")
        alias("use_quantization", "vibevoice_use_quantization")
        alias("auto_cleanup", "vibevoice_auto_cleanup")
        alias("auto_download", "vibevoice_auto_download")
        alias("enable_sage", "vibevoice_enable_sage")
        alias("attention_type", "vibevoice_attention_type")
        alias("cfg_scale", "vibevoice_cfg_scale")
        alias("diffusion_steps", "vibevoice_diffusion_steps")
        alias("temperature", "vibevoice_temperature")
        alias("top_p", "vibevoice_top_p")
        alias("top_k", "vibevoice_top_k")
        alias("stream_chunk_size", "vibevoice_stream_chunk_size")
        alias("stream_buffer_size", "vibevoice_stream_buffer_size")
    elif normalized_provider == "neutts":
        alias("device", "backbone_device")
        alias("backbone_repo", "backbone_repo")
        alias("codec_repo", "codec_repo")
        alias("sample_rate", "sample_rate")
    elif normalized_provider == "index_tts":
        alias("model_dir", "index_tts_model_dir")
        alias("cfg_path", "index_tts_cfg_path")
        alias("device", "index_tts_device")
        alias("use_fp16", "index_tts_use_fp16")
        alias("use_cuda_kernel", "index_tts_use_cuda_kernel")
        alias("use_deepspeed", "index_tts_use_deepspeed")
        alias("interval_silence", "index_tts_interval_silence")
        alias("quick_streaming_tokens", "index_tts_quick_streaming_tokens")
        alias("max_text_tokens_per_segment", "index_tts_max_text_tokens_per_segment")
        alias("more_segment_before", "index_tts_more_segment_before")
        alias("verbose", "index_tts_verbose")
        alias("sample_rate", "sample_rate")
    elif normalized_provider == "pocket_tts":
        alias("model_path", "pocket_tts_model_path")
        alias("tokenizer_path", "pocket_tts_tokenizer_path")
        alias("precision", "pocket_tts_precision")
        alias("device", "pocket_tts_device")
        alias("temperature", "pocket_tts_temperature")
        alias("lsd_steps", "pocket_tts_lsd_steps")
        alias("max_frames", "pocket_tts_max_frames")
        alias("stream_first_chunk_frames", "pocket_tts_stream_first_chunk_frames")
        alias("stream_target_buffer_sec", "pocket_tts_stream_target_buffer_sec")
        alias("stream_max_chunk_frames", "pocket_tts_stream_max_chunk_frames")
    elif normalized_provider == "pocket_tts_cpp":
        alias("binary_path", "pocket_tts_cpp_binary_path")
        alias("tokenizer_path", "pocket_tts_cpp_tokenizer_path")
        alias("device", "pocket_tts_cpp_device")
        alias("sample_rate", "pocket_tts_cpp_sample_rate")
        alias("enable_voice_cache", "pocket_tts_cpp_enable_voice_cache")
        alias("cache_ttl_hours", "pocket_tts_cpp_cache_ttl_hours")
        alias("cache_max_bytes_per_user", "pocket_tts_cpp_cache_max_bytes_per_user")
        alias("persist_direct_voice_references", "pocket_tts_cpp_persist_direct_voice_references")
    elif normalized_provider == "echo_tts":
        alias("model", "echo_tts_model")
        alias("model_path", "echo_tts_model_path")
        alias("device", "echo_tts_device")
        alias("module_path", "echo_tts_module_path")
        alias("sample_rate", "echo_tts_sample_rate")
        alias("cache_size", "echo_tts_cache_size")
        alias("cache_ttl_sec", "echo_tts_cache_ttl_sec")
        alias("cache_on_device", "echo_tts_cache_on_device")
        alias("fish_ae_repo", "echo_tts_fish_ae_repo")
        alias("pca_state_file", "echo_tts_pca_state_file")

    return normalized_cfg


class TTSAdapterRegistry:
    """
    Registry for TTS adapters.
    Manages registration, initialization, and access to TTS providers.
    """

    # Default adapter mappings (lazy, via dotted paths to avoid heavy imports at module import time)
    DEFAULT_ADAPTERS: dict["TTSProvider", "str|type[TTSAdapter]"] = {
        TTSProvider.OPENAI: "tldw_Server_API.app.core.TTS.adapters.openai_adapter.OpenAITTSAdapter",
        TTSProvider.KOKORO: "tldw_Server_API.app.core.TTS.adapters.kokoro_adapter.KokoroAdapter",
        TTSProvider.HIGGS: "tldw_Server_API.app.core.TTS.adapters.higgs_adapter.HiggsAdapter",
        TTSProvider.DIA: "tldw_Server_API.app.core.TTS.adapters.dia_adapter.DiaAdapter",
        TTSProvider.CHATTERBOX: "tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter.ChatterboxAdapter",
        TTSProvider.ELEVENLABS: "tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.ElevenLabsTTSAdapter",
        TTSProvider.VIBEVOICE: "tldw_Server_API.app.core.TTS.adapters.vibevoice_adapter.VibeVoiceAdapter",
        TTSProvider.VIBEVOICE_REALTIME: "tldw_Server_API.app.core.TTS.adapters.vibevoice_realtime_adapter.VibeVoiceRealtimeAdapter",
        TTSProvider.NEUTTS: "tldw_Server_API.app.core.TTS.adapters.neutts_adapter.NeuTTSAdapter",
        TTSProvider.INDEX_TTS: "tldw_Server_API.app.core.TTS.adapters.index_tts_adapter.IndexTTS2Adapter",
        TTSProvider.SUPERTONIC: "tldw_Server_API.app.core.TTS.adapters.supertonic_adapter.SupertonicOnnxAdapter",
        TTSProvider.SUPERTONIC2: "tldw_Server_API.app.core.TTS.adapters.supertonic2_adapter.Supertonic2OnnxAdapter",
        TTSProvider.POCKET_TTS: "tldw_Server_API.app.core.TTS.adapters.pocket_tts_adapter.PocketTTSOnnxAdapter",
        TTSProvider.POCKET_TTS_CPP: "tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_adapter.PocketTTSCppAdapter",
        TTSProvider.ECHO_TTS: "tldw_Server_API.app.core.TTS.adapters.echo_tts_adapter.EchoTTSAdapter",
        TTSProvider.QWEN3_TTS: "tldw_Server_API.app.core.TTS.adapters.qwen3_tts_adapter.Qwen3TTSAdapter",
        TTSProvider.OMNIVOICE: "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter.OmniVoiceAdapter",
        TTSProvider.LUX_TTS: "tldw_Server_API.app.core.TTS.adapters.luxtts_adapter.LuxTTSAdapter",
        TTSProvider.KITTEN_TTS: "tldw_Server_API.app.core.TTS.adapters.kitten_tts_adapter.KittenTTSAdapter",
    }

    @classmethod
    def resolve_provider(cls, provider: Union[TTSProvider, str, None]) -> Optional[TTSProvider]:
        """Resolve provider aliases (enum names, dashed/underscored forms, and common synonyms)."""
        if isinstance(provider, TTSProvider):
            return provider
        if provider is None:
            return None
        for token in _provider_alias_tokens(str(provider)):
            mapped = _TTS_PROVIDER_ALIASES.get(token)
            if mapped is not None:
                return mapped
        return None

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        *,
        include_defaults: bool = True,
    ):
        """
        Initialize the registry.

        Args:
            config: Configuration dictionary for all adapters
        """
        # Use unified configuration system
        if config:
            # Override config provided for testing
            self.config_manager = None
            # Ensure config is a dictionary
            if isinstance(config, dict):
                self.tts_config = config
                self.config = config
            else:
                # If config is not a dict (e.g., ConfigParser), convert it
                logger.warning(f"Non-dict config passed to TTSAdapterRegistry: {type(config)}")
                self.tts_config = {}
                self.config = {}
        else:
            self.config_manager = get_tts_config_manager()
            self.tts_config = self.config_manager.get_config()
            # Legacy config support - convert Pydantic model to dict
            self.config = model_dump_compat(self.tts_config)

        self._adapters: dict[TTSProvider, TTSAdapter] = {}
        # Store either classes or dotted paths; resolve lazily when needed
        self._adapter_specs: dict[TTSProvider, Any] = (
            self.DEFAULT_ADAPTERS.copy() if include_defaults else {}
        )
        self._initialized_providers: set[TTSProvider] = set()

        def _extract_retry_seconds(raw_cfg: Any) -> Optional[float]:
            if raw_cfg is None:
                return None
            try:
                return float(raw_cfg)
            except (TypeError, ValueError):
                return None

        retry_seconds: Optional[float] = None
        if isinstance(self.config, dict):
            retry_seconds = _extract_retry_seconds(self.config.get("adapter_failure_retry_seconds"))
            if retry_seconds is None:
                perf_cfg = self.config.get("performance")
                if isinstance(perf_cfg, dict):
                    retry_seconds = _extract_retry_seconds(perf_cfg.get("adapter_failure_retry_seconds"))
        if retry_seconds is None and self.config_manager:
            try:
                perf_cfg = self.config_manager.get_config().performance  # type: ignore[call-arg]
                retry_seconds = _extract_retry_seconds(
                    getattr(perf_cfg, "adapter_failure_retry_seconds", None)
                )
            except _TTS_REGISTRY_NONCRITICAL_EXCEPTIONS:
                pass

        if retry_seconds is not None and retry_seconds <= 0:
            retry_seconds = None

        self._failure_retry_seconds: Optional[float] = retry_seconds
        self._base: ProviderRegistryBase[TTSAdapter] = ProviderRegistryBase(
            config=ProviderRegistryConfig(failure_retry_seconds=retry_seconds),
            adapter_validator=lambda adapter: isinstance(adapter, TTSAdapter),
            adapter_materializer_async=self._materialize_adapter_async,
            provider_enabled_callback=self._is_provider_enabled_by_config,
        )
        for provider_name, adapter_spec in self._adapter_specs.items():
            self._base.register_adapter(provider_name.value, adapter_spec)

    def _is_provider_enabled_by_config(self, provider_key: str) -> Optional[bool]:
        """
        Return config-driven provider enablement for base registry checks.

        This preserves existing precedence:
        - Unified config manager uses `is_provider_enabled(...)`.
        - Direct dict config honors explicit `providers.<name>.enabled`, then
          legacy `{provider}_enabled` flags.
        - No explicit flag => no opinion (`None`) so wrapper logic is unchanged.
        """
        provider = self.resolve_provider(provider_key)
        if provider is None:
            return None

        if self.config_manager:
            try:
                return bool(self.config_manager.is_provider_enabled(provider.value))
            except _TTS_REGISTRY_NONCRITICAL_EXCEPTIONS:
                return None

        explicit_enabled = self._get_dict_provider_enabled_flag(provider)
        if explicit_enabled is not None:
            return explicit_enabled
        return None

    def _get_dict_provider_enabled_flag(self, provider: TTSProvider) -> Optional[bool]:
        """
        Resolve explicit provider enablement from dict-style config.

        Precedence:
        1. providers.<provider>.enabled
        2. legacy <provider>_enabled
        """
        if not isinstance(self.config, dict):
            return None

        providers_cfg = self.config.get("providers")
        if isinstance(providers_cfg, dict):
            provider_cfg = providers_cfg.get(provider.value)
            if provider_cfg is not None and not isinstance(provider_cfg, dict):
                provider_cfg = model_dump_compat(provider_cfg)
            if isinstance(provider_cfg, dict) and "enabled" in provider_cfg:
                return parse_bool(provider_cfg.get("enabled"), default=True)

        enabled_key = f"{provider.value}_enabled"
        if enabled_key in self.config:
            return parse_bool(self.config.get(enabled_key), default=True)
        return None

    def register_adapter(self, provider: Union[TTSProvider, str], adapter: Any):
        """
        Register a custom adapter class for a provider.

        Args:
            provider: The provider enum
            adapter: Adapter class or dotted import path string to register
        """
        resolved_provider = self.resolve_provider(provider)
        if resolved_provider is None:
            raise ValueError(f"Unknown provider '{provider}'")
        self._adapter_specs[resolved_provider] = adapter
        self._adapters.pop(resolved_provider, None)
        self._initialized_providers.discard(resolved_provider)
        self._base.register_adapter(resolved_provider.value, adapter)
        try:
            name = adapter.__name__  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            name = str(adapter)
        logger.info(f"Registered adapter {name} for provider {resolved_provider.value}")

    def _schedule_retry(self, provider: TTSProvider) -> None:
        """Record a failed provider with optional retry backoff."""
        self._base.mark_failure(provider.value)

    def _resolve_adapter_class(self, spec: Any) -> type[TTSAdapter]:
        """Resolve an adapter class from a class object or dotted path string."""
        if isinstance(spec, str):
            module_path, _, class_name = spec.rpartition(".")
            if not module_path:
                raise ImportError(f"Invalid adapter spec '{spec}'")
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls
        return spec

    async def _materialize_adapter_async(self, provider_key: str, spec: Any) -> TTSAdapter:
        """
        Async materialization hook used by the shared provider registry base.
        """
        resolved_provider = self.resolve_provider(provider_key)
        if resolved_provider is None:
            raise TTSProviderNotConfiguredError(
                f"Unknown provider '{provider_key}'",
                provider=str(provider_key),
            )

        # Reuse already initialized adapter if available.
        existing = self._adapters.get(resolved_provider)
        if existing and existing.status == ProviderStatus.AVAILABLE:
            return existing

        success = await self._initialize_adapter(resolved_provider)
        if not success:
            raise RuntimeError(f"Failed to initialize {resolved_provider.value} adapter")

        adapter = self._adapters.get(resolved_provider)
        if adapter is None or adapter.status != ProviderStatus.AVAILABLE:
            raise RuntimeError(f"{resolved_provider.value} adapter is not available")
        return adapter

    async def get_adapter(self, provider: Union[TTSProvider, str]) -> Optional[TTSAdapter]:
        """
        Get an adapter instance for the specified provider.

        Args:
            provider: The TTS provider

        Returns:
            Initialized adapter instance or None if unavailable

        Raises:
            TTSProviderNotConfiguredError: If provider is not registered
        """
        resolved_provider = self.resolve_provider(provider)
        if resolved_provider is None:
            error_msg = f"Unknown provider '{provider}'"
            logger.error(error_msg)
            raise TTSProviderNotConfiguredError(error_msg, provider=str(provider))

        if resolved_provider not in self._adapter_specs:
            error_msg = f"No adapter registered for provider {resolved_provider.value}"
            logger.error(error_msg)
            raise TTSProviderNotConfiguredError(
                error_msg,
                provider=resolved_provider.value
            )
        adapter = await self._base.get_adapter_async(resolved_provider.value)
        if adapter is None:
            return None
        if adapter.status == ProviderStatus.AVAILABLE:
            self._adapters[resolved_provider] = adapter
            self._initialized_providers.add(resolved_provider)
            return adapter

        logger.warning(
            "Adapter for {} is not available (status: {})",
            resolved_provider.value,
            adapter.status,
        )
        self._schedule_retry(resolved_provider)
        return None

    async def create_adapter_with_overrides(
        self,
        provider: Union[TTSProvider, str],
        overrides: Optional[dict[str, Any]] = None,
    ) -> Optional[TTSAdapter]:
        """Create a non-cached adapter instance with config overrides."""
        resolved_provider = self.resolve_provider(provider)
        if resolved_provider is None:
            error_msg = f"Unknown provider '{provider}'"
            logger.error(error_msg)
            raise TTSProviderNotConfiguredError(error_msg, provider=str(provider))
        if resolved_provider not in self._adapter_specs:
            error_msg = f"No adapter registered for provider {resolved_provider.value}"
            logger.error(error_msg)
            raise TTSProviderNotConfiguredError(error_msg, provider=resolved_provider.value)

        # Respect explicit enable/disable flags; BYOK can supply credentials.
        if self.config_manager:
            try:
                if not self.config_manager.is_provider_enabled(resolved_provider.value):
                    logger.info(f"Provider {resolved_provider.value} is disabled in configuration")
                    return None
            except _TTS_REGISTRY_NONCRITICAL_EXCEPTIONS:
                pass
        else:
            explicit_enabled = self._get_dict_provider_enabled_flag(resolved_provider)
            if explicit_enabled is False:
                logger.info(f"Provider {resolved_provider.value} is disabled in configuration")
                return None

        adapter_class = self._resolve_adapter_class(self._adapter_specs[resolved_provider])

        provider_cfg: dict[str, Any] = {}
        if self.config_manager:
            base_cfg = self.config_manager.get_provider_config(resolved_provider.value)
            if base_cfg:
                provider_cfg = (
                    dict(base_cfg)
                    if isinstance(base_cfg, dict)
                    else model_dump_compat(base_cfg)
                )
        else:
            providers = self.config.get("providers") if isinstance(self.config, dict) else None
            if isinstance(providers, dict):
                base_cfg = providers.get(resolved_provider.value)
                if isinstance(base_cfg, dict):
                    provider_cfg = dict(base_cfg)
                elif base_cfg is not None:
                    provider_cfg = model_dump_compat(base_cfg)

        if overrides:
            provider_cfg.update(overrides)

        adapter = adapter_class(config=provider_cfg)
        try:
            success = await adapter.ensure_initialized()
        except _TTS_REGISTRY_ADAPTER_EXCEPTIONS as exc:
            logger.error(
                "Error initializing {} adapter with overrides ({})",
                resolved_provider.value,
                _safe_exception_label(exc),
            )
            return None
        if not success:
            logger.error(f"Failed to initialize {resolved_provider.value} adapter with overrides")
            return None
        return adapter

    async def _initialize_adapter(self, provider: TTSProvider) -> bool:
        """
        Initialize an adapter for a provider.

        Args:
            provider: The TTS provider

        Returns:
            True if initialization successful

        Raises:
            TTSProviderInitializationError: If initialization fails
        """
        try:
            # Get adapter class (lazily resolve to avoid heavy imports during module import)
            adapter_spec = self._adapter_specs[provider]
            adapter_class = self._resolve_adapter_class(adapter_spec)

            # Get provider-specific config
            provider_config = self._get_provider_config(provider)

            # Check if provider is enabled using unified config
            if self.config_manager:
                if not self.config_manager.is_provider_enabled(provider.value):
                    logger.info(f"Provider {provider.value} is disabled in configuration")
                    return False
            else:
                # Heuristic for direct dict configs used in tests:
                # - If an explicit provider enable flag is present, honor it.
                # - Otherwise, enable lightweight/remote providers when credentials are present
                #   (e.g., OPENAI/ELEVENLABS) and keep heavy local providers disabled by default.
                explicit_enabled = self._get_dict_provider_enabled_flag(provider)
                if explicit_enabled is False:
                    logger.info(f"Provider {provider.value} is disabled in configuration")
                    return False
                if explicit_enabled is None:
                    remote_providers = {TTSProvider.OPENAI, TTSProvider.ELEVENLABS}
                    if provider in remote_providers:
                        # Consider provider enabled if API key is supplied via config or env
                        api_key: Optional[str] = None
                        if provider == TTSProvider.OPENAI:
                            api_key = (self.config.get("openai_api_key")
                                       or os.getenv("OPENAI_API_KEY"))
                        elif provider == TTSProvider.ELEVENLABS:
                            api_key = (self.config.get("elevenlabs_api_key")
                                       or os.getenv("ELEVENLABS_API_KEY"))
                        if not api_key:
                            logger.info(
                                f"Provider {provider.value} is disabled (no credentials found)"
                            )
                            return False
                    else:
                        # Keep local/heavy providers disabled unless explicitly enabled
                        logger.info(
                            f"Provider {provider.value} is disabled by default (no explicit enable flag)"
                        )
                        return False

            # Get resource manager for monitoring
            resource_manager = await get_resource_manager()

            # Check memory before initializing new adapter
            if resource_manager.memory_monitor.is_memory_critical():
                logger.warning(f"Skipping {provider.value} initialization due to memory constraints")
                return False

            # Create adapter instance
            logger.info(f"Initializing {provider.value} adapter...")
            adapter = adapter_class(config=provider_config)

            # Initialize the adapter
            success = await adapter.ensure_initialized()

            if success:
                self._adapters[provider] = adapter
                self._initialized_providers.add(provider)
                logger.info(f"Successfully initialized {provider.value} adapter")
                return True
            else:
                error_msg = f"Failed to initialize {provider.value} adapter"
                logger.error(error_msg)
                # Don't store failed adapter - it will be retried next time
                return False

        except Exception as e:
            if isinstance(e, TTSError):
                logger.error(
                    "Error initializing {} adapter ({})",
                    provider.value,
                    _safe_exception_label(e),
                )
                raise
            logger.error(
                "Error initializing {} adapter ({})",
                provider.value,
                _safe_exception_label(e),
            )
            # Don't store failed adapter - it will be retried next time
            return False

    def _get_provider_config(self, provider: TTSProvider) -> dict[str, Any]:
        """
        Get configuration for a specific provider.

        Args:
            provider: The TTS provider

        Returns:
            Provider-specific configuration dictionary
        """
        if self.config_manager:
            # Use unified configuration system
            provider_cfg = self.config_manager.get_provider_config(provider.value)

            if provider_cfg:
                # Convert to dict for adapter consumption
                cfg = model_dump_compat(provider_cfg)
                return _apply_provider_aliases(provider.value, cfg)

        # Fallback to legacy config
        provider_config = self.config.copy()

        # Add provider-specific overrides
        provider_key = f"{provider.value}_config"
        if provider_key in self.config:
            provider_config.update(self.config[provider_key])

        return _apply_provider_aliases(provider.value, provider_config)

    async def get_all_capabilities(self) -> dict[TTSProvider, TTSCapabilities]:
        """
        Get capabilities of all available adapters.

        Returns:
            Dictionary mapping providers to their capabilities
        """
        capabilities = {}

        def _get_enabled_flag(provider: TTSProvider) -> Optional[bool]:
            if self.config_manager:
                provider_cfg = self.config_manager.get_provider_config(provider.value)
                if provider_cfg is not None:
                    return provider_cfg.enabled
                return None
            if isinstance(self.config, dict):
                return self._get_dict_provider_enabled_flag(provider)
            return None

        for provider in TTSProvider:
            if provider not in self._adapter_specs:
                continue

            enabled_flag = _get_enabled_flag(provider)
            if enabled_flag is False:
                continue

            # Only try to get adapters that are likely to work quickly
            # Skip local model providers in testing unless explicitly enabled
            if provider in [TTSProvider.KOKORO, TTSProvider.KITTEN_TTS, TTSProvider.HIGGS, TTSProvider.DIA,
                           TTSProvider.CHATTERBOX, TTSProvider.VIBEVOICE, TTSProvider.VIBEVOICE_REALTIME,
                           TTSProvider.SUPERTONIC, TTSProvider.SUPERTONIC2, TTSProvider.POCKET_TTS,
                           TTSProvider.POCKET_TTS_CPP,
                           TTSProvider.QWEN3_TTS] and enabled_flag is not True:
                continue

            try:
                adapter_spec = self._adapter_specs.get(provider)
                if adapter_spec is not None:
                    adapter_class = self._resolve_adapter_class(adapter_spec)
                    if getattr(adapter_class, "STATIC_CAPABILITY_DISCOVERY", False):
                        static_adapter = adapter_class(config=self._get_provider_config(provider))
                        caps = await static_adapter.get_capabilities()
                        if caps:
                            capabilities[provider] = caps
                            continue

                # Skip providers currently marked as failed by registry backoff
                # only when static capability discovery is unavailable.
                status = self._base.get_status(provider.value)
                if status == RegistryProviderStatus.FAILED:
                    continue

                # Try to get adapter with a timeout to avoid hanging
                adapter = await asyncio.wait_for(self.get_adapter(provider), timeout=5.0)
                if adapter:
                    caps = adapter.capabilities
                    if caps:
                        capabilities[provider] = caps
            except asyncio.TimeoutError:
                logger.warning(f"Timeout getting capabilities for {provider.value}")
                if self._failure_retry_seconds is not None:
                    self._schedule_retry(provider)
            except _TTS_REGISTRY_ADAPTER_EXCEPTIONS as e:
                logger.debug(
                    "Error getting capabilities for {} ({})",
                    provider.value,
                    e.__class__.__name__,
                )
                if self._failure_retry_seconds is not None:
                    self._schedule_retry(provider)

        return capabilities

    async def list_capabilities(self, *, include_disabled: bool = True) -> list[dict[str, Any]]:
        """
        Return standardized capability envelopes for registered TTS providers.

        Envelope shape:
            {"provider": str, "availability": str, "capabilities": Any}
        """
        output: list[dict[str, Any]] = []
        provider_names = self._base.list_providers(include_disabled=include_disabled)

        for provider_name in provider_names:
            status = self._base.get_status(provider_name)
            capabilities: Any = None

            if status == RegistryProviderStatus.ENABLED:
                try:
                    provider_enum = self.resolve_provider(provider_name)
                    adapter = await self.get_adapter(provider_enum or provider_name)
                except _TTS_REGISTRY_ADAPTER_EXCEPTIONS as exc:
                    logger.warning(
                        "Capability discovery failed for provider '{}': {}",
                        provider_name,
                        exc,
                    )
                    adapter = None

                status = self._base.get_status(provider_name)
                if adapter is not None and status == RegistryProviderStatus.ENABLED:
                    try:
                        caps = adapter.capabilities
                        if caps is None:
                            caps = await adapter.get_capabilities()
                        capabilities = caps
                    except _TTS_REGISTRY_ADAPTER_EXCEPTIONS as exc:
                        logger.warning(
                            "Capability fetch failed for provider '{}': {}",
                            provider_name,
                            exc,
                        )

            output.append(
                {
                    "provider": provider_name,
                    "availability": status.value,
                    "capabilities": capabilities,
                }
            )

        return output

    async def find_adapter_for_requirements(
        self,
        language: Optional[str] = None,
        format: Optional[AudioFormat] = None,
        supports_streaming: Optional[bool] = None,
        supports_emotion: Optional[bool] = None,
        supports_voice_cloning: Optional[bool] = None,
        supports_multi_speaker: Optional[bool] = None
    ) -> Optional[TTSAdapter]:
        """
        Find an adapter that meets specific requirements.

        Args:
            language: Required language support
            format: Required audio format
            supports_streaming: Requires streaming support
            supports_emotion: Requires emotion control
            supports_voice_cloning: Requires voice cloning
            supports_multi_speaker: Requires multi-speaker support

        Returns:
            First adapter that meets all requirements, or None
        """
        for provider in self._get_provider_priority():
            adapter = await self.get_adapter(provider)
            if not adapter or not adapter.capabilities:
                continue

            caps = adapter.capabilities

            # Check requirements
            if language and language not in caps.supported_languages:
                continue
            if format and format not in caps.supported_formats:
                continue
            if supports_streaming and not caps.supports_streaming:
                continue
            if supports_emotion is not None and caps.supports_emotion_control != supports_emotion:
                continue
            if supports_voice_cloning is not None and caps.supports_voice_cloning != supports_voice_cloning:
                continue
            if supports_multi_speaker is not None and caps.supports_multi_speaker != supports_multi_speaker:
                continue

            return adapter

        return None

    def _get_provider_priority(self) -> list[TTSProvider]:
        """
        Get provider priority order.
        Can be customized via configuration.

        Returns:
            Ordered list of providers to try
        """
        # Use unified configuration priority
        if self.config_manager:
            priority_names = self.config_manager.get_provider_priority()
        else:
            # Use priority from config if available
            priority_names = self.config.get("provider_priority", [])

        priority = []
        for provider_name in priority_names:
            provider = self.resolve_provider(provider_name)
            if provider is not None:
                priority.append(provider)
            else:
                logger.warning(f"Unknown provider in priority list: {provider_name}")

        # Fallback to default if no valid providers
        if not priority:
            priority = [
                TTSProvider.OPENAI,
                TTSProvider.KOKORO,
                TTSProvider.CHATTERBOX,
                TTSProvider.DIA,
                TTSProvider.HIGGS
            ]

        return priority

    async def close_all(self):
        """Close all initialized adapters and clean up resources"""
        logger.info("Closing all TTS adapters...")

        # Teardown should only clean up an already-initialized manager.
        resource_manager = get_existing_resource_manager()

        tasks = []
        for provider, adapter in self._adapters.items():
            logger.info(f"Closing {provider.value} adapter...")
            tasks.append(adapter.close())

            # Unregister from resource manager if available
            if resource_manager:
                try:
                    await resource_manager.unregister_model(provider.value)
                except _TTS_REGISTRY_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(
                        "Error unregistering {} from resource manager ({})",
                        provider.value,
                        e.__class__.__name__,
                    )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._adapters.clear()
        self._initialized_providers.clear()
        self._base.clear_cache()
        self._base.reset_failures()

        # Clean up resource manager connections
        if resource_manager:
            try:
                await resource_manager.cleanup_all()
            except _TTS_REGISTRY_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(
                    "Error during resource manager cleanup ({})",
                    e.__class__.__name__,
                )

        logger.info("All TTS adapters closed")

    def get_status_summary(self) -> dict[str, Any]:
        """
        Get status summary of all adapters.

        Returns:
            Dictionary with status information
        """
        cached_by_name = self._base.get_cached_adapters()
        self._adapters = {}
        for provider_name, adapter in cached_by_name.items():
            provider_enum = self.resolve_provider(provider_name)
            if provider_enum is not None:
                self._adapters[provider_enum] = adapter
        self._initialized_providers = set(self._adapters.keys())

        summary = {
            "total_providers": len(TTSProvider),
            "initialized": len(self._initialized_providers),
            "available": 0,
            "providers": {}
        }

        for provider in TTSProvider:
            adapter = self._adapters.get(provider)
            if adapter:
                status_value = adapter.status.value
                is_available = adapter.status == ProviderStatus.AVAILABLE
                if is_available:
                    summary["available"] += 1
                provider_info = {
                    "status": status_value,
                    "initialized": bool(getattr(adapter, "_initialized", False)),
                    "failed": self._base.get_status(provider.value) == RegistryProviderStatus.FAILED,
                }
                try:
                    if adapter.capabilities:
                        provider_info["supports_streaming"] = adapter.capabilities.supports_streaming
                        provider_info["supported_formats"] = sorted(fmt.value for fmt in adapter.capabilities.supported_formats)
                        provider_info["sample_rate"] = adapter.capabilities.sample_rate
                except _TTS_REGISTRY_NONCRITICAL_EXCEPTIONS:
                    pass
            else:
                status_value = "not_initialized"
                provider_info = {
                    "status": status_value,
                    "initialized": False,
                    "failed": self._base.get_status(provider.value) == RegistryProviderStatus.FAILED,
                }
            summary["providers"][provider.value] = provider_info

        return summary


class TTSAdapterFactory:
    """
    Factory for creating and managing TTS adapters.
    Provides high-level interface for TTS operations.
    """

    MODEL_PROVIDER_MAP: dict[str, TTSProvider] = {
        # OpenAI models
        "tts-1": TTSProvider.OPENAI,
        "tts-1-hd": TTSProvider.OPENAI,

        # Kokoro models
        "kokoro": TTSProvider.KOKORO,
        "kokoro-v0_19": TTSProvider.KOKORO,
        "kokoro-v1_0": TTSProvider.KOKORO,
        "kokoro-v1.0": TTSProvider.KOKORO,
        "kokoro-1.0": TTSProvider.KOKORO,
        "kokoro-onnx": TTSProvider.KOKORO,
        "onnx-community/kokoro-82m-v1.0-onnx-timestamped": TTSProvider.KOKORO,

        # Higgs models
        "higgs": TTSProvider.HIGGS,
        "higgs-v2": TTSProvider.HIGGS,
        "higgs-audio-v2": TTSProvider.HIGGS,

        # ElevenLabs models
        "elevenlabs": TTSProvider.ELEVENLABS,
        "eleven_monolingual_v1": TTSProvider.ELEVENLABS,
        "eleven_multilingual_v1": TTSProvider.ELEVENLABS,
        "eleven_multilingual_v2": TTSProvider.ELEVENLABS,
        "eleven_turbo_v2": TTSProvider.ELEVENLABS,

        # Dia models
        "dia": TTSProvider.DIA,
        "dia-1.6b": TTSProvider.DIA,

        # Chatterbox models
        "chatterbox": TTSProvider.CHATTERBOX,
        "chatterbox-emotion": TTSProvider.CHATTERBOX,

        # VibeVoice models
        "vibevoice": TTSProvider.VIBEVOICE,
        "vibevoice-1.5b": TTSProvider.VIBEVOICE,
        "vibevoice-7b": TTSProvider.VIBEVOICE,
        "vibevoice-7b-q8": TTSProvider.VIBEVOICE,
        "microsoft/vibevoice-1.5b": TTSProvider.VIBEVOICE,
        # Official 7B repo id
        "vibevoice/vibevoice-7b": TTSProvider.VIBEVOICE,
        # Community 8-bit quantized 7B variant
        "fabiosarracino/vibevoice-large-q8": TTSProvider.VIBEVOICE,
        # VibeVoice Realtime models
        "vibevoice_realtime": TTSProvider.VIBEVOICE_REALTIME,
        "vibevoice-realtime": TTSProvider.VIBEVOICE_REALTIME,
        "vibevoice-realtime-0.5b": TTSProvider.VIBEVOICE_REALTIME,
        "microsoft/vibevoice-realtime-0.5b": TTSProvider.VIBEVOICE_REALTIME,

        # NeuTTS models
        "neutts": TTSProvider.NEUTTS,
        "neutts-air": TTSProvider.NEUTTS,
        "neuphonic/neutts-air": TTSProvider.NEUTTS,
        "neutts-nano": TTSProvider.NEUTTS,
        "neuphonic/neutts-nano": TTSProvider.NEUTTS,
        "neutts-air-q4-gguf": TTSProvider.NEUTTS,
        "neutts-air-q8-gguf": TTSProvider.NEUTTS,
        "neuphonic/neutts-air-q4-gguf": TTSProvider.NEUTTS,
        "neuphonic/neutts-air-q8-gguf": TTSProvider.NEUTTS,
        "neutts-nano-q4-gguf": TTSProvider.NEUTTS,
        "neutts-nano-q8-gguf": TTSProvider.NEUTTS,
        "neuphonic/neutts-nano-q4-gguf": TTSProvider.NEUTTS,
        "neuphonic/neutts-nano-q8-gguf": TTSProvider.NEUTTS,

        # Supertonic models (canonical + aliases)
        "tts-supertonic-1": TTSProvider.SUPERTONIC,
        "supertonic": TTSProvider.SUPERTONIC,
        "supertonic-onnx": TTSProvider.SUPERTONIC,
        # Supertonic2 models (canonical + aliases)
        "tts-supertonic2-1": TTSProvider.SUPERTONIC2,
        "supertonic2": TTSProvider.SUPERTONIC2,
        "supertonic-2": TTSProvider.SUPERTONIC2,
        "supertonic2-onnx": TTSProvider.SUPERTONIC2,

        # PocketTTS ONNX models
        "pocket-tts": TTSProvider.POCKET_TTS,
        "pocket-tts-onnx": TTSProvider.POCKET_TTS,
        "pocket_tts": TTSProvider.POCKET_TTS,
        "pockettts": TTSProvider.POCKET_TTS,
        "pockettts-onnx": TTSProvider.POCKET_TTS,
        "kevinahm/pocket-tts-onnx": TTSProvider.POCKET_TTS,

        # PocketTTS.cpp models
        "pocket_tts_cpp": TTSProvider.POCKET_TTS_CPP,
        "pocket-tts-cpp": TTSProvider.POCKET_TTS_CPP,

        # Echo-TTS models
        "echo-tts": TTSProvider.ECHO_TTS,
        "echo_tts": TTSProvider.ECHO_TTS,
        "jordand/echo-tts-base": TTSProvider.ECHO_TTS,

        # Qwen3-TTS models
        "qwen3-tts": TTSProvider.QWEN3_TTS,
        "qwen3_tts": TTSProvider.QWEN3_TTS,
        "omnivoice": TTSProvider.OMNIVOICE,
        "omni-voice": TTSProvider.OMNIVOICE,
        "omni_voice": TTSProvider.OMNIVOICE,
        "qwen/qwen3-tts-12hz-1.7b-customvoice": TTSProvider.QWEN3_TTS,
        "qwen/qwen3-tts-12hz-0.6b-customvoice": TTSProvider.QWEN3_TTS,
        "qwen/qwen3-tts-12hz-1.7b-voicedesign": TTSProvider.QWEN3_TTS,
        "qwen/qwen3-tts-12hz-1.7b-base": TTSProvider.QWEN3_TTS,
        "qwen/qwen3-tts-12hz-0.6b-base": TTSProvider.QWEN3_TTS,

        # KittenTTS models
        "kitten_tts": TTSProvider.KITTEN_TTS,
        "kitten-tts": TTSProvider.KITTEN_TTS,
        "kittentts": TTSProvider.KITTEN_TTS,
        "kittenml/kitten-tts-mini-0.8": TTSProvider.KITTEN_TTS,
        "kittenml/kitten-tts-micro-0.8": TTSProvider.KITTEN_TTS,
        "kittenml/kitten-tts-nano-0.8": TTSProvider.KITTEN_TTS,
        "kittenml/kitten-tts-nano-0.8-fp32": TTSProvider.KITTEN_TTS,
        "kittenml/kitten-tts-nano-0.8-int8": TTSProvider.KITTEN_TTS,
    }

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize the factory.

        Args:
            config: Configuration for all adapters
        """
        self.registry = TTSAdapterRegistry(config)

    def get_provider_for_model(self, model: Optional[str]) -> Optional[TTSProvider]:
        """
        Resolve which provider should serve a model name.

        Args:
            model: Model identifier from the request

        Returns:
            Matching TTSProvider enum or None
        """
        if not model:
            return None
        key = model.lower()
        provider = self.MODEL_PROVIDER_MAP.get(key)
        if provider is not None:
            return provider
        return self.registry.resolve_provider(model)

    async def get_adapter_by_model(self, model: str) -> Optional[TTSAdapter]:
        """
        Get adapter based on model name.
        Maps model names to providers.

        Args:
            model: Model name (e.g., "tts-1", "kokoro", "higgs")

        Returns:
            Appropriate adapter or None
        """
        provider = self.get_provider_for_model(model)
        if not provider:
            logger.warning(f"Unknown model: {model}")
            return None

        return await self.registry.get_adapter(provider)

    async def get_best_adapter(self, **requirements) -> Optional[TTSAdapter]:
        """
        Get the best adapter for given requirements.

        Args:
            **requirements: Requirements for the adapter

        Returns:
            Best matching adapter or None
        """
        return await self.registry.find_adapter_for_requirements(**requirements)

    async def close(self):
        """Close all adapters"""
        await self.registry.close_all()

    def get_status(self) -> dict[str, Any]:
        """Get factory status"""
        return self.registry.get_status_summary()


# Singleton instance management
_factory_instance: Optional[TTSAdapterFactory] = None
_factory_lock = asyncio.Lock()


async def get_tts_factory(config: Optional[dict[str, Any]] = None) -> TTSAdapterFactory:
    """
    Get or create the TTS adapter factory singleton.

    Args:
        config: Configuration for the factory

    Returns:
        TTSAdapterFactory instance
    """
    global _factory_instance

    if _factory_instance is None:
        async with _factory_lock:
            if _factory_instance is None:
                _factory_instance = TTSAdapterFactory(config)
                logger.info("TTS Adapter Factory initialized")

    return _factory_instance


async def close_tts_factory():
    """Close the TTS factory and all adapters"""
    global _factory_instance

    if _factory_instance:
        await _factory_instance.close()
        _factory_instance = None
        logger.info("TTS Adapter Factory closed")

#
# End of adapter_registry.py
#######################################################################################################################
