# tts_config.py
# Description: Unified TTS configuration management system
#
# Imports
import os
from pathlib import Path
from typing import Any, Optional

import yaml

#
# Third-party Imports
from loguru import logger
from pydantic import BaseModel, Field

try:
    from pydantic import field_validator
except Exception:
    from pydantic import validator as field_validator  # type: ignore
#
# Local Imports
import contextlib

from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.config_utils import (
    apply_default_sources,
    load_module_yaml,
    merge_config_layers,
)
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

from .utils import parse_bool

#
#######################################################################################################################
#
# TTS Configuration Schema

class ProviderConfig(BaseModel):
    """Configuration for a single TTS provider"""
    enabled: bool = False
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    runtime: Optional[str] = None
    model: Optional[str] = None
    model_revision: Optional[str] = None
    model_path: Optional[str] = None
    binary_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    mlx_model: Optional[str] = None
    capability_override: dict[str, Any] = Field(default_factory=dict)
    device: str = "cpu"
    timeout: int = 60
    max_retries: int = 3
    sample_rate: int = 24000
    max_concurrent_generations: Optional[int] = None
    use_fp16: bool = False
    use_bf16: bool = False
    use_onnx: bool = False
    batch_size: int = 1
    # Qwen3-TTS and tokenizer-related settings (optional)
    tokenizer_model: Optional[str] = None
    dtype: Optional[str] = None
    attn_implementation: Optional[str] = None
    stream_chunk_size_ms: Optional[int] = None
    auto_min_vram_gb: Optional[int] = None
    max_text_length: Optional[int] = None
    tokenizer_max_audio_seconds: Optional[int] = None
    tokenizer_max_tokens: Optional[int] = None
    tokenizer_max_payload_mb: Optional[int] = None
    voice_clone_prompt_max_kb: Optional[int] = None
    enable_voice_cache: bool = False
    cache_ttl_hours: Optional[int] = None
    cache_max_bytes_per_user: Optional[int] = None
    persist_direct_voice_references: bool = False
    # Allow providers (esp. local ones) to declare auto-download behavior
    auto_download: bool = True
    # Optional: for HTTP/API providers like OpenAI, perform a lightweight
    # API-key verification call during adapter.initialize(). This is disabled
    # by default so startup does not depend on external network availability.
    verify_api_key_on_init: bool = False
    # Optional: opt-in text sanitization before validation/generation.
    sanitize_text: bool = False
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator('api_key', mode='before')
    @classmethod
    def resolve_api_key(cls, v):
        """Resolve API key from environment variables"""
        if v and v.startswith('${') and v.endswith('}'):
            env_var = v[2:-1]
            return os.getenv(env_var)
        return v


class VoiceMappingConfig(BaseModel):
    """Voice mapping configuration"""
    generic: dict[str, dict[str, str]] = Field(default_factory=dict)
    emotions: dict[str, dict[str, str]] = Field(default_factory=dict)


class PerformanceConfig(BaseModel):
    """Performance settings"""
    max_concurrent_generations: int = 4
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600
    stream_chunk_size: int = 1024
    model_cache_max_entries: Optional[int] = None
    # Compatibility flag: when true, embed error messages as audio bytes in streams.
    # Default is False so APIs surface structured HTTP errors instead of "ERROR: ..." audio.
    # Set to true only if you explicitly rely on error-as-audio semantics.
    stream_errors_as_audio: bool = False
    token_estimation_enabled: bool = True
    token_estimate_per_char: float = 2.5
    token_estimate_safety: float = 1.3
    token_estimate_min_tokens: int = 256
    max_new_tokens_cap: int = 4096
    min_new_tokens_default: int = 60
    memory_warning_threshold: int = 80
    memory_critical_threshold: int = 90
    max_connections_per_provider: int = 5
    connection_timeout: float = 30.0
    # If set, failed provider initializations will be retried after this many seconds.
    # When unset/None, retries are disabled and providers are skipped for the process lifetime.
    adapter_failure_retry_seconds: Optional[float] = None


class FallbackConfig(BaseModel):
    """Fallback settings"""
    enabled: bool = True
    max_attempts: int = 3
    retry_delay_ms: int = 1000
    exclude_providers: list[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    """Logging settings"""
    level: str = "INFO"
    log_requests: bool = True
    log_responses: bool = False
    log_performance_metrics: bool = True


class TTSConfig(BaseModel):
    """Complete TTS configuration"""
    provider_priority: list[str] = Field(default_factory=lambda: ["openai", "kokoro"])
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    voice_mappings: VoiceMappingConfig = Field(default_factory=VoiceMappingConfig)
    format_preferences: dict[str, list[str]] = Field(default_factory=dict)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # Global input validation behavior; controls strict sanitization in TTSInputValidator.
    # When true (default), dangerous patterns raise errors; when false, they are stripped.
    strict_validation: bool = True

    # Settings from config.txt
    default_provider: Optional[str] = None
    default_voice: Optional[str] = None
    default_speed: float = 1.0
    local_device: str = "cpu"


class TTSConfigManager:
    """
    Manages TTS configuration from multiple sources.
    Priority order: Environment variables > config.txt > YAML > defaults
    """

    LEGACY_TTS_SETTINGS_ALIASES: dict[str, str] = {
        "default_tts_provider": "default_provider",
        "default_tts_voice": "default_voice",
        "default_tts_speed": "default_speed",
        "local_tts_device": "local_device",
        "tts_device": "local_device",
    }
    LEGACY_TTS_SETTINGS_REMOVAL_DATE = "2026-06-30"

    def __init__(self,
                 yaml_path: Optional[Path] = None,
                 config_txt_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            yaml_path: Path to YAML configuration file
            config_txt_path: Path to config.txt file
        """
        self.yaml_path = Path(yaml_path).expanduser() if yaml_path else None
        self.config_txt_path = Path(config_txt_path).expanduser() if config_txt_path else None
        self._config: Optional[TTSConfig] = None
        self._env_overrides: dict[str, Any] = {}
        self._sources: dict[str, Any] = {}

        # Load configurations
        self.reload()

    def reload(self):
        """Reload configuration from all sources"""
        yaml_override = str(self.yaml_path) if self.yaml_path else None
        yaml_config, yaml_path = load_module_yaml("tts", filename_override=yaml_override)
        if self.yaml_path is None:
            self.yaml_path = yaml_path

        cfg_txt = self._load_config_txt()
        cfg_env = self._load_env_overrides()

        merged, sources = merge_config_layers(
            [
                ("yaml", yaml_config),
                ("config", cfg_txt),
                ("env", cfg_env),
            ]
        )

        self._config = TTSConfig(**merged)
        sources = apply_default_sources(model_dump_compat(self._config), sources)
        self._sources = sources
        logger.info(f"TTS configuration loaded with {len(self._config.providers)} providers")
        logger.debug(f"TTS config sources: {sources}")

    @staticmethod
    def _get_first_present(
        section: Any,
        keys: tuple[str, ...],
    ) -> Any:
        """Return the first present key value from a config section."""
        for key in keys:
            if key in section:
                return section[key]
        return None

    def _warn_legacy_tts_setting_keys(self, section: Any) -> None:
        """Emit deprecation warnings for legacy [TTS-Settings] keys."""
        for legacy_key, canonical_key in self.LEGACY_TTS_SETTINGS_ALIASES.items():
            if legacy_key not in section:
                continue
            if canonical_key in section:
                logger.warning(
                    f"[TTS-Settings] '{legacy_key}' is deprecated and ignored when '{canonical_key}' is also set. "
                    f"Use '{canonical_key}'. Legacy aliases are scheduled for removal after "
                    f"{self.LEGACY_TTS_SETTINGS_REMOVAL_DATE}."
                )
                continue
            logger.warning(
                f"[TTS-Settings] '{legacy_key}' is deprecated; use '{canonical_key}'. "
                f"Legacy aliases are scheduled for removal after {self.LEGACY_TTS_SETTINGS_REMOVAL_DATE}."
            )

    def _load_config_txt(self) -> dict[str, Any]:
        """Load settings from config.txt"""
        config_dict = {}

        try:
            if self.config_txt_path:
                if not self.config_txt_path.exists():
                    logger.warning(f"Config.txt override not found at {self.config_txt_path}")
                    return {}
                import configparser

                config = configparser.ConfigParser()
                config.read(self.config_txt_path)
            else:
                config = load_comprehensive_config()

            # Check for TTS-Settings section
            tts_section = None
            if hasattr(config, "has_section") and config.has_section("TTS-Settings"):
                tts_section = config["TTS-Settings"]
            elif isinstance(config, dict):
                tts_section = config.get("TTS-Settings")

            if tts_section:
                self._warn_legacy_tts_setting_keys(tts_section)
                # Map config.txt settings to our schema
                # Deterministic alias precedence: canonical key first, then legacy key.
                provider_value = self._get_first_present(
                    tts_section,
                    ("default_provider", "default_tts_provider"),
                )
                if provider_value is not None:
                    config_dict['default_provider'] = provider_value

                voice_value = self._get_first_present(
                    tts_section,
                    ("default_voice", "default_tts_voice"),
                )
                if voice_value is not None:
                    config_dict['default_voice'] = voice_value

                speed_value = self._get_first_present(
                    tts_section,
                    ("default_speed", "default_tts_speed"),
                )
                if speed_value is not None:
                    with contextlib.suppress(ValueError):
                        config_dict['default_speed'] = float(speed_value)

                device_value = self._get_first_present(
                    tts_section,
                    ("local_device", "local_tts_device", "tts_device"),
                )
                if device_value is not None:
                    config_dict['local_device'] = device_value

                    # Apply device setting to local providers
                    if 'providers' not in config_dict:
                        config_dict['providers'] = {}

                    for provider in [
                        'kokoro',
                        'kitten_tts',
                        'higgs',
                        'dia',
                        'chatterbox',
                        'vibevoice',
                        'vibevoice_realtime',
                        'neutts',
                        'pocket_tts_cpp',
                        'lux_tts',
                    ]:
                        if provider not in config_dict['providers']:
                            config_dict['providers'][provider] = {}
                        config_dict['providers'][provider]['device'] = device_value

                # Global switch: auto download local models
                if "auto_download_local_models" in tts_section:
                    auto_dl = parse_bool(tts_section['auto_download_local_models'], default=False)
                    if 'providers' not in config_dict:
                        config_dict['providers'] = {}
                    for provider in [
                        'kokoro',
                        'kitten_tts',
                        'higgs',
                        'dia',
                        'chatterbox',
                        'vibevoice',
                        'vibevoice_realtime',
                        'neutts',
                        'pocket_tts_cpp',
                        'lux_tts',
                    ]:
                        config_dict['providers'].setdefault(provider, {})['auto_download'] = auto_dl

                # Provider-specific auto-download toggles
                def _bool_from_section(key: str) -> Optional[bool]:
                    if key in tts_section:
                        return parse_bool(tts_section[key], default=False)
                    return None

                for prov, key in (
                    ('vibevoice', 'vibevoice_auto_download'),
                    ('vibevoice_realtime', 'vibevoice_realtime_auto_download'),
                    ('kokoro', 'kokoro_auto_download'),
                    ('dia', 'dia_auto_download'),
                    ('higgs', 'higgs_auto_download'),
                    ('chatterbox', 'chatterbox_auto_download'),
                    ('neutts', 'neutts_auto_download'),
                    ('lux_tts', 'lux_tts_auto_download'),
                ):
                    bv = _bool_from_section(key)
                    if bv is not None:
                        if 'providers' not in config_dict:
                            config_dict['providers'] = {}
                        config_dict['providers'].setdefault(prov, {})['auto_download'] = bv

                # Optional: global strict_validation toggle
                sv = _bool_from_section("strict_validation")
                if sv is not None:
                    config_dict['strict_validation'] = sv

            # Check for API keys in main section
            if hasattr(config, "sections"):
                sections = config.sections()
            elif isinstance(config, dict):
                sections = list(config.keys())
            else:
                sections = []

            for section_name in sections:
                if section_name == "TTS-Settings":
                    continue
                section = config[section_name] if isinstance(config, dict) or hasattr(config, "__getitem__") else {}
                if not section:
                    continue
                # Look for API keys
                if "openai_api_key" in section:
                    config_dict.setdefault("providers", {}).setdefault("openai", {})[
                        "api_key"
                    ] = section["openai_api_key"]

                if "elevenlabs_api_key" in section:
                    config_dict.setdefault("providers", {}).setdefault("elevenlabs", {})[
                        "api_key"
                    ] = section["elevenlabs_api_key"]

            return config_dict

        except Exception:
            logger.error("Error loading config.txt")
            return {}

    def _load_env_overrides(self) -> dict[str, Any]:
        """Load environment variable overrides"""
        config_dict = {}

        # Check for provider API keys
        api_key_env_vars = {
            'OPENAI_API_KEY': ('openai', 'api_key'),
            'ELEVENLABS_API_KEY': ('elevenlabs', 'api_key'),
            'ANTHROPIC_API_KEY': ('anthropic', 'api_key'),
        }

        for env_var, (provider, field) in api_key_env_vars.items():
            value = os.getenv(env_var)
            if value:
                if 'providers' not in config_dict:
                    config_dict['providers'] = {}
                if provider not in config_dict['providers']:
                    config_dict['providers'][provider] = {}
                config_dict['providers'][provider][field] = value

        # Check for default settings
        if os.getenv('TTS_DEFAULT_PROVIDER'):
            config_dict['default_provider'] = os.getenv('TTS_DEFAULT_PROVIDER')

        if os.getenv('TTS_DEFAULT_VOICE'):
            config_dict['default_voice'] = os.getenv('TTS_DEFAULT_VOICE')

        if os.getenv('TTS_DEVICE'):
            config_dict['local_device'] = os.getenv('TTS_DEVICE')

        # Global strict validation toggle for TTS input sanitization
        if os.getenv('TTS_STRICT_VALIDATION') is not None:
            config_dict['strict_validation'] = parse_bool(
                os.getenv('TTS_STRICT_VALIDATION'),
                default=True,
            )

        return config_dict

    def get_config(self) -> TTSConfig:
        """Get the current configuration"""
        if self._config is None:
            self.reload()
        return self._config

    def get_sources(self) -> dict[str, Any]:
        """Return source tags for the current configuration."""
        if not self._sources:
            self.reload()
        return dict(self._sources)

    def get_provider_config(self, provider: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider"""
        config = self.get_config()
        return config.providers.get(provider)

    def is_provider_enabled(self, provider: str) -> bool:
        """Check if a provider is enabled"""
        provider_config = self.get_provider_config(provider)
        return provider_config.enabled if provider_config else False

    def get_enabled_providers(self) -> list[str]:
        """Get list of enabled providers"""
        config = self.get_config()
        return [
            name for name, cfg in config.providers.items()
            if cfg.enabled
        ]

    def get_provider_priority(self) -> list[str]:
        """Get provider priority order"""
        config = self.get_config()
        # Filter to only enabled providers
        enabled = self.get_enabled_providers()
        return [p for p in config.provider_priority if p in enabled]

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""
        cfg = self.get_config()
        return model_dump_compat(cfg)

    def save_yaml(self, path: Optional[Path] = None):
        """Save current configuration to YAML file"""
        path = path or self.yaml_path
        if not path:
            raise ValueError("No YAML path specified")

        config_dict = self.to_dict()

        # Convert ProviderConfig objects to dicts
        if 'providers' in config_dict:
            for provider_name in config_dict['providers']:
                if isinstance(config_dict['providers'][provider_name], ProviderConfig):
                    cfg = config_dict['providers'][provider_name]
                    config_dict['providers'][provider_name] = model_dump_compat(cfg)

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved TTS configuration to {path}")


# Singleton instance
_config_manager: Optional[TTSConfigManager] = None


def get_tts_config() -> TTSConfig:
    """Get the global TTS configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = TTSConfigManager()
    return _config_manager.get_config()


def get_tts_config_manager() -> TTSConfigManager:
    """Get the global TTS configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = TTSConfigManager()
    return _config_manager


def reload_tts_config():
    """Reload TTS configuration from all sources"""
    manager = get_tts_config_manager()
    manager.reload()
    logger.info("TTS configuration reloaded")
