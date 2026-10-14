"""
Configuration management for Evaluations module.

Provides dynamic configuration loading and reloading from YAML files,
environment-specific overrides, and runtime configuration updates.
"""

import asyncio
import copy
import hashlib
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from tldw_Server_API.app.core.config import get_config_section
from tldw_Server_API.app.core.config_paths import resolve_module_yaml
from tldw_Server_API.app.core.config_utils import (
    apply_default_sources,
    load_module_yaml,
    merge_config_layers,
    section_to_nested_dict,
)

_EVAL_CONFIG_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    yaml.YAMLError,
)


@dataclass
class RateLimitTierConfig:
    """Rate limit configuration for a user tier."""
    tier: str
    evaluations_per_minute: int
    batch_evaluations_per_minute: int
    evaluations_per_day: int
    total_tokens_per_day: int
    burst_size: int
    description: str = ""

    @classmethod
    def from_dict(cls, tier_name: str, data: dict[str, Any]) -> "RateLimitTierConfig":
        """Create config from dictionary data."""
        return cls(
            tier=tier_name,
            evaluations_per_minute=data.get("evaluations_per_minute", 10),
            batch_evaluations_per_minute=data.get("batch_evaluations_per_minute", 2),
            evaluations_per_day=data.get("evaluations_per_day", 100),
            total_tokens_per_day=data.get("total_tokens_per_day", 100000),
            burst_size=data.get("burst_size", 5),
            description=data.get("description", "")
        )


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration for a provider."""
    provider: str
    failure_threshold: int
    success_threshold: int
    timeout_seconds: float
    recovery_timeout_seconds: float
    expected_exceptions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, provider_name: str, data: dict[str, Any]) -> "CircuitBreakerConfig":
        """Create config from dictionary data."""
        return cls(
            provider=provider_name,
            failure_threshold=data.get("failure_threshold", 5),
            success_threshold=data.get("success_threshold", 2),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            recovery_timeout_seconds=data.get("recovery_timeout_seconds", 60.0),
            expected_exceptions=data.get("expected_exceptions", ["Exception"])
        )


class EvaluationsConfigManager:
    """
    Manages configuration for the Evaluations module.

    Supports dynamic loading, environment overrides, and hot reloading.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        environment: Optional[str] = None,
        enable_hot_reload: bool = False
    ):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file
            environment: Environment name for overrides (dev, test, prod)
            enable_hot_reload: Whether to watch for config file changes
        """
        if config_path is None:
            env_path = os.getenv("EVALUATIONS_CONFIG_PATH")
            config_path = Path(env_path).expanduser() if env_path else resolve_module_yaml("evaluations")
        if config_path is None:
            config_path = Path("evaluations_config.yaml")

        self.config_path = Path(config_path)
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.enable_hot_reload = enable_hot_reload

        # Configuration cache
        self._config: dict[str, Any] = {}
        self._config_sources: dict[str, Any] = {}
        self._config_hash: Optional[str] = None
        self._last_loaded: Optional[datetime] = None
        self._lock = threading.RLock()

        # Parsed configurations
        self._rate_limit_tiers: dict[str, RateLimitTierConfig] = {}
        self._circuit_breaker_configs: dict[str, CircuitBreakerConfig] = {}

        # Hot reload task
        self._reload_task: Optional[asyncio.Task] = None
        self._pending_hot_reload: bool = False

        # Load initial configuration
        self.load_config()

        if enable_hot_reload:
            self._schedule_hot_reload()

    def _schedule_hot_reload(self) -> None:
        """Start hot reload watcher when an event loop is available."""
        if self._reload_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("Evaluations hot reload enabled but no running event loop; deferring watcher startup")
            self._pending_hot_reload = True
            return
        self._reload_task = loop.create_task(self._start_hot_reload())
        self._pending_hot_reload = False

    def ensure_hot_reload_started(self) -> None:
        """Public helper to start hot reload once a loop is running."""
        if not self.enable_hot_reload or self._reload_task is not None:
            return
        if not self._pending_hot_reload:
            return
        self._schedule_hot_reload()

    def load_config(self) -> bool:
        """
        Load configuration from file.

        Returns:
            True if configuration loaded successfully
        """
        try:
            if not self.config_path.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return False

            # Calculate file hash to detect changes
            with open(self.config_path, 'rb') as f:
                content = f.read()
                new_hash = hashlib.md5(content, usedforsecurity=False).hexdigest()

            # Skip if file hasn't changed
            if self._config_hash == new_hash:
                return True

            with self._lock:
                yaml_config, _ = load_module_yaml(
                    "evaluations", filename_override=str(self.config_path)
                )
                if not yaml_config:
                    logger.error("Empty or invalid configuration file")
                    return False

                self._config = yaml_config
                self._config_hash = new_hash
                self._last_loaded = datetime.now(timezone.utc)

                # Apply environment-specific overrides
                self._apply_environment_overrides()

                config_txt_overrides = section_to_nested_dict(
                    get_config_section("Evaluations")
                )
                env_overrides = self._load_env_overrides()
                merged, sources = merge_config_layers(
                    [
                        ("yaml", dict(self._config)),
                        ("config", config_txt_overrides),
                        ("env", env_overrides),
                    ]
                )
                self._config = merged
                self._config_sources = apply_default_sources(self._config, sources)
                logger.debug(f"Evaluations config sources: {self._config_sources}")

                # Parse specialized configurations
                self._parse_rate_limit_configs()
                self._parse_circuit_breaker_configs()

            logger.info(f"Configuration loaded successfully from {self.config_path}")
            logger.info(f"Environment: {self.environment}")
            return True

        except _EVAL_CONFIG_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        environments = self._config.get("environments", {})
        env_config = environments.get(self.environment, {})

        if env_config:
            logger.info(f"Applying {self.environment} environment overrides")
            self._deep_merge(self._config, env_config)

    def _load_env_overrides(self) -> dict[str, Any]:
        """Load env-provided config overrides (YAML/JSON payload)."""
        raw = os.getenv("EVALUATIONS_CONFIG_OVERRIDES")
        if not raw:
            return {}
        try:
            data = yaml.safe_load(raw)
        except _EVAL_CONFIG_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"Invalid EVALUATIONS_CONFIG_OVERRIDES payload: {exc}")
            return {}
        if not isinstance(data, dict):
            logger.warning("EVALUATIONS_CONFIG_OVERRIDES must be a mapping")
            return {}
        return data

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]):
        """Recursively merge override config into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _parse_rate_limit_configs(self):
        """Parse rate limiting tier configurations."""
        rate_config = self._config.get("rate_limiting", {})
        tiers_config = rate_config.get("tiers", {})

        self._rate_limit_tiers = {}
        for tier_name, tier_data in tiers_config.items():
            try:
                tier_config = RateLimitTierConfig.from_dict(tier_name, tier_data)
                self._rate_limit_tiers[tier_name] = tier_config
                removed = [key for key in ("max_cost_per_day", "max_cost_per_month") if key in tier_data]
                if removed:
                    logger.warning(
                        f"Ignoring {', '.join(removed)} in rate limit tier '{tier_name}': evaluation cost caps were "
                        "removed; cap LLM spend with limits.llm_cost_per_day_usd / limits.llm_cost_per_month_usd"
                    )
            except _EVAL_CONFIG_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Invalid rate limit config for tier '{tier_name}': {e}")

        logger.info(f"Loaded {len(self._rate_limit_tiers)} rate limit tier configurations")

    def _parse_circuit_breaker_configs(self):
        """Parse circuit breaker configurations."""
        cb_config = self._config.get("circuit_breakers", {})
        providers_config = cb_config.get("providers", {})

        self._circuit_breaker_configs = {}
        for provider_name, provider_data in providers_config.items():
            try:
                cb_config = CircuitBreakerConfig.from_dict(provider_name, provider_data)
                self._circuit_breaker_configs[provider_name] = cb_config
            except _EVAL_CONFIG_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Invalid circuit breaker config for provider '{provider_name}': {e}")

        logger.info(f"Loaded {len(self._circuit_breaker_configs)} circuit breaker configurations")

    async def _start_hot_reload(self):
        """Start hot reload task to watch for configuration changes."""
        logger.info("Starting configuration hot reload")

        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                if self.config_path.exists():
                    # Check if file has been modified
                    with open(self.config_path, 'rb') as f:
                        content = f.read()
                        current_hash = hashlib.md5(content, usedforsecurity=False).hexdigest()

                    if current_hash != self._config_hash:
                        logger.info("Configuration file changed, reloading...")
                        if self.load_config():
                            logger.info("Configuration reloaded successfully")
                        else:
                            logger.error("Failed to reload configuration")

            except _EVAL_CONFIG_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Hot reload error: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    def get_config(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.

        Args:
            path: Configuration path (e.g., "rate_limiting.global.default_tier")
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        with self._lock:
            current = self._config

            for key in path.split('.'):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default

            return current

    def get_rate_limit_tier_config(self, tier: str) -> Optional[RateLimitTierConfig]:
        """
        Get rate limit configuration for a tier.

        Args:
            tier: Tier name

        Returns:
            Tier configuration or None if not found
        """
        return self._rate_limit_tiers.get(tier)

    def get_all_rate_limit_tiers(self) -> dict[str, RateLimitTierConfig]:
        """Get all rate limit tier configurations."""
        return self._rate_limit_tiers.copy()

    def get_circuit_breaker_config(self, provider: str) -> Optional[CircuitBreakerConfig]:
        """
        Get circuit breaker configuration for a provider.

        Args:
            provider: Provider name

        Returns:
            Circuit breaker configuration or default if not found
        """
        config = self._circuit_breaker_configs.get(provider)
        if config is None:
            # Return default configuration
            default_data = self._config.get("circuit_breakers", {}).get("providers", {}).get("default", {})
            if default_data:
                config = CircuitBreakerConfig.from_dict(provider, default_data)

        return config

    def get_all_circuit_breaker_configs(self) -> dict[str, CircuitBreakerConfig]:
        """Get all circuit breaker configurations."""
        return self._circuit_breaker_configs.copy()

    def update_tier_config(
        self,
        tier: str,
        updates: dict[str, Any],
        persist: bool = False
    ) -> bool:
        """
        Update rate limit tier configuration at runtime.

        Args:
            tier: Tier name
            updates: Configuration updates
            persist: Whether to save changes to file

        Returns:
            True if update successful
        """
        try:
            with self._lock:
                if tier not in self._rate_limit_tiers:
                    logger.error(f"Unknown tier: {tier}")
                    return False

                # Update in-memory configuration
                tier_data = self._config.get("rate_limiting", {}).get("tiers", {}).get(tier, {})
                tier_data.update(updates)

                # Update parsed configuration
                self._rate_limit_tiers[tier] = RateLimitTierConfig.from_dict(tier, tier_data)

                if persist:
                    return self._persist_config()

                logger.info(f"Updated tier '{tier}' configuration: {updates}")
                return True

        except _EVAL_CONFIG_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to update tier configuration: {e}")
            return False

    def _persist_config(self) -> bool:
        """Persist current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration persisted to {self.config_path}")
            return True

        except _EVAL_CONFIG_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to persist configuration: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """
        Get configuration manager status.

        Returns:
            Status information
        """
        with self._lock:
            return {
                "config_path": str(self.config_path),
                "environment": self.environment,
                "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None,
                "hot_reload_enabled": self.enable_hot_reload,
                "config_hash": self._config_hash,
                "rate_limit_tiers": list(self._rate_limit_tiers.keys()),
                "circuit_breaker_providers": list(self._circuit_breaker_configs.keys()),
                "config_sections": list(self._config.keys()) if self._config else []
            }

    def get_config_snapshot(self) -> dict[str, Any]:
        """Return a snapshot of the effective config and source tags."""
        with self._lock:
            return {
                "config": copy.deepcopy(self._config),
                "sources": copy.deepcopy(self._config_sources),
                "config_path": str(self.config_path),
            }

    def reload(self) -> bool:
        """Force reload configuration from file."""
        self._config_hash = None  # Force reload
        return self.load_config()

    def validate_config(self) -> list[str]:
        """
        Validate current configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            # Validate rate limiting configuration
            if "rate_limiting" not in self._config:
                errors.append("Missing 'rate_limiting' section")
            else:
                rate_config = self._config["rate_limiting"]

                if "tiers" not in rate_config:
                    errors.append("Missing 'rate_limiting.tiers' section")
                else:
                    for tier_name, tier_data in rate_config["tiers"].items():
                        if not isinstance(tier_data, dict):
                            errors.append(f"Invalid tier configuration: {tier_name}")
                            continue

                        required_fields = [
                            "evaluations_per_minute", "evaluations_per_day",
                            "total_tokens_per_day"
                        ]

                        for field in required_fields:
                            if field not in tier_data:
                                errors.append(f"Missing field '{field}' in tier '{tier_name}'")
                            elif not isinstance(tier_data[field], (int, float)):
                                errors.append(f"Invalid type for '{field}' in tier '{tier_name}'")

            # Validate circuit breaker configuration
            if "circuit_breakers" in self._config:
                cb_config = self._config["circuit_breakers"]
                if "providers" in cb_config:
                    for provider_name, provider_data in cb_config["providers"].items():
                        if not isinstance(provider_data, dict):
                            errors.append(f"Invalid circuit breaker configuration: {provider_name}")
                            continue

                        required_fields = ["failure_threshold", "timeout_seconds"]
                        for field in required_fields:
                            if field not in provider_data:
                                errors.append(f"Missing field '{field}' in circuit breaker '{provider_name}'")

        except _EVAL_CONFIG_NONCRITICAL_EXCEPTIONS as e:
            errors.append(f"Configuration validation error: {e}")

        return errors


# Global configuration manager instance
config_manager = EvaluationsConfigManager(
    environment=os.getenv("ENVIRONMENT", "development"),
    enable_hot_reload=os.getenv("ENABLE_CONFIG_HOT_RELOAD", "false").lower() == "true"
)

try:
    config_manager.ensure_hot_reload_started()
except _EVAL_CONFIG_NONCRITICAL_EXCEPTIONS:
    # Best-effort: avoid import-time crashes if no loop is running yet
    pass


# Convenience functions
def get_rate_limit_config(tier: str) -> Optional[RateLimitTierConfig]:
    """Get rate limit configuration for a tier."""
    return config_manager.get_rate_limit_tier_config(tier)


def get_circuit_breaker_config(provider: str) -> Optional[CircuitBreakerConfig]:
    """Get circuit breaker configuration for a provider."""
    return config_manager.get_circuit_breaker_config(provider)


def get_config_snapshot() -> dict[str, Any]:
    """Get the current evaluations config and source tags."""
    return config_manager.get_config_snapshot()


def get_config(path: str, default: Any = None) -> Any:
    """Get configuration value by path."""
    return config_manager.get_config(path, default)


def reload_config() -> bool:
    """Reload configuration from file."""
    return config_manager.reload()


def validate_config() -> list[str]:
    """Validate current configuration."""
    return config_manager.validate_config()
