"""
Secure secret management for production deployments.

Provides centralized secret management with environment variable validation,
secret rotation capabilities, and secure storage for API keys, tokens,
and other sensitive configuration data.

Integrates with existing tldw config system and prepares for aiosqlite migration.
"""

import base64
import os
import secrets
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# Import existing config system
from tldw_Server_API.app.core.config import load_comprehensive_config


class SecretType(Enum):
    """Types of secrets managed by the system."""
    API_KEY = "api_key"
    JWT_SECRET = "jwt_secret"  # nosec B105
    DATABASE_PASSWORD = "database_password"  # nosec B105
    WEBHOOK_SECRET = "webhook_secret"  # nosec B105
    ENCRYPTION_KEY = "encryption_key"
    OAUTH_SECRET = "oauth_secret"  # nosec B105
    CUSTOM = "custom"


class SecretSource(Enum):
    """Sources for secret retrieval."""
    ENVIRONMENT = "environment"
    CONFIG_FILE = "config_file"
    VAULT = "vault"  # For future integration
    DEFAULT = "default"


@dataclass
class SecretConfig:
    """Configuration for a secret."""
    name: str
    secret_type: SecretType
    env_var: Optional[str] = None
    config_section: Optional[str] = None
    config_key: Optional[str] = None
    required: bool = True
    default_value: Optional[str] = None
    min_length: int = 8
    rotation_days: Optional[int] = None
    description: str = ""


@dataclass
class SecretValue:
    """A secret value with metadata."""
    value: str
    source: SecretSource
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    rotation_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class SecretValidationError(Exception):
    """Raised when secret validation fails."""
    pass


class SecretManager:
    """
    Centralized secret management system.

    Provides secure storage, retrieval, and validation of secrets
    using the existing tldw config system.
    """

    def __init__(self, validate_on_startup: bool = True):
        """
        Initialize secret manager using existing tldw config system.

        Args:
            validate_on_startup: Whether to validate all secrets on startup
        """
        self._secrets_cache: dict[str, SecretValue] = {}

        # Load .env file if it exists
        self._load_env_file()

        self._config = load_comprehensive_config()

        # Define standard secret configurations
        self._secret_configs = self._init_secret_configs()

        if validate_on_startup:
            self._validate_required_secrets()

    def _load_env_file(self):
        """Load .env file from Config_Files directory if it exists."""
        # Note: .env loading is now handled by config.py's load_comprehensive_config()
        # This method is kept for backward compatibility but doesn't duplicate loading
        try:
            current_file_path = Path(__file__).resolve()
            project_root = current_file_path.parent.parent.parent.parent  # Up to tldw_Server_API
            env_path = project_root / 'Config_Files' / '.env'

            if env_path.exists():
                # Don't load again if already loaded by config.py
                # Just log that we found it
                logger.debug(f"SecretManager: .env file found at {env_path} (already loaded by config.py)")
            else:
                logger.debug(f"SecretManager: No .env file found at {env_path}")
        except Exception:
            logger.warning("SecretManager: Error checking .env file")

    def _get_config_value(self, section: str, key: str, fallback: Optional[str] = None) -> Optional[str]:
        """Get value from existing tldw config system."""
        if not self._config:
            return fallback

        try:
            return self._config.get(section, key, fallback=fallback)
        except Exception:
            return fallback

    def _init_secret_configs(self) -> dict[str, SecretConfig]:
        """Initialize standard secret configurations."""
        return {
            # Authentication secrets (using existing config sections)
            "single_user_api_key": SecretConfig(
                name="single_user_api_key",
                secret_type=SecretType.API_KEY,
                env_var="SINGLE_USER_API_KEY",
                config_section="API",
                config_key="api_bearer",  # Existing key in config
                required=True,
                min_length=24,
                rotation_days=365,
                description="API key for single-user authentication mode"
            ),

            # LLM API keys (using existing config sections)
            "openai_api_key": SecretConfig(
                name="openai_api_key",
                secret_type=SecretType.API_KEY,
                env_var="OPENAI_API_KEY",
                config_section="API-Keys",
                config_key="openai_api_key",
                required=False,
                min_length=20,
                description="OpenAI API key for evaluations"
            ),

            "anthropic_api_key": SecretConfig(
                name="anthropic_api_key",
                secret_type=SecretType.API_KEY,
                env_var="ANTHROPIC_API_KEY",
                config_section="API-Keys",
                config_key="anthropic_api_key",
                required=False,
                min_length=20,
                description="Anthropic API key for evaluations"
            ),

            "cohere_api_key": SecretConfig(
                name="cohere_api_key",
                secret_type=SecretType.API_KEY,
                env_var="COHERE_API_KEY",
                config_section="API-Keys",
                config_key="cohere_api_key",
                required=False,
                min_length=20,
                description="Cohere API key for evaluations"
            ),

            "groq_api_key": SecretConfig(
                name="groq_api_key",
                secret_type=SecretType.API_KEY,
                env_var="GROQ_API_KEY",
                config_section="API-Keys",
                config_key="groq_api_key",
                required=False,
                min_length=20,
                description="Groq API key for evaluations"
            )
        }


    def get_secret(
        self,
        name: str,
        required: Optional[bool] = None,
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Retrieve a secret value.

        Args:
            name: Secret name
            required: Override required flag
            default: Override default value

        Returns:
            Secret value or None if not found and not required

        Raises:
            SecretValidationError: If required secret is missing or invalid
        """
        # Check cache first
        if name in self._secrets_cache:
            cached = self._secrets_cache[name]
            if not cached.expires_at or cached.expires_at > datetime.now(timezone.utc):
                return cached.value

        # Get configuration
        config = self._secret_configs.get(name)
        if not config:
            # Handle ad-hoc secret requests
            config = SecretConfig(
                name=name,
                secret_type=SecretType.CUSTOM,
                env_var=name.upper(),
                required=required if required is not None else False,
                default_value=default
            )
        elif required is not None or default is not None:
            config = replace(
                config,
                required=required if required is not None else config.required,
                default_value=default if default is not None else config.default_value,
            )

        # Try to retrieve secret from various sources
        secret_value = None
        source = SecretSource.DEFAULT

        # 1. Environment variable
        if config.env_var:
            env_value = os.getenv(config.env_var)
            if env_value:
                secret_value = env_value
                source = SecretSource.ENVIRONMENT

        # 2. Configuration file (using existing tldw config system)
        if not secret_value and config.config_section and config.config_key:
            config_value = self._get_config_value(config.config_section, config.config_key)
            if config_value:
                secret_value = config_value
                source = SecretSource.CONFIG_FILE

        # 3. Default value
        if not secret_value and config.default_value:
            secret_value = config.default_value
            source = SecretSource.DEFAULT

        # Validate secret
        if not secret_value:
            if config.required:
                raise SecretValidationError(f"Required secret '{name}' not found")
            return None

        # Validate secret format and length
        if len(secret_value) < config.min_length:
            if config.required:
                raise SecretValidationError(
                    f"Secret '{name}' too short. Minimum length: {config.min_length}"
                )
            logger.warning(f"Secret '{name}' is shorter than recommended ({config.min_length} chars)")

        # Additional validation based on secret type
        self._validate_secret_format(config, secret_value)

        # Cache the secret
        expires_at = None
        if config.rotation_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=config.rotation_days)

        cached_secret = SecretValue(
            value=secret_value,
            source=source,
            expires_at=expires_at,
            metadata={"config": config.name, "type": config.secret_type.value}
        )

        self._secrets_cache[name] = cached_secret

        # Log secret retrieval (without the actual value)
        logger.info(f"Retrieved secret '{name}' from {source.value}")

        return secret_value

    def _validate_secret_format(self, config: SecretConfig, value: str):
        """Validate secret format based on type."""
        if config.secret_type == SecretType.API_KEY:
            # Basic API key validation
            if not value.replace('-', '').replace('_', '').isalnum():
                logger.warning(f"Secret '{config.name}' contains unexpected characters")

        elif config.secret_type == SecretType.JWT_SECRET:
            # JWT secrets should be high entropy
            if len(set(value)) < len(value) * 0.7:  # At least 70% unique characters
                logger.warning(f"JWT secret '{config.name}' may have low entropy")

        elif config.secret_type == SecretType.WEBHOOK_SECRET:
            # Webhook secrets should be hex or base64
            try:
                if len(value) % 2 == 0:
                    int(value, 16)  # Try hex
                else:
                    base64.b64decode(value)  # Try base64
            except ValueError:
                logger.warning(f"Webhook secret '{config.name}' should be hex or base64")

    def _validate_required_secrets(self):
        """Validate all required secrets on startup."""
        missing_secrets = []
        invalid_secrets = []

        for name, config in self._secret_configs.items():
            if not config.required:
                continue

            try:
                value = self.get_secret(name)
                if not value:
                    missing_secrets.append(name)
            except SecretValidationError as e:
                invalid_secrets.append(f"{name}: {e}")

        if missing_secrets or invalid_secrets:
            error_msg = "Secret validation failed:\n"
            if missing_secrets:
                error_msg += f"Missing required secrets: {', '.join(missing_secrets)}\n"
            if invalid_secrets:
                error_msg += f"Invalid secrets: {'; '.join(invalid_secrets)}\n"

            logger.error(error_msg)
            raise SecretValidationError(error_msg)

        logger.info("All required secrets validated successfully")

    def set_secret(
        self,
        name: str,
        value: str,
        expires_days: Optional[int] = None
    ):
        """
        Store a secret value in cache.

        Args:
            name: Secret name
            value: Secret value
            expires_days: Days until expiration
        """
        expires_at = None
        if expires_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)

        cached_secret = SecretValue(
            value=value,
            source=SecretSource.VAULT,  # Indicates programmatically set
            expires_at=expires_at,
            metadata={"cached": True}
        )

        self._secrets_cache[name] = cached_secret
        logger.info(f"Cached secret '{name}'")

    def generate_secret(
        self,
        name: str,
        length: int = 32,
        alphabet: str = None
    ) -> str:
        """
        Generate a new random secret.

        Args:
            name: Secret name
            length: Secret length
            alphabet: Character set to use

        Returns:
            Generated secret
        """
        if alphabet is None:
            # Use URL-safe alphabet by default
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"

        secret_value = ''.join(secrets.choice(alphabet) for _ in range(length))

        # Store the generated secret
        self.set_secret(name, secret_value)

        logger.info(f"Generated new secret '{name}' ({length} characters)")
        return secret_value

    def rotate_secret(self, name: str) -> str:
        """
        Rotate a secret by generating a new value.

        Args:
            name: Secret name to rotate

        Returns:
            New secret value
        """
        config = self._secret_configs.get(name)
        if not config:
            raise SecretValidationError(f"Unknown secret '{name}' cannot be rotated")

        # Generate new secret with appropriate length
        min_length = max(config.min_length, 32)
        new_value = self.generate_secret(name, min_length)

        logger.warning(f"Rotated secret '{name}' - update environment/config!")
        return new_value

    def list_secrets(self, include_values: bool = False) -> dict[str, dict[str, Any]]:
        """
        List all configured secrets with metadata.

        Args:
            include_values: Whether to include actual values (dangerous!)

        Returns:
            Dictionary of secret metadata
        """
        secrets_info = {}

        for name, config in self._secret_configs.items():
            cached = self._secrets_cache.get(name)

            info = {
                "type": config.secret_type.value,
                "required": config.required,
                "description": config.description,
                "min_length": config.min_length,
                "rotation_days": config.rotation_days,
                "has_value": bool(self.get_secret(name, required=False)),
                "source": cached.source.value if cached else None,
                "expires_at": cached.expires_at.isoformat() if cached and cached.expires_at else None,
                "rotation_required": cached.rotation_required if cached else False
            }

            if include_values:
                # Only include values for non-sensitive debugging
                value = self.get_secret(name, required=False)
                if value and len(value) > 10:
                    # Mask long values for security
                    info["value_preview"] = value[:4] + "..." + value[-4:]
                else:
                    info["value_preview"] = "***"

            secrets_info[name] = info

        return secrets_info

    def get_production_health_check(self) -> dict[str, Any]:
        """
        Get production readiness check for secrets.

        Returns:
            Health check results
        """
        health = {
            "status": "healthy",
            "issues": [],
            "recommendations": [],
            "secret_count": len(self._secret_configs),
            "required_secrets_ok": True,
            "config_available": self._config is not None,
            "rotation_warnings": []
        }

        try:
            # Check required secrets
            for name, config in self._secret_configs.items():
                if not config.required:
                    continue

                try:
                    value = self.get_secret(name, required=False)
                    if not value:
                        health["issues"].append(f"Required secret '{name}' is missing")
                        health["required_secrets_ok"] = False
                    elif len(value) < config.min_length:
                        health["issues"].append(f"Secret '{name}' is too short")
                    elif value == config.default_value:
                        health["recommendations"].append(f"Secret '{name}' using default value - should be changed for production")
                except Exception:
                    health["issues"].append(f"Error checking secret '{name}'")

            # Check for rotation warnings
            for name, cached in self._secrets_cache.items():
                if cached.expires_at and cached.expires_at < datetime.now(timezone.utc) + timedelta(days=7):
                    health["rotation_warnings"].append(f"Secret '{name}' expires soon: {cached.expires_at}")

            # Overall status
            if health["issues"]:
                health["status"] = "unhealthy"
            elif health["recommendations"] or health["rotation_warnings"]:
                health["status"] = "warning"

        except Exception:
            health["status"] = "error"
            health["issues"].append("Health check failed")

        return health

    def clear_cache(self):
        """Clear the secrets cache."""
        self._secrets_cache.clear()
        logger.info("Secrets cache cleared")


# Global secret manager instance
secret_manager = SecretManager(validate_on_startup=False)


# Convenience functions for common operations
def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a specific provider.

    Priority order:
    1. Environment variable (e.g., OPENAI_API_KEY)
    2. Secret manager cache
    3. Config file
    """
    # First check environment variable
    env_key = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(env_key)

    if api_key and api_key not in ['<' + provider + '_api_key>', '', 'your-api-key-here', 'None']:
        return api_key

    # Fall back to secret manager
    return secret_manager.get_secret(f"{provider}_api_key")


def get_auth_secret(key_type: str = "single_user") -> str:
    """Get authentication secret."""
    if key_type == "single_user":
        return secret_manager.get_secret("single_user_api_key", required=True)
    elif key_type == "jwt":
        return secret_manager.get_secret("jwt_secret_key", required=True)
    else:
        raise ValueError(f"Unknown auth key type: {key_type}")


def get_webhook_secret() -> str:
    """Get webhook signing secret."""
    return secret_manager.get_secret("webhook_master_secret", required=True)


def validate_production_secrets() -> bool:
    """Validate all production secrets are properly configured."""
    try:
        secret_manager._validate_required_secrets()
        return True
    except SecretValidationError:
        return False
