"""
Secure configuration management for unified MCP module

All sensitive configuration is loaded from environment variables or secure config files.
No hardcoded secrets allowed.
"""

import json
import os
import secrets
from functools import lru_cache
from ipaddress import ip_address, ip_network
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import Field, SecretStr

try:
    from pydantic import field_validator  # v2
except (ImportError, AttributeError):  # v1 fallback
    from pydantic import validator as field_validator  # type: ignore
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode

_MCP_CONFIG_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


def _default_ws_allowed_origins() -> list[str]:
    """Safe local-only defaults for WS Origin allowlist.

    Use exact Origin values commonly seen in local development to satisfy
    production validation without accidentally exposing the server.
    """
    return [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]


def _default_allowed_ips() -> list[str]:
    """Restrict MCP to loopback by default.

    This keeps the surface local-only unless explicitly configured via env.
    """
    return ["127.0.0.1", "::1"]


def _parse_env_list(value: Any) -> Any:
    """Parse env string into a list of strings for list-typed settings."""
    if value is None:
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return value


def _is_loopback_origin(origin: str) -> bool:
    """Return True when origin points to a loopback host."""
    if not origin:
        return False

    raw = origin.strip()
    if not raw or raw == "*":
        return False

    candidate = raw if "://" in raw else f"http://{raw}"
    try:
        parsed = urlparse(candidate)
    except _MCP_CONFIG_NONCRITICAL_EXCEPTIONS:
        return False

    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host == "localhost":
        return True

    try:
        return ip_address(host).is_loopback
    except _MCP_CONFIG_NONCRITICAL_EXCEPTIONS:
        return False


def _is_loopback_ip_entry(entry: str) -> bool:
    """Return True when an IP/CIDR entry is loopback-only."""
    if not entry:
        return False

    token = entry.strip()
    if not token:
        return False

    network = token
    if "/" not in token:
        network = f"{token}/128" if ":" in token else f"{token}/32"
    try:
        return ip_network(network, strict=False).is_loopback
    except _MCP_CONFIG_NONCRITICAL_EXCEPTIONS:
        return False


def _is_local_only_safe_profile(config: "MCPConfig") -> bool:
    """
    Check whether MCP is constrained to a local-only safe profile.

    This profile is intended for first-run development:
    - Loopback-only IP allowlist
    - Localhost/loopback WS + CORS origins
    - No trust of forwarded headers
    """
    allowed_ips = list(config.allowed_client_ips or [])
    if not allowed_ips:
        return False
    if not all(_is_loopback_ip_entry(entry) for entry in allowed_ips):
        return False

    ws_origins = list(config.ws_allowed_origins or [])
    if not ws_origins:
        return False
    if not all(_is_loopback_origin(origin) for origin in ws_origins):
        return False

    cors_origins = list(config.cors_origins or [])
    if not cors_origins:
        return False
    if not all(_is_loopback_origin(origin) for origin in cors_origins):
        return False

    return not bool(config.trust_x_forwarded_for)


class MCPConfig(BaseSettings):
    """
    MCP configuration with secure defaults and environment variable support.

    All sensitive values MUST come from environment variables or secure storage.
    """

    # Server Configuration
    server_name: str = Field(default="tldw-mcp-unified", validation_alias="MCP_SERVER_NAME")
    server_version: str = Field(default="3.0.0", validation_alias="MCP_SERVER_VERSION")
    debug_mode: bool = Field(default=False, validation_alias="MCP_DEBUG")

    # Security Configuration - CRITICAL: No hardcoded secrets!
    jwt_secret_key: Optional[SecretStr] = Field(default=None, validation_alias="MCP_JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", validation_alias="MCP_JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, validation_alias="MCP_JWT_ACCESS_EXPIRE")
    jwt_refresh_token_expire_days: int = Field(default=7, validation_alias="MCP_JWT_REFRESH_EXPIRE")

    # Protocol validation & policy
    validate_input_schema: bool = Field(default=True, validation_alias="MCP_VALIDATE_INPUT_SCHEMA")
    disable_write_tools: bool = Field(default=False, validation_alias="MCP_DISABLE_WRITE_TOOLS")
    # Idempotency (protocol-level) for write tools
    idempotency_ttl_seconds: int = Field(default=300, validation_alias="MCP_IDEMPOTENCY_TTL_SECONDS")
    idempotency_cache_size: int = Field(default=512, validation_alias="MCP_IDEMPOTENCY_CACHE_SIZE")

    # API Key Configuration
    api_key_salt: Optional[SecretStr] = Field(default=None, validation_alias="MCP_API_KEY_SALT")
    api_key_iterations: int = Field(default=100000, validation_alias="MCP_API_KEY_ITERATIONS")

    # Database Configuration
    database_url: str = Field(
        default="sqlite+aiosqlite:///./Databases/mcp_unified.db",
        validation_alias="MCP_DATABASE_URL"
    )
    database_pool_size: int = Field(default=20, validation_alias="MCP_DATABASE_POOL_SIZE")
    database_pool_timeout: int = Field(default=30, validation_alias="MCP_DATABASE_POOL_TIMEOUT")
    database_pool_recycle: int = Field(default=3600, validation_alias="MCP_DATABASE_POOL_RECYCLE")
    database_echo: bool = Field(default=False, validation_alias="MCP_DATABASE_ECHO")

    # Redis Configuration (Optional - for distributed deployments)
    redis_url: Optional[str] = Field(default=None, validation_alias="MCP_REDIS_URL")
    redis_password: Optional[SecretStr] = Field(default=None, validation_alias="MCP_REDIS_PASSWORD")
    redis_ssl: bool = Field(default=False, validation_alias="MCP_REDIS_SSL")
    redis_pool_size: int = Field(default=10, validation_alias="MCP_REDIS_POOL_SIZE")

    # Rate Limiting Configuration
    rate_limit_enabled: bool = Field(default=True, validation_alias="MCP_RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(default=60, validation_alias="MCP_RATE_LIMIT_RPM")
    rate_limit_burst_size: int = Field(default=10, validation_alias="MCP_RATE_LIMIT_BURST")

    # WebSocket Configuration
    ws_max_connections: int = Field(default=1000, validation_alias="MCP_WS_MAX_CONNECTIONS")
    ws_max_connections_per_ip: int = Field(default=10, validation_alias="MCP_WS_MAX_CONNECTIONS_PER_IP")
    ws_max_message_size: int = Field(default=1048576, validation_alias="MCP_WS_MAX_MESSAGE_SIZE")  # 1MB
    ws_ping_interval: int = Field(default=30, validation_alias="MCP_WS_PING_INTERVAL")
    ws_ping_timeout: int = Field(default=60, validation_alias="MCP_WS_PING_TIMEOUT")
    ws_close_timeout: int = Field(default=10, validation_alias="MCP_WS_CLOSE_TIMEOUT")
    ws_auth_required: bool = Field(default=True, validation_alias="MCP_WS_AUTH_REQUIRED")
    # WS security
    ws_allowed_origins: list[str] = Field(default_factory=_default_ws_allowed_origins, validation_alias="MCP_WS_ALLOWED_ORIGINS")
    ws_allow_query_auth: bool = Field(default=False, validation_alias="MCP_WS_ALLOW_QUERY_AUTH")
    # WS session policies
    ws_idle_timeout_seconds: int = Field(default=300, validation_alias="MCP_WS_IDLE_TIMEOUT_SECONDS")
    ws_session_rate_limit_count: int = Field(default=120, validation_alias="MCP_WS_SESSION_RATE_COUNT")
    ws_session_rate_limit_window_seconds: int = Field(default=60, validation_alias="MCP_WS_SESSION_RATE_WINDOW_SECONDS")

    # Network access controls
    # Default to loopback-only to avoid accidental exposure
    allowed_client_ips: list[str] = Field(default_factory=_default_allowed_ips, validation_alias="MCP_ALLOWED_IPS")
    blocked_client_ips: list[str] = Field(default_factory=list, validation_alias="MCP_BLOCKED_IPS")
    trust_x_forwarded_for: bool = Field(default=False, validation_alias="MCP_TRUST_X_FORWARDED")
    trusted_proxy_depth: int = Field(default=1, ge=0, validation_alias="MCP_TRUSTED_PROXY_DEPTH")
    trusted_proxy_ips: list[str] = Field(default_factory=list, validation_alias="MCP_TRUSTED_PROXY_IPS")

    # HTTP request limits
    http_max_body_bytes: int = Field(default=524288, validation_alias="MCP_HTTP_MAX_BODY_BYTES")  # 512 KiB default

    # Client certificate / mTLS hooks
    client_cert_required: bool = Field(default=False, validation_alias="MCP_CLIENT_CERT_REQUIRED")
    client_cert_header: Optional[str] = Field(default=None, validation_alias="MCP_CLIENT_CERT_HEADER")
    client_cert_header_value: Optional[str] = Field(default=None, validation_alias="MCP_CLIENT_CERT_HEADER_VALUE")
    client_cert_ca_bundle: Optional[str] = Field(default=None, validation_alias="MCP_CLIENT_CA_BUNDLE")

    # CORS Configuration
    cors_enabled: bool = Field(default=True, validation_alias="MCP_CORS_ENABLED")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        validation_alias="MCP_CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, validation_alias="MCP_CORS_CREDENTIALS")
    cors_allow_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        validation_alias="MCP_CORS_METHODS"
    )
    cors_allow_headers: list[str] = Field(
        default=["*"],
        validation_alias="MCP_CORS_HEADERS"
    )

    # Security Headers
    security_headers_enabled: bool = Field(default=True, validation_alias="MCP_SECURITY_HEADERS")
    csp_policy: str = Field(
        default="default-src 'self'; script-src 'self' 'unsafe-inline'",
        validation_alias="MCP_CSP_POLICY"
    )
    hsts_max_age: int = Field(default=31536000, validation_alias="MCP_HSTS_MAX_AGE")  # 1 year

    # Module Configuration
    module_timeout: int = Field(default=30, validation_alias="MCP_MODULE_TIMEOUT")
    module_max_retries: int = Field(default=3, validation_alias="MCP_MODULE_MAX_RETRIES")
    module_health_check_interval: int = Field(default=60, validation_alias="MCP_MODULE_HEALTH_INTERVAL")

    # Monitoring Configuration
    metrics_enabled: bool = Field(default=True, validation_alias="MCP_METRICS_ENABLED")
    metrics_port: int = Field(default=9090, validation_alias="MCP_METRICS_PORT")
    health_check_path: str = Field(default="/health", validation_alias="MCP_HEALTH_PATH")

    # Session Configuration
    session_ttl_minutes: int = Field(default=30, validation_alias="MCP_SESSION_TTL_MINUTES")
    max_sessions: int = Field(default=100, validation_alias="MCP_MAX_SESSIONS")
    max_session_uris: int = Field(default=500, validation_alias="MCP_MAX_SESSION_URIS")

    # Logging Configuration
    log_level: str = Field(default="INFO", validation_alias="MCP_LOG_LEVEL")
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        validation_alias="MCP_LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default=None, validation_alias="MCP_LOG_FILE")
    log_rotation: str = Field(default="100 MB", validation_alias="MCP_LOG_ROTATION")
    log_retention: str = Field(default="30 days", validation_alias="MCP_LOG_RETENTION")

    # Audit Logging
    audit_enabled: bool = Field(default=True, validation_alias="MCP_AUDIT_ENABLED")
    audit_log_file: str = Field(default="audit.log", validation_alias="MCP_AUDIT_LOG_FILE")

    # Rate limit categories (tool → category) mapping
    # Provide either JSON via MCP_TOOL_CATEGORY_MAP or file path via MCP_TOOL_CATEGORY_MAP_FILE
    tool_category_map: dict[str, str] = Field(default_factory=dict, validation_alias="MCP_TOOL_CATEGORY_MAP")
    tool_category_map_file: Optional[str] = Field(default=None, validation_alias="MCP_TOOL_CATEGORY_MAP_FILE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        enable_decoding=False,
        extra="ignore",
    )

    @field_validator("jwt_secret_key", mode="before")
    @classmethod
    def validate_jwt_secret(cls, v):
        """Ensure JWT secret is set and strong enough"""
        if not v:
            # Generate a secure random secret if not provided
            logger.warning("JWT secret not provided, generating a random one (not suitable for production)")
            return SecretStr(secrets.token_urlsafe(32))

        if isinstance(v, str):
            if len(v) < 32:
                raise ValueError("JWT secret must be at least 32 characters long")

            if v == "your-secret-key-change-this-in-production":
                raise ValueError("Default JWT secret detected! Please set MCP_JWT_SECRET environment variable")

            return SecretStr(v)

        return v

    @field_validator("api_key_salt", mode="before")
    @classmethod
    def validate_api_key_salt(cls, v):
        """Ensure API key salt is set and secure"""
        if not v:
            logger.warning("API key salt not provided, generating a random one")
            return SecretStr(secrets.token_urlsafe(32))

        if isinstance(v, str):
            if len(v) < 32:
                raise ValueError("API key salt must be at least 32 characters long")

            return SecretStr(v)

        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string"""
        return _parse_env_list(v)

    @field_validator("ws_allowed_origins", mode="before")
    @classmethod
    def parse_ws_allowed_origins(cls, v):
        """Parse WS allowed origins from comma-separated string."""
        return _parse_env_list(v)

    @field_validator("allowed_client_ips", "blocked_client_ips", "trusted_proxy_ips", mode="before")
    @classmethod
    def parse_ip_lists(cls, v):
        """Parse comma-separated IP/CIDR lists."""
        return _parse_env_list(v)

    @field_validator("cors_allow_methods", "cors_allow_headers", mode="before")
    @classmethod
    def parse_cors_list_fields(cls, v):
        """Parse comma-separated or JSON list values for CORS list fields."""
        return _parse_env_list(v)

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v):
        """Validate and potentially modify database URL"""
        if "sqlite" in v and not v.startswith("sqlite+aiosqlite"):
            # Ensure async SQLite driver is used
            v = v.replace("sqlite://", "sqlite+aiosqlite://")
        return v

    @field_validator("tool_category_map", mode="before")
    @classmethod
    def parse_tool_category_map(cls, v):
        """Allow JSON string in env for tool→category map."""
        if not v:
            return {}
        if isinstance(v, str):
            try:
                import json as _json
                data = _json.loads(v)
                return data if isinstance(data, dict) else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                return {}
        return v

    def get_redis_connection_params(self) -> Optional[dict[str, Any]]:
        """Get Redis connection parameters if Redis is configured"""
        if not self.redis_url:
            return None

        params = {
            "url": self.redis_url,
            "encoding": "utf-8",
            "decode_responses": True,
            "max_connections": self.redis_pool_size,
        }

        if self.redis_password:
            params["password"] = self.redis_password.get_secret_value()

        if self.redis_ssl:
            params["ssl"] = True
            params["ssl_cert_reqs"] = "required"

        return params

    def get_database_connection_params(self) -> dict[str, Any]:
        """Get database connection parameters"""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": 20,
            "pool_timeout": self.database_pool_timeout,
            "pool_recycle": self.database_pool_recycle,
            "echo": self.database_echo,
        }

    def configure_logging(self):
        """Configure logging using a safe, non-colorized formatter.

        Avoids angle-bracket color tags to prevent Colorizer errors when fields
        such as function may contain characters like '<module>'.
        """
        try:
            # Optional opt-out to inherit global logger configuration
            import os as _os
            if env_flag_enabled("MCP_INHERIT_GLOBAL_LOGGER"):
                return
        except _MCP_CONFIG_NONCRITICAL_EXCEPTIONS:
            pass

        # Placeholder-based format template (no color, safe for braces in messages)
        _fmt_template = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - {message}"
        )

        try:
            logger.remove()  # Reset to avoid duplicate/default handlers
        except (RuntimeError, ValueError):
            pass

        # Console logging (safe format, no color)
        logger.add(
            sink=os.sys.stderr,
            format=_fmt_template,
            level=self.log_level,
            colorize=False,
        )

        # File logging if configured (safe format)
        if self.log_file:
            logger.add(
                sink=self.log_file,
                format=_fmt_template,
                level=self.log_level,
                rotation=self.log_rotation,
                retention=self.log_retention,
                compression="zip",
            )

        # Audit logging if enabled (plain format)
        if self.audit_enabled:
            logger.add(
                sink=self.audit_log_file,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | AUDIT | {message}",
                level="INFO",
                filter=lambda record: "audit" in record["extra"],
                rotation="1 day",
                retention="90 days",
                compression="zip",
            )


@lru_cache
def get_config() -> MCPConfig:
    """Get cached configuration instance"""
    try:
        config = MCPConfig()
        config.configure_logging()
        # Load tool category map from YAML file if provided
        try:
            if config.tool_category_map_file:
                import os as _os  # type: ignore

                import yaml as _yaml
                if _os.path.exists(config.tool_category_map_file):
                    with open(config.tool_category_map_file) as f:
                        data = _yaml.safe_load(f) or {}
                    if isinstance(data, dict):
                        # Expect top-level mapping { tool_name: category }
                        for k, v in data.items():
                            if isinstance(k, str) and isinstance(v, str):
                                config.tool_category_map[k] = v
        except _MCP_CONFIG_NONCRITICAL_EXCEPTIONS as _e:
            logger.warning(f"Failed to load tool category map file: {_e}")
        logger.info("MCP configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(
            "Failed to load configuration ({})",
            type(e).__name__,
        )
        raise


def validate_config() -> bool:
    """Validate configuration on startup"""
    try:
        config = get_config()
        test_mode = False
        try:
            test_mode = (
                is_test_mode()
                or bool(os.getenv("PYTEST_CURRENT_TEST"))
            )
        except _MCP_CONFIG_NONCRITICAL_EXCEPTIONS:
            test_mode = False

        def _field_was_set(cfg: MCPConfig, field_name: str) -> bool:
            try:
                return field_name in cfg.model_fields_set  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                return field_name in getattr(cfg, "__fields_set__", set())

        # Check critical security settings
        if not config.jwt_secret_key:
            logger.error("JWT secret key not configured!")
            return False

        if not config.api_key_salt:
            logger.error("API key salt not configured!")
            return False

        # In non-debug/non-test contexts, allow generated secrets only when
        # MCP is constrained to local-only safe defaults.
        if not config.debug_mode and not test_mode:
            jwt_secret_explicit = _field_was_set(config, "jwt_secret_key")
            api_salt_explicit = _field_was_set(config, "api_key_salt")
            if not (jwt_secret_explicit and api_salt_explicit):
                if _is_local_only_safe_profile(config):
                    logger.warning(
                        "MCP_JWT_SECRET/MCP_API_KEY_SALT are not explicitly set; "
                        "using process-local generated secrets under local-only safe profile. "
                        "Set explicit secrets for shared or production deployments."
                    )
                else:
                    if not jwt_secret_explicit:
                        logger.error("MCP_JWT_SECRET must be set explicitly when MCP is not local-only")
                    if not api_salt_explicit:
                        logger.error("MCP_API_KEY_SALT must be set explicitly when MCP is not local-only")
                    return False

        # Validate database connection
        if not config.database_url:
            logger.error("Database URL not configured!")
            return False

        # Harden WebSocket + mTLS settings for production
        if not config.debug_mode:
            # WS must require auth
            if not config.ws_auth_required:
                logger.error("In production, MCP WebSocket authentication must be enabled (MCP_WS_AUTH_REQUIRED=true)")
                return False
            # WS must define allowed origins
            if not config.ws_allowed_origins:
                logger.error("In production, MCP WebSocket allowed origins must be configured (MCP_WS_ALLOWED_ORIGINS)")
                return False
            # If client certs are required, an explicit expected value must be set
            if config.client_cert_required and not (config.client_cert_header_value and config.client_cert_header_value.strip()):
                logger.error("Client certificate required but MCP_CLIENT_CERT_HEADER_VALUE not set")
                return False

            # Warn about other dev-leaning settings
            if "localhost" in config.cors_origins or "*" in config.cors_origins:
                logger.warning("Localhost or wildcard in CORS origins - not recommended for production")
            if config.database_echo:
                logger.warning("Database echo enabled - not recommended for production")

        logger.info("Configuration validation passed")
        return True

    except _MCP_CONFIG_NONCRITICAL_EXCEPTIONS as e:
        logger.error(
            "Configuration validation failed ({})",
            type(e).__name__,
        )
        return False
