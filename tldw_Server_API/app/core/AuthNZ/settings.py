# settings.py
# Description: Pydantic settings for user registration system with persistent JWT secret management
#
from __future__ import annotations
# Imports
import contextlib
import json
import os
from pathlib import Path
from typing import Annotated, Literal, Optional

#
# Local imports
from loguru import logger
from pydantic import Field, field_validator

#
# 3rd-party imports
from pydantic_settings import BaseSettings, NoDecode

_SETTINGS_IMPORT_EXCEPTIONS = (
    ImportError,
    RuntimeError,
    AttributeError,
)

_SETTINGS_JSON_PARSE_EXCEPTIONS = (
    json.JSONDecodeError,
    TypeError,
    ValueError,
)

_SETTINGS_CAST_EXCEPTIONS = (
    TypeError,
    ValueError,
    AttributeError,
)

_SETTINGS_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ImportError,
)

try:
    # Prefer centralized loader to honor project config precedence
    from tldw_Server_API.app.core.config import get_tldw_env_file_path
    from tldw_Server_API.app.core.config import load_comprehensive_config
    from tldw_Server_API.app.core.config import settings as core_settings
except _SETTINGS_IMPORT_EXCEPTIONS:
    get_tldw_env_file_path = None  # Fallback if import graph changes
    load_comprehensive_config = None  # Fallback if import graph changes
    core_settings = None

try:
    from tldw_Server_API.app.core.testing import (
        is_truthy as _is_truthy,
        is_explicit_pytest_runtime as _is_explicit_pytest_runtime,
    )
except _SETTINGS_IMPORT_EXCEPTIONS:
    def _is_truthy(value: str | None) -> bool:
        s = str(value or "").strip().lower()
        return s == "1" or s == "true" or s == "yes" or s == "y" or s == "on"

    def _is_explicit_pytest_runtime() -> bool:
        return bool(os.getenv("PYTEST_CURRENT_TEST"))

SECURE_KEY_INIT_COMMAND = "python -m tldw_Server_API.app.core.AuthNZ.initialize"
SECURE_KEY_GUIDANCE = f"Generate a secure key via:\n  {SECURE_KEY_INIT_COMMAND}"
SINGLE_USER_KEY_MISSING = (
    "SINGLE_USER_API_KEY is required for single-user mode but is not configured.\n"
    f"{SECURE_KEY_GUIDANCE}\n"
    "and follow the prompts (option \"Generate secure keys\").\n"
    "Then set SINGLE_USER_API_KEY in your environment or .env file."
)
SINGLE_USER_KEY_DEFAULT = (
    "Default API key detected! Please set SINGLE_USER_API_KEY via environment or .env.\n"
    f"{SECURE_KEY_GUIDANCE}"
)
SINGLE_USER_KEY_TOO_SHORT = (
    "SINGLE_USER_API_KEY must be at least 16 characters.\n"
    f"{SECURE_KEY_GUIDANCE}"
)
SINGLE_USER_KEY_PRODUCTION_FORMAT = (
    "In production (tldw_production=true), SINGLE_USER_API_KEY must use the "
    "server-generated format (tldw_<kid>.<secret>). "
    f"{SECURE_KEY_GUIDANCE}\n"
    "or set TLDW_ALLOW_LEGACY_SINGLE_USER_KEY=true to temporarily bypass."
)
SINGLE_USER_KEY_PRODUCTION_WEAK = (
    "In production (tldw_production=true), SINGLE_USER_API_KEY must be set to a secure value (>=24 chars) "
    f"and must not use defaults.\n{SECURE_KEY_GUIDANCE}"
)
SINGLE_USER_API_KEY_PLACEHOLDERS = {
    "CHANGE_ME_TO_SECURE_API_KEY",
    "default-secret-key-for-single-user",
    "change-me-in-production",
    "CHANGE-ME-to-a-secure-key-at-least-16-chars",
}
def _authnz_default_env_file() -> Path:
    if get_tldw_env_file_path:
        explicit_env_file = get_tldw_env_file_path()
        if explicit_env_file:
            return explicit_env_file
    return Path(__file__).resolve().parents[3] / "Config_Files" / ".env"


AUTHNZ_DEFAULT_ENV_FILE = _authnz_default_env_file()
ENTERPRISE_SUPPORTED_PROFILES = {
    "enterprise",
    "enterprise-postgres",
    "enterprise_postgres",
    "multi-user-postgres",
    "multi_user_postgres",
}

#######################################################################################################################
#
# Settings Class

class Settings(BaseSettings):
    """Configuration with persistent secret management for user registration system"""

    # ===== Core Settings =====
    PROFILE: Optional[str] = Field(
        default=None,
        description=(
            "Deployment profile hint (e.g., local-single-user, multi-user-postgres). "
            "AUTH_MODE remains the canonical switch for behavior; PROFILE is "
            "used for coordination/UX, feature gating, and future drift reduction. "
            "Callers may use PROFILE to apply additional *restrictions* in certain "
            "flows (for example, disabling self-registration in local-single-user "
            "deployments), but it must never be used to bypass or relax auth "
            "decisions relative to AUTH_MODE and claims."
        ),
    )

    AUTH_MODE: Literal["single_user", "multi_user"] = Field(
        default="single_user",
        description="Authentication mode: single_user (API key) or multi_user (JWT)"
    )

    DATABASE_URL: str = Field(
        default="sqlite:///./Databases/users.db",
        description="Database URL - PostgreSQL for multi-user: postgresql://user:pass@localhost/tldw"
    )

    # ===== JWT Settings with secure storage =====
    JWT_SECRET_KEY: Optional[str] = Field(
        default=None,
        description="JWT signing key - MUST be set via environment variable in production"
    )

    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )

    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        description="Refresh token expiration time in days"
    )

    MAGIC_LINK_EXPIRE_MINUTES: int = Field(
        default=15,
        description="Magic link token expiration time in minutes"
    )

    PUBLIC_WEB_BASE_URL: Optional[str] = Field(
        default=None,
        description="Public web application base URL used for hosted auth and billing links"
    )

    PUBLIC_PASSWORD_RESET_PATH: str = Field(
        default="/auth/reset-password",
        description="Public hosted path for password reset completion"
    )

    PUBLIC_EMAIL_VERIFICATION_PATH: str = Field(
        default="/auth/verify-email",
        description="Public hosted path for email verification completion"
    )

    PUBLIC_MAGIC_LINK_PATH: str = Field(
        default="/auth/magic-link",
        description="Public hosted path for magic-link verification"
    )

    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    JWT_PRIVATE_KEY: Optional[str] = Field(
        default=None,
        description="PEM-encoded private key for asymmetric JWT signing (RS256/ES256)"
    )
    JWT_PUBLIC_KEY: Optional[str] = Field(
        default=None,
        description="PEM-encoded public key for asymmetric JWT verification (RS256/ES256)"
    )
    JWT_SECONDARY_SECRET: Optional[str] = Field(
        default=None,
        description="Optional secondary HS secret for dual-validation during rotations"
    )
    JWT_SECONDARY_PRIVATE_KEY: Optional[str] = Field(
        default=None,
        description="Optional secondary private key (RS/ES) for legacy secret derivation during rotations"
    )
    JWT_SECONDARY_PUBLIC_KEY: Optional[str] = Field(
        default=None,
        description="Optional secondary public key (RS/ES) for dual-validation during rotations"
    )

    # Optional JWT claims enforcement (recommended in production)
    JWT_ISSUER: Optional[str] = Field(
        default=None,
        description="Expected JWT issuer (iss). If set, tokens must include matching iss"
    )
    JWT_AUDIENCE: Optional[str] = Field(
        default=None,
        description="Expected JWT audience (aud). If set, tokens must include matching aud"
    )

    # ===== Password Settings =====
    PASSWORD_MIN_LENGTH: int = Field(
        default=10,
        ge=8,
        description="Minimum password length"
    )

    ARGON2_MEMORY_COST: int = Field(
        default=32768,  # 32MB
        description="Argon2 memory cost parameter"
    )

    ARGON2_TIME_COST: int = Field(
        default=2,
        description="Argon2 time cost (iterations)"
    )

    ARGON2_PARALLELISM: int = Field(
        default=1,
        description="Argon2 parallelism factor"
    )

    # ===== Redis Configuration (Optional) =====
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis URL for session caching (optional)"
    )

    REDIS_MAX_CONNECTIONS: int = Field(
        default=50,
        description="Maximum Redis connections"
    )

    # ===== Security Settings =====
    ENABLE_REGISTRATION: bool = Field(
        default=False,
        description="Enable user registration"
    )

    REQUIRE_REGISTRATION_CODE: bool = Field(
        default=True,
        description="Require registration code for new users"
    )

    ENABLE_ORG_SCOPED_REGISTRATION_CODES: bool = Field(
        default=False,
        description="Allow registration codes to auto-assign org/team membership"
    )

    ORG_INVITE_ALLOW_MISSING_EMAIL: bool = Field(
        default=False,
        description="Allow org invite redemption when user email is missing"
    )

    MAX_LOGIN_ATTEMPTS: int = Field(
        default=5,
        ge=3,
        description="Maximum login attempts before lockout"
    )

    LOCKOUT_DURATION_MINUTES: int = Field(
        default=15,
        ge=5,
        description="Account lockout duration in minutes"
    )

    # ===== Storage Settings =====
    DEFAULT_STORAGE_QUOTA_MB: int = Field(
        default=5120,  # 5GB
        ge=100,
        description="Default storage quota in MB"
    )

    USER_DATA_BASE_PATH: str = Field(
        default="./user_databases",
        description="Base path for user data directories"
    )

    CHROMADB_BASE_PATH: str = Field(
        default="",
        description="Base path for ChromaDB data"
    )

    # ===== Registration Settings =====
    DEFAULT_USER_ROLE: str = Field(
        default="user",
        description="Default role for new users"
    )

    SINGLE_USER_DEFAULT_PERMISSIONS: list[str] = Field(
        default_factory=lambda: [
            "system.configure",
            "media.read",
            "media.create",
            "media.update",
            "media.delete",
            "claims.review",
            "claims.admin",
            "moderation.review.read",
            "moderation.review.decide",
            "moderation.review.bulk_decide",
            "moderation.audit.read",
        ],
        description="Default permissions granted to the single-user principal in single_user mode",
    )

    REGISTRATION_CODE_DEFAULT_EXPIRY_DAYS: int = Field(
        default=7,
        ge=1,
        description="Default registration code expiry in days"
    )

    REGISTRATION_CODE_DEFAULT_MAX_USES: int = Field(
        default=1,
        ge=1,
        description="Default maximum uses for registration code"
    )

    MAX_BATCH_REGISTRATION_CODES: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum registration codes per batch"
    )

    # ===== Database Pool Settings =====
    DATABASE_POOL_MIN_SIZE: int = Field(
        default=5,
        ge=1,
        description="Minimum database pool size"
    )

    DATABASE_POOL_MAX_SIZE: int = Field(
        default=20,
        ge=5,
        description="Maximum database pool size"
    )

    DATABASE_MAX_QUERIES: int = Field(
        default=50000,
        description="Maximum queries before connection reset"
    )

    DATABASE_MAX_INACTIVE_CONNECTION_LIFETIME: int = Field(
        default=300,
        description="Maximum inactive connection lifetime in seconds"
    )

    # ===== Session Settings =====
    SESSION_COOKIE_SECURE: bool = Field(
        default=True,
        description="Use secure cookies (HTTPS only)"
    )

    SESSION_COOKIE_HTTPONLY: bool = Field(
        default=True,
        description="HTTP-only cookies (no JavaScript access)"
    )

    SESSION_COOKIE_SAMESITE: str = Field(
        default="lax",
        description="SameSite cookie attribute"
    )

    SESSION_CLEANUP_INTERVAL_HOURS: int = Field(
        default=1,
        ge=1,
        description="Session cleanup interval in hours"
    )

    # ===== Encryption Settings =====
    SESSION_ENCRYPTION_KEY: Optional[str] = Field(
        default=None,
        description="Base64-encoded Fernet key for session token encryption (auto-generated if not set)"
    )

    API_KEY_PEPPER: Optional[str] = Field(
        default=None,
        description="Additional secret for API key hashing (recommended for production)"
    )

    # Optional: log a lightweight 'used' entry when an API key validates
    API_KEY_AUDIT_LOG_USAGE: bool = Field(
        default=False,
        description="If true, log a 'used' event in API key audit log on successful validation"
    )

    # ===== BYOK Settings =====
    BYOK_ENABLED: bool = Field(
        default=True,
        description="Enable per-user BYOK keys (requires BYOK_ENCRYPTION_KEY)"
    )
    BYOK_ALLOWED_PROVIDERS: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Optional allowlist of providers eligible for BYOK (comma-separated or list)"
    )
    BYOK_ALLOWED_BASE_URL_PROVIDERS: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Optional allowlist of providers that may set base_url in BYOK credential_fields"
    )
    BYOK_ENCRYPTION_KEY: Optional[str] = Field(
        default=None,
        description="Base64-encoded 32-byte key for BYOK secret encryption (AES-GCM)"
    )
    BYOK_SECONDARY_ENCRYPTION_KEY: Optional[str] = Field(
        default=None,
        description="Secondary BYOK encryption key for dual-read during rotations"
    )
    OPENAI_OAUTH_ENABLED: bool = Field(
        default=False,
        description="Enable OpenAI OAuth account-linking for BYOK users",
    )
    OPENAI_OAUTH_CLIENT_ID: Optional[str] = Field(
        default=None,
        description="OpenAI OAuth client ID",
    )
    OPENAI_OAUTH_CLIENT_SECRET: Optional[str] = Field(
        default=None,
        description="OpenAI OAuth client secret",
    )
    OPENAI_OAUTH_AUTH_URL: Optional[str] = Field(
        default=None,
        description="OpenAI OAuth authorize URL",
    )
    OPENAI_OAUTH_TOKEN_URL: Optional[str] = Field(
        default=None,
        description="OpenAI OAuth token URL",
    )
    OPENAI_OAUTH_SCOPES: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="OAuth scopes requested during OpenAI account linking",
    )
    OPENAI_OAUTH_STATE_TTL_MINUTES: int = Field(
        default=10,
        ge=1,
        le=60,
        description="OAuth state TTL in minutes for OpenAI account linking",
    )
    OPENAI_OAUTH_REFRESH_SKEW_SECONDS: int = Field(
        default=120,
        ge=0,
        le=3600,
        description="Proactive refresh lead time in seconds for OpenAI OAuth access tokens",
    )
    OPENAI_OAUTH_REDIRECT_URI: Optional[str] = Field(
        default=None,
        description="Optional fixed callback URI override for OpenAI OAuth flow",
    )
    OPENAI_OAUTH_ALLOWED_RETURN_PATH_PREFIXES: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["/"],
        description="Allowed app-relative return path prefixes for OpenAI OAuth flow",
    )
    OPENAI_OAUTH_MAX_OUTSTANDING_STATES: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum outstanding OpenAI OAuth state records per user/provider",
    )
    OPENAI_OAUTH_REFRESH_LOCK_BACKEND: Literal["memory", "redis", "db"] = Field(
        default="memory",
        description="Backend used for OpenAI OAuth refresh locking",
    )

    # ===== Enterprise Federation / MCP Broker =====
    AUTH_FEDERATION_ENABLED: bool = Field(
        default=False,
        description=(
            "Enable enterprise OIDC federation features. Supported only in "
            "multi-user PostgreSQL deployments."
        ),
    )
    MCP_CREDENTIAL_BROKER_ENABLED: bool = Field(
        default=False,
        description=(
            "Enable brokered MCP credential resolution. Supported only in "
            "multi-user PostgreSQL deployments and requires secret backends."
        ),
    )
    SECRET_BACKENDS_ENABLED: bool = Field(
        default=False,
        description=(
            "Enable managed secret-backend references. Supported only in "
            "multi-user PostgreSQL deployments."
        ),
    )

    # ===== Token Rotation =====
    ROTATE_REFRESH_TOKENS: bool = Field(
        default=True,
        description="Rotate refresh tokens on use (recommended). Returns new refresh token in /auth/refresh"
    )

    # ===== Logging / PII =====
    PII_REDACT_LOGS: bool = Field(
        default=True,
        description="Redact usernames/IPs in auth logs (recommended in production)"
    )

    # ===== CSRF Binding (Optional) =====
    CSRF_BIND_TO_USER: bool = Field(
        default=False,
        description="Bind CSRF token to user context via HMAC when user_id available"
    )

    # ===== RBAC / Usage Logging =====
    RBAC_SOFT_ENFORCE: bool = Field(
        default=False,
        description="If true, permission checks log warnings instead of 403 (deployment opt-in)"
    )
    ORG_RBAC_PROPAGATION_ENABLED: bool = Field(
        default=False,
        description="Enable org/team role-to-permission propagation for scoped RBAC"
    )
    ORG_RBAC_SCOPE_MODE: Literal["union", "active_only", "require_active"] = Field(
        default="require_active",
        description="Scoped permission mode: union, active_only, or require_active"
    )
    ORG_RBAC_SCOPED_PERMISSION_DENYLIST: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "system.",
            "users.",
            "api.",
            "security.",
            "billing.",
            "monitoring.",
            "maintenance.",
            "admin.",
            "workflows.admin",
            "embeddings.admin",
            "flashcards.admin",
            "claims.admin",
        ],
        description="Permission prefixes/names that cannot be granted via org/team roles"
    )
    USAGE_LOG_ENABLED: bool = Field(
        default=False,
        description="If true, record lightweight per-request usage into usage_log"
    )
    USAGE_LOG_EXCLUDE_PREFIXES: list[str] = Field(
        default_factory=lambda: [
            "/docs", "/redoc", "/openapi.json", "/metrics", "/static", "/favicon.ico"
        ],
        description="Request path prefixes to exclude from usage logging"
    )
    USAGE_AGGREGATOR_INTERVAL_MINUTES: int = Field(
        default=60,
        ge=1,
        le=24 * 60,
        description="Background usage aggregator interval in minutes"
    )
    # Disable capturing IP/User-Agent in usage_log.meta entirely
    USAGE_LOG_DISABLE_META: bool = Field(
        default=False,
        description="When true, do not store IP/User-Agent in usage_log meta (stores '{}')"
    )

    # ===== LLM Usage Logging =====
    LLM_USAGE_ENABLED: bool = Field(
        default=True,
        description="If true, record per-request LLM usage into llm_usage_log (can be overridden by env)"
    )

    # ===== LLM Usage Aggregation =====
    LLM_USAGE_AGGREGATOR_ENABLED: bool = Field(
        default=True,
        description="Enable background aggregation of llm_usage_log into llm_usage_daily"
    )
    LLM_USAGE_AGGREGATOR_INTERVAL_MINUTES: int = Field(
        default=60,
        ge=1,
        le=24 * 60,
        description="Background LLM usage aggregator interval in minutes"
    )

    # ===== Usage Log Retention =====
    USAGE_LOG_RETENTION_DAYS: int = Field(
        default=180,
        ge=1,
        le=3650,
        description="Retention window for usage_log rows (days)"
    )
    LLM_USAGE_LOG_RETENTION_DAYS: int = Field(
        default=180,
        ge=1,
        le=3650,
        description="Retention window for llm_usage_log rows (days)"
    )

    # Optional: retention for daily aggregates
    USAGE_DAILY_RETENTION_DAYS: int = Field(
        default=365,
        ge=1,
        le=3650,
        description="Retention window for usage_daily rows (days)"
    )
    LLM_USAGE_DAILY_RETENTION_DAYS: int = Field(
        default=365,
        ge=1,
        le=3650,
        description="Retention window for llm_usage_daily rows (days)"
    )
    PRIVILEGE_SNAPSHOT_RETENTION_DAYS: int = Field(
        default=90,
        ge=7,
        le=3650,
        description="Days to retain privilege snapshots at full fidelity before downsampling"
    )
    PRIVILEGE_SNAPSHOT_WEEKLY_RETENTION_DAYS: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Days to retain downsampled (weekly) privilege snapshots before purging"
    )

    # ===== Virtual Keys / LLM Budgeting =====
    VIRTUAL_KEYS_ENABLED: bool = Field(
        default=True,
        description="Enable Virtual API Keys features (org/team association, budgets)"
    )
    LLM_BUDGET_ENFORCE: bool = Field(
        default=True,
        description="Reject requests from over-budget virtual keys"
    )
    LLM_BUDGET_ENDPOINTS: list[str] = Field(
        default_factory=lambda: [
            "/api/v1/chat/completions",
            "/api/v1/embeddings"
        ],
        description="Paths where LLM budget middleware is applied"
    )

    # ===== Monitoring =====
    ENABLE_HEALTH_CHECK: bool = Field(
        default=True,
        description="Enable health check endpoints"
    )

    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )

    # ===== Security Alerting =====
    SECURITY_ALERTS_ENABLED: bool = Field(
        default=False,
        description="Enable AuthNZ security alerts (file/webhook/email)."
    )
    SECURITY_ALERT_MIN_SEVERITY: str = Field(
        default="high",
        description="Minimum severity to dispatch security alerts (low|medium|high|critical)."
    )
    SECURITY_ALERT_FILE_PATH: str = Field(
        default="Databases/security_alerts.log",
        description="File path for security alert JSONL sink."
    )
    SECURITY_ALERT_WEBHOOK_URL: Optional[str] = Field(
        default=None,
        description="Webhook URL for security alerts (optional)."
    )
    SECURITY_ALERT_WEBHOOK_HEADERS: Optional[str] = Field(
        default=None,
        description="JSON object of additional headers to include in security alert webhooks."
    )
    SECURITY_ALERT_EMAIL_TO: Optional[str] = Field(
        default=None,
        description="Comma-separated list of email recipients for security alerts."
    )
    SECURITY_ALERT_EMAIL_FROM: Optional[str] = Field(
        default=None,
        description="Email sender address for security alerts."
    )
    SECURITY_ALERT_EMAIL_SUBJECT_PREFIX: str = Field(
        default="[AuthNZ]",
        description="Subject prefix for security alert emails."
    )
    SECURITY_ALERT_SMTP_HOST: Optional[str] = Field(
        default=None,
        description="SMTP host for security alert emails."
    )
    SECURITY_ALERT_SMTP_PORT: int = Field(
        default=587,
        description="SMTP port for security alert emails."
    )
    SECURITY_ALERT_SMTP_STARTTLS: bool = Field(
        default=True,
        description="Use STARTTLS for security alert emails."
    )
    SECURITY_ALERT_SMTP_USERNAME: Optional[str] = Field(
        default=None,
        description="SMTP username for security alert emails."
    )
    SECURITY_ALERT_SMTP_PASSWORD: Optional[str] = Field(
        default=None,
        description="SMTP password for security alert emails."
    )
    SECURITY_ALERT_SMTP_TIMEOUT: int = Field(
        default=10,
        ge=1,
        description="Timeout (seconds) for SMTP connections when sending security alerts."
    )
    SECURITY_ALERT_BACKOFF_SECONDS: int = Field(
        default=30,
        ge=0,
        description="Backoff window after a sink failure before retrying (seconds)."
    )
    SECURITY_ALERT_FILE_MIN_SEVERITY: Optional[str] = Field(
        default=None,
        description="Minimum severity to write alerts to the file sink (default: use global threshold)."
    )
    SECURITY_ALERT_WEBHOOK_MIN_SEVERITY: Optional[str] = Field(
        default=None,
        description="Minimum severity to deliver alerts to the webhook sink (default: use global threshold)."
    )
    SECURITY_ALERT_EMAIL_MIN_SEVERITY: Optional[str] = Field(
        default=None,
        description="Minimum severity to deliver alerts via email (default: use global threshold)."
    )

    METRICS_PORT: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Prometheus metrics port"
    )

    # ===== Data Retention =====
    AUDIT_LOG_RETENTION_DAYS: int = Field(
        default=180,
        ge=30,
        description="Audit log retention in days"
    )

    PASSWORD_HISTORY_RETENTION_COUNT: int = Field(
        default=5,
        ge=3,
        description="Number of previous passwords to remember"
    )

    SESSION_LOG_RETENTION_DAYS: int = Field(
        default=90,
        ge=7,
        description="Session log retention in days"
    )

    # ===== Single-User Mode Settings =====
    SINGLE_USER_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for single-user mode - MUST be set via environment variable"
    )

    SINGLE_USER_FIXED_ID: int = Field(
        default=1,
        description="Fixed user ID for single-user mode"
    )
    # Optional IP allowlist for single-user mode (comma-separated env or list)
    SINGLE_USER_ALLOWED_IPS: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Optional list of allowed client IPs/CIDRs for SINGLE_USER_API_KEY"
    )

    # Trust proxy headers for resolving client IPs (used by API key allowlists).
    AUTH_TRUST_X_FORWARDED_FOR: bool = Field(
        default=False,
        description=(
            "If true, honor X-Forwarded-For/X-Real-IP when the peer is a trusted proxy "
            "(see AUTH_TRUSTED_PROXY_IPS)."
        ),
    )
    AUTH_TRUSTED_PROXY_IPS: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Optional list of trusted proxy IPs/CIDRs for X-Forwarded-For resolution."
    )

    # ===== Service Account Settings =====
    SERVICE_ACCOUNT_RATE_LIMIT: int = Field(
        default=1000,
        ge=100,
        description="Rate limit for service accounts per minute"
    )
    SERVICE_TOKEN_ALLOWED_IPS: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description=(
            "Optional list of allowed client IPs/CIDRs for service tokens. "
            "When empty, service tokens are restricted to loopback requests."
        ),
    )

    def __init__(self, **kwargs):
        """Initialize settings and ensure JWT secret persistence"""
        super().__init__(**kwargs)
        self._ensure_jwt_secret()
        self._validate_api_key()
        self._apply_enterprise_guardrails()

    def _ensure_jwt_secret(self):
        """Ensure JWT secret exists with secure handling"""
        # Skip JWT secret handling in single-user mode
        if self.AUTH_MODE == "single_user":
            logger.debug("Single-user mode - skipping JWT secret initialization")
            return

        alg_upper = (self.JWT_ALGORITHM or "").upper()
        if alg_upper.startswith(("RS", "ES")) and self.JWT_PRIVATE_KEY:
            # Asymmetric algorithms supply their own key material; a symmetric secret is unnecessary
            logger.debug(
                'Asymmetric JWT algorithm {} detected with private key; skipping JWT secret requirement',
                self.JWT_ALGORITHM,
            )
            return

        # Environment variable is REQUIRED for JWT secret
        if self.JWT_SECRET_KEY:
            # Secret provided via environment - validate it
            if len(self.JWT_SECRET_KEY) < 32:
                raise ValueError(
                    "JWT_SECRET_KEY must be at least 32 characters for security.\n"
                    "Set it via environment or .env. Example:\n"
                    "  export JWT_SECRET_KEY=$(python -c \"import secrets; print(secrets.token_urlsafe(32))\")\n"
                    "See README: Authentication Setup."
                )

            # Hard fail in production if default/weak
            prod_flag = os.getenv("tldw_production", "false").lower() in {"true", "1", "yes", "y", "on"}
            if prod_flag:
                default_values = {"CHANGE_ME_TO_SECURE_RANDOM_KEY_MIN_32_CHARS"}
                if self.JWT_SECRET_KEY in default_values:
                    raise ValueError(
                        "In production (tldw_production=true), JWT_SECRET_KEY must not use default template values.\n"
                        "Set a strong key via environment or .env. Example:\n"
                        "  export JWT_SECRET_KEY=$(python -c \"import secrets; print(secrets.token_urlsafe(32))\")"
                    )
            logger.info("Using configured JWT secret key")
            return

        # Allow deterministic fallback only in explicit pytest runtime.
        test_mode = _is_explicit_pytest_runtime()
        if test_mode:
            fallback = os.getenv(
                "JWT_SECRET_TEST_KEY",
                "test-secret-jwt-key-please-change-1234567890"
            )
            if len(fallback) < 32:
                fallback = (fallback * 2)[:32]
            self.JWT_SECRET_KEY = fallback
            logger.debug("Initialized deterministic JWT secret for test context")
            return

        # No JWT secret available - this is a configuration error
        raise ValueError(
            "JWT_SECRET_KEY must be set via environment variable for multi-user mode.\n"
            "Set it via environment or .env. Example:\n"
            "  export JWT_SECRET_KEY=$(python -c \"import secrets; print(secrets.token_urlsafe(32))\")\n"
            "This is required for security."
        )

    def _validate_api_key(self):
        """Validate API key for single-user mode with deterministic test fallback."""
        if self.AUTH_MODE == "single_user":
            # If an explicit key is provided via env/.env, honor it (legacy API_KEY supported)
            explicit_env_key = os.getenv("SINGLE_USER_API_KEY") or os.getenv("API_KEY")

            # Detect explicit test contexts where the server should expose a
            # stable key so client tests can authenticate. Only rely on server
            # environment variables; never trust request headers.
            in_test_context = _is_explicit_pytest_runtime() or os.getenv("E2E_TEST_BASE_URL") is not None

            if not self.SINGLE_USER_API_KEY:
                if explicit_env_key:
                    # Loaded by pydantic from env but still None here? Ensure it's applied
                    self.SINGLE_USER_API_KEY = explicit_env_key
                    logger.info("Using SINGLE_USER_API_KEY from environment for single-user mode")
                elif in_test_context:
                    # Deterministic key so tests can authenticate reliably.
                    test_key = os.getenv("SINGLE_USER_TEST_API_KEY")
                    if not test_key:
                        raise ValueError(
                            "SINGLE_USER_API_KEY is not configured for single-user mode.\n"
                            "In test contexts, set SINGLE_USER_TEST_API_KEY explicitly (no default is assumed)."
                        )
                    self.SINGLE_USER_API_KEY = test_key
                    logger.debug("Using SINGLE_USER_TEST_API_KEY for deterministic test context")
                else:
                    raise ValueError(SINGLE_USER_KEY_MISSING)
            # In test contexts, normalize known placeholder keys to a deterministic test key
            elif in_test_context and self.SINGLE_USER_API_KEY in SINGLE_USER_API_KEY_PLACEHOLDERS:
                test_key = os.getenv("SINGLE_USER_TEST_API_KEY")
                if test_key:
                    self.SINGLE_USER_API_KEY = test_key
                    logger.debug("Normalized SINGLE_USER_API_KEY to SINGLE_USER_TEST_API_KEY for pytest context")
            elif self.SINGLE_USER_API_KEY in SINGLE_USER_API_KEY_PLACEHOLDERS:
                raise ValueError(SINGLE_USER_KEY_DEFAULT)
            elif len(self.SINGLE_USER_API_KEY) < 16:
                # Allow short keys in explicit test contexts to avoid brittle fixtures
                if in_test_context:
                    return
                raise ValueError(SINGLE_USER_KEY_TOO_SHORT)

            # Hard fail in production if key is missing/weak/default
            prod_flag = os.getenv("tldw_production", "false").lower() in {"true", "1", "yes", "y", "on"}
            allow_legacy = os.getenv("TLDW_ALLOW_LEGACY_SINGLE_USER_KEY", "").lower() in {"true", "1", "yes", "y", "on"}
            if not in_test_context:
                try:
                    from tldw_Server_API.app.core.AuthNZ.api_key_crypto import parse_api_key

                    is_new_format = parse_api_key(self.SINGLE_USER_API_KEY) is not None
                except (ImportError, AttributeError):
                    logger.debug("Could not import parse_api_key; treating key as legacy format")
                    is_new_format = False

                if not is_new_format and not allow_legacy:
                    if prod_flag:
                        raise ValueError(SINGLE_USER_KEY_PRODUCTION_FORMAT)
                    logger.warning(
                        "SINGLE_USER_API_KEY uses a legacy format; "
                        "generate a new server-generated key for improved security."
                    )
            if prod_flag:
                weak = (
                    not self.SINGLE_USER_API_KEY
                    or self.SINGLE_USER_API_KEY in SINGLE_USER_API_KEY_PLACEHOLDERS.union({"test-api-key-12345"})
                    or len(self.SINGLE_USER_API_KEY) < 24
                )
                if weak:
                    raise ValueError(SINGLE_USER_KEY_PRODUCTION_WEAK)

    def _apply_enterprise_guardrails(self) -> None:
        """Log guardrail decisions for enterprise-only feature flags."""
        if self.AUTH_FEDERATION_ENABLED and not self.enterprise_federation_supported:
            logger.warning(
                "AUTH_FEDERATION_ENABLED is set but enterprise federation is disabled: {}",
                ", ".join(self.enterprise_support_matrix_errors),
            )

        if self.SECRET_BACKENDS_ENABLED and not self.enterprise_secret_backends_supported:
            logger.warning(
                "SECRET_BACKENDS_ENABLED is set but enterprise secret backends are disabled: {}",
                ", ".join(self.enterprise_support_matrix_errors),
            )

        if self.MCP_CREDENTIAL_BROKER_ENABLED and not self.enterprise_mcp_credential_broker_supported:
            reasons = list(self.enterprise_support_matrix_errors)
            if not self.SECRET_BACKENDS_ENABLED:
                reasons.append("SECRET_BACKENDS_ENABLED must also be enabled")
            logger.warning(
                "MCP_CREDENTIAL_BROKER_ENABLED is set but broker support is disabled: {}",
                ", ".join(reasons),
            )

    @property
    def enterprise_support_matrix_errors(self) -> tuple[str, ...]:
        """Return reasons why enterprise features are unavailable."""
        errors: list[str] = []
        if self.AUTH_MODE != "multi_user":
            errors.append("AUTH_MODE=multi_user is required")
        if not _database_url_is_postgres(self.DATABASE_URL):
            errors.append("PostgreSQL DATABASE_URL is required")
        if not _profile_supports_enterprise_features(getattr(self, "PROFILE", None)):
            errors.append("PROFILE is not supported for enterprise features")
        return tuple(errors)

    @property
    def enterprise_deployment_supported(self) -> bool:
        """Return True when the deployment matches the enterprise support matrix."""
        return not self.enterprise_support_matrix_errors

    @property
    def enterprise_federation_supported(self) -> bool:
        """Return True when OIDC federation may run in this deployment."""
        return self.AUTH_FEDERATION_ENABLED and self.enterprise_deployment_supported

    @property
    def enterprise_secret_backends_supported(self) -> bool:
        """Return True when secret-backend features may run in this deployment."""
        return self.SECRET_BACKENDS_ENABLED and self.enterprise_deployment_supported

    @property
    def enterprise_mcp_credential_broker_supported(self) -> bool:
        """Return True when MCP credential brokering may run in this deployment."""
        return (
            self.MCP_CREDENTIAL_BROKER_ENABLED
            and self.enterprise_deployment_supported
            and self.enterprise_secret_backends_supported
        )

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret(cls, v, info):
        """Validate JWT secret strength"""
        # Skip validation in single-user mode
        if info.data.get("AUTH_MODE") == "single_user":
            return v

        if v and len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
        return v

    @field_validator("SINGLE_USER_ALLOWED_IPS", mode="before")
    @classmethod
    def parse_single_user_allowed_ips(cls, v):
        """Allow CSV or JSON-list env forms for IP allowlists."""
        return _split_csv(v)

    @field_validator("SERVICE_TOKEN_ALLOWED_IPS", mode="before")
    @classmethod
    def parse_service_token_allowed_ips(cls, v):
        """Allow CSV or JSON-list env forms for IP allowlists."""
        return _split_csv(v)

    @field_validator("AUTH_TRUSTED_PROXY_IPS", mode="before")
    @classmethod
    def parse_auth_trusted_proxy_ips(cls, v):
        """Allow CSV or JSON-list env forms for IP allowlists."""
        return _split_csv(v)

    @field_validator("ORG_RBAC_SCOPE_MODE", mode="before")
    @classmethod
    def normalize_org_rbac_scope_mode(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("ORG_RBAC_SCOPED_PERMISSION_DENYLIST", mode="before")
    @classmethod
    def parse_org_rbac_scoped_permission_denylist(cls, v):
        return _split_csv(v)

    @field_validator("BYOK_ENABLED", mode="before")
    @classmethod
    def parse_byok_enabled(cls, v):
        if v is None:
            env_val = os.getenv("BYOK_ENABLED")
            if env_val is not None:
                return _bool_from_str(env_val)
        return v

    @field_validator("BYOK_ALLOWED_PROVIDERS", mode="before")
    @classmethod
    def parse_byok_allowed_providers(cls, v):
        if v is None:
            env_val = os.getenv("BYOK_ALLOWED_PROVIDERS")
            if env_val is not None:
                v = env_val
        return _split_csv(v)

    @field_validator("BYOK_ALLOWED_BASE_URL_PROVIDERS", mode="before")
    @classmethod
    def parse_byok_allowed_base_url_providers(cls, v):
        if v is None:
            env_val = os.getenv("BYOK_ALLOWED_BASE_URL_PROVIDERS")
            if env_val is not None:
                v = env_val
        return _split_csv(v)

    @field_validator("BYOK_ENCRYPTION_KEY", mode="before")
    @classmethod
    def normalize_byok_encryption_key(cls, v):
        if v is None:
            env_val = os.getenv("BYOK_ENCRYPTION_KEY")
            if env_val is not None:
                v = env_val
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("BYOK_SECONDARY_ENCRYPTION_KEY", mode="before")
    @classmethod
    def normalize_byok_secondary_encryption_key(cls, v):
        if v is None:
            env_val = os.getenv("BYOK_SECONDARY_ENCRYPTION_KEY")
            if env_val is not None:
                v = env_val
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("OPENAI_OAUTH_ENABLED", mode="before")
    @classmethod
    def parse_openai_oauth_enabled(cls, v):
        if v is None:
            env_val = os.getenv("OPENAI_OAUTH_ENABLED")
            if env_val is not None:
                return _bool_from_str(env_val)
        return v

    @field_validator(
        "AUTH_FEDERATION_ENABLED",
        "MCP_CREDENTIAL_BROKER_ENABLED",
        "SECRET_BACKENDS_ENABLED",
        mode="before",
    )
    @classmethod
    def parse_enterprise_feature_flags(cls, v):
        if v is None:
            return False
        if isinstance(v, str):
            return _bool_from_str(v)
        return bool(v)

    @field_validator("OPENAI_OAUTH_CLIENT_ID", "OPENAI_OAUTH_CLIENT_SECRET", mode="before")
    @classmethod
    def normalize_openai_oauth_secret_fields(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("OPENAI_OAUTH_AUTH_URL", "OPENAI_OAUTH_TOKEN_URL", "OPENAI_OAUTH_REDIRECT_URI", mode="before")
    @classmethod
    def normalize_openai_oauth_urls(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("OPENAI_OAUTH_SCOPES", mode="before")
    @classmethod
    def parse_openai_oauth_scopes(cls, v):
        if v is None:
            env_val = os.getenv("OPENAI_OAUTH_SCOPES")
            if env_val is not None:
                v = env_val
        return _split_csv(v)

    @field_validator("OPENAI_OAUTH_ALLOWED_RETURN_PATH_PREFIXES", mode="before")
    @classmethod
    def parse_openai_oauth_allowed_return_path_prefixes(cls, v):
        if v is None:
            env_val = os.getenv("OPENAI_OAUTH_ALLOWED_RETURN_PATH_PREFIXES")
            if env_val is not None:
                v = env_val
        parsed = _split_csv(v)
        if not parsed:
            return ["/"]
        return parsed

    @field_validator("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", mode="before")
    @classmethod
    def parse_openai_oauth_refresh_lock_backend(cls, v):
        if v is None:
            env_val = os.getenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND")
            if env_val is not None:
                v = env_val
        text = str(v or "memory").strip().lower()
        if text in {"memory", "redis", "db"}:
            return text
        return "memory"

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v, info):
        """Validate database URL based on auth mode"""
        auth_mode = info.data.get("AUTH_MODE", "single_user")

        if auth_mode == "multi_user" and v.startswith("sqlite"):
            # In production, disallow SQLite for multi-user mode
            prod_flag = os.getenv("tldw_production", "false").lower() in {"true", "1", "yes", "y", "on"}
            if prod_flag:
                raise ValueError(
                    "In production (tldw_production=true) with AUTH_MODE=multi_user, SQLite is not supported.\n"
                    "Please configure PostgreSQL via DATABASE_URL. Examples:\n"
                    "  export DATABASE_URL=postgresql://tldw_user:TestPassword123!@localhost:5432/tldw_users\n"
                    "  # With docker-compose service name:\n"
                    "  export DATABASE_URL=postgresql://tldw_user:TestPassword123!@postgres:5432/tldw_users\n"
                    "See Multi-User Deployment Guide for details."
                )
            else:
                logger.warning(
                    "Using SQLite in multi-user mode is not recommended. "
                    "Consider using PostgreSQL for better concurrency."
                )

        return v

    @field_validator("REDIS_URL")
    @classmethod
    def validate_redis_url(cls, v):
        """Validate Redis URL format if provided"""
        if v and not (v.startswith("redis://") or v.startswith("rediss://")):
            raise ValueError("REDIS_URL must start with redis:// or rediss://")
        return v

    model_config = {
        "env_file": str(AUTHNZ_DEFAULT_ENV_FILE),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "allow"  # Allow extra fields for backward compatibility
    }


# ===== Singleton Settings Instance =====
_settings: Optional[Settings] = None


def _bool_from_str(val: str) -> bool:
    return _is_truthy(str(val).strip())


def _database_url_is_postgres(database_url: str | None) -> bool:
    text = str(database_url or "").strip().lower()
    return text.startswith(("postgres://", "postgresql://"))


def _profile_supports_enterprise_features(profile: str | None) -> bool:
    if not isinstance(profile, str) or not profile.strip():
        return True
    return profile.strip().lower() in ENTERPRISE_SUPPORTED_PROFILES


def _split_csv(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except _SETTINGS_JSON_PARSE_EXCEPTIONS:
                parsed = None
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        return [s.strip() for s in val.split(",") if s.strip()]
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]
    return []


def _load_overrides_from_config() -> dict:
    """Load Settings overrides from Config_Files/config.txt (AuthNZ section).

    Environment variables retain precedence; only provide values that are
    NOT already present in the environment.
    """
    overrides = {}
    if not load_comprehensive_config:
        return overrides
    try:
        cfg = load_comprehensive_config()
        if not cfg or not cfg.has_section("AuthNZ"):
            return overrides

        def maybe_set(field: str, key: str, caster=lambda v: v):
            # Map config key to Settings field if env var not set
            env_name = field
            if os.getenv(env_name) is None and cfg.has_option("AuthNZ", key):
                with contextlib.suppress(_SETTINGS_CAST_EXCEPTIONS):
                    overrides[field] = caster(cfg.get("AuthNZ", key))

        maybe_set("AUTH_MODE", "auth_mode", lambda v: v.strip())
        maybe_set("DATABASE_URL", "database_url", lambda v: v.strip())
        maybe_set("JWT_SECRET_KEY", "jwt_secret_key", lambda v: v.strip())
        maybe_set("SINGLE_USER_API_KEY", "single_user_api_key", lambda v: v.strip())
        maybe_set("BYOK_ENABLED", "byok_enabled", _bool_from_str)
        maybe_set("BYOK_ALLOWED_PROVIDERS", "byok_allowed_providers", _split_csv)
        maybe_set("BYOK_ALLOWED_BASE_URL_PROVIDERS", "byok_allowed_base_url_providers", _split_csv)
        maybe_set("BYOK_ENCRYPTION_KEY", "byok_encryption_key", lambda v: v.strip())
        maybe_set("BYOK_SECONDARY_ENCRYPTION_KEY", "byok_secondary_encryption_key", lambda v: v.strip())
        maybe_set("OPENAI_OAUTH_ENABLED", "openai_oauth_enabled", _bool_from_str)
        maybe_set("OPENAI_OAUTH_CLIENT_ID", "openai_oauth_client_id", lambda v: v.strip())
        maybe_set("OPENAI_OAUTH_CLIENT_SECRET", "openai_oauth_client_secret", lambda v: v.strip())
        maybe_set("OPENAI_OAUTH_AUTH_URL", "openai_oauth_auth_url", lambda v: v.strip())
        maybe_set("OPENAI_OAUTH_TOKEN_URL", "openai_oauth_token_url", lambda v: v.strip())
        maybe_set("OPENAI_OAUTH_SCOPES", "openai_oauth_scopes", _split_csv)
        maybe_set("OPENAI_OAUTH_STATE_TTL_MINUTES", "openai_oauth_state_ttl_minutes", lambda v: int(v))
        maybe_set(
            "OPENAI_OAUTH_REFRESH_SKEW_SECONDS",
            "openai_oauth_refresh_skew_seconds",
            lambda v: int(v),
        )
        maybe_set("OPENAI_OAUTH_REDIRECT_URI", "openai_oauth_redirect_uri", lambda v: v.strip())
        maybe_set(
            "OPENAI_OAUTH_ALLOWED_RETURN_PATH_PREFIXES",
            "openai_oauth_allowed_return_path_prefixes",
            _split_csv,
        )
        maybe_set(
            "OPENAI_OAUTH_MAX_OUTSTANDING_STATES",
            "openai_oauth_max_outstanding_states",
            lambda v: int(v),
        )
        maybe_set(
            "OPENAI_OAUTH_REFRESH_LOCK_BACKEND",
            "openai_oauth_refresh_lock_backend",
            lambda v: str(v).strip().lower(),
        )
        maybe_set("AUTH_FEDERATION_ENABLED", "auth_federation_enabled", _bool_from_str)
        maybe_set(
            "MCP_CREDENTIAL_BROKER_ENABLED",
            "mcp_credential_broker_enabled",
            _bool_from_str,
        )
        maybe_set("SECRET_BACKENDS_ENABLED", "secret_backends_enabled", _bool_from_str)
        maybe_set("ENABLE_REGISTRATION", "enable_registration", _bool_from_str)
        # Legacy aliases in config.txt
        maybe_set("ENABLE_REGISTRATION", "registration_enabled", _bool_from_str)
        maybe_set("REQUIRE_REGISTRATION_CODE", "require_registration_code", _bool_from_str)
        maybe_set("REQUIRE_REGISTRATION_CODE", "registration_require_code", _bool_from_str)
        maybe_set(
            "ENABLE_ORG_SCOPED_REGISTRATION_CODES",
            "enable_org_scoped_registration_codes",
            _bool_from_str,
        )
        maybe_set(
            "ORG_INVITE_ALLOW_MISSING_EMAIL",
            "org_invite_allow_missing_email",
            _bool_from_str,
        )
        maybe_set("ACCESS_TOKEN_EXPIRE_MINUTES", "access_token_expire_minutes", lambda v: int(v))
        maybe_set("REFRESH_TOKEN_EXPIRE_DAYS", "refresh_token_expire_days", lambda v: int(v))
        maybe_set("MAGIC_LINK_EXPIRE_MINUTES", "magic_link_expire_minutes", lambda v: int(v))
        maybe_set("PUBLIC_WEB_BASE_URL", "public_web_base_url", lambda v: v.strip())
        maybe_set(
            "PUBLIC_PASSWORD_RESET_PATH",
            "public_password_reset_path",
            lambda v: v.strip(),
        )
        maybe_set(
            "PUBLIC_EMAIL_VERIFICATION_PATH",
            "public_email_verification_path",
            lambda v: v.strip(),
        )
        maybe_set(
            "PUBLIC_MAGIC_LINK_PATH",
            "public_magic_link_path",
            lambda v: v.strip(),
        )
        maybe_set("REDIS_URL", "redis_url", lambda v: v.strip())
        maybe_set("SECURITY_ALERTS_ENABLED", "security_alerts_enabled", _bool_from_str)
        maybe_set("SECURITY_ALERT_MIN_SEVERITY", "security_alert_min_severity", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_FILE_PATH", "security_alert_file_path", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_WEBHOOK_URL", "security_alert_webhook_url", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_WEBHOOK_HEADERS", "security_alert_webhook_headers", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_EMAIL_TO", "security_alert_email_to", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_EMAIL_FROM", "security_alert_email_from", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_EMAIL_SUBJECT_PREFIX", "security_alert_email_subject_prefix", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_SMTP_HOST", "security_alert_smtp_host", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_SMTP_PORT", "security_alert_smtp_port", lambda v: int(v))
        maybe_set("SECURITY_ALERT_SMTP_STARTTLS", "security_alert_smtp_starttls", _bool_from_str)
        maybe_set("SECURITY_ALERT_SMTP_USERNAME", "security_alert_smtp_username", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_SMTP_PASSWORD", "security_alert_smtp_password", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_SMTP_TIMEOUT", "security_alert_smtp_timeout", lambda v: int(v))
        maybe_set("SECURITY_ALERT_FILE_MIN_SEVERITY", "security_alert_file_min_severity", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_WEBHOOK_MIN_SEVERITY", "security_alert_webhook_min_severity", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_EMAIL_MIN_SEVERITY", "security_alert_email_min_severity", lambda v: v.strip())
        maybe_set("SECURITY_ALERT_BACKOFF_SECONDS", "security_alert_backoff_seconds", lambda v: int(v))
        maybe_set("AUTH_TRUST_X_FORWARDED_FOR", "auth_trust_x_forwarded_for", _bool_from_str)
        maybe_set("AUTH_TRUSTED_PROXY_IPS", "auth_trusted_proxy_ips", _split_csv)
        maybe_set("ORG_RBAC_PROPAGATION_ENABLED", "org_rbac_propagation_enabled", _bool_from_str)
        maybe_set("ORG_RBAC_SCOPE_MODE", "org_rbac_scope_mode", lambda v: v.strip().lower())
        maybe_set(
            "ORG_RBAC_SCOPED_PERMISSION_DENYLIST",
            "org_rbac_scoped_permission_denylist",
            _split_csv,
        )

        # If DATABASE_URL is not provided via env or explicit key, synthesize from db_type fields
        if os.getenv("DATABASE_URL") is None and "DATABASE_URL" not in overrides:
            if cfg.has_option("AuthNZ", "db_type"):
                db_type = cfg.get("AuthNZ", "db_type").strip().lower()
                if db_type == "sqlite":
                    # sqlite_path supports relative paths; default to Databases/users.db
                    sqlite_path = cfg.get("AuthNZ", "sqlite_path", fallback="./Databases/users.db").strip()
                    overrides["DATABASE_URL"] = f"sqlite:///{sqlite_path}"
                elif db_type in {"postgres", "postgresql"}:
                    host = cfg.get("AuthNZ", "pg_host", fallback="localhost").strip()
                    port = cfg.get("AuthNZ", "pg_port", fallback="5432").strip()
                    db   = cfg.get("AuthNZ", "pg_db", fallback="tldw_users").strip()
                    user = cfg.get("AuthNZ", "pg_user", fallback="tldw_user").strip()
                    # Do NOT silently synthesize with a placeholder password.
                    pwd  = cfg.get("AuthNZ", "pg_password", fallback="").strip()
                    sslm = cfg.get("AuthNZ", "pg_sslmode", fallback="prefer").strip()
                    if not pwd:
                        # In production, require explicit password; outside, skip synthesis to avoid insecure defaults
                        from os import getenv as _getenv
                        prod_flag = _getenv("tldw_production", "false").lower() in {"true", "1", "yes", "y", "on"}
                        logger.warning("AuthNZ: pg_password not set in config.txt; not synthesizing DATABASE_URL.")
                        if prod_flag:
                            # Leave DATABASE_URL unset to force explicit configuration
                            pass
                        else:
                            # Still avoid insecure default; allow env DATABASE_URL to win if present elsewhere
                            pass
                    else:
                        overrides["DATABASE_URL"] = f"postgresql://{user}:{pwd}@{host}:{port}/{db}?sslmode={sslm}"
                # else: ignore unknown types
    except _SETTINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"AuthNZ settings: failed to load overrides from config.txt: {e}")
    return overrides


# Internal generation counter for settings cache invalidation
_settings_generation: int = 0


def get_settings() -> Settings:
    """Get settings singleton instance with optional config.txt overrides"""
    global _settings
    if not _settings:
        overrides = _load_overrides_from_config()
        # Aliases for legacy names used in some tests/envs
        try:
            import os as _os
            def _alias_bool(env_name: str) -> bool:
                return _is_truthy(str(_os.getenv(env_name, "")).strip())
            # REGISTRATION_ENABLED -> ENABLE_REGISTRATION
            if _os.getenv("REGISTRATION_ENABLED") is not None and "ENABLE_REGISTRATION" not in overrides:
                overrides["ENABLE_REGISTRATION"] = _alias_bool("REGISTRATION_ENABLED")
            # REGISTRATION_REQUIRE_CODE -> REQUIRE_REGISTRATION_CODE
            if _os.getenv("REGISTRATION_REQUIRE_CODE") is not None and "REQUIRE_REGISTRATION_CODE" not in overrides:
                overrides["REQUIRE_REGISTRATION_CODE"] = _alias_bool("REGISTRATION_REQUIRE_CODE")
            if (
                _os.getenv("ENABLE_ORG_SCOPED_REGISTRATION_CODES") is not None
                and "ENABLE_ORG_SCOPED_REGISTRATION_CODES" not in overrides
            ):
                overrides["ENABLE_ORG_SCOPED_REGISTRATION_CODES"] = _alias_bool(
                    "ENABLE_ORG_SCOPED_REGISTRATION_CODES"
                )
            if (
                _os.getenv("ORG_INVITE_ALLOW_MISSING_EMAIL") is not None
                and "ORG_INVITE_ALLOW_MISSING_EMAIL" not in overrides
            ):
                overrides["ORG_INVITE_ALLOW_MISSING_EMAIL"] = _alias_bool(
                    "ORG_INVITE_ALLOW_MISSING_EMAIL"
                )
        except _SETTINGS_IMPORT_EXCEPTIONS:
            pass
        _settings = Settings(**overrides)
        try:
            base_dir = None
            if core_settings:
                base_dir = core_settings.get("USER_DB_BASE_DIR")
            if base_dir:
                _settings.USER_DATA_BASE_PATH = str(Path(base_dir).resolve())
        except _SETTINGS_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"AuthNZ settings: failed to align USER_DATA_BASE_PATH with core settings: {exc}")
        # Log a lightweight profile hint for coordination/UX and optional
        # hardening. AUTH_MODE remains the canonical behavioral switch; PROFILE
        # (explicit or inferred) must not be used to bypass or relax auth
        # decisions. It may be used as an additional tightening signal (e.g.,
        # disabling self-registration in local-single-user deployments) so long
        # as AUTH_MODE + claims remain the lower bound for permissions.
        try:
            profile_hint = getattr(_settings, "PROFILE", None)
            if not (isinstance(profile_hint, str) and profile_hint.strip()):
                profile_hint = _infer_profile_from_settings(_settings)
            if isinstance(profile_hint, str) and profile_hint.strip():
                logger.info(
                    'Settings initialized - Auth mode: {}, profile={}',
                    _settings.AUTH_MODE,
                    profile_hint.strip(),
                )
            else:
                logger.info("Settings initialized - Auth mode: {}", _settings.AUTH_MODE)
        except _SETTINGS_NONCRITICAL_EXCEPTIONS:
            logger.info("Settings initialized - Auth mode: {}", _settings.AUTH_MODE)
    return _settings


def reset_settings():
    """Reset settings singleton (mainly for testing)"""
    global _settings
    global _settings_generation
    _settings = None
    # Increment generation without swallowing errors
    _settings_generation = (_settings_generation or 0) + 1


def get_settings_generation() -> int:
    """Return a monotonic counter that increments when settings are reset.

    Middleware and helpers can use this value to cache the settings object
    per-process while still honoring explicit invalidations from tests or
    configuration reload hooks.
    """
    return _settings_generation


# ===== Utility Functions =====
def is_multi_user_mode() -> bool:
    """Return True when the effective runtime mode is multi-user.

    AUTH_MODE remains the canonical switch; PROFILE is advisory and
    may be used by higher-level helpers for UX/coordination. This
    helper deliberately ignores PROFILE so that existing behavior is
    preserved while we phase in profile-aware helpers elsewhere.
    """
    return get_settings().AUTH_MODE == "multi_user"


def is_single_user_mode() -> bool:
    """Return True when the effective runtime mode is single-user.

    AUTH_MODE remains the canonical switch; PROFILE is advisory and
    must not be used to bypass or relax auth behavior. Profile-aware
    helpers may use PROFILE as an additional tightening signal for
    deployment-specific flows (for example, forbidding self-registration
    in local-single-user) but must never grant privileges beyond those
    implied by AUTH_MODE and claims.
    """
    return get_settings().AUTH_MODE == "single_user"


def _infer_profile_from_settings(settings: Settings) -> Optional[str]:
    """Derive a coarse deployment profile from AUTH_MODE + DATABASE_URL.

    This helper is used when PROFILE is unset to provide a stable hint
    for coordination/UX and optional hardening that only *restricts*
    behavior (for example, disabling self-registration in single-user
    desktop deployments). It must not be used to bypass or relax auth
    or permission decisions relative to AUTH_MODE and claims.
    """
    try:
        mode = settings.AUTH_MODE
        db_url = str(settings.DATABASE_URL or "")
    except (AttributeError, TypeError, ValueError):
        return None

    db_lower = db_url.lower()

    if mode == "single_user":
        # Treat all single-user deployments as a local/desktop profile,
        # regardless of underlying DB, to preserve existing behavior.
        return "local-single-user"

    if mode == "multi_user":
        if db_lower.startswith(("postgres://", "postgresql://")):
            return "multi-user-postgres"
        if "sqlite" in db_lower:
            return "multi-user-sqlite"

    return None


def get_profile() -> Optional[str]:
    """Return the effective deployment profile string, if any.

    Resolution order:
    1. Explicit PROFILE setting/env (if set and non-empty).
    2. Derived from AUTH_MODE + DATABASE_URL via `_infer_profile_from_settings`.

    Callers should treat this primarily as a coordination/UX hint; auth
    and permission decisions remain driven by claims and AUTH_MODE
    helpers. It is acceptable to use PROFILE as an additional tightening
    signal for deployment-specific flows (for example, disabling
    self-registration in local-single-user deployments), but it must
    never be used to bypass or relax auth decisions or to grant
    permissions beyond those implied by AUTH_MODE and claims.
    """
    settings = get_settings()
    try:
        value = getattr(settings, "PROFILE", None)
    except (AttributeError, TypeError):
        value = None
    if isinstance(value, str) and value.strip():
        return value.strip()

    inferred = _infer_profile_from_settings(settings)
    if isinstance(inferred, str) and inferred.strip():
        return inferred.strip()
    return None


def is_single_user_profile_mode() -> bool:
    """Return True when PROFILE hints at a single-user deployment."""
    profile = get_profile()
    if profile:
        lowered = profile.strip().lower()
        if lowered in {"single_user", "local-single-user", "desktop"}:
            return True
    return False


def get_database_url() -> str:
    """Get the configured database URL"""
    return get_settings().DATABASE_URL


def get_jwt_secret() -> str:
    """Get JWT secret key (multi-user mode only). Raises if misconfigured."""
    settings = get_settings()
    if settings.AUTH_MODE != "multi_user":
        raise RuntimeError("JWT secret only available in multi-user mode")
    if not settings.JWT_SECRET_KEY:
        from tldw_Server_API.app.core.AuthNZ.exceptions import MissingConfigurationError
        raise MissingConfigurationError("JWT_SECRET_KEY")
    return settings.JWT_SECRET_KEY


#
# End of settings.py
#######################################################################################################################
