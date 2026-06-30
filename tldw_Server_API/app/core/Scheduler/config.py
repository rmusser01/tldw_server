"""
Configuration management for the scheduler system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from tldw_Server_API.app.core.testing import is_test_mode

# Prefer using the central project config when available so we place
# the scheduler DB alongside other Databases by default.
try:
    from tldw_Server_API.app.core.config import settings as core_settings  # type: ignore
except ImportError:
    core_settings = None  # Fallback if import graph changes

_SCHEDULER_CONFIG_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _contains_traversal_marker(path_value: str) -> bool:
    """Return True when a path string includes traversal or home-expansion segments."""

    normalized = path_value.replace("\\", "/")
    return any(part == ".." or part.startswith("~") for part in normalized.split("/"))


def _default_scheduler_db_url() -> str:
    """Compute a sensible default DB URL under the shared Databases folder.

    Priority:
    1) SCHEDULER_DATABASE_URL (or WORKFLOWS_SCHEDULER_DATABASE_URL) env
    2) Test contexts → temp sqlite per-process (preserves existing behavior)
    3) Project Databases folder → sqlite:///PROJECT_ROOT/Databases/scheduler.db
    4) Fallback to sqlite:///scheduler.db
    """
    # Explicit env overrides
    env_url = os.getenv('SCHEDULER_DATABASE_URL') or os.getenv('WORKFLOWS_SCHEDULER_DATABASE_URL')
    if env_url:
        return env_url

    # Preserve test behavior
    if (os.getenv('PYTEST_CURRENT_TEST') is not None or
            is_test_mode()):
        import os as _os
        import tempfile
        return f"sqlite:///{tempfile.gettempdir()}/scheduler_{_os.getpid()}.db"

    # Try to use the repo's Databases directory
    try:
        project_root = None
        if core_settings:
            project_root = core_settings.get('PROJECT_ROOT')
        if project_root:
            db_path = Path(project_root) / 'Databases' / 'scheduler.db'
            return f"sqlite:///{str(db_path.resolve())}"
    except _SCHEDULER_CONFIG_NONCRITICAL_EXCEPTIONS:
        pass

    # As a final fallback, use a relative Databases path if present
    try:
        cwd = Path(os.getcwd())
        candidate = cwd / 'Databases' / 'scheduler.db'
        return f"sqlite:///{str(candidate.resolve())}"
    except _SCHEDULER_CONFIG_NONCRITICAL_EXCEPTIONS:
        # Last resort: CWD scheduler.db
        return 'sqlite:///scheduler.db'


def _default_scheduler_base_path() -> Path:
    """Place scheduler payloads under PROJECT_ROOT/Databases/scheduler by default."""
    env_base = os.getenv('SCHEDULER_BASE_PATH')
    if env_base:
        try:
            return Path(env_base)
        except _SCHEDULER_CONFIG_NONCRITICAL_EXCEPTIONS:
            pass

    try:
        if core_settings:
            project_root = core_settings.get('PROJECT_ROOT')
            if project_root:
                return Path(project_root) / 'Databases' / 'scheduler'
    except _SCHEDULER_CONFIG_NONCRITICAL_EXCEPTIONS:
        pass

    # Fallback: put under local Databases if present; else default to ~/.local/share/scheduler
    try:
        local = Path(os.getcwd()) / 'Databases' / 'scheduler'
        return local
    except _SCHEDULER_CONFIG_NONCRITICAL_EXCEPTIONS:
        return Path.home() / '.local' / 'share' / 'scheduler'


@dataclass
class SchedulerConfig:
    """
    Complete configuration for the scheduler system.

    All paths are absolute to avoid working directory dependencies.
    Environment variables can override defaults.
    """

    # Database configuration
    database_url: str = field(default_factory=_default_scheduler_db_url)

    # Base path for all scheduler data
    base_path: Path = field(default_factory=_default_scheduler_base_path)

    # Write buffer configuration
    write_buffer_size: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_BUFFER_SIZE', '1000'))
    )
    write_buffer_flush_interval: float = field(
        default_factory=lambda: float(os.getenv('SCHEDULER_FLUSH_INTERVAL', '0.1'))
    )

    # Worker pool configuration
    min_workers: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_MIN_WORKERS', '1'))
    )
    max_workers: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_MAX_WORKERS', '10'))
    )
    worker_recycle_after_tasks: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_WORKER_RECYCLE', '1000'))
    )

    # Lease management
    lease_duration_seconds: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_LEASE_DURATION', '300'))
    )
    lease_renewal_interval: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_LEASE_RENEWAL', '30'))
    )
    lease_reaper_interval: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_REAPER_INTERVAL', '60'))
    )
    leader_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_LEADER_TTL', '300'))
    )

    # Payload management
    payload_threshold_bytes: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_PAYLOAD_THRESHOLD', '65536'))
    )
    payload_compression: bool = field(
        default_factory=lambda: os.getenv('SCHEDULER_PAYLOAD_COMPRESSION', 'true').lower() == 'true'
    )
    allow_legacy_pickle_payloads: bool = field(
        default_factory=lambda: os.getenv('SCHEDULER_ALLOW_LEGACY_PICKLE_PAYLOADS', 'false').lower() == 'true'
    )

    # Cleanup and retention
    payload_retention_days: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_PAYLOAD_RETENTION', '7'))
    )
    completed_task_retention_days: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_TASK_RETENTION', '30'))
    )
    cleanup_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_CLEANUP_INTERVAL', '3600'))
    )

    # Performance tuning
    batch_size: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_BATCH_SIZE', '100'))
    )
    max_concurrent_tasks: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_MAX_CONCURRENT', '100'))
    )

    # Database pool settings (PostgreSQL)
    db_pool_min_size: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_DB_POOL_MIN', '10'))
    )
    db_pool_max_size: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_DB_POOL_MAX', '100'))
    )

    # Monitoring
    metrics_enabled: bool = field(
        default_factory=lambda: os.getenv('SCHEDULER_METRICS_ENABLED', 'true').lower() == 'true'
    )
    metrics_port: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_METRICS_PORT', '9090'))
    )
    health_check_interval: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_HEALTH_INTERVAL', '30'))
    )

    # Queue defaults
    default_queue_name: str = field(
        default_factory=lambda: os.getenv('SCHEDULER_DEFAULT_QUEUE', 'default')
    )
    default_task_timeout: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_DEFAULT_TIMEOUT', '300'))
    )
    default_max_retries: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_DEFAULT_RETRIES', '3'))
    )
    default_retry_delay: int = field(
        default_factory=lambda: int(os.getenv('SCHEDULER_DEFAULT_RETRY_DELAY', '60'))
    )

    # Emergency backup file (for buffer fallbacks)
    emergency_backup_path: Optional[Path] = None

    @property
    def payload_storage_path(self) -> Path:
        """Path for external payload storage"""
        return self.base_path / 'payloads'



    @property
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL backend"""
        return 'postgresql' in self.database_url.lower() or 'postgres' in self.database_url.lower()

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite backend"""
        return 'sqlite' in self.database_url.lower()

    @property
    def is_memory(self) -> bool:
        """Check if using in-memory backend"""
        return self.database_url.lower() == 'memory://' or self.database_url == ':memory:'

    def __post_init__(self):
        """Validate configuration and create necessary directories"""
        # Validate and sanitize paths to prevent directory traversal
        # This must happen before attempting to create directories
        self._validate_and_sanitize_paths()

        # Default emergency backup path if not provided
        if self.emergency_backup_path is None:
            self.emergency_backup_path = self.base_path / 'emergency' / 'backup.json'
        else:
            # Ensure Path type and absolute
            self.emergency_backup_path = Path(self.emergency_backup_path).resolve()

        # Only create directories after validation passes
        try:
            # Ensure base path and subdirectories exist
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.payload_storage_path.mkdir(parents=True, exist_ok=True)
            self.emergency_backup_path.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            logger.error(f"Failed to create scheduler directories: {e}")
            raise ValueError(f"Cannot create scheduler directories at {self.base_path}: {e}") from e

        # Validate configuration
        self._validate()

        logger.info("Scheduler configuration initialized:")
        logger.info(f"  Database: {self._safe_database_url()}")
        logger.info(f"  Base path: {self.base_path}")
        logger.info(f"  Workers: {self.min_workers}-{self.max_workers}")
        logger.info(f"  Buffer: {self.write_buffer_size} items, {self.write_buffer_flush_interval}s flush")

    def _validate_and_sanitize_paths(self):
        """Validate and sanitize paths to prevent directory traversal attacks"""
        import platform

        # Check for directory traversal BEFORE resolving
        original_base = str(self.base_path)

        # Reject symlink paths, including symlink ancestors (e.g. symlink/child).
        # Allow top-level OS alias symlinks such as '/var' on macOS.
        import os as _os
        try:
            absolute_base = Path(_os.path.abspath(original_base))
        except OSError as exc:
            raise ValueError(f"Invalid base path: {self.base_path}") from exc

        for candidate in (absolute_base, *absolute_base.parents):
            try:
                if not candidate.exists():
                    continue
                if not candidate.is_symlink():
                    continue
                anchor = Path(candidate.anchor) if candidate.anchor else Path("/")
                if candidate.parent == anchor:
                    # OS-level alias symlink at the root boundary.
                    continue
                raise ValueError(f"Base path cannot be a symlink: {self.base_path}")
            except OSError as exc:
                raise ValueError(f"Invalid base path: {self.base_path}") from exc

        if _os.path.islink(original_base):
            raise ValueError(f"Base path cannot be a symlink: {self.base_path}")

        # Detect and prevent directory traversal attempts
        if _contains_traversal_marker(original_base):
            raise ValueError(f"Directory traversal detected in base_path: {original_base}")

        # Do not resolve symlinks here to preserve detection

        # Platform-specific path validation
        if platform.system() == 'Windows':
            # Windows-specific validations
            if ':' in str(self.base_path)[2:]:  # Allow drive letter
                raise ValueError(f"Invalid Windows path: {self.base_path}")
            # Use user's local app data if /var/lib doesn't exist
            if not self.base_path.exists() and str(self.base_path).startswith('/var/lib'):
                import tempfile
                self.base_path = Path(tempfile.gettempdir()) / 'scheduler'
                logger.warning(f"Using Windows temp path: {self.base_path}")
        else:
            # Unix-like systems
            base_str = str(self.base_path)
            is_var_lib = base_str.startswith('/var/lib') or base_str.startswith('/private/var/lib')
            var_paths = ['/var/lib', '/private/var/lib']
            writable = any(os.path.exists(p) and os.access(p, os.W_OK) for p in var_paths)
            if is_var_lib and not writable:
                # Fall back to user's home directory if var/lib is not writable
                self.base_path = Path.home() / '.local' / 'share' / 'scheduler'
                logger.warning(f"Using user directory: {self.base_path}")

        # Ensure path is not a symlink to prevent symlink attacks
        if self.base_path.exists() and self.base_path.is_symlink():
            raise ValueError(f"Base path cannot be a symlink: {self.base_path}")

        # Validate database URL if it's a file path
        if self.is_sqlite and 'sqlite:///' in self.database_url:
            db_path = self.database_url.replace('sqlite:///', '')
            if db_path and db_path != ':memory:':
                db_path_obj = Path(db_path)
                if _contains_traversal_marker(str(db_path_obj)):
                    raise ValueError(f"Directory traversal detected in database path: {db_path}")

    def _validate(self):
        """Validate configuration values"""
        errors = []

        # Validate numeric ranges
        # Allow 0 in test or when workers aren't started; in production
        # users typically run with >=1. Tests may configure 0.
        if self.min_workers < 0:
            errors.append(f"min_workers must be >= 0, got {self.min_workers}")

        if self.max_workers < self.min_workers:
            errors.append(f"max_workers ({self.max_workers}) must be >= min_workers ({self.min_workers})")

        if self.write_buffer_size < 1:
            errors.append(f"write_buffer_size must be >= 1, got {self.write_buffer_size}")

        if self.write_buffer_flush_interval <= 0:
            errors.append(f"write_buffer_flush_interval must be > 0, got {self.write_buffer_flush_interval}")

        if self.lease_duration_seconds < 10:
            errors.append(f"lease_duration_seconds should be >= 10, got {self.lease_duration_seconds}")

        if self.payload_threshold_bytes < 1024:
            errors.append(f"payload_threshold_bytes should be >= 1024, got {self.payload_threshold_bytes}")

        # Validate database URL
        if not self.database_url:
            errors.append("database_url cannot be empty")

        # Check for conflicting settings
        if self.lease_renewal_interval >= self.lease_duration_seconds:
            errors.append(
                f"lease_renewal_interval ({self.lease_renewal_interval}) should be less than "
                f"lease_duration_seconds ({self.lease_duration_seconds})"
            )

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

    def _safe_database_url(self) -> str:
        """Return database URL with password masked"""
        url = self.database_url
        if '@' in url and '://' in url:
            # Mask password in connection string
            parts = url.split('@')
            if len(parts) == 2:
                prefix = parts[0]
                if ':' in prefix:
                    # Has password
                    scheme_user = prefix.rsplit(':', 1)[0]
                    return f"{scheme_user}:****@{parts[1]}"
        return url

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'database_url': self._safe_database_url(),
            'base_path': str(self.base_path),
            'write_buffer_size': self.write_buffer_size,
            'write_buffer_flush_interval': self.write_buffer_flush_interval,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'worker_recycle_after_tasks': self.worker_recycle_after_tasks,
            'lease_duration_seconds': self.lease_duration_seconds,
            'lease_renewal_interval': self.lease_renewal_interval,
            'lease_reaper_interval': self.lease_reaper_interval,
            'leader_ttl_seconds': self.leader_ttl_seconds,
            'payload_threshold_bytes': self.payload_threshold_bytes,
            'payload_compression': self.payload_compression,
            'allow_legacy_pickle_payloads': self.allow_legacy_pickle_payloads,
            'payload_retention_days': self.payload_retention_days,
            'completed_task_retention_days': self.completed_task_retention_days,
            'cleanup_interval_seconds': self.cleanup_interval_seconds,
            'batch_size': self.batch_size,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'db_pool_min_size': self.db_pool_min_size,
            'db_pool_max_size': self.db_pool_max_size,
            'metrics_enabled': self.metrics_enabled,
            'metrics_port': self.metrics_port,
            'health_check_interval': self.health_check_interval,
            'default_queue_name': self.default_queue_name,
            'default_task_timeout': self.default_task_timeout,
            'default_max_retries': self.default_max_retries,
            'default_retry_delay': self.default_retry_delay
        }


# Global configuration instance
_config: Optional[SchedulerConfig] = None


def get_config() -> SchedulerConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = SchedulerConfig()
    return _config


def set_config(config: SchedulerConfig) -> None:
    """Set the global configuration instance"""
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration (useful for testing)"""
    global _config
    _config = None
