# session_manager.py
# Description: Session management with Redis caching, encryption, and automatic cleanup
#
# Imports
import asyncio
import base64
import hashlib
import hmac
import json
import os
import stat
import threading
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# File locking support (Unix/Windows)
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False
    fcntl = None  # type: ignore
#
# 3rd-party imports
import time

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from jose import jwt as jose_jwt
from jose.exceptions import JWSError, JWTError
from loguru import logger
from redis import asyncio as redis_async
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from tldw_Server_API.app.core.AuthNZ.crypto_utils import (
    derive_hmac_key,
    derive_hmac_key_candidates,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    InvalidSessionError,
    SessionError,
    SessionRevokedException,
)
from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.AuthNZ.token_blacklist import get_token_blacklist
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.testing import is_test_mode, is_truthy

_SESSION_MANAGER_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    JWTError,
    JWSError,
    RedisError,
    RedisConnectionError,
    InvalidSessionError,
    SessionError,
    SessionRevokedException,
)

try:
    from tldw_Server_API.app.core.config import settings as core_settings
except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - defensive fallback
    core_settings = {}

#######################################################################################################################
#
# Session Manager Class

class SessionManager:
    """Manages user sessions with database persistence, encryption, and optional Redis caching"""

    def __init__(
        self,
        db_pool: Optional[DatabasePool] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize session manager"""
        self.settings = settings or get_settings()
        self._provided_settings = settings
        self.db_pool = db_pool
        self._external_db_pool = db_pool is not None
        self.redis_client: Optional[redis_async.Redis] = None
        self.scheduler = AsyncIOScheduler()
        self._initialized = False
        self.cipher_suite: Optional[Fernet] = None
        self._fernet_candidates: list[Fernet] = []
        self._persisted_key_path: Optional[Path] = None
        self._ephemeral_cache: dict[str, tuple[str, float]] = {}
        self._ephemeral_lock = threading.Lock()
        self._init_encryption()

    async def initialize(self):
        """Initialize session manager and start cleanup scheduler"""
        if self._initialized:
            return

        # Get database pool
        self.db_pool = await self._ensure_db_pool()

        # Initialize Redis if configured
        if self.settings.REDIS_URL:
            try:
                self.redis_client = redis_async.from_url(
                    self.settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    max_connections=self.settings.REDIS_MAX_CONNECTIONS,
                    health_check_interval=30
                )

                # Test connection
                await self.redis_client.ping()
                logger.info("Redis connected for session caching")

            except (RedisConnectionError, RedisError) as e:
                logger.warning(f"Redis unavailable, using database only: {e}")
                self.redis_client = None

        # Schedule inline cleanup only for explicit fallback modes.
        # The dedicated AuthNZ scheduler owns periodic cleanup in normal startup.
        enable_inline_sched = False
        try:
            if is_truthy(str(os.getenv("AUTHNZ_SESSION_MANAGER_INLINE_SCHEDULER", "")).strip().lower()):
                enable_inline_sched = True
            if is_truthy(str(os.getenv("AUTHNZ_SCHEDULER_DISABLED", "")).strip().lower()):
                # Global AuthNZ scheduler is disabled: keep session cleanup alive via local fallback.
                enable_inline_sched = True
            # In general test mode, default to disabled unless explicitly overridden
            if is_test_mode() and not is_truthy(str(os.getenv("AUTHNZ_SCHEDULER_ENABLED", "")).strip().lower()):
                enable_inline_sched = False
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:
            pass
        if enable_inline_sched and self.settings.SESSION_CLEANUP_INTERVAL_HOURS > 0:
            self.scheduler.add_job(
                self.cleanup_expired_sessions,
                trigger=IntervalTrigger(hours=self.settings.SESSION_CLEANUP_INTERVAL_HOURS),
                id='session_cleanup',
                replace_existing=True,
                max_instances=1
            )
            self.scheduler.start()
            logger.info(
                f"Session cleanup scheduled every {self.settings.SESSION_CLEANUP_INTERVAL_HOURS} hours"
            )

        self._initialized = True
        logger.info("SessionManager initialized with encryption enabled")

    async def _ensure_db_pool(self) -> DatabasePool:
        """Ensure we have a database pool compatible with the current event loop."""
        current_settings = get_settings()

        if not self._external_db_pool:
            global_pool = await get_db_pool()
            if self.db_pool is not global_pool:
                logger.debug("SessionManager adopting refreshed AuthNZ DatabasePool instance")
                self.db_pool = global_pool
            self.settings = current_settings
        else:
            self.settings = current_settings

        if not self.db_pool:
            self.db_pool = await get_db_pool()
            return self.db_pool

        pool_ref = getattr(self.db_pool, "pool", None)
        pool_closed = bool(pool_ref) and getattr(pool_ref, "closed", False)

        loop_changed = False
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        stored_loop = getattr(self.db_pool, "_loop", None)
        if pool_ref and stored_loop and current_loop and stored_loop is not current_loop:
            loop_changed = True

        if pool_closed or loop_changed:
            logger.debug(
                "SessionManager refreshing database pool "
                f"(pool_closed={pool_closed}, loop_changed={loop_changed})"
            )
            if not self._external_db_pool:
                await reset_db_pool()
                self.db_pool = await get_db_pool()
                return self.db_pool
            await self.db_pool.close()
            await self.db_pool.initialize()
            return self.db_pool

        if not getattr(self.db_pool, "_initialized", False):
            await self.db_pool.initialize()

        return self.db_pool

    def _init_encryption(self):
        """Initialize encryption for session tokens"""
        key_materials = self._get_or_create_encryption_key()
        if not key_materials:
            raise ValueError("Session encryption key derivation failed")
        self._fernet_candidates = [Fernet(key) for key in key_materials]
        self.cipher_suite = self._fernet_candidates[0]
        logger.debug("Session token encryption initialized")

    def _get_or_create_encryption_key(self) -> list[bytes]:
        """Resolve ordered list of candidate encryption keys (primary first)."""
        key_bytes: list[bytes] = []
        seen: set[bytes] = set()

        def _append(candidate: Optional[bytes]) -> None:
            if not candidate:
                return
            if candidate not in seen:
                seen.add(candidate)
                key_bytes.append(candidate)

        # Explicit configuration wins
        explicit_key = getattr(self.settings, "SESSION_ENCRYPTION_KEY", None)
        if explicit_key:
            if isinstance(explicit_key, str):
                raw = explicit_key.strip().encode("utf-8")
            elif isinstance(explicit_key, bytes):
                raw = explicit_key
            else:
                raise ValueError("SESSION_ENCRYPTION_KEY must be str or bytes containing a Fernet key")
            try:
                decoded = base64.urlsafe_b64decode(raw)
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
                raise ValueError("SESSION_ENCRYPTION_KEY must be urlsafe base64-encoded") from exc
            if len(decoded) != 32:
                raise ValueError("SESSION_ENCRYPTION_KEY must decode to 32 bytes for Fernet compatibility")
            _append(raw)
        else:
            # If the preferred persistence location exists but is invalid, generate and
            # persist there first (even if a fallback key exists elsewhere). This repairs
            # broken or placeholder files and respects symlink resolution/security checks.
            preferred_path: Optional[Path] = None
            try:
                preferred_path = self._resolve_persisted_key_path()
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Session key: failed to resolve preferred persisted key path: {exc}")
                preferred_path = None

            def _exists_and_invalid(p: Optional[Path]) -> bool:
                try:
                    return bool(p) and Path(p).exists() and (not self._is_valid_key_file(Path(p)))
                except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(f"Session key: failed to inspect preferred path {p}: {exc}")
                    return False

            if _exists_and_invalid(preferred_path):
                generated = Fernet.generate_key()
                if self._persist_session_key(generated):
                    _append(generated)
                else:
                    # Persistence to the preferred path failed. Try to load any other
                    # persisted key while explicitly ignoring the known-bad preferred path.
                    def _read_valid_key_from_path(p: Optional[Path]) -> Optional[bytes]:
                        if not p:
                            return None
                        try:
                            if not p.exists():
                                return None
                            content = p.read_text(encoding="utf-8").strip()
                            if not content:
                                return None
                            decoded = base64.urlsafe_b64decode(content.encode("utf-8"))
                            if len(decoded) != 32:
                                return None
                            # Record discovered valid path
                            self._persisted_key_path = p
                            return content.encode("utf-8")
                        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as _exc:
                            logger.debug(f"Session key: failed reading candidate key at {p}: {_exc}")
                            return None

                    # Build alternate candidates explicitly excluding preferred_path
                    other_candidates: list[Path] = []
                    try:
                        ap = self._resolve_api_key_path()
                        if ap and preferred_path and ap != preferred_path:
                            other_candidates.append(ap)
                    except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as _e:
                        logger.debug(f"Session key: failed resolving API key path: {_e}")
                    try:
                        pp = self._resolve_project_root_key_path()
                        if pp and preferred_path and pp != preferred_path:
                            other_candidates.append(pp)
                    except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as _e:
                        logger.debug(f"Session key: failed constructing project-root key path: {_e}")

                    persisted_key: Optional[bytes] = None
                    for cand in other_candidates:
                        persisted_key = _read_valid_key_from_path(cand)
                        if persisted_key:
                            break

                    if persisted_key:
                        logger.warning(
                            'Session key: preferred path invalid and persistence failed; using alternate persisted key from {}',
                            str(self._persisted_key_path),
                        )
                        _append(persisted_key)
                    else:
                        # No alternate persisted key available. Use the generated key in-memory
                        # to keep the service functional, then attempt to persist to an alternate
                        # safe location (or repair the invalid file) best-effort.
                        logger.warning(
                            'Session key: persistence failed at preferred path {} and no alternate persisted key found; proceeding with in-memory key and attempting repair.',
                            str(preferred_path),
                        )
                        _append(generated)

                        # Try to persist to an alternate destination first (API path or project root)
                        alt_candidates: list[Path] = []
                        try:
                            ap = self._resolve_api_key_path()
                            if ap and (not preferred_path or ap != preferred_path):
                                alt_candidates.append(ap)
                        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as _e:
                            logger.debug(f"Session key: could not resolve API key path for alternate persistence: {_e}")
                        try:
                            pp = self._resolve_project_root_key_path()
                            if pp and (not preferred_path or pp != preferred_path):
                                alt_candidates.append(pp)
                        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as _e:
                            logger.debug(f"Session key: could not compute project-root path for alternate persistence: {_e}")

                        persisted_anywhere = False
                        original_target: Optional[Path] = self._persisted_key_path
                        for dest in alt_candidates + ([preferred_path] if preferred_path else []):
                            if not dest:
                                continue
                            try:
                                # If attempting to rewrite the known-bad file, create a backup first
                                if preferred_path and dest == preferred_path:
                                    try:
                                        if dest.exists():
                                            backup = dest.with_suffix(dest.suffix + ".bak")
                                            try:
                                                dest.rename(backup)
                                                logger.info(f"Session key: backed up invalid key file to {backup}")
                                            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as _be:
                                                logger.debug(f"Session key: backup of invalid key file failed: {_be}")
                                    except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as _ce:
                                        logger.debug(f"Session key: could not check/backup invalid key file: {_ce}")

                                # Force persistence target
                                self._persisted_key_path = dest
                                if self._persist_session_key(generated):
                                    logger.info(f"Session key: persisted generated key to alternate path {dest}")
                                    persisted_anywhere = True
                                    break
                                else:
                                    logger.debug(f"Session key: alternate persistence attempt failed for {dest}")
                            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as _pe:
                                logger.debug(f"Session key: exception during alternate persistence to {dest}: {_pe}")
                            finally:
                                # If persistence failed, restore original pointer before next attempt
                                if not persisted_anywhere:
                                    self._persisted_key_path = original_target

                        if not persisted_anywhere:
                            logger.warning(
                                "Session key: unable to persist generated key after repair attempts; running with in-memory key only."
                            )
            else:
                # Normal path: use persisted key if found; otherwise, generate and persist
                # Use file lock to prevent race conditions during concurrent initialization
                key_path = self._persisted_key_path or self._resolve_persisted_key_path()
                if key_path:
                    try:
                        with self._key_file_lock(key_path):
                            # Re-check after acquiring lock (another process may have created it)
                            persisted_key = self._load_persisted_session_key()
                            if persisted_key:
                                _append(persisted_key)
                            else:
                                generated = Fernet.generate_key()
                                if self._persist_session_key(generated):
                                    _append(generated)
                                else:
                                    logger.warning("Failed to persist session encryption key; falling back to derived secrets.")
                    except RuntimeError as lock_err:
                        # Lock acquisition failed - fall back to unlocked behavior with warning
                        logger.warning(f"Could not acquire key file lock: {lock_err}; proceeding without lock")
                        persisted_key = self._load_persisted_session_key()
                        if persisted_key:
                            _append(persisted_key)
                        else:
                            generated = Fernet.generate_key()
                            if self._persist_session_key(generated):
                                _append(generated)
                            else:
                                logger.warning("Failed to persist session encryption key; falling back to derived secrets.")
                else:
                    # No key path available - use unlocked behavior
                    persisted_key = self._load_persisted_session_key()
                    if persisted_key:
                        _append(persisted_key)
                    else:
                        generated = Fernet.generate_key()
                        if self._persist_session_key(generated):
                            _append(generated)
                        else:
                            logger.warning("Failed to persist session encryption key; falling back to derived secrets.")

        # Always include derived secrets for backward compatibility / fallback (includes secondary secrets)
        for derived in self._derive_secret_key_candidates():
            _append(derived)

        if not key_bytes:
            test_mode = is_test_mode()
            pytest_active = os.getenv("PYTEST_CURRENT_TEST") is not None
            if test_mode or pytest_active:
                logger.warning("Generating temporary session encryption key for test context.")
                _append(Fernet.generate_key())
            else:
                raise ValueError(
                    "Session encryption key is not configured. "
                    "Set SESSION_ENCRYPTION_KEY or ensure Config_Files/session_encryption.key is writable."
                )
        return key_bytes

    def _derive_secret_key_candidates(self) -> list[bytes]:
        """Derive deterministic Fernet keys from configured secret material."""
        secrets_order: list[Optional[str | bytes]] = []

        def _add_secret(value: Optional[str | bytes]) -> None:
            if value:
                secrets_order.append(value)

        if getattr(self.settings, "AUTH_MODE", "single_user") == "single_user":
            _add_secret(getattr(self.settings, "SINGLE_USER_API_KEY", None))

        _add_secret(getattr(self.settings, "API_KEY_PEPPER", None))
        _add_secret(getattr(self.settings, "JWT_SECRET_KEY", None))
        _add_secret(getattr(self.settings, "JWT_PRIVATE_KEY", None))
        # NOTE: JWT_PUBLIC_KEY is intentionally excluded - public keys are not secrets
        # and should never be used as cryptographic key material for encryption
        _add_secret(getattr(self.settings, "JWT_SECONDARY_SECRET", None))
        _add_secret(getattr(self.settings, "JWT_SECONDARY_PRIVATE_KEY", None))
        # NOTE: JWT_SECONDARY_PUBLIC_KEY is also excluded for the same reason

        derived_keys: list[bytes] = []
        seen: set[bytes] = set()

        for secret in secrets_order:
            if not secret:
                continue
            raw = secret if isinstance(secret, bytes) else str(secret).encode("utf-8")
            # NOTE: Static salt is used for backward compatibility with existing sessions.
            # This is acceptable because:
            # 1. Input (raw) is already high-entropy secret material (JWT keys, API keys)
            # 2. PBKDF2 with 600k iterations provides sufficient key stretching
            # For new deployments, prefer setting SESSION_ENCRYPTION_KEY directly
            # with a cryptographically random 32-byte key.
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"session_encryption_salt_v1",
                iterations=600000,  # OWASP 2023 recommendation for PBKDF2-HMAC-SHA256
            )
            key_material = base64.urlsafe_b64encode(kdf.derive(raw))
            if key_material not in seen:
                seen.add(key_material)
                derived_keys.append(key_material)
        return derived_keys

    @contextmanager
    def _key_file_lock(self, path: Path, timeout: float = 5.0):
        """Context manager for file locking during key operations.

        Uses fcntl.flock on Unix systems for proper file locking.
        Falls back to a simple lock file on systems without fcntl.

        Args:
            path: Path to the key file (lock file will be path.lock)
            timeout: Maximum time to wait for lock in seconds

        Raises:
            RuntimeError: If lock cannot be acquired within timeout
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_fd = None

        try:
            # Ensure parent directory exists
            lock_path.parent.mkdir(parents=True, exist_ok=True)

            if _HAS_FCNTL:
                # Unix: Use proper file locking
                lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
                start_time = time.time()
                while True:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        if time.time() - start_time > timeout:
                            raise RuntimeError(
                                f"Failed to acquire lock on {lock_path} within {timeout}s"
                            ) from None
                        time.sleep(0.1)
            else:
                # Windows/other: Use exclusive file creation as a lock
                start_time = time.time()
                while True:
                    try:
                        lock_fd = os.open(
                            str(lock_path),
                            os.O_CREAT | os.O_EXCL | os.O_RDWR,
                            0o600
                        )
                        break
                    except FileExistsError:
                        # Check if lock is stale (older than timeout * 2)
                        try:
                            lock_stat = os.stat(lock_path)
                            if time.time() - lock_stat.st_mtime > timeout * 2:
                                os.unlink(lock_path)
                                continue
                        except (OSError, FileNotFoundError):
                            pass

                        if time.time() - start_time > timeout:
                            raise RuntimeError(
                                f"Failed to acquire lock on {lock_path} within {timeout}s"
                            ) from None
                        time.sleep(0.1)

            yield  # Lock acquired, execute protected code

        finally:
            # Release lock
            if lock_fd is not None:
                if _HAS_FCNTL:
                    with suppress(_SESSION_MANAGER_NONCRITICAL_EXCEPTIONS):
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                with suppress(_SESSION_MANAGER_NONCRITICAL_EXCEPTIONS):
                    os.close(lock_fd)
            # Clean up lock file (best effort)
            if not _HAS_FCNTL:
                with suppress(_SESSION_MANAGER_NONCRITICAL_EXCEPTIONS):
                    lock_path.unlink(missing_ok=True)

    def _persist_session_key(self, key: bytes) -> bool:
        """Persist generated session key to disk for reuse across restarts."""
        path = self._persisted_key_path or self._resolve_persisted_key_path()
        if not path:
            return False
        try:
            try:
                if path.is_symlink():
                    raise RuntimeError(f"Session encryption key path {path} must not be a symlink")
                if path.exists():
                    existing_stat = os.stat(path, follow_symlinks=False)
                    if not stat.S_ISREG(existing_stat.st_mode):
                        raise RuntimeError(f"Session encryption key path {path} is not a regular file")
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise RuntimeError(f"Unable to inspect existing session key at {path}: {exc}") from exc

            # Ensure parent directory exists with restricted permissions (best-effort 0o700)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                parent_stat = os.stat(path.parent, follow_symlinks=False)
                if not stat.S_ISDIR(parent_stat.st_mode):
                    raise RuntimeError(f"Session key directory {path.parent} is not a directory")
            except FileNotFoundError as exc:
                raise RuntimeError(f"Failed to prepare session key directory {path.parent}: {exc}") from exc
            try:
                os.chmod(path.parent, 0o700)
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:
                # Ignore if chmod not supported (e.g., on Windows)
                pass

            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW

            # SECURITY: Set restrictive umask before file creation to prevent TOCTOU race
            # This ensures the file is created with restricted permissions even if
            # another process tries to modify it between creation and chmod
            old_umask = None
            if hasattr(os, "umask"):
                old_umask = os.umask(0o077)  # Only owner can read/write/execute

            # Write the key with 0o600 permissions so only the owner can read it
            try:
                fd = os.open(str(path), flags, 0o600)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        handle.write(key.decode("utf-8"))
                except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:
                    with suppress(_SESSION_MANAGER_NONCRITICAL_EXCEPTIONS):
                        os.close(fd)
                    raise
            except OSError as exc:
                raise RuntimeError(f"Failed to open session encryption key file {path}: {exc}") from exc
            finally:
                # Restore umask after file operations complete
                if old_umask is not None:
                    os.umask(old_umask)

            try:
                os.chmod(path, 0o600)
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as chmod_exc:
                # Log warning instead of silently ignoring - this could be a security issue
                logger.warning(f"Failed to set permissions on session key file {path}: {chmod_exc}")

            try:
                st = os.stat(path, follow_symlinks=False)
                if not stat.S_ISREG(st.st_mode):
                    raise RuntimeError(f"Session encryption key {path} is not a regular file")
                if hasattr(os, "getuid") and st.st_uid != os.getuid():
                    raise RuntimeError(f"Session encryption key {path} is not owned by the current user")
                # Also check group ownership for extra security
                if hasattr(os, "getgid") and hasattr(os, "getuid"):
                    # Verify file mode is restrictive (no group/other access)
                    mode = stat.S_IMODE(st.st_mode)
                    if mode & (stat.S_IRWXG | stat.S_IRWXO):
                        logger.warning(
                            f"Session encryption key {path} has permissive mode {oct(mode)}. "
                            "Expected 0o600 (owner read/write only)."
                        )
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                with suppress(_SESSION_MANAGER_NONCRITICAL_EXCEPTIONS):
                    path.unlink(missing_ok=True)
                raise RuntimeError(f"Persisted session encryption key failed validation: {exc}") from exc
            self._persisted_key_path = path
            return True
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"Unable to persist session encryption key to {path}: {exc}")
            return False

    def _load_persisted_session_key(self) -> Optional[bytes]:
        """Load persisted session encryption key if available.

        Preferred location (default): tldw_Server_API/Config_Files/session_encryption.key
        Back-compat fallback: PROJECT_ROOT/Config_Files/session_encryption.key

        Legacy behavior can be forced by setting SESSION_KEY_STORAGE=project (or
        compatible aliases), which makes PROJECT_ROOT/Config_Files primary again.
        """
        # Build candidate paths honoring optional override
        prefer_api_path = self._session_key_storage_mode() == "api"
        candidate_paths: list[Path] = []
        primary_path: Optional[Path] = None
        api_path: Optional[Path] = None
        # Resolve both paths safely
        try:
            api_path = self._persisted_key_path or self._resolve_api_key_path()
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"failed to resolve persisted API key path: {e}")
            api_path = None
        try:
            primary_path = self._resolve_project_root_key_path()
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"failed to construct primary session_encryption.key path: {e}")
            primary_path = None

        if prefer_api_path:
            if api_path:
                candidate_paths.append(api_path)
            if primary_path:
                candidate_paths.append(primary_path)
        else:
            if primary_path:
                candidate_paths.append(primary_path)
            if api_path and (not candidate_paths or api_path != candidate_paths[0]):
                candidate_paths.append(api_path)

        # If API path preference is enabled, migrate a valid key from project root
        # to the API component path when the latter is missing or invalid.
        if prefer_api_path and api_path and primary_path:
            try:
                self._maybe_migrate_key_to_api_path(primary_path, api_path)
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Session key migration skipped due to error: {exc}")

        for path in candidate_paths:
            try:
                if not path.exists():
                    continue
                if not self._is_valid_key_file(path):
                    logger.warning(
                        f"Persisted session encryption key at {path} is invalid or insecure; ignoring."
                    )
                    continue
                content = path.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                # Use the first valid candidate found
                self._persisted_key_path = path
                # Warn only on true fallbacks. If storage preference is API path and
                # the key was loaded from that API path, do not warn.
                if (
                    primary_path
                    and path != primary_path
                    and not (prefer_api_path and api_path and path == api_path)
                ):
                    logger.warning(f"Using persisted session_encryption.key at alternate location: {path}")
                return content.encode("utf-8")
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Failed to read persisted session encryption key from {path}: {exc}")
                continue
        return None

    def _resolve_api_key_path(self) -> Optional[Path]:
        """Return the tldw_Server_API/Config_Files path for the key."""
        try:
            api_root = Path(__file__).resolve().parent.parent.parent.parent
            config_dir = (api_root / "Config_Files").resolve()
            return config_dir / "session_encryption.key"
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:
            return None

    def _resolve_project_root_key_path(self) -> Optional[Path]:
        """Return the PROJECT_ROOT/Config_Files path for legacy compatibility."""
        try:
            project_root = core_settings.get("PROJECT_ROOT") if core_settings else None
            if not project_root:
                return None
            config_dir = (Path(project_root) / "Config_Files").resolve()
            return config_dir / "session_encryption.key"
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:
            return None

    def _session_key_storage_mode(self) -> str:
        """Resolve session key storage mode (`api` default, `project` legacy)."""
        raw = str(os.getenv("SESSION_KEY_STORAGE", "")).strip().lower()
        if raw in {"project", "project_root", "root", "repo", "legacy"}:
            return "project"
        return "api"

    def _resolve_persisted_key_path(self) -> Optional[Path]:
        """Determine filesystem location for persisted session key.

        By default, persist under the API component directory
        (tldw_Server_API/Config_Files). This avoids creating root-level Config_Files
        directories in deployments where PROJECT_ROOT points at the repository root.

        Set environment variable SESSION_KEY_STORAGE=project (legacy aliases are
        supported) to force PROJECT_ROOT/Config_Files/session_encryption.key.
        """
        if self._session_key_storage_mode() == "api":
            path = self._resolve_api_key_path()
            if path is not None:
                return path
        else:
            path = self._resolve_project_root_key_path()
            if path is not None:
                return path
        # Fallback to whichever location is still resolvable.
        return self._resolve_project_root_key_path() or self._resolve_api_key_path()

    def _is_valid_key_content(self, content: str) -> bool:
        try:
            decoded = base64.urlsafe_b64decode(content.encode("utf-8"))
            return len(decoded) == 32
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:
            return False

    def _is_valid_key_file(self, path: Path) -> bool:
        try:
            if not path.exists():
                return False
            if not path.is_file():
                return False
            try:
                st = os.stat(path, follow_symlinks=False)
                if not stat.S_ISREG(st.st_mode):
                    return False
                if hasattr(os, "getuid") and st.st_uid != os.getuid():
                    logger.warning(f"Session key file {path} is not owned by the current user; ignoring.")
                    return False
                mode = stat.S_IMODE(st.st_mode)
                if mode & (stat.S_IRWXG | stat.S_IRWXO):
                    logger.warning(
                        f"Session key file {path} has insecure permissions {oct(mode)}; ignoring."
                    )
                    return False
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Failed to inspect session key file {path}: {exc}")
                return False
            content = path.read_text(encoding="utf-8").strip()
            return bool(content) and self._is_valid_key_content(content)
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:
            return False

    def _maybe_migrate_key_to_api_path(self, source_primary: Path, dest_api: Path) -> None:
        """If a valid key exists at project root but not at API path, copy it over.

        Preconditions: This runs only when API storage mode is active.
        """
        try:
            # If API path already has a valid key, nothing to do
            if self._is_valid_key_file(dest_api):
                return
            # If source has a valid key, copy to dest
            if not self._is_valid_key_file(source_primary):
                return
            with suppress(_SESSION_MANAGER_NONCRITICAL_EXCEPTIONS):
                dest_api.parent.mkdir(parents=True, exist_ok=True)
            payload = source_primary.read_text(encoding="utf-8").strip()

            # Write atomically; if file exists but is invalid, replace it
            tmp_path = dest_api.with_suffix(".tmp")
            try:
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                os.chmod(tmp_path, 0o600)
                # Replace destination
                try:
                    tmp_path.replace(dest_api)
                except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:
                    # If replace fails, try unlink + rename
                    with suppress(_SESSION_MANAGER_NONCRITICAL_EXCEPTIONS):
                        dest_api.unlink(missing_ok=True)
                    tmp_path.rename(dest_api)
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:
                    pass
            # Validate destination and record
            if not self._is_valid_key_file(dest_api):
                raise RuntimeError("Migrated session key failed validation at API path")
            self._persisted_key_path = dest_api
            logger.info(f"Migrated session_encryption.key to API path: {dest_api}")
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
            # Preserve visibility but allow critical validation failures to propagate
            logger.warning(f"Failed to migrate session_encryption.key to API path: {exc}")
            if isinstance(exc, RuntimeError):
                # Re-raise to allow callers to handle invalid-migration errors explicitly
                raise

    def _token_hash_candidates(self, token: str) -> list[str]:
        """Return ordered hash candidates for a token across active/legacy secrets."""
        hashes: list[str] = []
        candidate_keys: list[bytes] = []

        def _extend_from_settings(s: Optional[Settings]) -> None:
            if not s:
                return
            try:
                keys = derive_hmac_key_candidates(s)
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:
                keys = [derive_hmac_key(s)]
            for key in keys:
                if key not in candidate_keys:
                    candidate_keys.append(key)

        if self._provided_settings is not None:
            _extend_from_settings(self._provided_settings)
        _extend_from_settings(self.settings)
        pool_settings = getattr(self.db_pool, "settings", None)
        if pool_settings is not None:
            _extend_from_settings(pool_settings)

        for key in candidate_keys:
            digest = hmac.new(key, token.encode("utf-8"), hashlib.sha256).hexdigest()
            if digest not in hashes:
                hashes.append(digest)
        return hashes

    def hash_token(self, token: str) -> str:
        """Create HMAC-SHA256 of token for lookup/indexing (aligned with AuthNZ)."""
        candidates = self._token_hash_candidates(token)
        if not candidates:
            raise ValueError("Unable to derive token hash candidates")
        return candidates[0]

    def encrypt_token(self, token: str) -> str:
        """Encrypt a token for secure storage"""
        if not self.cipher_suite:
            self._init_encryption()

        encrypted = self.cipher_suite.encrypt(token.encode())
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')

    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt a stored token.

        Tries all key candidates in order for key rotation support.
        Logs metrics for monitoring key rotation health.
        """
        if not self.cipher_suite or not self._fernet_candidates:
            self._init_encryption()

        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_token.encode('utf-8'))
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to decode stored session token: {e}")
            log_counter("session_token_decode_error")
            raise InvalidSessionError("Failed to decrypt session token") from e

        last_error: Optional[Exception] = None
        num_candidates = len(self._fernet_candidates or [])
        errors_by_candidate: list[str] = []

        for idx, cipher in enumerate(self._fernet_candidates or []):
            try:
                decrypted = cipher.decrypt(encrypted_bytes)
                # Track which candidate succeeded for key rotation monitoring
                if idx > 0:
                    log_counter("session_decrypt_secondary_key_used")
                    logger.info(f"Session token decrypted with secondary key candidate {idx}")
                return decrypted.decode('utf-8')
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                last_error = exc
                errors_by_candidate.append(f"candidate[{idx}]: {type(exc).__name__}")
                logger.debug(f"Session token decryption failed with candidate {idx}: {exc}")
                continue

        # All candidates failed - log detailed error for debugging
        log_counter("session_decrypt_all_candidates_failed")
        logger.warning(
            f"Failed to decrypt token after examining {num_candidates} key candidates. "
            f"Errors: {', '.join(errors_by_candidate)}"
        )
        raise InvalidSessionError("Failed to decrypt session token") from last_error

    @staticmethod
    def _extract_token_metadata(token: Optional[str]) -> tuple[Optional[str], Optional[datetime]]:
        """Return (jti, expires_at) tuple without verifying signature."""
        if not token:
            return None, None
        try:
            claims = jose_jwt.get_unverified_claims(token)
            if claims is None:
                return None, None
            jti = claims.get("jti")
            exp = claims.get("exp")
            expires_at = None
            if isinstance(exp, (int, float)):
                # Use timezone-aware datetime to avoid naive datetime issues
                expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
            return jti, expires_at
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:
            return None, None

    @staticmethod
    def _get_unverified_claims(token: Optional[str]) -> Optional[dict[str, Any]]:
        """Return unverified JWT claims dict (best-effort)."""
        if not token:
            return None
        try:
            claims = jose_jwt.get_unverified_claims(token)
            return claims if isinstance(claims, dict) else None
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS:
            return None

    @staticmethod
    def _validate_token_binding(
        claims: Optional[dict[str, Any]],
        session_data: dict[str, Any],
        *,
        token_label: str,
    ) -> None:
        """Ensure token subject/session_id claims align with the session record."""
        if not claims:
            logger.debug(f"Refresh session: {token_label} token claims missing")
            raise InvalidSessionError()
        subject = claims.get("sub") or claims.get("user_id")
        if subject is None:
            logger.debug(f"Refresh session: {token_label} token missing subject claim")
            raise InvalidSessionError()
        try:
            subject_id = int(subject)
        except (TypeError, ValueError):
            logger.debug(f"Refresh session: {token_label} token subject is invalid")
            raise InvalidSessionError() from None
        if subject_id != int(session_data.get("user_id")):
            logger.debug(f"Refresh session: {token_label} token subject mismatch")
            raise InvalidSessionError()
        session_claim = claims.get("session_id")
        if session_claim is not None:
            try:
                session_claim_id = int(session_claim)
            except (TypeError, ValueError):
                logger.debug(f"Refresh session: {token_label} token session_id is invalid")
                raise InvalidSessionError() from None
            if session_claim_id != int(session_data.get("id")):
                logger.debug(f"Refresh session: {token_label} token session_id mismatch")
                raise InvalidSessionError()

    async def create_session(
        self,
        user_id: int,
        access_token: str,
        refresh_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None,
        expires_at_override: Optional[datetime] = None,
        refresh_expires_at_override: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Create a new session for a user

        Args:
            user_id: User's database ID
            access_token: JWT access token
            refresh_token: JWT refresh token
            ip_address: Client IP address
            user_agent: Client user agent string
            device_id: Optional device identifier
            expires_at_override: Optional override for access token expiry
            refresh_expires_at_override: Optional override for refresh token expiry

        Returns:
            Session information dictionary
        """
        start_time = time.perf_counter()
        log_counter("auth_session_create_attempt")
        if not self._initialized:
            await self.initialize()

        # Hash tokens for indexing/lookup
        access_hash = self.hash_token(access_token)
        refresh_hash = self.hash_token(refresh_token)

        # Encrypt tokens for secure storage
        encrypted_access = self.encrypt_token(access_token)
        encrypted_refresh = self.encrypt_token(refresh_token)
        expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
        refresh_expires_at = datetime.now(timezone.utc) + timedelta(
            days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS
        )

        access_jti, access_exp_override = self._extract_token_metadata(access_token)
        if access_exp_override:
            expires_at = access_exp_override
        refresh_jti, refresh_exp_override = self._extract_token_metadata(refresh_token)
        if refresh_exp_override:
            refresh_expires_at = refresh_exp_override
        # Explicit overrides win when supplied (e.g., MFA pending sessions).
        if expires_at_override:
            expires_at = expires_at_override
        if refresh_expires_at_override:
            refresh_expires_at = refresh_expires_at_override

        try:
            db_pool = await self._ensure_db_pool()
            repo = AuthnzSessionsRepo(db_pool)
            session_id = await repo.create_session_record(
                user_id=user_id,
                token_hash=access_hash,
                refresh_token_hash=refresh_hash,
                encrypted_token=encrypted_access,
                encrypted_refresh=encrypted_refresh,
                expires_at=expires_at,
                refresh_expires_at=refresh_expires_at,
                ip_address=ip_address,
                user_agent=user_agent,
                device_id=device_id,
                access_jti=access_jti,
                refresh_jti=refresh_jti,
            )

            # Cache in Redis if available
            if self.redis_client:
                await self._cache_session(
                    access_hash,
                    user_id,
                    session_id,
                    expires_at,
                    user_active=True,
                    revoked=False,
                )

            if self.settings.PII_REDACT_LOGS:
                logger.info("Created session [redacted]")
            else:
                logger.info(f"Created session {session_id} for user {user_id}")
            log_counter("auth_session_create_success")
            log_histogram(
                "auth_session_create_duration",
                time.perf_counter() - start_time,
            )

            return {
                "session_id": session_id,
                "user_id": user_id,
                "expires_at": expires_at.isoformat(),
                "access_token": access_token,
                "refresh_token": refresh_token,
            }

        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to create session: {e}")
            log_counter("auth_session_create_error")
            log_histogram(
                "auth_session_create_duration",
                time.perf_counter() - start_time,
            )
            raise SessionError(f"Failed to create session: {e}") from e

    async def validate_session(self, access_token: str) -> Optional[dict[str, Any]]:
        """
        Validate a session by access token

        Args:
            access_token: JWT access token

        Returns:
            Session data if valid, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        token_hash_candidates = self._token_hash_candidates(access_token)
        if not token_hash_candidates:
            logger.debug("validate_session received token with no hash candidates; treating as invalid")
            return None
        token_hash_primary = token_hash_candidates[0]
        matched_hash: Optional[str] = None
        cache_normalize_required = False
        cached: Optional[dict[str, Any]] = None
        if self.redis_client:
            for candidate_hash in token_hash_candidates:
                cached = await self._get_cached_session(candidate_hash)
                if cached:
                    break

        try:
            db_pool = await self._ensure_db_pool()
            repo = AuthnzSessionsRepo(db_pool)
            session_data: Optional[dict[str, Any]] = None

            # Attempt to reuse cached session_id to minimize lookups,
            # but always verify current DB state.
            if cached and cached.get("session_id") is not None:
                session_data = await repo.fetch_session_for_validation_by_id(
                    int(cached["session_id"])
                )
                if session_data:
                    matched_hash = session_data.get("token_hash")
                if not session_data and cached.get("session_id") is not None:
                    # Cache is stale; purge it.
                    await self._clear_session_cache(int(cached["session_id"]))

            if not session_data:
                for candidate_hash in token_hash_candidates:
                    session_data = (
                        await repo.fetch_session_for_validation_by_token_hash(
                            candidate_hash
                        )
                    )
                    if session_data:
                        matched_hash = (
                            session_data.get("token_hash") or candidate_hash
                        )
                        break

            if not session_data:
                return None

            if matched_hash is None:
                matched_hash = session_data.get("token_hash")

            user_active = bool(session_data.get("user_active"))
            revoked_flag = bool(session_data.get("revoked_at"))
            if not user_active:
                if self.settings.PII_REDACT_LOGS:
                    logger.warning("Session valid but user is inactive [redacted]")
                else:
                    logger.warning(
                        f"Session valid but user {session_data['user_id']} is inactive"
                    )
                return None

            if revoked_flag:
                if self.settings.PII_REDACT_LOGS:
                    logger.warning("Session revoked [redacted]")
                else:
                    logger.warning(f"Session {session_data['id']} was revoked")
                raise SessionRevokedException()

            # Use timing-safe comparison to prevent timing attacks
            if matched_hash and not hmac.compare_digest(matched_hash, token_hash_primary):
                try:
                    await repo.normalize_session_token_hash(
                        session_id=session_data["id"],
                        new_token_hash=token_hash_primary,
                    )
                    session_data["token_hash"] = token_hash_primary
                    cache_normalize_required = True
                except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as normalize_exc:
                    logger.warning(
                        'Failed to normalize session token hash for session {}: {}',
                        session_data.get("id"),
                        normalize_exc,
                    )

            await repo.update_last_activity(session_data["id"])

            # Outside of DB operations - refresh cache with validation status
            expires_at = session_data.get('expires_at')
            expires_at_dt: Optional[datetime] = None
            if isinstance(expires_at, str):
                expires_at_dt = datetime.fromisoformat(expires_at)
                # Ensure timezone-aware datetime
                if expires_at_dt.tzinfo is None:
                    expires_at_dt = expires_at_dt.replace(tzinfo=timezone.utc)
            elif isinstance(expires_at, datetime):
                expires_at_dt = expires_at
                if expires_at_dt.tzinfo is None:
                    expires_at_dt = expires_at_dt.replace(tzinfo=timezone.utc)

            # Always clear old cache if normalization was required (cache consistency)
            if self.redis_client and cache_normalize_required:
                await self._clear_session_cache(session_data['id'])

            # Update cache with new validation status
            if self.redis_client and expires_at_dt:
                await self._cache_session(
                    token_hash_primary,
                    session_data['user_id'],
                    session_data['id'],
                    expires_at_dt,
                    user_active=user_active,
                    revoked=revoked_flag,
                )

            return session_data

        except SessionRevokedException:
            raise
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error validating session: {e}")
            return None

    async def revoke_session(
        self,
        session_id: int,
        revoked_by: Optional[int] = None,
        reason: Optional[str] = None
    ):
        """Revoke a specific session"""
        if not self._initialized:
            await self.initialize()

        session_details: Optional[dict[str, Any]] = None
        try:
            db_pool = await self._ensure_db_pool()
            repo = AuthnzSessionsRepo(db_pool)
            session_details = await repo.revoke_session_record(
                session_id=session_id,
                revoked_by=revoked_by,
                reason=reason,
            )

            # Clear from cache
            if self.redis_client:
                await self._clear_session_cache(session_id)

            if self.settings.PII_REDACT_LOGS:
                logger.info("Revoked session [redacted]")
            else:
                logger.info(f"Revoked session {session_id}")

        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to revoke session: {e}")
            raise SessionError(f"Failed to revoke session: {e}") from e
        else:
            if session_details:
                await self._blacklist_session_tokens(
                    [session_details],
                    reason=reason,
                    revoked_by=revoked_by,
                )

    async def revoke_all_user_sessions(
        self,
        user_id: int,
        except_session_id: Optional[int] = None,
        reason: str = "User requested logout from all devices",
    ) -> int:
        """Revoke all sessions for a user, optionally except one"""
        if not self._initialized:
            await self.initialize()

        affected = 0
        try:
            db_pool = await self._ensure_db_pool()
            repo = AuthnzSessionsRepo(db_pool)
            affected = int(
                await repo.revoke_all_sessions_for_user(
                    user_id=user_id,
                    except_session_id=except_session_id,
                )
            )

            # Clear from cache
            if self.redis_client:
                await self._clear_user_sessions_cache(user_id)

            if self.settings.PII_REDACT_LOGS:
                logger.info("Revoked all sessions [redacted]")
            else:
                logger.info(f"Revoked all sessions for user {user_id}")

        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to revoke user sessions: {e}")
            raise SessionError(f"Failed to revoke sessions: {e}") from e

        # After sessions are marked revoked, ensure associated JTIs are blacklisted
        try:
            blacklist = get_token_blacklist()
            await blacklist.revoke_all_user_tokens(user_id, reason=reason)
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as bl_error:
            if self.settings.PII_REDACT_LOGS:
                logger.warning(f"Failed to blacklist tokens for authenticated user (details redacted): {bl_error}")
            else:
                logger.warning(f"Failed to blacklist tokens for user {user_id}: {bl_error}")

        return affected

    async def refresh_session(
        self,
        refresh_token: str,
        new_access_token: str,
        new_refresh_token: Optional[str] = None
    ) -> dict[str, Any]:
        """Refresh a session with new tokens"""
        if not self._initialized:
            await self.initialize()

        refresh_hash_candidates = self._token_hash_candidates(refresh_token)
        if not refresh_hash_candidates:
            raise InvalidSessionError()
        primary_refresh_hash = refresh_hash_candidates[0]
        new_access_hash = self.hash_token(new_access_token)
        encrypted_access_token = self.encrypt_token(new_access_token)
        access_jti, access_exp = self._extract_token_metadata(new_access_token)
        refresh_jti, refresh_exp = self._extract_token_metadata(new_refresh_token) if new_refresh_token else (None, None)
        expires_at = access_exp or (datetime.now(timezone.utc) + timedelta(
            minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES
        ))
        refresh_expires_at = None
        if new_refresh_token:
            refresh_expires_at = refresh_exp or (
                datetime.now(timezone.utc) + timedelta(days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS)
            )

        try:
            db_pool = await self._ensure_db_pool()
            repo = AuthnzSessionsRepo(db_pool)

            # Locate session using any legacy hash candidate
            session_data = await repo.find_active_session_by_refresh_hash_candidates(
                refresh_hash_candidates
            )
            if not session_data:
                raise InvalidSessionError()
            expected_access_hash = session_data.get("token_hash")
            expected_refresh_hash = session_data.get("refresh_token_hash")
            if not expected_access_hash or not expected_refresh_hash:
                raise InvalidSessionError()

            # Validate token subject/session binding before updating session records.
            try:
                access_claims = self._get_unverified_claims(new_access_token)
                self._validate_token_binding(access_claims, session_data, token_label="access")
                if new_refresh_token:
                    refresh_claims = self._get_unverified_claims(new_refresh_token)
                    self._validate_token_binding(refresh_claims, session_data, token_label="refresh")
            except InvalidSessionError:
                raise
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Refresh session: token binding validation failed: {exc}")
                raise InvalidSessionError() from exc

            if new_refresh_token:
                refresh_hash_update = self.hash_token(new_refresh_token)
                encrypted_refresh_token = self.encrypt_token(new_refresh_token)
            else:
                refresh_hash_update = primary_refresh_hash
                encrypted_refresh_token = self.encrypt_token(refresh_token)

            # Update session with new tokens
            updated = await repo.update_session_tokens_for_refresh(
                session_id=session_data["id"],
                expected_access_hash=expected_access_hash,
                expected_refresh_hash=expected_refresh_hash,
                new_access_hash=new_access_hash,
                access_jti=access_jti,
                expires_at=expires_at,
                encrypted_access_token=encrypted_access_token,
                refresh_hash_update=refresh_hash_update,
                refresh_jti=refresh_jti,
                refresh_expires_at=refresh_expires_at,
                encrypted_refresh_token=encrypted_refresh_token,
            )
            if not updated:
                # Compare-and-swap failed: session was refreshed/revoked/expired concurrently.
                raise InvalidSessionError()

            # Update cache
            if self.redis_client:
                await self._clear_session_cache(session_data["id"])
                await self._cache_session(
                    new_access_hash,
                    session_data["user_id"],
                    session_data["id"],
                    expires_at,
                    user_active=True,
                    revoked=False,
                )

            if self.settings.PII_REDACT_LOGS:
                logger.info("Refreshed session [redacted]")
            else:
                logger.info(f"Refreshed session {session_data['id']}")

            return {
                "session_id": session_data["id"],
                "user_id": session_data["user_id"],
                "expires_at": expires_at.isoformat(),
            }

        except InvalidSessionError:
            raise
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to refresh session: {e}")
            raise SessionError(f"Failed to refresh session: {e}") from e

    async def update_session_tokens(
        self,
        session_id: int,
        access_token: str,
        refresh_token: str
    ):
        """Update session with actual tokens after creation"""
        if not self._initialized:
            await self.initialize()

        try:
            # Hash tokens for storage
            access_token_hash = self.hash_token(access_token)
            refresh_token_hash = self.hash_token(refresh_token)
            access_jti, access_exp = self._extract_token_metadata(access_token)
            refresh_jti, refresh_exp = self._extract_token_metadata(refresh_token)
            encrypted_access_token = self.encrypt_token(access_token)
            encrypted_refresh_token = self.encrypt_token(refresh_token)

            db_pool = await self._ensure_db_pool()
            repo = AuthnzSessionsRepo(db_pool)
            user_id = await repo.update_session_tokens_after_creation(
                session_id=session_id,
                access_token_hash=access_token_hash,
                refresh_token_hash=refresh_token_hash,
                access_jti=access_jti,
                refresh_jti=refresh_jti,
                access_expires_at=access_exp,
                refresh_expires_at=refresh_exp,
                encrypted_access_token=encrypted_access_token,
                encrypted_refresh_token=encrypted_refresh_token,
            )

            expires_at_dt = access_exp
            if isinstance(expires_at_dt, str):
                try:
                    expires_at_dt = datetime.fromisoformat(expires_at_dt)
                except ValueError:
                    expires_at_dt = None
            if not expires_at_dt:
                expires_at_dt = datetime.now(timezone.utc) + timedelta(
                    minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES
                )

            if self.redis_client and user_id is not None:
                try:
                    await self._clear_session_cache(session_id)
                    await self._cache_session(
                        access_token_hash,
                        int(user_id),
                        session_id,
                        expires_at_dt,
                        user_active=True,
                        revoked=False,
                    )
                except RedisError:
                    pass

            logger.debug(f"Updated session {session_id} with token hashes")

        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to update session tokens: {e}")
            raise SessionError(f"Failed to update session tokens: {e}") from e

    async def is_token_blacklisted(self, token: str, jti: Optional[str] = None) -> bool:
        """
        Check if a token has been blacklisted/revoked

        Args:
            token: JWT token to check
            jti: Optional JWT ID (if already parsed by caller)

        Returns:
            True if token is blacklisted, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Fail-closed on missing token material
            if not token:
                logger.warning("is_token_blacklisted invoked without token; treating as revoked")
                return True

            # Determine JWT ID (JTI) if not provided
            jti_value = jti
            if not jti_value:
                try:
                    from jose import jwt as _jwt  # Lazy import to avoid top-level dependency
                    claims = _jwt.get_unverified_claims(token)
                    jti_value = claims.get("jti")
                except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                    logger.warning(f"Failed to extract JTI from token; treating as revoked: {exc}")
                    return True

            if not jti_value:
                logger.warning("Token missing JTI claim; treating as revoked")
                return True

            # Consult shared token blacklist (fail-closed on error)
            try:
                blacklist = get_token_blacklist()
                if await blacklist.is_blacklisted(jti_value):
                    return True
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                logger.error(f"Token blacklist check failed; treating token as revoked: {exc}")
                return True

            token_hashes = self._token_hash_candidates(token)

            # Check Redis cache first if available (JTI-aligned with TokenBlacklist)
            if self.redis_client:
                try:
                    redis_key = f"blacklist:{jti_value}"
                    if await self.redis_client.exists(redis_key):
                        return True
                except RedisError:
                    pass  # Fall back to database

            # Check database for revoked sessions
            db_pool = await self._ensure_db_pool()
            repo = AuthnzSessionsRepo(db_pool)
            return bool(await repo.has_revoked_session_for_token_hash_candidates(token_hashes))

        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error checking token blacklist; treating token as revoked: {e}")
            return True

    async def get_user_sessions(self, user_id: int) -> list[dict[str, Any]]:
        """Get all sessions for a user (alias for get_active_sessions)"""
        return await self.get_active_sessions(user_id)

    async def get_active_sessions(self, user_id: int) -> list[dict[str, Any]]:
        """Get all active sessions for a user"""
        if not self._initialized:
            await self.initialize()

        try:
            db_pool = await self._ensure_db_pool()
            repo = AuthnzSessionsRepo(db_pool)
            return await repo.get_active_sessions_for_user(user_id)
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions from database and cache"""
        if not self._initialized:
            await self.initialize()

        try:
            logger.info("Starting session cleanup...")

            db_pool = await self._ensure_db_pool()
            repo = AuthnzSessionsRepo(db_pool)
            deleted = await repo.cleanup_expired_sessions()

            if deleted:
                logger.info(f"Cleaned up {deleted} expired sessions")

            # Clean Redis cache
            if self.redis_client:
                await self._cleanup_redis_cache()

            return int(deleted or 0)

        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Session cleanup failed: {e}")
            return 0

    # Redis cache helpers
    async def _cache_session(
        self,
        token_hash: str,
        user_id: int,
        session_id: int,
        expires_at: datetime,
        *,
        user_active: bool = True,
        revoked: bool = False,
    ):
        """Cache session metadata in Redis (validation state included)."""
        if not self.redis_client:
            return

        try:
            cache_data = {
                "user_id": user_id,
                "session_id": session_id,
                "expires_at": expires_at.isoformat(),
                "user_active": bool(user_active),
                "revoked": bool(revoked),
            }

            # Calculate TTL - ensure expires_at is timezone-aware for comparison
            expires_at_aware = expires_at.replace(tzinfo=timezone.utc) if expires_at.tzinfo is None else expires_at
            ttl = int((expires_at_aware - datetime.now(timezone.utc)).total_seconds())
            if ttl > 0:
                # Cache session data
                await self.redis_client.setex(
                    f"session:{token_hash}",
                    ttl,
                    json.dumps(cache_data)
                )

                # Add to user's session set
                await self.redis_client.sadd(f"user:{user_id}:sessions", session_id)
                await self.redis_client.expire(f"user:{user_id}:sessions", ttl)

        except RedisError as e:
            logger.warning(f"Failed to cache session: {e}")

    async def _get_cached_session(self, token_hash: str) -> Optional[dict[str, Any]]:
        """Get session metadata from Redis cache (if still valid)."""
        if not self.redis_client:
            return None

        try:
            cached = await self.redis_client.get(f"session:{token_hash}")
            if cached:
                data = json.loads(cached)
                expires = datetime.fromisoformat(data['expires_at'])
                # Ensure timezone-aware comparison
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
                if expires > datetime.now(timezone.utc):
                    data.setdefault("user_active", True)
                    data.setdefault("revoked", False)
                    return data
                else:
                    # Expired, remove from cache
                    await self.redis_client.delete(f"session:{token_hash}")
            return None

        except RedisError:
            return None

    async def _clear_session_cache(self, session_id: int):
        """Clear specific session from cache"""
        if not self.redis_client:
            return

        try:
            # Find and delete session by scanning (not ideal but necessary)
            async for key in self.redis_client.scan_iter("session:*"):
                data = await self.redis_client.get(key)
                if data:
                    session_data = json.loads(data)
                    if session_data.get('session_id') == session_id:
                        await self.redis_client.delete(key)
                        break

        except RedisError:
            pass

    async def _clear_user_sessions_cache(self, user_id: int):
        """Clear all sessions for a user from cache"""
        if not self.redis_client:
            return

        try:
            # Clear each session
            async for key in self.redis_client.scan_iter("session:*"):
                data = await self.redis_client.get(key)
                if data:
                    session_data = json.loads(data)
                    if session_data.get('user_id') == user_id:
                        await self.redis_client.delete(key)

            # Clear user's session set
            await self.redis_client.delete(f"user:{user_id}:sessions")

        except RedisError:
            pass

    async def _cleanup_redis_cache(self):
        """Clean up expired sessions from Redis"""
        if not self.redis_client:
            return

        try:
            count = 0
            async for key in self.redis_client.scan_iter("session:*"):
                ttl = await self.redis_client.ttl(key)
                if ttl == -1:  # No expiry set
                    await self.redis_client.delete(key)
                    count += 1

            if count:
                logger.info(f"Cleaned {count} sessions from Redis cache")

        except RedisError as e:
            logger.warning(f"Redis cache cleanup failed: {e}")

    @staticmethod
    def _ephemeral_redis_key(key: str) -> str:
        return f"authnz:ephemeral:{key}"

    def _purge_ephemeral_cache_locked(self, now: float) -> None:
        if not self._ephemeral_cache:
            return
        expired = [k for k, (_, exp) in self._ephemeral_cache.items() if exp <= now]
        for key in expired:
            self._ephemeral_cache.pop(key, None)

    async def store_ephemeral_value(self, key: str, value: str, ttl_seconds: int) -> None:
        """Store an encrypted ephemeral value with TTL in Redis or memory."""
        if not self._initialized:
            await self.initialize()
        if not key:
            raise ValueError("Ephemeral cache key is required")
        ttl = max(int(ttl_seconds), 1)
        encrypted = self.encrypt_token(value)
        if self.redis_client:
            try:
                await self.redis_client.setex(self._ephemeral_redis_key(key), ttl, encrypted)
                return
            except RedisError as exc:
                logger.warning("Failed to cache ephemeral value in Redis: {}", exc)
        now = time.monotonic()
        with self._ephemeral_lock:
            self._purge_ephemeral_cache_locked(now)
            self._ephemeral_cache[key] = (encrypted, now + ttl)

    async def get_ephemeral_value(self, key: str) -> Optional[str]:
        """Retrieve and decrypt an ephemeral value if present and unexpired."""
        if not self._initialized:
            await self.initialize()
        if not key:
            return None
        if self.redis_client:
            try:
                cached = await self.redis_client.get(self._ephemeral_redis_key(key))
                if cached:
                    return self.decrypt_token(cached)
                return None
            except RedisError as exc:
                logger.warning("Failed to read ephemeral value from Redis: {}", exc)
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to decrypt ephemeral value: {}", exc)
                return None
        now = time.monotonic()
        with self._ephemeral_lock:
            entry = self._ephemeral_cache.get(key)
            if not entry:
                return None
            encrypted, expires_at = entry
            if expires_at <= now:
                self._ephemeral_cache.pop(key, None)
                return None
        try:
            return self.decrypt_token(encrypted)
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Failed to decrypt ephemeral value: {}", exc)
            return None

    async def consume_ephemeral_value(self, key: str) -> Optional[str]:
        """Atomically retrieve and delete an ephemeral value if present and unexpired."""
        if not self._initialized:
            await self.initialize()
        if not key:
            return None
        if self.redis_client:
            redis_key = self._ephemeral_redis_key(key)
            try:
                cached = await self.redis_client.eval(
                    """
                    local value = redis.call('GET', KEYS[1])
                    if value then
                        redis.call('DEL', KEYS[1])
                    end
                    return value
                    """,
                    1,
                    redis_key,
                )
                if cached:
                    return self.decrypt_token(cached)
                return None
            except RedisError as exc:
                logger.warning("Failed to consume ephemeral value from Redis: {}", exc)
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to decrypt consumed ephemeral value: {}", exc)
                return None
        now = time.monotonic()
        with self._ephemeral_lock:
            entry = self._ephemeral_cache.pop(key, None)
            if not entry:
                return None
            encrypted, expires_at = entry
            if expires_at <= now:
                return None
        try:
            return self.decrypt_token(encrypted)
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Failed to decrypt consumed ephemeral value: {}", exc)
            return None

    async def delete_ephemeral_value(self, key: str) -> None:
        """Delete an ephemeral value from Redis and in-memory cache."""
        if not self._initialized:
            await self.initialize()
        if not key:
            return
        if self.redis_client:
            try:
                await self.redis_client.delete(self._ephemeral_redis_key(key))
            except RedisError as exc:
                logger.warning("Failed to delete ephemeral value from Redis: {}", exc)
        with self._ephemeral_lock:
            self._ephemeral_cache.pop(key, None)

    @staticmethod
    def _coerce_datetime(value: Optional[Any]) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return None
        return None

    async def _blacklist_session_tokens(
        self,
        sessions: list[dict[str, Any]],
        *,
        reason: Optional[str],
        revoked_by: Optional[int],
    ) -> None:
        if not sessions:
            return
        try:
            blacklist = get_token_blacklist()
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"AuthNZ blacklist unavailable for session revocation: {exc}")
            return

        for entry in sessions:
            user_id = entry.get("user_id")
            access_jti = entry.get("access_jti")
            refresh_jti = entry.get("refresh_jti")
            access_exp = self._coerce_datetime(entry.get("expires_at"))
            refresh_exp = self._coerce_datetime(entry.get("refresh_expires_at"))

            if access_jti and access_exp:
                with suppress(_SESSION_MANAGER_NONCRITICAL_EXCEPTIONS):
                    blacklist.hint_blacklisted(access_jti, access_exp)
                try:
                    await blacklist.revoke_token(
                        jti=access_jti,
                        expires_at=access_exp,
                        user_id=user_id,
                        token_type="access",
                        reason=reason,
                        revoked_by=revoked_by,
                        ip_address=None,
                    )
                except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(f"Failed to persist access-token blacklist entry {access_jti}: {exc}")

            if refresh_jti and refresh_exp:
                with suppress(_SESSION_MANAGER_NONCRITICAL_EXCEPTIONS):
                    blacklist.hint_blacklisted(refresh_jti, refresh_exp)
                try:
                    await blacklist.revoke_token(
                        jti=refresh_jti,
                        expires_at=refresh_exp,
                        user_id=user_id,
                        token_type="refresh",
                        reason=reason,
                        revoked_by=revoked_by,
                        ip_address=None,
                    )
                except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(f"Failed to persist refresh-token blacklist entry {refresh_jti}: {exc}")

    async def shutdown(self):
        """Shutdown session manager and cleanup"""
        # Guard against shutdown being called after the event loop has closed
        try:
            if self.scheduler.running:
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = None
                if loop is None or not loop.is_closed():
                    self.scheduler.shutdown(wait=False)
        except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
            # In tests, teardown may run after the loop is closed; ignore scheduler shutdown errors
            logger.debug(f"SessionManager scheduler shutdown skipped: {e}")

        if self.redis_client:
            try:
                await self.redis_client.close()
            except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Ignoring Redis client shutdown error: {e}")
            finally:
                self.redis_client = None

        logger.info("SessionManager shutdown complete")


#######################################################################################################################
#
# Module Functions

# Global instance
_session_manager: Optional[SessionManager] = None


async def get_session_manager() -> SessionManager:
    """Get session manager singleton instance"""
    global _session_manager
    if not _session_manager:
        _session_manager = SessionManager()
        await _session_manager.initialize()
    return _session_manager


async def reset_session_manager():
    """Reset session manager (for testing)"""
    global _session_manager
    if _session_manager:
        await _session_manager.shutdown()
    _session_manager = None


#
# End of session_manager.py
#######################################################################################################################
