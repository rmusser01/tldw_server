# DB_Deps.py
# Description: Manages user-specific database instances based on application mode.
#
# Imports
import os
import threading
from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

# 3rd-party Libraries
from fastapi import Depends, HTTPException, Request, status
from loguru import logger

try:
    from cachetools import LRUCache
    _HAS_CACHETOOLS = True
except ImportError:
    _HAS_CACHETOOLS = False
    logger.warning("cachetools not found. User DB instance cache will grow indefinitely. Install with: pip install cachetools")

# Local Imports
# Import the settings dictionary
# Import the primary user identification dependency and User model
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    reset_managed_sqlite_backends,
)
from tldw_Server_API.app.core.DB_Management.DB_Manager import get_content_backend_instance
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import MediaDbFactory, MediaDbSession
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError,
    SchemaError,
)
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope
from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode

#######################################################################################################################

# Transitional compatibility alias for legacy monkeypatch/test surfaces.
MediaDatabase = Any

# Note: Do not cache USER_DB_BASE_DIR at import time. Tests may set USER_DB_BASE_DIR
# via environment after module import. We will resolve it at request time in helpers.

# --- Global Cache for Media DB Factories ---
MAX_CACHED_DB_INSTANCES = 100  # Adjust as needed

if _HAS_CACHETOOLS:
    # Legacy cache retained for compatibility with older reset/test helpers.
    _user_db_instances: LRUCache = LRUCache(maxsize=MAX_CACHED_DB_INSTANCES)
    logger.info(f"Using LRUCache for user DB instances (maxsize={MAX_CACHED_DB_INSTANCES}).")
    # Keyed by user ID (int)
    _media_db_factories: LRUCache = LRUCache(maxsize=MAX_CACHED_DB_INSTANCES)
else:
    # Legacy cache retained for compatibility with older reset/test helpers.
    _user_db_instances: dict[int, MediaDatabase] = {} # Fallback to standard dict
    _media_db_factories: dict[int, MediaDbFactory] = {}

_user_db_lock = threading.Lock() # Protects access to _media_db_factories and legacy cache state

#######################################################################################################################

# --- Helper Functions ---

def _get_db_path_for_user(user_id: int) -> Path:
    """
    Determines the database file path for a given user ID.
    Ensures the user's specific directory exists.
    Uses USER_DB_BASE_DIR assigned from settings.
    """
    base_dir_env = os.environ.get("USER_DB_BASE_DIR")
    # Test-mode safety: isolate user DBs to a per-process temp dir unless explicitly overridden
    if not base_dir_env and env_flag_enabled("TESTING"):
        run_tag = (
            os.getenv("TLDW_TEST_RUN_ID")
            or os.getenv("PYTEST_XDIST_WORKER")
            or "default"
        )
        safe_run_tag = "".join(
            ch if ch.isalnum() or ch in "-_." else "_"
            for ch in str(run_tag)
        )
        try:
            import tempfile
            # Use project Databases/user_databases_test/<run_tag> to keep nearby but stable across processes
            project_root = settings.get("PROJECT_ROOT")  # type: ignore[attr-defined]
            if project_root:
                base_dir_env = str(
                    Path(project_root) / "Databases" / "user_databases_test" / safe_run_tag
                )
            else:
                base_dir_env = str(
                    Path(tempfile.gettempdir()) / "tldw_user_databases_test" / safe_run_tag
                )
            # Set env so subsequent calls use the same directory
            os.environ.setdefault("USER_DB_BASE_DIR", base_dir_env)
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            logger.warning(
                'TESTING mode: failed to derive project-root user DB dir; falling back to temp dir. Error: {}',
                e,
                exc_info=True,
            )
            import tempfile
            base_dir_env = str(
                Path(tempfile.gettempdir()) / "tldw_user_databases_test" / safe_run_tag
            )
            os.environ.setdefault("USER_DB_BASE_DIR", base_dir_env)
    try:
        return DatabasePaths.get_media_db_path(user_id)
    except Exception as e:
        logger.error(f"Could not resolve database directory for user_id {user_id}: {e}", exc_info=True)
        raise OSError(f"Could not initialize storage directory for user {user_id}.") from e

def _get_or_create_media_db_factory(current_user: User) -> MediaDbFactory:
    if not current_user:
        logger.error("get_media_db_for_user called without a valid User object.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User identification failed.")

    user_id = getattr(current_user, "id_int", None)
    if user_id is None and isinstance(getattr(current_user, "id", None), int):
        user_id = current_user.id
    if user_id is None:
        logger.error("get_media_db_for_user could not resolve a numeric user identifier.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User identification failed.")

    backend_mode_hint = (os.getenv("CONTENT_DB_MODE") or str(settings.get("CONTENT_DB_BACKEND", "sqlite"))).strip().lower()
    require_shared_backend = backend_mode_hint in {"postgres", "postgresql"}

    try:
        shared_backend = get_content_backend_instance()
    except RuntimeError as exc:
        logger.error(f"Content backend initialization failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PostgreSQL content backend required but unavailable. Check server logs."
        ) from exc

    shared_backend_type = getattr(shared_backend, "backend_type", None)
    if require_shared_backend and shared_backend_type != BackendType.POSTGRESQL:
        logger.error("CONTENT_DB_MODE=postgres but shared backend is not PostgreSQL or missing.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PostgreSQL content backend required but unavailable. Check server configuration.",
        )

    use_shared_backend = shared_backend_type == BackendType.POSTGRESQL

    # --- Check Cache ---
    with _user_db_lock:
        factory = _media_db_factories.get(user_id)
    # TEST_MODE: log cache hit/miss visibility for debugging
    try:
        if is_test_mode():
            logger.warning(
                f"TEST_MODE: DB_Deps factory cache {'hit' if factory else 'miss'} for user_id={user_id}"
            )
    except (OSError, RuntimeError, TypeError, ValueError):
        pass

    if factory:
        logger.debug(f"Using cached MediaDbFactory for user_id: {user_id}")
        return factory

    # --- Factory Not Cached: Create New One ---
    with _user_db_lock:
        factory = _media_db_factories.get(user_id)
        if factory:
            logger.debug(f"MediaDbFactory for user {user_id} created concurrently.")
            try:
                if is_test_mode():
                    logger.warning(
                        "TEST_MODE: DB_Deps returning concurrently-created cached factory user_id={}",
                        user_id,
                    )
            except (OSError, RuntimeError, TypeError, ValueError):
                pass
            return factory

        db_path: Optional[Path] = None
        try:
            if use_shared_backend:
                db_path = Path(":memory:")
                init_target = "shared_postgresql"
                factory = MediaDbFactory(
                    db_path=str(db_path),
                    client_id=str(current_user.id),
                    backend=shared_backend,
                )
            else:
                db_path = _get_db_path_for_user(user_id)
                init_target = str(db_path.resolve())
                factory = MediaDbFactory.for_sqlite_path(
                    db_path=str(db_path),
                    client_id=str(current_user.id),
                )

            logger.info(
                "Initializing MediaDbFactory user_id={} backend={} target={}",
                user_id,
                "postgresql" if use_shared_backend else "sqlite",
                init_target,
            )
            _media_db_factories[user_id] = factory
            try:
                if is_test_mode():
                    logger.warning(
                        "TEST_MODE: DB_Deps cached new factory user_id={} db_path={} shared_backend={}",
                        user_id,
                        db_path,
                        use_shared_backend,
                    )
            except (OSError, RuntimeError, TypeError, ValueError):
                pass

        except (DatabaseError, SchemaError) as e:
            log_path = db_path or f"directory for user_id {user_id}"
            logger.error(f"Failed to initialize database for user {user_id} at {log_path}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not initialize database for user: {e}"
            ) from e
        except OSError as e:
            logger.error(f"Failed to get DB path for user {user_id}: {e}", exc_info=True)
            raise HTTPException(
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                 detail=str(e)
             ) from e
        except Exception as e:
            log_path = db_path or f"directory for user_id {user_id}"
            logger.error(f"Unexpected error initializing database for user {user_id} at {log_path}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during database setup for user."
            ) from e

    return factory


def _resolve_media_db_for_user(current_user: User) -> MediaDbSession:
    factory = _get_or_create_media_db_factory(current_user)
    org_id: int | None = None
    team_id: int | None = None
    try:
        scope = get_scope()
        if scope:
            org_id = scope.effective_org_id
            team_id = scope.effective_team_id
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return factory.for_request(org_id=org_id, team_id=team_id)


# --- Main Dependency Function ---

async def get_media_db_for_user(
    request: Request,
    current_user: User = Depends(get_request_user)
) -> AsyncGenerator[MediaDbSession, None]:
    """
    FastAPI dependency that provides a request-scoped Media DB handle.

    Yields a fresh handle for the current request and ensures connection cleanup
    on exit.
    """
    db_instance = _resolve_media_db_for_user(current_user)
    try:
        yield db_instance
    finally:
        try:
            if hasattr(db_instance, "release_context_connection"):
                db_instance.release_context_connection()
        except (DatabaseError, OSError, RuntimeError, TypeError, ValueError):
            pass


def get_media_db_for_owner(owner_user_id: int) -> MediaDbSession:
    """
    Return a request-scoped Media DB session for an arbitrary owner user.

    This is the low-level session constructor used by
    ``managed_media_db_for_owner()``. Non-dependency callers should normally
    prefer the managed helper so ``release_context_connection()`` is always
    invoked for the returned request-scoped session.
    """
    if not isinstance(owner_user_id, int):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid owner_user_id.",
        )
    fake_user = User(id=owner_user_id, username=f"owner-{owner_user_id}", email="")
    return _resolve_media_db_for_user(fake_user)


@contextmanager
def managed_media_db_for_owner(owner_user_id: int) -> Iterator[MediaDbSession]:
    """
    Yield an owner-scoped Media DB session and always release request context.

    This mirrors the FastAPI dependency teardown path for non-request call
    sites such as sharing helpers and background tasks.
    """
    db_instance = get_media_db_for_owner(owner_user_id)
    try:
        yield db_instance
    finally:
        try:
            if hasattr(db_instance, "release_context_connection"):
                db_instance.release_context_connection()
        except (DatabaseError, OSError, RuntimeError, TypeError, ValueError):
            pass


def get_media_db_path_for_rag(media_db: Any) -> str | None:
    """
    Return a filesystem path for RAG callers when one actually exists.

    Shared PostgreSQL-backed Media DB sessions use ``:memory:`` as an internal
    placeholder path; forwarding that sentinel into path-based RAG entry points
    can trigger incorrect SQLite fallback behavior, so return ``None`` instead.
    """
    raw_path = getattr(media_db, "db_path_str", None)
    if raw_path is None:
        raw_path = getattr(media_db, "db_path", None)
    if raw_path is None:
        return None

    normalized = str(raw_path).strip()
    if not normalized:
        return None

    backend_type = getattr(media_db, "backend_type", None)
    if backend_type is None:
        backend = getattr(media_db, "backend", None)
        backend_type = getattr(backend, "backend_type", None)

    if backend_type == BackendType.POSTGRESQL and normalized in {":memory:", "/:memory:"}:
        return None
    return normalized


def reset_media_db_cache() -> None:
    """Clear cached Media DB factories and any legacy cached instances."""
    def _warn(step: str, exc: Exception) -> None:
        logger.warning(
            "Failed {} during media DB cache reset: {}",
            step,
            exc,
            exc_info=True,
        )

    managed_backends = []
    with _user_db_lock:
        try:
            # Attempt to close outstanding connections for cache entries
            values_iter = (
                _user_db_instances.values()
                if hasattr(_user_db_instances, "values")
                else []  # type: ignore[assignment]
            )
            for db in list(values_iter):  # type: ignore[arg-type]
                try:
                    backend = db.backend
                except AttributeError:
                    backend = None
                except RuntimeError as exc:
                    _warn("legacy DB backend", exc)
                    backend = None
                if backend is not None:
                    managed_backends.append(backend)
                try:
                    if hasattr(db, "release_context_connection"):
                        db.release_context_connection()
                    if hasattr(db, "close_connection"):
                        db.close_connection()
                except (DatabaseError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    _warn("legacy DB connection cleanup", exc)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _warn("legacy DB cache iteration", exc)
        try:
            _user_db_instances.clear()  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _warn("legacy DB cache clear", exc)
        try:
            for factory in list(_media_db_factories.values()):  # type: ignore[attr-defined]
                try:
                    backend = factory.backend
                except AttributeError:
                    backend = None
                except RuntimeError as exc:
                    _warn("factory backend", exc)
                    backend = None
                if backend is not None:
                    managed_backends.append(backend)
                try:
                    if hasattr(factory, "close"):
                        factory.close()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    _warn("factory close", exc)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _warn("factory cache iteration", exc)
        try:
            _media_db_factories.clear()  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _warn("factory cache clear", exc)
        reset_managed_sqlite_backends(mode="hard", backends=managed_backends)


async def try_get_media_db_for_user(
    request: Request,
    current_user: User = Depends(get_request_user),
) -> AsyncGenerator[Optional[MediaDbSession], None]:
    """
    Optional version of get_media_db_for_user for endpoints that can operate
    without DB. Returns None instead of raising on initialization failures.
    """
    db_instance: Optional[MediaDbSession] = None
    try:
        db_instance = _resolve_media_db_for_user(current_user)
    except HTTPException as e:
        if e.status_code in {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN}:
            raise
        logger.warning(f"Optional Media DB unavailable for user {getattr(current_user, 'id', '?')}: {e.detail}")
        yield None
        return
    except (DatabaseError, OSError, RuntimeError, SchemaError, TypeError, ValueError) as e:
        logger.warning(
            f"Optional Media DB unexpected error for user {getattr(current_user, 'id', '?')}: {e}",
            exc_info=True
        )
        yield None
        return
    try:
        yield db_instance
    finally:
        try:
            if db_instance and hasattr(db_instance, "release_context_connection"):
                db_instance.release_context_connection()
        except (DatabaseError, OSError, RuntimeError, TypeError, ValueError):
            pass

#
# End of DB_Deps.py
########################################################################################################################
