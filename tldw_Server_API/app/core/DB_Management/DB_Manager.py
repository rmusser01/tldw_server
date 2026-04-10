# DB_Manager.py
# Description: Helpers and factories for content databases (Media/ChaCha/etc.)
# backed by SQLite or PostgreSQL via a shared backend abstraction. Note:
# Elasticsearch/OpenSearch are not wired here; use SQL content backends only.
#
# Imports
import configparser
import os
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger

#
# 3rd-Party Libraries
#from elasticsearch import Elasticsearch
#
# Local Imports
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseBackend

# ChaChaNotes database
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.ChatWorkflows_DB import ChatWorkflowsDatabase

#from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
#list_prompts as sqlite_list_prompts,
#fetch_prompt_details as sqlite_fetch_prompt_details,
#add_prompt as sqlite_add_prompt,
#search_prompts as sqlite_search_prompts,
#add_or_update_prompt as sqlite_add_or_update_prompt,
#load_prompt_details as sqlite_load_prompt_details,
# insert_prompt_to_db as sqlite_insert_prompt_to_db,
#delete_prompt as sqlite_delete_prompt
#)
from tldw_Server_API.app.core.DB_Management.content_backend import ContentDatabaseSettings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_repository
from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.media_db.runtime import defaults as media_db_runtime_defaults
from tldw_Server_API.app.core.DB_Management.media_db.runtime.defaults import (
    build_media_runtime_config,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.factory import (
    create_media_database as runtime_create_media_database,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.factory import (
    validate_postgres_content_backend as runtime_validate_postgres_content_backend,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_reads import (
    get_latest_transcription as sqlite_get_latest_transcription,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_reads import (
    get_media_prompts as sqlite_get_media_prompts,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_reads import (
    get_media_transcripts as sqlite_get_media_transcripts,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_reads import (
    get_specific_prompt as sqlite_get_specific_prompt,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_reads import (
    get_specific_transcript as sqlite_get_specific_transcript,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers import (
    get_document_version as sqlite_get_document_version,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers import (
    import_obsidian_note_to_db as sqlite_import_obsidian_note_to_db,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers import (
    ingest_article_to_db_new as sqlite_ingest_article_to_db,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_state import (
    check_media_exists as sqlite_check_media_exists,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_state import (
    get_unprocessed_media as sqlite_get_unprocessed_media,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_state import (
    mark_media_as_processed as sqlite_mark_media_as_processed,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_content_queries import (
    fetch_keywords_for_media as sqlite_fetch_keywords_for_media,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_content_queries import (
    get_all_content_from_database as sqlite_get_all_content_from_database,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_maintenance import (
    check_media_and_whisper_model as sqlite_check_media_and_whisper_model,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_maintenance import (
    empty_trash as sqlite_empty_trash,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_backup import (
    create_automated_backup as sqlite_create_automated_backup,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase

#
# End of imports
############################################################################################################

############################################################################################################
#
# Database Config loading (standardized on load_comprehensive_config)
single_user_config: configparser.ConfigParser
content_db_settings: ContentDatabaseSettings
_CONTENT_DB_BACKEND: Optional[DatabaseBackend]
_POSTGRES_CONTENT_MODE: bool
single_user_db_path: str
single_user_backup_path: str
single_user_backup_dir: Union[str, bytes]
single_user_chacha_path: str
single_user_workflows_path: str
single_user_chat_workflows_path: str
db_type: str
"""Backup directory resolution (standardized)

Precedence:
1) TLDW_DB_BACKUP_PATH (env)
2) content_backend backup_path
3) default ./tldw_DB_Backups/
"""
_DEFAULT_BACKUP_DIR = './tldw_DB_Backups/'

DB_MANAGER_RUNTIME_EXCEPTIONS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    configparser.Error,
)


def _sync_media_runtime_defaults() -> None:
    """Mirror shared Media DB runtime defaults onto DB_Manager compatibility globals."""
    global single_user_config, content_db_settings, _CONTENT_DB_BACKEND
    global _POSTGRES_CONTENT_MODE, single_user_db_path

    single_user_config = media_db_runtime_defaults.single_user_config
    content_db_settings = media_db_runtime_defaults.content_db_settings
    _CONTENT_DB_BACKEND = media_db_runtime_defaults.content_db_backend
    _POSTGRES_CONTENT_MODE = media_db_runtime_defaults.postgres_content_mode
    single_user_db_path = str(media_db_runtime_defaults.single_user_db_path)


def _sync_db_manager_path_defaults(
    config: Optional[configparser.ConfigParser] = None,
) -> None:
    """Refresh DB_Manager-owned path globals from the shared Media DB defaults."""
    global single_user_backup_path, single_user_backup_dir
    global single_user_chacha_path, single_user_workflows_path
    global single_user_chat_workflows_path

    cfg = config or single_user_config
    single_user_backup_path = content_db_settings.backup_path or _DEFAULT_BACKUP_DIR
    single_user_backup_dir = (
        os.environ.get('TLDW_DB_BACKUP_PATH')
        or single_user_backup_path
        or _DEFAULT_BACKUP_DIR
    )
    single_user_chacha_path = cfg.get(
        'Database',
        'chacha_path',
        fallback=str(DatabasePaths.get_chacha_db_path(DatabasePaths.get_single_user_id())),
    )
    single_user_workflows_path = cfg.get(
        'Database',
        'workflows_path',
        fallback=str(DatabasePaths.get_workflows_db_path(DatabasePaths.get_single_user_id())),
    )
    single_user_chat_workflows_path = cfg.get(
        'Database',
        'chat_workflows_path',
        fallback=str(DatabasePaths.get_chat_workflows_db_path(DatabasePaths.get_single_user_id())),
    )


def _sync_db_type() -> None:
    """Refresh the DB_Manager compatibility backend discriminator."""
    global db_type

    raw_backend = content_db_settings.raw_backend_type
    if content_db_settings.backend_type == BackendType.POSTGRESQL:
        db_type = 'postgres'
    elif content_db_settings.backend_type == BackendType.SQLITE:
        db_type = 'sqlite'
    elif raw_backend in {'elasticsearch', 'opensearch'}:
        db_type = 'elasticsearch'
        logger.warning(
            "Elasticsearch content backend is not supported in DB_Manager; operations will raise NotImplementedError"
        )
    else:
        logger.warning(f"Unknown Database.type '{raw_backend}', defaulting to sqlite content backend")
        db_type = 'sqlite'


_sync_media_runtime_defaults()
_sync_db_manager_path_defaults()
_sync_db_type()


def _ensure_content_backend_loaded() -> Optional[DatabaseBackend]:
    """Lazily resolve the shared content backend when PostgreSQL mode is active."""
    backend = media_db_runtime_defaults.ensure_content_backend_loaded()
    _sync_media_runtime_defaults()
    return backend


def get_content_backend_instance() -> Optional[DatabaseBackend]:
    """Return the shared content DatabaseBackend instance (if configured)."""
    backend = media_db_runtime_defaults.get_content_backend_instance()
    _sync_media_runtime_defaults()
    return backend


def shutdown_content_backend() -> None:
    """Gracefully close pooled connections for the shared content backend.

    Call this from application shutdown hooks to ensure database connections
    are returned/closed for long-lived processes (e.g., FastAPI on_shutdown).
    Safe to call when no backend is configured or when using SQLite per-user
    files without a shared backend.
    """
    try:
        backend = get_content_backend_instance()
    except RuntimeError:
        backend = None
    if backend is None:
        return
    try:
        pool = backend.get_pool()
        pool.close_all()
        logger.info("Content backend connection pool closed")
    except DB_MANAGER_RUNTIME_EXCEPTIONS as exc:  # pragma: no cover - defensive shutdown
        logger.warning(f"Failed to close content backend pool: {exc}")


def _is_media_database_instance(candidate: Any) -> bool:
    return candidate is not None and callable(getattr(candidate, "execute_query", None)) and (
        hasattr(candidate, "backend") or hasattr(candidate, "backend_type")
    )


def _unwrap_media_database(candidate: Any) -> Any:
    if _is_media_database_instance(candidate):
        return candidate

    wrapped = getattr(candidate, "database", None)
    if _is_media_database_instance(wrapped):
        return wrapped

    return candidate


def reset_content_backend(
    *,
    config: Optional[configparser.ConfigParser] = None,
    reload: bool = True,
) -> Optional[DatabaseBackend]:
    """Reset the shared content backend and recompute settings.

    - Clears the cached backend instance in content_backend to allow
      reconfiguration at runtime (useful in tests or dynamic reloads).
    - Recomputes module-level settings and paths derived from config.
    - Optionally rebuilds the backend immediately when reload=True.
    """
    global _CONTENT_DB_BACKEND
    global single_user_backup_path, single_user_backup_dir
    global single_user_chacha_path, single_user_workflows_path
    global single_user_chat_workflows_path

    cfg = config or single_user_config
    try:
        _CONTENT_DB_BACKEND = media_db_runtime_defaults.reset_media_runtime_defaults(
            config=cfg,
            reload=reload,
            reset_mode="graceful",
        )
        _sync_media_runtime_defaults()
        _sync_db_manager_path_defaults(cfg)
        _sync_db_type()
    except DB_MANAGER_RUNTIME_EXCEPTIONS as e:
        logger.debug(f"reset_content_backend: failed to recompute content settings: {e}")
    return _CONTENT_DB_BACKEND


def create_media_database(
    client_id: str,
    *,
    db_path: Union[str, Path, None] = None,
    backend: Optional[DatabaseBackend] = None,
    config: Optional[configparser.ConfigParser] = None,
) -> Any:
    """Factory for MediaDatabase instances using the shared backend wiring."""
    return runtime_create_media_database(
        client_id,
        db_path=db_path,
        backend=backend,
        config=config,
        runtime=build_media_runtime_config(),
    )


def validate_postgres_content_backend() -> None:
    """Ensure the configured PostgreSQL content backend is ready for use.

    Raises RuntimeError when the shared backend is expected but migrations or
    row-level security policies are missing. Acts as a no-op for SQLite.
    """

    runtime_validate_postgres_content_backend(
        get_content_backend_instance=get_content_backend_instance,
        runtime=build_media_runtime_config(),
    )


def create_chacha_database(
    client_id: str,
    *,
    db_path: Union[str, Path, None] = None,
    backend: Optional[DatabaseBackend] = None,
    config: Optional[configparser.ConfigParser] = None,
) -> CharactersRAGDB:
    """Factory for ChaChaNotes database instances with backend support."""

    target_path = Path(db_path) if db_path else Path(single_user_chacha_path)
    backend_to_use = backend or _ensure_content_backend_loaded()
    cfg = config or single_user_config

    return CharactersRAGDB(
        db_path=str(target_path),
        client_id=client_id,
        backend=backend_to_use,
        config=cfg,
    )


def create_prompt_studio_database(
    client_id: str,
    *,
    db_path: Union[str, Path],
    backend: Optional[DatabaseBackend] = None,
    config: Optional[configparser.ConfigParser] = None,
) -> PromptStudioDatabase:
    """Factory for Prompt Studio databases with backend-aware wiring."""

    target_path = Path(db_path)
    backend_to_use = backend or _ensure_content_backend_loaded()
    cfg = config or single_user_config

    return PromptStudioDatabase(
        db_path=str(target_path),
        client_id=client_id,
        backend=backend_to_use,
        config=cfg,
    )


def create_workflows_database(
    *,
    db_path: Union[str, Path, None] = None,
    backend: Optional[DatabaseBackend] = None,
) -> WorkflowsDatabase:
    """Factory for Workflow database instances with backend-aware wiring."""

    target_path = Path(db_path) if db_path else Path(single_user_workflows_path)
    backend_to_use = backend or _ensure_content_backend_loaded()

    # Only pass backend through when it is a PostgreSQL adapter; SQLite continues to
    # rely on the local file connection for compatibility with existing tests.
    if backend_to_use and backend_to_use.backend_type != BackendType.POSTGRESQL:
        backend_to_use = None

    return WorkflowsDatabase(
        db_path=str(target_path),
        backend=backend_to_use,
    )


def create_chat_workflows_database(
    client_id: str,
    *,
    db_path: Union[str, Path, None] = None,
    backend: Optional[DatabaseBackend] = None,
) -> ChatWorkflowsDatabase:
    """Factory for Chat Workflows database instances."""

    target_path = Path(db_path) if db_path else Path(single_user_chat_workflows_path)
    backend_to_use = backend or _ensure_content_backend_loaded()

    if backend_to_use and backend_to_use.backend_type != BackendType.POSTGRESQL:
        backend_to_use = None

    return ChatWorkflowsDatabase(
        db_path=str(target_path),
        client_id=client_id,
        backend=backend_to_use,
    )


def create_evaluations_database(
    *,
    db_path: Union[str, Path],
    backend: Optional[DatabaseBackend] = None,
) -> EvaluationsDatabase:
    """Factory for EvaluationsDatabase with backend-aware wiring.

    When a PostgreSQL content backend is configured, this returns an
    EvaluationsDatabase instance bound to that backend; otherwise it
    falls back to the SQLite file at `db_path`.
    """
    backend_to_use = backend or _ensure_content_backend_loaded()
    return EvaluationsDatabase(str(Path(db_path)), backend=backend_to_use)


def get_db_config():
    try:
        config = load_comprehensive_config()

        if 'Database' not in config:
            logger.warning("'Database' section not found in config. Using default values.")
            return default_db_config()

        return {
            'type': config.get('Database', 'type', fallback='sqlite'),
            'sqlite_path': config.get(
                'Database',
                'sqlite_path',
                fallback=str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id())),
            ),
            'elasticsearch_host': config.get('Database', 'elasticsearch_host', fallback='localhost'),
            'elasticsearch_port': config.getint('Database', 'elasticsearch_port', fallback=9200)
        }
    except FileNotFoundError:
        logger.warning("Config file not found. Using default database configuration.")
        return default_db_config()
    except DB_MANAGER_RUNTIME_EXCEPTIONS as e:
        logger.error(f"Error reading config: {str(e)}. Using default database configuration.")
        return default_db_config()

def default_db_config():
    return {
        'type': 'sqlite',
        # Per-user default: single-user fixed ID
        'sqlite_path': str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id())),
        'elasticsearch_host': 'localhost',
        'elasticsearch_port': 9200
    }

# Content backends supported by this module. Elasticsearch/opensearch are
# intentionally not wired here to avoid confusing fallthrough paths.
SQL_CONTENT_BACKENDS = {'sqlite', 'postgres'}


#
# End of Database Config loading
############################################################################################################

# Centralized message for unsupported Elasticsearch content backend
def _raise_elasticsearch_not_supported(op: str) -> None:
    message = (
        f"Elasticsearch content backend is not enabled. Operation: {op}. "
        "Set TLDW_CONTENT_DB_BACKEND=sqlite or postgresql to use content databases."
    )
    raise NotImplementedError(message)


def _require_db_instance(args, kwargs, func_name: str) -> Any:
    """Extract a MediaDatabase instance from kwargs or first positional arg.

    Accepts either a raw MediaDatabase or a request-scoped wrapper exposing
    the underlying instance at ``.database``. Emits a deprecation warning if
    provided positionally. Returns the unwrapped MediaDatabase or raises
    ValueError if invalid/missing.
    """

    provided_positionally = False
    if 'db_instance' in kwargs and kwargs['db_instance'] is not None:
        dbi = kwargs.pop('db_instance')
    elif args and _is_media_database_instance(_unwrap_media_database(args[0])):
        dbi = args[0]
        provided_positionally = True
    else:
        dbi = None

    if provided_positionally:
        logger.warning(
            "DEPRECATION: db_instance should be passed as keyword 'db_instance' to {}. "
            "Passing positionally is deprecated and will be removed in a future release.",
            func_name,
        )

    dbi = _unwrap_media_database(dbi)
    if not _is_media_database_instance(dbi):
        raise ValueError(f"{func_name} requires 'db_instance' (MediaDatabase)")
    return dbi
#
# DB Search functions

def get_all_content_from_database(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        # The extracted legacy helper still expects an explicit db_instance.
        return sqlite_get_all_content_from_database(*args, **kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("get_all_content_from_database")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def check_media_exists(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        return sqlite_check_media_exists(*args, **kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("check_media_exists")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


# DEPRECATED MEDIA COMPATIBILITY SURFACE
# Keep these forwards until all callers have moved to media_db.api.
def get_full_media_details(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance = kwargs.get('db_instance') or (args[0] if args else None)
        media_id = kwargs.get('media_id') or (args[1] if len(args) > 1 else None)
        include_content = kwargs.get('include_content', True)
        return media_db_api.get_full_media_details(
            db=db_instance,
            media_id=media_id,
            include_content=include_content,
        )
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("get_full_media_details")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


# Deprecated: use get_full_media_details
def get_full_media_details2(*args, **kwargs):
    return get_full_media_details(*args, **kwargs)


def get_full_media_details_rich(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance = kwargs.get('db_instance') or (args[0] if args else None)
        media_id = kwargs.get('media_id') or (args[1] if len(args) > 1 else None)
        include_content = kwargs.get('include_content', True)
        include_versions = kwargs.get('include_versions', True)
        include_version_content = kwargs.get('include_version_content', False)
        return media_db_api.get_full_media_details_rich(
            db=db_instance,
            media_id=media_id,
            include_content=include_content,
            include_versions=include_versions,
            include_version_content=include_version_content,
        )
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("get_full_media_details_rich")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


# Deprecated: use get_full_media_details_rich
def get_full_media_details_rich2(*args, **kwargs):
    return get_full_media_details_rich(*args, **kwargs)

def get_paginated_files(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'get_paginated_files')
        page = kwargs.get('page', 1)
        results_per_page = kwargs.get('results_per_page', 50)
        if hasattr(db_instance, "get_paginated_files"):
            return db_instance.get_paginated_files(page=page, results_per_page=results_per_page)
        if hasattr(db_instance, "get_paginated_media_list"):
            return db_instance.get_paginated_media_list(page=page, results_per_page=results_per_page)
        raise AttributeError("MediaDatabase instance does not expose a paginated files API")
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("get_paginated_files")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_paginated_trash_files(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'get_paginated_trash_files')
        page = kwargs.get('page', 1)
        results_per_page = kwargs.get('results_per_page', 50)
        if hasattr(db_instance, "get_paginated_trash_list"):
            return db_instance.get_paginated_trash_list(page=page, results_per_page=results_per_page)
        raise AttributeError("MediaDatabase instance does not expose a paginated trash list API")
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("get_paginated_trash_files")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

#
# End of DB-Searching functions
############################################################################################################


############################################################################################################
#
# DB-Ingestion functions

def import_obsidian_note_to_db(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        return sqlite_import_obsidian_note_to_db(*args, **kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("import_obsidian_note_to_db")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def add_media_with_keywords(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'add_media_with_keywords')
        media_writer = get_media_repository(db_instance)
        return media_writer.add_media_with_keywords(**kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("add_media_with_keywords")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def check_media_and_whisper_model(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        return sqlite_check_media_and_whisper_model(*args, **kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("check_media_and_whisper_model")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def ingest_article_to_db(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        return sqlite_ingest_article_to_db(*args, **kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("ingest_article_to_db")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def add_media_chunk(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'add_media_chunk')
        return db_instance.add_media_chunk(**kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("add_media_chunk")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def batch_insert_chunks(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'batch_insert_chunks')
        return db_instance.batch_insert_chunks(**kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("batch_insert_chunks")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_unprocessed_media(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'get_unprocessed_media')
        return sqlite_get_unprocessed_media(db_instance)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("get_unprocessed_media")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def mark_media_as_processed(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'mark_media_as_processed')
        media_id = kwargs.pop('media_id', None)
        if media_id is None:
            raise ValueError("mark_media_as_processed requires 'media_id'")
        return sqlite_mark_media_as_processed(db_instance, media_id)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("mark_media_as_processed")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def update_keywords_for_media(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'update_keywords_for_media')
        media_id = kwargs.pop('media_id', None)
        keywords = kwargs.pop('keywords', None)
        if media_id is None or keywords is None:
            raise ValueError("update_keywords_for_media requires 'media_id' and 'keywords'")
        return db_instance.update_keywords_for_media(media_id, keywords)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("update_keywords_for_media")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def rollback_to_version(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'rollback_to_version')
        media_id = kwargs.pop('media_id', None)
        target_version_number = kwargs.pop('target_version_number', None)
        if media_id is None or target_version_number is None:
            raise ValueError("rollback_to_version requires 'media_id' and 'target_version_number'")
        return db_instance.rollback_to_version(media_id, target_version_number)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("rollback_to_version")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def delete_document_version(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'delete_document_version')
        version_uuid = kwargs.pop('version_uuid', None)
        if version_uuid is None:
            raise ValueError("delete_document_version requires 'version_uuid'")
        return db_instance.soft_delete_document_version(version_uuid)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("delete_document_version")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

#
# End of DB-Ingestion functions
############################################################################################################


############################################################################################################
#
# Prompt-related functions #FIXME rename /resort

# def list_prompts(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_list_prompts(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def search_prompts(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_search_prompts(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def fetch_prompt_details(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_fetch_prompt_details(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def add_prompt(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_add_prompt(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
#
# def add_or_update_prompt(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_add_or_update_prompt(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#     return None
#
#
# def load_prompt_details(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_load_prompt_details(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#     return None
#
#
# def insert_prompt_to_db(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_insert_prompt_to_db(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#     return None
#
#
# def delete_prompt(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_delete_prompt(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

def mark_as_trash(*args, **kwargs) -> bool:
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'mark_as_trash')
        return db_instance.mark_as_trash(**kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("mark_as_trash")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_latest_transcription(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        return sqlite_get_latest_transcription(*args, **kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("get_latest_transcription")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def fetch_paginated_data(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'fetch_paginated_data')
        # The underlying content DB exposes paginated file listing directly.
        page = kwargs.get('page', 1)
        results_per_page = kwargs.get('results_per_page', 50)
        return db_instance.get_paginated_files(page=page, results_per_page=results_per_page)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("fetch_paginated_data")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_media_transcripts(*args, **kwargs) -> list[dict]:
    if db_type in SQL_CONTENT_BACKENDS:
        return sqlite_get_media_transcripts(*args, **kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("get_media_transcripts")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_specific_transcript(*args, **kwargs) -> dict:
    if db_type in SQL_CONTENT_BACKENDS:
        return sqlite_get_specific_transcript(*args, **kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("get_specific_transcript")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def get_all_document_versions(db_instance: Any, media_id: int, **kwargs):
    """
    Wrapper to get all document versions for a given media_id from a MediaDatabase instance.
    """
    # db_type check might be relevant if you support multiple DB backends via DB_Manager.
    # For now, this wrapper expects a Media DB-like instance.
    db_instance = _unwrap_media_database(db_instance)
    if _is_media_database_instance(db_instance):
        # Call the INSTANCE method, passing only the relevant kwargs
        # The instance method itself is get_all_document_versions(self, media_id, include_content=True, include_deleted=False, limit=None, offset=None)
        # So we need to ensure only those valid arguments are passed from kwargs.

        # Extract known arguments for the instance method
        limit = kwargs.get('limit')
        offset = kwargs.get('offset')
        include_content = kwargs.get('include_content', True)  # Default if not in test call
        include_deleted = kwargs.get('include_deleted', False)  # Default if not in test call

        return db_instance.get_all_document_versions(
            media_id=media_id,
            include_content=include_content,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset
        )

    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_all_document_versions not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
#
#
############################################################################################################
#
# Prompt Functions:

def get_media_prompts(*args, **kwargs) -> list[dict]:
    if db_type in SQL_CONTENT_BACKENDS:
        return sqlite_get_media_prompts(*args, **kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("get_media_prompts")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_specific_prompt(*args, **kwargs) -> dict:
    if db_type in SQL_CONTENT_BACKENDS:
        return sqlite_get_specific_prompt(*args, **kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("get_specific_prompt")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


#
# End of Prompt-related functions
############################################################################################################

############################################################################################################
#
# Keywords-related Functions

def add_keyword(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = _require_db_instance(args, kwargs, 'add_keyword')
        keyword = kwargs.get('keyword') or kwargs.get('keyword_text')
        if keyword is None:
            raise ValueError("add_keyword requires 'keyword'")
        return db_instance.add_keyword(keyword)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("add_keyword")

def fetch_keywords_for_media(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        return sqlite_fetch_keywords_for_media(*args, **kwargs)
    elif db_type == 'elasticsearch':
        _raise_elasticsearch_not_supported("fetch_keywords_for_media")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

#
# End of Keywords-related Functions
############################################################################################################

############################################################################################################
#
# Chat-related Functions
# FIXME
# def search_notes_titles(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_search_notes_titles(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def save_message(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_save_message(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def load_chat_history(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_load_chat_history(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def start_new_conversation(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_start_new_conversation(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def get_all_conversations(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_all_conversations(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def get_notes_by_keywords(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_notes_by_keywords(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def get_note_by_id(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_note_by_id(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def add_keywords_to_conversation(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_add_keywords_to_conversation(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def get_keywords_for_note(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_keywords_for_note(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def delete_note(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_delete_note(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def search_conversations_by_keywords(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_search_conversations_by_keywords(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def delete_conversation(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_delete_conversation(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def get_conversation_title(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_conversation_title(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def update_conversation_title(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_update_conversation_title(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def fetch_all_conversations(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_fetch_all_conversations()
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def fetch_all_notes(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_fetch_all_notes()
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def delete_messages_in_conversation(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_delete_messages_in_conversation(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of delete_messages_in_conversation not yet implemented")
#
# def get_conversation_text(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_conversation_text(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of get_conversation_text not yet implemented")

#
# End of Chat-related Functions
############################################################################################################


############################################################################################################
#
# Character Chat-related Functions
# FIXME
# def add_character_card(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_add_character_card(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_character_card not yet implemented")
#
# def get_character_cards():
#     if db_type == 'sqlite':
#         return sqlite_get_character_cards()
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of get_character_cards not yet implemented")
#
# def get_character_card_by_id(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_character_card_by_id(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of get_character_card_by_id not yet implemented")
#
# def update_character_card(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_update_character_card(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of update_character_card not yet implemented")
#
# def delete_character_card(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_delete_character_card(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of delete_character_card not yet implemented")
#
# def add_character_chat(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_add_character_chat(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_character_chat not yet implemented")
#
# def get_character_chats(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_character_chats(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of get_character_chats not yet implemented")
#
# def get_character_chat_by_id(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_character_chat_by_id(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of get_character_chat_by_id not yet implemented")
#
# def update_character_chat(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_update_character_chat(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of update_character_chat not yet implemented")
#
# def delete_character_chat(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_delete_character_chat(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of delete_character_chat not yet implemented")
#
# def update_note(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_update_note(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of update_note not yet implemented")
#
# def save_notes(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_save_notes(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of save_notes not yet implemented")
#
# def clear_keywords(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_clear_keywords_from_note(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of clear_keywords not yet implemented")
#
# def clear_keywords_from_note(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_clear_keywords_from_note(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of clear_keywords_from_note not yet implemented")
#
# def add_keywords_to_note(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_add_keywords_to_note(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_keywords_to_note not yet implemented")
#
# def fetch_conversations_by_ids(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_fetch_conversations_by_ids(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of fetch_conversations_by_ids not yet implemented")
#
# def fetch_notes_by_ids(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_fetch_notes_by_ids(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of fetch_notes_by_ids not yet implemented")

#
# End of Character Chat-related Functions
############################################################################################################


############################################################################################################
#
def empty_trash(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        # Provide a sensible default threshold when not specified by caller
        if 'days_threshold' not in kwargs:
            kwargs['days_threshold'] = 0
        return sqlite_empty_trash(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def fetch_item_details(*args, **kwargs) -> tuple[str, str, str]:
    """
    Fetch basic details of a media item including content, prompt, and summary.

    Expects kwargs: db_instance (MediaDatabase), media_id (int)
    Returns empty strings when not found or inactive.
    """
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = kwargs.get('db_instance') or (args[0] if args else None)
        media_id: Optional[int] = kwargs.get('media_id')
        db_instance = _unwrap_media_database(db_instance)
        if not _is_media_database_instance(db_instance) or media_id is None:
            raise ValueError("fetch_item_details requires 'db_instance' and 'media_id'")
        details = media_db_api.get_full_media_details_rich(
            db=db_instance,
            media_id=media_id,
            include_content=True,
            include_versions=True,
            include_version_content=False,
        )
        if not details:
            return "", "", ""
        media = details.get('media') or {}
        latest = details.get('latest_version') or {}
        content = media.get('content') or ""
        prompt = latest.get('prompt') or ""
        summary = latest.get('analysis_content') or ""
        return content, prompt, summary
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of fetch_item_details not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

#
# End of Trash-related Functions
############################################################################################################


############################################################################################################
#
# DB-Backup Functions

def create_automated_backup(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_create_automated_backup(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

#
# End of DB-Backup Functions
############################################################################################################


############################################################################################################
#
# Document Versioning Functions

def create_document_version(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        db_instance: Any = kwargs.pop('db_instance', None) or (args[0] if args else None)
        db_instance = _unwrap_media_database(db_instance)
        if not _is_media_database_instance(db_instance):
            raise ValueError("create_document_version requires 'db_instance' (MediaDatabase)")
        return db_instance.create_document_version(**kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of create_document_version not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_document_version(*args, **kwargs):
    if db_type in SQL_CONTENT_BACKENDS:
        return sqlite_get_document_version(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_document_version not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

#
# End of Document Versioning Functions
############################################################################################################


############################################################################################################
#
# Workflow Functions
#
# def get_workflow_chat(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_workflow_chat(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of get_workflow_chat not yet implemented")
#
#
# def save_workflow_chat_to_db(*args, **kwargs):
#     if db_type == 'sqlite':
#         # FIXME
#         return sqlite_save_workflow_chat_to_db(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of save_workflow_chat_to_db not yet implemented")
#
# #
# End of Workflow Functions
############################################################################################################

# Dead code FIXME
# def close_connection():
#     if db_type == 'sqlite':
#         db.get_connection().close()

#
# End of file
############################################################################################################
