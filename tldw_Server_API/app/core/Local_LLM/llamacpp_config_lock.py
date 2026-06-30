from __future__ import annotations

from pathlib import Path

from tldw_Server_API.app.core.Infrastructure.distributed_lock import FileLock, LockAcquisitionError
from tldw_Server_API.app.core.Setup import setup_manager


def llamacpp_config_lock_path() -> Path:
    """Return the config-owned lock path shared by llama.cpp config writers."""
    config_path = setup_manager.get_config_file_path().expanduser().resolve()
    return config_path.with_name(".llamacpp.lock")


def llamacpp_config_write_lock(timeout: float = 10) -> FileLock:
    """Return the shared cross-process lock for llama.cpp config writes."""
    lock_path = llamacpp_config_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(lock_path, timeout=timeout)
