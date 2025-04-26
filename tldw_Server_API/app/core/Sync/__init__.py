# sync_library/__init__.py
from .core import SyncManager
from .models import SyncLogEntry
from .exceptions import SyncError, ConflictError, TransportError, ApplyError, StateError
from .transport import SyncTransport, HttpApiTransport
from .conflict import ConflictResolver, LastWriteWinsStrategy
from .state import SyncStateManager

__all__ = [
    "SyncManager",
    "SyncLogEntry",
    "SyncError",
    "ConflictError",
    "TransportError",
    "ApplyError",
    "StateError",
    "SyncTransport",
    "HttpApiTransport",
    "ConflictResolver",
    "LastWriteWinsStrategy",
    "SyncStateManager",
]

# Optional: Perform default logging setup on import
# setup_logging()
