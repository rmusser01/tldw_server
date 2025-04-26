# sync_library/exceptions.py

class SyncError(Exception):
    """Base exception for the sync library."""
    pass

class ConflictError(SyncError):
    """Represents a data conflict during synchronization."""
    pass

class TransportError(SyncError):
    """Represents an error during data transport (fetch/send)."""
    pass

class ApplyError(SyncError):
    """Represents an error applying a remote change locally."""
    def __init__(self, message, change_id=None, entity_uuid=None, *args):
        super().__init__(message, *args)
        self.change_id = change_id
        self.entity_uuid = entity_uuid

    def __str__(self):
        base = super().__str__()
        details = []
        if self.change_id: details.append(f"ChangeID: {self.change_id}")
        if self.entity_uuid: details.append(f"EntityUUID: {self.entity_uuid}")
        return f"{base} ({', '.join(details)})" if details else base

class StateError(SyncError):
    """Represents an error reading/writing sync state."""
    pass