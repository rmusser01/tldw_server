from __future__ import annotations

"""Sync v2 core substrate."""

from typing import TYPE_CHECKING

from .errors import (
    SyncConflictNotFoundError,
    SyncDatasetNotFoundError,
    SyncStoreError,
    SyncV2Error,
)
from .models import (
    SyncConflict,
    SyncConflictCreate,
    SyncDataset,
    SyncDatasetCreate,
    SyncDevice,
    SyncDeviceCursor,
    SyncDeviceUpsert,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncKeyRecord,
    SyncKeyRecordCreate,
)

if TYPE_CHECKING:
    from .store import SyncV2Store


def __getattr__(name: str) -> object:
    if name == "SyncV2Store":
        from .store import SyncV2Store

        return SyncV2Store
    raise AttributeError(name)

__all__ = [
    "SyncConflict",
    "SyncConflictCreate",
    "SyncConflictNotFoundError",
    "SyncDataset",
    "SyncDatasetCreate",
    "SyncDatasetNotFoundError",
    "SyncDevice",
    "SyncDeviceCursor",
    "SyncDeviceUpsert",
    "SyncEnvelope",
    "SyncEnvelopeCreate",
    "SyncKeyRecord",
    "SyncKeyRecordCreate",
    "SyncStoreError",
    "SyncV2Error",
    "SyncV2Store",
]
