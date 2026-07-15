"""Compatibility alias for the playlist ingest DB-management implementation."""

from __future__ import annotations

import sys

from tldw_Server_API.app.core.DB_Management import playlist_ingest_store as _implementation

sys.modules[__name__] = _implementation
