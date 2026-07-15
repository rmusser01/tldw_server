"""Architecture contracts for playlist ingest persistence and domain errors."""

import pytest

pytestmark = pytest.mark.unit


def test_playlist_domain_exceptions_are_centralized() -> None:
    """Playlist service and store errors originate from the shared exception module."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_service, playlist_ingest_store

    exception_names = (
        "InvalidPlaylistUrlError",
        "PlaylistIngestConflictError",
        "PlaylistIngestNotFoundError",
        "PlaylistPreflightBusyError",
        "PlaylistPreflightCapacityError",
        "PlaylistPreflightIncompleteError",
        "PlaylistPreflightLeaseLostError",
        "PlaylistPreflightRequiredError",
        "PlaylistPreflightUnavailableError",
        "PlaylistRunPendingError",
        "PlaylistRunStatusUnavailableError",
        "PlaylistRunValidationError",
        "PlaylistSelectionError",
        "ReviewRequiredError",
    )

    modules = (playlist_ingest_service, playlist_ingest_store)
    for name in exception_names:
        exception_type = next(getattr(module, name) for module in modules if hasattr(module, name))
        assert exception_type.__module__ == "tldw_Server_API.app.core.exceptions"


def test_playlist_store_implementation_lives_in_db_management_with_legacy_alias() -> None:
    """Persistence stays in the DB layer without breaking the historical import path."""
    from tldw_Server_API.app.core.DB_Management.playlist_ingest_store import (
        PlaylistIngestStore as DatabasePlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore as LegacyPlaylistIngestStore,
    )

    assert DatabasePlaylistIngestStore.__module__ == (
        "tldw_Server_API.app.core.DB_Management.playlist_ingest_store"
    )
    assert LegacyPlaylistIngestStore is DatabasePlaylistIngestStore
