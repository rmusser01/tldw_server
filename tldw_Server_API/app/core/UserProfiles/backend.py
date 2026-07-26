"""Strict AuthNZ backend resolution shared by UserProfiles gateways."""

from __future__ import annotations

from typing import Any, Literal

ProfileBackend = Literal["postgres", "sqlite"]


class ProfileBackendUnavailable(RuntimeError):
    """The AuthNZ pool did not expose a supported backend contract."""

    def __init__(self) -> None:
        super().__init__("Profile storage backend is unavailable")


def resolve_profile_backend(db_pool: Any) -> ProfileBackend:
    """Resolve the public DatabasePool backend contract without guessing."""
    try:
        backend_type = db_pool.backend_type
    except Exception:  # noqa: BLE001 - pool adapters are an external boundary
        raise ProfileBackendUnavailable() from None
    if type(backend_type) is not str:
        raise ProfileBackendUnavailable()
    if backend_type == "postgres":
        return "postgres"
    if backend_type == "sqlite":
        return "sqlite"
    raise ProfileBackendUnavailable()


__all__ = [
    "ProfileBackend",
    "ProfileBackendUnavailable",
    "resolve_profile_backend",
]
