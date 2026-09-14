"""Shared opaque observability context for planned STT network calls."""

from contextvars import ContextVar

OPAQUE_STT_ENDPOINT_ID: ContextVar[str | None] = ContextVar(
    "opaque_stt_endpoint_id",
    default=None,
)


def get_opaque_stt_endpoint_id() -> str | None:
    """Return the planned STT endpoint identity in the current context."""
    return OPAQUE_STT_ENDPOINT_ID.get()
