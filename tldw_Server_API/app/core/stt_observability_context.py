"""Moved to ``core/Logging/stt_observability_context.py`` (TASK-13950.12).

Re-exported so the old path keeps the same ``ContextVar``.
"""

from tldw_Server_API.app.core.Logging.stt_observability_context import (  # noqa: F401
    OPAQUE_STT_ENDPOINT_ID,
    get_opaque_stt_endpoint_id,
)
