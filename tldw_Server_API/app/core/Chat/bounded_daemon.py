"""Moved to core.LLM_Calls.bounded_daemon (TASK-13950.13); re-exported for existing importers.

Patch module attributes (STREAM_DAEMON_POOL, _drain_owned_task, ...) on the LLM_Calls module:
the functions read their own module's globals, so a patch on this shim does not reach them.
"""

from tldw_Server_API.app.core.LLM_Calls import bounded_daemon as _moved
from tldw_Server_API.app.core.LLM_Calls.bounded_daemon import *  # noqa: F403 - re-exported

__all__ = _moved.__all__
