"""Moved to core.LLM_Calls.bounded_daemon (TASK-13950.13); this path is an alias of that module."""

import sys

from tldw_Server_API.app.core.LLM_Calls import bounded_daemon as _moved
from tldw_Server_API.app.core.LLM_Calls.bounded_daemon import *  # noqa: F403 - names for type checkers

__all__ = _moved.__all__

sys.modules[__name__] = _moved
