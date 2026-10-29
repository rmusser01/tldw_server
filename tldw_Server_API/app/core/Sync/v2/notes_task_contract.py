"""Moved to tldw_Server_API.app.core.Sync_Contracts.notes_task_contract (TASK-13950.16); this path is the same module."""

import sys

from tldw_Server_API.app.core.Sync_Contracts import notes_task_contract as _moved
from tldw_Server_API.app.core.Sync_Contracts.notes_task_contract import *  # noqa: F403 - names for type checkers; runtime is the alias below

sys.modules[__name__] = _moved
