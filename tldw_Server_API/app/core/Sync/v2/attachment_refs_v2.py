"""Moved to tldw_Server_API.app.core.Sync_Contracts.attachment_refs_v2 (TASK-13950.16); this path is the same module."""

import sys

from tldw_Server_API.app.core.Sync_Contracts import attachment_refs_v2 as _moved
from tldw_Server_API.app.core.Sync_Contracts.attachment_refs_v2 import *  # noqa: F403 - names for type checkers; runtime is the alias below

sys.modules[__name__] = _moved
