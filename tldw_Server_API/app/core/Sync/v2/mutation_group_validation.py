"""Moved to tldw_Server_API.app.core.Sync_Contracts.mutation_group_validation (TASK-13950.16); this path is the same module."""

import sys

from tldw_Server_API.app.core.Sync_Contracts import mutation_group_validation as _moved
from tldw_Server_API.app.core.Sync_Contracts.mutation_group_validation import *  # noqa: F403 - names for type checkers; runtime is the alias below

sys.modules[__name__] = _moved
