"""Moved to tldw_Server_API.app.core.Usage.llm_budget_guard (TASK-13950.17).

An alias, not a re-export: the old name *is* the moved module, so old imports and
monkeypatch targets (string or module object) reach the code that runs.
"""

import sys

from tldw_Server_API.app.core.Usage import llm_budget_guard as _moved

sys.modules[__name__] = _moved
