"""Moved to tldw_Server_API.app.core.Provider_Credentials.byok_runtime (TASK-13950.17).

An alias, not a re-export: the old name *is* the moved module, so old imports and
monkeypatch targets (string or module object) reach the code that runs.
"""

import sys

from tldw_Server_API.app.core.Provider_Credentials import byok_runtime as _moved

sys.modules[__name__] = _moved
