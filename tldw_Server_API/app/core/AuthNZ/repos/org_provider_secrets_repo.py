"""Moved to tldw_Server_API.app.core.Provider_Credentials.repos.org_provider_secrets_repo (TASK-13950.17).

An alias, not a re-export: the old name *is* the moved module, so old imports and
monkeypatch targets (string or module object) reach the code that runs.
"""

import sys

from tldw_Server_API.app.core.Provider_Credentials.repos import org_provider_secrets_repo as _moved

sys.modules[__name__] = _moved
