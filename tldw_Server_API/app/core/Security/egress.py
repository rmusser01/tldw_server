"""Moved to core/Utils/egress.py (TASK-13950.12).

An alias, not a re-export: the old name *is* the moved module, so old imports and
monkeypatch targets (string or module object) reach the code that runs. core.http_client
is foundation and enforces this policy on every outbound request, so the policy lives in
foundation too. Retire this once no importer or patch target uses the old path.
"""

import sys
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.Utils import egress as _egress

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Utils.egress import *  # noqa: F403

sys.modules[__name__] = _egress
