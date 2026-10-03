# quota_manager.py
# Description: Quota helpers for chatbook operations
#
"""
Quota Manager for Chatbook Operations
--------------------------------------

Per-user daily export/import limits and concurrent-job limits are UserProfiles
``limits.chatbooks_*`` values, enforced in ``chatbook_service`` job admission.
This module keeps only the off switch that admission reads and the fixed file-size cap.
"""

import os
from typing import Any, Optional

from tldw_Server_API.app.core.config import usage_quotas_enabled
from tldw_Server_API.app.core.testing import is_truthy


def _env_flag(name: str) -> bool:
    return is_truthy(os.getenv(name))


# Fixed per-request guardrail (spec 2 §7 keeps it; it is not a usage quota).
MAX_CHATBOOK_FILE_SIZE_MB = 100


class QuotaManager:
    """Chatbook quota helpers.

    Per-user daily export/import limits and concurrent-job limits are UserProfiles
    ``limits.chatbooks_*`` values, enforced in ``chatbook_service`` job admission.
    This class keeps only the off switch that admission reads and the fixed file-size cap.
    """

    def __init__(self, user_id: str, user_tier: str = "free", db: Optional[Any] = None):
        """Bind to a user; ``user_tier`` is accepted for compatibility and ignored."""
        self.user_id = user_id
        self.db = db
        self._quotas_disabled = (
            not usage_quotas_enabled()
            or _env_flag("CHATBOOKS_DISABLE_QUOTAS")
            or _env_flag("TEST_MODE")
            or _env_flag("TESTING")
            or bool(os.getenv("PYTEST_CURRENT_TEST"))
        )

    async def check_file_size(self, file_size_bytes: int) -> tuple[bool, str]:
        """Refuse a file larger than MAX_CHATBOOK_FILE_SIZE_MB."""
        if file_size_bytes > MAX_CHATBOOK_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File too large. Maximum size is {MAX_CHATBOOK_FILE_SIZE_MB}MB"
        return True, "File size OK"
