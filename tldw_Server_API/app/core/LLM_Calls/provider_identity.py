"""Moved to ``core/Utils/provider_identity.py`` (TASK-13950.12) because ``core.config`` is foundation."""

from __future__ import annotations

from tldw_Server_API.app.core.Utils.provider_identity import (  # noqa: F401
    PROVIDER_ALIASES,
    canonical_provider_name,
    provider_lookup_names,
)
