"""
Compatibility imports for Resource Governor policy administration.

The SQL-backed implementation lives under ``DB_Management`` so Resource
Governance code does not own raw database statements directly.
"""

from __future__ import annotations

from tldw_Server_API.app.core.DB_Management.Resource_Governance_Policy_Admin import (
    AuthNZPolicyAdmin,
    PolicyVersionConflictError,
)

__all__ = ["AuthNZPolicyAdmin", "PolicyVersionConflictError"]
