"""Compatibility import for the Explainer repository.

The implementation lives under DB_Management because it owns SQLite access.
"""

from __future__ import annotations

from tldw_Server_API.app.core.DB_Management.Explainer_Repository import ExplainerRepository

__all__ = ["ExplainerRepository"]
