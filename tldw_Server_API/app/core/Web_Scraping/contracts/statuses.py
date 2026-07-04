from __future__ import annotations

from enum import Enum


class WebScrapingStatus(str, Enum):
    """Internal statuses shared by Web_Scraping refactor contracts."""

    OK = "ok"
    BLOCKED = "blocked"
    POLICY_DENIED = "policy_denied"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"
    EXTERNAL_TOOL_DISABLED = "external_tool_disabled"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
