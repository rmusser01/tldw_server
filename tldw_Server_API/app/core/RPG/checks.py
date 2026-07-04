from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.RPG.errors import RPGValidationError
from tldw_Server_API.app.core.RPG.models import CheckResult


def resolve_check(adapter: Any, roller: Any, payload: dict[str, Any]) -> CheckResult:
    result = adapter.resolve_check(roller, payload)
    if not isinstance(result, CheckResult):
        raise RPGValidationError("rules adapter returned an invalid check result")
    return result
