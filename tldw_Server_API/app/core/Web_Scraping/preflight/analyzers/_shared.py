"""Private failure handling shared by governed preflight analyzers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from ..context import PreflightDeadlineExceeded
from ..probes import ProbeError


async def _safe_analyzer_call(
    call: Awaitable[dict[str, Any]],
) -> dict[str, Any]:
    try:
        return await call
    except asyncio.CancelledError:
        raise
    except PreflightDeadlineExceeded:
        raise
    except ProbeError as exc:
        return {
            "status": "error",
            "message": exc.public_message,
            "error_code": exc.error_code,
        }
    except Exception:  # noqa: BLE001 - analyzer boundary sanitizes defensive failures
        return {
            "status": "error",
            "message": "Analyzer failed.",
            "error_code": "analyzer_error",
        }
