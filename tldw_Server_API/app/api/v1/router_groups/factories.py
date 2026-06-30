"""Shared lazy router factories for grouped router registration."""
from __future__ import annotations

from fastapi import APIRouter


def evaluations_router_factory() -> APIRouter:
    """Return the unified evaluations router after route policy allows it."""
    from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import (
        router as evaluations_router,
    )

    return evaluations_router
