from __future__ import annotations

from fastapi import APIRouter

from tldw_Server_API.app.api.v2.endpoints import user_profiles

api_v2_router = APIRouter(prefix="/api/v2")
api_v2_router.include_router(user_profiles.router, tags=["user-profiles-v2"])
