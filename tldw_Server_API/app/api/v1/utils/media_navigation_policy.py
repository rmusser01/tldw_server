"""Shared dependency policy for upcoming media navigation endpoints.

Stage 1 contract freeze artifact:
- Keeps auth/db/rate-limit policy explicit and stable before endpoint wiring.
- Lets Stage 2 endpoints import one shared dependency definition.
"""

from __future__ import annotations

from dataclasses import dataclass

# Stable, named resource key for media navigation read-rate controls.
MEDIA_NAVIGATION_RATE_LIMIT_RESOURCE = "media.navigation"


@dataclass(frozen=True)
class MediaNavigationRoutePolicy:
    """Contract-only policy descriptor for Stage 2 endpoint wiring."""

    auth_dependency_name: str = "get_request_user"
    db_dependency_name: str = "get_media_db_for_user"
    rate_limit_factory_name: str = "rbac_rate_limit"
    rate_limit_resource: str = MEDIA_NAVIGATION_RATE_LIMIT_RESOURCE


MEDIA_NAVIGATION_ROUTE_POLICY = MediaNavigationRoutePolicy()
