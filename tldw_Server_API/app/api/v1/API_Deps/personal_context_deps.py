"""Authenticated dependency assembly for the Personal Context API."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from fastapi import Depends

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user,
)
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    get_personalization_db_for_user,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization.companion_user_ids import (
    resolve_existing_companion_storage_user_id,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
)


def get_workspace_access_checker(
    database: Any = Depends(get_chacha_db_for_user),
) -> Callable[[str], bool]:
    """Return an ownership check bound to the authenticated user's workspace DB."""

    def owns_workspace(workspace_id: str) -> bool:
        try:
            return database.get_workspace(workspace_id) is not None
        except (KeyError, ValueError):
            return False

    return owns_workspace


def personal_context_service_for_user(
    user_id: str | int,
    *,
    database: PersonalizationDB | None = None,
    workspace_access: Callable[[str], bool] | None = None,
    clock: Callable[[], datetime] | None = None,
    id_factory: Callable[[str], str] | None = None,
) -> PersonalContextService:
    """Build a service over exactly one already-authenticated user's storage."""

    storage_user_id = resolve_existing_companion_storage_user_id(user_id)
    owning_database = database or PersonalizationDB.for_user(storage_user_id)
    return PersonalContextService(
        PersonalContextRepository(owning_database),
        workspace_access=workspace_access,
        clock=clock,
        id_factory=id_factory,
    )


def get_personal_context_service(
    user: User = Depends(get_request_user),
    database: PersonalizationDB = Depends(get_personalization_db_for_user),
    workspace_access: Callable[[str], bool] = Depends(get_workspace_access_checker),
) -> PersonalContextService:
    """Authenticate first, then bind Personal Context to that exact user."""

    return personal_context_service_for_user(
        user.id,
        database=database,
        workspace_access=workspace_access,
    )
