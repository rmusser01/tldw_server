"""In-process Chat publication context; it cannot be supplied in HTTP input."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.DB_Management.Buddy_Turns_DB import BuddyTurnRepository


@dataclass(frozen=True)
class BuddyPublication:
    repository: BuddyTurnRepository
    turn: dict[str, Any]


current_buddy_publication: ContextVar[BuddyPublication | None] = ContextVar("buddy_publication", default=None)
