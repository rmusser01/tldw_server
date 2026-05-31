from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "CharacterStore",
    "ConversationStore",
    "KeywordStore",
    "MessageStore",
    "NoteStore",
    "PersonaStateStore",
]


_STORE_MODULES = {
    "CharacterStore": "tldw_Server_API.app.core.DB_Management.chacha.character_store",
    "ConversationStore": "tldw_Server_API.app.core.DB_Management.chacha.conversation_store",
    "KeywordStore": "tldw_Server_API.app.core.DB_Management.chacha.keyword_store",
    "MessageStore": "tldw_Server_API.app.core.DB_Management.chacha.message_store",
    "NoteStore": "tldw_Server_API.app.core.DB_Management.chacha.note_store",
    "PersonaStateStore": "tldw_Server_API.app.core.DB_Management.chacha.persona_state_store",
}


def __getattr__(name: str) -> Any:
    module_name = _STORE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
