from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CharacterStore",
    "ConversationStore",
    "KeywordStore",
    "MessageStore",
    "NoteAttachment",
    "NoteAttachmentStore",
    "NoteStore",
    "NotesLinkStore",
    "PersonaStateStore",
    "SharedWorkspaceChatClaim",
    "SharedWorkspaceChatStore",
    "SharedWorkspaceChatThread",
    "SharedWorkspaceMessagePage",
    "SharedWorkspaceStoredMessage",
    "StaleSharedWorkspaceChatClaim",
    "StoredSharedWorkspaceTurn",
]


_STORE_MODULES = {
    "CharacterStore": "tldw_Server_API.app.core.DB_Management.chacha.character_store",
    "ConversationStore": "tldw_Server_API.app.core.DB_Management.chacha.conversation_store",
    "KeywordStore": "tldw_Server_API.app.core.DB_Management.chacha.keyword_store",
    "MessageStore": "tldw_Server_API.app.core.DB_Management.chacha.message_store",
    "NoteAttachment": "tldw_Server_API.app.core.DB_Management.chacha.note_attachment_store",
    "NoteAttachmentStore": "tldw_Server_API.app.core.DB_Management.chacha.note_attachment_store",
    "NoteStore": "tldw_Server_API.app.core.DB_Management.chacha.note_store",
    "NotesLinkStore": "tldw_Server_API.app.core.DB_Management.chacha.note_link_store",
    "NoteGraphProjectionStore": "tldw_Server_API.app.core.DB_Management.chacha.note_graph_projection_store",
    "PersonaStateStore": "tldw_Server_API.app.core.DB_Management.chacha.persona_state_store",
    "SharedWorkspaceChatClaim": "tldw_Server_API.app.core.DB_Management.chacha.shared_workspace_chat_store",
    "SharedWorkspaceChatStore": "tldw_Server_API.app.core.DB_Management.chacha.shared_workspace_chat_store",
    "SharedWorkspaceChatThread": "tldw_Server_API.app.core.DB_Management.chacha.shared_workspace_chat_store",
    "SharedWorkspaceMessagePage": "tldw_Server_API.app.core.DB_Management.chacha.shared_workspace_chat_store",
    "SharedWorkspaceStoredMessage": "tldw_Server_API.app.core.DB_Management.chacha.shared_workspace_chat_store",
    "StaleSharedWorkspaceChatClaim": "tldw_Server_API.app.core.DB_Management.chacha.shared_workspace_chat_store",
    "StoredSharedWorkspaceTurn": "tldw_Server_API.app.core.DB_Management.chacha.shared_workspace_chat_store",
}


def __getattr__(name: str) -> Any:
    module_name = _STORE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
