# Character Cards Documentation Design

## Purpose

Create source documentation for the Character Cards and Character Chat surfaces without editing generated `Docs/Published` output.

## Audience

- Server users and API clients who need a practical guide for character cards, character-backed chats, world books, dictionaries, and common error handling.
- Contributors who need to understand the `tldw_Server_API/app/core/Character_Chat/` module boundary and its FastAPI touch points before changing code.

## Source Of Truth

- `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- `tldw_Server_API/app/api/v1/endpoints/character_messages.py`
- `tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py`
- `tldw_Server_API/app/api/v1/schemas/character_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/world_book_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/chat_dictionary_schemas.py`
- `tldw_Server_API/app/core/Character_Chat/`
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`

## Scope

In scope:

- Add a server user guide for Character Cards and Character Chat workflows.
- Refresh the `Character_Chat` module README for current responsibilities, data flow, extension guidance, endpoints, and tests.
- Link the new user guide from the source user-guide index.
- Record verification and leave `Docs/Published` untouched.

Out of scope:

- API or database behavior changes.
- Generated published-doc updates.
- Frontend UX changes.

## Accuracy Constraints

- Do not claim automatic editing of `Docs/Published`; it is generated separately.
- Do not promise UI controls that are not guaranteed by the backend.
- Use route families and source file names rather than fragile line-number references.
- Call out streaming persistence behavior accurately: `stream=true` returns SSE and does not persist assistant content automatically; clients persist streamed content with `POST /api/v1/chats/{chat_id}/completions/persist`.
- Distinguish character cards from Personas. Character-card chats can coexist with persona-backed chats, but Character Cards are not the Persona Garden runtime.

## Deliverables

- `Docs/User_Guides/Server/Character_Cards_User_Guide.md`
- `tldw_Server_API/app/core/Character_Chat/README.md`
- `Docs/User_Guides/index.md`
- Backlog task notes with verification and known skips.
