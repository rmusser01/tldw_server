# Character Cards and Character Chat User Guide

Last updated: 2026-06-01

This guide covers the server-side Character Cards and Character Chat workflows in `tldw_server`: creating and importing character cards, starting character-backed chats, using world books and chat dictionaries, and understanding safety, privacy, and common API errors.

## What Character Cards Are

Character Cards are stored assistant profiles. A card can hold identity, style, scenario, greeting text, example dialogue, tags, creator metadata, an avatar image, and optional extension metadata. Character-backed chats use the selected card to assemble prompt context before sending a request to the configured chat provider.

Character Cards are separate from the Persona module. Character-backed chats are built around card fields and chat settings. Persona-backed chats may use some of the same chat-session infrastructure, but Persona Garden has its own profile/session model.

## Storage And Ownership

Character cards, conversations, messages, world books, dictionaries, prompt presets, settings, and related metadata are stored through the per-user `ChaChaNotes_DB` database layer. In multi-user mode, API access is scoped through the authenticated user. In single-user mode, the server still uses the same storage abstractions but authenticates with the configured single-user API key.

Images and card text are local/self-hosted data unless you explicitly send chat completions to an external provider. If you use a commercial or hosted LLM provider, the assembled chat prompt may include character fields, selected greetings, author notes, world-book entries, memory blocks, message history, and any appended user message.

## Character Card Fields

The core fields are:

- `name`: required for create/import.
- `description`: identity and background.
- `personality`: behavioral and voice guidance.
- `scenario`: current situation or setting.
- `system_prompt`: direct system-level instruction for the character.
- `post_history_instructions`: extra instruction appended after history in supported presets.
- `first_message`: default greeting/opening message.
- `message_example`: example dialogue or cadence reference.
- `alternate_greetings`: optional greeting variants.
- `tags`: organization and filtering labels.
- `creator`, `character_version`, `creator_notes`: attribution and maintenance metadata.
- `extensions`: structured metadata used by presets or external card formats.
- `image_base64`: optional avatar image data in API create/update payloads.

Text fields have server-side length limits. List-like fields such as `tags` and `alternate_greetings` may be sent as JSON arrays or JSON strings depending on the client path; invalid JSON is rejected rather than silently ignored.

## Importing Character Cards

Use:

```text
POST /api/v1/characters/import
```

Upload a file as `character_file`. Supported extensions are `.png`, `.webp`, `.jpeg`, `.jpg`, `.json`, `.yaml`, `.yml`, `.txt`, and `.md`.

The server validates both extension and content where possible. Image imports support embedded character metadata in PNG/WEBP-style card files; JPEG is accepted as an image type but normally does not carry the same embedded `chara` metadata. If no card metadata is found in an image, the endpoint returns a `422` response with `code=missing_character_data` and `can_import_image_only=true`. Retry with `allow_image_only=true` only when you intentionally want a card shell inferred from the image file name.

Supported structured card parsing includes Character Card v1/v2/v3 plus common Tavern/SillyTavern, Pygmalion, Text Generation WebUI, and Alpaca-style shapes. Imports are sanitized for obvious script-like content and enforce image and metadata size limits.

Common import responses:

- `201`: card imported.
- `200`: a name conflict was resolved by returning an existing matching card when the conflict can be mapped.
- `400`: invalid extension, malformed data, unsupported content, or parse failure.
- `413`: upload exceeds configured import size.
- `422`: image-only import requires explicit confirmation.

## Creating And Managing Cards

Common endpoints:

```text
POST   /api/v1/characters
GET    /api/v1/characters
GET    /api/v1/characters/query
GET    /api/v1/characters/filter
GET    /api/v1/characters/search/
GET    /api/v1/characters/{character_id}
PUT    /api/v1/characters/{character_id}
DELETE /api/v1/characters/{character_id}
POST   /api/v1/characters/{character_id}/restore
```

Updates and deletes use the database version field for optimistic locking where the endpoint requires `expected_version`. On a version mismatch, refresh the character and retry with the current version. Deletes are soft deletes by default; restore is available while the configured retention window allows it.

Bulk tag maintenance is available through:

```text
POST /api/v1/characters/tags/operations
```

Supported tag operations are `rename`, `merge`, and `delete`.

## Exporting Cards

Use:

```text
GET /api/v1/characters/{character_id}/export?format=v3
```

Supported export formats are:

- `v3`: Character Card v3 JSON shape.
- `v2`: Character Card v2 JSON shape.
- `json`: raw stored character payload.
- `png`: PNG character card with v2 JSON embedded in a `chara` metadata chunk.

Add `include_world_books=true` to include associated world-book data where supported by the export path. PNG export returns `image/png`; the other formats return JSON.

## Character Exemplars

Character exemplars are example text snippets with source, label, safety, and rights metadata. They can be used for search and debug selection workflows around character voice and behavior.

Common endpoints:

```text
POST   /api/v1/characters/{character_id}/exemplars
GET    /api/v1/characters/{character_id}/exemplars/{exemplar_id}
PUT    /api/v1/characters/{character_id}/exemplars/{exemplar_id}
DELETE /api/v1/characters/{character_id}/exemplars/{exemplar_id}
POST   /api/v1/characters/{character_id}/exemplars/search
POST   /api/v1/characters/{character_id}/exemplars/select/debug
```

Exemplar embedding sync is best-effort. If semantic scoring is unavailable, lexical or stored-search behavior may still work depending on the configured database and embedding setup.

## Starting A Character Chat

Create a chat session with:

```text
POST /api/v1/chats
```

For a character-backed chat, provide either `character_id` or `assistant_kind="character"` with a numeric `assistant_id`. Optional creation fields include `title`, lifecycle metadata such as `state`, workspace scope fields, fork metadata, and source/external reference fields.

Important creation query parameters:

- `seed_first_message=true`: seed the chat with the card's first greeting when possible.
- `greeting_strategy=default|alternate_random|alternate_index`: choose how seeded greeting text is selected.
- `alternate_index`: zero-based greeting index when using `alternate_index`.

The API also supports untracked/global chats and persona-backed chats through the same `/api/v1/chats` route family, but this guide focuses on character-backed chats.

## Messages

Common message endpoints:

```text
POST   /api/v1/chats/{chat_id}/messages
GET    /api/v1/chats/{chat_id}/messages
GET    /api/v1/messages/{message_id}
PUT    /api/v1/messages/{message_id}
DELETE /api/v1/messages/{message_id}
GET    /api/v1/chats/{chat_id}/messages/search
```

Messages use roles `user`, `assistant`, or `system`. The server maps those roles to stored senders and applies placeholder replacement when returning character-aware context. A message can contain text, an image attachment, or both. Supported message image detection includes PNG, JPEG, GIF, WebP, BMP, and ICO; oversized or invalid base64 image data is rejected.

For completion-ready history, call:

```text
GET /api/v1/chats/{chat_id}/messages?format_for_completions=true&include_character_context=true
```

Optional flags can include tool-call metadata, extra metadata, and message IDs in completion-formatted responses.

## Character Chat Completions

There are two main completion paths:

```text
POST /api/v1/chats/{chat_id}/completions
POST /api/v1/chats/{chat_id}/complete-v2
```

`/completions` prepares messages for use with the main OpenAI-compatible chat endpoint and does not call an LLM. It is useful when a client wants to inspect or forward assembled messages itself.

`/complete-v2` assembles context, resolves provider/model settings, enforces rate and message limits, calls the configured provider, and optionally persists messages. It accepts controls such as `append_user_message`, provider/model, `model="auto"` routing overrides, generation options, `tools`, `tool_choice`, prompt cache intents, message-steering flags, `prompt_preset`, and `stream`.

Persistence rules:

- Non-streaming completions persist the appended user message and assistant reply only when `save_to_db=true` or the server default save setting resolves true.
- Streaming completions return Server-Sent Events and do not persist assistant content automatically, even when `save_to_db=true`.
- After streaming, persist assistant text with:

```text
POST /api/v1/chats/{chat_id}/completions/persist
```

The persist endpoint can store speaker metadata, mood metadata, tool calls, usage, ranking, and chat rating.

## Prompt Preview, Presets, Greetings, And Author Notes

Prompt preview:

```text
POST /api/v1/chats/{chat_id}/prompt-preview
```

This returns supplemental prompt sections, token estimates, truncation flags, warnings, and lorebook diagnostics. It is the safest way to inspect what context will be injected before sending a real completion.

Prompt presets:

```text
GET    /api/v1/chats/presets/tokens
GET    /api/v1/chats/presets
POST   /api/v1/chats/presets
PUT    /api/v1/chats/presets/{preset_id}
DELETE /api/v1/chats/presets/{preset_id}
```

Built-in presets include `default` and `st_default`; custom presets can define section order and templates. Built-ins cannot be modified or deleted.

Greetings:

```text
GET /api/v1/chats/{chat_id}/greetings
PUT /api/v1/chats/{chat_id}/greetings/select
```

The server returns all greetings from the card, tracks the selected index in chat settings, and reports staleness if the card greetings changed after the chat started.

Author note:

```text
GET /api/v1/chats/{chat_id}/author-note/info
```

Author note info reports display text, prompt text, estimated token counts, budget, enabled/GM-only/excluded flags, scope, source, and warnings. Chat settings control the author note payload and injection behavior.

## Chat Settings

Use:

```text
GET /api/v1/chats/{chat_id}/settings
PUT /api/v1/chats/{chat_id}/settings
```

Settings are stored as JSON and merged with timestamp-aware conflict behavior. Current settings used by the prompt assembly path include greeting scope, preset scope, memory scope, selected greeting, chat preset overrides, participant character IDs, turn-taking mode, author note controls, character memory notes, auto-summary options, and linked deep-research attachment metadata.

Invalid settings are rejected or normalized. A settings row that only contains internal greeting checksum bootstrap metadata is not returned as user-visible settings.

## Multi-Character Chats

Character Chat can resolve additional participants from `participantCharacterIds` or `participant_character_ids` in chat settings. When multiple participants are configured, completion prep and `/complete-v2` can use `directed_character_id` to force the next response to a selected participant. If the ID is not part of the selected participant set, the server returns `400`.

Without a directed participant, the server chooses an active character from turn context and recent assistant senders.

## World Books

World books, also called lorebooks in some chat UIs, inject relevant context when keywords or regex patterns match the conversation text.

Common endpoints:

```text
GET    /api/v1/characters/world-books
POST   /api/v1/characters/world-books
GET    /api/v1/characters/world-books/{world_book_id}
PUT    /api/v1/characters/world-books/{world_book_id}
DELETE /api/v1/characters/world-books/{world_book_id}

POST   /api/v1/characters/world-books/{world_book_id}/entries
GET    /api/v1/characters/world-books/{world_book_id}/entries
PUT    /api/v1/characters/world-books/entries/{entry_id}
DELETE /api/v1/characters/world-books/entries/{entry_id}

POST   /api/v1/characters/{character_id}/world-books
GET    /api/v1/characters/{character_id}/world-books
DELETE /api/v1/characters/{character_id}/world-books/{world_book_id}

POST   /api/v1/characters/world-books/process
POST   /api/v1/characters/world-books/import
GET    /api/v1/characters/world-books/{world_book_id}/export
GET    /api/v1/characters/world-books/{world_book_id}/statistics
```

World-book entries contain keywords, content, priority, enabled status, case sensitivity, regex/whole-word controls, optional grouping, appendable metadata, recursive-scanning metadata, and arbitrary metadata. The processing endpoint returns injected content, matched entry IDs, token usage, budget exhaustion flags, skipped-entry counts, and diagnostics.

Attached world books are considered during character-chat prompt assembly. Diagnostics for world-book injection can be previewed with prompt preview and exported across turns with:

```text
GET /api/v1/chats/{chat_id}/diagnostics/lorebook
```

## Chat Dictionaries

Chat dictionaries transform text with literal or regex replacement entries. They are mounted under the chat API:

```text
POST   /api/v1/chat/dictionaries
GET    /api/v1/chat/dictionaries
GET    /api/v1/chat/dictionaries/{dictionary_id}
PUT    /api/v1/chat/dictionaries/{dictionary_id}
DELETE /api/v1/chat/dictionaries/{dictionary_id}

POST   /api/v1/chat/dictionaries/{dictionary_id}/entries
GET    /api/v1/chat/dictionaries/{dictionary_id}/entries
PUT    /api/v1/chat/dictionaries/entries/{entry_id}
DELETE /api/v1/chat/dictionaries/entries/{entry_id}

POST /api/v1/chat/dictionaries/process
POST /api/v1/chat/dictionaries/import
GET  /api/v1/chat/dictionaries/{dictionary_id}/export
GET  /api/v1/chat/dictionaries/{dictionary_id}/export/json
POST /api/v1/chat/dictionaries/import/json
GET  /api/v1/chat/dictionaries/{dictionary_id}/statistics
GET  /api/v1/chat/dictionaries/{dictionary_id}/activity
GET  /api/v1/chat/dictionaries/{dictionary_id}/versions
GET  /api/v1/chat/dictionaries/{dictionary_id}/versions/{revision}
POST /api/v1/chat/dictionaries/{dictionary_id}/versions/{revision}/revert
```

Dictionary entries support literal or regex matching, probability, group, timed effects, max replacements, enabled status, case sensitivity, and ordering. Regex patterns are validated with server-side safety checks, including length limits and nested-quantifier heuristics.

Use JSON import/export for full fidelity. Markdown import/export is useful for readable sharing but may not preserve every advanced field.

## Safety, Privacy, And Provider Boundaries

- Treat cards, world books, dictionaries, author notes, and memory notes as prompt material. They can be sent to the configured LLM provider during completion.
- Do not put secrets, credentials, private keys, or access tokens in card fields or world-book entries.
- Validate imported cards from untrusted sources before chatting with them. The server sanitizes obvious script-like content, but imported card instructions can still influence model behavior.
- Regex dictionaries and regex world-book entries are bounded, but complex patterns can still be costly. Prefer literal entries unless regex is required.
- Use prompt preview before sending sensitive or expensive completions.
- External provider calls use configured server keys or BYOK resolution. If no credentials are available for a provider that requires them, `/complete-v2` returns `503` with `error_code=missing_provider_credentials`.

## Common Errors And Fixes

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `400` invalid extension or content mismatch | File extension and magic bytes do not match, or data cannot be parsed | Export from the source tool again or upload JSON/YAML/text instead |
| `413` file too large | Import or image attachment exceeds configured limit | Reduce image size or raise the relevant server limit |
| `422` `missing_character_data` | Image has no embedded card metadata | Retry with `allow_image_only=true` only if you want an image-only card |
| `404` character/chat/message not found | Wrong ID, deleted item, wrong user, or wrong workspace scope | Refresh list endpoints and include the correct scope fields |
| `409` version mismatch | Optimistic-lock version is stale | Reload the object and retry with the current version |
| `429` rate limit exceeded | Resource Governor or character-chat rate limit denied the request | Wait for the `Retry-After` window or adjust policy |
| `503` missing provider credentials | Selected provider requires an API key | Configure server provider keys or BYOK credentials |
| Streamed reply not in history | Streaming does not auto-persist assistant text | Call `/api/v1/chats/{chat_id}/completions/persist` after the stream finishes |

## Related Source Files

- Character routes: `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
- Chat sessions and completions: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Messages: `tldw_Server_API/app/api/v1/endpoints/character_messages.py`
- Dictionaries: `tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py`
- Character schemas: `tldw_Server_API/app/api/v1/schemas/character_schemas.py`
- Chat session/message schemas: `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- World-book schemas: `tldw_Server_API/app/api/v1/schemas/world_book_schemas.py`
- Dictionary schemas: `tldw_Server_API/app/api/v1/schemas/chat_dictionary_schemas.py`
- Core module README: `tldw_Server_API/app/core/Character_Chat/README.md`
