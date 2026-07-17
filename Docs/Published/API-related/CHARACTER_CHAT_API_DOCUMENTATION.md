# Character Chat API Documentation

## Overview

The Character Chat API provides comprehensive endpoints for managing character-based chat sessions, messages, and interactions. This API enables creating persistent conversations with AI characters, managing message history, and performing character-specific completions.

Implementation status: Character CRUD, chat sessions, messages, search, chat history export, character export, and rate limiting are implemented. Character-specific LLM responses are performed via the core Chat API (`POST /api/v1/chat/completions`) using conversation context from character chats.

## Table of Contents

1. [Authentication](#authentication)
2. [Character Management](#character-management)
3. [Chat Session Management](#chat-session-management)
4. [Message Management](#message-management)
5. [Chat Completions](#chat-completions)
6. [Search and Filtering](#search-and-filtering)
7. [Export/Import](#exportimport)
8. [Rate Limiting](#rate-limiting)
9. [Error Handling](#error-handling)

---

## Authentication

Supported headers:
- Single-user: `X-API-KEY: <key>`
- Multi-user: `Authorization: Bearer <JWT>`

If authentication is required and missing/invalid, endpoints return `401`.

---

## Character Management

### Create Character

Create a new character card.

**Endpoint:** `POST /api/v1/characters/`

**Request Body (selected fields):**
```json
{
  "name": "Assistant",
  "description": "A helpful AI assistant",
  "personality": "Friendly and knowledgeable",
  "first_message": "Hello! How can I help you today?",
  "scenario": "You are chatting with a helpful assistant",
  "message_example": "USER: What can you do?\nASSISTANT: I can help with various tasks!",
  "tags": ["assistant", "helpful"]
}
```

**Response:** `201 Created`
```json
{
  "id": 1,
  "name": "Assistant",
  "description": "A helpful AI assistant",
  "personality": "Friendly and knowledgeable",
  "first_message": "Hello! How can I help you today?",
  "version": 1
}
```

### Get Character

Retrieve a specific character by ID.

**Endpoint:** `GET /api/v1/characters/{character_id}`

**Response:** `200 OK`
```json
{
  "id": 1,
  "name": "Assistant",
  "description": "A helpful AI assistant",
  "personality": "Friendly and knowledgeable",
  "first_message": "Hello! How can I help you today?",
  "scenario": "You are chatting with a helpful assistant",
  "message_example": "...",
  "tags": ["assistant", "helpful"],
  "version": 1
}
```

### List Characters

Get a paginated list of characters.

**Endpoint:** `GET /api/v1/characters/`

**Query Parameters:**
- `limit` (int, default: 100, max: 1000): Number of characters to return
- `offset` (int, default: 0): Number of characters to skip

**Response:** `200 OK` (array of characters)
```json
[
  {
    "id": 1,
    "name": "Assistant",
    "description": "A helpful AI assistant",
    "version": 1
  }
]
```

### Update Character

Update an existing character's information.

**Endpoint:** `PUT /api/v1/characters/{character_id}?expected_version={version}`

**Query Parameters:**
- `expected_version` (int, required): Expected version for optimistic locking

**Request Body:**
```json
{
  "name": "Updated Assistant",
  "description": "An even more helpful AI assistant"
}
```

**Response:** `200 OK`
```json
{
  "id": 1,
  "name": "Updated Assistant",
  "description": "An even more helpful AI assistant",
  "version": 2,
  ...
}
```

### Delete Character

Soft delete a character (marks as deleted but preserves data).

**Endpoint:** `DELETE /api/v1/characters/{character_id}?expected_version={version}`

**Response:** `200 OK`
```json
{
  "message": "Character '<name>' (ID: <id>) soft-deleted.",
  "character_id": <id>
}
```

---

### Deprecated: Legacy Completion Endpoint

The legacy endpoint `POST /api/v1/chats/{chat_id}/complete` is deprecated.

- The request body is no longer supported. Sending a non-empty body now returns `422 Unprocessable Entity`.
- Deprecation headers are sent with responses: `Deprecation: true`, a `Sunset` date ~90 days from release, and a `Link` header pointing to the successor endpoint.
- Please migrate to one of the following:
  - `POST /api/v1/chats/{chat_id}/complete-v2` to execute a completion (with optional persistence and streaming).
  - `POST /api/v1/chats/{chat_id}/completions` to prepare chat messages for the unified Chat API (`/api/v1/chat/completions`).

---

## Chat Session Management

### Create Chat Session

Start a new chat session with a character.

**Endpoint:** `POST /api/v1/chats/`

**Request Body:**
```json
{
  "character_id": 1,
  "title": "Evening Chat",
  "parent_conversation_id": null
}
```

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "character_id": 1,
  "title": "Evening Chat",
  "rating": null,
  "created_at": "2024-09-04T12:00:00Z",
  "last_modified": "2024-09-04T12:00:00Z",
  "message_count": 0,
  "version": 1
}
```

### Get Chat Session

Retrieve a specific chat session.

**Endpoint:** `GET /api/v1/chats/{chat_id}`

**Response:** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "character_id": 1,
  "title": "Evening Chat",
  "rating": null,
  "created_at": "2024-09-04T12:00:00Z",
  "last_modified": "2024-09-04T12:00:00Z",
  "message_count": 5,
  "version": 1
}
```

### List User Chats

Get all chat sessions for the current user.

**Endpoint:** `GET /api/v1/chats/`

**Query Parameters:**
- `character_id` (int): Filter by character
- `limit` (int, default: 50): Number of chats to return
- `offset` (int, default: 0): Number of chats to skip

**Response:** `200 OK`
```json
{
  "chats": [...],
  "total": 10,
  "limit": 20,
  "offset": 0
}
```

### Update Chat Session

Update chat session metadata.

**Endpoint:** `PUT /api/v1/chats/{chat_id}`

**Query Parameters:**
- `expected_version` (int, required): Expected version for optimistic locking

**Request Body:**
```json
{
  "title": "Updated Chat Title",
  "rating": 5
}
```

**Response:** `200 OK`

### Delete Chat Session

Soft delete a chat session.

**Endpoint:** `DELETE /api/v1/chats/{chat_id}?expected_version={version}` (version optional)

**Response:** `204 No Content`

---

## Message Management

### Send Message

Add a new message to a chat session.

**Endpoint:** `POST /api/v1/chats/{chat_id}/messages`

**Request Body:**
```json
{
  "role": "user",
  "content": "Hello! Tell me about yourself.",
  "parent_message_id": null,
  "image_base64": null
}
```

**Response:** `201 Created`
```json
{
  "id": "msg_123456",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "parent_message_id": null,
  "sender": "user",
  "content": "Hello! Tell me about yourself.",
  "timestamp": "2024-09-04T12:00:00Z",
  "ranking": null,
  "has_image": false,
  "version": 1
}
```

### Get Messages

Retrieve messages from a chat session, with optional character context for AI completions.

**Endpoint:** `GET /api/v1/chats/{chat_id}/messages`

**Query Parameters:**
- `limit` (int, default: 50): Number of messages to return
- `offset` (int, default: 0): Number of messages to skip
- `include_deleted` (bool, default: false): Include soft-deleted messages
- `include_character_context` (bool, default: false): Include character personality information
- `format_for_completions` (bool, default: false): Format response for use with `/api/v1/chat/completions`
- `include_tool_calls` (bool, default: false): Include a `tool_calls` field per message (standard format only)
- `include_metadata` (bool, default: false): Include stored per-message `metadata.extra` where available
- `include_message_ids` (bool, default: false): Include `message_id` fields when `format_for_completions=true` for messages backed by stored chat rows (typically user/assistant, and persisted system messages). Synthetic system prompts and tool role messages do not include `message_id`.

**Response:** `200 OK`

Standard format:
```json
{
  "messages": [
    {
      "id": "msg_123456",
      "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
      "sender": "user",
      "content": "Hello!",
      "timestamp": "2024-09-04T12:00:00Z",
      "version": 1
    },
    {
      "id": "msg_123457",
      "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
      "sender": "assistant",
      "content": "Hello! I'm your helpful assistant.",
      "timestamp": "2024-09-04T12:00:05Z",
      "version": 1
    }
  ],
  "total": 2,
  "limit": 50,
  "offset": 0
}
```

With `format_for_completions=true&include_character_context=true&include_message_ids=true` (tool calls and tool results shown; `message_id` is only present for stored user/assistant messages):
```json
{
  "character_name": "Assistant",
  "character_id": 1,
  "chat_id": "550e8400-e29b-41d4-a716-446655440000",
  "messages": [
    {
      "role": "system",
      "content": "You are Assistant.\nA helpful AI assistant.\nFriendly and knowledgeable."
    },
    {
      "role": "user",
      "message_id": "msg_123456",
      "content": "Hello!"
    },
    {
      "role": "assistant",
      "message_id": "msg_123457",
      "content": "Hello! I'm your helpful assistant.",
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {"name": "search", "arguments": "{\"query\": \"hello\"}"}
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_123",
      "name": "search",
      "content": "{\"content\": \"result text\"}"
    }
  ],
  "total": 2,
  "usage_instructions": "Use these messages with POST /api/v1/chat/completions"
}
```

When `include_metadata=true`, the response also includes a top-level `metadata_extra` object keyed by `message_id`, containing stored JSON sidecar data.

### Get Chat Context (compact)

Return compact context for a chat, including character name and messages formatted for completions when available.

**Endpoint:** `GET /api/v1/chats/{chat_id}/context`

**Response:** `200 OK`
```json
{
  "character_name": "Assistant",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

### Get Specific Message

Retrieve a single message by ID.

**Endpoint:** `GET /api/v1/messages/{message_id}`

**Response:** `200 OK`

### Edit Message

Edit the content of an existing message.

**Endpoint:** `PUT /api/v1/messages/{message_id}?expected_version={version}`

**Request Body:**
```json
{
  "content": "Updated message content"
}
```

**Response:** `200 OK`

### Delete Message

Soft delete a message.

**Endpoint:** `DELETE /api/v1/messages/{message_id}?expected_version={version}`

**Response:** `204 No Content`

### Search Messages

Search for messages within a chat session.

**Endpoint:** `GET /api/v1/chats/{chat_id}/messages/search`

**Query Parameters:**
- `query` (string, required): Search query
- `limit` (int, default: 50): Maximum results

**Response:** `200 OK`
```json
{
  "messages": [...],
  "total": 5,
  "limit": 50,
  "offset": 0
}
```

---

## Chat Completions

To generate AI responses in character chat sessions, use the main OpenAI-compatible chat completions endpoint:

**Endpoint:** `POST /api/v1/chat/completions`

This endpoint supports:
- Multiple LLM providers (OpenAI, Anthropic, local models, etc.)
- Streaming responses
- System prompts for character personality
- Conversation history
- Ephemeral or persistent operation (see `save_to_db` below)

Streaming behavior follows the core Chat API: the server sends an initial `event: stream_start`, emits delta chunks as OpenAI-style `choices[].delta.content`, and terminates with a single `data: [DONE]` (heartbeat comments are sent periodically). Duplicate terminal markers are suppressed.

### Workflow for Character Chat Completions

1. **Get formatted messages from the chat session:**
```bash
curl -X GET "http://localhost:8000/api/v1/chats/{chat_id}/messages?format_for_completions=true&include_character_context=true&include_message_ids=true" \
  -H "X-API-KEY: your-api-key"
```

2. **Use the formatted messages with the completions endpoint:**
```bash
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
 -d '{
    "model": "gpt-4o",
    "messages": [messages from step 1],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

By default, chats are ephemeral (not saved). To persist conversation/messages automatically, add `"save_to_db": true` to the request body:

```bash
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [messages from step 1],
    "save_to_db": true
  }'
```

Server default for persistence can be configured via environment variable `CHAT_SAVE_DEFAULT=true` or in `Config_Files/config.txt` under `[Chat-Module]` with `chat_save_default = True`.

Persistence guard: If `save_to_db=true` is set but there is no valid character/chat context (e.g., missing `character_id` and `conversation_id` in the request), the server will disable persistence for that request and continue normally to avoid invalid writes. A warning is logged; no partial records are created. When calling completions for character chats, always include `conversation_id` (the chat ID) or `character_id` in the request body when you want persistence.

3. **Save the AI response as a new message (optional):**
```bash
curl -X POST "http://localhost:8000/api/v1/chats/{chat_id}/messages" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "role": "assistant",
    "content": "[AI response from step 2]"
  }'
```

See the main [Chat API documentation](/api/v1/docs#/chat) for complete details on the chat completions endpoint.

Also see: `Docs/API-related/Chat_API_Documentation.md` for a focused Chat API reference.

Legacy test-only endpoint with minimal per-chat throttling exists for rate-limit tests:

- `POST /api/v1/chats/{chat_id}/complete` → returns `{ "status": "ok", "chat_id": "..." }` or 429 on bursts

### Character Chat Completions (Rate-Limited)

Prepare messages for use with the main Chat API while enforcing a per-minute completion limiter.

**Endpoint:** `POST /api/v1/chats/{chat_id}/completions`

**Request Body:**
```json
{
  "include_character_context": true,
  "limit": 100,
  "offset": 0,
  "append_user_message": "Tell me more about your background.",
  "prompt_preset": "st_default"
}
```

Selected request fields:
- `prompt_preset` (optional string): Single-turn override for prompt preset selection.
- `directed_character_id` (optional int): Direct next reply to a selected participant in multi-character chats.
- `continue_as_user` / `impersonate_user` / `force_narrate` (optional bools): Single-response steering controls.

**Response:** `200 OK`
```json
{
  "chat_id": "...",
  "character_id": 1,
  "character_name": "Assistant",
  "messages": [
    {"role": "system", "content": "You are Assistant. ..."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "Tell me more about your background."}
  ],
  "total": 2,
  "usage_instructions": "Use these messages with POST /api/v1/chat/completions"
}
```

This endpoint enforces the per-minute completion limit (HTTP 429 on exceed). Use the response `messages` directly with `POST /api/v1/chat/completions`.

### Character Chat Completion (Operational + Persistence)

Call the LLM directly and optionally persist both the appended user message and the assistant response into the chat.

**Endpoint:** `POST /api/v1/chats/{chat_id}/complete-v2`

**Request Body (selected):**
```json
{
  "include_character_context": true,
  "append_user_message": "Tell me more about your background.",
  "prompt_preset": "st_default",
  "save_to_db": true,
  "provider": "local-llm",
  "model": "local-test",
  "temperature": 0.7,
  "max_tokens": 300
}
```

Notes:
- `provider` and `model` are optional; when omitted, defaults prefer `local-llm` for offline/dev usage.
- When `save_to_db` is omitted, server default is used (see Chat API `DEFAULT_SAVE_TO_DB`).
- Set `"stream": true` in the request body to stream the result via `text/event-stream` (SSE). In offline/dev mode without a running provider, streaming is disabled and a non-streaming response is returned.
- `prompt_preset` applies only for this request and takes precedence over chat/character preset scope defaults.

#### Streaming Behavior

When `stream=true` and the provider supports streaming, the server emits Server-Sent Events (SSE):

- Each chunk is sent as a line prefixed with `data: `, followed by a blank line (SSE framing).
- If the upstream provider already emits SSE-formatted lines (beginning with `data:`), they are forwarded as-is.
- Exactly one terminal marker is sent at the end: `data: [DONE]`. Duplicate terminal markers are suppressed.
- On transform/iteration errors mid-stream, an error payload is sent and the stream still terminates with `data: [DONE]`.

**Response:** `200 OK`
```json
{
  "chat_id": "...",
  "character_id": 1,
  "provider": "local-llm",
  "model": "local-test",
  "saved": true,
  "user_message_id": "...",
  "assistant_message_id": "...",
  "assistant_content": "Here is more about my background..."
}
```

### Prompt Assembly Preview

Preview the assembled supplemental prompt sections (preset, author note, steering, greeting, lorebook), including token estimates, section budgets, truncation flags, and conflict warnings.

**Endpoint:** `POST /api/v1/chats/{chat_id}/prompt-preview`

**Request Body (selected):**
```json
{
  "include_character_context": true,
  "prompt_preset": "st_default",
  "directed_character_id": 2,
  "continue_as_user": false,
  "impersonate_user": false,
  "force_narrate": false
}
```

**Response (selected):**
```json
{
  "chat_id": "...",
  "sections": [
    {
      "name": "preset",
      "content": "You are ...",
      "tokens_estimated": 140,
      "tokens_effective": 140,
      "budget": 180,
      "truncated": false
    }
  ],
  "total_supplemental_tokens": 220,
  "total_supplemental_effective_tokens": 220,
  "supplemental_budget": 1200,
  "budget_status": "ok",
  "warnings": [],
  "conflicts": []
}
```

### Chat Settings Metadata (Selected Keys)

`GET/PUT /api/v1/chats/{chat_id}/settings` stores merged per-chat settings. Selected fields:

- `presetScope`: `"chat"` or `"character"`.
- `chatPresetOverrideId`: chat-level preset ID used when `presetScope="chat"`.
- `chatGenerationOverride`: canonical chat generation override object.
- `generationOverrides`: legacy alias accepted for backward compatibility.

`chatGenerationOverride` / `generationOverrides` object:
- `enabled` (optional bool)
- `temperature` (`0.0-2.0`)
- `top_p` (`0.0-1.0`)
- `repetition_penalty` (`0.0-3.0`)
- `stop` (array of strings)
- `updatedAt` (ISO timestamp)

Compatibility and precedence:
- Server evaluates `chatGenerationOverride` first.
- If canonical override is absent, server reads `generationOverrides` as fallback.
- If both are present, canonical `chatGenerationOverride` wins.

---

## World Books

Manage world books (lorebooks) to inject structured context into character chats.

### Create World Book

**Endpoint:** `POST /api/v1/characters/world-books`

```json
{
  "name": "WB Test",
  "description": "World book for tests",
  "scan_depth": 3,
  "token_budget": 500,
  "recursive_scanning": false,
  "enabled": true
}
```

**Response:** `201 Created`

### List World Books

**Endpoint:** `GET /api/v1/characters/world-books`

**Query:** `include_disabled` (bool, default: false)

**Response:** `200 OK`
```json
{
  "world_books": [ { "id": 1, "name": "WB Test", "entry_count": 0 } ],
  "total": 1,
  "enabled_count": 1,
  "disabled_count": 0
}
```

### Get/Update/Delete World Book

- `GET /api/v1/characters/world-books/{world_book_id}`
- `PUT /api/v1/characters/world-books/{world_book_id}`
- `DELETE /api/v1/characters/world-books/{world_book_id}`

Note: world book names are unique; renaming to an existing name returns `409`.

### Entries

- `POST /api/v1/characters/world-books/{world_book_id}/entries`
- `GET  /api/v1/characters/world-books/{world_book_id}/entries`
- `PUT  /api/v1/characters/world-books/entries/{entry_id}`
- `DELETE /api/v1/characters/world-books/entries/{entry_id}`
- `POST /api/v1/characters/world-books/entries/bulk`

### Attach to Character

- `POST /api/v1/characters/{character_id}/world-books` (attach)
- `DELETE /api/v1/characters/{character_id}/world-books/{world_book_id}` (detach)
- `GET /api/v1/characters/{character_id}/world-books` (list attached)

### Process Context

`POST /api/v1/characters/world-books/process` → returns injected content for input text

---

## Search and Filtering

### Search Characters

Search for characters by name, description, or tags.

**Endpoint:** `GET /api/v1/characters/search/`

**Query Parameters:**
- `query` (string, required): Search query
- `limit` (int, default: 20): Maximum results

**Response:** `200 OK`
```json
[
  {
    "id": 1,
    "name": "Assistant",
    "description": "A helpful AI assistant",
    "tags": ["assistant", "helpful"],
    ...
  }
]
```

### Filter by Tags

Filter characters by specific tags.

**Endpoint:** `GET /api/v1/characters/filter`

**Query Parameters:**
- `tags` (array): List of tags to filter by (passed as multiple query params)
- `match_all` (bool, default: false): If true, require all tags; if false, match any tag
- `limit` (int, default: 50, max: 200): Maximum results
- `offset` (int, default: 0): Pagination offset

**Response:** `200 OK`
```json
[
  {
    "id": 2,
    "name": "Wizard",
    "description": "A wise wizard",
    "tags": ["fantasy", "wizard", "magic"],
    ...
  }
]
```

---

## Export/Import

### Export Character

Export a character in various formats, optionally including attached world books.

**Endpoint:** `GET /api/v1/characters/{character_id}/export`

**Query Parameters:**
- `format` (string, default: `"json"`): Export format. Supported values: `"v3"` (Character Card V3), `"v2"` (Character Card V2), `"json"` (raw JSON from DB)
- `include_world_books` (bool, default: `false`): Include world book data and entries in the export

For `format=v3`, the response follows the Character Card V3 spec structure with fields under `data` (e.g., `name`, `description`, `personality`, `first_mes`, etc.). For `format=v2`, it returns a simplified V2 structure. For `format=json`, it returns the raw character record (with binary image, if present, encoded as Base64 under `character_image`).

Example (V3):
```json
{
  "spec": "chara_card_v3",
  "spec_version": "3.0",
  "data": {
    "name": "Assistant",
    "description": "A helpful AI assistant",
    "personality": "Friendly and knowledgeable",
    "first_mes": "Hello! How can I help you today?",
    "mes_example": "...",
    "system_prompt": "You are a helpful assistant.",
    "alternate_greetings": [],
    "tags": ["assistant"],
    "extensions": {}
  }
}
```

Example (include world books):
```json
{
  "id": 1,
  "name": "Assistant",
  "world_books": [
    {
      "id": 10,
      "name": "Fantasy Lore",
      "entries": [ { "id": 1, "keywords": ["magic"], "content": "..." } ]
    }
  ]
}
```

### Export Chat History

Export a chat session's history.

**Endpoint:** `GET /api/v1/chats/{chat_id}/export`

**Query Parameters:**
- `format` (string, default: "json"): Export format ("json", "markdown", or "text")
- `include_metadata` (bool, default: true): Include chat metadata
- `include_character` (bool, default: true): Include character information

**Response:** `200 OK`

For JSON format:
```json
{
  "chat_id": "550e8400-e29b-41d4-a716-446655440000",
  "character_name": "Assistant",
  "character_id": 1,
  "title": "Evening Chat",
  "created_at": "2024-09-04T12:00:00Z",
  "messages": [
    {
      "id": "msg_123456",
      "sender": "user",
      "content": "Hello!",
      "timestamp": "2024-09-04T12:00:00Z"
    },
    {
      "id": "msg_123457",
      "sender": "assistant",
      "content": "Hello! How can I help you?",
      "timestamp": "2024-09-04T12:00:05Z",
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {"name": "search", "arguments": "{\"query\": \"hello\"}"}
        }
      ]
    }
  ],
  "metadata": {
    "version": 1,
    "message_count": 2
  },
  "message_metadata_extra": {
    "msg_123457": {"tool_results": {"call_123": {"content": "result text"}}}
  }
}
```

JSON export fields (metadata extras):
- When `include_metadata=true` and one or more messages have stored extras, the response includes a top-level `message_metadata_extra` object.
- Keys are `message_id`; values are arbitrary JSON previously stored for that message. By convention, tool execution outputs are stored under `tool_results`, keyed by `tool_call_id`.

Example shape of `message_metadata_extra`:
```json
{
  "message_metadata_extra": {
    "msg_987": {
      "tool_results": {
        "call_abc": {"content": "result text", "score": 0.92},
        "call_def": {"items": [1, 2, 3]}
      },
      "version": 1
    }
  }
}
```

Notes:
- Messages may also include `tool_calls` directly. If a message has no stored metadata but contains an inline suffix like `"[tool_calls]: [...]"`, the export parser populates `messages[].tool_calls` from that inline value for convenience.
- The `message_metadata_extra` block is omitted when `include_metadata=false` or when no messages have stored extras.

For Markdown format, returns a plain text markdown representation of the conversation.

### Import Character

Import a character from various formats including V3.

**Endpoint:** `POST /api/v1/characters/import`

**Request:** Multipart form data
- `character_file`: Character card file (supports PNG, WEBP, JSON, MD formats)

**Response:** `201 Created`
```json
{
  "id": 1,
  "name": "Imported Character",
  "message": "Character 'Imported Character' imported successfully"
}
```

**Note:** The endpoint automatically detects the format. For JSON files, it supports Character Card V3 format among others.

---

## Rate Limiting

The API implements several rate limits to prevent abuse. Redis is optional - if Redis is unavailable or the `redis` package is not installed, the server automatically falls back to an in-memory limiter suitable for single-instance deployments.

Configuration summary:
- General character ops: `CHARACTER_RATE_LIMIT_OPS`, `CHARACTER_RATE_LIMIT_WINDOW`.
- Chat-specific per-minute limits: `MAX_CHAT_COMPLETIONS_PER_MINUTE`, `MAX_MESSAGE_SENDS_PER_MINUTE`.
- Soft cap for non-persisted completions: `MAX_MESSAGES_PER_CHAT_SOFT` (defaults to `MAX_MESSAGES_PER_CHAT`; set lower to cap ephemeral completions, e.g., `MAX_MESSAGES_PER_CHAT_SOFT=200` with a 1000 hard cap).
- Optional Redis: set `REDIS_ENABLED=true` and `REDIS_URL` to enable distributed rate limiting. Without Redis, limits apply per process.

The API enforces the following defaults:

### Character Operations
- **Max operations per hour**: 100 per user
- **Max characters per user**: 1000
- **Max import file size**: 10MB

### Chat Operations
- **Max concurrent chats per user**: 100
- **Max messages per chat (hard)**: 1000
- **Max messages per chat (soft, non-persisted completions)**: 1000 (defaults to hard cap; example override: 200)
- **Max chat completions per minute**: 20
- **Max message sends per minute**: 60

### Checking Rate Limit Status

To check your current rate limit usage:

**Endpoint:** `GET /api/v1/characters/rate-limit-status`

**Response:** `200 OK`
```json
{
  "operations_used": 12,
  "operations_remaining": 88,
  "reset_time": 1736520000.0
}
```

When rate limited, the API returns `429 Too Many Requests`:
```json
{
  "detail": "Rate limit exceeded. Max 20 chat completions per minute."
}
```

**Note:** Rate limit information is not currently returned in response headers. Use the rate limit status endpoint to check your usage.

---

## Tokenizer Configuration

Dictionary and World Book modules estimate tokens when applying budgets. You can view and adjust the server’s token counting strategy via configuration endpoints.

Endpoints:
- GET `/api/v1/config/tokenizer` → returns current mode and settings
- PUT `/api/v1/config/tokenizer` → updates mode (non-persistent; in-memory only)

Modes:
- `whitespace` (default): counts whitespace-separated tokens
- `char_approx`: approximates by character length (≈ length/4). Adjustable with `divisor`.

Examples:
1) Read current config
```
GET /api/v1/config/tokenizer
{
  "mode": "whitespace",
  "divisor": 4,
  "available_modes": ["whitespace", "char_approx"]
}
```

2) Switch to character-approximation with divisor 4
```
PUT /api/v1/config/tokenizer
{
  "mode": "char_approx",
  "divisor": 4
}
```

Notes:
- This setting is applied process-wide and is not persisted across restarts.
- These endpoints adjust estimates for token budgets in chat dictionary and world book processing only.

---

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

### Common Status Codes

- `200 OK`: Successful GET/PUT request
- `201 Created`: Successful POST request creating new resource
- `204 No Content`: Successful DELETE request
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Authenticated but not authorized for resource
- `404 Not Found`: Resource doesn't exist
- `409 Conflict`: Version conflict (optimistic locking)
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### Error Response Format

```json
{
  "detail": "Detailed error message",
  "error": "ErrorType",
  "chat_id": "optional-related-chat-id",
  "message_id": "optional-related-message-id"
}
```

### Optimistic Locking

Many update/delete operations require an `expected_version` parameter to prevent concurrent modification conflicts. If the version doesn't match, a `409 Conflict` error is returned:

```json
{
  "detail": "Version mismatch. Expected 2, found 3"
}
```

---

## Usage Examples

### Complete Chat Flow Example

1. **Create a character:**
```bash
curl -X POST "http://localhost:8000/api/v1/characters/" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Helper",
    "description": "A helpful assistant",
    "personality": "Friendly",
    "first_message": "Hello!"
  }'
```

2. **Create a chat session:**
```bash
curl -X POST "http://localhost:8000/api/v1/chats/" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "character_id": 1,
    "title": "My Chat"
  }'
```

3. **Send a message:**
```bash
curl -X POST "http://localhost:8000/api/v1/chats/{chat_id}/messages" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "content": "Hello!"
  }'
```

4. **Get AI response using chat completions:**
```bash
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "How are you?"}
    ],
    "max_tokens": 150
  }'
```

5. **Export chat history:**
```bash
curl -X GET "http://localhost:8000/api/v1/chats/{chat_id}/export?format=markdown" \
  -H "X-API-KEY: your-api-key"
```

### Python Client Example

```python
import json
import socket
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

class CharacterChatClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-API-KEY": api_key
        }

    def _request_json(self, method, path, payload=None, params=None, timeout=None):
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json", **self.headers}
        req = Request(url, data=data, headers=headers, method=method)
        try:
            if timeout is None:
                with urlopen(req) as resp:
                    body = resp.read().decode("utf-8")
            else:
                with urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
        except HTTPError as err:
            error_body = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {err.code} error for {url}: {error_body}"
            ) from err
        except URLError as err:
            raise ConnectionError(f"Connection error for {url}: {err.reason}") from err
        except socket.timeout as err:
            raise TimeoutError(f"Request to {url} timed out") from err

    def create_character(self, name, description, personality, first_message):
        return self._request_json(
            "POST",
            "/api/v1/characters/",
            {
                "name": name,
                "description": description,
                "personality": personality,
                "first_message": first_message,
            },
        )

    def create_chat(self, character_id, title=None):
        return self._request_json(
            "POST",
            "/api/v1/chats/",
            {
                "character_id": character_id,
                "title": title,
            },
        )

    def send_message(self, chat_id, content, role="user"):
        """Send a message to a chat session."""
        return self._request_json(
            "POST",
            f"/api/v1/chats/{chat_id}/messages",
            {
                "role": role,
                "content": content,
            },
        )

    def get_messages_for_completions(self, chat_id):
        """Get messages formatted for use with chat completions."""
        return self._request_json(
            "GET",
            f"/api/v1/chats/{chat_id}/messages",
            params={
                "format_for_completions": True,
                "include_character_context": True,
                "limit": 50,
            },
        )

    def get_completion(self, chat_id, message, max_tokens=500):
        # First get the formatted messages with character context
        context = self.get_messages_for_completions(chat_id)

        # Add the new message
        messages = context["messages"]
        messages.append({"role": "user", "content": message})

        # Call the main chat completions endpoint
        result = self._request_json(
            "POST",
            "/api/v1/chat/completions",
            {
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
        )

        # Extract the response
        if "choices" in result and len(result["choices"]) > 0:
            ai_response = result["choices"][0]["message"]["content"]

            # Save the AI response back to the conversation
            self.send_message(chat_id, ai_response, role="assistant")

            return {
                "response": ai_response,
                "usage": result.get("usage", {})
            }
        return result

# Usage
client = CharacterChatClient("http://localhost:8000", "your-api-key")

# Create a character
character = client.create_character(
    name="Assistant",
    description="A helpful AI assistant",
    personality="Friendly and knowledgeable",
    first_message="Hello! How can I help you?"
)

# Start a chat
chat = client.create_chat(character["id"], "Evening Chat")

# Send message
message = client.send_message(chat["id"], "Hello!")

# Get AI response through chat completions
response = client.get_completion(chat["id"], "Tell me a joke")
if "response" in response:
    print(response["response"])
else:
    print("Error getting completion:", response)
```

---

## Related Documentation

- Core Chat API: `Docs/API-related/Chat_API_Documentation.md`
- Chatbook features (dictionaries, documents, import/export): `Docs/API-related/Chatbook_Features_API_Documentation.md`

For provider integration testing, see the “Commercial Tests” section in `Docs/API-related/Chat_API_Documentation.md`.

Configuration notes for providers: API keys are read from environment variables and from `.env`/`.ENV` files (project root or `tldw_Server_API/Config_Files/`), falling back to `Config_Files/config.txt` `[API]` entries. See the Chat API doc for precedence and a quick sanity-check snippet.

---

## Notes

- All timestamps are in UTC ISO 8601 format
- Character IDs are integers
- Chat IDs are UUIDs; message IDs are opaque strings (often `msg_`-prefixed in examples)
- Soft deletes preserve data but mark as deleted
- Optimistic locking prevents concurrent modification conflicts
- Rate limits are per-user, not per-API-key
- Streaming responses use Server-Sent Events (SSE)

---

## Version History

- **v1.0.0** (2024-09-04): Initial release with complete character chat API

---

*For more information about the tldw_server project, visit the [main documentation](API_README.md).*
