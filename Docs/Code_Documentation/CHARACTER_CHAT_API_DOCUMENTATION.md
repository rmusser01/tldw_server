# Character Chat API Documentation

## Overview

The Character Chat API provides comprehensive endpoints for managing character-based chat sessions, messages, and interactions. This API enables creating persistent conversations with AI characters, managing message history, and performing character-specific completions.

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

All endpoints require authentication via API key in the header:

```http
X-API-KEY: your-api-key
```

For multi-user mode, JWT Bearer tokens are used instead:

```http
Authorization: Bearer your-jwt-token
```

---

## Character Management

### Create Character

Create a new character card.

**Endpoint:** `POST /api/v1/characters/`

**Request Body:**
```json
{
  "name": "Assistant",
  "description": "A helpful AI assistant",
  "personality": "Friendly and knowledgeable",
  "first_message": "Hello! How can I help you today?",
  "scenario": "You are chatting with a helpful assistant",
  "example_messages": [
    {"role": "user", "content": "What can you do?"},
    {"role": "assistant", "content": "I can help with various tasks!"}
  ],
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
  "created_at": "2024-09-04T12:00:00Z",
  "updated_at": "2024-09-04T12:00:00Z",
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
  "example_messages": [...],
  "tags": ["assistant", "helpful"],
  "created_at": "2024-09-04T12:00:00Z",
  "updated_at": "2024-09-04T12:00:00Z",
  "version": 1
}
```

### List Characters

Get a paginated list of characters.

**Endpoint:** `GET /api/v1/characters/`

**Query Parameters:**
- `limit` (int, default: 20): Number of characters to return
- `offset` (int, default: 0): Number of characters to skip
- `search` (string): Search term for character names/descriptions

**Response:** `200 OK`
```json
{
  "characters": [...],
  "total": 50,
  "limit": 20,
  "offset": 0
}
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

**Response:** `204 No Content`

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
- `limit` (int, default: 20): Number of chats to return
- `offset` (int, default: 0): Number of chats to skip
- `active_only` (bool, default: true): Only show active chats

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

**Endpoint:** `DELETE /api/v1/chats/{chat_id}?expected_version={version}`

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

Retrieve messages from a chat session.

**Endpoint:** `GET /api/v1/chats/{chat_id}/messages`

**Query Parameters:**
- `limit` (int, default: 50): Number of messages to return
- `offset` (int, default: 0): Number of messages to skip
- `include_deleted` (bool, default: false): Include soft-deleted messages

**Response:** `200 OK`
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

### Character-Specific Completion

Generate a response from a character in an existing chat session.

**Endpoint:** `POST /api/v1/chats/{chat_id}/complete`

**Request Body:**
```json
{
  "message": "What's the weather like today?",
  "max_tokens": 500,
  "temperature": 0.7,
  "stream": false,
  "include_history": true,
  "history_limit": 20
}
```

**Response:** `200 OK`
```json
{
  "response": "I'm an AI assistant and don't have access to real-time weather data. To get current weather information, I recommend checking a weather website or app for your location.",
  "message_id": "msg_123458",
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 32,
    "total_tokens": 77
  }
}
```

### Streaming Completion

For streaming responses, set `stream: true` in the request. The response will be Server-Sent Events (SSE):

```
data: {"chunk": "I'm an AI assistant"}
data: {"chunk": " and don't have access"}
data: {"chunk": " to real-time weather data."}
data: {"done": true, "message_id": "msg_123458"}
```

---

## Search and Filtering

### Search Characters

Search for characters by name, description, or tags.

**Endpoint:** `GET /api/v1/characters/search`

**Query Parameters:**
- `q` (string, required): Search query
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

**Request Body:**
```json
{
  "tags": ["fantasy", "wizard"],
  "match_all": false
}
```

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

Export a character in various formats.

**Endpoint:** `GET /api/v1/characters/{character_id}/export`

**Query Parameters:**
- `format` (string, default: "v3"): Export format ("v3", "v2", or "json")

**Response:** `200 OK`
```json
{
  "spec": "chara_card_v3",
  "spec_version": "3.0",
  "data": {
    "name": "Assistant",
    "description": "A helpful AI assistant",
    ...
  }
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
```json
{
  "chat_id": "550e8400-e29b-41d4-a716-446655440000",
  "character_name": "Assistant",
  "character_id": 1,
  "title": "Evening Chat",
  "created_at": "2024-09-04T12:00:00Z",
  "messages": [...],
  "metadata": {...}
}
```

### Import Character V3

Import a character in V3 format.

**Endpoint:** `POST /api/v1/characters/import/v3`

**Request Body:**
```json
{
  "spec": "chara_card_v3",
  "spec_version": "3.0",
  "data": {
    "name": "Imported Character",
    "description": "Character from V3 format",
    ...
  }
}
```

**Response:** `201 Created`

---

## Rate Limiting

The API implements several rate limits to prevent abuse:

### Character Operations
- **Max operations per hour**: 100 per user
- **Max characters per user**: 1000
- **Max import file size**: 10MB

### Chat Operations
- **Max concurrent chats per user**: 100
- **Max messages per chat**: 1000
- **Max chat completions per minute**: 20
- **Max message sends per minute**: 60

Rate limit information is returned in response headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1693843200
```

When rate limited, the API returns `429 Too Many Requests`:
```json
{
  "detail": "Rate limit exceeded. Max 20 chat completions per minute."
}
```

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

4. **Get character response:**
```bash
curl -X POST "http://localhost:8000/api/v1/chats/{chat_id}/complete" \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How are you?",
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
import requests
import json

class CharacterChatClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
    
    def create_character(self, name, description, personality, first_message):
        response = requests.post(
            f"{self.base_url}/api/v1/characters/",
            headers=self.headers,
            json={
                "name": name,
                "description": description,
                "personality": personality,
                "first_message": first_message
            }
        )
        return response.json()
    
    def create_chat(self, character_id, title=None):
        response = requests.post(
            f"{self.base_url}/api/v1/chats/",
            headers=self.headers,
            json={
                "character_id": character_id,
                "title": title
            }
        )
        return response.json()
    
    def send_message(self, chat_id, content):
        response = requests.post(
            f"{self.base_url}/api/v1/chats/{chat_id}/messages",
            headers=self.headers,
            json={
                "role": "user",
                "content": content
            }
        )
        return response.json()
    
    def get_completion(self, chat_id, message, max_tokens=500):
        response = requests.post(
            f"{self.base_url}/api/v1/chats/{chat_id}/complete",
            headers=self.headers,
            json={
                "message": message,
                "max_tokens": max_tokens
            }
        )
        return response.json()

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

# Send message and get response
message = client.send_message(chat["id"], "Hello!")
response = client.get_completion(chat["id"], "Tell me a joke")
print(response["response"])
```

---

## Notes

- All timestamps are in UTC ISO 8601 format
- Character IDs are integers
- Chat and message IDs are UUIDs
- Soft deletes preserve data but mark as deleted
- Optimistic locking prevents concurrent modification conflicts
- Rate limits are per-user, not per-API-key
- Streaming responses use Server-Sent Events (SSE)

---

## Version History

- **v1.0.0** (2024-09-04): Initial release with complete character chat API

---

*For more information about the tldw_server project, visit the [main documentation](index.md).*
