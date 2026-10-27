# /api/v1/voice (Voice Assistant API)

Purpose: real-time and REST-driven voice commands using a shared core pipeline:

- STT (optional) → intent parse → action execute → TTS (optional)
- Actions can route to MCP tools, workflows, custom handlers, or LLM chat

Core implementation:

- REST + WebSocket endpoints: `tldw_Server_API/app/api/v1/endpoints/voice_assistant.py`
- Core module: `tldw_Server_API/app/core/VoiceAssistant/`

## Endpoints (REST)

### POST `/api/v1/voice/command`

Process a text command as if it were spoken (bypasses STT).

Request body (`VoiceCommandRequest`):

- `text` (required): transcribed text to process
- `session_id` (optional): reuse an existing voice session
- `include_tts` (default `true`): include base64 TTS in the response
- `tts_provider`, `tts_voice`, `tts_format` (optional overrides)

Response (`VoiceCommandResponse`):

- `session_id`: resolved session identifier
- `intent`: parsed intent (action type + metadata)
- `action_result`: action result (success, response text, error)
- `output_audio` / `output_audio_format`: base64 TTS (when enabled)
- `processing_time_ms`: end-to-end processing time

### Voice Commands CRUD

Voice commands can be system-level (YAML defaults, `user_id=0`) or user-level
(DB-backed).

- GET `/api/v1/voice/commands`
  - Query:
    - `include_system` (default `true`)
    - `include_disabled` (default `false`)
- POST `/api/v1/voice/commands`
  - Creates a user command
- GET `/api/v1/voice/commands/{command_id}`
  - Fetch a specific command (user or system)
- PUT `/api/v1/voice/commands/{command_id}`
  - Update a user command (system commands are forbidden)
- POST `/api/v1/voice/commands/{command_id}/toggle`
  - Enable/disable a user command
- DELETE `/api/v1/voice/commands/{command_id}`
  - Soft-delete a user command

Command definition notes:

- `phrases` are matched as prefixes.
- Conflicts are resolved by match score, then `priority`.
- See `tldw_Server_API/Config_Files/voice_commands.yaml` for system defaults.

### Sessions + Analytics

- GET `/api/v1/voice/sessions`
  - Query:
    - `active_only` (default `true`): uses the session timeout window
    - `limit` (default `100`, max `1000`)
- GET `/api/v1/voice/sessions/{session_id}`
  - Session detail snapshot
- DELETE `/api/v1/voice/sessions/{session_id}`
  - End a session (204 on success)

Analytics:

- GET `/api/v1/voice/analytics`
  - Query:
    - `days` (default `7`, range `1..365`)
  - Returns aggregate metrics, top commands, and daily usage
- GET `/api/v1/voice/commands/{command_id}/usage`
  - Query:
    - `days` (default `30`, range `1..365`)
  - Returns per-command usage stats

### Workflows (Voice Bridge)

- GET `/api/v1/voice/workflows/templates`
  - Lists voice-oriented workflow templates
- GET `/api/v1/voice/workflows/{run_id}/status`
  - Workflow run status for the current user
- POST `/api/v1/voice/workflows/{run_id}/cancel`
  - Best-effort cancellation

## WebSocket: `/api/v1/voice/assistant`

Purpose: low-latency voice turns with streamed STT input and streamed TTS output.

High-level protocol:

1. Client sends `auth`
2. Server sends `auth_ok` (or closes)
3. Client sends `config`
4. Client streams `audio` and/or sends `text`
5. Client sends `commit` to finalize an utterance
6. Server sends `transcription` → `intent` → `action_*` → `tts_*`

### Client → Server Messages

First message must be `auth`. It is the only frame the server reads before it knows the caller, and its token is
verified like `Authorization: Bearer` on HTTP (revocation, API-key `read` scope, single-user key and IP allowlist):

- any other first message: `auth_error` "Expected AUTH message", then close `4400`;
- a missing, refused or revoked token: `auth_error` "Invalid credentials", then close `4401`;
- no `auth` message within 10 seconds: `auth_error` "Authentication timeout", then close `4401`.

A frame without a token falls back to the single-user WebUI session cookie, then to a credential on the upgrade
request, then to the deprecated `?token=` query parameter (logged as deprecated; URLs end up in access logs).

Auth:

```json
{"type": "auth", "token": "YOUR_API_KEY_OR_JWT"}
```

Config:

```json
{
  "type": "config",
  "stt_model": "parakeet",
  "stt_language": "en",
  "tts_provider": "kokoro",
  "tts_voice": "af_heart",
  "tts_format": "mp3",
  "sample_rate": 16000
}
```

Audio (base64 PCM float32 frames):

```json
{"type": "audio", "data": "<base64_float32_pcm>", "sequence": 1}
```

Commit:

```json
{"type": "commit"}
```

Text (bypass STT):

```json
{"type": "text", "text": "search for notes about rag reranking"}
```

Other controls:

- `{"type":"cancel"}` clears buffered audio and pending confirmations
- `{"type":"workflow_subscribe","run_id":"..."}`
- `{"type":"workflow_cancel","run_id":"..."}`

### Server → Client Messages (Typical Turn)

State changes:

```json
{"type": "state_change", "state": "listening"}
{"type": "state_change", "state": "processing"}
```

Transcription:

```json
{"type": "transcription", "text": "search for rag benchmarks", "is_final": true}
```

Intent + action:

```json
{"type": "intent", "action_type": "mcp_tool", "command_name": "Search Media"}
{"type": "action_start", "action_type": "mcp_tool"}
{"type": "action_result", "success": true, "response_text": "I found 3 results."}
```

TTS streaming:

```json
{"type": "tts_chunk", "sequence": 1, "format": "mp3", "data": "<base64_audio>"}
{"type": "tts_end", "total_chunks": 4, "total_bytes": 58231}
```

On completion, the server returns to idle:

```json
{"type": "state_change", "state": "idle"}
```

## Configuration + Route Gating Notes

Voice routes are mounted under the `voice-assistant` and `voice-assistant-ws`
route keys in `tldw_Server_API/app/main.py`.

Important for tests:

- When `MINIMAL_TEST_APP=1`, voice routes are not mounted.
- For voice endpoint tests, set:
  - `MINIMAL_TEST_APP=0`
  - `ULTRA_MINIMAL_APP=0`
  - and reload `tldw_Server_API.app.main` if already imported

## Data Model + Analytics Notes

Voice persistence and analytics live in the ChaChaNotes DB:

- `voice_commands`: command definitions
- `voice_sessions`: session snapshots
- `voice_command_events`: per-turn analytics events

Analytics endpoints are powered by aggregate queries in
`tldw_Server_API/app/core/VoiceAssistant/db_helpers.py`.

## Testing Pointers

Targeted test commands:

- REST endpoints:
  - `python -m pytest -q tldw_Server_API/tests/VoiceAssistant/test_rest_endpoints.py`
- Pipeline behaviors:
  - `python -m pytest -q tldw_Server_API/tests/VoiceAssistant/test_e2e_pipeline.py`

