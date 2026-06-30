# VoiceAssistant

VoiceAssistant provides the core command-routing, session, registry, intent parsing, workflow handling, schemas, and database helpers for the voice assistant API. It coordinates text-first assistant commands and workflow progress while endpoint code handles REST/WebSocket transport and audio-specific boundaries.

## Start Here

- `router.py` routes parsed intents to registered voice assistant commands.
- `registry.py` defines command registration and lookup.
- `intent_parser.py` parses command text into intents.
- `session.py` manages assistant session state.
- `workflow_handler.py` coordinates workflow execution/progress payloads.
- Related API surface: `app/api/v1/endpoints/voice_assistant.py`.
- Related tests: `tests/VoiceAssistant/`.

## Responsibilities

- Parse user command text into structured voice assistant intents.
- Register and route supported assistant commands.
- Manage session state and persistence helpers for command history.
- Coordinate workflow command execution and progress sanitization.
- Provide core schemas used by endpoint and workflow layers.
- Support persona voice-command persistence through shared database helpers.

## Module Map

- `router.py` - command routing and dispatch.
- `registry.py` - command registry.
- `intent_parser.py` - intent parsing logic.
- `session.py` - session state handling.
- `workflow_handler.py` - workflow execution/progress bridge.
- `db_helpers.py` - persistence helpers.
- `schemas.py` - core voice assistant models.

## How It Connects

- `app/api/v1/endpoints/voice_assistant.py` exposes REST and WebSocket API behavior.
- `app/api/v1/schemas/voice_assistant_schemas.py` defines endpoint contracts.
- Persona endpoints and tests use voice command persistence behavior.
- Audio transcription and TTS resolution happen at adjacent endpoint/service boundaries, while this package handles command/session logic.
- Documentation exists in `Docs/Code_Documentation/VoiceAssistant_Module.md` and `Docs/API/Voice_Assistant.md`.

## Extension Points

- For a new command, add registry/router support and cover it in `tests/VoiceAssistant/test_registry.py` and REST/WS tests as appropriate.
- For parsing changes, update `intent_parser.py` and `tests/VoiceAssistant/test_intent_parser.py`.
- For session behavior, update `session.py`, `db_helpers.py`, and session/persistence tests.
- For workflow commands, inspect `workflow_handler.py` and progress sanitization tests.

## Testing

- `tests/VoiceAssistant/test_intent_parser.py`
- `tests/VoiceAssistant/test_registry.py`
- `tests/VoiceAssistant/test_session.py`
- `tests/VoiceAssistant/test_rest_endpoints.py`
- `tests/VoiceAssistant/test_ws_integration.py`
- `tests/VoiceAssistant/test_e2e_pipeline.py`
- `tests/VoiceAssistant/test_voice_command_dry_run_endpoint.py`
- `tests/VoiceAssistant/test_workflow_progress_sanitization.py`
- `tests/VoiceAssistant/test_tts_resolution.py`
- `tests/VoiceAssistant/test_transcription_sanitization.py`
- Persona voice command tests also live under `tests/Persona/`.

## Gotchas

- This package is not the STT/TTS provider implementation; it coordinates voice assistant command/session behavior around those services.
- Workflow progress payloads need sanitization before they are exposed through endpoints.
- WebSocket test API-key behavior has runtime guards; keep endpoint and core-session changes aligned with those tests.
