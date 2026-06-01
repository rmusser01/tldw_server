# Personas User Guide

Last Updated: 2026-06-01

## Overview

Personas are persistent assistant profiles for tldw_server. A Persona can hold
assistant instructions, durable state documents, scoped memories, style
exemplars, tool policies, voice commands, live-session preferences, and visual
Buddy packs. The same Persona can be used from ordinary chat, Persona Live, and
other workflow surfaces that need a stable assistant identity.

The Personas API is mounted under `/api/v1/persona`. It is user-scoped: profile
records, sessions, memory entries, exemplars, connections, commands, and visual
packs belong to the authenticated user.

## Personas vs Characters

Characters and Personas overlap conceptually, but they are not the same system.

| Concept | Main Use | Storage/Behavior |
| --- | --- | --- |
| Character | Roleplay or character-card chat using imported card data, world books, and dictionaries. | Character cards and conversations live in the per-user `ChaChaNotes` database. |
| Persona | Persistent assistant identity for chat, live sessions, scoped tools, state docs, exemplars, voice commands, and Buddy visuals. | Persona profiles and related records also live in the per-user `ChaChaNotes` database, but evolve independently after creation. |

A Persona can reference a character card at creation time, but later Persona
state, policies, exemplars, and visual packs are managed by the Persona feature.

## Prerequisites

1. Start the API server.
2. Authenticate with either:
   - `X-API-KEY` in single-user mode.
   - `Authorization: Bearer <token>` in multi-user/JWT mode.
3. Ensure Personas are enabled. In `Config_Files/config.txt`, `[persona] enabled = true`; the equivalent runtime setting is `PERSONA_ENABLED`.

Examples below use an API key:

```bash
export BASE_URL="http://127.0.0.1:8000"
export API_KEY="<your-api-key>"
```

## Quickstart

### 1. List the active Persona catalog

```bash
curl -s "$BASE_URL/api/v1/persona/catalog" \
  -H "X-API-KEY: $API_KEY" | jq
```

If the user has no active Personas yet, the server creates or returns the
default Persona profile.

### 2. Create a Persona profile

```bash
curl -s -X POST "$BASE_URL/api/v1/persona/profiles" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "id": "research_helper",
    "name": "Research Helper",
    "mode": "session_scoped",
    "system_prompt": "Help analyze sources carefully. Ask before using external tools.",
    "is_active": true,
    "use_persona_state_context_default": true
  }' | jq
```

Save the returned `id` as `PERSONA_ID`.

```bash
export PERSONA_ID="research_helper"
```

### 3. Add durable state docs

State docs are long-lived Markdown fields used to describe a Persona's durable
self-model. They are separate from ordinary chat turns.

```bash
curl -s -X PUT "$BASE_URL/api/v1/persona/profiles/$PERSONA_ID/state" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "identity_md": "You are a careful research assistant for technical reading.",
    "soul_md": "Prefer source-grounded answers, explicit uncertainty, and concise next steps.",
    "heartbeat_md": "Current focus: help the user inspect and summarize project documentation."
  }' | jq
```

View state history:

```bash
curl -s "$BASE_URL/api/v1/persona/profiles/$PERSONA_ID/state/history" \
  -H "X-API-KEY: $API_KEY" | jq
```

### 4. Add an exemplar

Exemplars are compact examples of style, boundaries, scenario behavior, or tool
behavior. They guide prompt assembly and runtime planning; they are not
permissions.

```bash
curl -s -X POST "$BASE_URL/api/v1/persona/profiles/$PERSONA_ID/exemplars" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "kind": "boundary",
    "content": "If a request would use an external tool, briefly explain the action and wait for confirmation.",
    "scenario_tags": ["tools", "safety"],
    "priority": 10,
    "enabled": true,
    "source_type": "manual"
  }' | jq
```

### 5. Start a Persona session

```bash
curl -s -X POST "$BASE_URL/api/v1/persona/session" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "persona_id": "'"$PERSONA_ID"'",
    "surface": "docs-example"
  }' | jq
```

Save the returned `session_id`:

```bash
export SESSION_ID="<returned-session-id>"
```

### 6. List and inspect sessions

```bash
curl -s "$BASE_URL/api/v1/persona/sessions?persona_id=$PERSONA_ID" \
  -H "X-API-KEY: $API_KEY" | jq
```

```bash
curl -s "$BASE_URL/api/v1/persona/sessions/$SESSION_ID" \
  -H "X-API-KEY: $API_KEY" | jq
```

## Core Concepts

### Profiles

A profile is the top-level Persona identity. Key fields include:

- `id`: optional stable identifier supplied at creation, otherwise generated.
- `name`: display name.
- `mode`: `session_scoped` or `persistent_scoped`.
- `system_prompt`: core assistant instructions.
- `is_active`: controls catalog visibility.
- `use_persona_state_context_default`: whether state docs are used by default.
- `voice_defaults`: saved STT/TTS, wake phrase, and turn-detection defaults.
- `setup`: setup wizard progress and status.

List, create, update, soft-delete, and restore profiles with:

- `GET /api/v1/persona/profiles`
- `POST /api/v1/persona/profiles`
- `GET /api/v1/persona/profiles/{persona_id}`
- `PATCH /api/v1/persona/profiles/{persona_id}`
- `DELETE /api/v1/persona/profiles/{persona_id}`
- `POST /api/v1/persona/profiles/{persona_id}/restore`

### Sessions

A session binds a Persona to a runtime surface and stores a scope snapshot. The
normal session endpoints are:

- `POST /api/v1/persona/session`
- `GET /api/v1/persona/sessions`
- `GET /api/v1/persona/sessions/{session_id}`
- `GET /api/v1/persona/sessions/{session_id}/export`

Transcript export is RBAC-gated by `[persona.rbac] allow_export` or
`PERSONA_RBAC_ALLOW_EXPORT`.

### Live Sessions

Persona Live adds focused runtime control around the same Persona identity:

- `GET /api/v1/persona/live/sessions`
- `POST /api/v1/persona/live/sessions`
- `POST /api/v1/persona/live/sessions/{session_id}/focus`
- `POST /api/v1/persona/live/sessions/{session_id}/stop`

Live sessions are the intended path for WebSocket-backed interaction and Buddy
state.

## Live Sessions and WebSocket Stream

The WebSocket endpoint is:

```text
WS /api/v1/persona/stream
```

The stream accepts JWT or API-key authentication. Query auth is supported for
WebSocket clients that cannot set headers:

```bash
websocat "ws://127.0.0.1:8000/api/v1/persona/stream?api_key=$API_KEY"
```

Send a user turn:

```json
{
  "type": "user_message",
  "session_id": "<session-id>",
  "text": "Summarize the active research context."
}
```

Common client frame types include:

- `user_message`: send a text turn.
- `voice_config`: update voice runtime settings for a session.
- `wake_activation` and `wake_deactivation`: manage manually armed wake phrase state.
- `voice_commit`: commit a transcript as a Persona Live turn.
- `audio_chunk`: send bounded audio chunks for live STT/TTS handling.
- `confirm_plan`: approve selected tool-plan steps.
- `retry_tool_call`: retry an approval-backed tool call.

Common server events include:

- `notice`: status, warning, error, and recovery messages.
- `tool_plan`: proposed tool plan requiring client confirmation when needed.
- `tool_call` and `tool_result`: tool execution lifecycle.
- `assistant_delta`: assistant text output.
- `tts_audio`: TTS chunk metadata followed by binary audio frames.

The stream enforces feature flags, authentication, policy checks, tool
confirmation, audio limits, and rate limits.

## State, Memory, and Exemplars

Personas use several context layers:

- **Profile:** identity and default configuration.
- **State docs:** durable Markdown fields: `soul_md`, `identity_md`, and `heartbeat_md`.
- **Memory:** persisted Persona turn and tool outcome entries used by retrieval.
- **Exemplars:** curated snippets that guide voice, boundaries, scenario style, or tool behavior.
- **Scope snapshot:** rule-derived context captured when a session is materialized.

Useful endpoints:

- `GET /api/v1/persona/profiles/{persona_id}/state`
- `PUT /api/v1/persona/profiles/{persona_id}/state`
- `GET /api/v1/persona/profiles/{persona_id}/state/history`
- `POST /api/v1/persona/profiles/{persona_id}/state/restore`
- `POST /api/v1/persona/profiles/{persona_id}/state/archive`
- `GET /api/v1/persona/profiles/{persona_id}/exemplars`
- `POST /api/v1/persona/profiles/{persona_id}/exemplars`
- `POST /api/v1/persona/profiles/{persona_id}/exemplars/import`

Transcript import can create candidate exemplars from a bounded transcript:

```bash
curl -s -X POST "$BASE_URL/api/v1/persona/profiles/$PERSONA_ID/exemplars/import" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "transcript": "User: Explain your safety policy. Assistant: I can help, but I need confirmation before tool actions.",
    "source_ref": "manual-demo",
    "max_candidates": 3
  }' | jq
```

## Scope and Policy Rules

Scope rules decide where a Persona applies. Supported rule types are:

- `conversation_id`
- `character_id`
- `media_id`
- `media_tag`
- `note_id`

Replace scope rules:

```bash
curl -s -X PUT "$BASE_URL/api/v1/persona/profiles/$PERSONA_ID/scope-rules" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "rules": [
      {"rule_type": "media_tag", "rule_value": "risk-review", "include": true}
    ]
  }' | jq
```

Policy rules gate capabilities. Supported rule kinds are:

- `mcp_tool`
- `skill`

Replace policy rules:

```bash
curl -s -X PUT "$BASE_URL/api/v1/persona/profiles/$PERSONA_ID/policy-rules" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "rules": [
      {
        "rule_kind": "mcp_tool",
        "rule_name": "*",
        "allowed": true,
        "require_confirmation": true,
        "max_calls_per_turn": 3
      }
    ]
  }' | jq
```

Policy and scope rules do not grant authentication privileges by themselves.
AuthNZ, MCP tool permissions, and Persona RBAC still apply.

## Voice Commands and Connections

Persona voice commands map heard phrases to actions. Commands can optionally
reference Persona connections, which define a bounded external HTTP target and
redacted secret/header configuration.

Useful endpoints:

- `GET /api/v1/persona/profiles/{persona_id}/voice-commands`
- `POST /api/v1/persona/profiles/{persona_id}/voice-commands`
- `PUT /api/v1/persona/profiles/{persona_id}/voice-commands/{command_id}`
- `POST /api/v1/persona/profiles/{persona_id}/voice-commands/{command_id}/toggle`
- `DELETE /api/v1/persona/profiles/{persona_id}/voice-commands/{command_id}`
- `POST /api/v1/persona/profiles/{persona_id}/voice-commands/test`
- `GET /api/v1/persona/profiles/{persona_id}/connections`
- `POST /api/v1/persona/profiles/{persona_id}/connections`
- `POST /api/v1/persona/profiles/{persona_id}/connections/{connection_id}/test`

Use the dry-run endpoint before enabling a voice command in a live surface:

```bash
curl -s -X POST "$BASE_URL/api/v1/persona/profiles/$PERSONA_ID/voice-commands/test" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{"heard_text": "summarize this source"}' | jq
```

## Visual Packs

Persona Visual Packs are user-owned Buddy assets attached to Personas. The
current activation path is manifest-backed `sprite_frames`; use the capabilities
endpoint before assuming another renderer is active:

```bash
curl -s "$BASE_URL/api/v1/persona/visual-renderers" \
  -H "X-API-KEY: $API_KEY" | jq
```

Common visual pack routes:

- `GET /api/v1/persona/visual-starter-packs`
- `POST /api/v1/persona/visual-starter-packs/{starter_pack_id}/copy`
- `GET /api/v1/persona/profiles/{persona_id}/visual-packs`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/assets`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/activate`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/export`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs/import-previews`

Generated or imported packs are not activated automatically. The user must
explicitly activate a valid pack.

## Privacy and Safety

- Persona records are user-scoped.
- Profile/session/state/exemplar data is stored in the authenticated user's
  `ChaChaNotes` database.
- Persona tool execution is gated by authentication, RBAC, policy rules, MCP
  permissions, and confirmation flow.
- State docs, memory, and exemplars should not be used to store secrets.
- Connection responses redact secrets and expose only bounded metadata.
- Visual assets are served through ownership-checked endpoints.
- Dialogue-tree/runtime explorer features are defensive and feature-flagged.
  They do not authorize actions and should not be treated as policy bypasses.
- Wake phrase support is manually armed in the active live surface; pre-wake
  audio is not sent to the server.

## Common Errors

| Status | Typical Cause | What To Check |
| --- | --- | --- |
| `401` | Missing or invalid API key/JWT. | Auth mode and headers/query credentials. |
| `403` | RBAC denied an operation such as transcript export. | `[persona.rbac]` config and user permissions. |
| `404` | Persona disabled or resource not found for this user. | `PERSONA_ENABLED`, profile id, session id, and ownership. |
| `409` | Optimistic locking, import conflict, non-completed preview, or incompatible live-session state. | Version fields, import preview status, and requested target mode. |
| `413` | State doc, upload, archive, or audio chunk is too large. | Configured size limits and request payload size. |
| `422` | Request validation failed. | Field names, enum values, required fields, and max lengths. |

## Related Docs

- `tldw_Server_API/app/core/Persona/README.md` - developer guide for the core module.
- `Docs/Code_Documentation/Persona_Visual_Packs.md` - Persona Visual Pack ownership, activation, import/export, and renderer notes.
- `Docs/User_Guides/WebUI_Extension/Persona_Live_Wake_Phrases.md` - wake phrase behavior in the WebUI/extension surfaces.
- `Docs/Operations/Persona_Memory_ChaCha_Cutover_Rollback_Runbook_2026_02_22.md` - memory cutover and rollback notes.
