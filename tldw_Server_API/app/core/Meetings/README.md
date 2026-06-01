# Meetings

Meetings coordinates meeting sessions, templates, artifacts, event streams, and outbound integration dispatch for the `/meetings` API. It keeps business rules in services while persistence lives in the per-user Meetings database and request or WebSocket authentication lives in API dependency code.

## Start Here

- `session_service.py` owns session creation, listing, lookup, and status transitions.
- `template_service.py` manages built-in and user-created meeting templates.
- `artifact_service.py` creates artifacts and generates simple final summaries, action items, decisions, and speaker stats.
- `events_service.py` persists meeting events and returns session snapshots.
- `integration_service.py` queues Slack and webhook dispatch records.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/meetings.py`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/meetings_schemas.py`.
- Related tests: `tldw_Server_API/tests/Meetings/`.

## Responsibilities

- Create, read, and list meeting sessions.
- Enforce allowed status transitions across scheduled, live, processing, completed, and failed states.
- Provide built-in default and standup templates and user-created templates.
- Persist meeting event log entries and build event snapshots.
- Generate final meeting artifacts from transcript text.
- Queue outbound Slack and webhook dispatch rows after egress policy evaluation.
- Convert meeting events to SSE frames and support WebSocket stream events.

## Module Map

- `session_service.py`: session lifecycle and status transition rules.
- `template_service.py`: built-in and persisted template behavior.
- `artifact_service.py`: artifact creation and final artifact generation.
- `events_service.py`: event persistence and session snapshots.
- `integration_service.py`: Slack and webhook dispatch validation and queuing.
- `stream_adapter.py`: event timestamps, event payload construction, and SSE formatting.
- `__init__.py`: package marker.

## How It Connects

- `meetings.py` exposes REST, SSE, and WebSocket routes under `/meetings`.
- `API_Deps/Meetings_DB_Deps.py` opens the per-user Meetings DB and handles WebSocket auth from bearer headers, API keys, subprotocols, or query tokens.
- `DB_Management/Meetings_DB.py` owns tables for sessions, templates, artifacts, integration dispatch rows, and event logs.
- `integration_service.py` depends on `Security.egress.evaluate_url_policy` before webhook dispatch.
- `app/services/meetings_webhook_dlq_service.py` handles failed webhook dispatch retry or dead-letter behavior.
- Developer context lives in `Docs/Code_Documentation/Meetings_Developer_Guide.md`.

## Extension Points

- Add a session status transition in `session_service.py` and database validation together.
- Add a template scope or built-in template in `template_service.py`.
- Add final artifact kinds in `artifact_service.py` and the meetings schemas.
- Add an outbound integration in `integration_service.py` and cover egress policy behavior.
- Change stream event formatting in `stream_adapter.py` and WebSocket or SSE tests.

## Testing

- Direct module and endpoint coverage lives under `tldw_Server_API/tests/Meetings/`.
- Dependency error mapping is covered in `tldw_Server_API/tests/API_Deps/test_meetings_db_deps_error_mapping.py`.
- Tests include DB behavior, session/template/artifact services, REST routes, SSE event streams, WebSocket streams, ingest finalization, integrations, and webhook dead-letter behavior.

## Gotchas

- Built-in templates are read-only.
- Organization and team template scopes require the endpoint authorization checks in `meetings.py`.
- Live WebSocket partial transcript events are stream events; final artifacts are generated through the finalize/commit flow.
- Webhook dispatch is subject to egress policy and dead-letter handling.
