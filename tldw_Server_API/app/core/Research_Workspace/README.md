# Research Workspace

Research_Workspace currently owns the capability-readiness contract for the Research Workspace WebUI. It collects lightweight health snapshots from adjacent subsystems, derives user-safe capability modes for workspace actions, and exposes them through a dedicated capability endpoint.

## Start Here

- `capabilities.py` builds and collects Research Workspace capability readiness.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/research_workspace.py`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/research_workspace_capabilities.py`.
- Related tests: `tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py` and `tldw_Server_API/tests/Research_Workspace/test_capability_endpoint.py`.

## Responsibilities

- Define the supported Research Workspace capability ids.
- Normalize adjacent health responses into `ready`, `degraded`, `unavailable`, or `unknown`.
- Map capability status to `allow`, `warn`, or `block` modes.
- Compose action readiness for source browsing, chat, text artifact generation, slides generation, audio summary, export download, and sync/share.
- Collect aggregate health, RAG health, LLM provider health, Slides DB readiness, and TTS provider availability.
- Return a TTL-bound capability response with an overall readiness status.

## Module Map

- `capabilities.py`: capability constants, readiness derivation, health collection, and composition helpers.
- `__init__.py`: package marker.

## How It Connects

- `research_workspace.py` exposes `GET /research-workspace/capabilities` with `MEDIA_READ` permission and rate limiting.
- `research_workspace_capabilities.py` defines the response contract consumed by the WebUI.
- Health collection calls existing health endpoints for aggregate API health, RAG health, and LLM provider health.
- Slides readiness checks `try_get_slides_db_for_user`; TTS readiness checks configured enabled TTS providers.
- Broader Research Workspace migration, source status, MCP hub, ACP bridge, sandbox handoff, and UI plans live outside this core package.

## Architecture Notes

### Core Flow

- The endpoint calls `collect_research_workspace_health()`, then `build_research_workspace_capabilities()` maps subsystem health into capability modes and reason codes.
- Capability ids are composed from dependency health rather than direct feature execution. The response tells the WebUI whether to allow, warn, or block an action.
- TTS readiness is derived from configured provider availability and should not initialize TTS providers during capability checks.

### State And Data

- This package does not persist workspace state; the capability response schema is the contract.
- The response includes a TTL so clients know when to refresh readiness.
- `sync_share` currently reports unknown/warn readiness until a concrete sync health collector is wired into this package.

### Security And Operations

- Health payloads must stay user-safe by excluding raw errors, filesystem paths, secrets, and provider credentials.
- Unknown health should usually warn, while unavailable required dependencies can block only the capabilities that depend on them.
- Keep endpoint permission and rate-limit checks in `research_workspace.py` when adding capability routes.

### Extension Checklist

- New capability: update constants, schema literals, builder logic, and derivation tests.
- New health dependency: add a collector, sanitize its payload, and add endpoint coverage.
- New readiness mode or reason code: update schema, WebUI contract expectations, and capability derivation tests.

## Extension Points

- Add a capability id in `RESEARCH_WORKSPACE_CAPABILITY_IDS`, the schema literal, and `build_research_workspace_capabilities()`.
- Add a health dependency by adding a collector and composing it into the relevant capability.
- Change user-facing modes or reason codes in the capability helper functions and update capability derivation tests.
- Extend endpoint authorization or rate limiting in `research_workspace.py`.

## Testing

- Capability derivation tests live in `tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py`.
- Endpoint tests live in `tldw_Server_API/tests/Research_Workspace/test_capability_endpoint.py`.

## Gotchas

- This package does not own Research Workspace persistence or migration workflows; it only owns capability readiness helpers.
- Unknown adjacent health is usually surfaced as `warn`, while unavailable required dependencies can block composed capabilities.
- `sync_share` currently returns unknown readiness because no concrete sync health collector is wired here.
