# Persona Module

The Persona module implements persistent assistant profiles, live Persona
sessions, state docs, exemplar retrieval, scoped tool policy, voice command
support, and Persona Buddy visual-pack services.

For user-facing workflows and curl examples, see
`Docs/User_Guides/Server/Personas_User_Guide.md`. For the visual asset contract,
see `Docs/Code_Documentation/Persona_Visual_Packs.md`.

## Purpose

Personas give tldw_server a first-class assistant identity that can persist
across sessions and surfaces. A Persona is broader than a character card: it can
carry durable state, scoped memories, tool policies, exemplar guidance, voice
commands, live-session preferences, and an optional Buddy visual pack.

The core module keeps business rules and reusable runtime helpers out of the
FastAPI router where possible. The router remains the API integration layer; the
module files here define the behavior that chat, Persona Live, visual packs, and
offline evaluation paths share.

## Responsibilities

- Materialize Persona sessions with a profile, policy rules, scope snapshot, and
  runtime preferences.
- Keep live session snapshots in process while persisting durable session data
  through `ChaChaNotes_DB`.
- Persist and retrieve Persona memory/state rows.
- Classify turns, select exemplars, and assemble bounded prompt sections.
- Evaluate Persona tool policy before tool execution.
- Validate and serve Persona Visual Pack manifests and user-owned assets.
- Queue visual generation, import-preview, import-commit, and export jobs.
- Maintain defensive dialogue-tree and runtime-explorer helpers behind feature
  flags.
- Provide redaction and trace-safety boundaries for runtime diagnostics.

## Module Map

| File or Folder | Responsibility |
| --- | --- |
| `session_materialization.py` | Creates or resumes persisted Persona sessions, ensures default profiles, builds scope snapshots, and normalizes session preferences. |
| `session_manager.py` | Process-local live session snapshots, turns, pending plans, and runtime preferences. |
| `live_control.py` | Focus/stop/list helpers for Persona Live control sessions and active Buddy target state. |
| `memory_integration.py` | Persona memory retrieval, turn persistence, tool outcome persistence, and legacy/ChaCha read-write mode handling. |
| `policy_evaluator.py` | Normalizes and evaluates Persona policy rules for MCP tool and skill actions. |
| `exemplar_turn_classifier.py` | Classifies latest turns into style, boundary, scenario, and tool-behavior retrieval needs. |
| `exemplar_retrieval.py` | Deterministically selects exemplar rows from enabled candidates. |
| `exemplar_prompt_assembly.py` | Formats selected exemplars into bounded prompt sections. |
| `exemplar_runtime.py` | Async runtime bridge used by chat and live Persona flows. |
| `exemplar_ingestion.py` | Converts transcript text into candidate exemplar rows and appends review notes. |
| `exemplar_eval_harness.py` | Offline fixture/evaluation helpers for exemplar behavior. |
| `connections.py` | Validates Persona external connection targets, templates headers/payloads, resolves secrets, and redacts headers. |
| `buddy.py` | Derives and resolves stable Persona Buddy identity summaries from profile data. |
| `visuals.py` | Validates manifest shape, animation frames, state catalog, fallbacks, and authored triggers. |
| `visual_service.py` | Creates, duplicates, activates, deactivates, and reviews visual packs, assets, and generated candidates. |
| `visual_library_service.py` | Saves and reuses user-owned visual packs through a personal library. |
| `visual_starter_catalog.py` and `visual_starter_fixtures.py` | Bundled starter pack catalog and fixture assets. |
| `visual_renderer_capabilities.py` | Renderer capability registry surfaced by `/persona/visual-renderers`. |
| `visual_jobs.py` and `visual_jobs_worker.py` | Jobs integration for candidate generation, pack export, import preview, and import commit. |
| `visual_portability/` | Archive validation, import preview, import commit, export, Codex Pet adapter, provider envelope normalization, and fingerprinting. |
| `dialogue_tree*.py` | Defensive tree exploration, context redaction, pruning, scoring, and trace serialization. |
| `runtime_explorer.py` | Optional runtime plan exploration layer, disabled by default. |
| `robustness_eval.py` | Offline robustness evaluation helpers. |
| `archetype_loader.py` | Loads bundled Persona archetype templates from configuration. |

## Runtime Lifecycle

1. A user creates or lists profiles through `/api/v1/persona/profiles` or the
   active catalog through `/api/v1/persona/catalog`.
2. The session API calls `materialize_persona_session(...)` to select a profile,
   build a scope snapshot, merge preferences, and create or resume a persisted
   session row.
3. Persona Live creates or focuses a live control session and mirrors current
   state into the process-local `SessionManager`.
4. WebSocket turns enter through `/api/v1/persona/stream`.
5. Runtime code loads the persisted session context, state docs, memory,
   exemplars, and policy rules.
6. Tool plans are proposed, evaluated, and emitted for confirmation when needed.
7. Confirmed steps are rechecked against policy and RBAC before execution.
8. Turns, tool outcomes, summaries, and compact metadata are persisted back to
   Persona memory/session storage.

The live `SessionManager` is a latency optimization, not the authority for
durable state. Persisted session rows and memory entries remain the recovery
source for API reads.

## Persistence Model

Persona persistence is handled by `CharactersRAGDB` in
`tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`.

Per-user records include:

- Persona profiles.
- Persona sessions and session preferences.
- Scope rules and policy rules.
- Persona memory entries, including state docs and connection records.
- Persona exemplars.
- Persona setup and voice analytics.
- Persona visual packs, assets, candidates, library rows, and portability job
  metadata.

Do not add raw SQL in the Persona module for new persistence behavior. Add or
reuse `ChaChaNotes_DB` methods so API routes, core helpers, tests, and future
Postgres paths stay aligned.

## API Touch Points

Primary router:

- `tldw_Server_API/app/api/v1/endpoints/persona.py`

Schemas:

- `tldw_Server_API/app/api/v1/schemas/persona.py`
- `tldw_Server_API/app/api/v1/schemas/voice_assistant_schemas.py`

Important route groups:

- Profiles and catalog: `/profiles`, `/profiles/{persona_id}`, `/catalog`
- Sessions: `/session`, `/sessions`, `/sessions/{session_id}`
- Live control: `/live/sessions`, `/live/sessions/{session_id}/focus`,
  `/live/sessions/{session_id}/stop`
- WebSocket stream: `/stream`
- State docs: `/profiles/{persona_id}/state`
- Exemplars: `/profiles/{persona_id}/exemplars`
- Scope and policy: `/profiles/{persona_id}/scope-rules`,
  `/profiles/{persona_id}/policy-rules`
- Voice commands and connections: `/profiles/{persona_id}/voice-commands`,
  `/profiles/{persona_id}/connections`
- Visual packs and library: `/visual-renderers`, `/visual-starter-packs`,
  `/visual-library`, `/profiles/{persona_id}/visual-packs`

Router registration:

- `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- `tldw_Server_API/app/api/v1/router_groups/content.py`

## Runtime Concepts

- **Profile:** top-level identity, mode, system prompt, active flag, state-doc
  default, voice defaults, and setup status.
- **State docs:** durable Markdown state fields: `soul_md`, `identity_md`, and
  `heartbeat_md`.
- **Memory:** retrieved Persona experience and persisted turn/tool outcomes.
- **Exemplars:** selected examples that shape style, boundaries, scenario
  behavior, or tool behavior.
- **Scope rules:** include/exclude rules for conversations, characters, media,
  media tags, and notes.
- **Policy rules:** allow/deny/confirmation rules for MCP tools and skills.
- **Session preferences:** persisted and process-local runtime knobs, including
  voice runtime configuration and session policy overrides.
- **Buddy summary:** derived stable visual/behavior summary projected from the
  profile for UI rendering.

Keep these layers distinct. In particular, exemplars guide output style and
boundaries; they do not grant tool authority. Policy and RBAC remain the
authority for capability use.

## Visual Packs and Buddy Runtime

Persona Visual Packs are attached to Personas and remain user-owned. The
supported activation path is manifest-backed `sprite_frames`; clients should
read `list_persona_visual_renderer_capabilities()` or
`GET /api/v1/persona/visual-renderers` before assuming a renderer is usable.

Important invariants:

- Generated candidates are review artifacts until accepted.
- Imported packs are previewed and committed through Jobs-backed portability
  flows.
- Draft, review, and archived packs do not affect the live Buddy until explicit
  activation.
- Activation validates the manifest before marking a pack active.
- Asset content is served only after user, persona, pack, and asset ownership
  checks.
- Provider envelopes and generation provenance are bounded and redaction-safe.

See `Docs/Code_Documentation/Persona_Visual_Packs.md` for the full visual-pack
contract and Codex Pet import notes.

## Security and Privacy Boundaries

- All HTTP routes require `get_request_user`; the WebSocket stream authenticates
  before accepting active behavior.
- API-key and JWT auth are both supported for the WebSocket stream.
- Persona records are scoped by authenticated user id.
- Scope and policy rules are evaluated before tool execution.
- Transcript export is gated by Persona RBAC settings. Delete-capable runtime
  scopes are exposed only when Persona RBAC allows them, then rechecked in the
  policy flow before tool execution.
- Runtime explorer inputs pass through redaction and truncation boundaries.
- Dialogue-tree traces and reports must not contain raw secrets, raw memory
  rows, raw tool output, local paths, or provider secrets.
- Connection secrets are stored as redacted memory content and responses expose
  only bounded metadata.
- Red-team and offline robustness fixtures must not write into Persona memory,
  exemplars, state docs, or chat history.

## Configuration

Main config lives under `[persona]` and `[persona.rbac]` in
`tldw_Server_API/Config_Files/config.txt`. Environment variables with uppercase
names override the matching settings where supported.

Key settings:

- `PERSONA_ENABLED` / `[persona] enabled`
- `PERSONA_DEFAULT_PERSONA` / `[persona] default_persona`
- `PERSONA_VOICE` / `[persona] voice`
- `PERSONA_STT` / `[persona] stt`
- `PERSONA_MAX_TOOL_STEPS` / `[persona] max_tool_steps`
- `PERSONA_MEMORY_READ_MODE` / `[persona] persona_memory_read_mode`
- `PERSONA_MEMORY_WRITE_MODE` / `[persona] persona_memory_write_mode`
- `PERSONA_DIALOGUE_TREE_EVAL_ENABLED`
- `PERSONA_RUNTIME_EXPLORER_ENABLED`
- `PERSONA_RUNTIME_EXPLORER_MAX_DEPTH`
- `PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING`
- `PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS`
- `PERSONA_RUNTIME_EXPLORER_TIMEOUT_MS`
- `PERSONA_RUNTIME_EXPLORER_MAX_TOKENS`
- `PERSONA_RUNTIME_EXPLORER_LLM_JUDGES_ENABLED`
- `PERSONA_DIALOGUE_TREE_TRACE_RETENTION_DAYS`
- `PERSONA_RBAC_ALLOW_EXPORT` / `[persona.rbac] allow_export`
- `PERSONA_RBAC_ALLOW_DELETE` / `[persona.rbac] allow_delete`

Runtime explorer and LLM judges are off by default. LLM judges are offline or
warning/ranking helpers and must not authorize runtime actions.

## Testing

Use the project virtual environment before running Python tests:

```bash
source .venv/bin/activate
```

Recommended targeted suites:

```bash
python -m pytest tldw_Server_API/tests/Persona/test_persona_sessions.py -q
python -m pytest tldw_Server_API/tests/Persona/test_persona_ws.py -q
python -m pytest tldw_Server_API/tests/Persona/test_persona_memory_integration.py -q
python -m pytest tldw_Server_API/tests/Persona/test_persona_policy_evaluator.py -q
python -m pytest tldw_Server_API/tests/Persona/test_persona_profiles_api.py -q
python -m pytest tldw_Server_API/tests/Persona/test_persona_visual*.py -q
python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree*.py -q
python -m pytest tldw_Server_API/tests/Chat -k persona -q
python -m pytest tldw_Server_API/tests/Character_Chat/test_persona_dialogue_tree_character_eval.py -q
```

Pick the smallest relevant subset for narrow changes. Broaden to the WebSocket,
visual, Chat, or Character Chat suites when changing shared runtime behavior.

## Extension Guidance

- Keep FastAPI route handlers thin. Put reusable logic in this module or in
  `DB_Management` when persistence is involved.
- Add DB behavior through `ChaChaNotes_DB` methods instead of ad hoc SQL.
- Preserve user isolation on every read and write.
- Treat Persona session preferences as persisted state with process-local cache
  projection, not as cache-only state.
- Keep visual renderer support behind the capability registry.
- Keep runtime explorer behavior feature-flagged and preserve disabled-mode
  WebSocket behavior.
- Store compact IDs, reasons, and metadata rather than raw exemplar text or raw
  tool output when tracing runtime decisions.
- If a new feature needs user-facing status, consider Jobs for queue visibility;
  use Scheduler only for internal orchestration where dependency handling is the
  central concern.
- Update this README and `Docs/User_Guides/Server/Personas_User_Guide.md` when
  public behavior or extension points change.

## Common Pitfalls

- Confusing Personas with Character Chat cards. Personas can seed from
  characters, but they evolve independently.
- Letting exemplars act like permissions. They are prompt guidance only.
- Treating the process-local `SessionManager` as durable storage.
- Returning raw secrets, raw tool payloads, raw memory rows, or local paths in
  traces, diagnostics, or API responses.
- Activating generated or imported visual assets without an explicit user
  activation step.
- Adding WebSocket message types without preserving existing `notice`,
  `tool_plan`, `tool_call`, `tool_result`, and `assistant_delta` behavior.
- Enabling runtime explorer features without bounded context, deterministic
  hard blockers, and policy re-evaluation before execution.
