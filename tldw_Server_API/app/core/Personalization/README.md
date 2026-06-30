# Personalization

Personalization provides opt-in user memory and companion-context helpers for profiles, semantic memories, explicit activity capture, knowledge-card derivation, companion goals, reflections, and follow-up prompts. The core package works with `PersonalizationDB`, Collections DB, Jobs, and adjacent Notes, Reading, Watchlists, Reminders, and Persona flows that record or consume companion context.

## Start Here

- `companion_activity.py` records explicit companion activity events from reading, notes, reminders, persona, watchlists, and manual check-ins.
- `companion_context.py` loads bounded companion context for chat or persona use.
- `companion_derivations.py` derives knowledge cards from explicit activity and active goals.
- `companion_relevance.py` ranks cards, goals, and activity rows against a live query.
- `companion_reflection_jobs.py` generates daily or weekly companion reflections through Jobs.
- `companion_lifecycle.py` purges or rebuilds scoped companion-derived state.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/personalization.py`, `tldw_Server_API/app/api/v1/endpoints/companion.py`, and `tldw_Server_API/app/api/v1/endpoints/admin/admin_personalization.py`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/personalization.py`.
- Related tests: `tldw_Server_API/tests/Personalization/`.

## Responsibilities

- Respect the global personalization feature flag and per-user opt-in state before recording companion activity.
- Store semantic memories and profile preferences through `PersonalizationDB`.
- Normalize explicit activity payloads and dedupe repeated captures.
- Derive knowledge cards such as current focus, emerging topic, stale follow-up, source focus, and active goal signals.
- Load compact companion context with bounded character and item limits.
- Rank companion candidates lexically for query-aware conversations.
- Build follow-up prompts from reflections, goals, knowledge cards, or recent activity.
- Enqueue and handle companion reflection and rebuild jobs.

## Module Map

- `companion_activity.py`: activity builders and recorders for adjacent feature flows.
- `companion_context.py`: compact context loading and formatting.
- `companion_derivations.py`: deterministic knowledge-card derivation.
- `companion_relevance.py`: lexical relevance ranking.
- `companion_followups.py`: follow-up prompt selection for companion conversations.
- `companion_proactive.py`: proactive reflection delivery policy.
- `companion_lifecycle.py`: scoped purge and rebuild helpers.
- `companion_reflection_jobs.py`: Jobs-backed reflection generation.
- `companion_reflection_jobs_worker.py`: WorkerSDK loop for companion jobs.
- `companion_user_ids.py`: stable companion storage user-id mapping.

## How It Connects

- `personalization.py` exposes opt-in, profile, preferences, memory CRUD/import/export/validate, purge, and explanation routes.
- `companion.py` exposes companion activity, check-ins, knowledge, reflections, conversation prompts, goals, purge, and rebuild routes.
- `admin/admin_personalization.py` exposes admin personalization consolidation controls.
- `app/services/companion_reflection_scheduler.py` uses APScheduler to enqueue daily and weekly companion reflection jobs.
- Notes, Reading, Watchlists, Reminders, Persona sessions, and Collections feeds call `companion_activity.py` helpers to record explicit user activity.
- Persona chat can call `load_companion_context()` to include bounded companion context in prompts.
- Companion reflection notifications are written through Collections DB.

## Architecture Notes

### Core Flow

- API dependencies and route helpers check the global feature flag and the user's opt-in/profile state before recording activity or returning companion context.
- Adjacent features record explicit activity through `companion_activity.py`; derivation helpers turn that activity into bounded knowledge cards, goals, reflections, and follow-up prompts.
- `companion_context.py` ranks and formats a compact context bundle for chat/persona callers without exposing the entire activity log.
- The reflection scheduler enqueues daily and weekly Jobs with deterministic idempotency keys; worker code generates reflections and optional notification items.
- Lifecycle helpers purge or rebuild derived scopes while keeping raw activity unless a full purge path is requested.

### State And Data

- `PersonalizationDB` owns profiles, opt-in state, semantic memories, explicit activity, derived cards, goals, and reflections.
- Collections DB stores companion reflection notifications and related delivery artifacts.
- `companion_user_ids.py` provides stable companion storage identifiers so adjacent modules do not invent user-id mappings.
- Raw activity and derived cards are separate data classes; preserve that distinction when adding new companion features.

### Security And Operations

- Disabled or non-opted-in users should receive empty context and skipped captures, not partial data.
- Logs use redacted user identifiers in companion context paths.
- Context limits are a privacy and prompt-size control; do not widen them without updating tests and caller expectations.
- Scheduler idempotency keys prevent duplicate reflection jobs across daily and weekly slots.

### Extension Checklist

- New activity source: add a builder/recorder, wire the owning endpoint, and add bridge tests.
- New derived card or reflection type: update derivation, lifecycle rebuild/purge behavior, and Personalization tests.
- New context consumer: use `load_companion_context()` and keep opt-in handling in the dependency or service boundary.

## Extension Points

- Add a new explicit activity source by adding a builder or recorder in `companion_activity.py` and wiring it from the endpoint that owns the user action.
- Change companion context limits or formatting in `companion_context.py`.
- Add derived card types in `companion_derivations.py` and update lifecycle rebuild tests.
- Adjust reflection delivery policy in `companion_proactive.py`.
- Extend reflection payloads or notification behavior in `companion_reflection_jobs.py`.
- Add companion API fields in `companion.py` and `personalization.py` schemas together.

## Testing

- Direct personalization and companion coverage lives under `tldw_Server_API/tests/Personalization/`.
- Related bridge tests include `tldw_Server_API/tests/Notifications/test_companion_reminders_activity_bridge.py` and endpoint tests that record companion activity from adjacent features.
- Dependency sanitization coverage lives in `tldw_Server_API/tests/API_Deps/test_personalization_deps_sanitization.py`.

## Gotchas

- Activity capture is skipped when personalization is disabled or the profile is not opted in.
- `companion_lifecycle.py` preserves raw activity by default while purging derived scopes.
- Derived goal and computed progress regeneration currently return zero rebuilt rows rather than recreating those states.
- `list_explanations` currently returns an empty response scaffold.
