# UAT207 default-ID caller inventory

- Endpoint GET profiles and GET catalog invoke `_ensure_default_persona_profile`; that wrapper now delegates to the already-existing core helper.
- core/Persona/session_materialization.py also calls its ensure helper for a missing/unknown requested profile, then propagates `profile.id` into scope queries, persisted session, runtime session manager and result. This was a second actual collision path; the added cold materialization test retained PostgreSQL RED / SQLite PASS before source changes.
- PersonaSessionRequest.persona_id is required; existing session lookup and runtime contexts use persisted persona_id. Endpoint occurrences of `_DEFAULT_PERSONA_ID` outside bootstrap are missing-value fallback payloads. The actual scoped-ID policy/session test verifies the persisted ID is selected rather than that fallback.
- `_persona_info_from_profile` uses profile.id before its fallback. The unused `_persona_catalog_items` definition remains unchanged.
- Frontend sidepanel initial selection is research_assistant. usePersonaLiveSession reads catalog, validates that selected ID exists and otherwise resolves personas[0].id before starting the actual session (lines 487–517 at review). Companion mode uses the same validity check. No frontend changes are required for returned owner-specific IDs.
- No backend schema/type change or existing persona/session ID rewrite. Ordinary list/get/batch retain bound user_id and existing flags/pagination. SQL translation already handles deleted/is_active.
