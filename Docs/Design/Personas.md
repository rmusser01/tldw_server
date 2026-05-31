# Personas



https://github.com/VectorSpaceLab/general-agentic-memory
https://arxiv.org/abs/2601.10080
https://github.com/Neph0s/awesome-llm-role-playing-with-persona
https://github.com/rmusser01/Clawatar
https://www.deviantart.com/ryoma3d/art/ani-x-1220087954
https://oshikoi.io/community/mate/8969a098-ad97-4f49-9b59-cfcb5d53a65b

https://zby.github.io/commonplace/agent-memory-systems/
https://github.com/elevenyellow/handcrafted-persona-engine/blob/624a76c8b364edbf22b779f2ed19df0c5ccad53e/src/PersonaEngine/PersonaEngine.Lib/Resources/Prompts/personality.txt
https://github.com/vspeech/Qwen3-TTS-Train

## Problem statement

The platform currently supports character cards and session context, but it lacks a first-class persona model that can persist user intent, policy constraints, and scoped behavior across conversations and workflows. This creates inconsistent behavior between sessions and increases repeated setup effort for users who want stable persona-driven assistant behavior.

## Goals and non-goals

Goals:
- Define a canonical persona object that can be attached to sessions, workflows, and chat contexts.
- Support both session-scoped and persistent personas with explicit activation/deactivation rules.
- Provide guardrails for persona policy (tool/skill allow/deny and confirmation requirements).
- Preserve user ownership and privacy boundaries (per-user isolation and soft-delete semantics).

Non-goals:
- Building avatar rendering pipelines or real-time 3D editing in this phase.
- Replacing existing character card import formats; persona links to them when available.
- Shipping provider-specific prompt tuning logic in v1 (persona text is provider-agnostic).

## Design approach

Persona model:
- `persona_profiles`: identity and core persona instructions (`id`, `name`, `mode`, `system_prompt`, `is_active`).
- `persona_scope_rules`: include/exclude rules that bind personas to conversation/media/note/tag contexts.
- `persona_policy_rules`: tool/skill policy controls (allowed, confirmation required, optional per-turn limits).
- `persona_sessions`: runtime binding between a persona and an active conversation/session.

Lifecycle:
1. Create or import persona profile.
2. Attach scope and policy rules.
3. Start a persona session (session-scoped or persistent).
4. Apply persona context at request time through deterministic rule matching.
5. Close/archive session; retain history and version metadata for auditability.

Privacy and safety:
- Persona records are owned by `user_id` and must be isolated across users/tenants.
- Soft delete and version columns are retained for recovery and synchronization.
- Policy rules are evaluated before tool execution to avoid accidental capability escalation.

## User stories and use cases

- Research user: "Use my compliance persona for financial document analysis, but only in conversations tagged `risk-review`."
- Team lead: "Use a concise standup persona for meeting summaries and action-item extraction."
- Writer: "Use a creative persona for drafting, but require confirmation before any external tool action."
- Analyst: "Reuse my persistent persona across weekly workflows without re-entering prompt instructions."

## Implementation considerations

Data structures and storage:
- Persist persona tables in ChaChaNotes DB with indexes on `user_id`, `persona_id`, and status fields.
- Keep schema versioned and migration-safe with additive changes where possible.

API surface:
- Persona CRUD endpoints should align with existing auth/rate-limit patterns.
- Session endpoints should support idempotency and explicit status transitions (`active`, `paused`, `closed`, `archived`).
- Read APIs should expose effective scope/policy snapshots for debugging.

UI integration:
- Add persona selector and "effective persona" indicators in chat/workflow surfaces.
- Allow scope-rule editing with clear include/exclude affordances.
- Provide policy visibility (what tools are blocked/allowed and why).

Wake phrase support:
- Persona Live wake phrase support is manually armed per session.
- V1 uses the selected persona profile's saved `voice_chat_trigger_phrases` as wake phrases and detects them in the active browser or extension surface before sending a scoped `wake_activation` frame to `/api/v1/persona/stream`.
- V1 is not true background always-on listening. Wake listening is visible, manually armed, and only active while the Persona Live surface is open and the browser speech recognition API is available.
- Pre-wake audio is not sent to the tldw server.

Compatibility and migration:
- Existing character-card flows remain valid; personas can optionally reference `character_card_id`.
- Legacy sessions without persona links should continue operating with default behavior.
- Rollout should be feature-flagged to support staged adoption and telemetry validation.

## References

- <https://hub.vroid.com/en> - VRoid Hub examples for avatar/persona visual reference material.
- <https://github.com/uwu/vrh-deobfuscator?tab=readme-ov-file> - Tooling reference for VRH format inspection when evaluating import options.
