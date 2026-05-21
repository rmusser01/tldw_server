# Personalization Memory Layer PRD

Status: Draft

Owner: Persona module / Personalization

Tracking: #1918, split from #1902

Backlog: TASK-471

## Summary

Define the future Personalization Memory Layer for the Persona system: a user-owned, opt-in, reviewable memory contract that can curate cross-app personalization facts and make them available to Persona sessions, RAG, and companion context without surprising writes or hidden inheritance.

The current codebase already has several pieces of the foundation: per-user `Personalization.db`, personalization opt-in and purge endpoints, semantic memory CRUD, usage-event logging, companion knowledge/goals/activity, Persona session memory retrieval toggles, ChaCha Persona memory entries, and a legacy RAG personalization store. The remaining product gap is a single policy and data contract for automatic memory creation, merge/prune, validation, provenance, injection, and RAG biasing.

This PRD does not make broad personalization a current Persona module completion blocker. It documents the future slice moved out of `Docs/Product/Persona_Agent_Design.md`.

## Problem

Persona memory currently spans multiple concepts that are easy to conflate:

- Persona state docs describe the assistant's self-model and durable Persona state.
- Persona memory entries describe Persona/session-scoped interaction memory.
- Personalization memories describe user-owned cross-app facts, preferences, corrections, constraints, and working context.
- Companion knowledge/goals/activity provide a bounded context layer for opted-in users.
- RAG personalization has both a planned scorer and an older JSON-backed boost store.

Without a dedicated PRD, future work can accidentally add hidden long-term memory writes, double-boost RAG results, or inject inferred user facts without validation. The system needs an explicit memory lifecycle and integration policy before automatic curation is expanded.

## Goals

- Define a unified Personalization Memory contract for user-owned semantic memory.
- Keep all personalization memory opt-in, inspectable, exportable, and purgeable.
- Preserve explicit Persona `read_only` / `read_write` memory-mode boundaries.
- Add review-first handling for inferred identity, constraint, and sensitive memories.
- Support automatic candidate creation, merge, prune, and confidence updates through background jobs.
- Keep Persona state docs, Persona memory entries, companion context, and personalization memory distinct.
- Resolve the planned personalization scorer versus legacy RAG boost store before live RAG biasing.
- Require provenance and explanation metadata for every automatic memory action.
- Define staged backend-first implementation steps and validation requirements.

## Non-goals

- No Buddy animation, Buddy runtime, or visual-pack implementation.
- No design-system backlog work.
- No implementation in this PRD slice.
- No cross-user modeling or global recommendation system.
- No silent `read_write` memory inheritance from ordinary chat, Workspace defaults, or scheduled work.
- No automatic injection of unvalidated inferred identity or constraint memories.
- No broad Persona tool administration or multi-agent workflow behavior.
- No replacement for Persona state docs or Persona-owned memory entries.
- No new parallel memory database unless an implementation plan proves migration is safer than extending existing storage.

## Current Contract Evidence

- `Docs/Product/Personalization_Design.md` defines the active personalization scaffold, opt-in model, memory taxonomy direction, pending validation policy, and planned unified `memories` table.
- `tldw_Server_API/app/core/DB_Management/Personalization_DB.py` provides per-user `profiles`, `usage_events`, `semantic_memories`, `episodic_memories`, `topic_profiles`, `companion_activity_events`, `companion_knowledge_cards`, and `companion_goals`.
- `tldw_Server_API/app/api/v1/endpoints/personalization.py` exposes opt-in, purge, profile, preferences, semantic memories CRUD, memory import/export, validation, and explanations placeholder endpoints.
- `tldw_Server_API/app/core/Persona/memory_integration.py` gates Persona memory persistence on feature enablement and profile opt-in, with read modes `legacy_only`, `chacha_only`, `chacha_first_fallback_legacy` and write modes `legacy_only`, `chacha_only`, `dual_write`.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` stores Persona-scoped memory entries and chat-level `persona_memory_mode` values of `read_only` or `read_write`.
- `tldw_Server_API/app/core/Personalization/companion_context.py` loads bounded companion knowledge, goals, and explicit activity for opted-in users.
- `tldw_Server_API/app/core/Personalization/companion_activity.py` records explicit companion activity with provenance and Persona surface allowlists.
- `tldw_Server_API/app/core/Personalization/companion_reflection_jobs.py` uses Jobs-backed helpers for companion reflection and rebuild work.
- `tldw_Server_API/app/core/RAG/personalization_scorer.py` is a Stage 1 scaffold and is not yet wired into RAG.
- `tldw_Server_API/app/core/RAG/rag_service/user_personalization_store.py` remains a separate JSON-backed personalization signal and must not be applied alongside the new scorer without a clear precedence rule.
- `Docs/Product/Persona_Backed_Chat_Startup_PRD.md`, `Docs/Product/Workspace_Persona_Defaults_PRD.md`, and `Docs/Product/Persona_Scheduled_Work_PRD.md` all keep broad personalization memory out of their V1 scopes and require visible `read_write` memory configuration.

## Product Shape

The Personalization Memory Layer should be a backend contract first:

```json
{
  "id": "memory-id",
  "user_id": "current-user",
  "memory_type": "preference",
  "subtype": "style",
  "content": "Prefers concise implementation summaries.",
  "status": "confirmed",
  "source": "explicit",
  "confidence": 0.92,
  "provenance": {
    "source_event_ids": ["event-id"],
    "source_persona_session_ids": ["session-id"],
    "source_surface": "api.persona"
  },
  "controls": {
    "pinned": false,
    "hidden": false,
    "injectable": true
  },
  "timestamps": {
    "created_at": "2026-05-21T00:00:00Z",
    "last_validated": "2026-05-21T00:00:00Z",
    "expires_at": null
  }
}
```

The exact schema may differ during implementation, but V1 must preserve these concepts: type, status, source, confidence, provenance, user controls, timestamps, and an explicit injectable decision.

## Memory Taxonomy

Use the taxonomy already introduced by `Docs/Product/Personalization_Design.md`:

- `identity`: stable facts about the user.
- `preference`: style, format, tool, model, domain, or workflow preferences.
- `relational`: people, organizations, projects, and relationship context.
- `correction`: user corrections that should prevent repeated mistakes.
- `constraint`: scheduling, resource, privacy, safety, or operational constraints.
- `working`: short-lived context for active work.

V1 should migrate or map current `semantic_memories` into this taxonomy rather than keeping "semantic" as the only user-facing type. Legacy API compatibility can keep `semantic` as an alias during migration, but the canonical contract should use the taxonomy above.

## Lifecycle

Personalization memories should move through explicit states:

- `candidate`: automatically proposed and not yet trusted for injection.
- `pending_validation`: requires user review before injection.
- `confirmed`: allowed for retrieval and injection according to policy.
- `pinned`: confirmed and boosted by user action.
- `hidden`: retained but excluded from retrieval and suggestions.
- `rejected`: excluded from retrieval and future prompts.
- `merged`: superseded by another memory with provenance retained.
- `archived`: no longer active but retained for audit/export unless purged.
- `deleted`: removed by user action or purge.

Automatic extractors may create `candidate` memories from usage events, Persona sessions, companion activity, or explicit user edits. They must not directly create injectable inferred identity or constraint memories. Those go to `pending_validation`.

## Opt-in And Consent

The existing personalization profile remains the primary gate:

- If personalization is disabled globally, memory APIs return disabled behavior.
- If a user profile is not opted in, no automatic memory extraction, retrieval, RAG biasing, or companion-context enrichment should run.
- Purge must remove or disable all personalization memory, topic, companion, and embedding artifacts covered by the layer.
- Import/export must cover canonical memory fields and enough provenance for user inspection.
- Ordinary Persona-backed chat, Workspace Persona defaults, and scheduled work must not silently turn on `read_write`.

`read_write` Persona memory mode means a session may write Persona-scoped memory according to the existing Persona contract. It does not automatically grant broad cross-app Personalization Memory writes unless the user has opted in and the specific flow is configured to produce reviewable candidates.

## Review And Curation

The user-facing review model should be conservative:

- Explicit user-authored memories can become `confirmed` immediately.
- Inferred preference, relational, and working memories can be candidates, with confidence and provenance shown.
- Inferred identity and constraint memories default to `pending_validation`.
- Correction memories from direct user corrections may become confirmed but should keep source text provenance.
- Merge/prune jobs should create reviewable diffs for high-impact or pinned memories.
- Every memory should support confirm, edit, pin, hide, reject, merge, archive, and delete semantics.

For first implementation, backend state and API semantics are more important than building a full dashboard. Existing Personalization endpoints can expose the review queue and update operations incrementally.

## Persona Integration

Persona should consume personalization memory through an explicit, bounded context builder:

- Respect `use_memory_context`, `use_companion_context`, `use_persona_state_context`, and `memory_top_k` preferences.
- Never inject hidden, rejected, deleted, expired, or unvalidated sensitive inferred memories.
- Label injected context distinctly from Persona state docs and Persona-owned memory.
- Return trace-safe metadata that identifies memory IDs, types, confidence bands, and reason codes without leaking raw hidden content.
- Keep retrieval limits bounded by count and character budget.
- Prefer explicit corrections and constraints over general preferences when both are relevant.

Persona state docs remain the assistant's self-model. Personalization memories remain user-owned. Implementation should not copy broad personalization facts into Persona state docs unless a future explicit migration path is approved.

## RAG Integration

RAG integration must first reconcile the current split between:

- `personalization_scorer.py`, which is a planned scorer/context builder scaffold, and
- `rag_service/user_personalization_store.py`, which stores older JSON-backed priors.

Rules:

- Do not apply both boost systems to the same result set.
- Prefer the canonical Personalization Memory Layer when enabled and opted in.
- Emit explanations that identify which memory/topic signals affected ranking.
- Keep BM25/vector ranking usable when personalization is disabled.
- Add a kill switch for personalization-based reranking.
- Avoid using unvalidated inferred identity or constraint memories as ranking signals.

## Background Jobs

Automatic memory work should use Jobs for user-visible background processing because users need status, retries, pause/cancel, and audit visibility.

Suggested job types:

- `personalization_memory_extract`: propose candidate memories from recent events or Persona sessions.
- `personalization_memory_merge`: detect duplicates, conflicts, and stale variants.
- `personalization_memory_prune`: expire working memories and archive low-confidence stale candidates.
- `personalization_memory_embed`: maintain embedding records for confirmed and candidate memories.
- `personalization_memory_rebuild`: rebuild derived memories after import, purge, or policy changes.

Jobs should be idempotent by user, window, and source cursor. They should store reviewable summaries and source references rather than raw prompt transcripts where possible.

## API Direction

Preferred V1 API additions or changes:

- Extend memory list/create/update schemas to expose taxonomy, lifecycle status, source, confidence, provenance, validation state, and injectable flag.
- Add a review queue endpoint for `candidate` and `pending_validation` memories.
- Add merge/archive/reject actions that preserve provenance.
- Add scoped purge/export controls for personalization memories, companion knowledge, reflections, topics, and embeddings.
- Add explanation responses for Persona and RAG memory use.
- Add admin/job status surfaces for extraction, merge, prune, embed, and rebuild jobs.

Existing `/api/v1/personalization/memories` compatibility can remain, but the canonical V1 shape should not be limited to `semantic|episodic`.

## Data Model Direction

Implementation should extend `Personalization.db` toward the planned unified `memories` table instead of creating another memory store:

- Canonical memory table with taxonomy, status, source, confidence, provenance, lifecycle timestamps, and user controls.
- Source-link table for usage events, Persona sessions, companion activity, and imported records.
- Conflict/merge table or JSON field for superseded memory IDs and merge rationale.
- Optional embedding cache or Chroma references for retrieval.
- Migration path from `semantic_memories` and `episodic_memories`.

ChaCha Persona memory entries should stay Persona-scoped. They can be a source for candidate personalization memories, but they should not become the canonical personalization store.

## Privacy And Safety

- Never log secrets, raw provider payloads, full hidden memory content, or unredacted local paths in memory diagnostics.
- Provenance should use stable IDs and summaries, not raw transcript copies by default.
- Sensitive inferred memories require validation before injection.
- Purge must remove derived embeddings and job artifacts that could reconstruct memory content.
- Import must validate schema version, type, lifecycle status, source, and unsafe fields.
- Export must make source, confidence, and validation state visible.
- Memory extraction prompts and model outputs need strict schema validation and bounded text.

## Staged Delivery

### Stage 1: Contract And Migration Design

Goal: finalize canonical memory taxonomy, lifecycle, schema, and migration strategy.

Deliverables:

- Unified memory schema proposal.
- Compatibility plan for `semantic_memories` and `episodic_memories`.
- Review queue and lifecycle transition matrix.
- Provenance and explanation schema.
- RAG boost precedence decision.

### Stage 2: Backend Memory API

Goal: expose the canonical memory contract through Personalization APIs.

Deliverables:

- Memory list/detail/create/update/delete using taxonomy and lifecycle status.
- Review queue endpoints.
- Confirm/reject/hide/pin/archive/merge actions.
- Import/export and scoped purge support.
- Tests for opt-in, disabled feature behavior, and lifecycle transitions.

### Stage 3: Candidate Extraction Jobs

Goal: generate reviewable memory candidates without automatic injection.

Deliverables:

- Jobs-backed extractor for explicit events and Persona session summaries.
- Idempotency and cursor handling.
- Confidence/source classification.
- Pending-validation handling for sensitive inferred types.
- Job status, failure summaries, and retry behavior.

### Stage 4: Persona Context Integration

Goal: let Persona sessions retrieve confirmed personalization memory safely.

Deliverables:

- Bounded personalization memory context builder.
- Trace-safe injected-memory metadata.
- Tests for `read_only`, `read_write`, opt-in, disabled profile, hidden/rejected memory, and pending validation.
- Documentation distinguishing Persona state docs, Persona memory entries, companion context, and personalization memory.

### Stage 5: RAG Integration

Goal: use personalization memory and topics for explainable RAG biasing.

Deliverables:

- Resolve legacy JSON store versus canonical scorer precedence.
- Reranking or boost integration behind feature/profile gates.
- Explanation output for applied signals.
- Tests proving no double boost and disabled-mode parity with baseline ranking.

### Stage 6: Merge, Prune, And Rebuild

Goal: keep memory quality high over time.

Deliverables:

- Duplicate/conflict detection.
- Reviewable merge diffs.
- Working memory expiration.
- Low-confidence stale candidate pruning.
- Scoped rebuild from retained provenance.

## Validation Plan

- Schema and migration tests for canonical memory table and legacy semantic/episodic compatibility.
- API tests for opt-in, purge, import/export, review queue, lifecycle actions, and disabled feature responses.
- Persona tests proving memory context is gated by profile opt-in, session preferences, and validation status.
- RAG tests proving personalization signals are explainable and not double-applied.
- Background job tests for idempotency, retry, failure summaries, and cursor handling.
- Privacy tests proving hidden/rejected/deleted memories and raw sensitive provenance are not injected.
- Bandit on touched backend implementation paths when code is added.

## Risks And Mitigations

- Risk: duplicate memory systems keep diverging.
  Mitigation: choose `Personalization.db` as the canonical store and document ChaCha Persona memory as a source/consumer boundary.

- Risk: users are surprised by automatic memory writes.
  Mitigation: opt-in gates, explicit `read_write` boundaries, candidate/review states, and no silent inheritance.

- Risk: inferred sensitive facts are injected too early.
  Mitigation: pending validation for inferred identity and constraint memories.

- Risk: RAG personalization changes ranking invisibly.
  Mitigation: explanations, kill switch, and no double boost with the legacy JSON store.

- Risk: memory extraction stores too much raw content.
  Mitigation: bounded summaries, source IDs, redaction, and exportable provenance.

## Acceptance Criteria

- The memory taxonomy and lifecycle are implemented in a canonical backend contract.
- Personalization memory remains opt-in, purgeable, exportable, and inspectable.
- Automatic extraction creates candidates and validation queues before broad injection.
- Persona memory context never uses hidden, rejected, deleted, expired, or unvalidated sensitive inferred memories.
- Persona `read_write` remains explicit and visible; no Workspace, scheduled, or ordinary chat flow silently inherits it.
- RAG personalization uses one canonical signal path with explanations and a kill switch.
- Tests cover migration, API lifecycle, Persona gating, RAG no-double-boost, background jobs, and privacy boundaries.

## Open Questions

- Should memory extraction start only from explicit user actions, or also from opted-in chat/session summaries in the first implementation?
- Should the canonical memory table live beside legacy tables immediately, or should compatibility views preserve old endpoint behavior while APIs migrate?
- What confidence threshold should promote non-sensitive inferred preference memories from candidate to confirmed?
- How should conflicting memories be displayed when both are confirmed and recent?
- Which memories should be eligible for cross-surface RAG biasing versus Persona-only context?
