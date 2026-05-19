# Characters Full-Stack Delta Review Design

Date: 2026-04-08
Topic: Staged full-stack Characters review for `tldw_server`

## Goal

Produce an evidence-based staged review of the Characters module across backend
and frontend surfaces that identifies:

- net-new bugs and edge-case failures
- still-open regressions previously thought to be remediated
- backend or frontend contract drift
- maintainability and performance risks
- missing or misleading automated tests

The review is intentionally delta-oriented. It should not re-report historical
findings that were already documented and remediated unless the current code
still appears affected.

## Scope

This review covers the Character surface across both backend and frontend,
organized into separate backend and frontend stages plus a cross-layer
synthesis.

### Backend in scope

- `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- `tldw_Server_API/app/api/v1/endpoints/character_messages.py`
- `tldw_Server_API/app/api/v1/schemas/character_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/character_memory_schemas.py`
- `tldw_Server_API/app/core/Character_Chat/Character_Chat_Lib_facade.py`
- `tldw_Server_API/app/core/Character_Chat/character_limits.py`
- `tldw_Server_API/app/core/Character_Chat/character_rate_limiter.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_db.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_io.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_chat.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_validation.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_memory_extraction.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_templates.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_prompt_presets.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_generation_presets.py`
- `tldw_Server_API/app/core/Character_Chat/world_book_manager.py`
- `tldw_Server_API/app/core/Chat/chat_characters.py`
- related Character retrieval, exemplar, and world-book logic touched by the
  current Character API and chat flows
- backend Character tests under `tldw_Server_API/tests/Characters/`,
  `tldw_Server_API/tests/Character_Chat*`, `tldw_Server_API/tests/ChaChaNotesDB/`,
  and nearby integration/property/e2e suites where Character behavior is
  exercised
- any additional backend module directly imported by the in-scope Character
  entry points when that dependency materially affects Character behavior

### Frontend in scope

- `apps/tldw-frontend/pages/characters.tsx`
- `apps/tldw-frontend/pages/settings/characters.tsx`
- `apps/packages/ui/src/routes/option-characters.tsx`
- `apps/packages/ui/src/components/Option/Characters/`
- `apps/packages/ui/src/components/Common/CharacterSelect.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx`
- `apps/packages/ui/src/hooks/useSelectedCharacter.ts`
- `apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts`
- `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts`
- `apps/packages/ui/src/hooks/chat/useSelectServerChat.ts`
- `apps/packages/ui/src/utils/selected-character-storage.ts`
- `apps/packages/ui/src/utils/characters-route.ts`
- `apps/packages/ui/src/utils/character-greetings.ts`
- `apps/packages/ui/src/utils/character-mood.ts`
- `apps/packages/ui/src/utils/default-character-preference.ts`
- `apps/packages/ui/src/utils/character-export.ts`
- nearby Character-related selection, storage, server-chat loader, greeting, and
  workspace hooks/utilities
- `apps/packages/ui/src/services/tldw/domains/characters.ts`
- Character-focused frontend tests, including unit, integration, and e2e
  coverage where current Character behavior is exercised
- any additional frontend hook, utility, or adapter directly imported by the
  in-scope Character entry points when that dependency materially affects
  Character behavior

## Non-Goals

This review does not cover:

- implementing fixes during the audit
- broad unrelated chat or UI subsystems that do not materially depend on
  Character behavior
- re-documenting historical Character findings that are already covered and no
  longer appear live
- purely visual or stylistic frontend commentary unless it indicates a state,
  correctness, or contract problem

## Baseline Artifacts

Earlier Character review and remediation artifacts are the baseline for deciding
whether a finding is net-new or a regression. The audit should consult them
before claiming novelty.

Primary baseline artifacts:

- `Docs/superpowers/specs/2026-03-23-characters-backend-review-design.md`
- `Docs/superpowers/plans/2026-03-23-characters-backend-sequential-review.md`
- `Docs/superpowers/specs/2026-04-07-characters-backend-remediation-design.md`
- `Docs/superpowers/specs/2026-03-27-chat-character-menu-and-system-prompt-editor-design.md`
- `Docs/superpowers/plans/2026-03-27-chat-character-menu-and-system-prompt-editor-implementation-plan.md`
- `Docs/Product/WebUI/PRD-Characters Playground UX Improvements.md`
- related Character review and remediation docs created from that work

Rule:

- `net-new finding`: not already covered by the historical review or
  remediation docs
- `possible regression`: historically covered, but the current code still
  appears affected or has drifted back into the same failure mode
- `historical only`: already known and apparently remediated; do not report as a
  current finding except as brief context

Novelty rule:

- each reported finding must note which historical artifacts were checked before
  it was classified as net-new or a regression
- if no comparable frontend baseline artifact exists for a frontend finding,
  state that explicitly instead of implying stronger novelty confidence than the
  evidence supports

## Approaches Considered

### 1. Recommended: Staged Delta Audit With Targeted Validation

Review backend first, frontend second, using the earlier Character review and
remediation artifacts as the baseline. Run a small set of focused validation
commands only when code inspection alone is not strong enough to support a
claim.

Strengths:

- strongest fit for the user's request for net-new findings
- keeps findings attributable to either backend, frontend, or contract drift
- avoids repeating already-resolved review material
- allows stronger evidence without broad noisy suite runs

Weaknesses:

- depends on accurate interpretation of the earlier artifacts
- requires discipline to avoid drifting into a fresh full audit

### 2. Fresh Full Audit Then Dedupe

Review the entire Character surface from scratch, then subtract anything already
covered historically.

Strengths:

- highest independent confidence
- less risk of missing a resurfaced issue because of documentation bias

Weaknesses:

- slower
- likely to repeat large amounts of already-known material

### 3. Fast Hot-Path Audit

Inspect only the highest-risk flows: import/export, CRUD/versioning, chat
session coupling, frontend state sync, and API client adapters.

Strengths:

- fastest route to high-severity findings
- good if time is tightly constrained

Weaknesses:

- likely to miss lower-frequency correctness issues and test gaps
- weaker final coverage story

## Recommended Approach

Use the staged delta audit with targeted validation.

Execution order:

1. establish the prior-review baseline
2. run a backend delta pass
3. run a frontend delta pass
4. perform a cross-layer synthesis for contract drift
5. support ambiguous findings with small focused validation only when it
   materially improves confidence

## Review Architecture

The audit proceeds in three passes.

### Pass 1: Backend Delta Review

Review the current backend Character surfaces against the historical review and
remediation artifacts.

Focus:

- CRUD, restore, revert, versioning, and persistence integrity
- import/export and image handling contracts
- rate limiting, chat/session coupling, greeting or memory injection, and
  completion persistence
- exemplar, world-book, search, and related retrieval behavior
- test coverage around risky backend branches

Output:

- backend-only findings, residual risks, and coverage gaps

### Pass 2: Frontend Delta Review

Review the current frontend Character surfaces against the current backend
contracts and the historical Character artifacts.

Focus:

- Character workspace state ownership and edit flows
- selection, storage, and identity synchronization
- chat integration and server-chat restoration behavior
- API client normalization, fallback logic, caching, and invalidation
- test coverage around risky frontend branches

Output:

- frontend-only findings, residual risks, and coverage gaps

### Pass 3: Cross-Layer Synthesis

Compare backend and frontend assumptions to identify Character-specific contract
drift or boundary mismatches.

Focus:

- path or schema mismatches
- stale fallback assumptions
- inconsistent identity or version semantics
- backend responses that the frontend currently tolerates but should not rely on

Output:

- short cross-layer findings section plus a combined synthesis

## Review Slices

To keep the audit bounded and attributable, use these six slices.

### 1. Backend Lifecycle and Persistence

Primary files:

- `characters_endpoint.py`
- `character_db.py`
- `Character_Chat_Lib_facade.py`
- `ChaChaNotes_DB.py`
- related schemas and helpers

Primary questions:

- do CRUD, delete, restore, revert, and version flows preserve Character
  invariants
- are normalization and concurrency checks consistent
- do current tests still cover the highest-risk persistence branches

### 2. Backend Chat-Coupled Behavior

Primary files:

- `character_chat_sessions.py`
- `character_messages.py`
- `character_chat.py`
- `character_rate_limiter.py`
- `character_limits.py`

Primary questions:

- does Character-specific chat behavior still align with backend contracts
- are quotas, greeting injection, memory injection, and completion persistence
  coherent under current code
- are there state-coupled regressions not covered by prior remediation

### 3. Backend Retrieval Surfaces

Primary areas:

- exemplars
- world books
- Character search and retrieval fallbacks

Primary questions:

- do returned contracts still match current behavior
- are fallback paths semantically correct
- are there net-new correctness, performance, or permission risks

### 4. Frontend Character Workspace

Primary areas:

- `CharactersWorkspace`
- editor, list, dialog, inline-edit, bulk-ops, import, and version-history hooks

Primary questions:

- is state ownership clear and robust
- can edits, import, bulk actions, or version flows desynchronize UI state
- are there avoidable performance traps or missing tests

### 5. Frontend Chat Integration

Primary areas:

- `useCharacterChatMode`
- Character selection and storage hooks
- shared Character selection components
- server chat loader and synchronization helpers

Primary questions:

- can Character identity or session state drift between local UI and server
- are greeting and chat-start flows robust
- do frontend assumptions still match backend chat/session behavior

### 6. Frontend API Client and Domain Layer

Primary areas:

- `apps/packages/ui/src/services/tldw/domains/characters.ts`
- nearby Character routing and normalization utilities

Primary questions:

- are path fallbacks still necessary and correct
- is cache invalidation coherent for create, update, revert, import, and delete
- are payload normalization and error mapping defensible under current API

## Evidence and Validation Rules

The audit should be evidence-driven and conservative about claims.

For each suspected issue:

1. trace the current code path end to end
2. inspect nearby tests and relevant historical Character artifacts
3. classify the issue as `net-new`, `possible regression`, or `historical only`
4. run targeted validation only when it materially increases confidence
5. report uncertainty explicitly when a claim cannot be proven locally

### Targeted Validation Rules

Permitted validation:

- focused `pytest` invocations for backend Character behavior under review
- focused frontend test commands for the exact Character behavior under review,
  including package-scoped unit or integration tests and narrow Playwright or
  e2e runs when that is the smallest reliable validation path
- small read-oriented commands that clarify code paths, references, or coverage
- minimal execution to confirm a suspected branch mismatch or regression

Avoid:

- broad suite runs that add noise without improving confidence
- speculative findings based only on style or code shape
- presenting ambiguous concerns as confirmed bugs

## Deliverables

The audit should produce a compact review package.

### 1. Design Spec

This document defines scope, evidence rules, and deliverables before execution.

### 2. Execution Plan

A follow-on implementation plan should enumerate the stage order, files to
inspect, validation commands, and review artifacts to write.

### 3. Stage Review Docs

Write separate review artifacts for:

- `Docs/superpowers/reviews/characters-fullstack-delta/README.md`
- `Docs/superpowers/reviews/characters-fullstack-delta/YYYY-MM-DD-backend-review.md`
- `Docs/superpowers/reviews/characters-fullstack-delta/YYYY-MM-DD-frontend-review.md`
- `Docs/superpowers/reviews/characters-fullstack-delta/YYYY-MM-DD-synthesis.md`

If an existing directory or filename would create ambiguity with older Character
reviews, prefer the dated `characters-fullstack-delta` path over reusing the
older `characters-backend` location.

### 4. Final Findings Format

Findings first, ordered by severity. Each finding should include:

- severity
- type
- whether it is net-new or a regression
- impact
- concrete reasoning
- file reference(s)
- baseline artifact note
- validation note when a command or targeted test was used

Then include:

- open questions or residual risks
- coverage gaps
- improvement opportunities that are worth addressing even when they are not
  confirmed bugs
- short cross-layer contract drift section

## Severity Model

- `High`: likely bug, data loss/corruption risk, security issue, or major
  behavioral regression risk
- `Medium`: correctness edge case, contract inconsistency, meaningful
  maintainability/performance issue, or important state-synchronization risk
- `Low`: localized cleanup, smaller code smell, or minor missing-test issue

## Success Criteria

This design is successful if the eventual audit:

- covers the full Character surface requested by the user while reporting only
  net-new findings and still-open regressions
- keeps backend findings separate from frontend findings
- includes a short but concrete cross-layer synthesis
- uses targeted validation only where it materially improves confidence
- clearly separates confirmed defects, likely risks, open questions, and test
  gaps
- leaves review artifacts specific enough for later remediation work to pick up
  individual findings without redoing the investigation

## Boundaries

- This phase remains review-only; no Character fixes are implemented during the
  audit.
- Previously resolved historical findings are not re-reported just for
  completeness.
- The audit stays focused on Character-dependent behavior and does not expand
  into unrelated chat or UI subsystems.
- If a suspected issue cannot be validated confidently, it must be labeled as an
  open question or residual risk instead of a confirmed defect.
- The execution plan must assume a dirty workspace is possible and should avoid
  staging or committing unrelated files when writing review artifacts.
