# Characters Full-Stack Findings Remediation Design

**Date:** 2026-04-09

**Goal:** Fix the approved Character audit findings and improvements across backend correctness, frontend behavior, frontend workspace tooling, and regression coverage, then verify the fixes with the same targeted validation slices that informed the review.

## Problem Statement

The completed Character full-stack delta review identified five live implementation issues plus one enabling infrastructure gap:

- backend manual memory extraction authorizes against `user_id` even though conversations are stored with `client_id`
- backend streamed assistant persistence can skip hard validation when internal quota/count checks fail
- frontend Character handoff payloads are narrower than the fields downstream consumers already read before hydration
- frontend quick-chat promotion seeds `serverChatId` before assistant identity metadata
- the Character journey E2E does not actually validate Character selection or prompt propagation
- the prescribed frontend Vitest and Playwright command paths do not currently resolve through usable repo-local tooling in this checkout

These findings should be remediated as one coordinated effort because the frontend behavior fixes and their regression coverage depend on making the workspace toolchain runnable first.

## Scope

### In Scope

- normalize frontend workspace tooling so the prescribed Character Vitest and Playwright commands execute against repo-local tooling
- fix manual Character memory-extraction ownership validation
- change streamed Character assistant persistence to save once, then return `503` on internal quota/count-check failure without duplicating the assistant reply on retry
- update Character streamed-persist callers so a saved-but-degraded `503` does not trigger duplicate fallback persistence
- widen Character handoff payloads so existing pre-hydration consumers receive the fields they already read
- seed assistant identity metadata during quick-chat promotion before navigation when the data is already available
- add backend, frontend, and browser regressions for the reviewed findings
- rerun the targeted review validation slices after implementation

### Out of Scope

- unrelated Character UI redesign or workflow changes
- broad chat-storage architecture changes outside the streamed Character persist path
- generic cross-module deduplication frameworks
- refactoring unrelated dirty-worktree files

## Approved Decisions

### Delivery Order

Use a staged verification-first sequence:

1. repair frontend workspace/tool resolution
2. fix backend correctness
3. fix frontend handoff behavior
4. add and run regression coverage

This order reduces risk because the frontend fixes should be verified with the same commands that previously failed for environment reasons.

### Frontend Behavior Strategy

Preserve fast navigation. Do not delay route changes just to wait for full Character hydration.

Instead:

- seed richer selected-character state immediately
- seed quick-chat assistant metadata before navigation when already available
- keep later hydration as the fallback path, not the primary source of truth

### Streamed Persist Failure Strategy

For internal quota/count-check failures in the streamed assistant persist path:

- save the assistant reply
- return `503` to signal degraded validation
- prevent duplicate assistant persistence on retry of the same streamed reply

This is intentionally different from the direct-send path, which still fails closed before persistence.

## Architecture

The remediation is split into four workstreams.

### Workstream 1: Frontend Workspace Tooling

Repair the shared `apps/` frontend workspace so the existing commands already named in the review execute against repo-local tools:

- `cd apps/packages/ui && bun run test -- ...`
- `cd apps/tldw-frontend && bun run e2e:pw -- ...`

The fix should address command resolution, workspace dependency visibility, and runner invocation so the checkout does not fall back to broken or non-repo binaries.

The target is broader than the Character module alone: the shared frontend workspace install should be normalized so sibling frontend test commands resolve through repo-local tooling too, even if the implementation validates only the Character slices immediately.

### Workstream 2: Backend Correctness

#### Manual Memory Extraction Ownership

Align the memory-extraction endpoint with the ownership field actually stored and surfaced by conversation persistence. The endpoint should:

- accept chats owned by the current user via stored conversation ownership
- reject foreign chats
- preserve the rest of the extraction flow unchanged

#### Streamed Persist Save-Then-503

Keep assistant persistence in the Character streamed persist path, but treat internal quota/count-check failures as degraded validation rather than as permission to silently continue.

Required behavior:

- assistant reply persists once
- response returns `503`
- retry of the same streamed reply does not write a duplicate assistant message
- the degraded `503` exposes a machine-detectable saved outcome in error details so Character clients can distinguish "saved but validation degraded" from "not saved"
- Character callers that currently fall back to `addChatMessage()` treat that saved-degraded outcome as terminal and do not perform duplicate fallback persistence

### Workstream 3: Frontend Handoff Correctness

#### Selected-Character Payload Fidelity

The shared Character handoff payload should preserve the fields current pre-hydration consumers already read, including:

- greeting variants
- `extensions`
- image-related fields
- other existing Character identity fields already present on the source record

The purpose is not to create a new canonical frontend Character shape. It is to stop handing downstream consumers a knowingly weaker temporary object than the data already available at the handoff point.

Implement this through one shared field-preserving builder/helper used by both normal Character selection and quick-chat promotion, so the handoff contract cannot drift immediately across duplicate mappers.

#### Quick-Chat Assistant Metadata Seeding

Quick-chat promotion should set:

- `serverChatCharacterId`
- `serverChatAssistantKind`
- `serverChatAssistantId`

before navigation whenever the promoted state already has enough information to do so.

Later `getChat()`-based hydration remains the fallback for incomplete or stale state.

Pre-seeding must not mark server-chat metadata as fully loaded by itself. Keep `serverChatMetaLoaded` false unless the promoted state is already equivalent to authoritative server-chat metadata, so the later `getChat()` hydration path can still correct stale or incomplete assistant identity.

### Workstream 4: Regression Coverage

Add the minimal targeted regressions that directly encode the reviewed findings:

- backend endpoint regression for memory-extraction ownership
- backend persist regression for save-once, `503`, and no duplicate on retry
- frontend streamed-persist regression for saved-degraded `503` handling without `addChatMessage()` duplication
- frontend regression for Character handoff payload richness
- frontend regression for quick-chat metadata seeding before navigation
- browser journey regression for Character selection and prompt propagation into `/chat/completions`

## Data and Idempotency Strategy

Duplicate suppression for streamed persist should remain narrow and local to the Character persist endpoint.

Recommended contract:

- derive or reuse a stable per-reply persist identity
- scope that identity to the same conversation and same assistant reply
- store the identity in persisted assistant-message metadata
- on retry, detect an existing assistant message for the same persist identity and return the existing logical outcome instead of writing a second assistant reply

Because the current streamed-persist request schema does not carry a dedicated idempotency key, server-side deterministic fingerprinting should be the default identity source.

Fingerprint inputs should include the smallest stable fields that identify the same streamed reply:

- conversation id
- parent/user message id when present
- speaker Character identity when present
- normalized assistant reply payload

Identity source preference:

1. derive the deterministic server-side fingerprint above
2. reuse an existing stable per-turn or per-message reference only if the implementation confirms the request already carries one consistently across retries

This should be narrow enough to avoid blocking legitimate repeated assistant turns while still preventing duplicates when the initial save succeeded but the endpoint responded with `503`.

## Testing Strategy

### Frontend Tooling Verification

First, make these exact commands runnable in this checkout:

```bash
cd apps/packages/ui && bun run test -- \
  src/components/Option/Characters/__tests__/Manager.first-use.test.tsx \
  src/components/Option/Characters/__tests__/Manager.crossFeatureStage1.test.tsx \
  src/components/Option/Characters/__tests__/CharacterGalleryCard.test.tsx \
  src/components/Option/Characters/__tests__/import-state-model.test.ts \
  src/components/Option/Characters/__tests__/search-utils.test.ts \
  src/services/__tests__/tldw-api-client.characters-list-all.test.ts \
  src/services/__tests__/tldw-api-client.characters-delete.test.ts \
  src/hooks/__tests__/useCharacterGreeting.test.tsx \
  src/hooks/__tests__/useServerChatLoader.test.ts \
  src/utils/__tests__/character-greetings.test.ts \
  src/utils/__tests__/character-mood.test.ts \
  src/utils/__tests__/default-character-preference.test.ts \
  src/utils/__tests__/characters-route.test.ts \
  --maxWorkers=1
```

```bash
cd apps/tldw-frontend && bun run e2e:pw -- \
  e2e/workflows/tier-2-features/characters.spec.ts \
  e2e/workflows/journeys/character-chat.spec.ts \
  --reporter=line
```

### Backend Regression Verification

After backend fixes:

- run focused backend tests for the new endpoint and streamed-persist behaviors
- rerun the previously reviewed backend slices to confirm no local regression

### Frontend Regression Verification

After frontend fixes:

- add targeted unit/integration tests for payload fidelity, quick-chat metadata seeding, and saved-degraded streamed-persist handling
- rerun the prescribed Character frontend Vitest slice

### Browser Verification

After E2E changes:

- rerun the Character Playwright slice
- verify the journey test now asserts actual Character selection and `/chat/completions` prompt propagation

### Security and Completion Verification

Before completion:

- run Bandit on touched Python paths from the project virtual environment
- rerun the affected validation commands and capture exact outcomes

## Risks and Mitigations

### Risk: Toolchain repair bleeds into unrelated frontend setup

Mitigation:

- keep the goal narrowly tied to repo-local command resolution for the shared `apps/` workspace
- prefer the smallest install/config change that restores workspace-local Vitest and Playwright resolution

### Risk: Persist duplicate suppression blocks legitimate repeated turns

Mitigation:

- scope dedupe by conversation and stable reply identity
- apply it only to the streamed Character persist path
- cover retry and non-retry cases in tests

### Risk: Richer handoff payloads drift from canonical Character normalization

Mitigation:

- keep the handoff change additive and field-preserving
- avoid inventing new semantics
- rely on later hydration to remain canonical

### Risk: Fast-navigation quick-chat fix still misses edge timing

Mitigation:

- seed metadata before navigation when the data is already present
- keep existing hydration paths intact
- verify with targeted tests and the Character journey E2E

## Success Criteria

The remediation is complete when all of the following are true:

- the shared frontend workspace resolves the prescribed Character Vitest and Playwright commands through repo-local tooling
- manual memory extraction authorizes owned chats correctly
- streamed Character persist saves once, returns `503` on internal quota/count-check failure, and does not duplicate on retry
- Character streamed-persist callers recognize the saved-degraded `503` outcome and do not fall back to duplicate `addChatMessage()` persistence
- Character handoff payloads include the fields already consumed before hydration
- quick-chat promotion seeds assistant metadata before navigation when possible
- backend, frontend, and browser regressions exist for the reviewed findings
- the previously prescribed backend/frontend validation slices run successfully or fail only for clearly unrelated reasons

## Implementation Notes

- keep commits scoped by logical workstream where practical
- do not touch unrelated dirty-worktree files
- if dependency installation or lockfile changes are required, isolate them and verify they are intentional
- preserve existing user-facing navigation speed while improving first-paint correctness
