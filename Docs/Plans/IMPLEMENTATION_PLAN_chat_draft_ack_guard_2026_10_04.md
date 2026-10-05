# Chat draft acknowledgement guard

**Task/spec:** TASK-13454; approved Chat UX Stage 1. Full conversation switch/bind and source recovery remain required follow-ups.
**Baseline:** Latest origin/dev `ba553fdc51a16ba04330754671f0c2ee3398e17f`, integrated after reviewed character setup `d2763dfe92`.
**Goal:** A delayed send completion cannot remove a newer unsent text draft or cross to another draft key.
**Architecture:** Reuse `useComposerText`'s existing optimistic attempt token and latest committed message ref. Fence `useDraftPersistence.clearDraft` with its current key/owner identity. Pass the acknowledged token at Sidepanel's existing afterSend boundary and clear only acknowledged success. Explicit current-owner clear and synchronous queue paths keep their contract. No new draft store/API/dependency. This protects text retention; context/file recovery and optimistic failure restoration remain separately scoped.
**ADR assessment:** No new ADR; existing account/tab isolation and draft bucket stay authoritative.

## Stage 1: Reproduce delayed draft removal
**Goal:** Establish the existing composer/registry baseline and add red regressions.
**Success Criteria:** A held submission followed by NEXT DRAFT and a stale key callback reproduce persisted text loss.
**Tests:** Existing real-registry native-tab composer suite; retain account and tab isolation suites.
**Status:** Complete

## Stage 2: Guard the shared clear boundary
**Goal:** Clear only the current draft and acknowledged optimistic attempt.
**Success Criteria:** Next text and its pending write survive a late callback; old-key clear leaves retained source records untouched; normal successful clear still works.
**Tests:** The red regressions plus existing composer, privacy and persistence behavior.
**Status:** Complete

## Stage 3: Verify and record
**Goal:** Reviewed incremental commit with maintained compatibility.
**Success Criteria:** Relevant tests and typecheck pass; no added touched lint warnings/errors; two existing independent reviewers find no major unresolved issue; task notes and plan updated.
**Tests:** Affected unit/component contracts, WebUI typecheck, baseline-relative lint, diff check, canonical task normalization. Bandit not applicable to TS/docs-only scope.
**Status:** Complete

## Current evidence

Existing four-suite baseline: 66 passing. Two real-registry regressions fail before the guards because retained text restores empty; six affected suites pass after the changes (82 tests), including queue and prompt-assist source contracts. WebUI typecheck passes. Both independent reviewers found no Critical/Important issue in this bounded precursor. Touched lint adds no errors/warnings; the Sidepanel retains its 34 baseline warnings. Diff check and canonical task normalization pass. Bandit is not applicable to TS/docs-only changes. Character preflight plan is archived in d2763dfe92 and removed by this unit. No per-conversation key migration, bind, fresh draft handle, or attachment/context recovery is claimed by this change.
