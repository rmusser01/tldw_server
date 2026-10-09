# TASK-13211: Retain Buddy loads through effect replay

ADR required: no
ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md; Docs/ADR/046-persona-live-conversation-and-voice-runtime.md
Reason: Routine loader lifecycle repair within existing contracts. Requests stay instance-owned; no persistent or global cache.

## Stage 1: Regressions
**Goal**: Reproduce duplicate network dispatch during effect replay.
**Success Criteria**: Strict Mode and replay tests fail for duplicate pack/session reads; existing stale ownership cases remain covered.
**Tests**: BuddyShellHost and usePersonaLiveControl focused tests.
**Status**: Complete

## Stage 2: Minimal repair
**Goal**: Reuse the automatic request for unchanged normalized identity in one mounted lifetime.
**Success Criteria**: Replay reattaches to pending work; settled work stays settled. Persona/surface changes, pack activation, explicit reload, disable/re-enable and actual remount still load fresh data. Old completions cannot replace newer work.
**Tests**: Focused suites, scoped lint/type checks.
**Status**: Complete

## Stage 3: Live verification
**Goal**: Repeat the controlled Fast Refresh experiment on repaired source.
**Success Criteria**: Three refreshes add no pack/session list dispatches while artwork remains loaded; controls retain their draft. Record exact source and clean up owned listeners.
**Tests**: Real browser plus source-bound request metadata.
**Status**: Complete

Historical incident: no preserved lifecycle metadata establishes the original 250 ms trigger. The reproducible effect replay mechanism is the repair target; do not rewrite that historical uncertainty.
