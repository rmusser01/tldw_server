## Stage 1: Buddy/Persona Context Lock
**Goal**: Ground the slice in the merged Persona Buddy and persona visual-pack implementation instead of adjacent VN surfaces.
**Success Criteria**: Branch uses a clean `origin/dev` worktree, tracks GitHub issue #1410 and Backlog TASK-158, and targets the floating Buddy/Persona surface.
**Tests**: Read-only inspection of Buddy shell, Persona Garden Visuals tab, persona visual service, and MCP module.
**Status**: Complete

## Stage 2: Red Buddy Shell Coverage
**Goal**: Capture the missing direct entry point from floating Persona Buddy into the visual-pack workflow.
**Success Criteria**: Focused Buddy shell test expects an active persona popover action linking to `/persona?persona_id=<id>&tab=visuals`.
**Tests**: `bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx`
**Status**: Complete

## Stage 3: Buddy Visuals Entry Point
**Goal**: Add the minimal Buddy popover/action wiring that opens the existing Visuals tab for the active persona.
**Success Criteria**: The popover preserves the active persona id, works with and without an active visual pack, and leaves existing rendering/fallback behavior intact.
**Tests**: Focused Buddy shell tests plus related persona visual state/rendering tests as needed.
**Status**: Complete

## Stage 4: Verification and Closeout
**Goal**: Verify the direct Buddy-to-Visuals path and record completion evidence.
**Success Criteria**: Focused frontend tests pass, Backlog TASK-158 is updated with verification/final summary, and the branch has a focused commit ready for PR.
**Tests**: Focused Vitest command; Bandit documented as not applicable if only frontend/task/plan files are touched.
**Status**: Complete
