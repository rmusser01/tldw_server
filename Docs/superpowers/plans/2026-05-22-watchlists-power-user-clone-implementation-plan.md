## Stage 1: Verify Existing Clone Surface
**Goal**: Confirm what clone behavior already exists on `/watchlists` Sources and Monitors, and narrow this slice to the remaining unsafe gap.
**Success Criteria**: Existing monitor/source clone utilities and UI actions are identified, and the implementation scope avoids duplicating already-merged work.
**Tests**: Read existing clone utility tests and relevant SourcesTab/JobsTab code paths.
**Status**: Complete

## Stage 2: Source Clone Review Form
**Goal**: Change source clone from immediate duplicate creation to a prefilled create form that preserves source rules and assignments while keeping the cloned source inactive by default.
**Success Criteria**: Clicking clone opens the source form with copied name, URL, type, tags, settings, and hidden group assignment preservation; no create API call is made until the user saves.
**Tests**: Add a focused SourcesTab clone-flow test that fails before implementation and passes after.
**Status**: Complete

## Stage 3: Preservation Regression Coverage
**Goal**: Keep monitor clone behavior covered as a paused copy and source clone payload behavior covered as a reset runtime copy.
**Success Criteria**: Existing clone utility tests remain passing, and new tab-level behavior coverage protects the source clone handoff.
**Tests**: Run SourcesTab clone-flow test, source clone utility test, and relevant monitor clone utility test.
**Status**: Complete

## Stage 4: Verification And Task Finalization
**Goal**: Verify focused frontend behavior, record results in Backlog, and commit a clean reviewable slice.
**Success Criteria**: Focused tests pass, diff is reviewed, Backlog task records outcome and any skips, and branch is ready for PR.
**Tests**: Focused Vitest commands plus `git diff --check`; Bandit skipped if only frontend/markdown files are touched.
**Status**: Complete
