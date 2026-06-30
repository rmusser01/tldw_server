## Stage 1: Verify Tour Failure Surface
**Goal**: Identify why Research Workspace tour controls can launch without a visible walkthrough.
**Success Criteria**: Document whether failure is caused by store wiring, route registration, missing targets, lazy rendering, Joyride timing, or CSS/stacking.
**Tests**: Focused unit tests around tutorial target readiness and launch controls, plus live CDP/Playwright reproduction.
**Status**: Complete

## Stage 2: Repair Tour Launch And Replay
**Goal**: Make the header tour button, first-run prompt, and Settings > Replay tour open a visible, navigable tour.
**Success Criteria**: Tour starts on a visible Research Workspace target, remains navigable, and does not silently skip all steps when targets are briefly unavailable.
**Tests**: Tutorial runner tests for delayed target readiness and Research Workspace launch wiring tests.
**Status**: Complete

## Stage 3: Improve Contextual State Guidance
**Goal**: Clarify first-run, empty, processing, missing-model, failed-source, and partial-success states without adding a persistent trust banner.
**Success Criteria**: Copy states the current system state and next action in the relevant pane or modal, including local/self-hosted storage only at source-addition or recovery decision points.
**Tests**: Focused component/helper tests for empty-state and status copy.
**Status**: Complete

## Stage 4: Live Validation
**Goal**: Validate the fixed flow against a running backend and WebUI using CDP/Playwright.
**Success Criteria**: First-run empty state, tour/replay, missing model, processing source status, and failed-source copy are exercised or explicitly documented if unavailable in the local fixture.
**Tests**: CDP/Playwright walkthrough with screenshots or DOM assertions.
**Status**: Complete

## Stage 5: Finalize TASK-478.11
**Goal**: Update Backlog, run focused verification, self-review, commit, and push the task.
**Success Criteria**: Backlog task records touched files and verification results; commit contains only TASK-478.11 changes and excludes unrelated Watchlists/TASK-505 work.
**Tests**: Focused frontend tests, relevant type/test command where feasible, and Bandit skipped as frontend-only unless backend files are touched.
**Status**: Complete
