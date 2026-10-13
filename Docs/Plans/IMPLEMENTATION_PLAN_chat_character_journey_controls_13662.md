# Character journey controls — TASK-13662

ADR required:no; E2E navigation uses existing disclosure/button roles, without product or persistence change.

## Stage 1: Reproduce
**Goal**: Confirm old helper misses collapsed character controls.
**Success Criteria**: Source-derived old class fails trigger on the owned current Casual page; original CI traces explain other7navigation failures.
**Tests**: Owned live old helper failure and critical job115425762846.
**Status**: Complete

## Stage 2: Repair shared navigation
**Goal**: Reveal actual Casual/Pro controls and current setup/context actions.
**Success Criteria**: No forcedPro/storage injection/reload and no lost selection/setup/readiness/geometry checks.
**Tests**: Shared page object and phase6 helper checks.
**Status**: Complete

## Stage 3: Qualify
**Goal**: Verify current helper behavior and static/source gates.
**Success Criteria**: Owned page selection and viewport controls work, source review clear, runner boundaries explicit.
**Tests**: Desktop/tablet/mobile actual controls, unit boundary tests and critical test discovery.
**Status**: In Progress
