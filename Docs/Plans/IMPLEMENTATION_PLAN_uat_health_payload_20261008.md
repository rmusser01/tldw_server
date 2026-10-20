# UAT health payload reporting repair

Related task: TASK-13260.278.18.83.45.
Actual C09/A12 Health Diagnostics reports all seven checks passing although Audio HTTP 200 contains status unhealthy. Use the existing endpoint payload status in the shared HealthStatus outcome. HTTP errors, diagnostic redaction, and status-less Chat Metrics must retain their behavior.
ADR assessment: no new ADR; this restores the existing health endpoint contracts without changing architecture.

## Stage 1: Reproduce the rendered outcome
**Goal**: Reproduce HTTP-success unhealthy/degraded payloads in the actual component.
**Success Criteria**: Causal tests fail on the current component; healthy and status-less metrics controls remain.
**Tests**: Rendered status, summary, recheck feedback and recovery using existing Vitest configs.
**Status**: Complete

## Stage 2: Honor the payload contract
**Goal**: Make one shared outcome govern rows, summary and feedback.
**Success Criteria**: Six health endpoints require ok/healthy; Chat Metrics retains its status-less response; transport failure cannot pass.
**Tests**: Complete health component suite in UI and WebUI; scoped lint and configured types.
**Status**: Complete

## Stage 3: Review and actual UAT
**Goal**: Independently review and verify the corrected health page on the owned runtime.
**Success Criteria**: Audio unhealthy is shown as failing, raw redacted details remain, and broader provider acceptance remains separate.
**Tests**: Actual page after normal owned frontend build/start; final integration hooks remain open.
**Status**: In Progress
