# Retry refusal details — TASK-13260.278.18.83.45
ADR assessment: no new architecture decision; existing retained recovery and error review UI.
## Stage 1: Actual failure and diagnostic red
**Goal**: Preserve the actual refusal code without exposing arbitrary provider messages.
**Success Criteria**: Known refusal is inspectable; untrusted text remains hidden.
**Tests**: Mounted real history review with controlled consumer result.
**Status**: In Progress
## Stage 2: Minimal diagnostic and qualification
**Goal**: Show only finite client refusal codes; preserve recovery/reset behavior.
**Success Criteria**: Affected configured UI/Web tests and lint pass; source independently reviewed.
**Tests**: Complete review component suite, scoped lint/types.
**Status**: Not Started
## Stage 3: Actual Retry diagnosis and acceptance
**Goal**: Identify and fix the actual runtime refusal before accepting Retry.
**Success Criteria**: Corrected real Retry preserves original input and saves reply.
**Tests**: One declared new-source-corrected real workflow, then reload.
**Status**: Not Started
