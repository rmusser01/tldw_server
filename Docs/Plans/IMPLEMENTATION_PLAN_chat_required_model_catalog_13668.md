# Required model catalog — TASK-13668

Design: Docs/Design/2026-10-13-chat-required-model-catalog.md. ADR assessment: no new durable rule; ADR-049 ownership remains.

## Stage 1: Reproduce
**Goal**: Capture the unopened-picker and pending-catalog failures.
**Success Criteria**: The Form observer lacks connected activation before engagement; loading is distinguishable from unavailable. Parent observers remain explicitly acknowledged.
**Tests**: Actual Form query options and queued/ordinary held-loading controls.
**Status**: Complete

## Stage 2: Repair dependency
**Goal**: Query required models after connection and preserve guarded admission.
**Success Criteria**: Picker visibility is irrelevant; loading never dispatches; ready and frozen paths remain available.
**Tests**: Native/WebUI connected regressions and source checks.
**Status**: Complete

## Stage 3: Qualify browser
**Goal**: Verify cold first Send and streaming on the owned production build.
**Success Criteria**: Actual mobile and streaming workflows pass with preserved canonical records.
**Tests**: Real backend, strict dispatch/settlement, immutable baseline.
**Status**: In Progress

## Stage 4: Review and deliver
**Goal**: Review and commit the bounded root repair.
**Success Criteria**: Independent source review clear, canonical evidence current, normal hooks pass.
**Tests**: Exact staging; whole-PR review/rebase/merge remain separate.
**Status**: Not Started
