# ACP Frontend UX Implementation Plan

## Stage 1: Current UX Contract
**Goal**: Ground the #1473 UI work in the existing Agent Tasks, ACP Playground, Agent Registry, and ACP service contracts.
**Success Criteria**: Identify the frontend seams for shared connection/auth handling, setup readiness, and task run drill-through without inventing parallel backend APIs.
**Tests**: Existing focused connection tests for Agent Tasks, ACP Playground, and Agent Registry.
**Status**: Complete

## Stage 2: Shared Readiness And Setup State
**Goal**: Give first-time users actionable ACP setup state in the task execution surface.
**Success Criteria**: Agent Tasks can explain missing orchestration routes, missing ACP health, runner/agent/API-key setup gaps, and route users to Registry or Playground diagnostics.
**Tests**: Add failing Agent Tasks frontend tests for setup/readiness states and shared auth transport.
**Status**: Complete

## Stage 3: Run/Review Drill-Through
**Goal**: Let regular users inspect task runs, session diagnostics, artifacts, and failures from Agent Tasks without copying IDs.
**Success Criteria**: Task cards expose a detail action that fetches task detail, shows run/review history, failure context, session IDs, and ACP diagnostic/artifact/audit links.
**Tests**: Add failing Agent Tasks frontend tests for task detail fetch and diagnostics rendering.
**Status**: Complete

## Stage 4: Cross-Surface Navigation And Coverage
**Goal**: Connect Agent Registry, ACP Playground, and Agent Tasks into a coherent setup/run/diagnose path.
**Success Criteria**: Registry launch links, Agent Tasks setup links, and Playground health handling use the same connection assumptions; focused frontend and E2E coverage document the main path.
**Tests**: Focused Vitest suite plus targeted Playwright coverage where feasible.
**Status**: Complete
