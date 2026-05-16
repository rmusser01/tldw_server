# Main Chat Cockpit Visual and Copy Polish Plan

Roadmap: `Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md`
Backlog: TASK-405
Scope: Main WebUI `/chat` cockpit only. No browser extension/sidebar/sidepanel work.

## Stage 1: Pin Visual-Copy Contract
**Goal**: Add focused regression coverage for the PR7 cockpit labels and hierarchy before changing UI code.
**Success Criteria**: Runtime and context rail tests assert mature, task-oriented terminology: model route, provider:model settings, MCP tools, and composition-source language.
**Tests**: Focused Vitest tests for `PlaygroundRuntimeInspector`, `PlaygroundContextRail`, and `playground-composition-preview`.
**Status**: Complete

## Stage 2: Shared Rail Visual Tokens
**Goal**: Replace duplicated left/right rail class constants with shared cockpit rail helpers and reduce repeated borders/low-value card treatment.
**Success Criteria**: Context and runtime rails import the same rail section, heading, action, inset, badge, and state styles; repeated hard-coded section/card classes are removed from the rail components.
**Tests**: Focused Vitest suite remains green; targeted ESLint passes for touched files.
**Status**: Complete

## Stage 3: Terminology and Hierarchy Polish
**Goal**: Tighten visible copy without removing controls or changing behavior.
**Success Criteria**: Right rail uses `Model route`, `Provider:model settings`, and `MCP tools`; empty-state copy explains assistant, prompt, context, and MCP status without generic settings-dump wording; duplicate session/status copy is reduced.
**Tests**: Focused Vitest suite plus real-server Playwright `/chat` cockpit spec.
**Status**: Complete

## Stage 4: Real-Server Visual QA and Closeout
**Goal**: Prove the polished cockpit still works against the running real backend on desktop and mobile.
**Success Criteria**: Real-server Playwright screenshots are produced for desktop conversation and mobile rails; no mocked route data is used; task and plan are updated with verification evidence.
**Tests**: `chat-cockpit.real-server.spec.ts`, targeted ESLint, `git diff --check`; Bandit skipped only if touched scope stays frontend TS/docs.
**Status**: Complete
