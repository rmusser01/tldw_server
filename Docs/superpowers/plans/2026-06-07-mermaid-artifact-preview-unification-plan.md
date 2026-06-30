# Mermaid Artifact Preview Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Mermaid diagram artifacts in the chat artifact panel use the same preview/action surface as assistant Mermaid markdown blocks.

**Architecture:** Reuse `MermaidDiagramBlock` as the shared diagram artifact surface inside `ArtifactsPanel` instead of rendering bare `Mermaid`. Keep artifact-panel footer actions intact for existing panel-level behavior, and pass `enableArtifactAction={false}` so the panel does not show a recursive artifact-open control.

**Tech Stack:** React 18, TypeScript, Ant Design tooltips/modal mocks, Vitest/JSDOM, existing shared UI Mermaid components.

---

### Stage 1: Artifact Panel Contract Tests
**Goal**: Prove diagram artifacts use shared Mermaid block controls.
**Success Criteria**: The artifact-panel Mermaid test fails until the panel renders `MermaidDiagramBlock` controls for preview/copy/download and omits the recursive artifact action.
**Tests**: `bunx vitest run src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx`
**Status**: Complete

- [x] Add a test/mock contract for `MermaidDiagramBlock` in `ArtifactsPanel.mermaid.test.tsx`.
- [x] Assert the panel renders shared controls: `Open Mermaid preview`, `Copy Mermaid source`, and `Download Mermaid SVG`.
- [x] Assert the panel does not render `View Mermaid diagram`.
- [x] Run the focused test and confirm it fails before implementation.

### Stage 2: Panel Rendering Change
**Goal**: Replace bare panel Mermaid rendering with the shared Mermaid diagram block.
**Success Criteria**: Diagram artifacts render through `MermaidDiagramBlock` with artifact actions disabled.
**Tests**: Focused artifact-panel Mermaid test.
**Status**: Complete

- [x] Import `MermaidDiagramBlock` in `ArtifactsPanel.tsx`.
- [x] Use it for `active.kind === "diagram"` with `source={active.content}` and `enableArtifactAction={false}`.
- [x] Remove the now-unused bare `Mermaid` import.
- [x] Run the focused test and confirm it passes.

### Stage 3: Verification And Task Closeout
**Goal**: Verify the focused Mermaid/chat scope and record known skips.
**Success Criteria**: Relevant tests pass, whitespace check passes, task status records verification.
**Tests**:
- `bunx vitest run src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx src/components/Common/__tests__/MermaidDiagramBlock.test.tsx src/components/Common/__tests__/MermaidPreviewDialog.test.tsx`
- `git diff --check`
**Status**: Complete

- [x] Run the focused regression set.
- [x] Run `git diff --check`.
- [x] Run Bandit only if backend Python is touched; otherwise record frontend-only skip.
- [x] Update `TASK-2277` with results and final summary.

Verification results:
- Initial focused test failed as expected because the panel still rendered bare `Mermaid` and no `shared-mermaid-diagram-block`.
- Focused artifact-panel test passed after implementation: 1 file, 2 tests.
- Focused shared Mermaid regression set passed: 3 files, 25 tests.
- `git diff --check` passed.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --project tsconfig.json` failed in existing latest-`dev` KnowledgeQA test fixtures outside this change: `KnowledgeQALayout.behavior.test.tsx` and `knowledgeQaStateFixtures.ts`.
