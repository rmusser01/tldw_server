# Research Mermaid Artifact Preview Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Research Workspace mind-map artifact modals use the shared Mermaid diagram preview/action surface already used by assistant-facing chat Mermaid blocks.

**Architecture:** Keep Mermaid detection and raw-output fallback local to the Research Workspace artifact viewer, but delegate valid Mermaid rendering and actions to `MermaidDiagramBlock`. Do not change user-message markdown rendering or chat artifact behavior in this slice.

**Tech Stack:** React, Research Workspace artifact modal components, shared Mermaid UI components, Vitest.

**Backlog Task:** TASK-2278

---

## Boundaries

- Only assistant/generated Research Workspace mind-map artifact modal content is in scope.
- User messages remain unchanged.
- The existing raw fallback for non-Mermaid mind-map output remains unchanged.
- The modal should reuse shared Mermaid preview/copy/SVG-download behavior instead of local zoom/export controls.

## Files

- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/ArtifactModalContent.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx`
- Modify: `backlog/tasks/task-2278 - Unify-Research-Workspace-Mermaid-artifact-modal-preview.md`

## Task 1: Capture Shared Modal Contract In Tests

- [x] Update the Research Workspace stage 2 test mock so `MermaidDiagramBlock` can be asserted directly.
- [x] Change the fenced mind-map modal test to expect shared Mermaid preview/copy/SVG controls.
- [x] Assert the modal does not expose the chat artifact action or legacy modal-only PNG/SVG export labels.
- [x] Run the focused test and confirm it fails because the modal still renders bare `Mermaid`.

Result: `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx -t "renders mind map diagrams from fenced mermaid content"` failed with `Unable to find an element by: [data-testid="research-shared-mermaid-block"]`, confirming the test catches the old bare Mermaid path.

## Task 2: Reuse Shared Mermaid Diagram Block

- [x] Replace the valid Mermaid branch in `MindMapArtifactViewer` with `MermaidDiagramBlock`.
- [x] Pass `enableArtifactAction={false}` so the modal does not create a nested chat artifact action.
- [x] Remove now-unused local zoom/export state, refs, and icon imports.
- [x] Keep the raw non-Mermaid fallback branch intact.
- [x] Run the focused mind-map modal tests and confirm they pass.

Result: `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx -t "renders mind map diagrams from fenced mermaid content"` passed after the modal delegated valid Mermaid content to `MermaidDiagramBlock`.

## Task 3: Verify And Close

- [x] Run the Research Workspace focused test file or focused mind-map pattern.
- [x] Run shared Mermaid component tests to guard the reused action surface.
- [x] Run `git diff --check`.
- [x] Run UI type-check and record any existing baseline failures separately from this slice.
- [x] Record Bandit as not applicable because only TypeScript/React/docs/Backlog files are touched.
- [x] Update TASK-2278 with verification results, final summary, and PR link.

Verification:

```bash
bunx vitest run src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx -t "mind map"
```

Result: 1 file, 4 tests passed.

```bash
bunx vitest run src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx
```

Result: 1 file, 26 tests passed.

```bash
bunx vitest run src/components/Common/__tests__/Mermaid.test.tsx src/components/Common/__tests__/MermaidDiagramBlock.test.tsx src/components/Common/__tests__/MermaidPreviewDialog.test.tsx
```

Result: 3 files, 27 tests passed.

```bash
git diff --check
```

Result: passed.

```bash
env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit
```

Result: failed on existing unrelated KnowledgeQA fixture type errors in `KnowledgeQALayout.behavior.test.tsx` and `knowledgeQaStateFixtures.ts`; no diagnostics were reported for touched Research Workspace or Mermaid files.

Bandit: not applicable for this frontend-only TypeScript/React/docs/Backlog slice.

Pull Request: https://github.com/rmusser01/tldw_server/pull/2293

## PR Review Follow-Up

- [x] Rebased onto latest `origin/dev` after PR #2294 merged.
- [x] Addressed CodeRabbit prop-contract comment by removing the now-unused `title` prop from `MindMapArtifactViewer` and its call site.
- [x] Reran focused Research Workspace mind-map tests, shared Mermaid tests, `git diff --check`, and higher-heap UI type-check.

Review verification:

```bash
bunx vitest run src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx -t "mind map"
```

Result: 1 file, 4 tests passed.

```bash
bunx vitest run src/components/Common/__tests__/Mermaid.test.tsx src/components/Common/__tests__/MermaidDiagramBlock.test.tsx src/components/Common/__tests__/MermaidPreviewDialog.test.tsx
```

Result: 3 files, 27 tests passed.

```bash
git diff --check
```

Result: passed.

```bash
env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit
```

Result: failed on the existing unrelated KnowledgeQA fixture type errors in `KnowledgeQALayout.behavior.test.tsx` and `knowledgeQaStateFixtures.ts`; no diagnostics were reported for touched Research Workspace or Mermaid files.
