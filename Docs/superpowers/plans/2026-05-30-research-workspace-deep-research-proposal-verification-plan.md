# Research Workspace Deep Research Proposal Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show imported Deep Research verification summaries beside compatible Research Proposal Pack sections without mutating proposal content, source coverage, or review checklist state.

**Architecture:** Add a small pure helper that finds the matching imported Deep Research bundle for a proposal artifact, normalizes bounded verification metadata, and splits proposal markdown into display sections with conservative section annotations. Wire that helper into the generic artifact View modal only for `research_proposal_pack` artifacts, leaving all other artifact viewers unchanged.

**Tech Stack:** TypeScript, React, Vitest, Testing Library, existing Research Workspace StudioPane and artifact metadata contracts.

---

## File Structure

- Create `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/proposal-deep-research-verification.ts`
  - Pure section parsing and Deep Research import compatibility helpers.
  - No React imports; safe to unit test directly.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/ArtifactModalContent.tsx`
  - Add a proposal-specific viewer component that renders markdown sections with optional verification companions.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx`
  - Find the matching Deep Research import when viewing proposal artifacts and route the modal through the proposal viewer.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx`
  - Add RED helper and UI tests for matching, non-matching, and modal section rendering.
- Modify `backlog/tasks/task-574 - Show-Deep-Research-verification-in-proposal-sections.md`
  - Record acceptance criteria, plan link, verification, and final closeout.

## Stage 1: Pure Verification Contract

**Goal:** Define the strict compatibility and bounded summary behavior.

**Success Criteria:** Tests prove only Deep Research bundle imports whose `data.deepResearch.sourceArtifact.id` matches the proposal are considered compatible, and the helper exposes run ID, counts, unresolved questions, contradictions, and source trust.

**Tests:** Focused Vitest helper tests in `StudioPane.literature-workproducts.test.tsx`.

**Status:** Complete

## Stage 2: Proposal Section Mapping

**Goal:** Split proposal markdown into displayable sections and attach conservative verification summaries to relevant sections.

**Success Criteria:** Tests prove proposal sections remain readable, verification appears beside Source Audit/Literature Overview/Proposed Hypothesis/Methodology, and unrelated imports produce no section companions.

**Tests:** Focused Vitest helper tests.

**Status:** Complete

## Stage 3: Modal UI Integration

**Goal:** Display section-level verification companions in the View modal for compatible proposal artifacts.

**Success Criteria:** Testing Library coverage proves the modal shows Deep Research verification beside proposal sections and preserves sourceCoverage/reviewChecklist artifacts unchanged.

**Tests:** Focused StudioPane UI tests.

**Status:** Complete

## Stage 4: Verification And Closeout

**Goal:** Run focused tests, type-check if feasible, diff check, Bandit applicability, and update TASK-574.

**Success Criteria:** Verification results are recorded in the task and the branch is ready for PR.

**Tests:** Focused Vitest suite, TypeScript check, `git diff --check`; Bandit skip rationale for frontend-only TypeScript/backlog changes.

**Status:** Complete
