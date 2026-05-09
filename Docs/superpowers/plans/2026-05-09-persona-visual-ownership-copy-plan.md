# Persona Visual Ownership Copy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clarify Persona/Buddy visual-pack ownership, activation, import/export, and review semantics in the WebUI and docs.

**Architecture:** Keep the implementation copy/docs only. Add a compact explanatory block and stage-specific helper copy to the existing `VisualPackEditor`, cover the visible language with focused Vitest assertions, and add a docs page that repeats the same product model without changing runtime behavior.

**Tech Stack:** React, Ant Design/Tailwind utility classes already used by the editor, Vitest, Testing Library, Markdown docs.

---

## File Structure

- Modify `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
  - Add compact ownership/activation help copy near the pack selector/header.
  - Add import/export/review helper copy in existing editor sections.
- Modify `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
  - Add focused assertions for ownership, active pack, import/export, and generated-candidate review copy.
- Create `Docs/Code_Documentation/Persona_Visual_Packs.md`
  - Document the ownership model, active pack semantics, import preview/commit, export archive, and generated candidate review.
- Modify `backlog/tasks/task-189 - Document-Persona-Visuals-pack-ownership-and-activation-semantics.md`
  - Track implementation notes, verification, and final summary.

## Stage 1: WebUI Ownership Copy

**Goal:** Make the core product model visible in the Persona Visuals editor.

**Success Criteria:** Users can see that assets are user-owned, attached to one persona by default, manifest-backed, and that active pack means the Buddy renderer uses it now.

**Tests:** `VisualPackEditor.test.tsx` copy assertion.

**Status:** Not Started

- [ ] Add a failing test in `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx` that renders an active pack and expects:
  - `Assets are user-owned`
  - `attached to Garden Helper by default`
  - `stored as manifests`
  - `The active pack is the one Persona Buddy renders now`
- [ ] Run:
  `./node_modules/.bin/vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx -t "explains visual pack ownership"`
  Expected: FAIL because the copy does not exist yet.
- [ ] Update `VisualPackEditor.tsx` near the pack header/selector with the compact ownership help block.
- [ ] Re-run the focused test.
  Expected: PASS.

## Stage 2: Import/Export And Review Copy

**Goal:** Clarify staged workflows without changing behavior.

**Success Criteria:** Existing import/export/generation sections distinguish preview from commit, export from shared library publication, and generated-candidate review from immediate activation.

**Tests:** `VisualPackEditor.test.tsx` copy assertion.

**Status:** Not Started

- [ ] Add a failing test in `VisualPackEditor.test.tsx` that expects visible helper copy for:
  - import preview validates a portable archive before changing this persona
  - commit import creates or updates a reviewed pack
  - export downloads a portable archive and does not publish to a shared library
  - generated candidates stay in review until accepted
- [ ] Run:
  `./node_modules/.bin/vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx -t "clarifies import export and generated candidate review"`
  Expected: FAIL because the copy does not exist yet.
- [ ] Add the helper copy to the existing import/export/generated-candidate sections in `VisualPackEditor.tsx`.
- [ ] Re-run the focused test.
  Expected: PASS.

## Stage 3: Documentation

**Goal:** Preserve the same product model outside the editor.

**Success Criteria:** Docs explain Persona/Buddy visual-pack ownership and explicitly say this is not VN/CYOA asset-pack behavior.

**Tests:** docs grep check.

**Status:** Not Started

- [ ] Create `Docs/Code_Documentation/Persona_Visual_Packs.md` with sections:
  - Ownership model
  - Active versus available packs
  - Manifest-backed pack format
  - Import preview and commit
  - Generated candidates and review
  - Export archives and future portability
  - Scope: Persona/Buddy, not VN/CYOA
- [ ] Run:
  `rg -n "user-owned|attached to one persona|manifest|active pack|import preview|generated candidates|not VN|CYOA" Docs/Code_Documentation/Persona_Visual_Packs.md`
  Expected: all required phrases are present.

## Stage 4: Verification And Packaging

**Goal:** Leave the branch reviewable and tied back to issue #1429.

**Success Criteria:** Focused UI tests pass, docs checks pass, Backlog task is updated, and the branch is ready for PR.

**Tests:** Focused Vitest, docs grep, diff hygiene. Bandit is not required if only docs and TSX copy/tests are touched; record the skip reason.

**Status:** Not Started

- [ ] Run focused Vitest:
  `./node_modules/.bin/vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
- [ ] Run docs grep:
  `rg -n "user-owned|attached to one persona|manifest|active pack|import preview|generated candidates|not VN|CYOA" Docs/Code_Documentation/Persona_Visual_Packs.md`
- [ ] Run diff hygiene:
  `git diff --check`
- [ ] Update TASK-189 acceptance criteria, notes, verification, and final summary.
- [ ] Commit with:
  `git commit -m "docs: clarify persona visual pack ownership"`
- [ ] Push and open a PR against `dev` linked to #1429.
