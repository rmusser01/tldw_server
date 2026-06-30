# Persona Buddy Sprite Atlas V1.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden Persona/Buddy sprite atlas support under the existing `sprite_frames` renderer.

**Architecture:** Keep atlas support as raster frame rendering, not a new renderer. Backend validation continues to validate `frames[].region` against known asset dimensions, while Buddy renders region-backed frames through the existing renderer registry and fails soft when a region is invalid at runtime.

**Tech Stack:** Python manifest validation tests, React/Vitest Buddy renderer tests, Markdown product/code documentation.

---

### Task 1: Backend Atlas Validation Contract

**Files:**
- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`

- [x] **Step 1: Add an activatable atlas-manifest test**

Add a test that uses `renderer_type: "sprite_frames"`, a `sprite_sheet`-style atlas asset ID, required visual states, and `frames[].region` rectangles. Validate with `require_activatable=True` and known dimensions.

- [x] **Step 2: Add a missing-dimension behavior test**

Add a test showing that region bounds are only rejected when source dimensions are known; when dimensions are unavailable, validation still accepts positive integer regions so drafts/import previews can remain fail-open until asset metadata exists.

- [x] **Step 3: Run backend focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py -q`

Expected: all tests pass.

### Task 2: Buddy Renderer And Diagnostics Contract

**Files:**
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts`

- [x] **Step 1: Add registry-path atlas rendering coverage**

Add a `PersonaVisualRendererHost` test proving a pack with `asset_role: "sprite_sheet"` and `frames[].region` renders as a cropped background through the registered `sprite_frames` renderer.

- [x] **Step 2: Add unsupported-region diagnostics coverage**

Add coverage showing `unsupported_region` render errors produce a fail-soft warning diagnostic instead of blocking Buddy.

- [x] **Step 3: Run frontend focused tests**

Run: `bunx vitest run apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts`

Expected: all tests pass.

### Task 3: Documentation And Tracker Closeout

**Files:**
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
- Modify: `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
- Modify: `Docs/Design/2026-05-10-persona-visual-renderer-provider-adapter-evaluation.md`
- Modify: `backlog/tasks/task-308 - Harden-Persona-Buddy-sprite-atlas-support-under-sprite_frames.md`

- [x] **Step 1: Add a minimal atlas-backed manifest example**

Document a `sprite_frames` manifest using a `sprite_sheet` asset role plus frame-level `region` rectangles. State that `sprite_sheet` is an asset role in this slice, not an activatable renderer.

- [x] **Step 2: Refresh roadmap/evaluation wording**

Mark sprite atlas V1.1 as the current hardening slice and keep Live2D, non-sprite manifest V2, external providers, and shared libraries as future work.

- [x] **Step 3: Run final checks**

Run: `git diff --check`

Run focused backend and frontend tests from Tasks 1 and 2. Run Bandit on touched Python source only if source code changes; otherwise record a non-code skip for Bandit.
