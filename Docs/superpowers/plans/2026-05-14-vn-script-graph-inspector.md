# VN Script Graph Inspector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only WebUI graph/outline inspector for VN script drafts and published versions using the backend-owned graph API from issue #1680.

**Architecture:** The frontend adds typed graph API helpers and renders server-shaped graph data only. The existing VN script workbench remains the integration point; graph derivation, validation, cache keys, and diagnostics semantics stay owned by `/api/v1/vn/vn-scripts`.

**Tech Stack:** Next.js/React, existing `apiClient`, Vitest + Testing Library, VN API docs in `Docs/API/VN.md`.

---

### Task 1: Typed VN Script Graph Client Contract

**Files:**
- Modify: `apps/tldw-frontend/types/vn-scripts.ts`
- Modify: `apps/tldw-frontend/lib/api/vnScripts.ts`
- Modify: `apps/tldw-frontend/__tests__/vn-scripts/vnScriptsApi.test.ts`

- [ ] Add TypeScript interfaces for the graph response envelope, outline labels, graph nodes, graph edges, graph diagnostics, and preview request.
- [ ] Add `getVNScriptDraftGraph`, `previewVNScriptDraftGraph`, and `getVNScriptVersionGraph` helpers.
- [ ] Add failing API helper tests for the three endpoint paths and payload behavior.
- [ ] Run `bunx vitest run __tests__/vn-scripts/vnScriptsApi.test.ts` from `apps/tldw-frontend` and confirm the new tests fail before implementation.
- [ ] Implement the helpers and verify the focused API test passes.

### Task 2: Read-Only Graph Inspector UI

**Files:**
- Modify: `apps/tldw-frontend/components/vn-scripts/VNScriptsWorkbench.tsx`
- Modify: `apps/tldw-frontend/__tests__/vn-scripts/VNScriptsWorkbench.test.tsx`

- [ ] Add state and loading handlers for saved draft graph, unsaved draft graph preview, and per-version graph reads.
- [ ] Gate the panel on `vnCapabilities.features.script_authoring_graph`.
- [ ] Render graph source metadata, cache/staleness keys, outline rows, graph diagnostics, validation diagnostics, and truncated-state warnings distinctly from existing validation/diagnostics panels.
- [ ] Provide actions to load the saved draft graph, preview current editor JSON without saving, and load a published version graph.
- [ ] Keep the UI read-only: do not derive edges client-side, mutate drafts, or add node-editing controls.
- [ ] Add failing workbench tests for capability gating, saved draft graph rendering, unsaved preview payloads, version graph rendering, and graph-vs-validation diagnostics separation.
- [ ] Run `bunx vitest run __tests__/vn-scripts/VNScriptsWorkbench.test.tsx` from `apps/tldw-frontend` and confirm the new tests fail before implementation.
- [ ] Implement the panel and verify the focused workbench test passes.

### Task 3: Documentation And Verification

**Files:**
- Modify: `Docs/API/VN.md`
- Modify: `backlog/tasks/task-333 - Add-VN-script-graph-and-outline-inspector.md`

- [ ] Update the VN API docs with a short WebUI/custom-frontend graph-inspector flow, including when to call saved draft, unsaved preview, and version graph endpoints.
- [ ] Run focused frontend tests for VN script API and workbench.
- [ ] Run `git diff --check`.
- [ ] Run Bandit on touched Python scope if any backend Python is touched; otherwise record the frontend/docs-only skip in TASK-333.
- [ ] Update TASK-333 acceptance criteria, verification notes, known skips, and final summary.
