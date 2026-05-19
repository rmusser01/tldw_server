# tldw Product Roadmap First Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first narrow implementation slice of the approved workspace-first product roadmap: canonical workspace discovery, one executive-brief golden path, a minimal artifact review contract, and a typed bridge between the existing server workspace API and the browser-local WorkspacePlayground store.

**Architecture:** Treat the current `WorkspacePlayground` as the likely canonical shell, but make that decision explicit through a short decision record before changing route semantics. Reuse the existing `/api/v1/workspaces` backend and frontend workspace stores; add typed adapters and contracts where current code uses `any` or browser-only state. Add one work-product template end to end before expanding to the other roadmap templates.

**Tech Stack:** FastAPI, Pydantic, SQLite-backed `CharactersRAGDB`, Next.js/WebUI extension routes, React, Zustand, Vitest, pytest.

---

## Scope Boundary

This plan implements only the 6-8 week first-value slice from the roadmap spec.

In scope:

- Workspace route/state discovery and a canonical-workspace decision record.
- Typed frontend contract for the existing `/api/v1/workspaces` API.
- Minimal generated-artifact contract for template lineage, review status, citations, and export intent.
- Template metadata for all four flagship work products, with only `executive_brief` wired as the golden path.
- WorkspacePlayground entry point for selecting the executive brief template and generating a reviewable artifact.
- Focused tests and docs for the above.

Out of scope:

- Full route consolidation between `WorkspacePlayground`, `ChatWorkspace`, and `DocumentWorkspace`.
- End-to-end implementation of all four templates.
- Full shared-team collaboration, real-time editing, billing, or seat management.
- Broad connector work across Drive, Notion, GitHub, email, and Slack.
- New backend workspace service parallel to the existing `/api/v1/workspaces` endpoints.

## Existing Anchors

- Roadmap spec: `Docs/superpowers/specs/2026-05-06-tldw-product-roadmap-design.md`
- Workspace docs: `Docs/Product/WebUI/Workspace_Playground_Redesign.md`
- Local persistence docs: `Docs/Design/Workspace_Persistence_Architecture.md`
- Workspace route: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Workspace shell: `apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx`
- Workspace studio: `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx`
- Artifact generation hook: `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/hooks/useArtifactGeneration.tsx`
- Workspace store: `apps/packages/ui/src/store/workspace.ts`
- Store slices: `apps/packages/ui/src/store/workspace-slices/`
- Workspace types: `apps/packages/ui/src/types/workspace.ts`
- Existing frontend workspace API helper: `apps/packages/ui/src/store/workspace-api.ts`
- Existing sync payload helper: `apps/packages/ui/src/store/workspace-sync-contract.ts`
- Existing tldw API workspace domain: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Backend workspaces endpoint: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Backend workspace schemas: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Backend workspace tests: `tldw_Server_API/tests/Workspaces/`

## File Structure

Create:

- `Docs/Design/Workspace_Canonical_Model_Decision_2026_05.md`
  - Decision record for canonical workspace shell, route boundaries, and how chat-first/document-focused experiences fit.
- `apps/packages/ui/src/workspace-templates/work-product-templates.ts`
  - Template metadata and helper functions for executive brief, research dossier, competitive/market memo, and technical/project spec.
- `apps/packages/ui/src/workspace-templates/types.ts`
  - Shared template ID and citation policy types used by both workspace artifact types and template metadata.
- `apps/packages/ui/src/workspace-templates/__tests__/work-product-templates.test.ts`
  - Unit tests for template metadata, source requirements, and golden-path defaults.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/WorkProductTemplateChooser.tsx`
  - Compact template chooser for the Studio pane.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkProductTemplateChooser.test.tsx`
  - Component tests for template selection and disabled/missing-source states.
- `apps/packages/ui/src/store/__tests__/workspace-artifact-review-contract.test.ts`
  - Store/type tests for artifact review status and lineage persistence.
- `apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts`
  - Tests for typed workspace API methods if the service layer is updated beyond current `any` return types.

Modify:

- `Docs/Product/WebUI/Workspace_Playground_Redesign.md`
  - Add a short pointer to the canonical workspace decision record and first-slice cut line.
- `Docs/Design/Workspace_Persistence_Architecture.md`
  - Add a short server/local boundary note that references the existing `/api/v1/workspaces` API and the local cache role.
- `apps/packages/ui/src/types/workspace.ts`
  - Add template IDs, artifact review status, lineage, review checklist, and export-intent fields.
- `apps/packages/ui/src/store/workspace.ts`
  - Preserve new artifact fields through local persistence, split storage, IndexedDB offload, import/export, duplicate, and restore paths.
- `apps/packages/ui/src/store/workspace-slices/studio-slice.ts`
  - Add focused artifact review mutations if the implementation needs explicit accept/revise/export transitions.
- `apps/packages/ui/src/store/workspace-api.ts`
  - Replace loose `any` workspace hydration shapes with typed server/local adapters that match backend schema names.
- `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
  - Add typed request/response interfaces for workspace, source, artifact, and note methods.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx`
  - Place the template chooser where artifact generation already lives.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/hooks/useArtifactGeneration.tsx`
  - Thread selected template metadata into generated artifact creation for executive brief only.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage1.test.tsx`
  - Extend existing Studio pane coverage for the template entry point if a new dedicated test is not enough.
- `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
  - Only modify if backend artifact schema needs fields that cannot fit the current `status` and `content` contract.
- `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
  - Only modify if schema changes require response mapping changes.
- `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`
  - Only modify if backend schema/endpoint behavior changes.

Do not create a second workspace API, a new global store, or another route that competes with `WorkspacePlayground`.

---

### Task 1: Canonical Workspace Decision Record

**Files:**

- Create: `Docs/Design/Workspace_Canonical_Model_Decision_2026_05.md`
- Modify: `Docs/Product/WebUI/Workspace_Playground_Redesign.md`
- Modify: `Docs/Design/Workspace_Persistence_Architecture.md`

- [ ] **Step 1: Inventory current workspace entry points**

  Read:

  ```bash
  sed -n '460,510p' apps/tldw-frontend/extension/routes/route-registry.tsx
  sed -n '1,220p' apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx
  sed -n '1,220p' apps/packages/ui/src/components/DocumentWorkspace/DocumentWorkspacePage.tsx
  sed -n '760,900p' apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx
  ```

  Record the current user intent for each route:

  - `/workspace-playground`: broad research workspace and best candidate for canonical shell.
  - `/chat-workspace`: chat-first/staged-context workspace.
  - `/document-workspace`: document-focused reading/annotation workspace.

- [ ] **Step 2: Write the decision record**

  Create `Docs/Design/Workspace_Canonical_Model_Decision_2026_05.md` with these sections:

  ```markdown
  # Workspace Canonical Model Decision - May 2026

  ## Decision

  `WorkspacePlayground` is the canonical shell for the roadmap first slice.
  `ChatWorkspace` and `DocumentWorkspace` remain separate routes during this slice
  and are treated as specialized entry points/modes, not deleted or fully merged.

  ## Reasons

  - WorkspacePlayground already contains sources, selected sources, chat, quick notes,
    generated artifacts, saved workspaces, source transfer, local persistence, and
    artifact payload offload.
  - ChatWorkspace validates a chat-first route but should not own a separate product model.
  - DocumentWorkspace validates deep document reading and annotation but should feed
    workspace sources/artifacts rather than define a parallel commercial workspace.

  ## First-Slice Boundary

  This slice does not consolidate routes. It defines the shared model and implements
  one golden path inside WorkspacePlayground.

  ## Server/Local Boundary

  The server already exposes `/api/v1/workspaces` for workspace metadata, sources,
  artifacts, and notes. The browser-local Zustand store remains the responsive cache
  and offline-friendly UI state. The first implementation must reconcile field names,
  artifact status semantics, and persistence behavior between the two.

  ## Follow-Up Decisions

  - Whether ChatWorkspace becomes a mode inside WorkspacePlayground.
  - Whether DocumentWorkspace writes selected documents into workspace sources by default.
  - Which collaboration semantics are required before enterprise pilots.
  ```

- [ ] **Step 3: Add doc pointers**

  Add a short "Current first-slice decision" note to:

  - `Docs/Product/WebUI/Workspace_Playground_Redesign.md`
  - `Docs/Design/Workspace_Persistence_Architecture.md`

  The note should point to the decision record and state that server sync uses the existing `/api/v1/workspaces` family first.

- [ ] **Step 4: Verify docs**

  Run:

  ```bash
  rg -n "Workspace Canonical Model Decision|/api/v1/workspaces|first-slice" Docs/Design Docs/Product/WebUI
  git diff --check
  ```

  Expected: references are present and `git diff --check` reports no whitespace errors.

- [ ] **Step 5: Commit**

  ```bash
  git add Docs/Design/Workspace_Canonical_Model_Decision_2026_05.md Docs/Product/WebUI/Workspace_Playground_Redesign.md Docs/Design/Workspace_Persistence_Architecture.md
  git commit -m "docs: record canonical workspace first slice"
  ```

---

### Task 2: Type The Existing Workspace API Contract

**Files:**

- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Modify: `apps/packages/ui/src/store/workspace-api.ts`
- Modify: `apps/packages/ui/src/store/workspace-sync-contract.ts`
- Test: `apps/packages/ui/src/store/__tests__/workspace-api-first.test.ts`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts`
- Backend fallback only if needed: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Backend fallback only if needed: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`

- [ ] **Step 1: Write failing frontend contract tests**

  Extend `apps/packages/ui/src/store/__tests__/workspace-api-first.test.ts` so hydration maps backend snake_case into UI camelCase:

  ```ts
  it("maps backend workspace source and artifact fields into local workspace state", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      id: "ws-1",
      name: "Server WS",
      version: 2,
      sources: [
        {
          id: "src-1",
          workspace_id: "ws-1",
          media_id: 42,
          title: "Quarterly strategy doc",
          source_type: "document",
          url: "https://example.test/doc",
          selected: true,
          added_at: "2026-05-06T12:00:00Z",
          version: 1
        }
      ],
      artifacts: [
        {
          id: "art-1",
          workspace_id: "ws-1",
          artifact_type: "report",
          title: "Executive Brief",
          status: "draft",
          content: "Brief body",
          total_tokens: 120,
          total_cost_usd: 0.02,
          created_at: "2026-05-06T12:05:00Z",
          completed_at: "2026-05-06T12:06:00Z",
          version: 3
        }
      ],
      notes: []
    })

    const state = await hydrateWorkspaceFromServer("ws-1", { fetch: mockFetch })

    expect(state.sources[0]).toMatchObject({
      id: "src-1",
      mediaId: 42,
      title: "Quarterly strategy doc",
      type: "document",
      status: "ready"
    })
    expect(state.artifacts[0]).toMatchObject({
      id: "art-1",
      type: "report",
      title: "Executive Brief",
      reviewStatus: "draft",
      content: "Brief body",
      totalTokens: 120,
      totalCostUsd: 0.02
    })
  })
  ```

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/store/__tests__/workspace-api-first.test.ts
  ```

  Expected: FAIL until typed mapping and review-status fields exist.

- [ ] **Step 2: Add typed server interfaces**

  In `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`, add exported interfaces matching `workspace_schemas.py`:

  ```ts
  export interface WorkspaceApiResponse {
    id: string
    name: string | null
    archived: boolean
    study_materials_policy: "general" | "workspace"
    deleted: boolean
    created_at: string
    last_modified: string
    version: number
  }

  export interface WorkspaceSourceApiResponse {
    id: string
    workspace_id: string
    media_id: number
    title: string
    source_type: string
    url: string | null
    position: number
    selected: boolean
    added_at: string
    version: number
  }

  export interface WorkspaceArtifactApiResponse {
    id: string
    workspace_id: string
    artifact_type: string
    title: string
    status: string
    content: string | null
    total_tokens: number | null
    total_cost_usd: number | null
    created_at: string
    completed_at: string | null
    version: number
  }
  ```

  Then replace `Promise<any>` / `Record<string, any>` only for workspace methods in this file. Leave unrelated skills/watchlists methods alone.

- [ ] **Step 3: Implement adapter mapping in `workspace-api.ts`**

  Add local helper functions:

  ```ts
  const mapServerSourceToLocal = (source: WorkspaceSourceApiResponse): WorkspaceSource => ({
    id: source.id,
    mediaId: source.media_id,
    title: source.title,
    type: normalizeWorkspaceSourceType(source.source_type),
    status: "ready",
    url: source.url || undefined,
    addedAt: new Date(source.added_at)
  })

  const mapServerArtifactToLocal = (
    artifact: WorkspaceArtifactApiResponse
  ): GeneratedArtifact => ({
    id: artifact.id,
    type: normalizeArtifactType(artifact.artifact_type),
    title: artifact.title,
    status: mapServerGenerationStatus(artifact.status),
    reviewStatus: mapServerReviewStatus(artifact.status),
    serverId: artifact.id,
    content: artifact.content || undefined,
    totalTokens: artifact.total_tokens || undefined,
    totalCostUsd: artifact.total_cost_usd || undefined,
    createdAt: new Date(artifact.created_at),
    completedAt: artifact.completed_at ? new Date(artifact.completed_at) : undefined
  })
  ```

  Keep normalizers conservative:

  - Unknown source types become `"document"`.
  - Unknown artifact types become `"report"`.
  - Existing generation statuses stay compatible with `pending`, `generating`, `completed`, and `failed`.
  - Review states are stored separately from generation lifecycle.

- [ ] **Step 4: Run focused frontend tests**

  ```bash
  bunx vitest run apps/packages/ui/src/store/__tests__/workspace-api-first.test.ts apps/packages/ui/src/store/__tests__/workspace-sync-contract.test.ts
  ```

  Expected: PASS.

- [ ] **Step 5: Run backend workspace tests if backend files changed**

  Only if `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py` or `workspaces.py` changed:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py tldw_Server_API/tests/Workspaces/test_workspace_sub_resources_api.py -q
  ```

  Expected: PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add apps/packages/ui/src/services/tldw/domains/workspace-api.ts apps/packages/ui/src/store/workspace-api.ts apps/packages/ui/src/store/workspace-sync-contract.ts apps/packages/ui/src/store/__tests__/workspace-api-first.test.ts apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts
  git commit -m "feat: type workspace api bridge"
  ```

---

### Task 3: Artifact Review And Template Contract

**Files:**

- Create: `apps/packages/ui/src/workspace-templates/work-product-templates.ts`
- Create: `apps/packages/ui/src/workspace-templates/types.ts`
- Create: `apps/packages/ui/src/workspace-templates/__tests__/work-product-templates.test.ts`
- Modify: `apps/packages/ui/src/types/workspace.ts`
- Modify: `apps/packages/ui/src/store/workspace.ts`
- Modify: `apps/packages/ui/src/store/workspace-slices/studio-slice.ts`
- Test: `apps/packages/ui/src/store/__tests__/workspace-artifact-review-contract.test.ts`
- Test: `apps/packages/ui/src/store/__tests__/workspace.test.ts`
- Test: `apps/packages/ui/src/store/__tests__/workspace.split-storage.test.ts`

- [ ] **Step 1: Write template metadata tests**

  Create `apps/packages/ui/src/workspace-templates/__tests__/work-product-templates.test.ts`:

  ```ts
  import { describe, expect, it } from "vitest"
  import {
    DEFAULT_WORK_PRODUCT_TEMPLATE_ID,
    getWorkProductTemplate,
    WORK_PRODUCT_TEMPLATES
  } from "../work-product-templates"

  describe("work product templates", () => {
    it("defines all roadmap flagship templates", () => {
      expect(WORK_PRODUCT_TEMPLATES.map((template) => template.id)).toEqual([
        "executive_brief",
        "research_dossier",
        "competitive_market_memo",
        "technical_project_spec"
      ])
    })

    it("uses executive brief as the first golden path", () => {
      const template = getWorkProductTemplate(DEFAULT_WORK_PRODUCT_TEMPLATE_ID)
      expect(template.id).toBe("executive_brief")
      expect(template.outputArtifactType).toBe("report")
      expect(template.reviewChecklist.length).toBeGreaterThanOrEqual(3)
      expect(template.citationPolicy).toBe("required")
    })
  })
  ```

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/workspace-templates/__tests__/work-product-templates.test.ts
  ```

  Expected: FAIL because the module does not exist yet.

- [ ] **Step 2: Add template metadata**

  Create `apps/packages/ui/src/workspace-templates/types.ts` first:

  ```ts
  export type WorkProductTemplateId =
    | "executive_brief"
    | "research_dossier"
    | "competitive_market_memo"
    | "technical_project_spec"

  export type WorkProductCitationPolicy = "required" | "recommended"
  ```

  Then create `apps/packages/ui/src/workspace-templates/work-product-templates.ts`:

  ```ts
  import type { ArtifactType } from "@/types/workspace"
  import type {
    WorkProductCitationPolicy,
    WorkProductTemplateId
  } from "./types"

  export interface WorkProductTemplate {
    id: WorkProductTemplateId
    label: string
    description: string
    outputArtifactType: ArtifactType
    minSelectedSources: number
    sections: string[]
    reviewChecklist: string[]
    citationPolicy: WorkProductCitationPolicy
  }

  export const DEFAULT_WORK_PRODUCT_TEMPLATE_ID: WorkProductTemplateId =
    "executive_brief"

  export const WORK_PRODUCT_TEMPLATES: WorkProductTemplate[] = [
    {
      id: "executive_brief",
      label: "Executive Brief",
      description: "Decision-ready summary with context, evidence, risks, and next actions.",
      outputArtifactType: "report",
      minSelectedSources: 1,
      sections: ["Situation", "Key Findings", "Evidence", "Risks", "Recommended Actions"],
      reviewChecklist: [
        "Every material claim has a source or explicit uncertainty.",
        "Recommendations are separated from evidence.",
        "Risks and open questions are visible before export."
      ],
      citationPolicy: "required"
    }
  ]
  ```

  Add the other three templates in the same array with metadata only. Do not wire them into generation yet.

- [ ] **Step 3: Extend artifact types without breaking generation lifecycle**

  In `apps/packages/ui/src/types/workspace.ts`, keep:

  ```ts
  export type ArtifactStatus = "pending" | "generating" | "completed" | "failed"
  ```

  Add:

  ```ts
  export type ArtifactReviewStatus =
    | "draft"
    | "reviewing"
    | "accepted"
    | "needs_revision"
    | "exported"
    | "assigned"

  export interface ArtifactSourceLineage {
    sourceId: string
    mediaId?: number
    title?: string
    citationCount?: number
  }

  export interface ArtifactReviewChecklistItem {
    id: string
    label: string
    checked: boolean
  }
  ```

  Extend `GeneratedArtifact` with optional fields:

  ```ts
  templateId?: WorkProductTemplateId
  reviewStatus?: ArtifactReviewStatus
  sourceLineage?: ArtifactSourceLineage[]
  reviewChecklist?: ArtifactReviewChecklistItem[]
  exportTargets?: Array<"markdown" | "docx" | "pdf" | "slides" | "chatbook">
  ```

  Import `WorkProductTemplateId` from `apps/packages/ui/src/workspace-templates/types.ts`,
  not from `work-product-templates.ts`. This avoids a cycle because the metadata
  module imports `ArtifactType` from `workspace.ts`.

- [ ] **Step 4: Write artifact persistence tests**

  Create `apps/packages/ui/src/store/__tests__/workspace-artifact-review-contract.test.ts` to verify:

  - `addArtifact` preserves `templateId`, `reviewStatus`, `sourceLineage`, and `reviewChecklist`.
  - `updateArtifactStatus` does not erase review fields.
  - Persistence sanitize/revive keeps review fields.

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/store/__tests__/workspace-artifact-review-contract.test.ts
  ```

  Expected: FAIL until store persistence preserves the fields.

- [ ] **Step 5: Implement minimal store support**

  Update `apps/packages/ui/src/store/workspace.ts` and `apps/packages/ui/src/store/workspace-slices/studio-slice.ts` only as needed:

  - Do not rewrite the store.
  - Do not change split-storage keys.
  - Preserve new fields in `sanitizeArtifactForPersistence`, `reviveArtifacts`, duplicate workspace paths, import/export paths, and artifact offload paths.
  - Add explicit review mutation only if UI needs it:

    ```ts
    updateArtifactReviewStatus: (
      id: string,
      reviewStatus: ArtifactReviewStatus
    ) => void
    ```

- [ ] **Step 6: Run focused store tests**

  ```bash
  bunx vitest run apps/packages/ui/src/workspace-templates/__tests__/work-product-templates.test.ts apps/packages/ui/src/store/__tests__/workspace-artifact-review-contract.test.ts apps/packages/ui/src/store/__tests__/workspace.test.ts apps/packages/ui/src/store/__tests__/workspace.split-storage.test.ts
  ```

  Expected: PASS.

- [ ] **Step 7: Commit**

  ```bash
  git add apps/packages/ui/src/workspace-templates apps/packages/ui/src/types/workspace.ts apps/packages/ui/src/store/workspace.ts apps/packages/ui/src/store/workspace-slices/studio-slice.ts apps/packages/ui/src/store/__tests__/workspace-artifact-review-contract.test.ts
  git commit -m "feat: add work product artifact contract"
  ```

---

### Task 4: Executive Brief Golden Path In WorkspacePlayground

**Files:**

- Create: `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/WorkProductTemplateChooser.tsx`
- Test: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkProductTemplateChooser.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/hooks/useArtifactGeneration.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage1.test.tsx`
- Optional test: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/StudioPane.executive-brief-template.test.tsx`

- [ ] **Step 1: Write chooser component tests**

  Create `WorkProductTemplateChooser.test.tsx`:

  ```tsx
  import { render, screen } from "@testing-library/react"
  import userEvent from "@testing-library/user-event"
  import { describe, expect, it, vi } from "vitest"
  import { WorkProductTemplateChooser } from "../StudioPane/WorkProductTemplateChooser"

  describe("WorkProductTemplateChooser", () => {
    it("offers executive brief as the golden path", async () => {
      const onSelect = vi.fn()
      render(
        <WorkProductTemplateChooser
          selectedTemplateId="executive_brief"
          selectedSourceCount={2}
          onSelectTemplate={onSelect}
        />
      )

      expect(screen.getByRole("button", { name: /executive brief/i })).toBeEnabled()
      expect(screen.getByText(/decision-ready/i)).toBeInTheDocument()
    })

    it("marks templates unavailable when source requirements are not met", () => {
      render(
        <WorkProductTemplateChooser
          selectedTemplateId="executive_brief"
          selectedSourceCount={0}
          onSelectTemplate={() => undefined}
        />
      )

      expect(screen.getByRole("button", { name: /executive brief/i })).toHaveAttribute(
        "aria-disabled",
        "true"
      )
    })
  })
  ```

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkProductTemplateChooser.test.tsx
  ```

  Expected: FAIL because the component does not exist.

- [ ] **Step 2: Implement chooser with existing UI conventions**

  Build a compact Studio-pane control:

  - Use existing button, card, badge, or state primitives already used in `StudioPane/index.tsx`.
  - Do not introduce a page-level hero, marketing copy, or nested cards.
  - Keep all four templates visible, but only make executive brief actionable in this slice.
  - Use disabled or "planned" state for the other three templates.
  - Avoid in-app explanatory text about roadmap/internal functionality.

- [ ] **Step 3: Thread selected template into artifact generation**

  In `StudioPane/index.tsx`, keep local UI state for selected template ID if the store does not need it globally yet.

  In `useArtifactGeneration.tsx`, when generating the executive brief:

  - Use `outputArtifactType: "report"`.
  - Create artifact title from template label, for example `Executive Brief`.
  - Include `templateId: "executive_brief"`.
  - Include `reviewStatus: "draft"` once content generation completes.
  - Include `sourceLineage` from selected workspace sources.
  - Include `reviewChecklist` from template metadata.

  Do not change generation for existing summary, audio overview, mind map, flashcards, quiz, timeline, slides, or data table outputs unless needed to keep types compiling.

- [ ] **Step 4: Add Studio pane integration coverage**

  Extend `StudioPane.stage1.test.tsx` or add `StudioPane.executive-brief-template.test.tsx` to verify:

  - Template chooser renders in the Studio pane.
  - Executive brief generation is disabled until source requirements are met.
  - Generated executive brief artifacts carry template/review fields.
  - Existing non-template output buttons still render.

- [ ] **Step 5: Run focused UI tests**

  ```bash
  bunx vitest run apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkProductTemplateChooser.test.tsx apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage1.test.tsx apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage2.test.tsx
  ```

  Expected: PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/WorkProductTemplateChooser.tsx apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/hooks/useArtifactGeneration.tsx apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkProductTemplateChooser.test.tsx apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage1.test.tsx
  git commit -m "feat: add executive brief workspace template path"
  ```

---

### Task 5: Validation, Documentation, And Implementation Closeout

**Files:**

- Modify: `Docs/superpowers/specs/2026-05-06-tldw-product-roadmap-design.md`
- Modify: `Docs/Product/WebUI/Workspace_Playground_Redesign.md`
- Modify: active implementation Backlog task for this execution slice

- [ ] **Step 1: Update docs with implementation result**

  Add a short "First implementation slice" note to the roadmap spec after the 6-8 week horizon introduction:

  ```markdown
  First implementation slice: canonical workspace decision record, typed
  server/local workspace bridge, executive brief template, and generated
  artifact review contract.
  ```

  Update the Workspace Playground redesign doc only with user-facing/product-relevant facts. Do not paste implementation details that are already in tests.

- [ ] **Step 2: Run full focused verification**

  Frontend:

  ```bash
  bunx vitest run apps/packages/ui/src/workspace-templates/__tests__/work-product-templates.test.ts apps/packages/ui/src/store/__tests__/workspace-api-first.test.ts apps/packages/ui/src/store/__tests__/workspace-artifact-review-contract.test.ts apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkProductTemplateChooser.test.tsx apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage1.test.tsx apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage2.test.tsx
  ```

  Backend, only if workspace API/schema files changed:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py tldw_Server_API/tests/Workspaces/test_workspace_sub_resources_api.py -q
  ```

  Static/checks:

  ```bash
  git diff --check
  ```

- [ ] **Step 3: Run Bandit if backend Python changed**

  If any `tldw_Server_API/**/*.py` files changed:

  ```bash
  source .venv/bin/activate
  python -m bandit -r tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py -f json -o /tmp/bandit_product_roadmap_first_slice.json
  ```

  Expected: no new findings in touched backend files.

  If no backend Python files changed, record that Bandit was skipped because the slice was frontend/docs only.

- [ ] **Step 4: Final review checklist**

  Confirm:

  - The implementation still uses `WorkspacePlayground` as the golden-path shell.
  - `ChatWorkspace` and `DocumentWorkspace` still work as separate routes.
  - The existing `/api/v1/workspaces` API is reused.
  - Only executive brief is implemented end to end.
  - Other templates are metadata/planned state only.
  - Artifact generation lifecycle and artifact review lifecycle are not conflated.
  - No broad connector, billing, seat, or collaboration work entered this slice.

- [ ] **Step 5: Update the active implementation Backlog task and commit**

  Use the Backlog task that was created for executing this first slice. Do not
  reuse `TASK-97.1`, which tracked creation of this plan document.

  Replace `TASK-123` and the task filename below with the real implementation
  task ID and file created for the execution slice.

  ```bash
  backlog task edit TASK-123 --ac 1 --ac 2 --ac 3 --ac 4 --ac 5 --notes "Verification: commands run and outcomes. Bandit: run or skipped reason." --final-summary "Implemented the first-slice roadmap plan: canonical workspace decision, typed workspace API bridge, executive brief work-product template, and generated artifact review contract."
  ```

  Commit:

  ```bash
  git add Docs/superpowers/specs/2026-05-06-tldw-product-roadmap-design.md Docs/Product/WebUI/Workspace_Playground_Redesign.md "backlog/tasks/task-123 - Implement-product-roadmap-first-slice.md"
  git commit -m "feat: implement product roadmap first slice"
  ```

---

## Handoff Notes

- Before executing this plan, create a new Backlog task for the implementation
  slice or promote an existing matching task to `In Progress`. This plan's
  creation task is not the execution task.
- Implement tasks sequentially unless using explicit subagent-driven execution with disjoint ownership.
- Prefer keeping backend changes out of the first pass unless frontend typing proves the current schema cannot represent review state or template lineage.
- If backend artifact schema changes become necessary, create a focused follow-up Backlog task before expanding scope.
- If route consolidation becomes tempting during implementation, stop and update the decision record instead of merging routes in this slice.
