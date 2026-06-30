# Research Workspace Deep Research Bundle Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a matching Deep Research return handoff import a completed `bundle.json` into Research Workspace as a source-backed generated artifact.

**Architecture:** Keep the slice frontend-only and reuse the existing `tldwClient.getResearchBundle(run_id)` API plus the Research Workspace `addArtifact` store action. Add a pure bundle-to-artifact adapter so validation, provenance, source coverage, and bounded content formatting can be tested without rendering the full workspace.

**Tech Stack:** TypeScript, React, Zustand workspace store, Vitest/Testing Library, existing Research Workspace route-state helpers, existing Deep Research bundle API client.

---

## Source Requirements

- Backlog: `TASK-573`
- Existing follow-up: `TASK-572` (`Import Deep Research bundles into Research Workspace artifacts`)
- PRD: `Docs/Product/Research_Workspace_Literature_Workproducts_PRD.md`
- Umbrella plan: `Docs/superpowers/plans/2026-05-30-research-workspace-literature-workproducts-plan.md`

## File Structure

- Create `apps/packages/ui/src/components/Option/ResearchWorkspace/deep-research-bundle-import.ts`
  - Pure validation and normalization for Deep Research bundle imports.
  - Builds a `GeneratedArtifact` payload without `id`/`createdAt`.
  - Preserves run/source artifact provenance in `producerMetadata` and `data.deepResearch`.
  - Builds bounded markdown content and source coverage from the source artifact when available, otherwise from `bundle.source_inventory`.
- Create `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/deep-research-bundle-import.test.ts`
  - Covers successful artifact payload construction, malformed bundle rejection, source coverage fallback, and bounded content.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
  - Add explicit Import action to the existing Deep Research return handoff.
  - Fetch bundle with `tldwClient.getResearchBundle`.
  - Insert imported artifact through `addArtifact`.
  - Show importing, imported, and failed states without mutating unrelated workspace state.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx`
  - Mock bundle fetch and artifact insertion.
  - Cover successful import and failed/malformed bundle handling.
- Modify `backlog/tasks/task-573 - Implement-Deep-Research-bundle-import-into-Research-Workspace-artifacts.md`
  - Record implementation notes, verification, and final status at closeout.

## Task 1: Add Bundle Import Adapter Tests

**Goal:** Lock the import contract before production code.

- [x] **Step 1: Write failing adapter tests**

Add tests for:

- valid bundle returns a `report` artifact titled from the source artifact/run;
- artifact contains `producerMetadata.producerType === "deep_research_bundle_import"`;
- artifact `data.deepResearch.runId`, `verificationSummary`, `sourceTrust`, `claims`, `unsupportedClaims`, `contradictions`, and `sourceArtifact` are populated;
- source artifact `sourceCoverage` is preserved when available;
- fallback coverage is derived from `bundle.source_inventory`;
- missing `question` or `report_markdown` throws a user-facing import error;
- long report content is bounded and marked as truncated.

- [x] **Step 2: Run tests and confirm RED**

Run:

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/deep-research-bundle-import.test.ts
```

Expected: FAIL because the adapter does not exist.

## Task 2: Implement Bundle Import Adapter

**Goal:** Convert a Deep Research bundle into a safe Research Workspace artifact payload.

- [x] **Step 1: Create `deep-research-bundle-import.ts`**

Implement:

- `DeepResearchBundleImportError`
- `buildDeepResearchBundleArtifactPayload(options)`
- internal helpers for bounded strings, string lists, record/list checks, source inventory normalization, provenance metadata, and markdown assembly.

- [x] **Step 2: Preserve source coverage**

Rules:

- If the source artifact has `sourceCoverage`, copy it directly.
- Otherwise derive `selectedSourceIds` and `usableSources` from `bundle.source_inventory`.
- `minimumUsableSourcesMet` is true when at least one imported source exists.
- Do not invent skipped/truncated sources when the bundle does not provide them.

- [x] **Step 3: Run adapter tests and confirm GREEN**

Run:

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/deep-research-bundle-import.test.ts
```

Expected: PASS.

## Task 3: Wire Import Action Into Return Handoff

**Goal:** Let the user import a matching returned bundle from the existing banner.

- [x] **Step 1: Add failing Research Workspace UI tests**

In `ResearchWorkspace.stage2.responsive.test.tsx`, add tests that:

- clicking `Import bundle` calls `tldwClient.getResearchBundle("research-run-7")`;
- `addArtifact` receives the normalized artifact payload;
- successful import shows a compact imported state;
- rejected/malformed bundles show an error and do not call `addArtifact`;
- mismatched workspace return contexts still do not render the handoff.

- [x] **Step 2: Run UI tests and confirm RED**

Run:

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx
```

Expected: FAIL because the Import action is not wired.

- [x] **Step 3: Implement the Import action**

Modify `ResearchWorkspace/index.tsx`:

- select `addArtifact` from `useWorkspaceStore`;
- keep local import status/error state scoped to the active return context;
- find the source artifact by `sourceArtifactId` when present;
- fetch the bundle through `tldwClient.getResearchBundle`;
- pass bundle, return context, and source artifact into the adapter;
- call `addArtifact(payload)`;
- show loading, success, and error copy in the handoff banner.

- [x] **Step 4: Run UI tests and confirm GREEN**

Run:

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx
```

Expected: PASS.

## Task 4: Verification And Closeout

**Goal:** Prove the slice and record results.

- [x] **Step 1: Run focused Research Workspace tests**

Run:

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/deep-research-bundle-import.test.ts \
  src/components/Option/ResearchWorkspace/__tests__/research-workspace-route-state.test.ts \
  src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx
```

- [x] **Step 2: Run TypeScript check**

Run:

```bash
cd apps/packages/ui
NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit -p tsconfig.json
```

- [x] **Step 3: Run diff check**

Run:

```bash
git diff --check
```

- [x] **Step 4: Record Bandit applicability**

If only frontend TS/TSX/docs/backlog files changed, record Bandit skipped as not applicable. If Python changes are added, run Bandit on the touched Python scope.

- [x] **Step 5: Update Backlog and commit**

Update TASK-573 with notes, final summary, and DoD status, then commit the task, plan, tests, and implementation.
