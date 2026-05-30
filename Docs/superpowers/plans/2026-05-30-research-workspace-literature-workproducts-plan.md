# Research Workspace Literature Work Products Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four MVP literature-review work products to Research Workspace: Literature Matrix, Corpus Gap Finder, Evidence-Bound Hypothesis Generator, and Research Proposal Pack.

**Architecture:** MVP stays inside Research Workspace Studio and reuses the existing selected-source, Studio artifact, work-product template, chat-completion generation, review checklist, and export patterns. Typed literature work products use JSON-first generation, explicit source coverage, and template-declared availability/generation strategies. Deep Research integration is planned as a later stage and must not block MVP delivery.

**Tech Stack:** TypeScript, React, Zustand workspace store, Vitest/Testing Library, existing tldw API client, existing Research Workspace Studio hooks, existing Python/FastAPI document-insights and file-artifacts APIs where reused.

---

## Source PRD

- `Docs/Product/Research_Workspace_Literature_Workproducts_PRD.md`
- Backlog: TASK-486

## File Structure

Expected implementation touch points:

- Modify `apps/packages/ui/src/workspace-templates/types.ts`
  - Add work-product template IDs plus category, availability, and generation strategy types.
- Modify `apps/packages/ui/src/workspace-templates/work-product-templates.ts`
  - Add four Literature Review work-product templates with requirements, availability, generation strategies, and review checklists.
- Modify `apps/packages/ui/src/workspace-templates/__tests__/work-product-templates.test.ts`
  - Cover existing template preservation, new template IDs, source requirements, availability, generation strategies, review checklists, and output type mapping.
- Modify `apps/packages/ui/src/types/workspace.ts`
  - Add artifact source coverage metadata for selected, usable, skipped, and truncated sources.
- Create `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/literature-workproducts.ts`
  - Isolate prompts, JSON schemas/validators, compatibility helpers, table/list normalization, and artifact metadata builders for the four work products.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`
  - Dispatch generation for the new templates, attach lineage/review/coverage metadata, and preserve existing generic output behavior.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/WorkProductTemplateChooser.tsx`
  - Make new actionable templates visible under the work-product model without enabling unrelated planned templates.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx`
  - Add any necessary display grouping or artifact viewer handoff.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/ArtifactModalContent.tsx`
  - Render typed table/list data where available and expose CSV/JSON export affordances.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactExport.tsx`
  - Keep MVP export scope honest: client CSV/JSON/markdown unless a server artifact ID exists.
- Modify or create `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx`
  - Cover generation gates, success states, JSON validation failures, source coverage, source lineage, and review checklists.
- Optional later backend files:
  - `tldw_Server_API/app/api/v1/endpoints/media/document_insights.py`
  - `tldw_Server_API/app/core/File_Artifacts/adapters/data_table_adapter.py`
  - These are reuse targets, not required MVP modifications unless implementation finds a hard blocker.

## Stage -1: Implementation Tracking Setup

**Goal:** Create or reopen an active implementation Backlog.md task before code edits begin.

**Success Criteria:**

- There is an active Backlog.md task for implementation work.
- The task links this PRD and plan.
- Follow-up Deep Research work stays out of the MVP implementation task unless explicitly approved.

**Tests:**

- Backlog task view shows status `In Progress` and documentation links.

- [x] **Step 1: Search for an existing implementation task**

Run:

```bash
backlog search "Research Workspace literature work products" --plain
```

Expected: Either find an existing implementation task or confirm that only the
docs/planning task exists.

- [x] **Step 2: Create or update the implementation task**

Use the Backlog.md MCP workflow when available. If using CLI fallback, create a
task similar to:

```bash
backlog task create "Implement Research Workspace literature work products MVP" \
  --doc Docs/Product/Research_Workspace_Literature_Workproducts_PRD.md \
  --doc Docs/superpowers/plans/2026-05-30-research-workspace-literature-workproducts-plan.md \
  --plain
```

Expected: Task exists and is `In Progress` before any code/test file edits.

- [x] **Step 3: Record scope notes**

Add notes that MVP is Research Workspace only and Deep Research work is deferred
to Stage 6 follow-up tasks.

## Stage 0: Shared Work-Product Foundation

**Goal:** Add shared template IDs, template availability metadata, source coverage types, and tests without changing generation behavior.

**Success Criteria:**

- Four templates exist and are covered by tests.
- Existing templates keep their IDs, labels, ordering, and behavior.
- Existing planned templates remain planned/unavailable unless they have an implemented generation strategy.
- New Literature Review templates are categorized and actionable.
- Generated artifacts can carry source coverage metadata.

**Tests:**

- `apps/packages/ui/src/workspace-templates/__tests__/work-product-templates.test.ts`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkProductTemplateChooser.test.tsx`

- [x] **Step 1: Add failing template tests**

Add expectations that `WORK_PRODUCT_TEMPLATES` includes:

```ts
expect(templateIds).toEqual(
  expect.arrayContaining([
    "literature_matrix",
    "corpus_gap_finder",
    "evidence_bound_hypotheses",
    "research_proposal_pack"
  ])
)
```

Also assert:

- Literature Matrix maps to `data_table` and requires 2 sources.
- Corpus Gap Finder maps to `data_table` or `report` and requires 2 sources.
- Evidence-Bound Hypotheses maps to `report` and requires 2 sources.
- Research Proposal Pack maps to `report` and requires 2 sources.
- Every new template has at least three review checklist items.
- Every new template has `category === "literature_review"`.
- Every new template has `availability === "actionable"`.
- Every new template has a non-empty `generationStrategy`.
- Existing roadmap templates keep their current IDs and remain planned if they
  still lack a generation strategy.

- [x] **Step 2: Add failing chooser tests for availability**

Update `WorkProductTemplateChooser.test.tsx` to assert:

- `executive_brief` remains enabled with 1 selected source.
- the four new Literature Review templates become enabled when selected-source
  requirements are met.
- existing planned templates such as Research Dossier, Competitive Market Memo,
  and Technical Project Spec remain visible but unavailable until their
  strategies are implemented.
- unavailable buttons explain whether the blocker is source count, generation
  in progress, or planned availability.

- [x] **Step 3: Add failing source coverage type tests**

Add a focused test or type assertion around `GeneratedArtifact` fixtures used by
Studio tests so a work product artifact can include:

```ts
sourceCoverage: {
  selectedSourceIds: ["source-a", "source-b"],
  usableSources: [{ sourceId: "source-a", mediaId: 1, title: "Paper A" }],
  skippedSources: [
    { sourceId: "source-b", reason: "missing_text", title: "Paper B" }
  ],
  truncatedSources: [],
  sourceContextCharLimit: {
    perSource: 6000,
    total: 18000
  },
  minimumUsableSourcesMet: false
}
```

- [x] **Step 4: Run template/chooser tests and confirm failure**

Run:

```bash
cd apps/packages/ui
bun run test -- src/workspace-templates/__tests__/work-product-templates.test.ts \
  src/components/Option/ResearchWorkspace/__tests__/WorkProductTemplateChooser.test.tsx
```

Expected: FAIL because the new template IDs, availability fields, and source
coverage types do not exist.

- [x] **Step 5: Add template IDs, availability fields, and source coverage types**

Modify:

- `apps/packages/ui/src/workspace-templates/types.ts`
- `apps/packages/ui/src/workspace-templates/work-product-templates.ts`
- `apps/packages/ui/src/types/workspace.ts`

Add:

- `WorkProductTemplateCategory = "general" | "literature_review" | ...`
- `WorkProductTemplateAvailability = "actionable" | "planned" | "disabled"`
- `WorkProductGenerationStrategy` literals for implemented paths.
- `minUsableSources` on templates.
- `sourceCoverage` on `GeneratedArtifact`.

Keep the default template as `executive_brief`. Do not enable existing roadmap
templates unless their generation strategy is implemented.

- [x] **Step 6: Update the chooser to use availability fields**

Modify:

- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/WorkProductTemplateChooser.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx`

Remove hard-coded `template.id === "executive_brief"` actionability checks.
Instead:

- enable templates only when `availability === "actionable"`;
- show planned templates as visible but disabled;
- group or label Literature Review templates clearly;
- preserve source-count gating.

- [x] **Step 7: Run template/chooser tests and confirm pass**

Run:

```bash
cd apps/packages/ui
bun run test -- src/workspace-templates/__tests__/work-product-templates.test.ts \
  src/components/Option/ResearchWorkspace/__tests__/WorkProductTemplateChooser.test.tsx
```

Expected: PASS.

- [x] **Step 8: Commit Stage 0**

```bash
git add apps/packages/ui/src/workspace-templates/types.ts \
  apps/packages/ui/src/workspace-templates/work-product-templates.ts \
  apps/packages/ui/src/workspace-templates/__tests__/work-product-templates.test.ts \
  apps/packages/ui/src/types/workspace.ts \
  apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/WorkProductTemplateChooser.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkProductTemplateChooser.test.tsx
git commit -m "feat: add literature work product templates"
```

## Stage 1: Literature Matrix

**Goal:** Generate a structured comparison table across selected sources.

**Success Criteria:**

- Literature Matrix is actionable when at least 2 selected sources are ready and
  generation proceeds only when at least 2 usable source contexts remain after
  extraction/truncation.
- Generation returns a table-like artifact with source lineage.
- The artifact records selected, usable, skipped, and truncated source coverage.
- Unknown attributes are represented as unknown, not invented.
- JSON validation failure preserves a useful error state.

**Tests:**

- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx`
- Existing `StudioPane.stage2.test.tsx` if the repo prefers extending current tests.

- [x] **Step 1: Write failing UI tests for source-count gating**

Test cases:

- 0 selected sources: Literature Matrix disabled with source requirement.
- 1 selected source: Literature Matrix disabled with "requires 2 selected sources".
- 2 selected sources: Literature Matrix enabled.
- 2 selected sources but only 1 usable source context: generation fails before
  model call with a useful coverage error.

- [x] **Step 2: Write failing generation test**

Mock selected source text and chat completion returning JSON:

```json
{
  "rows": [
    {
      "source": "Paper A",
      "year_or_date": "2024",
      "research_question_or_scope": "Question A",
      "methodology": "Survey",
      "sample_corpus_or_setting": "240 users",
      "primary_finding": "Finding A",
      "limitations": "Small sample",
      "future_work": "Replicate",
      "contradictions_or_tension": "Tension with Paper B",
      "evidence_references": ["Source 1"],
      "confidence": "medium"
    }
  ]
}
```

Assert generated artifact:

- `type === "data_table"`
- `templateId === "literature_matrix"`
- `sourceLineage` exists for audit/review
- `sourceCoverage.usableSources` includes only source contexts sent to the model
- `sourceCoverage.skippedSources` records selected sources without usable text
- `data.table.headers` is populated
- review checklist exists

- [x] **Step 3: Run tests and confirm failure**

Run:

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
```

Expected: FAIL because template-specific generation is missing.

- [x] **Step 4: Create literature work-products helper**

Create:

- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/literature-workproducts.ts`

Responsibilities:

- template-specific system prompts;
- user prompt builders;
- required table headers;
- strict JSON extraction and validation helpers;
- table normalization helpers that turn validated JSON into `data.table`;
- source coverage builder for used/skipped/truncated source contexts;
- artifact title helpers;
- source-lineage/review metadata helper if not already generic;
- compatibility helper for later stages.

Keep this file pure where possible.

- [x] **Step 5: Wire Literature Matrix generation**

Modify:

- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`

Add a template-specific branch for `literature_matrix` that:

- loads selected source contexts and source coverage;
- blocks generation unless at least 2 usable contexts remain;
- asks for strict JSON using `response_format: { type: "json_object" }`;
- parses into `data.table`;
- stores readable markdown/table content, lineage, and source coverage;
- fails with a clear error when no usable table is returned.

- [x] **Step 6: Run focused tests**

Run:

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx \
  src/workspace-templates/__tests__/work-product-templates.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit Stage 1**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx \
  apps/packages/ui/src/workspace-templates
git commit -m "feat: generate literature matrix work product"
```

## Stage 2: Corpus Gap Finder

**Goal:** Generate source-grounded corpus gaps from selected sources, optionally using an existing Literature Matrix artifact.

**Success Criteria:**

- Corpus Gap Finder is actionable with at least 2 selected sources.
- Generation proceeds only when at least 2 usable source contexts remain after
  extraction/truncation.
- Output distinguishes source-stated gaps from inferred gaps.
- Output includes gap type, evidence basis, sources, missing area, confidence, and follow-up question.
- Existing Literature Matrix artifact can be used as optional context only when
  it is completed, newest, same workspace, and source-compatible.

**Tests:**

- Add cases to `StudioPane.literature-workproducts.test.tsx`.

- [ ] **Step 1: Add failing tests for gap generation**

Mock a JSON response with fields:

- Gap
- Gap type
- Evidence basis
- Sources
- Missing area
- Why it matters
- Confidence
- Suggested follow-up question

Assert:

- `templateId === "corpus_gap_finder"`
- `sourceCoverage.usableSources.length >= 2`;
- high-confidence gaps include source basis;
- single-source inferred gaps are marked low or limited confidence.

- [ ] **Step 2: Run tests and confirm failure**

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
```

Expected: FAIL.

- [ ] **Step 3: Add gap prompt and schema helpers**

Modify:

- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/literature-workproducts.ts`

Add:

- `buildCorpusGapPrompt(...)`
- required gap table headers
- strict gap JSON schema/validator
- normalization for known gap types:
  - `unanswered_question`
  - `underrepresented_population`
  - `underrepresented_context`
  - `unused_method`
  - `weak_or_conflicting_evidence`
  - `missing_comparison`
  - `future_work_pattern`

- [ ] **Step 4: Wire gap generation**

Modify:

- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`

Generation should include selected usable sources plus optional Literature Matrix
content only when a compatible matrix is present in `generatedArtifacts`.
Compatibility requires:

- `status === "completed"`;
- `templateId === "literature_matrix"`;
- same workspace if workspace identity is available;
- matrix usable source IDs are the same as or a subset of the current usable
  source IDs.

- [ ] **Step 5: Run focused tests**

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit Stage 2**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
git commit -m "feat: generate corpus gap work product"
```

## Stage 3: Evidence-Bound Hypothesis Generator

**Goal:** Generate testable hypotheses from selected sources and optional matrix/gap artifacts.

**Success Criteria:**

- Hypotheses are testable statements.
- Each hypothesis includes supporting findings, supporting sources, prediction,
  suggested methodology, validity risks, and confidence.
- Speculative hypotheses are visibly marked low confidence.
- The artifact includes stress-test notes.
- Optional Matrix/Gap context follows the same compatibility rules as Stage 2.

**Tests:**

- Add cases to `StudioPane.literature-workproducts.test.tsx`.

- [ ] **Step 1: Add failing hypothesis tests**

Mock JSON output containing:

```json
{
  "hypotheses": [
    {
      "hypothesis": "...",
      "supporting_findings": ["..."],
      "supporting_sources": ["Source 1", "Source 2"],
      "prediction": "...",
      "suggested_methodology": "...",
      "threats_to_validity": ["..."],
      "what_would_falsify_it": "...",
      "confidence": "medium"
    }
  ]
}
```

Assert:

- `templateId === "evidence_bound_hypotheses"`
- `type === "report"`
- review checklist exists
- source lineage and source coverage exist
- no hypothesis without source basis is treated as high confidence.

- [ ] **Step 2: Run tests and confirm failure**

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
```

Expected: FAIL.

- [ ] **Step 3: Add hypothesis prompt helpers**

Modify:

- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/literature-workproducts.ts`

Prompt must require:

- testable statement;
- evidence basis;
- prediction;
- method;
- risks;
- falsification/stress-test notes.
- strict JSON output with a `hypotheses` array.

- [ ] **Step 4: Wire hypothesis generation**

Modify:

- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`

Include optional Matrix and Gap Finder artifacts when present. Do not require
them. Do not silently include incompatible prior artifacts.

- [ ] **Step 5: Run focused tests**

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit Stage 3**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
git commit -m "feat: generate evidence-bound hypotheses"
```

## Stage 4: Research Proposal Pack

**Goal:** Generate a structured proposal draft from selected sources and optional prior literature work products.

**Success Criteria:**

- Proposal includes required PRD sections.
- Evidence, inference, and proposed work are separated.
- Source audit is included.
- Source coverage notes are included.
- User can save or export the result through existing artifact actions.
- Optional prior work products follow source-compatibility rules.

**Tests:**

- Add cases to `StudioPane.literature-workproducts.test.tsx`.
- Extend export/save tests only if current coverage does not exercise report artifacts.

- [ ] **Step 1: Add failing proposal tests**

Mock markdown with headings:

- Title
- Research Question
- Literature Overview
- Evidence Matrix Summary
- Identified Gaps
- Proposed Hypothesis
- Methodology
- Expected Results Or Decision Value
- Contribution
- Risks And Limitations
- Source Audit

Assert:

- `templateId === "research_proposal_pack"`
- `type === "report"`
- source lineage and source coverage exist
- review checklist exists
- output contains Source Audit section.

- [ ] **Step 2: Run tests and confirm failure**

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
```

Expected: FAIL.

- [ ] **Step 3: Add proposal prompt helper**

Modify:

- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/literature-workproducts.ts`

The prompt must:

- mark source-grounded claims;
- mark proposed work separately;
- include source audit;
- avoid publication-ready claims.
- include source coverage notes and list which prior artifacts, if any, were
  used as context.

- [ ] **Step 4: Wire proposal generation**

Modify:

- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`

Use selected usable sources plus optional Matrix, Gap Finder, and Hypothesis
artifacts when compatible. Keep the proposal artifact valid even when no prior work
products exist.

- [ ] **Step 5: Verify save/export behavior**

Run existing artifact export tests relevant to report artifacts. If none exist,
add a focused test that the proposal artifact appears in the artifact list and
uses existing view/download actions.

- [ ] **Step 6: Run focused tests**

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit Stage 4**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
git commit -m "feat: generate research proposal work product"
```

## Stage 5: Polish, Export, And Regression Coverage

**Goal:** Make the work products discoverable, exportable, and stable across source counts and mobile/desktop layouts.

**Success Criteria:**

- Literature Review work-product group is understandable and not crowded.
- Table-like artifacts render as tables when structured data is present.
- Data-table artifacts export CSV and JSON in MVP.
- XLSX is not shown unless the artifact has a server-backed File Artifacts ID
  that can support it.
- Report artifacts preserve existing markdown/text export and save-to-note behavior.
- JSON validation and generation errors are actionable.
- Mobile Studio remains usable.

**Tests:**

- `StudioPane.literature-workproducts.test.tsx`
- Existing Research Workspace layout/mobile tests as needed.

- [ ] **Step 1: Add discoverability tests**

Assert the Studio work-product chooser shows a Literature Review grouping or
clear labels/descriptions for the four new templates.

- [ ] **Step 2: Add parse-failure test**

Mock invalid JSON for Literature Matrix or Gap Finder. Assert the artifact fails
cleanly with a validation error and does not mark the artifact completed with
invented table data.

- [ ] **Step 3: Add source-lineage regression test**

Assert every completed new artifact has `sourceLineage` for selected sources.
Assert every completed new artifact also has `sourceCoverage` describing usable,
skipped, and truncated sources.

- [ ] **Step 4: Add export-scope tests**

Assert:

- Literature Matrix and Corpus Gap Finder expose CSV export from table data.
- JSON export is available when structured data exists.
- XLSX is absent unless a server-backed artifact ID/export capability exists.
- Proposal Pack preserves existing markdown/text download and save-to-note
  behavior.

- [ ] **Step 5: Run Research Workspace focused suite**

```bash
cd apps/packages/ui
bun run test -- src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx \
  src/workspace-templates/__tests__/work-product-templates.test.ts
```

Expected: PASS.

- [ ] **Step 6: Run formatting/type checks used by the package**

Use package-local commands that already exist in `apps/packages/ui/package.json`.
If a command has known unrelated baseline failures, record them explicitly.

- [ ] **Step 7: Commit Stage 5**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace \
  apps/packages/ui/src/workspace-templates
git commit -m "test: cover literature work product regressions"
```

## Stage 6: Later Deep Research Integration Plan

**Goal:** Define but do not implement the post-MVP bridge to Deep Research.

**Success Criteria:**

- Follow-up task(s) exist for Deep Research integration.
- MVP code does not depend on Deep Research.
- Integration contract identifies launch seed, bundle import, and verification
  display paths.

- [ ] **Step 1: Create follow-up Backlog tasks**

Suggested follow-ups:

- Launch Deep Research from Literature Matrix or Gap Finder.
- Seed `ResearchRunCreateRequest.follow_up` from a hypothesis or proposal.
- Import Deep Research `bundle.json` back into Research Workspace artifacts.
- Display Deep Research verification summaries next to proposal sections.

- [ ] **Step 2: Link follow-ups from TASK-486 or the implementation parent**

Use Backlog.md MCP or CLI. Do not manually edit task files unless MCP/CLI is not
available.

- [ ] **Step 3: Commit task/documentation updates**

```bash
git add backlog/tasks Docs/Product/Research_Workspace_Literature_Workproducts_PRD.md \
  Docs/superpowers/plans/2026-05-30-research-workspace-literature-workproducts-plan.md
git commit -m "docs: plan research workspace literature work products"
```

## Verification Checklist

Before opening a PR for implementation:

- [ ] Focused Vitest suites pass.
- [ ] Relevant package type/lint command has been run or documented as skipped
      with reason.
- [ ] No unrelated dirty files are staged.
- [ ] Backlog task includes final summary and verification notes.
- [ ] Bandit is run for touched Python code, or explicitly skipped for UI/docs-only
      work.
- [ ] PR body includes a human-owned Change summary if materially AI-authored.
