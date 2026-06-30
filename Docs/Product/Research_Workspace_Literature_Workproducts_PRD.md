# Research Workspace Literature Work Products PRD

Date: 2026-05-30
Status: Draft for review
Backlog: TASK-486

## Summary

Add a coherent set of literature-review work products to Research Workspace:

1. Literature Matrix
2. Corpus Gap Finder
3. Evidence-Bound Hypothesis Generator
4. Research Proposal Pack

The MVP is Research Workspace first. A user selects ready sources in the existing
workspace, generates source-grounded work products in Studio, reviews source
coverage, source lineage, and confidence signals, and exports or saves the
resulting artifacts.
Deep Research integration is a later-stage expansion and must not block first
value.

The product direction is inspired by current research-assistant tools that turn
multi-paper review into structured tables, gap analysis, hypotheses, and proposal
drafts. The tldw_server version should remain provider-agnostic, local-first,
auditable, and source-bound.

## Problem

Research Workspace already lets users collect sources, chat against selected
sources, and generate generic outputs such as summaries, reports, comparison
notes, data tables, mind maps, slides, quizzes, and audio summaries. Those
outputs are useful, but literature-review work still requires users to manually
compose a workflow:

- compare papers one at a time;
- extract method, sample, findings, limitations, and future work by hand;
- infer cross-source gaps from free-form output;
- turn gaps into testable hypotheses;
- assemble proposal drafts while preserving citations and uncertainty.

The missing product layer is not another generic artifact button. It is a small
set of named research work products with predictable structures, source lineage,
and review expectations.

## Goals

- Make literature review a first-class Research Workspace workflow.
- Ship MVP value without adding a new long-running Deep Research dependency.
- Reuse selected workspace sources, source readiness, existing Studio artifact
  generation, document insights, Data Table rendering, File Artifacts exports,
  and workspace notes where practical.
- Produce structured artifacts that are inspectable, regenerable, exportable,
  and tied to source lineage.
- Make the difference between selected sources, usable sources, and skipped
  sources visible before the user trusts an artifact.
- Separate direct evidence, synthesis, uncertainty, and recommendations.
- Keep the four work products composable: matrix feeds gaps, gaps feed
  hypotheses, and hypotheses feed proposal drafts.
- Leave a clean later path into Deep Research sessions and bundles.

## Non-Goals

- Do not merge `/research` and Research Workspace.
- Do not require a Deep Research run to generate the MVP artifacts.
- Do not create a parallel workspace model or parallel artifact store.
- Do not add broad automatic paper discovery in the MVP. Users provide or select
  sources first.
- Do not promise publication-ready academic writing without human review.
- Do not present uncited claims as facts.
- Do not build a full citation manager replacement in this slice.

## Target Users

- Students and researchers reviewing 5 to 30 papers or documents.
- Independent learners comparing studies, technical reports, or policy sources.
- Product and engineering teams synthesizing research papers, standards, specs,
  incident writeups, or market evidence.
- Users migrating from NotebookLM-like source notebooks who expect source-bound
  chat plus generated research artifacts.

## Product Boundary

Research Workspace is the MVP surface:

- User selects sources in the Research Workspace source pane.
- Studio exposes the literature work products.
- Generated artifacts remain workspace artifacts with source lineage.
- Quick Notes can capture drafts, review notes, and next actions.
- Data-table-like outputs can render as structured tables and export through
  existing file-artifact paths where available.

Deep Research is a later integration surface:

- A workspace can later launch a Deep Research run seeded by selected sources,
  a matrix, gaps, hypotheses, or a proposal draft.
- A completed Deep Research bundle can later be imported back into the workspace
  as supporting evidence.
- Deep Research checkpointing, source review, verification summary, and bundle
  contracts are not MVP prerequisites.

## Current Repo Anchors

Existing foundations this PRD should reuse:

- Research Workspace UI:
  - `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/WorkProductTemplateChooser.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/ArtifactModalContent.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactExport.tsx`
- `apps/packages/ui/src/workspace-templates/work-product-templates.ts`
- `apps/packages/ui/src/types/workspace.ts`
- Existing artifact types:
  - `compare_sources`
  - `data_table`
  - `report`
  - `mindmap`
  - `audio_overview`
  - `slides`
- Document analysis:
  - `tldw_Server_API/app/api/v1/endpoints/media/document_insights.py`
  - `tldw_Server_API/app/api/v1/endpoints/media/document_references.py`
- Data table and export support:
  - `tldw_Server_API/app/core/File_Artifacts/adapters/data_table_adapter.py`
  - `tldw_Server_API/app/core/File_Artifacts/adapters/markdown_table_adapter.py`
  - `Docs/Product/PRD_file_artifacts_endpoint.md`
- Deep Research later-stage anchors:
  - `tldw_Server_API/app/core/Research/`
  - `tldw_Server_API/app/api/v1/endpoints/research_runs.py`
  - `Docs/Plans/2026-03-07-deep-research-module-roadmap.md`
  - `Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md`

## Product Principles

### Work Products Before Output Types

The user should choose "Literature Matrix" or "Research Proposal Pack", not
guess which generic output type to use. Generic output types remain the rendering
and export mechanisms.

### Source Bound By Default

Every generated artifact must store selected source lineage. Claims,
contradictions, gaps, hypotheses, and proposal assertions should include source
references or explicit uncertainty.

### Source Coverage Before Trust

Lineage must not imply that every selected source was used by the model. Each
artifact should distinguish selected sources, usable source contexts included in
generation, skipped sources, and truncation or extraction limits.

### MVP Uses Selected Sources

MVP scope starts after source selection. The product does not need to discover or
ingest papers automatically before the first version is useful.

### Structured First, Prose Second

Tables and typed sections should be canonical for matrix, gaps, and hypotheses.
Free-form prose is appropriate for the proposal pack, but it must be generated
from structured evidence where possible.

### JSON First For Typed Artifacts

For Literature Matrix, Corpus Gap Finder, and Evidence-Bound Hypotheses, strict
JSON should be the primary generation contract. Markdown tables are a rendering
or fallback format, not the canonical parse target.

### Composable But Not Magical

Each work product can be generated independently from selected sources. When
prior generated artifacts exist, later work products may use them as additional
context, but they must still remain grounded in selected sources.

Prior artifacts are compatible only when they are completed, generated in the
same workspace, and their usable source set is the same as or a subset of the
currently selected usable sources. Otherwise they may be shown as available
context but must not be silently included.

## MVP User Flow

1. User opens Research Workspace.
2. User adds or selects 2 to 30 ready sources.
3. User opens Studio and sees a "Literature Review" work-product group.
4. User chooses one of the four work products.
5. System generates an artifact with:
   - title;
   - type;
   - structured content;
   - source coverage summary;
   - source lineage;
   - review checklist;
   - confidence and missing-evidence notes where available.
6. User reviews, regenerates, exports, saves to notes, or uses the artifact as
   context for a later work product.

## Feature Track 1: Literature Matrix

### Purpose

Turn selected sources into a structured comparison table that supports fast
cross-paper review.

### Inputs

- Minimum selected sources: 2.
- Recommended selected sources: 5 to 30.
- Source types: PDF, document, text, website, audio/video transcript when text
  extraction is available.
- Optional existing per-document insights when available.

### Output

Primary rendering: structured table.

Required columns:

- Source
- Year or date
- Research question or scope
- Methodology
- Sample, corpus, or setting
- Primary finding
- Limitations
- Future work
- Contradictions or tension with other sources
- Evidence references
- Confidence or extraction status

MVP export target: CSV from the table viewer and JSON from the structured data
payload where available. XLSX is later-stage unless the artifact is backed by a
server-side File Artifacts `data_table` object with an output/download ID.

### Acceptance Criteria

- With 2 or more selected ready sources, user can select Literature Matrix.
- Generation requires at least 2 usable source contexts after extraction and
  truncation, not merely 2 selected cards.
- Each row is traceable to one source context that was actually included in
  generation.
- Empty or unavailable attributes are marked as unknown, not invented.
- Contradictions reference at least one other source when available.
- The artifact stores source coverage for selected, used, skipped, and truncated
  sources.
- The artifact can be viewed as a table, not only raw markdown.

### MVP Notes

MVP should use the existing Studio direct-source generation pattern but request
strict JSON with a `rows` array and required fields. The client can render that
JSON as a table and optionally derive markdown/CSV. A later backend pass can move
matrix extraction into a typed endpoint.

## Feature Track 2: Corpus Gap Finder

### Purpose

Identify what the selected corpus does not cover: unanswered questions,
underrepresented populations or contexts, methodological gaps, missing
comparisons, weak evidence areas, and repeated future-work requests.

### Inputs

- Minimum selected sources: 2.
- Strongly benefits from Literature Matrix output when present.
- Optional per-document `research_gap`, `limitations`, and `future_work`
  insights.

### Output

Primary rendering: structured gap table plus short summary.

Required fields:

- Gap
- Gap type
- Evidence basis
- Sources mentioning or implying the gap
- Missing population, context, method, data, or comparison
- Why it matters
- Confidence
- Suggested follow-up question

Gap types:

- unanswered_question
- underrepresented_population
- underrepresented_context
- unused_method
- weak_or_conflicting_evidence
- missing_comparison
- future_work_pattern

### Acceptance Criteria

- With 2 or more selected ready sources, user can select Corpus Gap Finder.
- Generation requires at least 2 usable source contexts after extraction and
  truncation.
- Each gap lists the source basis or marks itself as synthesis with low
  confidence.
- Future-work recommendations from sources are distinguished from model-inferred
  gaps.
- The artifact makes uncertainty visible.
- The output can be reused by the Hypothesis Generator.

### MVP Notes

MVP should prefer conservative wording. A gap is stronger when multiple sources
mention similar limitations or future work; single-source gaps should be marked
as limited evidence.

## Feature Track 3: Evidence-Bound Hypothesis Generator

### Purpose

Generate testable hypotheses that follow from selected sources and identified
gaps, while making evidence basis and testing risks explicit.

### Inputs

- Minimum selected sources: 2.
- Optional Literature Matrix artifact.
- Optional Corpus Gap Finder artifact.
- Optional user-selected gap or focus question.

### Output

Primary rendering: structured hypothesis list.

Required fields:

- Hypothesis
- Supporting findings
- Supporting sources
- Prediction
- Suggested methodology
- Data or population needed
- Threats to validity
- Feasibility
- Confidence

### Acceptance Criteria

- Generation requires at least 2 usable source contexts after extraction and
  truncation.
- Generated hypotheses are phrased as testable statements.
- Each hypothesis has at least one explicit supporting source or is marked as
  speculative and low confidence.
- Methodology suggestions are separated from evidence.
- Testing challenges are visible before export.
- User can regenerate with a narrower focus question.
- Prior Matrix or Gap artifacts are included only when their source coverage is
  compatible with the current usable source set.

### MVP Notes

MVP does not need a full debate or adversarial review mode. Instead, include a
"stress-test notes" section listing assumptions, likely confounders, and what
would falsify the hypothesis.

## Feature Track 4: Research Proposal Pack

### Purpose

Turn selected sources and prior work products into a proposal-like artifact that
can seed a class project, research memo, grant outline, or internal study plan.

### Inputs

- Minimum selected sources: 2.
- Optional Literature Matrix.
- Optional Corpus Gap Finder.
- Optional selected hypothesis.

### Output

Primary rendering: markdown report, with optional export.

Required sections:

- Title
- Research question
- Literature overview
- Evidence matrix summary
- Identified gaps
- Proposed hypothesis
- Methodology
- Expected results or decision value
- Contribution
- Risks and limitations
- Source audit and bibliography notes

### Acceptance Criteria

- Generation requires at least 2 usable source contexts after extraction and
  truncation.
- Proposal claims are source-grounded or explicitly marked as proposed work.
- Evidence, inference, and recommendation sections are visibly separated.
- The proposal includes a source audit section.
- User can save the proposal to workspace notes or export as markdown.
- Later stages can hand off the proposal to Deep Research.

### MVP Notes

MVP should not claim publication readiness. It should produce a structured draft
that helps users review and revise.

## Shared Artifact Metadata

Each generated work product should carry:

- `templateId`
- `artifact.type`
- `sourceLineage`
- `sourceCoverage`
- `reviewChecklist`
- `createdAt`
- `completedAt`
- optional structured `data`
- optional `previousVersionId`
- token/cost metadata when available

Suggested template IDs:

- `literature_matrix`
- `corpus_gap_finder`
- `evidence_bound_hypotheses`
- `research_proposal_pack`

Suggested output type mapping:

- Literature Matrix: `data_table`
- Corpus Gap Finder: `data_table` or `report` with table data
- Evidence-Bound Hypothesis Generator: `report` with structured list data
- Research Proposal Pack: `report`

## Template Catalog Contract

The template catalog should not rely on hard-coded checks such as "only
executive_brief is actionable." Each template should declare:

- `category`: for MVP, the four new templates use `literature_review`.
- `availability`: `actionable`, `planned`, or `disabled`.
- `generationStrategy`: stable identifier for the generation path, for example
  `literature_matrix_json`, `corpus_gap_json`, `hypotheses_json`, or
  `proposal_markdown`.
- `minSelectedSources`: selected-source UI requirement.
- `minUsableSources`: post-extraction generation requirement.
- `outputArtifactType`: existing artifact rendering type.

Existing roadmap templates that do not yet have implemented generation
strategies should stay visible as planned templates. Adding Literature Review
must not accidentally enable Research Dossier, Competitive Market Memo, or
Technical Project Spec unless those strategies are also implemented.

## Source Coverage Contract

Every artifact must record source coverage in addition to source lineage:

- `selectedSourceIds`: source cards selected when generation started.
- `usableSources`: source IDs/media IDs that contributed text to the prompt.
- `skippedSources`: selected sources omitted because text extraction failed,
  the source was unready, or the source exceeded context limits.
- `truncatedSources`: usable sources whose text was clipped before generation.
- `sourceContextCharLimit`: per-source and total character limits used.
- `minimumUsableSourcesMet`: whether the work product met its generation gate.

The UI should surface the coverage summary on each artifact card or in the
artifact detail view. A user should not have to infer whether a missing row was a
model choice, a source-readiness issue, or a context-budget issue.

## Prior Artifact Compatibility

When a work product uses another generated artifact as optional context, the
implementation should select only artifacts that satisfy all of these rules:

- same workspace;
- completed status;
- matching expected `templateId`;
- compatible usable source set;
- newest artifact when multiple compatible candidates exist.

If an artifact is useful but not compatible, show it as a candidate for manual
review rather than silently injecting it into the next prompt.

## Review Checklists

Literature Matrix:

- Every row maps to a usable source context included in generation.
- Unknown values are not filled with guesses.
- Contradictions name the involved source(s).

Corpus Gap Finder:

- Gaps distinguish source-stated gaps from inferred gaps.
- Each high-confidence gap has more than one evidence basis or a strong source.
- Missing population/context/method details are visible.

Hypothesis Generator:

- Hypotheses are testable.
- Predictions and methods are separated from existing findings.
- Confounders and falsification criteria are visible.

Research Proposal Pack:

- Literature claims are cited.
- Proposed work is not presented as established evidence.
- Risks and limitations are visible before export.

## Data And Generation Requirements

MVP generation should:

- use selected source text and titles;
- build source coverage from the same source contexts that are actually sent to
  the model;
- require at least 2 usable source contexts for matrix, gaps, hypotheses, and
  proposal generation;
- ignore instructions embedded in source content;
- refuse to invent missing methods, dates, sample sizes, or findings;
- include citation/source labels in generated rows or sections;
- preserve useful partial output when some selected sources lack text;
- surface missing-content failures clearly.

Typed generation should:

- request strict JSON for Matrix, Gap Finder, and Hypothesis outputs;
- validate required fields before marking the artifact complete;
- render typed data as tables/lists in the UI;
- keep the raw model text only as diagnostic fallback when it is safe to show.

Large corpus behavior:

- Selected source count can be 5 to 30, but MVP generation may use only the
  subset that fits current context limits.
- If any selected source is skipped or truncated, show a coverage warning.
- If fewer than 2 usable source contexts remain after extraction and truncation,
  block generation with a clear message.

Optional enrichment:

- use cached Document Insights when available;
- use Document References for bibliography and citation metadata when available;
- use existing source status to block or warn on unready sources.

## Error Handling

- No selected sources: show the existing source-selection guidance.
- One selected source for matrix/gap/hypothesis: explain that at least two
  sources are required.
- Fewer than 2 usable source contexts after extraction/truncation: explain which
  selected sources lacked usable text and suggest reprocessing or narrowing
  selection.
- Unready sources: warn and allow generation only from ready sources unless the
  user explicitly includes partial content.
- Missing full text: mark source rows as unavailable; do not silently omit them
  from lineage.
- JSON validation failure: keep the failed artifact with a useful error and
  offer regeneration. Show raw generated content only if it is safe and
  source-grounded.
- Export failure: keep the artifact and show export status/error separately.

## UX Requirements

- Add a "Literature Review" work-product group in Studio.
- Keep generic output buttons available but secondary.
- The four work products should expose short descriptions and selected-source
  requirements.
- Disabled states should explain both selected-source requirements and usable
  source-context failures.
- Artifact details should show source coverage and skipped/truncated source
  counts.
- Users should be able to generate work products without first learning Deep
  Research terminology.
- Existing mobile tabs should keep Studio usable on small screens.
- Artifact viewers should show table output as tables when structured data is
  available.

## Later Deep Research Integration

After MVP stabilizes, add integration points:

- launch a Deep Research run from selected sources and a work-product artifact;
- seed `ResearchRunCreateRequest.follow_up` from a gap, hypothesis, or proposal;
- import completed `bundle.json` claims, source inventory, unresolved questions,
  contradictions, and source trust into Research Workspace artifacts;
- show Deep Research verification summaries next to proposal sections;
- allow Deep Research outline checkpoints to use Literature Matrix or Gap Finder
  artifacts as review context.

## Metrics

Track:

- work-product generation started/completed/failed by template ID;
- selected source count per generation;
- usable source count, skipped source count, and truncated source count per
  generation;
- parse success/failure for table-like artifacts;
- JSON validation success/failure for typed artifacts;
- export started/completed/failed;
- artifact regenerate and version creation;
- save-to-note or export conversion;
- Deep Research handoff events in later stages.

## Risks

### Overclaiming

The model may present synthesized gaps or hypotheses as facts.

Mitigation: enforce explicit source basis, confidence, and uncertainty fields.

### Artifact Proliferation

Four new work products can make Studio feel crowded.

Mitigation: group them under Literature Review and keep generic output types
secondary.

### Fragile Table Parsing

Markdown table parsing can fail for complex generated content.

Mitigation: use strict JSON as the MVP canonical format and render markdown,
CSV, or tables from validated structured data. Keep markdown parsing only as a
fallback for generic Data Table output.

### Research Workspace And Deep Research Confusion

Users may not understand why there are two research surfaces.

Mitigation: MVP uses Research Workspace only. Deep Research appears later as
"run a deeper investigation" rather than a prerequisite.

## Rollout

Stage 1: Literature Matrix

- Add template.
- Generate strict JSON and render matrix table.
- Export CSV/JSON in MVP; reserve XLSX for server-backed File Artifacts.

Stage 2: Corpus Gap Finder

- Add gap template.
- Generate strict JSON gap rows and summary.
- Allow reuse of matrix artifact when present.

Stage 3: Evidence-Bound Hypothesis Generator

- Add hypothesis template.
- Use selected usable sources plus compatible optional matrix/gap context.
- Include stress-test notes.

Stage 4: Research Proposal Pack

- Add proposal template.
- Compose selected source evidence plus optional prior artifacts.
- Save/export markdown proposal.

Stage 5: Deep Research Integration

- Launch or seed Deep Research from selected work products.
- Import bundle artifacts back into Research Workspace.

## MVP Exit Criteria

- A user can select several ready sources and generate all four work products
  without leaving Research Workspace.
- Literature Matrix and Gap Finder produce table-like artifacts with source
  lineage.
- Artifacts distinguish selected, usable, skipped, and truncated sources.
- Hypothesis and Proposal artifacts separate evidence from proposed inference.
- Artifacts have review checklists.
- At least one export or save path is available for each artifact.
- Tests cover template visibility, source-count gating, generation success,
  parse-failure handling, and source lineage.
