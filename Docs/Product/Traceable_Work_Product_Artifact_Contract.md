# Traceable Work Product Artifact Contract

Status: Product contract for staged implementation, created 2026-05-14
Primary issue: [#1525](https://github.com/rmusser01/tldw_server/issues/1525)
ACP mapping: [#1538](https://github.com/rmusser01/tldw_server/issues/1538)

## Purpose

Generated work products are durable user-facing outputs such as briefs,
reports, specs, tables, slide outlines, extracted action plans, or reviewed
agent deliverables. They are different from low-level run logs, raw model
messages, transient export files, or ACP session artifacts.

The product contract below defines the minimum shape required before generated
outputs can be treated as serious workspace artifacts: they must preserve source
lineage, review state, ownership, version history, export posture, and enough
audit context to explain how the artifact was produced.

## Non-Goals

- This contract does not replace ACP session history, raw file-artifact exports,
  Chatbooks, media derivatives, meeting artifacts, or evaluation artifacts.
- This contract does not require every artifact type to be implemented in one
  release.
- This contract does not create enterprise document-management behavior beyond
  what is needed for source-grounded review, revision, export, and assignment.

## Artifact Classes

| Class | Description | Contract posture |
| --- | --- | --- |
| Work product artifact | Curated generated output intended for a workspace, user, team, or export workflow. | Must follow this contract. |
| Execution artifact | Raw or semi-structured output emitted by a run, session, tool, or model call. | Linked as evidence; promoted only after mapping to this contract. |
| Export artifact | A generated file representation such as Markdown, CSV, JSON, XLSX, HTML, ICS, DOCX, PPTX, or Chatbook output. | References a source work product and export format; may be transient. |
| Diagnostic artifact | Logs, prompts, traces, redacted transcripts, reviewer notes, failure context, or audit evidence. | Linked as support evidence; not presented as polished user work. |

## Minimum Schema

Every work product artifact needs these fields or direct equivalents:

| Field group | Minimum fields |
| --- | --- |
| Identity | artifact_id, artifact_type, title, created_at, updated_at, created_by, owner_id, owner_scope. |
| Workspace placement | `workspace_id`, optional `project_id`, optional `task_id`, optional `source_collection_id`, and optional display location. |
| Generation provenance | `producer_type`, `producer_id`, `run_id`, optional `session_id`, prompt/template IDs, model/provider identifiers when available, and normalized completion reason. |
| Content envelope | `content_type`, `content_ref` or inline payload pointer, preview text, summary, size metadata, and schema version. |
| Source lineage | Source references, citation spans, selected-source snapshot, retrieval/query metadata when used, MCP/tool evidence references, and confidence or coverage notes when available. |
| Review state | `state`, reviewer identifiers or review run IDs, decision timestamps, rejection or revision reason codes, and checklist status. |
| Versioning | `version`, `previous_version_id`, `root_artifact_id`, `revision_reason`, and comparison-safe metadata. |
| Export state | Export format, export job/file IDs, generated-at timestamp, expiration or retention policy, and export errors. |
| Governance and audit | Visibility, redaction posture, retention class, audit event references, and policy decision references when relevant. |

The schema can be implemented as relational columns plus JSON metadata, but API
responses should expose stable names for the groups above so frontend and
external clients do not parse model text or ACP transcripts to reconstruct
artifact state.

## Source Lineage

Source lineage must answer four questions:

1. Which sources were eligible for generation?
2. Which sources were actually used or cited?
3. Which generated sections or claims map back to which source references?
4. Which tools, MCP servers, or agent actions affected the output?

Minimum lineage records should include stable source IDs, display labels,
source type, source revision or timestamp when available, citation spans or
section anchors, and retrieval/tool evidence IDs. When an agent or model
generates uncited content, the artifact should mark that section as generated
without direct citation rather than implying source support.

## Review States

Work product artifacts use these canonical review states:

| State | Meaning | Allowed next states |
| --- | --- | --- |
| `draft` | Generated or manually created but not submitted for review. | `reviewing`, `accepted`, `needs_revision`, `archived`. |
| `reviewing` | Awaiting manual review, reviewer-agent decision, or policy review. | `accepted`, `needs_revision`, `rejected`, `archived`. |
| `accepted` | Approved as a usable workspace work product. | `exported`, `assigned`, `archived`, new version via revision. |
| `needs_revision` | Reviewer or user requested changes; artifact remains visible but not accepted. | new version in `draft` or `reviewing`, `rejected`, `archived`. |
| `rejected` | Not suitable for use; retained only for audit/history if retention allows. | `archived` or new version from source context. |
| `exported` | Accepted or explicitly exported artifact has at least one generated export. | `assigned`, `archived`, new export. |
| `assigned` | Artifact has been routed to an owner, task, workspace, or follow-up. | `accepted`, `needs_revision`, `archived`, new version. |
| `archived` | Hidden from active workspace flows but retained according to policy. | restore to prior state when policy allows. |

Implementations may expose a smaller first-release state set, but they must map
their local states to these canonical states in documentation and API responses.

## Versioning

Each revision creates a new artifact version, not an in-place overwrite. The
new version references `previous_version_id` and keeps `root_artifact_id`
stable. Revision metadata should preserve why the artifact changed, which
sources or prompts changed, whether a reviewer requested the change, and which
exports were generated from which version.

Rejected artifacts can still be retained as versions when audit or reviewer
history matters, but user-facing accepted/exported views should default to the
latest accepted version.

## Export Mapping

Exports are representations of a work product, not the work product itself.

| Export target | Mapping expectation |
| --- | --- |
| Markdown/HTML/JSON | Preserve source-lineage metadata and artifact version identifiers. |
| CSV/XLSX/data tables | Preserve table schema, source lineage for generated rows/columns, and export job ID. |
| Slides/DOCX/PDF | Preserve artifact ID/version in metadata or manifest and keep source/citation appendix when possible. |
| Chatbooks | Include artifact manifest, source references, version chain, and export references without assuming all large binaries are bundled. |
| Prompt/output surfaces | Link back to source artifact and version rather than copying detached generated text where possible. |

## Existing Surface Mapping

| Existing surface | Reuse path |
| --- | --- |
| Workspace Playground generated outputs | Primary golden-path surface for first work product artifacts. It already frames source-grounded briefs, review checklist metadata, and export affordances. |
| ACP Agent Tasks and ACP session artifacts | Use as producer evidence. Promote only structured deliverables that satisfy this contract. |
| File artifacts endpoint | Use for validated export representations and transient generated files; do not treat raw exports as the durable work product record by themselves. |
| Chatbooks | Use as portable packaging for artifact metadata, version chains, and selected export references. |
| Meeting artifacts | Can map meeting summaries/action items into this contract when they become workspace work products. |
| Evaluations/persona trace artifacts | Remain diagnostic/evaluation evidence unless a user explicitly promotes a generated artifact into a workspace work product. |

## First Golden Path

The first implementation slice should use a source-grounded workspace brief:

1. User selects workspace sources in Workspace Playground.
2. User generates an executive brief or technical spec.
3. Backend stores a work product artifact with source snapshot, citations,
   prompt/template provenance, review state, and version metadata.
4. UI shows the artifact detail with source lineage, preview, review checklist,
   revision controls, and export actions.
5. Export jobs produce Markdown/HTML/JSON first, with richer document/slides
   exports added later.

This golden path is narrow enough to validate the contract while leaving
template-specific artifacts, enterprise routing, and broader export formats as
separate follow-ups.

## Implementation Slices

1. Storage and API: define artifact tables or typed records, stable schemas,
   ownership, versioning, review state transitions, and export references.
2. Workspace UI: add artifact detail, review checklist, version comparison,
   source-lineage display, and export controls.
3. ACP mapping: promote selected ACP deliverables into work product artifacts
   while preserving ACP session/run/review evidence.
4. Export adapters: map accepted artifact versions to file-artifact exports,
   Chatbooks, docs/slides/tables, and prompt/output surfaces.
5. Verification: add contract tests, source-lineage fixtures, UI regression
   coverage, and redaction/retention checks.

## Release Caveats

- Do not claim every generated output is traceable until it is represented by
  this contract or an explicitly mapped equivalent.
- Do not present low-level ACP artifacts as accepted workspace work products.
- Do not expose full prompts, transcripts, tool payloads, local paths, or
  environment values in support-safe artifact views.
- Do not claim export portability unless the export includes artifact identity,
  version, and source-lineage metadata.
