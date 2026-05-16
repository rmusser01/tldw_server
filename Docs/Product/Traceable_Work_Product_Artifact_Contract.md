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

`task_id` in the workspace placement group identifies the workspace task or
product task that owns, receives, or displays the artifact. It is not the same
thing as the generating ACP Task ID unless the workspace task and ACP task are
explicitly the same object. `producer_type` and `producer_id` identify the
generating subsystem and producer record. For `producer_type="acp"`,
`producer_id` should be the ACP Task ID for the promoted deliverable, while
`run_id` and `session_id` identify the concrete attempt and protocol session.
If an artifact is only assigned to a workspace task after generation, keep that
assignment in `task_id` and keep the generating ACP task in `producer_id`.

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

Artifact version history must be represented in workspace history or an
equivalent audit timeline. Creating an artifact, creating a new version,
changing review state, exporting, assigning, archiving, or restoring an artifact
should emit a history event with `artifact_id`, `root_artifact_id`, `version`,
`previous_version_id` when applicable, actor identity, workspace/project/task
placement, and the review or revision reason. Workspace history entries should
link back to the exact artifact version they describe so users can navigate from
the timeline to the reviewed/exported version instead of only to the latest
artifact. Support-safe history views should retain stable IDs and decision
state while redacting prompts, transcripts, local paths, and sensitive tool
payloads.

## Export Mapping

Exports are representations of a work product, not the work product itself.

| Export target | Mapping expectation |
| --- | --- |
| Markdown/HTML/JSON | Preserve source-lineage metadata and artifact version identifiers. |
| CSV/XLSX/data tables | Preserve table schema, source lineage for generated rows/columns, and export job ID. CSV exports should include lineage columns only when they do not corrupt the table contract; otherwise require a sidecar JSON manifest keyed by stable row/column IDs. XLSX exports can use a hidden lineage worksheet or the same sidecar manifest. |
| Slides/DOCX/PDF | Preserve artifact ID/version in metadata or manifest and keep source/citation appendix when possible. |
| Chatbooks | Include artifact manifest, source references, version chain, and export references without assuming all large binaries are bundled. |
| Prompt/output surfaces | Link back to source artifact and version rather than copying detached generated text where possible. |

### Implemented Markdown/HTML/JSON Export Contract

Issue #1705 implements the first accepted-version export contract at
`POST /api/v1/workspaces/{workspace_id}/artifacts/{artifact_id}/exports`.
The request accepts `format: "md" | "html" | "json"` and an optional
`artifact_version_id`; omitting the version exports the current artifact
version. Only `accepted` artifact versions are exportable. Draft, reviewing,
needs-revision, rejected, assigned, archived, or otherwise non-accepted states
fail closed with `workspace_artifact_not_accepted`.

Each export response includes the rendered content, UTF-8 byte count, content
type, generated timestamp, export reference, and metadata preserving workspace
ID, artifact ID, artifact version ID, review state, source lineage, producer
metadata, review metadata, version metadata, and redaction posture. The backend
records the export reference back onto the workspace artifact without creating
a new content version and without replacing existing export references.

This implementation covers portable text representations. File-artifact
materialization, Chatbook packaging, document/slides exports, table-specific
sidecar manifests, retention controls, and support-safe redaction views remain
separate follow-up slices.

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

### Current Storage/API Foundation

Issue #1703 implements the first backend foundation on the existing
`workspace_artifacts` surface rather than introducing an ACP-only artifact
store. The API and database now preserve traceable contract fields for content
envelope, owner scope, source lineage, ACP producer metadata, review metadata,
version metadata, export references, redaction posture, schema version, stable
root artifact ID, artifact version ID, and previous-version links. Artifact
creation writes the initial version record; artifact updates create a new
version record and carry the previous version ID forward.

This storage/API foundation is now consumed by the ACP promotion, Workspace UI
detail, and accepted-export slices below. It still does not imply that every
generated output is traceable unless that output is represented by this
contract or an explicitly mapped equivalent.

### Current ACP Promotion Foundation

Issue #1706 implements promotion for accepted ACP completion artifacts that are
structured as source-grounded workspace briefs, reports, specs, action plans,
or tables. Promotion writes `producer_type="acp"` metadata with task/run/session
and review-run references, preserves source lineage and redaction posture, maps
accepted reviewer-loop state to `review_state="accepted"`, and updates an
existing artifact as a new version when the same artifact ID is promoted again.

Rejected, needs-revision, malformed, unsupported, or workspace-less ACP outputs
remain execution evidence and are not presented as accepted workspace work
products.

### Current Workspace UI Foundation

Issue #1707 implements the Workspace artifact detail surface for traceable
artifacts. The UI shows review state, ACP provenance, authenticated session and
diagnostics drill-through links, source lineage, version/root/previous-version
metadata, redaction posture, export references, and review-state controls. When
the redaction posture is restricted, the detail view hides ACP provenance and
source-lineage details while keeping safe state labels visible.

### Current Accepted Export Foundation

Issue #1705 implements accepted workspace artifact version exports for Markdown,
HTML, and JSON. The export endpoint renders from the exact accepted artifact
version, embeds or exposes the traceability metadata required by this contract,
and appends a version-specific export reference to the workspace artifact.

This does not complete all export channels. Rich document/slides/table exports,
external file-artifact storage, Chatbook bundling, export retention policies,
and UI-triggered download workflows remain separate follow-up slices.

### Current Release Verification

Issue #1704 records release signoff for the first ACP artifact golden path in
`Docs/Development/ACP_Artifact_Release_Verification_2026_05_15.md`. The focused
verification covers ACP-to-artifact promotion, source lineage, ACP provenance,
redaction, reviewer-loop state mapping, versioning, accepted export identity,
and UI hydration/detail behavior.

This verification is intentionally scoped to accepted structured ACP
deliverables that satisfy the contract. Non-golden-path artifact types, rich
document exports, Chatbook packaging, external file-artifact materialization,
and live downstream-agent certification remain separately tracked work.

## Release Caveats

- Do not claim every generated output is traceable until it is represented by
  this contract or an explicitly mapped equivalent.
- Do not present low-level ACP artifacts as accepted workspace work products.
- Do not expose full prompts, transcripts, tool payloads, local paths, or
  environment values in support-safe artifact views.
- Do not claim export portability unless the export includes artifact identity,
  version, and source-lineage metadata.
