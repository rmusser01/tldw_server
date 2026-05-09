# Persona Visual Ownership Copy Design

Date: 2026-05-09
Status: Approved for implementation planning
Owner: Codex brainstorming pass
Backlog: TASK-189
GitHub: #1429

## Summary

This lightweight PRD/spec clarifies the product language for Persona/Buddy visual packs.

The existing Persona Visuals implementation already supports persona-scoped packs, manifest-backed assets, activation, import preview/commit, export, generation readiness, and generated-candidate review. The remaining gap is explanatory: users need to understand what a pack belongs to, what "active" means, and why imported or generated visual assets still pass through review before the Buddy renderer uses them.

This sprint adds concise UI and docs copy. It does not add duplicate-to-persona, shared visual libraries, marketplace behavior, or VN/CYOA semantics.

## Problem

Persona Visuals now has several product concepts that are technically correct but easy to misread:

- A pack is attached to one persona by default, but the manifest format is intentionally portable.
- "Active" means the Persona Buddy renderer uses that pack now; inactive packs still exist and can be edited.
- Import preview is inspection and validation, not a commit.
- Import commit creates or updates a pack only after the review step.
- Generated candidates are not applied until accepted.
- Export creates a portable archive, not a shared library entry.

Without explicit copy, a user can reasonably infer that assets are global, that an imported pack immediately becomes active, or that generation bypasses review.

## Goals

- Explain that Persona Visuals assets are user-owned.
- Explain that visual packs are attached to the selected persona by default.
- Explain that packs are manifest-based so future duplicate, import/export, and shared-library workflows can reuse the same format.
- Clarify active pack versus available/editable pack behavior.
- Clarify import preview, import commit, export archive, and generated-candidate review semantics.
- Keep the copy concise and aligned with existing Persona/Buddy terminology.

## Non-Goals

- Do not implement duplicate-to-persona.
- Do not implement shared visual libraries.
- Do not implement marketplace or public sharing behavior.
- Do not change storage ownership, RBAC, or backend data models.
- Do not change Persona Visuals pack activation behavior.
- Do not change generated-candidate review behavior.
- Do not mix this with VN Play, CYOA, or story scene asset semantics.

## Users And Jobs

Primary user: a local/self-hosted user configuring the visible Persona Buddy for a selected persona.

User jobs:

- "Help me understand where this visual pack will be used."
- "Let me safely import or generate assets without accidentally replacing my active Buddy."
- "Let me tell the difference between a pack that exists and the pack currently being rendered."
- "Let me understand why the pack format is portable even though the current pack belongs to this persona."

## Product Model

Use these exact model statements across UI/docs:

- Assets are user-owned.
- A visual pack is attached to one persona by default.
- A pack is stored as a manifest plus referenced assets.
- The active pack is the pack currently used by the Persona Buddy renderer.
- Other packs remain available for editing, export, activation, or archival.
- Imported packs and generated candidates require review before they affect the active Buddy.
- Portable pack archives preserve the manifest format for import/export and future duplicate/shared-library workflows.

Avoid these implications:

- Do not imply that a pack is global today.
- Do not imply that import preview mutates the active Buddy.
- Do not imply that import commit automatically activates a pack unless the existing implementation actually does so.
- Do not imply that shared libraries exist today.
- Do not imply that VN/CYOA asset packs use this same product surface.

## UI Design

Add one compact help block near the top of `VisualPackEditor`, close to the pack selector/header.

Recommended title:

> How Persona Visual packs work

Recommended body:

> Assets are user-owned and this pack is attached to {personaName} by default. Packs are stored as manifests with referenced assets, so they can be edited, exported, imported, and later duplicated or shared without changing the core format. The active pack is the one Persona Buddy renders now; other packs stay available for editing or review.

The copy can be split into two short paragraphs if the existing layout reads better that way.

### Import/Export Copy

Near import preview/commit controls, use language that distinguishes stages:

- Import preview validates a portable pack archive before it changes this persona.
- Commit import creates or updates a reviewed pack for this persona.
- Export downloads a portable archive for this pack; it does not publish it to a shared library.

### Generation Review Copy

Near generated candidates, use language that preserves the review model:

- Generated candidates stay in review until accepted.
- Accepting a candidate updates this pack's manifest/assets; activation remains the explicit pack-level action.

## Documentation Design

Add or update a Persona/Buddy visual-pack documentation section. The doc should repeat the same product model and scope statements:

- Persona/Buddy visual packs only.
- User-owned assets.
- One-persona attachment by default.
- Manifest-backed pack structure.
- Active versus available packs.
- Import preview/commit/review.
- Generated-candidate review.
- Export as portable archive, not shared library.
- Future duplicate/shared-library workflows are format-compatible but not implemented in this sprint.

The doc should link or point operators toward the existing Persona Visuals editor behavior, not introduce a parallel workflow.

## Acceptance Criteria

- The Persona Visuals editor contains concise copy explaining user-owned assets and default persona attachment.
- The editor copy explains active pack versus available/editable pack behavior.
- The editor copy clarifies import preview/commit, export archive, and generated-candidate review semantics.
- Docs contain the same model language and explicitly exclude VN/CYOA scope.
- Focused tests or docs checks protect the visible copy from accidental removal.

## Open Questions

None. The work is intentionally copy/docs/test scoped.
