# Persona Visual Post-Setup Management Design

## Status

Draft design for GitHub issue #1769 and Backlog TASK-406.

## Purpose

The Persona Visual setup path is now mostly present: users can start from bundled
defaults, import archives, create draft packs, save packs to the personal
library, duplicate packs to another persona, queue generation jobs, review
generated candidates, and explicitly activate a pack. The remaining product gap
is management after first setup. The current `VisualPackEditor` exposes the
right operations, but the lifecycle is dense and action-oriented users have to
infer which pack is live, which packs are drafts, which jobs need attention, and
which recovery state matters next.

This design defines a post-setup management UX for Persona Garden Visuals. It is
parallel to PR #1767 and intentionally does not change the recipe-backed
generation request or job-payload design.

## Current System

The existing Persona Visual system already provides:

1. active and available pack listing via `listPersonaVisualPacks()`.
2. draft creation, manifest editing, asset upload, activation, and deactivation.
3. export jobs with status refresh and authenticated archive download.
4. import-preview and import-commit jobs with conflict choices.
5. same-user duplicate-to-persona draft creation.
6. reference-backed personal-library save, edit, remove, and use flows.
7. generated candidate review before manifest changes.
8. generation readiness diagnostics before queueing generation jobs.
9. Buddy runtime fallback and an `Open Visuals` entry point.

The design should keep these backend/API semantics and improve how the existing
Visuals tab frames and sequences them.

## Goals

1. Make the current live Buddy visual pack obvious.
2. Separate post-setup management from first-run setup.
3. Give users a lifecycle view of active, draft, review, archived, and failed
   packs.
4. Put outstanding work in one visible attention path: invalid manifests,
   incomplete drafts, pending jobs, candidates awaiting review, unavailable
   library sources, and failed import/export/generation jobs.
5. Preserve review-first and explicit activation semantics.
6. Keep the first implementation slice mostly frontend and documentation
   oriented, using existing endpoints.

## Non-Goals

1. No overlap with #1765 / PR #1767 recipe-backed generation request fields,
   prompt composition, idempotency, or job payloads.
2. No backend generation orchestration changes.
3. No final default buddy art production.
4. No renderer expansion, Live2D/Rive/Spine adapter, or manifest version bump.
5. No MCP provider execution or resource download.
6. No marketplace, shared public library, or cross-user sharing.
7. No VN/CYOA behavior.
8. No automatic activation after import, duplicate, library use, or generation.

## Recommended UX Model

### Management Header

The Visuals tab should open with a compact management header once setup is no
longer the dominant state. The header should summarize:

1. selected persona name.
2. active pack title or "No active visual pack".
3. active pack health: ready, invalid, unavailable, or fallback.
4. counts for drafts, review candidates, failed items, and library items.
5. primary action: open the active pack, activate a valid selected draft, or
   start setup when no packs exist.

This header should not duplicate the full editor. It should orient users before
they choose a pack or task.

### Pack Workspace

The existing pack selector should become a clearer workspace with filtered
sections:

1. **Active**: the one pack Buddy renders now. Show deactivation and diagnostics
   here, not buried near unrelated actions.
2. **Drafts and Review**: editable packs, imported drafts, duplicated drafts,
   and packs with generated candidates.
3. **Archived/Inactive**: previously active or inactive packs that remain
   available but are not rendered.
4. **Failed/Needs Attention**: failed packs, invalid manifests, source-stale
   library rows, and failed jobs.

The first implementation does not need a new route or backend filter. It can
derive these groups from the existing pack list, selected pack, candidate list,
library list, and job state already loaded by `VisualPackEditor`.

### Task Queue

Post-setup users need a small "needs attention" queue before the lower-level
editor controls:

1. generated candidates awaiting accept/reject.
2. import preview ready for conflict choices.
3. import commit completed and ready for draft review.
4. export completed and ready to download.
5. generation unavailable because jobs/provider/backend are not configured.
6. invalid manifest blocking activation.
7. stale or unavailable personal-library sources.

Each row should link or focus the existing control area. The first slice should
use local state only; durable job history can remain future work.

### Reuse Panel

`VisualPackReusePanel` is the correct concept, but in post-setup management it
should read as a management action row rather than first-run help:

1. Start fresh: creates a draft.
2. Personal library: opens reusable same-user references.
3. Portable archive: opens import preview and export affordances.
4. Another persona: duplicates selected pack to another persona as a draft.

The copy should continue to emphasize draft-first review and explicit
activation. It should not suggest that library entries are snapshots; the
reference-backed model in `Persona_Visual_Packs.md` remains authoritative.

### Editor Sections

The lower-level manifest editor can stay mostly as-is, but should be grouped so
users do not have to scan a long mixed surface:

1. Pack basics and active status.
2. Assets and animations.
3. State mappings and fallbacks.
4. Authored triggers.
5. Validation and activation.
6. Jobs and review.
7. Reuse and portability.

This grouping is also the likely implementation path for splitting
`VisualPackEditor.tsx` into smaller components later.

## State Model

The UI should derive a small `PersonaVisualManagementSummary` view model from
existing data:

```ts
type PersonaVisualManagementSummary = {
  activePackId: string | null
  activePackTitle: string | null
  packCounts: {
    active: number
    draft: number
    review: number
    archived: number
    failed: number
  }
  attentionCounts: {
    invalidSelectedManifest: number
    reviewCandidates: number
    failedCandidates: number
    unavailableLibraryItems: number
    changedLibraryItems: number
    pendingJobs: number
    failedJobs: number
  }
}
```

The first implementation can keep this view model in shared UI code and test it
with deterministic inputs. Backend persistence is not required for V1.

## Error And Recovery Semantics

The management surface should preserve current behavior:

1. Failed pack list/detail loads fall back to derived Buddy behavior.
2. Failed generation readiness checks block queueing and explain what is
   unavailable.
3. Import previews and commits remain explicit; completed commits create drafts.
4. Library items with unavailable sources remain removable but unusable.
5. Changed library sources remain usable but visibly stale.
6. Activation is disabled until required states resolve.
7. Deactivation does not delete packs.

## Accessibility And Copy

The first implementation should avoid a visual-only dashboard. Counts and
statuses need text labels, keyboard-focusable actions, and test IDs for the
attention rows. Labels should use existing localization patterns in
`sidepanel:personaGarden.visuals.*`.

Copy should be direct:

1. "Rendered now" for the active pack.
2. "Draft, not live" for editable inactive packs.
3. "Review required" for candidates/imports.
4. "Source unavailable" for stale library references.
5. "Ready to activate" only when validation passes.

## Implementation Slices

### Slice 1: Management Summary And Attention Model

Add a pure shared UI helper that derives the management summary and attention
rows from existing pack, candidate, library, import/export job, and readiness
state. Add focused tests. No visual rewrite is required yet.

### Slice 2: Visuals Tab Management Header

Render the summary at the top of `VisualPackEditor`, including active-pack
status, counts, and the highest-priority attention item. Keep existing controls
intact below it.

### Slice 3: Sectioned Pack Workspace

Refactor the editor into clearer grouped sections while preserving behavior and
test coverage. This is the point to extract small components from
`VisualPackEditor.tsx`.

### Slice 4: Attention Navigation Polish

Add focus/scroll links from attention rows to existing controls: candidate
review, import preview, import commit, export download, validation, library, and
generation readiness.

## Testing Strategy

Focused tests should cover:

1. management summary derivation with no packs.
2. active pack plus inactive drafts.
3. invalid selected manifest blocking activation.
4. generated candidates awaiting review.
5. completed import commit with draft review copy.
6. completed export with download affordance.
7. unavailable and changed library sources.
8. generation readiness unavailable states.
9. keyboard-accessible attention actions.
10. existing Visual Pack editor behavior remains covered.

## Rollout

Keep each slice behind existing Persona Garden Visuals behavior; do not add a
new feature flag unless the rendered layout becomes large enough to risk user
confusion. Slice 1 is non-rendering and low risk. Slices 2-4 should preserve all
existing test IDs or add compatibility wrappers where tests rely on current
controls.

## Recommendation

Implement Slice 1 first. A tested management-summary helper gives future UI work
a stable contract and makes the next PR reviewable without touching backend
behavior or PR #1767's recipe-backed generation design.
