# Writing Full-Stack Findings Remediation Design

**Date:** 2026-04-10
**Status:** Draft for review
**Related:** `Docs/superpowers/specs/2026-04-09-writing-fullstack-review-design.md`

## Goal

Fix the approved Writing review findings and improvements across backend correctness, frontend editor behavior, shared route or service cleanup, and regression coverage without turning the work into a broad Writing-module rewrite.

## Problem Statement

The completed Writing full-stack review surfaced a small set of real defects plus a few narrower cleanup items that now need to be remediated together:

- soft-deleted manuscript projects still allow descendant reads or writes through child-by-id, child-create, search, link, and cached-analysis paths
- cached manuscript analyses stay falsely fresh after reorder, cascade delete, and prompt-relevant character or world-info changes
- writing snapshot import can fail on same-name soft-deleted templates or themes because import only checks live rows before hitting globally unique `name` constraints
- the current snapshot rollback regression test no longer targets the live import path
- TipTap editor mode bypasses dirty-state and autosave behavior, loses external rehydrate updates, falls back to textarea in split mode, and leaves selection-based helpers bound to textarea-only APIs
- the extension route parity guard asserts an obsolete contract, and the shared Writing service file keeps duplicate manuscript type exports
- snapshot import refresh logic does not refresh the active session detail query when merge-mode imports keep the same active session selected

These issues should be fixed as one bounded remediation tranche because the frontend editor fixes depend on the same shared session state flows, and the backend correctness fixes depend on consistent project-boundary and cache-invalidation rules.

## Approved Product Decisions

- Deleting a manuscript project should make its descendants unreadable and unwritable.
- Snapshot import should restore same-name soft-deleted templates and themes rather than fail on uniqueness.
- Public route shapes should stay stable where possible; fixes should prefer helper, DB, or local component boundaries over new APIs.

## Scope

### In Scope

- harden manuscript project-boundary enforcement for descendant reads, writes, links, search, and cached-analysis visibility
- fix manuscript analysis invalidation for order-changing mutations, cascade deletes, and prompt-relevant character or world-info changes
- make snapshot import restore same-name soft-deleted templates and themes in the live DB import path
- replace the stale snapshot rollback regression with one that injects failure inside the current `import_writing_snapshot()` flow
- route TipTap edits through the same dirty-state and autosave path as textarea edits
- make TipTap external rehydrate and split-view behavior match the approved editor contract
- give selection-based editor helpers a shared editor adapter instead of a textarea-only dependency
- refresh active session detail after merge-mode snapshot import when the active session survives
- fix the stale extension route parity guard and collapse duplicate manuscript type exports in the shared Writing service
- add focused regression coverage and rerun the targeted backend and frontend validation slices

### Out of Scope

- new Writing features or UX redesign
- a broad TipTap-first rewrite of the entire Writing Playground
- changing public Writing API schemas unless a narrow bug fix requires it
- repo-wide route-shell unification between web and extension surfaces
- unrelated refactors in large Writing files beyond what is needed to land the reviewed fixes

## Review-Driven Corrections

This design includes the design-review corrections raised before writing the spec:

- active-project enforcement is not limited to project-scoped list routes; it also covers child-by-id, link, create, reorder, search, and cached-analysis paths
- the TipTap remediation does not merely remove a focus gate; it explicitly routes TipTap edits through the session dirty-state and autosave pipeline while preserving rich JSON state
- snapshot rollback coverage is rewritten against the current `CharactersRAGDB.import_writing_snapshot()` path rather than dead helper internals
- route parity cleanup updates the test to the real shared-route contract instead of forcing obsolete `PageShell` markup onto the shared route
- snapshot refresh behavior distinguishes merge from replace instead of invalidating or clearing state uniformly
- frontend verification uses repo-local package scripts rather than assuming a global `vitest` binary

## Approaches Considered

### 1. Boundary-first remediation with a narrow editor bridge

Fix manuscript invariants in the helper or DB layer, then repair the Writing Playground by introducing a small editor adapter between the parent page and the concrete editor implementation.

Strengths:

- fixes the project-boundary bug where future callers can still bypass route checks
- keeps analysis invalidation close to the mutations that cause stale caches
- solves TipTap parity without a broad frontend architecture rewrite

Weaknesses:

- touches both backend and frontend coordination seams
- requires careful regression coverage around the new editor adapter

### 2. Route and component local patching

Patch only the currently failing routes, handlers, and render branches.

Strengths:

- smaller initial diff
- fast for obvious local issues

Weaknesses:

- leaves the same project-boundary and editor-state rules duplicated across call sites
- high chance of missing another descendant path or another selection-based helper

### 3. Larger Writing cleanup pass

Refactor manuscript ownership helpers, snapshot import, editor state, route wrappers, and shared types more broadly before fixing the bugs.

Strengths:

- potentially cleaner long-term structure

Weaknesses:

- too much churn for a findings-driven remediation batch
- higher regression risk and slower verification

## Recommended Approach

Use the boundary-first remediation with a narrow editor bridge.

Execution shape:

1. harden manuscript project and scope invariants at the helper or DB boundary
2. extend stale-analysis handling only where the review found real freshness defects
3. keep snapshot import atomic in the DB import path while reconciling same-name soft-deleted templates and themes
4. add a small editor adapter so TipTap and textarea share the same parent editing contract
5. clean up the stale parity guard and duplicate type exports without changing intended route behavior

This keeps the work focused on approved defects while still fixing the shared seams that caused them.

## Architecture

The remediation is split into five workstreams.

### Workstream 1: Manuscript Project-Boundary Hardening

`ManuscriptDBHelper` remains the source of truth for project-owned manuscript invariants.

The implementation will add narrow helper-level checks that answer two questions:

- is the owning project still active?
- if a child entity is being accessed by id, does its owning project and any required active parent still exist?

These checks will then be used by the reviewed high-risk paths:

- child-by-id reads and writes for parts, chapters, scenes, characters, relationships, world info, plot lines, plot events, plot holes, and linked-scene metadata where those routes are exposed through the Writing module
- parent-scoped child collection routes keyed by project-owned parents, such as scenes-by-chapter, scene-characters-by-scene, scene-world-info-by-scene, and other equivalent collection reads
- create flows that accept `project_id`, `chapter_id`, `part_id`, or other project-owned parent references
- link and unlink flows that currently trust only the immediate child row
- project-scoped search and cached-analysis listing
- reorder and reparent flows

Approved route-contract behavior:

- project-scoped list and search routes keep their current collection-style contract and return empty results when the project is deleted
- parent-scoped child collection routes keep their current collection-style contract and return empty results when the parent or owning project is deleted
- child-by-id read, update, delete, analyze, and link flows should behave as missing or deleted resources through the existing route error mapping
- create, reorder, and link flows under deleted projects or deleted parents should fail at the helper boundary before mutation

This keeps deleted descendants unreadable and unwritable without introducing a new family of route shapes.

### Workstream 2: Analysis Freshness and Visibility

The manuscript helper layer will take ownership of the stale-analysis rules that are directly implied by the review findings.

#### 2.1 Order-changing invalidation

The following mutations change the effective text ordering used by project or chapter analysis prompts and must stale cached analyses:

- part reorder: mark project analyses stale
- chapter reorder or chapter reparent: mark project analyses stale
- scene reorder: mark chapter and project analyses stale

Scene reorder does not need to stale scene-scoped analyses because the scene body itself did not change.

#### 2.2 Membership-changing invalidation

The following mutations change which text is included in aggregate analysis prompts and must stale cached analyses:

- scene create: mark chapter and project analyses stale
- scene content update: keep the current scene/chapter/project invalidation behavior for content-bearing changes
- scene delete: mark scene, chapter, and project analyses stale as today
- chapter soft delete: mark deleted chapter analyses stale and mark project analyses stale
- part soft delete: mark affected chapter analyses stale and mark project analyses stale

#### 2.3 Prompt-summary invalidation

Project-level consistency and plot-hole prompts currently summarize only:

- character `name` and `role`
- world-info `name` and `kind`

So this tranche will stale project analyses only when those prompt-relevant fields or membership change:

- character create, delete, or update of `name` or `role`
- world-info create, delete, or update of `name` or `kind`

This intentionally avoids broader invalidation for fields not currently included in project analysis prompts.

#### 2.4 Cached-analysis visibility

User-visible analysis retrieval will stop surfacing cached analyses whose scope is no longer readable:

- deleted projects suppress all cached analyses for that project
- deleted chapter or scene scopes are suppressed from analysis reads or lists that would otherwise surface them

This aligns cached-analysis visibility with the approved descendant unreadable or unwritable rule.

### Workstream 3: Snapshot Import Semantics and Rollback Defense

`CharactersRAGDB.import_writing_snapshot()` remains the authoritative reconciliation point for snapshot import.

#### 3.1 Same-name soft-deleted template and theme restore

For templates and themes, import will stop treating a soft-deleted same-name row as “missing.”

Required behavior:

- if a live row with the same `name` exists, update it in place as today
- if only a soft-deleted row with that `name` exists, restore that row in place, update its fields, clear `deleted`, and bump version
- if no row exists, insert a new row

This applies to both merge and replace modes because the uniqueness constraint problem is not replace-only.

#### 3.2 Replace-mode atomicity

Replace-mode import keeps one transaction around:

1. soft-delete live sessions, templates, and themes
2. reconcile incoming snapshot rows
3. commit only when the full import succeeds

If any failure happens after replace-mode mutations begin, the pre-import dataset must remain intact after rollback.

#### 3.3 Regression coverage target

The stale rollback test in `test_writing_endpoint_integration.py` will be replaced with a regression that injects failure inside the current `CharactersRAGDB.import_writing_snapshot()` path after replace-mode mutations have already begun.

The test must not patch dead route internals such as `_restore_soft_deleted_writing_session`.

### Workstream 4: Writing Playground Editor-State Parity

The frontend fix will not attempt a broad editor rewrite. It will introduce one small shared editor contract between `WritingPlayground` and the concrete editor implementation.

#### 4.1 Shared editor adapter

`WritingPlayground` will stop depending directly on textarea DOM APIs for selection-sensitive actions.

Instead, the active editor implementation will expose a narrow adapter that can support the existing parent workflows:

- current plain-text selection or cursor range
- focus or selection updates for search navigation
- text insertion or replacement at the active selection
- selected-text extraction for speech and editor actions

Textarea and TipTap will both satisfy that contract, which lets search, replace, placeholder insertion, token insertion, speech selection, and similar helpers stop branching on a textarea ref.

#### 4.2 TipTap content and dirty-state flow

TipTap edits will update two pieces of parent state together:

- authoritative rich JSON for TipTap rendering
- plain-text prompt content through the same session dirty-state and autosave path used by textarea edits

The remediation will also persist the rich editor document as a companion session-payload field:

- plain `prompt` remains the canonical text used by generation, search, and existing payload consumers
- rich TipTap-compatible JSON is stored under `prompt_rich`
- readers must tolerate `prompt_rich` being absent and fall back to deriving a minimal rich document from plain `prompt`

That means TipTap `onContentChange` will no longer call raw `setEditorText()` directly. It will feed the session save path while keeping rich JSON synchronized in parent state and inside the session payload.

The parent sync path will continue to distinguish editor-originated changes from external plain-text changes so a local TipTap edit is not immediately replaced by lossy `plainTextToTipTapJson(editorText)` conversion.

Plain-text-only edits remain allowed. When the user edits through the plain-text editor path, the saved payload should clear `prompt_rich` so the plain prompt and rich prompt cannot silently diverge.

#### 4.3 External rehydrate behavior

`WritingTipTapEditor` will accept external content updates whenever the parent has authoritative content that differs from the editor’s current document.

The current focus gate will be removed from the editor component itself. Protection against clobbering unsaved local work stays at the existing parent session-management boundary, where external session rehydrate is already guarded by dirty-state rules.

When a session payload includes `prompt_rich`, rehydrate should prefer that stored rich document over reconstructing from plain text. Reconstruction from plain `prompt` remains the fallback only for legacy or plain-text sessions.

#### 4.4 Split-view parity

When `editorMode === "tiptap"` and `editorView === "split"`, the left side of split view will render TipTap instead of forcing a textarea fallback.

Textarea remains the implementation for plain-text mode only.

#### 4.5 Snapshot import refresh behavior

Snapshot import refresh will distinguish merge from replace:

- replace mode keeps the current behavior of clearing active session selection
- merge mode invalidates the active session detail query when the active session id is still present, so the open editor is refreshed if the imported snapshot changed that session

### Workstream 5: Route-Parity and Shared-Type Cleanup

#### 5.1 Extension route parity guard

The extension parity guard will be rewritten to assert the actual shared contract:

- both route wrappers mount `WritingPlayground`
- both use their expected route layout wrapper
- any shared route-shell invariant checked by the test reflects the current shared route source

This cleanup does not require forcing the shared route and extension route to use identical wrapper markup.

#### 5.2 Shared Writing service type cleanup

`apps/packages/ui/src/services/writing-playground.ts` will collapse its duplicate manuscript type exports into one canonical set.

Requirements:

- preserve existing exported names that current consumers import
- prefer one rich canonical response shape plus narrower aliases or `Pick<>`-style derived types where needed
- avoid consumer-wide renaming churn in this tranche

## File Plan

Primary backend files expected to change:

- `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- `tldw_Server_API/app/api/v1/endpoints/writing.py`
- related Writing backend tests under `tldw_Server_API/tests/Writing/`

Primary frontend files expected to change:

- `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
- `apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx`
- `apps/packages/ui/src/components/Option/WritingPlayground/writing-tiptap-utils.ts`
- `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingSessionManagement.ts`
- `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingImportExport.ts`
- `apps/packages/ui/src/services/writing-playground.ts`
- `apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts`
- targeted shared UI and extension tests for the touched behavior

New frontend helper files are acceptable only if they keep the editor adapter or TipTap mapping logic smaller and more testable than further expanding `index.tsx`.

## Testing Strategy

### Backend

Add or update focused tests for:

- deleted-project descendant by-id reads returning no readable resource
- deleted-project or deleted-parent child collection routes returning empty results instead of leaking descendants
- deleted-project create, link, reorder, and search rejection or suppression behavior
- cached-analysis suppression for deleted projects and deleted scopes
- project-analysis invalidation after part, chapter, and scene reorder or reparent
- project-analysis invalidation after chapter or part cascade delete
- project-analysis invalidation after character `name` or `role` changes and world-info `name` or `kind` changes
- snapshot import restoring same-name soft-deleted templates and themes in merge and replace flows
- snapshot rollback after a failure injected inside the current DB import path

Target verification commands will stay on repo-native backend runners:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py -v
python -m pytest tldw_Server_API/tests/Writing/test_manuscript_db.py tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py -v
```

### Frontend

Add or update focused tests for:

- TipTap edits marking the session dirty and scheduling save through the shared session path
- TipTap saves persisting `prompt_rich` while keeping plain `prompt` aligned
- TipTap reload or session-switch rehydrate preferring stored `prompt_rich` over lossy plain-text reconstruction
- TipTap split-view rendering instead of textarea fallback
- selection-based editor helpers working through the shared editor adapter
- merge-mode snapshot import invalidating the active session detail query
- the updated route parity guard
- duplicate manuscript type cleanup staying source-compatible for touched consumers

Use repo-local package scripts instead of global binaries:

```bash
cd apps/packages/ui && bun run test -- src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx src/components/Option/WritingPlayground/__tests__/writing-editor-actions-utils.test.ts src/components/Option/WritingPlayground/__tests__/writing-snapshot-import-utils.test.ts --maxWorkers=1
cd apps/tldw-frontend && bun run test:run -- extension/__tests__/writing-playground-route-parity.guard.test.ts
```

The implementation plan will append the exact new TipTap and snapshot-refresh test files introduced in this tranche to the same package-local UI command, using the concrete filenames chosen during implementation.

If the package-local runner still cannot start because workspace dependencies are unavailable, that remains an explicit verification blocker and frontend completion cannot be claimed.

### Security Verification

Before closing the backend tranche, run Bandit on the touched Python scope from the project virtual environment.

## Risks and Mitigations

### Risk: boundary hardening changes route behavior more than intended

Mitigation:

- keep project-scoped list and search routes on their current collection-style contract
- let by-id routes continue to surface through existing not-found or conflict handling instead of inventing new responses

### Risk: stale-analysis invalidation becomes too broad and churns caches

Mitigation:

- restrict prompt-summary invalidation to fields the current analysis prompts actually read
- restrict order-based invalidation to scopes whose aggregate text ordering really changes

### Risk: the editor adapter becomes a hidden rewrite

Mitigation:

- keep the interface narrowly focused on selection and text-replacement actions already used by `WritingPlayground`
- avoid pushing editor state into a new global store or cross-module abstraction

### Risk: TipTap external rehydrate reintroduces lossy plain-text conversion

Mitigation:

- keep authoritative TipTap JSON in parent state
- preserve the distinction between editor-originated changes and external plain-text rehydrate events

### Risk: template or theme restore by name regresses version or default-flag semantics

Mitigation:

- reuse the same version-bump and field-reconciliation rules for restored rows as for live-row updates
- lock the behavior with merge and replace regressions

## Success Criteria

This design is successful if the implementation:

- makes deleted manuscript projects a real read or write boundary for descendants
- removes the stale cached-analysis cases identified in the review without widening invalidation beyond current prompt inputs
- restores same-name soft-deleted templates and themes during snapshot import instead of failing uniqueness checks
- replaces the stale rollback regression with one that exercises the current import path
- makes TipTap behave like a first-class Writing editor with shared dirty-state, autosave, split view, and selection-helper support
- removes the stale route parity assumption and duplicate manuscript type definitions without broad route churn
- verifies the changes with targeted backend and frontend tests run through repo-native commands
