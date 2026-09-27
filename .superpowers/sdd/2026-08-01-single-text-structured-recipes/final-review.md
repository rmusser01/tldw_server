# Track B final whole-branch review

## Verdict

**NEEDS FIXES.** The integrated feature range
`9da94ebcb41ba45d279ab707ec92a285d31ba5c7..872901da8723e60afba4bdbbd889695c8fc48d3a`
has two Important findings in the durable recipe-uncertainty recovery boundary.
No Critical or Minor findings were found.

The schema, deterministic renderers, storage and interop boundaries, Prompt
Studio/MCP behavior, builder/library adapters, real WebUI and packaged-extension
journeys, capability authorization, local Apply/Undo, accessibility, and
mixed-version behavior otherwise passed the review and representative gates.

## Findings

### Important 1 — An untrusted or non-matching pull erases the durable uncertainty lock

`apps/packages/ui/src/services/prompt-sync.ts:423-448` always includes
`syncStatus: "synced"` in `serverToLocalFields`. Both existing-row branches in
`pullFromStudio` write those fields before calling `clearReconciledOwner`
(`prompt-sync.ts:884-923`) and then return `syncStatus: "synced"`. However,
`clearReconciledOwner` intentionally clears only the exact trustworthy
`actualOwnerId` (`prompt-sync.ts:145-155`). It cannot clear an unknown-owner
quarantine or a scoped marker belonging to another owner.

Consequently a recipe already durably marked `error` is rewritten to `synced`
when pull metadata has no owner, when a different owner performs the pull, or
when a same-owner pull leaves the separate unknown-owner quarantine in place.
The process-local registry still blocks writes immediately, but Web reload or
MV3 background termination drops that registry. The local row then falsely
looks reconciled and a later Save/Update may dispatch again, creating the exact
duplicate mutation the durable restart lock is intended to prevent. The same
path is used by `keep_server` conflict resolution.

Reviewer probe: I temporarily strengthened the existing “pull with no reported
owner cannot clear scoped or unknown markers” test by seeding
`syncStatus: "error"` and asserting it remains `error`. The run failed exactly
as expected: **43 passed / 1 failed**, received `synced`. The test change was
removed afterward.

Required correction: a pull may copy server content, but it must not mark a v2
row synced or clear its durable error unless the exact-ID uncertainty state is
fully reconciled under the trustworthy matching owner. Unknown-owner and
other-owner state must remain durable across restart. Add regressions for
unknown owner, different owner, same owner plus unknown quarantine, and
`keep_server`.

### Important 2 — Unlink bypasses the same durable lock and removes the reconciliation pointer

`unlinkPrompt` unconditionally removes all server/project references and writes
`syncStatus: "local"` (`apps/packages/ui/src/services/prompt-sync.ts:1027-1057`).
It never validates the local record's v2 identity, checks an existing durable
`error`, or consults the uncertainty authority. The library exposes Pull and
Unlink whenever `serverId` exists, including error rows
(`apps/packages/ui/src/components/Option/Prompt/PromptActionsMenu.tsx:64-112`).

An ambiguously dispatched recipe can therefore be unlinked while its scoped or
unknown marker remains. This erases the restart-safe `error` and also discards
the `serverId` needed for an authoritative pull. After reload/background
restart, the record is local and retryable; Save as new can duplicate the
possibly successful remote mutation. Unlink is not one of the approved
same-owner reconciliation operations and is not the explicit, warned
unknown-owner Forget action.

Reviewer probe: I temporarily added a test that seeds a linked v2 recipe with
`syncStatus: "error"` plus a scoped marker and expects Unlink to fail closed.
The run failed exactly as expected: **44 passed / 1 failed**, returning
`success: true, syncStatus: "local"`. The test change was removed afterward.

Required correction: fail closed for v2 Unlink while durable or runtime
uncertainty exists, preserving both `error` and the server linkage. Add service
and library-action regressions for scoped, unknown-owner, and restarted
durable-error cases while retaining v1 behavior.

## Integrated contract review

- Python and TypeScript keep the exact v2 discriminants, strict outer/inner
  identity, stable order/tie behavior, disabled-block omission, XML collision
  rejection, Markdown/free-form formatting, string-only selected values,
  bounded output, and one-field legacy snapshots. The shared fixture gates
  cover deterministic parity and retain v1 coercion compatibility without v2
  fallthrough.
- Raw/API/DB/import/Prompt Studio persistence boundaries reject runtime-value
  maps before mutation and derive stored v2 legacy snapshots from authored
  template text. Runtime values stay sibling UI state and are absent from
  save/update/clone payloads.
- Prompt Studio previews return one rendered string plus the target legacy
  field and no synthetic message list. Prompt execution rejects unapplied v2
  recipes. MCP returns one protocol-safe user text message while preserving
  target/render metadata.
- Owner IDs remain opaque SHA-256 identities over effective base, auth mode and
  source, organization, and authoritative principal/key fingerprint. The
  extension background remains the dispatch and uncertainty authority;
  immutable snapshots, refresh constraints, provisional receipts, response
  overlay validation, exact-ID quarantine, and no ambiguous direct fallback
  are implemented and broadly exercised. Findings 1 and 2 are downstream
  cleanup/restart holes, not an owner-derivation or double-dispatch-attempt bug.
- Capability support, create/update authorization, current owner, and
  credential revision remain separate gates. Old, unknown, offline, private,
  and future-schema states keep local edit/preview/Apply available while
  disabling persistence or quarantining unsupported records.
- System Apply preserves selected-template identity and exact raw override;
  user Apply changes only the unsent draft. Both have exact one-step Undo and
  clear Undo after a later user edit/send lifecycle transition.

## Improve-button placement and scope audit

The deferred Improve-button relocation was not implemented in Track B. Git
blame/diff shows the composer action remains in its pre-Track-B container and
the system action remains in the existing modal footer. Track B adds the third
menu action and builder host without moving the trigger. Task 9's production
changes are limited to its approved viewport-width containment, semantic
contrast tokens, and returning focus to the actual Improve trigger. No package
manifest or lockfile changes occur in the reviewed feature range.

The packaged extension Quick Chat pop-out still has no recipe adapter, and the
review does not falsely claim it. The exercised extension chat surfaces are the
existing recipe-capable sidepanel and `/options.html#/chat` paths, consistent
with the approved Task 9 scope.

## Fresh verification

All positive commands below ran against restored production/test sources at
`872901da87` before this artifact was created.

- Backend schema/render/persistence/interop/Prompt Studio/MCP matrix:
  **876 passed** in 205.41 seconds. Only unrelated pytest temporary-directory
  cleanup warnings were emitted.
- Capability support/authentication/authorization subset: **7 passed** in 7.71
  seconds. An initial command named one nonexistent node ID and collected no
  usable evidence; the corrected command produced this result.
- Frontend renderer/transport/owner/registry/sync/builder/editor/library/search
  matrix using repository-pinned Vitest 4.0.18: **13 files / 675 tests passed**
  in 24.24 seconds.
- Real WebUI `/chat` Playwright recipe spec: **6/6 passed** in 1.2 minutes,
  including starters, exact rendering/reorder, persistence payloads,
  system/user Apply and Undo, quarantine, old/unknown/offline, mobile, focus,
  and two-theme axe checks.
- Packaged-extension recipe spec: **5/5 passed** in 19.8 seconds after rerunning
  outside the filesystem/network sandbox so the loopback fixture could bind.
  The first sandboxed attempt failed only with `listen EPERM`; its production
  extension build itself completed successfully before that environment error.
- Extension `bun run compile`: exit 0.
- Bandit over changed production Python files other than the large inherited
  `PromptStudioDatabase.py`: exit 0 with no findings. The inclusive scan found
  only 23 low B311 reports in untouched random-jitter lines of that database
  file, matching the earlier Task 4 baseline; there were no medium/high
  findings and no finding in changed code.
- `git diff --check` for both worktree and feature range: exit 0. Temporary
  reviewer probes were removed and status was clean before this report.

## Baseline determinations

- The full frontend typecheck is not claimed as passing. The owner-contract
  final review recorded a byte-identical 86-diagnostic repository baseline and
  an empty changed-path scan; fresh extension compile and focused TypeScript
  tests pass here. The documented Presentation Studio/v1 editor/presentation
  E2E/skills diagnostics are therefore not Track B findings.
- The Web production build previously compiled all 154 routes and passed token
  sync before the shared 683.0 KB/600.0 KB app-shell budget failure. Task 8
  recorded the same gate already failing at 678.1 KB and confirmed lazy-loading
  the recipe builder did not change the measured shell before/after its
  integration. Task 10 changed no `apps` Git tree. This remains repository
  budget debt, not a new correctness or release-boundary finding in this
  review.
- The capability catalog rate-limit test remains a real pre-existing defect:
  route dependency ordering leaves its second request at 200 instead of 429.
  Task 10 proved identical test/auth blobs and identical failure at its base and
  head. It does not bypass recipe write authorization or owner checks and is not
  attributed to the one-boolean capability rollout.

## Handoff

Return both Important findings to the Track B implementation under strict
RED/GREEN. Re-run the focused owner/sync suite, the 675-test frontend matrix,
both browser specs, extension compile, backend representative matrix, Bandit,
and diff hygiene. Keep `TASK-12984.2` In Progress until a new independent
whole-branch review approves the corrected range.
