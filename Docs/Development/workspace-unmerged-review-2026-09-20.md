# Unmerged Workspace Stream Review

Tracking: TASK-12020.50, with existing TASK-12020.49 changes included.

## Scope And Baseline

- Worktree: `.worktrees/shared-workspace-clone-ux`.
- Branch: `codex/shared-workspace-clone-ux`.
- Reviewed baseline: `c70387f496d82fcee92926bf3715bf5cd240ba88` plus tracked
  uncommitted changes and untracked implementation/tests, not only `git diff`.
- Fetched dev: `d72b1d2850ea947b6d12cac19f6b95867b68a580`, 446 commits ahead.
  No rebase or merged-tree certification was performed. GitHub reports no PR for
  this branch. No CI was cancelled.
- Independent review scopes: previous backend patch; client opening/persistence/
  connection foundation; editor/directory UI and clone recovery lifecycle.
- The owned route remains deliberately disabled. That known incomplete integration
  is not itself classified as a newly discovered regression. Clone UX and the
  connection store are separately exposed surfaces.
- New deletion-foundation work is reviewed separately; it must not obscure or
  retroactively change the conclusions about the preceding patch.

## Findings

The findings below describe the original reviewed patch. R1-R6 are corrected
and verified within the bounded checks recorded below, in the working tree.
Original reproductions used Vitest instrumentation; subsequent
remediation also uses an isolated real backend and a Chromium/CDP storage probe.
Passing pre-existing tests alone does not establish coverage of these scenarios.

### R1 - P1: Prehydration Invalidations Can Overwrite Legacy Storage

`useOwnedWorkspaceOpening.ts:94` invokes `stop()` before waiting for hydration.
The call to `invalidateOwnedWorkspace()` reaches the persisted setter even when
there is no owned state to invalidate. With delayed hydration, a populated legacy
index can be rewritten from empty initial state; split-storage cleanup can remove
the omitted records. A reload before recovery can leave lost legacy data.

Correction: avoid invoking the persisted setter at all for a no-op invalidation;
enforce the no-prehydration-write invariant at the persistence boundary. Returning
the same state from inside a setter is insufficient if middleware still persists.
Regression: delayed hydration with a populated legacy index and snapshots, mount,
unmount/cancel and reload; persisted bytes and records must remain intact.

### R2 - P1: Cookie Opening Reads Are Not Principal-Bound

`owned-workspace-opening.ts:193` loads metadata, sources, artifacts and notes
without the captured expected-user header. Cookies can change between the initial
and final principal reads. An A-to-B-to-A switch with a colliding workspace ID
can accept B's content into an A-scoped bundle even though both identity checks
return A.

Correction: send the captured principal on every bundle request and enforce it
before database access on every matching route, including sources and artifacts.
Keep the final identity recheck as an additional guard, not the sole protection.
Regression: each bundle read rejects the intervening B account with 412; no
foreign content reaches activation, drafts, or visible UI.

### R3 - P1: Outgoing Stale Draft Overwrites Another Window

`owned-workspace-slice.ts:317` unconditionally saves the outgoing draft when
leaving a scope/target. Two visible windows share the same draft storage key.
Window B can save new composer text, then window A's invalidation writes its old
or empty composer over B's value despite no edit in A.

Correction: do not resave unchanged outgoing state; define concurrent ownership
and recovery for genuinely competing edits. Prefer writer-specific retained
revisions or serialized revision checks with explicit conflict preservation, not
an unprotected read-then-write localStorage CAS. Never silently select a winner
and discard the other writer's text.
Regression: unchanged stale departure, simultaneous edits, storage failure and
recovery, reload, and same workspace UUID in different accounts.

### R4 - P1: Cross-Tab Account Changes Leave Clone Commands Active

`useSharedWorkspaceClones.ts:111` ignores storage events other than its recovery
key or full clear. Another tab can change configuration/session credentials while
the visible tab retains the old verified scope and commands. The clone transport
resolves current connection/auth settings when dispatching, so admission or retry
can target a different account/server from the recovery record and displayed share.

Correction: synchronously invalidate on the existing configuration/session storage
keys and bind the actual request context to the verified scope, including server,
principal and redirect behavior. Event handling alone cannot close the
verification-to-dispatch race. Never replay uncertain admission with new credentials.
Regression: cross-tab config and session changes during queued admission, polling
and retry; real transport boundary and redirect checks, not only mocked hook APIs.

### R5 - P2: An Unrelated Failed Draft Blocks Note Creation

`owned-workspace-slice.ts:590` gates a new note's durable uncertain-create marker
on aggregate `ownedWorkspaceDraftStatus`. An older failed write in A keeps the
aggregate unavailable even after the B marker saves successfully. Retrying B
cannot flush A's pending entry, so valid B creation remains blocked.

Correction: use the current marker's persistence result, retaining the aggregate
warning separately, as archive preparation already does. Do not remove the
uncertain-create protection when the current marker itself is not durable.
Regression: failed A write, switch to B, recover storage, save B marker; B can
proceed and A's retained recovery warning remains accurate.

### R6 - P2: JWT Readiness Uses A Disableable Legacy Endpoint

`connection.tsx:858` selects `/api/v1/users/me` for authenticated multi-user
readiness. With `ENABLE_LEGACY_USER_ME_ENDPOINTS=false`, the backend returns 410
despite valid credentials, causing false disconnected state.

Correction: use `/api/v1/users/me/profile?sections=identity`, matching the canonical
opening principal endpoint. Preserve 401/403 failure behavior and do not fall back
to public liveness as authentication evidence.
Regression: supported profile endpoint with legacy routes disabled, plus valid,
expired and forbidden sessions in both applicable browser transports.

## Verification And Limits

- Previous backend patch reviewer: 205 tests passed; 17 PostgreSQL cases
  deliberately deselected. No concrete new backend patch defect found.
- Client-foundation reviewer: all five targeted probes reproduced R1/R2/R3/R5/R6;
  100 existing loader/draft-state tests also passed.
- UI/clone reviewer: targeted cross-tab probe reproduced R4. Manager suite 55
  tests passed; no additional high-confidence editor/archive/root-panel defect.
- Parent regression rerun: **1328 tests in 51 frontend files passed**.
  Log: `/tmp/tldw-workspace-stream-review-frontend.log`.
- Parent independently reran all six counterexamples: the five foundation probes
  failed with the expected behavioral assertions (not import/runtime errors), and
  the clone config-storage probe retained the old scope. Reproduction scripts:
  `/tmp/tldw-workspace-review-probes.mjs` and
  `/tmp/tldw-clone-cross-tab-review-probe.mjs`; matching `.log` files retain output.
  These inject tests into Vite's in-memory source transform without modifying
  tracked tests. Run from `apps/tldw-frontend` with
  `NODE_OPTIONS=--no-experimental-webstorage node <script>`.
- R4's dynamic probe specifically tests `tldwConfig` and proves stale verified
  scope. A wrong-account mutation is a source-traced risk, not a live mutation
  demonstrated by that probe. Other session storage keys need separate coverage.
- Existing test warnings remain; these are not warning-free or full-project runs.
- No live cookie race, real multi-window IndexedDB recovery, PostgreSQL execution,
  full project build, latest-dev integration or full WebUI/CDP certification here.

## P1 Remediation (2026-09-20)

The approved remediation is tracked by
`../superpowers/plans/2026-09-20-workspace-p1-remediation.md` under TASK-12020.50.
It preserves the existing dirty worktree and does not enable the owned route.

- R1: the actual persistence adapter rejects writes until hydration completes;
  a no-op invalidation returns before invoking the persisted setter. Delayed
  hydration and direct-setter regressions protect populated legacy storage.
- R2: every owned bundle read sends the captured expected-user header. Sources
  and artifacts now enforce it before resource access, like metadata and notes.
  A typed account-change 412 is classified as denial, not a network failure;
  explicit retry re-verifies identity. Unrelated 412 responses are unchanged.
- R3: each writer has one replaceable journal slot per target. Concurrent heads
  are retained, and ambiguous activation/archive preparation fails closed.
  Modern code never writes or deletes the legacy key. One exact, validated
  baseline distinguishes a migrated legacy draft from a later old-tab edit;
  unexpected legacy bytes remain a recoverable conflict. This is bounded per
  writer, not a global journal garbage-collection or byte-size guarantee.
- R4: clone commands and polling use a verified, captured transport with
  expected-user enforcement, no redirects, and no mutable credential lookup.
  Configuration/session events suspend commands; same-account verification
  installs a fresh transport without accepting late results from the old one.
  Foreign recovery envelopes must be rejected before mutation, not deleted or
  handled as ordinary storage failure. The UI exposes a recovery conflict rather
  than a destructive automatic reset. Uncertain admission keys must survive
  authentication failures and re-verification.

Independent review additionally reproduced mixed-version legacy draft loss,
foreign-scope recovery deletion, and loss of an uncertain clone key after an auth
failure. These are included in R3/R4 remediation, not silently deferred.
Final rereview also covered plain-string CSRF 403 recovery and a two-manager race
where one tab's first 403 cleared another tab's ambiguous admission key. Every
403 now preserves the key and suspends dispatch until re-verification, without
changing server permissions. All identified R1-R4 follow-up findings are resolved;
independent targeted rereviews found no remaining issue in those corrections.

### Evidence And Limits

- Final combined frontend regressions: **1510 passed in 53 files**.
  `/tmp/workspace-p1-regression-complete.log`. Test fixtures now read durable
  drafts through the recovery API and inject failures into actual journal writes;
  retained-content and pre-dispatch durability assertions were not removed.
- Expanded scoped TypeScript passed: `/tmp/workspace-p1-tsc-complete.log`.
  ESLint exited zero with the same 13 warnings present at HEAD (10 store imports,
  three existing test warnings), plus the existing pages-directory notice.
  `/tmp/workspace-p1-eslint-verified.log`; last 403 edits also pass scoped lint.
  The full design-system guard still fails outside this changed screen; current
  and HEAD SharedWithMe both have zero findings. No baseline exemption was added.
- Selected backend regressions: 141 passed; 11 existing warnings.
  `/tmp/workspace-p1-backend-green.log`.
- Real isolated FastAPI/API-key/SQLite backend: 23 HTTP assertions, including
  expected-user mismatch rejection before owned/clone resource access, optional
  header compatibility and unauthenticated denial. Backend stopped afterward.
  `/tmp/workspace-p1-http-final.log`.
- Actual Chromium connected over CDP: seven storage cases using the bundled
  production modules, real same-origin localStorage and native Web Locks. Cases
  cover competing drafts, reload, account isolation, fixed per-writer slot count,
  legacy-tab edits, and foreign-account clone recovery reads/writes.
  `/tmp/workspace-p1-cdp-verified.log`, fixture `workspace-p1-cdp-61AGOE`.
  The mixed-version case failed before the fix with `ready` instead of `conflict`:
  `/tmp/workspace-p1-cdp-mixed-red.log`.
- Production Python Bandit: zero findings/errors in the touched endpoint/route
  scope. `/tmp/workspace-p1-bandit-verified.json`. This is not a TypeScript security scan.
- OpenAPI export, generated TypeScript and fingerprint recheck passed:
  `/tmp/workspace-p1-openapi-check-final.log`.

The probes are not full WebUI acceptance or a real JWT/cookie A-B-A browser run.
PostgreSQL, the production startup lifecycle and integration with current dev
remain unverified in this remediation pass. Draft conflict resolution and journal
consolidation still need an explicit UI/lifecycle before owned-route enablement.
R5/R6 were still open at the P1 checkpoint; their follow-up is recorded below.
TASK-12020.50 remains In Progress.

## P2 Remediation (2026-09-20)

- R5: `setOwnedNoteCreateUncertain` now checks the current captured marker's
  persistence result, not aggregate `ownedWorkspaceDraftStatus`. Another target's
  failed write remains visible and recoverable without blocking a durable current
  note. Current-target quota failures and concurrent edits still reject admission
  and roll back the uncertain flag without discarding note content.
- R6: bearer and applicable cookie readiness now use
  `/api/v1/users/me/profile?sections=identity`. The probe remains authenticated;
  401/403 still deny readiness without public-health or RAG fallback. No response
  body consumer or backend contract change was necessary.

Test-first evidence: the new R5 store and QuickNotes dispatch regressions failed
before the fix, then the three-file store/draft/QuickNotes suite passed 235 tests.
R6's updated legacy-disabled and auth-denial matrix failed eight cases before the
path correction; connection/persona tests then passed 50 tests. Independent scoped
read-only reviews of both fixes found no actionable issues.

Broader regression: 1521 tests passed in 54 files, including directory lifecycle,
clone recovery, owned opening, legacy persistence and mutation controls.
`/tmp/workspace-p2-regression.log`. Scoped TypeScript passes; ESLint reports zero
errors and the 14 existing `any` warnings in connection tests. These changes touch
TypeScript/tests/docs only; Bandit cannot analyze TypeScript, and the earlier
Python scan is not claimed as a new security scan. No OpenAPI change was made.
Final post-format focused rerun: 285 tests passed in five files;
`/tmp/workspace-p2-final-focused.log`. Final type/lint evidence:
`/tmp/workspace-p2-tsc-final.log`, `/tmp/workspace-p2-eslint-final.log`.

Real FastAPI/SQLite HTTP verification, with `ENABLE_LEGACY_USER_ME_ENDPOINTS=false`:
18 requests passed across two isolated backend processes. Multi-user registration
and login used actual endpoints and JWTs; single-user sessions used the actual
cookie-mint endpoint. Both authenticated canonical reads returned 200 and legacy
reads returned 410. Missing, malformed, expired JWT and revoked-session cases
returned 401. Restricted profile query parameters returned 403 in both modes;
that validates a real forbidden response, not an account-permission revocation
scenario. JWT expiry used a deliberately expired, correctly signed synthetic
test token. Both processes stopped afterward.

Evidence: `/tmp/workspace-r6-live.log`, probe `/tmp/tldw-workspace-r6-live.py`,
fixtures `workspace-r6-multi_user-om41zfqi` and `workspace-r6-single_user-z8c63op4`
in the OS temporary directory. Bootstrap used production helpers with application
lifespan disabled. This is not full-startup, PostgreSQL, WebUI/CDP account-switch
or end-to-end clone acceptance. The first probe attempts exposed harness-only
CSRF-header and profile-response-shape mistakes; both were corrected without
disabling authentication/CSRF or changing production backend behavior.

Plan: `../superpowers/plans/2026-09-20-workspace-p2-remediation.md`.

## Sequence Before Enabling Owned Workspaces

1. Current-dev integration and expanded PostgreSQL regression are complete;
   see `workspace-dev-integration-2026-09-20.md` for the active worktree and evidence.
2. Finish deletion writer fences/cleanup, explicit draft conflict resolution and
   journal consolidation, and remaining owned mutation boundaries.
3. Enable the route only for real backend/WebUI/CDP acceptance, including actual
   JWT/cookie identity changes and PostgreSQL coverage.

The unmerged stream is **not merge-ready**. Closing these six findings does not
complete the remaining integration and acceptance work.

## Current-Dev Integration Follow-Up

The full dirty checkpoint was preserved and transferred to a separate managed
worktree at fetched dev `d72b1d2850e`. Upstream global `/auth/sessions` readiness
supersedes the earlier R6 profile probe to support authenticated unverified users;
owned workspace identity checks still require the canonical profile and expected
user. Real HTTP checks verify that distinction rather than weakening auth denial.

The integrated PostgreSQL run exposed request-owned connection cleanup and absent
message-deletion sync logging. Both now have bounded fixes and regressions;
independent reviews found no actionable issues. The expanded final backend run
passed 262 tests without skips, including real PostgreSQL fixtures. The integration
report records the complete evidence. This is not full WebUI/CDP acceptance and
does not enable the owned route.
