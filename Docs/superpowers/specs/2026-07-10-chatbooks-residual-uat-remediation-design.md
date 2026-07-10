# Chatbooks Residual UAT Remediation Design

**Status:** Approved design and independent spec review
**Date:** 2026-07-10
**Tasks:** TASK-12098.3, TASK-12098.4, TASK-12098.5
**Evidence:** `Docs/Reviews/CHATBOOKS_POST_MERGE_UAT_UX_REVIEW_2026_07_09.md`

## 1. Purpose

The exact WebUI full-account backup and clean-destination import passed host UAT. The remaining work is to:

1. certify the same full-account workflow in the normal production browser-extension artifact;
2. correct notification authorization and stop terminal authorization failures from causing silent request loops; and
3. remove extension build-integrity warnings and enforce measurable startup budgets.

These are release-hardening changes. They do not redesign the Chatbooks archive contract or reduce the meaning of **Full export**.

## 2. Non-Negotiable Full-Export Invariant

A full export contains all data associated with the exporting user's personal account that the product supports exporting, including media records, the account-owned stored and derived media data represented by the archive contract, transcripts, chunks, embeddings or vector records, profile and settings data, characters, and the other account-owned records in the inventory.

This work must not:

- convert Full export into an inventory-only export;
- replace media data with pointers when the archive contract requires the data itself;
- fetch, synthesize, or infer media bytes from an external location during certification;
- relax the clean-destination restore checks; or
- treat a build, unit test, or reduced extension package as equivalent to a browser round trip.

Certification uses the bytes included in the browser-downloaded full-account archive. The UAT must extract and hash each fixture stored-media payload directly from that archive, then prove three-way SHA-256 equality between the source fixture bytes, archive payload bytes, and restored destination bytes. Before import, the source user's stored-media directory must be moved outside the server-visible root and external media retrieval must be disabled. The restore must therefore succeed from the downloaded archive alone. Restored vector identifiers must match the exported records.

The approved archive contract may retain only its existing explicit security exclusions, such as non-exportable authentication secrets. This remediation adds no new exclusion, pointer-only substitution, or unsupported-data loophole.

## 3. Evidence and Root Causes

### 3.1 Notification authorization

The notification endpoints require `notifications.read` or `notifications.control`. The constants exist, but the current SQLite permission seed and PostgreSQL parity path do not seed or grant them. The host-UAT user also had no effective `user_roles` row because the fixture used a low-level user-creation helper without the canonical role-assignment step.

The UI therefore issued four initial 403 requests and then retried the notification SSE request approximately every 1.2 seconds. `createNotificationStreamSubscription` currently reconnects after every error, regardless of HTTP status, while the shared background subscription records itself as active and ignores the error.

### 3.2 Production extension launch

Chromium and Playwright launch normally without the extension and with a minimal MV3 extension. The exact production extension starts a Chrome process but never establishes a usable persistent context before timeout. Replacing the background worker, reducing the manifest, removing locales, and removing manifest-declared content surfaces did not clear the stall. The remaining evidence points to a condition in the generated production asset tree, but the exact file or asset class has not yet been isolated.

The implementation must diagnose that condition before choosing the production fix. It must not declare success using a post-build-stripped or mocked artifact.

### 3.3 Build integrity

The production build reports:

- duplicate auto-import exports for `MediaNavigationFormat` and `estimateStorageCost`;
- circular chunk warnings caused by consumers importing through the broad `@/services/tldw` barrel;
- invalid absolute font URLs and JetBrains declarations with no bundled JetBrains files; and
- generic large-chunk warnings without startup-specific budgets.

The current unpacked output is about 45 MB. `background.js` is about 653 KB, `content-scripts/copilot-popup.js` about 460 KB, and several optional lazy chunks are 1.3 MB to 2.6 MB. A useful performance gate must distinguish startup-critical dependency graphs from optional lazy features.

## 4. Goals

- Standard authenticated users can use personal notifications without 403s.
- Built-in role grants remain explicit and idempotent across SQLite and PostgreSQL.
- Restricted custom roles do not generate retry storms or misleading notification badges.
- Transient notification failures recover with bounded backoff.
- The normal production MV3 output loads in Chromium through a deterministic package-health probe.
- The exact loaded artifact completes the full-account export/import UAT.
- Build warnings in the affected scope are removed at their source.
- Startup-critical extension graphs have explicit, reviewable byte budgets.

## 5. Non-Goals

- Changing the Chatbooks archive schema, export inventory, or restore semantics.
- Redesigning the Chatbooks screens that passed WebUI UAT.
- Granting notification permissions to every custom role.
- Replacing the project's RBAC model with UI-side feature flags.
- Broad bundle optimization of every optional editor, renderer, or visualization.
- Certifying Firefox or a store-distributed package in this task.

## 5.1 Delivery Structure

TASK-12098.4 and TASK-12098.5 are independent implementation tracks with separate commits and focused verification. Their only shared gate is the final exact-artifact browser acceptance run in TASK-12098.3. The implementation plan must preserve this separation so notification remediation can be reviewed without depending on the extension asset-tree diagnosis.

## 6. Notification Authorization Design

### 6.1 Permission seeding

Add SQLite migration 090 and an idempotent PostgreSQL parity function in the existing AuthNZ migration/ensure flow.

Both paths must:

1. insert `notifications.read` and `notifications.control` if absent;
2. grant both permissions to the built-in interactive roles `admin`, `user`, `moderator`, and `reviewer` if those roles exist; and
3. leave custom role grants and explicit user-level allow/deny overrides unchanged.

These are global permissions for a user's personal notification inbox. They must not be copied into organization or team scoped-role tables, because the inbox is not an organization or team resource.

The migration is additive and idempotent. Reapplying it must not duplicate permission or role-permission rows. Rollback must not delete notification permissions that may already be referenced by custom roles; if the migration framework requires a rollback function, it should remove only grants proven to have been created by the migration, or document the migration as forward-only following current project convention.

### 6.2 Role assignment

The browser-UAT fixture must create users through, or explicitly complete, the same canonical role-assignment path used by production registration. It must assert effective `user` role membership before starting the server.

Implementation must also add a focused test proving that production registration results in an effective default role. If that test exposes the same missing join outside the UAT helper, the production registration path is fixed in this task. Authorization logic must not infer RBAC membership from a legacy `users.role` field at request time.

### 6.3 Backend authorization tests

Tests must cover:

- migration from a database at version 089;
- fresh SQLite initialization;
- PostgreSQL permission and grant parity;
- idempotent reapplication;
- existing custom roles and explicit deny overrides remaining unchanged;
- standard-user read and control access; and
- a custom restricted role continuing to receive 403.

## 7. Notification Client and UX Design

### 7.1 Error classification

Use the structured `status` already attached by `background-proxy`. Do not introduce a second request-error hierarchy.

The stream policy is:

| Failure | Behavior |
| --- | --- |
| Abort or explicit unsubscribe | Stop silently |
| 401 or 403 | Terminal unavailable state; no automatic reconnect |
| Other non-retryable 4xx | Terminal error; no automatic reconnect |
| 408, 425, 429, 5xx, or network error with no HTTP status | Retry with bounded exponential backoff |

Backoff starts at the existing 1.2-second interval, doubles to a maximum of 30 seconds, and resets after a successful connection or event. Tests use injected delays or fake timers; production jitter may be added to avoid synchronized clients, but test assertions must remain deterministic.

### 7.2 Subscription lifecycle

The shared notification subscription exposes one storage-backed state:

- `idle`: not started or explicitly stopped;
- `connecting`: initial unread count or stream connection in progress;
- `active`: initial count succeeded and the stream is running;
- `degraded`: a retryable failure is backing off; or
- `unavailable`: a terminal authorization or other non-retryable response occurred.

The initial unread-count request and the SSE request use the same classification. A terminal initial response must prevent stream startup. A terminal stream response must clear the active flag and cancel the reconnect loop. An explicit auth/session/configuration change may start a new subscription; time alone may not restart a terminal subscription.

### 7.3 User-facing state

For the default built-in roles, the existing bell and inbox behave normally.

For an intentionally restricted role:

- suppress the unread badge and stale unread count;
- keep the notification control focusable so the missing capability is not ambiguous;
- expose `Notifications unavailable for this account` on hover, focus, and activation using the existing tooltip/popover system;
- use `aria-disabled="true"` and an accessible name that includes the unavailable state; and
- do not emit repeated toasts, system notifications, console warnings, or network requests.

A retryable outage keeps the last known count only if it is visibly marked stale in the inbox surface. The header must not invent a zero count while disconnected. Notification failure must never block Chatbooks export or import.

This behavior applies the relevant usability heuristics: visibility of system status, error prevention, consistency, recognition rather than recall, and useful recovery guidance without exposing internal RBAC terminology.

## 8. Production Extension Package-Health Design

### 8.1 Exact-artifact probe

Add a deterministic package-health command under the extension workspace. It must:

1. run or require the normal production Chrome MV3 build;
2. validate `manifest.json` and every manifest-referenced local file before launch;
3. launch Chromium with a fresh temporary user-data directory and only the normal output directory loaded;
4. wait for a usable extension target, deriving the extension ID from the service worker or an extension page rather than assuming the worker exists immediately;
5. open a known extension page and assert a stable app-ready marker;
6. enforce separate launch and app-ready timeouts; and
7. always close processes and preserve concise diagnostics on failure.

The build fingerprint is the SHA-256 of a deterministic package manifest containing every production-output file's sorted relative path, byte length, and file SHA-256. The probe writes a machine-readable result containing that fingerprint. The extension UAT must recompute the fingerprint before launch, require equality with the probe result, and recompute it after UAT to prove the tested directory was not mutated.

Failure output must include the sanitized output label (for example, `chrome-mv3`), build fingerprint, file count and total bytes, relative manifest references, elapsed phase, sanitized browser stderr, observed target types, and relative names of the largest files. It must never emit an absolute extension root, repository root, home directory, credentials, or user data into retained logs or release artifacts.

The probe passes only against the untouched output of the normal production build. A build-time change may stop emitting files proven unreachable from production, but no test step may delete asset classes, rewrite the manifest, or replace entrypoints after the build.

### 8.2 Asset-tree isolation

Before changing production output, add a diagnostic isolator that copies candidate subsets into temporary directories and reruns the same launch probe. It should bisect by top-level asset class and then by file batches while preserving the minimum valid manifest graph.

The diagnostic output must identify a reproducible minimal failing set or a measurable threshold condition. The production correction then belongs at the source that emits or references the offending files, such as WXT/Vite configuration, import structure, locale generation, or asset copying. Temporary diagnostic packages are evidence only and cannot satisfy release acceptance.

### 8.3 Service-worker startup

The launcher must support normal MV3 behavior where the service worker is initially absent or suspended. It should first inspect existing workers and extension pages, then open a manifest-declared extension page to activate the extension if needed, and finally wait for either a service worker or the opened extension page. Absence of an initial worker is neither a skip condition nor a failure by itself. Certification fails only if activation still produces no usable extension target before the explicit timeout.

## 9. Build Integrity and Performance Budgets

### 9.1 Duplicate exports

- Keep `MediaNavigationFormat` canonical in `utils/media-navigation-scope.ts`; import it there rather than re-exporting a second declaration from `useMediaNavigation.ts`.
- Keep `estimateStorageCost` canonical in `utils/storage-budget.ts`; `storage-guard.ts` may consume it but must not expose a duplicate auto-import symbol.

Add compile or auto-import-generation coverage that fails if either symbol becomes ambiguous again.

### 9.2 Circular service imports

Affected extension-bundled consumers must import from the narrow implementation module they use, such as `TldwApiClient`, `TldwModels`, or `TldwChat`, instead of the broad `@/services/tldw` barrel. The public barrel may remain for external compatibility, but internal startup paths must not route through it.

The build gate fails on newly introduced circular-chunk warnings in the touched service scope.

### 9.3 Fonts

Move existing Inter, Space Grotesk, and Arimo files through the established bundled-asset path and reference them with build-resolved relative URLs. Remove JetBrains `@font-face` declarations because no JetBrains files are present. Preserve the existing system monospace fallback stack.

Tests must verify that every local font URL emitted by the production CSS resolves to a packaged file and that no root-absolute `/fonts/` URL remains in the extension output.

### 9.4 Startup budgets

Add a checked-in budget manifest and a post-build graph analyzer. The analyzer follows static imports from each startup root and reports raw and gzip bytes without double-counting shared files.

Budget roots are:

- MV3 background service worker;
- each manifest content script;
- sidepanel initial graph; and
- options initial graph.

Initial limits are recorded after the launch-stall root cause is fixed, using the measured passing baseline rounded up with at most 10 percent headroom. Increasing a limit requires a reviewed budget change that names the responsible dependency and reason. Optional lazy chunks are reported separately and do not fail a startup budget unless they become statically reachable from a startup root.

The generic Vite large-chunk threshold may only be adjusted after these graph budgets are active. It must not be globally suppressed, and individual optional chunks larger than the documented lazy-chunk threshold remain visible in the build report.

## 10. Test and Release Gates

### 10.1 Focused automated tests

- SQLite migration and permission-resolution tests.
- PostgreSQL ensure-function and grant-parity tests using the project fixture.
- Registration/default-role and UAT-fixture assertions.
- Notification stream unit tests for terminal 401/403, other 4xx, transient retry, backoff cap, cursor retention, reset after success, and unsubscribe.
- Shared subscription tests for initial terminal failure, stale-count clearing, state transitions, and explicit restart.
- Header/inbox accessibility tests for restricted and degraded states.
- Extension build-warning, font-resolution, manifest-reference, and startup-budget tests.
- Package-health probe against the production output.

### 10.2 Exact browser acceptance

Run the host UAT outside the sandbox for both surfaces. The extension run must use the same normal production output that passed package health.

The extension acceptance run must:

1. recompute the production-output fingerprint, require equality with the package-health result, and retain that fingerprint with the UAT result;
2. create a source standard user with verified effective notification permissions;
3. create the media-bearing full-account fixture;
4. export through the extension `/chatbooks` route with the Full export contract;
5. retain the exact browser-downloaded archive and extract the fixture stored-media payload from it;
6. prove source-fixture and archive-payload SHA-256 equality;
7. move source stored-media data outside the server-visible root and disable external media retrieval before import;
8. import that archive into a distinct clean destination user;
9. verify restored profile/settings values, characters, media, transcripts, chunks, and other fixture records;
10. prove archive-payload and restored stored-media SHA-256 equality;
11. verify vector identifiers;
12. confirm notification requests do not return 401/403 or retry terminal failures;
13. activate and continue when the MV3 service worker is initially absent, failing only if no usable extension target appears before timeout; and
14. recompute the production-output fingerprint after UAT and require it to remain unchanged.

WebUI UAT is rerun as a regression gate. Neither surface may pass if export did not fire, import used a different archive, destination state was not clean, or only inventory counts were checked.

### 10.3 Quality gates

- Extension compile and production build.
- Relevant frontend unit and browser tests.
- Relevant backend unit and integration tests.
- Bandit on touched backend production paths.
- `git diff --check`.
- No new warnings in the touched build scope.
- UAT logs, package-health results, and archive checks contain no credentials, absolute source paths, home-directory paths, or unredacted sensitive values.

## 11. Rollout and Rollback

The permission migration is additive. Client behavior remains backward compatible with servers that return transient errors and becomes quieter for servers that return terminal 4xx responses.

If the notification UI state causes a regression, the UI presentation can be rolled back independently while retaining terminal retry classification and backend permission grants. If the extension output correction causes a runtime regression, revert that build-source change, not the package-health probe or certification gate. Startup budgets may be lowered after optimization but must not be raised merely to restore a green build.

## 12. Completion Criteria

TASK-12098.4 is complete when notification authorization, lifecycle, accessible unavailable state, and tests pass for both runtimes.

TASK-12098.5 is complete when the exact production extension loads, build-integrity findings are addressed, startup budgets are enforced, and its machine-readable package-health fingerprint is ready for the acceptance harness. It owns package readiness, not the final account-data round-trip result.

TASK-12098.3 owns the shared final UAT evidence and closes only after its remaining AC 15 and AC 17 are satisfied using the fingerprinted artifact from TASK-12098.5. WebUI success alone is not packaged-extension certification.
