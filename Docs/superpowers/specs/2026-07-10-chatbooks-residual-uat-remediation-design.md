# Chatbooks Residual UAT Remediation Design

**Status:** Approved design and technical/UX spec review
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

Certification uses the bytes included in the browser-downloaded full-account archive. The UAT must extract and hash each fixture stored-media payload directly from that archive, then prove three-way SHA-256 equality between the source fixture bytes, archive payload bytes, and restored destination bytes. Before import, the source server stops and its entire data root is quarantined. The destination server starts with a fresh independent data root that contains no source databases or storage. A local trap endpoint records and fails any attempted external-media retrieval. The restore must therefore succeed from the downloaded archive alone. Restored vector identifiers must match the exported records.

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

Before implementation, rebase this worktree onto the latest `dev` and re-check migration numbering, package output, and existing notification behavior. The SQLite migration is 090 in the reviewed worktree, but implementation must use the next available migration number after that rebase.

## 6. Notification Authorization Design

### 6.1 Permission seeding

Add the next available SQLite migration after rebasing (090 in the reviewed worktree). PostgreSQL requires both a fresh-install seed update and an idempotent backfill in the existing AuthNZ ensure flow.

Both database backends must:

1. insert `notifications.read` and `notifications.control` if absent;
2. grant both permissions to each backend's built-in interactive roles when present: `admin`, `user`, `reviewer`, `moderator`, and `viewer`; and
3. leave custom role grants and explicit user-level allow/deny overrides unchanged.

These are global permissions for a user's personal notification inbox. They must not be copied into organization or team scoped-role tables, because the inbox is not an organization or team resource.

The migration is additive and idempotent. Reapplying it must not duplicate permission or role-permission rows. Rollback must not delete notification permissions that may already be referenced by custom roles; if the migration framework requires a rollback function, it should remove only grants proven to have been created by the migration, or document the migration as forward-only following current project convention.

PostgreSQL role grants must run after baseline roles are seeded. Fresh installation must include the two permissions in the baseline role seed, and the idempotent ensure/backfill must run after role seeding for existing installations. Calling only the current pre-role ensure hook is insufficient. Backend parity means equivalent effective access for the interactive system roles each backend actually defines; it does not require renaming `viewer` or adding `moderator` to PostgreSQL.

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

Normalize the structured HTTP metadata already used by both runtimes. Extension failures expose `status` through `background-proxy`; direct WebUI API failures expose `status` or `statusCode` through `ApiError`. The WebUI SSE opener must stop throwing a message-only `Error` and preserve HTTP status in that existing error shape. A shared pure classifier may read `status` and `statusCode`, but it must not create a competing request-error hierarchy or parse status only from human-readable text.

The stream policy is:

| Failure | Behavior |
| --- | --- |
| Abort or explicit unsubscribe | Stop silently |
| 401 | Stop notification requests, enter `auth-required`, and invoke the runtime's existing session-expiry/auth recovery path |
| 403 | Stop notification requests and enter `unavailable`; no automatic reconnect |
| Other non-retryable 4xx | Terminal error; no automatic reconnect |
| 408, 425, 429, 5xx, or network error with no HTTP status | Retry with bounded exponential backoff |

Backoff starts at the existing 1.2-second interval, doubles to a maximum of 30 seconds, honors a valid longer `Retry-After` value for 429/503, and resets after a successful connection. Tests use injected delays or fake timers; production jitter may be added to avoid synchronized clients, but test assertions must remain deterministic.

This policy applies to every initiator, not only the shared stream loop:

- the extension background unread bootstrap and SSE subscription;
- the WebUI `WebLayout` unread poll;
- the WebUI toast bridge bootstrap and SSE subscription; and
- the WebUI notifications page bootstrap, mutations, and SSE subscription.

A terminal bootstrap response must prevent the corresponding poll or stream from starting. A terminal response after startup must cancel its timer or stream. A transient bootstrap failure may recover through bounded retry without blocking Chatbooks.

Automatic backoff applies only to idempotent reads, bootstrap requests, polls, and SSE connections. Notification mutations are never replayed automatically unless their endpoint first has an explicit idempotency-key contract. A transient mutation failure is shown once in the initiating surface and requires an explicit user retry; a terminal mutation failure updates the shared lifecycle state through the same classifier, preserving `auth-required` for 401 and `unavailable` for 403.

### 7.2 Runtime adapters and lifecycle

Use one shared pure state model with runtime-specific adapters. The extension adapter persists state and unread count in safe extension storage. The WebUI adapter holds state in its authenticated layout/provider and passes it to the header, toast bridge, and notifications page. Do not assume WebUI uses the extension background subscription or extension storage.

The states are:

- `idle`: not started or explicitly stopped;
- `connecting`: bootstrap or stream connection is in progress;
- `active`: the transport has confirmed an open HTTP stream and bootstrap data is current;
- `degraded`: a retryable failure is backing off and data may be stale;
- `auth-required`: the current session is no longer authorized and sign-in recovery is required; or
- `unavailable`: the authenticated principal lacks permission or the endpoint returned another non-retryable response.

The stream transport must expose an explicit connection-open callback after a successful HTTP response/body is acquired. `active` cannot be inferred from creating an unsubscribe handle or from waiting for the stream promise to return. It returns to `connecting` during an intentional reconnect and to `degraded` only after a retryable failure.

401 and 403 have different restart contracts:

- `auth-required` restarts automatically only after a successful login, token refresh, or authenticated principal change;
- `unavailable` restarts after server/endpoint change, a successful explicit authenticated-user/effective-permissions refresh, or an explicit user `Try again` action; and
- logout synchronously stops all notification work and clears rendered notification metadata before the next principal can render.

There is no passive role-change polling. A permission grant becomes visible through an existing explicit account/permission refresh if that flow is available, or through the user's `Try again` action.

Extension unread count and lifecycle keys must be namespaced by normalized server identity and stable principal identity. If either identity is unavailable, clear and suppress the prior value rather than reusing a global count. WebUI state resets synchronously when its authenticated principal or server identity changes. Tests must cover login, logout, token refresh, account switch, server switch, role grant, and a failed explicit retry.

### 7.3 User-facing state

For the default built-in roles, the existing bell and inbox behave normally.

For an intentionally restricted role:

- suppress the unread badge and stale unread count;
- keep the notification control as an enabled native button so the missing capability is discoverable;
- expose `Notifications unavailable for this account` on hover and focus, and open a compact status popover on activation;
- allow one explicit `Try again` request from that popover without enabling an automatic retry loop; and
- do not emit repeated toasts, system notifications, console warnings, or network requests.

A 401 state says `Sign in again to view notifications` and defers to the existing authentication recovery flow. It must not use the restricted-account message. A retryable outage suppresses the header badge and marks any retained inbox count as `Last updated before the connection was lost`. The header must not invent a zero count while disconnected.

The active bell's accessible name includes its unread count, for example `Notifications, 3 unread`; the visual badge itself is `aria-hidden`. Non-active names include the state, for example `Notifications, reconnecting`, `Notifications, sign-in required`, or `Notifications unavailable`. State transitions are announced once through a polite status region, not on every retry.

`idle` is internal-only before the authenticated shell starts the notification lifecycle or after logout. It renders no bell and no badge. An authenticated shell must transition synchronously from `idle` to `connecting`; it may not remain silently idle.

If a popover is used, the button exposes `aria-haspopup`, `aria-expanded`, and `aria-controls`; keyboard activation opens it, Escape dismisses it, focus moves predictably within it, and dismissal returns focus to the bell. Tooltip content remains available on hover and focus and satisfies dismissible/hoverable behavior. The status button itself performs no notification API request except when the user activates the explicit `Try again` action.

Direct navigation to the notifications page must render the corresponding `auth-required`, `unavailable`, or `degraded` state and suppress actions that cannot succeed. Notification failure must never block Chatbooks export or import.

This behavior applies the relevant usability heuristics: visibility of system status, error prevention, consistency, recognition rather than recall, and useful recovery guidance without exposing internal RBAC terminology.

## 8. Production Extension Package-Health Design

### 8.1 Exact-artifact probe

Add a deterministic package-health command under the extension workspace. It must:

1. require an explicit normal production Chrome MV3 output directory and never rebuild it implicitly;
2. validate `manifest.json` and every manifest-referenced local file before launch;
3. reject symlinks in the production output and compute its fingerprint before any browser launch;
4. canonicalize the explicit output realpath, pass that same value as the only path in both `--disable-extensions-except` and `--load-extension`, and launch Chromium with a fresh temporary user-data directory;
5. prohibit certification-time copying, staging, manifest-key injection, host-permission patching, entrypoint replacement, or any other output mutation;
6. discover the extension ID dynamically and grant optional host access through the existing runtime-permission path rather than changing the manifest;
7. use one shared direct-output launcher for package health and final acceptance;
8. wait for a usable extension target, deriving the extension ID from the service worker or an extension page rather than assuming the worker exists immediately;
9. open a known extension page and assert both a stable app-ready marker and an extension-storage read/write sentinel;
10. enforce separate launch, storage-sentinel, and app-ready timeouts; and
11. always close processes and preserve concise diagnostics on failure.

The build fingerprint is the SHA-256 of a deterministic package manifest containing every regular production-output file's sorted POSIX-style relative path, byte length, and file SHA-256. Symlinks are rejected rather than followed. The probe writes its machine-readable result outside the fingerprinted output directory. It recomputes the fingerprint after package health and requires equality. The extension UAT must recompute the fingerprint before launch, require equality with the probe result, and recompute it after UAT to prove the same directory was loaded without mutation.

Failure output must include the sanitized output label (for example, `chrome-mv3`), build fingerprint, file count and total bytes, relative manifest references, elapsed phase, sanitized browser stderr, observed target types, and relative names of the largest files. It must never emit an absolute extension root, repository root, home directory, credentials, or user data into retained logs or release artifacts.

The probe passes only against the untouched output of the normal production build. A build-time change may stop emitting files proven unreachable from production, but no test step may delete asset classes, rewrite the manifest, copy to a staged directory, inject a key, patch permissions, or replace entrypoints after the build. Loaded-path proof consists of unit-tested launcher arguments containing the same canonical realpath in both extension flags, a pre-launch fingerprint of that realpath, an observed extension target from that process, and an equal post-launch fingerprint. Retained evidence records only a sanitized label and fingerprint, never the absolute realpath.

### 8.2 Asset-tree isolation

Before changing production output, add a diagnostic isolator that copies candidate subsets into temporary directories and reruns the launcher in diagnostic mode. It should bisect by top-level asset class and then by dependency-closed file batches while preserving the minimum valid manifest graph.

The diagnostic output must identify a reproducible minimal failing set or a measurable threshold condition. The production correction then belongs at the source that emits or references the offending files, such as WXT/Vite configuration, import structure, locale generation, or asset copying. Temporary diagnostic packages are evidence only, use a visibly different diagnostic result type, and cannot satisfy package health or release acceptance.

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
- PostgreSQL fresh-seed ordering, post-role backfill, idempotency, and grant-parity tests using the project fixture.
- Registration/default-role and UAT-fixture assertions.
- Notification stream unit tests for structured direct-WebUI and extension errors, terminal 401/403, other 4xx, transient retry, `Retry-After`, backoff cap, cursor retention, connection-open, reset after success, and unsubscribe.
- Extension adapter tests for initial terminal failure, principal/server namespacing, synchronous stale-count clearing, state transitions, and restart after auth/config/permission events.
- WebUI tests for the layout unread poll, toast bridge, notifications page, terminal bootstrap suppression, timer cancellation, and shared availability presentation.
- Header/inbox accessibility tests for active count naming, connecting, degraded, auth-required, restricted, explicit retry, popover keyboard behavior, focus return, and one-time polite announcements.
- Extension build-warning, font-resolution, manifest-reference, and startup-budget tests.
- Direct-output package-health tests for no staging/mutation, loaded-path identity, fingerprint equality, result placement outside output, symlink rejection, storage sentinel, and initially absent service worker activation.
- Certification-report tests for checked-in expected test-ID manifest validation, phase-qualified aggregation, exact selected-set equality, and failure on skipped, missing, renamed, duplicated, or extra certification tests.
- Central diagnostic-sanitizer negative tests for API keys, bearer/cookie headers, absolute paths, home paths, browser command lines, archive names, and raw child-process output.

### 10.2 Exact browser acceptance

Run the host UAT outside the sandbox for both surfaces. The extension run must use the same normal production output that passed package health.

The extension acceptance run must:

1. accept an explicit production-output directory and package-health result, without rebuilding, staging, copying, or patching either;
2. recompute the production-output fingerprint, require equality with the package-health result, and retain that fingerprint with the UAT result;
3. create a source standard user with verified effective notification permissions;
4. create the media-bearing full-account fixture and hash its stored-media bytes before export;
5. export through the extension `/chatbooks` route with the Full export contract;
6. retain the exact browser-downloaded archive, extract the fixture stored-media payload from it, and prove source-fixture and archive-payload SHA-256 equality;
7. stop the source server and quarantine its entire data root so the destination process cannot resolve source databases, stored media, or absolute source paths;
8. start the destination server with a fresh independent data root containing no source account data and a distinct clean destination user;
9. configure the fixture's external-media location to a local trap endpoint and fail the run if the destination process attempts any external retrieval;
10. import the exact downloaded archive into that destination user;
11. verify restored profile/settings values, characters, media, transcripts, chunks, and other fixture records;
12. prove archive-payload and restored stored-media SHA-256 equality;
13. verify vector identifiers;
14. confirm standard-user notification requests do not return 401/403 or retry terminal failures;
15. exercise a restricted principal, keyboard-accessible unavailable state, badge suppression, explicit retry, and recovery after a permission grant;
16. exercise 401 auth-required behavior and automatic recovery after successful reauthentication;
17. activate and continue when the MV3 service worker is initially absent, failing only if no usable extension target appears before timeout;
18. aggregate the machine-readable source-export, destination-import, notification, and extension-activation phase reports into one certification result with phase-qualified test IDs, compare that result against a checked-in manifest of required IDs, require exact set equality with every required ID passed and zero skipped, and fail preflight when required environment is absent; and
19. recompute the production-output fingerprint after UAT and require it to remain unchanged.

WebUI UAT is rerun as a regression gate. Neither surface may pass if export did not fire, import used a different archive, destination state was not isolated and clean, only inventory counts were checked, a required test skipped, or the machine-readable result omitted a required assertion.

### 10.3 Quality gates

- Extension compile and production build.
- Relevant frontend unit and browser tests.
- Relevant backend unit and integration tests.
- Bandit on touched backend production paths.
- `git diff --check`.
- Zero duplicate-export, circular-service-import, or unresolved-font warnings in the touched build scope; no startup-budget violation.
- One central sanitizer processes every retained package-health/UAT diagnostic write, including child-process stdout/stderr and browser command lines.
- UAT logs, machine-readable reports, package-health results, and archive checks contain no credentials, authorization/cookie headers, archive names containing account data, absolute source paths, home-directory paths, or unredacted sensitive values.

## 11. Rollout and Rollback

The permission migration is additive. Client behavior remains backward compatible with servers that return transient errors and becomes quieter for servers that return terminal 4xx responses.

If the notification UI state causes a regression, the UI presentation can be rolled back independently while retaining terminal retry classification and backend permission grants. If the extension output correction causes a runtime regression, revert that build-source change, not the package-health probe or certification gate. Startup budgets may be lowered after optimization but must not be raised merely to restore a green build.

## 12. Completion Criteria

TASK-12098.4 is complete when notification authorization, fresh-install/backfill ordering, structured error parity, every WebUI and extension initiator, principal-scoped state, accessible recovery states, and focused browser tests pass for both runtimes.

TASK-12098.5 is complete when the exact production directory loads directly without staging or mutation, its pre/post fingerprints match, the storage sentinel and app-ready marker pass, build-integrity findings are addressed, startup budgets are enforced, and its machine-readable package-health result is ready for the acceptance harness. It owns package readiness, not the final account-data round-trip result.

TASK-12098.3 owns the shared final UAT evidence and closes only after its remaining acceptance criteria are updated and satisfied using the fingerprinted artifact from TASK-12098.5, an isolated destination data root, archive-extracted media hashes, an external-retrieval trap, and a machine-readable zero-skip report. WebUI success alone is not packaged-extension certification.
