# UAT237: disconnected Media search notifications — read-only diagnosis

## Outcome and confidence

**Confirmed native symptom; high-confidence frozen-source cause:** after ordinary Disconnect and reload, Media correctly settles on the credential gate but also displays three “Failed to search media” notifications. The wrapper permits protected Media content to mount during initial connection testing, and the search hook starts manual searches before credential readiness is established. Its uncancelled asynchronous work can also refetch or report errors after the wrapper switches to the gate.

This is a first-render and asynchronous-lifetime gap in Media. The evidence does not establish three HTTP requests, three server 401 responses, a server outage, or failed credential removal. No runtime or browser was operated, no tests were executed, and no database, credential files, private helpers, or raw logs were accessed for this diagnosis. Only this ignored audit is written. Parent owns UAT237 task/tracker and any repair.

## Observed native sequence

- `auth-disconnect-click.txt`: normal Settings Disconnect action result at **2026-09-17 13:58:20.174 UTC**.
- `auth-disconnect-snapshot.txt`: Settings key input has no displayed value; Disconnect is still loading. This is a UI observation, not an inspection of credential storage.
- `auth-disconnected-reload.txt`: reload capture at **13:59:03.140 UTC**. It is an early capture and does not establish settled Media state.
- `auth-disconnected-media-snapshot.txt`: final snapshot URL is `/media`; title “Add your credentials to use Media,” explanation that valid credentials are required, and **three distinct notifications** with “Failed to search media.” Parent reports the initial navigation was `/media?id=1`; the supplied settled snapshot itself has no query string.
- `auth-recovery-settings-open.txt`: immediate navigation capture at **14:00:19.507 UTC** still includes the prior gate and all three notifications.
- `auth-recovery-settings-snapshot.txt`: Settings is open with no displayed key value. It does not prove a successful reconnect.

These selected captures contain no search request/response trace sufficient to assign each notification to a particular operation. Exact multiplicity remains unproven.

## Causal source path on the frozen revision

1. `/media` is registered by `option-media-view-route-registry.tsx:9–14` and renders `ViewMediaPage` through `option-media.tsx`.
2. `store/connection.tsx:457–477` starts at `SEARCHING`, `isConnected:false`, `isChecking:false`. `types/connection.ts:87–90` maps `SEARCHING` to **testing**. `useServerOnline.ts` returns `isConnected && mode !== "demo"` and starts the existing connection check in an effect.
3. `ViewMediaPage.tsx:122` guards only `!isOnline && uxState !== 'testing'`. Therefore its initial offline/testing state falls through to `MediaPageContent` at line 292. This is a concrete source-level path, not an observed native state-transition trace. Normal background checks of an already-connected store retain connected UX (`types/connection.ts:80–84`, `connection.tsx:682–696`), so an initial loading gate need not unmount healthy background refreshes.
4. `MediaPageContent` invokes `useMediaSearch({t,message})` at line 312. The hook dependency interface (`useMediaSearch.ts:224–227`) has no readiness or authority input. Its query is `enabled:false` (655–680), but its mount/criteria effect **explicitly invokes `refetch()`** (803–833); disabled automatic queries do not disable these manual calls. The default selected kind is media, with a blank query that reaches the Media listing request.
5. A separate media-type population effect starts another listing request (838–878), and **unconditionally awaits `refetch()` after its try/catch** (905–908). It has no cleanup or cancellation/generation check. Thus work started during testing can continue and re-enter search after the content unmounts. Keyword-loading effects also start independently of readiness (1040–1043). These are identified request starters, not a proof that precisely three searches occurred natively.
6. Meanwhile `connection.tsx:744–776` recognizes missing credentials and changes to `UNCONFIGURED`/`configStep:'auth'`, producing the correct settled credential gate. Unmounting the content does not undo a global toast already emitted, and the hook's own async callbacks are not fenced by current mount/authority.
7. `useMediaSearch.ts:483–496` catches non-404 search failures and always calls `message.error('Failed to search media')`; it has no mounted, request-generation, authority, or cancellation guard. The query function does not consume a TanStack query context/AbortSignal. The media-search key also contains no connection authority (655–679), so an authority/lifetime regression should inspect stale success/cache behavior as well as error toasts; this audit does not claim a demonstrated cross-account data leak.
8. The real transport supports caller cancellation via `abortSignal`, but this hook does not pass it. `request-core.ts:445–461` returns a structured local `ok:false,status:401` for a missing single-user key before network dispatch. `background-proxy.ts` maps failed responses to errors. Consequently the three notifications can exist without three backend requests; these captures do not prove which failure branch was taken.

## Prior scope and existing test gap

- **UAT047, TASK13260.5** records passive Notifications/Buddy requests after Disconnect, first disconnected-render fetches, and stale authority health completion (task lines 37–41). Its accepted native window records no new Notifications/Buddy private calls for 4m13s (line 47). That acceptance does not cover Media search mount or its asynchronous type loader.
- **TASK13260.77 is UAT138**, the earlier Media outage repair. It intentionally downgraded only recognized transport errors from `console.error` to a fixed warning while preserving contextual feedback/retry/unexpected-error diagnostics (task lines 17–32). Its stop/restart acceptance used configured credentials. This diagnosis does not invalidate that bounded result, and suppressing all search errors would weaken its contract.
- **UAT077**, if the number refers to the finding rather than task suffix, is TASK13260.22's Provider Keys loading-translation TypeError and is unrelated.
- `ViewMediaPage.connection.test.tsx` checks already-settled auth/setup/unreachable states; it mocks TanStack `useQuery`/refetch and does not cover initial `SEARCHING → configuring_auth` with the actual hook and deferred requests.
- `useMediaSearch.outage.test.tsx` uses the actual hook/QueryClient to preserve outage feedback and recovery, but supplies no connection transition and does not cover unmount/authority change or late type-loader completion. No tests were run in this read-only task.

## Bounded repair guidance and required causal controls

1. Prevent private Media content/search initialization until actual connection/auth readiness is established. Treat initial testing as loading rather than a bypass; preserve already-connected background refresh and the current settled auth/setup/unreachable guidance.
2. Fence the hook's existing asynchronous searches/type/keyword initialization and their side effects to the current mounted authority/request lifetime. Use existing cancellation/lifetime facilities; do not repair only by deduplicating notifications or globally silencing errors. A declarative `enabled` change alone will not stop existing explicit refetch calls.
3. Before a repair, reproduce with actual wrapper + hook + QueryClient: initial testing → missing credentials must initiate no protected reads and no search-error toast; initial ready → Disconnect with a deferred search and type loader must produce no late toast/refetch/cache contamination; a current reconnect must load normally. Include old success and old rejection after authority change, and retain configured transport-outage, unexpected-error, 404 unsupported-endpoint and explicit retry controls.
4. Verify the normal Disconnect/reload/Media route again natively after reviewed source loads. No proposed code, passing test, repair, or native acceptance is claimed here.

## Frozen source and evidence identity

Revision: `8f8774e6c868b304a96d95ab82e28389c129a78b`. Run: `fresh-final-20260917`.
All 15 selected source/history files below match their exact original archive-manifest SHA-256 entries. Only these selected source paths were hashed; no runtime-generated tree was walked.

Archive manifest SHA-256: `26255fe54e27f7e92d849bbf810a7224c655602e3e2f9514eb6cbcce3c96bba1`.

Source paths relative to `sources/sqlite-single`:

| Path | SHA-256 |
|---|---|
| `apps/packages/ui/src/routes/option-media.tsx` | `8e863f9586a4a8d0b56accf58ce95a18688854f4e08ccb83f8045f2dadde1f61` |
| `apps/packages/ui/src/routes/option-media-view-route-registry.tsx` | `72924cc83fd47e4e457a7a853d309d8edd76501209c11e0dff2f2b0d5ff4751a` |
| `apps/packages/ui/src/components/Review/ViewMediaPage.tsx` | `b9e58148dbac0f55a4da39ea5867da0e752492006ac25e4f3aaf7969a08da734` |
| `apps/packages/ui/src/components/Review/hooks/useMediaSearch.ts` | `316dc3845a374d6e169d74cd1d95e5a96345eac7668edf6c801523721c7d7f86` |
| `apps/packages/ui/src/hooks/useServerOnline.ts` | `777b3dd55599d39ad6fae11d893cabe68161af270aa937347e5e458ced05f95c` |
| `apps/packages/ui/src/hooks/useConnectionState.ts` | `c29b208787bf8355c11b8499806d069485ea72128c4f869dae53d8f47ca15649` |
| `apps/packages/ui/src/store/connection.tsx` | `25361312aa847d5d9bb184db0c19b6ac01792f1d83f3e2b706db98fca00002ea` |
| `apps/packages/ui/src/types/connection.ts` | `432d3efbf63bdc5bcb154e89a5831bd1e1cf6141b8cfc8e6caf61aee96b8eef1` |
| `apps/packages/ui/src/services/background-proxy.ts` | `70f1483af5b48f6573340a0e6728a2b363757d61ad4748f32249208ab1bae638` |
| `apps/packages/ui/src/services/tldw/request-core.ts` | `3567b4d2defdab1449031bf51ab5db69548cdb1ec123f856091ef403075487df` |
| `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.connection.test.tsx` | `2cc4de2965485afb78289278ea4ead783a027c9debe8b03836b62c12f30f1068` |
| `apps/packages/ui/src/components/Review/hooks/__tests__/useMediaSearch.outage.test.tsx` | `86bb056dba9c9047521568c4e76b4ea3eadeff81fed92256d16ae191ffa9ff21` |
| `backlog/tasks/task-13260.5 - Repair-UAT-account-scoped-Notes-state-and-connection-recovery.md` | `b715fd4a8a5e121b68e53cc3dfc93200fd1bb912310f949b0840a1b8c5ad006e` |
| `backlog/tasks/task-13260.77 - Handle-Media-API-outages-without-a-runtime-error-overlay.md` | `9596310db2774b79db07b3be5bd1f0f47ac91ca8c749f78009f1451deb4b9592` |
| `backlog/tasks/task-13260.22 - Keep-Provider-Keys-usable-with-actual-loading-translations-and-denied-access.md` | `d0b4bab48757c698d831b45d9355d7c6f6367ecfd81383be26e5ac605472a157` |

Native inputs relative to `native/sqlite-single`:

| Path | SHA-256 |
|---|---|
| `auth-disconnect-click.txt` | `589a61a99e0cfc2534b7210327e73ab7d8fc25c4d326462893ffd5cdd042f30a` |
| `auth-disconnect-snapshot.txt` | `8983ded1dc4cc5cd55d51cf11051a9f5d14bf749956d1c71261f5749f6095eed` |
| `auth-disconnected-reload.txt` | `0cc3212be51cd7572087de34d4c8fbc2ecb0e7d6835e4e0ab3a098fd26b6348c` |
| `auth-disconnected-media-snapshot.txt` | `382338e1feb02e91481f3fd5b7efe663a37b4dc058e85a25ab651ec062433626` |
| `auth-recovery-settings-open.txt` | `885009aba447c9458ee3406cec1d53356c424d98e8c85e185fa0e785022331ab` |
| `auth-recovery-settings-snapshot.txt` | `29dd92cd90dc1587b0f7b2a05c0d8a8c9706fc138b8167187348b1feff93f151` |
