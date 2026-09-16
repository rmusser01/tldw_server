# UAT134 — WebUI connection probe cannot refresh expired JWT

## Evidence and limit

`/private/tmp/cycle5-multi-native-expiry-late-status.txt` observes `/notes` at09:17:26.924Z. Its newly installed6.5-second response listener captures **two actual `/api/v1/auth/sessions`401 responses**. The listener only observes responses and is removed afterward; it does not issue these requests. Thus the requests are not stale listener output. The97 console errors are cumulative and must not be counted as97 newly confirmed failures. No refresh request, current refresh-token validity or refresh rejection is proved by this limited artifact; prior08:39:30 successful rotation and~09:09:30 access expiry are context supplied by root.

## Confirmed client boundary

- `apps/packages/ui/src/store/connection.tsx:869–891` selects `/api/v1/auth/sessions` for multi-user readiness and calls `apiSend`.
- `apps/packages/ui/src/services/api-send.ts:193–200` uses the fresh direct-browser config resolver and `tldwRequest`, but supplies **no refreshAuth callback**. `directRecipeRequestAuthority` only supplies recipe principal/dispatch ownership, not token refresh.
- `apps/packages/ui/src/services/tldw/request-core.ts:553–560` refreshes a multi-user401 only when `runtime.refreshAuth` exists. Thus this WebUI probe returns401 even with a refresh token available.
- `connection.tsx:1003` classifies the401 as auth ERROR/disconnected. `useServerOnline.ts:99–115` continues the shared mounted poller; `connection-timing.ts` switches from30-second connected to5-second disconnected interval. Layout and Notes subscribe to that poller. With no refresh attempt, no refresh-terminal invalidation occurs, and stored credentials keep the missing-auth early stop from applying. This explains repeated sessions401 until another transport refreshes, login changes, or subscribers leave.
- The existing `background-proxy.ts:839–871` direct runtime *does* provide the shared refresh path. This is a separate caller boundary from the prior hasNewerCurrentAccessToken / rotated-pair invalidation repair, not evidence that the previously repaired rotation algorithm failed again.

## Private actual transport control

`/private/tmp/cycle5-auth-probe-refresh.config.ts` transforms the existing api-send suite in memory to run actual apiSend, direct config resolver and request-core; existing storage/runtime seams and synthetic fetch responses remain. No real network or credential export. From `apps/packages/ui`: `bun run test --config /private/tmp/cycle5-auth-probe-refresh.config.ts`.

`/private/tmp/cycle5-auth-probe-refresh.log`: **1 RED / 1 GREEN**. Expired synthetic multi-user access with a refresh token makes exactly one sessions call, returns401 and never calls `/auth/refresh`. Valid access control returns200 with no refresh. The failed case is chiefly proof of the missing refresh dispatch; it does not exercise a real issuer, terminal revocation or the full timed poller. No success-after-fix claim.

## Bounded correction and acceptance

Connect the readiness GET to the existing refresh-capable direct transport/runtime while preserving normalized status/error behavior, target binding and current authority/generation guards. Reuse existing canonical token-pair rotation, shared single-flight, invalidation and late-owner handling; do not build a second token store or refresh implementation. If extracting a reusable runtime is necessary, keep it narrow and review dependency cycles. Do not accidentally extend automatic mutation replay across apiSend/recipe writes merely to repair this safe GET.

Permanent controls: actual connection store→apiSend/core→synthetic refresh200→retry200 across first and subsequent rotations; simultaneous probes single-flight; expired/invalid refresh terminal401 invalidates current pair and stops further unauthenticated polling; access401 from an older owner/target cannot invalidate replacement or send its credentials to the old target; A→B→A and stale read/event; valid access, hosted/cookie auth, single-user key, 403 and network failures retain their established behavior; recipe uncertainty/no automatic write replay remains green. Native isolated-context second expiry and true terminal revocation checks are still required after correction.

Read-only diagnosis. No token contents printed, live auth storage read/export, repo/task/source/test/browser/runtime edits or inference. Hashes: `/private/tmp/cycle5-controller-additional-diagnosis-hashes.txt`.
