# Independent review — UAT246 Character SSE buffering

**Verdict: CLEAR for the bounded source/test repair. Native acceptance remains pending.**

Task TASK13260.188. Reviewer `/root/retry031_repair`; author `/root/account_access`. Reviewed author source freeze `95260c44c7a0746fcbf85946c79e05cbeaa0a8f3a77002e213ebc4d565a1e445` and final evidence manifest `b852eb08c1c92de33629d1b73f8d1fd8b40d2fb8bb1087ff501a3c7cf9015307`. All three source hashes and all thirty author evidence entries matched before and after replay. Original snapshots match base `86458ab88ce3fa62e6518c9d813c3860254ddb2c`.

## Source review

The production delta is confined to `character_chat_completion` in `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`. Its endpoint-local header map supplies `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no` to exactly three SSE response constructions: text fallback (line 6762), unified provider stream (line 6921), and legacy provider stream (line 7002). No provider choice, authentication, timeout, persistence, request-body, or transport orchestration changed.

The installed Next 16.1.4 compression implementation recognizes `no-transform`. Its real configuration matches the frozen base; the dependency hashes are retained. Adding the directive prevents the demonstrated default compressor from holding small SSE frames. The existing no-cache behavior remains, and legacy/fallback responses receive the same explicit stream policy.

The twelve backend controls construct the actual Character endpoint response with real SQLite/PostgreSQL persistence and local adapter logic. They cover unified/legacy × complete/disconnect/fallback. Header assertions bind the backend response contract to the separate installed-Next HTTP fixture. Complete/disconnect preserve the acknowledged user row, make exactly one controlled provider request, and close response/client/runtime in order. The added fallback never creates the controlled upstream thread, emits content and DONE once, closes runtime, and preserves the same canonical user. Finally blocks release gates and settle the pending iterator; the response background cleanup remains exercised.

The Next tests hold terminal output behind an explicit gate and inspect the first body reader result before releasing it. This prevents eventual `response.text()` success from passing as early delivery. Tests release gates, abort/cancel readers, stop only their own ephemeral Next child and HTTP fixture, and remove their unique temporary app. Existing authorization/body forwarding, upstream 503, cancellation, and responses lasting beyond 30 seconds remain covered.

## Independent replay

| Suite | Result | Evidence |
| --- | --- | --- |
| Installed Next full suite | 8 passed, 0 skipped, exit 0, 65.97s | `next-green.log` |
| Required official PostgreSQL + SQLite backend/adjacent suite | 35 passed, 0 skipped, 5 warnings, 160.44s; runner exit 0 | `backend-receipt.log`, `backend.redacted.log`, `backend-command.json` |

Gzip-negotiated canonical and identity controls each delivered the exact role frame at 3 ms while terminal output was still held. The old-header gzip adverse control had no first body bytes at 303 ms, then returned the exact role/content/DONE sequence after release. This independently reproduces the differential delivery behavior through installed Next.

The author's corrected causal RED inspected first body delivery and failed with terminal held for 2 seconds; its earlier header-only draft is separately retained. Backend RED contains twelve missing-header failures. These receipts support the changed contract; the independent adverse control confirms the buffering effect without relying solely on those logs.

The author's apparent pending-process status was reconciled with its now-completed original handle: official exit 0, no kill/replacement. The independent backend run also exited normally after its test summary. No cleanup defect was found or suppressed.

## Static and scope verification

- Fresh Ruff: four existing findings, zero added/removed; original/current Python source parsed under real logical filenames.
- Fresh ESLint: zero errors/warnings on baseline/current MJS, no ignored files; actual frontend configuration with Next rootDir context.
- Fresh Bandit: production zero findings/errors; Python test zero findings/errors with only test-assertion rule B101 excluded. The author's mixed MJS/Python scan cannot provide JavaScript security evidence and is qualified in its report.
- Python compile, Node syntax and scoped diff whitespace checks pass.
- AST audit confirms only the Character completion definition changed and all three SSE calls use the local map. Three frozen source hashes remain exact.

No source, task/tracker, Git, browser, native runtime/profile/database, or provider changes by the reviewer. The installed-Next replay used only test-owned ephemeral processes/ports and controlled upstream I/O; PostgreSQL used official disposable fixtures with required mode.

## Limits

This review accepts the demonstrated compression-buffering repair. It does not retrospectively establish the cause of every original 45-second timeout, promise timely upstream model output, or change timeout policy. The adverse measurement is a bounded 303 ms held-stream observation. Coverage uses actual backend response tests and a separate installed-Next proxy test with the same asserted headers; deployment/native frame delivery remains the parent's acceptance gate.

No actionable correction requested.
