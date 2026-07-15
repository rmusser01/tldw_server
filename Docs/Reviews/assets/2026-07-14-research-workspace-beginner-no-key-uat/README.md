# Research Workspace beginner/no-key UAT evidence

This bundle is the durable evidence for TASK-12968 and RW-UAT-027. The run used:

- FastAPI on `http://127.0.0.1:18160` in `single_user` mode with a server-only key.
- The advanced-mode Next.js WebUI on `http://127.0.0.1:18161`, with public API-key and bearer variables empty.
- A clean Chrome `145.0.7632.6` profile controlled through CDP on `http://127.0.0.1:18162`, with browser and component extensions disabled.
- Fresh disposable AuthNZ and per-user databases.

`run-beginner-uat.mjs` is the exact runner snapshot used for this evidence, not a portable CI harness. It observes network traffic at browser-context scope for direct backend `/api/v1/`, same-origin `/api/v1/`, and hosted `/api/proxy/` request shapes. A browser-level CDP target observer rejects transient service workers and extension background pages. Dedicated disposable browser contexts emit unique start/end health sentinels that bracket the audited backend log segment without contaminating the desktop or mobile persona contexts. The runner uses `uat-diagnostic-gate.mjs` to fail the run when any context records a page error, failed request, or HTTP error; the helper's focused test covers each failure bucket. The run passed 17 of 17 checkpoints. The correlated access-log segment contained 11 API request lines and zero workspace migration lines.

The raw JSON manifests retain their original `/private/tmp` screenshot paths. The representative committed copies are:

- `desktop-02-settled-workspace.png` -> `desktop-settled-workspace.png`
- `desktop-06-first-run-tour.png` -> `desktop-first-run-tour.png`
- `desktop-09-visible-search.png` -> `desktop-visible-search.png`
- `desktop-11-add-url-auth-recovery.png` -> `desktop-add-url-auth-recovery.png`
- `mobile-01-direct-entry.png` -> `mobile-direct-entry.png`
- `mobile-02-sources-tab.png` -> `mobile-sources-tab.png`
- `mobile-03-chat-tab.png` -> `mobile-chat-tab.png`
- `mobile-04-studio-tab.png` -> `mobile-studio-tab.png`
- `mobile-05-search.png` -> `mobile-search.png`

The manifests are authoritative for assertions and timing. Screenshots are supporting visual evidence.
