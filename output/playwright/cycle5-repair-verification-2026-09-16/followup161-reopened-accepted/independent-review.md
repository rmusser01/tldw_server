# UAT161 independent review

Verdict: **clear within the bounded configuration and retained native controls**. Read-only review; no browser, runtime, source, task or git changes by this reviewer.

## Source

`apps/tldw-frontend/next.config.mjs` SHA256 `7a78b19818d04e6d23e12ededa457217996eb1abf2154e929ce9d47d60d7ead1` matches the frozen manifest. The only functional change replaces the badge position with `devIndicators: false`; two comment lines explain the boundary.

I inspected the installed Next16.1.4 `server/config-schema.js`, `build/define-env.js`, and `next-devtools/userspace/pages/pages-dev-overlay-setup.js`. The schema explicitly accepts false. The environment mapping disables the optional indicator. Pages overlay registration still installs error/unhandled-rejection listeners and the console.error bridge, and the overlay renderer/error boundary are not conditional on this flag. This avoids a route-dependent choice of badge corner while preserving error diagnostics.

## Retained native and test evidence

- Original `../uat-study-native-20260917/image-card-create-result.txt`: ordinary Create pointer action timed out after5000ms because `nextjs-portal` intercepted events.
- `create-geometry.txt`, 2026-09-17T01:14:54.335Z: badge count0 and actual Create button owns its center.
- `create-normal-pointer.txt`: ordinary `.click()` reaches the actual Create action and shows both required Front/Back validation messages. This proves the original obstruction is gone without saving a duplicate card; it is not a successful populated-card submission claim.
- `settings-normal-pointer.txt`: ordinary sidebar Settings click reaches `/settings`.
- `error-visibility-result.txt`, 01:16:34.589Z: a deliberately labeled synthetic TypeError ErrorEvent is dispatched in a disposable tab and its diagnostic text is visibly rendered. I independently viewed `error-visible.png`: the actual Next Runtime TypeError panel shows that exact label. This is a diagnostic control, not an observed product defect or a no-errors claim.
- `schema.json`: schema valid and indicator flag false. `config-tests.log`: root's12 tests/2 suites pass. I inspected these retained results; I did not rerun broad tests.

No force click, CSS pointer bypass, portal removal or console suppression appears in the reviewed change. These receipts support native acceptance of the badge obstruction repair while preserving runtime-error visibility. No claim is made about every screen size or every route.
