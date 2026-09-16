# UAT161 independent review

Task: TASK-13260.98. Reviewer: source013_diagnosis. Date: 2026-09-16.

## Outcome

**No blocking findings.** The frozen change is the smallest supported configuration adjustment: one comment and `devIndicators: { position: 'bottom-right' }` in the existing Next config. It changes no product layout, portal CSS, authentication, networking, or error handler. No mirrored string test or permanent test source was added for this review.

Frozen production and review-copy SHA-256: `215a4527131f8944ded65c6991e534a47d9206554eab5b3b98505d57b0f70d73`.

## Independent installed-version checks

The frontend resolves installed **Next.js 16.1.4** at `apps/node_modules/.bun/next@16.1.4+d9efe0b44b7e6f19/node_modules/next`.

- `dist/server/config-schema.js:444` accepts object-valued devIndicators and explicitly includes bottom-right. `dist/server/config-shared.d.ts:858` documents the same supported configuration. The existing default is bottom-left.
- `dist/server/config.js:831` also accepts bottom-right in its runtime allowed-position check.
- Independently imported the actual exported next.config.mjs in separate Node processes using safe explicit advanced-mode and quickstart-mode environments. Both module imports and the installed `configSchema.safeParse` succeeded. An invalid bottom-middle position was rejected by that same installed schema as a negative control. No server or build process was started. Advanced rewrites remained empty; quickstart retained the configured loopback API rewrites.
- `dist/build/define-env.js:127–129` sets __NEXT_DEV_INDICATOR from `config.devIndicators !== false` and supplies the configured position separately. This change therefore leaves the indicator enabled. The installed pages hot-reloader still dispatches compilation errors through `dispatcher.onBuildError`; no error handler or overlay configuration is changed. The installed compiled DevTools code keeps its buildError/errors state independently of the position setting.

These checks validate the real installed configuration contract and executable exported module, rather than comparing the added source string with a duplicate assertion. Error reporting remains enabled by the framework configuration; this review did not deliberately create a browser compilation/runtime error.

## Retained native evidence

Root owns browser acceptance and runtime restart. The reviewer read root's retained artifacts without operating the browser:

- `after-geometry.txt`: Settings rectangle x=7.5, y=905, width=32, height=32; Next indicator x=1146, y=899, width=32, height=32. Their horizontal intervals are disjoint, so the indicator does not cover Settings in this observed state.
- `after-click.txt`: ordinary locator pointer click reports pointerClick:true and navigates Home to /settings. The retained command uses no force-click or keyboard workaround.

This supports the observed collapsed-sidebar Settings acceptance. It is not a claim that every viewport or product route is collision-free. Root remains responsible for any broader browser acceptance.

## Limits and provenance

Next may persist an explicit developer-selected indicator position under the distDir cache; the author documented that this preference takes precedence over the configured initial position. This patch preserves that preference. The retained native geometry directly confirms bottom-right placement for the observed owned runtime.

The author reports 12 passing existing config tests across two suites, plus clean parse/schema/lint/diff checks. Those checks are author evidence; the reviewer independently performed module/schema validation and read the relevant installed framework code. Bandit cannot analyze this JavaScript config; its parse error provides no JavaScript security assurance. The two-line change introduces no new data or request handling.

Only this private independent-review.md artifact was written by the reviewer. No production/test changes, permanent tests, runtime/browser operations, staging, commit, or tracker edits were performed.

## Frozen evidence hashes

Paths below are relative to this private review folder.

| Artifact | SHA-256 |
| --- | --- |
| `owned.patch` | `5789716c85b031220ab72ba0099d87b6a5069b96866fa1be47941e97b5079637` |
| `IMPLEMENTATION.md` | `9dbc27687b9e5696375b47ee0a878b1f6d4934065f677ef7ee93697c4b984f91` |
| `owned-manifest.json` | `52e940da214c5fd597fd862601db0f63b40f83666fa4a4a8aca2562446d95583` |
| `next.config.review.mjs` | `215a4527131f8944ded65c6991e534a47d9206554eab5b3b98505d57b0f70d73` |

Native evidence:

| Artifact | SHA-256 |
| --- | --- |
| `output/playwright/cycle5-repair-verification-2026-09-16/followup161/after-geometry.txt` | `2e3b82ac7ea10aa4f4dec1db4f39818404cf8f3576475c034bc4ee23f76cdb31` |
| `output/playwright/cycle5-repair-verification-2026-09-16/followup161/after-click.txt` | `9fb0d3a39687b8b77b9ed2fb93c1f236dbb54673a92eb740a1adcbc1133c4fef` |
