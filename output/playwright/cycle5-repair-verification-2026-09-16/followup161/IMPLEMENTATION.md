# UAT161 — development indicator position

Task: TASK-13260.98. Base HEAD: d76746ac9a69d2ead927e0f2b15434d6740556ce.

## Diagnosis and bounded design
The retained native failure at 1200x969 shows the collapsed sidebar Settings button visible and enabled, but the Next portal intercepts its center pointer click. Keyboard Enter opens Settings. The captured normal indicator has data-error=false and fixed bottom:20px/left:20px; this is not a runtime error overlay.

Installed Next.js is 16.1.4. Its config-shared.d.ts and config-schema.js support devIndicators.position in all four corners. build/define-env.js keeps __NEXT_DEV_INDICATOR true for any object-valued setting and passes the position to the DevTools state. Repositioning therefore retains the development error badge and existing overlay behavior.

Selected design: one existing config option, bottom-right. Upper corners contain sidebar toggle or header Settings/notification controls; the Buddy default is y=96. The observed bottom-right area is clear. Actual chat/composer and native geometry remain browser acceptance controls, not inferred guarantees. No product layout, portal CSS, suppression, or new abstraction is needed.

## Change
Only production file: apps/tldw-frontend/next.config.mjs, comment plus devIndicators: { position: 'bottom-right' }.
No test source edits: a config-string assertion would duplicate this low-impact option. The meaningful RED is the retained native pointer failure; root owns its actual post-change geometry/click GREEN.

## Verification
- node --check apps/tldw-frontend/next.config.mjs: exit 0.
- From apps/tldw-frontend, bunx eslint next.config.mjs: exit 0, no warnings/errors, same baseline.
- Installed Next configSchema.safeParse on the loaded config with safe advanced-mode environment: success, indicatorEnabled=true, bottom-right. See schema.log.
- From apps/tldw-frontend, bunx vitest run __tests__/next-config-dev-watch-guard.test.ts __tests__/next-config-quickstart-health.test.ts: 12/12 tests, 2/2 suites pass. See config-tests.log.
- git diff --check -- apps/tldw-frontend/next.config.mjs: exit 0.
- source .venv/bin/activate; python -m bandit apps/tldw-frontend/next.config.mjs -f json -o .tmp/uat161-dev-indicator-20260916/bandit.json: exit 0, unsupported JavaScript AST parse error; this provides no JavaScript security assurance. There are no new request/data handling paths in the two-line configuration change.

## Root-owned native acceptance
Root reported and retained post-restart GREEN. I read the receipts: Settings rectangle x=7.5,y=905,width=32,height=32; visible DevTools button x=1146,y=899,width=32,height=32. They do not intersect. The normal unforced pointer click navigated Home to /settings. Evidence: output/playwright/cycle5-repair-verification-2026-09-16/followup161/after-geometry.txt, after-click.txt, after-click.png. The geometry operation began before editing but completed after automatic Next restart; it is correctly labeled AFTER, never claimed as a baseline. The prior screenshot and click timeout remain the separate RED.

A chat composer route check remains a suggested control for migrated collisions; no claim of all-route geometry acceptance is made by this report. Root owns closure and any further native work.

Next can persist an explicit developer position in distDir/cache/next-devtools-config.json. Installed DevTools reducer uses that position in preference to the configured initial position. This patch intentionally preserves the developer preference; root should inspect actual rendered placement/use the fresh owned build rather than silently deleting preferences. I issued no runtime/browser commands or commits.

## Evidence
Original native RED: .tmp/uat142-native-20260916/settings-open.txt, settings-pointer-intercept.png, portal-diagnosis.txt (retained, not rerun by author).
Current directory: owned.patch, next.config.baseline.mjs, next.config.review.mjs, parse.log, schema.log, eslint.log, config-tests.log, bandit.json, diff-check.log, next-devtools-source-excerpts.json, owned-manifest.json.
Independent review pending at handoff; root owns task closure/staging after the retained actual native GREEN.
