# UAT114 read-only harness reassessment

## Verdict

The proposed supported headed Playwright correction was already attempted: disable `Emulation.setFocusEmulationEnabled` on every page, then perform a real `bringToFront()` tab switch. Both detached and retained CDP-session variants left all six pages `visible`, with six active notification streams. Five pages correctly reported `hasFocus() === false`; this establishes loss of focus, not hidden-page behavior. Repeating this same angle cannot close UAT114.

UAT114 native hidden-tab cancellation/catch-up remains **unverified because of the available harness**, not a product pass or a demonstrated product failure. No browser, runtime, permissions, source, or installed tooling was changed during this reassessment.

## Evidence and headed/headless distinction

Repository evidence base: `output/playwright/cycle4-repair-verification-2026-09-16/native-multi/`.

- `six-headed-tabs-streams.txt`: six actual pages; successful stream responses; all six visibility states `visible`.
- `six-tabs-native-focus.txt`: actual executed code creates one CDP session per page, sends `{enabled:false}`, detaches, then calls `c.pages()[5].bringToFront()`. Result: all visible; five unfocused, one focused; active streams 6.
- `six-tabs-native-focus-retained.txt`: actual executed code retains each CDP session, sends the same command, then calls `page.bringToFront()`. Same visibility/focus/stream outcome. This excludes merely assuming the detached session retained its override.
- `native-harness-limitations.md`, `FINAL_REPORT.md`, and `RUNNING_TRACKER.md` contemporaneously identify these as **headed** attempts. `/private/tmp/cycle4-multi-targeted-headed-browser.mjs` selects the distinct `cycle4-multi-repair-headed` session. That wrapper forwards arguments and does not itself force headed mode; an original launch-options receipt/session file was not found in this bounded reassessment. Thus the headed characterization comes from retained contemporaneous records, not a surviving launch metadata audit.
- Current `cycle5-repair-{single,multi}-20260916.session` metadata separately reports Chromium/channel Chrome and `launchOptions.headless:true`. These current sessions have no native visible window to minimize. They do **not** establish that the prior distinct cycle4 session was headless.
- The cycle5 native-single running tracker records two actual Codex IAB tabs also staying visible after switching. No IAB implementation or launch-setting evidence establishes its cause.

## What the tooling explains—and what it does not

Installed local Playwright 1.58.0 source at `apps/node_modules/.bun/playwright-core@1.58.0/node_modules/playwright-core/lib/server/chromium/crPage.js:410–412` enables focus emulation for the main frame. Chrome documents that focus emulation forces page visibility to visible and suppresses visibility changes. This is a direct plausible explanation for the initial result. [Chrome DevTools documentation](https://developer.chrome.com/docs/devtools/rendering/apply-effects)

The same installed version's `chromiumSwitches.js` includes `--disable-background-timer-throttling`, `--disable-backgrounding-occluded-windows`, and `--disable-renderer-backgrounding`. Chromium describes these as scheduling/process/occlusion backgrounding controls. Their presence does not independently prove that hidden tabs must report visible, and the retained experiment did not isolate these flags as the remaining cause after focus emulation was disabled. Removing defaults would be an unverified new experiment, not a diagnosed fix. [Chromium switch definitions](https://chromium.googlesource.com/chromium/src/%2B/lkgr/content/public/common/content_switches.cc)

The wrapper uses `npx --package @playwright/cli`; the local 1.58.0 source is retained implementation evidence, not proof of the exact package version used by every current daemon.

Electron documents that `backgroundThrottling:false` can keep a window's visibility state visible even when hidden or minimized. That is only a possible explanation for IAB; its actual setting was not inspected and must not be asserted. [Electron page visibility documentation](https://www.electronjs.org/docs/latest/api/browser-window#page-visibility)

## Honest supported alternative and present limit

A future ordinary headed Chrome session with working native app control can use real tab selection or real window minimization and read the resulting `document.visibilityState`. It must first demonstrate one genuinely hidden page and one visible page before expanding to six and checking actual stream cancellation/catch-up. Disabling focus emulation is a legitimate removal of an automation override, not visibility spoofing—but that exact Playwright approach already failed here.

Native minimization through the actual window control (or Chromium's supported `Browser.setWindowBounds` minimized state) is also an honest browser-window operation. It is not currently a verified path: the present sessions are headless, and minimizing one window containing six tabs would hide the whole window rather than prove the intended five-hidden/one-visible case. [CDP window bounds API](https://chromedevtools.github.io/devtools-protocol/tot/Browser/#method-setWindowBounds)

Prior supported native fallback attempts are retained: CUA Chrome returned “Browser is not available: chrome”; cua-driver's supported app relaunch did not produce a daemon, and status confirmed it was not running. The outside-daemon permissions probe explicitly warned its result could be inaccurate, so it is insufficient to diagnose the daemon failure as a permissions issue. No available, functioning native Chrome-control path was established. No further launch or retry was performed.

Acceptance disposition: retain native UAT114 as unverified, with automated lifecycle coverage kept separate. Reopen a bounded native attempt only after a genuinely headed browser plus functioning supported native control is available; do not repeat the two already-failed CDP focus variants or the all-visible IAB angle, fake visibility/events, or claim six successful stream connections prove hidden-tab behavior.
