# Task 2E — Phase 6 Character Chat readiness

## Status

**DONE_WITH_CONCERNS.** The Phase 6 failures were stale journey interactions, not a Character Chat direct-entry, responsive-layout, provider, SSE, or streaming regression. The scoped journey now follows the supported layout transitions and the three real-backend viewport cases pass.

## Root-cause trace and working comparison

1. `getCharacterChatRouteIntent("?mode=character")` returns first-class Character Chat intent even without a character id. `Playground` consequently sets `characterWorkflowActive`; the retained controlled desktop/tablet traces pass the `playground-active-chat-mode` assertion before the session assertion runs.
2. Character sessions live in the context rail. The default desktop cockpit can have that rail collapsed. The prior helper saw no session region and unconditionally clicked `playground-chat-layout-mode-trigger`.
3. That control is bidirectional. In cockpit mode it is labelled **Enter focus chat**, not “show panels”; the click switched the page into focus, which intentionally removes the rails. The retained trace records that click followed by no visible `Context` tab and the 30-second missing-session failure.
4. At mobile width, focus is the intentional default. In that state the Character Chat chip and readiness panel are intentionally not rendered, which explains the prior early mobile timeout. After opening cockpit mode, compact layout exposes the context rail through the first `Mobile cockpit panels` tab. The compact cockpit composer does not show the role-play overflow; the existing mobile toolbar contract exposes it after the supported **Enter focus chat** transition.

The working comparison is the existing cockpit contract: **Exit focus** returns to `data-mode="cockpit"`; desktop restores the Context sidechannel through `playground-cockpit-left-rail-restore`; compact layouts select the first mobile rail tab; and the mobile Role-play setup remains available from **More options** in focus.

## Hypothesis and repair

**Confirmed hypothesis:** the Phase 6 journey assumed the layout-mode control only reveals panels. It actually toggles between cockpit and focus, so the stale helper hid the very sessions surface it was trying to assert.

The minimal repair is in the journey only:

- leave focus before asserting the Character Chat chip/readiness surface;
- restore the context rail by the mode-appropriate supported control;
- return to focus only when the mobile role-play overflow is needed.

No product, provider, mock, timeout, sleep, DOM-removal, or assertion-weakening change was made.

## RED/GREEN evidence

### RED

Task 2A's final controlled one-worker fixture run used the tracked `e2e-critical-config.txt` config and a shell-owned Redis/mock/API/WebUI graph. Before this Task 2E journey correction it reported:

```text
3 failed, 2 passed (2.0m)
  - character-chat-phase6: desktop waited for Character chat sessions
  - character-chat-phase6: tablet waited for Character chat sessions
  - character-chat-phase6: mobile waited for playground-active-chat-mode to contain Character Chat
```

The retained desktop trace proves the direct-entry label assertion passed, then records the old helper clicking `playground-chat-layout-mode-trigger` before waiting for the missing sessions region. The bounded mobile diagnostic after the first transition correction confirmed the remaining stale path: in compact cockpit there is no `More options` control until the supported focus transition is entered.

### GREEN

Against the same isolated Task 2A fixture graph (`TLDW_CONFIG_FILE=e2e-critical-config.txt`, mock `127.0.0.1:18091`, API `127.0.0.1:62458`, WebUI `127.0.0.1:62459`, Redis `62457`), the final command was:

```bash
bunx playwright test \
  e2e/workflows/journeys/character-chat-phase6.spec.ts \
  --project=journeys --workers=1 --reporter=line
```

Result:

```text
3 passed (10.5s)
  - desktop
  - tablet
  - mobile
```

## Live viewport results

- Desktop (1440×900): direct Character Chat status, restored context sessions, readiness panel, composer, and Role-play drawer passed without horizontal overflow.
- Tablet (768×1024): direct Character Chat status, compact Context tab/session reachability, readiness panel, composer, and drawer passed without horizontal overflow.
- Mobile (390×844): exited the intentional initial focus view for Character Chat setup/session checks, used the compact Context tab, then used the supported focus-mode overflow to open Role-play setup; all assertions passed without horizontal overflow.

## Verification

- Final three-viewport real-backend Playwright run: **3 passed (21.6s)** in the fix round, with passing traces retained below.
- The focused controller, direct-route, and mobile-toolbar checks: **13 passed** across three files.
- `git diff --check`: exited 0 after the final fix-round edits.
- Bandit: not applicable; this package touches no Python.

## Fix round 1 — strict behavioral TDD and retained evidence (2026-08-31)

### Owning boundary and RED/GREEN sequence

The narrowest owning behavior is the E2E journey's layout-transition decision:
given a visible Cockpit or Focus surface and a viewport class, it must expose
the Character Chat sessions without turning a Cockpit surface into Focus.  The
new E2E-owned `revealCharacterChatSessions` controller has only the six
visible-surface operations the journey already performs.  It deliberately
does not change product code, selectors, or API behavior.

Before correcting that controller or wiring it into the journey, the harness
represented the prior unconditional bidirectional layout-toggle behavior.  Its
two assertions failed: desktop ended in `focus` instead of `cockpit`, and
mobile recorded `toggle-layout-mode` rather than the supported `Exit focus`
then compact Context selection.  This is a behavior-level state/operation
regression, not a source-string test and not a weakened E2E assertion.

After the correction, the same test command passes both transition cases:

```bash
bunx vitest run e2e/utils/__tests__/character-chat-phase6-surface.test.ts
```

```text
Test Files  1 passed (1)
     Tests  2 passed (2)
```

The complete pre-change RED and GREEN transcripts, exact command lines, and
the adjacent direct-route/mobile focused run are committed at
`task-2e-phase6-fix-round-1-evidence.md` (Git blob
`a779ae091b7ed4a1e7badb0a0bb2539b28cb3058`).  The latter focused command
passes 13 tests: controller 2, `character-chat-mode-intent` 6, and
`ComposerToolbar.role-play-mobile` 5.

### Package-visible live evidence

The same evidence artifact records the owned graph's listener PIDs, tracked
`e2e-critical-config.txt` selection, authenticated health check, and the
one-worker real-backend command.  It passed desktop, tablet, and mobile in
21.6 seconds with `--trace=on`.  Passing trace paths and SHA-256 values are:

```text
apps/tldw-frontend/test-results/character-chat-phase6-Char-aa839-covery-surfaces-fit-desktop-journeys/trace.zip
  SHA256 2d604572f39f557435582b040e6611527675c74eb5f41b8ed5059b2f62f02e32
apps/tldw-frontend/test-results/character-chat-phase6-Char-d0ec5-ecovery-surfaces-fit-tablet-journeys/trace.zip
  SHA256 ccde52353cc27f5be42d420ce3a96208dd2b9674f53de5638afb138709c9c065
apps/tldw-frontend/test-results/character-chat-phase6-Char-aa36c-ecovery-surfaces-fit-mobile-journeys/trace.zip
  SHA256 b814e7fb829627296fc8aeeb13bc406090e4fb5288831094e640beb0665bc632
```

The passing traces are local worktree artifacts, while their capture command,
result, paths, hashes, and the prior RED/GREEN output are reviewable in the
committed SDD artifact above.  That artifact is intentionally force-staged
because the SDD directory ignores generated-by-default material.

### Broad-gate result and origin

The broad typecheck was rerun solely to make this report's prior concern
reviewable.  It exited 2 with exactly eight pre-existing errors, all outside
this package's changed paths:

```text
../packages/ui/src/components/Option/Documentation/DocumentationPage.tsx
  TS2322 at lines 48 and 55; TS2558 at lines 50 and 57
scripts/__tests__/skills-certification-profile.test.ts
  TS2741 at lines 277 and 362
scripts/__tests__/skills-certification-runner.test.ts
  TS2353 at lines 217 and 237
```

The exact command and compiler-output grouping are retained in the same SDD
artifact.  The check was not reinterpreted as green; it remains a broader
owner concern.

Two broad UI checks remain pre-existing/unrelated concerns and were not changed:

- `Playground.cockpit-shell.test.tsx` fails at suite import because its existing `@/services/tldw-server` mock omits `LEGACY_SERVICE_PROMPT_DEFAULTS`; the failing module is `src/services/title.ts`.
- `bun run typecheck` fails in `DocumentationPage.tsx` plus the concurrently dirty skills-certification tests. Neither path is touched by Task 2E.

## Changed files and commit

- `apps/tldw-frontend/e2e/workflows/journeys/character-chat-phase6.spec.ts`
- `apps/tldw-frontend/e2e/utils/character-chat-phase6-surface.ts`
- `apps/tldw-frontend/e2e/utils/__tests__/character-chat-phase6-surface.test.ts`
- `.superpowers/sdd/2026-08-30-software-supply-chain-release/task-2e-phase6-fix-round-1-evidence.md`
- This report
- Initial Task 2E commit: `c645e43ad4`, `test(frontend): align Character Chat Phase 6 journey (TASK-13013.7.1)`.
- Fix-round commit: the commit containing this report,
  `test(frontend): cover Phase 6 layout recovery (TASK-13013.7.1)`.

## Self-review

- Confirmed the direct URL contract is current and functioning; this does not revive a legacy alias.
- Preserved all pre-existing dirty work, including both Backlog task notes and Task 2 fix-round files.
- Did not touch unrelated repositories or workstreams, Playground product code, the deterministic mock, or provider configuration.
- The minimal controller extraction remains E2E-test-owned and only centralizes
  the supported visible-layout transition already exercised by the journey.
- Did not increase timeouts, add sleeps, force clicks, skip viewports, weaken overflow/readiness checks, or mock Character Chat APIs.

## Concerns

The untouched broad typecheck and cockpit-suite failures should be resolved by their owners before treating those broader gates as clean. They do not affect the controlled live Phase 6 regression evidence, which is green across all required viewports.
