# Task 2E Phase 6 — fix round 1 retained verification evidence

This committed artifact is the reviewable transcript for the Task 2E fix
round.  Commands were run from `apps/tldw-frontend` unless stated otherwise.
The test key below is the checked-in local E2E fixture value, not a production
credential.

## Behavioral boundary RED — before the controller correction

**Origin.** Before wiring the corrected transition logic into
`e2e/workflows/journeys/character-chat-phase6.spec.ts`, the E2E-owned
`revealCharacterChatSessions` controller faithfully used the stale journey's
unconditional bidirectional layout toggle.  The harness models the visible
cockpit/focus state and supported surface controls; it is not a source-text
contract test.

```bash
bunx vitest run e2e/utils/__tests__/character-chat-phase6-surface.test.ts
```

```text
FAIL  e2e/utils/__tests__/character-chat-phase6-surface.test.ts (2 tests | 2 failed)
  × restores a collapsed desktop context rail without entering focus
    AssertionError: expected 'focus' to be 'cockpit'
  × uses Exit focus before selecting the compact Context rail
    AssertionError: expected [ 'toggle-layout-mode', 'select-compact-context-tab' ]
    to deeply equal [ 'exit-focus-mode', 'select-compact-context-tab' ]

Test Files  1 failed (1)
     Tests  2 failed (2)
```

The first failure is the prior desktop bug: clicking the bidirectional control
from cockpit enters focus and makes the rails unavailable.  The second proves
that a focus-state mobile surface must use the explicit **Exit focus** path,
not another toggle.  This RED was captured before the controller was changed
to inspect focus state, exit it first, and then restore/select the supported
context surface.

## Focused GREEN after wiring the correction

```bash
bunx vitest run \
  e2e/utils/__tests__/character-chat-phase6-surface.test.ts \
  ../packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx \
  ../packages/ui/src/utils/__tests__/character-chat-mode-intent.test.ts
```

```text
✓ e2e/utils/__tests__/character-chat-phase6-surface.test.ts (2 tests)
✓ ../packages/ui/src/utils/__tests__/character-chat-mode-intent.test.ts (6 tests)
✓ ../packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx (5 tests)

Test Files  3 passed (3)
     Tests  13 passed (13)
Duration  815ms
```

The new two-test controller regression proves both state transitions; the
existing six direct-route and five mobile-toolbar tests retain the adjacent
contracts used by the real journey.

## Controlled live three-viewport GREEN

The shell-owned graph used Redis `62457`, deterministic Mock OpenAI `18091`,
the API `62458` with the tracked
`tldw_Server_API/Config_Files/e2e-critical-config.txt`, and the WebUI `62459`.
The started listener PIDs were Redis `79110`, mock `79190`, API `79249`, and
WebUI `79307`.  The API health request, authenticated with the local fixture
key, returned `status: ok` before the browser suite ran.

```bash
TLDW_WEB_AUTOSTART=false \
TLDW_WEB_URL=http://127.0.0.1:62459 \
TLDW_SERVER_URL=http://127.0.0.1:62458 \
TLDW_API_KEY=test-api-key-for-e2e-testing-12345 \
bunx playwright test \
  e2e/workflows/journeys/character-chat-phase6.spec.ts \
  --project=journeys --workers=1 --reporter=line --trace=on
```

```text
Running 3 tests using 1 worker
[1/3] ... character mode setup and recovery surfaces fit desktop
[2/3] ... character mode setup and recovery surfaces fit tablet
[3/3] ... character mode setup and recovery surfaces fit mobile
  3 passed (21.6s)
```

Passing trace artifacts retained in the worktree at capture time:

```text
apps/tldw-frontend/test-results/character-chat-phase6-Char-aa839-covery-surfaces-fit-desktop-journeys/trace.zip
  2,728,735 bytes; SHA256 2d604572f39f557435582b040e6611527675c74eb5f41b8ed5059b2f62f02e32
apps/tldw-frontend/test-results/character-chat-phase6-Char-d0ec5-ecovery-surfaces-fit-tablet-journeys/trace.zip
  2,915,921 bytes; SHA256 ccde52353cc27f5be42d420ce3a96208dd2b9674f53de5638afb138709c9c065
apps/tldw-frontend/test-results/character-chat-phase6-Char-aa36c-ecovery-surfaces-fit-mobile-journeys/trace.zip
  2,546,623 bytes; SHA256 b814e7fb829627296fc8aeeb13bc406090e4fb5288831094e640beb0665bc632
```

## Broad typecheck and scoped hygiene

```bash
bun run typecheck
```

```text
exit 2
DocumentationPage.tsx: TS2322 (twice), TS2558 (twice)
scripts/__tests__/skills-certification-profile.test.ts: TS2741 (twice)
scripts/__tests__/skills-certification-runner.test.ts: TS2353 (twice)
```

Those eight errors are outside this package's changed paths and are preserved
as explicit broader-gate concerns rather than treated as green.  The exact
compiler diagnostics are reproduced in the readiness report.  No Python path
changed in this fix round, so Bandit is not applicable.  `git diff --check`
was run after the final edits; its zero-output, exit-zero result is recorded in
the readiness report.
