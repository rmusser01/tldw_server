# UX regression reproductions

Red-first reproductions of verified defects from the 2026-10-02 /notes and
/chat UX review (`Docs/Design/2026-10-02-notes-chat-ux-review.md`). Tracking:
#3101; this harness: #3125.

## Running

```bash
# From apps/tldw-frontend
bun run e2e:ux-regression
```

The live-tier runner (`scripts/live-tier-uat/run.mjs`) starts an isolated
backend with fresh databases, the mock OpenAI-compatible LLM and the web UI on
reserved ports, runs the `ux-regression` Playwright project with zero retries,
fails on skipped tests, and writes `test-results/live-tier-uat/<run-id>/`.

Before it starts any service, the runner builds the browser extension
(`bun run build:chrome:prod` in `apps/extension`, about two minutes) for the
side-panel specs. The build prints to the runner's output; a failed build
stops the run and leaves its log in `extension-build.log`. (Playwright clears
`test-results/` when it starts, so logs written before the tests, this one
included, do not survive a run that gets that far.) To iterate locally without
rebuilding, point `TLDW_UXR_EXTENSION_DIR` at an existing production build,
for example `apps/extension/.output/chrome-mv3`. A missing or dev-server build
fails the side-panel specs; they never skip.

The side-panel specs (`sidepanel-p0.spec.ts`) load that build into Chromium
through `e2e/utils/extension-sidepanel.ts`: a fresh profile per test, the
full Chromium build in new headless mode (the default headless shell cannot
load extensions), the runner's backend and API key in
`chrome.storage.local`, and `sidepanel.html#/chat` opened in a 420 px wide
page. A real side panel keeps its chat state under a global storage key, and
the helper makes the page behave the same way.

In CI:

- **Pull requests.** The `ux-regression` job ("E2E UX Regression
  (notes/chat)") in `.github/workflows/frontend-e2e-tiers.yml` runs the suite
  when a PR to `dev` or `main` changes the notes/chat UI, the extension side
  panel or the extension itself, the shared UI layers under them
  (`components/Common`, `hooks`, `services`, `db`, `store`), the notes/chat
  endpoints, `DB_Management`, or this harness. Other PRs skip
  the run after a quick diff. The path list is in the job's
  `Detect notes/chat UX changes` step. The job is advisory: it is not a
  required check. On failure it uploads `test-results/` as
  `e2e-ux-regression-results`.
- **Manual.** Dispatch "Frontend E2E Tiers" with `tier=all-tiers` and the
  branch as the ref. That runs every tier, this suite included, regardless
  of paths.
- **Nightly.** `.github/workflows/ux-regression-nightly.yml` runs it daily,
  but only from the default branch (`main`). GitHub ignores scheduled
  workflows on other branches, so the nightly starts once this harness is
  released to `main`, and it tests `main`'s code.

These specs never run in the default `chromium` project: they need a real
backend, not the smoke route mocks.

## Writing a reproduction

1. Do setup through the API (`e2e/utils/seed-api.ts`) and the page objects.
   Call `warmBackendOnce(api)` first: a cold backend freezes for ~30 s on its
   first `/openapi.json` build (#3135), which would make the first spec flaky.
2. Keep preconditions as ordinary assertions, so a broken environment fails
   the test instead of looking like the defect.
3. Wrap only the correct-behaviour assertion in `expectKnownDefect`
   (`e2e/utils/known-defect.ts`) with the review id and GitHub issue. While
   the defect reproduces, the test passes and records the evidence.
4. Prefer stable observables (API state, request payloads, counts) over
   copy or layout.
5. Control timing that `next dev` would otherwise decide, for example by
   compiling a route before a scenario that depends on fast navigation.
6. For the extension side panel, use `test` and `SidePanelChat` from
   `e2e/utils/extension-sidepanel.ts`. Seed server chats with
   `createChatWithMessages`, which links each message to the previous one as
   the chat UI does; the side panel cannot load a chat of unlinked messages.
   Read the outcome from the panel's saved tabs (`readTabs`) and from the
   server (`e2e/utils/chat-api.ts`).

## When a defect is fixed

The reproduction starts failing with "`<id>` no longer reproduces". Replace
`expectKnownDefect(...)` with the plain assertion in the fix PR. The test then
guards the fix as an ordinary regression test.

## Ratchets: accessibility and request budget

Some problems are too many to fix in one PR but must not grow. For those,
`a11y.spec.ts` and `request-budget.spec.ts` compare what they observe with a
committed baseline in `baselines/`. Each baseline entry names the review id
and GitHub issue that tracks its fix, plus a note.

| Spec | Measures | Baseline |
|---|---|---|
| `a11y.spec.ts` | Serious and critical axe violations (WCAG 2.0-2.2 A/AA plus best practices, including `color-contrast` and `target-size`), node count per rule, in five states: /notes empty, populated (25 notes) and with a note in the editor; /chat on a fresh load and with the model picker open. Light theme, 1280x720. | `baselines/a11y.json` |
| `request-budget.spec.ts` | API requests started in the first `windowMs` after loading warm /notes and /chat: the total, duplicate GETs (same path and query within `duplicateWindowMs`) and polling (an endpoint hit `pollingMinHits`+ times after `idleFromMs`). | `baselines/request-budget.json` |

The comparison (`e2e/utils/ratchet.ts`) fails in both directions:

- **A new problem fails.** An axe rule, duplicate GET or polling endpoint that
  is not in the baseline, or a count above its baseline plus `tolerance`, is
  a regression. Fix it. Add it to the baseline only if it is a known review
  finding, with its review id and issue.
- **A fixed problem fails too**, with "no longer present, remove it from the
  baseline" (or "improved … Lower its count" when only some nodes or requests
  went away). Delete the entry, or lower its count, in the fix PR. The ratchet
  then guards the fix, as `expectKnownDefect` does for reproductions.

### Updating a baseline when you fix an issue

1. Run `bun run e2e:ux-regression -- --grep "ratchet"` (both specs) or
   `--grep "Accessibility ratchet"` / `--grep "Request budget ratchet"`.
2. The failure lists every difference. Each test also attaches what it
   observed: `a11y-<state>.json` (every node, with its markup) or
   `request-budget-<page>.json` (counts and a request timeline). The runner's
   `test-results/live-tier-uat/<run-id>/playwright-results.json` embeds them.
3. Remove or lower only the entries your change fixed. Never raise a count
   or add an entry to make a regression pass.
4. `tolerance` absorbs measured run-to-run noise only. Keep it as small as
   the noise allows and say why in the entry's note. A tolerance that reaches
   0 lets a race-dependent offender be absent (the /chat `config/providers`
   duplicate appears in about 1 of 4 loads); such an entry cannot report its
   own fix, so delete it by hand when you fix it.

Determinism:

- Every test warms the backend (`warmBackendOnce`) and then loads /notes and
  /chat once in its own page (`primeUxRoutes` in `ux-routes.ts`) before it
  measures. That keeps `next dev` compile time out of the measurement and
  puts the browser's cached responses in the same state on every run; a
  context's first visit requests more than later visits.
- The specs never send chat messages (#3106).
- The library size is fixed: the empty-library scan asserts 0 notes and the
  populated scan exactly 25. Both rely on the fresh backend the runner
  starts, on its single worker, and on `a11y.spec.ts` sorting first.
- Request counts come from `next dev`, where React StrictMode runs effects
  twice; the baseline notes which duplicates look dev-only. The request
  budget requires the page to be ready before `idleFromMs`, so a slow
  environment fails clearly instead of looking like polling.

## Current reproductions

| Id | Issue | Spec |
|---|---|---|
| NE-04 Print / Save as PDF always fails with a pop-up error | #3117 | `notes-p0.spec.ts` |

Fixed, and now guarded by a plain assertion in the same test:

| Id | Issue | Spec |
|---|---|---|
| NL-01 notes list capped at 100 | #3103 | `notes-p0.spec.ts` |
| NS-01 edits lost on in-app navigation | #3102 | `notes-p0.spec.ts` |
| NS-N1 reloading after a save conflict overwrote the other tab | #3102 | `notes-p0.spec.ts` |
| NE-01 WYSIWYG typed text backwards | #3102 | `notes-wysiwyg.spec.ts` |
| CS-02 chat history search ignored message content | #3108 | `chat-p0.spec.ts` |
| XS-01 opening a past chat from side-panel search overwrote the current tab | #3105 | `sidepanel-p0.spec.ts` |
| XS-07 side-panel "Delete" only closed the tab | #3105 | `sidepanel-p0.spec.ts` |
| XS-07 side-panel "Rename" only relabelled the tab | #3105 | `sidepanel-p0.spec.ts` |
| XP-08 a reopened side panel never refreshed its chat from the server | #3105 | `sidepanel-p0.spec.ts` |

Chat reproductions here seed saved chats through the API
(`createCharacter`, `createChatWithMessages`) and never send from the
composer. On current dev the first send in a fresh browser is not reliably
delivered (#3106, #3111, #3135), so defects that need a browser send cannot
be reproduced deterministically here yet. CS-01, CS-03 and others live in
the Playground integration harness instead. CS-04 (reply lost on reload
mid-stream, #3104) is left out of this harness: it needs a reply in flight,
which only a send can start.

The side-panel specs never send either. XP-08 adds the other client's turn
through the API and checks what the reopened panel shows. The silent fork the
review saw when the stale panel then sent a message needs a send from the
panel, so it is left out until a browser send is reliable.
