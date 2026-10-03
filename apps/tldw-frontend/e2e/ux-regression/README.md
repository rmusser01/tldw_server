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
  when a PR to `dev` or `main` changes the notes/chat UI, the shared UI
  layers under it (`components/Common`, `hooks`, `services`, `db`, `store`),
  the notes/chat endpoints, `DB_Management`, or this harness. Other PRs skip
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

## Current reproductions

| Id | Issue | Spec |
|---|---|---|
| NL-01 notes list capped at 100 | #3103 | `notes-p0.spec.ts` |
| NS-01 edits lost on in-app navigation | #3102 | `notes-p0.spec.ts` |
| XS-01 opening a past chat from side-panel search overwrites the current tab | #3105 | `sidepanel-p0.spec.ts` |
| XS-07 side-panel "Delete" only closes the tab | #3105 | `sidepanel-p0.spec.ts` |
| XS-07 side-panel "Rename" only relabels the tab | #3105 | `sidepanel-p0.spec.ts` |
| XP-08 a reopened side panel never refreshes its chat from the server | #3105 | `sidepanel-p0.spec.ts` |

Chat reproductions (CS-01, CS-03 and others) live in the Playground
integration harness instead: on current dev the first send in a fresh
browser is not reliably delivered (#3106, #3111, #3135), so a browser-level
chat reproduction cannot be deterministic yet.

The side-panel specs never send either. XP-08 adds the other client's turn
through the API and checks what the reopened panel shows. The silent fork the
review saw when the stale panel then sent a message needs a send from the
panel, so it is left out until a browser send is reliable.
