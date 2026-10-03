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

## When a defect is fixed

The reproduction starts failing with "`<id>` no longer reproduces". Replace
`expectKnownDefect(...)` with the plain assertion in the fix PR. The test then
guards the fix as an ordinary regression test.

## Current reproductions

| Id | Issue | Spec |
|---|---|---|
| NL-01 notes list capped at 100 | #3103 | `notes-p0.spec.ts` |
| NS-01 edits lost on in-app navigation | #3102 | `notes-p0.spec.ts` |
| NS-N1 "Reload notes" after a save conflict overwrites the other tab | #3102 | `notes-p0.spec.ts` |
| NE-04 Print / Save as PDF always fails with a pop-up error | #3117 | `notes-p0.spec.ts` |
| CS-02 chat history search ignores message content | #3108 | `chat-p0.spec.ts` |

Chat reproductions here seed saved chats through the API
(`createCharacter`, `createChatWithMessages`) and never send from the
composer. On current dev the first send in a fresh browser is not reliably
delivered (#3106, #3111, #3135), so defects that need a browser send cannot
be reproduced deterministically here yet. CS-01, CS-03 and others live in
the Playground integration harness instead. CS-04 (reply lost on reload
mid-stream, #3104) is left out of this harness: it needs a reply in flight,
which only a send can start.
