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
CI runs it nightly (`.github/workflows/ux-regression-nightly.yml`).

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
| NL-01 notes list capped at 100 | #3103 | `notes-p0.spec.ts` |
| NS-01 edits lost on in-app navigation | #3102 | `notes-p0.spec.ts` |

Chat reproductions (CS-01, CS-03 and others) live in the Playground
integration harness instead: on current dev the first send in a fresh
browser is not reliably delivered (#3106, #3111, #3135), so a browser-level
chat reproduction cannot be deterministic yet.
