# UAT159 independent review

Reviewer: source013_diagnosis. Task: TASK-12918 (reopened). Review date: 2026-09-16.

## Outcome

No blocking findings in the approved scope. This was a read-only review of the frozen implementation, retained test evidence, and relevant existing auth and request helpers. The reviewer made no production or test edits and ran no tests, browser actions, runtime starts, or provider calls.

All three production files and three test files matched the frozen manifest during review. The implementation preserves the existing shell authentication decision, including deferred runtime bootstrap, canonical exact-origin quickstart cookie resolution, and hosted multi-user cookie validation. It adds no separate raw-key gate or credential-copy logic.

The disabled hook sends no profile GET and clears setup, resume, and loading state. Effect cleanup suppresses late success, error, and finalization state updates from the previous invocation. The existing `coalesce: false` option is client-local and is not forwarded to the server; using it for this hook prevents a newly enabled invocation from joining the old unscoped GET promise. The actual App → Gate → Hook → apiSend → mocked request-core tests meaningfully cover that boundary after an observed disable/re-enable transition.

The effect depends only on `allowCompletedSetup` and `enabled`. State updates and stable authenticated refreshes do not create a request loop. Existing source-destination, bypass-route, error, dismissal, and resume behavior remains within the stated scope.

## Explicit limitations

- **Development Strict Mode:** `apps/tldw-frontend/next.config.mjs` enables `reactStrictMode`. Development effect replay can issue two bounded profile GETs because this hook now opts out of coalescing. The retained one-request assertions cover ordinary test mounts and stable rerenders, not a universal native development request count. This is a bounded replay, not a request loop, and it does not re-enable blank-key Home probes.
- **Continuous authenticated owner changes:** the hook receives a boolean eligibility signal, not an owner identity. An authenticated owner-to-owner change without an intervening disabled state is not fenced by this repair. This is an existing boundary outside the approved scope; the review does not claim it is fixed. The covered transition is an observed disable/logout followed by authenticated re-entry.
- **Other caller:** the composer nudge retains the hook's default `enabled: true` behavior. This repair does not claim universal Chat or extension pre-auth request suppression. Concurrent enabled consumers each make their own read, as documented by the author.
- Existing compiler diagnostics remain: the author reported 90 baseline and 90 current diagnostics, with none added. Bandit's three TypeScript parse errors are not TypeScript security assurance.

## Verification provenance

The reviewer inspected the frozen patch, source, existing auth/request helpers, test semantics, retained logs, and six source/test hashes. The author retained 110 passing tests across five suites. After this review, root independently reported **110 tests / five suites passing** and a fresh native Home check with **zero persona requests before adding a key**; authenticated Media/notification activity followed. These later results are root-reported, not executions by this reviewer.

## Frozen hashes

Manifest HEAD: `e717525400746ab790a38d4a2be2149124d00935`.

| Artifact | SHA-256 |
| --- | --- |
| `apps/tldw-frontend/pages/_app.tsx` | `ae5a722f00b16829495fc5ea0e5be2472f217b55d0a4bedd3d6965b41c2c33db` |
| `apps/packages/ui/src/components/PersonaGarden/FirstRunGate.tsx` | `4302a207c689b8d7324c7327d12fd2795572e7751a656265f2ef89341c89e13f` |
| `apps/packages/ui/src/hooks/useFirstRunCheck.ts` | `21e1aae24a32def73a5ea649c8e8c90e05ef43c6af817c4fb7d9e07cd7af4bef` |
| `apps/tldw-frontend/__tests__/app/app-layout.test.tsx` | `67dec97c527524c68d5573049b8ac86669b2441a7df8457bd046e7776bf310de` |
| `apps/packages/ui/src/hooks/__tests__/useFirstRunCheck.test.tsx` | `e3c029e197375c043e7ac8ed71a52939ab00d9a44dce38d10b5c8e9afe2c185a` |
| `apps/packages/ui/src/components/PersonaGarden/__tests__/FirstRunGate.test.tsx` | `bd4f4b512c3a567899608b8ead4ab6b907e12bea9b6ee68cb261efbcbe4fa150` |
| `.tmp/uat159-first-run-auth-20260916/owned-manifest.json` | `ea324859e8e271efd49ef576a5bfad7c7a51f8cd2ed40b1eb98a833ff5337bdb` |
| `.tmp/uat159-first-run-auth-20260916/IMPLEMENTATION.md` | `82579765e7d04659225c78e13749df220b2acf6a23f6f96c9c722fc89821dce8` |

The source/test copies under `review-snapshot/` and the manifest retain the reviewed version independently of later root commits. This review artifact is the only file written for this handoff.
