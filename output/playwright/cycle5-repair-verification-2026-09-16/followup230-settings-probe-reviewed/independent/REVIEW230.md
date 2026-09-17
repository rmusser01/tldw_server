# UAT230 / TASK13260.171 — independent review

**CLEAR for the four frozen source/test files.** No remaining production finding. Independent mounted/adjacent tests pass **87/87 across six files, zero skips, 18.68s**. Native Settings acceptance remains separate and parent-owned.

Author manifest: `9a41995dec568803117d3604acf8cbf3a75c287ac42af62dfd1b05ccd9be3671`. All four working files and snapshots match before and after review. The source baseline was captured at `e1ccad4be7cf5b5c1b4c1e3b7405741a3ad19f0d`; the final manifest names `5cfac7bbb54da83fb10c9d7eeb3f4f298d3734c2`. An independent read-only diff confirms the four owned files are identical between these commits.

## Source and causal assessment

The production patch exports the existing pure `isQuickstartWebUiSameOriginServerUrl` predicate without changing its body, imports it into Settings, and adds it to the existing Billing-probe precondition. The predicate checks the actual browser surface, deployment mode, normalized origin and root pathname; trailing slashes are handled. It does not choose a different server or consult mutable singleton configuration. Next's current quickstart rewrites cover `/health` and `/api/...`, while `/openapi.json` remains unsupported there. The guard therefore applies the client's existing transport policy to this second caller.

The effect still clears Billing availability and retains multi-user/login/explicit-server checks, a five-second AbortController timeout, cancellation of obsolete results, and all-four-advertised-GET gating. Direct backend and advanced-mode discovery are preserved. No proxy, backend route, authentication, configuration, capability architecture, or global error suppression was introduced.

The retained expanded RED has **two failures/eight controls**, with 23 other tests excluded by `-t`. The failing assertions show the actual mounted component issued `http://localhost:3000/openapi.json` for quickstart same-origin targets with and without a slash. Connection testing still succeeded. Final verification runs every selected suite without exclusions. New tests use the real exported predicate via partial module passthrough, not a Boolean predicate mock. The Form fixture forwards submit into the actual Settings save handler for the server-change case; the other fixture adjustment is export passthrough only.

Permanent controls cover quickstart root suppression, direct-backend quickstart/advanced advertised Billing, same-origin advanced discovery, direct-backend404, five-second abort, logout, unmount, and old-target responses arriving after a normal server save. Existing actual-form/auth/timeout/tab/client connection controls also pass. These are mounted component tests with controlled fetch/client collaborators; they are not native browser or live backend evidence.

## Verification correction and final results

The original author's four ESLint warnings were all **“File ignored because outside of base path.”** The initial invocation did not lint the owned files. This review retained those outputs and informed the author; it does not count the ignored run as successful lint coverage.

The corrected independent lint imports the unchanged frontend configuration, sets repository `cwd` so all shared UI paths fall within its base, and explicitly sets `settings.next.rootDir` to the actual frontend. No rule is disabled or weakened. It lints the original and current source text at the same real logical filenames. Result: **0 errors /565 warnings in both versions**, exact normalized file/rule/severity/message equality, **0 added or removed**. `eslint-scoped-*.json` contains the final results. The first corrected run without the explicit Next root is separately retained; its missing-pages contextual warning is resolved in the final run.

| Independent check | Result |
| --- | --- |
| Six-file Vitest run | 87 passed, zero skipped |
| Actual scoped ESLint | 0 errors; 565 unchanged baseline warnings |
| Full TypeScript baseline/current, in-memory source substitution | 90/90 diagnostics, exact semantic equality, none added/removed |
| Bandit attempt from project venv | Four TS/TSX parse failures; no meaningful security coverage claim |
| Owned diff whitespace check | PASS |
| Four current/snapshot hashes | Stable before/after |

The first reviewer hash-preflight command used the frontend cwd as the repository root and failed before test execution. `reviewer-harness-note.txt` preserves that harness error; the directory was corrected before the successful run. No product source or test was changed.

## Limits and record

Approval is for the bounded implementation and automated checks. Existing TypeScript/lint debt is explicitly retained, and unsupported Bandit parsing is not called a security pass. The original Settings404 must still be repeated natively by the parent. This review performed no browser/runtime/service action, credential access, task/tracker edit, source/test edit, or git mutation. `source-before.json`, `source-after.json`, commands, causal receipts and `reviewer-manifest.json` bind the result.
