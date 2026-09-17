# Fresh SQLite single-user five-card failure diagnosis

Source: frozen `8f8774e6c868b304a96d95ab82e28389c129a78b`, run `fresh-final-20260917`, cell `sqlite-single`. Read-only diagnosis; parent associates UAT234. No repair, tests, inference, service/browser/DB action or private runtime-log/config/credential read was performed.

## Finding

**The Next rewrite proxy cut off the request at its 30-second default before the backend completed successfully.** This is a proxy/client deadline mismatch, supported by source, native request timing and the parent-sanitized log excerpt. It is not evidence that the Flashcards generation adapter returned a backend500 or that the configured provider was unavailable.

| Evidence | Observation |
|---|---|
| Browser POST | `2026-09-17T13:09:48.169Z`, `/api/v1/flashcards/generate`, the five protocol facts, `num_cards:5`, basic/mixed, explicit llama.cpp and existing Gemma model |
| Browser response | `2026-09-17T13:10:18.159Z`, **500**, exact text `Internal Server Error`; elapsed **29.990s** |
| Frontend sanitized log | line27: `Failed to proxy http://127.0.0.1:18600/api/v1/flashcards/generate Error: socket hang up`; lines29/33: `ECONNRESET` |
| Backend sanitized access log | line6963: same POST **200 in30558ms**, at local06:10:18.710 / UTC13:10:18.710, request `7ca1efbb-3f5c-42a4-92fe-f48e1bf6abe9` |

The backend success log follows the browser500 by **551ms**. Browser event time is an observer timestamp, not an exact server-arrival clock; the29.990s window is consistent with the installed30s proxy timer. Initial plain500 text alone would not distinguish all possible unhandled backend/middleware errors, but the matching proxy error and later backend200 resolve this case.

## Frozen source path

Paths below are relative to this cell's frozen root.

1. `apps/packages/ui/src/services/flashcards.ts:25,1000–1012` sets `FLASHCARD_GENERATION_TIMEOUT_MS=180000` and supplies it to this POST. `background-proxy.ts:1427–1443` forwards it in the direct WebUI request. `services/tldw/request-core.ts:100–105,489–506` honors a positive explicit override and arms the browser AbortController with that budget. Thus the declared generation budget is180s, not30s.
2. `apps/tldw-frontend/next.config.mjs:147–168` rewrites quickstart `/api/*` to the internal API origin. It sets no `experimental.proxyTimeout`; the existing conditional experimental setting at179–182 only disables the UAT Turbopack dev filesystem cache.
3. Installed copied **Next16.1.4** `dist/server/lib/router-server.js:357–359` invokes `proxyRequest(..., config.experimental.proxyTimeout)` for the external rewrite. `dist/server/config-shared.js:165` defaults that option to undefined. `dist/server/lib/router-utils/proxy-request.js:26–33` converts undefined to **30000ms**. Its proxy error handler at73–82 logs `Failed to proxy`, sets status500 and emits the exact text observed by the browser. This is the direct source match for the failure surface.
4. `tldw_Server_API/app/api/v1/endpoints/flashcards.py:2609–2677` awaits the real generation adapter, normalizes the result, then awaits claim verification before constructing its successful JSON response. This is one nonstreaming HTTP request, so the proxy sees no completed response while both stages execute.
5. `core/Workflows/adapters/content/generation.py:286–295` awaits the configured model call, requesting up to4000 output tokens. Lines333–336 return normalized results or a structured adapter error. The endpoint maps that error to HTTP400, no usable cards to422, failed claim verification to a structured422, and caught unexpected errors to JSON500 detail `Failed to generate flashcards` (2678–2682). Its caught exception tuple is bounded (254–264); source alone does not promise every possible error is JSON. Here the observed backend status is200.
6. `core/Claims_Extraction/artifact_verification.py:310–314,432–438` resolves the verifier provider/model and awaits verification against the source. We cannot split the30.558s backend duration between generation and verification from the supplied excerpt, and should not attribute it to a particular stage or model defect.

## Checks and repair boundary for later review

The source/log checks needed to distinguish proxy cutoff from backend generation failure are now satisfied. The frozen matrix failure should remain recorded; ordinary Chat/Knowledge QA success does not demonstrate that a nonstreaming request exceeding30s survives this rewrite boundary.

For a future separately reviewed repair, the installed Next schema supports the existing `experimental.proxyTimeout` numeric option (`dist/server/config-schema.js:246`). Review a finite proxy budget consistent with supported long-running client requests, preserving the existing experimental settings and client cancellation/timeouts. Do not assume zero disables this timeout: the installed runtime expression uses `proxyTimeout || 30000`. Do not change model output, verification policy or backend error handling to conceal the proxy deadline mismatch.

Appropriate later causal verification is a controlled slow upstream through the actual installed Next rewrite path, showing that a response after30s but within the declared request budget reaches the client; retain fast success, real upstream failure and client-abort controls. Then rerun the exact native five-fact/five-card request only on explicitly reviewed source. None of these follow-up actions was run here.

## Limits

The backend access200 proves successful response construction/status for this request, not that the browser received its lost body. No response payload is available here to certify actual returned card count, card quality, save/persistence, or overall workflow acceptance. No claim is made about all other proxy routes or deployments. The original native failure remains a failure.

## Hash-bound inputs

| Input | SHA256 |
|---|---|
| `native/sqlite-single/biology-generate-failure-evidence.txt` | `35ac00a09c559d6b918b71adc8c4cb8092e3630a1088ff192bd5c873d68dec28` |
| `native/sqlite-single/flashcard-failure-log-excerpt.json` | `3345b5e41ba66af52a1c645db9299d620b6763efb3772daa3f84ecfd0ddc636d` |
| `copy-preparation/sqlite-single-archive-manifest.json` | `26255fe54e27f7e92d849bbf810a7224c655602e3e2f9514eb6cbcce3c96bba1` |
| `sources/sqlite-single/apps/tldw-frontend/next.config.mjs` | `7a78b19818d04e6d23e12ededa457217996eb1abf2154e929ce9d47d60d7ead1` |
| `sources/sqlite-single/apps/packages/ui/src/services/flashcards.ts` | `923f290dce3b692d9d1d5160f43dc0fdd03fb4678cb1c34fffc221f762068aaa` |
| `sources/sqlite-single/apps/packages/ui/src/services/background-proxy.ts` | `70f1483af5b48f6573340a0e6728a2b363757d61ad4748f32249208ab1bae638` |
| `sources/sqlite-single/apps/packages/ui/src/services/tldw/request-core.ts` | `3567b4d2defdab1449031bf51ab5db69548cdb1ec123f856091ef403075487df` |
| `sources/sqlite-single/tldw_Server_API/app/api/v1/endpoints/flashcards.py` | `cc3f242d2c115a324db2f2ed180256c32408c1cb53a9515e7b795bea4183b107` |
| `sources/sqlite-single/tldw_Server_API/app/core/Workflows/adapters/content/generation.py` | `afc770de0650a2cc0f8dddc9ef825dac243f0584c61542e1af922cf34e307197` |
| `sources/sqlite-single/tldw_Server_API/app/core/Claims_Extraction/artifact_verification.py` | `9f58bf190d6b359a04c6a252e710a664375c584199e53b08cfedb407ae42131d` |
| `sources/sqlite-single/apps/node_modules/.bun/next@16.1.4+d9efe0b44b7e6f19/node_modules/next/package.json` | `caa51e521d4ab3d7e5d0b647af543561a20b6df049180c2191a99aa84343d07b` |
| `sources/sqlite-single/apps/node_modules/.bun/next@16.1.4+d9efe0b44b7e6f19/node_modules/next/dist/server/lib/router-utils/proxy-request.js` | `26c521dc09d71e5a798a384b3a8f5e85a535dcc1b5943cbc7acb0dbc4c28ff57` |
| `sources/sqlite-single/apps/node_modules/.bun/next@16.1.4+d9efe0b44b7e6f19/node_modules/next/dist/server/lib/router-server.js` | `74f584998fb395fe92bbcd6a1096575491f3a537fab2a3bdd141df4a413cfa05` |
| `sources/sqlite-single/apps/node_modules/.bun/next@16.1.4+d9efe0b44b7e6f19/node_modules/next/dist/server/config-shared.js` | `8549edd12d194d77f0f7dcc5fb3719ccf94d4ede6e473c38a885b2928e51f4ce` |
| `sources/sqlite-single/apps/node_modules/.bun/next@16.1.4+d9efe0b44b7e6f19/node_modules/next/dist/server/config-schema.js` | `5705f9473cdf176c382e181fba7fb03b549aa92a0188295cae5e935e64939007` |

All seven inspected application-source hashes match their original archived entries. Installed Next files are separately hash-bound copied dependencies, not tracked archive entries. Only this diagnosis report was written.
