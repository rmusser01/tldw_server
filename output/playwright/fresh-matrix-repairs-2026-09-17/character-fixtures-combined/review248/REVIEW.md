# Independent UAT248 implementation review

**CLEAR for the bounded six-file repair. Native acceptance remains pending.**

Frozen author manifest: `6dd7a5e19cdd9f5a2bf91192ad0dc3dbb417891561957f3c335686846a1247a4`. All six live/snapshot hashes matched before and after review. All60 author evidence entries match their manifest; the three baseline production snapshots match recorded commit `f1cefecb22ae2447a53e2ffdb5cda67866ccf4a4`.

## Source and authority assessment

The change repairs the previously demonstrated late Character persistence dispatch after actual logout without introducing a new authority model:

- Character mode takes its execution signal and request scope from the existing captured lease. The original caller signal remains the controller ownership/Stop identity. Create, greeting/user writes, stream and completion/fallback writes carry the same captured options.
- Cancellation checks after asynchronous prerequisites and acknowledgements prevent later dispatch and publication. Late persistence success/failure cannot publish old IDs or trigger the fallback; queued streaming updates are suppressed on authority invalidation. The existing success/error save boundaries receive captured scope and signals.
- Deliberate caller cancellation remains distinct from authority loss. Same-owner non-abort partial recovery remains supported; caller Stop does not remotely persist partial text. Existing routing, acknowledged IDs, one-user-write, greeting, emote and Retry behavior remains exercised.
- The stream domain adapter forwards captured config/expected-user header and signal outside the inference body, preserving model payload and idle budget. The existing route policy adds only POST `/api/v1/chats/<single-segment>/complete-v2`; wrong method, sibling/suffix path, missing segment, encoded slash and traversal remain rejected by existing canonical-path checks.

Actual-lease tests invoke WebUI logout rather than manually aborting a fake scope controller. Existing adjacent lease controls cover server/API-key/account changes, same-user token rotation, A→B→A non-revival and invalidation after caller cancellation. Request-scope tests separately prove captured server/user metadata is passed through the domain adapter. No actionable finding remains in the reviewed six-file change.

## Independent verification

| Check | Result |
|---|---|
| Focused Character/domain/policy tests | **139 passed**,0 skipped,3 files,2.86s |
| Bounded adjacent ordinary/persona/overlay/coordinator/lease suites | **338 passed**,0 skipped,7 files,36.88s |
| Final tests against original three production files | **14 expected failures,9 passing controls**,116 filtered,2.77s |
| Scoped ESLint differential | 0 errors;274→273 warnings;0 new,1 existing `any` warning removed |
| Frontend TypeScript differential | 90 baseline/90 current diagnostics;0 added/removed |
| Bandit on all six TS/TSX files | 6 parse errors; no meaningful TypeScript security coverage |
| Source/snapshot verification | All six stable; all60 evidence entries valid |

Exact commands and independent logs are included. The seven-file adjacent command explicitly excludes `background-proxy.test.ts`, whose seven known console-spy fixture failures are separately tracked as UAT250. I inspected the author's current456/7failure and exact-baseline118/7failure receipts; no claim that the entire eight-file suite is green is made. The original UAT248 causal failures and intermediate fixture corrections are retained, not hidden.

## Limits

This establishes actual hook → existing lease → Character adapter → controlled fetch/stream behavior, with side effects observed at mocked persistence boundaries. It does not prove a native cross-account database write, its rejection by a running backend, real provider behavior, ASGI/proxy delivery, or the cause of native UAT246. Requests already dispatched under valid authority may have completed at the server before cancellation; this frontend fix cannot roll those back. Compiler/lint results are differential, not a clean-build assertion.

UAT231/232/236/243 and UAT246 diagnostic verdicts remain separate. Root owns native acceptance and integration. This review changed no production/test source, browser, native runtime, held database, Git or task records.
