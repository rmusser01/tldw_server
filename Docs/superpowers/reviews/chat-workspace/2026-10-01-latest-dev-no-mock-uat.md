# Chat Workspace Latest-Dev Acceptance

Tracking: TASK-13418; previous qualification TASK-13417 and finalization
TASK-13408 archived intact;
historical Chat Workspace TASK-13398 and its children.
Epic: https://github.com/rmusser01/tldw_server/issues/1239.
Initial acceptance baseline: dev `ec86ba4e871d844ff4a3a53c607e80eb85bd0e9b`.
Current integration baseline: dev `3caebcfc1e16596f4a352744a9f017017d766165`.

## Current Gate

The initial acceptance matrix below passed on ec86ba4. A subsequent latest-dev
integration is published in `5f52dde95936030353581ecaa1f8d7cf2efd9bfc`;
its fresh grounded, route, cancellation/recovery, workspace, preview/mobile and
owner-return requalification now pass. The follow-up keyless-owner recovery fix
is covered by 207 owning-package checks and the exact protected native GET.
[PR #3071](https://github.com/rmusser01/tldw_server/pull/3071) remains draft and
unmerged pending current gates. The requester-authored Change summary is
published verbatim and is no longer an outstanding requirement.

## Initial Acceptance Recorded

All browser acceptance below used real Chrome raw CDP, native API authentication,
SQLite, browser IndexedDB/localStorage, the actual local Gemma provider and real
embeddings. No request interception, mocked responses, injected application
state, fabricated receipts, focus emulation or plugin installation was used.
Unit-test doubles are separate from browser acceptance.

| Initial ec86 baseline check | Result | Evidence |
| --- | --- | --- |
| Grounded selected-history send | PASS: one retrieval with generation disabled, one actual Gemma dispatch, canonical input/result receipts and full source metadata | `native-grounded-keyboard-roundtrip.json` |
| Citation expansion and protected fresh-document restore | PASS: captured memo excerpt, real H1 read, native IndexedDB draft restored; no automatic retrieval/inference on reload | Same grounded evidence |
| Stop, reload, Verify and explicit Reprepare | PASS: two real dispatches, one canonical input; earlier unknown observation retained | `native-stop-latest.json` |
| Deep link, hash, Back/Forward, reload, malformed route, mobile | PASS: six checks, zero completion dispatches, rejected route retains checkpoint, no horizontal overflow | `native-route-latest-complete.json` |
| Workspace A/B/A, shared Research handoff, New Chat precedence | PASS: eleven checks, zero completion dispatches, inactive/outgoing checkpoints preserved | `native-workspace-latest.json` |
| Original data/browser preservation | PASS: original conversation projections retain ten/eight rows and exact hashes; both original targets remain open | `final-preservation/native-after-dev-refresh.json` |
| Actual served OpenAPI | PASS: unchanged production capability predicates and twelve negative controls | `final-preservation/served-openapi-gate.json` |
| Valid account A/B/A and backend A/B/A | PASS: three explicit real Gemma turns, six verified canonical rows, qualified drafts/reloads, inactive checkpoint hashes unchanged, zero transition/reload autosends; foreign cached UUID controls fail closed | `owner-uat/positive-owner-controls-1790878910253.json`, `owner-uat/owner-functional-audit.json` |
| Fresh account-return visual proof | PASS: actual Ready/loaded original-A transcript and draft, fresh capture inspected; cached Loading frame excluded | `owner-uat/fresh-visual-final.json`, corresponding PNG |
| Browse, staging, external link and mobile keyboard | PASS: two real HTTP 200 previews, Stage/Unstage/Clear preserve a nonempty native draft, real external target loads, composer/Send fit 390x844, no overflow or inference | `native-preview-mobile-repair-acceptance.json`, three inspected/captured PNGs |
| Latest FastAPI runtime | PASS: actual 0.142.1 API, one generation-disabled retrieval and one Gemma dispatch, canonical input/result, citations and native fresh-document draft restore; no reload resend | `native-grounded-fastapi1421-latest-dev.json` |
| Post-upstream original data/contract preservation | PASS: ten/eight original rows and exact hashes, both original tabs and actual capability predicates/negative controls intact | `final-fastapi1421-preservation/` |

Private evidence root: `/private/tmp/chat-workspace-task12-v1-20261001`.
Original failed runners are retained, not relabeled as passing: date-format and
Ant Design selector assumptions were corrected; native keyboard scrolling was
used for offscreen citation controls. The final preview runner hit a readiness/
compositor timeout before any preview request or send; its failure and the
subsequent real readiness/migration observations remain recorded separately.
The first final external-link assertion incorrectly expected the title in the
site's body; the actual external title/content were verified in a distinct run.
Native mobile acceptance then found a genuine defect: retained recovery notices
outside the bounded transcript pushed Send below the viewport. The notices now
remain inside both existing workspace transcript scrollers, with no recovery
or owner-guard change. Containment tests reproduce two failures before the fix;
158 focused assertions and the corrected native mobile run pass afterward.

This is not an exactly-once inference claim. Unknown outcomes remain unknown;
explicit re-preparation can produce another provider answer.

## Final Code Verification

- All selected unit files ran directly from their owning package configs in the
  actual checkout: 82 shared files/2,228 assertions and three frontend files/50
  assertions, both commands exit 0, no failures or skips. Final results/logs:
  `final-latest-{shared,frontend}*`; original owning commands remain recorded.
- Independent immutable broad review: 240 selected paths, zero hash mismatches.
  Initial 2,266 owned assertions passed; the final three-file replacement was
  independently verified with 54 preview/URL assertions and 23 IndexedDB checks.
- Independent strict SourceV1 review passed 166 frontend and 159 Python checks,
  capability/boundary controls and all eight metadata digest/receipt mutations.
- Official SQLite/PostgreSQL scoped regression: 213 passed, one inherited
  SQLite-only connection-lifetime skip. Actual PostgreSQL arms ran; earlier
  sandbox reachability skips are not the acceptance gate.
- Additional changed backend scope: 403 passed, one inherited heartbeat skip.
  Static migration-test SQL cleanup: 20 passed, no skips. Counts overlap and are
  not a unique repository-wide total.
- Final production Turbopack build, shared-token sync and unchanged bundle gates
  exit 0: 539.9 KB shared (600 KB limit), 842.3 KB heaviest route (900 KB limit).
  The queue imports the existing exact ID helper directly without initializing
  the DB/history schema graph; its legacy export and algorithm remain identical.
  Exact source hashes bind the final copy to the matched isolated dependency
  snapshot. Direct checkout Turbopack cannot follow its preexisting external
  node_modules symlink; no repository config or budget was changed to hide this.
  Permission-corrected isolated builds use the normal CSS worker port. The prior
  webpack build is not substituted for the Turbopack budget gate.
- Independent final nine-file increment review: no actionable findings, 48/48
  checks, entry/exit hashes identical. Full frontend TypeScript exit 0 and all
  applicable cached pre-commit hooks pass. `final-increment-review.md` records
  focus, scroll, accessibility and guard analysis with its explicit limits.
- Latest-dev FastAPI 0.142.1 route/dependency/source regression: 34 passed, no
  skips; route-helper Bandit zero findings/errors. Owned API restarted gracefully
  with its exact preserved environment/data. Existing OTel SDK/exporter were
  aligned with the new mandatory API dependency; no provider/ML upgrade.
- Bandit: twelve touched production modules, zero findings/errors. Five setup
  test-fixture B105 warnings exactly match HEAD; assert checks are excluded for
  pytest files. No new findings in the changed scope.
- Final preview and loader ESLint: zero errors/warnings. Capability scope has
  nineteen inherited warnings, verified unchanged; no blanket lint-clean claim.
  The final mobile increment has zero errors and eleven inherited Research
  warnings; the Chat Workspace and new containment tests have no warnings.

The old remote `fd32aad` Onboarding and UX jobs failed at the production bundle
gate (654.1/958.0 KB and 651.1/955.0 KB respectively), before their browser tests.
The later UX health failure followed the stopped build, not a running API test.
The pure-ID import repair above addresses the measured eager dependency chain;
the new-head remote rerun remains a separate gate. Actual old-job logs are
retained as `remote-job-110516627382.log` and `remote-job-110516627352.log`.

The real-source JSON fixture is ignored by the repository-wide JSON rule and
must be explicitly included in the PR tree with SHA256
`d5bfed3bb97db124bcc990f13427dfd880b17af2b32984755211a1d06a83771d`.
The IndexedDB unit loader now resolves the existing dependency from its actual
frontend package rather than the runner's working directory; no install.

## Known Limits

The broader neighboring aggregate is not fully green. Nine untouched
SourceViewControls/Kitten assertions reproduce on HEAD; separate untouched
Studio export/lazy-modal assertions are timing-sensitive or not reproduced in
isolation. They remain enabled, with immutable controls in `pr-review/triage.md`.
This PR does not silently change unrelated Studio behavior or claim the full
repository suite passes. Native acceptance does not certify screen-reader speech
or unrelated administrative/provider routes. Remote CI is pending, not passing
by inference. The existing virtualenv still has unrelated ML/typer and stale
editable-package metadata conflicts; the new OTel conflict was repaired, and
the actual API/Gemma/embedding acceptance above passes. No global pip-clean claim.

Upstream introduced a different FastAPI task with the same historical root ID
13398. TASK-13408 is the unique finalization tracker. The completed Chat Workspace
root and all 31 descendants are archived through the official CLI with their
exact contents preserved; upstream's FastAPI record remains unchanged. Both
histories are retained without an active ID collision. The completed task-owned
implementation plan is removed; its approved design and acceptance record remain.

All 68 stashes were retained. Runtime builds, databases, logs, credentials,
screenshots and unrelated historical task records are excluded from the PR.

## Subsequent Latest-Dev Integration

The seven textual conflicts against dev `5f3ed81e` were resolved in
`a4cf2adbb175d49ce48ff81255ff904c83e7e726`, preserving upstream owner leases,
retry instruction transactions, image expansion and current-history callbacks.
The subsequent `d81c13f` advance contains eight Backlog changes and no source
changes; its merge is `5f52dde95936030353581ecaa1f8d7cf2efd9bfc`.

The old `792b6c4` remote run completed with 281 passing, 36 skipped and five
failed checks (three backend shards and two aggregates). Repairs address the
three actual causes without pinning dependencies or weakening guards:

- `SourceMetadataV1.total_chunks` now expresses its strict positive safe-integer
  bounds once, avoiding Pydantic 2.13.5's conflicting annotation precedence.
  Cached actual 2.13.5 schema/OpenAPI checks pass 142 tests; root-venv schema and
  SSE checks pass 151 tests.
- Four moderation audit fixture overrides use pytest's scoped dictionary
  restoration. The exact forward/reverse orders each pass eight checks;
  mandatory production audit remains fail-closed.
- SSE assertions require the existing `no-cache` and `no-transform` tokens.
  Production headers were not changed to satisfy obsolete expectations.

Independent overlap review also found two semantic regressions: captured
recovery reads needed the exact canonical message GET allowlist entry, and
protected image-reader errors needed the existing bounded 503/413 translation.
The shared image-expansion helper operates on already-authorized rows without
rereading them. SQLite image/recovery checks pass 33 tests; 24 PostgreSQL arms
were explicitly deselected because the local daemon is unavailable.

Fresh actual Chrome retrieval exposed a further raw-adapter mismatch before
inference: current RAG sources include bounded `source_id`, `evidence_origin`,
`section_path` and nonempty `ancestry_titles`. These are validated and omitted
only from the unchanged 17-field native wire. Citation/navigation consumers do
not read them; KnowledgeQA's separate trust flow is unchanged. The initial
failure had one retrieval and zero completions and remains retained. The
repaired native run has one generation-disabled retrieval, one real Gemma
completion, verified canonical receipts, citations and draft/reload restoration
with no resend. Source tests pass 159 checks and adjacent restoration tests 136.

Fresh native route acceptance passes six checks, including actual Back/Forward,
invalid-route checkpoint preservation and 390x844 layout, with zero completions.
The three owned APIs were gracefully restarted with their exact saved
environments and databases to load the final backend; both Next servers,
Chrome, Gemma and the existing embedding worker were preserved.

Additional owning-package regression runs pass 2,282 shared assertions in 80
files and six pure-ID/import assertions in two files, plus 50 frontend checks.
These overlap the focused suites and are not a unique all-repository count.
Full TypeScript, applicable cached pre-commit hooks and four production-module
Bandit checks pass. Actual ESLint inspection has zero errors and 26 inherited
warnings, verified against both merge parents; no warning-free claim is made.

The broader Chat_NEW lane remains non-green: 237 passed, 11 skipped and one
Hypothesis input-generation timing health check failed. Exact-seed replay and
the complete property file then pass (one and 17 checks), with health checks,
deadlines and strategies unchanged. No behavior counterexample was reported;
the failed run is not discarded or relabeled passing.

Local PostgreSQL is currently unreachable and Docker commands hang. No forced
daemon restart, replacement database, fixture bypass or fresh PostgreSQL pass
is claimed. The initial historical PostgreSQL results above remain distinct.
Original ten/eight-row projections still match their hashes, but historical
Chrome target IDs were already absent at this follow-up's first inventory.
The final inventory also lacks the two historical same-URL tabs; this follow-up
did not close or navigate those original tabs. All six current-baseline targets
remain. Neither historical ID nor same-URL tab preservation is claimed for this
run, and no replacement tab was fabricated to satisfy the old assertion.

Fresh evidence root: `/private/tmp/chat-workspace-pr3071-ci-followup-20261001`.
Current PR CI is running on the published integration; it is not assumed green.

### Exact Recovery Requalification

The first fresh cancellation driver attempt dispatched zero completions; an
actual foreground/window restoration then admitted and stopped one real input.
Verify exposed another genuine callback defect: native loading retains a keyless
owner, while its protected capture supplies the authoritative key in the view.
The read helper rejected the keyless object before any GET. Only that callback
argument now receives the protected view key; the controller owner, lease,
scope, signal and stale-response fences stay unchanged. Regression evidence:
60 passed/one failed before repair, 207 passed afterward, including foreign
profile/owner/conversation controls. Independent source review found no remaining
issue in this narrow fix.

The resumed real run retains the original admission for input `51f24478...`.
Its repeated identical recovery controls initially selected an older pending
case; that read and draft preparation produced no inference. Native keyboard
then selected the correct panel by its exact original input text: protected
message GET200, zero Verify completions, explicit Reprepare. The continuing
runner's explicit Send produces the second total dispatch and a verified saved
result with the same canonical input. The earlier unresolved operation remains
in its ledger. Both `native-merged-stop-recovery-resume.json` and
`native-exact-recovery-ui-action.json` are required evidence; the broad driver
alone is not proof that the correct original input was inspected.

Failed surface-capture/load traces remain separate. Screenshot pumping is now
nonblocking and uses Chrome's actual view so a stalled capture cannot prevent
the UI deadline from advancing. No focus/visibility emulation or synthetic state
was introduced. Final TypeScript passes with an8192MB heap; the default4096MB
process's out-of-memory failure is retained, not counted as a passing run.

### Completed Fresh Matrix

All of the following ran against the actual integrated source and the existing
live services, without mocks, injected application state or automatic resends:

| Current check | Result | Evidence |
| --- | --- | --- |
| Grounded RAG, receipts, citations, draft/reload | PASS: one generation-disabled retrieval and one real Gemma completion; no reload send | `native-merged-grounded-repaired.json` |
| Exact Stop/Verify/Reprepare | PASS: protected original-input GET200, zero Verify inference, explicit second dispatch using one canonical input; original unknown retained | `native-merged-stop-recovery-resume.json`, `native-exact-recovery-ui-action.json` |
| Route/hash/Back/Forward/invalid route/mobile | PASS: six checks, zero completions | `native-merged-route.json` |
| Workspace A/B/A, Research handoff, New Chat | PASS: eleven checks, preserved inactive checkpoints, zero completions | `native-merged-workspace.json` |
| Browse/staging/Clear/external link/mobile keyboard | PASS: two HTTP200 previews, real Example Domain target, nonempty draft retained, composer/Send fit390x844, zero completions | `native-merged-preview.json`, three fresh inspected PNGs |
| Actual current owner baseline | PASS: original A canonical row hash and draft verified, full pre-switch checkpoint unchanged under B, zero completions | `owner-uat/fresh-negative-owner-baseline.json` |
| Account A/B/A and target A/B/A | PASS: nineteen checks, two new explicit Gemma sends/four new canonical rows, old A restored unchanged, foreign reads404, no switch/reload autosends | `owner-uat/positive-owner-controls-1790906645517.json` |
| Fresh owner-return visual | PASS: loaded original A transcript/draft before and after fresh capture, zero completions; image inspected | `owner-uat/fresh-visual-final.json`, corresponding PNG |
| Current original-data/contract/stash gates | PASS: exact ten/eight-row projections, served capability predicates/12negative controls, all68stashes and six follow-up baseline targets | `current-preservation/current-preservation.json` |

The first owner rerun compared against the old negative fixture's client-session
ID and failed before any send. Inspection found only the expected session
renewal, with original rows, owner and draft unchanged. A fresh real A baseline
was captured before A-to-B switching; the same full-checkpoint SHA comparisons
then pass throughout the matrix. The historical failed fixture run remains
separate. The first preservation runner's historical same-URL assertion also
failed; current preservation reports that limitation explicitly rather than
weakening or relabeling the original gate.

Final source-bound production Turbopack build, token-sync and unchanged bundle
gates pass:553246bytes shared (540.3KB,600KB limit),862892bytes heaviest route
(842.7KB,900KB limit). All tracked frontend bytes and the directory symlink were
checked against the checkout, including the uncommitted recovery fix. The reused
dependency clone required both tracing roots to include its sibling directory
only in the external snapshot; repository roots and budgets are unchanged. The
earlier symlink-verification and mismatched external-root failures are retained.
Three existing documentation-pattern tracing warnings and stale Browserslist
data remain; no warning-free build claim is made.

At the preceding publication the latest inspected upstream was `d81c13f`. Its
remote CI was pending,
not certified green; fresh PostgreSQL is unavailable and the broader local
Hypothesis timing failure remains recorded above. These are explicit remaining
qualification limits, not fabricated UAT successes.

## Cookie-Model Dev Requalification

Frozen dev `dcae0cbd` merges cleanly. Its two model-discovery services retain
live authenticated cookie-profile admission, cookie/key in-flight separation,
fresh capability checks and cache invalidation. The earlier handled-error
`console.warn` remains intact. Independent source review found no actionable
regressions. Both task histories are preserved: the previous finalization
TASK-13408 archive has identical SHA256
`f6b0da4206b72589bbdff853a1ca158f7312df96d47d7e9b976d48d2fe4cbd89`;
incoming cookie TASK-13408 is untouched and continuation uses TASK-13417.

Fresh owning-package model/recovery scope: 292 passed, no failures. Full frontend
TypeScript passes with the existing 8192MB allowance. Six incoming frontend
files have zero ESLint errors and six warnings. Bandit on the four final touched
Python production modules has zero findings/errors; this dev advance changes no
backend source. Final affected regression rerun again passes292 checks; normal
applicable pre-commit hooks pass without bypass. Existing deprecated hook-stage
warnings remain. All7217 tracked frontend files/symlink bytes still match the
qualified build snapshot immediately before publication. Final actual read-only
preservation again passes the original ten/eight-row hashes, served contract,
68stashes and six follow-up tabs; historical missing tabs remain unavailable.
Latest fetched `dev` is still `dcae0cbd`.

Private evidence: `/private/tmp/chat-workspace-pr3071-cookie-dev-20261001`.
Actual Chrome raw CDP and native input use live auth, SQLite/IndexedDB, Gemma and
embeddings. No interception, mocks, state injection, focus emulation or install.

| Fresh dcae check | Result | Evidence |
| --- | --- | --- |
| Grounded turn, canonical sources, draft/reload | PASS combined evidence: one generation-disabled retrieval and one completionHTTP200, captured four-source payload matches two canonical rows; independent native post-send/full fact-bearing excerpt and draft/reload pass with zero additional sends | `combined-grounded-verification.json`, `main-uat/postsend.json`, `main-uat/fullsource.json` |
| Exact Stop/recovery/explicit reprepare | PASS: admission then Stop, exact logical-input GET200, no reload/Verify autosend, two deliberate dispatches with one canonical input/result; original unknown ledger retained | `native-dcae-stop.json` |
| Route/history/invalid/mobile | PASS six checks, zero completions | `native-dcae-route.json` |
| Workspace A/B/A, Research handoff, New Chat | PASS eleven checks, zero completions and inactive checkpoints retained | `native-dcae-workspace.json` |
| Browse/staging/Clear/external link and mobile keyboard | PASS combined evidence: both HTTP200 previews, actual memo facts and external Example Domain load; nonempty draft retained. Separate native mobile continuation reaches composer then Send at390x844 with zero overflow and zero completions | `native-dcae-preview.json` (two completed steps; overall runner FAIL retained), `mobile-keyboard-continuation.json`; three fresh inspected PNGs |
| Real owner baseline and account/backend return | PASS original rows/draft plus full inactive checkpoint hashes; nineteen checks, two explicit new Gemma turns/four canonical rows, foreign404 controls and no switch/reload autosends | `owner-uat/fresh-negative-owner-baseline.json`, `owner-uat/positive-owner-controls-1790910636302.json` |
| Fresh restored owner visual | PASS loaded original A transcript/draft before and after actual capture, zero sends | `owner-uat/fresh-visual-final.json` |
| Real cookie model discovery and reload | PASS existing bootstrap200, authenticated profile200 before model metadata200, no browser API key/token; fresh reload revalidates profile, zero sends. Separate settled Healthy/Gemma fresh capture excludes the transitional reload frame | `cookie-uat/catalog-final.json`, `cookie-uat/fresh-ready.json` |

Retained failures are not relabeled. The heavily reused conversation's next
input `4364266d...` received provider502: actual Gemma rejected an8612-token
request against8192-token context. That canonical input remains; native New Chat
created a bounded new conversation rather than resending it or raising limits.
Its first driver waited for a network-finish event and timed out after one
retrieval/completion despite a settled saved answer. Combined verification uses
the captured actual HTTP200/request payload and independent canonical/native
checks, while `native-dcae-fresh-grounded.json` still reports its original FAIL.
Full fact-bearing source proof selects the memo chunk, not only its title chunk.

The preview runner completed both real preview/staging flows but its fixed
80-Tab mobile traversal stopped on an older message action, so its overall FAIL
is retained. A separate continuation records the actual visible tab order and
uses native Tab input only: four observations reach the composer, followed by
Send, without activating Send. Both controls fit390x844 with zero overflow,
the nonempty draft is unchanged and no completion is dispatched. No application
change or injected focus/state was needed. Fresh memo/web/mobile images and the
full-source, Stop, route and workspace screenshots were inspected.

The isolated cookie production frontend initially inherited the build-only
`app:8000` proxy origin. Its private build was corrected to the real loopback API,
without changing repository configuration, auth policy or existing services.
The first launcher lacked the existing Next binary PATH; its failure is retained.
The first cookie driver read manual config rather than the real cookie binding;
its stopped progress is retained. A second asserted metadata before its response;
that FAIL is retained. Final mounted checks wait for real metadata200 and use
the canonical cookie binding. The initial reload frame was transitional, not a
settled-readiness claim; the separate fresh Healthy/Gemma image was inspected.

Both byte-bound production builds, token sync and unchanged budgets pass. The
standard build is540.5/844.1KB; the real-loopback cookie build is540.3/842.7KB,
under600/900KB. Existing dependencies are reused; only external tracing roots
include their sibling clone. Original limits (fresh PostgreSQL, historical tabs,
broader Hypothesis timing failure and current-head hosted CI) remain explicit.

Published integration: `1fb829cf256efe452f6d9b137d7ba2e9f6d70081` on existing
PR3071. GitHub readback verifies the exact head/body, base `dev`, open draft
state and unchanged requester-supplied Change summary; the PR remains attached
to this chat and unmerged. Head-bound hosted runs are queued, not passing. The
replacement license audit36960989844 and backend await-license36960906531
have no assigned runner; backend admission is skipped. The earlier same-head
license audit was cancelled and is not counted as a code/test failure or pass.
This publication-record follow-up changes no tested application source.

## Route-Auth Dev Retry

Frozen and finally re-fetched dev `3caebcfc` merges cleanly. Its nine incoming
paths affect authentication/inspection, not Chat Workspace runtime code. The
prior TASK-13417 qualification archive remains byte-identical, SHA256
`546a0220d918a561522667767d85ea78fc1bb25d2d798af999ce1b0ff9dde93a`;
incoming unrelated route-map TASK-13417 is unchanged. TASK-13418 owns this retry.

Independent review found two real inspection gaps: CSV-only copying corrupted
JSON route lists, and pinning the temporary config before dotenv selection
caused the repository config to shadow FILE/PATH/DIR selections. Four real
subprocess inventory regressions fail before the fix. The shared loader now
reuses the canonical route-policy parser, early dotenv loading and cache reset
APIs. All17 owning tests pass; the final lint/benchmark delta has87 passes and
one inherited skip. Follow-up independent static review has no actionable
findings; newline lists and dotenv test-mode flags are not newly covered.

Changed Ruff checks pass. Connector I001/SIM114 diagnostics are byte-equivalent
to the published source; no unrelated cleanup or blanket lint-clean claim.
Final Bandit on all four incoming production Python modules has zero findings
and errors. Current Chat source remains unchanged, so the earlier292 regression
checks and production builds are reused as source-bound historical evidence,
not reported as fresh runs. All7217 qualified frontend entries match except
the metadata-only OpenAPI fingerprint updated below.

Fresh actual native Chrome/CDP reload obtains a new document loader and the
protected history GET200, restores the nonempty draft and remains Ready/loaded.
At390x844 composer/Send fit with zero overflow;92 native Tabs reach the
composer without sending. Desktop/mobile captures are inspected; no mocks,
interception, state injection, focus emulation or completions. The existing
services were preserved, not restarted to claim browser coverage of unrelated
incoming admin routes. Original ten/eight-row hashes, actual served contract,
six baseline tabs and all68 stashes remain intact. Historical missing tabs
remain missing and are not claimed preserved.

Hosted `backend-required` on head7274 fails its OpenAPI drift gate, not its
explicitly non-blocking mypy step. An isolated CI-version dependency directory
(FastAPI0.142.2/Pydantic2.13.5/Starlette1.7) reproduces the exact failed hash;
the frozen-dev production control reproduces the checked-in old hash. Reviewed
delta:14 added durable/recovery schemas, three updated models and nine intended
capability/owner parameter contracts; zero removed paths or schemas. The
existing exporter/openapi-typescript regenerate the fingerprint/types, and a
fresh drift check passes with sha256
`f4609bf67c33c5d62ecc243f98c918836f09181200768dc0c28b615732d5007f`.
The root virtualenv and live UAT services were not upgraded.
Fresh CI-version route-auth/route-map rerun passes23 tests with six warnings;
full frontend TypeScript passes after generated-type refresh using the existing
8192MB allowance. These are fresh checks, separate from reused historical tests.

Local official PostgreSQL fixtures skip all seven selected cases because the
service is unavailable. Docker's read-only probe times out and5432 refuses
connections; no restart, shared-container removal, replacement DB or bypass.
Direct hosted logs/JUnit on head7274 confirm56 auth-postgres tests and149
auth-integration-b-z tests actually pass, including all six durable-turn
PostgreSQL cases without skips. This qualifies that unchanged owning scope;
the separate image-recovery PG case and next-head hosted CI remain unqualified.
The head7274 snapshot has211 successes, one backend-required failure,
17 running and59 queued (plus35 skips/two status successes). These are not
certification of the next published head.

Private evidence: `/private/tmp/chat-workspace-pr3071-retry-20261002-0552`.
Native evidence: `native-reload.json` and two inspected PNGs; red/green/final
delta JUnit, Bandit, schema control/delta, hosted logs/JUnit and source bindings
are retained there. Original runner failures remain separate. Preservation is
under the preceding evidence root's `route-auth-retry-preservation-20261002`.
