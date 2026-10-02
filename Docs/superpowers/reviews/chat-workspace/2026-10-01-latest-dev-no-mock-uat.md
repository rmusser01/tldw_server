# Chat Workspace Latest-Dev Acceptance

Tracking: TASK-13421.1; previous qualifications TASK-13421, TASK-13418 and TASK-13417 and finalization
TASK-13408 archived intact;
historical Chat Workspace TASK-13398 and its children.
Epic: https://github.com/rmusser01/tldw_server/issues/1239.
Initial acceptance baseline: dev `ec86ba4e871d844ff4a3a53c607e80eb85bd0e9b`.
Current integration baseline: dev `38b09af8e92b3d9a2a00aced22d6b992dbae9435`.

## Current Gate

The latest rebase and review qualification is recorded below. Fresh hosted CI
and Qodo review must qualify the final published head before the newly
authorized merge. Earlier frozen-dev and approved mobile evidence remains
source-bound and historical; pending checks are not inferred from it.

Historical source integration `5dbf0b076883907dccbb174332d83dae20daf541` on frozen
dev `3caebcfc` now qualifies the required hosted backend/frontend checks and
all seven owning PostgreSQL cases, including the image-recovery snapshot.
The recorded real Chrome acceptance covers grounded chat, citations, durable
receipts, cancellation/recovery, owner/workspace return, previews and mobile
keyboard behavior without mocks. Historical and reused results remain labeled
with their actual source baselines below.
[PR #3071](https://github.com/rmusser01/tldw_server/pull/3071) was draft and
unmerged at the earlier publication. The requester-authored Change summary is published
verbatim. Evidence-only publication updates reuse the verified application
source; their own hosted status must not be inferred from an earlier head.

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

The latest inspected upstream remains `d81c13f`. Current remote CI is pending,
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
Before publication, direct hosted logs/JUnit on head7274 confirm56 auth-postgres
tests and149 auth-integration-b-z tests actually pass, including all six
durable-turn PostgreSQL cases without skips. This qualified that unchanged
owning scope; the separate image-recovery PG case and next-head hosted CI were
still unqualified at that snapshot.
The head7274 snapshot has211 successes, one backend-required failure,
17 running and59 queued (plus35 skips/two status successes). These are not
certification of the next published head.

Private evidence: `/private/tmp/chat-workspace-pr3071-retry-20261002-0552`.
Native evidence: `native-reload.json` and two inspected PNGs; red/green/final
delta JUnit, Bandit, schema control/delta, hosted logs/JUnit and source bindings
are retained there. Original runner failures remain separate. Preservation is
under the preceding evidence root's `route-auth-retry-preservation-20261002`.

## Published Source Qualification

The exact published source head `5dbf0b076883907dccbb174332d83dae20daf541`
has 293 successful checks, 35 skipped checks and one cancelled superseded license
audit; zero failed, queued or running checks. The replacement license audit and
trusted license status succeed. Both [backend-required](https://github.com/rmusser01/tldw_server/actions/runs/36974608567/job/110738851124)
and [frontend-required](https://github.com/rmusser01/tldw_server/actions/runs/36974608496/job/110742239202)
succeed, closing the earlier OpenAPI drift failure. Skipped checks are not
counted as tests passed, and the cancelled earlier audit is not a test failure.

Actual downloaded JUnit from run36974608504 closes the remaining PG gate; both
artifact inventories identify the exact source head above:

- [Auth integration B-Z](https://github.com/rmusser01/tldw_server/actions/runs/36974608504/job/110741695710): 149 passed, zero skips/failures/errors; all six durable-user-turn PostgreSQL cases actually ran.
- [Chat integration](https://github.com/rmusser01/tldw_server/actions/runs/36974608504/job/110741703457): 861 passed, 30 skipped, zero failures/errors. All 60 image-recovery cases pass, including `test_postgres_strict_image_snapshot` in 3.175s with no skip.

The local seven unavailable-service skips remain skips. Hosted PostgreSQL
qualification does not claim a local Docker repair or a no-mock hosted LLM
session; the separate native Chrome/Gemma/embedding UAT supplies that evidence.
No unresolved review threads or reviews were returned by the final PR readback.
No application changes, inference resends, service restarts, tab closures or
stash changes were needed to close these gates. Only the owned completed retry
plan is retired; other plans and historical evidence are preserved.
The final read-only preservation check again verifies the original10/eight-row
hashes, actual served contract, six baseline tabs and all68 stashes. Historical
missing target IDs/same-URL tabs remain absent, not claimed preserved.

Private evidence: `/private/tmp/chat-workspace-pr3071-continue-20261002-oxxtxz`.
`pr-current.json`, `review-current.json` and the two downloaded JUnit/log
artifacts bind this conclusion to 5dbf. Final documentation/task publication
does not certify its next-head hosted checks before they actually finish.
Final preservation evidence is under the preceding cookie-dev evidence root's
`final-ci-closure-preservation-20261002` directory.

## Release And Redis Dev Qualification

TASK-13421 freezes dev `413c2c9123509f17d96d514a7722e076598ea28f` after
one fresh fetch. It integrates release0.1.46, Redis request-window semantics,
relay recovery budgets, startup-secret output, Drawer props and scoped smoke
exceptions. Clean preview tree `69ab09f175fea8bad4c3225787abbb7dfceda29b`
and the actual merge agree. No additional application changes were needed.
The completed own TASK-13418 is archived byte-identically (SHA256
`dad9f464d19c86e97666c71cb90eb809c222b57719651fe815d873ac5f80e55a`);
both unrelated incoming same-ID records retain their upstream bytes.

Fresh verification:

- Incoming governor, policy, middleware, relay/recovery and startup-secret
  checks: 296 passed, two expected xfails. Both inherited xfails concern Redis
  burst/retry-after determinism; neither is relabeled as passing. Unit Redis
  isolation/doubles are not no-mock browser evidence.
- Official real-Redis Lua/TTL integration: nine passed, zero skips, unique
  fixture namespaces; no shared key flush. Independent incoming-module review
  found no actionable introduced issue. Existing window-key expiration work
  remains separately tracked upstream, not silently folded into this PR.
- Seven touched production files: Bandit has zero new findings/errors versus
  frozen3cae. Its one unchanged B113 warning mistakes a local policy dictionary
  named `requests` for an HTTP client. Ruff has nine inherited findings versus
  ten in the baseline, zero new findings. No suppressions or guards were added.
- Full TypeScript, production build, token sync and unchanged bundle budgets
  pass: shared540.3KB/heaviest842.7KB gzip against600/900KB. Owning ESLint passes
  for both changed UI files and smoke files. One owning Drawer test and six
  directly executed existing smoke-classifier scenarios pass. An initial
  root-level lint invocation selected an incompatible cached ESLint; its
  failure is retained, and the actual installed frontend runner passes.
- Canonical CI-version schema export/check passes: 2105 paths, 3259 schemas,
  unchanged fingerprint `f4609bf67c33c5d62ecc243f98c918836f09181200768dc0c28b615732d5007f`.
  No generated contract/type churn is necessary.

Fresh no-mock acceptance uses new loopback API18094 and production frontend18095,
leaving existing services, ports, Chrome tabs/profiles and drafts untouched.
The API advertises0.1.46, native protected profile authentication succeeds and
governor diagnostics report `real_redis=true`, `multi_lua_loaded=true` and
`last_used_multi_lua=true`, with fail-closed Redis configured on isolatedDB14.
The frontend build is byte-bound to all merged application files; the only
external-clone config adjustment includes already installed sibling dependency
paths, without changing repository configuration or budgets.

Actual Chrome target `8B93C54868C475C7D9A2E213EF98C613` loads and connects the
real workspace through its normal route. A first explicit Send with no model
performs generation-disabled retrieval but admits no completion. Allowing the
normal Chat route/provider initialization to settle selects the real Gemma;
the draft survives, and a second explicit Send stages the real memo and produces
one HTTP200 completion. Across those two explicit clicks, both retrieval
requests specify `enable_generation=false`; they are not automatic resends.
The original driver wrongly expected one retrieval across both clicks and
failed; that run remains failed. Its closed CDP session no longer exposes the
response bodies, so no missing response body is inferred or fabricated.

Separate no-send continuation verifies protected canonical rows: exactly one
`input_verified` user and one matching `result_verified` assistant, with all
source metadata equal to the captured real completion request. Gemma answers
18 November2026 and Mira Chen. Native citation expansion exposes the captured
source, native input writes a nonempty IndexedDB checkpoint, and a fresh-loader
reload restores the same draft, verified answer and citations through actual
HTTP200 history reads. Request counts remain one completion/two explicit-click
retrievals: reload triggers neither. The final desktop screenshot was inspected.
Conversation: `0ff15c7a-9a06-40de-aa88-1085df6dc042`.

The new API's actual served contract passes the existing production capability
predicates and twelve negative controls. Original conversation projections
remain byte-identical at ten/eight rows, six follow-up baseline targets remain
open and all68 stashes remain retained. Historical missing target IDs/same-URL
tabs remain missing and are not claimed preserved. Current mobile viewport
UAT was initially blocked by auto-review pending explicit viewport-only approval;
the approved fresh413 pass is recorded below. Earlier390x844 mobile results
remain historical. No alternate emulation or focus/visibility workaround is used.

Owning source binding verifies394 unchanged backend/API-dependency/DB/chat and
fixture files against qualified source5dbf, including both owning PostgreSQL
test files. Its seven actual hosted PostgreSQL passes remain reusable historical
owning evidence, not a fresh whole-runtime CI result. Incoming shared runtime
changes have the fresh focused checks and live Redis/Gemma acceptance above.

Private evidence: `/private/tmp/chat-workspace-pr3071-head8757-20261002-w0zazO`.
The `native-grounded-413` failure, `native-verify-reload-413` pass, source bindings,
real Redis JUnit, security/lint comparisons, build/schema logs and
`latest-api-actual-preservation` are separate records. PR3071 stays draft and
unmerged; requester Change summary must remain byte-identical. New-head queued,
skipped and completed hosted checks must be reported distinctly.

Source publication `93780ae94b8e7d37e5dfe9c6e8fe8109a575beb3` is normally
pushed to the existing draft PR. All applicable configured pre-commit checks
pass over35 integration/evidence files; scoped Ruff/Black hooks correctly skip
non-wizard paths, with the separate incoming lint comparison above retained.
Exact PR readback verifies source head, dev base, OPEN/draft state and unchanged
requester summary bytes. Subsequent source-head CI snapshot has57 successful,
28 skipped,22 running and7 queued checks, zero failures; backend-required is
still running. No new reviews or unresolved review threads are returned.
This is not a completed hosted CI claim, nor does it qualify a later metadata
head before that head's own checks finish.

Independent full fact-bearing memo citation expansion also passes with native
Tab/Enter and the existing complete CDP key payload: rendered source contains
18 November2026 and Mira Chen, actual document visibility is `visible` and the
nonempty draft is unchanged. Its fresh screenshot was inspected. Three earlier
driver attempts remain failed: pointer hit testing checked viewport bounds but
missed transcript clipping, then incomplete Enter lacked native key codes/text.
Read-only hit testing showed the composer at the pointer position; source review
confirmed ordinary native details/summary behavior. After stopping/reassessing,
the already-working complete-key helper supplies the successful alternate
approach without application edits, emulation or another send. Evidence:
`native-full-memo-keyboard` (FAIL), `native-full-memo-complete-key` (PASS).

## Approved Mobile Closure

After the user continued following the explicit viewport-only approval request,
auto-review permits the original native Chrome runner. It completes against
metadata head `d6a7beca75d380322328017cdd9ce32ef28702ed`, whose application/test
bytes are unchanged from source integration93780ae on frozen dev413c2c9.

Fresh no-mock mobile acceptance passes at390x844: actual `visible`/`loaded`
document, retained verified Gemma transcript and nonempty draft, fresh document
loader and protected historyHTTP200. Composer bounds(17,650,356,88) and Send
bounds(242.34,746,130.66,44) fit completely with zero horizontal overflow.
Twenty-one native Tabs reach the composer; a separate native Tab then reaches
Send without activating it or altering the draft. Both checks observe zero
completion dispatches. Fresh mobile transcript/composer and Send-focus
screenshots were inspected. No focus/visibility emulation, mocked services,
request interception, state injection or provider resend is used.

The original mobile Send-focus driver has a read-expression quoting error and
fails before keyboard navigation; its evidence remains FAIL. The corrected
read-only selector expression and distinct output record pass. No application
change is needed. The accompanying `fromSurface=false` desktop capture crops
to the native physical window/compositor; it is not substituted for the earlier
inspected full desktop acceptance image.

Final latest-API preservation again passes actual capability predicates/twelve
negative controls, original ten/eight-row hashes, six baseline tabs and all68
stashes. Historical missing tabs remain missing, not claimed preserved. TASK-13421
acceptance is complete; only its owned completed plan is retired. Prior plans,
failures, profiles, services and unrelated files remain intact.

At the resumed d6a7 head, actual workflow metadata identifies a successful
replacement license audit; required backend/frontend/e2e/security/coverage runs
are queued, not failed or qualified. Hosted CI completion is not claimed for
d6a7 or a later documentation-only closeout. PR3071 remains draft and unmerged,
and the requester Change summary remains byte-identical.

Private evidence remains under the preceding root: `native-reload` and
`native-mobile-send-focus-qualified` pass; `native-mobile-send-focus` retains
the quoting failure. `final-mobile-closure-preservation`, exact-head PR/workflow
readbacks and the inspected mobile PNGs bind this closure to the real runtime.

## Latest Dev Rebase And Review Qualification

TASK-13421.1 records the user's new authorization to rebase, address PR findings,
and merge only after review and verification. Original head0c9d592231 is retained
in local backup branch `codex/pr3071-before-rebase-20261002`. Rebase with merge
topology preservation onto devdf17c8ac3f completes. Restoring the owned13418
archive move byte-identically makes the complete tracked result equal the clean
expected integration treecbf4388b, before this corrective task's additions.

Upstream08c9fe0f3d already fixes the stale published ADR056 that caused the two
docs-refresh CI failures. Fresh docs suite33passed; combined docs/CI/route-auth
and durable-wire273passed; incoming fixture regressions52passed. Correctly
isolated durable Chat persistence/completion/API tests180passed/24skipped. Local
skips are not claimed as PostgreSQL passes; the final hosted owning cases remain
required. Two earlier test-invocation runs hit the unchanged DB path guard
because basetemp was outside macOS's default tempfile root. Their failure and
interrupted-run logs are retained. The corrected invocation sets TMPDIR to its
isolated test root, without changing application validation.

Independent read-only integration review identifies one actionable incoming
test-helper defect: parent PATH can find Homebrew timeout while child PATH
excludes it. A new parent-only availability regression fails before the fix
with `exec: timeout: not found`. The existing helper now reuses one child_path
for both executable resolution and subprocess environment;56owningCI tests and
all5formattedhelper cases pass afterward. Ruff check/format pass. The reviewer
rechecks the small fix and reports no actionable regressions. This is a unit
test correction, not mocked UAT or a changed runtime.

Fresh Bandit on14 production/helper paths has zero findings/errors. Raw Bandit
on the changed test preserves inherited subprocess diagnostics and existing
pytest assertions, with only two additional intentional pytest B101 assertion
diagnostics. These are test expectations, not production enforcement; no
suppression or security guard is added or removed.

All production/frontend/Helper_Scripts bytes remain identical to the prior
real Chrome desktop/mobile acceptance source0c9. Fresh read-only live API and
raw-CDP preservation checks verify the actual served contract/twelve negative
controls, original ten/eight-row hashes, six baseline targets, and all68stashes.
Historical missing tabs remain absent, not claimed preserved. No provider
resend, browser-state injection, runtime restart, or mocked UAT is performed.

Dev advances during verification to38b09af8e9; its delta fromdf17 is only a
completed relay task record, with no runtime/test changes. The second rebase
includes it cleanly and matches expected treea082f5f9. Qodo has not yet reviewed the draft;
marking ready and final-head CI/review remain pending, and no merge is claimed.
ADR required: no new durable decision; ADR-002/004/006 remain governing.

Private evidence: `/private/tmp/pr3071-rebase-qodo-20261002-hWnKnr`. The preserved
live check is `rebased-dev-preservation-20261002` under the preceding evidence
root. Original human Change summary remains unchanged.
