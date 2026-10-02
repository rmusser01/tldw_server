# Chat Workspace Latest-Dev Acceptance

Tracking: TASK-13408; historical Chat Workspace TASK-13398 and its children.
Epic: https://github.com/rmusser01/tldw_server/issues/1239.
Initial acceptance baseline: dev `ec86ba4e871d844ff4a3a53c607e80eb85bd0e9b`.
Current integration baseline: dev `d81c13fddd1dac1948b30401af0388932a0af8f2`.

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

The latest inspected upstream remains `d81c13f`. Current remote CI is pending,
not certified green; fresh PostgreSQL is unavailable and the broader local
Hypothesis timing failure remains recorded above. These are explicit remaining
qualification limits, not fabricated UAT successes.
