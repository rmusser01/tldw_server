# Chat Workspace Latest-Dev Acceptance

Tracking: TASK-13408; historical Chat Workspace TASK-13398 and its children.
Epic: https://github.com/rmusser01/tldw_server/issues/1239.
Final baseline: dev `ec86ba4e871d844ff4a3a53c607e80eb85bd0e9b`, merged and rechecked before final PR packaging.

## Current Gate

The approved no-mock native acceptance matrix is complete. Final account/target,
Browse/staging/external-link, mobile keyboard and fresh visual gates pass.
The merged latest-dev FastAPI 0.142.1 API also passes a fresh actual grounded
Gemma/citation/draft/reload round trip. [PR #3071](https://github.com/rmusser01/tldw_server/pull/3071) remains draft/unmerged pending
remote CI and the separately required requester-authored Change summary.

## Real Acceptance Recorded

All browser acceptance below used real Chrome raw CDP, native API authentication,
SQLite, browser IndexedDB/localStorage, the actual local Gemma provider and real
embeddings. No request interception, mocked responses, injected application
state, fabricated receipts, focus emulation or plugin installation was used.
Unit-test doubles are separate from browser acceptance.

| Latest-dev check | Result | Evidence |
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
