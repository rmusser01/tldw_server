# Owner-Fenced Workspace Chat Checkpoints

Tracking: TASK-13398.12.
Approved specification: Docs/Design/CHAT_WORKSPACE_OWNER_CHECKPOINTS_2026_09_29.md.
Current baseline: dev 3763d4b013ad31ad48cc7f9d895d2ae096223da8 plus preserved dirty candidate.
Approved compatibility contract: Docs/Design/CHAT_WORKSPACE_H1_DURABLE_COMPATIBILITY_2026_09_30.md.
Human authorized fixing all identified issues, full no-mock UAT, commit/push and PR creation on 2026-10-01. Do not merge; the human-written Change summary remains a merge gate.

## Stage 1: Qualified Existing Records
**Goal**: Extend existing session clone/normalization/persistence with versioned
checkpoint metadata: verified owner key, workspace/reference, H1 reference, draft.
Use an optional qualification argument on the existing reader for exact-key reads;
never use its legacy fallback for qualified checkpoints. Keep legacy data intact.
**Success Criteria**: Qualified reads reject absent/mismatched metadata and invalid
conversation references without writes. Empty draft-only records survive storage.
**Tests**: RED/GREEN exact-key, mismatch, clone independence, bounded serialization,
rehydration, draft-only and rejected-legacy preservation regressions.
**Status**: Complete

Verification: qualified-record RED/GREEN 62/62; parent existing-store regression
216/216 across eight suites; frontend TypeScript exit 0. Independent frozen-diff
review found no actionable findings. Checkpoint-specific tests use inline
storage; adjacent IndexedDB tests do not constitute native checkpoint acceptance.

## Stage 2: Shared Owner And H1 Handoff
**Goal**: Reuse existing verified service scope, owner key, H1 controller, store
session actions and route replacement event for both workspace chat surfaces.
Capture outgoing state, fence async restoration, and restore validated history
through H1 rather than cached rows. No credential storage or automatic sending.
**Success Criteria**: Account/target changes clear sensitive mounted state; stale
loads cannot overwrite new selections, drafts, New Chat or A/B/A transitions.
**Tests**: Deferred ownership and history loads, typing, explicit route intent,
New Chat, unmount/switch handoff, cancellation and hydration failure.
**Status**: In Progress

Integration dependency: existing H1 selected-history admission rejects grounded
RAG and conflicts with workspace durable `tldwTurn` admission, even for plain
saved sends. The backend also rejects simultaneous H1/durable protocol fields.
Do not mount the provider or remove guards as a shortcut. Stage 2 needs an
explicit admission/settlement design covering selected ancestry, citations,
Retry identity and unknown-outcome recovery before implementation. Read-only
assessment: /private/tmp/chat-workspace-h1-rag-feasibility-review-20260930.md.

Compatibility design (authorized design work, not protocol implementation):
Docs/Design/CHAT_WORKSPACE_H1_DURABLE_COMPATIBILITY_2026_09_30.md. Independent
review's three recovery ambiguities are addressed: verify protected live result
state, distinguish browser inference observations from logical user UUIDs, and
disable unavailable exact-payload replay after reload. Reviewer confirmed these
corrections; concrete citation bounds/digest/capability/recovery wire shapes and
assistant/context matrix remain gates. Stage 2/3 application guards stay intact.
Current candidate baseline advanced to dev f3f1b4fdbe3fe461b371ece30887c5fff8476d9d;
its 21 RG/auth paths do not change the frozen Stage 1 or H1 protocol code.

Concrete V1 candidate is now incorporated into the same design: strict bounded
source projection, mandatory empty result for plain sends, accepted-reference
attempt digest, versioned OpenAPI capability, protected live recovery-read DTO
and explicit assistant/context matrix. Independent review's body/query-scope P2
is corrected and re-reviewed: only the dispatched JSON body is hashed, while
pinned target/account/scope leases and server checks remain separate. Source
bounds and support exclusions still require human approval; backend digest parity,
normalized saver/read proof and native checkpoint behavior are implementation
gates, not tests run by this design work. Historical Stage 2/3 status was Not Started.

Human approved concrete bounded V1 implementation on 2026-09-30. Execute strict
schema and ECMAScript digest parity, then single-input admission, normalized
atomic result and protected live recovery reads, then shared owner/H1 handoff.
Capability advertisement remains disabled until coherent contract verification.
Track wire, browser serialization and recovery proof as separate reviewable
children of TASK-13398.12. Existing guards remain for unsupported combinations.

Implementation progress on 2026-09-30: wire/digest, protected recovery reads and
browser observation ledger have passed independent replacement reviews. Full
official SQLite/PostgreSQL recovery regression: 162 passed, one SQLite-only
connection-lifetime skip; prior sandbox reachability skips are superseded.
Completion review's moderation, raw-result eligibility and mandatory SSE error
fixes passed an independent frozen review (409 passing tests, one inherited
heartbeat skip). Source operation annotations passed independent review; the
restored latest-dev API now advertises the verified contract. Actual
generated OpenAPI plus 12 negative controls pass after declaring the UUID format
already enforced by the result DTO. The raw-to-wire pipeline follow-up has 347
passing tests; its independent frozen review passed. The recorded real RAG response
contains consumed media/locator metadata outside the approved SourceV1 contract;
that route stays gated pending contract preservation rather than silently
dropping evidence. Unit fixtures and recorded responses are not native UAT.

## Stage 3: Surface Integration
**Goal**: Connect Chat Workspace draft/transcript and Research Workspace's existing
handoff to the same qualification. Remove unowned cache restoration from Research
Workspace. Preserve H1 selected ancestry and native-fork authority.
**Success Criteria**: Both surfaces use the same qualified record contract;
unqualified legacy records remain untouched and unrelated chat routes unchanged.
**Tests**: Focused surface regressions plus shared H1/history/loader suites,
TypeScript, scoped lint and touched Python Bandit where applicable.
**Status**: In Progress

Both root shells now allocate path-keyed H1 controllers for the two workspace
routes without the playground reference fallback. Parent focused integration:
174 passed; shared checkpoint/surface review and complete regressions pending.
Independent checkpoint review found two P2s: a winning authorized H1 load leaves
the checkpoint lease unsettled, and staged composer edits retain a recovery UUID
for changed text. Both fixes passed an independent replacement review with 232
passing tests and 19 independent checks. Explicit workspace route selection is
under failing-first integration; cached rows never supply authority.

Latest-dev refresh on 2026-09-30: remote dev moved to
85ede1f1df10c03505c603e4183920edcb7cbfef. A read-only three-way preview identifies
one semantic collision: upstream startup receipts and owned ordering both claim
SQLite 74 / PostgreSQL 78. Preserve upstream historical migrations and advance
ordering to 75 / 79, with an in-transaction bridge for the actual ordering-only
native 74 / 78 shape. Source fast-forward is complete. All 2224 nonoverlapping
dirty/untracked paths and 64 preexisting stashes were preserved. The frozen
bridge has actual SQLite/PostgreSQL RED 20 failures then GREEN 20 passes, no
skips, and zero Bandit findings; independent review passed. Migration of a copy
of the retained native backup preserved all 198 existing tables, 49 messages,
49 ordering rows and schema objects. The restored live API migrated to version
75 with those same message/order counts. Fresh latest-dev completion/wire/digest/admission/moderation/OpenAPI
regressions: 415 passed, one inherited heartbeat skip, zero errors or failures.
Touched backend Bandit has zero findings and errors. Next and API are running
with their preserved original environments. Both retained native conversation
projections and original Chrome targets remain unchanged. The running API's
served OpenAPI passed the strict client capability gate and 12 negative controls.

The route-query freeze passed 316 regressions, but independent review identified
ROUTE-F1: same-query Back/Forward invalidates a supplied-search native lease
without scheduling a replacement load. Its one-condition repair passed 324
regressions and independent replacement review. Real Chrome then exposed a
separate same-conversation metadata-reset bug in the shared server chat loader.
Its one-guard repair reproduced one failure with 57 controls, then passed 235
regressions and full frontend TypeScript. Independent loader review passed the
235 regressions and eight additional probes against the exact frozen delta.
Remote dev advanced again to b365af1827b607fc221f9bf31ca76dde881edb4f;
the 46 upstream paths have no overlap with the dirty candidate. The safe refresh
is complete: all 1888 stopped-runtime dirty/untracked paths matched their original
hashes after restoration; all 66 preexisting stashes remain and the new retained
candidate stash makes 67. No stash was popped or dropped. Latest backend gates:
410 completion/wire/admission regressions pass with one inherited heartbeat skip;
367 RG/auth/policy regressions pass with two inherited Redis-determinism xfails.
Touched Python Bandit reports zero findings/errors. The served OpenAPI strict
client gate plus 12 negative controls pass. Both original native conversation
projections (10 and eight rows) and original Chrome targets remain unchanged.

Native Research activation exposed a real autosave feedback loop: the parent
subscribes to the session map while persistence depends on a fresh chat wrapper.
The minimum shared fix uses persisted-field dependencies in qualified and legacy
effects, preserving all qualification and restoration checks. Actual-store RED
reproduces both loops; final GREEN is 206 tests across six suites, full TypeScript
exit 0, owned lint zero errors/warnings. Independent exact-freeze review passes
those 206 tests plus eight field-change probes with matching entry/exit hashes.

## Stage 4: Review And Real Chrome Acceptance
**Goal**: Independently review the frozen diff and run actual Chrome CDP UAT with
real API/auth/database/local model/embeddings and native browser storage.
**Success Criteria**: Fresh-document reload restores qualified conversation and
draft; A/B switching retains each identity/draft without cross-context bleed.
Native failures remain explicit and no unit mock is called browser acceptance.
**Tests**: Authenticated primary Chat Workspace reload and A/B/A switching;
existing source/citation/recovery/Stop checks, screenshots and request traces.
**Status**: In Progress

Native preflight only: authenticated real Chrome page/bootstrap and an exact
draft-only New Chat/reload round trip pass using raw CDP and native storage.
Desktop 1440x1000 and mobile 390x844 screenshots are nonblank, with no horizontal
overflow and a visible composer; both were visually inspected.
The initial blank-document timing failure remains recorded separately.
Two actual native plain-send attempts dispatched zero completion requests; the
second exposes invalid_history_durable_request. The real ordinary-text formatter
produces a one-element array while the strict durable serializer requires text.
An actual formatter-to-serializer regression reproduced three failures. The
minimal exact-single-text normalization passed its 20 boundary tests, broader
108 regressions and independent review. Server-owned follow-up preparation also
needed to omit the client history prefix; that one-line repair passed 203
regressions, full TypeScript and independent review. Native plain generation and
ancestry follow-up now pass with actual Gemma, SSE admissions/results, protected
canonical rows, native checkpoint/draft reload and no automatic resend.

Native route UAT now passes fresh explicit deep link, hash navigation, actual
Back/Forward, post-navigation draft persistence, reload, malformed-route checkpoint
retention and mobile zero-overflow rendering, with zero completion dispatches.
The malformed route's collapsed Diagnostics was opened through native CDP input.
Original route failure evidence is retained separately.

Real Stop after admission, unknown-outcome reload, protected outcome inspection,
explicit reprepare and a new inference pass. Two actual dispatches retain one
canonical logical user input; the new result is verified and the earlier unknown
observation remains separate. The first Stop runner's asynchronous IDB read
failed; a corrected awaitPromise read continued that same stopped turn, without
creating another input or replacing the original failure trace.

Post-refresh b365 native evidence passes actual Gemma plain generation, protected
input/result proof, native IndexedDB offload, saved draft/fresh-document reload,
routes/Back/Forward/mobile rendering, and 11 primary/Research/A-B-A/New Chat steps.
New Chat preserves the outgoing active qualified record byte-for-byte, does not
resurrect its draft, and performs no automatic inference. Original runner failures
remain separate; corrected storage/readiness checks did not replace their traces.

Real selected-source RAG retrieves the Lumen memo with HTTP 200, but strict
SourceV1 rejects consumed media/author/character-locator metadata. The native
failure is explicit, draft/context remain in the mounted view, and the protected
eight-row conversation projection stays unchanged. The initial RAG response also
revealed actual backend answer generation before durable admission despite zero
frontend completion dispatches. Its original misleading zero-inference verdict
is explicitly corrected in a separate evidence note. A bounded shared-mode fix
now forces server-owned retrieval to send enable_generation=false and omits its
generation model/provider; the legacy path is unchanged. RED two failures/19
controls, GREEN 202 tests/eight suites plus 60 actions tests, full TypeScript exit
0, scoped lint zero errors/three inherited test warnings. The native rerun reports
generation_executed=false, generated_answer=null, zero chat completion dispatches,
retained draft and unchanged native rows. A fresh-document draft-only check also
passes with a new protected H1 read and zero retrieval/completion dispatches;
source staging correctly remains transient under the approved design.
Replacement independent retrieval-only review passes 21 exact tests and
before-shadow two expected failures/19 controls, with entry/exit hashes unchanged.
Full UAT remains open: shared SourceV1 extension approval, grounded happy
path/citation/recovery acceptance and the complete native owner/account matrix.

Additional real Chrome Settings check reached the candidate-key probe HTTP 401
and restored the original draft, transcript and unchanged eight native rows with
zero completion dispatches. Its owner-isolation wait timed out: source tracing
shows commitManualServerTransition validates before saving, so the invalid key
never changes the effective owner. This is a wrong runner assumption, not an
auth/checkpoint defect or an owner-transition acceptance pass. The earlier
startup-frame and input-acknowledgment failures remain separate; real compositor
frames and 600-second timeouts were used, with no fourth retry. Existing candidate
validation unit controls freshly pass eight/eight, separately from native UAT.
Post-check preservation confirms original conversation projections/Chrome
targets, the actual served contract, six reviewed hashes and all 67 stashes.
Stage 4 remains In Progress; use valid independent account/target transitions
for the remaining matrix rather than a candidate rejected before commit.

2026-10-01 refresh preserves all 1779 stopped-runtime paths and all 67 prior
stashes; the retained refresh stash makes 68. The only overlapping upstream file
is ChaChaNotes_DB.py; its PostgreSQL catalog query merges cleanly with the reviewed
75/79 migration bridge. API and Next authenticated readiness pass. The bounded
eight-field SourceV1 extension is now authorized. Failing-first native-fixture
checks reproduce six frontend and five backend failures; shared schema/projection
repair passes 90 frontend and 136 backend boundary checks. Broader digest,
capability, review and actual Chrome acceptance remain required.
