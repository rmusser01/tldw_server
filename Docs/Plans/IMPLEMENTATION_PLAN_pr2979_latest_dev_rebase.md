# PR2979 latest-dev integration — 2026-09-28

Task: TASK-13260.278.18. Resume PR-first repairs; full UAT remains paused.
ADR required: no new ADR for integration. Preserve ADR020 storage and ADR049
Chat ownership and ADR050 native/local fork lifecycle; reassess if a resolution
changes a durable architecture rule.

## Stage 1: Preserve and rebase
**Goal:** Retain b71c04cde8 and replay UAT work onto verified newest dev.
**Success Criteria:** Recovery ref exists; conflicts retain both intended behaviors.
**Tests:** Clean worktree, ancestry, range-diff and overlap review.
**Status:** Complete

Recovery ref `codex/pr2979-pre-dev-38c1455-20260928` retains the pre-rebase
work. All62 commits replayed onto dev38c1455, then the newly landed dev3d102e0d31.
Rebased head23417aebb1 has dev3d102 as its exact merge base. Stashfa97b77e
preserves the first verified repair batch and was applied without conflicts.
The second rebase retains both automatic lease validation and cancellation in
Playground, plus both isolation-ratchet and journey-provider CI contracts.

## Stage 2: Verify overlapping behavior
**Goal:** Check changed Chat, AuthNZ, Notes, providers and CI boundaries.
**Success Criteria:** Focused controls pass, including required official PostgreSQL.
**Tests:** Existing relevant suites, frontend types/lint, touched Python Bandit.
**Status:** Complete

Initial 12-file Chat integration run: 312 passed, 10 failed. TASK13260.278.18.1
tracks account-stamped local history admission and ordinary Character Retry
receipt preservation (UAT473). TASK13260.278.18.2 tracks the real PostgreSQL
Jobs diagnostic parameter and two causal test-oracle repairs (UAT474). Three
existing Jobs failures reproduce on the official PostgreSQL fixture, zero skips.
UAT474 now passes79 real-PG and30 companion checks after also correcting index
sort/null catalog validation. UAT473 Character/source-auth controls pass111;
shared account/revision review fixes continue. Historical migration modules64
pass, deployment84 pass, backend overlap182 pass (including required PostgreSQL).
After the second rebase 25 CI contracts pass. TASK13260.278.18.3 tracks API drift
(UAT475). Declared Pydantic 2.13.5 and checkout-local package sources reproduce
the upstream fingerprint; the earlier extra component came from the stale local
environment. Exactly three retained response/request components change. The
existing generation pipeline and drift gate pass, with full schemas/types kept
ignored; 79 contract checks pass, including three official PostgreSQL cases.
Final WebUI and extension types pass. Expanded UAT473 controls pass 281 across
nine shared suites and 186 across seven Character/mirror suites; these overlap.
The profile-only public-loader follow-up passes 130 across five affected suites,
zero skips. Scoped lint adds no findings; independent review is clear. Preserve
the distinction between transactional fixtures and native IndexedDB acceptance.

## Stage 3: Publish and handle review
**Goal:** Update PR2979 and resolve current CI/Qodo findings before full UAT.
**Success Criteria:** Current-base reviewable head; honest tracker and PR gate status.
**Tests:** Hosted checks and exact PR head/base; generated-artifact exclusion.
**Status:** In Progress

2026-09-29 refresh: latest fetched `dev` is `0f9e6917cef2deb5da36d6fc2f85b4457f0ce884`. The clean 121-commit PR replay is conflict-free and `git range-diff` reports 121 exact patch-equivalent matches, zero changed/added/removed commits; a recovery ref retains the prior head. The new upstream delta is limited to VZ boot-stall files with no repaired-path overlap. UAT419/UAT441 local repairs and real PostgreSQL checks are recorded in the tracker; the rebased head is published as `2d033d9993`, with hosted review and later local fixes still open.

Previous source checkpoint `00fed1c6af` is published on dev `3d102e0d31`.
The original publication had zero missing dev commits. Current verified local
VAD and Character repairs were committed965ea81fc0/f435078aac before the
clean100-commit replay onto devca3b7f834a. Rebased checkpointdbbf5ad1fde7
has0missingdev, exact6upstream path changes and all13frozen qualification hashes
retained; upstream AuthNZ cause-chain/privacy controls31pass with normal exit.
The PR body will be refreshed with the settled scope and excluded artifacts.
Fresh hosted CI/Qodo and the requester-owned Change summary remain required;
the prior 14 review threads are resolved. Fresh Qodo on head00fed1c6af reports
zero bugs and retains only the two previously source-dispositioned recommendations.
Fresh Playground accessibility CI reproduces four H1 fixture ownership failures;
TASK13260.278.18.4 / UAT476 tracks the bounded test-only repair. Additional
frontend shards and AuthNZ route binding require causal attribution under UAT419.

Bounded new-head repairs: UAT476 all three Playground quality scripts pass and
UAT477 exact route/header/owner controls pass; both source units are committed.
UAT478 fixes obsolete Sidepanel/mirror fixtures with228 passes, preserving real
ownership. UAT479 aligns Sync fixture clocks and restores a shadowed test while
retaining production expiry:343 full-module passes, zero skips; final narrow
constraint oracle has31 overlapping passes, clean Ruff and unchanged Bandit.
UAT480 supplies and persists the captured account stamp for new local Chat
creation without relaxing public import sanitation;134 production-boundary
fixture controls and2 first-attempt native browser journeys pass. UAT481 preserves native
Character provider recovery and updates exact canonical acceptance. UAT482
repairs the rejected Settings-return loader callback loop with stable existing
H1 methods/getter and setter wrappers, keeping account and selection fences;
156 source checks pass in both implementation and independent review. Normal
whole WebUI/extension types pass for these integrated repairs. UAT483 reuses the
existing JSON SSE emitter for Character provider failures; UAT484 preserves all
privacy assertions while updating two exact normalized logger tokens. Combined
streaming21 pass, zero skips, independent review clear, Ruff clean and unchanged
test-only Bandit findings. Auth tenant/profile fixtures use current gateways;
six modules138 pass with all original assertions retained. Protected health/MFA/
resend and remaining database/runtime failures are separately tracked and traced.
Historical Character/PG fixture follow-up TASK18.6 proves exact old migrations
before current APIs, keeps specific RLS denials and full old-row/new-default
checks. Final five-module84 pass, including28 actual PostgreSQL, zero skips;
independent review clear, Ruff clean, Bandit348 assertions only and no errors.
Fresh CI attribution continues before publication/merge/UAT.

Further current-base units: UAT481122 native controls and both whole typechecks
pass; all three bounded Character cases are verified with two earlier Phase7
harness failures retained. Four cockpit sends pass first attempts with native
backend/provider completion calls. Native keyboard failure is traced to palette
host replacement on hideHeader transitions (UAT490/TASK18.25), with loader-focus
observation still requiring causal attribution. UAT485 uses existing borrowed
schema checkout binding;145 normal controls pass. UAT486 releases only owned
nonmemory SQLite bootstrap checkouts;74 unique owner/lifecycle controls pass.
UAT487's two native URL callers reuse the existing initialized validator;69
security/quota/egress controls pass, no skips. Read-only UAT488 has9 direct
passes; its earlier broader127 accounting is126pass/one independent deadline
failure, retaining the original interrupted run and disjoint38pass follow-up.
UAT491 reuses the existing30s acquisition budget while retaining5s DDL and30s
statement limits. Final integrated133/133 pass, zero failures/errors/skips,
including all72 PostgreSQL bootstrap cases and finite held-lock invalidation/
reuse controls. Root reviewed and committed485/486/488/491 as40ac75b9ed.
UAT489 factory retirement passes47 controls with central ownership, runtime
rejection and committed-data reuse intact; independently reviewed, committed
e88d29142f. UAT490 stable palette passes68 controls and unchanged native
keyboard first attempt1/1, zero retries/skips; normal types pass with Web8GB,
retaining the default4GB OOM. Reviewed and committed5b340ea0b1. Actual loader
focus steal UAT494/TASK18.30 is independently verified and committed2ca5c6929f:
only the dashboard fallback opts out;129 controls and unchanged native keyboard
first attempt pass without skips/retries/flakiness. Normal types/lint/build pass.

Further actual PostgreSQL causes: UAT492/TASK18.27 preserves only a static
exact saved-view uniqueness subtype without driver payload; recovery lifetime
and all account/RLS/rollback guards remain. UAT493/TASK18.28 types five
standalone nullable identity parameters in two shared Sync queries, retaining
all filters/locks/expiry. Historical saved-view checkpoint TASK18.29 tests
existing exact53→54 branch over its actual synthetic seed without replaying
later migrations on mislabeled current tables. Production edits were held
until133 qualification and common-source review/commit, then released. Final
saved-view/backend/API152 and Sync/blob/quota/expiry246 controls total398 pass,
zero failures/errors/skips, including mandatory actual PostgreSQL. Root review,
compile/lint and security comparison clear; committed5badebd487/9818eb292a.
RAG TASK18.31 has two native reds: its barrier omits current max_bytes keyword;
bounded trace proves TypeError before the event. The existing keyword now reaches
the real secure reader, preserving deadlines/result and every assertion. Full22
controls pass without skips, independent review clear, committede4e2364b7b.
Slides hosted concurrency remains unconfirmed after three bounded native probes
(macOS, actual connection barrier, Linux3.12); stop/reassess rather than equate
those passes with CI acceptance. Media15 original failures all reproduce
natively without skips. UAT495/TASK18.32 repairs fixed-window VAD framing and
detector-local TorchScript recovery shared by three callers, preserving ASR bytes,
ONNX, rates and timing guards. Native speech also confirmed UAT496/TASK18.39:
ongoing None events were incorrectly classified as silence despite triggered=True.
The shared boolean state now prevents premature commits. The inherited actual-VAD
control now reuses tracked speech through the existing guarded converter and a
nonzero deterministic clock, retaining exact thresholds/guards. Final frozen64
checks pass with no failures/errors/skips, including all three unchanged native
streaming tests. Real-model timeline commits once after silence, never while
active. Both independent reviews clear; production Bandit0 and compile/lint
comparison clear. Earlier54pass/1invalid-fixturefail qualification is retained.
TASK18.33–.35 upload/config/HTML fixture repairs are independently reviewed and
committed5a8274a16b/e894435942/9b1b029fae; exact success/warning/content assertions
remain. Recorded module gates30pass/7skip/1xpass,147pass/5skip,64pass/8skip/1xpass
retain inherited skips as unexecuted; URL security127pass/1inheritedxfail.
Windows10 child rc1 failures still need native A/B confirmation: TASK18.36 adds
a temporary bounded diagnostic to the existing Windows3.12 shard, committed
61811b0819; local148CI/admission controls pass. It withholds child output, uses
strict fixed-field artifacts and never inherits operator secrets. Native results
and diagnostic removal remain required. TASK18.37 UTF-8 vector reads reproduce
21fail/70pass under cp1252 and pass91 after seven explicit encodings, with all
assertions retained; independently reviewed and committed f13ff16ee7.
TASK18.38/.40 Character mocks/settings are frozen with62passing controls and
zero failures/errors/skips; seeded/plain settings and later deletion remain real,
all production files unchanged. Initial hosted failure was settings line134,
not deletion. Reviews, compile/lint and security comparison clear.
Latestdevca3b7f834abc10ba0889caeae868afe9d290009b is fully included after
clean100-commit replay, checkpointdbbf5ad1fde7. Exact6upstream diagnostic paths
changed, all13frozen VAD/Character qualification hashes retained;31upstream
AuthNZ privacy/cause controls pass,0fail/error/skip and normal exit.
Final native repairs are reviewed and verified: UAT497/TASK18.41 initializes
only the shared safe HTTP success URL before optional metadata; UAT499/TASK18.46
uses native INFO/WARNING level names in both existing branches. All callers and
privacy/egress/retry boundaries remain. TASK18.48 scopes actual stdlib HTTPX INFO
capture in the existing concurrent test and restores its handler/level. Final
HTTP307 and provider41 pass with0fail/error/skip and normal exits; all130 original
assertions and concurrency barriers/deadlines/instrumentation checks remain.
Compile passes; Ruff retains7 baseline findings, Bandit adds34 test assertions
only with0production findings. Frozen hashes and root/independent review clear.
Earlier4/9 causal reds, failed logging observation and297/1 gate are retained.
Tasks18.41/.46/.48 Done, committed13f50abb00; these qualifications overlap.

Fixture43 canonical seeds82pass and44 precise detached error65pass, all normal
exits/zero skips, all original14/170asserts retained plus2privacy controls;
production unchanged. Reviewed, Done and committed013f19b825. Fixture42 adds
three traced Notes/webhook inventory names and47 one automation scheduler
mapping. Final21-module lifecycle/ownership gate370passes0fail/error/skip and
normalexit; all72originalasserts AST-identical, all23productionpaths unchanged,
all44frozenhashes match. Compile/Ruff pass, Bandit unchanged72testasserts. Root
review clear;18.42/.47 Done, committed3c1a588f22. Earlier369/1 and original failures remain recorded.

UAT498/TASK18.45 spaces only two shared PostgreSQL INSERT parameter lists after
managed guard parser rejection reproduced before I/O. Caller auth/scopes,
bound values, conflicts, SQL guards and transactions remain. Two focused guard
reds become4causal passes including2actualPostgreSQL; final9modules198pass with
0fail/error/skip and normalexit. Original15unitasserts retained, compile/Ruff
pass, Bandit adds2testasserts with0source/errors. Root/independent review clear;
Done and committedc5bd693404. Tracker499/471/28 describes bounded engineering
scopes; hosted Windows, current-head CI/Qodo, native diagnostic confirmation
and removal, and requester-owned final Change summary remain merge gates.
Full UAT remains paused.

Fixture-only follow-ups retain original guards/oracles: protected health/MFA/
resend295 passes; canonical Privilege seeds42 plus seven overlapping controls;
worker/redaction72 including24 actual PostgreSQL, final nine overlapping;
offline Email41 including native22 with every38assertion and tripwire unchanged.
Current exact inventory refresh363 passes with independently attributed source
deltas; new pure coercion bootstrap dependency remains frozen alongside all
unchanged discovery/network mutation controls;348 full frozen HTTP/coercion
dependency checks also pass without skips, independently reviewed and committed
f30e8257e7. Latestdevca3b7f834a is included,0missing at the verified replay.
Full UAT remains paused while
these repairs, remaining CI causes, publication and current-head merge gates
finish. Captures, runtime databases and private diagnostic artifacts stay ignored.


## Latest dev replay and integration controls — 2026-09-29 (TASK-13260.278.18.83.5/.6)

### Stage 1: Preserve and reconcile
**Goal:** Replay all 168 commits onto current dev while retaining canonical writers, privacy, ownership and migration contracts.
**Success Criteria:** Recovery refs retained; no skipped commits; latest upstream delta adopted.
**Tests:** Conflict review, ancestry and range-diff.
**Status:** Complete

Reconciled onto `5910412f`, then replayed all 168 patch-equivalent commits onto `60006a2fed`. Recovery refs preserve `4c33d51d` and the first replay `b791876b`; integration checkpoint `adab81ce26` contains both rebases. Existing audio cleanup and Persona activation helpers are reused; typed unique errors and atomic transactions preserve Watchlists defaults. Genuine historical migration schemas and independent ctime/payload extraction guards remain. No new ADR is required: ADR020/049/050 continue to govern storage and ownership.

### Stage 2: Repair combined integration artifacts
**Goal:** Remove merge artifacts without changing product contracts.
**Success Criteria:** Changed Python compiles; no added Ruff diagnostics; exact CI coverage and affected SQLite/real PostgreSQL controls pass.
**Tests:** TTS duplicate-keyword compilation, fixture bindings, genuine v57 graph backfill, official PostgreSQL migrations/keys, archive readiness and exact CI inventory.
**Status:** Complete

UAT516 repairs duplicate declarations and stale test bindings. The first SQLite/CI run passes 133 with two contract failures; the reconciled CI module passes 49 and Notes scope passes 25. Current dev's 3.12 preflight/event policy is retained. The coverage guard covers 825 shards/4,879 files with zero new omissions. All 269 changed Python files compile; package-configured Ruff adds zero diagnostics (298 inherited). Production Bandit retains 20 findings across 58 changed files, zero added; the repair tests add one LOW historical-schema assertion only. Actionlint and diff checks pass.

UAT517 removes duplicate DESC encoding from the shared PostgreSQL archive-index name query while retaining exact sort/null bits and all integrity checks. Four real failures become a full 62-case green coordination run with zero skips; independent caller review is clear. Other focused gates: AuthNZ/migrations/keys/Watchlist 102, Character/Persona 59, Audio/MCP/TTS/gateway 100, deterministic VAD 32 and cancellation/heartbeat/health 11 pass. Python 3.12 preflight characterization passes 19 with one optional curl-cffi skip; dev already fixed version-sensitive signatures.

### Stage 3: Publish and qualify
**Goal:** Publish the verified batch and handle final-head CI/Qodo before merge.
**Success Criteria:** Final head tested, actionable review resolved, protected merge verified on dev.
**Tests:** Fresh remote head/base, automatic PR CI, existing manually dispatched native matrix and Qodo review.
**Status:** In Progress

Old head `60eedcdc` terminal results (800 pass/14 fail/34 cancelled/30 skipped) do not qualify the new branch. Optional real VAD remains locally unverified because torchaudio is absent. Frontend type checking retains the same 23 baseline diagnostics; extension type checking passes. Focused frontend suites pass 131 and fail 40 legacy render cases due to duplicate local React installations; three resolver probes confirm the environment split. No product workaround, assertion relaxation or test suppression is added. Require fresh hosted installs and native matrix completion. Full fresh-install UAT remains paused.

## FastAPI route audit follow-up — UAT518 (TASK-13260.278.18.83.7)

### Stage 1: Reproduce the upstream gap
**Goal:** Establish whether included routes escape the pagination audit under dev's FastAPI version.
**Success Criteria:** The same missing-pagination endpoint is detected at root but missed when nested on 0.141.1.
**Tests:** Two-level router probe on FastAPI 0.136.3 and 0.141.1; durable existing model-collision controls expanded to nested routes.
**Status:** Complete

Private probe confirms a false pass only for the nested 0.141.1 route. Local all-169-commit replay onto dev `607431154c` succeeds with recovery ref preserving published `e49fab`; candidate `bef6421cd95dfeb9605df00896d5e47bb4382b64` remains unpushed. Published diagnostic CI continues unchanged.

### Stage 2: Reuse the served-route walker
**Goal:** Make the existing audit inspect every public schema using dev's shared route traversal.
**Success Criteria:** Existing exact endpoint/model/property expectations detect the same missing pagination through root and nested routers.
**Tests:** Durable red/green on FastAPI 0.141.1, full pagination module, shared route traversal and auth-ratchet controls.
**Status:** Complete

Keep APIRoute metadata, use served full paths for OpenAPI lookup, and preserve native path-format normalization. Remove the unused duplicate endpoint walker. No production schema or assertion is relaxed. Durable red has two nested failures/eight passes; the final actual-FastAPI route/pagination/auth/coverage scope passes 46 cases, zero skips. Independent review is clear.

### Stage 3: Qualify and publish
**Goal:** Verify the latest-dev candidate while preserving current remote diagnostics, then publish a single tested batch.
**Success Criteria:** Compilation/lint/Bandit clear; affected application contracts pass; final head gets hosted and Qodo acceptance before merge.
**Tests:** Focused route/API contracts with actual 0.141.1, ancestry/range-diff, protected CI and review gates.
**Status:** In Progress

Private dependency overlay leaves the shared 0.136.3 environment unchanged. Full UAT remains paused. Current remote head is `e49fab`, with final newer-dev acceptance still required.

Latest-dev HTTP qualification passes all 68 PostgreSQL lifecycle/media controls with required official fixtures, zero skips and normal exit. The 269 changed Python files compile, Ruff passes, and the route test retains five baseline LOW Bandit assertions. Fresh Qodo e49 review adds image-memory, async Character reads, Sidepanel subscription/storage, missing handoff locales and downstream Chatbook capability findings; tasks .83.8-.11 track the server repair units before publication.

## Fresh Qodo e49 repair batch — 2026-09-30 (TASK-13260.278.18.83.8-.12)

### Stage 1: Trace each finding
**Goal:** Verify new findings against current source and actual callers before repairs.
**Success Criteria:** Shared pixel budget, all Character read callers, UI subscribers, locale namespaces and current downstream schema traced.
**Tests:** Compact PNG decoding red, route worker ownership red, real-store rerender red, missing locale contract and exact Chatbook typed-model probe.
**Status:** Complete

The saved-image helper lacks a decoded pixel guard despite compressed byte limits; reuse existing 16MP cap. Three async Character callers missed the existing fourth caller's threadpool pattern. Sidepanel whole-store subscription and five absent handoff locale keys are confirmed. Current Chatbook main still lacks the optional field and drops it on typed spec resave; this downstream gap remains open until a companion repair or accepted scope disposition.

### Stage 2: Apply minimal existing patterns
**Goal:** Repair the shared causes without changing account, image or failure semantics.
**Success Criteria:** Relevant controls pass, official PostgreSQL lifecycle retained, no new production Ruff/Bandit findings.
**Tests:** Shared saved-image limit, complete Character images/operation release, Sidepanel readiness/resume and supported locale/mirror contracts.
**Status:** In Progress

UAT521 passes 33 native UI controls after a causal render failure; strict-equality lint passes. Existing 8GB frontend type script retains 23 source baseline diagnostics plus a sandbox ignored-cache write error; no changed-file diagnostic is reported. Bandit is inapplicable to TS/TSX. UAT519 compactPNG red becomes three size-control passes with clean Ruff and no production Bandit findings; full Character SQLite/PostgreSQL controls and independent review remain running. UAT523 companion patch is concrete: exact client schema red fourfail/sixpass, patched/replayed tenpass, zero production Bandit findings. Separate-repository scope is awaiting requester clarification; no client checkout or PR changed. UAT520 final Character module passes108 with required official PostgreSQL and no skips, including all12offload/operation-return cases. UAT522 adds exactlyfivekeys to eachof18source/21publiclocales; focusedlocalecontracts3pass, officialsyncdryrun0writes, priorcontent preserved. Independent UAT519 review reproduces eager ICO/GIF/APNG allocation before the first guard; refined CORE metadata bound is in progress before acceptance. Release priority is explicit: PR2979 blocks the next release/private beta, carries release-blocker label, and dev has no configured merge queue. Protected final-head qualification remains required.

### Stage 3: Review, publish and qualify
**Goal:** Publish the reviewed batch on newest dev and settle every applicable Qodo finding before merge.
**Success Criteria:** Source/evidence replies, honest downstream disposition, final-head automatic and native matrix accepted and merge landed on dev.
**Tests:** Independent scoped review, recovery/lease verification, hosted checks and exact final-head review.
**Status:** Not Started

Published diagnostic head remains e49fab; local UAT518 is committed7920983dbc on dev607431. Full UAT stays paused and generated captures remain excluded.

## Native e49 matrix follow-up — UAT524–529 (TASK-13260.278.18.83.13–.16)

### Stage 1: Diagnose native failures
**Goal:** Trace every new native failure to its actual API, fixture or platform cause.
**Success Criteria:** Exact failure evidence retained; tests and production contracts stay active.
**Tests:** Nine failed job logs, actual PostgreSQL binding red, Windows-like fixture byte/capability probes and real SQLite handle lifetime.
**Status:** Complete

### Stage 2: Apply bounded shared repairs
**Goal:** Repair verified causes using existing writers, transports, fixture lifetimes and platform security patterns.
**Success Criteria:** Causal regressions and adjacent modules pass; no added production lint/security findings.
**Tests:** Full auth/database bindings, exact Research fixture hashes/tripwires, Media processing/config and handle release, secure legacy artifact controls.
**Status:** In Progress

UAT524–526 preserves committed-version and owner/SQL assertions; exact four failures become258 full-module passes with required official PostgreSQL and zero skips. Research and Media repairs are delegated in separate scopes. Windows legacy private-artifact semantics require further diagnosis. The final UAT519 refinement passes73 affected controls and75 independent controls, with zero production Bandit findings and a clear independent review; initial failed guards remain recorded. Local Character, Sidepanel and locale repairs are committed868310db45,84a6920747 and80f1699052. Downstream UAT523 companion schema patch is concrete, but separate-repository scope remains pending requester clarification.

### Stage 3: Review and publish
**Goal:** Publish one reviewed batch on latest dev after current useful diagnostics settle.
**Success Criteria:** Recovery/lease checks, final automatic and native matrix, exact Qodo disposition, protected merge verified on dev.
**Tests:** Current-head hosted gates and native matrix; all newly actionable findings.
**Status:** Not Started

Published e49 diagnostics have388 native passes,9failures and384 queued/running jobs at the latest saved snapshot. These results do not qualify newer local dev607431. PR2979 is a labelled release blocker; no dev merge queue exists. Merge immediately when gates pass; full UAT remains paused.

Latest local repair commits are d7ab76c6d4 (UAT519),7bf152e251 (UAT524–526) and0b2d9c91ac (UAT527). The image scope has a clear independent75-case review; the complete auth/database scope passes258 and Research passes1615, all with zero skips. Native e49 diagnostics now have513pass/14fail/254queued-or-running plus2configuredskip; four additional failures repeat existing causes, one new UAT530 is moderation temporary DB ownership. UAT528 releases its fixture's main-thread handle before client shutdown to prevent backend eviction hiding it from final reset; strict actual pytest lifetime acceptance continues. UAT529 applies the requester-approved POSIX artifact boundary with native negative coverage and individual supported-platform positives. Child17 tracks UAT530. No final-head publication or hosted acceptance is claimed.
