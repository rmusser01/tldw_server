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
