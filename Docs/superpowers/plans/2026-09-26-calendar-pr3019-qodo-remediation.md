# Calendar PR #3019 Qodo Remediation

**Goal:** Address all original and follow-up Qodo findings, verify the rebased Calendar module, and merge PR #3019 only when its required gates are satisfied.
**Architecture:** Retain the SQLite repository, existing calendar permission model, read-only CalDAV adapter, bounded Jobs polling, and thin frontend. Share credential resolution and offload blocking provider operations. Preserve recurrence masters and exception identities rather than collapsing a series into one row.
**Tech Stack:** FastAPI, SQLite, HTTPX, AnyIO, icalendar, python-dateutil, pytest, Next.js, Vitest.
**Spec:** `Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md`.
**Tracking:** TASK-13356; https://github.com/rmusser01/tldw_server/pull/3019.

## Global Constraints

- Work only in `.worktrees/calendar-dev-pr`; preserve unrelated user changes.
- Write regression tests and observe their failures before implementation.
- Keep CalDAV imports read-only, same-origin, HTTPS-only, and credential scoped.
- Never log credentials or raw provider exception messages. Log safe operation context and traceback frame locations.
- Keep database SQL in `Calendar_DB.py`; make item/recurrence writes atomic.
- Do not invent a human Change summary or bypass required checks or merge policy.
- Reply to each Qodo thread with the verified disposition and commit; obtain a fresh review after fixes.
- The human requester replaced the fresh Qodo gate with the
  `requesting-code-review` independent reviewer workflow after Qodo exhausted
  workspace credits. Historical Qodo dispositions remain evidence; no new Qodo
  request or paid credit purchase is required.

## Approved CI Follow-Up (2026-10-20)

ADR required: no. This restores the existing core import direction, isolated
AuthNZ test setup and lifecycle inventory contracts (Docs/Architecture.md and
Docs/ADR/021-services-lifecycle-startup-and-shutdown.md); it changes no API fields.

1. **Current-dev integration (Complete):** Rebase onto
   `e1ca22f2986554f85a7f9473104bfd3e984b95ae`, retaining upstream quota and
   Calendar SQLite RLS exemptions. Preserve the dirty publication note and
   Calendar shard coverage; normalize both PR-owned task records with backlog-py.
2. **Four shard fixes (Complete):** Move unchanged scheduled-task models into
   core with compatible API exports; use the isolated fixture's direct PostgreSQL
   setup connection; filter privilege pagination to a seeded resource; include
   Calendar workers in fixed inventories/factory mocks. Reproduce before fixes.
3. **Local verification (Complete):** Run Python 3.12 Calendar, scheduled-task API,
   AuthNZ PostgreSQL, privilege and startup tests; frontend/TypeScript, applicable
   CI contracts, shard guard, Ruff, Bandit and normal hooks. Record actual results.
4. **Publication (In Progress):** Push with an explicit lease against the verified
   remote head, inspect CI on the published commit and complete a full independent
   review via `requesting-code-review`. Provider smoke and human-authored
   what-and-why rationale remain gates.

Local Python 3.12 evidence: Calendar/scheduled-task/permission unit scope 585 pass;
real PostgreSQL and privilege scope 49 pass; worker/startup scope 87 pass;
CI workflow/task/RLS contracts 115 pass; additional required ratchets 33 pass and
the full test-quality ratchet passes after strengthening existing assertions and
using real isolated SQLite quota persistence. French/Paris frontend 128 pass,
TypeScript and canonical OpenAPI checks pass. Bandit reports zero findings/errors;
shard guard reports 5,145 files with no newly uncovered files. No test-quality
baseline or guard was weakened. Hosted exact-head checks remain mandatory.
Final real-quota API/startup rerun: 144 pass. Normal hooks pass over the entire
PR merge-base diff, including the new shared models and canonical task records.

Published candidate: `2f73fcc8d62d6cf2c454db9aef9c23228768585d`, verified on PR3019
after an exact old-head lease push. GitHub reports MERGEABLE against current dev
`e1ca22f2986554f85a7f9473104bfd3e984b95ae`. Fresh CI run 39523251502 and required
gate runs are queued; trusted-license audit 39523304079 is queued with no steps
started. No hosted pass is claimed. Validation-only PR-body refresh preserved all
other bytes. Full Qodo request 6305157939 received billing-blocked reply
6305159034: workspace credits exhausted, so no full review ran. Existing provider
smoke and human-authored what-and-why summary gates remain unwaived. No merge.

Human-authorized review replacement: one fresh, read-only general-purpose
reviewer is reviewing the entire base `e1ca22f2986554f85a7f9473104bfd3e984b95ae`
through head `2f73fcc8d62d6cf2c454db9aef9c23228768585d` range using the
requesting-code-review template, without inheriting implementation history.
Reviewer agent: `01a15bee-211d-70c1-b099-27feb5c10a1f`. Full review completed;
Qodo credit exhaustion no longer blocks merge under the updated human instruction.

### Independent Review Result

Review: https://github.com/rmusser01/tldw_server/pull/3019#pullrequestreview-5593667681
Exact head: `2f73fcc8d62d6cf2c454db9aef9c23228768585d`.
Verdict: **With fixes**, no verified Critical findings, five Important/P2 findings:

1. `CalendarItemDrawer.tsx:34`: displayed original wall times lose their offset
   on edits when item timezone metadata exists, shifting instants or failing
   start/end validation. Keep display and serialization in the same timezone.
2. `calendar_sync_worker.py:326` / `Calendar_DB.py:2117`: failures leave the oldest
   due timestamp unchanged, so repeated bounded scans can starve later bindings.
   Reschedule failed attempts without weakening active-job deduplication.
3. `CalendarItemDrawer.tsx:135` / `calendar.py:667`: persisted annotations have no
   product read path. Expose the existing authorized list operation and render
   saved notes when reopening the item.
4. `Calendar_DB.py:1863`: revoked accounts retain references to wiped secrets;
   subsequent deletion fails and rolls back. Make scoped revocation/deletion
   cleanup composable while retaining ownership checks.
5. `view_service.py:261`: scheduled-source partial/errors are discarded; a source
   outage is shown as a complete empty agenda. Propagate degradation warnings.

The coordinator checked the cited implementations and published five inline
comments. Reviewer reports 412 focused Python 3.12 tests passing, zero Bandit
findings/errors across 19 Calendar runtime files, and clean whitespace checks.
Evidence detail: 299 tests plus 112 tests passed under an isolated `--noconftest`
harness; one multiprocessing harness invocation could not reopen `<stdin>` and
passed when rerun through the pytest module entrypoint (412 distinct passes).
This is focused review evidence, not a canonical hosted-suite result. Saved
security report: `/private/tmp/calendar_pr3019_review_bandit.json`.
No source approval or fixes for these five findings are claimed. No production
files changed in this review-only turn. Hosted CI, provider smoke and human
what-and-why summary remain separate, unwaived gates.

## Approved Independent-Review Remediation (2026-10-20)

Human approval covers all five P2 findings, regression tests, fresh CI and a new
requesting-code-review pass. ADR required: no; these restore existing Calendar
timezone, polling, authorized commentary and degraded-result contracts.

1. **Regressions (Complete):** Ten backend cases and three frontend cases failed
   against the published head for the expected five reported defects.
2. **Scoped fixes (Complete):** Normalize editor display into the item timezone;
   reuse polling cadence after failures; expose authorized annotation listing
   without private tag overlays; atomically clear revoked secret references;
   propagate partial scheduled-source warnings while preserving loaded items.
3. **Local verification (Complete):** Full Calendar and scheduled-task suites,
   French/Paris frontend, TypeScript, API/OpenAPI/route contracts, quality ratchets,
   Bandit and normal hooks. Preserve existing DST and authority regressions.
4. **Publication and review (In Progress):** Lease-protected push, exact-head CI,
   individual evidence replies, fresh independent full requesting-code-review.
   Provider smoke and human-owned rationale gates remain unchanged.

Integrated current dev `721bb583e9090d3794d43775ff33df8f425b3259` (PR3292),
with no conflicts and all 23 Calendar commits range-diff equivalent. Combined
Calendar/scheduled-task/required workflow/task/quality suite: **675 pass** on
Python 3.12. French/Paris frontend: **133 pass**, TypeScript pass. Shared
route-auth/privilege/import boundaries: **63 pass**; required real PostgreSQL
migration fixture: **2 pass**, no skips. Canonical OpenAPI/path union and normal
full-PR hooks pass; Ruff pass; Bandit zero findings/errors; shard coverage 5,145
files with zero newly uncovered files. These are local results, not hosted CI
or independent source approval.

### Independent Re-review Follow-Up

Full re-review of `8096e315836dc64c49eac7394d85fa4370d91072`:
https://github.com/rmusser01/tldw_server/pull/3019#pullrequestreview-5593885698
IR2-IR5 are independently verified closed. IR1 needed preservation of the second
repeated-hour fold. The reviewer also reproduced delegated Tasks permission
bypass (P1), cross-connection Calendar state reuse (P2), and null PATCH required
fields returning 500 (P3). These block technical source approval until fixed.

1. **Regressions (Complete):** Ten backend cases and three connection-switch
   cases reproduced the gaps; positive authorized cases also remain covered.
2. **Repairs (Complete):** Reuse canonical Tasks permission checks, retain local
   Calendar reads independently, and enforce Tasks control quota for delegated
   creation. Key workspace/caches/drafts to the existing credential-free
   connection authority; retire old callbacks/multi-step mutations on unmount.
   Preserve original fold in backend temporal edits without freezing later-date
   DST offsets. Reject explicit null for nonnullable PATCH fields.
3. **Verification (Complete):** Calendar/scheduled-task/test-quality592 pass;
   French/Paris frontend140 pass; route/workflow/task/import contracts80 pass;
   TypeScript, canonical OpenAPI and Ruff pass. Shared authority23 pass;
   Bandit zero findings/errors; normal full-PR hooks pass.
4. **Publication and exact-head review (In Progress):** Published and verified
   `d7302b0103160084f74b2e0125cc45816ee64e78` with the exact 8096 old-head lease.
   Fresh full CI `39528241681` and backend-required `39528241457` are queued;
   trusted license audit `39528570121` is queued. Independent full reviewer
   `01a15c19-9ce9-7041-a689-c899c0c1dc71` checked the entire exact-head PR;
   the final follow-up below records its verdict and remaining repairs.
   Individual evidence replies cover the remaining fold and new review findings.
   No hosted success, live-provider smoke, human rationale or source approval is
   claimed by these local results. Publication tracking stays local until the
   candidate is settled, avoiding a tracking-only push that invalidates CI.

### Final Review Admission and Binding Validation

Full independent review of `d7302b0103160084f74b2e0125cc45816ee64e78` verified
all prior repairs on the tested paths, but reproduced pre-dispatch quota
starvation (P2) and required binding PATCH nulls returning 500 (P3):
https://github.com/rmusser01/tldw_server/pull/3019#pullrequestreview-5594003361

1. **Regressions (Complete):** Four failures reproduced before implementation:
   a real Jobs quota with 100 older blocked bindings hides a new owner's binding,
   and three required binding fields accept null. The manual-only null interval
   positive control passes. Preserve reservations, owner quotas and deduplication.
2. **Repairs (Complete):** Reuse the existing scan cadence after failed scheduled
   admission; retain durable pending admissions and continue safely if backoff
   persistence fails. Apply the existing PATCH explicit-null validation pattern
   only to required binding policy fields. No new ADR or schema migration needed.
3. **Verification (Complete):** Seven focused regressions pass. Full Calendar,
   scheduled-task and test-quality scope: 598 pass; workflow/task/route/import
   boundaries: 80 pass. OpenAPI, Ruff, Bandit zero findings/errors, whitespace
   and normal full-PR hooks pass. No reservation or ratchet guard weakened.
   Frontend source is unchanged from the independently verified 140 Calendar and
   23 shared-authority passes; avoid repeating unchanged suites.
4. **Publication and closure review (In Progress):** Commit with normal hooks,
   push against the exact published d730 lease, and request independent review of
   the final delta alongside the unchanged full-range review. Hosted CI and the
   unwaived live-provider/human-rationale gates remain required.

Published `59ee290323998ffae380fa6f296fa20d5dd87d72`. Full independent review
verified the preceding repairs but reproduced polling-interval overflow below.
CI `39529871149`, backend `39529871130` and license `39529948369` remain queued;
these runs do not establish hosted success or source approval.

### Polling Interval Overflow

1. **Regressions (Complete):** Eight failing cases and three positive controls
   cover create/PATCH bounds, legacy persisted integers, UTC limit arithmetic,
   and a two-owner real-quota periodic runner. All eleven now pass.
2. **Repair (Complete):** Bound new intervals to 3,700 days, matching existing
   binding scan-policy bounds. The shared next-scan helper saturates dates beyond
   the UTC datetime range, preserving legacy values without terminating success,
   failure bookkeeping or the shared scheduler. No migration or new ADR needed.
3. **Verification (Complete):** Calendar/scheduled-task/test-quality 609 pass;
   boundary contracts 80 pass; OpenAPI, Ruff, normal hooks and whitespace pass.
   Bandit on three changed sources: zero findings/errors. Unchanged frontend
   retains independent evidence.
4. **Source closure and publication (Complete):** Published and remote-verified
   `5587463288310b9aff99d8e639280e0dde98988a` on current dev `721bb583`.
   The full 87-file review at 59ee plus exact-delta/caller closure reports no
   actionable findings. Independent 22 focused tests and unmodified periodic-runner
   real-quota probes pass, including huge legacy intervals. Reviewer closed;
   public technical review `5594091981`, all 76 review threads resolved.

Overall Stage 4 remains **In Progress**. Fresh exact-head full CI `39531103973`,
backend-required `39531103960` and trusted license `39531212018` are queued, not
passing. PR is open/current/MERGEABLE but BLOCKED; no merge attempted. All seven
hosted contexts, current-dev recheck, live CalDAV smoke or allowed explicit waiver,
and human-authored what-and-why summary remain required. Heartbeat stays paused.
These publication notes stay local to avoid invalidating the reviewed CI candidate
with a tracking-only push; preserve them on the next authorized continuation.

### CI Watch Follow-Up (2026-10-23)

The human authorized watching CI, fixing failures and notification when green.
On exact `5587463`, all seven required contexts and the full Python 3.12 aggregate
passed. Two auxiliary jobs failed: critical E2E `39531103803` and shared-UI
coverage shard 3 in `39531103910`. This is a CI-only follow-up, not a merge or
waiver of provider smoke/human rationale. The existing watcher is active again.

1. **Reproduce (Complete):** Retained trace proves PII filtering changed numeric
   UUID groups in the fixture's title/body, breaking literal citation membership.
   Two deterministic fixture-label regressions failed; 51 positive/existing tests
   passed. The admin retry test clicked while AntD still held its loading state.
2. **Repair and local verification (Complete):** Map UUID digits injectively to
   alphabetic characters; retain all source/citation assertions and security
   filters. Wait for a usable refresh button before the admin test's retry.
   Grounding 53 pass; real RAG policy probe preserves repaired content exactly;
   shared-UI admin 23 pass under coverage; TypeScript pass. Bandit is not applicable
   to this TypeScript/test-only scope; no Python application code changed.
3. **Review and publish (Complete):** Independent review has no findings; exact
   `14895fcd4630ba9278d544e5e9429e6398c10f41` is published with the protected 558
   old-head lease. Review `5632130399` records independent reversible-label,
   random-UUID, real RAG-policy and AntD click-state probes. Reviewer closed.
4. **Hosted CI watch (In Progress):** Fresh CI `39996031207`, backend
   `39996031401`, frontend `39996031142`, E2E tiers `39996031083` and license
   `39996076939` are queued, not green. Existing watcher is active with the fresh
   checkpoint. Do not continually rebase for this CI-only watch. Notify only
   actionable changes or verified all-green; stop the watcher when green.

#### Matrix Revision Mismatch (2026-10-23 20:33Z)

All seven required contexts on exact `14895fc` are SUCCESS, and critical E2E
`39996031083` passed. Full CI `39996031207` failed `admin-bundle-ops` before
collection: pytest exit 4, `test_bundle_restore_backends.py` not found. Job
`120605922220` checked out candidate `14895fc`, but GitHub evaluated the workflow
from synthetic merge `205d874` (dev `4d7405b` plus that candidate). The merged
matrix names 15 test paths absent from the candidate across seven shards; the
candidate's own 208-shard manifest has no absent test paths. This is workflow/code
revision drift, not a demonstrated Calendar test failure.

The supported non-required `ci.yml` dispatch on `codex/calendar-dev-pr` created
replacement run `40020128320`, verified against exact head `14895fc`. Cancellation
of the original mixed-revision run was requested as superseded, not passed; GitHub
still reported it in progress at the final check. No required gate was
dispatched or replaced, no missing paths were ignored, and no source/test/workflow
files changed. The manual lane also enables OS/Python matrices; inspect applicable
failures rather than assuming optional admission skips imply success. Full CI is
pending. Preserve the failed job log/artifact `12530480543`; do not declare green
until actual replacement success. No rebase, merge or policy waiver.

#### Release-Lane Portability (2026-10-23)

Run `40020128320` exposed two further failures on exact `14895fc`: macOS Python
3.13 `core-utils-tooling` lacked Docker Compose; Windows Python 3.13 had nine
email-probe failures because native chmod mode bits do not enforce POSIX privacy.

1. **Reproduce (Complete):** Retained both job logs. Two deterministic non-POSIX
   regressions reached credential reads/admin connections before rejection.
   The missing macOS install-step contract also failed before the workflow fix.
2. **Repair and local verification (Complete):** Reuse the macOS 3.12 Compose CLI
   recipe in shared release steps, gated to macOS and `core-utils-tooling` so
   Windows never runs brew. Standalone PostgreSQL probe setup/validation reject
   non-POSIX hosts before credential or database changes; real POSIX mode checks
   remain intact. Tests assert native unsupported rejection, not skips or relaxed
   permissions. The provenance/timing test injects a validated synthetic manifest
   boundary without changing its remaining security assertions. Probe 90 pass;
   actual local monitoring/Compose 8 pass; workflow/license contracts 93 pass;
   Ruff, normal hooks and diff checks pass; scoped source Bandit 0 findings/errors.
   No ADR is needed: existing POSIX-only privacy policy, no new ACL support.
3. **Independent review and publish (Complete):** Requesting-code-review over
   the complete eight-file source/test/workflow delta found no issues. Reviewer
   independently passed 159 tests with no skips, verified red/green in memory and
   source/security checks, then closed. Published exact head
   `c85925c990960f8f6cce763949a9d8ac529dae6e` using the protected old-148 lease;
   review `5634915991` records technical clearance, not human approval.
   No Calendar runtime changes; shard coverage 5147 files/new_uncovered 0 passes.
4. **Hosted verification (In Progress):** Requires fresh exact-head PR statuses
   plus full Python 3.12 and applicable release-lane results. Local Python 3.12 on
   macOS is not real Windows or hosted Python 3.13 proof. Fresh PR CI `40030875894`
   and required workflows are queued, latest trusted license `40030909390` queued.
   Wait for actual new-head license SUCCESS before one exact-branch full CI
   dispatch; recheck the PR workflow matrix before replacing any mismatched run.
   Cancellation of old-head manual `40020128320` was requested, not counted as
   success. Old-head seven-context success is not proof for this candidate.
   No rebase or merge; preserve local publication notes without another push.

   22:34Z checkpoint: actual `frontend-license-policy/trusted/dev` is SUCCESS on
   unchanged `c85925c990`. With no prior matching dispatch, exact-branch full CI
   `40034567805` was created and its event/head verified directly; initial run-list
   indexing lag is not a reason to dispatch again. Current PR CI `40030875894`
   uses merge-ref `77056d9`, whose matrix still names 15 candidate-absent paths
   across seven shards; head's 208-shard manifest has none. Only that non-required
   mixed-revision run was requested cancelled as superseded, not passing.
   Required workflows remain untouched. Old manual `40020128320` is confirmed
   cancelled. Full `40034567805` and six other protected contexts are pending;
   inspect applicable release lanes and do not declare green yet.

#### Soak Collector Clock Fixture (2026-10-24)

Full CI `40034567805` failed only the Windows Python 3.12 healthy soak-fixture
case in `core-utils-tooling` at this checkpoint (2245 passed/19 expected skips).
The original assertion did not retain the harness failure list, so its precise
Windows cause remains inferred pending a hosted rerun.

1. **Reproduce and repair (Complete):** A controlled 100ms collector clock
   reproduced invalid/repeated telemetry and recovery failures. The test collector
   now adds a bounded per-sample sequence offset, as other tests in the file do.
   An explicit repeated-timestamp negative control still fails validation. No
   production harness, threshold, duration, timeout, filter or gate changed.
2. **Local verification and review (Complete):** All 42 soak tests pass with
   the real loopback integration server permitted; 40 healthy/coarse repetitions
   pass. Ruff and normal hooks pass; test-only Bandit excludes expected pytest
   assert warnings and has no other findings. Independent requesting-code-review
   found no issues, reproduced baseline failure, passed the full file/40 repetitions
   and all nine negative scenarios, and confirmed the sequence offset stays below
   30ms in this fixed profile. Reviewer closed. Earlier sandbox loopback denial is
   not counted as a code failure.
3. **Publish and hosted verification (In Progress):** Exact-old-head lease after
   review clearance, then require fresh actual PR statuses and exact-head full CI.
   Preserve failed run evidence; do not treat this inferred fixture fix as Windows
   proof or merge authorization. ADR not needed for a test-only collector change.

   Published exact `d8f91dd9aff02bd29106a31724b8b1373dd37f4e` with protected old-c859
   lease; technical review `5635743008` is clear and the reviewer is closed. Fresh
   PR CI `40043796259` and required workflows are queued; latest trusted license
   `40043821067` is queued. No same-head full dispatch yet: wait for actual license
   SUCCESS, check for an existing matching dispatch and create only one. Check the
   PR merge-matrix revision before replacing only a mismatched non-required run.
   Cancellation of old-c859 full `40034567805` was requested, not counted as pass.
   New-head Windows and full-suite proof remain pending; no rebase or merge.

   At 2026-10-24 01:05Z, exact-d8 trusted license `40043821067` passed. With no
   existing matching dispatch, full CI `40048217901` was created once and directly
   verified as `workflow_dispatch` on exact d8. Do not dispatch again. Current
   merge `126e6c5` still adds 15 candidate-absent paths across seven shards versus
   zero missing paths in the candidate's 208-shard manifest. After replacement
   verification, only mismatched non-required PR CI `40043796259` was requested
   cancelled; protected PR workflows are untouched. Old-c859 full `40034567805`
   is confirmed cancelled, not passing. Remaining protected contexts, full Ubuntu
   Python 3.12 and applicable OS/Python lanes are pending. No source edit or merge.

#### Windows Runtime Script Transport (2026-10-24)

Exact-d8 full CI `40048217901` failed Windows Python 3.12
`auth-integration-b-z` job `120750258667`: 34 VN storage tests could not launch
their Python subprocess (`WinError 206`); 97 passed and 117 expected skips.
The 32,695-character script expands beyond Windows' 32,767-character command
limit. This is a shared test launcher defect, not a storage assertion failure.

1. **Reproduce and repair (Complete):** Two long-Unicode-script regressions
   fail at a simulated Windows command-length boundary around the real process.
   Send UTF-8 script bytes through stdin, retaining raw diagnostic bytes and
   parsing only the prefixed JSON result. Private environment/cwd, timeout and
   nonzero/single-result assertions are unchanged. Add diagnostic-withholding
   coverage. All 43 selected helper/VN SQLite tests pass; no runtime code changes.
2. **Shared callers and review (Complete):** All 16 helper, 34 VN SQLite and 46
   other storage/bootstrap/VN caller tests pass (96 distinct tests). Ruff, syntax,
   hooks, diff and shard coverage pass. Test-scope Bandit retains the same four
   existing non-assert warnings and no new findings beyond expected pytest
   assertions. Independent review is clear after 14 real subprocess probes of
   base/current transport, Unicode, cp1252 and nonzero/missing/duplicate results;
   reviewer closed. Native Windows and PostgreSQL remain hosted prerequisites.
   ADR not needed for this test-only transport repair; all authority is unchanged.
3. **Publication and CI (In Progress):** Recheck no concurrent branch writer or
   auto-merge owner, normal commit and protected exact-d8 lease push only after
   review/verification. Fresh required PR statuses and applicable exact-head full
   CI remain mandatory. Preserve failed run evidence; no merge or policy waiver.

   Published `5636c06584780426787ff75a1d8465e340baadc7` with protected old-d8 lease;
   exact-head technical review `5636965027` is clear. New PR CI `40065435407` and
   protected workflows are queued; latest license `40065451129` is pending.
   No new-head full dispatch yet: wait for actual trusted/dev license SUCCESS,
   check for an existing matching dispatch, then dispatch exact-branch CI once.
   Independently check PR merge-matrix drift before replacing only non-required
   CI. Old-d8 full `40048217901` cancellation was requested after publication;
   its Windows failure evidence remains retained, not counted as a pass. Hosted
   Windows/PostgreSQL proof remains outstanding; leave publication notes local.

   At 2026-10-24 05:46Z, trusted license `40065451129` passed on unchanged 5636.
   Full CI `40067635101` was dispatched once after checking for duplicates and
   directly verified on exact 5636. Current merge `b93f85c` still adds 15 absent
   paths across seven shards while the candidate manifest has none. After verified
   replacement, cancellation was requested only for mismatched non-required PR
   CI `40065435407`; required PR workflows are untouched. Old-d8 full
   `40048217901` is confirmed cancelled, not passing. Full Ubuntu Python 3.12,
   applicable OS/Python lanes and six other required contexts remain pending.

#### Logging Fixture Isolation (2026-10-24)

Exact-5636 full CI `40067635101` exposed two Python 3.13 smoke failures: macOS
`120801012928` did not reject the timeout test's placeholder lock; Windows
`120801013261` observed one unrelated cleanup log instead of 100 retained entries.
Each job passed its other 227 tests. The precise Windows interleaving remains
inferred until hosted verification; the isolated local baseline passed 21 tests.

1. **Regressions and fixture repair (Complete):** Controlled pending-queue reset
   and initialization-aged-lock cases fail before the changes. Drain the existing
   writer queue after removing the sink and before resetting shared module state;
   restore the fake worker copied by the reload test. Reuse the existing fallback
   fixture's real held lock, matching upstream `9f9ca7c4aa`. Production logging,
   locking, retention, redaction and all timing/assertion limits are unchanged.
2. **Validation and review (Complete):** All 23 system-log tests and all 105
   Logging/Health tests pass. Ruff, syntax, normal hooks and diff pass; test Bandit
   has only the same three fixture-secret warnings and expected pytest assertions.
   Independent review is clear: five focused cases, a real persistent-writer and
   redaction probe, 20 worker-reuse repetitions without thread growth, and negative
   controls for draining, fake-worker restoration, stale locks and ignored runtime
   timeout. Reviewer closed. ADR not needed for test-only isolation; no Calendar,
   credential, tenant or Jobs behavior changed.
3. **Publication and CI (In Progress):** The complete affected smoke-shard manifest
   passes all 230 tests locally. Independent clearance, normal hooks, scoped
   security comparison, syntax, shard coverage and fresh branch/auto-merge checks
   pass. Use a normal commit and protected exact-old-head lease before push.
   Fresh required PR contexts and applicable exact-head full CI remain mandatory;
   keep failed logs and local publication notes. No merge or policy waiver.

   Published `a7e49c86db217c4bd87855fcae33e1c7f2467f33` with protected old-5636 lease;
   exact-head technical review `5637248537` is clear. Latest license `40072247193`
   is queued; other PR workflows were not yet indexed at the first read. Resolve
   them freshly. No same-head full dispatch yet: wait for actual trusted/dev
   license SUCCESS, check for an existing matching dispatch, then create one
   exact-branch CI run and verify its head. Recheck PR matrix drift before replacing
   only non-required CI. Old-5636 full `40067635101` cancellation was requested
   after publication, not counted as pass. Hosted Python 3.13 remains mandatory.

   Post-publication blocker: GitHub reports the candidate conflicting with dev;
   only pull_request_target license audits exist, with no normal PR workflows.
   Live dev `879d375635a71e8b6cdf1ab73b9e80f9a8ecfc00` has one content conflict
   with this candidate, in `tests/Logging/test_system_log_buffer.py`: upstream's
   held-lock test overlaps the candidate's initialization-aged-lock regression.
   `git merge-tree` confirmed this without changing branch, index or worktree.
   No dev integration was performed under this CI-only watch. Request one explicit
   integration decision to restore PR CI; do not repeat the question unchanged.
   Obsolete 5636 full `40067635101` is now confirmed completed/cancelled, not passing.

   2026-10-24T07:22Z: Actual `frontend-license-policy/trusted/dev` SUCCESS from
   `40072247193` verified on unchanged a7. After checking no matching dispatch,
   created full exact-branch CI `40073930678` once and directly verified its
   workflow_dispatch event, branch and exact a7 head; queued. Do not dispatch
   again. Required PR contexts remain absent because of the unresolved conflict;
   manual full CI is not a substitute. Integration approval remains unanswered;
   do not repeat the question. No source edit, rebase, push, merge or waiver.

#### BYOK Worker-Entry Investigation (2026-10-24)

1. **Evidence and reproduction (In Progress):** Exact-a7 full `40073930678`
   failed Windows Python 3.12 auth-unit-a-l job `120822833087`: one timeout at
   the lifecycle test's 1s adapter-entry wait, before its timeout/capacity
   assertions; 759 passed and one expected skip. Preserve
   `/private/tmp/calendar-pr3019-auth-unit-windows312-120822833087.log`.
   The local timeout/cancellation cases both pass. Full local shard baseline
   finished with 759 passes, one skip and a different sandbox IPC socket failure;
   the IPC case separately passed when permitted. Preserve both local logs; this
   is not an all-green full baseline. The test matches upstream. Hosted cause is
   not established; do not widen the watchdog or change runtime admission based
   only on an inference about scheduling.
2. **Hosted diagnosis (Pending):** A single debug-enabled failed-job rerun was
   requested after fresh exact-head verification but rejected as not rerunnable;
   full workflow remains active and run_attempt is still 1. Retry one diagnostic
   job rerun only after completion; do not duplicate the full dispatch or cancel
   valid lanes. Inspect the original first-task outcome if more evidence appears.
3. **Remediation and verification (Pending):** Only after a root cause is
   established, add a failing regression and minimal repair, scoped security
   validation, normal hooks and independent requesting-code-review. No source
   edit, runtime change, rebase, push, merge or waiver at this checkpoint.

#### Existing-Only DB Fixture Portability (2026-10-24)

1. **Regression evidence (Complete):** Exact-a7 full `40073930678` Windows
   Python 3.12 db-management-m job `120822834087` failed three setup operations,
   with 998 passes and 139 expected skips. A URI test creates a Windows-invalid
   question-mark filename; deletion tests unlink while SQLite still holds a
   handle. Preserve `/private/tmp/calendar-pr3019-db-management-windows312-120822834087.log`.
   Deterministic strict filename and real closed-handle guards reproduced all
   three defects before repair (three failures, one pass).
2. **Minimal fixture repair (Complete):** Use a portable URI-sensitive filename
   and release the existing cached connection before unlink, without closing the
   pool. Require query failure and no recreated file. Keep actual SQLite, all
   negative assertions, `mode=rw` and production code unchanged. Run focused/full
   boundary tests, Ruff, test Bandit comparison, shard guard and normal hooks.
   All 37 existing-only tests pass; Ruff, syntax, normal hooks and diff pass,
   Bandit has zero nonassert findings/errors on old/current test scope, and shard
   guard has zero newly uncovered files. Full affected shard passes all 1,140
   tests locally with no skips; local Python 3.12 is not native Windows proof.
3. **Review and publication (In Progress):** Independent requesting-code-review
   `01a16d6b-30b5-7410-932b-f5ab11072c90` is clear/closed: four focused cases,
   two real reconnect probes and 11 negative controls pass. Both query methods
   attempt a fresh native connection and receive `SQLITE_CANTOPEN`, not a closed
   pool error. Escaping remains meaningful on both path flavors. Technical
   publication clearance only; actual Windows confirmation remains mandatory.
   Proceed with
   then fresh writer/auto-merge/head checks, normal commit and exact-old-head
   lease. ADR not required for test-only fixture validity. No rebase or merge;
   existing dev conflict and separate BYOK uncertainty remain unresolved.

## Stage 1: Provider Safety and Worker Reliability
**Goal:** Fix findings 6, 7, 10, 11, 12, 15, and 19.
**Success Criteria:** Streamed responses stop at a byte limit; XML/ICS are bounded before parsing; synchronous provider calls run off the event loop; API and worker reuse scoped credential resolution; a deleted account or failed scan does not stop later polling; diagnostics contain safe context and stack locations without secrets.
**Tests:** Add oversized Content-Length/chunk/ICS tests to `tests/Calendar/unit/test_calendar_caldav_provider.py`; add event-loop responsiveness, safe logging, missing-account continuation, and scan-retry tests to `test_calendar_sync_worker.py`; add verify/discover off-thread and credential precedence/scope tests to `integration/test_calendar_api.py`.
**Files:** `core/Calendar/providers/caldav.py`, new scoped `core/Calendar/provider_operations.py`, `core/Calendar/calendar_sync_worker.py`, `services/calendar_sync_scheduler.py`, `api/v1/endpoints/calendar.py`, and the tests above.
**Steps:** Write and run regressions (expect failures); implement the smallest shared operations and bounded transport; rerun affected tests (expect pass); run Bandit and pre-commit; commit with TASK-13356.
**Status:** Complete

## Stage 2: Temporal and Recurrence Integrity
**Goal:** Fix findings 1, 2, 3, 5, 13, 14, and 16.
**Success Criteria:** Provider UID plus recurrence identity is stable; masters, exclusions, additions, and detached exceptions survive repeated imports; all-day dates and exclusive ends survive; local RDATE/EXDATE work; malformed times and reversed intervals are rejected before persistence; explicit null removes recurrence while omission preserves it; failed recurrence writes roll back item changes; afternoon windows contain all-day events.
**Tests:** Extend provider parsing/import tests, `test_calendar_recurrence.py`, `test_calendar_service.py`, `test_calendar_db.py`, and API tests with recurrence-only dates, exclusions, detached exceptions, date-only DTSTART/DTEND, invalid updates, all-day noon queries, null/omitted recurrence, and injected write failures. Extend recurrence property coverage where applicable.
**Files:** `core/Calendar/recurrence.py`, `view_service.py`, `calendar_service.py`, CalDAV provider/worker, `core/DB_Management/Calendar_DB.py`, schemas/endpoints, and corresponding tests.
**Steps:** Write regressions and observe failures; use dateutil recurrence sets and bounded iteration; preserve provider metadata and instance identities; implement nested repository transactions and deletion; validate merged item state; run Calendar unit/integration/property suite; run Bandit and pre-commit; commit with TASK-13356.
**Status:** Complete

## Stage 3: Permissions, Links, and Review Hygiene
**Goal:** Fix findings 4, 8, 9, 17, 18, 20, and 21.
**Success Criteria:** Active AuthNZ organization roles are resolved request-locally with org/tenant boundaries; permission tests have type hints and unit markers; existing item calendar selection cannot imply an unsupported move; persisted links load after refresh and can be removed through authorized APIs; centralized calendar exception exports retain compatibility; module/dependency/endpoint functions have concise meaningful docstrings.
**Tests:** API role member/nonmember/wrong-org and revoked membership tests; authorized link list/delete and refresh tests; drawer calendar-selector and persisted-link tests; exception export identity and endpoint docstring checks.
**Files:** Calendar API/schemas, permission tests, shared exception modules, `apps/packages/ui/src/services/calendar.ts`, Calendar drawer/types/tests, and backend integration tests.
**Steps:** Write failing tests; wire existing AuthNZ membership APIs; add link GET/DELETE and UI retrieval; disable only the unsupported move control; centralize exception definitions using existing lightweight export pattern; document API functions; run backend/frontend tests and typecheck; run security/format checks; commit with TASK-13356.
**Status:** Complete

## Stage 4: Re-review and Integration
**Goal:** Publish verified fixes, address follow-up review, and satisfy merge gates.
**Success Criteria:** Each earlier Qodo finding has a verified disposition in its thread; the human-authorized requesting-code-review replacement and required CI checks pass; branch is current with dev; PR is merged, or an exact remaining external/policy gate is recorded without claiming completion.
**Tests:** Full Calendar pytest suite; focused frontend Vitest tests from frontend and shared-UI working directories; frontend TypeScript; touched-scope Bandit; pre-commit; shard coverage guard; git diff checks. Inspect baseline failures separately rather than masking them.
**Steps:** Self-review complete diff; verify and push; reply to Qodo threads; request fresh review; inspect CI and latest dev; rebase/reverify if required; merge only when allowed; update Backlog with PR/verification/final state.
**Status:** In Progress

### Follow-Up Review Fixes

- Bound provider recurrence before entering dateutil, including rules that never yield; report occurrence truncation as partial rather than complete.
- Preserve master timestamps when editing occurrence text and preserve offsets on unchanged temporal fields; disable unsupported occurrence time editing.
- Add authorized native-item soft deletion using the existing service/repository operation.
- Apply item timezone to nonrecurring overlap and serialize timed views with offsets.
- Treat date-only all-day values as civil dates in agenda/week rendering.
- Derive exclusive provider ends from valid VEVENT DURATION when DTEND is absent.
- Add failing regressions first, run focused/full checks, and obtain a scoped independent re-review before publishing.
- Ownership: controller owns backend changes; a frontend implementer owns only Calendar drawer/agenda/week and their frontend regression tests. No overlapping edits or concurrent commits.

### Exact-Head Review on 38c29d84bd

Review 5325803386 adds nine findings; the previous 26 remain resolved. Qodo clarification 5846051515 confirms the findings index has only these nine active entries and every other indexed entry is implemented. Keep the approved feature scope and merge gates unchanged.

1. **Authorization and destructive actions (Complete):** Add failing service/API regressions rejecting organization creation without active authenticated membership and drawer regressions proving recurrence occurrences cannot delete the shared master. Enforce organization authorization in CalendarService with a request-scoped membership resolver; hide ordinary occurrence deletion. Frontend worker owns only the drawer and its tests; controller owns service/API and their tests.
2. **Worker event-loop responsiveness (Complete):** Reproduce blocking initialization, lookup, persistence, and failure bookkeeping with deterministic off-thread regressions. Offload complete synchronous phases, keeping each transaction on one worker thread and preserving bounded provider dispatch, atomic imports, and scoped credentials. Worker agent owns calendar_sync_worker.py and its tests, including missing docstrings there. Independent review additionally reproduced native asyncio cancellation abandoning a live database phase and AnyIO scope cancellation spinning the drain. Drain started work and failure bookkeeping before propagating cancellation; shield subsequent drain attempts from level-triggered cancellation while retaining repeated native-cancel handling. Six native and two scope regressions failed before their fixes. Final narrow re-review is clear, with one drain retry in each of four 200ms probes and no orphan work or transaction-context leakage.
3. **URL and view correctness/performance (Complete):** Write red tests for malformed base/collection ports and end-exclusive scheduled projections. Validate URL ports before account persistence and convert parse errors to domain validation errors. Replace per-row item refetches with authorization of loaded rows, calendar-context caching and batch provider ownership lookup; prove query count is independent of candidate count while retaining privacy and tenant checks. Controller owns provider, view/service, DB helper and corresponding tests. Independent review found and verified an additional collection-authority urljoin error; four regressions failed before translation, six focused URL/API cases now pass. Narrow re-review reports no remaining findings.
4. **Compliance and publish (In Progress):** Annotate/classify DB tests and document new schema/DB symbols with concise meaningful docstrings; verify contracts. Run full Calendar/backend/frontend, TypeScript, Ruff, Bandit, pre-commit and shard guard; obtain scoped independent review; commit and publish verified changes, reply to each new inline finding, request exactly one fresh exact-head review, and update the heartbeat. Stage 4 remains In Progress until merge gates are genuinely satisfied.

### Exact-Head Review on 7e15ca42d6

Review 5325936683 is complete with eight new inline findings; the prior 35 remain implemented. Clarification 5846439303 confirms all 27 omitted index entries are implemented and only the eight current inline findings remain active. Do not request a duplicate full review on this head.

1. **Credential and persistence safety (Complete):** Reproduce cross-origin forwarding of stored credentials in verify/discovery; pin reused credentials to the persisted account origin while retaining deliberate complete replacement credentials. Reproduce orphan secrets on account-insert failure and wrap account plus secret creation in one repository transaction. Controller owns provider_operations.py, Calendar API and integration tests. Initial regressions: 22 failed, then 124 API/service tests passed. Independent review additionally reproduced a fallback URL acting as its own origin for legacy unpinned credentials: two regressions failed before requiring a saved origin for all reuse, then 126 API/service tests passed. Narrow independent re-review approved with eight targeted tests and 16 helper checks.
2. **Temporal correctness (Complete):** Reproduce omitted timezone for local wall-clock item creation and inherit the selected calendar timezone. Controller owns calendar_service.py and its unit tests. A temporal implementer preserved embedded custom VTIMEZONE definitions and payload-scoped timezone reconstruction, with four later-window regressions failing before the fix. Preserve DST, folds, durations, exclusions, detached identity and refresh stability. Pre-resolution guards reject nonproductive rules, observance exclusions, oversized definitions and excessive observance counts. Independent review reproduced dense historical transitions and two rule-family gaps; the controller added a cumulative 20,000-transition budget covering RRULE history and explicit dates, rejected mixed ordinal/plain weekdays, and corrected implicit month-day density. Their regressions failed first; final narrow review approved with 38 focused tests, 180 finite-rule comparisons without undercounts, and DST probes through 2100. All temporal agents are closed.
3. **Frontend input and migration boundaries (Complete):** A frontend implementer owns CalendarSyncSettings.tsx/CalendarItemDrawer.tsx and their tests, normalizing sync days to integers and verifying selected-calendar timezone payloads. Six regressions failed first; 35 focused and 70 full Calendar frontend tests pass under French locale, TypeScript clean. A separate migration implementer moved only the new Calendar SQL to a DB_Management helper without changing unrelated migration code. Query-boundary/helper tests failed first; 18 focused unit tests and 52 related migration tests pass. Two real-PostgreSQL checks skip because the existing fixture cannot reach PostgreSQL. Both slices have clear independent scoped reviews; implementers/reviewers are closed.
4. **Compliance and release (In Progress):** Controller annotated service tests/helpers, renamed the vague API scenario, and mechanically fixed two property-test lint issues. Final verification: 312 Calendar backend tests, 70 French-locale frontend tests, 11 default temporal-view tests, 3 shared-UI route tests, TypeScript, Ruff, pre-commit and diff checks; Bandit zero findings/errors across 12 touched backend source files. Migration verification: 18 focused unit tests and 52 related tests pass, with two existing-fixture PostgreSQL skips. Shard guard covers 4806 test files with zero newly uncovered. All independent scoped reviews approve and every agent is closed. Commit with normal hooks, publish, reply with individual evidence and request one exact-new-head full review. Required CI, live-provider smoke and human rationale gates still apply.

### Exact-Head Review on 09d43aa0a0

Review 5326092846 completed with two new inline correctness findings; prior 43 findings remain implemented. Clarification 5846842617 confirms the index has two active findings, 42 implemented and one dismissed, with no additional active omitted entries.

1. **Floating recurrence dates (Complete):** Parser and four import/persistence/refresh/reopened-repository agenda/week regressions reproduced UTC coercion. Preserve naive wall time when no TZID is supplied; retain explicit UTC and named-zone offsets. IANA and embedded custom-zone cases retain intended additions, exclusions and one-hour durations across DST.
2. **Annual timezone productivity (Complete):** Eight finite-rule/import regressions reproduced valid month-day sets rejected because some selected months lack a requested day. Accept sets with at least one valid non-leap-year transition, retaining syntax validation, rejection of wholly nonproductive combinations, and conservative cumulative transition bounds. Positive/negative days, mixed valid/invalid dates, implicit/explicit months, pre-parser rejection and finite count estimates are covered.
3. **Verification and publication (In Progress):** Initial focused run: 14 failed and 15 passed; corrected run: 29 passed. Full Calendar backend: 330 passed. Ruff, normal pre-commit and diff checks pass; Bandit zero findings/errors on both changed source files; shard guard 4806 test files, zero newly uncovered. Independent scoped review approved: 29 focused and 16 existing regression tests, 1032 accepted finite-rule comparisons without undercounts, and 192 nonproductive combinations rejected. Reviewer closed with no active sessions. Frontend and migration production/test files are unchanged from their previously verified suites; do not repeat them. Publish verified fixes, reply individually, and request one fresh exact-new-head review. Existing CI, live-provider smoke and human-summary merge gates remain unchanged.

### Exact-Head Review on 9e0e55eea8

Review 5326171306 reports zero bugs and one documentation finding (4111618387) covering two helpers. Clarification 5847009576 confirms no other active omitted findings. Prior 45 findings retain verified fixes/dispositions.

1. **Helper contracts (Complete):** Two static contract cases failed before documentation changes. Expanded only _validate_timezone_rule and CalDavProvider._component_dates docstrings using established Args/Returns/Raises style. Every parameter, actual return behavior/bounds and CalendarValidationError conditions are documented; production ASTs are unchanged excluding docstrings.
2. **Verification and publication (In Progress):** All 10 contract tests and 29 focused temporal regressions pass; Ruff, Bandit zero findings/errors, diff and shard guard zero newly uncovered pass. Independent scoped reviewer approved documentation, tests and AST equivalence, then closed all sessions. Run final normal pre-commit, publish with individual evidence, and request one exact-new-head full review. Prior 330-test Calendar runtime verification remains applicable; do not repeat unchanged broad suites or weaken remaining merge gates.

### Current-Dev Rebase on f5fa1f3a41

1. **Integration (Complete):** Dev advanced to `f5fa1f3a41855aa02871d8b76d0ec0cebbaf9e07` through PR #3006 (Sync blob-upload expiry and documentation). All 15 Calendar commits rebased without conflicts and compare equal by range-diff. Only the 12 upstream-owned files differ from published `710885e988`; Calendar production and test files are unchanged.
2. **Fresh verification (Complete):** Full Calendar backend: 332 passed. French/Europe-Paris frontend: 70 passed; default temporal: 11 passed; shared-UI route: 3 passed. TypeScript, Ruff, full PR-scope normal pre-commit, diff checks and shard guard (4807 test files, zero newly uncovered) pass. Bandit reports zero findings/errors across 27 PR-owned backend source files. Related AuthNZ/service verification: 48 passed, two failed and two PostgreSQL fixture skips. Both failures are the previously documented migration-090 helper assertion at `test_rbac_seed_helper.py:432` (expected version 96, registry version 98); relevant helper/registry files are unchanged from the prior published head. No tests were disabled and no unrelated fix was introduced. PostgreSQL skips are not live-database success.
3. **Publication and gates (In Progress):** Publish the rebase and tracking evidence using normal hooks and an explicit lease against `710885e988`, then request one full review on the exact published head. The prior head's Qodo execution remains unverified; neither its resolved-only summary nor receipt reaction establishes completed full review. Required exact-head CI, live CalDAV smoke and the human-owned rationale gate remain prerequisites. Stage 4 and TASK-13356 remain In Progress.

### Exact-Head Deep Review on aa48471476

Review 5327644148 completed at 22:04:52Z and posted six new inline candidates. The summary shows one bug/two rules and omits 37 entries; clarification 5850352501 confirms exactly these six active findings and no additional active omitted entries. There are now 52 reported findings and 51 inline threads. Do not duplicate the completed full review.

1. **Polling cadence (Complete):** Reproduced omitted-interval bindings remaining continuously due. Schema, repository and UI now use the smoke runbook's hourly default; explicit/legacy null remains manual-only and is excluded from periodic due queries. Ten cadence regressions failed before production edits; all 113 DB/worker tests pass, retaining native/AnyIO cancellation and atomic import behavior.
2. **Backend request contracts (Complete):** Invalid ZoneInfo keys are rejected before calendar writes. Manual sync overrides require aware ISO timestamps, positive UTC spans no longer than the existing combined lookback/lookahead maximum (7400 days), and at most 64 characters per boundary; invalid input creates neither Jobs nor queued audits. The bounded sync-history handler uses synchronous FastAPI worker execution, including ownership checks. All 31 permission dependencies are annotated with the actual AuthPrincipal return type. Controller regressions failed first; the 165-test service/API/contract suite passes.
3. **Verified account setup (Complete):** Generic account creation remains compatible through verify_before_create=False. The UI sends True with complete supplied CalDAV credentials; bounded off-loop verification precedes the account/secret transaction. False verification or provider failure creates neither account nor encrypted secret. UI failure preserves the draft and skips post-persistence verification. Frontend 13 red regressions preceded fixes; 29 focused and 83 full French/Europe-Paris tests pass, with TypeScript and scoped ESLint clean. Strengthened backend regressions prove exact credentials, off-loop execution, ordering and no orphan secrets.
4. **Verification and publication (In Progress):** Full Calendar backend: 385 passed. Ruff, normal 17-file pre-commit, diff and shard guard (4807 files, zero newly uncovered) pass; Bandit reports zero findings/errors across all six changed backend source files. Independent backend review approved 115 targeted tests and nine boundary/authorization/rollback/cancellation probes; independent frontend review approved 29 focused tests and six Unicode/password/retry/serialization probes. All implementers/reviewers and their sessions are closed. Commit and publish with individual replies to all six findings and one exact-new-head full review. Existing hosted CI, live CalDAV smoke and human-owned rationale remain merge gates. Unchanged migration verification retains 48 related passes, two known migration-090 assertion failures and two unavailable-fixture PostgreSQL skips, not live-PG success.

### Exact-Head Deep Review on 573b90de8a

Review 5327879344 completed at 22:54:51Z and posted eight new inline candidates. Findings-index clarification 5850714013 confirms exactly eight active findings and no other active omitted entries; the remaining 52 are implemented or dismissed. The completed request must not be duplicated.

1. **Canonical rate limits (Complete):** Missing catalog policy produced three failing cases; catalog-only enforcement still failed three HTTP 429 assertions because rate dependencies ran before authentication. Authentication-first router dependencies and all 32 per-user budgets now enforce existing standard read (120/minute, burst 240) and privileged write/sync (60/minute, burst 120) classes. Nine integration cases cover all routes, auth-kind sharing, user isolation and stricter/higher database overrides. Independent review approved 178 related tests, 12 boundary probes and two seed checks; no live PostgreSQL success claimed.
2. **Off-loop polling (Complete):** The complete synchronous scan/account/queue phase uses the existing cancellation-drained DB phase; 12 regressions failed before the move. Independent review reproduced a manual/scheduler active-check/create race; shared same-binding admission and the manual HTTP off-loop phase now pass two race regressions and seven baseline-failing API cancellation/responsiveness cases. Independent approval includes 32 focused tests, 20 concurrent manual requests with scheduler contention and a symlinked repository producing one job/audit, unrelated binding progress, idle-lock reclamation and 21 repeated native cancels plus AnyIO draining of a real uncommitted audit. Preserve bounded scans and hourly/manual-only cadence. The weak-registry binding gate is process-local only; it is not cross-process atomicity or a transaction across separate Jobs/Calendar stores.
3. **Temporal consistency and coverage (Complete):** Six DST/fold mismatch regressions and one malformed P1DT regression failed before fixes; 63 focused and 66 temporal/property tests pass. Direct and set timed expansions share UTC elapsed-duration calculations. Thirty-seven direct duration cases passed baseline and explicitly protect civil days/weeks, elapsed subday time, validation and folds. Independent review reproduced second-fold DTSTART loss/duplication and same-zone fold-window rejection. Fold-aware candidates retain the exact initial instant/count slot; datetime query ordering/span use UTC. Twenty-eight baseline failures plus UNTIL/spring/exact-seed follow-up failures preceded fixes; initial 49 new cases and 237 related tests pass, including persisted agenda/week. Final review additionally reproduced same-zone UNTIL premature exclusion of first-fold candidates: five failures/two passes changed to seven passes after giving dateutil a UTC UNTIL bound, retaining civil/floating interpretation first. Final independent approval covers 58 focused tests and 13 civil/naive/COUNT/second-fold probes; no remaining actionable temporal findings.
4. **Frontend validation and coverage (Complete):** Projection URLs use the established safe-URL helper; local field-level schedule validation retains drafts without API calls. Thirteen initial regressions failed before fixes. Scoped review then reproduced due-only display fallback, hidden kind-switch draft and Honolulu test assumptions: three drawer cases, four persisted provenance cases and one Honolulu case failed before their respective fixes. Authorized view metadata now distinguishes canonical nullable item_start_at from a due-derived display timestamp; active-kind serialization preserves actual start-only todos and drops hidden event drafts. Full French/Europe-Paris frontend 125, Honolulu temporal 13 and TypeScript pass. Scoped ESLint has zero errors and three unchanged baseline warnings. Independent narrow review approved 12 probes, nine drawer cases, two Honolulu cases and scoped TypeScript; reviewer is closed. Direct filters/ownership coverage passes; same-day events already render in both views on unchanged production code, supporting a reasoned disposition rather than a false fix.
5. **Integration and publication (In Progress):** Final full Calendar backend: 530 passed with 13 warnings and no failures. Frontend 125 French/Europe-Paris, Honolulu temporal 13 and TypeScript pass. Ruff, scoped ESLint (zero errors, three unchanged warnings), diff checks and shard guard (4810 files, zero newly uncovered) pass; Bandit reports zero findings/errors in all six changed backend source files. Independent rate, frontend and temporal/admission reviewers approve; all agents and sessions are closed. Commit with normal hooks, publish and reply individually to all eight findings, including the baseline-verified same-day disposition. Then request one full review on the exact new head. Hosted required CI, live CalDAV smoke and human-owned rationale remain merge gates; Stage 4 and TASK-13356 remain In Progress until actual normal merge.

### Wave9: Durable Sync Admission (Approved 2026-09-27)

**Spec:** `Docs/superpowers/specs/2026-09-27-calendar-sync-admission-design.md`.
**Goal:** Fix the three exact-c87 findings without changing provider or UI scope.
**Status:** In Progress

1. **Atomic reservation and audit (Complete).** Modify
   `core/DB_Management/Calendar_DB.py` with a typed admission row, binding-unique
   reservation, conditional Job attachment/release and bounded pending lookup.
   Keep SQL in the repository. Add failing tests in
   `tests/Calendar/unit/test_calendar_sync_worker.py` for audit failure leaving
   zero runnable Jobs, dispatch failure retaining one recoverable audit, and
   simultaneous processes creating exactly one Job/audit. Run those exact tests
   before changing worker admission; implement and rerun them.
2. **Dispatch and lifecycle (Complete).** Replace worker process-local locking
   with committed reservation plus stable idempotent Jobs dispatch. Use exact
   scoped live/archive lookup and conditional release only for terminal states.
   Replace the capped legacy lookup with existing keyset pagination. Scheduler
   recovers pending admissions before ordinary due scans, including manual-only
   bindings. Red/green tests cover old Jobs beyond 100, post-dispatch crash,
   processing/delayed retries, archive/missing Job behavior, same-window terminal
   resubmission, tenant scope and unrelated bindings during blocked dispatch.
   Independent review reproduced stale-dispatch resurrection after terminal
   pruning, status-separated legacy retry omission, and completed manual-only
   recovery resubmission. Three controller regressions RED then GREEN using the
   existing Jobs durable receipt API, status-independent keyset traversal and
   exact recovery distinct from fresh triggers. Receipt commands explicitly use
   priority 5 to preserve the Jobs 1..10 contract; initial default-100 failures
   were corrected before final integration. Final independent review approved
   the exact source hashes and six race/encryption/authority/fairness probe groups.
3. **Typed core trigger (Complete).** Add
   `CalendarService.trigger_binding_sync(...)->CalendarSyncJobResponse` and type
   `queue_binding_sync`. Move configured-window resolution and ownership into
   core; endpoint delegates through `_run_db_phase` with validated fields.
   Add an endpoint delegation regression and preserve aware ISO <=64-character,
   positive UTC <=7400-day validation and native/AnyIO cancellation regressions.
4. **Verification and publication (In Progress).** Current-dev rebase completed
   without conflicts onto `718c191082f1d6372fb6fe000ac763dcc07ffcbd`; local rebased
   head `1509d98dd7504d946dcb001c5a00d7a4034452e1`, published c87 unchanged. Run
   full Calendar backend, current frontend/TypeScript and migration boundaries,
   Ruff, touched-source Bandit, shard guard, diff and normal pre-commit. Obtain
   fresh independent scoped review, fix findings with TDD, commit/publish only
   verified work with an explicit remote-head lease. Reply individually with
   evidence and verify thread resolution; request exactly one full new-head
   review. Required CI, latest dev, provider smoke and human rationale still gate
   normal merge. Never publish an intermediate tracking-only review head.
   Final current-base verification: 558 Calendar backend tests (13 warnings),
   68 existing SQLite Jobs receipt/prune tests (2 warnings), 18 Calendar migration
   unit tests (38 warnings), 13 Honolulu temporal tests, TypeScript, Ruff, diff,
   normal 11-file pre-commit and shard guard (4829 files, zero newly uncovered)
   pass. Bandit reports zero findings/errors in five changed backend sources.
   Prior current-frontend French/Europe-Paris run passed 125 tests on dev9668;
   subsequent dev718c changes have no frontend overlap. Bacon independent final
   scoped review approved and closed; no live PostgreSQL success is claimed.

## Decisions and Evidence

- 2026-09-26: Rebase onto dev `59bd584503` completed; original three commits unchanged by range-diff. Published head `6602a4c5956e99be6ed5d11fef879102d4732a36`.
- Baseline on rebased head: Calendar backend 112 passing, frontend 30 passing, shared-UI route tests 3 passing, TypeScript clean, Bandit zero findings, shard guard zero new uncovered files.
- Qodo review posted 21 findings. Finding 21 was omitted from the initial summary and was supplied in issue comment 5843301983: endpoint docstrings.
- External provider smoke remains unrun without credentials. Preserve the requester-authored Change summary verbatim; do not manufacture human rationale.
- Follow-up Qodo review on a4c76e9c00 resolved the original findings and raised four new findings: typed recurrence property helper, recurrence regression docstring, explicit truncation warning, and old high-frequency recurrence seeking. These are included in Stage 4 fixes.
- Ruling: Expand only productive provider rules (frequency/interval/count/until, weekly weekday selection, week start); preserve complex BYxxx rules verbatim with partial-result warnings rather than evaluating potentially non-yielding dateutil generators. This follows the spec's pragmatic recurrence/preserve-complex-provider-data boundary.
- Ruling: Seek second/minute/hour rules arithmetically before dateutil, preserving interval phase, COUNT, UNTIL, and duration-adjusted overlap. Dateutil between() itself still scans from DTSTART and does not provide a work bound.
- Remediation verification: 152 Calendar backend tests; 34 frontend tests; 3 route tests from the shared-UI cwd; TypeScript and full Calendar-source Ruff pass; touched-scope Bandit has zero findings/errors; pre-commit and shard coverage guard pass.
- Additional regression: provider all-day RRULE UNTIL dates are normalized for aware recurrence expansion while raw provider metadata remains unchanged.
- Refetched dev remains `59bd5845038342013a2d84d0130f6164f14b54fd`. Desktop/mobile Calendar drawer screenshots use isolated, nonsecret API fixtures, not a live provider connection.
- Final follow-up verification: 173 backend tests and 58 Calendar frontend tests pass; TypeScript passes. Scoped reviewers identified and prompted fixes for DST duration arithmetic, custom VTIMEZONE offsets, agenda window clamping, and DST-crossing edits. Added regressions failed before fixes.
- DURATION uses structured icalendar content-line parsing to retain P1D versus PT24H, then shared nominal-day/elapsed-hour arithmetic. Raw duration is preserved and reapplied to each recurrence instance. See RFC 5545 section 3.3.6.
- CalendarPage tests now pin Date to their June fixture window without faking async timers; previously they displayed mocked out-of-window records using the current date.
- Scoped frontend review approved after an additional overnight-window regression failed and was fixed; final frontend suite has 59 passing tests. Midnight-exclusive boundaries retain correct placement.
- Final backend fold regressions: compare recurrence durations and window overlap as UTC instants; normalize recurrence-set candidates/additions/exclusions before ordering so repeated local hours remain distinct. Zero nominal-day duration additions preserve fold. Both regressions failed first; full Calendar backend now has 175 passing tests.
- Independent scoped re-reviews are clear: frontend spec/quality approved; backend reports no remaining P1/P2 and seven targeted fold/exclusion/all-day checks pass. Required hosted CI and live-provider/human-summary gates remain external prerequisites.
- Fresh full Qodo review on 34e644438e reported no bugs and one locale-dependent test finding. French-locale reproduction failed 10 of 11 temporal-view cases; replacing hard-coded English labels with a locale-aware test formatter preserves civil-date assertions without changing production behavior. All 59 Calendar frontend tests and TypeScript pass under French locale after the fix.
- Integration checkpoint: dev advanced to `a2826f103f02a67f57adb40ed048dbfa2ecfc6e5` with unrelated VZ startup-drill changes. All ten Calendar commits rebased without conflicts and compare equal by range-diff; every PR-owned file is unchanged from published `eb1bb493a1`.
- Rebase verification: 175 backend tests, 59 French-locale frontend tests, 11 default-locale temporal-view tests, and 3 shared-UI route tests pass. TypeScript, Calendar-source Ruff, and diff checks pass; touched-backend Bandit has zero findings/errors; shard coverage has zero newly uncovered files among 4803 test files. Publish with an explicit lease, then obtain exact-new-head Qodo review and required CI. Live-provider smoke and human-summary policy gates remain outstanding; Stage 4 is not complete.
- New review wave checkpoint: 168 stable service/provider/API/DB/documentation-contract tests pass; 64 French-locale frontend tests, 11 default temporal tests, 3 shared-UI route tests and TypeScript pass. Scoped authorization/URL/query review is clear; large agendas require 8/9 SELECTs for 51 candidates, preserving tenant/private-provider/actor-overlay boundaries. DB/schema AST comparison confirms documentation-only changes except two reviewed batch helpers and return annotations. Worker cancellation regression remains in progress before full-module verification and publication.
- Final new-wave verification: 245 Calendar backend tests (43 worker), 64 French-locale frontend tests, 11 default temporal tests, 3 shared-UI route tests and TypeScript pass. Ruff, normal pre-commit, diff and shard checks pass; Bandit reports zero findings/errors in touched backend source. Both scoped re-reviews are clear and all agents are closed. All 35 reported Qodo findings have verified fixes, including the nine new findings plus collection-resolution/native-cancellation/AnyIO-drain gaps found by independent review. Publish, reply to the nine threads, then request one fresh exact-head full review; required hosted CI, live-provider smoke and human-summary rationale gates remain outstanding. Stage 4 stays In Progress.
- 2026-09-29 Wave10: Human approved the three remaining exact-23ae findings (safe verification-log correlation, unsupported-server calendar creation, mounted-page local-date refresh); remote deletion remains explicitly deferred. Rebased 19 local commits onto dev `0da68530e80c713ed3a323a741998e1fed37e3e9` without conflicts; local unpublished head `b3652528e3fd`. Focused regressions failed before the fixes and pass after them. The new backend-required RLS ratchet exposed six CalendarDatabase tables defined only in SQLite; documented those as non-PostgreSQL exemptions without changing ratchet logic. RLS suite 17/17, other integrated required contracts 67/67, Ruff and the project TypeScript typecheck pass. Full current-dev frontend/backend and shared-boundary verification, Bandit, normal hooks, scoped review, publication and exact-new-head Qodo remain pending. Stage 4 stays In Progress; smoke and human-authored what/why summary still gate merge.
- 2026-09-29 latest-dev verification: Rebased all 20 commits onto dev `6110d2ae436c805c3beeda8f84427b4890534ddf`; resolved the sole AuthNZ seed-test conflict by retaining both upstream and Calendar callers. Full Calendar backend 558 PASS; Calendar frontend 128 PASS in French/Europe-Paris and TypeScript PASS; shared frontend history/loader 146 PASS. CI contracts/ratchets 130 PASS, SQLite Jobs receipt/prune 90 PASS/23 SKIP, AuthNZ unit 184 PASS, startup/catalog 24 PASS, migrations and DB errors 36 PASS/10 SKIP, provider/HTTP/history 151 PASS/50 SKIP, OpenAPI/RLS startup 93 PASS. Ruff PASS; Bandit on Calendar backend scope reports zero findings/errors; shard guard covers 4889 test files with zero newly uncovered. Local Python tests used the existing 3.11 venv while dev now requires 3.12; the available bare 3.12 interpreter lacks project dependencies, so hosted 3.12 checks remain mandatory. PostgreSQL skips are not live PostgreSQL validation. Final hooks, publication, exact-head Qodo, hosted gates, live-provider smoke, and human-authored what/why summary remain; Stage 4 stays In Progress.
