# Post-publication Qodo follow-up

Tracking: TASK-13263.3. Scope: four PR2974 findings, six PR2972 findings found
in the final publication review inventory, and thirteen findings on follow-up
PR2978 plus nine final sync PR2971 comments. PR2972's six comments were posted at
07:08 UTC, after its approved merge/tag/release. Published v0.1.43 and its source
record remain immutable. The approved main-to-dev sync carries that release;
this separate branch prepares verified follow-up fixes.

## Stage 1: Validate review claims
**Goal:** Trace the reported behavior through its actual boundaries.
**Success Criteria:** Every finding is either reproduced or supported by a
concrete existing contract and test. The proposed full-suite coverage gate is
assessed against the requester's explicit bounded-gate approval.
**Tests:** Chat SQL at PostgreSQL driver boundaries; denied-chat active/stale
loads; sign-in error behavior; fixture ownership and category selection.
**Status:** Complete

## Stage 2: Apply bounded repairs
**Goal:** Fix confirmed defects using existing project helpers.
**Success Criteria:** No speculative auth-shell extraction or production API
added solely to support a fixture. Safe error text and account/request authority
remain intact. No published tag, legal record or package version changes.
**Tests:** Red/green behavior regressions, relevant existing test files,
TypeScript, scoped lint and Bandit against baseline.
**Status:** Complete

## Stage 3: Review and prepare follow-up
**Goal:** Produce a reviewable PR with every disposition and its evidence.
**Success Criteria:** Independent review, fresh focused verification, documented
limits and inline Qodo replies linked to the fix or evidence. Publication and
follow-up merge are separate from preparing these changes.
**Tests:** Combined impacted tests, clean diff, unchanged published source record.
**Status:** In Progress

## Verification checkpoint

215 frontend tests and the full WebUI TypeScript check pass. Scoped ESLint has
zero errors and seven unchanged warnings. The 48 fixture/Media tests pass;
33 chat SQL tests pass (six driver-boundary cases fail under normalization
mutation). The official PostgreSQL fixture reports unavailable; no live-server
pass is claimed. Independent reviews of frontend and fixture changes found no
actionable issues. Bandit/Ruff comparisons introduce no new findings. Combined parent verification passes 94 tests with one official PostgreSQL
unavailable skip and a clean process exit. Pushed PR checks and final thread
replies remain pending.

Detailed ten-finding dispositions and commands:
[review evidence](../reviews/2026-09-21-post-publication-qodo.md).


## Pushed review and sync CI follow-up

[PR #2978](https://github.com/rmusser01/tldw_server/pull/2978) holds the changes
and depends on the approved sync #2971. All ten original review threads are
answered and resolved with fixes or dispositions; this does not claim the new
code is in the immutable release.

Sync CI exposed a research-console test race: the run-list title renders before
independent artifact metadata arrives. Controlled delayed-snapshot reproduction
fails with the original synchronous query. The follow-up test waits for artifact
availability and bundle enabled state; all 19 owning tests, ESLint and full
TypeScript pass. No timeout, retry or production change is added. The unchanged
sync reruns only its affected shard; the permanent repair is in this follow-up.
Remote final-head CI and the PR-specific human-summary/merge gate remain pending.

## Second review batch

The thirteen PR2978 comments also cover release changes visible while the base
branch awaits the approved sync. Confirmed repairs cover missing production
Docker profile data, safe structured requeue warning context, readiness response
validation, setup helper documentation, and DSR/CodeQL test boundaries and
categories. Intent comments explain nullish timestamp and native browser-storage
checks. Evidence-backed dispositions retain the HTTP setup projection, existing
shared-package filename convention, and the Qwen local-model confinement guard.

Docker packaging failed its expanded contract before the fix; a wheel built from
production COPY inputs includes all 47 declared profile JSON files afterward.
Both requeue logging regressions failed on missing operation context before the
fix, then all five owning tests passed with no invalid-payload value in logs.
Readiness validation passes all 45 Health tests; the exact CI-compatible API fingerprint and full WebUI TypeScript check pass. Parent combined verification passes 112 tests.
No published artifact, source manifest, legal date or version is changed.


## Final sync review checkpoint

The final nine sync comments are included in the review ledger. Saved-image
normalization reproduces ten failing cases and now passes 57 owning tests with
one official PostgreSQL availability skip. Soak diagnostics/documentation pass
40 tests, with bounded safe categories and Loguru output. No new production
Bandit findings or lint findings were introduced. Duplicate findings and proposed
gate/classifier changes have explicit, evidence-backed dispositions. The sync's
head now passes 71 checks with 39 skips and none pending or failed; final review
replies, approved sync merge and final follow-up CI remain to be completed.
