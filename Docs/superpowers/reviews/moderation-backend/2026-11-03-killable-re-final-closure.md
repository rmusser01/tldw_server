# Moderation Isolation Final Review Closure

Date: 2026-11-03. Associated task: TASK-15477 (previously TASK-15417/15466).
Source revision: `6b9d2c9b30d134c354cccc191397decdf8f510a3`.
Branch: `codex/moderation-ci-triage-regex-design`.
This records implementation completion, not unconditional merge or platform approval.

## Outcome

Runtime matching and redaction use stdlib re in bounded, parent-owned killable
children. Existing result/import/override contracts remain, with fail-closed
current Chat, Guardian, Audio and Workflow integration, terminal stream handling,
metadata-only operational incidents and transactional neutral-phase review support.
The requester-approved Audio amendment batches independent fields rather than
launching an interpreter per segment; no deadline or aggregate size cap widened.

All five task reviews completed. Whole-branch review found F1-F3 and M1-M2.
The fix wave addressed F1/F3/M1/M2; independent re-review retained F2 for a
cancellation-ordering gap. The requester explicitly approved one targeted follow-up,
and its fresh scoped review accepted F2 with no new fix-diff findings.

| Finding | Final Disposition |
| --- | --- |
| F1: ordinary segmented Audio times out | Approved independent-text batch, max 4,096 fields, same two-second scope and aggregate caps; accepted at `0233b7ba09`. |
| F2: cancelled unconfirmed cleanup is silent | Cleanup owner attempts one sanitized incident for any registered quarantine, independent of exception/cancellation order; accepted at `6b9d2c9b30`. |
| F3: unavailable incident DB prevents startup | Sanitized degraded preparation retains the execution shutdown owner; incident writes do not migrate. Accepted. |
| M1: installed ASCII test is nondiscriminating | Actual installed-child Unicode/ASCII distinction, retaining MULTILINE/DEBUG checks. Accepted. |
| M2: branch-added finally-return warning | Removed without changing resource retirement; closed-loop and warnings-as-errors regression. Accepted. |

The F2 follow-up preserves CancelledError, the original typed failure, actual
quarantine/reaper/offload leases, aggregate metadata, privacy, independent bounded
reporting and shared report-once suppression. Healthy cancellation stays quiet.
No queued retry, extra reporting transport, or early resource release was added.

## Verification

Fresh controller checks at source revision above:

- Async execution, installed-wheel packaging and task-format suites:
  **44 passed, 100 warnings**, exit 0, 22.58 seconds.
  Log: `/tmp/moderation-final-controller-verification.log`.
- All cumulatively touched backend Python paths compile with SyntaxWarning
  treated as error, exit 0.
- Cumulative touched production Bandit: **0 findings, 0 scan errors**, exit 0.
  JSON: `/tmp/bandit_moderation_final_controller.json`.
  Existing unused-nosec advisories are visible; this is not pristine log output.

Final F2 implementer evidence at the same source revision: 151 affected host
tests, 15 independent-containment tests, and 12 postcommit selected checks pass.
These overlap; they are not summed as unique coverage. Touched Ruff and
production/test Bandit are clean, with only B101 excluded in test scanning.

Revision-qualified earlier evidence: 917 affected host nodes and 329 selected
Linux 3.12 source nodes passed for the Audio fix wave. Linux selection overlaps
imported test nodes and is not disjoint coverage or an installed Linux wheel.
The real warmed 100-segment/5,099-character payload took 43.714 ms with one actual
child and no retained ownership; a Linux 102-field probe took 118.466 ms with
one child and verified AS/CPU/core posture. Those are measured samples, not an SLA.

Before the final fix wave, the broader focused manifest passed 2,049 nodes with
two preexisting skips. Shared UI tests passed 37; mocked browser flows passed 3
across desktop/mobile widths, and frontend typecheck passed. Current F2 changes
do not alter those UI or worker/protocol paths, but those runs are not relabelled
as fresh all-current-head suites. See the detailed validation record:
[Release Evidence](2026-11-02-killable-re-validation.md).

## Remaining Integration Gates

- Windows runtime remains unverified. Latest F2 execution tests are host macOS;
  Python 3.12 source compilation is not a fresh Linux/Windows runtime run.
- Whole-project collection remains blocked by missing `httpx2` in the unrelated
  Workspaces suite; no all-project pass is claimed.
- Existing broader Ruff/test-security heuristics and dependency/configuration
  warnings remain explicitly triaged; narrow clean scans do not erase them.
- Current-dev reconciliation, required CI statuses and a human-owned Change
  summary are still needed before an AI-authored PR can merge. Nothing was
  pushed, published or merged by this work.
- Neutral-phase rollout requires compatible quiesced readers/writers, backup,
  pre-traffic preparation and compatible rollback; do not drop or relabel rows.
- Previously emitted stream bytes cannot be recalled. Existing compiler/admin
  lint isolation, new LLM-surface coverage and legacy Workflow accessor repair
  remain outside this slice.

ADR assessment: existing accepted ADR-097 and ADR-021 govern; their rationale
was not rewritten. The approved design and Audio amendment remain under
`Docs/superpowers/specs/`. Completed controller-owned implementation plans are
removed per AGENTS; their committed history and local evidence archive preserve
the record. All other plans/worktrees are left alone.

Local SDD evidence archive:
`/tmp/moderation-15417-sdd-evidence-2026-11-03.tar.gz`.
Its gzip integrity and complete inventory were verified before deleting only
this plan's 53 ignored scratch files and its two completed owned plans.
SHA256: `49e531b87673e9ed983b810d6cfe486294b89ca441174dae4b1db339f54c2992`.

## Rulings I Made

Every ledger `Ruling:` entry is represented below with its cost if wrong.
The requester separately approved the Audio design and the final targeted F2
follow-up; neither was silently implemented past an approval boundary.

1. Inherited agent models, as tool policy requires; cost: potentially higher cost.
2. Deferred rebasing until integration; cost: later conflict/revalidation work.
3. Renumbered only this branch's colliding task; cost: historical reference repair.
4. Kept baseline warnings visible; cost: warning noise remains.
5. Corrected the version test to use a valid frame; cost: test/plan adjustment.
6. Added schema-aware mutations with the count fix; cost: broader protocol tests.
7. Applied the repo's three-attempt cap; cost: stop/reassess rather than blind retries.
8. Reviewed trusted subprocess Bandit heuristics without suppressing them; cost:
   findings remain visible and require contextual review.
9. Required actual platform evidence; cost: Windows remains unverified.
10. Used existing Bun/Vitest instead of pnpm; cost: equivalent command substitution.
11. Added the new moderation worker to exact inventory expectations; cost: one
    adjacent test file, exact equality retained.
12. Corrected already-stale Audio catalog expectations; cost: test-only scope growth.
13. Corrected already-stale Notes expectations; cost: test-only scope growth.
14. Closed review-store SQLite connections deterministically; cost: narrow lifetime
    correction under existing transaction semantics.
15. Isolated lifecycle review preparation in temporary DBs; cost: test fixture change.
16. Treated post-kill signal-zero EPERM as unknown until the existing deadline;
    cost if wrong: delayed refusal, never credited cleanup or released ownership.
17. Made worker evaluation decision-only and covered indirect Audio boundaries;
    cost: internal batch correction and adjacent typed-failure branches.
18. Required decision responses to carry no eager redaction; cost: protocol/test
    invariant correction, unchanged wire fields/version.
19. Extended typed failure handling through real Workflow controls; cost: broader
    engine/control regressions, unrelated retry behavior preserved.
20. Clarified the execution-scope deadline, not end-to-end cold call latency;
    cost if stricter embedding latency is required: separately reviewed setup work.
21. Kept the residual cleanup race blocking and requested another explicit approval;
    cost: an extra targeted fix/review, not a silently waived observability gap.
