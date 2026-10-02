# UAT/E2E Verification Hardening Implementation Plan (Credit Batch 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the test suite able to catch what the 2026-09 UAT found: fix the four E2E assertion/accounting gaps (UAT389-392), promote the test-quality ratchet to a CI gate, fix the sandbox WS teardown hang, and map the unmapped UAT sweep families.

**Architecture:** Test-infrastructure-only changes, ordered by blast radius: collection/discovery first (UAT392 — without it the suite silently runs nothing), then per-acceptance assertion hardening, then the repo-wide ratchet. Backend test infra lives under `tldw_Server_API/tests/`; frontend E2E under `apps/` is edited only in test files, coordinated with the in-flight workstreams.

**Tech Stack:** pytest, Playwright (backend-driven E2E), Vitest, GitHub Actions, Backlog.md.

**Spec:** Backlog tasks `task-13260.278.2` … `.5` (each carries its own AC — read before editing); `audits/2026-07-04-test-quality-triage-report.md` + `Helper_Scripts/ci/test_quality_baseline.txt`; `Docs/Reviews/UAT_ENGINEERING_SWEEP_COVERAGE_2026_09_21.md`; `Docs/Development/Testing_Known_Issues.md`.

## Global Constraints

- `source .venv/bin/activate` first.
- These tasks already exist in Backlog — update them, don't duplicate.
- Test fixes must make failures *loud*; never weaken an assertion to make a suite pass, never disable tests.
- Frontend test-file edits under `apps/**` risk colliding with the two active workstreams: before each edit, `git log --oneline -5 -- <file>` and rebase-check against the in-flight PR lists; when a file is hot, take the backend-side assertion instead.
- Each stage ends with Bandit (touched paths) and a commit referencing the UAT/task ID.

---

## Stage 1: Release test discovery — Playwright collection exits with 0 suites (UAT392, TASK-13260.278.5)

**Goal:** The release E2E collection actually runs the planned cases and reports exact accounting.
**Success Criteria:** Full Playwright collection discovers and runs the intended suites (non-zero); incompatible Vitest files excluded from Playwright discovery; a Skills collection profile exists; planned-vs-run accounting is exact and printed.
**Tests:** The collection run itself + an accounting diff (planned manifest vs executed).
**Status:** Not Started

- [ ] Read `backlog/tasks/task-13260.278.5*.md` for AC, and reproduce the failure: run the release Playwright collection, confirm 0 suites and the incompatible-file errors.
- [ ] Exclude Vitest-style spec files from Playwright discovery (config testMatch/playwright config exclusion — mirror how other profiles scope discovery).
- [ ] Add the missing Skills collection profile.
- [ ] Add planned-case accounting output (planned manifest vs executed; non-empty diff fails).
- [ ] Commit `test(e2e): fix release collection discovery and accounting (UAT392)`.

## Stage 2: Content-Review acceptance can't silently pass (UAT389, TASK-13260.278.2)

**Goal:** Content-Review acceptance fails unless draft actions actually happened.
**Success Criteria:** Acceptance step asserts draft actions exist and fails with a clear message when none occurred; regression test proves the failure mode.
**Tests:** Negative test (no draft actions → suite fails) added next to the happy-path test.
**Status:** Not Started

- [ ] Read `task-13260.278.2*.md`; locate the content-review acceptance step; identify where failures are swallowed (bare try/except, `|| true`, or unasserted awaits).
- [ ] Write the negative regression test first; confirm it currently *passes wrongly* (swallowed).
- [ ] Fix the assertion path; negative test now fails as intended, happy path still green.
- [ ] Commit `test(e2e): content-review acceptance requires draft actions (UAT389)`.

## Stage 3: Canonical-ID and sourced-cards assertions (UAT390 + UAT391, TASK-13260.278.3/.4)

**Goal:** Ingest→Chat acceptance asserts exact source identity and canonical media IDs (UAT390); Notes→study acceptance requires five *distinct* sourced cards and verifies scheduling (UAT391) instead of accepting no-ops.
**Success Criteria:** Both acceptance tests fail on no-op/duplicate outputs; assertions check canonical IDs returned by ingest, not just HTTP 200s.
**Tests:** Strengthened assertions are the deliverable; include a duplicate-card negative case for UAT391.
**Status:** Not Started

- [ ] UAT390: capture canonical media ID at ingest; assert the Chat handoff references exactly that ID (and source identity metadata) — per the task's AC.
- [ ] UAT391: assert ≥5 distinct card IDs (uniqueness check) and the scheduled digest job exists in Jobs; add the duplicate-card negative case.
- [ ] Commit per UAT item.

## Stage 4: Test-quality ratchet → CI gate

**Goal:** Promote the existing test-quality triage from a static baseline to an enforcing ratchet.
**Success Criteria:** CI step fails on any *new* offense beyond `Helper_Scripts/ci/test_quality_baseline.txt`; baseline count monotonically decreases via exemplar fixes; no existing offense mass-fixed in one PR (ratchet only).
**Tests:** The ratchet script itself tested with a synthetic offense fixture.
**Status:** Not Started

Proposed Backlog task: "Promote test-quality ratchet to CI gate" (check for an existing one first — the 2026-07-04 audit may have a follow-up task).

- [ ] Read `audits/2026-07-04-test-quality-triage-report.md` (745 enforceable offenses; baseline in `Helper_Scripts/ci/test_quality_baseline.txt`) and the ratchet script in `Helper_Scripts/ci/`.
- [ ] Fix 3-5 exemplar files (one per offense class: tautology_suspect, stub_injection, status_only) so the fix pattern is documented.
- [ ] Add the CI step (fail on delta > 0 vs baseline) with a tested script.
- [ ] Commit `ci: enforce test-quality ratchet (TASK-<id>)`.

## Stage 5: Sandbox WS teardown hang + sweep automation mapping

**Goal:** Fix the known `tests/sandbox/test_ws_heartbeat_seq.py` teardown hang; produce automation mappings for the six unmapped UAT sweep families (X-01, X-02, S-01…S-06).
**Success Criteria:** Heartbeat test terminates within a deadline (no hang) using the recommended cancellation fix; `Docs/Reviews/UAT_ENGINEERING_SWEEP_COVERAGE_2026_09_21.md` (or successor) lists a concrete automation target (existing suite / new test / explicitly-manual) for each unmapped family.
**Tests:** The heartbeat test with `WS_SHUTDOWN_TIMEOUT_MS` deadline; mapping is documentation with review sign-off.
**Status:** Not Started

- [ ] Implement the documented recommendation from `Docs/Development/Testing_Known_Issues.md`: task cancellation with deadline + `WS_SHUTDOWN_TIMEOUT_MS` env (previously unowned).
- [ ] For each unmapped family in the sweep coverage doc, write the mapping row with the owning suite path or `manual:<reason>`; don't invent automation for families that genuinely need a human.
- [ ] Commit `test: fix sandbox WS teardown; map UAT sweep automation gaps`.
