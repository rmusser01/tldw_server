# PR 3434 CI Monitoring Repairs

Tracked by TASK-15466. The requester approved CI/review repair and gated auto-merge;
the human-written Change summary is explicitly waived, not other quality gates.

ADR required: no. This restores existing test, publication, and shard contracts.
ADR-094 governs shared Jobs ownership; ADR-095 stays Proposed. No new architecture
or supported SDK/platform promise is introduced.

## Stage 1: Diagnose Current Failures
**Goal**: Establish the exact current-head CI failures and actionable feedback.
**Success Criteria**: Reproductions distinguish runtime defects from test/publication omissions.
**Tests**: Failed SQLite race case, docs manifest/strict build, shard coverage guard.
**Status**: Complete

Observed additional complete-CI failures: DB partition expected assignment list,
missing RLS registration for both publication tables, and ten test-quality
offenses across eight tests. These are included in Stage 2 without raising
baselines, adding exclusions, or weakening gates.

## Stage 2: Repair Validated Causes
**Goal**: Make the smallest corrections without weakening any gates or assertions.
**Success Criteria**: Original repros pass; race coverage controls actual transaction boundaries.
**Tests**: Red/green regression, applicable Jobs module, docs suite, all shard matrices.
**Status**: Complete

Evidence: deterministic two-case SQLite race RED (both admissions escaped), then
90-test module GREEN; dependency/active/archive state preserved. Docs 213 passed;
shard guard zero newly uncovered; five files exactly once across six matrices.
Workflow contracts 60 passed after aligning the explicit expected partition list.
Both backend poison gaps reproduced RED and now pass 207 coordination/prune/dependency
tests. Shared RLS repair passed 130 scoped tests, including 16 restricted-role
migration/repair cases. Quality cleanup passed all 350 selected SQLite cases;
the entire CI-contract suite passed 984 cases with six existing skips.

## Stage 3: Review And Publish
**Goal**: Independently review corrections, verify and publish a merge-ready head.
**Success Criteria**: Review findings resolved, evidence recorded, clean scoped commit, auto-merge rearmed with exact head.
**Tests**: Scoped suites, compilation, lint, Bandit, canonical task format and full PR-scoped hooks.
**Status**: In Progress

## Stage 4: Monitor Gated Merge
**Goal**: Follow new CI/review outcomes until GitHub confirms the final state.
**Success Criteria**: Record actual merge SHA/time and stop monitor on MERGED/CLOSED or required user action.
**Tests**: Fresh GitHub state, feedback, effective rulesets, exact head and base verification.
**Status**: In Progress

Automation `monitor-pr-3434-through-merge` is active. Auto-merge is disarmed
until the corrected, independently reviewed and verified head is published.
Actual MERGED state, SHA and time remain pending; do not claim them early.

### Latest Dev Requalification

On 2026-11-08, dev advanced to `7a020da14ab28e220fab652269723e3a2a4abbcb`.
Auto-merge was disarmed before the clean rebase. All 48 prior PR commits remain
patch-equivalent; the recovery branch retains published head `7e44324c2e`.
The dev change removes endpoint-local test-mode predicates and updates their
tests; repository instructions and test/build configuration are unchanged.
All 146 PR-owned paths are byte-identical before and after rebase; none overlaps
the 32 upstream paths. Existing independent reviews remain applicable.

Fresh qualification from the root venv completed:
- Claims/publication/RLS plus both-backend coordination/prune/dependency/TTL:
  1,923 passed, one existing spaCy skip.
- Full Jobs SQLite selection: 2,539 passed, two inherited crypto skips;
  1,560 PostgreSQL-marked cases deselected by the normal SQLite selection.
- Complete CI/Docs/Config, Audit/backup and endpoint regressions: 1,739 passed,
  six existing runner/environment skips.
- Additional official PostgreSQL MediaFiles and cluster publication/snapshot:
  105 passed, zero skips. No PostgreSQL availability skip in either cohort.
- Chroma manager: 33 passed, one legacy skip, 15 inherited xfails, two xpasses.
- Compile 108 files; full configured Ruff 113 versus dev 116 with no new/worse
  signatures; Bandit 58 production files, one identical inherited B608,
  zero errors/new findings; 15 PR-touched task records canonical.

Selections overlap and are not a combined-suite total. An initial invocation
with a nonexistent test path collected no tests; its corrected selection passed.
Bandit JSON was rerun with an explicit output file after its progress-prefixed
stdout defeated the collector. Neither required repository corrections.
The known full-root missing-pg_dump prerequisite was not rerun or called green.

Refetched dev and the remote PR head remain unchanged. All applicable full
146-path staged PR-scoped hooks passed; only no-matching TOML and wizard-only
Ruff/Black hooks skipped normally. Exact-lease publication and exact-head MERGE
rearming follow; their readback is recorded on PR #3434. Monitoring remains
active until GitHub reports the actual final state. No required CI gate or
user-summary waiver is changed.
