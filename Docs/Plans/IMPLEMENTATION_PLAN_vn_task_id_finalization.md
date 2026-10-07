# VN Task-ID Finalization Implementation Plan

Backlog: TASK-13515. Requester approved the scoped manual task-ID exception on
2026-10-06. This plan changes tracking and current references only.

## Stage 1: Identify Owned Records
**Goal**: Reserve unique identities without selecting unrelated colliding tasks.
**Success Criteria**: Inspect current dev, all registered worktree inventories,
the exact three VN filenames, and PR3016's verified merged state.
**Tests**: The 136-worktree inventory found maximum root ID 13514. PR3016 is
merged as 1e06e03b587310ec023f3810c4480ea4e69b05f5; all seven required gates passed
on a6250c853bbb875686a3fc793de4c8749cf41868; 102 review threads are resolved.
**Status**: Complete

## Stage 2: Migrate and Finalize
**Goal**: Resolve only the three VN-owned collisions and finalize merged tracking.
**Success Criteria**: TASK-13356 -> TASK-13516, TASK-13358 -> TASK-13517, and
TASK-13369 -> TASK-13518 have matching filenames. Update current VN dependencies
and design/plan associations, preserving every historical section and unrelated
task. Use official Backlog edits for notes, criteria, and finalization after IDs
are unique. Leave other agents' historical plans intact.
**Tests**: Structured task parsing verifies filenames, unique IDs, dependency
targets, original historical-note prefixes, and byte-identical unrelated records.
**Status**: Complete

## Stage 3: Verify and Publish
**Goal**: Publish a separately reviewable documentation-only follow-up.
**Success Criteria**: Scoped diff/format checks and an independent review pass;
runtime/tests/configuration/workflows/dependencies remain identical to dev.
Publish without force or hook bypass. The summary waiver applies only to PR3060;
this new PR needs a requester-written Change summary and current-head hosted
reviews/CI before a normal merge. Remove only this owned plan after completion.
**Tests**: Run canonical Backlog normalization checks, applicable pre-commit
checks, git diff --check, exact changed-path audit, and independent review.
Bandit has no Python changes to scan; do not claim a fresh runtime/security test.
**Status**: In Progress

## Review Follow-up (2026-10-06)

The requester authorized setting Qodo aside and requested a fresh subagent
review. The independent read-only review of the complete published migration
found no actionable issue. CodeRabbit then identified unchecked Definition of
Done entries in the already-completed TASK-13518 record (4202448522). The
official Backlog CLI reconciles all six entries against the retained acceptance,
verification, documentation, security and final-summary evidence. This
documentation-only correction makes no fresh product-test or Bandit claim.

PR3060 merged normally as 7ba48f251ec47a1e0bb680f49f9b7d86ec2b988d. A normal
merge inherits that two-file documentation closeout before publishing this
checklist correction. Stage 3 remains In Progress: the new head needs review
and CI, and PR3207 still requires its own requester-written Change summary.
