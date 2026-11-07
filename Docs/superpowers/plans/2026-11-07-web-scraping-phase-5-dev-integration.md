# Phase 5 Dev Integration

**Task:** TASK-15452. This is integration of the already approved implementation,
not a new refactor or feature. Preserve analyzer, compatibility and security
behavior. No new architecture decision; the existing ownership ADR is renumbered
from 093 to 094 only to avoid an unrelated open-PR record. Preserve its accepted
decision and rationale.

## Stage 1: Preserve And Rebase
**Goal:** Integrate fetched dev `dfac3ca7f34a296a2dc3a3fbc340efdd1a0bae5c`.
**Success Criteria:** Backup retains original HEAD `51064ceeb9`; no unrelated task
or ADR is overwritten; upstream egress tightening and summary-quota tests survive.
**Tests:** Task-ID/ADR namespace scan, canonical task checks, diff review.
**Status:** In Progress.

## Stage 2: Verify Integration
**Goal:** Verify the rebased source and actual consumers.
**Success Criteria:** Scraping/shared HTTP, native browser, selected service and
consumer gates pass with explicit skips; lint/compile/Bandit show no new issues.
**Tests:** Reuse the accepted final certification commands and affected upstream
egress and summary tests; record failures rather than suppressing them.
**Status:** Not Started.

## Stage 3: Record And Hand Off
**Goal:** Commit accurate integration evidence and preserve a clean local branch.
**Success Criteria:** Task updated, normal hooks pass, main checkout untouched;
publication occurs only if separately approved. No automatic merge; future merge
still requires the requester's own Change summary.
**Tests:** Canonical records, duplicate IDs, links, diff check and clean status.
**Status:** Not Started.
