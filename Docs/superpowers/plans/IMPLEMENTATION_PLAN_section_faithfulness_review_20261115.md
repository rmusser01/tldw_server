# Section faithfulness and reviewer corrections

Task: TASK-15505.6. Design: Docs/Design/POLITICAL_PDF_VIP_FIXES_2026_11_15.md, follow-up addendum. ADR required: no; reuse ADR-007/065/012. Parent authorized implementation; no paid provider calls or automatic judge.

## Stage 1: Freeze evidence and define honest evaluation
**Goal**: Preserve original failures and prepare reproducible held-out candidates with explicit provenance.
**Success Criteria**: Agent labels are never represented as human labels; source coverage and semantic review are separate; missing labels stay missing.
**Tests**: Fixture validation, exact quote/ID checks, unlabeled and stale fingerprint handling.
**Status**: In Progress — root owns frozen fixture, audit and human-review packet.

## Stage 2: Preserve identities through synthesis
**Goal**: Keep source/author context and prevent citations outside the current supplied evidence.
**Success Criteria**: Bounded reductions retain identity; instructions distinguish quoted voices, agreement/disagreement, agency identity, trends and proposals; checkpoints remain compatible.
**Tests**: RED current-group citation rejection, source/author prompt preservation, generic stance/trend instruction construction; focused section-job suite.
**Status**: Complete — current-call ID scope and generic attribution/stance/trend tests GREEN; 50 section tests.

## Stage 3: Persist and display reviewer corrections
**Goal**: Nontechnical review controls use existing canonical artifact metadata while retaining raw output.
**Success Criteria**: Cold reopen, current ownership, stale target/version conflicts, drafts retained on save failure, separate correction rendering and export; original evidence is unchanged.
**Tests**: Reader interaction, actual persistence request payload/optimistic conflicts/lifecycle guards, export and canonical round-trip.
**Status**: Complete — connected Studio suite 122 GREEN; final focused review guards 30 GREEN.

## Stage 4: Verify and hand off review cases
**Goal**: Focused tests, lint/Bandit and independent review establish implementation behavior without claiming model quality from mocks.
**Success Criteria**: Report exact tests and remaining human-label/live-provider requirements; parent approves any later bounded retest.
**Tests**: Connected frontend/backend suites and deterministic offline evidence audit; no automatic regeneration.
**Status**: In Progress — implementation released for independent review/typecheck and bounded live retest; no provider calls made.
