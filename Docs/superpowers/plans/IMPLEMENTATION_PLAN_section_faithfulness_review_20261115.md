# Section faithfulness and reviewer corrections

Task: TASK-15505.6. Design: Docs/Design/POLITICAL_PDF_VIP_FIXES_2026_11_15.md, follow-up addendum. ADR required: no; reuse ADR-007/065/012. User authorized implementation and bounded real DeepSeek calls. No automatic judge or invented human labels.

## Stage 1: Freeze evidence and define honest evaluation
**Goal**: Preserve original failures and prepare reproducible held-out candidates with explicit provenance.
**Success Criteria**: Agent labels are never represented as human labels; source coverage and semantic review are separate; missing labels stay missing.
**Tests**: Fixture validation, exact quote/ID checks, unlabeled and stale fingerprint handling.
**Status**: Complete — 13 frozen cases (originally five regression, eight held-out), source evidence and human-review packet recorded. H05 subsequently informed the page-boundary fix and now counts as regression evidence. Human labels remain pending.

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
**Success Criteria**: Report exact tests, bounded live-provider results and remaining human-label requirements. Preserve failed outputs and distinguish agent source audits from human judgments.
**Tests**: Connected frontend/backend suites and deterministic offline evidence audit; no automatic regeneration.
**Status**: In Progress — implementation checks and authorized fixed replay completed: 13 structurally valid direct-context answers (agent audit: 11 pass, one attribution failure, one context-limited), plus 30-selected-chapter synthesis with an unsupported disagreement. 41 calls used 172,213 tokens; no retry sampling. Original extraction checkpoints and failures retained. Human source labels and semantic acceptance remain pending; see Docs/Reviews/POLITICAL_PDF_CLAIM_REVIEW_2026_11_15.md.
