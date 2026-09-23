# UAT372: Keep sidepanel first-use guidance clear of controls

Task: TASK-13260.277.23. Root approved the bounded design on 2026-09-20.

## Stage 1: Reproduce and specify
**Goal**: Preserve the native obstruction evidence and add causal coverage with the real FeatureHint and ControlRow.
**Success Criteria**: Fresh hints appear one at a time in normal flow; dismissal remains persistent and keyboard accessible without submitting the composer.
**Tests**: Real-component tests for fresh/seen/disconnected states, keyboard dismissal, focus and existing menu Escape; native PG evidence `pg-multi-ext-review366-013-model-list.txt` records the actual paragraph intercepting the model picker.
**Status**: Complete

## Stage 2: Minimal repair
**Goal**: Render one compact inline hint before the control buttons, retaining wording, setting keys and eligibility.
**Success Criteria**: No floating hint geometry; next hint gets independent dismissal state; model and More tools keyboard behavior remains intact. Existing shared Alert may be reused without changes.
**Tests**: New causal tests plus surrounding ControlRow and Alert tests.
**Status**: Complete

## Stage 3: Verify and freeze
**Goal**: Record scoped checks and exact review snapshot for root.
**Success Criteria**: Focused/surrounding suites pass, no new lint/type/security diagnostics in touched code, source snapshot supplied for independent review. Native narrow/wide acceptance remains root-owned.
**Tests**: Focused Vitest, scoped ESLint, matched TypeScript comparison as needed; Bandit applicability documented for TS-only edits.
**Status**: In Progress

## Scope and constraints

- Production: `apps/packages/ui/src/components/Common/FeatureHint.tsx` and `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx`.
- No preserved PostgreSQL profile, frozen candidate, shared runtime, root tracker or git mutations.
- Use existing `tldw:seenHints` keys `knowledge-search` and `more-tools`; knowledge remains eligible only when connected.
- Unit DOM tests cannot prove browser hit testing. Root must validate fresh visible hints at narrow and wide native sizes, including model selection without dismissal and dismissal persistence after reload.

## Verification and source freeze

- Native causal evidence: `.tmp/uat-frontend-repair1-20260920/pg-multi-ext-review366-013-model-list.txt` explicitly identifies the More tools paragraph intercepting the model-picker click.
- Real FeatureHint integration before repair: 2 failed / 6 passed (`/tmp/uat372-causal-red.log`): simultaneous hints and keyboard Dismiss submitting the enclosing composer form.
- Focused green: 8 passed (`/tmp/uat372-causal-green.log`). Final surrounding ControlRow/Alert verification: 4 suites / 26 passed (`/tmp/uat372-final-tests.log`).
- Scoped ESLint: 0 errors / 0 warnings in all three changed TSX files (`/tmp/uat372-lint-final.log`). The lint runner emits an existing repository pages-directory configuration notice.
- Matched TypeScript compiler/config comparison with only the two production files replaced by their pre-edit snapshots: 97 baseline / 97 current diagnostics; no added, removed or touched-file diagnostics (`/tmp/uat372-type-comparison.json`).
- Security review: no auth, network, storage format or HTML injection changes. Existing typed settings persistence remains unchanged. Bandit is not applicable to these TypeScript-only edits; no Python changed.
- Formatter package was unavailable in the UI installation; changed blocks were formatted manually, with no broad file reformat.
- Review artifact: `/tmp/uat372-review.patch`; exact source hashes: `/tmp/uat372-source-manifest.txt`.
- Stage 3 remains in progress for independent review and root-owned native narrow/wide acceptance. No shared runtime, preserved PG profile, frozen candidate or git mutation occurred.

Root independent review is clear; 8 focused tests independently passed (`/tmp/uat372-root-review.log`). Source remains unchanged. Final candidate2 native narrow/wide acceptance remains pending.
