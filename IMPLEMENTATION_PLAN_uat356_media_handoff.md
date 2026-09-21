# UAT356 Media handoff implementation plan

**Task:** TASK-13260.277.6
**Goal:** Deliver media source text only to the initiating tab's intended Chat destination and preserve unrelated and destination drafts.
**Approved design:** Store each full payload under an opaque token in native `window.sessionStorage`; put only that token in the Chat route. Validate owner before consumption. Existing destination text requires explicit insert, replace, or cancel, following the sidepanel import controls. Shared singleton delivery and untargeted events must no longer seed a composer. Extension Quick Ingest outside options retains its new-options-tab route using owner-bound per-token shared storage with expiry and Web Locks; only the route token addresses the record.
**Constraints:** No shared runtime mutations, dependencies, global tracker edits, or git staging/commits. Keep task In Progress for root-coordinated native PostgreSQL acceptance. Other agents own sidepanel and backend Retry changes.

## Stage 1: Causal regression
**Goal:** Capture the cross-tab draft overwrite before implementation.
**Success Criteria:** A mounted Character composer cannot consume or clear a Media tab's pending payload; expected failure against old consumer.
**Tests:** Extend PlaygroundForm handoff tests with a same-owner unrelated draft and forced reader rerender. Add tab-storage service cases for separate simultaneous handoffs, exact full text, and owner checks.
**Status:** Complete

## Stage 2: Tab-owned delivery and conflict handling
**Goal:** Replace shared delivery with opaque route tokens in `services/tldw/media-chat-handoff.ts` and `components/Option/Playground/PlaygroundForm.tsx`.
**Success Criteria:** Only route token and tab storage match may seed text; unknown owner waits, mismatch cannot expose content, consumed tokens cannot replay. Nonempty destination draft requires explicit action.
**Tests:** Owner unresolved/mismatch and A-B-A invalidation; repeated render/remount; destination insert/replace/cancel; normal versus RAG source intent.
**Status:** Complete

## Stage 3: Producer migration
**Goal:** Migrate ViewMediaPage, ReviewPage, useMediaReviewActions, useIngestResults, and option-index to the same delivery helper.
**Success Criteria:** All source actions navigate with only a token; no global plaintext pending source or untargeted composer mutation; preserve full source and selected media IDs.
**Tests:** Existing ViewMediaPage permalink/chat action, Home first-source, quick-ingest, and Media review handoff cases updated to assert destination behavior.
**Status:** Complete

## Stage 4: Verification and handoff
**Goal:** Review touched diff and run installed Vitest/TypeScript/ESLint entrypoints; document Bandit's non-Python applicability.
**Success Criteria:** Focused suites pass with recorded red/green evidence; no new touched-file type/lint errors. Report native acceptance as pending.
**Tests:** `node node_modules/vitest/vitest.mjs run <focused tests> --maxWorkers=1 --no-file-parallelism` from apps/packages/ui; installed frontend lint and type entrypoints; `git diff --check`.
**Status:** In Progress

## Verification evidence

- Causal regression reproduced the unrelated Character draft overwrite before the repair (`/tmp/uat356-causal-red.log`); token-service missing-function red evidence is `/tmp/uat356-service-red.log`.
- Review-driven delayed acceptance and producer lifetime regressions failed before their guards (`/tmp/uat356-manual-unmount-red.log`, `/tmp/uat356-producer-lifetime-red.log`).
- Final direct and surrounding regression selection: **14 suites, 285 tests passed** (`/tmp/uat356-final-tests.log`). Independent review: **10 lifecycle/owner/hydration/conflict cases passed**, no remaining reviewed blocker (`/private/tmp/uat356-review-lifecycle-green.log`).
- Nine wider Review fixture failures reproduce with all seven changed production modules overridden from df1245d7fd (`/tmp/uat356-surrounding-baseline.log`); no edits to those unrelated fixture files.
- ESLint comparison uses the same config on current and df1245d7fd source: **zero added diagnostics**, 3 existing errors, warnings decrease307→299 (`/tmp/uat356-lint-comparison-final.json`). Diff whitespace check passed.
- Bandit touched-directory run completed with0 Python LOC (`/tmp/bandit_uat356.json`); it supplies no TypeScript security coverage.
- Two pre-existing UAT357 integration fixture mismatches were repaired after source-comparison proof: verified Cedar owner key and real account invalidation preserving a replacement-owner sentinel.
- Native PostgreSQL multi-tab acceptance remains assigned to root. Backlog stays In Progress, with acceptance criteria1–2 checked and3 pending.
- Matched frontend TypeScript final/baseline: **93 diagnostics each, zero additions or removals**, none in delegated files (`/tmp/uat356-frontend-type-comparison-final.json`). Global typecheck remains nonzero for those baseline diagnostics.
