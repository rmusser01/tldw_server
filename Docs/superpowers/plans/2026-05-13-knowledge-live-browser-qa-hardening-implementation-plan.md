# Knowledge Live-Browser QA Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and document a verification-first live-browser QA hardening pass for `/knowledge` in PR #1617, fix only observed `/knowledge` regressions or low-risk friction that belongs in the current PR, and create an evidence-based product-expansion issue for deferred source-picker organization/sharing work.
**Architecture:** Evidence-first QA gate around the existing WebUI, extension, shared KnowledgeQA components, and current Playwright/Vitest harnesses. Reuse existing seeded tests and browser infrastructure before adding coverage. Keep current-PR fixes small, local, and test-backed; move product expansion to a follow-up issue.
**Tech Stack:** Next.js WebUI, WXT extension, React/TypeScript shared UI, Playwright, Vitest, Backlog.md, GitHub CLI.

---

## Reference Documents

- Design spec: `Docs/superpowers/specs/2026-05-13-knowledge-live-browser-qa-hardening-design.md`
- Backlog task: `TASK-297.6`
- Current PR: `https://github.com/rmusser01/tldw_server/pull/1617`

## Scope Rules

- Keep this plan limited to `/knowledge` and flows directly reachable from it.
- Treat `/knowledge` as QA-only, not the canonical creation/import/management hub.
- Include WebUI and extension behavior where routes exist or differ.
- Do not implement saved-view sharing, profile sharing/export/import, advanced source organization, or backend sharing APIs in PR #1617.
- Do not expose or commit private local database content, screenshots, titles, prompts, chats, notes, document text, or source excerpts.
- Fix in this PR only when live QA finds a clear `/knowledge` regression, accessibility blocker, viewport break, misleading copy, or deterministic source-picker bug.

## File Map

- `Docs/superpowers/plans/2026-05-13-knowledge-live-browser-qa-evidence.md`
  - New committed evidence artifact for the synthetic/seeded QA matrix, privacy-safe local real-data notes, triage decisions, and final PR QA summary.
- `backlog/tasks/task-297.6 - Plan-knowledge-live-browser-QA-hardening-for-PR-1617.md`
  - Keep acceptance criteria, implementation plan link, notes, verification, and final summary current.
- `apps/tldw-frontend/e2e/smoke/stage6-interaction-stage2.spec.ts`
  - Existing WebUI seeded `/knowledge` smoke coverage; extend only if browser QA finds a deterministic regression that belongs here.
- `apps/tldw-frontend/e2e/smoke/m3-2-a11y-focus-evidence.spec.ts`
  - Existing focus evidence harness; reuse before adding a new a11y/focus test.
- `apps/extension/tests/e2e/quick-chat-guides-tutorials.spec.ts`
  - Existing extension `/knowledge`, `/knowledge/thread/:threadId`, and `/knowledge/shared/:shareToken` route evidence.
- `apps/extension/tests/e2e/knowledge-rag-ux.spec.ts`
  - Older extension knowledge workspace coverage; use as supplemental evidence only if still runnable against current routes.
- `apps/tldw-frontend/__tests__/extension/knowledge-route-parity.test.ts`
  - Existing static route parity check.
- `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/`
  - Existing focused KnowledgeQA unit/component tests; add coverage here for any source-picker/profile/filter defects fixed in this PR.
- `apps/packages/ui/src/components/Option/KnowledgeQA/`
  - Current-PR UI fix target only if QA identifies a local defect.
- `/private/tmp/pr1617_knowledge_product_expansion_issue.md`
  - Temporary issue body for the product-expansion GitHub issue; do not commit if it contains environment-specific notes.

---

## Task 1: Create Evidence Artifact And Sync Task Metadata

**Goal:** Establish the execution record before QA begins.
**Status:** Complete.

- [x] Create `Docs/superpowers/plans/2026-05-13-knowledge-live-browser-qa-evidence.md` with sections for summary, environment, seeded matrix, extension matrix, local real-data pass, keyboard pass, findings, fix decisions, product-expansion issue, verification commands, and skipped rows.
- [x] Include the design-spec evidence table fields: surface, route, viewport, data profile, task, result, evidence, and decision.
- [x] Define the exact required QA matrix in the evidence artifact before running browser tests. At minimum, pre-create rows for:
  - [x] Empty or first-run state.
  - [x] Seeded realistic library with media, notes, chats, characters, task boards, prompts, world books, and dictionaries where current fixtures can expose them.
  - [x] Sources with unavailable or empty status.
  - [x] Weak or no-result retrieval.
  - [x] Workspace-scoped artifacts hidden globally and visible only under explicit workspace scope.
  - [x] Web fallback disabled.
  - [x] Web fallback enabled, with the server default provider/privacy disclosure visible or clearly absent as a finding.
  - [x] One privacy-safe local real-data pass.
- [x] For any required data profile that current seeded fixtures cannot support, leave the matrix row in place and mark it `Blocked` with the missing fixture/runtime reason during execution.
- [x] Add privacy rules directly to the evidence artifact so local real-data notes stay redacted.
- [x] Update `TASK-297.6` with this implementation plan path and the evidence artifact path.
- [x] Record the current branch, PR URL, and baseline commit in the evidence artifact.
- [x] Record runtime setup decisions in the evidence artifact:
  - [x] WebUI Playwright default URL: `http://localhost:8080`.
  - [x] WebUI Playwright default command from `apps/tldw-frontend/playwright.config.ts`: `bun run dev -- -p 8080`.
  - [x] WebUI API default: `http://127.0.0.1:8000` unless overridden by `NEXT_PUBLIC_API_URL`, `TLDW_SERVER_URL`, or `TLDW_E2E_SERVER_URL`.
  - [x] Manual browser URL for this pass: `http://localhost:8080/knowledge`, or the actual alternative URL if port 8080 is unavailable.
  - [x] Seeded fixture source: existing Playwright mocks/harness data, test-only mocks, or explicit blocker if no current fixture supports the required row.
- [x] Run `git diff --check`.
- [x] Run `rg -n "[^[:ascii:]]" Docs/superpowers/plans/2026-05-13-knowledge-live-browser-qa-evidence.md "backlog/tasks/task-297.6 - Plan-knowledge-live-browser-QA-hardening-for-PR-1617.md"` and replace any accidental non-ASCII text.

**Success Criteria:**

- [x] The evidence artifact exists and can be filled incrementally during QA.
- [x] Required data profiles are predeclared before tests begin, so skipped coverage is explicit rather than accidental.
- [x] The Backlog task links the approved design spec, this implementation plan, and the evidence artifact.
- [x] The repository diff is whitespace-clean.

---

## Task 2: Run Existing Seeded Baseline Harnesses

**Goal:** Reuse current deterministic coverage before adding tests or doing manual browser exploration.
**Status:** Complete.

- [x] From `apps/packages/ui`, run:
  - `bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.scalable-source-picker.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.profiles.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx src/components/Knowledge/__tests__/KnowledgePanelTabRouting.test.tsx src/services/rag/__tests__/sourceMetadata.test.ts`
- [x] From `apps/tldw-frontend`, run:
  - `bunx vitest run __tests__/extension/knowledge-route-parity.test.ts`
- [x] From `apps/tldw-frontend`, run the existing `/knowledge` seeded interaction smoke row:
  - `bunx playwright test e2e/smoke/stage6-interaction-stage2.spec.ts -g "search typing and deterministic no-results answer remain functional" --reporter=line`
- [x] From `apps/extension`, run the extension route tutorial-card coverage:
  - `bunx playwright test tests/e2e/quick-chat-guides-tutorials.spec.ts -g "knowledge tutorial card" --reporter=line`
- [x] If a harness is blocked by dependencies, environment, or stale assumptions, record the exact blocker and do not silently replace it with weaker evidence.
- [x] Record pass/fail/blocker results in the evidence artifact.

**Success Criteria:**

- [x] Existing seeded WebUI, shared UI, and extension checks are recorded before any new QA-driven fix.
- [x] Failures are triaged as current-PR fix candidates, follow-up candidates, or environment blockers.

---

## Task 3: Run WebUI Seeded Live-Browser Matrix

**Goal:** Observe the actual `/knowledge` page across realistic viewport sizes using repeatable seeded/synthetic data.
**Status:** Complete.

- [x] Confirm the backend/API mode for the pass:
  - [x] If using the local backend, verify `http://127.0.0.1:8000/api/v1/health` and record the result.
  - [x] If using mocked/seeded frontend data only, record the mock source and do not claim real backend coverage.
- [x] Start the WebUI/dev stack for manual browser QA from `apps/tldw-frontend` with `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev -- -p 8080`, unless that port/API is unavailable.
- [x] If port 8080 or the default backend is unavailable, set and record the actual `TLDW_WEB_URL`, `TLDW_WEB_CMD`, and `NEXT_PUBLIC_API_URL` values used for the pass before continuing.
- [x] Prefer the Browser plugin for live interaction if available; otherwise use the existing Playwright CLI wrapper or direct Playwright tests and record the fallback reason.
- [x] Visit WebUI `/knowledge` at these viewports:
  - [x] 1440 x 900
  - [x] 1280 x 720
  - [x] 1024 x 768
  - [x] 390 x 844
- [x] For each viewport, exercise:
  - [x] First arrival and ready/empty state comprehension.
  - [x] Add Sources discovery and handoff visibility.
  - [x] Source category selection across canonical source types that seeded data exposes.
  - [x] Unavailable/empty source status clarity.
  - [x] Specific source picker query filter.
  - [x] Source status filter.
  - [x] Recent imports filter.
  - [x] Workspace filter and hidden-artifact behavior.
  - [x] Bulk select visible, clear visible, and select recent imports.
  - [x] Local saved profile save/load behavior.
  - [x] Simple/Detailed toggle.
  - [x] Local-only QA query.
  - [x] Web fallback disabled.
  - [x] Web fallback enabled with server default provider/privacy disclosure checked.
  - [x] Weak/no-result query and recovery.
  - [x] Citation/source card inspection.
  - [x] Continue in editor handoff reachability.
- [x] Check browser console and network errors during each run.
- [x] Capture screenshots only for synthetic/seeded states that are safe to commit or reference.
- [x] Record each row in the evidence artifact with result and decision.

**Success Criteria:**

- [x] The planned WebUI viewport matrix is complete, or every skipped row has a concrete blocker.
- [x] No P0/P1 WebUI `/knowledge` findings remain untriaged.

---

## Task 4: Run Extension Route And Viewport Pass

**Goal:** Verify extension `/knowledge` behavior and route parity without turning the extension into a separate product redesign.
**Status:** Complete.

- [x] Use the extension E2E harness to verify:
  - [x] `#/knowledge`
  - [x] `#/knowledge/thread/:threadId`
  - [x] `#/knowledge/shared/:shareToken`
- [x] Confirm the extension route exposes the expected tutorial/card or knowledge UI for each route and does not regress route state.
- [x] Check extension-sized behavior for overflow, hidden primary actions, and unreachable source controls.
- [x] Record any difference between WebUI and extension behavior in the evidence artifact.
- [x] If the older `knowledge-rag-ux.spec.ts` harness still maps to shipped behavior, run it as supplemental evidence; otherwise record it as legacy/unverified instead of treating it as current product evidence.

**Success Criteria:**

- [x] Extension `/knowledge` routes are verified or clearly marked blocked/legacy with evidence.
- [x] Any WebUI/extension mismatch is triaged for current-PR fix or follow-up.

---

## Task 5: Run Privacy-Safe Local Real-Data Pass

**Goal:** Catch scale and realism issues without leaking private content or mutating user data.
**Status:** Complete with privacy-blocked query, answer, and citation inspection documented. Backend/database read-only; one temporary saved profile was created and loaded only in isolated Playwright browser storage.

- [x] Identify the local database/profile to use and prefer a copied or sanitized profile over live mutable databases.
- [x] If a copied/sanitized profile is not practical, record that live local data is being used and avoid backend/database create, edit, delete, reindex, export, or share flows; the saved-profile check used isolated browser storage only.
- [x] Run at least one WebUI `/knowledge` pass against local real data at:
  - [x] 1440 x 900
  - [x] 390 x 844, unless the local environment cannot support it.
- [x] Exercise source picker scale, filtering, status clarity, and browser-local profile persistence reachability; weak/no-result recovery plus citation/source card inspection against private local data were blocked and documented because they create/persist QA state and expose private content.
- [x] Record only privacy-safe behavioral observations: source type, workflow, viewport, symptom, and whether it can be reproduced with synthetic data.
- [x] Keep private screenshots and raw notes outside git. Do not quote private titles, prompts, chats, notes, source excerpts, or document text.
- [x] If a real-data issue needs a current-PR fix, first reproduce it with seeded/synthetic data or a mocked unit/component test.

**Success Criteria:**

- [x] At least one local real-data pass is recorded with no private content committed.
- [x] Real-data-only observations are either reproduced synthetically or deferred with a clear privacy-safe summary.

---

## Task 6: Run Keyboard And Power-User Friction Pass

**Goal:** Determine whether the current source picker and QA workflow need same-PR accessibility fixes or later product work.
**Status:** Complete. Seeded keyboard probe covered desktop and mobile workflows; direct browser Escape closure was inconclusive because Playwright delivered 0 Escape keydown events to the page, so the Escape defect was verified with focused Vitest red/green coverage.

- [x] Test Tab order through search, source menus, filters, bulk actions, saved profiles, settings, answer actions, and source cards.
- [x] Test Enter behavior for search submission and source filtering.
- [x] Test Escape behavior for menus, dialogs, and pickers.
- [x] Verify focus return after closing source picker, settings, source viewer, and existing export/share dialogs if present.
- [x] Verify keyboard access to Simple/Detailed mode.
- [x] Repeat a query with modified source scope.
- [x] Save a local profile, load it, and update the profile-like workflow using existing behavior.
- [x] Run bulk source actions after filtering.
- [x] Navigate citations/source cards with keyboard where controls exist.
- [x] Classify findings using the design spec's P0/P1/P2/P3 gates.

**Success Criteria:**

- [x] Keyboard blockers and repeated-action friction are severity-ranked.
- [x] Current-PR fixes are limited to clear defects or low-risk affordance gaps.
- [x] Larger keyboard/power-user workflow ideas are deferred to follow-up unless they block current usability.

---

## Task 7: Fix Only Allowed Current-PR Issues

**Goal:** Apply minimal, test-backed fixes only for defects proven by Tasks 2-6.
**Status:** Complete. Fixed the P2 Escape handler risk with capture-phase handlers in `KnowledgeContextBar` and `SettingsPanel`, and fixed the P3 duplicate settings accessible name in `CompactToolbar`; no product-expansion feature work was added.

- [x] For each P0/P1 finding, decide whether it must be fixed before merge or blocks the PR.
- [x] For each P2 finding, fix only if the change is small, local, and low-risk; otherwise document a follow-up rationale.
- [x] For each P3 finding, document unless the fix is trivial and clearly safe.
- [x] For deterministic UI/source-picker defects, add or update a focused test in `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/` before changing implementation.
- [x] For deterministic WebUI route defects, add or update the smallest relevant Playwright smoke coverage.
- [x] For extension route defects, update the focused extension E2E route test if the behavior is stable enough for automation.
- [x] Keep implementation edits inside `/knowledge`, shared KnowledgeQA, source metadata, or directly affected route glue.
- [x] Do not add saved-view sharing, profile export/import, advanced source grouping, backend sharing APIs, or knowledge CRUD/import hub behavior.
- [x] Re-run the focused tests that cover each fix.
- [x] If Python production files are touched, run Bandit on the touched Python scope and record results.

**Success Criteria:**

- [x] Every current-PR fix has an observed finding, a focused test or documented manual verification, and an evidence-artifact entry.
- [x] No product-expansion feature work enters PR #1617.

---

## Task 8: Create Product-Expansion GitHub Issue

**Goal:** Preserve future saved-view/profile-sharing/advanced-organization work without expanding the current PR.
**Status:** Complete. Created https://github.com/rmusser01/tldw_server/issues/1631 after QA synthesis.

- [x] After QA synthesis, write `/private/tmp/pr1617_knowledge_product_expansion_issue.md`.
- [x] Include:
  - [x] Evidence summary from PR #1617 QA.
  - [x] Workflows that did or did not show saved-view/profile-sharing/organization friction.
  - [x] Proposed scope for saved views, profile sharing/export/import, advanced source organization, workspace grouping, or keyboard shortcuts only when evidence supports it.
  - [x] Non-goals: canonical CRUD/import hub, automatic web fallback recommendation, generated/test/workspace artifacts by default, backend sharing APIs unless explicitly required by product scope.
  - [x] Accessibility requirements.
  - [x] Privacy/data ownership requirements for sharing or export behavior.
  - [x] Acceptance criteria for a later implementation slice.
- [x] Create the GitHub issue with `gh issue create --repo rmusser01/tldw_server --title "Plan /knowledge source-picker product expansion" --body-file /private/tmp/pr1617_knowledge_product_expansion_issue.md`.
- [x] Add the issue URL to the evidence artifact, Backlog task notes, and final PR comment.

**Success Criteria:**

- [x] A follow-up product-expansion issue exists even if QA only justifies a low-priority tracker.
- [x] The issue is evidence-based and does not imply the work belongs in PR #1617.

---

## Task 9: Close Out PR QA Hardening

**Goal:** Finish the current PR with a clear evidence trail and no ambiguous follow-up state.

- [ ] Update the evidence artifact with final matrix status, findings, fixes, deferrals, skipped rows, and product-expansion issue URL.
- [ ] Update `TASK-297.6` acceptance criteria and Definition of Done.
- [ ] Run `git diff --check`.
- [ ] Run all focused tests touched or relied on by this plan.
- [ ] Run Bandit only if Python production files changed; otherwise document the non-code skip.
- [ ] Commit or amend the QA evidence and any approved fixes.
- [ ] Push the PR branch.
- [ ] Refresh PR checks and review threads:
  - [ ] `gh pr checks 1617 --repo rmusser01/tldw_server`
  - [ ] `gh pr view 1617 --repo rmusser01/tldw_server --json reviewDecision,mergeStateStatus,url`
  - [ ] Query unresolved review threads if needed before declaring review comments addressed.
- [ ] Add a PR comment summarizing:
  - [ ] Tested WebUI and extension surfaces.
  - [ ] Viewports and data profiles.
  - [ ] Fixes made in PR #1617.
  - [ ] Deferrals and rationale.
  - [ ] Product-expansion issue URL.
  - [ ] Remaining blockers, if any.

**Success Criteria:**

- [ ] Evidence, task state, PR state, and follow-up issue all agree.
- [ ] No unresolved P0/P1 `/knowledge` findings remain.
- [ ] P2 findings are fixed or linked to follow-up with rationale.
- [ ] PR #1617 has a concise QA hardening summary comment.

---

## Implementation Notes For Future Agents

- Start from existing tests before creating new harnesses. The goal is QA hardening, not broad testing-infrastructure expansion.
- Treat blocked harnesses as evidence when the blocker is concrete. Do not hide a blocked row by replacing it with an unrelated passing test.
- Keep private real-data observations privacy-safe. If a user-specific issue matters, reduce it into a synthetic reproduction before committing a test or screenshot.
- When in doubt about whether a finding is product expansion or current-PR repair, default to follow-up unless the page is broken, misleading, inaccessible, or unusable in a tested `/knowledge` flow.
