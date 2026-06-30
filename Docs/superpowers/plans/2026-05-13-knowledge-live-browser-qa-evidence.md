# Knowledge Live-Browser QA Evidence For PR 1617

Date: 2026-05-13
Status: QA hardening evidence complete; PR operations tracked on PR #1617
PR: https://github.com/rmusser01/tldw_server/pull/1617
Branch: codex/knowledge-source-contract
Baseline commit: 7667cb9da6287dbc6d26fd9cb5b45d12791ebe1e
Design spec: Docs/superpowers/specs/2026-05-13-knowledge-live-browser-qa-hardening-design.md
Implementation plan: Docs/superpowers/plans/2026-05-13-knowledge-live-browser-qa-hardening-implementation-plan.md
Backlog task: TASK-297.6

## Summary

This artifact is the committed execution record for the PR #1617 `/knowledge` live-browser QA hardening pass. Task 3 added a seeded WebUI live-browser pass across desktop, tablet, and mobile viewports using synthetic route-mocked data. Task 4 added extension route and extension-sized coverage for `#/knowledge`, `#/knowledge/thread/:threadId`, and `#/knowledge/shared/:shareToken`. Task 5 added a privacy-safe local real-data backend-read-only pass that captured counts/control state only. Task 6 added seeded keyboard/power-user coverage and produced two low-risk current-PR accessibility fixes. Task 8 created the evidence-based product-expansion follow-up issue: https://github.com/rmusser01/tldw_server/issues/1631.

`/knowledge` is QA-only in this PR. This pass must not implement saved-view sharing, profile sharing/export/import, advanced source organization, backend sharing APIs, or a general knowledge CRUD/import hub. Those ideas belong in a follow-up product-expansion issue only after QA evidence supports them.

## Environment

| Field | Value |
| --- | --- |
| Worktree | `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/knowledge-source-contract` |
| Current branch | `codex/knowledge-source-contract` |
| PR URL | `https://github.com/rmusser01/tldw_server/pull/1617` |
| Baseline commit | `7667cb9da6287dbc6d26fd9cb5b45d12791ebe1e` |
| WebUI Playwright default URL | `http://localhost:8080` |
| WebUI Playwright default command | `bun run dev -- -p 8080` |
| WebUI API default | `http://127.0.0.1:8000` unless overridden by `NEXT_PUBLIC_API_URL`, `TLDW_SERVER_URL`, or `TLDW_E2E_SERVER_URL` |
| Manual browser URL for this pass | `http://localhost:8080/knowledge` unless port 8080 is unavailable; record any alternate URL here before QA continues |
| Seeded fixture source | Existing Playwright mocks/harness data and test-only mocks where available; if no current fixture supports a required row, keep the row and mark it `Blocked` with the missing fixture/runtime reason during execution |
| Backend health for Task 3 | `http://127.0.0.1:8000/api/v1/health` reachable in `single_user` mode; response status `degraded` because `chacha_notes` was degraded, with database and metrics healthy |
| Browser automation for Task 3 | Browser plugin had no callable browser tool in this session; used direct Playwright from `apps/tldw-frontend/node_modules` through `/private/tmp/knowledge_task3_live_qa.cjs` |
| Browser QA status | QA hardening evidence complete |
| Task 3 coverage boundary | Seeded frontend coverage only. API health was verified, but `/knowledge` media, notes, RAG, notifications, persona, chat, and provider responses were synthetic Playwright route mocks; this is not real backend/database coverage. |
| Task 4 coverage boundary | Extension route coverage used the shipped extension E2E harness plus a temporary synthetic Playwright/Bun probe. The route probe used `http://dummy-tldw` server mocks and does not prove real backend/database behavior. |
| Task 5 coverage boundary | Live backend-read-only UI probe with `TLDW_E2E_API_KEY`. Captured no screenshots, titles, source excerpts, generated answers, response bodies, or private content. Did not run UI query/citation flows because those persist Knowledge QA chat/thread state and expose private result content. The saved-profile check created and loaded only a temporary profile in isolated Playwright browser storage; it did not call backend create/update APIs or mutate local databases. |
| Task 6 coverage boundary | Seeded keyboard probe used synthetic route mocks and no private content. Playwright did not deliver `Escape` keydown events to the page in the final diagnostic run (`Captured Escape events ...: 0`), so direct browser Escape closure was treated as an automation limitation and covered with focused Vitest regression tests instead. |

## Privacy Rules For Local Real-Data QA

- Prefer a copied or sanitized profile over live mutable databases.
- If live local data must be used, avoid create, edit, delete, reindex, export, and share flows.
- Do not commit screenshots that reveal private titles, prompts, chats, notes, document text, source excerpts, filenames, or other identifying content.
- Do not quote private content in this artifact, Backlog notes, PR comments, issue bodies, screenshots, or test fixtures.
- Record only source type, route, viewport, workflow, symptom, and whether the symptom can be reproduced with synthetic data.
- Keep raw local real-data notes and private screenshots outside git.
- If a real-data issue needs a current-PR fix, reproduce it with seeded or synthetic data before committing a regression test.

## Evidence Table Fields

Every QA row uses these design-spec fields:

| Surface | Route | Viewport | Data profile | Task | Result | Evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Example | `/knowledge` | 1440 x 900 | Seeded | Source picker filters by status | Planned | Pending QA | Pending |

Result values: `Planned`, `Pass`, `Fail`, `Blocked`, or `Observation`.

Decision values: `No action`, `Current PR fix`, `Follow-up issue`, `Document only`, `Blocked`, or `Pending`.

## Baseline Harness Results

These rows record Task 2's existing deterministic coverage before manual browser QA. The first Playwright smoke attempt was blocked by sandbox port binding on `0.0.0.0:8080`; the same focused command passed when rerun outside the sandbox.

| Surface | Route | Viewport | Data profile | Task | Result | Evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Shared UI | Component tests | N/A | Seeded/test mocks | Source picker filters, saved profiles, streaming QA provider, KnowledgePanel routing, and source metadata | Pass | `apps/packages/ui`: `bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.scalable-source-picker.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.profiles.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx src/components/Knowledge/__tests__/KnowledgePanelTabRouting.test.tsx src/services/rag/__tests__/sourceMetadata.test.ts`; 5 files passed, 29 tests passed. Non-blocking stderr: missing i18next instance warnings and `/api/v1/llm/providers 400 tldw server not configured` harness noise. | No action |
| WebUI | Extension route parity static test | N/A | Static route registry | Verify active extension options knowledge paths map to shared KnowledgeQA route graph and legacy mirror remains safe | Pass | `apps/tldw-frontend`: `bunx vitest run __tests__/extension/knowledge-route-parity.test.ts`; 1 file passed, 4 tests passed. | No action |
| WebUI | `/knowledge` | Chromium desktop smoke | Seeded Playwright smoke data | Search typing and deterministic no-results answer remain functional | Pass | `apps/tldw-frontend`: first sandboxed run blocked with `Error: listen EPERM: operation not permitted 0.0.0.0:8080`; escalated rerun of `bunx playwright test e2e/smoke/stage6-interaction-stage2.spec.ts -g "search typing and deterministic no-results answer remain functional" --reporter=line` passed, 1 test passed in 22.6s. | No action |
| Extension | `#/knowledge/thread/:threadId`, `#/knowledge/shared/:shareToken` | Chromium extension harness | Extension harness data | Knowledge tutorial card appears on thread and shared routes | Pass | `apps/extension`: `bunx playwright test tests/e2e/quick-chat-guides-tutorials.spec.ts -g "knowledge tutorial card" --reporter=line`; 2 tests passed in 40.4s after extension build. Non-blocking build noise: `npm: command not found`, duplicate import warnings, unresolved font runtime warnings, Rollup circular chunk warnings, and chunk size warnings. | No action |

## Seeded Matrix

Rows in this section must stay in place even when current fixtures cannot cover them. During QA, mark unsupported rows `Blocked` with the missing fixture or runtime reason instead of deleting them.

| Surface | Route | Viewport | Data profile | Task | Result | Evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WebUI | `/knowledge` | 1440 x 900 | Empty or first-run state | Verify first arrival, ready state, and empty-state comprehension | Pass | Synthetic no-source run `desktop-empty`: search and ready-state title visible, Add Sources reachable, no-source copy visible, provider/privacy disclosure visible, Ask disabled when no sources and web fallback disabled. Screenshot: `/private/tmp/knowledge-task3-live-qa/desktop-empty.png`. | No action |
| WebUI | `/knowledge` | 390 x 844 | Empty or first-run state | Verify Add Sources discovery and empty-state actions remain reachable on mobile | Pass | Synthetic no-source run `mobile-empty`: search and ready-state title visible, Add Sources reachable, source scope reachable, no-source copy visible, provider/privacy disclosure visible, Ask disabled when no sources and web fallback disabled. Screenshot: `/private/tmp/knowledge-task3-live-qa/mobile-empty.png`. | No action |
| WebUI | `/knowledge` | 1440 x 900 | Seeded realistic library: media, notes, chats, characters, task boards, prompts, world books, dictionaries where current fixtures expose them | Verify canonical source category selection and source status visibility | Pass | `desktop-wide`: source menu showed `Documents & Media`, `Notes`, `Chats`, `Characters`, `Task Boards`, `Prompts`, `World Books`, and `Dictionaries`; status filtering exposed indexing and unavailable synthetic sources. Screenshot: `/private/tmp/knowledge-task3-live-qa/desktop-wide.png`. | No action |
| WebUI | `/knowledge` | 1280 x 720 | Seeded realistic library: media, notes, chats, characters, task boards, prompts, world books, dictionaries where current fixtures expose them | Verify query filter, status filter, recent imports filter, and specific source picker behavior | Pass | `desktop-constrained`: specific picker showed seeded media, hid generated and workspace artifacts by default, query filter narrowed docs, status filter exposed indexing and unavailable sources, recent imports filter was reachable, and workspace filter exposed the scoped artifact. Screenshot: `/private/tmp/knowledge-task3-live-qa/desktop-constrained.png`. | No action |
| WebUI | `/knowledge` | 1024 x 768 | Seeded realistic library: media, notes, chats, characters, task boards, prompts, world books, dictionaries where current fixtures expose them | Verify Simple/Detailed toggle, citation/source card inspection, and Continue in editor handoff reachability | Pass | `tablet`: Simple/Detailed toggle reachable, answer rendered, citation chip reachable, Evidence panel opened with 2 source cards, seeded source visible, and Continue in editor was reachable. Screenshot: `/private/tmp/knowledge-task3-live-qa/tablet.png`. | No action |
| WebUI | `/knowledge` | 390 x 844 | Seeded realistic library: media, notes, chats, characters, task boards, prompts, world books, dictionaries where current fixtures expose them | Verify source controls, filters, bulk actions, and answer actions do not overflow or become unreachable | Pass | `mobile`: Add Sources, source scope, compact settings, all source categories, web toggle, answer, citation chip, Evidence source cards, and Continue in editor reachable. Mobile has no Simple/Detailed toggle by current responsive design. Screenshot: `/private/tmp/knowledge-task3-live-qa/mobile.png`. | No action |
| WebUI | `/knowledge` | 1440 x 900 | Sources with unavailable or empty status | Verify unavailable/empty source status clarity and selection behavior | Pass | `desktop-wide` and `desktop-local-only`: status filter exposed indexing source and unavailable synthetic source. Current picker exposes ready, indexing, error, and unavailable status options; no separate empty-status option was observed. | No action |
| WebUI | `/knowledge` | 1280 x 720 | Weak or no-result retrieval | Run weak/no-result query and verify recovery path and messaging | Pass | `desktop-no-results`: no-result answer rendered, no citation chip expected, Evidence panel showed 0 sources, and no-result recovery messaging was visible. Screenshot: `/private/tmp/knowledge-task3-live-qa/desktop-no-results.png`. | No action |
| WebUI | `/knowledge` | 1440 x 900 | Workspace-scoped artifacts hidden globally and visible only under explicit workspace scope | Verify workspace filter hides scoped artifacts globally and shows them only under explicit workspace scope | Pass | `desktop-wide`: synthetic generated fixture and workspace artifact were hidden by default; selecting workspace scope exposed the scoped artifact. | No action |
| WebUI | `/knowledge` | 390 x 844 | Workspace-scoped artifacts hidden globally and visible only under explicit workspace scope | Verify mobile workspace scope controls are reachable and scoped artifacts remain hidden globally | Pass | `mobile`: compact source settings reachable and source categories exposed. Direct granular workspace filter is a detailed desktop control; mobile settings path was reachable and no generated/workspace artifact leaked into answer evidence. | No action |
| WebUI | `/knowledge` | 1440 x 900 | Web fallback disabled | Run local-only QA query and verify no web fallback is invoked or recommended | Pass | `desktop-local-only`: web fallback left disabled before query, answer rendered from seeded local sources, and captured RAG request had `enable_web_fallback: false`. Screenshot: `/private/tmp/knowledge-task3-live-qa/desktop-local-only.png`. | No action |
| WebUI | `/knowledge` | 1440 x 900 | Web fallback enabled | Verify server default provider/privacy disclosure is visible, or record its absence as a finding | Pass | `desktop-wide`, `desktop-constrained`, and `tablet`: web fallback toggle reachable, `AI: Server default` visible, privacy copy visible, and captured RAG requests after toggling had `enable_web_fallback: true`. | No action |
| WebUI | `/knowledge` | 390 x 844 | Web fallback enabled | Verify provider/privacy disclosure remains visible or clearly absent on mobile | Observation | `mobile`: web toggle and `AI: Server default` pill were visible, disclosure was available in settings and empty state, but the privacy copy was not adjacent to the compact main toolbar after sources were selected. | Document only |
| WebUI | `/knowledge` | 1440 x 900 | Seeded realistic library | Verify bulk select visible, clear visible, select recent imports, and local saved profile save/load behavior | Pass | `desktop-wide`: Select visible, Clear visible, Select recent imports, Profiles, Save current settings, saved profile listing, and saved profile load action were reachable. | No action |

## Extension Matrix

| Surface | Route | Viewport | Data profile | Task | Result | Evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Extension | `#/knowledge` | Extension harness | Seeded synthetic route mocks | Verify main knowledge route exposes expected tutorial/card or knowledge UI and route state is stable | Pass | Temporary route probe `bun /private/tmp/extension_knowledge_task4_qa.ts`: `#/knowledge` rendered search input, ready-state title, Add Sources, source scope, web fallback toggle, and all canonical source labels with zero console errors and zero failed requests. Screenshot: `/private/tmp/knowledge-task4-extension-qa/knowledge-main.png`. | No action |
| Extension | `#/knowledge/thread/:threadId` | Extension harness | Extension harness data plus seeded synthetic route mocks | Verify thread route parity and no route-state regression | Pass | Shipped harness: `bunx playwright test tests/e2e/quick-chat-guides-tutorials.spec.ts -g "knowledge tutorial card" --reporter=line` passed 2 tests including thread route. Temporary route probe also rendered search input, ready-state title, Add Sources, source scope, web fallback toggle, and all canonical source labels on `#/knowledge/thread/thread-123` with zero console errors and zero failed requests. Screenshot: `/private/tmp/knowledge-task4-extension-qa/knowledge-thread.png`. | No action |
| Extension | `#/knowledge/shared/:shareToken` | Extension harness | Extension harness data plus seeded synthetic route mocks | Verify shared route parity and no route-state regression | Pass | Shipped harness: `bunx playwright test tests/e2e/quick-chat-guides-tutorials.spec.ts -g "knowledge tutorial card" --reporter=line` passed 2 tests including shared route. Temporary route probe also rendered search input, ready-state title, Add Sources, source scope, web fallback toggle, and all canonical source labels on `#/knowledge/shared/share-token-123` with zero console errors and zero failed requests. Screenshot: `/private/tmp/knowledge-task4-extension-qa/knowledge-shared.png`. | No action |
| Extension | `#/knowledge` | 390 x 844 | Seeded synthetic route mocks | Check overflow, hidden primary actions, and source-control reachability | Pass | Temporary route probe at 390 x 844 showed Add Sources, source scope, web fallback, selected provider, ready-state guidance, Select sources, composer, and footer help link without horizontal overflow in screenshot. | No action |

## Local Real-Data Pass

Local real-data QA must remain privacy-safe and observation-focused. Record only redacted behavioral notes.

| Surface | Route | Viewport | Data profile | Task | Result | Evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WebUI | `/knowledge` | 1440 x 900 | Privacy-safe local real-data backend-read-only pass | Verify source picker scale, filters, status clarity, browser-local profile persistence reachability, and backend request safety | Observation | `node /private/tmp/knowledge_task5_real_readonly.cjs` with `TLDW_E2E_API_KEY`: search, ready-state title, source scope, web fallback toggle, temporary browser-local saved profile create/list/load in isolated Playwright storage, all 8 source-category labels, specific-source filters, source status filter, recent imports filter, workspace scope filter, bulk select, and text filter were reachable. Real backend endpoint hits were read-only `GET /api/v1/media`, `GET /api/v1/media/`, and `GET /api/v1/notes/`; 200 media options and 0 note options were counted. No screenshots, response bodies, titles, excerpts, generated answers, failed requests, console errors, backend mutation requests, or local-database mutation attempts were captured. Weak/no-result, answer, and citation/source-card inspection were intentionally not run; see skipped rows. | Document only |
| WebUI | `/knowledge` | 390 x 844 | Privacy-safe local real-data backend-read-only pass | Verify mobile source picker scale, status clarity, compact source controls, and backend request safety | Observation | Same backend-read-only probe at 390 x 844: search, ready-state title, source scope, web fallback toggle, all 8 source-category labels, compact source settings, and compact source categories were reachable. No screenshots, response bodies, titles, excerpts, generated answers, failed requests, console errors, backend mutation requests, or local-database mutation attempts were captured. Answer/source-card inspection was intentionally not run; see skipped rows. | Document only |

## Keyboard Pass

| Surface | Route | Viewport | Data profile | Task | Result | Evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WebUI | `/knowledge` | 1440 x 900 | Seeded realistic library | Test Tab order through search, source menus, filters, bulk actions, saved profiles, settings, answer actions, and source cards | Observation | Final seeded keyboard probe sampled 60 desktop tab stops. Search and source scope were reachable; specific-source and profile controls were not reached in the first 60 stops while starting from compact mode, but both were directly keyboard-activatable after switching to detailed mode. No console errors or failed requests. | Document only |
| WebUI | `/knowledge` | 1440 x 900 | Seeded realistic library | Test Enter behavior for search submission and source filtering | Pass | Enter opened source menu, specific source selector, profile menu, evidence panel, source preview, and export dialog. Enter submitted the search input and sent a RAG request. A repeated query after keyboard source-scope change sent a second RAG request with `notes` removed from `sources`. | No action |
| WebUI | `/knowledge` | 1440 x 900 | Seeded realistic library | Test Escape behavior for menus, dialogs, and pickers | Observation | Browser diagnostic run generated `/private/tmp/knowledge-task6-keyboard-qa/results.json` at `2026-05-13T06:30:19.346Z`, but captured 0 Escape keydown events reaching the page, so direct browser Escape closure was inconclusive. Focused Vitest red/green coverage now verifies source menu, specific source selector, and settings panel close on Escape using capture-phase handlers even when nested controls stop propagation. | Current PR fix |
| WebUI | `/knowledge` | 1440 x 900 | Seeded realistic library | Verify focus return after closing source picker, settings, source viewer, and existing export/share dialogs if present | Observation | Source preview opened from a source card and returned focus to the View source button after close. Export dialog opened by keyboard and was closeable. Browser Escape focus-return assertions are limited by the same Escape keydown delivery issue. | Document only |
| WebUI | `/knowledge` | 1440 x 900 | Seeded realistic library | Verify keyboard access to Simple/Detailed mode, repeated query with modified source scope, profile save/load, bulk source actions after filtering, and citation/source card navigation | Pass | Keyboard opened detailed controls, typed into the specific-source filter, activated Select visible, created and loaded a browser-local profile, activated citation/source-card controls, toggled Simple/Detailed mode, and repeated the query after changing source scope. | No action |
| WebUI | `/knowledge` | 390 x 844 | Seeded realistic library | Repeat core keyboard and focus checks on mobile-sized viewport where practical | Pass | Final mobile keyboard probe reached search, source scope, and the renamed `Open Knowledge QA settings` control; only one Knowledge QA settings button had that name after the fix. Enter opened the settings panel, close button was keyboard-activatable, Enter submitted mobile search, and Evidence opened with 2 source cards. | No action |

## Findings

No P0/P1 WebUI `/knowledge` defects were found in Tasks 2-6. Task 6 produced two small current-PR keyboard/accessibility fixes and one non-blocking tab-order density observation for later synthesis.

| ID | Severity | Surface | Route | Viewport | Data profile | Finding | Evidence | Decision | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KQA-T3-001 | P3 | WebUI | `/knowledge` | 390 x 844 | Seeded web fallback enabled | Mobile compact toolbar can enable web fallback without adjacent visible privacy copy once sources are selected. | Task 3 `mobile`: web toggle and `AI: Server default` pill visible; disclosure available in settings and empty state; disclosure not visible near compact toolbar after sources are selected. | Follow-up issue | Tracked in https://github.com/rmusser01/tldw_server/issues/1631 |
| KQA-T6-001 | P2 | WebUI | `/knowledge` | 1440 x 900 and 390 x 844 | Seeded keyboard | Source menus/settings should close on Escape even if nested controls intercept key events. | Browser Escape probe was inconclusive because Playwright delivered 0 Escape keydown events to the page; focused red/green Vitest reproduced the underlying nested-control risk by stopping key propagation inside menus/dialogs. | Current PR fix | Fixed by capture-phase Escape handlers in `KnowledgeContextBar` and `SettingsPanel` |
| KQA-T6-002 | P3 | WebUI | `/knowledge` | 390 x 844 | Seeded keyboard | Compact toolbar settings gear used the same accessible name as the global app settings control. | Mobile keyboard probe found duplicate `Open settings` buttons before the fix; after the fix the Knowledge QA control label and tooltip are `Open Knowledge QA settings`, and the final probe saw one matching Knowledge QA settings button. | Current PR fix | Fixed in `CompactToolbar` |
| KQA-T6-003 | P3 | WebUI | `/knowledge` | 1440 x 900 | Seeded keyboard | Specific source and profile controls are not reached within the first 60 Tab stops from initial compact mode. | Final desktop keyboard probe reached search and source scope, but not specific-source/profile controls in the first 60 stops until switching to detailed mode. Direct keyboard activation worked after detailed mode. | Follow-up issue | Tracked in https://github.com/rmusser01/tldw_server/issues/1631 |

Severity gates:

- P0/P1: fix in PR #1617 or explicitly block merge.
- P2: fix in PR #1617 when the fix is small and local; otherwise document with a follow-up issue and rationale.
- P3: document in the QA summary unless the fix is trivial and clearly risk-free.
- Product expansion: do not implement in PR #1617; create the product-expansion issue with evidence.

## Fix Decisions

Current-PR fixes are allowed only for defects proven by QA evidence and limited to `/knowledge` or directly reachable flows.

| Finding ID | Decision | Rationale | Test or verification required |
| --- | --- | --- | --- |
| KQA-T3-001 | Follow-up issue | P3 observation with no demonstrated query failure or privacy-blocking path; provider is visible in the compact toolbar and privacy copy is present in settings and empty state. Mobile compact disclosure refinements belong in the product-expansion issue if user trials show confusion. | Tracked in https://github.com/rmusser01/tldw_server/issues/1631. |
| KQA-T6-001 | Current PR fix | The change is small, local to shared KnowledgeQA controls, and protects Escape close behavior when nested widgets intercept keyboard events. | Red/green Vitest for `KnowledgeContextBar` and `SettingsPanel`; final browser keyboard run documents the Playwright Escape-key delivery limitation. |
| KQA-T6-002 | Current PR fix | The duplicate `Open settings` accessible name created real mobile keyboard/screen-reader ambiguity and the copy change is local to the `/knowledge` compact toolbar. | Red/green Vitest for `CompactToolbar`; final browser keyboard run verifies `Open Knowledge QA settings`. |
| KQA-T6-003 | Follow-up issue | P3 density/friction observation only. Controls remain reachable through visible detailed mode and direct keyboard activation; redesigning tab order or adding shortcuts belongs in product-expansion evaluation. | Tracked in https://github.com/rmusser01/tldw_server/issues/1631. |

## Product-Expansion Issue

Status: Created at https://github.com/rmusser01/tldw_server/issues/1631 after QA synthesis, so the issue cites observed PR #1617 evidence rather than guesses.

Candidate scope, only if supported by QA findings:

- Saved views beyond local single-user profiles.
- Profile sharing/export/import when source scopes need reuse across devices, users, or installations.
- Advanced source organization when large libraries are hard to navigate with current filters.
- Workspace/source grouping improvements when current filters are insufficient.
- Keyboard command palette or saved-view shortcuts when repeated-task friction is observed.

Non-goals for the issue:

- Making `/knowledge` the canonical CRUD/import/management hub.
- Replacing source owner pages.
- Automatic web fallback recommendations.
- Showing generated, test, or workspace artifacts globally by default.

## Verification Commands

Task 1 verification:

```bash
git diff --check
rg -n "[^[:ascii:]]" Docs/superpowers/plans/2026-05-13-knowledge-live-browser-qa-evidence.md "backlog/tasks/task-297.6 - Plan-knowledge-live-browser-QA-hardening-for-PR-1617.md"
```

Results:

| Command | Result | Notes |
| --- | --- | --- |
| `git diff --check` | Pass | Exit 0; no whitespace errors |
| `rg -n "[^[:ascii:]]" Docs/superpowers/plans/2026-05-13-knowledge-live-browser-qa-evidence.md "backlog/tasks/task-297.6 - Plan-knowledge-live-browser-QA-hardening-for-PR-1617.md"` | Pass | Exit 1 with no matches, which means no non-ASCII text was found |

Task 2 baseline verification:

| Command | Result | Notes |
| --- | --- | --- |
| `bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.scalable-source-picker.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.profiles.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx src/components/Knowledge/__tests__/KnowledgePanelTabRouting.test.tsx src/services/rag/__tests__/sourceMetadata.test.ts` from `apps/packages/ui` | Pass | 5 files passed, 29 tests passed; non-blocking stderr noted in baseline results |
| `bunx vitest run __tests__/extension/knowledge-route-parity.test.ts` from `apps/tldw-frontend` | Pass | 1 file passed, 4 tests passed |
| `bunx playwright test e2e/smoke/stage6-interaction-stage2.spec.ts -g "search typing and deterministic no-results answer remain functional" --reporter=line` from `apps/tldw-frontend` | Pass after sandbox rerun | Initial sandboxed run blocked with `listen EPERM` on `0.0.0.0:8080`; escalated rerun passed, 1 test passed |
| `bunx playwright test tests/e2e/quick-chat-guides-tutorials.spec.ts -g "knowledge tutorial card" --reporter=line` from `apps/extension` | Pass | 2 tests passed; non-blocking build warnings noted in baseline results |

Task 3 WebUI seeded live-browser verification:

| Command | Result | Notes |
| --- | --- | --- |
| `/bin/zsh -lc "curl -sf http://127.0.0.1:8000/api/v1/health"` | Pass with degraded service status | Backend reachable in `single_user` mode; database and metrics healthy; `chacha_notes` degraded. Used only as environment evidence because `/knowledge` data in this task was route-mocked. |
| `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev -- -p 8080` from `apps/tldw-frontend` | Pass | Worktree WebUI served `http://localhost:8080`; port 3000 had another WebUI, so 8080 avoided testing the wrong checkout. |
| `node /private/tmp/knowledge_task3_live_qa.cjs` from `apps/tldw-frontend` | Pass | Generated `/private/tmp/knowledge-task3-live-qa/results.json` at `2026-05-13T05:25:20.692Z`; covered `desktop-empty`, `mobile-empty`, `desktop-wide`, `desktop-local-only`, `desktop-constrained`, `tablet`, `mobile`, and `desktop-no-results`; all rows had zero console errors and zero failed API requests. |

Task 4 extension route and viewport verification:

| Command | Result | Notes |
| --- | --- | --- |
| `bunx playwright test tests/e2e/quick-chat-guides-tutorials.spec.ts -g "knowledge tutorial card" --reporter=line` from `apps/extension` | Pass | 2 tests passed in 7.0s; covered `#/knowledge/thread/:threadId` and `#/knowledge/shared/:shareToken` tutorial-card reachability. Non-blocking Node warning: `NO_COLOR` ignored because `FORCE_COLOR` was set. |
| `bun /private/tmp/extension_knowledge_task4_qa.ts` from `apps/extension` | Pass | Generated `/private/tmp/knowledge-task4-extension-qa/results.json` at `2026-05-13T05:34:17.208Z`; covered `#/knowledge`, `#/knowledge/thread/thread-123`, and `#/knowledge/shared/share-token-123` at `390 x 844` with zero console errors and zero failed requests. |
| `bunx playwright test tests/e2e/knowledge-rag-ux.spec.ts -g "workflow hub" --reporter=line` from `apps/extension` | Blocked | Older supplemental harness appears stale and failed before reaching `/knowledge`: it navigated to now-missing `#/settings/manageKnowledge`, showed a 404 route page, and timed out waiting for `workflow-button`. This is recorded as stale supplemental harness coverage, not a `/knowledge` product failure or planned matrix skip. |

Task 5 privacy-safe local real-data verification:

| Command | Result | Notes |
| --- | --- | --- |
| `/bin/zsh -lc "curl -s -L -o /dev/null -w '%{http_code} %{url_effective}' -H 'X-API-KEY: <local-e2e-key>' 'http://127.0.0.1:8000/api/v1/media?limit=1'"` | Pass | Returned `200` after redirect to `/api/v1/media/?limit=1`; no response body printed. |
| `/bin/zsh -lc "curl -s -L -o /dev/null -w '%{http_code} %{url_effective}' -H 'X-API-KEY: <local-e2e-key>' 'http://127.0.0.1:8000/api/v1/notes?limit=1'"` | Pass | Returned `200` after redirect to `/api/v1/notes/?limit=1`; no response body printed. |
| `TLDW_E2E_API_KEY=<local-e2e-key> node /private/tmp/knowledge_task5_real_readonly.cjs` from `apps/tldw-frontend` | Pass | Generated `/private/tmp/knowledge-task5-real-readonly/results.json` at `2026-05-13T05:59:35.851Z`; covered `desktop-real-readonly` and `mobile-real-readonly`; zero console errors, zero failed requests, zero backend mutation requests, zero local-database mutation attempts, and no screenshots or response bodies captured. Temporary saved-profile create/load used isolated Playwright browser storage only. |

Task 6 keyboard/power-user and Task 7 fix verification:

| Command | Result | Notes |
| --- | --- | --- |
| `bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.test.tsx src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx src/components/Option/KnowledgeQA/__tests__/SettingsPanel.behavior.test.tsx` from `apps/packages/ui` before the implementation fix | Failed as expected | Red run failed 3 tests: nested controls could stop Escape before `KnowledgeContextBar` closed source selectors, nested settings controls could stop Escape before `SettingsPanel` closed, and `CompactToolbar` still used `Open settings`. Existing tests otherwise passed. |
| `bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.test.tsx src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx src/components/Option/KnowledgeQA/__tests__/SettingsPanel.behavior.test.tsx` from `apps/packages/ui` after the implementation fix | Pass | 3 files passed, 30 tests passed. Non-blocking stderr: missing i18next instance warnings. |
| `bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.test.tsx src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx src/components/Option/KnowledgeQA/__tests__/SettingsPanel.behavior.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.scalable-source-picker.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.profiles.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx src/components/Knowledge/__tests__/KnowledgePanelTabRouting.test.tsx src/services/rag/__tests__/sourceMetadata.test.ts` from `apps/packages/ui` | Pass | 8 files passed, 59 tests passed. Non-blocking stderr: missing i18next instance warnings and `/api/v1/llm/providers 400 tldw server not configured` test-harness noise. |
| `bunx vitest run __tests__/extension/knowledge-route-parity.test.ts` from `apps/tldw-frontend` | Pass | 1 file passed, 4 tests passed. |
| `node /private/tmp/knowledge_task6_keyboard_qa.cjs` from `apps/tldw-frontend` | Pass with Escape automation limitation | Generated `/private/tmp/knowledge-task6-keyboard-qa/results.json` at `2026-05-13T06:30:19.346Z`; covered desktop and mobile seeded keyboard workflows with zero console errors and zero failed requests. Desktop Enter opened source, specific-source, profile, evidence, source preview, and export controls; repeated query after source-scope change sent a second RAG request with `notes` removed. Mobile Enter opened Knowledge QA settings, submitted search, and opened Evidence with 2 source cards. Playwright captured 0 Escape keydown events reaching the page, so Escape behavior is verified by focused Vitest instead of the browser probe. |
| Bandit | Not applicable | Task 6/7 changed only TypeScript/React UI and test files; no Python production files were touched. |

Task 8 product-expansion issue verification:

| Command | Result | Notes |
| --- | --- | --- |
| `gh issue create --repo rmusser01/tldw_server --title "Plan /knowledge source-picker product expansion" --body-file /private/tmp/pr1617_knowledge_product_expansion_issue.md` | Pass | Created https://github.com/rmusser01/tldw_server/issues/1631. |

## Skipped Rows

Rows must be added here when a planned matrix row cannot run. Do not delete planned rows from the seeded, extension, local real-data, or keyboard matrices.

No planned matrix rows were skipped through Task 4. Stale supplemental harness coverage is documented in the verification command table instead of this planned-row skip table.

| Surface | Route | Viewport | Data profile | Task | Result | Evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WebUI | `/knowledge` | 1440 x 900 and 390 x 844 | Privacy-safe local real-data pass | Weak/no-result query, citation/source card inspection, and answer inspection against private local data | Blocked | Not run in Task 5 because the current UI query flow creates/persists Knowledge QA chat/thread state and would expose private answer/source content. Seeded synthetic Tasks 2-3 already verify no-result recovery, citations, Evidence/source cards, and Continue in editor without private data. | Blocked |
