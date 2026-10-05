# Complete character-session discovery — TASK-13455.3

Baseline: local `3485d54c42`; latest `origin/dev c2260945ee` freshly verified. Approved design: `Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md`. This child implements the complete character-session requirement without closing the whole Stage 2/program.

Use the existing native `character_id` list filter and account request scope before paging, with five recent selected-character sessions initially and an explicit All sessions for this character view using ordinary 25-row pages. Keep other-character recents as an independently labelled, bounded preview. Retain established selection and existing ownership gates. Invalid/local character IDs cannot silently fall back to all-server history. Reset view/page/disclosures on account lease and character replacement; preserve global chat scope and shared-hook workspace compatibility; reject stale callback/response publication. The list endpoint can fall back to a page-length total, so a full page must still offer Next and a later empty page must never mean no character conversations. Honest loading/offline/stale/failed-page feedback and Retry preserve reachability. Add no backend API, search service, persistence store or credential-bearing query key.

## Stage 1: Reproduce filtered-history and paging failures
**Goal**: red-first mounted hook/UI cases for interleaved histories, duplicates, old pages and account/scope changes.
**Success Criteria**: current global-sample defect and unavailable/failed-page states fail meaningfully; keep existing character metadata and current-row behavior covered.
**Tests**: real QueryClient/history hook/account observer and English translations with request/connection IO doubles; existing panel and cockpit compatibility.
**Status**: Complete

## Stage 2: Implement owned selected-character paging
**Goal**: server filtering before pagination, explicit full-history access and independently bounded recents.
**Success Criteria**: all selected-character pages reachable, resume-last stays within character, old lease/page callbacks cannot act, invalid local IDs never query unfiltered as complete history.
**Tests**: paging with more than five interleaved rows, duplicate titles, held responses, character/account replacement, foreign-workspace rejection and native selector invocation.
**Status**: Complete

## Stage 3: Verify states and shared-surface compatibility
**Goal**: distinguish pending, offline, stale, hard/recoverable failures, exact empty versus later-page empty and retry.
**Success Criteria**: visible usable controls, no false final-page conclusion from fallback totals, shared English and extension catalogs agree, affected tests/typecheck/touched lint pass.
**Tests**: failure/refresh/Retry, short/full pages and stale account/character closures and scoped hook request/cache compatibility; real account observer regressions and relevant owned-local/draft compatibility.
**Status**: Complete

## Stage 4: Review, record and commit
**Goal**: concrete reviewed incremental delivery with honest coverage.
**Success Criteria**: both authorized reviewers have no remaining substantiated Critical/Important finding; task/plan/diff validation records evidence and remaining whole-program scope. Remove only the completed previous own plan after archival at `3485d54c42`.
**Tests**: bounded regression, typecheck, no added touched lint, catalog parity and task normalization. Bandit is N/A if production remains TypeScript/docs only.
**Status**: Complete

Review scope correction: the deployed full-page selector is global. The panel exposes no workspace prop, requests explicit global scope and excludes foreign-workspace rows. Exact workspace/character/account paging remains tested on the existing shared hook; workspace H1 resumption is not claimed by this unit.

## Verification and review — 2026-10-05

- Red-first mounted discovery run: 25 failures / 1 existing pass before implementation. Existing character metadata, current-row behavior and timestamp coverage were retained with native numeric ID fixtures. A persona row is now correctly excluded from both character lists rather than represented as another character. The default selected-character list filters before its five-row preview; All sessions uses ordinary 25-row paging, including full-page Next when count lookup falls back. Other-character recents are an explicit lazy bounded sample. Resume last stays within the selected character.
- Independent review reproduced two initial verification pending/retry failures, three duplicate-label/dispatch failures (offline transition plus Resume inside one act, and disabled failed-panel Retry), and one foreign-workspace response failure. Corrected the same fenced owner observer, flattened extension message entries, distinct duplicate references, imperative connection/enabled checks and explicit global panel scope. The shared hook preserves exact character/workspace/account/offset and search-cache compatibility. No new workspace-facing selector or storage authority was introduced.
- Final bounded verification: **9 suites / 245 tests passed**, including 30 mounted discovery/account observer cases, legacy panel metadata, cockpit shell, locale mirror, shared history paging/search, owned on-device selection, composer recovery, assistant ownership and Playground persistence handoff. Frontend typecheck passed. No added touched-scope ESLint findings relative to `3485d54c42`; the history hook's two existing findings remain. All 50 new English source entries match flattened packaged message entries; diff whitespace and exact task normalization checked.
- Both authorized independent reviewers found no remaining substantiated Critical/Important issue after corrections. They inspected source/logs; they did not establish live backend/native H1 resumption, browser geometry, screen-reader behavior or user-choice outcomes. Mounted discovery uses real history/query/account-observer and English translation paths with connection, request, account event delivery and final selection IO doubles. More than five interleaved sessions, duplicate selectors and two-page reachability are demonstrated; whole-program live journeys and very large-history acceptance remain later gates.
- Bandit N/A: TypeScript/documentation-only production scope. Previous own navigation plan removed after archival at `3485d54c42`. This child is complete; focused composer, combined source discovery/offline mirrors/import, durable native contracts, roleplay setup/greeting, extension continuity and whole-program accessibility/live-journey gates remain open.
