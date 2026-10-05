# Owned on-device history discovery — TASK-13455.1

**Baseline:** HEAD `0d0f98d0bb`; latest `origin/dev` `c2260945ee`, already integrated.
**Approved program:** all ten UX groups in five stages, equal novice/power/roleplay scope. This is a bounded child of Stage 2 and addresses the local reachability part of Stage 1. It does not close either whole stage.
**Design:** add a persisted, lazy **On this device** tab to the shared sidebar. Extend the existing paginated DB read with optional profile/account qualification before count, paging and search; current profile-only forks browse without an inference account, while scope-stamped local rows require the matching verified realm. Server mirrors, server-source rows, `temp`, foreign profiles and fully unowned rows are excluded. Enumeration is read-only and never adopts/stamps records. Reuse the existing draft-owner observer for the canonical realm, its synchronous invalidation and object-identity lease. Distinct qualified query keys prevent warm legacy caches from becoming visible.
**Temporary:** clicking a row opens an explicitly labelled read-only stored-message preview, retaining the current Temporary transcript, draft, address, mode and selected-history controller. Do not use the owned loader, profile initializer or bookmark writer for previews. Show speaker/role, images and alternative/comparison distinctions without inventing a canonical H1 path. Preserve source/profile/account leases and initiating focus. Close only the preview on Escape.
**Normal selection:** request the existing settings-exit acceptance before loading. Prepare required prompt/model/file metadata before the owned acceptance boundary; navigate to `/chat` and close the drawer only for an accepted current result. Preserve existing draft/controller/account/navigation fences and verify the Layout provider remount.
**Scope:** simplify the currently production-unused LocalChatList into a discovery/reader component; omit bare-ID rename/pin/delete/Undo, global date deletion and server folder assignment. Classic Sidebar management remains unchanged. Combined All discovery, older unowned-cache handling, offline mirrors, import, Temporary Q3 retention, durable P9 status, history-first navigation and remaining expert/roleplay stages stay open.
**Tracking:** official backlog-py CLI after MCP search stalled. Do not alter the dirty primary checkout or create a new authority, history schema, migration, provider API or framework. Bandit is N/A unless Python production code is touched.

## Stage 1: Qualify database reads and reproduce boundaries
**Goal**: page/search/count and read-only previews stay within the existing profile and verified realm.
**Success Criteria**: mixed-owner/mirror pages, exact-page counts, title/content search beyond page 1, profile changes, no initialization/adoption and zero write previews have meaningful red/green evidence.
**Tests**: existing test tools plus a real IndexedDB browser harness where available; pure/mocked transaction tests supplement rather than claim browser evidence.
**Status**: Complete

## Stage 2: Render discovery and Temporary previews
**Goal**: current qualified results, distinct loading/empty/error/partial states, retry and safe read-only previews.
**Success Criteria**: current keys cannot reuse unqualified data; duplicate IDs do not render twice; live changes update results; stale account/profile/source reads cannot publish. Temporary action and preview meaning are explicit.
**Tests**: mounted real React Query list with real Core/H1 boundaries where relevant; held account ABA/unmount/New races; initial/next-page retry; keyboard/focus and zero-write checks.
**Status**: Complete

## Stage 3: Connect shared sidebar and owned navigation
**Goal**: persisted lazy tab plus accepted normal selection from chat/settings.
**Success Criteria**: local tab mounts only when shown, server overview stays off, trimmed scoped search works, settings cancellation does not load, accepted selection survives route/provider remount, metadata failure does not partially switch.
**Tests**: actual shared sidebar lazy/coordinator tests, actual loader races and route/provider integration; setting normalization.
**Status**: Complete

## Stage 4: Review, verify and commit
**Goal**: independently reviewed incremental change with honest coverage and remaining program scope.
**Success Criteria**: affected suites, typecheck, no added touched-scope lint, diff and task normalization pass; both authorized reviewers resolve Critical/Important findings. Task/program notes distinguish delivered eligibility from open full discovery/migration/retention/status work. Own prior completed plan removed after archival reference.
**Tests**: bounded regressions and browser evidence, not a full-program completion claim.
**Status**: Complete

## Verification and review (2026-10-05)

- Meaningful red/green coverage includes qualification before paging/search, loader metadata preparation, actual-English labels/focus, profile initializer/observer completion orders, route/account/profile ABA, held promotion, failed candidate qualification and the late H1 profile-read boundary. The final boundary reproduced three failures, then all 86 affected tests passed.
- Final bounded regression: 14 suites / 326 tests passed. Frontend TypeScript passed. Touched TypeScript ESLint introduced no findings relative to HEAD; existing baseline findings are retained. All 26 English on-device catalog values agree across source and packaged-extension catalogs. Diff whitespace and exact Backlog task normalization checked.
- Native Chrome with real Dexie/schema/chat/local-owner modules and IndexedDB passed qualification before page counts, title/content search, readonly preview with zero storage changes, unchanged-count edit events, promotion invalidation, expected-profile owner qualification, readonly qualified-open validation, and exactly one explicit profile initialization followed by idempotent verification. The temporary harness is storage verification, not full application UI or a backend/provider journey.
- Mounted tests use actual Core/H1, React Query, AntD and English i18next with IO doubles. The explicit qualified handoff survives actual H1/Playground persistence remount in the presence of a valid older unsent draft. This does not claim complete Layout/browser navigation, geometry, VoiceOver or native extension-shell acceptance.
- Both authorized independent reviewers found no remaining substantiated Critical/Important issue after the final readonly profile correction. Prior A remains selected on candidate/profile preflight failure; the accepted owner remains usable after list closure without weakening Core or account fences.
- Bandit is N/A: production changes are TypeScript and documentation; no Python production code was touched. The prior completed mode plan is removed only after its archival commit `0d0f98d0bb`.
- TASK-13455.1 is complete as a bounded discovery unit. TASK-13453/13454/13455 and remaining approved stages stay open: full combined history/offline mirrors/import, history-first preference, focused composer, older character sessions, roleplay setup, durable P9 state, native D7 contracts, Temporary Q3 retention, responsive/extension and full-journey acceptance remain required.
