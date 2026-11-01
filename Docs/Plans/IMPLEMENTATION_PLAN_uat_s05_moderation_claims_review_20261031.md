# S05 Moderation and claims review UAT

Task: TASK-13260.278.18.83.45. S04 is accepted; 41/43 workflows are accepted. Finish S05 before S06.

Candidate: product/frontend ca3c20768f50b682ad6f005642580bef80963263, build JagRpn2BPJIOfZhd17hkU; backend 51f0c1d8c51c4101c164039df7947ec6094c105c at /private/tmp/tldw-s04-search-budget-20261031/site. Frozen dev a412f6580b7ed0cf48fd7d222a47a20cfc0dde92 includes the Chat redesign; this is not current live-dev or publication qualification.

Current local repair: moderation queue/read/mutation ownership is committed at 8a73f66a0e7f5e28eebc6f737f84879ab4677912 and qualified by 53 affected tests in each client; the current runtime build does not include it yet. Authority/preset assessment and live S05 acceptance remain open.

## Contract and scope

- /moderation: inspect actual sanitized items, matched rules/reasons/evidence, decide selected items, inspect audit attribution, and reload. The specialized power-user variant uses its bulk decision controls.
- /moderation/rules: persist a controlled fictional literal rule, test benign and flagged samples, reload and independently verify its effective configuration. /moderation-playground is the shipped redirect.
- /claims-review: review one clearly supported and one unsupported Rowan claim against its actual source, save status/notes/correction, inspect history and reload.
- Recovery: denied role with no private-item leak or decision mutation; stale/conflicting decision recovery; real partial bulk success/failure with untouched unselected items. Pin the actual supported failure mechanism before execution.
- Inventory advertised model-derived classification before assigning L coverage. Deterministic rules are not model classification. No fake provider/browser responses or unrecorded capability substitutions.
- Use private synthetic fixtures through existing services/DB abstractions. Set MODERATION_REVIEW_DB_PATH inside the exact private profile; capture defaults off and must be configured explicitly if testing actual capture. Claims use the existing per-user MediaDatabase path. Never write the repository's default moderation database.
- Execute SQLite/PostgreSQL single-/multi-user modes through official restricted fixtures where the actual storage supports them. Moderation's own SQLite review store remains SQLite in a PostgreSQL AuthNZ/Media deployment; record that distinction.
- Compare full selected/unselected rows and audit attribution independently. Capture settled screenshots for human UX review while functional checks continue. Human review is separate from automated acceptance.

## Authority and resource guards

All necessary diagnostic/retry/fix/build/runtime authorization persists. Reassess failures using evidence and reviewed corrections without repeated permission questions. Reuse this checkout and current functional package/build. One private window/tab per native case; preserve original API/frontend, original Chrome absence and the untouched/unverified original DOM. No stale original-draft questions.

Keep 600s workflow/build, 120s startup, 180s generation and 15s operations/graceful cleanup limits. The shared observer has a fixed 4096-entry bound with latched overflow and exact terminal/cohort checks. Official PostgreSQL15492 only; no operator SQL or homemade database. No GitHub/fetch/publication/queue/merge, unrelated native/installer replay, model/service changes or downloads. Dispose completed private profiles/exports/images under identity and resource guards; keep only current reusable functional inputs.

ADR assessment: no new durable rule for validation. ADR018 governs endpoint authority, ADR003 Jobs/Scheduler, ADR059 task editing. Assess any proven repair before implementation. Use existing debugging/TDD, scoped security, independent actual/staged review and normal hooks for repository repairs.

## Stage 1: Pin real controls, authority and fixtures
**Goal**: Map every S05 catalog variant to current UI/API, provider capability and exact expected result.
**Success Criteria**: Private fixture paths, supported creation/capture methods, claim source/offset data, role matrix, partial-failure mechanism and bounded driver are reviewed before live writes.
**Tests**: Small actual-helper admission/privacy/cleanup controls for new operator logic; inspect relevant existing tests without claiming mocked results as live UAT.
**Status**: In Progress

## Stage 2: Validate and repair sequentially
**Goal**: Complete rules, moderation decisions/bulk/audit, then claims decisions/history/recovery in supported modes.
**Success Criteria**: Actual state changes match selected items only; reload and canonical comparisons pass; denial/partial failure preserve private and unselected data. Causal red precedes minimal shared repair; affected checks and scoped security/review precede changed-source packaging.
**Tests**: Appropriate moderation review/rules and claims review API/DB/client controls; real native service and UI checks.
**Status**: In Progress

## Stage 3: Accept and dispose
**Goal**: Settle functional, UX, review and cleanup gates before S06.
**Success Criteria**: All named variants have truthful results, human UX accepted, all requests/processes retire and completed private fixtures are removed. Parent task remains open for broader work; acceptance42/43 only after S05 passes. Remove only this owned plan on completion.
**Tests**: Independent canonical decisions/audit/content/ownership comparison, exact resource release and guarded disposal.
**Status**: Not Started


## Queue and detail response ownership repair

S05 request-order unit in progress. Six read-order cases causally failed unchanged production, then passed the initial queue/detail ticket fix. Four connected mutation/selection/unmount cases failed that incomplete fix, then passed current-selection and lifecycle guards. Captured Undo-token/new-selection control also passes. Independent review then identified stale post-mutation filter refresh and cancellation of unrelated B detail: four focused cases failed before correction and now pass. Current affected five review modules pass 44 tests in UI and 44 in WebUI (before the latest additional red below). Touched ESLint/syntax baseline/current are 0 errors/0 warnings; configured semantic types have 0/0 touched diagnostics and 7/7 connected inherited diagnostics in each client, not whole-project qualification. New test formatting preserves structural AST and all original test regions. TS-only Bandit is inapplicable. No commit/build/native S05 qualification yet. Review still requires preserving pending newer Refresh/next-page append intent after bulk ACK and retiring only successful bulk targets. An additional actual loaded-filter regression now fails: bulk ACK can clear a current approved detail via captured needs_review filters. Assess a mutation-version fence with same-intent post-write query replacement and current page observation before further source changes. Separate canonical authority lease/filter-preset concern remains unqualified. Acceptance stays 41/43; no S06.


## Qualified request-ownership unit

S05 moderation read/selection repair is locally qualified and actual-source review CLEAR. The original race allowed delayed item details to disagree with the selected decision target; connected mutation callbacks, filter snapshots and bulk checkbox state had the same ownership problem. The final hook preserves queue/detail tickets, current selection and page state, captured action/Undo identity, and lifecycle retirement. A successful write invalidates stale GET results within the same logical filter/append request; unsuccessful bulk targets retain their own detail reads. Bulk updates preserve new checkbox selections and Clear. A successful response lacking refreshed item data retains its ACK/result but requires explicit Refresh before stale detail can be used. Twenty-four new cases include 21 causal failures across baseline/incomplete candidates plus three boundary controls; these are not 21 separate shipped bugs or live UAT results. Final five-module checks: 53 passed UI (5.088s outer), 53 passed WebUI (5.307s outer), natural0. Touched ESLint/syntax are 0 errors/0 warnings, no ignored paths; inherited Next pages-directory console diagnostic remains. Configured semantic types: touched0/0 and connected7/7 inherited diagnostics in each client; no whole-project typecheck claim. New test formatting has identical structural AST; every original fixture/test/assertion region is byte-identical. TypeScript-only Bandit is inapplicable. No new durable rule/ADR: existing ownership and request contracts govern. Prepare normal hooks, exact five-path staging/review/local commit (two source/test paths, own S05 plan, parent task, completed own S04 plan deletion). No publication/build/native acceptance. Canonical authority/preset concern and all S05 live/rules/claims/role/partial-failure/UX gates remain open; 41/43.


## Canonical authority and private presets

S05 request-ownership unit committed locally as 8a73f66a0e7f5e28eebc6f737f84879ab4677912, parent ca3c20768f50b682ad6f005642580bef80963263. Normal commit exactly matches reviewed five-path 414,336-byte patch SHA256 b1bcdb366b71f1f5871d43f50ffe00336410e291c7b9c7a929d6a7ce0832db65; tracked files clean at commit, no publication. Normal pre-commit passed in 1.764s; final actual/staged reviews CLEAR. Backend unchanged51f0; retained frontend buildJag still represents ca3, so it does not qualify the new fix live. Next distinct S05 authority/preset unit: inspect canonical-loading/account/server transitions with real moderation controllers and controlled transport mocks. Current Layout does not key route content by authority; review hook owns no canonical identity and loads globally keyed filter presets. Rules queries also use unscoped settings/policy/override cache keys. Shared direct transport already watches account retirement, so reuse its guarantees rather than assume transport retargeting or rewrite it speculatively. First causal tests target resident private review data and unowned legacy presets; follow existing Claims/Kanban canonical boundary patterns if confirmed. No runtime/build yet; 41/43, S05 open.


## Review view and preset isolation repair

S05 Review-view/preset authority defect confirmed by six causal failures on unchanged8a73: reads before canonical readiness, private resident data during rehydration/server/account change, late old queue publication, and legacy global preset labels exposed to another account. Minimal repair reuses Claims/Kanban-style canonical gating and existing usePromptLocalAuthority transport-owned identity: require ready/non-null owner and matching canonical boundary, key the inner controller by boundary+opaque owner, and require immutable owned preset namespace for every read/write/delete. There is no local-only/null fallback; legacy unowned key remains byte-identical and unread by new scoped presets. Existing request-ownership lifecycle guards retire old controller publication; AuditTimeline is props-only, with no separate transport work needed. Six causal cases now pass plus five boundary controls: cookie principals with unchanged connection hash, resolver-boundary disagreement, owner-unavailable Retry, null-owner refusal, and owner-specific preset round trips. The tests use the real owner hook with a controlled transport-owner resolver. Existing ready-controller tests bind an explicit canonical/owner fixture; no original action/assertion body is weakened. This qualifies view/preset isolation only; captured mutation dispatch authority and Rules query-cache scope remain separate open concerns. No shared transport/hook rewrite, build, native acceptance or publication. Acceptance41/43.


## Review isolation qualification

S05 Review view/preset isolation qualified locally: six causal baseline failures; twelve authority controls plus all original Review modules pass, 65 UI (4.121s outer) and 65 WebUI (4.318s), natural0. Reviewer requested delete callback dependency and deletion isolation; corrected dependency and added owner-B deletion preserving exact owner-A and legacy bytes, passing within both full scopes. Removed unsupported Testing Library exact option without changing its default exact-name behavior; focused deletion final1passed/11deselected. New wrapper/new test formatting preserves structural AST, original Review body retained, existing test bodies byte-identical. Touched ESLint/syntax0 errors0 warnings; TypeScript-only BanditN/A. Configured semantic validation includes the normal Vitest setup for both baseline/current; initial baseline-root omission is an operator artifact and not accepted. No build/native acceptance or publication; captured dispatch authority and Rules/private cache assessment remain open; 41/43 accepted.


## Rules view and cache isolation

S05 Rules view/cache defect confirmed by six causal baseline failures; all now pass. Minimal Root canonical/transport-owner gate keys a private Rules subtree; its lazy house createQueryClient preserves normal retry/focus defaults, existing keys/invalidation work inside it, and retirement cancels/clears only its client. Seventh control verifies real StrictMode replay plus subsequent policy invalidation/refetch and exact surrounding-cache preservation. All21 affected modules235cases qualified compositionally in each client: initial222passed13ready-fixture failures, then three corrected modules20passed (including7authority overlap), UI3.526s/Web3.494s outer. Four existing controller fixtures now explicitly ready and partial ReactQuery mocks; every existing describe/test/assertion body byte-identical, original Rules workspace body unchanged. Lint/syntax: no new findings; Root4 and quick-list4 inherited warnings. Semantictypes: UI touched0/0; Web touched1/1 inherited MemoryRouter initialEntries type diagnostic; connected7/7 inherited each, no new diagnostics, not wholeprojectqualification. TS-onlyBanditN/A. Independent actualsource reviewCLEAR. No new durable ADR; existing ownership contracts. Captured follow-up dispatch remains next: override bulkDelete catches failures and continues across awaits, so view/cache disposal alone does not qualify that boundary. No build/live/publication; S05open41/43.


## Retired override bulk continuation

S05 bulk override continuation causal red2failed2passed on unchanged07d785: after unmount or principal retirement, the catch-and-continue loop sent another DELETE (including replacement-account credentials). Actual bgRequest already refused a principal change during decision credential resolution; no shared transport rewrite needed. Minimal useUserOverrides lifetime revision plus existing watchChatAccountChanges now retires immediately/on cleanup; bulk asserts before every DELETE and before state/refetch, throwing AbortError on retirement without retrying an uncertain first write. Six controls include real transport/storage/fetch boundaries, immediate account event without unmount, retained callback/noHTTP and final-ACK/no-refetch. Real same-owner409 partial failure still continues only intended targets. Three affected modules32passed UI3.234s/Web3.211s outer natural0; lint/syntax0errors0warnings; configuredtypes touched0/0, connected7/7 inherited each, no new, not wholeprojectpass. New test formatted with unchanged structural AST; existing test files untouched. TypeScript-onlyBanditN/A. Independent review pending. Build/live/Claims/role/partialfailure/UX acceptance remain open41/43; no publication.


Independent S05 continuation review found a final-completion gap: retirement during awaited final refetch could still resolve success. Seventh causal control failed on partial fix (resolved[]), then passes with one post-refetch assertCurrent in UI0.763s/Web0.759s; passing previous32-case scopes not replayed. Combined affected33cases each, seven new controls incl3causal failures across baseline/incompletecandidate. Retired operations now abort before next delete/refetch and before final completion reporting. Final source/staged review and normalhooks follow; all build/live acceptance gates stay open.
