# X03 usability, accessibility and responsive validation

**Task:** TASK-13260.278.18.83.45.
**Goal:** Finish selected X03 before X04, preserving the approved Chat redesign and all earlier functional acceptance.
**Spec:** RELEASE_UAT_PLAYBOOK.md X-03 and release-catalog-data.mjs X-03: six core variants plus error-focus recovery.
**ADR assessment:** No new ADR for validation. ADR-049 governs Chat identity/continuation; existing profile and task rules remain. Any behavioral repair requires a separate assessment, causal regression, minimal correction, affected checks and independent review.

## Fixed scope and safeguards

- X02 is closed at its recorded e388823/dev17141/frontend9U composition;35/43 families accepted, eight open. Do not replay its completed controls or recreate disposed payloads.
- Before live X03, integrate verified devc90e57e3824a77102939e5b4017c5c0973a00c64 containing merged3310. Its production matches the prior c0cd source review; the final two e2e/task increment is independently clear. PR3311 fad63/basec90 is source-reviewed reference-only and stays unintegrated while open.
- Preserve original API50010/49993, frontend50863/50855, Chrome54716/54710, original source/profile/data/keys, URL/five messages/draft/canonical objects. Work only in the isolated checkout and one disposable browser context/tab through the existing Chrome CDP session. No original service cycle or browser launch.
- Chrome via the existing CDP session; pin its actual version before use. Viewports1440x900,1024x768,390x844; light/dark and English plus one actually advertised non-English locale, pinned before its run. Verify real200% zoom by observed browser metrics, not CSS scaling or a DPI substitute. Responsive emulation is not mobile-device certification.
- Use normal navigation and keyboard controls, representative empty/populated/loading/error states, long multiline/non-ASCII content, repository accessibility tools and measurable geometry/focus/contrast checks. Layout fixture content is not a live provider-result claim. Existing accepted real output/listening judgments retain their recorded scopes.
- Preserve incumbent tokens/layout/product behavior; no unrelated visual redesign or product-context setup. Impeccable context/audit guidance is loaded; no PRODUCT/DESIGN document was invented. Review one batched desktop/mobile pass, fix substantiated issues together, then one confirmation pass. Further causal debugging follows the human-authorized bounded retry/reassessment rules.
-600s workflow/build/run,120s startup/child,180s generation if genuinely needed,300s pytest case,15s operation/graceful cleanup. No force, installs/downloads/models, assertion weakening, raw SQL, proof bundles or foreign evidence/log recovery. No source publication or GitHub mutation. Heartbeat remains paused.

## Stage1: Integrate and qualify incoming rails
**Goal:** Preserve every local fix on verified devc90 and qualify genuinely affected shared/WebUI behavior.
**Success Criteria:** Backup exact current head, preserve notes/untracked inputs/owners, rebase all21 commits with inspected range-diff, unchanged backend/package bytes and dependencies, clear incoming/combined review and affected checks. Current build9U is stale for new rails after rebase until a normal qualified frontend build.
**Tests:** Exact source/head guards, incoming/new-assertion review, range-diff/tree overlap, nine changed shared test modules plus the affected coordinator ownership and locale mirror modules in both configured clients, touched TypeScript parse/lint/locale contracts, scoped security assessment; normal hooks. No unchanged backend suite/export replay.
**Status:** In Progress

## Stage2: Audit declared surfaces and states
**Goal:** Measure discoverability, keyboard/focus, responsive/zoom, state clarity and theme/locale behavior.
**Success Criteria:** Pin browser/locale/surface list and actual artifact bindings; record each of seven variants with positive and recovery observations. Run repository accessibility checks and manually verify findings. Preserve all drafts, accounts and original state.
**Tests:** Normal navigation without pasted IDs; keyboard action/dialog/Escape/focus return; exact viewport and real zoom checks; light/dark/non-English/long text; honest empty/loading/error/completed states. Capture only the screenshots needed for human review.
**Status:** Not Started

## Stage3: Fix and confirm substantiated defects
**Goal:** Repair actual failures at their shared root without changing the redesign.
**Success Criteria:** Causal red/green, affected configured checks, static/security results, independent source review and a bounded live confirmation pass. Keep failed runs failed.
**Tests:** Regression matched to each real defect; no timeout/guard weakening or speculative rebuild/model replay.
**Status:** Not Started

## Stage4: Human review, dispose and close
**Goal:** Selected X03 accepted before X04.
**Success Criteria:** Seven variant dispositions, separate functional and human UX judgments, exact original preservation, final review, accurate master/task counts and guarded removal of completed payloads/operators/own plan.
**Tests:** Review screenshots with the human; normalize task/diff checks, normal hooks/commit where applicable, release/open-handle/ownership cleanup guards. Broader release and native-device limits stay explicit.
**Status:** Not Started


## Merged rails component qualification

X03 merged-rails integration is locally qualified through component/static scope: conflict-free rebase onto dev c90e57e3824a77102939e5b4017c5c0973a00c64 at HEAD775fe34a59222e25fb69e0a6c8412303544ca10c; all21 range-diff entries identical. Backup codex/uat-before-rails-20261024 retains6e8f769. Exact notes stash244e9f82a99678b3d01cbb081d18deeaf090b58a (10414bytes/SHA256004ffb1b7364885f4d6b78097a434e9d1b62afd273f488eb2d69404cc10814ee) is retained and restored byte-identically; untracked plan preserved. Committed tree delta is exactly27 incoming paths. All3817 installed API/data files at /private/tmp/tldw-x02-outbox-20261024/site match current source; no Python/dependency delta or API rebuild. Nine changed rails modules pass181 tests in UI9.971s and WebUI7.504s. Genuinely affected coordinator ownership/retirement plus locale mirror pass83 in UI29.704s and WebUI32.034s; total264/configuration, no failures or pending. TypeScript syntax22files clear; actual lint22files0errors/90warnings exactly baseline-current/no ignored/new findings. First comparator incorrectly distinguished3 embedded hook line numbers; only location text normalization corrected, with pure guards, no source edit. Inherited Pages-discovery diagnostic remains. Bandit N/A: no Python source change; no TypeScript security/full typecheck claim. Independent source/combined preservation assessments clear. Frontend9U is stale for the new rails; reviewed normal build/qualification and X03 live audit remain. Original owners unchanged. No publication, original service cycle, installs/downloads or models.

The next step is a reviewed normal frontend build using existing x01_current_frontend_build.py. It retains source/extension-link/dependency/original-build/owner checks, exact five18908 rewrites, token-sync/bundle budgets and600s overall/15s graceful release. No new install, API package refresh, original-runtime transition or browser action is needed for packaging. Native X03 remains unattempted.
