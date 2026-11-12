# Chat UX follow-up qualification — TASK-13710

Base: remote `dev` at `2e48dc91848887dcb74ebe39d6ff877fed4d64bd`, fetched again before publication. Follow-up to the six findings and enhancements in [TASK-13709's reassessment](CHAT_LATEST_DEV_UX_REASSESSMENT_13709_2026_10_15.md). The requester selected character-first roleplay and filtered, paginated compact character history.

ADR assessment: no new ADR required. This work reuses the existing shared UI, connection hook, settings-navigation event, controlled composer event, owner-scoped history query and server pagination. It changes no persistence, authentication, protocol or ownership rule.

## Changes and rationale

| Finding | Resolution |
|---|---|
| Offline character readiness and wrong recovery destination | Require the actual connection state together with health readiness. Server recovery uses existing settings navigation, supporting inline WebUI configuration and extension routing. |
| Compact starters lose their text on rerender | Send both suggestions through the existing controlled-composer event rather than mutating a DOM value. |
| First-run competing failures/actions | Present compact connection-first setup. Suppress duplicate offline guidance only while the empty timeline owns initial setup; retain configured-account errors and draft recovery guards. |
| Character choice buried in roleplay | Put Character and Behavior first, then optional Scene details and generation. Preview changes and saved setups use disclosures; saved setups open when present. Editors stay mounted and staged Apply/Cancel ownership checks remain. |
| Ambiguous before/after preview | Label Current and After applying for assistive technology and wrap long values. |
| Compact history samples before filtering | Request character-only global history in six-row server pages with verified ownership. Show loading, failure, stale, verification and retry states; expose next/previous and truthful empty pages. |

Remaining requested enhancements are included: Context/Chat settings task labels; seven common customization actions with specialist controls under Advanced customization; preserved expert controls and focus repair; session timestamps, selected state and duplicate-title references; legacy character/persona picker loading/error/retry feedback. Unsupported tracked persona chat remains unavailable with guidance, as previously chosen by the requester.

## Verification

- Regression tests reproduce both starter failures on the original source after a controls rerender. Fixed source passes all 34 controlled-form/model-validation cases, including each actual starter through the production message effect and submit callback, exactly one send each.
- All 16 touched shared UI test files pass under the WebUI shim profile: **362 tests**. Extension-profile qualification covers the changed setup, notices, toolbar, picker, history, shell and controls suites. The roleplay worker also ran 86 cases under native, extension and WebUI profiles.
- Independent history review found an exact-multiple pagination boundary error; repaired and re-reviewed. Independent full source review found two P2 qualification gaps (duplicate setup guidance and insufficient starter integration evidence); both repaired and cleared on scoped re-review. No remaining actionable source finding was reported.
- Fresh isolated source copy, Bun frozen lockfile, production WebUI build, WebUI typecheck, strict shared-boundary types, production Chrome extension build and extension compile all pass. The WebUI build skips typechecking by repository configuration, so its separate successful typechecks are necessary evidence.
- Source provenance verifies 4318 non-test app files against current source and the build copy; lockfile unchanged. All 1130 static asset references across 969 emitted JSON manifests exist. Build ID `GTjD5DT8kMnjhBPajljq0`. Token synchronization and bundle budget pass; shared app 591.7 KB gzip within 600 KB.
- Scoped ESLint:31 changed TS/TSX files, **0 errors**,163 existing warnings. Warning identities match the base after normalizing reported line shifts. No rule was disabled. `git diff --check` passes. No Python changed, so Bandit has no touched Python scope.
- Bounded headless WebUI walkthrough at 1280×720 and 390×844: connection setup visible with one primary action, inline Server settings opens, typing a draft restores offline recovery, normal Save/Test Connection succeeds, a disposable saved conversation loads and survives reload, Advanced summary works with Enter/Space, Character is visible first, and the settled mobile drawer fits the viewport (x 0, width 390).
- Fresh installed production Chrome extension: both starter prompts survive actual Character controls opening and closing; all six authoritative older rows load on character-history page 2 (147 character sessions available); Previous returns to page 1; full-page roleplay exposes Character first at desktop/mobile sizes. Only the temporary installed fixture copy receives loopback host permission; the production manifest is unchanged.
- Original nine protected conversations and 28 message rows remain deep-equal to the immutable baseline before and after qualification. Browser checks read existing disposable fixtures; they send no paid model requests. Owned headless browser contexts are closed; no visible Chrome window was opened.
- One scoped mechanical UI scan reports no findings on the coordinator's changed components; the roleplay worker's scoped scan also reports none. Screenshots were inspected; loading/transition captures were excluded from settled-layout evidence.

Private logs, source hashes, screenshots and receipts are retained under `/private/tmp/chat13710-*`; credential-bearing browser state is not included in the repository.

## Limits and landing

Physical VoiceOver/NVDA, actual browser permission-grant UI, and separate Firefox/Edge runtime builds were not checked. Accessibility semantics are covered by automated tests and rendered accessibility snapshots. Existing whole-package typecheck attempts exhausted a 4 GB heap; fresh locked production typecheck/strict/extension compile gates subsequently passed with the repository's configured memory budget.

The PR must pass current required CI and the repository's human-written Change summary merge gate before landing. Merge-queue variable was unset at publication, so latest-dev rebase and exact-head checks apply.

PR: https://github.com/rmusser01/tldw_server/pull/3248. Independent final source review cleared commit `98eda36cfa19705c9ff8308c0c45905e6e3ac5ea`. Explicit pre-commit checks passed on all 41 implementation/review/task files; local fast CI passed under project Python 3.12 (compileall, syntax and repository guards; Python tests/lint skipped because no Python changed).

## CI qualification follow-up

GitHub CI exposed stale fixture prerequisites and assertions: two tour fixtures omitted connected state/the connection selector, a research query mock omitted the required query-client export, Buddy integration still expected an advanced control to be immediately visible, and four responsive smoke cases queried the old Runtime label. The production behavior and CI ratchet/pinned policy were unchanged. Fixtures now seed and restore the correct state, reuse the adjacent partial React Query mock pattern, and preserve primary controls plus hidden-before/visible-after Advanced assertions.

After observed red runs, all **257 native UI + 47 WebUI cases (304 total)** pass with fresh frozen dependencies and the exact deterministic CI config. All **four** previously failing responsive cockpit smoke cases pass against the unchanged production artifact. Independent scoped review reports no weakened assertion or actionable boundary/cleanup issue. Final lint covers 36 PR TS/TSX files: 0 errors and 198 warnings matching the base after line-shift normalization. Production source/artifact provenance remains unchanged. The requester supplied the Change summary verbatim in PR3248; the human ownership gate is satisfied. Fresh final-head GitHub CI must pass before merge.

The verified implementation task is complete. Its owned execution plan was retired after local qualification; commit history preserves it. Final integration continues through the PR’s exact-head checks and authorized merge.

The broader final UX run then exposed one shared E2E contract still assuming specialist controls lived in Research & context. Both focused failures stopped there before their actual workflow assertions. The helper now retains primary controls, opens Advanced customization only when closed, and checks Tools and mobile server diagnostics in Advanced composer controls. The only Scene-notes editing caller also explicitly opens Scene details. Common-only setup and production source remain unchanged.

The actual helper reproduced the failure against the production preview, then passed twice each at desktop and mobile sizes after repair; Scene notes was editable through its disclosure at both sizes. Independent scoped review found no actionable issue or weakened assertion. Earlier final-head CI passed all eight frontend unit shards and the earlier UX stages (39 passed/1 pre-existing skipped, 14 passed, 6 passed); the repaired focused workflows still require fresh full CI before merge.

## Latest-dev integration and cold-drawer qualification

Rebased onto remote dev f5051e3 (including PR3033 restoration and temporary character/persona fixes). All five existing patches remain identical by range-diff, and independent integration review confirms upstream guards and temporary-session preservation remain intact. Fresh frozen production builds, WebUI/strict/extension typechecks, and artifact provenance pass: 4318 production files match; 1130 asset references have no missing files. Nine focused integration suites pass 338 cases. Fresh headless WebUI/extension walkthroughs requalify saved-chat reload, controlled starters, older filtered history, keyboard disclosure, and character-first roleplay at desktop/mobile sizes. Original nine conversations and 28 message rows remain unchanged.

Inspection of a report-only native coverage run found a pre-existing cold React.lazy drawer race: synchronous button queries ran before its Suspense boundary resolved. Exact isolated coverage reproduces the same three missing-button failures on both the PR and current-dev production source. Five first-control queries across the related late-settings and neutral-identity cases now await findByRole, following the existing helper pattern. All ten affected scenarios pass native coverage without changing default deadlines; application, late-completion, unmount, identity and persistence assertions are unchanged. Independent scoped review is clear. The broad report-only shared UI run reached its existing 15-minute cap; required package-owned shards passed. Final exact-head CI remains required before merge.

The full native Form coverage run completed with 210 passing cases and four runner deadline failures under its default five-second limit; all ten repaired cold-drawer cases passed. All seven variants of the slow cases then passed in isolation with the same coverage and deadline. The required package-owned 214-case shard had already passed on the rebased head. No deadline was increased, and the full native coverage run is not claimed green.

Fresh required WebUI CI also exposed an unchanged VN recovery test race: finding an error precedes a separate passive effect that clears initial loading. Its loading-absence assertion now uses the existing waitFor utility. Every retry, disabled-action, no-request, receipt and account-scope assertion remains intact. Independent review cleared this one-line settlement fix.

All 257 VN cases pass native coverage after the settlement repair. The ten cold-drawer scenarios and seven isolated slow-case variants also pass native coverage with unchanged default deadlines. Final source qualification remains the fresh builds/typechecks, 338 integration cases, browser receipts and preservation checks above. Only test readiness and tracking changed afterward; fresh exact-head CI is required before merge.
