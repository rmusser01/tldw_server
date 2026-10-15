# Chat UX follow-up — TASK-13710

Approved design: TASK-13709 review and the user's acceptance of character-first roleplay and filtered, paginated compact character history. Implement existing-flow refinements only. New PR targets latest remote dev. ADR required: no; existing API, ownership, persistence and security boundaries remain unchanged.

## Stage 1: Connection-first setup
**Goal**: Accurate prerequisite/readiness and working inline Server settings recovery; compact unconfigured first-run guidance.
**Success Criteria**: Fresh and disconnected roleplay never diagnoses model setup before connection; configured auth failures retain recovery detail; draft/owner guards remain.
**Tests**: Disconnected/null-health regression, server-settings action routing, initial/failed-account states.
**Status**: Complete

## Stage 2: Controlled draft and session discovery
**Goal**: Starter suggestions update form state; compact character history uses filtered server pagination, states and row metadata; active legacy picker exposes fetch errors.
**Success Criteria**: Suggestions survive rerenders; old tracked chats are reachable beyond first page; loading/error do not claim empty; duplicates are distinguishable.
**Tests**: Both prompt suggestions through controls/send boundary; paged filtered query, next/previous, loading/error/retry, selected and duplicate rows; picker failure/retry.
**Status**: Complete

## Stage 3: Roleplay and customization hierarchy
**Goal**: Character first; optional scene/behavior, collapsible preview/reuse; accessible before/after semantics; task-focused terminology and advanced customization disclosure.
**Success Criteria**: Primary selection visible at laptop/mobile sizes; all advanced actions remain reachable; staged Apply/Cancel and ownership checks remain intact.
**Tests**: Roleplay drawer interaction and changed-value semantics; customization disclosure and existing action events; shared WebUI/extension profiles.
**Status**: Complete

## Stage 4: Qualification and landing
**Goal**: Independent review, relevant tests/lint/types/builds and bounded desktop/mobile browser validation; new PR and latest-dev merge when qualified.
**Success Criteria**: Review findings resolved; no new touched-scope security findings; source and protected-data evidence recorded; required checks pass; merge policy/human summary satisfied.
**Tests**: Appropriate unit/integration/E2E, production builds, original nine-chat/28-row preservation, normal precommit hooks, exact-head CI.
**Status**: In Progress

Execution note: setup/controlled-starter regression group passes47 cases. Character history passes21 with independent pagination review and exact-multiple boundary repair. Customization test reassessment: the remaining four failures all asserted Compare inside the former primary Response group; the new approved Advanced section relocates it. All assertions remain and now scope the visible advanced control. No production fallback, deadlines or assertions were disabled.

Qualification update: PR https://github.com/rmusser01/tldw_server/pull/3248 is open against dev. Implementation and independent exact-source review are complete; all local production/type/test gates pass. Requester-written Change summary is included verbatim in PR3248. Required GitHub CI remains before merge.
