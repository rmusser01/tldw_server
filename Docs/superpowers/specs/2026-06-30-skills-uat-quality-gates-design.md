# Skills UAT Quality Gates Design

## Context

The `/skills` page has already received staged UX and reliability improvements for beginner onboarding, guided authoring, success actions, server-backed discovery, dense power-user management, safer test-run semantics, dry-render execution, import conflict review, seeded overwrite confirmation, version-aware delete paths, export metadata feedback, and state/copy/accessibility polish.

The remaining risk is regression confidence. Existing component tests cover many detailed states, but workflow-level user acceptance coverage is still narrow. The current Playwright Skills spec includes a mocked beginner seed/test-run path plus live smoke checks. `SkillsManager` and `SkillPreview` Vitest coverage already exercises search, filters, sorting, density, columns, import conflicts, version conflicts, export feedback, seeding confirmation, execution failures, and success actions.

This design turns the Skills audit and staged plans into a full UAT bundle without duplicating every component permutation in Playwright.

Backlog task: `TASK-530.14 - Implement Skills UAT and quality gates`

## Goals

- Add deterministic workflow-level UAT coverage for `/skills` that can run without a populated live backend.
- Cover the beginner activation path, advanced large-library management path, and high-trust failure/recovery states.
- Preserve detailed branch coverage in Vitest instead of moving every state into Playwright.
- Provide a manual QA checklist for accessibility, responsive behavior, browser-specific affordances, and moderated user confidence checks.
- Document success metrics without enabling telemetry by default.

## Non-Goals

- Redesign the `/skills` page UI.
- Add product telemetry or analytics collection.
- Require a live backend for deterministic UAT quality gates.
- Build a general-purpose fake Skills backend that mirrors every API behavior.
- Re-test all existing Vitest permutations through browser E2E.

## Proposed Architecture

Use a three-layer quality model.

### Layer 1: Deterministic Automated UAT

Playwright covers a few complete user journeys with route-mocked API responses:

1. Beginner activation from an empty library to first successful skill use.
2. Power-user discovery and bulk-management preparation in a large library.
3. Representative failure and recovery states that affect user trust.

These tests should live in `apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts` and use narrowly scoped helpers from `apps/tldw-frontend/e2e/utils/skills-fixtures.ts`.

The mocked UAT block must not depend on `skipIfServerUnavailable`. Live-server smoke coverage should remain separate and may keep its current skip behavior.

### Layer 2: Component Regression Coverage

Vitest remains responsible for state permutations that are cheaper and more stable below the browser workflow layer. Existing Skills tests already cover most detailed UI states. Add new Vitest coverage only if implementation reveals a missing branch that is not worth validating in Playwright.

### Layer 3: Manual Release Checklist And Metrics

Create `Docs/Reviews/skills-page-uat.md` for human QA. The checklist should include setup prerequisites, browser/auth assumptions, beginner and advanced workflows, accessibility checks, responsive checks, failure checks, and a coverage column showing whether each row is automated, manual, or both.

The same document should define success metrics that can be measured in moderated QA or optional self-hosted analytics, without turning telemetry on by default.

## Fixture Design

Create scenario-level fixture helpers instead of a broad fake backend:

- `mockBeginnerSkillsJourney`
- `mockPowerUserSkillsLibrary`
- `mockSkillsUnsupportedCapability`
- `mockSkillsImportValidationFailure`
- `mockSkillsExecutionFailure`
- `mockSkillsStaleVersionConflict`
- `mockSkillsSlowList`

Each helper should mock only the endpoints required by its scenario, including readiness/capability gates when needed:

- `GET /api/v1/health`
- `GET /api/v1/config/docs-info`
- `GET /openapi.json`
- `GET /api/v1/skills`
- `POST /api/v1/skills/seed`
- `GET /api/v1/skills/context`
- `GET|PUT|DELETE /api/v1/skills/{name}`
- `POST /api/v1/skills/{name}/execute`
- `POST /api/v1/skills/import`
- `POST /api/v1/skills/import/file`

The helper API should expose enough request capture to assert important workflow behavior, such as search query params, filter params, sort params, delete version payloads, and execution payloads.

## Automated UAT Scenarios

### Beginner Activation

Expected flow:

1. Visit `/skills` with an empty Skills list.
2. See beginner orientation and first actions.
3. Seed built-in skills.
4. Confirm a seeded skill appears.
5. Open a test run for the seeded skill.
6. Enter arguments and run a dry render or explicit test run.
7. Confirm result or rendered output appears.
8. Use `Copy invocation` and verify either clipboard contents or a reliable success state.

Acceptance criteria:

- The flow completes without reading external documentation.
- No unrelated blocking overlays appear.
- The UI gives clear success feedback after seeding and running.
- Clipboard verification is automated only if the existing Playwright environment supports it reliably; otherwise the manual checklist owns clipboard-value validation.

### Power-User Large Library

Expected flow:

1. Mock at least 30 skills with a target skill outside page 1.
2. Search by exact target name.
3. Assert the request includes the query and the target skill becomes visible.
4. Apply at least one meaningful filter, such as fork-mode or tool-using skills.
5. Sort by a supported column.
6. Select two rows.
7. Open bulk export or bulk delete confirmation without confirming destructive deletion.

Acceptance criteria:

- Assertions focus on user-visible workflow outcomes and captured API query semantics.
- The test does not depend on every row order unless row order is the behavior under test.
- No destructive action is submitted before confirmation.

### Failure And Recovery States

Cover one representative scenario per trust-risk category:

- Unsupported Skills API capability: Skills-specific capability message appears.
- Invalid import preview: validation feedback appears and no import mutation is applied.
- Execution failure: alert appears with useful recovery affordance.
- Stale delete conflict: user is told to reload before retrying.
- Slow list request: loading state appears and duplicate submissions are prevented.

Acceptance criteria:

- Each failure explains what happened and what the user can do next.
- Tests do not enumerate every API error status.
- Scenarios remain deterministic and independent.

## Manual UAT Checklist

`Docs/Reviews/skills-page-uat.md` should include:

- Environment setup: backend URL, auth mode, seeded data assumptions, browser list, and whether the Skills API is expected to be available.
- Beginner pass/fail rows.
- Advanced pass/fail rows.
- Accessibility checks for keyboard navigation, focus return after dialogs, visible focus, form labels, table/action names, alerts, and screen-reader clarity.
- Responsive checks for narrow desktop and mobile-width extension contexts.
- Failure checks for offline server, unsupported API, invalid import, stale version conflict, execution failure, and slow loading.
- Coverage column: `Automated`, `Manual`, or `Both`.
- Evidence notes field for tester screenshots or observations.

## Success Metrics

Document metrics only; do not add telemetry by default.

- Beginner task completion rate: percentage of first-time users who seed or create a skill and run or dry-render it successfully.
- Time to first successful skill use: time from landing on `/skills` to visible successful result or rendered prompt.
- Power-user search/filter success: percentage of attempts where the target skill is found without browsing pages manually.
- Configuration recovery success: percentage of users who recover from unavailable or unsupported Skills API states.
- Error rate categories: import validation, execution failure, stale version conflict, connectivity/capability, search miss.
- User confidence rating: moderated test prompt after workflow completion, using a 1-5 confidence scale.

## Quality Gates

Implementation verification should include:

- `npx playwright test apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts`
- Focused Skills Vitest only when touched or when new component-level coverage is added.
- Docs review for `Docs/Reviews/skills-page-uat.md`.
- `git diff --check` on touched files.
- Bandit skipped if the slice remains frontend/docs-only; run scoped Bandit if Python files are touched.

The deterministic mocked UAT block is the release gate. Live-server smoke tests remain valuable but should not be required for deterministic CI success unless the environment explicitly provisions a compatible backend.

## Risks And Mitigations

- Risk: Playwright suite duplicates component tests and becomes slow.
  Mitigation: limit browser tests to full workflows and representative failure categories.

- Risk: Fixtures become a fake backend.
  Mitigation: scenario-level helpers with endpoint coverage only where needed.

- Risk: Large-library tests break on harmless table presentation changes.
  Mitigation: assert API semantics, target visibility, and user actions rather than full row ordering.

- Risk: Clipboard assertions are flaky.
  Mitigation: automate success feedback and use manual UAT for exact clipboard contents if browser permissions are unreliable.

- Risk: Manual checklist drifts.
  Mitigation: include coverage mapping and test file references for automated rows.

## Open Implementation Notes

- Confirm the exact button labels and test IDs in the current `/skills` page before writing assertions.
- Prefer additive fixture helpers over editing global E2E setup.
- Keep unrelated Skills UI changes out of the UAT PR unless a failing test exposes a true blocker.
- Record any skipped manual checks or environment limitations in the Backlog task final notes.
