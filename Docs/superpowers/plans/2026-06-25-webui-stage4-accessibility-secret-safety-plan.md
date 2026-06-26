## Stage 1: Command Palette Trigger Contract
**Goal**: Ensure the WebUI header exposes a stable command palette trigger and opens the palette from pointer and keyboard entry points.
**Success Criteria**: Header search trigger has the accessible name `Open command palette`, keeps the visible shortcut hint, and dispatches the command palette open event on click.
**Tests**: Focused component tests for `ChatHeader` and `Header` trigger behavior.
**Status**: Complete

## Stage 2: Palette Focus Management
**Goal**: Preserve keyboard and screen-reader flow when the command palette opens and closes.
**Success Criteria**: Opening the palette focuses its search input; Escape closes the palette and restores focus to the element that invoked it.
**Tests**: Focused `CommandPalette` tests for Ctrl/Cmd+K and custom event open flows.
**Status**: Complete

## Stage 3: High-Risk Route Governance
**Goal**: Extend responsive/a11y route governance to the audit's remaining high-risk WebUI routes.
**Success Criteria**: Responsive governance includes `/`, `/settings/health`, `/admin/server`, and `/scheduled-tasks` in addition to existing chat/media coverage; Axe high-risk route list includes health, admin server, scheduled tasks, and media where metadata allows.
**Tests**: Focused route governance helper/unit coverage and smoke spec list verification.
**Status**: Complete

## Stage 4: Secret-Safe Diagnostics Confirmation
**Goal**: Keep visible and copied diagnostics redacted for secret-shaped keys.
**Success Criteria**: Existing health diagnostics redaction remains covered for API keys, Authorization, bearer/token/password/cookie-shaped fields in raw details and clipboard payloads.
**Tests**: Focused health diagnostics redaction tests.
**Status**: Complete

## Stage 5: Verification
**Goal**: Prove the scoped accessibility and governance work is ready without expanding unrelated blast radius.
**Success Criteria**: Focused Vitest/Playwright checks pass or any environment blockers are documented; lint and diff checks pass; Bandit is confirmed not applicable for TS/TSX/docs-only changes.
**Tests**: Focused Vitest, focused smoke checks where feasible, ESLint on touched files, `git diff --check`.
**Status**: Complete
