## Stage 1: Sidepanel Role-Play Handoff Contract
**Goal**: Preserve Character Chat intent when the extension sidepanel opens the full `/chat` app.
**Success Criteria**: Sidepanel full-app links target `/chat?mode=character`, include `characterId` when a character is selected, and keep persona selections in Character Chat mode.
**Tests**: Focused unit tests for the handoff URL helper.
**Status**: Complete

## Stage 2: Visible Sidepanel Role-Play State
**Goal**: Make active Character Chat state visible in the sidepanel without forcing users into settings.
**Success Criteria**: The sidepanel control row exposes a compact Character Chat chip with current Character/Persona labeling and clear/switch affordances.
**Tests**: Source/contract tests for visible chip and action wiring.
**Status**: Complete

## Stage 3: Verification And PR Closeout
**Goal**: Verify focused tests, run applicable static/security checks, and package the branch for review.
**Success Criteria**: Focused frontend tests pass; Bandit is skipped/documented for frontend-only changes or run if backend code changes; task notes and final summary are updated.
**Tests**: `bunx vitest run` on focused test files from `apps/tldw-frontend`.
**Status**: Complete
