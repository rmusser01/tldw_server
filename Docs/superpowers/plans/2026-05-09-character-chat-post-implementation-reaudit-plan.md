# Character Chat Post-Implementation Re-Audit Plan

> For implementation agents: use Puppeteer/Chrome-driver evidence, not Computer Use, unless the user explicitly changes that constraint.

**Goal:** Repeat the first-time and returning-user character-chat walkthrough after remediation packages land, compare against the 2026-05-09 baseline, and document remaining defects or regressions.

**Primary baseline:**
- `Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_AUDIT_2026_05_09.md`
- `Docs/Reviews/assets/2026-05-09-character-chat-ux/puppeteer-states.json`

**Likely surfaces:**
- `Docs/Reviews/`
- `Docs/Reviews/assets/`
- `apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts`
- Temporary Puppeteer scripts under `/private/tmp` or a repo test utility if formalized

## Stage 1: Define Re-Audit Protocol

**Goal:** Make the second walkthrough comparable to the baseline.

**Success Criteria:**
- First-time and returning-user task scripts are written before browser execution.
- Environment requirements are explicit: backend profile, model/provider state, database state, and browser driver.
- The audit distinguishes observed evidence from interpretation.

**Tests:** Protocol review only.

**Status:** Complete

Steps:

- Reuse the original persona definitions.
- Decide whether to test against recovered default DB, fresh temp DB, or both.
- Decide whether to use a mocked/test model provider so final message generation can be exercised.
- Define screenshot names and state-capture format.

Notes:

- Protocol written in `Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_REAUDIT_2026_05_09.md` before browser execution.
- Re-audit uses a fresh temp DB profile. The default DB corruption remains a baseline blocker and is not overwritten.
- Model/provider availability will be observed live; no provider mocking is introduced for this documentation-only re-audit.

## Stage 2: Run First-Time User Walkthrough

**Goal:** Verify the character-chat first-run path from connection through first message or model-ready blocker.

**Success Criteria:**
- User can discover character-chat path from first-run state.
- User can create or import a character.
- Missing model state is in-context, or model-ready state reaches first send.
- Screenshots and DOM state are captured.

**Tests:** Puppeteer walkthrough.

**Status:** Complete

Steps:

- Start frontend and backend in a controlled profile.
- Capture Home/connection state, Characters entry, create/import, model readiness, and chat start.
- Record blockers exactly if dependencies are missing.

Notes:

- Puppeteer captured direct `/characters`, first-run splash, explicit character-chat onboarding intent, character creation route, successful UI character creation, and chat/no-provider states.
- The clean first-run path remains blocked by the generic `Build Your Assistant` splash and Persona detour before the new character-chat onboarding lane can render.
- Final message generation remained blocked by missing LLM provider configuration.

## Stage 3: Run Returning-User Walkthrough

**Goal:** Verify returning-user resumption, search, edit, and quick chat.

**Success Criteria:**
- Search count matches filtered state.
- Primary `Chat` action is visible and preserves selected character.
- Edit flow remains available.
- Chat header character mode sequences character before scene.
- Screenshots and DOM state are captured.

**Tests:** Puppeteer walkthrough.

**Status:** Complete

Steps:

- Seed or create at least one known unique character.
- Search, edit, start row chat, start header character mode, and switch/resume if available.
- Compare route transitions against the baseline failures.

Notes:

- Created `Reaudit Character 20260509173128` via the UI drawer; the create request returned `201`.
- Search and edit were captured.
- Row-level `Chat as...` still navigated to `/` Companion Home in the live app, so the baseline intent-loss failure remains.

## Stage 4: Produce Comparison Report

**Goal:** Give the team a clear pass/fail view against the original work packages.

**Success Criteria:**
- New report lists resolved findings, partially resolved findings, new regressions, and remaining blockers.
- Each claim links to screenshots or JSON state captures.
- The report says whether another implementation pass is required.

**Tests:** `git diff --check` on report and artifact paths.

**Status:** Complete

Steps:

- Create a dated report under `Docs/Reviews/`.
- Store assets under a dated `Docs/Reviews/assets/` directory.
- Include a matrix for the eight work packages.
- Record verification commands, model/provider state, and any skipped send behavior.

Notes:

- Report written to `Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_REAUDIT_2026_05_09.md`.
- Assets written to `Docs/Reviews/assets/2026-05-09-character-chat-reaudit/`.
- Report records remaining blockers and the missing-provider send limitation.

## Risks

- Without a configured test model/provider, final message generation may remain untested.
- If the DB recovery package has not landed, the default profile may still fail startup.

## Handoff Notes

Run this only after the other work packages are implemented or explicitly marked out of scope. The comparison report should not silently excuse missing dependencies; it should document them as blockers.
