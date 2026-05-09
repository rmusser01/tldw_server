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

**Status:** Not Started

Steps:

- Reuse the original persona definitions.
- Decide whether to test against recovered default DB, fresh temp DB, or both.
- Decide whether to use a mocked/test model provider so final message generation can be exercised.
- Define screenshot names and state-capture format.

## Stage 2: Run First-Time User Walkthrough

**Goal:** Verify the character-chat first-run path from connection through first message or model-ready blocker.

**Success Criteria:**
- User can discover character-chat path from first-run state.
- User can create or import a character.
- Missing model state is in-context, or model-ready state reaches first send.
- Screenshots and DOM state are captured.

**Tests:** Puppeteer walkthrough.

**Status:** Not Started

Steps:

- Start frontend and backend in a controlled profile.
- Capture Home/connection state, Characters entry, create/import, model readiness, and chat start.
- Record blockers exactly if dependencies are missing.

## Stage 3: Run Returning-User Walkthrough

**Goal:** Verify returning-user resumption, search, edit, and quick chat.

**Success Criteria:**
- Search count matches filtered state.
- Primary `Chat` action is visible and preserves selected character.
- Edit flow remains available.
- Chat header character mode sequences character before scene.
- Screenshots and DOM state are captured.

**Tests:** Puppeteer walkthrough.

**Status:** Not Started

Steps:

- Seed or create at least one known unique character.
- Search, edit, start row chat, start header character mode, and switch/resume if available.
- Compare route transitions against the baseline failures.

## Stage 4: Produce Comparison Report

**Goal:** Give the team a clear pass/fail view against the original work packages.

**Success Criteria:**
- New report lists resolved findings, partially resolved findings, new regressions, and remaining blockers.
- Each claim links to screenshots or JSON state captures.
- The report says whether another implementation pass is required.

**Tests:** `git diff --check` on report and artifact paths.

**Status:** Not Started

Steps:

- Create a dated report under `Docs/Reviews/`.
- Store assets under a dated `Docs/Reviews/assets/` directory.
- Include a matrix for the eight work packages.
- Record verification commands, model/provider state, and any skipped send behavior.

## Risks

- Without a configured test model/provider, final message generation may remain untested.
- If the DB recovery package has not landed, the default profile may still fail startup.

## Handoff Notes

Run this only after the other work packages are implemented or explicitly marked out of scope. The comparison report should not silently excuse missing dependencies; it should document them as blockers.
