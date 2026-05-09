# Character Chat WebUI UX Re-Audit

Date: 2026-05-09
Backlog task: TASK-170
Audience lens: PhD-level UX/HCI review using cognitive walkthrough, task analysis, heuristic evaluation, and error-recovery analysis.

## Re-Audit Protocol

This re-audit repeats the original 2026-05-09 character-chat walkthrough after the implemented remediation packages. Browser evidence must come from Puppeteer with Chrome for Testing, not Computer Use.

### Environment

- Frontend: `apps/tldw-frontend`, `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev -- -H 127.0.0.1 -p 8080`.
- Backend: `AUTH_MODE=single_user`, `SINGLE_USER_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY`, `DATABASE_URL=sqlite:////private/tmp/tldw-character-reaudit-20260509/users.db`.
- Database profile: isolated temporary SQLite profile for the re-audit. The original default DB corruption is treated as a known baseline blocker and is not overwritten during this run.
- Browser: fresh Puppeteer user-data directory for first-time persona; persistent profile within the same run for returning-user persona.
- Model/provider state: observed live. If no chat-capable model is configured, message-generation attempts are recorded as readiness blockers rather than inferred failures.

### Evidence Format

- Screenshots: `Docs/Reviews/assets/2026-05-09-character-chat-reaudit/*.png`.
- State capture: `Docs/Reviews/assets/2026-05-09-character-chat-reaudit/puppeteer-states.json`.
- Each captured state records persona, step, URL, title, visible text excerpt, important accessibility names, console errors, and network failures.

### First-Time Character-Chat Persona Script

Persona: a new local/self-hosted user whose first goal is to create or import a character and start a character chat.

Steps:

1. Open `/characters` in a clean browser profile.
2. Observe whether the route preserves character-chat intent or redirects to generic setup.
3. Complete or bypass connection setup using the controlled local backend.
4. Verify whether character-chat onboarding offers create, import, model setup, and start-chat actions.
5. Create a minimal character from the Characters surface.
6. Attempt to start chat from the character row/card.
7. Record whether model-readiness blockers are kept in character-chat context.

### Returning Character-Chat Persona Script

Persona: a regular user with at least one known character who wants to search, edit, and start or resume a chat quickly.

Steps:

1. Reuse the same browser profile after character creation.
2. Search for the unique test character and compare visible result count with table state.
3. Open edit and verify terminology, hierarchy, and advanced controls.
4. Use the primary row/card chat action.
5. Enter chat header character mode and observe task sequencing.
6. If a model is unavailable, verify whether the app gives an in-context recovery path.

## Evidence

Puppeteer/Chrome artifacts:

- [State capture JSON](assets/2026-05-09-character-chat-reaudit/puppeteer-states.json)
- [01: direct `/characters` first-time entry](assets/2026-05-09-character-chat-reaudit/01-first-time-characters-entry.png)
- [02: after first-run Get Started](assets/2026-05-09-character-chat-reaudit/02-first-time-after-get-started.png)
- [03: explicit character-chat onboarding intent route](assets/2026-05-09-character-chat-reaudit/03-first-time-character-intent-onboarding.png)
- [04: onboarding connection attempt state](assets/2026-05-09-character-chat-reaudit/04-first-time-onboarding-result.png)
- [05: character creation route](assets/2026-05-09-character-chat-reaudit/05-first-time-create-character-entry.png)
- [06: character created](assets/2026-05-09-character-chat-reaudit/06-first-time-character-created.png)
- [07: returning-user search](assets/2026-05-09-character-chat-reaudit/07-returning-user-search-character.png)
- [08: returning-user edit](assets/2026-05-09-character-chat-reaudit/08-returning-user-edit-character.png)
- [09: row chat action result](assets/2026-05-09-character-chat-reaudit/09-returning-user-row-chat-action.png)
- [10: chat empty/model blocker state](assets/2026-05-09-character-chat-reaudit/10-returning-user-chat-empty-state.png)
- [11: header character-mode attempt](assets/2026-05-09-character-chat-reaudit/11-returning-user-header-character-mode.png)

Observed environment:

- Frontend and backend were reachable to the Puppeteer process.
- The backend ran against an isolated temp SQLite profile, not the known malformed default user-1 DB.
- UI character creation succeeded through the drawer with `POST /api/v1/characters` returning `201`.
- No LLM provider was configured, so final assistant response generation was not testable.
- Browser console repeatedly logged notification API CORS failures for `/api/v1/notifications`, `/unread-count`, and `/stream`; those failures did not block the character creation path but are real console noise.

## Comparison Against Baseline

### Resolved Or Improved

- Character creation itself is functional in the isolated backend profile. The UI drawer submitted successfully and the new character appeared in the table.
- Returning-user edit is available and exposes the expected character maintenance surface.
- The chat empty state now says `No LLM provider configured` with `Open Settings` and `Refresh`, which is clearer than the earlier generic "no model" wording.
- Some terminology is improved in the character/edit surfaces: `Characters`, `Character chat`, and `Behavior / instructions` are coherent in the tested path.

### Still Blocking Character-Chat Task Completion

1. **First-run route intent is still preempted before the new onboarding lane can appear.**
   - Direct `/characters` in a clean profile shows `Build Your Assistant`, not character-chat setup.
   - Clicking `Get Started` navigates to `/persona`, where the user sees `Add your credentials to use Persona`.
   - Even explicit `/?intent=character-chat&returnTo=...` showed the same generic first-run splash.
   - Interpretation: the route-aware onboarding package exists, but an earlier first-run/splash layer intercepts character-chat intent before `OptionIndex` can render the character-chat lane.

2. **Row-level `Chat as...` still loses the character-chat task context in the live app.**
   - The row action for `Reaudit Character 20260509173128` navigated to `/`.
   - The resulting page was `Companion Home`, not an in-context "choose a model before chatting as this character" blocker.
   - This is the same high-impact behavioral failure as the baseline from the user's perspective.

3. **Search result count remains misleading.**
   - Searching the unique character displayed only that row, but the status text still said `3 characters found`.
   - The correct status should be closer to `1 of 3 characters shown`.

4. **No-provider state still fragments the character-chat path.**
   - Chat route clearly states no LLM provider is configured.
   - The row action does not keep that blocker attached to the selected character.
   - Header character-mode entry could not be meaningfully evaluated because the generic no-provider chat empty state dominated the route.

5. **Global first-run language still competes with the character-chat mental model.**
   - The first visible path says `Build Your Assistant`.
   - `Get Started` routes to `Persona Garden`.
   - The later fallback route shows `Companion Home`.
   - For a character-chat-centered user, those terms still force avoidable concept reconciliation before the first successful task.

### New Or Newly Visible Issues

- Notification API requests fail CORS preflight when launched from `http://127.0.0.1:8080` because credentials mode expects `Access-Control-Allow-Credentials: true`. This did not visibly block character creation, but it creates repeated console errors and can undermine notification/trust diagnostics.

## Work-Package Matrix

| Work package | Re-audit status | Evidence | UX/HCI interpretation |
| --- | --- | --- | --- |
| DB recovery and root cause | Not retested as a restore; isolated DB profile starts cleanly | Backend temp profile allowed character creation; default malformed DB intentionally untouched | The corruption diagnosis remains valid. A restore/doctor implementation still needs its own verification before using default user-1 data. |
| Character-chat intent preservation | Fails in live row action | `09-returning-user-row-chat-action.png` lands on `/` Companion Home | The task context is still lost at the most important returning-user entry point. |
| Route-aware first-run onboarding | Implemented path is preempted | `01`, `02`, `03`, `04` all show generic first-run/splash or Persona state | The new lane needs integration with the earlier first-run experience, not only `WorkspaceConnectionGate`/`OptionIndex`. |
| Character mode sequencing | Not verifiable in live no-provider state | `10`, `11` stay in generic chat/no-provider state | Need a configured model or deterministic test provider to verify character-first sequencing. |
| Model readiness and in-context blockers | Partial | Chat route has clearer provider blocker; row action does not | The readiness contract is not consistently applied across entry points. |
| Library clarity polish | Partial/unresolved | `07` search shows one visible row with `3 characters found`; row chat remains icon-only | Returning users get control density, but status feedback and primary action salience remain weak. |
| Terminology alignment | Partial | Character/edit surfaces improved; first-run uses Assistant/Persona/Companion | The taxonomy is useful but not yet dominant across the first-run path. |
| Post-implementation re-audit | Complete | Current report and JSON/screenshots | Another implementation pass is required before a clean character-chat UX signoff. |

## Remaining Issues

### P1: First-Run Splash Discards Character Intent

The top priority is to route `/characters` first-time users into the character-chat setup lane before generic assistant/persona onboarding. The user should see `Create character`, `Import character`, `Choose model`, and `Start character chat` without detouring through Persona Garden.

Recommended fix: make the first-run splash consume `resolveOnboardingEntryIntent(location)` or bypass the splash when the current route/return target is `/characters`.

### P1: Row `Chat As` Must Stay Local When No Model Is Ready

The row action should preserve the selected character and show a local blocker with `Open model settings`, `Retry character chat`, and `Return to character`. The live browser evidence still lands on Companion Home.

Recommended fix: trace why the tested live path differs from the unit expectation in `Manager.first-use.test.tsx`; likely candidates are selected-model storage resolution, query timing, or the `useCharacterQuickChat` navigation fallback.

### P2: Search Count Should Reflect Filtered State

The visible table state and result-count text conflict after search.

Recommended fix: expose both filtered and total counts, e.g. `1 of 3 characters shown`, and update the aria live region with the same semantics.

### P2: Notification CORS Noise Should Be Removed

The repeated notification request failures are not character-chat-specific, but they pollute debugging and can create latent UI trust issues.

Recommended fix: align notification requests and backend CORS credentials behavior for the WebUI dev origins, or avoid `credentials: include` for API-key-only notification calls.

### P2: Add A Deterministic Test Provider For Character-Chat E2E

Final message send remains untested because no chat-capable model/provider was configured.

Recommended fix: add a local mock/OpenAI-compatible provider profile for E2E that can validate character selection, first message send, and system prompt inclusion without external dependencies.

## Verification

- Puppeteer/Chrome re-audit command: `node /private/tmp/character-chat-reaudit.mjs`
- Captured unique character: `Reaudit Character 20260509173128`
- UI create response: `POST /api/v1/characters` returned `201`
- Message generation: not tested because the server reported no LLM provider configured.
- Bandit: skipped because this task touched docs and generated browser evidence only.
