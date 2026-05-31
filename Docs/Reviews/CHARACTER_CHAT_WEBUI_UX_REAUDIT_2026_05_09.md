# Character Chat WebUI UX Re-Audit

Date: 2026-05-09
Backlog tasks: TASK-170, TASK-170.1, TASK-170.1.1
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
- Post-P1 refresh captured unique character `Reaudit Character 20260509184619`.
- No LLM provider was configured, so final assistant response generation was not testable.
- Browser console logged repeated Ant Design `useForm` disconnected warnings during character drawer interactions.
- Notification event-stream requests were aborted during navigation; no failed HTTP response status was captured in the final post-P1 refresh JSON.

## Comparison Against Baseline

### Resolved Or Improved

- Direct `/characters` now reaches the Characters surface in a clean profile instead of the generic `Build Your Assistant` splash.
- Explicit `/?intent=character-chat&returnTo=...` now shows `Character Chat Onboarding` before the connection form, preserving the character-chat intent.
- Character creation itself is functional in the isolated backend profile. The UI drawer submitted successfully and the new character appeared in the table.
- Returning-user edit is available and exposes the expected character maintenance surface.
- Row-level `Chat as...` now stays on `/characters` and shows the selected-character blocker: `Choose a chat model before chatting as Reaudit Character 20260509184619`, with `Return to character`, `Retry character chat`, and `Open model settings`.
- The chat route says `No LLM provider configured` with `Open Settings` and `Refresh`, which is clearer than the earlier generic "no model" wording.
- Some terminology is improved in the character/edit surfaces: `Characters`, `Character chat`, and `Behavior / instructions` are coherent in the tested path.

### Still Blocking Character-Chat Task Completion

1. **Search result count remains misleading.**
   - Searching the unique character displayed only that row, but the status text still said `7 characters found`.
   - The correct status should be closer to `1 of 7 characters shown`.

2. **No-provider state still blocks final character-chat task completion.**
   - Chat route clearly states no LLM provider is configured.
   - Row action now keeps the model blocker attached to the selected character, but final message generation remains untested.
   - Header character-mode entry could not be meaningfully evaluated because the no-provider chat empty state dominated the route.

3. **Onboarding success still mixes character-chat and ingestion priorities.**
   - The preserved character-chat lane appears and offers character-specific actions.
   - The same connected setup surface still includes `RECOMMENDED FIRST RUN` guidance for ingesting a source and asking Chat.
   - For a character-chat-centered user, this is no longer a route blocker, but it still adds avoidable cognitive branching during first setup.

### New Or Newly Visible Issues

- Ant Design logs repeated `useForm` disconnected warnings during the character edit/create flow. This did not visibly block the walkthrough, but it is noisy enough to weaken debugging signal during future UX audits.
- Notification stream requests abort during navigation. The final refresh did not capture CORS status failures, but the aborted requests remain observable request noise.

## Work-Package Matrix

| Work package | Re-audit status | Evidence | UX/HCI interpretation |
| --- | --- | --- | --- |
| DB recovery and root cause | Not retested as a restore; isolated DB profile starts cleanly | Backend temp profile allowed character creation; default malformed DB intentionally untouched | The corruption diagnosis remains valid. A restore/doctor implementation still needs its own verification before using default user-1 data. |
| Character-chat intent preservation | Resolved for tested no-model row action | `09-returning-user-row-chat-action.png` stays on `/characters` with selected-character model blocker | The returning-user entry point now preserves task context when no explicit chat model is selected. |
| Route-aware first-run onboarding | Resolved for direct and explicit intent routes | `01` reaches Characters; `03` and `04` show Character Chat Onboarding | The first-time path now preserves character-chat intent, though connected setup still includes ingestion-first guidance. |
| Character mode sequencing | Not verifiable in live no-provider state | `10`, `11` stay in generic chat/no-provider state | Need a configured model or deterministic test provider to verify character-first sequencing. |
| Model readiness and in-context blockers | Improved, still provider-limited | Row action has local blocker; `/chat` has no-provider state | The main local blocker is now consistent, but provider-free E2E cannot verify message send. |
| Library clarity polish | Partial/unresolved | `07` search shows one visible row with `7 characters found`; row chat remains icon-only | Returning users get control density, but status feedback and primary action salience remain weak. |
| Terminology alignment | Partial | Character/edit surfaces improved; first-run still has broader Assistant/Chat ingestion copy | The taxonomy is useful but not yet dominant across the connected setup path. |
| Post-implementation re-audit | Refreshed after P1 fixes | Current report and JSON/screenshots | Core P1 route/context blockers are resolved in the tested no-provider profile; remaining signoff needs model-backed send coverage. |

## Remaining Issues

### P2: Search Count Should Reflect Filtered State

The visible table state and result-count text conflict after search.

Recommended fix: expose both filtered and total counts, e.g. `1 of 7 characters shown`, and update the aria live region with the same semantics.

### P2: Console And Request Noise Should Be Removed

The repeated `useForm` warnings and notification stream aborts are not character-chat-specific, but they pollute debugging and can create latent UI trust issues.

Recommended fix: attach form instances only when their drawer form is mounted, and either avoid opening notification streams during route churn or treat expected navigation aborts as non-error telemetry.

### P2: Add A Deterministic Test Provider For Character-Chat E2E

Final message send remains untested because no chat-capable model/provider was configured.

Recommended fix: add a local mock/OpenAI-compatible provider profile for E2E that can validate character selection, first message send, and system prompt inclusion without external dependencies.

## Verification

- Puppeteer/Chrome re-audit command: `node /private/tmp/character-chat-reaudit.mjs`
- Captured unique character: `Reaudit Character 20260509184619`
- UI create response: `POST /api/v1/characters` returned `201`
- Focused regression: `bunx vitest run src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "row chat intent|stale selected model|implicit row chat selection" --testTimeout=30000`
- Pinned UI typecheck: `../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json --pretty false`
- Artifact validation: `jq empty Docs/Reviews/assets/2026-05-09-character-chat-reaudit/puppeteer-states.json`
- Whitespace check: `git diff --check`
- Message generation: not tested because the server reported no LLM provider configured.
- Bandit: skipped because no Python files were changed; touched runtime code is TypeScript/React plus docs and Puppeteer evidence.
