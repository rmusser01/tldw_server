# Fresh-install UAT: single-user and multi-user

## Run status

- Status: **In progress — repairing identified issues before the next full workflow UAT**, per the user's follow-up. Initial failures below remain historical evidence; repair verification is tracked separately.
- Requested scope: fresh single-user and multi-user setup, then core workflow loops A, B, and C; record every observed bug, failure, and UX issue.
- Backlog record: `backlog/tasks/task-13260 - Run-fresh-single-user-and-multi-user-UAT-across-core-workflow-loops.md` (see ENV-003).
- Checkout: `codex/email-offline-validation-13250`, commit `54ecc7e7735fc082b711d922a27e97ee9999e5ae`.
- Host: macOS; existing project Python environment and frontend dependencies; about 17 GiB free at preflight.
- Existing backend on port 8000 was left running. Test runtimes use separate ports, configuration, databases, and browser state. Unrelated source files were not edited; a task-record collision and recovery are detailed in ENV-003.
- Installation path: current-checkout fresh configuration/data with existing dependencies. This does **not** certify installation of dependencies into a clean machine/environment.
- Workflow source: frontend E2E/UAT and shared integration tests, as clarified by the user. Exact named journeys and coverage limitations are recorded below; no literal A/B/C loop mapping was found.
- AI provider: existing llama.cpp on port 9099; `/v1/models` verified with model `../../Language_Models/Qwen3.8-27B-UD-Q8_K_XL.gguf`. No mock response counts as real model acceptance.

## Initial-run coverage (before repairs)

| Scenario | Single-user | Multi-user | Evidence / issue |
| --- | --- | --- | --- |
| Fresh configuration and empty data | Pass (reused dependencies) | Pass (reused dependencies) | Isolated SQLite initialization, migrations through 98 |
| Setup entry and onboarding | Fail; advanced using workaround | Fail; recovered via direct login settings | UAT-001 through UAT-004 |
| Authentication and refresh/reload | Pass after manual key entry | Pass: admin password login survives reload | Connection diagnostics still fail; UAT-008 |
| First real chat response | Pass on Retry; first attempt failed | Pending | UAT-003 |
| Normal chat and reload persistence | Pass with local Qwen | Not exercised | Direct Chat navigation works; response persists after reload |
| Text ingestion and persisted search | Pass through visible Quick Ingest and API search | Pass through API for Alice and Bob | Synthetic fixtures; no embeddings requested for multi-user isolation |
| Knowledge QA with cited answer | Fail: provider configuration rejected | Pending | UAT-009; RAG/storage setup had been deferred |
| Workflow loops | Deferred until repairs | Deferred until repairs | Use the executable workflow inventory below; letter labels are not authoritative |
| Admin + two ordinary accounts | N/A | Pass via documented CLI/admin API | Users & roles UI lacks account creation; UAT-011 |
| Cross-account data isolation | N/A | Pass for note reads, media searches, and admin access denial | API-level checks; other resource types not certified |
| Logout and session recovery | N/A: API-key mode | Pass: admin logout, Alice login, identity survives navigation | Alice's Notes UI blocked by UAT-014 |
| Ordinary-user Notes UI | N/A | Fail: connection gate blocks list/editor | UAT-014; note API access passes |

Statuses: **Pass** means directly observed and verified; **Fail** means an observed product defect; **Blocked** means a dependency prevents execution; **Pending** means not yet attempted. Workarounds never erase the original failure.

## Checkpoint summary

- **15 product findings:** five P1, six P2, four P3. P1 denotes a blocked core step, P2 a significant defect or friction, and P3 a lower-impact UX/diagnostic observation. Priorities are provisional triage assessments, not confirmed root-cause diagnoses.
- Passing observations: both empty SQLite databases initialize; single-user real local chat succeeds and persists; text ingestion/search succeeds; admin and two ordinary accounts authenticate; ordinary-user note/media API isolation checks pass.
- Main blockers: onboarding rejects a discovered model ID; first-chat timeout; multi-user setup skip fails; Knowledge QA cannot answer; ordinary-user Notes is blocked by a privileged health probe.
- This is **not release sign-off**. A/B/C remain undefined, dependencies were reused, optional RAG/audio were deferred, and Docker/Postgres were not exercised. Do not interpret API-level isolation as passing the blocked browser Notes workflow.

## Product findings

### UAT-001 — P1: Validated llama.cpp model cannot advance onboarding

- Mode / step: single-user, local install, Chat provider → Continue.
- Reproduce: select llama.cpp; set base URL `http://127.0.0.1:9099/v1`; use the discovered model `../../Language_Models/Qwen3.8-27B-UD-Q8_K_XL.gguf`; Validate → Save providers → Continue.
- Expected: advance with the successfully validated/saved provider and model.
- Actual: validation and provider save return 200; progress save returns 400 `unsupported_first_run_step_data`. UI remains on provider selection with `Setup progress could not be saved. Try again.` In development a Runtime Error overlay intercepts subsequent clicks.
- Failing payload: `{"step":"providers","data":{"acknowledged":true,"default_provider":"llamacpp","default_model":"../../Language_Models/Qwen3.8-27B-UD-Q8_K_XL.gguf","default_provider_credential_configured":false}}`.
- Evidence: browser requests 119/120/123; screenshot `/private/tmp/.playwright-cli/page-2026-09-15T01-08-31-339Z.png`; browser error stack points to `handleProviderContinue` in `UnifiedSetupWizard.tsx`.
- Suspected cause: first-run state rejects path-like strings, while llama.cpp legitimately reports a path-like model ID. Validation accepts the same ID and the UI offers it under Discovered models. The controlled retest below isolates the model-string difference; the exact backend rejection rule has not been fully audited.
- Controlled retest: changed only model entry to `Qwen3.8-27B-UD-Q8_K_XL.gguf`, then Validate → Save → Continue. State save returned 200 and advanced to Ingest defaults. First chat later succeeded with that basename. Original discovered ID remains unusable here.
- Status: open; verified basename workaround. No product fix made.

### UAT-002 — P2: Readiness fails throughout otherwise reachable setup

- Mode / step: single-user, initial page through provider setup; clean browser storage and no prebundled API key.
- Expected: load readiness through the first-run setup connection, or provide a concrete connection/auth recovery action.
- Actual: persistent generic readiness failure advises waiting for server startup/connection details. `/health`, first-run metadata/state/catalog, provider validation, and provider save all succeed. No readiness request appears in the browser request inventory.
- Evidence: screenshot `/private/tmp/.playwright-cli/page-2026-09-15T01-07-01-535Z.png`; browser request inventory through request 123.
- Status: open. Provider work can continue, but readiness remains unavailable. No claim yet about the precise cause.

### UAT-003 — P1: First-chat UI aborts before successful local inference finishes

- Reproduce: complete provider/ingestion/optional setup, then send the default `Say hello in one short sentence.` prompt to local Qwen.
- Expected: wait for model completion within a suitable inference timeout; report a meaningful timeout if exceeded.
- Actual: browser request 159 aborts after approximately five seconds; screen says `signal is aborted without reason`, `Category: unknown`, and `First chat did not complete`. Backend completes the same request with 200 after 10,803 ms, including upstream model 200.
- Evidence: screenshot `/private/tmp/.playwright-cli/page-2026-09-15T01-12-32-864Z.png`; backend request ID `5c153dcd-0518-4f66-8d53-5e21903ba4cb`; backend timestamps 18:11:55–18:12:06 PDT.
- Retest: visible Retry succeeded and marked first chat complete. The initial failed run remains a defect; slower local inference is a normal first-install condition.
- Status: open; retry is a verified workaround for this warm-model run, not a general fix.

### UAT-004 — P1: Multi-user setup traps the user before login; skip fails CSRF

- Reproduce: initialize fresh multi-user SQLite server and admin; open WebUI in clean browser; choose Multi-user; click Skip for now.
- Expected: guide an already provisioned server to user login, or skip cleanly.
- Actual: Multi-user displays only external guides and Back, while header still says Solo onboarding. Skip sends an unauthenticated first-run POST rejected by CSRF middleware with 403. That response lacks CORS headers, so the UI reports `Failed to fetch` and opens a runtime-error overlay.
- Evidence: multi-user browser request 115; backend log `CSRF token validation failed for POST /api/v1/setup/first-run/skip`; request ID `a79c25ae-f903-4047-bbb7-ef88f2911749`; browser snapshot `/private/tmp/.playwright-cli/page-2026-09-15T01-13-00-978Z.yml`.
- Verified recovery: navigate directly to `/login`, which redirects to `/settings/tldw`. Select Multi User (Login), confirm mode change, and select Password. Admin login succeeds and survives page reload.
- Status: open; affects configured multi-user server with first-run enabled. Docker/Postgres profile has not been tested.

### UAT-005 — P2: First-chat success immediately contradicts provider banner

- After successful first chat and API-key recovery, the home screen simultaneously says `First chat is working` and `Configure an LLM provider to start chatting`.
- Banner tells the user to add a hosted API key to `.env` and restart, despite a just-validated, saved, and exercised local provider that needs no key.
- Evidence: snapshot `/private/tmp/.playwright-cli/page-2026-09-15T01-13-54-090Z.yml`.
- Retest: the missing-provider banner is absent after a full reload. Home instead simultaneously reports `Your first source is ready for grounded chat` and `Getting Started: 0 of 2 complete` / `What's next: Add your first source`. The ingested source is verified in the backend.
- Status: open; initial provider banner is transient, while first-source completion remains inconsistent after reload.

### UAT-006 — P2: Solo auth handoff requires unexplained manual API-key recovery

- Setup metadata displays `Bundled auth available`. After first chat, UI instead shows `Restore media access` and asks for a Single-user API key without explaining where to find it.
- Environment: documented local style with no `NEXT_PUBLIC_X_API_KEY`; runtime auth exposure explicitly disabled in frontend to avoid importing the existing installation's credentials.
- Entering the isolated backend key succeeds and reaches Add your first source. This validates the recovery form but not an automatic first-install auth handoff.
- Evidence: first-chat retry snapshot at 01:12:58 UTC; key entry itself is deliberately omitted from public evidence.
- Status: open UX observation; distinguish intentional manual-auth configuration from a broken promised handoff before prescribing a fix.

### UAT-007 — P3: Connection/login copy is mismatched to WebUI and fresh setup

- WebUI connection settings repeatedly call the app `this extension` and offer `Grant Site Access`.
- `/login` lands with Single User (API Key) selected despite the backend being multi-user. Switching an empty fresh browser to Multi User asks to clear nonexistent credentials; confirmation is labeled `Continue Response`.
- Multi-user login initially selects Magic link, although this test server has only a bootstrapped password admin and no configured mail service. Selecting Password reveals the usable login fields. No email was sent.
- Evidence: multi-user settings snapshots at 01:14:32 and 01:14:53 UTC.
- Status: open UX findings; no claim that magic-link delivery was tested.

### UAT-008 — P2: Authenticated multi-user connection check reports invalid API key

- Reproduce: log in with the bootstrapped admin password; click Test Connection in tldw Server settings. Repeat after login has settled.
- Expected: check server health using the current multi-user credentials and report the actual connection state.
- Actual: Logged In is displayed, but the check returns `Invalid API key` / HTTP 401 and `Core: unreachable`; RAG reports healthy. The same session loads protected Server Admin data successfully. Repeated check gives the same result.
- Evidence: settings snapshot at 01:17:29 UTC and protected admin responses; reload at 01:28 UTC still displays Logged In.
- Status: open. Normal authenticated calls work; this finding concerns the diagnostic path.

### UAT-009 — P1: First-source Knowledge QA cannot produce an answer

- Reproduce: paste the Project Juniper fixture via Add your first source → Quick Ingest → Use defaults & process; observe one successful result; click Search in Knowledge; skip the additional assistant setup; ask `When does Project Juniper launch, and who owns it? Cite the source.`
- Expected: return the source's 18 October 2026 date and Mira Chen attribution with a usable citation, or direct the user to the precise missing prerequisite.
- Actual: `Search error. The selected provider configuration is invalid.` No answer or citation. Selecting explicit Llama.cpp with model basename and retrying yields the same error, despite successful first-chat validation/inference with that provider.
- Supporting check: authenticated media search returns the persisted fixture (media ID 1). Ingestion itself succeeded.
- Additional UX: `Ranking retrieved sources` remains alongside the terminal failure. The failed request becomes a previous conversation turn with `No answer recorded`, and the retry says `Using context from turn 1`.
- Environment qualification: optional RAG/storage setup was deferred during onboarding. This demonstrates a broken/unexplained path from the advertised source-ready state; it does not establish that a fully configured embedding/RAG installation also fails.
- Evidence: screenshot at 01:22:23 UTC; two browser QA attempts, including explicit provider selection; `media-search-result.json` in the isolated single-user runtime.
- Status: open; no successful grounded QA loop yet.

### UAT-010 — P3: Additional setup interrupts the source-to-QA handoff

- After completing server onboarding and ingesting the first source, Search in Knowledge opens a full-screen `Build Your Assistant` setup before showing QA. Skip works.
- Home also defaults to numerous `Setup required` personalization cards and a `Temporarily unavailable` reading queue, without a direct enable-personalization action in those cards.
- Expected: make optional assistant/personalization setup clearly distinct from completed server setup, and preserve the user's next source task.
- Evidence: Knowledge navigation at 01:18 UTC; Home snapshots at 01:28–01:29 UTC.
- Status: open UX observation; personalization was not enabled or tested.

### UAT-011 — P2: Multi-user administrator cannot discover an account-creation action

- Reproduce: log in as admin and open Server Admin → Users & roles on a fresh server with registration disabled.
- Observed controls allow refreshing, role management, password reset, and account activation; no Create user action was found in the rendered Users & roles section.
- Impact: provisioning the two ordinary test users required the documented admin API instead of a discoverable WebUI path.
- Scope: UI discoverability finding. The admin create-user API succeeded for both accounts; not a claim that provisioning is impossible.
- Status: open.

### UAT-012 — P3: Connection settings show billing errors for an unavailable module

- On authenticated multi-user Connection settings, Billing & usage renders multiple `Not Found` alerts: billing unavailable, subscription, plans, usage, and invoices.
- Browser requests to the corresponding billing endpoints return 404. The user did not open the Billing tab or enable billing.
- Expected: suppress unavailable-module controls or present one clear capability explanation.
- Evidence: full settings snapshot at 01:28 UTC.
- Status: open; no billing transaction attempted.

### UAT-013 — P2: Home source starter question has no visible effect

- Reproduce: reload single-user Home after ingestion; under `Your first source is ready for grounded chat`, click `Summarize this source.`
- Expected: navigate to a grounded conversation or visibly populate/send the starter question.
- Actual: page remains on Companion Home; no conversation, answer, dialog, or explanatory error appears in the subsequent snapshot.
- Evidence: click at 01:29:01 UTC and unchanged page snapshot at 01:29:16 UTC.
- Status: open; direct Chat navigation works and a new manually entered prompt succeeds, but the source starter question and context are not carried into that conversation.

### UAT-014 — P1: Ordinary-user Notes is blocked by a privileged health probe

- Reproduce: log out as admin, select Password, log in as ordinary user Alice, navigate to Notes, skip the tour, and click Retry connection.
- Expected: show Alice's existing private note and allow normal note operations.
- Actual: `Not connected — Connect to use Notes`, zero notes, and disabled save/export/import controls. Repeated health probes return 403 `Permission denied: missing system.logs`. Retry connection does not recover.
- Control evidence: browser `/api/v1/auth/me` returns 200 with ID 2 / username `uat-alice` / role `user`; root `/health` returns 200; normal protected notifications/persona APIs return 200. Alice's note exists and is accessible through the authenticated note API. The failing endpoint is `/api/v1/health/live`.
- Impact: an ordinary account cannot use the Notes screen despite successful login and data access. Do not grant log privileges merely to work around this UI connection gate.
- Evidence: browser request 167 response, request 150 identity response, snapshot at 01:33:02 UTC and screenshot at 01:33:03 UTC.
- Status: open; reproduced after account switch, navigation, and explicit connection retry. Scope of other gated screens remains untested.

### UAT-015 — P3: Successful first normal chat emits a settings 404

- The first normal saved chat returns a real local-model answer and survives page reload, but the browser logs 404 for `/api/v1/chats/ddf2f7f7-b8af-4e1a-b5df-8f9cb5f4db9b/settings?scope_type=global`.
- No visible interruption or lost answer was observed. Record as diagnostic noise/possible default-settings handling issue, not a failed chat.
- Evidence: single-user browser console error at 01:32 UTC.
- Status: open observation; root cause and whether 404 is an intended absent-settings sentinel are not established.

## Environment and tooling observations

### ENV-001 — Backlog MCP initially unresponsive

- Impact: task tracking through MCP is unavailable during preflight.
- Reproduction: list Backlog MCP resources and search tasks for `fresh install`; neither returned after several minutes.
- Workaround: official Backlog CLI search and task creation succeeded.
- Later outcome: MCP returned after the initial delay; subsequent mutations use MCP with the explicit project path.
- Classification: test tooling; not a tldw product failure.

### ENV-002 — Sandbox blocks host service access

- Impact: Docker socket access is denied; a sandboxed curl cannot reach the host model server on port 9099.
- Evidence: Docker reports `permission denied ... docker.sock`; curl reports connection failure while `lsof` shows a listening llama.cpp process.
- Recovery: authorized host-service access verified the local model and enabled isolated browsers and APIs. Docker deployment remains untested.
- Classification: environment constraint.

### ENV-003 — Backlog CLI allocated an already-used untracked task ID

- Observed: CLI created UAT task `13259`, but preflight already showed the unrelated untracked file `backlog/tasks/task-13259 - Document-second-brain-parity-roadmap-across-server-and-Chatbook.md`.
- Impact: the CLI replaced the pre-existing task record, rather than retaining two files. This was discovered by comparing the final filesystem inventory against preflight.
- Recovery: created UAT task with explicit unused ID 13260 through official MCP. Recovered original task 13259 from its creating Codex task history and restored its title, description, all checked criteria/DoD, plan, notes, documentation link, and final summary through MCP. The roadmap document itself was untouched.
- Limitation: MCP does not expose timestamp restoration. The original Created 2026-09-13 23:33 / Updated 2026-09-13 23:38 timestamps are preserved in a recovery note; frontmatter now reflects collision/recovery. Formatting also follows MCP serialization. No manual task-file edit was used.
- Classification: task tooling incident; content recovered, historical timestamp formatting not restored.

### ENV-004 — Explicit unavailable Redis URL blocks migration locking

- The isolation harness set `REDIS_ENABLED=false` but also supplied an unavailable `REDIS_URL` to avoid the host Redis. AuthNZ migration locking intentionally treats an explicit URL as authoritative and fails closed.
- Initial auth initialization exited 1 with `Redis unavailable for migration lock; refusing local file fallback`.
- Removed the harness-supplied `REDIS_URL`; rerunning initialization exited 0. This is a corrected test setup error, not a product finding.

### ENV-005 — Existing editable environment predates bundled profile package

- First backend start exited 1: `ModuleNotFoundError: No module named 'tldw_profile_core'`, imported by Sync v2 / personal context publication.
- The current `pyproject.toml` includes that local package, but the reused environment does not resolve it.
- Workaround: added `packages/tldw_profile_core/src` and `apps/mcp-unified/src` to the isolated process `PYTHONPATH`.
- Classification: stale development environment. Clean dependency installation remains untested.

### ENV-006 — Existing frontend .env settings conflict with UAT API origin

- First WebUI start exited 1: quickstart mode forbids an absolute `NEXT_PUBLIC_API_URL`.
- Cause: inherited frontend `.env` deployment mode. Explicitly selected `advanced` mode and the isolated API origin, and disabled runtime auth exposure from inherited files.
- Classification: environment contamination corrected for UAT; not proof the documented clean setup fails.

## Execution log

### Preflight

- Read root and frontend agent instructions, setup guides, existing onboarding isolation profile, and UAT verification conventions.
- Confirmed existing dependencies, spare disk space, current commit, and occupied host ports.
- Created the associated Backlog task through the official CLI before creating this tracker.
- Requested A/B/C definitions and preferred installation path; proceeding with independent setup preparation.
- Single-user AuthNZ initialization succeeded on an empty isolated SQLite database after ENV-004 correction.
- Single-user runtime: API `http://127.0.0.1:18000`; WebUI `http://127.0.0.1:18080`; private runtime data/config/logs under `/private/tmp/tldw-onboarding-uat-fresh-single-20260914`.
- Harness reuses existing runtime-profile helpers for path isolation, scrubs host provider credentials, and uses no test-mode flag or mock provider. Local provider endpoint templates point to port 9099; normal private-address/port egress restrictions are retained for onboarding validation.
- Single-user: provider validation/save passed; ingest defaults saved; audio deferred; optional RAG/storage deferred; five default read-only MCP packs saved (23 tools); sample tool check passed.
- Multi-user runtime: API `http://127.0.0.1:18001`; WebUI `http://127.0.0.1:18081`; private files under `/private/tmp/tldw-onboarding-uat-fresh-multi-20260914`. Admin `uat-admin` created through the documented create-admin CLI. Credentials remain only in private runtime files.
- Single-user first-source input: synthetic Project Juniper text with date 18 October 2026, owner Mira Chen, budget 4200 credits, and searchable marker `juniper-uat-20260914`. Processing requested through visible Quick Ingest controls.
- Single-user ingestion result: one succeeded, zero failed; persisted media ID 1 found by authenticated API search. Subsequent Knowledge QA failed twice (default and explicit local provider).
- Multi-user accounts: admin ID 1, Alice ID 2, Bob ID 3. Ordinary accounts created through the documented admin API and authenticated successfully. Generated passwords and tokens remain in the private manifest and are excluded from evidence.
- Note isolation: Alice and Bob each created a private note (201), fetched their own note (200), and received 404 fetching the other user's note. Both receive 403 from `/api/v1/admin/users`.
- Media isolation: each ordinary user uploaded a distinct text fixture (200), searched their own marker (one correct result), and searched the other account's marker (zero results). Each user can have local media ID 1 with a distinct UUID; this is expected per-user storage.
- Single-user auth persists after full reload. Multi-user admin auth persists after full reload; Logout returns the UI to Login Required.
- Multi-user account switch: admin Logout → Password → Alice Login succeeds. A later `/auth/me` confirms Alice; her Notes screen is blocked by a missing `system.logs` health permission, so browser note isolation cannot be marked passed.
- Single-user normal Chat: selected LLaMa.cpp/model is restored automatically and marked Healthy. Prompt `Reply in one sentence: the single-user fresh-install chat check is working.` returns `The single-user fresh-install chat check is working.` Full page reload retains both messages. Direct chat navigation is available, but it did not carry the source starter question or attach its context.

## Evidence retained in the workspace

Selected evidence is copied from temporary browser output into `output/playwright/fresh-install-2026-09-14/`. The [evidence index](../../output/playwright/fresh-install-2026-09-14/evidence-index.json) maps each copied browser artifact to its original timestamped filename. Original temporary paths mentioned above are provenance, not the only retained copy.

| Finding / check | Retained evidence |
| --- | --- |
| UAT-001 provider progress rejection | [Screenshot](../../output/playwright/fresh-install-2026-09-14/single-provider-continue-error.png) |
| UAT-002 / UAT-006 readiness and auth metadata | [Screenshot](../../output/playwright/fresh-install-2026-09-14/single-readiness-auth.png) |
| UAT-003 first-chat abort | [Screenshot](../../output/playwright/fresh-install-2026-09-14/single-first-chat-abort.png) |
| UAT-004 multi-user skip | [Snapshot](../../output/playwright/fresh-install-2026-09-14/multi-setup-skip-error.yml) |
| UAT-005 contradictory provider status | [Snapshot](../../output/playwright/fresh-install-2026-09-14/single-home-provider-contradiction.yml) |
| UAT-007 auth-mode confirmation and magic-link default | [Confirmation](../../output/playwright/fresh-install-2026-09-14/multi-auth-mode-confirmation.yml), [login](../../output/playwright/fresh-install-2026-09-14/multi-login-default.yml) |
| UAT-008 / UAT-012 connection and billing | [Snapshot](../../output/playwright/fresh-install-2026-09-14/multi-logged-in-health-failure.yml) |
| UAT-009 Knowledge QA | [Screenshot](../../output/playwright/fresh-install-2026-09-14/single-knowledge-qa-error.png) |
| UAT-010 / UAT-013 Home and source handoff | [Home](../../output/playwright/fresh-install-2026-09-14/single-home-starter-noop.yml), [assistant setup](../../output/playwright/fresh-install-2026-09-14/single-assistant-interstitial.yml), [Knowledge entry](../../output/playwright/fresh-install-2026-09-14/single-knowledge-entry.yml) |
| UAT-011 admin account creation | [Rendered controls](../../output/playwright/fresh-install-2026-09-14/multi-admin-users-controls.yml) |
| UAT-014 ordinary-user Notes gate | [Screenshot](../../output/playwright/fresh-install-2026-09-14/multi-alice-notes-blocked.png), [retry snapshot](../../output/playwright/fresh-install-2026-09-14/multi-alice-notes-retry.yml), [selected control responses](../../output/playwright/fresh-install-2026-09-14/multi-health-gate-controls.json) |
| UAT-015 nonblocking chat settings 404 | [Console observation](../../output/playwright/fresh-install-2026-09-14/single-chat-console-error.txt) |
| Real chat survives reload | [Screenshot](../../output/playwright/fresh-install-2026-09-14/single-chat-reload-pass.png) |
| API note isolation | [Results](../../output/playwright/fresh-install-2026-09-14/multi-isolation-results.json) |
| API media isolation | [Results](../../output/playwright/fresh-install-2026-09-14/multi-media-isolation-results.json) |
| Single-user persisted source | [Search result](../../output/playwright/fresh-install-2026-09-14/single-media-search-result.json) |

Screenshots were visually reviewed. Text artifacts and this tracker are checked against generated runtime credentials. Private manifests, browser auth storage, raw request headers, and complete runtime logs are excluded.

## Continuation and remaining coverage

1. Complete repairs and targeted verification of the 15 findings before another full UAT, as requested by the user.
2. Build both-mode acceptance matrices from the executable named workflows below. Preserve original failure evidence and distinguish regression verification from a full fresh-install sign-off.
3. Configure optional RAG prerequisites through the intended user path before any fully configured Knowledge QA retest; record configuration friction. Multi-user provider/chat setup is still untested.
4. Complete browser workflows for ordinary users after recording or resolving the health-gate blocker. API isolation does not substitute for browser workflow acceptance.
5. Remaining independent checks include Bob's browser login, token revocation/expiry, note editing/export/recovery, cross-user writes/direct media reads, clean dependency install, Docker/Postgres, audio, and any other steps named by A/B/C. They are not marked passed.

The isolated services are retained for continuation: single-user WebUI `http://127.0.0.1:18080` / API `http://127.0.0.1:18000`; multi-user WebUI `http://127.0.0.1:18081` / API `http://127.0.0.1:18001`. Runtime launcher and configuration are under `/private/tmp/tldw-fresh-uat-launch.mjs` and the two runtime directories listed above. Browser sessions are `fresh-single-20260914` and `fresh-multi-20260914`. After repair verification, the single-user browser displays the cited Cedar answer; the multi-user browser is Alice on the working Notes screen.

Only these UAT frontend build directories belong to this run: `apps/tldw-frontend/.next-live-tier-fresh-single-20260914/` and `apps/tldw-frontend/.next-live-tier-fresh-multi-20260914/`. Keep them while the services run. Cleanup after the final pass should stop only the UAT runtimes, then remove their build directories and synthetic data when no longer needed. Do not stop port 8000 or the user's model server on 9099.

## Closure checklist

- [x] Both setup modes exercised; limitations stated.
- [ ] All agreed A/B/C steps attempted and accounted for.
- [x] Findings backed by reproduction steps and evidence; uncertain causes explicitly labeled.
- [x] Account isolation checked for the stated API scope.
- [x] Test processes and data locations documented.
- [x] Tracker checked for secrets and accuracy at checkpoint.
- [x] Backlog record updated with checkpoint results; overall task remains In Progress.

Bandit was not applicable to the initial documentation-only UAT checkpoint. Touched Python code in the subsequent repair pass is checked below.

## Repair pass — 2026-09-15 UTC (2026-09-14 local)

Branch: `codex/fresh-install-uat-fixes`; baseline `cb8335cf8d`. Tracking: TASK-13260.1 (setup), .2 (accounts), .3 (Home/source continuity), .4 (QA/chat). Design: [fresh-install repair design](../Design/2026-09-15-fresh-install-uat-repairs.md). This pass uses failing regressions, focused suites, independent code review, and narrow live reproductions. A full UAT has **not** been rerun.

| Finding | Repair / current verification |
| --- | --- |
| UAT-001 | Local-provider opaque model IDs survive save, first-chat verification, and public-state reload. Hosted arbitrary paths and credential-shaped values remain rejected. Setup regression suite: 157 passed. |
| UAT-002 | Exact first-run readiness endpoints use unauthenticated setup transport; admin endpoints retain auth. Client regression passes. |
| UAT-003 | First-chat deadline is 180 seconds; abort/timeout has actionable recovery text. Request and component regressions pass. |
| UAT-004 | Multi-user metadata selects sign-in handoff, saves auth mode before navigation, and suppresses solo writes/readiness requests. Focused regressions pass. |
| UAT-005 | Provider loading, absent configuration, and fetch failure are distinct. First-chat and saved-ingest milestones are scoped to the initiating server/account, including normal Chat and Quick Ingest; delayed results cannot credit a replacement account. Focused regressions pass. |
| UAT-006 | Copy distinguishes local eligibility from browser credential exposure and explains manual operator key configuration. Exposure policy remains intentional. |
| UAT-007 | Password login default, credential-aware mode confirmation, neutral copy, and WebUI controls corrected. Focused tests and synthetic Alice login pass. |
| UAT-008 | Saved sessions use `/api/v1/auth/sessions`; edited foreign targets receive no saved credentials. Live Alice and the active, unverified bootstrap admin both report Core reachable / RAG healthy. No verification or operator-health permissions were changed. |
| UAT-009 | Canonical provider/model selection, clarification handling, failed-turn exclusion, progress cleanup, streaming evidence mapping, and source IDs corrected. Public Cedar control produces the correct date/owner and an inspectable source excerpt. Security exclusions are explained using aggregate counts; an entirely filtered result skips generation. See UAT-017 for the citation prompt follow-up. |
| UAT-010 | Empty optional assistant profiles no longer block `/knowledge` or `/chat`; unavailable profiles are not treated as empty. No privileged setup probe or globally cached completion bypass is used. Personalization links its guide. |
| UAT-011 | Admin Create user form calls the existing protected API. Live creation succeeded with default user role after visible reserved-email validation. Strict Mode config loading and unverified-admin diagnostics were corrected during retest. |
| UAT-012 | Billing controls and loaders require advertised OpenAPI capability. Missing-capability regression passes. |
| UAT-013 | Home awaits an owned persisted media handoff before Chat navigation; consumer verifies account/server ownership, media ID, prompt, and clearing. Stale clicks cannot navigate after identity changes. Legacy unowned Review handoffs remain usable under cookie authentication. Live initial handoff and consumer/race regressions pass. |
| UAT-014 | Connection gate uses `/api/v1/auth/sessions` rather than privileged operator health. Alice's live Notes page loads its private note and editor. Operator health permission is unchanged. |
| UAT-015 | Ephemeral chat IDs no longer become persisted server conversation links. Focused stream/history regressions pass. |
| UAT-016 (new) | Notes reads admin title policy only for an active administrator, with account-scoped caching and before/after-await identity checks. Fourteen focused Notes tests pass. Live ordinary-user Notes renders with no title-settings request or403. |
| UAT-017 (new) | Streaming passes the explicit citation option to generation, numbers sources consistently with the visible evidence, and instructs supported inline citations. Disabled citations retain prior full-context behavior. Actual-provider-prompt and invalid-reference regressions pass. Live Cedar answer shows Cited answer, one source / one citation, and a working `[1]` jump to its evidence. |
| UAT-018 (new) | Raw ranking remains unchanged for ordering. Only explicit bounded relevance probabilities drive percentages or low-relevance warnings; unknown relevance is labeled “Relevance not measured” consistently across cards, details, and exports. Raw candidate scores such as 25 and -2.5 remain intact. 159 focused tests pass; live Cedar retains its correct cited answer without a false percentage/confidence warning. |
| UAT-019 (new) | Selecting Alice's private note triggers two unauthorized neighbors requests (`notes.graph.read` missing). Editor remains usable. Capability-aware graph loading repair in progress under reopened TASK-13260.2; preserve backend permissions. |

### Additional observations during repair verification

- **UAT-016 — P3:** Open `/notes` as ordinary user Alice. Expected: default title behavior without an admin request. Actual: `GET /api/v1/admin/notes/title-settings` returns403 and logs an access-denied warning. The list/editor remain usable. Keep backend admin permissions; correct client capability/role handling.
- **UAT-009 security diagnosis:** The Juniper fixture contains the literal word `token`, which the existing content classifier marks confidential; `credits` raises another classification. The standalone RAG access controller defaults an unconfigured principal to guest/public-only. Keyword retrieval worked; later security filtering removed the document. Preserve this policy and its independent ACL model: do not silently promote AuthNZ roles or relax classification. The repair explains exclusions without exposing removed source IDs/text. A separate public Cedar fixture was added through the normal ingestion API (not a UI-ingestion pass) to verify retrieval, generation, excerpt mapping, and source inspection.
- **UAT-017 — P2:** Ask “When does Project Cedar launch, and who leads it? Cite the source.” with Llama.cpp / the catalog model. Actual answer gives 22 November 2026 and Mira Chen, then a prose source title; the UI correctly reports one source and zero mapped citations. Investigation confirms `enable_citations` is lost in streaming generation config and its source labels do not match `[1]` syntax. Repair the prompt contract; never infer citations from arbitrary prose.
- Live admin validation rejected a reserved `.test` email with an actionable server message; a synthetic `example.com` address succeeded. No email was sent.
- **UAT-018 — P2:** The correct cited Cedar answer shows “0% match” and “Low answer confidence.” Its wire score is an uncalibrated hybrid ranking value; the UI multiplies it by 100 and compares it to a probability threshold. Preserve raw ranking for sorting, distinguish calibrated relevance, and show relevance unavailable when no such measure exists. Do not invent confidence.
- **UAT-019 — P3:** Select Alice's existing note in `/notes`. The editor opens, but two `GET /api/v1/notes/<id>/neighbors` requests return403, missing `notes.graph.read`. An ordinary note-open should not eagerly request unavailable graph data or retry a denied capability. Preserve the permission boundary and expose the unavailable state honestly.
- Development hot reload temporarily reset browser UI references/provider selection and emitted an Ant Design unconnected-form warning. Refreshing and selecting the model again recovered. Treat the warning as unconfirmed outside hot reload; do not label it a stable product regression without reproduction.

QA closure checkpoint: combined changed/new frontend regressions **487 passed across 41 suites**; combined setup/provider/RAG Python regressions **248 passed**. Bandit on all six touched Python production modules reports **zero findings**. ESLint across 92 touched frontend files reports **zero errors** and 1,086 warnings. Frontend `tsc --noEmit --incremental false` reports 90 diagnostics, exactly matching a compiler-host baseline overlay of `cb8335cf8d` (**zero additions/removals**). These pre-existing presentation/prompt/E2E errors are not a passing typecheck. The initial baseline overlay exceeded Node's 4 GiB heap; the 8 GiB rerun completed. Notes offline/denied-reconnect follow-up verification is pending separately.

Adjacent-suite limits: Notes content-assist has one missing-fixture-button failure reproduced against its baseline implementation. The generation prompt-loader suite has 11 passes and one concurrent-resolver file-read-barrier failure reproduced with baseline `generation.py`. Neither suite is represented as fully passing. No fixes to unrelated fixtures/loaders are included.

Live security negative control: a normal streaming request restricted to Juniper media ID 1 completed in 414 ms with HTTP 200, empty contexts, `{excluded_count: 1, retained_count: 0}`, then terminal `output_emitted: false`. It emitted no answer or excluded source IDs/text.

Repair screenshots (visually inspected): [Cedar cited answer](../../output/playwright/fresh-install-repairs-2026-09-15/cedar-cited-answer.png), [Alice Notes](../../output/playwright/fresh-install-repairs-2026-09-15/alice-notes-after-repair.png), [Alice connection](../../output/playwright/fresh-install-repairs-2026-09-15/alice-connection-success.png).

### Executable workflow inventory for the next UAT

- Primary guide: `apps/Testing_Guide.md:23`; shared implementation registered by `apps/tldw-frontend/e2e/real-server-workflows.spec.ts:131`.
- Journey project: `apps/tldw-frontend/playwright.config.ts:112`.
- `e2e/workflows/journeys/ingest-search-chat.spec.ts:12`: Ingest → Search → Chat.
- `e2e/workflows/journeys/notes-flashcards.spec.ts:27`: Notes → generate/save flashcards.
- `e2e/workflows/journeys/prompts-chat.spec.ts:16`: Prompts → Chat.
- Additional journeys: Ingest → Evaluate → Review, Watchlist → Ingest → Notify, Create character → Chat.
- Shared real-server workflows in `apps/test-utils/real-server-workflows.ts`: Chat → save note → linked conversation (3512), Chat → save flashcards → review (3970), Media → delete → restore (4266), Media ingestion → analysis → review → re-analysis (4417).

There is no verified source assigning A/B/C to three loops. `Docs/Plans/2026-03-12-e2e-test-coverage-expansion-design.md` uses A/B/C for **coverage tiers**; current config uses numeric tiers 1–5. Onboarding UAT defines only `tierAScenarios`.

Acceptance gaps to account for in the next full run: shared/journey fixtures seed single-user API-key mode and completed onboarding, so they cannot certify fresh multi-user setup unchanged. The ingest journey does not assert citation provenance; the prompt journey does not prove the saved prompt was applied; Notes→Flashcards has skip branches that must not count as passes. Use their named steps with explicit fresh-install, multi-user, provenance, and no-skip acceptance checks.
