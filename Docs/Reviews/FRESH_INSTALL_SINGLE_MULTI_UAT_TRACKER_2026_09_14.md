# Fresh-install UAT: single-user and multi-user

## Run status

- Current status: **Cycle3 single-user and multi-user execution has ended with failures and explicit coverage limits. Repair review is underway.** Application behavior was frozen at `d40e17dc81`; a documented UAT-only development-cache configuration change addressed repeated disk exhaustion without changing application behavior. The pass records 31 new findings UAT056–086 plus reopened055 Manage scope: fourteen P2 and eighteen P3. Current coverage is in the [cycle3 matrix](#cycle-3-full-fresh-uat--started-2026-09-15t0743z), with [retained multi-user evidence](../../output/playwright/cycle3-full-uat-2026-09-15/multi/README.md). Earlier runs and repairs below remain historical evidence. This is not release sign-off.
- Requested scope: fresh single-user and multi-user setup, then core workflow loops A, B, and C; record every observed bug, failure, and UX issue.
- Backlog record: `backlog/tasks/task-13260 - Run-fresh-single-user-and-multi-user-UAT-across-core-workflow-loops.md` (see ENV-003).
- Initial checkout: `codex/email-offline-validation-13250`, commit `54ecc7e7735fc082b711d922a27e97ee9999e5ae`. Repairs are on `codex/fresh-install-uat-fixes`.
- Host: macOS; existing project Python environment and frontend dependencies; about 17 GiB free at preflight.
- Existing backend on port 8000 was left running. Test runtimes use separate ports, configuration, databases, and browser state. Unrelated source files were not edited; a task-record collision and recovery are detailed in ENV-003.
- Installation path: current-checkout fresh configuration/data with existing dependencies. This does **not** certify installation of dependencies into a clean machine/environment.
- Workflow source: frontend E2E/UAT and shared integration tests, as clarified by the user. Exact named journeys and coverage limitations are recorded below; no literal A/B/C loop mapping was found.
- AI provider: existing llama.cpp on port 9099; `/v1/models` verified with model `../../Language_Models/Qwen3.8-27B-UD-Q8_K_XL.gguf`. No mock response counts as real model acceptance.

## Cycle 3 repair checkpoints — 2026-09-15 UTC

The frozen run has 32 open findings. Targeted repair verification added UAT087–092, bringing the current total to 38 (14 P2 / 24 P3); all remain open pending their required live verification and the subsequent full fresh run. These implementation checkpoints do not change the frozen UAT outcomes.

- **UAT080 authentication:** backend checkpoint `3751292380` removes the SQLite refresh self-lock while retaining atomic rotation and replay controls. Typed transient failures preserve retryable status; actual invalid sessions retain401 semantics. Independent real SQLite/dependency controls passed40 tests with3 fixture-reported PostgreSQL skips; Bandit introduced no findings. Frontend transport checkpoint `b3a757b017` preserves status/Retry-After, stops stale-token write replay and separates cancellation from timeout (113 focused tests, independent review clear).
- **UAT056 ingestion configuration:** checkpoint `4a21a23540` validates the enabled analysis provider before advancing to Ready; independent run89 passed.
- **UAT057/061 ingestion progress:** checkpoint `dfab9ed3cb` removes simulated stages and size-derived duration estimates. Pending work shows elapsed time, indeterminate progress and confirmed finished-item counts. Actual reported item progress remains supported. Independent run205 passed across15 QuickIngest suites.
- **UAT080 terminal-session handling:** checkpoint `a7f62270af` adds exact credential-pair invalidation markers, preserving newer logins, rotations and hosted cookies. The first natural-expiry check successfully refreshed at 15:25:50 UTC, then exposed UAT087 notification recovery. Revoking that actual rotated session exposed a stale-token winner bug: the raw pre-rotation JWT was misidentified as a newer login, suppressing invalidation. Follow-up `bfcb393a44` fixes that comparison; 113 broader tests and an independent 89-test run pass, with no new lint findings. Live Notes now redirects to sign-in, the second Settings tab becomes Login Required, and its private-fetch counts remain unchanged over 53 seconds. Normal Bob login restores all three saved Notes and active notifications. The Settings tab's stale status after that login is separately tracked as UAT088. Full fresh acceptance remains pending.
- **UAT067/068 Chat selection, ready for live checks:** checkpoint `e4d5642c40` removes disconnected Edit-form writes and competing greeting selection hydration. Canonical selection and delayed loaders respect replacement, clearing and account changes. Parent verification passed130 focused tests plus42 adjacent persistence/loader controls; final changed-hook run33 passed. ESLint0errors with181 unchanged warnings. Saved-Chat creation, mirror and backlink findings remain under repair.
- **UAT065 extraction feedback:** `7db8f5d851` preserves safe structured failure categories through the real queue and Results UI; mixed or unknown failures require review before retry. Backend 52 and frontend 255 regressions pass, with independent re-review and no new lint/Bandit findings. Follow-up `b993995dfe` checks the browser's terminal HTTP response before extraction; missing/non-2xx responses cannot store refusal bodies. Independent browser-guard suite: 26 passed. Targeted real extraction remains pending.
- **UAT077 Provider Keys:** `7e48f29cb1` repairs scalar loading translations, upstream ICU delegation and superseded load state. Parent review passed 26 focused tests; the broader agent checks passed 63 frontend and 26 shared tests. Actual admin login and Provider Keys navigation now render precise BYOK-disabled guidance for the real 403 response without the prior translation crash. The empty route title remains UAT058.
- **UAT062 Saved Chat:** `68906b148b` establishes a neutral saved identity before inference and joins the full pending history promotion. Review additionally reproduced partial-copy, reconnect and placeholder-rendering recovery gaps; every resumed promotion now verifies the fresh raw server prefix before continuing. The visible Retry preserves acknowledged IDs and blocks incomplete inference. Parent306 frontend/9 backend tests pass; independent recovery probes, unchanged lint/type baselines and clean Bandit verified. Actual saved-Chat browser acceptance remains pending.
- **UAT055/074/083/084 Study:** `9d9f5222f5` counts ready cards as due plus new, uses the refreshed remaining queue directly, and replaces both deprecated Manage lists with native markup. Next-review copy describes the hour beginning at the first future card and the uncertainty of a capped scan. Parent and independent70-test runs pass;22 final affected controls cover the cap-copy correction. ESLint matches44 existing warnings with none added; TypeScript matches90 baseline signatures. Live Study/Manage acceptance remains pending.
- **UAT060/066/071/072/073 Media:** `9c21e08f0c` renders analysis through the existing safe Markdown component while preserving raw copy/edit text; aligns missing source-type labels; retains Trash and the existing stale-selection notice in an empty library; uses application-context feedback and exactly-once Undo; and sends percentage zoom100. Independent review passed89 tests, parent48 WebUI controls passed, and the implementation's broader110 shared controls passed. No added lint warnings; TS-only scope, Bandit not applicable. Targeted live acceptance remains pending.
- **UAT058/059/075 route and setup guidance:** `545a7a59d2` adds titles to eight core wrappers using existing Next Head ownership, orders Reading Queue prerequisites before outage copy, and points both server-guide translation keys to the maintained self-hosting index across18 locales. Parent26 title/Home tests and76 docs/app/route tests pass; touched lint has0 errors/0 warnings. The combined checkout matches the known90 TypeScript diagnostic signatures exactly; it is not a clean typecheck. Targeted core route titles now pass. The actual guide button opens the maintained self-hosting index with the expected local/single/multi profile links; deployment of those profiles is not certified by the click.
- **UAT063/069/079 Notes:** `af14ac1258` preserves acknowledged IDs/version baselines through proven same-authority rotation, keeps edits made while saving dirty, scopes optional monitoring to the verified entitled owner, and reserves scrollable results space. Final independent89-test run and parent65 WebUI controls pass; original29 controls also pass with overlap. Parent lint17files0errors66warnings versus70baseline, none added. Targeted live ordinary-user saves201, five notes/five recents,209px resultscroll, actual Tab/Enter selection and exact saved-content reload pass. Mobile390px selection closes its drawer and fits the viewport. No optional monitoring read was sent. The newly visible header wrapping defect is UAT091; full fresh acceptance remains pending.
- **UAT078 Automation Inbox:** `f45f7749c1` distinguishes denied, unsupported, unknown and failed sources while retaining successful results. Protected reads require the captured verified owner and target. Parent independent review and87 WebUI controls pass; broader141shared and ScheduledTasks50 controls pass. No new lint/type diagnostics. Targeted live ordinary-user Home shows tasks.read restriction, capability/notification200, no scheduled-task requests, and Reading Queue Setup required. Other required roles and full fresh acceptance remain pending.
- **UAT076 mixed-deck Study:** backend regression now keeps seven mixed cards in one explicit all-decks session. Review found and reproduced expired-session and tag-membership gaps; corrected controls pass35tests with1PostgreSQL-fixture skip, Bandit0findings and Ruff0findings. A nested transaction correction addresses a source-confirmed PostgreSQL premature commit; PostgreSQL execution remains unverified. Backend final review is clear and committed as `982b03a940`; independent35tests/1fixture skip plus3originalprobes pass. Frontend536ad461e9 and targeted multi-user mixed-session acceptance now pass as detailed below; broader native scenarios and full fresh matrices remain pending.
- **UAT076 frontend session ownership:** `536ad461e9` retains explicit scope and the first acknowledged session ID through ratings and End. Review exposed terminal-expiry handling and a comparison against the wrong verified-owner field; the original probes now pass unchanged. Parent82 WebUI tests pass, lint18files adds0 signatures with285 unchanged warnings, and the full compiler matches90 baseline diagnostics. Practice-only Cram, Undo, delayed scope/account changes and explicit inactive-session recovery are covered. The restarted multi API and real browser pass five Biology cards plus two undecked cards in one new global session2/count7. Automatic End200 and actual reload preserve the completed7-card result. Explicit early End/Undo/practice native checks and full fresh runs remain separate.
- **UAT064 private Flashcard transfers:** `aae6b72d05` replaces all five producers' plaintext URLs with expiring, consume-once transfers bound to the verified account and target. Review corrected same-tab delivery, Quiz ownership and actual sidepanel source capture. Parent63 WebUI controls and12 shared storage controls pass; broader172 tests and independent probes are retained in TASK-13260.13. The WebUI run additionally caught and corrected a leaking test storage spy. Actual Bob Note transfer preserves exact source text/provenance in a clean URL, real generation/save creates two owned cards and independent GET confirms their source UUID. Other producers and cross-account native acceptance remain pending.
- **UAT081 saved source destinations:** `16485e90d4` uses actual Media and conversation routes and opens explicit Note sources through owned detail/task reads. Independent review found a P1 loss of newer typing during the confirmation save and a P2 cancellation of navigation after saving a new draft. Both were reproduced and corrected; unchanged confirmation2/guard2 probes, parent47 Notes tests and39 source/route consumer controls pass. Lint adds0 signatures,144 unchanged warnings; compiler90 baseline. The actual saved Biology card Note link and reload load the exact saved content. Native Media/Chat/negative source checks and full fresh acceptance remain pending.
- **UAT082 private ingest state:** `af1e7bb08b` binds persisted sessions, uploads, worker jobs, polling/cancellation and DocumentPicker handoffs to their captured verified account and target. Independent review reproduced three stale-expiry/marker continuity failures; all original probes pass after canonical revalidation and generation checks. Parent61/5 tests pass, broader480/32 pass, lint0 errors/878 unchanged warnings and compiler90 exact baseline. Native account-switch/resume verification remains pending; worker-only controls do not certify actual popup destruction.
- **UAT086 QA ownership:** `d05c13ecc0` scopes history, active results, source filters and pending requests to the verified account and target. Independent review reproduced draft loss when a stale expiry hint arrives after valid same-owner rotation. The correction revalidates canonical authority; the original probe passes unchanged, independent re-review is clear and the combined746-test run passes. Parent123 WebUI controls pass, compiler90 baseline unchanged, and lint retains7 pre-existing test errors/938 warnings with no new signatures after code-frame line normalization. Targeted live account-history acceptance and full fresh runs remain pending.
- **UAT087 notifications:** `fb79cc565a` reuses effective rotation/invalidation, rejects wrong-target dispatch and preserves terminal permission denial. Review reproduced environment/runtime credential overrides bypassing canonical invalidation and a batched A→B→A scope mismatch; corrective regressions now pass in the parent's independent 131-test run. Initial 117 passing tests did not cover those cases. Targeted live natural expiry at16:22:26UTC refreshed successfully at16:22:28UTC; both tabs remained active and each made a subsequent notification request200. No token/clock mutation was used. Revoking only the current isolated session at16:24:15UTC stopped private polling: the two tabs' request counts remained37/36 from16:25:11 through16:26:22. Normal UI login restored the Settings form, but exposed separate shell recovery issue089. Full fresh acceptance remains pending.

[Retained targeted auth and Provider Keys evidence](../../output/playwright/cycle3-repair-verification-2026-09-15/auth-and-provider/README.md) includes failed and repaired observations, credential-free network metadata and capture hashes. It does not modify the frozen full-run evidence.

[Targeted Notes/Home evidence](../../output/playwright/cycle3-repair-verification-2026-09-15/notes-home-round3/README.md) retains UI saves, layout/keyboard/reload/mobile controls, Home state and three route titles, plus the newly encountered091header defect. Nine files are credential-scanned and hash-indexed; retained PNGs were visually inspected.

[Targeted source and guide evidence](../../output/playwright/cycle3-repair-verification-2026-09-15/source-guide-round4/README.md) retains the remaining core route titles, actual guide-button destination and actual saved-card Note click/reload. Seven captures are credential-scanned and hash-indexed. These focused results do not change the frozen full-run outcomes. Interleaved Quick Ingest implementation briefly produced a duplicate binding, a request-signature compiler error and a stale Next build overlay; those development observations were recorded in TASK-13260.26 and excluded from acceptance passes.

[Second targeted auth recovery evidence](../../output/playwright/cycle3-repair-verification-2026-09-15/auth-recovery-round2/README.md) retains natural refresh, terminal revocation, stale and repaired cross-tab login, and confirmation-dialog checks. Fourteen safe JSON records and six supporting files are scanned for credentials and hash-indexed. Settings and shell fixes are committed as `a3542ea385` and `af725e4330`.

[Targeted Flashcard transfer evidence](../../output/playwright/cycle3-repair-verification-2026-09-15/flashcard-handoff-round5/README.md) retains actual Note transfer, real generation/save with independent owned read, and UAT092 selector RED/live correction. Fourteen files are hash-indexed and credential-scanned. This is not the required mixed Study run or a full fresh UAT.

[Targeted mixed Study evidence](../../output/playwright/cycle3-repair-verification-2026-09-15/mixed-study-round6/README.md) retains five decked/two undecked preparation,7 real ratings, one completed global session, automatic End200 and the reloaded7-card summary. Twenty files are hash-indexed and credential-scanned; no explicit early-End or full fresh run claim.

## Cycle 3 targeted repair findings

#### UAT-092 — P3: Generator deck Clear immediately restores the same selection

- Mode / step: existing isolated multi-user Bob, Flashcards → Create cards, existing Biology deck. At19:19:28UTC the exact `flashcards-generate-deck` Clear control was clicked.
- Actual: the selected deck is Cycle3 Bob Biology before and after clearing, and the Clear control remains visible. Source confirms the initial-default effect reselects the first deck whenever selection is null; saving also requires an existing or new deck. The control promises a state this form does not retain.
- Expected: a required deck selector offers the valid existing/new deck choices without a misleading Clear action.
- Status: targeted live pass, `f9b0dd2f53`, TASK-13260.31. Exact-control RED confirmed the issue; the one-line correction removes the misleading Clear action. Existing24/3 regressions and independent review pass. After rebuilding the isolated frontend, native Clear count0, new-deck name fields and restored Biology scheduler summary all pass. Full fresh UAT remains pending. Evidence is retained in flashcard-handoff-round5.

#### UAT-087 — P3: Notifications still require sign-in after a successful session refresh

- Mode / step: repaired multi-user runtime, Bob normal UI login14:55:46UTC, natural access expiry15:25:46UTC, Notes and server Settings tabs. No token/clock mutation or mock response.
- Actual: Settings requests397/398 receive401, refresh399 returns200, retry401 and session408/414 return200. Test Connection reports Core reachable/RAG healthy. Both tabs retain Notifications require sign in after unread-count401 (Settings411; Notes981), and notification polling stops while private session reads continue200. Existing saved Bob Notes remain present.
- Expected: same-account refresh resumes notifications with the effective rotated credentials; genuine revoked-session401 and permission403 keep their terminal behavior.
- Status: targeted live pass, open pending the full fresh run, TASK-13260.28; related prior lifecycle implementation TASK-12098.4. Original failure evidence: /private/tmp/uat080-settings-requests-after-expiry.txt, uat080-settings-after-expiry.txt, uat080-notes-after-expiry.txt and uat080-notes-requests-after-expiry.txt. The second retained recovery bundle shows natural rotation and subsequent notification200 in both tabs, followed by clean terminal revocation. This does not change the frozen full UAT outcome.

#### UAT-088 — P3: An open Settings tab retains Login Required after another tab signs in

- Mode / step: multi-user Bob, terminal revoked-session handling followed by ordinary UI login at 15:52:26 UTC in the Notes tab.
- Actual: Notes loads all three saved Bob rows and notifications become active. The existing Settings tab still displays Login Required and a login form at 15:54–15:56 UTC.
- Expected: the open Settings tab reflects current authentication for its displayed connection without reloading or overwriting unsaved settings.
- Status: open, TASK-13260.29. Evidence: /private/tmp/uat080-notes-after-normal-relogin.txt, uat080-settings-after-normal-relogin.txt and uat080-settings-relogin-settled.txt. The initial task note claiming second-tab recovery was premature and has been explicitly corrected.
- Repair verification: at16:26:59UTC, the same mounted Settings form changed from Login Required to Logged In after normal Bob login in the other tab, without a Settings reload. Independent real-Form review additionally found that cancelling an auth-mode change restores the field but leaves stale login status; the correction passes real Form tests and live Cancel at16:32:36. Committed `a3542ea385`; parent82 tests/6 suites pass. The application header's separate stale auth state is089 below.

#### UAT-089 — P3: Application navigation stays hidden after cross-tab login

- Mode / step: multi-user, same two-tab terminal-session revocation and normal Bob login used for088. The existing Settings page was not reloaded.
- Actual: the Settings form says Logged In, while the application header, Companion Home shortcut and notification controls remain absent through16:27:31UTC, including tab focus. The Notes tab successfully returns to its normal authenticated workspace. This is not evidence of failed notification transport; the Settings header is not rendered.
- Expected: authenticated application navigation returns when the current canonical session becomes valid; unsaved Settings fields are preserved.
- Status: open, TASK-13260.30. Credential-free evidence: /private/tmp/uat088-second-login-two-tab-metadata.txt and /private/tmp/uat088-second-login-settings.yaml. Investigate the app auth owner's cached configuration read separately from the repaired Settings-local subscription.
- Targeted repair pass: `af725e4330` resolves effective canonical storage and preserves the app's validation/generation guards. A second own-session revocation at16:44:50 and normal UI login at16:46:53 restore the existing Settings header, Logged In and active notifications without reloading it. At16:48:44, each tab has two fresh notification200 responses. Parent61 tests and broader179 tests pass; full fresh acceptance remains pending.

#### UAT-090 — P3: Authentication-mode confirmation uses a static modal outside the application context

- Mode / step: existing multi-user Settings tab, select Single User (API Key) at16:32:06UTC, then Cancel. The session and original mode are preserved.
- Actual: opening the dialog emits `Warning: [antd: Modal] Static function can not consume context like dynamic theme. Please use 'App' component instead.`
- Expected: the confirmation uses the active application context and theme without a console warning.
- Status: open, handled with the adjacent confirmation fix in TASK-13260.29. Evidence: /private/tmp/uat088-mode-confirm-dialog.yaml, /private/tmp/uat088-cancel-live-metadata.txt, console event4435142ms in .playwright-cli/console-2026-09-15T15-18-10-922Z.log. Root is adding a real App/Form/dialog regression and using the existing context-backed modal hook.
- Targeted repair: the actual context-backed dialog opened and cancelled at16:42:24–16:43:02UTC without the static warning, preserving Logged In and active notifications. Parent82 tests/6 suites pass, with zero lint errors and33 unchanged warnings. A separate non-failing browser advisory recommended autocomplete hints for the Settings login inputs; the adjacent form now supplies standard username/current-password attributes. No login failure was observed from that advisory.

#### UAT-091 — P3: Notes toolbar compresses the heading and wraps saved time vertically

- Mode / step: targeted multi-user Notes at1280×720, five saved notes and expanded sidebar. Open the fourth note after saving and wait for the settled Saved3mago status.
- Actual: the single desktop toolbar row squeezes the title to approximately four letters and stacks Saved/3m/ago into three lines. The content is saved and the209px results scroll remains usable; this is a distinct header presentation issue.
- Expected: preserve a readable heading and keep the status together, moving toolbar groups onto another row when space is insufficient.
- Status: open pending full fresh UAT; targeted repair `062e8b7cb2` passes under TASK-13260.18. Header/action groups wrap and the status stays together. Actual1280/1024/390px views preserve the full heading and single-line saved time, with document width equal to viewport. Parent14tests and independent7header tests pass; lint0errors2unchangedwarnings. Independent code/screenshot review clear. [Before/after evidence](../../output/playwright/cycle3-repair-verification-2026-09-15/notes-header091/README.md).

## Cycle 2 repair status — 2026-09-15 UTC

The user requested an ongoing UAT → review → fix loop until a complete fresh single/multi run encounters no issues. Repairs for UAT-020–045 are in progress under TASK-13260.5–.11 and `IMPLEMENTATION_PLAN_uat_cycle_2.md`. The findings below remain open pending integrated review and real-runtime verification; targeted tests do not replace the next full run.

Targeted follow-up found ten additional issues046–055 (twoP1, sixP2, twoP3). All now have targeted live verification, including the remaining cross-tab logout repair in `d40e17dc81`. Recent committed fixes include052 `26f35ae81d`,055 `2651047438` and private activity handling `7dc8db2efb`. Parent verification after the follow-ups:166shared,155web and59app/login controls pass (overlapping suites are not summed). Final TypeScript check retains the same90 baseline diagnostic signatures with none added/removed; it does not pass outright. [Retained targeted evidence](../../output/playwright/cycle2-repair-verification-2026-09-15/README.md) is credential-scanned and hash-indexed. Full fresh acceptance remains pending.

- **Notes/account state:** Recent history, pinned IDs and offline drafts use verified server/account storage scopes. Notes/login titles and manual-key Disconnect are repaired. Review found offline transport and delayed A→B→A detail races; guarded transport, expected-user API checks, cancellation and stale-result handling now cover them. Final checks: 105 Notes/policy/worker tests, 52 web transport tests and 58 backend tests passed. Live account-switch verification remains pending.
- **Chat:** Repairs address neutral sampling rejection before provider dispatch, first-turn Saved persistence, retained server message/conversation linkage on failure, and explicit character context replacement. Agent verification: 69 backend and 152 frontend tests; independent parent repeats: 69 backend and 140 frontend tests. Production diff reviewed without remaining findings.
- **Flashcards:** Repairs verify contextual question/answer claims, retain semantic checking for quoted questions, constrain generated facts to source, show concise verification failures, save visible Chat content, require a reviewed question/answer pair, and show truthful initial Study state. Independent review found and repaired a late-save overwrite of an edited card draft; 43 UI and 16 backend review tests passed. Shared E2E helper is being aligned with the required card review dialog.
- **Media/retrieval:** Repairs cover ingestion error classification, current model configuration, non-success URL response rejection, explicit skipped chunking, source/permalink/Chat navigation, permissions-aware delete, deleted-selection clearing and Trash dates. Verification: 95 backend and **144 unique frontend** tests; independent production review found no remaining issue. UAT-030 worker ownership is repaired for new ingestion and narrowly backfilled for existing null-owner worker rows. Tests preserve explicit owners and malformed labels; a private clone of the UAT database returned Alice's source to Alice and excluded Bob. Original runtime data was not changed by that clone check.
- **Setup/QA defaults:** Readiness warnings explain required verification; local model discovery precedes selection. Blank template QA defaults inherit configured Chat; explicit overrides retain provider/model pairing. Review caught provenance loss when the real config loader merged an environment provider with another provider's model. Real-loader regression: two failed before repair; broader 280 configuration/generation tests passed after repair, alongside 7 readiness and 23 provider-step tests.
- **Prompt editor:** Successful save establishes saved identity and a clean baseline; Back returns to the library. Review also repaired repeated keyboard saves and late success/rejection affecting another draft. All 87 Prompt tests passed; final independent review has no remaining findings.

Combined parent verification: 60 backend boundary regressions passed; Bandit scanned all 24 touched production Python files with zero findings. Frontend typecheck reports the same 90 baseline diagnostics, with no added/removed diagnostic signatures; it is not a passing typecheck. ESLint across 68 touched frontend files reports five pre-existing `no-require-imports` errors in three Media test fixtures (all reproduced against HEAD), plus 1,554 warnings. Domain lint comparisons found no new production findings. Logs: `/private/tmp/uat-round2-parent-integration.log`, `/private/tmp/bandit_uat_round2_combined.json`, `/private/tmp/uat-round2-typecheck-final.log`, `/private/tmp/uat-round2-eslint-final.json`. Live checks and the complete fresh UAT rerun still remain.

### Cycle 2 targeted live check follow-up

- Repaired product revision: `a53aa33e58`; isolated APIs restarted on 18100/18101 and both `/health` returned 200. Single-user `/media/capabilities` returned 200 with `can_delete: true`.
- Prompt create/save adopted a real saved identity (`pa_8d8f-a875-78f-2a1d`), changed URL from `?new=1` to `?edit=…`, displayed the saved title and success notification. Back verification was interrupted when Playwright CLI lost its browser sessions; this is a tooling interruption, not a passing navigation check.

#### UAT-046 — P3: Prompt save emits a notification context warning

- Mode / step: single-user; create and save a Prompt on `a53aa33e58`.
- Expected: successful save displays the notification through the active application theme/context without a console error.
- Actual: visible save succeeds, but the console records `Warning: [antd: notification] Static function can not consume context like dynamic theme. Please use 'App' component instead.`
- Evidence: `/private/tmp/.playwright-cli/console-2026-09-15T05-58-03-961Z.log`, event at 47,806 ms; saved-editor snapshot `/private/tmp/.playwright-cli/page-2026-09-15T05-58-52-754Z.yml`.
- Status: **targeted live pass** on `5576c93c23`: context-backed notification, 95 regression tests, and real browser save/back without the warning. Saved record `pa_d25c-b410-ae7-2a53`; Back returned to `/prompts` with one synced row and no false unsaved prompt. Full fresh UAT remains pending.

#### UAT-047 — P2: Private polling continues after Disconnect

- Mode / step: single-user, manual-key Settings → Disconnect on repaired build.
- Expected: clear credentials and stop authenticated background polling until a verified reconnection.
- Actual: the key clears and re-entry succeeds, but notifications unread-count continues every 30 seconds and triggers CORS preflight errors using credentials mode `include`; missing-key warnings repeat twice every five seconds while disconnected.
- Evidence: `/private/tmp/.playwright-cli/console-2026-09-15T06-07-28-118Z.log`, errors at 30,392 ms and 60,388 ms; reconnect success snapshot `/private/tmp/.playwright-cli/page-2026-09-15T06-08-48-027Z.yml`.
- Status: **targeted live pass** under TASK-13260.5. Notifications/Buddy require verified connection authority; stale checks cannot restore a disconnected/replaced account. 140 shared and 145 web regressions pass, independent review clear. Disconnect cleared the key; a 4m13s request comparison (2026-09-14 23:57:23 to 2026-09-15 00:01:36 Pacific) showed no new private polling, only public docs-info200. Normal UI key re-entry restored Core reachable/RAG healthy. Evidence: `/private/tmp/uat047-requests-start.txt`, `/private/tmp/uat047-requests-end.txt`, `/private/tmp/uat047-reconnected-final.txt` (redacted). Full fresh UAT remains pending.

#### UAT-048 — P2: Offline logout produces a runtime error

- Mode / step: multi-user Alice, queued offline Note draft, then Settings → Logout while offline.
- Expected: clear the local session, preserve Alice's draft under Alice's storage scope, and recover sign-in cleanly when online again.
- Actual: Settings shows a `Failed to fetch (POST /api/v1/auth/logout)` runtime error, with a separate uncached PageHelpModal chunk failure. The other Notes tab navigates to a browser error page. Credential/draft isolation checks continue after reconnect; these errors do not count as passing logout UX.
- Evidence: `/private/tmp/uat-round2-offline-logout.txt` and the multi-user agent's saved snapshot.
- Status: **targeted live pass**, repairs `7dc8db2efb` and `d40e17dc81`, TASK-13260.5. The intermediate retest fixed the Settings overlay but retained the other-tab browser error; that failed evidence remains in `uat048-final-multi-report.md`. The final repair replaces protected content with an already loaded signed-out screen and defers only automatic offline navigation; stale auth completions cannot restore identity or perform another logout. Final two-tab retest: Notes stays on `/notes`, title Signed out | tldw, private draft/list/editor unmounted, no browser error, optional Help failure, or runtime overlay. Reconnect opens `/login`. Bob sees only his own note; Alice's queued draft syncs with observed POST201 (`f6d2d7cd-a29a-4014-be48-b338d71633e9`,version1), and independent Bob login200 → foreign-note GET404 → verifier logout200 confirms ownership. Offline remote revocation is not claimed. Evidence: `/private/tmp/uat048-boundary-final-report.md`, `/private/tmp/uat048-boundary-final-title.txt`, `/private/tmp/uat048-boundary-bob-foreign-note.json`.

#### UAT-049 — P2: Closed Help modal can break an offline app remount

- Observed during the UAT-048 offline logout/recovery path; the user never opened Help.
- Root-cause trace: app-shell hosts mount the lazy PageHelpModal even while closed. An offline remount requests its uncached chunk, then the load rejection reaches the route error boundary and shows `Something went wrong`.
- Expected: a closed optional Help modal should not block the working app shell or offline draft/session recovery.
- Evidence: `/private/tmp/uat-round2-offline-logout.txt`; source trace in `EventHosts.tsx`, `Layout.tsx` and shared entry hosts.
- Status: **targeted loader pass** under TASK-13260.5. Closed hosts no longer import the body. Requested loading failure stays local; actual Turbopack rejection caching requires explicit Reload page, with an unsaved-edits confirmation. Unit red/green, 22 Help/shell regressions and independent review pass. Live `/setup` global Help-open event while offline shows the local notice and keeps the page usable; cancelling reload preserves the synthetic field. Reconnect → confirmed Reload page → reopen shows real Page Help (Tutorials/Shortcuts). Evidence: `/private/tmp/uat049-setup-help-result.txt`, `/private/tmp/uat049-cancel-keeps-draft.txt`, `/private/tmp/uat049-recovered-modal.txt`. This is a targeted loader check; multi-user offline logout and full fresh workflows remain separate.

#### UAT-050 — P1: Normal saved Chat corrupts an advertised path-like model ID

- Mode / step: single-user replacement browser, Media → Chat, first saved normal Chat turn using the server-advertised local model.
- Expected: send the selected model identifier unchanged and receive the local model's answer.
- Actual: browser POST `/api/v1/chat/completions` includes `model: ../../Language_Models/Qwen3.8-27B-UD-Q8_K_XL.gguf`, `api_provider: llama`, `save_to_db: true`. The API rejects it with 400 `model_not_available`, naming **`../Language_Models/Qwen3.8-27B-UD-Q8_K_XL.gguf`** (one parent segment lost). UI says `Stream completion failed`; the sidebar still labels the model healthy. The user message and error are persisted, but no answer is generated.
- Evidence: `cycle2-single-recovery` request 299, exact request/response bodies; snapshot `/private/tmp/.playwright-cli/page-2026-09-15T06-15-54-912Z.yml`; console `/private/tmp/.playwright-cli/console-2026-09-15T06-12-34-327Z.log` at 149,886 ms.
- Status: **targeted model-routing pass**, committed `1e628951be`. Provider-prefix parsing preserves opaque local paths/repository names and registered aliases; 87 backend regressions pass, Ruff/Bandit have zero findings, and independent review is clear. Exact-model Retry request 1381 returned 200 and the correct answer, "The garden opens on 18 December 2026 and is coordinated by Mira Chen." Reload retained that answer in browser state, but a subsequent server read disproved server persistence (UAT-052). Evidence: `/private/tmp/uat050-request-body.txt`, `/private/tmp/uat050-browser-after-retry.txt`, `/private/tmp/uat050-reloaded-answer.txt`. Separate stale error UX is UAT-051.

#### UAT-051 — P2: Successful Chat retry leaves an active error banner after reload

- Mode / step: single-user; retry UAT-050 successfully, then reload the saved conversation.
- Expected: the current composer reflects the successful latest assistant reply; an earlier failed attempt may remain historical.
- Actual: the persisted correct answer appears as assistant message 3, but the composer displays an active Error banner saying "Something went wrong while talking to your tldw server" with Retry chat actions. This describes the earlier failed message as a current failure after successful recovery.
- Evidence: `/private/tmp/uat050-reloaded-answer.txt`, saved conversation `5536eadd-4be4-4eea-bce2-bc6f454e8990`; completion request 1381 returned 200. No new completion failure was observed.
- Status: **targeted live pass**, committed `384d9010ad`: latest conversational attempt determines the active composer banner, historical failed bubbles remain intact. All 55 related regressions pass; lint has no errors/new warnings. Parent review and live reload confirm the old composer banner is absent (`/private/tmp/uat051-reloaded.txt`).

#### UAT-052 — P1: Saved normal Chat answer forks into another conversation and lacks export actions

- Mode / step: single-user normal saved Chat; successful UAT-050 Retry, reload, then open the assistant reply's More actions.
- Expected: `save_to_db: true` persists the reply to its owned conversation and provides Note/Flashcard save actions after server message identity is established.
- Actual: browser shows three messages including the successful answer and labels Chat Saved; server GET `/chats/5536eadd-4be4-4eea-bce2-bc6f454e8990/messages` returns only two (original user message and error). The reply was instead persisted in unexpected conversation `e4a86441-a45a-4d72-b0de-58bb39dac45f`, confirmed by independent GET200. More actions lacks Save to Notes/Flashcards because local state retains the original conversation and has no canonical reply identity.
- Evidence: `/private/tmp/uat-chat-saved-messages.txt` (request1733), `/private/tmp/uat050-request-body.txt`, menu snapshot `/private/tmp/.playwright-cli/page-2026-09-15T06-34-36-470Z.yml`. Browser reload alone was insufficient persistence evidence; earlier wording is corrected above.
- Root cause: backend resolves default character3 for the existing neutral conversation, then forks on character mismatch (`expected char:3, got char:None`, same owner1). The successful reply exists under the fork; local metadata still points to the original conversation. No data-loss claim is made. Additional evidence: `/private/tmp/uat052-forked-conversation.json`.
- Status: **targeted live pass**, committed `26f35ae81d` under TASK-13260.12. Exact owned neutral conversation reuse, diagnostic filtering and narrowly matched failed-turn retries preserve owner/character boundaries. Server reply IDs reach the frontend save actions. 90 backend/76 frontend regressions pass; independent reviewer repeated 58 backend/52 combined frontend tests. Bandit0; lint no new findings. Live retry request199 returned200; independent GET of the original conversation contains its original system/user/error plus assistant `6aedba81-4546-41a6-98c4-c080f26a05ba`, with no duplicate user. Save to Notes263 and Flashcards395 both return201 with the original conversation/reply IDs. Note `d299e8d8-db3b-4c32-b7ea-f7a1188f54d5` contains the visible answer and reports Origin: Saved from Chat; its Open conversation action plus reload restores the original question/error/correct answer. Card `52f987ff-2d7a-4ddb-a202-ca977f8711d8` has the explicitly reviewed question and correct nonblank answer, independently read from the server. Evidence: `/private/tmp/uat052-original-server-after-retry.json`, `/private/tmp/uat052-note-server.json`, `/private/tmp/uat052-card-server.json`, `/private/tmp/uat052-note-open-final.txt`, `/private/tmp/uat052-linked-chat-reloaded.txt`; report `/private/tmp/uat052-repair-report-20260915T065632Z.md`. Full fresh UAT remains pending.

#### UAT-053 — P2: Background Buddy polling opens a blocking offline error dialog

- Mode / step: single-user Chat, set the browser offline while checking optional Help loading; no Buddy action requested.
- Expected: passive polling failures remain nonblocking with honest connection status; Help/draft interactions remain available. Credential removal separately requires polling to stop (UAT-047).
- Actual: an unsolicited modal "Can't reach your tldw server" blocks toolbar interaction and names `GET /api/v1/buddies/attachment?client_slot=default`. The page already reports offline and offers settings. Dismissing this unrelated background error is required before continuing the intended action.
- Evidence: `/private/tmp/uat049-overlay-current.txt`; no user Buddy request preceded this modal.
- Status: **targeted live pass** under TASK-13260.5. Passive Buddy reads suppress only the global unavailable event; failed reads still reject and explicit GET/DELETE controls still notify. 49 tests pass across the real service/client/proxy chain in web and extension modes; independent review clear, lint no new findings. After required Chat loading completed, a33second offline observation showed no blocking dialog; notifications reported reconnecting. Evidence: `/private/tmp/uat053-buddy-final.log`, `/private/tmp/uat054-settled-offline-start.txt`, `/private/tmp/uat053-settled-offline-final.txt`. Taking the browser offline during its initial settings load still reports that active request's failure; that is not a background-polling pass or suppression.

Tooling/environment follow-up: a sandboxed Playwright CLI list operation removed nonpersistent session records after failing to connect to daemon sockets. Replacement persistent browser profiles were opened and authenticated through the UI; browser-state continuity from the lost sessions is not claimed. Restarting both isolated WebUIs resolved the development build graph's missing newly added Notes module; Notes returned HTTP 200 and the correct title. Old 18080/18081 UAT frontends were retired and only their untracked `.next-live-tier-fresh-*` caches removed for disk space; databases, configuration and evidence were preserved.

#### UAT-054 — P2: Implicit Chat feedback opens a blocking offline error dialog

- Mode / step: single-user saved Chat, reload then set browser offline; no feedback action requested.
- Expected: automatic feedback fails quietly and the shell stays usable with honest offline status.
- Actual: after36seconds a blocking "Can't reach your tldw server" dialog names `POST /api/v1/rag/feedback/implicit`. This is a separate automatic request from the Buddy reads repaired for UAT-053.
- Evidence: `/private/tmp/uat053-offline-start.txt`, `/private/tmp/uat053-offline-end.txt`, observed2026-09-15T07:11:58Z.
- Status: **targeted live pass** under TASK-13260.5. Implicit feedback suppresses only the global unavailable event; explicit feedback retains failures. 13 regressions pass, independent review clear, lint no new findings. Copying the real reply while offline triggered implicit POST458 with ERR_INTERNET_DISCONNECTED and no blocking modal; reconnect succeeds. Evidence: `/private/tmp/uat054-copy-offline.txt`, `/private/tmp/uat054-copy-requests.txt`. Expected browser network diagnostics from deliberate offline testing remain visible.

Environment interruption (2026-09-15T07:06Z): simultaneous single/multi Next caches exhausted disk space. Evidence writes/npm and two research-run polls failed (HTTP500); backend log buffer reported OSError. Multi frontend80042/80047 was retired and only its generated cache removed. Disk space recovered; independent research-runs GET returned200 with runs:[] and continued UI polling recovered. No test database was removed. Evidence: `/private/tmp/uat-enospc-research-runs-recovered.json`. Remaining WebUI runs are sequential.

Validation follow-up: the broader UAT048 login regression run passed58tests and exposed one stale navigation-test fixture. It directly imports the unchanged Login page, expects a superseded heading synchronously, and lacks deterministic configuration for asynchronous login-target resolution. The repaired test awaits the existing unconfigured-server redirect panel and retains target/query and hosted-login controls; final59/59parent app/login tests pass. This is a test maintenance finding, not a newly observed product failure. Evidence: `/private/tmp/uat048-parent-login-regression.log`, `/private/tmp/uat048-parent-login-final.log`; TASK-13260.5.

#### UAT-055 — P3: Flashcards Study emits a deprecated List console error

- Mode / step: single-user Flashcards → Study, existing recent study session and due queue.
- Actual: console reports `Warning: [antd: List] The List component is deprecated. And will be removed in next major version.` Study remains usable.
- Expected: current supported rendering without a console error from visiting Study.
- Evidence: `/private/tmp/uat052-study-card.txt`, browser console on2026-09-15T07:13Z.
- Status: **targeted live pass** under TASK-13260.6. RecentStudySessions uses a labeled native list with the same controls/content and state branches. 26 regressions pass; lint0; independent review clear. Fresh Flashcards navigation renders Recent study sessions and View completed session with0console errors/warnings. Evidence: `/private/tmp/uat055-study-fixed.txt`, `/private/tmp/uat055-recent-sessions-final-20260915T071600Z.log`.

## Cycle 3 full fresh UAT — started 2026-09-15T07:43Z

- Frozen application behavior: `d40e17dc81`. New findings during this run will be recorded before another repair pass. The separately documented development-cache configuration exception preserves application logic and default non-UAT behavior.
- Empty profiles: `/private/tmp/tldw-onboarding-uat-cycle3-single-20260915` and `/private/tmp/tldw-onboarding-uat-cycle3-multi-20260915`; audits confirmed no users database before initialization, no test/mock/provider overrides, and blank RAG defaults inheriting Chat.
- Both AuthNZ initializations succeeded through migration98. Only the multi-user bootstrap admin was provisioned through the documented CLI. Alice/Bob creation remains a UI acceptance step.
- New APIs18200/18201 and WebUIs18280/18281; WebUIs run sequentially for disk space. New persistent browser profiles will contain no previous configuration or account state.
- Existing dependencies are reused; this certifies fresh configuration/data workflows, not clean-machine dependency installation. Real llama.cpp9099 and the exact Wikipedia journey URL remain required; no substitute or mocked answer certifies that URL.

| Required workflow | Single-user | Multi-user |
| --- | --- | --- |
| Fresh setup, provider discovery, real first Chat | PASS: blank-model discovery, selected-model validation/save, real first response200; manual key UI recovery succeeds | Functional setup/real Chat PASS through documented operator config; separate Provider Keys route FAIL077 |
| Normal saved Chat, second turn, reload/persistence | FAIL062: both real replies persist, but first turn is duplicated in another conversation and mode switches to Character | FAIL062: real two-turn continuity/reload works, but mode changes and participant-mismatch persistence400 occurs; blocking overlay also observed in later Media→Chat |
| Public file ingest → content search → Chat/QA with citations | Functional PASS: exact Aster content, real answers, one mapped citation and correct Media source; ingest/Chat UX findings remain | Functional PASS: Alice Cedar ingest/search/Media→Chat/default QA, one citation, correct preview and actual Media jump succeed. Specific confidential Indigo Media2 yields no answer/sources. Ingest/Chat/source-type UX findings remain; Bob own Media ingest/read succeeds |
| Exact Wikipedia URL → search → grounded Chat | BLOCKED: remote extraction fails, zero articles stored; failure is honestly counted, but generic error065 obscures the reason | BLOCKED: exact URL attempted; process-web-scraping returns200 transport but zero stored articles and extraction failure. Dependent article search/grounded Chat cannot pass; generic065 recurs |
| Notes → generated Flashcards → save/review/reload | Functional PASS: exact five facts generate5grounded cards, all saved/linked/reviewed and persisted; UX/privacy055/058/063/064 remain | Bob: generation, five saved grounded cards, five Good reviews, schedules and completed deck session survive reload. Source link FAIL081 opens blank Note; privacy064 recurs |
| Create/save/back/apply Prompt → actual request and real answer | Application PASS: synced pirate prompt, clean Back, exact system payload and real pirate answer/reload;062 recurs, output uses Arrr casing | Application PASS: saved/applied pirate prompt, exact real system request and ARRR reply retained after reload;062 mode change recurs |
| Tracked character Chat, context replacement, saved history | FAIL068/070: direct Aster/Robot entry and real replies work, but replacement reverts and settled reload hides a persisted final answer | FAIL068/070: direct Cedar reply/persistence works; picker reverts Robot5 to Cedar4. Backlink initially shows both saved turns, but settled reload shows only user despite two server rows |
| Chat → Note/backlink and reviewed Flashcard → study | Artifacts/pairs/scheduling PASS for normal and deliberate characters, all7cards studied; completed-session accounting FAIL076 | Earlier Media-Chat reply Note/card/review/reload/server schedule PASS. Deliberate Cedar Note saves; backlink restores content but retains Robot context and lacks saved-message actions085, blocking its card/study. Source-link081 also recurs |
| Media analysis → Review → re-analysis → reload | Functional PASS: original grounded analysis in Review; distinct correct second analysis survives reload/restore; Markdown presentation060 remains | Functional PASS: Review/full-content search, real distinct second analysis, version2 and settled reload succeed; Markdown060 remains |
| Permission-aware delete → Trash date → restore | Functional recovery PASS through direct Trash route: truthful date, cleared selection, preserved content/analysis; last-item navigation071 and console072 fail | Admin: sole owned item deleted/restored with exact content and truthful Trash date. FAIL071 hides navigation; direct route used for recovery. Console072 recurs; ordinary-user delete disabled correctly |
| Auth/disconnect/offline logout/reconnect | PASS: manual-key disconnect clears two tabs and stops private polling; invalid-key guidance, re-entry, offline disconnect and recovery succeed | FAIL080 at natural access-token refresh: SQLite self-lock is reported as invalid session, with continued401 polling and unrelated modal. Offline two-tab logout/reconnect and Alice queued draft recovery PASS; no remote revocation claim |
| Admin creates Alice/Bob; owned/foreign API and browser isolation | N/A | UI creation and owned/foreign API controls PASS: Notes GET/versioned PUT404 and foreign-only Media2 GET/valid PUT404; originals unchanged. Alice recovered draft POST201 and Bob GET404. Browser isolation FAIL: old Quick Ingest metadata082, QA question metadata086, and full Note handoff text restored under Bob via browser history064 |

Frozen multi-user browser cutoff: **2026-09-15T11:14:21Z**. All twelve named workflow rows have observed outcomes or explicit blocks. Exact Wikipedia search/grounded Chat remains blocked by zero extracted articles; deliberate Cedar card/study is blocked by085. Mixed deck plus undecked session accounting076 was observed failing in single-user and was not separately executed in multi-user; the latter is a coverage limit, not a pass. Repair verification and the next full run must cover it.

### Cycle 3 observations

Single-user setup: Chat readiness explains that provider verification/first chat is required. Blank-model Validate discovers the exact local model; selecting, saving, and validating succeeds. First-chat request154 returns200 with `Hello, it's great to meet you!` from the actual configured model, then completion157 returns200. The manual-key access screen accepts the isolated key and opens Home/first-source onboarding. Audio, advanced paths and MCP remain deferred within the documented scope. Evidence: `/private/tmp/uat-cycle3-single-first-chat.json` and browser session `cycle3-single-20260915`.

#### UAT-056 — P3: Ingest review claims readiness before validating required analysis provider

- Mode / step: fresh single-user Home → first source → Paste → queue260B Aster text → Configure → Standard preset → Next.
- Expected: validate required analysis configuration before the Review step claims readiness; preserve an actionable provider choice.
- Actual: Review displays Ready to Process, Standard · OCR + Extract + Analyze + Chunk, and enabled Start Processing. Clicking it returns to Configure, focuses the blank Analysis provider, and reports Choose an analysis provider before running ingest analysis. No ingest request was sent in this failed attempt; no stored-content failure is claimed.
- Status: open, frozen cycle3; continue with explicit provider selection to exercise downstream steps. Evidence: review snapshot `/private/tmp/.playwright-cli/page-2026-09-15T07-57-42-450Z.yml`, `/private/tmp/uat056-ingest-late-validation.txt`.

Harness observation: clicking the visually hidden Paste radio input timed out because its visible icon intercepted the pointer. Clicking the visible Paste label selected it normally. This is a locator correction, not a product defect or skipped workflow.

#### UAT-057 — P3: Ingest duration estimate ignores actual analysis cost

- Single-user260B document with Standard analysis/chunking reports about3seconds; actual successful run reports50seconds. The processing view changes to generic advice about large files even though this source is tiny.
- Read-only trace confirms the estimate uses media type/file bytes/preset only, without provider/model/inference cost. Expected: avoid a precise misleading estimate for unmeasured model work, or show a defensible range/uncertainty.
- Status: open. Evidence: same Review snapshot as056 and `/private/tmp/uat-cycle3-single-ingest-complete.txt`. Actual ingestion succeeds; no failure is inferred from duration alone.

#### UAT-058 — P3: Fresh Home/setup browser title is blank

- Fresh root onboarding and subsequent Companion Home have no Page Title in the browser snapshots. Direct `document.title` inspection while Home's ingestion modal is open returns an empty string.
- Read-only trace confirms Home/setup routes, shared layouts and document shell supply no title; signed-out and several other routes own theirs separately. Expected: useful current-route title on fresh setup and Home.
- Status: open; no prior-account title leak is claimed in this fresh browser. Session `cycle3-single-20260915`, observed08:00UTC.

#### UAT-059 — P3: Disabled Reading Queue is presented as a temporary outage

- After successful setup/key entry, Home correctly explains personalization is unavailable. Other dependent cards show Setup required, but Reading Queue says Temporarily unavailable / Reading queue data is temporarily unavailable.
- No reading-list request occurred. Read-only trace: `CompanionHome/hooks.ts` initializes reading as degraded then skips fetching without personalization; `CompanionHomePage.tsx` renders that as a failed fetch without checking the prerequisite.
- Expected: distinguish feature setup/unavailability from an actual transient request failure. Status: open. Home snapshots07:55–07:56UTC; first-ingest request inventory has no reading-list call.

Single-user source control: explicit llama.cpp selection after056 starts ingest job1. Results report1succeeded/0failed/50seconds; Open in Media directly opens `/media?id=1` on the first try. Media contains the exact260characters/43words and a correct real generated summary of all six fixture facts. Full-text Aster search returns that item. Evidence: `/private/tmp/uat-cycle3-single-ingest-complete.txt`, `/private/tmp/uat-cycle3-single-media-source.txt/.png` (PNG visually inspected). Source→Chat/QA/citations remain in progress.

#### UAT-060 — P3: Media analysis displays raw Markdown formatting

- The real ingest analysis contains Markdown headings, bold facts and bullets. The default Media Analysis display shows literal `**Overview**` and `**Project Aster**` rather than rendered formatting; the screenshot confirms this visually.
- Expected: readable rendered analysis by default, or a clearly labeled raw/source view. Status: open. Content is complete/correct; this is a presentation finding. Evidence: `/private/tmp/uat-cycle3-single-media-source.png`.

#### UAT-061 — P3: Ingestion progress reports unconfirmed processing stages

- During job1, UI snapshots show50–55% while the recorded actual job response reports20%, progress_message:process. Read-only inspection confirms an interval fabricates increasing percentages and transitions through Analyze/Store without server confirmation.
- Expected: use actual progress or honest indeterminate activity when backend stages are unavailable. Completion remains tied to actual results and is not falsely reported in this run.
- Status: open. Evidence: processing snapshots07:59:29–07:59:54UTC and browser response391; `QuickIngestWizardModal.tsx` interval beginning1088.

#### UAT-062 — P2: First saved Chat duplicates history and silently changes to Character mode

- Fresh single-user Media → Chat correctly opens `/chat` with the source in the composer, Standard chat, Saved, no character selection. First request617 sends the exact local model, save_to_db:true and no conversation/character ID. A second factual question is queued while the first response generates.
- Both answers succeed: the first summarizes the supplied fixture; the second says the garden opens18December2026 and Mira Chen coordinates it. Independent server GET200 confirms both turns in `d77f1ba1-2de6-4fbe-b9b4-f39c50baab95`.
- Unexpectedly, browser POST632 creates `ada6a95c-7e4b-4580-ac1b-065183de8e8b` and POST634/643 copies the first user/assistant pair there. Independent GET200 confirms the duplicate. The visible workspace changes from Standard chat to Character Chat / Assistant with no user selection; title becomes Helpful AI Assistant(timestamp), while its header says Untitled.
- Expected: a fresh saved standard conversation has one canonical history and preserves its selected mode. Status: open; root cause under read-only investigation. No loss of the actual replies is claimed. This differs from052's existing-neutral-conversation fork.
- Evidence: `/private/tmp/uat062-first-saved-chat.txt`, `/private/tmp/uat-cycle3-single-original-chat-server.json`, `/private/tmp/uat-cycle3-single-extra-chat-server.json`, request inventory `/private/tmp/uat-cycle3-single-chat-final-requests.txt`.
- Harness detail: typing a newline pressed Enter and submitted the source-only first turn. The factual question was then queued through the visible Queue request control; both actual requests were inspected. No claim that the first request already contained that question.

UAT-055 scope follow-up: fresh Study initially renders without the deprecated List error and truthfully offers Choose a study path with1available card. Opening Manage emits the same AntD List deprecation as a console error. The prior RecentStudySessions repair remains verified; **055 is open for the newly exercised Manage list**. Evidence: `/private/tmp/uat055-cycle3-manage-console.txt`.

UAT-058 scope follow-up: fresh navigation to `/flashcards` also has an empty document title. Include this required route in the title repair rather than treating it as a Home-only issue.

#### UAT-063 — P3: Loaded saved Note has conflicting save status

- Open the newly Chat-derived note from its server-loaded Notes list. The accessibility status says No server save status yet, while the footer correctly says Version1 / Last saved and Origin: Saved from Chat. The user has made no edits. Read-only inspection confirms this status is an sr-only live region; it is an accessibility inconsistency, not a visually painted header warning.
- Expected: distinguish a saved, loaded note from a new unsaved draft and display consistent status. The artifact exists on the server; no persistence failure is claimed.
- Status: open. Evidence: `/private/tmp/uat-cycle3-single-note-open.txt`, note `5724dbeb-4613-463f-8fbc-895e1f114ebe`.

Single Chat-derived control: Save to Notes request950 returns201; Note body is exactly the visible answer, origin Saved from Chat, backlink restores the original server conversation including both real answers. Save to Flashcards requires a Question before enabling Save; the reviewed pair is independently persisted as card `72e26812-c23c-45a2-a97d-178502008af6`, linked to original conversation/reply and supporting note `661e7b26-f149-4110-a041-16744c61348e`. Evidence: `/private/tmp/uat-cycle3-single-note-save.json`, `/private/tmp/uat-cycle3-single-note-backlink.txt`, `/private/tmp/uat-cycle3-single-card-review-dialog.txt`, `/private/tmp/uat-cycle3-single-flashcards-server.json`. Study scheduling/reload remains underway.

Study follow-up: Show Answer displays the correct pair. Good persists repetitions1/version2 at08:12:04.690Z with due08:22:04.690Z; browser reload shows Reviewed today1, completed session and next review in10minutes. Evidence: `/private/tmp/uat-cycle3-single-card-answer.txt`, `/private/tmp/uat-cycle3-single-study-reloaded.txt`, `/private/tmp/uat-cycle3-single-flashcards-reviewed-server.json`.

#### UAT-064 — P2: Notes-to-Flashcards exposes private Note content through URLs and account-switch history

- Save/reload exact biology fixture, then Notes → More actions → Generate flashcards. The destination URL includes the full plaintext body in `generate_text`, plus source title/ID. The route announcer also repeats that full URL. The source arrives correctly in the form.
- Expected: transfer owned source context without exposing the full private Note in navigation URLs/browser history; use a scoped transient handoff that preserves the exact draft and rejects a different account.
- Final multi-user control confirms cross-account content exposure: Alice's recovered private Juniper Note opens Generate with its full body/title/source UUID in the URL. Normal Logout → Bob Login → browser Back restores the complete Alice body in Bob's Generate textarea. Browser principal verification confirms Bob user3; independent Bob GET of the same Note returns404. No generation or save was submitted as Bob, and this synthetic fixture does not establish external disclosure. Evidence: `uat-cycle3-multi-private-handoff-bob-history-result.txt`, `uat-cycle3-multi-private-handoff-bob-visible.png`, `uat-cycle3-multi-private-handoff-browser-principal.json` in the main multi-user bundle.
- Status: open. Evidence: Notes-to-generation navigation snapshot `/private/tmp/.playwright-cli/page-2026-09-15T08-13-58-922Z.yml`, biology note `7d92dc41-0c91-469c-a246-1b414c76485d`.

Single Notes→Flashcards result: exact five-fact Note saves/reloads. Generate uses blank optional provider/model inputs and real server defaults; request434 returns5correct cards with grounded verdict, verified5/refuted0/unverified0, and actual source snippets. No draft editing or verification bypass was needed. All five save to Cycle3 Biology Cards with the original Note source ID. Their Study Show Answer/Good actions finish successfully; independent server reads show each repetitions1/version3 and a10minute due interval. Reloaded Study retains the completed session. Evidence: `/private/tmp/uat-cycle3-single-generated-cards.json`, `/private/tmp/uat-cycle3-single-generated-cards-review.txt`, `/private/tmp/uat-cycle3-single-all-cards-saved-server.json`, `/private/tmp/uat-cycle3-single-all-cards-reviewed-server.json`, `/private/tmp/uat-cycle3-single-biology-study-reloaded.txt`. The action harness read each front/back but did not print its collected text; persisted pairs and the separately inspected generation draft supply the content evidence. This is functional success with recorded route/status/privacy issues, not issue-free acceptance.

Single Knowledge QA control: default AI setting (Server default), one selected source (Media1), and question “When does Project Aster garden open, and who coordinates it? Cite the source.” produce the correct18December2026/MiraChen answer with one mapped citation. The citation jumps to its source; View source preview shows the exact original excerpt and chunk `late_chunk:1:0`. Open in Media opens a new tab at `/media?id=1` containing that source and its saved analysis. Relevance is truthfully not measured. Evidence: `/private/tmp/uat-cycle3-single-qa-stream.txt`, `/private/tmp/uat-cycle3-single-qa-cited-answer.txt/.png` (PNG inspected), `/private/tmp/uat-cycle3-single-qa-source-preview.txt`, `/private/tmp/uat-cycle3-single-qa-open-media-confirmed.txt`. Confidential-content negative control remains pending. Knowledge and Media also have empty document titles, expanding058.

#### UAT-065 — P3: Failed URL extraction loses its useful error context

- Single-user exact Wikipedia journey URL, Quick preset, analysis/chunking off. The result correctly reports0succeeded/1failed and creates no article. This verifies the false-success portion of044 is repaired for this run.
- Actual visible explanation: “An unexpected error occurred. Try again or check the server logs.” and Error · Retryable. Response381 says `Failed to extract: https://en.wikipedia.org/wiki/Playwright_(software)`, with `stored_articles:0` and empty media_ids. Runtime retrieval logs show Wikimedia robots requests receiving403. No bypass was attempted.
- Expected: explain that source retrieval/extraction failed, preserve a safe useful reason when known, and avoid unsupported retry guidance. The remote block itself is external; article search/grounded Chat remains explicitly blocked.
- Status: open; diagnosis pending. Evidence: `/private/tmp/uat-cycle3-single-wikipedia-failed.txt`, `/private/tmp/uat-cycle3-single-wikipedia-response.txt`, `/private/tmp/uat-cycle3-single-wikipedia-requests.txt`.

UAT-065 read-only diagnosis: ingest transport retains `errors[0]`, but the results component renders only the classifier's generic message. “Failed to extract” matches no known category, so UNKNOWN also labels it retryable. The scraping service separately reduces extraction failures to the URL, losing the specific cause/code. Preserve safe structured extraction information and give an honest failure category; do not weaken retrieval controls.

#### UAT-066 — P3: Citation preview changes the source type from Document to Other

- Open the Aster answer's Document source card, then View source preview. The same source is labeled Other in the preview, despite the exact content and Media link being correct.
- Read-only diagnosis: SourceCard defaults missing sourceType to media_db, while SourceViewerModal has no equivalent fallback. This is inconsistent source labeling, not evidence loss or a wrong-source claim.
- Expected: both views use the same normalized source type. Status: open. Evidence: `/private/tmp/uat-cycle3-single-qa-cited-answer.txt`, `/private/tmp/uat-cycle3-single-qa-source-preview.txt`.

Single Prompt control: created `Cycle3 Pirate Prompt` (`pa_ac8a-8cb1-480-4cb8`) with the journey's exact system instruction. Save→Back returns one Synced row without a false dirty warning. Use in chat→Use as System Instruction preserves the prompt. After starting a new saved conversation and explicitly choosing General chat, the pre-send view says Standard chat/Saved/Custom prompt. Request459 sends the exact pirate system instruction plus “Tell me about the weather today.” to the real configured model. Its answer begins “Arrr, matey!” and requests a location instead of inventing weather; it persists through reload. The model uses title-case Arrr, so literal uppercase ARRR is not claimed. Evidence: `/private/tmp/uat-cycle3-single-prompt-saved-back.txt`, `/private/tmp/uat-cycle3-single-pirate-before-send.txt`, `/private/tmp/uat-cycle3-single-pirate-request.json`, `/private/tmp/uat-cycle3-single-pirate-answer.txt`, `/private/tmp/uat-cycle3-single-pirate-reloaded.txt`.

UAT-062 independent control: this new first turn has no queued second request. It still creates canonical conversation `232de83b-7e3c-45a9-bd6c-0cb77718b03e` (system/user/assistant) and extra `2178b118-a9a7-4bb7-afa9-301e5279ef0d` (duplicate user/assistant), then silently displays Helpful AI Assistant Character mode. Independent authenticated GETs confirm both copies. Thus queuing is unnecessary to trigger062. Evidence: `/private/tmp/uat-cycle3-single-pirate-requests.txt`, `/private/tmp/uat-cycle3-single-pirate-original-server.json`, `/private/tmp/uat-cycle3-single-pirate-duplicate-server.json`.

#### UAT-067 — P3: New character opens with a disconnected-form console error

- Navigate to Characters→New character. Console emits: “Instance created by useForm is not connected to any Form element. Forget to pass form prop?”
- The form remains usable and successfully creates Cycle3 Aster Guide. Expected: normal creation without a lifecycle/form wiring error. Root cause not yet diagnosed; no save failure is inferred.
- Status: open. Evidence: `/private/tmp/uat-cycle3-single-character-console.txt`, `/private/tmp/uat-cycle3-single-character-created.txt`.

Single tracked control: UI-created Cycle3 Aster Guide (character4) opens a fresh saved tracked conversation `e004e361-604c-4a6a-85ce-019bc75fdeea`. Request324 uses complete-v2 with character context and the real model; answer is “Project Aster has seven raised beds.” Save to Notes375 and reviewed Flashcard400 both return201, linked to reply `pa_ec2a-b676-39f-15e0`. Note `080ce690-1d8f-4920-86b9-bc90bcc4f7ba` contains only that visible answer, reports Saved from Chat, and its Open conversation action restores the correct character/history; reload retains the factual answer. Card `eae2875e-e170-4e34-aed3-6df4579d007a` requires a question and has the correct nonblank answer; study control pending. Evidence: `/private/tmp/uat-cycle3-single-aster-character-request.json`, `/private/tmp/uat-cycle3-single-aster-character-answer.txt`, `/private/tmp/uat-cycle3-single-aster-note-open.txt`, `/private/tmp/uat-cycle3-single-aster-backlink.txt`, `/private/tmp/uat-cycle3-single-aster-tracked-reloaded.txt`, `/private/tmp/uat-cycle3-single-aster-card-dialog.txt`.

#### UAT-068 — P2: Character picker reverts a replacement to the previous character

- With saved Aster Guide character4 active, create Cycle3 Beep Robot through the UI using the exact journey instruction “Always respond with exactly: BEEP BOOP.” Return to Chat, open its current-character selector, and choose the visible Cycle3 Beep Robot option.
- Actual: conversation content clears, but the settled view shows Aster Guide and its greeting again. The next “Hello, who are you?” creates `ff6e44a4-803d-45dc-9d85-d8b881a71ae1` with `character_id:4`; its real answer is “I am the Cycle3 Aster Guide for Project Aster.” The selected robot was not applied. The original Aster conversation remains on the server; no history deletion is claimed.
- Expected: selecting the robot establishes its identity/context and a distinct robot conversation; UI selection cannot be overwritten by stale identity hydration. This is a fresh failure at the character replacement boundary addressed by032; root cause under read-only investigation.
- Status: open. Evidence: `/private/tmp/uat-cycle3-single-character-switch-picker.txt`, `/private/tmp/uat-cycle3-single-robot-created.json`, `/private/tmp/uat-cycle3-single-robot-before.txt`, `/private/tmp/uat-cycle3-single-wrong-character-create.json`, `/private/tmp/uat-cycle3-single-wrong-character-answer.txt`.

UAT-068 direct-entry control: Characters→Chat as Cycle3 Beep Robot correctly selects character5 and creates `ff85f37d-2c47-4467-af8b-db3fc305729b`. Real complete-v2 returns “BEEP BOOP” (no final period); the requested character behavior is present. This does not pass the failed in-chat replacement. Read-only trace identifies a competing legacy selectedCharacter reconciliation in useCharacterGreeting that can overwrite the canonical picker selection during asynchronous mirroring. Evidence: `/private/tmp/uat-cycle3-single-robot-direct-before.txt`, `/private/tmp/uat-cycle3-single-robot-direct-answer.txt`, `/private/tmp/uat-cycle3-single-robot-direct-requests.txt`.

Multi-user recurrence: Alice's UI-created Cedar Guide4 returns a grounded answer and persists200. Choosing UI-created Robot5 through the active picker clears the old visible history, then the label/URL revert to Cedar4. The next real request creates conversation `dcd5ec14-e564-4372-bc2f-18f1106dbefc` with character context and answers “I do not know.” instead of BEEP BOOP. The original Cedar conversation remains available through its saved Note; no server deletion is claimed.

#### UAT-069 — P2: Notes controls shrink the results list to zero height

- At1280×720 with5notes and3recent notes, reopen Notes while Views/Filters are expanded (the default state used in this run). The visible sidebar contains only controls and Recent Notes. Clicking an existing full-list note times out because those controls intercept its pointer position.
- Visual inspection confirms the layout; DOM measurement gives the `flex-1 overflow-y-auto` result ancestor height0, while note rows remain below/behind the header. Window scrolling does not expose a usable list.
- Expected: the note list retains usable space or the whole sidebar scrolls at normal laptop window heights. Collapsing both Views and Filters restores the list; the same note then opens normally. No force-click, viewport enlargement or data loss is involved.
- Status: open. Evidence: `/private/tmp/uat-cycle3-single-note-pointer-block.txt/.png`, `/private/tmp/uat-cycle3-single-note-list-top.png` (both PNGs inspected). Root cause under read-only investigation.

#### UAT-070 — P2: Reload hides the tracked character's persisted final answer

- Reopen Aster conversation `e004e361-604c-4a6a-85ce-019bc75fdeea` via the Note backlink after the Robot control. Explicitly wait for the visible seven-beds answer: succeeds. Reload, then wait for that same visible answer for30seconds: times out.
- Settled timeline contains only the greeting and user question. Actual reload GET1948 returns all3messages, including final reply `pa_ec2a-b676-39f-15e0` with sender Cycle3 Aster Guide, speaker_character_id4 and the correct answer after its reasoning block. Thus server persistence succeeds; frontend hydration drops the final visible reply.
- Expected: reload restores all saved turns, normalizes tracked speaker identity and keeps model reasoning separate from the visible answer. Status: open; read-only diagnosis pending.
- Evidence: `/private/tmp/uat-cycle3-single-aster-reload-settled-failure.txt`, `/private/tmp/uat-cycle3-single-aster-reload-settled-requests.txt`, `/private/tmp/uat-cycle3-single-aster-reload-response.json`. Earlier immediate reload snapshots were inconclusive; this settled control supersedes the preliminary reload-success statement above. The earlier backlink response body was evicted from the browser capture and is not counted as retained evidence.

UAT-070 causal confirmation: bounded read-only IndexedDB inspection finds active local history `pa_0176-43d1-093-c548` with exactly the server's greeting/user IDs, both lacking serverMessageId; the assistant is absent. The server loader omits serverMessageId when mirroring, then classifies the restored cache as unsynced and preserves it instead of the fetched complete timeline. Existing matching mirrors need safe reconciliation; genuinely unsynced content must remain protected. Evidence: `/private/tmp/uat-cycle3-single-aster-local-mirror.txt` emits only IDs, roles and content lengths.

Multi-user recurrence: Alice's original Cedar conversation initially shows both saved turns after its Note backlink, then settled reload shows only the user. Independent GET200 returns two server rows, including the exact answer with sender Cycle3 Cedar Guide; this conversation has no system/greeting row. No multi-user IndexedDB capture is available, so the exact cached fields are not independently established. Read-only inspection confirms the API adapter already maps custom non-user/system/tool senders to assistant and timestamp to created_at. The independent GET omitted include_metadata; its absent metadata is not evidence of missing stored speaker metadata or a separate sender-projection defect. Preserve the exact two-server/one-visible fixture in the existing hydration repair.

Single Media cycle: Review content search Aster finds media1 and displays its complete first analysis. Inspector Generate uses the already selected exact local Qwen model and a new source-grounded instruction. Result “ASTER_ANALYSIS_TWO. Project Aster opens on18December2026, is coordinated byMiraChen, and has seven raised beds.” survives an explicit settled reload. Delete confirmation truthfully promises Trash recovery, and success clears `?id=1` plus the inspector. Direct Trash route shows Deleted: Sep15,2026,1:59AM. Restore removes the item from Trash; reopening media1 retains exact original content and the second analysis. Original analysis also displays raw Markdown in Review, expanding060. Evidence: `/private/tmp/uat-cycle3-single-review-original-analysis.txt`, `/private/tmp/uat-cycle3-single-analysis-reloaded.txt`, `/private/tmp/uat-cycle3-single-delete-confirmation.txt`, `/private/tmp/uat-cycle3-single-media-deleted.txt`, `/private/tmp/uat-cycle3-single-trash-actual.txt`, `/private/tmp/uat-cycle3-single-media-restored.txt`.

Harness/approval note: the second-analysis request was initially rejected by automatic approval review because its model destination was considered unspecified. Read-only runtime configuration confirmed the exact selected model maps to local llama.cpp9099 and the source is our synthetic260B fixture. The retry explicitly stated that evidence and was approved; no model rerouting or approval bypass occurred.

#### UAT-071 — P2: Deleting the last Media item hides the Trash entry

- Delete the only active source. Media shows the first-ingest empty screen without its usual Trash button. The temporary Undo toast works as an available immediate action, but normal Trash navigation disappears when the toast expires.
- Expected: recovery remains discoverable when there are no active media items, especially immediately after deletion. Direct `/media-trash` navigation works and restores the item; that independent control does not pass the missing UI entry.
- Status: open. Evidence: `/private/tmp/uat-cycle3-single-media-deleted.txt`, `/private/tmp/uat-cycle3-single-trash-date.txt` (this latter capture is still the empty Media route, not Trash), `/private/tmp/uat-cycle3-single-trash-actual.txt`.

#### UAT-072 — P3: Media deletion emits deprecated/context-free notification errors

- Successful deletion emits two console errors: AntD Notification `btn` is deprecated in favor of `actions`; static message functions cannot consume dynamic theme context and should use App context.
- Expected: the normal deletion/Undo flow uses supported context-backed notifications without console errors. Delete/restore succeeds; no toast-action failure is inferred.
- Status: open. Evidence: `/private/tmp/uat-cycle3-single-delete-console.txt`. This is the Media notification path; the earlier Prompt-specific046 repair is not claimed regressed.

Single security negative control: UI-uploaded synthetic Indigo source stores as media2 with chunking enabled and analysis disabled. It includes the classification-trigger word token but no real credentials. With only media2 selected, default Knowledge QA reports “Security settings excluded all retrieved sources,” returns no contexts/citations/answer and exposes no source excerpt. The earlier public Aster positive control passed. No ACL/classifier changes or override were used. Evidence: `/private/tmp/uat-cycle3-single-confidential-source.txt`, `/private/tmp/uat-cycle3-single-confidential-ingest-result.txt`, `/private/tmp/uat-cycle3-single-confidential-media.txt`, `/private/tmp/uat-cycle3-single-confidential-qa-result.txt`, `/private/tmp/uat-cycle3-single-confidential-qa-stream.txt`.

Single tracked-card study: all7available cards display correct reviewed question/answer pairs, including “How many raised beds does Project Aster have?” / “Project Aster has seven raised beds.” Good persists the tracked card's repetitions1/version2, last_reviewed09:08:49.202Z and due09:18:49.202Z. Completed session survives reload. Evidence: `/private/tmp/uat-cycle3-single-seven-card-review.txt` (returned visible pairs, unlike the earlier console.log-only harness), `/private/tmp/uat-cycle3-single-tracked-study-reloaded.txt`, `/private/tmp/uat-cycle3-single-after-tracked-study-server.json`.

#### UAT-073 — P2: Media reading-progress saves send invalid zoom units

- While switching restored Media1 to the newly ingested Media2, the old item's progress flush returns422: “Input should be greater than or equal to25.” The automatic PUT fails and logs a warning; it is not a wrong-item-routing claim.
- Read-only diagnosis: useMediaReadingProgress always sends zoom_level1, while the API requires percentage units25–400 (default100). Preserve backend validation and use matching client units.
- Status: open. Original request body was not retained after navigation; the producer/schema mismatch independently explains the captured validation error. Console evidence: `/private/tmp/.playwright-cli/console-2026-09-15T09-00-19-405Z.log`, lines6–7.

#### UAT-074 — P3: Deck dashboard double-counts due learning cards

- After the first learning interval expires, Biology deck displays Total5, Due5, Learning5, New0, but its action says Review10ready. The actual all-deck queue has7cards (six learning plus one new) and reviews each once.
- Read-only diagnosis: dashboard adds analytics due+learning+new, but analytics due already includes expired learning cards. Correct eligible count for this contract is due+new; future learning cards are not ready. Preserve the descriptive Learning count and backend queue semantics.
- Status: open. Evidence: `/private/tmp/uat-cycle3-single-tracked-study-start.txt`, `/private/tmp/uat-cycle3-single-before-tracked-study-server.json`, `/private/tmp/uat-cycle3-single-seven-card-review.txt`.

Execution observations: the file-chooser harness registered the synthetic Indigo file twice before its CLI modal was cleared; the duplicate was visibly marked Already queued and removed before processing, leaving one source. The development-server badge intercepted the compact sidebar Settings button; the visible header Open settings worked. These are retained execution constraints, not inferred production failures.

Disk recovery checkpoint09:12–09:15UTC: host exhaustion (116MiB free) prevented the first attempted disconnect from executing. Only the verified single-UAT frontend18280 was stopped; its4.7GiB generated `.next-live-tier-cycle3-single-20260915` cache was removed and rebuilt. Existing runtime data, browser profile and evidence were preserved; API8000 and model9099 were untouched. Replacement frontend listener PID27711 (launcher27709), exec session99987. Browser reload settled after an HMR/navigation interruption. Subsequent actual UI Disconnect clears the key; the other Knowledge tab shows the credential-needed gate and no Aster answer. Passive polling/re-entry checks continue.

Single-user completion checkpoint2026-09-15T09:27Z: all named single-user workflows have an observed outcome or explicit dependency block. This is completed execution with open product findings, not acceptance. Default public Aster QA and confidential Indigo exclusion both pass. The final seven-card Study control records every visible question/answer and persists the tracked-derived card's first review. Reload shows a completed session, but evidence review found its count is2rather than7; UAT076 below records the confirmed split. Exact Wikipedia article retrieval remains externally blocked.

Auth recovery evidence: actual Disconnect clears Settings and the other Knowledge tab; private polling remains stopped for98.31seconds in Settings and62.97seconds in Knowledge. An invalid key produces clear HTTP401 guidance; entering the actual isolated key restores Core reachable / RAG healthy. Reload retains the key while its immediate health status correctly says not checked yet. Offline Disconnect leaves both tabs on the local credential gate without prior answer content or a browser error; reconnect and key entry restore access and fresh healthy checks. Opening the visible header Show keyboard shortcuts control loads the actual Keyboard Shortcuts dialog. Evidence: retained `disconnected-requests-*`, `disconnected-other-*`, `offline-*` and `help-loaded.txt` captures in the cycle3 single-user evidence directory. This manual-key control does not claim offline JWT revocation.

#### UAT-075 — P3: Server setup guide opens the moved browser-extension repository

- Mode / step: Settings → tldw Server → View server setup guide.
- Actual: a new tab opens `https://github.com/rmusser01/tldw_browser_assistant`. Its README identifies the moved browser extension, requires an already running server, and describes extension development/configuration rather than server installation.
- Expected: the labeled server setup action opens maintained server setup documentation. No broken network request is claimed; the destination loads successfully.
- Status: open. Evidence: `/private/tmp/uat-cycle3-single-setup-guide-destination.txt`; the `serverDocsUrl` translation also points to this extension root. This external documentation link is separate from the successfully loaded in-app keyboard Help dialog.

#### UAT-076 — P2: One all-decks Study run is split into incomplete server sessions

- Mode / step: single-user Study → all decks; continuously reveal and rate all7available cards Good, then reload Recent study sessions.
- Expected: the completed run accounts for all7reviews in its selected all-decks scope and leaves none of its component reviews in an active session.
- Actual: all7visible question/answer pairs and individual scheduling updates are verified, but the completed All decks entry says2cards reviewed. Independent GET200 of review sessions confirms session3 (`due:global`) completed with2cards and session4 (`due:deck:1`) still active with5cards. Both were created/updated during the same09:08:48–49UTC seven-card UI run. The earlier five-card Biology session2 is separately completed at08:17:58UTC.
- Read-only trace: `/flashcards/review` chooses a session from each card's deck, while ReviewTab remembers the last returned session ID and ends that one. Correct per-card schedules do not establish correct run/session accounting.
- Status: open; no session records were modified during investigation. Evidence: retained `uat-cycle3-single-seven-card-review.txt`, `uat-cycle3-single-tracked-study-reloaded.txt` and the independent read `/private/tmp/uat-cycle3-single-reviewed-sessions-server.json`. Original per-review wire bodies are not claimed retained by this new GET.

- Targeted multi-user repair acceptance: `982b03a940` + `536ad461e9`, five decked/two undecked → one new due:global session2, first1 then completed7,7 review200 and automatic End200. Reload retains All decks Completed7; all7 card versions/timestamps advance once. Retained mixed-study-round6. The full fresh rerun and explicit early-End/Undo/practice live checks remain separate.

#### UAT-077 — P2: Provider Keys loading crashes on an object-valued translation

- Mode / step: fresh multi-user admin, authenticated through normal Settings login → Provider Keys.
- Expected: the route remains usable while loading, then displays available keys or an actionable permission/configuration result.
- Actual: the route error boundary and Next development overlay report `TypeError: res.replace is not a function`. The stack identifies `ProviderKeysSettings.tsx:285` and `i18n/icu-format.ts:31`. Requests for profile preferences, OAuth status and user provider keys also return403; their policy reasons remain separate from the rendering defect.
- Read-only cause: the loading branch calls `t("common:loading", "Loading...")`, but English `common.loading` is an object containing title/description/content. The ICU adapter calls `.replace` on that object. The observed stack is from loading text, not proof that the403 response itself caused the exception.
- Status: open, frozen cycle3. Evidence: `/private/tmp/uat-cycle3-multi-provider-keys-settled.txt` and `/private/tmp/uat-cycle3-multi-provider-keys-console.txt`. No permission was changed and no provider key was submitted.

UAT077 follow-up: the key-list403 explicitly says BYOK is disabled in this deployment, so that response is an expected configuration result and must retain accurate guidance. Read-only review found six additional object-valued `common:loading` calls in PersonaGarden Scopes/Policies/Commands/Connections; these are static related findings, not exercised UAT passes/failures. The shared ICU adapter should transform strings only, then preserve upstream handling for object/error-handler and syntax-tree inputs; that compatibility guard does not replace correcting scalar label callers.

#### UAT-078 — P3: Home describes denied automation data as a temporary outage

- Mode / step: fresh ordinary user Bob signs in through the UI and opens Home.
- Actual: Automation Inbox says scheduled-task results are temporarily unavailable. Its scheduled-task/results requests return403 with missing `tasks.read`, rather than a transient server outage.
- Expected: distinguish unavailable account capability from retryable transport/server failures, and avoid eager privileged reads when authoritative capability information is available. Preserve backend permissions.
- Status: open. UI evidence: `/private/tmp/uat-cycle3-multi-bob-home.txt`; exact request/response evidence is being retained by the Bob UAT runner. Read-only trace: `useScheduledTaskHomeSignals` runs whenever capability loading ends; `AutomationInboxCard` renders generic temporary-unavailability copy for an error.
- Repair tracking: TASK-13260.20. OpenAPI/deployment capability flags do not establish user permissions; preserve independently available sources and clear stale account results.

#### UAT-079 — P3: Ordinary Note save eagerly requests privileged monitoring alerts

- Mode / step: ordinary user Bob creates and saves Cycle3 Bob private note through the UI.
- Actual: the Note saves201 and own reads return200, but the post-save monitoring request returns403 missing `system.logs`. The editor remains usable and no blocking overlay appears.
- Expected: successful ordinary Notes actions should not eagerly request an unavailable admin monitoring endpoint. Keep its permission boundary and preserve legitimate monitoring feedback for authorized accounts.
- Status: open. Note `a80707b3-c761-4d8a-8bbe-fe2222886bc2`, version1; UI evidence `/private/tmp/uat-cycle3-multi-bob-note-saved.txt`, request evidence retention in progress. Read-only trace: `loadMonitoringNoticeForSavedNote` requests `/monitoring/alerts` without a capability guard, then silently catches permission failures. Account changes during authorized monitoring reads also need stale-result protection in the repair.
- Repair tracking: TASK-13260.18. Preserve custom-role monitoring permission and successful ordinary saves; role names alone cannot establish the required entitlement.

#### UAT-080 — P2: Transient refresh failures are treated as invalid sessions and disrupt recovery

- Mode / step: fresh multi-user Alice and Bob reach approximately30minutes after their normal UI logins while opening Media or studying already-saved Biology cards.
- Actual: refresh returns401 `Invalid or expired refresh token`. Bob sees Credentials required plus a blocking “Can't reach your tldw server” modal saying `Request was aborted during token refresh` for the Flashcards analytics request. `/auth/sessions` continues401 polling at roughly5-second intervals. Alice's normal UI re-login succeeds; saved data remains intact. Bob's five generated cards were saved before the interruption and are not regenerated for recovery.
- Root-cause evidence: Alice03:19:33.937 and Bob03:21:29.474 Pacific logs report a locked database through session refresh, followed by invalid-session401. Read-only source tracing finds `refresh_token` uses `Depends(get_db_transaction)`, whose SQLite path holds `BEGIN IMMEDIATE` for the request, then awaits `SessionManager.refresh_session` opening a second write transaction in `update_session_tokens_for_refresh`. The outer request holds the writer lock the inner service needs. The existing non-locking login connection dependency explicitly avoids this same separate-service self-lock. This is stronger evidence than the initial external-contention hypothesis; a real SQLite request regression remains required during repair.
- Expected: retain valid credentials across truthful retryable refresh-service failures; real invalid/expired/revoked sessions must still require reauthentication and stop private polling. Cancellation caused by refresh must not open an unrelated server-unreachable modal or discard saved cards/scoped drafts.
- Status: open, TASK-13260.24. Evidence: `/private/tmp/uat-cycle3-multi-auth-session-boundary.json`, `uat-cycle3-multi-auth-refresh-boundary.json`, `uat-cycle3-multi-bob-refresh-denial.json` and `uat-cycle3-multi-bob-biology-review-entry.txt`. Exact sanitized backend snippets are being retained. Additional verifier logins were initially considered as a possible cause; no session-limit/eviction cause is established, and the main helper now reuses one private verifier session per account.

#### UAT-081 — P2: Flashcard source link opens a blank Note instead of its saved source

- Mode / step: multi-user Bob; after generating, saving and reviewing five Biology cards, click the actual Note source link in Manage.
- Actual: navigation reaches `/notes?source_ref_id=9ae43c6a-2458-4910-97fd-013d888914f7`, but the settled editor says New note with an empty title/body. No detail request is dispatched by that route. Independent owned GET returns the original version1 Note, and selecting its visible library row opens it correctly. Generation, card provenance, reviews and persistence passed; source navigation did not.
- Expected: open the specific saved source through normal owned loading and editor protections, with truthful unavailable handling if it is missing or inaccessible.
- Read-only trace: `Flashcards/utils/source-reference.ts` emits `source_ref_id`, but Notes does not consume that query. Its existing deep-link effect reads a separate last-note setting. The same builder's Media and message branches also require comparison with their actual route contracts; those branches are static related scope, not live-confirmed failures in this finding.
- Status: open, TASK-13260.25. Evidence: `/private/tmp/uat-cycle3-multi-bob-biology-source-link-opened.txt`, `uat-cycle3-multi-bob-biology-source-link-wait.txt`, `uat-cycle3-multi-bob-biology-source-link-settled.txt` and `uat-cycle3-multi-bob-biology-source-link-requests.txt`. A GET recorded before the click does not establish route hydration.

#### UAT-082 — P2: Quick Ingest reveals the previous account's completed result after login

- Mode / step: in the same browser, Bob completes his Copper Finch ingestion, uses Settings Logout, and signs in as admin. Admin's identity is independently confirmed and its Media library is empty.
- Actual: opening Quick Ingest displays Bob's prior source filename `uat-cycle3-multi-bob-source.txt`, one succeeded result, and Open in Media/Workspace/Knowledge actions.
- Expected: logout and account/server changes immediately hide private ingest inputs, progress and results; delayed work must not restore them under another account. Same-account close/resume should remain usable.
- Status: open, TASK-13260.26. Evidence: `/private/tmp/uat-cycle3-multi-bob-admin-ingest-add.txt` and `uat-cycle3-multi-bob-admin-authority.txt`. No stale source action was clicked, so this establishes frontend metadata disclosure, not unauthorized backend access. Visible Ingest More starts the separate admin fixture; reload behavior of the old result was not tested before replacement.

#### UAT-083 — P3: Study remaining count subtracts already-reviewed cards twice

- Mode / step: multi-user Bob's actual five-card Biology study run; discovered during independent evidence review.
- Actual: after one review the header says 3 remaining / 1 reviewed / Available now4. After three reviews it says 0 remaining / 3 reviewed / Available now2 while the bones card is visible; the next water card still displays 0 remaining. All five actual reviews and persisted schedules succeed.
- Expected: the remaining count and accessible announcement match the active queue, including the visible card. Cumulative reviews must not be subtracted from a count that already excludes them.
- Read-only trace: ReviewTab supplies the refreshed shrinking due-count total; ReviewProgress subtracts the cumulative reviewedCount again. Cram supplies a fixed queue length and needs its distinct accounting preserved.
- Status: open, TASK-13260.19. Evidence: [fourth card](../../output/playwright/cycle3-full-uat-2026-09-15/multi/bob/biology-review-4-answer.txt), adjacent second/third/fifth answer captures. This is separate from074's dashboard due/learning overlap.

#### UAT-084 — P3: Next-review label assigns a one-hour count to one due timestamp

- Mode / step: Bob's completed Biology run and settled reload; discovered during independent evidence review.
- Actual: the UI says Tuesday, September15 at3:37AM · 5 cards due. Stored due times are10:37:47,10:38:27,10:39:17,10:39:45 and10:40:14UTC. Only the first card becomes due at the earliest timestamp.
- Expected: accurately label the count's time window or show only the cards due at the stated time.
- Read-only trace: useFlashcardQueries intentionally counts a one-hour window, but ReviewTab's nextDueCardCount label omits that window. The schedules themselves are correct.
- Status: open, TASK-13260.19. Evidence: [settled reload](../../output/playwright/cycle3-full-uat-2026-09-15/multi/bob/biology-after-reload.txt) and [independent schedules](../../output/playwright/cycle3-full-uat-2026-09-15/multi/bob/final-controls.json). Preserve staggered due-time coverage in the repair.

#### UAT-085 — P2: Note backlink restores Chat text with the wrong character and missing save actions

- Mode / step: Alice completes the direct Robot5 control, then opens her saved Cedar Guide Note and its actual conversation backlink.
- Actual: both original Cedar messages appear, but Robot5 remains the active character. The saved assistant reply's More actions offers branch/continue/transform/delete/pin, with no Save to Notes or Save to Flashcards. A Save action attempt times out; no deliberate-Cedar card is created or studied.
- Expected: a linked saved Chat restores its conversation/character identity and saved-message actions together. Another conversation's character cannot remain active for the restored messages.
- Status: open, TASK-13260.15. Evidence: `/private/tmp/uat-cycle3-multi-cedar-card-source-state.txt` and `uat-cycle3-multi-cedar-card-actions-state.txt`. Read-only diagnosis is checking the backlink's direct state mapping and action eligibility; no server data loss is claimed. The dependent deliberate-character card/study remains blocked rather than being replaced by an extra generated turn.

Direct Robot control passes independently: character5 conversation `2b88abd0-9bcb-4782-95cc-22077dc5145d` receives real complete-v2 BEEP BOOP and persist200. This does not pass the failed in-chat picker068 or the later backlink085.

#### UAT-086 — P2: Knowledge QA exposes the previous account's recent questions

- Mode / step: after Alice's offline logout, reconnect and normal Bob login in the same browser profile. Bob has not performed a QA query. His Notes correctly show only Bob records.
- Actual: Knowledge QA Recent displays Alice's exact confidential Indigo and public Cedar questions, answer status, citation count and time metadata. Clicking the Cedar item dispatches `/chat/conversations/a02b8e97-32db-4920-9ed7-99f4f64a5947/messages-with-context?include_rag_context=true`, which returns404. No RAG search/inference occurs and no answer/source body is restored; the UI reports Unable to load conversation and zero sources. The failure also produces a development overlay in the retained snapshot.
- Expected: local QA history and active results belong to the verified originating account/server. Account changes must hide old queries and prevent their restoration or delayed persistence under new credentials.
- Read-only trace: the active KnowledgeQAProvider loads and writes the global `knowledge_qa_history` localStorage key. Its history hydration/mutation guards do not establish an account owner. Other similarly named hooks are not assumed active solely from their filename.
- Status: open, TASK-13260.27. Evidence: `/private/tmp/uat-cycle3-multi-bob-recent-qa.txt` and `uat-cycle3-multi-bob-alice-recent-restored.txt`. This establishes frontend metadata disclosure; the server's404 correctly denies the foreign conversation body.

Independent Bob/admin review: all68 hashes across69 files match and20 JSON files parse. Exact saved/generated pairs, reviews, schedules, completed session, source-link failure/positive control, account metadata leakage and owned delete/restore claims are supported. Findings083/084 were identified after the original sealed report and are recorded here without altering its original captures. Parent credential scan checks14 known runtime values plus JWT/private-key patterns across239 current evidence/tracking files with zero matches; full main multi evidence remains pending. Bob/tracker checkpoint committed `ebb3d3643a`; no application behavior change.

Environment interruption around10:00UTC: disk exhaustion prevented the browser wrapper from opening its npm cache and an independent API login returned500. Bob's first ingest submission had not occurred. The verified current multi frontend processes31535/31537 were stopped, only `.next-live-tier-cycle3-multi-20260915` was removed, and frontend43042 restarted with approximately3.6GiB free. API42500, user data and both browser profiles were preserved. Both runners resumed from fresh snapshots/reload. These failures are retained as environment limitations, not invented product findings.

Second interruption around10:10–10:11UTC: the same isolated dist grew to3.4GiB, dominated by2.6GiB in `dev/cache`; free space fell to117MiB. Alice's citation Media-jump click and Bob's generation-form fills did not execute. Both runners paused with no model/save pending. Parent verified/stopped frontend43042/43043, confirmed18281 closed, removed only its exact generated dist and recovered3.6GiB. One shell stop attempt failed before execution because the disk could not create a heredoc temporary file; the verified direct Node command succeeded.

Environment repair committed `c10e1464fc`: `next.config.mjs` disables supported `experimental.turbopackFileSystemCacheForDev` only when existing `TLDW_NEXT_DIST_DIR` selects an isolated UAT run. Independent review found no issue. Actual Next configuration comparison confirms ordinary dev cache=true, isolated UAT cache=false, all other normalized configuration plus headers/redirects/rewrites/webpack equal. Syntax and scoped lint checks pass; no Python was touched, so Bandit is inapplicable to this configuration-only change. Frontend48482 restarted through the same launcher, `/login` returned200, and both retained browser profiles resumed. API42500, provider, databases, credentials and application behavior remain unchanged. After resumed routes, cache size is8KB and free space about2.9GiB. [Retained environment evidence](../../output/playwright/cycle3-full-uat-2026-09-15/environment/README.md). This exception is not a product repair or a fresh full-run restart.

Bob fixture checkpoint: own Copper Finch Media1 (`1430f1e4-37e3-473e-b6d0-764f8f922f8d`) contains the exact Bob source and differs from Alice's numeric Media1, as expected for per-user databases. Bob's Alice-Note GET and valid version1 PUT both return404; Alice's independent read remains200/version1 with original content. Bob's self-profile GET returned200 at10:07:21Z, without exposing profile values in evidence. Bob Biology Note `9ae43c6a-2458-4910-97fd-013d888914f7` was UI-saved/version1 with the exact five-fact fixture; its handoff reproduces064. Generation/review is still pending. Evidence: `/private/tmp/uat-cycle3-multi-bob-read-controls.txt`, `uat-cycle3-multi-bob-biology-saved.txt`, `uat-cycle3-multi-bob-biology-handoff.txt`.

Multi-source checkpoint after environment recovery: public Cedar QA's source action opens the actual Media item in another tab. Confidential Indigo Media2 is tested with Server default provider/model, Specific Media2 and Web off: excluded_count1, retained_count0, contexts empty and output_emitted=false; the UI reports all retrieved sources excluded and displays no answer/sources. Exact `https://en.wikipedia.org/wiki/Playwright_(software)` extraction is attempted; transport200 does not establish successful ingestion because stored_articles=0 with extraction failure. The dependent article journey remains blocked.

Bob learning completion after080 recovery: real generation produced five distinct cards grounded in the five-fact Note, saved into deck1 Cycle3 Bob Biology. Normal UI re-login resumed those same cards; all five were reviewed Good. Independent reads at10:31:07Z confirm repetitions1, lapses0, version3, learning state and due times ten minutes after each review. One completed session has scope `due:deck:1` and cards_reviewed5. Reload retains all five cards and the completed session. The subsequent actual Note source action fails081; library selection provides a separate positive control.

Admin recovery completion: after Bob→admin login, admin ingests its own sole Silver Wren source without analysis/chunking (job6 owner1; Media1 UUID `3d0e3e25-338e-4ebf-8e80-41c57d47ced2`). Canonical Media capabilities return200/can_delete=true. Actual Delete/confirm returns204 and clears the library, but071 hides Trash navigation; direct `/media-trash` is explicitly a recovery continuation. The date matches deleted_at `2026-09-15T10:40:09.178Z`; visible Restore succeeds. Independent10:41:58Z GET confirms the exact original text/version1 and empty Trash. Notification warnings072 recur. Changing text size emits no progress PUT during this bounded check, so073 recurrence is not claimed. A mistaken verifier request to an unsupported per-item capability URL returned404; the canonical endpoint then passed. Admin storage/quota403 responses say Email verification required and are retained separately from the verified delete permission.

Multi-user setup observation: Alice and Bob were created as ordinary Users through the admin UI. Synthetic `.test` email validation failed visibly and was corrected to `example.com`; no email was sent. The multi-user wizard intentionally hands off to operator documentation. The local-LLM guide prescribes `config.txt`; the API setup editor exposes the endpoint but only comments the model key. Exactly three values were configured in the isolated0600 runtime file (`default_api`, `llama_api_IP`, `llama_model`), then only API18201 restarted. The exact real Qwen model appears and is selectable after a visible model refresh; temporary catalog lag after restart is an environment observation, not a new model-selector bug. Evidence: `/private/tmp/uat-cycle3-multi-operator-provider-config.json` and `uat-cycle3-multi-local-provider-controls.txt`.

UAT062 multi-user recurrence: Alice's two real normal-Chat turns and settled reload retain CEDAR-27. The workspace nevertheless changes to Character mode and uses tracked completion for the next turn; persistence also reports400 `speaker_character_name must reference a selected participant in this chat` before a message fallback201. After Media→Chat produces a correct real answer, this same400 also opens a blocking Next Runtime Error overlay and intercepts the More Actions click. The fallback still saves the answer; the UI error is independently actionable. Evidence: `/private/tmp/uat-cycle3-multi-chat-overlay.txt`, stack through `background-proxy.ts`, `chat-rag.ts` and `useChatActions.ts`. The runner captures then visibly dismisses the overlay to continue. Independent reads of both created server conversations remain underway; do not infer final duplicate counts solely from the main tracked conversation's five rows.

Bob's post-recovery independent reads corroborate the permission findings: own Note200 with exact saved version/content; admin users403 requiring admin; scheduled tasks/results403 missing `tasks.read`; monitoring alerts403 missing `system.logs`. Retained follow-up bodies: `/private/tmp/uat-cycle3-multi-bob-own-controls.txt`. Original pre-reload response bodies had been evicted and are not claimed as captured. The first independent probe used an incorrect credentials mode and triggered two CORS errors; the corrected normal bearer probe used `credentials: omit`. Those two probe errors are harness noise.

Isolation verification constraint: numeric Media and study-session IDs are scoped by each user's database. Equal IDs in Alice and Bob can legitimately resolve to different owned records. Verify content/ownership and choose a known foreign-only identifier before expecting404; do not mutate Bob's own record under a colliding numeric ID as a purported foreign-write check. Note UUID controls are unambiguous.

Runtime transition09:28UTC: after completing single-user checks, stopped only verified frontend parent27709/listener27711, confirmed port18280 was closed, and removed its1.2GiB regenerated build cache to make room for sequential multi-user UAT. Single-user API18200, runtime data, browser profiles and evidence remain available. Application revision stays frozen at `d40e17dc81`. Fresh multi API18201/WebUI18281 and a new browser are now running; normal admin Settings login succeeds, with Core reachable / RAG healthy. Full multi-user workflows remain in progress.

Repair planning: [cycle3 design](../Design/2026-09-15-uat-cycle-3-repairs.md) and [implementation plan](../../IMPLEMENTATION_PLAN_uat_cycle_3.md) split known issues into reviewable Backlog units. These are preparation only; product/test code remains unchanged during both-mode UAT.

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

## Initial checkpoint summary (before repairs)

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

### ENV-007 — Host disk exhaustion interrupts repair verification

- During targeted repairs at approximately19:36UTC, the host volume had102–116MiB free. Shell staging failed before any index change with `no space left on device`; the restarted Next compiler then panicked and returned HTTP500.
- Recovery removed only the original, inactive cycle3 multi-user compiled `dev/static` and `dev/server` directories (about1.4GB). They were untracked, had no open files, and the active runtime manifest pointed to the separate repair build. Logs, trace, manifests, databases, private configuration and evidence were retained.
- About1.2GiB became available. Quick Ingest staging/commit then succeeded. The isolated frontend was restarted; `/flashcards?tab=importExport` returned HTTP200 after rebuilding, followed by successful current-source selector checks. This environment interruption is not a product acceptance pass or a new product finding.

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

1. Repairs and targeted verification are complete for the original 15 findings and four additional findings discovered during retest. Begin the next full UAT from fresh profiles, following the named workflows below.
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
| UAT-016 (new) | Notes reads admin title policy only for an active administrator, with account-scoped caching and before/after-await identity checks. Fourteen focused Notes tests pass. Live ordinary-user Notes renders without a title-settings request or corresponding403. |
| UAT-017 (new) | Streaming passes the explicit citation option to generation, numbers sources consistently with the visible evidence, and instructs supported inline citations. Disabled citations retain prior full-context behavior. Actual-provider-prompt and invalid-reference regressions pass. Live Cedar answer shows Cited answer, one source / one citation, and a working `[1]` jump to its evidence. |
| UAT-018 (new) | Raw ranking remains unchanged for ordering. Only explicit bounded relevance probabilities drive percentages or low-relevance warnings; unknown relevance is labeled “Relevance not measured” consistently across cards, details, and exports. Raw candidate scores such as 25 and -2.5 remain intact. 159 focused tests pass; live Cedar retains its correct cited answer without a false percentage/confidence warning. |
| UAT-019 (new) | Selecting a note sends no graph request; Connections starts collapsed per note/account. Explicit expansion loads once and shows an unavailable state after403, without retries, false empty-link claims, or stale cached chips. Offline/reconnect and identity-race regressions pass. Live Alice note-open sends no graph/title request; explicit Connections expansion yields one denied request and the intended message. Backend permissions are unchanged. |

### Additional observations during repair verification

- **UAT-016 — P3:** Open `/notes` as ordinary user Alice. Expected: default title behavior without an admin request. Actual: `GET /api/v1/admin/notes/title-settings` returns403 and logs an access-denied warning. The list/editor remain usable. Keep backend admin permissions; correct client capability/role handling.
- **UAT-009 security diagnosis:** The Juniper fixture contains the literal word `token`, which the existing content classifier marks confidential; `credits` raises another classification. The standalone RAG access controller defaults an unconfigured principal to guest/public-only. Keyword retrieval worked; later security filtering removed the document. Preserve this policy and its independent ACL model: do not silently promote AuthNZ roles or relax classification. The repair explains exclusions without exposing removed source IDs/text. A separate public Cedar fixture was added through the normal ingestion API (not a UI-ingestion pass) to verify retrieval, generation, excerpt mapping, and source inspection.
- **UAT-017 — P2:** Ask “When does Project Cedar launch, and who leads it? Cite the source.” with Llama.cpp / the catalog model. Actual answer gives 22 November 2026 and Mira Chen, then a prose source title; the UI correctly reports one source and zero mapped citations. Investigation confirms `enable_citations` is lost in streaming generation config and its source labels do not match `[1]` syntax. Repair the prompt contract; never infer citations from arbitrary prose.
- Live admin validation rejected a reserved `.test` email with an actionable server message; a synthetic `example.com` address succeeded. No email was sent.
- **UAT-018 — P2:** The correct cited Cedar answer shows “0% match” and “Low answer confidence.” Its wire score is an uncalibrated hybrid ranking value; the UI multiplies it by 100 and compares it to a probability threshold. Preserve raw ranking for sorting, distinguish calibrated relevance, and show relevance unavailable when no such measure exists. Do not invent confidence.
- **UAT-019 — P3:** Select Alice's existing note in `/notes`. The editor opens, but two `GET /api/v1/notes/<id>/neighbors` requests return403, missing `notes.graph.read`. An ordinary note-open should not eagerly request unavailable graph data or retry a denied capability. Preserve the permission boundary and expose the unavailable state honestly.
- Development hot reload temporarily reset browser UI references/provider selection and emitted an Ant Design unconnected-form warning. Refreshing and selecting the model again recovered. Treat the warning as unconfirmed outside hot reload; do not label it a stable product regression without reproduction.

Final verification: combined changed/new frontend regressions **488 passed across 41 suites**; combined setup/provider/RAG Python regressions **248 passed**. Bandit on all six touched Python production modules reports **zero findings**. ESLint across 92 touched frontend files reports **zero errors** and 1,086 warnings. Frontend `tsc --noEmit --incremental false` reports 90 diagnostics, exactly matching a compiler-host baseline overlay of `cb8335cf8d` (**zero additions/removals**). These pre-existing presentation/prompt/E2E errors are not a passing typecheck. The initial baseline overlay exceeded Node's 4 GiB heap; the 8 GiB rerun completed. Notes offline/denied-reconnect follow-ups pass their 21-test focused sweep and are included in the final combined frontend run.

Independent review and subsequent re-review closed the identified integration gaps: scoped handoff consumption, legacy cookie-auth handoff compatibility, normal-success milestone producers, Notes policy identity races, citation-disabled context preservation, raw candidate score preservation, and Notes offline/reconnect/cache handling. The repair pass has no remaining actionable review finding.

Validation logs retained locally: `/private/tmp/uat-repairs-vitest-final-verified.log`, `/private/tmp/uat-repairs-pytest-citation-final.log`, `/private/tmp/bandit_uat_repairs_final.json`, `/private/tmp/uat-repairs-typecheck-final-verified.log`. Test-file manifest: `/private/tmp/uat-repair-vitest-all-final.json`. Backend coverage comprises the unified first-run setup integration suite, provider registry defaults, clarification gate, streaming executor, unified pipeline, security-filter sanitizers, generation controls, and stream parity. This is targeted regression verification, not another full UAT.

Runtime observation: Next development servers briefly retained stale shared-module exports during hot reload. Restarting only the two isolated UAT frontends restored normal compilation. Both UAT backends and frontends now run current repairs; production port8000 and the user's model on9099 were untouched.

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

## Full workflow rerun — 2026-09-15 UTC

- Product revision frozen at `68863b90b7`; new issues will be recorded before further product repairs.
- New empty configuration/data profiles: `/private/tmp/tldw-onboarding-uat-full-single-20260915` and `/private/tmp/tldw-onboarding-uat-full-multi-20260915`.
- New API ports18100/18101 and WebUI ports18180/18181; separate browser sessions `full-single-20260915` and `full-multi-20260915`.
- Existing Python/frontend dependencies are reused. This run validates fresh configuration/data and user workflows, not installation onto a clean machine.
- Real local llama.cpp on9099; no mocked response qualifies as model acceptance. Credentials stay in private runtime manifests.
- The two modes run independently, following the actual named frontend tests. UI evidence, API corroboration, and any bootstrap/workaround steps are distinguished.

| Acceptance workflow | Single-user | Multi-user |
| --- | --- | --- |
| Fresh setup and real first chat | PASS with setup UX findings020/021; actual response Hello! | PASS documented operator bootstrap, UI admin sign-in and real ordinary Chat |
| Authentication, reload, logout/recovery; admin user creation where applicable | PASS key persistence, invalid-key feedback and correct-key recovery; manual-key disconnect unavailable045 | PASS admin creates Alice/Bob, password login, reload and logout/login; account metadata privacy fails034/042 |
| Public file ingest → Search → Chat/QA with cited source | PARTIAL: storage/search PASS; QA cited answer after provider workaround025; excerpt PASS, original link FAIL029; Media→Chat lands Home043, manual Open Chat recovers source and real answer | FAIL source search/QA030; citations BLOCKED despite owned public source stored; first Media handoff needs reload028 |
| Exact Wikipedia URL ingest → Search → contextual Chat | FAIL article ingestion044; search finds stored denial-page metadata, contextual Chat BLOCKED by missing article | FAIL article ingestion044; Media/QA search empty, contextual Chat BLOCKED by missing article |
| Notes → generate/save flashcards → review | Notes persistence PASS; generation422, no accepted drafts; save/review BLOCKED; raw error024 | Notes persistence PASS; grounded-word verification FAIL022; save/review BLOCKED; raw error024 |
| Create/apply prompt → Chat with payload verification | Saved data PASS, save/back navigation FAIL023; explicit apply + real pirate reply PASS after navigation workaround | Same: navigation FAIL023; saved prompt, exact system payload and real pirate reply PASS after workaround |
| Chat → save note → open linked conversation | Save201 + linked conversation PASS after persistence workaround035; saved content FAIL037, provenance label041 | Real tracked turn FAIL026, original save/link BLOCKED; partial greeting saves FAIL031 |
| Chat → save flashcards → review | Save201, reveal/rate200 and progress after reload PASS mechanically; content FAIL037/038 and initial empty-state UX039 | Real tracked turn FAIL026, save/review BLOCKED; partial greeting save FAIL031 |
| Media ingestion → analysis → review → re-analyze | Explicit first analysis, Review search/output, second analysis and reload PASS; initial Quick ingest intentionally had no analysis | Initial requested analysis FAIL027; explicit Analyze, real second analysis and Inspector reload PASS. Dedicated Review later displays that second output via visible-list selection; search remains FAIL030 |
| Media → delete → restore | Delete204 + Trash restore PASS, source and second analysis retained; confirmation/selection/date UX033/036/040 | Delete denied403 by intended role policy; capability UX FAIL033; restore BLOCKED (nothing deleted) |
| Cross-account read/write isolation | N/A | API foreign Notes/Media GET and valid versioned updates404; own originals unchanged PASS. Browser content denied PASS; Recent-note title/UUID privacy FAIL034 |

### Rerun observations and new findings

The earlier 19 findings retain their IDs. Regressions will reference those IDs; new findings start at UAT-020. A blocked or skipped step will not be counted as a pass.

- **UAT-022 — P1, open — Notes→Flashcards rejects grounded generated answers.** Alice saves the journey's biology note, then Flashcards → Import / Export → Generate with the same content, three basic cards, llama.cpp and its actual model. After about114seconds, `POST /flashcards/generate` returns422 `claim_verification_failed`. Its report labels “Mitochondria” and “Photosynthesis” hallucinations although both appear in the source. Expected: grounded cards can be reviewed and saved. Original save/review path blocked; no verification bypass applied. Evidence: `output/playwright/full-workflow-uat-2026-09-15/multi/flashcards-claim-verification-422.json`.
- **UAT-023 — P2, open — Saved prompt stays in a dirty new-prompt editor.** Create Alice's pirate prompt and Save. Toast says Prompt Added and the list gains the record, but the populated New Prompt editor remains at `/prompts?new=1`. Back to Prompts warns about unsaved changes; accepting clears the editor and reopens blank New Prompt. Expected: successful save leaves a clean saved state and Back closes it. Direct navigation to `/prompts` is a recorded workaround, not a pass for that transition.
- **UAT-024 — P2, open — Flashcard failure dumps raw verification JSON.** UAT-022 shows the entire nested report as a long error paragraph instead of a concise actionable explanation. A Next development error overlay also appears; that overlay is development-specific, while raw JSON rendering is application behavior.

Single setup completed with conservative ingest defaults, optional audio deferred, advanced RAG/storage deferred, and optional MCP tools skipped. Those optional features are not claimed as tested. First-chat request154 returned200 with `status: ready` and real response `Hello!`; manual key handoff succeeded using the isolated runtime's key. First-chat progress shows1of2complete and offers Add your first source.

Single source ingestion: Home → File label → Add source → Browse files → upload `full-single-uat-study.txt` → Configure → Quick (Extract + Chunk) → Review (Storage: Server) → Start Processing. UI reports1succeeded/0failed in3seconds and offers Media/Knowledge actions. The direct automation click on the visually hidden radio input was intercepted by its label; clicking the visible File label succeeds normally (automation adjustment, not a user-facing failure). The synthetic public Project Aster fixture replaces the journey's Wikipedia URL for this initial provenance control; URL ingestion remains a separate coverage item. Followed Search in Knowledge and asked about the opening date/coordinator with citations.

- **UAT-025 — P2, open — Fresh setup's working chat provider leaves QA's server default unusable.** After successful first chat and first source ingestion, follow Search in Knowledge with AI: Server default and ask the Aster question. The stream returns200 but terminates with provider-configuration failure; UI correctly explains how to choose an answer provider/model or configure RAG defaults. Expected: a first-use default resolves to a configured provider, or readiness indicates that QA requires configuration before offering it as ready. Explicit UI provider selection is the next control; no configuration/security bypass. Evidence: single request351 and the failed-search UI. RAG setup was deferred, which explains the separate configuration boundary but does not explain the misleading ready indication.

Single Notes→Flashcards: biology note saves and survives reload. Three basic cards using blank optional provider/model resolve to llama.cpp, but generation returns422 after approximately72seconds; save/review is blocked. Here generated answers add unsupported ATP/glucose/genetic-function details, so this run confirms failure to deliver grounded drafts and UAT-024's raw error presentation, **not** a false-positive verification diagnosis. Evidence: single request188 and `single/flashcards-claim-verification-422.json`; the multi-user exact-word rejection remains UAT-022's independent evidence. No verifier bypass or repeated random retries.

Single explicit QA control: AI → Llama.cpp automatically fills the catalog model. Retry returns the correct18December2026/MiraChen answer with `[1]`, three retrieved chunks, a cited source excerpt, and “Relevance not measured.” This passes retrieval/generation after the recorded provider-selection workaround. UAT-023 also reproduces in single-user: successful synced prompt save, false unsaved confirmation, blank editor reopens; direct `/prompts` navigation recovers the saved record.

- **UAT-026 — P1, open — Tracked Chat fails with the working local provider.** Multi-user ordinary `/chat/completions` succeeds, but tracked persisted Chat sends `/chats/{id}/complete-v2` with provider `llama`, the actual model, and `save_to_db:true`, and receives400 `Chat provider error`. One UI retry reproduces400. Original shared Chat→saveNote/Flashcards flows are blocked at real model completion; greeting-only controls cannot qualify as full passes.

- **UAT-029 — P2, open — QA original-source action opens a nonexistent frontend route for an uploaded file.** The correct Aster answer's `[1]` opens/focuses evidence and View opens the correct source excerpt (Source ID1, chunk `late_chunk:1:1`). Open original then creates a tab at `http://127.0.0.1:18180/full-single-uat-study.txt`, which shows404 Route not found. Expected: supported media/source view or original download; a plain uploaded filename must not become a WebUI route. Screenshot: `single/source-original-404.png`. The excerpt/citation controls themselves pass.

- **UAT-027 — P1, open — Ingestion reports success while saving an analysis error.** Multi-user Markdown ingestion with Standard analysis and llama reports success200 in2seconds, job `warnings:null`, yet saved Analysis contains `Error: Model is required for provider 'llama.cpp'`. Expected: actual analysis or a clearly surfaced partial failure; error text must not become a successful analysis. Explicit Analyze with a discovered model later succeeds with three correct bullets, which is a separate recovery control.
- **UAT-028 — P2, open — First Media handoff loses the new item until reload.** After multi-user successful ingestion, Open in Media navigates to `/media?id=1` but shows the first-content splash/zero results. Skip for now reveals No media found. Reload finally opens the saved source. Expected: the handoff opens the newly stored item without onboarding dismissal/reload. The single-user direct `/media?id=1` navigation after QA opens its item correctly.

Single Prompts→Chat application passes after UAT-023 navigation workaround: request271 contains the exact saved pirate system instruction and real response begins `ARRR, hello to the garden volunteers...`. Payload evidence: `single/prompt-chat-request.json`. This does not certify the broken save/back transition. Ordinary Saved Chat uses browser history (`save_to_db:false`); shared server-linked saves require the tracked-chat prerequisite.

- **UAT-032 — P1, open — Characters-list Chat action displays a new character while sending the previous conversation.** From the completed pirate prompt Chat, create Aster UAT Guide with a distinct greeting, then Characters → Chat as Aster UAT Guide. UI shows Character Chat, Aster greeting, and Character context: Included. Asking how many beds the garden has sends request551 with the old pirate system and prior pirate conversation, omitting the character instructions/greeting entirely (`/chat/completions`, `save_to_db:false`). The displayed greeting contains seven beds but the reply says it does not know. Expected: the visible character/context and conversation match the actual request. The picker path used by shared tests is a separate control.

- **UAT-033 — P2, open — Delete UI offers an action unavailable to an ordinary user.** Alice can choose Delete for her own media and sees an irreversible confirmation, then receives403 missing `media.delete`; Trash stays empty and restore is blocked. The permission boundary is intentional and unchanged. Expected: capability-aware controls/explanation before confirmation, and wording aligned with soft-delete/restore behavior.
- **UAT-034 — P1, open — Recent Notes leaks another account's title and ID.** Alice logout → Bob login in the same browser → Notes. Bob's server-backed list is empty, but Recent notes still shows Alice's study-note title and links its UUID. Clicking returns404/Failed to load note, so full content remains protected. Expected: browser recent-note history is scoped/cleared on account change. Screenshot: `multi/bob-recent-note-title-leak.png`. Foreign note/media read and update requests return404 and original records survive; these server controls pass independently of the UI metadata leak.
- **UAT-035 — P2, open — Initial Saved Chat state does not enable the expected server persistence.** Multi-user fresh Chat labeled Saved initially calls ordinary completion without server-linked save actions. Temp → bottom Ephemeral toggle back to Saved (tooltip Locally+Server) → select character through composer then calls tracked `complete-v2` with `save_to_db:true`. Expected: visible persistence state and selected character reliably determine transport without a toggle workaround. Single-user likewise initially shows Saved while requests use `save_to_db:false`; local-history persistence itself is not claimed to fail.

- **UAT-036 — P2, open — Deleted Media remains selected as a broken inspector.** Single-user Delete succeeds204 and toast offers Moved to trash/Undo, while list shows No media found. The detail pane still says Showing full-single-uat-study, exposes Analyze/Delete controls, and reports zero words/no content/no analysis. It sends repeated404 requests for the deleted item/progress/navigation. Expected: clear the active item or show its explicit Trash state. The confirmation also says deletion cannot be undone despite the Undo/Trash restore path (shared UAT-033 wording issue).

Single Media analysis control: real generation produces `ASTER_ANALYSIS_ONE`; Review search Aster finds the item and displays that exact saved output. Second generation produces `ASTER_ANALYSIS_TWO`, which survives full Inspector reload. Inspector content search Aster also returns the source; multi-user UAT-030 is not reproduced here.

- **UAT-021 — P3, open — Local model discovery requires already knowing a model name.** Select llama.cpp, enter a valid base URL, leave Default model empty, then Validate. The UI stops with “Default model is required before validation.” Only after manually consulting the provider's `/v1/models` and entering its exact ID does validation expose Discovered models. Expected: discover available models from the endpoint before requiring a selection. Manual catalog lookup unblocks setup, and the opaque local model ID now validates/saves successfully (UAT-001 regression check passed).

- **UAT-020 — P3, open — Setup readiness has unexplained warnings.** On an empty single-user profile, before choosing a provider, Chat and Embeddings/RAG display “ready with warnings” followed by “the server did not include the warning details.” Both readiness calls return200. `/api/v1/setup/readiness/profiles` explicitly returns `ready_with_warnings` with empty `warnings`, `blockers`, and `consequences` for those lanes (Chat selection is default OpenAI). Expected: readiness reflects configuration/verification and names any actionable warning. Speech does include an explanatory warning, providing a positive control. Evidence: browser requests115/116, initial setup snapshots in `full-single-20260915`. Does not block continuing setup.

- **UAT-030 — P1, open — Ordinary-user search cannot retrieve its ingested public source.** Alice's Media full-text query `Cedar` sends `{query:"Cedar",fields:["title","content"],sort_by:"relevance"}` and returns200 with `items:[]`, `total:0`, despite the visible owned source containing Project Cedar. An independent normally authenticated API request corroborates it. Knowledge QA lists and selects media1, but explicit Llama.cpp/model returns an empty-context no-results stream, without a security-exclusion count/explanation. Expected: retrieve the user's searchable public source, or explain an intentional restriction. Citation inspection is blocked; no ACL bypass was applied. Evidence: `multi/qa-selected-source-stream.xndjson`, `multi/qa-no-results-selected-source.png`, and the multi-user report. Single-user Aster content search and explicit QA pass.
- **UAT-031 — P2, open — Failed tracked-chat retry leaves greeting save identifiers inconsistent.** After UAT-026's Retry creates a different conversation, the greeting still offers Save to Notes/Flashcards. Both calls return400 `Message is not in conversation`; no artifact is created. Expected: actions refer to a message belonging to the active conversation, or remain unavailable. This is a partial diagnostic control, not a substitute for the blocked real-answer workflow. Detailed IDs and sequence are in the multi-user report.
- **UAT-037 — P2, open — Chat saves raw reasoning into Notes and Flashcards.** Single-user real tracked response visibly separates optional collapsed model reasoning from “The garden has seven raised beds.” Save to Notes and Save to Flashcards each return201, but derived content includes the full literal `<think>…</think>` block. Expected: save the visible answer by default, with reasoning included only by an explicit choice. The note opens its correct linked conversation, independently of this content defect. Evidence: `single/chat-derived-card.json`, `single/chat-card-raw-reasoning-empty-back.png`.
- **UAT-038 — P2, open — Save to Flashcards creates an unusable blank-backed card.** The same direct Chat save creates a Standard card with the assistant's answer on Front and an empty Back. Review all due → Show answer displays an empty answer area. Expected: a usable question/answer pair, or an editor requiring completion before accepting the card. A Good rating returns200, schedules the card ten minutes later, and persists one reviewed card after reload; those scheduling controls pass without certifying content quality. Evidence: `single/chat-derived-card.json`, `single/chat-card-review-response.json`, and the card screenshot.
- **UAT-039 — P3, open — Study initially claims no cards while one is available.** First visit to Flashcards after the save shows “1 cards remaining / Available now:1 / new:1” and Review all due alongside “You're all caught up! No cards are due for review.” Clicking Review all due immediately presents the card. Expected: prompt the user to start the available queue rather than show a completion message before a session. Evidence: full browser snapshot returned to the UAT session at approximately04:43:41UTC, before clicking Review all due. That settled snapshot was not separately saved; the initial navigation snapshot was captured before its data loaded. Subsequent saved card/review artifacts corroborate availability, not the initial contradictory text.
- **UAT-040 — P3, open — Trash lacks the just-deleted item's date.** Immediately after successful single-user deletion, Trash lists the source with “Deleted date unavailable.” Expected: show its deletion date if the UI promises that metadata. Restore succeeds and retains source content and `ASTER_ANALYSIS_TWO`; no data-loss claim. Root cause not diagnosed.
- **UAT-041 — P3, open — Chat-derived note reports manual origin.** The saved note editor correctly shows Linked to conversation and its message ID, but its footer says “Origin: Typed manually.” Expected: provenance labels agree with the actual Chat save. Observed note `cd50f1fe-26ef-42b2-99d0-8772c5dd8dab`, version1; no manual note editing occurred.
- **UAT-042 — P3, open — Document title retains previous chat context after navigation/logout.** Single-user Chat → Notes leaves the browser title “How many raised beds does the garden have?” while Notes is displayed. Multi-user logout redirects the previous chat tab to login but leaves its old chat title. Expected: document title follows current route/auth state and clears prior-account metadata. Multi-user recent-note list leakage remains separately tracked as UAT-034; no additional content-access claim.
- **UAT-043 — P2, open — Media's Chat action lands on Companion Home.** Restored single-user Media → Chat with this media navigates to `/` and displays Companion Home, without a chat composer. Clicking Home's Open Chat manually navigates to `/chat` and restores the intended source text in the composer. Expected: the media action opens Chat directly. This differs from UAT-013's repaired Home starter-question producer; that specific producer is not being claimed regressed by this observation.

Single shared save results: tracked `complete-v2` request731 returns200 with the real seven-beds answer after the persistence-toggle workaround. Save to Notes request944 returns201 (note `cd50f1fe-26ef-42b2-99d0-8772c5dd8dab`); Save to Flashcards request982 returns201 (card `9d6cb1d0-f54f-49d5-9505-dafd546c97d4` and supporting note). Notes → More actions → Open conversation restores the correct greeting, question, and actual answer. Both artifact contents fail UAT-037; card answer fails UAT-038. Flashcard review request202 returns200, and reload retains Reviewed today1 / completed session / next review. Browser request numbers belong to the relevant page capture, not a global run-wide sequence.

- **UAT-044 — P2, open — URL ingestion reports success for a remote denial page.** In both modes, use the exact journey URL `https://en.wikipedia.org/wiki/Playwright_(software)` with analysis/chunking disabled as in its helper. `POST /media/process-web-scraping` returns200 and the UI reports Succeeded1/Failed0. Open in Media shows media2 titled N/A, containing Wikimedia's robot-policy denial text instead of the article. Expected: surface the retrieval denial/invalid article, rather than report a useful article ingestion. The remote site's refusal itself is an external limitation; no access-control workaround was attempted. Single-user Playwright search finds this stored denial page through URL metadata, whereas multi-user Media and Knowledge searches return0; neither result certifies article retrieval. Actual article-grounded Chat is BLOCKED. Evidence: both modes' `wikipedia-*` artifacts.
- **UAT-045 — P2, open — Manual single-user API-key settings lack a disconnect action.** Settings → tldw Server exposes the remembered key, Save and Test Connection. Help says it is stored “until you disconnect or clear browser data,” but there is no disconnect/logout control for this mode. Source inspection confirms Logout is rendered for single-user cookie sessions and logged-in multi-user sessions, not manual keys. Expected: an explicit way to forget the active manual key, consistent with the help text. This blocks the requested UI disconnect/re-entry check. Invalid-key Test Connection correctly explains HTTP401; restoring the real key and Save succeeds, then reload retains it. Browser-data clearing/mode-switching is not counted as the missing disconnect action.

### Rerun conclusion and evidence

All named workflow steps now have an observed outcome or an explicit dependency block. The run completed; product acceptance did not pass. The highest-priority new work is the account metadata leak034, ordinary-user source retrieval030, multi-user tracked completion026, grounded flashcard verification022, hidden analysis failure027, and mismatched character context032. Remaining P2/P3 findings are recorded above rather than silently deferred.

Source-grounded single-user Chat recovery: after UAT-043's manual Open Chat, the source text was present in the composer. Appending a question about volunteers produced the real answer “Volunteers meet every Saturday at09:30.” The tracked request570 contained that source text and returned200. This is a direct supplied-context Chat control; the separate Knowledge QA control provides citation/provenance evidence. It does not establish a passing automatic RAG handoff.

Detailed multi-user steps and request evidence: [multi-user report](../../output/playwright/full-workflow-uat-2026-09-15/multi/multi-uat-results.md). Key single-user evidence: [cited Aster answer](../../output/playwright/full-workflow-uat-2026-09-15/single/aster-cited-answer.png), [linked conversation restored](../../output/playwright/full-workflow-uat-2026-09-15/single/linked-conversation-restored.png), [invalid saved card content](../../output/playwright/full-workflow-uat-2026-09-15/single/chat-card-raw-reasoning-empty-back.png), [restored Media analysis](../../output/playwright/full-workflow-uat-2026-09-15/single/restored-media-analysis.png), [Wikipedia denial stored](../../output/playwright/full-workflow-uat-2026-09-15/single/wikipedia-robot-response.png).

Coverage limits: these are fresh configuration/data/browser profiles using existing dependencies and Next development servers. Clean-machine dependency installation, Docker/Postgres, optional STT/TTS/MCP tools, Evals, Watchlists, and browser extensions were not certified. The exact Wikipedia URL was attempted in both modes and failed to yield article content; public synthetic fixtures supplied the independent source controls. Fresh-run multi-user confidential-content RAG exclusion verification is blocked because even its public positive control returns no contexts; API cross-account denials do not re-certify that RAG policy. No literal A/B/C mapping was invented. No product fixes, role changes, or ACL bypasses occurred during this frozen rerun. Original UAT-001–019 targeted repair verification remains distinct from these workflow results.

Execution plan closure: fresh setup, source-to-answer attempts, knowledge reuse/recovery attempts, and evidence reconciliation were completed. Failures and blocked acceptance remain open in this tracker; completion here describes execution of the UAT plan, not successful product acceptance.

Evidence review: checked all48 retained evidence files (plus the generated manifest), parsed every JSON artifact, compared text with generated runtime credentials and token patterns, verified finding IDs/counts and no pending rerun-matrix cells, and ran `git diff --check`. All checks pass. Independent review caught the missing dedicated multi-user Review check; it was then completed by selecting Cedar in `/media-multi`, with screenshot/snapshot/API evidence retained. This check occurred after the second analysis; no claim is made that the original failed ingest-analysis path passed in order. The expired single-user422 request-index capture was recovered from its browser request-warning diagnostic and its provenance is documented in the evidence README. One observed study empty-state message lacks a standalone saved snapshot, as UAT-039 explicitly states. The only repository changes in this rerun are the tracker, evidence and Backlog records, so new application tests/Bandit are not applicable.

Additional URL observation: both Media inspectors show “Chunking: Completed” for the Wikipedia denial item despite the submitted `perform_chunking:false`. The processing-state origin was not independently diagnosed; retain this discrepancy with UAT-044 for investigation rather than infer a separate confirmed root cause.
