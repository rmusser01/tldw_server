# Research Workspace Live UAT Matrix

Date: 2026-05-25

This matrix is the current acceptance ledger for the Research Workspace remediation
stream. It is intentionally tied to live backend/WebUI validation, not only code
inspection. Rows owned by completed child tasks retain their recorded live evidence;
rows marked `Current run` were rechecked during TASK-478.13.
TASK-478.18 refreshed the migration row with TASK-515/TASK-516 live true-move
evidence. TASK-478.25 then closed the guided recovery/import/export validation
gap with a live backend plus WebUI CDP walkthrough.
TASK-478.30 closed the real embeddings-backed vector completion validation gap
with a live backend, Redis embeddings worker, WebUI CDP source-status check, and
selected-source RAG request.
TASK-478.31 restored the shared UI TypeScript gate that covers Research
Workspace source, components, stores, and route tests; the broader WebUI
E2E-inclusive typecheck still has unrelated route/admin/agent fixture blockers.
TASK-12020.11 re-ran final remediation certification on 2026-06-25/26 with a
live backend and in-app CDP browser. It confirmed beginner/no-key entry,
tour, mobile layout, and add-source auth recovery, but kept final certification
open as `Partial`/`Blocked` for authenticated power-user workflows because the
current settings credential check can report healthy while Research Workspace
still receives 401s. Standalone Playwright was also environment-blocked in the
macOS Codex sandbox before any product assertion executed.
TASK-12020.13 restored discoverable local workspace search for the beginner
no-key state by adding a Research Workspace header search action, wiring
Cmd/Ctrl+K to the workspace-owned search modal, and preserving the app command
palette route block on `/research-workspace`. Live browser certification still
depends on the TASK-12020.14 browser-runner follow-up.
TASK-12020.14 added a focused Research Workspace final UAT runner that defaults
to localhost-safe WebUI binding, writes a JSON evidence artifact, and classifies
standalone Playwright outcomes as `passed`, `product_failed`, or
`environment_blocked` so skipped/blocked browser launches cannot be counted as
product passes.
The first runner execution on 2026-06-26 reached the localhost-bound WebUI and
attempted all 25 configured Chromium tests, but the browser failed before page
code executed with macOS `MachPortRendezvousServer` permission denial. The
runner wrote `test-results/research-workspace-final-uat-evidence.json` and
classified the run as `environment_blocked` with reasons
`macos_mach_port_denied` and `browser_launch_failed`.
TASK-12130 re-ran issue #2605 certification on 2026-07-04 against the full live
FastAPI backend on `http://127.0.0.1:8000`, the full Next.js WebUI quickstart
proxy on `http://127.0.0.1:8080`, and a local llama.cpp endpoint on
`http://127.0.0.1:9099/v1` exposing
`gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`. Standalone Chromium
launched, loaded the real application, and executed all 25 configured tests.
After stabilizing the final Flashcards manage-page assertion to reopen the
moved general-scope deck directly, product failures were zero: 24 passed, 1
environment skip, 0 flaky, and 0 unexpected failures. The runner still exited
75 with `status=environment_blocked` because the live environment skipped the
sandbox-run API path (`POST /api/v1/sandbox/runs` 404). Evidence:
`apps/tldw-frontend/test-results/research-workspace-final-uat-evidence-2026-07-04-llamacpp9099-rerun.json`
and `apps/tldw-frontend/test-results/research-workspace-final-uat-report-2026-07-04-llamacpp9099-rerun.json`.
TASK-12020.16 then used the in-app browser/CDP fallback against
`http://127.0.0.1:8080` and a live backend on `http://127.0.0.1:8000` to
confirm that invalid-auth workspace context failures stay scoped to Workspace
degraded UI, that restoring a valid single-user API key recovers to
`Server context ready`/`Connected`, and that workspace reconciliation failures
no longer open the global `Can't reach your tldw server` modal.
TASK-12020.17 then closed the remaining stale-modal recovery blocker found in
the controlled power-user pass: a transient `GET /api/v1/chat/commands`
status-0 failure no longer promotes an optional Research Workspace bootstrap
request into the global backend-unreachable modal after Settings and workspace
context recover.
TASK-12020.18 then closed a power-user source-organization recovery defect found
during the same controlled CDP pass: `Clear search and filters` now exits an
empty active folder and restores visible source rows instead of leaving users in
a no-match state while a source remains selected for chat.
TASK-12020.19 then scoped optional audio voice bootstrap failures out of the
global backend-unreachable modal after the grounded-chat follow-up showed
`GET /api/v1/audio/voices/catalog?provider=kitten_tts` and
`GET /api/v1/audio/voices` status-0 warnings alongside a workspace that was
otherwise connected.
TASK-12020.20 then scoped optional ingestion-source capability bootstrap
failures out of the global modal after direct RAG probes succeeded for the
selected source, but the browser path opened `Can't reach your tldw server`
from `GET /api/v1/ingestion-sources/capabilities` status-0. The same controlled
workspace then produced a grounded answer with the selected evidence phrase and
no global modal.
TASK-12020.21 then covered the next Studio gap found in the controlled
power-user pass: a stale legacy `tldw:gemma3:1b` model selection was still
enabled for Studio generation, sent `POST /api/v1/chat/completions`, and failed
with `no_provider_configured`. Studio now blocks stale `tldw:` selections that
are absent from the current chat model catalog, keeps configured
provider-qualified models available, and suppresses degraded "try it" capability
copy while a prerequisite already blocks generation.
TASK-12020.24 then isolated the remaining configured-provider Studio generation
gap to provider/runtime setup rather than a completed Studio path: direct chat
probes found the advertised `ollama/gemma3:1b` model accepted by
`/api/v1/chat/completions`, but backend egress rejected the configured
`192.168.2.216:11434` endpoint with `Port not allowed: 11434`, and direct
network probes could not reach that Ollama host from the UAT session.
TASK-12020.29 then closed the product feedback gap around that provider/runtime
blocker. `/api/v1/llm/providers` and `/api/v1/llm/models/metadata` now surface
readiness metadata for egress-blocked endpoints, unreachable local endpoints,
missing external credentials, and catalog aliases that are not valid chat
providers. The shared frontend model pipeline preserves those fields, filters
unavailable models out of selectable chat models, and Studio blocks a saved
unavailable model with the backend readiness message before creating an
artifact.
TASK-12020.22 then closed the first destructive-action defect found while
recertifying share/export/import: Share Workspace > Active Shares removed a
revoked share token and showed success feedback, but left the confirmation
dialog visible. Successful team/org share and share-link revocations now call
the confirmation close callback only after the mutation succeeds; failures keep
the confirmation open with error feedback.
TASK-12020.23 then closed the paste-source-to-bulk-actions blocker found during
the final power-user pass. Add Sources > Paste now dismisses promptly after the
source is added, best-effort workspace tagging and processing-status polling do
not raise the global backend-unreachable modal, and the controlled CDP pass
verified Select all, bulk Remove confirmation, and Undo recovery for disposable
sources.
TASK-12968 completed issue #2606 beginner/no-key certification on 2026-07-14
against an isolated live FastAPI backend, advanced-mode Next.js WebUI, and a
fresh Chrome 145 profile controlled exclusively through CDP. All 17 explicit
desktop/mobile checkpoints passed, including readiness, all three empty-state
surfaces, the complete first-run tour and replay, visible and keyboard search,
preserved-input Add URL auth recovery, 390x844 layout, and global modal/runtime
overlay suppression. The run also fixed false legacy migration caused by an
empty hydration write in either split or monolithic storage mode, scoped the
fresh-initialization migration exemption to the exact workspace ID, restored the
shared tour runner under hidden global chrome, replaced persistent tour feedback
with a transient message, and reduced the mobile header without removing core
workspace controls. The final machine-readable manifest records one active and
saved workspace at both direct entry and final state, no visible migration
messages, and zero migration API requests.
TASK-12020.27 then inventoried the remaining destructive/recovery workflows
outside source bulk remove and share revoke. Live CDP confirmed archive and
artifact-delete cancel confirmations, but archive success and artifact-delete
success both removed or switched away from affected state while rendering toast
text without a visible `Undo` control. The shared recovery-control defect is
split to TASK-12020.31 and blocks clean destructive-action certification.
TASK-12020.31 then fixed the source-level issue by moving destructive Undo
actions from Ant Design's ignored `message.open` `btn` field into rendered
message content, with regressions that render the toast content and require an
accessible `Undo` button. A clean temporary webpack WebUI on
`127.0.0.1:8083` loaded the current bundle and live-CDP confirmed archive
success renders a visible `Undo` and restores the archived duplicate. After
local network permission was restored, a shell-launched standalone Chromium also
attached a disposable `.workspace.json` export through Import Workspace,
rendered the imported failed artifact, deleted it, showed one visible `Undo`,
and restored the failed artifact.
TASK-12020.27 then resumed on the current clean bundle and live-confirmed the
remaining chat/note/source-organization recovery paths: chat clear/Undo, message
delete/Undo, Quick Notes clear/Undo, per-source remove/Undo, and source
move-transfer/Undo all restored disposable state. Evidence is recorded in
RW-UAT-030.
The same pass split two remaining product defects: TASK-12020.32 for imported
workspace chat sessions being wiped on first render after a valid Import
Workspace flow, and TASK-12020.33 for selected-source batch `Remove (n)`
scheduling an undo action without applying the removal, leaving the enabled
button inert.
TASK-12020.32 then added a RED/GREEN ChatPane regression proving imported
workspace chat messages load before any empty local autosave can overwrite the
session, and guarded autosave until the active workspace session has been
reconciled. TASK-12020.33 added a click-through selected-source batch Remove
regression proving the current source calls `removeSources(["s1", "s2"])` and
Undo restores direct selection plus folder memberships. A fresh live browser
recheck was attempted from a disposable temp WebUI on `127.0.0.1:8084`, but
Chromium remains environment-blocked by macOS Mach-port denial and direct
Chrome-for-Testing crashpad startup failure.
TASK-12020.26 then certified the owner-side broader team/org sharing contract:
the live backend on `127.0.0.1:18061` accepted an isolated workspace upsert,
rejected invalid team and org share targets with scoped HTTP 403 messages, and
left the active share list empty after those failed attempts. A headless
browser pass against the live WebUI on `127.0.0.1:3000` opened Share Workspace
and, with a seeded active team share in the isolated AuthNZ DB, changed
`Team #7` access from `view_chat` to `view_chat_add` and enabled cloning; both
PATCH requests returned HTTP 200 and the DB row became `view_chat_add|1`.
Screenshots: `/tmp/tldw_task12020_26_share_dialog.png` and
`/tmp/tldw_task12020_26_active_share_updated.png`. Remaining share coverage is
recipient-side behavior with real team/org membership fixtures, not owner-side
target validation or active-share update controls.

## TASK-12020.11 Final Recheck Notes

- Setup: Next.js WebUI served from `http://127.0.0.1:8080` with
  `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`; backend health returned HTTP 200
  in `single_user` mode. The WebUI required explicit localhost binding after
  the default `0.0.0.0:8080` bind failed with `EPERM`.
- Standalone Playwright: `bunx playwright test
  e2e/workflows/research-workspace.spec.ts --project=chromium --reporter=line
  --workers=1` could start the WebUI after the bind override, but every test
  failed before page execution because Chromium headless shell crashed with
  `bootstrap_check_in ... Permission denied (1100)`. This is recorded as QA
  environment-blocked, not a product regression.
- Repeatable runner: `bun run e2e:research-workspace:uat` was run on
  2026-06-26 with the backend already healthy at `http://127.0.0.1:8000` and
  WebUI autostart set to `bun run dev -- -H 127.0.0.1 -p 8080`. It attempted
  both configured Research Workspace workflow specs, executed 25 Chromium test
  cases, and exited 75 with status `environment_blocked`, reasons
  `macos_mach_port_denied` and `browser_launch_failed`, and report artifacts
  `apps/tldw-frontend/test-results/research-workspace-final-uat-evidence.json`
  plus `apps/tldw-frontend/test-results/research-workspace-final-uat-report.json`.
  The WebUI server also emitted repeated Watchpack `EMFILE: too many open files,
  watch` warnings, which should be treated as local QA environment pressure
  until reproduced outside this sandbox.
- Beginner/no-key CDP state: fresh `http://localhost:8080/research-workspace`
  showed the readiness gate, then loaded Research Workspace with no selected
  model, disconnected/degraded workspace status, explanatory empty states,
  skip links, the Sources/Chat/Studio mental model, and API-key recovery copy.
  Route timing note: the initial DOM became inspectable immediately after
  navigation, the readiness gate resolved to the workspace after a 3s CDP wait,
  and the mobile reload settled after a 2.5s CDP wait.
  Screenshots:
  `/private/tmp/task12020_11_beginner_entry.png`,
  `/private/tmp/task12020_11_beginner_tour.png`,
  `/private/tmp/task12020_11_beginner_missing_key_add_url.png`, and
  `/private/tmp/task12020_11_beginner_mobile.png`.
- Beginner auth recovery: Add Sources > URL preserved the entered URL
  `https://example.com/research-workspace-uat-beginner` after a 401 and showed
  `You do not have permission to add this source. Check your session and retry.`
  inside the modal. Console/network diagnostics repeatedly logged missing-key
  and 401 warnings such as `GET /api/v1/workspaces/.../context 401 Add or
  update your API key in Settings -> tldw server, then try again.`
- Beginner search gap: on the no-key desktop state, the global `Search Cmd+K`
  app-shell affordance was absent and manual Ctrl/Cmd+K attempts in CDP did not
  open the workspace search modal. TASK-12020.13 now covers the remediation in
  focused UI tests, with live browser recheck still pending TASK-12020.14.
- Power/API-key blocker: Settings > tldw Server accepted the local E2E
  placeholder key enough to show `Server responded successfully. You can
  continue.`, `Core: reachable`, and `RAG: healthy`, but returning to
  `/research-workspace` still produced 401 `Invalid or missing API Key`
  responses for workspace/model/storage endpoints plus the `Can't reach your
  tldw server` dialog and a Next.js runtime overlay for
  `GET /api/v1/llm/models/metadata`. Route timing note: the failed workspace
  state was visible after a 4s CDP wait following navigation. Screenshots:
  `/private/tmp/task12020_11_power_auth_setup_e2e_key.png` and
  `/private/tmp/task12020_11_power_workspace_entry.png`. Follow-up:
  TASK-12020.12.
- Controlled power-user recovery follow-up: TASK-12020.17 used a backend on
  `http://127.0.0.1:8001` with
  `SINGLE_USER_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY` and WebUI
  `http://127.0.0.1:8081`. Settings > tldw Server > Recheck showed
  `Server responded successfully. You can continue.`, `Core: reachable`, and
  `RAG: healthy` with no backend-unreachable dialog
  (`/private/tmp/task12020_17_cdp_settings_recheck_healthy.png`). Research
  Workspace was rechecked after restarting the WebUI so the shared-package
  proxy fix was in the served bundle; it then showed `Server context ready`,
  `Connected`, and no stale `Can't reach your tldw server` modal after the
  earlier transient `GET /api/v1/chat/commands` status-0 warning
  (`/private/tmp/task12020_17_cdp_workspace_no_stale_modal_after_restart.png`).
- Controlled power-user source workflow follow-up: TASK-12020.11 then added a
  pasted source `TASK-12020.11 Power Source Alpha` and confirmed the source row,
  text-searchable/ready status, selection, source-status details, preview with
  captured text, and local annotation creation. Screenshots:
  `/private/tmp/task12020_11_final_power_paste_source_result.png`,
  `/private/tmp/task12020_11_final_power_source_selected.png`,
  `/private/tmp/task12020_11_final_power_source_status_details.png`,
  `/private/tmp/task12020_11_final_power_preview_selected.png`, and
  `/private/tmp/task12020_11_final_power_annotation_added.png`. The same pass
  found that `Clear search and filters` did not exit an empty selected folder;
  TASK-12020.18 fixed and rechecked that recovery path. Authoritative evidence:
  `/private/tmp/task12020_18_cdp_clear_folder_filter_recovered.png`.
- Controlled power-user grounded-chat follow-up: with source
  `TASK-12020.11 Power Source Alpha` selected and the chat model set to
  `Custom / tldw:gemma3:1b`, the first prompt `What exact evidence phrase does
  the selected source contain? Cite the source title.` produced an inline
  recoverable response instead of an answer:
  `I couldn't retrieve evidence from the selected sources... Details: Cannot
  reach server.` Browser diagnostics for that attempt included
  `GET /api/v1/audio/voices/catalog?provider=kitten_tts 0 Failed to fetch`,
  `POST /api/v1/rag/search 0 Failed to fetch`, `GET /api/v1/audio/voices 0
  Failed to fetch`, and `GET /api/v1/ingestion-sources/capabilities 0 Failed
  to fetch`. The original pre-fix screenshot is
  `/private/tmp/task12020_11_final_power_grounded_chat_result.png`.
  TASK-12020.19 then kept the optional audio voice bootstrap failures scoped out
  of the global backend-unreachable modal; after restarting the WebUI, the same
  authenticated workspace showed `Server context ready`, `Connected`, selected
  source state retained, and no global modal
  (`/private/tmp/task12020_19_cdp_audio_voice_no_global_modal.png`). Direct
  probes to `POST /api/v1/rag/search` on the controlled backend returned HTTP
  200 with Media ID 1, source title `TASK-12020.11 Power Source Alpha`, and the
  selected evidence phrase, so the remaining browser interruption was not a
  selected-source data/RAG availability failure. The next browser retry opened a
  global modal from optional `GET /api/v1/ingestion-sources/capabilities` status
  0 while the endpoint returned HTTP 200 directly; evidence:
  `/private/tmp/task12020_11_final_power_grounded_chat_retry_after_task19.png`.
  TASK-12020.20 now keeps that optional capability bootstrap failure scoped out
  of the global modal. After restarting the WebUI, the same controlled browser
  pass produced an answer containing `TASK-12020.11 evidence alpha. The research
  workspace paste source should become selectable and citation-ready for a
  controlled final UAT pass.`, cited `TASK-12020.11 Power Source Alpha`, and
  showed no global backend-unreachable dialog. Evidence:
  `/private/tmp/task12020_20_cdp_grounded_chat_success_no_capabilities_modal.png`.
- Controlled power-user Studio follow-up: the next visible workflow attempt
  clicked Studio `Summary` with the selected source still active and the stale
  `Custom / tldw:gemma3:1b` model selected. The browser created a failed
  artifact with `no_provider_configured (POST /api/v1/chat/completions)` while
  the backend model catalog showed the configured local model as
  `ollama/gemma3:1b`, not `tldw:gemma3:1b`. Direct backend probes confirmed
  `tldw:gemma3:1b` returned HTTP 503 `no_provider_configured`; a direct
  `api_provider=ollama`, `model=gemma3:1b` request returned HTTP 500 in this
  environment, so live Studio generation remains uncertified. Evidence:
  `/private/tmp/task12020_11_studio_summary_no_provider_configured.png`.
  TASK-12020.21 now blocks stale `tldw:` Studio selections before generation.
  Post-fix CDP DOM trace on `http://127.0.0.1:8081/research-workspace` showed
  the warning `The selected Studio model is no longer available. Choose a
  configured model in Studio Options before generating outputs.`, `Summary`
  disabled, no Studio degraded "You can still try it" capability warning, no
  backend-unreachable modal, and `Server context ready`/`Connected`.
  TASK-12020.24 rechecked the configured-provider path directly on 2026-06-26:
  `/api/v1/llm/providers` advertised `ollama/gemma3:1b` as configured, but
  minimal `api_provider=ollama`, `model=gemma3:1b` chat completion still
  returned HTTP 500. Backend trace `debb6c420f3d0010109a0f63c4f576e3`
  identified the root cause as `EgressPolicyError: Port not allowed: 11434`
  while contacting `http://192.168.2.216:11434/v1`; direct network probes to
  both that host and local `127.0.0.1:11434` failed. `mlx` and
  `custom_openai_api` catalog entries are not accepted chat provider IDs, and
  the accepted `custom-openai-api` alias returned HTTP 401. Treat Studio
  generation as provider/configuration-blocked in this environment, not
  certified. TASK-12020.29 now covers the product-side affordance by surfacing
  those provider blockers through readiness metadata and Studio prerequisite
  copy before artifact creation.
- Controlled power-user share/destructive follow-up: Share Workspace opened from
  the connected workspace, the Team/Org tab blocked submit with `Enter a team or
  organization ID before sharing.`, the Share Link tab created a full read-only
  link and showed a copy affordance, and Active Shares listed the generated
  token with access, clone, usage, password, expiry, and revoke controls. The
  pre-fix revoke path removed token `AU1H7sTg...` and showed `Share link
  revoked`, but left the confirmation dialog visible indefinitely. TASK-12020.22
  now explicitly closes revoke confirmations after successful team/org and token
  mutations while keeping them open on errors. Post-fix CDP generated and
  revoked token prefix `tdvf83sx`: it was visible before revoke, the confirmation
  opened, the token disappeared from Active Shares after success, the
  confirmation closed, no backend-unreachable modal appeared, and the workspace
  remained `Server context ready`/`Connected`.
- Controlled power-user export/import follow-up: Workspace settings exposed
  `Export Workspace`, `Export Citations (BibTeX)`, and `Import Workspace`.
  `Export Workspace` produced the success toast `Workspace exported:
  new-research-2026-06-26t18-15-54-391z.workspace.zip` and no
  backend-unreachable modal. `Import Workspace` opened a visible modal with
  helper text and a file input accepting `.json,.workspace.json,.zip,.workspace.zip`.
  TASK-12020.25 created the disposable supported bundle
  `/tmp/research-workspace-import-TASK-12020-25.workspace.json`, loaded
  `/research-workspace` through the in-app browser at `127.0.0.1:8081`,
  confirmed the visible import dialog/input contract, and saved evidence at
  `/private/tmp/task12020_25_cdp_import_dialog.png`. Full live import remains
  uncertified because the in-app browser file-input locator exposes no
  `setInputFiles`, upload, dispatch, or mutable evaluate method, and a separate
  local Playwright runner failed before page execution with macOS Chromium
  `bootstrap_check_in ... Permission denied (1100)`. Existing focused UI/store
  tests still cover accepted/rejected import files and imported workspace/source
  state, but the matrix must keep import open until a browser surface can attach
  a real file.
- Controlled power-user bulk-actions follow-up: TASK-12020.23 reproduced the
  post-paste blocker where the source appeared but an Ant modal wrapper, then
  later best-effort tagging/status-poll failures, kept blocking bulk controls.
  After the fix, a clean in-app CDP run against backend
  `http://127.0.0.1:8001` and WebUI `http://127.0.0.1:8081` added
  `TASK-12020.23 Final 2026-06-26T18-51-09-294Z`, saw three ready disposable
  sources, no Add Sources dialog, no global `Can't reach your tldw server`
  dialog, and no runtime overlay. DOM hit-testing over `Select all` returned
  the checkbox input as the top element, not a modal wrapper. Clicking it showed
  `3 selected`, grounded-chat selection state, `Move / Copy`, `Preview
  selected`, and `Remove (3)`. Confirming `Remove (3)` removed all three
  disposable sources and showed `3 sources removed` with `Undo`; clicking Undo
  restored all three sources with no backend modal. Evidence:
  `/private/tmp/task12020_23_cdp_bulk_selection_recovered.png`.
- Controlled power-user destructive/recovery follow-up: TASK-12020.27
  inventoried workspace archive/delete/restore, workspace collection delete,
  banner reset, chat clear/message delete, note clear, artifact delete, source
  transfer undo, and the already-covered source bulk remove/share revoke paths.
  It then used disposable workspace/artifact state in the in-app browser. Archive
  cancel left `New Research (Copy)` active and closed the confirmation
  (`/private/tmp/task12020_27_archive_cancel_dialog.png`). Confirming Archive on
  disposable duplicates switched back to `New Research`, showed
  `Workspace archived.`, and rendered no visible `Undo`
  (`/private/tmp/task12020_27_archive_success_missing_undo.png`). Artifact
  delete cancel kept the failed output in place
  (`/private/tmp/task12020_27_artifact_delete_cancel_dialog.png`). Confirming
  artifact delete removed the failed output and showed `Output deleted.`, but no
  visible `Undo` control appeared
  (`/private/tmp/task12020_27_artifact_delete_missing_undo.png`). TASK-12020.31
  now owns the remaining live recertification after fixing the current source
  path with rendered-content regressions. A post-fix live archive recheck against
  the locked `127.0.0.1:8081` WebUI was not accepted as product evidence because
  it still served a stale compiled `WorkspaceHeader` chunk older than the source
  change and reproduced the old missing-Undo toast
  (`/private/tmp/task12020_31_archive_stale_missing_undo.png`). A clean
  temporary webpack server on `127.0.0.1:8083` then loaded the current bundle:
  archiving a disposable duplicate showed `Workspace archived.` with a visible
  `Undo`, and clicking `Undo` restored `New Research (Copy)` with
  `Workspace restored` feedback. Evidence:
  `/private/tmp/task12020_31_archive_undo_visible.png` and
  `/private/tmp/task12020_31_archive_undo_restored.png`. After restoring local
  network permission, a shell-launched standalone Chromium imported an attached
  disposable `.workspace.json` bundle containing one failed artifact, confirmed
  `Failed output` rendered, deleted it, verified `Output deleted.` with a
  visible `Undo`, clicked `Undo`, and verified `Output restored` plus the
  restored `Failed output` card. Evidence:
  `/private/tmp/task12020_31_failed_artifact_imported.png`,
  `/private/tmp/task12020_31_failed_artifact_deleted_undo_visible.png`, and
  `/private/tmp/task12020_31_failed_artifact_restored.png`. Console/network
  observations during this disconnected clean-origin run included expected
  missing-API-key 401 warnings for workspace/notes/migration endpoints and CORS
  failures for notification polling, but those did not block local import or
  artifact undo recovery.

## Validation Rules

- Behavior claims require a live backend plus WebUI run using Playwright/CDP.
- Static code review can justify `Not covered` or `Gap`; it cannot justify `Pass`.
- `/research-workspace` is the only active WebUI route. `/workspace-playground`
  must remain removed with no alias and no redirect.
- Extension handoff requires a current Chrome MV3 build plus live CDP validation
  against the backend.
- MCP, ACP, and Sandbox are part of the canonical workspace model. If a behavior
  is only documented as a handoff contract and not yet exposed as a working UI/API
  flow, it must be marked `Partial` or `Gap`.

## Status Legend

| Status | Meaning |
| --- | --- |
| Pass | Works in the live product or focused regression suite, with evidence recorded. |
| Partial | Core contract exists, but part of the user workflow is not live or not fully covered. |
| Blocked | Cannot be validated until an external prerequisite is available. |
| Gap | Known missing behavior or endpoint/UI surface. |
| Watch | Works for the tested path, but needs continued matrix coverage because it is high-risk. |

## Matrix

| ID | Scenario | Owner | Surface | Status | Evidence | Regression coverage | Follow-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-UAT-001 | Backend starts and authenticated workspace APIs respond | TASK-478.3 | FastAPI Workspaces API | Pass | TASK-478.3 live validation created a workspace, added an idempotent source, saw one `workspace_source_ingest` job, and read `/sources/status` job/error details. | `tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py`; `test_workspaces_api.py`; media ingest worker tests. | Keep status rows updated when Jobs projection changes. |
| RW-UAT-002 | Canonical `/research-workspace` route loads | TASK-478.7, TASK-478.13 | WebUI route | Pass | TASK-478.13 live probe booted `/research-workspace`, found `#workspace-main-content` and `workspace-workspaces-button`, and recorded screenshot `/private/tmp/task47813-live-matrix-research-workspace.png`. | `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`. | Keep this as a high-risk smoke check for future route metadata work. |
| RW-UAT-003 | Old `/workspace-playground` route is removed with no redirect | TASK-478.7, TASK-478.13 | WebUI route | Pass | TASK-478.13 live probe returned HTTP 404 for `/workspace-playground`, retained path `/workspace-playground`, and observed no `Location` header. Focused Playwright regression passed. | `research-workspace.real-backend.spec.ts` checks 404, retained path, then canonical route boot. | Never add route aliases or redirects for compatibility. |
| RW-UAT-004 | Model catalog loads selectable configured models | TASK-478.1 | Model selector | Pass | Live CDP selected configured `Ollama / gemma3:1b`; fresh tab reported 0 console errors after Add Sources -> My Media. | `modelSelectorUtils.test.ts`; `ChatPane.stage2.test.tsx`. | Maintain provider metadata normalization across shared selectors. |
| RW-UAT-005 | Missing-model and failed-response states are recoverable | TASK-478.2 | Chat composer/RAG send | Pass | Live CDP confirmed missing-model sends made zero chat requests, draft was preserved, invalid provider rendered recoverable 503 error, empty stream rendered `No response was returned.` | Focused chat/RAG frontend tests for missing model, failed submit, empty stream, and request normalization. | Continue checking provider-specific failure copy during model/provider changes. |
| RW-UAT-006 | First-class workspace source ingestion/indexing status exists | TASK-478.3, TASK-478.30 | Workspaces API, Jobs, Embeddings worker, WebUI source status | Pass | TASK-478.3 live validation exposed workspace-source Jobs in `/api/v1/jobs/list` and `/sources/status`, including progress/error details for missing media. TASK-478.30 then ran a live backend on `127.0.0.1:18033`, WebUI on `127.0.0.1:18034`, and Redis embeddings worker with task-scoped streams; a bounded media source completed `POST /api/v1/media/1/embeddings` job `e6861c66-c3c7-4cfa-965d-0e445078bb91` with `embedding_count=1`, Media DB reported `chunking_status=completed` and `vector_processing=1`, `/api/v1/workspaces/research-workspace-task47830-1779931800375/sources/status` reported `state=queryable`, `readiness.vector_ready=true`, `progress_percent=100`, and `Ready for grounded questions.`, and WebUI CDP showed the source card as `READY` with the store source status from `workspace-status-projection`. Screenshot: `/private/tmp/task47830-research-workspace-cdp.png`. | Workspace status API tests; media state repository test; embeddings backpressure tests; Redis worker completion tests; focused CDP live run. | Keep watching vector completion because provider/content-policy behavior can affect exact answer text; TASK-478.30 saw RAG cite the selected source while redacting the seeded numeric token under the content-policy filter. |
| RW-UAT-007 | Source cards do not claim fully ready when API says partial | TASK-478.3 | Sources pane | Pass | TASK-478.3 live CDP confirmed source cards aligned with status projection after backend projection fix. | Frontend regression prevents legacy media fallback from overriding authoritative partial status. | Recheck when vector/indexing providers change. |
| RW-UAT-008 | Individual, bulk, and persisted source selection share one contract | TASK-478.4 | Sources pane, Workspaces API, RAG | Pass | Live CDP seeded two media documents, selected one, saw `/sources/selection` and `/sources/status` agree, and saw `/api/v1/rag/search` include only the selected media ID. | Store/API-first selection tests and server reconciliation tests. | Keep selection intent separate from queryable media IDs for processing sources. |
| RW-UAT-009 | Grounded selected-source RAG Q&A returns evidence/citations | TASK-478.5 | Chat/RAG | Pass | Live WebUI returned `/api/v1/rag/search` documents with `include_media_ids [8,7]`, visible answer containing `PASTE-EVIDENCE-ORION`, and expanded citations titled with source names. | `ragMode.sanitization.test.ts` plus chat-mode tests. | Continue verifying with at least one configured local provider per release. |
| RW-UAT-010 | Studio enables and generates from selected sources | TASK-478.6 | Studio pane | Pass | Live Playwright/CDP generated a completed summary artifact with local provider response HTTP 200; screenshot `/private/tmp/task4786-studio-summary-live.png`. | StudioPane focused suites and ResearchWorkspace stage tests. | Recheck saved artifact persistence when workspace artifact storage changes. |
| RW-UAT-011 | Add Sources URL validation is explicit and recoverable | TASK-478.8 | Add Sources modal | Pass | Live CDP validated invalid URL inline feedback with 0 console errors and no silent stall. | `AddSourceModal.stage2.intake.test.tsx`. | Extend to duplicate/partial batch ingestion when backend receipts are finalized. |
| RW-UAT-012 | My Media search handles exact matches and already-attached media | TASK-478.8 | Add Sources modal | Pass | Live CDP exact search for `research-workspace-uat-source.md` showed the already-in-workspace explanation instead of unrelated drift. | `AddSourceModal.stage2.intake.test.tsx`. | Add pagination regression if media result sets become large. |
| RW-UAT-013 | Paste/file acquisition creates visible workspace sources | TASK-478.8, TASK-478.9 | Add Sources modal, Sources pane | Pass | TASK-478.8 live CDP paste created `Gate D Paste Smoke`; TASK-478.9 uploaded `/private/tmp/task4789-live-source.md` and saw source creation return 201. | AddSourceModal and source-preview tests. | Keep tied to status projection so acquisition and readiness do not diverge. |
| RW-UAT-014 | Source preview shows captured content/evidence without dumping unbounded payloads | TASK-478.9 | Source preview modal, API | Pass | Live validation opened preview for uploaded source, saw captured content, chunk snippets, citation-ready state, and persisted browser-local annotation; screenshot `/private/tmp/task4789-source-preview-live.png`. | 13 backend tests, 55 frontend tests, OpenAPI guard for preview/status capability. | Add server-persisted annotations if browser-local is no longer acceptable. |
| RW-UAT-015 | Context envelope exposes page capability/readiness/service state | TASK-478.9 | Workspaces API, UI warnings | Pass | TASK-478.9 kept page context envelope separate from bounded preview endpoint and surfaced partial errors as compact Sources-pane warnings. | Workspace API status/capability tests. | Keep envelope small; do not move source previews into page bootstrap payloads. |
| RW-UAT-016 | Responsive layout remains usable at desktop and 390px mobile | TASK-478.10 | Workspace shell | Pass | Live backend/WebUI validation passed at 1365x900 and 390x844; screenshots `/private/tmp/task47810-desktop-source-advanced-after.png` and `/private/tmp/task47810-mobile-source-advanced-after.png`. | Desktop layout and SourcesPane transfer/layout tests. | Continue to include advanced filters open state in responsive checks. |
| RW-UAT-017 | Keyboard/focus paths are reachable for dense workspace controls | TASK-478.10, TASK-478.19 | Workspace shell | Pass | TASK-478.19 added a dedicated keyboard-only source-to-chat E2E: Tab reaches the Research Workspace skip link, Enter focuses Sources, Tab reaches a named source checkbox, Space selects it, Tab reaches the composer, and Enter submits a grounded RAG request with the selected media ID. | Focused UI tests, live layout assertions, and `research-workspace.spec.ts` keyboard source-to-chat regression. | Keep the route-level skip links and source checkbox accessible names covered when pane shell or source rows change. |
| RW-UAT-018 | First-run copy gives next actions without extra banner clutter | TASK-478.11 | Empty states, Sources pane, Chat pane | Pass | Live CDP asserted local/self-hosted copy present, Sources pane wording present, missing-model copy present, and rejected `workspace trust`/`left panel` copy absent. | Source-location copy, SourcesPane, ChatPane tests. | Keep local-first copy contextual at source/model decisions, not as persistent bars. |
| RW-UAT-019 | Start tour and Settings > Replay tour open visible walkthrough | TASK-478.11 | Guided tour | Pass | Live CDP against `/research-workspace` saw `tooltipCount: 1` and `overlayCount: 1` for both first-run Start tour and Settings Replay tour; screenshots `/private/tmp/task47811-first-run-tour.png`, `/private/tmp/task47811-settings-replay-tour.png`. | WebLayout runner and Research Workspace copy tests. | Recheck after any WebLayout or tutorial runner changes. |
| RW-UAT-020 | Research Workspace aligns with canonical Shared Workspaces model | TASK-478.7 | Architecture/API/UI contract | Pass | Contract documented in `Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md`; live backend confirmed `/api/v1/research-workspace/capabilities` 200 and old Research Studio capabilities 404. TASK-478.27 confirms the MCP Hub/Shared Workspaces slice of the model with live fixture-backed evidence. TASK-478.28 confirms the ACP run-history/diagnostics slice with a live fixture-backed Research Workspace to ACP canonical bridge. TASK-478.29 and TASK-478.32 confirm the Sandbox diagnostics/admission slice, including strict enabled-route validation with a real Docker-backed run, while preserving Sandbox ownership of execution state. | Capability endpoint/derivation tests; route metadata and extension route tests; TASK-478.27 MCP Hub workspace-set/policy E2E; TASK-478.28 ACP run-history diagnostics E2E; TASK-478.29 Sandbox workspace diagnostics backend tests; TASK-478.32 strict real-Docker Research Workspace real-backend Playwright fixture. | Keep Research Workspace as a canonical ID/context handoff surface. MCP Hub, ACP, and Sandbox must continue owning their policy, execution, diagnostics, audit, and recovery state. |
| RW-UAT-021 | MCP Hub Shared Workspaces is included in the workspace model | TASK-478.7, TASK-478.21, TASK-478.27 | MCP hub | Pass | TASK-478.27 live backend/WebUI Playwright validation created a canonical Research Workspace ID, registered the same ID as an MCP Hub Shared Workspace with a trusted root, created a team-scoped MCP workspace set, added the Research Workspace as a member, created a named policy assignment, resolved effective policy with `selected_workspace_source_mode: named`, `selected_workspace_trust_source: shared_registry`, the active workspace ID, and `allowed_tools: ["run"]`, executed `/api/v1/mcp/tools/execute` with `x-tldw-workspace-id`/`x-tldw-cwd` and received HTTP 200 from the virtual CLI `run` tool, then opened `/mcp-hub?workflow=setup&view=workspace-sets&workspace_id=<id>&source=research-workspace` and saw the workspace included in an MCP workspace set. The URL and UI text were checked for absence of `workspace-playground`. | Focused live test `mcp-hub.spec.ts` / `binds a Research Workspace into an MCP workspace set and resolves policy evidence` passed against `127.0.0.1:18001` backend and `127.0.0.1:18002` WebUI after tightening the tool assertion to 2xx: `1 passed (2.6s)`. API-only confirmation also returned workspace 200, shared workspace 201, workspace set 201, member 201, assignment 201, effective policy 200, and tool execution 200. | Keep MCP Hub as canonical owner of workspace sets, path trust, and policy/tool execution. Research Workspace should keep passing IDs/context into MCP Hub rather than duplicating MCP policy state. |
| RW-UAT-022 | ACP canonical bridge uses Research Workspace IDs/source labels | TASK-478.7, TASK-478.22, TASK-478.28 | ACP | Pass | TASK-478.28 live fixture created a canonical Research Workspace, created the ACP execution workspace through `/api/v1/agent-orchestration/workspaces/canonical-bridge` with `canonical_workspace_source: research_workspace`, created an ACP-owned project/task/run, confirmed the server-side project filter returned the created project for `canonical_workspace_id=<active-id>&canonical_workspace_source=research_workspace`, opened Research Workspace `ACP run history`, saw the ACP-owned project/task/session/run status, and followed `Open diagnostics` to `/acp-playground?session=<session>&view=diagnostics`. The seeded run may be failed/triaged when the live agent prompt fails, but the run/session/diagnostics ownership remains ACP-owned and visible rather than duplicated in Research Workspace. | Agent orchestration canonical/workspace DB/artifact promotion tests; backend canonical project-filter tests; Research Workspace and Agent Tasks Vitest request-url coverage; focused live Playwright/CDP test `binds a Research Workspace to a real ACP run history and diagnostics path` passed against backend `127.0.0.1:18001` and WebUI `127.0.0.1:18080`: `1 passed (8.4s)`. | Keep ACP as owner of execution workspaces, projects, tasks, runs, sessions, diagnostics, artifacts, audit, and reviewer state. Research Workspace should keep passing canonical workspace IDs/source labels into ACP surfaces rather than storing parallel run history. |
| RW-UAT-023 | Sandbox diagnostics/admission are part of workspace handoff model | TASK-478.23, TASK-478.24, TASK-478.29, TASK-478.32 | Sandbox | Pass | Research Workspace opens sandbox-owned diagnostics for the active canonical workspace ID and sends `source_label=research_workspace`. Backend contract coverage proves a run created through `POST /api/v1/sandbox/runs` with `workspace_id`, `workspace_group_id`, and `scope_snapshot_id` is returned by `GET /api/v1/sandbox/workspaces/{workspace_id}/diagnostics`. TASK-478.29 tightened admission so route-enabled but execution-disabled environments return `admission.state=blocked` and `reason_code=sandbox_execution_disabled`; route-disabled strict validation failed closed with HTTP 404 as expected. TASK-478.32 repeated the strict live backend + WebUI/CDP path with a real Docker daemon: backend `127.0.0.1:18041`, WebUI `127.0.0.1:18042`, config `/private/tmp/tldw_task47829_config.txt`, `SANDBOX_ENABLE_EXECUTION=1`, and `TLDW_SANDBOX_DOCKER_FAKE_EXEC=0`. The strict Playwright case created and observed a workspace-linked sandbox run successfully. A CDP probe captured active workspace `420d15bb-aaae-4f02-ab0d-35376732fb0a`, real Docker run `d9af7f15-2ed0-44e7-acf9-5295fb633bce`, `phase=completed`, `exit_code=0`, message `Docker execution finished`, `runtime.state=available`, `admission.state=available`, `runs.total=1`, and screenshot `/private/tmp/task47832-real-docker-sandbox-diagnostics-full.png`. | Backend tests `test_workspace_diagnostics_includes_run_created_through_sandbox_api`, `test_workspace_diagnostics_blocks_admission_when_sandbox_route_disabled`, and `test_workspace_diagnostics_blocks_admission_when_execution_disabled`; real Docker lifecycle test `test_docker_runner_integration.py::test_full_lifecycle` passed with fake execution disabled; strict Playwright `shows workspace-linked sandbox run in diagnostics when sandbox run API is available` passed against the live real-Docker backend with `TLDW_E2E_REQUIRE_SANDBOX_WORKSPACE_RUN=1` and `TLDW_E2E_EXPECT_SANDBOX_RUN_PHASE=completed`: `1 passed (31.8s)`. | Keep Sandbox as owner of run creation, execution, diagnostics, and runtime/admission state. Preserve the strict fail-closed route-disabled check and the real-Docker enabled-route check as separate evidence paths; do not duplicate sandbox execution state in Research Workspace. |
| RW-UAT-024 | Browser extension capture targets canonical workspace/source IDs | TASK-478.12 | Browser extension, WebUI | Pass | TASK-478.12 live Chrome MV3/CDP run saved a Web Clipper workspace clip against `http://127.0.0.1:18002`, opened canonical `#/research-workspace`, verified `GET /api/v1/web-clipper/{clip_id}` workspace placement, verified the clipped body through `GET /api/v1/workspaces/{workspace_id}/notes`, and verified `GET /api/v1/workspaces/{workspace_id}/sources/status` contains `web-clipper:{clip_id}` as a first-class `web_clip` source with a non-null `media_id`, `state: partially_queryable`, FTS/citation readiness, and progress messaging. CDP also loaded `/research-workspace` in the WebUI against the live backend after clearing stale local state with no new warnings/errors. | `apps/extension/tests/e2e/research-workspace.real-backend.spec.ts` now requires the promoted source row/status projection; focused Web Clipper service/API tests cover Media DB promotion, idempotent retry reuse, and job enqueue attempts. | Vector readiness remains pending until embeddings/indexing completes; this is surfaced as `partially_queryable` rather than hidden failure. |
| RW-UAT-025 | Migration/import/export recovery is clear and resumable | TASK-478.3, TASK-515, TASK-516, TASK-478.25 | Migration APIs/UI | Pass | TASK-515 live/backend work made finalized, server-readback-verified migration sessions `client_delete_eligible=true` only after declared chunks and manifest hash match. TASK-516 live Playwright/CDP validation confirmed the eligible path posts `client-delete-ack`, writes a `contentRetained:false` tombstone, and leaves no covered `tldw-workspace` or matching split workspace content keys after page activity. TASK-478.25 live backend/WebUI CDP validation confirmed the eligible true-move path shows migration recovery details with `Result deleted`, server receipt/status, retained/deleted surface lists, no unknown surfaces, and posts one `client-delete-ack`; the blocked unknown-inventory path shows `Result blocked`, retains local content, shows unknown surfaces and retry, and sends no ack. TASK-478.25 also exported the current format `tldw.research-workspace.bundle`, imported the current ZIP through the UI, imported a supported legacy `tldw.workspace-playground.bundle` recovery JSON through the UI, and confirmed `/workspace-playground` returns 404 without redirect. Screenshots: `/private/tmp/research-workspace-migration-eligible.png`, `/private/tmp/research-workspace-migration-blocked.png`, `/private/tmp/research-workspace-import-export.png`. | `tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py`; `src/store/__tests__/workspace-migration.test.ts`; `src/store/__tests__/workspace.test.ts`; `WorkspaceStatusBar.test.tsx`; `ResearchWorkspace.stage3.test.tsx`; TASK-516 and TASK-478.25 live CDP evidence. | Keep true-move deletion as a high-risk regression path. Watch for fresh local-cache classification so current Research Workspace persistence is not mistaken for legacy migration input on clean first-run pages. |
| RW-UAT-026 | Maintained matrix and regression gate exist | TASK-478.13, TASK-478.18, TASK-478.26, TASK-478.31 | Docs, E2E, TypeScript | Pass | Current matrix exists in this document. Live probe passed route, empty-state copy, rejected-copy absence, tour overlay, and critical console/page-error checks. Focused Playwright route regression passed: `1 passed (2.4s)`. TASK-478.18 refreshed the migration row after TASK-515/TASK-516 live true-move validation. TASK-478.26 split the remaining fixture-backed risks into explicit follow-up tasks without moving Partial rows beyond their evidence. TASK-478.31 reproduced the current frontend TypeScript blockers, fixed the `CharacterListContent.design-system.test.tsx` density fixture, and restored a clean shared UI TypeScript gate for Research Workspace-owned code. | `research-workspace.real-backend.spec.ts` route contract test; task-linked matrix updates after high-risk follow-ups; `apps/packages/ui`: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`; focused Research Workspace UI Vitest gate. | Keep this matrix current as future child tasks close. Broader `apps/tldw-frontend` E2E-inclusive TypeScript failures remain classified outside the Research Workspace UAT gate unless they touch `/research-workspace` or its handoff contracts. |
| RW-UAT-027 | Fresh beginner/no-key entry, learnability, tour, mobile layout, and auth recovery | TASK-12020.11, TASK-12020.13, TASK-12968 | WebUI Research Workspace, Settings/auth states | Pass | TASK-12968 ran the full persona on 2026-07-14 with FastAPI `http://127.0.0.1:18160` in `single_user` mode using a server-only key, the advanced-mode WebUI at `http://127.0.0.1:18161` with public API-key and bearer variables explicitly empty, and a fresh Chrome 145 profile through CDP `http://127.0.0.1:18162`. The machine-readable manifest passed 17/17 desktop/mobile checkpoints: readiness; clean direct entry with exactly one active/saved workspace, no visible migration status, and zero migration API requests at both entry and final state; separate Sources, Chat, and Studio empty states; all five first-run tour steps, completion, and Settings replay; visible and Cmd/Ctrl+K workspace search; Add URL permission recovery with the typed URL preserved; and 390x844 navigation/layout with no overlap or horizontal overflow. Browser state contained no cookies, service workers, or current/legacy credentials; browser requests carried neither `X-API-KEY` nor `Authorization`; diagnostics recorded zero page errors, request failures, unexpected HTTP errors, global backend dialogs, or runtime overlays. Expected no-key client guards/401 warnings and one Next development HMR warning remained scoped and recoverable. Evidence: `/private/tmp/task12968-research-workspace-uat/checkpoints.json`, `diagnostics.json`, `desktop-02-settled-workspace.png`, `desktop-06-first-run-tour.png`, `desktop-08-replay-tour.png`, `desktop-09-visible-search.png`, `desktop-11-add-url-auth-recovery.png`, and `mobile-01-direct-entry.png` through `mobile-06-final.png`. | Regressions cover always-mounted route tours, empty-hydration persistence suppression in split and monolithic modes, workspace-ID-scoped migration suppression, StrictMode-safe fresh initialization, transient tour feedback, preserved partial-ingestion rows, and compact mobile header structure. Focused shared-UI suites passed 9 files / 164 tests; the WebLayout regression passed 14 tests; the maintained real-backend `UAT entry evidence` Playwright check passed separately in 28.4s. Targeted ESLint reported zero errors; its warnings are existing findings outside the changed lines. | Keep the complete clean-profile CDP checkpoint manifest as the certification contract. Re-run after changes to workspace hydration/migration, WebLayout tutorial ownership, header responsiveness, auth bootstrap, or global backend-modal routing; do not add `/workspace-playground` aliases or redirects. |
| RW-UAT-028 | Authenticated power-user setup leads to usable Research Workspace APIs | TASK-12020.11, TASK-12020.12, TASK-12020.15, TASK-12020.16, TASK-12020.17, TASK-12020.18, TASK-12020.19, TASK-12020.20, TASK-12020.21, TASK-12020.22, TASK-12020.23, TASK-12020.24, TASK-12020.25, TASK-12020.26, TASK-12020.29 | Settings, auth persistence, Research Workspace API clients, global backend recovery modal, Studio, Share Workspace, import/export, Sources pane bulk actions | Partial | TASK-12020.16 reproduced invalid-auth recovery in the in-app browser: Settings with `tldw_invalid_task12020` showed `Connection failed`, `Invalid API key -- HTTP 401`, `Core: unreachable`, and `RAG: healthy` (`/private/tmp/task12020_16_cdp_settings_invalid_key.png`); `/research-workspace` then stayed in workspace-scoped degraded state with `Server context unavailable` and no global unreachable modal (`/private/tmp/task12020_16_cdp_workspace_invalid_key.png`). Restoring `THIS-IS-A-SECURE-KEY-123-FAKE-KEY` showed `Server responded successfully`, `Core: reachable`, and `RAG: healthy` (`/private/tmp/task12020_16_cdp_settings_valid_key_recovery.png`), then `/research-workspace` recovered to `Server context ready`, `Connected`, and no modal (`/private/tmp/task12020_16_cdp_workspace_valid_key_recovered.png`). TASK-12020.17 repeated the controlled path against backend `http://127.0.0.1:8001` and WebUI `http://127.0.0.1:8081`: Settings > Recheck showed `Server responded successfully`, `Core: reachable`, and `RAG: healthy` (`/private/tmp/task12020_17_cdp_settings_recheck_healthy.png`), then after restarting the WebUI to pick up the shared-package proxy change, `/research-workspace` showed `Server context ready`, `Connected`, and no stale modal after a prior transient `GET /api/v1/chat/commands` status-0 warning (`/private/tmp/task12020_17_cdp_workspace_no_stale_modal_after_restart.png`). The same controlled pass added a pasted source, selected it, opened status details and preview, added a local annotation, and TASK-12020.18 fixed the no-match clear action so an empty active folder recovers to visible source rows (`/private/tmp/task12020_18_cdp_clear_folder_filter_recovered.png`). A grounded-chat attempt with the selected source then produced an inline recoverable RAG error while optional audio voice catalog/list bootstrap requests returned status 0 and previously caused a distracting global modal; TASK-12020.19 reloaded the fixed WebUI and confirmed `Server context ready`, `Connected`, selected source state retained, and no global modal (`/private/tmp/task12020_19_cdp_audio_voice_no_global_modal.png`). TASK-12020.20 confirmed direct selected-source RAG returned HTTP 200, then reloaded the fixed WebUI and confirmed the same selected source produced a grounded answer with the evidence phrase and source title while `GET /api/v1/ingestion-sources/capabilities` status-0 no longer opened the global modal (`/private/tmp/task12020_20_cdp_grounded_chat_success_no_capabilities_modal.png`). TASK-12020.21 reproduced Studio `Summary` failing with `no_provider_configured` for stale `tldw:gemma3:1b` (`/private/tmp/task12020_11_studio_summary_no_provider_configured.png`), then confirmed through CDP DOM trace that the fixed page shows the stale-model warning, disables `Summary`, suppresses Studio "You can still try it" degraded capability copy, keeps `Server context ready`/`Connected`, and opens no backend-unreachable modal. TASK-12020.24 confirmed the remaining configured-provider path is environment/configuration-blocked because `ollama/gemma3:1b` maps to an endpoint rejected by backend egress (`Port not allowed: 11434`) and the configured Ollama host was not reachable from the UAT session. TASK-12020.29 added backend/frontend readiness metadata so egress-blocked, unreachable, missing-credential, and unsupported-provider states are filtered from selectable Studio models or shown as a prerequisite warning before artifact creation. TASK-12020.22 confirmed Share Workspace team/org validation, share-link generation, Active Shares metadata, successful token revoke cleanup, and workspace export success; the pre-fix confirmation remained visible after success, while the post-fix CDP trace generated and revoked token prefix `tdvf83sx`, removed it from Active Shares, closed the confirmation, showed no backend-unreachable modal, and kept `Server context ready`/`Connected`. TASK-12020.26 added owner-side target-scope enforcement and live-confirmed invalid team/org share target rejection with no ghost active shares, plus seeded active-share access/clone updates through the WebUI against the live backend. Export produced `Workspace exported: new-research-2026-06-26t18-15-54-391z.workspace.zip`; TASK-12020.25 reconfirmed import dialog visibility and accepted file types with `/tmp/research-workspace-import-TASK-12020-25.workspace.json`, saved `/private/tmp/task12020_25_cdp_import_dialog.png`, and TASK-12020.31 later completed a real attached-file import through the same Import Workspace dialog by loading a disposable `.workspace.json` bundle with a failed artifact (`/private/tmp/task12020_31_failed_artifact_imported.png`). TASK-12020.23 added three disposable pasted sources, confirmed no Add Sources dialog, no global backend-unreachable modal, and no runtime overlay after creation, hit-tested Select all with the checkbox input as the top element, selected all three sources, confirmed `Move / Copy`, `Preview selected`, and `Remove (3)` became available, then bulk-removed and used Undo to restore all three sources. Evidence: `/private/tmp/task12020_23_cdp_bulk_selection_recovered.png`. | Focused coverage now verifies invalid-key Settings failure on an authenticated workspace storage endpoint, model-cache invalidation after settings updates, migration chunk status-0 suppression, workspace source refresh/upsert reconciliation failure suppression from the global backend modal, Research Workspace chat-command bootstrap status-0 suppression from the global backend modal, optional audio voice catalog/list bootstrap status-0 suppression from the global backend modal, optional ingestion-source capabilities bootstrap status-0 suppression from the global backend modal, caller-handled best-effort request suppression from the global backend modal, stale `tldw:` Studio model gating, provider-qualified configured Studio model availability, provider-readiness metadata propagation, unavailable Studio model prerequisite copy, share revoke confirmation close-on-success and stay-open-on-error behavior, share target-scope rejection and active-share update controls, source no-match clearing across search/advanced filters/active folder, workspace import accepted/rejected file handling and imported workspace/source state, header context refresh after page-level status recovery, and WebLayout modal clearing after connected recovery. TASK-12020.16 focused tests passed: 4 files / 133 tests; TASK-12020.17 added the chat-command bootstrap proxy regression to `background-proxy.test.ts`; TASK-12020.18 added the active-folder no-match regression to `SourcesPane.stage4.filters-and-sort.test.tsx`; TASK-12020.19 added the audio voice bootstrap proxy regression to `background-proxy.test.ts`; TASK-12020.20 added the ingestion-source capabilities proxy regression to `background-proxy.test.ts`; TASK-12020.21 added the stale-model and provider-qualified model regressions to `StudioPane.stage1.test.tsx`; TASK-12020.22 added share/team revoke confirmation close/error regressions to `ShareDialog.test.tsx`; TASK-12020.23 added caller-handled request suppression coverage to `background-proxy.test.ts`, paste modal close/tagging suppression coverage to `AddSourceModal.stage1.ingestion.test.tsx`, and processing-status media-detail suppression to `ResearchWorkspace.stage3.test.tsx`; TASK-12020.26 added backend Sharing regressions for valid owned org scopes plus invalid team/org scope rejection and a ShareDialog regression for active-share access/clone patching; TASK-12020.29 added `test_llm_providers_readiness.py`, model-normalization readiness propagation coverage, `TldwModelsService` readiness preservation coverage, and Studio unavailable-model prerequisite coverage. | The auth/setup-to-usable-workspace path plus paste-source, bulk source selection/remove/undo recovery, status-details, preview, annotation, source-filter recovery, optional bootstrap modal scoping, controlled selected-source grounded chat/RAG, stale Studio model blocking, provider-readiness prerequisite surfacing, share-link generation/revoke recovery, owner-side team/org target rejection, active-share access/clone updates, workspace export, and attached-file import paths are live-CDP-confirmed or focused-regression-confirmed as noted. Keep this row Partial until successful Studio generation with a reachable configured provider, true recipient-side team/org share behavior with real membership fixtures, and remaining destructive/recovery workflows outside source bulk remove receive a clean full browser pass. Residual console warnings remain for some status-0 background fetches, including models metadata, notes search, storage, slides, flashcards, and audio voices. |
| RW-UAT-029 | Final UAT browser runner is repeatable in the Codex macOS sandbox | TASK-12020.11, TASK-12020.14, TASK-12020.16, TASK-12020.17, TASK-12130 | QA tooling, Playwright/CDP | Partial | Initial Playwright run failed to bind `0.0.0.0:8080` with `EPERM`. After local network permission and `-H 127.0.0.1`, WebUI started, but Chromium headless shell failed before page code executed with `bootstrap_check_in ... Permission denied (1100)`. TASK-12020.14 added `bun run e2e:research-workspace:uat`, which defaults WebUI autostart to `bun run dev -- -H 127.0.0.1 -p 8080`, writes JSON evidence, and classifies outcomes as `passed`, `product_failed`, or `environment_blocked`. TASK-12130 re-ran issue #2605 certification on 2026-07-04 with the full live FastAPI backend at `http://127.0.0.1:8000`, the full Next.js WebUI quickstart proxy at `http://127.0.0.1:8080`, and a local llama.cpp endpoint at `http://127.0.0.1:9099/v1` advertising `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`. Standalone Chromium launched and executed all 25 configured tests against the real application. An initial llama-backed run exposed one Flashcards manage-page assertion race after moving a workspace deck to general scope; the test now reopens the moved general deck directly before asserting the preserved card UUID. The final runner wrote `apps/tldw-frontend/test-results/research-workspace-final-uat-evidence-2026-07-04-llamacpp9099-rerun.json` and `apps/tldw-frontend/test-results/research-workspace-final-uat-report-2026-07-04-llamacpp9099-rerun.json`; Playwright exited 0 with 24 passed, 1 skipped, 0 flaky, and 0 unexpected failures, while the wrapper exited 75 with `status=environment_blocked`, `scope=environment`, and reasons `environment_skips_present,sandbox_run_api_unavailable`. | `bunx vitest run ../packages/ui/src/services/acp/__tests__/connection.test.ts` passed for the ACP runtime single-user auth fallback after a red failure without the fix. The focused real-backend Flashcards scope-move case passed after the final Manage-page stabilization. The final full-app command was `TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://127.0.0.1:8080 TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY TLDW_RESEARCH_WORKSPACE_UAT_EVIDENCE=test-results/research-workspace-final-uat-evidence-2026-07-04-llamacpp9099-rerun.json TLDW_RESEARCH_WORKSPACE_UAT_REPORT=test-results/research-workspace-final-uat-report-2026-07-04-llamacpp9099-rerun.json bun run e2e:research-workspace:uat -- --no-autostart`; it included live llama.cpp chat generation, the canonical `/research-workspace` route, and legacy `/workspace-playground` removal regression. Docs: `Docs/Development/Research_Workspace_Final_UAT_Runner.md`. | Standalone Playwright is no longer blocked at browser launch in this session, and the chat-model blocker was removed by the local llama.cpp endpoint. Final certification remains environment-blocked until the UAT environment exposes the sandbox run API, or the documented in-app browser/CDP fallback covers that full path with equivalent backend capability. No product failures remained in the 2026-07-04 llama-backed standalone full-app run. |
| RW-UAT-030 | Remaining destructive/recovery actions expose usable recovery controls | TASK-12020.27, TASK-12020.31, TASK-12020.32, TASK-12020.33 | Workspace settings, Studio artifacts, chat/note/source organization | Partial | TASK-12020.27 live-CDP inventory covered workspace archive/delete/restore, collection delete, banner reset, chat clear/message delete, note clear, artifact delete, source transfer undo, and excluded the already-certified source bulk remove/share revoke paths. Archive cancel and artifact-delete cancel both closed safely with state preserved. Archive success on disposable duplicated workspaces showed `Workspace archived.` and switched back to `New Research`, but no visible `Undo` rendered. Artifact delete success removed the failed output and showed `Output deleted.`, but no visible `Undo` rendered. Screenshots: `/private/tmp/task12020_27_archive_cancel_dialog.png`, `/private/tmp/task12020_27_archive_success_missing_undo.png`, `/private/tmp/task12020_27_artifact_delete_cancel_dialog.png`, `/private/tmp/task12020_27_artifact_delete_missing_undo.png`. TASK-12020.31 identified the root cause as Ant Design `message.open` ignoring the notification-only `btn` field, added rendered-content regressions for workspace archive and artifact delete, and moved Research Workspace destructive Undo actions into rendered message content. A follow-up live archive recheck on locked WebUI `127.0.0.1:8081` still showed missing Undo because the served compiled `WorkspaceHeader` chunk was older than the source change (`/private/tmp/task12020_31_archive_stale_missing_undo.png`). A clean temporary webpack server on `127.0.0.1:8083` loaded the current bundle; archiving a disposable duplicate showed `Workspace archived.` with visible `Undo`, and clicking `Undo` restored `New Research (Copy)` with `Workspace restored` feedback. Screenshots: `/private/tmp/task12020_31_archive_undo_visible.png`, `/private/tmp/task12020_31_archive_undo_restored.png`. A shell-launched standalone Chromium then imported an attached disposable `.workspace.json` bundle containing one failed artifact, rendered `Failed output`, deleted it, showed `Output deleted.` with one visible `Undo`, and restored the card after `Undo` with `Output restored` feedback. Screenshots: `/private/tmp/task12020_31_failed_artifact_imported.png`, `/private/tmp/task12020_31_failed_artifact_deleted_undo_visible.png`, `/private/tmp/task12020_31_failed_artifact_restored.png`. The resumed TASK-12020.27 pass then live-confirmed chat clear/Undo (`/private/tmp/task12020_27_chat_clear_undo_visible.png`, `/private/tmp/task12020_27_chat_clear_restored.png`), message delete/Undo (`/private/tmp/task12020_27_message_delete_undo_visible.png`, `/private/tmp/task12020_27_message_delete_restored.png`), Quick Notes clear/Undo (`/private/tmp/task12020_27_note_preloaded_visible.png`, `/private/tmp/task12020_27_note_clear_undo_visible.png`, `/private/tmp/task12020_27_note_clear_restored.png`), per-source remove/Undo (`/private/tmp/task12020_27_single_source_remove_undo_visible.png`, `/private/tmp/task12020_27_single_source_remove_restored.png`), and source move-transfer/Undo (`/private/tmp/task12020_27_source_transfer_undo_visible.png`, `/private/tmp/task12020_27_source_transfer_restored.png`). The same pass found two defects: imported workspace chat sessions are wiped after a successful Import Workspace flow (`/private/tmp/task12020_27_probe_after_import.png`, split to TASK-12020.32), and selected-source batch `Remove (1)` is enabled but inert because the batch handler schedules Undo without applying removal (`/private/tmp/task12020_27_source_probe_before.png`, `/private/tmp/task12020_27_source_probe_after.png`, split to TASK-12020.33). | Focused TASK-12020.31 regressions now render message `content` and require an accessible `Undo` button for workspace archive and failed-artifact delete. Broader focused coverage verifies archive, chat clear/message delete, quick-note clear, artifact delete, source transfer, source annotation/source removal, template start-over, duplicate open-original, and note shortcut paths no longer depend on `btn` in current source. Source inspection confirms no `btn:` fields remain under `apps/packages/ui/src/components/Option/ResearchWorkspace`. TASK-12020.27 live browser evidence now covers the previously pending chat, note, per-source remove, and source-transfer recovery paths. TASK-12020.32 adds RED/GREEN coverage that imported chat messages load before empty local autosave can overwrite the session. TASK-12020.33 adds click-through coverage that selected-source batch Remove applies `removeSources(["s1", "s2"])` and Undo restores direct selection plus folder memberships. | Keep this row Partial pending a fresh live browser recheck of TASK-12020.32 and TASK-12020.33 in an unblocked browser environment. Archive Undo restore, failed-artifact delete Undo restore, chat clear Undo, message delete Undo, Quick Notes clear Undo, per-source remove Undo, and source transfer Undo are live-confirmed on the current bundle; the locked stale 8081 server should not be used for product evidence. |

### TASK-12020.28 Fresh UAT Evidence (2026-07-05)

TASK-12020.28 repeated the final Research Workspace runner against the full FastAPI backend at `http://127.0.0.1:8000`, the full Next.js WebUI at `http://127.0.0.1:8080`, and the user-provided llama.cpp server at `http://127.0.0.1:9099/v1`. The backend used a temporary config file at `/tmp/task12020_28_config.txt` so `llama.cpp` was both `default_api` and `default_api_for_tasks`, private-local egress to port `9099` was allowed, and the repository config was not modified. Provider metadata showed the llama.cpp model `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`; a direct backend chat completion returned exactly `UAT llama ready`, proving generation flowed through the configured local model.

| Coverage area | Result | Evidence |
| --- | --- | --- |
| Beginner and power-user entry evidence | Pass | The final runner captured fresh beginner/no-key and power/API-key entry screenshots plus JSON state as inline Playwright attachments, extracted to `/tmp/task12020_28_uat_attachments_2026-07-05/`. |
| Canonical routing and removed legacy route | Pass | The runner confirmed `/research-workspace` remains canonical and `/workspace-playground` is removed. |
| Live backend health, bootstrap, auth, and context APIs | Pass with one recovered warning | Backend health returned HTTP 200 in `single_user` mode; the responsive probe rendered the workspace while logging a recovered `GET /api/v1/workspaces/<uuid>/context` HTTP 404 for a fresh generated workspace ID. |
| Grounded chat and Studio source scoping | Pass | The real-backend tests grounded chat on a selected source and scoped Studio compare-sources generation to selected media IDs. |
| Workspace search, source visibility moves, quizzes, and flashcards | Pass | The runner searched live chat turns and verified generated quiz/flashcard assets stayed workspace-hidden until explicitly moved to general scope without record ID changes. |
| ACP and sandbox handoff request scoping | Pass/Environment skip | ACP run-history and sandbox diagnostics requests carried the active workspace ID. The strict sandbox-run creation case skipped because `POST /api/v1/sandbox/runs` returned HTTP 404 in this backend profile. |
| Mocked destructive/recovery flows | Pass | The mocked workflow spec covered add-source flows, keyboard-only selection, generation cancel, failed-artifact marking, and failed-artifact recovery after reload. |
| Responsive layout | Pass | Standalone Playwright screenshots were captured at `/tmp/task12020_28_research_workspace_playwright_desktop_2026-07-05.png` and `/tmp/task12020_28_research_workspace_playwright_mobile_2026-07-05.png`; desktop showed the three-pane workspace and mobile showed the tabbed Sources/Chat/Studio layout with no obvious overlap. |

Final runner stats: 25 configured tests, 24 passed, 1 skipped, 0 unexpected failures, 0 flaky, Playwright exit code 0, wrapper exit code 75, evidence `/tmp/task12020_28_research_workspace_uat_evidence_2026-07-05.json`, report `/tmp/task12020_28_research_workspace_uat_report_2026-07-05.json`. The wrapper correctly classified the run as `environment_blocked` with reasons `environment_skips_present` and `sandbox_run_api_unavailable`. The skipped case is now tracked separately as TASK-12020.36; the final strict run requires a sandbox-capable backend profile with `[API-Routes] stable_only = true` plus `enable = sandbox`, `SANDBOX_ENABLE_EXECUTION=1`, and real/fake execution settings chosen deliberately. No product failures remained in the 2026-07-05 full-app llama-backed runner. The in-app browser fallback was attempted as supplemental evidence but showed a runtime fetch overlay for model metadata in that tool context, while the standalone Playwright run had clean product evidence.

## Current High-Risk Remainders

1. TASK-12020.12 removed the Settings false-success path for invalid
   single-user keys and the stale chat-model cache risk after settings updates.
   TASK-12020.15 added automated coverage for migration chunk status-0
   scoping and stale global backend-modal recovery. TASK-12020.16 then live
   CDP-confirmed invalid-auth workspace degradation, valid-key recovery to
   `Server context ready`, and suppression of workspace reconciliation failures
   from the global backend modal. TASK-12020.17 added the remaining
   chat-command bootstrap scoping after the controlled workspace recovered but
   retained a stale modal. TASK-12020.18 fixed the source no-match recovery
   path found while testing folder organization in the controlled power-user
   pass. TASK-12020.19 scoped optional audio voice bootstrap failures out of
   the global modal after a grounded-chat attempt surfaced audio voice status-0
   warnings alongside a recoverable RAG failure. TASK-12020.20 then scoped the
   optional ingestion-source capabilities bootstrap out of the same global modal
   after direct selected-source RAG returned HTTP 200, and the controlled browser
   pass produced a grounded answer with the selected evidence phrase and source
   title. TASK-12020.21 then blocked stale legacy `tldw:` Studio model
   selections before generation and preserved provider-qualified configured
   model availability in regression coverage. TASK-12020.24 records that the
   remaining configured-provider Studio path is blocked by UAT provider/runtime
   setup: the advertised Ollama endpoint uses port `11434`, which backend egress
   currently blocks, and the configured Ollama host is not reachable from this
   session. TASK-12020.29 now surfaces egress, reachability, credential, and
   provider-alias blockers through provider/model readiness metadata and Studio
   prerequisite copy before artifact creation. TASK-12020.22 then fixed and
   live-rechecked share-link revoke confirmation cleanup. TASK-12020.23 then
   fixed paste-source modal cleanup and scoped best-effort tagging/status-poll
   failures out of the global backend modal, allowing source bulk selection,
   bulk remove confirmation, and Undo recovery to pass live CDP. Paste-source
   creation, bulk source selection/remove/undo, source details, preview,
   annotation, filter recovery, optional bootstrap modal scoping, controlled
   selected-source grounded chat, stale Studio model blocking, provider-readiness
   prerequisite surfacing, share-link generation/revoke recovery, and workspace
   export now have live CDP or focused-regression evidence as noted; do not mark
   successful Studio generation, broader share/team permissions, or remaining
   destructive-action recovery
   outside source bulk remove as freshly certified until a clean full
   browser/CDP pass covers those workflows. TASK-12020.25 reconfirmed import
   dialog behavior and TASK-12020.31 later completed a real attached-file
   import through Import Workspace. TASK-12020.27 then
   confirmed archive/artifact destructive cancel paths but found a broader
   missing visible `Undo` recovery control after archive success and artifact
   delete success; TASK-12020.31 fixed the current source-level message action
   pattern with rendered-content regressions and live-confirmed archive plus
   failed-artifact Undo restore on a clean current bundle. TASK-12020.27 then
   live-confirmed chat clear/Undo, message delete/Undo, Quick Notes clear/Undo,
   per-source remove/Undo, and source transfer/Undo. TASK-12020.32 now has a
   ChatPane RED/GREEN regression and autosave guard for imported chat-session
   preservation, and TASK-12020.33 has focused click-through coverage for
   selected-source batch Remove plus Undo folder restoration. This matrix keeps
   the remaining destructive/recovery row Partial until those two paths receive
   a fresh live browser recheck in an unblocked browser environment.
2. TASK-12020.13 added automated coverage for the no-key beginner search gap
   found in CDP: a visible workspace search action, local Cmd/Ctrl+K modal
   opening, shortcut help copy, and the `/research-workspace` command-palette
   route guard. Final beginner certification still needs a clean live browser
   pass through TASK-12020.14.
3. TASK-12020.14 provides the repeatable runner and in-app browser/CDP fallback
   for Codex macOS sessions. A fresh 2026-06-26 standalone run reached the WebUI
   but returned `environment_blocked` before any page assertions because
   Chromium could not start under the macOS sandbox. TASK-12130 re-ran the same
   certification path on 2026-07-04 against the full live backend, WebUI, and
   local llama.cpp model on `127.0.0.1:9099`. Chromium launched, page assertions
   executed, and Playwright reported 24 expected passes, 1 environment skip, 0
   unexpected failures, and 0 flaky tests. The wrapper still returned
   `environment_blocked` because the current environment lacks
   `POST /api/v1/sandbox/runs`. Keep that environment skip separate from product
   failures, and do not count the row as fully certified until that backend
   capability is present or the documented in-app browser/CDP fallback covers the
   same full persona matrix with equivalent capabilities.
4. The Research Workspace-owned shared UI TypeScript gate is clean as of
   TASK-478.31. The broader `apps/tldw-frontend` E2E-inclusive TypeScript check
   still fails on unrelated route-governance, E2E auth, chat cockpit,
   agent-task fixture, and admin llama.cpp fixture typings; those remain outside
   the Research Workspace UAT gate unless they regress `/research-workspace` or
   its MCP/ACP/Sandbox handoff contracts. During TASK-12020.13, a fresh
   `apps/packages/ui` typecheck also reported unrelated Notes, Scheduled Tasks,
   background-service, and voice-cloning type errors; no Research Workspace
   search files appeared in the error list.
5. Long-running vector indexing with real embedding completion is now
   live-verified by TASK-478.30, but should remain a Watch risk because
   provider availability, Redis stream configuration, Media DB readiness flags,
   and content-policy redaction can still affect end-to-end answer quality.
6. Sandbox workspace handoff now has both fixture-backed fail-closed coverage
   and TASK-478.32 real-Docker live validation. Keep the route-disabled,
   execution-disabled, and real-runtime paths separate so future failures show
   whether policy, admission, Docker reachability, or run execution regressed.
7. Migration true-move deletion and guided import/export recovery are now
   live-verified, but remain high-risk because they intentionally delete local
   legacy content only after server receipt verification and bounded inventory
   checks.

## How To Update This Matrix

1. Add or update one row per user-visible workflow, not one row per component.
2. Record the owning Backlog task and exact live evidence. Use screenshot paths
   only as supporting evidence; prefer DOM/API assertions or test names.
3. Mark extension, MCP, ACP, Sandbox, or provider-dependent rows as `Blocked`,
   `Partial`, or `Gap` unless a current live run proves the complete workflow.
4. If a row regresses, update the status immediately and create/link a child
   Backlog task before editing product code.
5. Keep `/workspace-playground` removal as a permanent regression check: 404,
   no `Location` header, no client-side redirect, and no route alias.
