# Ordinary provider and first-document qualification (TASK-13376)

## Result

**Failed; full setup remains unqualified.** The approved media-readiness fix
clears the manual-master-key blocker. Provider setup, Markdown ingestion,
lexical content search and a first ordinary application chat work. Application
model discovery falsely reports no configured models, and provider settings
are lost when the signed stop/start helpers recreate the backend container.
No additional production change, requirement waiver, publication or merge was
made during this qualification.

## Exact candidate and environment

- Source: `cb581a1e16032df27e4026afd784c9a682b584c1`.
- Container platform: `linux/arm64` on macOS Docker Desktop.
- Final signed manifest SHA256:
  `4e5c4173e5437196412de3211a83534f7adb7f4b44b94995d127386a79214de7`.
- Final helper archive SHA256:
  `aadeaa3a05754943017475eab33eb30c4c9e04bebba61bec03427c0c9c8578ee`.
- Final archive extracted outside the checkout at
  `/private/tmp/task13376-workflow-cb581a1e16/bundle`.
- New origin: `http://127.0.0.1:19083`; fresh origin storage, no storage preseed.
- Browser: Codex in-app browser (`iab`). The supported browser API did not
  expose the exact engine version; no exact version is claimed.
- Runtime images built from the committed source; signed helpers and image
  identities verified by the candidate pipeline and ordinary signed start.
- Disposable repository mock provider, private Docker network only; no paid
  provider, model download, global outbound-policy relaxation, manual backend
  master key or frontend/server wiring.
- Scoped anonymous Docker config was required because the default host
  credential-helper issue remains unresolved. Default Docker configuration was
  not modified; this run does not qualify that host setup issue.

The pipeline exited 0 with built-backend MCP/setup checks, 13 lifecycle checks,
38 initial-browser checks, signature/hash verification and owned cleanup.
It removed its signing key. Those initial checks retain their original
`managed_connection_and_initial_wizard_only` scope and
`planned_setup_complete=false`. G12 remains false and promotion was correctly
refused. This manual workflow is separate evidence.

## Ordinary steps and observations

1. Run the final extracted `start.sh`, open its printed loopback URL and choose
   **Set up in WebUI**, then **Solo, Docker**.
2. Acknowledge the displayed local/remote setup and provider-secret notice.
3. Select **Custom OpenAI-compatible**. For this disposable test only, enter
   the fake mock token, private URL `http://qualification-provider:18995/v1`
   and model `gpt-4`. Validate: **Provider validation is ready**, with discovered
   models. Save: masked credential confirmation appears.
4. Continue with existing defaults: local-path ingestion off, balanced
   chunking, automatic metadata, audio skipped, RAG/storage deferred. Skip
   optional MCP tools. These optional choices do not qualify those features.
5. **Send test chat** succeeds and navigates Home. Home shows **Add your first
   source**, with no manual server API-key prompt. This verifies the approved
   readiness correction on the actual signed candidate.
6. Choose **File**, **Add source**, **Browse files**, and the harmless
   `task13376-first-document.md` fixture. The file chooser took an unexpectedly
   long time to return; this is not claimed as acceptable timing or a measured
   application failure. The returned queue contained exactly the 292-byte file.
7. **Use defaults & process** completes: **1 succeeded, 0 failed**, 2 seconds
   reported by the ingest UI. The results shortcut opens Knowledge QA; that
   answer-generation path was not run. Open the ordinary Media library linked
   from Home for lexical search instead.
8. Keep **Full-text** selected, search `cobaltparcel13376`, and open the returned
   document. One result contains the exact fixture text. Chunking is completed;
   vector indexing is pending. No embedding/RAG success is claimed.
9. **Chat with this media** prepares the stored document in the composer.
   Application chat reports **No LLM provider configured** and **No chat models
   configured**. Refresh reproduces those messages. The ordinary Send button
   nevertheless returns the repository mock response from the saved
   `custom-openai-api / gpt-4` target. The contradictory readiness state fails
   acceptance even though this request succeeds.
10. Use signed `stop.sh`, then signed `start.sh`, retaining the same origin,
    private instance state and data/config volumes. The owned external mock is
    detached/reconnected so it does not prevent network recreation; its
    provider configuration is not re-entered.
11. Repeat full-text search: the document and its exact content remain present.
    The earlier conversation also remains visible. Send a new ordinary chat:
    **The selected model is not available**. Technical details identify
    `gpt-4` as unavailable for `custom-openai-api`. Provider persistence fails.

Home also reports Companion personalization unavailable. It was not enabled,
changed or waived; the required core workflow was investigated independently.

## Diagnosed issues and bounded correction proposal

### Model catalog ignores cookie-session authentication

`apps/packages/ui/src/services/tldw/TldwModels.ts` checks only a JWT or API key
in `isConfiguredForModels`. The single-user cookie-session source that already
works for real media/chat requests fails this precheck, returning an empty
model list without making a discovery request. Its cache scope also groups
cookie-session authentication with unauthenticated access.

Proposed correction: recognize the existing single-user cookie-session source
in this service and its cache scope, preserving actual authenticated discovery,
error handling, key/JWT paths and separation from unauthenticated caches. Add
behavioral cases for cookie discovery, rejected sessions and scope changes;
verify the ordinary chat model selector/readiness on a rebuilt signed candidate.
Do not change provider-status API semantics merely to hide the banner.

### Provider config is outside persistent storage

The signed Compose template persists `/app/managed-config` and points only
`TLDW_ENV_FILE` there. Setup writes provider URL, key, model and default to
`/app/tldw_Server_API/Config_Files/config.txt` using the existing config-path
resolver. That directory is in the disposable container filesystem.

A read-only, sanitized probe after recreation confirmed: no explicit config
path override, no `config.txt` in the persistent volume, persistent `.env`
present, and the saved mock URL, model, key and default provider all absent from
the active packaged `config.txt`. No secret value was printed or copied into
the browser.

Proposed correction: persist the existing editable configuration and required
configuration assets in the existing backend config volume, seed only missing
packaged defaults, and preserve existing settings and credentials. Test fresh
initialization and actual backend recreation with saved provider configuration.
Do not overwrite established state, change auth identity, bypass strict model
selection, or change the normal provider setup flow. Additional implementation
requires the user's requested review/approval before proceeding.

## Verification limits and native CI

The approved TypeScript fix has 18 passing hook/Home tests and an independent
review without findings. Scoped lint/format/diff checks pass. The broader
core-route-identity heading test reproduces the same failure on unchanged
`d3d1083adc` (1 failed / 6 passed); it remains open. Bandit is not applicable to
the TypeScript-only production change or this documentation/evidence update.

[Native CI run 36265212062](https://github.com/rmusser01/tldw_server/actions/runs/36265212062):
Windows helper syntax passed; amd64 lifecycle checks passed but initial browser
qualification failed `manual_master_key_absent_2` with
`manual_master_key_required`. Root cause is unverified; no retry, requirement
relaxation or success claim was made. Arm64 was still running at this record.
Whole-frontend typecheck diagnostics remain baseline reporting, not a clean
typecheck claim. Neither Linux container job qualifies a Windows Docker host.

## Evidence and recovery

Private evidence root: `/private/tmp/task13376-workflow-cb581a1e16`.
`workflow-evidence.json` records positive observations and failures separately;
`passed=false` and `planned_setup_complete=false`. Screenshots include
`chat-readiness.png`, `search-after-restart.png` and
`chat-after-restart-failure.png`. Signed helper logs and exact owned container
identities are retained privately. Candidate build evidence remains under
`/private/tmp/task13376-candidate-cb581a1e16`.

The completed amd64 job log and bounded artifact are retained in
`/private/tmp/task13376-ci-cb581a1e16-amd64-failure.log` and
`/private/tmp/task13376-ci-cb581a1e16-amd64-artifact.zip`. Candidate signatures
invalidated by that failed job are not treated as qualified artifacts.

Owned workflow containers are stopped/removed through the signed helper and
exact recorded mock/registry IDs. Private state, named data/config volumes,
images, workspaces and evidence are retained. Unrelated containers are preserved.
Native/core-format qualification and G12 remain separate mandatory product gates.

## Approved root-cause corrections and implementation review

On 2026-09-26 the user approved the two bounded corrections above. TASK-13376.7
adds cookie-session model discovery and cache scope; TASK-13376.8 directs
managed configuration to the existing persistent volume and initializes only
missing packaged assets. Existing settings/credentials are preserved, default
publication is atomic without overwrite, and legacy non-managed startup is
unchanged. The signed lifecycle helper now verifies provider fields as well as
data after recreation. The bundle README describes the retained configuration.

Verification: 52 model/readiness tests, 108 managed/legacy Docker, release-helper
and setup-writer tests, and 59 browser-helper contract tests passed. The first
model/persistence runs demonstrated failures before implementation. Python
lint/format, shell syntax and Bandit pass; Bandit has zero production findings.
Changed TypeScript regions match formatting; ESLint has zero errors and the
same pre-existing unrelated `inputMods` warning. Whole-repository tests and
whole-frontend typecheck are not claimed clean.

An independent read-only reviewer found no critical/important issues and
independently passed 40 model and 14 managed-configuration tests. Its minor
diagnostics issue was corrected and the affected test first failed, then the
108-test suite passed again. Initialization errors report safe asset/reason or
filesystem errno without configuration contents.

These are implementation results, not a replacement for fresh signed ordinary
browser qualification. The prior failed evidence remains unchanged. That old
native CI run ultimately ended with amd64's browser failure and an arm64
180-minute timeout during Bun install. Those failures remain unwaived.

The user subsequently approved unused build-cache cleanup. The default
builder's unused cache prune reclaimed 14.72 GB; exact before/after image,
container and volume identities matched (32 images, 2 containers, 6 volumes).
No image, volume or system prune was performed. This resolved the local
build-space blocker without removing retained state or prior evidence.

## Corrected signed candidate: ordinary workflow and restart passed

The local candidate was built from clean committed source
`9709d0dcb7e846d3a7366a3412afa933208f4b10` on linux/arm64 under macOS Docker
Desktop. Built-backend qualification, 13 lifecycle checks (including saved
provider fields after recreation) and 38 initial-browser checks passed. The
pipeline exited 0; its separate promotion gate correctly refused G12=false.
The initial-browser artifact retains its original limited scope and
`planned_setup_complete=false`; it was not rewritten to stand in for the
ordinary workflow below.

Final manifest SHA256:
`95c5fc1e5037032174afca8fcf6510d84a29b8244a28469ad6b5afcbf5135527`.
Final archive SHA256:
`cc1496f929d3bf68eea1661433e00f2ae3a5519365307603d2edc2b122b7303a`.
Candidate evidence remains at `/private/tmp/task13376-candidate-9709d0dcb7`.
The final signed archive was extracted outside the checkout into
`/private/tmp/task13376-workflow-9709d0dcb7/bundle` with fresh private instance
state, serving only `http://127.0.0.1:19084`. The signing private key was removed
by the candidate pipeline. Exact browser engine version is unavailable through
supported APIs; the browser was Codex's in-app browser, with no version inferred.

The ordinary WebUI workflow passed without entering a server master key or
wiring frontend/backend URLs. It used the repository's mock provider on only
the instance private network, a disposable placeholder credential and model
`gpt-4`. Provider validation discovered models; saving masked the placeholder.
The wizard's test chat completed. Home allowed the first document workflow.
Markdown ingest reported 1 succeeded, 0 failed and 2 seconds elapsed.
Full-text search for `cobaltparcel13376` returned the stored document and exact
content. `Chat with this media` opened the composer and ordinary Send returned
the expected mock response. Provider status was Healthy; the earlier false
missing-provider/model banner was absent without refreshing models or changing
strict selection.

Signed stop/start recreated the application containers with the same private
state and origin. The mock container was disconnected before stop and
reconnected to the recreated private network under its existing alias. No
provider settings were re-entered. A fresh full-text search again found the
document and content; a distinct post-restart application chat request returned
the expected mock response with provider status Healthy. Retained old chat
history alone was not counted as a new request.

The bounded replay steps were:

1. Run the signed start helper and open its loopback setup URL; choose **Set up
   in WebUI**, **Solo, Docker**, and acknowledge the local-access/privacy notice.
2. Configure **Custom OpenAI-compatible** with the test fixture's private URL,
   disposable placeholder key and `gpt-4`; validate, save and continue.
3. Keep local-path ingestion off and balanced/automatic defaults; defer optional
   audio, RAG and storage configuration, skip optional MCP tools, then send the
   wizard test chat.
4. On Home choose **File**, **Add source**, and **Browse files**; upload the
   harmless Markdown fixture and choose **Use defaults & process**.
5. Open the succeeded document in Media. Select full-text search, enter
   `cobaltparcel13376` and verify the result/content. Use **Chat with this media**
   and **Send message**, then verify the returned mock response.
6. Run signed stop/start, reload, repeat the ordinary document search and send
   a new chat request without re-entering provider configuration.

These steps qualify local mock-provider integration and Markdown lexical
search/chat persistence. They do not qualify vector retrieval, commercial
provider access, answer quality or the full format/platform matrix. Optional
Companion personalization remained disabled by the existing backend feature
configuration; it was not enabled as part of this task.

Private `workflow-evidence.json` now records `passed=true` and
`planned_setup_complete=true` for this separate bounded ordinary workflow,
with `full_product_qualification=false`. Screenshots and visible DOM records
include `search-before-restart`, `chat-before-restart`, `search-after-restart`
and `chat-after-restart`. Prior failed `cb581a1e16` evidence remains unchanged.
The browser file-chooser tool took 2090.6105 seconds despite its requested
timeouts before queuing the file; this is a browser-control limitation, not
the application's measured 2-second ingestion time.

After verification, the signed stop helper removed owned application containers
and the exact recorded mock/registry IDs were removed. Named backend data and
configuration volumes, private state, images, workspace and evidence remain.
Both unrelated PostgreSQL containers remain running. The owned browser tab was
closed. No release publication or PR merge occurred.

## Remaining native failures and separately proposed loading correction

[Native CI run 36280791905](https://github.com/rmusser01/tldw_server/actions/runs/36280791905)
ended in failure. amd64 passed all 13 lifecycle checks, including persistent
provider fields, but failed initial-browser `manual_master_key_absent_2` with
`manual_master_key_required`. Arm64 was cancelled by its 180-minute job deadline
during `bun install --frozen-lockfile --cwd /app/apps`, after its last recorded
download/extraction output. The underlying Bun stall remains unverified; local
Bun install and Next build completing do not resolve that native failure.
Windows syntax passed; actual Windows Docker qualification remains open.
No CI retry, manual cancellation or requirement waiver was performed.

A read-only diagnostic of the actual setup route reproduces a separate
loading defect: two new cases fail while all 18 existing route cases pass.
When initial setup state/metadata are incomplete, the route exposes the manual
API-key form. `showLoader = loading && !state` also drops the loader when state
arrives before metadata, while setup choice remains unavailable. This is
consistent with the native browser failure; direct native runtime attribution
has not yet been proved.

The concrete proposal is to show only a loading surface during incomplete
initial state/metadata loading, retain manual recovery after loading/errors
finish, and add regression tests without changing browser acceptance checks.
User approval for this separate correction is pending. No implementation edit
has been made. Read-only diagnostic output is retained at
`/private/tmp/task13376-setup-loading-diagnosis.log`.

The two approved root-cause fixes now have successful local ordinary and restart
evidence. Native failures, the loading proposal, full native/core-format gates
and G12 remain open; the application is not release-qualified.

This final evidence update changes only Markdown documentation and Backlog task
records. `git diff --check` passed; there is no new Python scope for Bandit or
behavior change requiring another regression run. The production verification
and independent review recorded above still apply to the tested source commit.
