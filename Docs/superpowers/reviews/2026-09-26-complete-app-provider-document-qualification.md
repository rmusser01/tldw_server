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

The next local build is constrained by approximately 12 GiB free host storage
versus the measured 8.8 GiB backend rootfs plus build/export overhead. Reclaiming
the shared unused build cache requires the separately requested permission;
no pruning or retained-state/evidence/image deletion has occurred.
