---
id: TASK-13376
title: Qualify complete-app provider setup and first-document workflow
status: In Progress
assignee: []
created_date: '2026-09-26 16:38'
updated_date: '2026-09-27 16:16'
labels:
  - distribution
  - qualification
  - webui
dependencies:
  - TASK-13343
references:
  - Docs/Design/2026-09-20-complete-app-distribution-design.md
documentation:
  - >-
    Docs/superpowers/plans/2026-09-26-complete-app-provider-document-qualification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Explicit unresolved product acceptance checkpoint from the user-approved review corrections. TASK-13265 is the completed design task and cannot stand in for pending implementation or qualification. Initial wizard progression in WP1 does not prove that a newcomer can configure a provider and use documents. This task qualifies that ordinary browser workflow before an installer is advertised as a complete usable application; it does not authorize publication or change native-platform/core-format requirements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 From a fresh signed extracted paired candidate outside a checkout, complete ordinary provider setup with a deterministic mock using the real WebUI and no manual backend master key or frontend-server URL wiring.
- [x] #2 Use ordinary UI controls to ingest a Markdown document, find its content through search, and complete a chat with the configured mock provider; fail on setup or application errors.
- [x] #3 Stop/start preserves provider configuration and document data; record exact candidate source, platform, browser and novice instructions with full-setup evidence distinct from initial-wizard checks.
- [x] #4 Keep full-provider/document qualification false until all required workflows pass; retain the complete native platform and core-format matrix as separate required product gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the approved provider, Markdown ingestion, search and chat workflow on the existing signed extracted candidate outside the checkout.
2. Record concrete failures and trace them through existing UI/backend paths; present any product behavior change for review before implementing it.
3. Add bounded qualification coverage and rerun the workflow, including stop/start persistence, with exact source/platform/browser evidence. Keep full setup false until every acceptance criterion passes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created as an explicit acceptance checkpoint while correcting TASK-13343. No test success is claimed and no requirement is waived.

Continuation on 2026-09-26: working only in complete-app-wp1. Existing qualification source 84d54884346f548379f44fea53c09320b4a5fb6c; local documentation HEAD d3d1083adc24fbb64e5d35c5d0e46311a736a453. No new product decisions, publication or requirement waivers authorized. Inspecting ordinary existing controls first.

Fresh signed source-84 candidate extracted outside checkout at /private/tmp/task13376-provider-probe-20260926/bundle. Disposable local registry and instance on 127.0.0.1:19080; repository mock provider on private instance network only. Actual WebUI selected Solo Docker, acknowledged privacy, configured Custom OpenAI-compatible with mock URL, validated models, saved provider, kept existing ingest defaults, deferred audio/RAG/storage and skipped optional MCP; Send test chat completed and navigated to Home. Home blocks document workflow with Restore media access / Single-user API key. UI Retry reproduces the same blocker. No master key entered. Source inspection identifies usePostOnboardingMediaReadiness.hasRequiredAuth checking only apiKey/runtime override, omitting cookie-session recognized by existing TldwAuth.isAuthenticated. Proposed bounded correction presented for approval; no production/test code changed. Full provider/document evidence remains false.

Reproduction manifest SHA256 39e09e851d90d25e0e3c213bb359bcf25494622ddac85d5930d37160841774a5, linux/arm64 on macOS Docker Desktop, Codex in-app Chromium (exact version not collected). Failure evidence retained at /private/tmp/task13376-provider-probe-20260926/workflow-failure.json. Existing hook suite: 2 tests pass, Node 26 emits localStorage experimental warning; neither test covers cookie-session. Markdown ingestion, search, ordinary application chat and their persistence checks not run because Home requires a manual master key. Proposed fix remains pending approval; no production or test edits. Bandit N/A for task notes and private observation JSON only.

Cleanup verified: signed stop helper stopped the owned application while retaining its instance state and data/config volumes. Exact recorded mock and private registry container IDs removed. Unrelated tldw_postgres_test and tldw_workspace_12020_50_pg remain running. Pending approval; task remains In Progress.

User approved the bounded readiness correction on 2026-09-26: recognize the existing single-user cookie-session source, retain a real listMedia access probe before ready, add regression tests, then resume signed-candidate qualification. Approval does not waive failures or authorize unrelated redesign/publication.

Approved readiness fix adds only the existing single-user cookie-session auth source to the local precheck; the real listMedia probe and its error handling remain mandatory. New behavior tests observed red 5 failed/7 passed, then focused hook+Home setup flow green 18 passed. Scoped ESLint and matching shared-code formatting pass. Broader core-route-identity result and baseline comparison tracked separately. Bandit not applicable: changed production code is TypeScript only, no Python changes.

Stage 1 verified: hook12+Home setup6 =18 passed; read-only provider_readiness_review independently reran18 and found no Critical/Important/Minor issue. Shared style retained; ESLint and formatting/diff checks pass. Baseline extracted d3d1083adc reproduces identical core-route-identity first-heading failure (1 failed/6 passed), with modified hook mocked; private baseline output /private/tmp/task13376-baseline-d3d1083adc/core-route.log retained. No unrelated test fix or success claim. Stage2 fresh signed build and full UI qualification pending.

Approved fix committed and pushed as cb581a1e16032df27e4026afd784c9a682b584c1. Fresh linux/arm64 candidate pipeline exited 0: built-backend setup/MCP, 13 lifecycle and 38 initial-browser checks passed; signature and all local artifact hashes verified, owned fixtures removed, signing key removed. Final manifest SHA256 4e5c4173e5437196412de3211a83534f7adb7f4b44b94995d127386a79214de7; archive SHA256 aadeaa3a05754943017475eab33eb30c4c9e04bebba61bec03427c0c9c8578ee. Promotion correctly refused G12=false. Initial setup scope remains unchanged and full provider/document qualification false. Final archive extracted fresh at /private/tmp/task13376-workflow-cb581a1e16/bundle and signed start underway at loopback port19083. Native CI 36265212062 remains running; Windows syntax passed, native amd64/arm64 build/smoke jobs in progress.

Fresh signed-candidate ordinary browser run: provider validation/save/wizard chat passed; Home media readiness no longer demands the master key. First-source File upload of the harmless Markdown fixture passed (1 succeeded/0 failed, UI2s); full-text search cobaltparcel13376 returned exact stored content. Chat with this media returned the deterministic mock response, but falsely displayed No LLM provider configured / No chat models configured, reproduced by Refresh. TldwModels.ts isConfiguredForModels and cache scope omit cookie-session auth. Signed stop/start passed and document/search retained, but new chat failed model_not_available for custom-openai-api/gpt-4. Read-only sanitized backend probe confirms packaged config.txt outside persistent volume lost saved URL/model/key/default on recreation; persistent .env retained. Full qualification false. Review/proposal recorded in Docs/superpowers/reviews/2026-09-26-complete-app-provider-document-qualification.md. Two bounded additional corrections presented for approval; no additional production edits. CI amd64 lifecycle passed but manual_master_key_absent_2 failed; root cause unverified. Arm64 still running, not cancelled. Exact engine version unavailable via supported browser API, not claimed. Owned application/mock/registry cleanup verified with retained state/volumes/images/evidence; unrelated PostgreSQL containers remain running. AC2/3 and task completion remain open.

User approved both root causes on 2026-09-26. Child tasks TASK-13376.7 and TASK-13376.8 now implement cookie catalog auth/cache and persistent managed config. 52 model/readiness tests, 108 Python entrypoint/helper/setup regressions pass; Bandit0 production findings. Native run36265212062 ultimately failed amd64 manual_master_key_absent_2 and arm64 timed out after180min during Bun install; no waiver/retry/cancellation. Fresh build currently constrained by12GiB host free disk versus8.8GiB backend rootfs plus build/export overhead. Cleanup of unused shared build cache requested separately; no pruning occurred. Existing evidence/state/images retained.

Corrected clean signed source 9709d0dcb7e846d3a7366a3412afa933208f4b10 passed built-backend qualification, 13 lifecycle checks and 38 initial-browser checks locally on linux/arm64. Fresh ordinary WebUI provider validation/save/wizard chat, harmless Markdown ingest (1 succeeded, 0 failed, 2s UI elapsed), full-text search and application chat passed without a manual server key or frontend/server wiring. Signed stop/start recreated containers; search and a distinct new chat passed without re-entering provider settings. Provider stayed Healthy; false missing-provider/model warnings were absent. Separate ordinary evidence is passed=true/planned_setup_complete=true at /private/tmp/task13376-workflow-9709d0dcb7/workflow-evidence.json; initial-wizard evidence is unchanged and full_product_qualification=false. Exact source, hashes, browser/version limitation and replay steps are in the updated review. File-chooser control stalled 2090.6105s despite requested timeouts; that does not measure ingestion time. Owned test containers removed; named volumes, state, images, evidence and workspace retained; unrelated PostgreSQL services preserved. User-approved unused build-cache cleanup reclaimed 14.72 GB with all prior 32 image, 2 container and 6 volume identities preserved. Native CI 36280791905 failed amd64 manual_master_key_absent_2 after 13 lifecycle checks passed; arm64 hit its 180-minute deadline during Bun install, root cause unverified. Windows syntax only passed. Separate loading defect: 2 failing read-only diagnostic cases, 18 existing cases passing; concrete loading-only correction awaits user approval with no implementation or acceptance-check edits. Children TASK-13376.7/.8 are complete; parent remains In Progress for native/loading/full matrix/G12 follow-ups. No waiver, CI retry, manual cancellation, publication or merge.

Loading correction committed/pushed as 3ef013bd94d389454e0b10b67060f9a95e9089af. All 60 affected regressions passed after red 6 failed/18 passed; independent review found no actionable issue and independently passed 24 route tests. Managed production WebUI compilation, token synchronization and bundle budgets passed using existing installed dependencies; existing build skips whole-frontend typecheck. Fresh signed linux/arm64 candidate was blocked during unchanged Bun frozen install after resolved/extracted256, with no progress for1090s. Only strictly revalidated owned Buildx was sent SIGTERM under reviewed15-minute bound; pipeline exited130 and removed owned registry, preserving both unrelated PostgreSQL services and private recovery material/images/evidence. No new signed candidate or browser/lifecycle result, no retry/runtime/dependency change. Native run36293327915 is In Progress on exact3ef source: Windows helper syntax passed; both Linux candidate builds running. Full release/native/core-format/G12 gates remain open. Review and plan contain final evidence and limits. Child TASK-13376.9 is Done; parent remains In Progress. Earlier ordinary workflow evidence stays specific to9709d0dcb7.

Completed the bounded retained-evidence diagnostic, not a Bun fix. Install passed85.08s without reproducing the stall; successful-run Puppeteer/canvas identity is now known and private evidence survives timeout fixtures. Scoped cleanup verified. Native amd64 loading fix independently passes13 lifecycle/38 browser checks; arm64 and full native/core-format/G12 gates remain open. Exact evidence/hashes and qualification limits are recorded in the acceptance review; parent remains In Progress.

TASK13376.11 completed one private BuildKit diagnostic: reproduced resolved256/no-progress, install exit124 at300s, Buildx1 at303s. Retained966records identify Puppeteer24.36.0 node install.mjs executing Bun as pending child, static CPU/I/O232-299s; internal cause remains unknown. Baseline Docker resources retained. Prepared Docker-only PUPPETEER_SKIP_DOWNLOAD RUN proposal awaits requester review and remains unapplied. Source3ef nativeamd64 passes; nativearm64 stillbuilding. No additionalcandidate retry/CIcancel/push/release. Broader qualification Stage4 remainsInProgress.

Approved Docker-only correction completed at source 41e3b9bc0598071034f250702ccc71a7b862a7e3. Local production install 57.06 seconds; built-backend checks and 13 lifecycle / 38 initial-browser checks passed. Native CI 36328715529 completed success on both Linux architectures with the same 13/38 checks each; Windows syntax and combined platform job passed. Independent source, signatures, artifact ZIP/file/archive hashes and paired inventory verified. Child TASK-13376.12 is Done. Initial-wizard scope and full release gates remain distinct.

Fresh signed ordinary WebUI run on that exact source passed provider validation/save/wizard chat, Markdown upload (1 succeeded, 0 failed, 5 seconds UI elapsed), Media full-text search, application chat, signed stop/start with all three containers recreated, retained-content search and a distinct new chat without re-entering settings. Provider remained Healthy; no manual backend key or server wiring. Evidence retained at /private/tmp/task13376-workflow-41e3b9bc05/workflow-evidence.json. Browser is isolated headless Chrome; user-agent reports 153.0.0.0, full patch version not captured. Owned browser/app/mock/registry cleanup verified; baseline resources, unrelated running PostgreSQL services, named data/config volumes, state/images/prior evidence retained.

Additional upload Search in Knowledge action reproducibly returns 401 Authentication required on QA history. Same browser-cookie profile request returns 200, history 401. Source and a controlled diagnostic identify TokenScopeGuard resolving missing admin principal only with Authorization present, before the later cookie-aware user dependency runs. Proposed bounded canonical-principal resolution in its existing admin check is prepared in the acceptance review and awaits requester approval; no auth code changed. New ordinary evidence conservatively has bounded_required_workflow_passed=true but passed=false/planned_setup_complete=false/full_product_qualification=false/G12=false. Parent stays In Progress for this observed error, native Windows Docker host and full native/core-format/release gates. No release, promotion or merge.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
