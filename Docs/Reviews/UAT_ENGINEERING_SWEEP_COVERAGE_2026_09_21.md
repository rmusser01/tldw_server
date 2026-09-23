# Engineering sweep: baseline and coverage audit

**Authoritative task:** TASK13260.278; baseline/audit TASK13260.278.1. **Date:** 2026-09-21.

The requester approved a comprehensive engineering sweep before further full UAT and migrated execution into the main UAT task. This changes sequence, not the required workflows or PostgreSQL coverage. The [implementation plan](../../IMPLEMENTATION_PLAN_uat_engineering_sweep_20260921.md) preserves that approval and its gates. The [release playbook](../Development/RELEASE_UAT_PLAYBOOK.md) is the workflow authority; task13262 produced a specification, not a complete runner or binary fixture pack.

**Current outcome:** Source checkpoint verified. Family inventory created. Test discovery and assertion gaps identified. Latest-dev integration, complete call-path review, executable case-manifest freeze, integrated regression and full UAT remain outstanding. No UAT browser or application workflow ran during this audit.

## 1. Revision and preservation checkpoint

| Item | Recorded value |
| --- | --- |
| Working branch | `codex/post2970-uat-20260920` |
| HEAD | `3c871df7173d7b87b64bbb421882c2088542cb7d` |
| Cached `origin/dev` | `08e980a453d12155cccf10d0c5eb7fe32d05ac75` |
| Common ancestor | `d72b1d2850ea947b6d12cac19f6b95867b68a580` |
| Cached divergence | Ahead9/behind111; not a fresh remote comparison |
| Native candidates10/11/12 base | `2896dac108307cb437c91d9665700961da97b390` |
| Checkpoint manifest SHA-256 | `b8a4f5d1afa29497daf1f6f7a9d6fb228fdcbde921957b4643f4cba0afba6720` |
| Tracked binary patch SHA-256 | `5db19875a12c9f114438cc6a964c69f486d800750d42300bf1df813ffab61ecb` |
| Incremental commit bundle SHA-256 | `1a4e269eae0e495bffdb42b0818f08c67b603bb1b8b4431abf628eb914da20c8` |

The private packet `.tmp/uat-engineering-sweep-20260921/checkpoint/` contains184 copied changed files, separate HEAD/index/working patches, original status, per-file hashes and a verified incremental commit bundle requiring the recorded ancestor. Copy verification found0mismatches. HEAD and the empty index patch remained unchanged. Later audit/task documents are new work after this timestamp and are not falsely described as part of that checkpoint.

This is a source/commit checkpoint, not a database backup. Twenty-one runtime/cache/private entries remain in place. Running API/browser/official PostgreSQL fixture owners and llama.cpp were not changed. Existing failed browser profiles and immutable candidate packets remain attached to their original hashes. Do not run broad cleanup or infer that a cached remote branch is latest.

### Workstream ownership

| Group at checkpoint | Entries | Handling |
| --- | ---: | --- |
| Associated with frozen UAT candidate12 | 122 | Candidate association is evidence, not individual author attribution. Reconcile task-specific changes before adopting. |
| UAT tracking/plans | 45 | Preserve task evidence and incomplete plans; do not mark pending native acceptance complete. |
| Concurrent Chatbook work | 9 | Preserve separately; no implicit inclusion in UAT repair PR. |
| Concurrent TTS work | 2 | Preserve separately; no implicit inclusion in UAT repair PR. |
| Approved playbook input | 2 | Task13262 and playbook; reuse as specification. |
| Initially unresolved ownership | 4 | Adjudication below; preserve all files. |
| Runtime/cache/private entries | 21 | Retain in place; excluded from source-copy/PR evidence. |

The new `form.model-validation.test.tsx` belongs to this main task's UAT388 repair and was one of the four initially unmatched paths. The other three are `Docs/Design/2026-09-13-second-brain-parity-roadmap.md` and task13259 (concurrent roadmap work), plus task13256 (email-upload/quota work). None is silently adopted into the UAT repair scope. Shared files can contain several UAT repairs; per-task source diffs and frozen overlay records, not broad staging, determine integration.

Among candidate12-associated files in the checkpoint, only the sidebar `form.tsx` differs from candidate12's frozen hash, reflecting UAT388. The new UAT388 test is also outside candidate12. All earlier candidate results remain valid only for their recorded source and scope.

### Existing evidence retained separately

| Repair / packet | Code review | Regression evidence | Native evidence and limit |
| --- | --- | --- | --- |
| UAT375 / candidate12 | Reviewed | 88 retained queue/caller tests; prior official SQLite/PG evidence remains linked in its task | SQLite saved seed only; interruption/account-return/metadata-retry/canonical-readback acceptance pending on both databases |
| UAT385/386 / candidate10 | Reviewed | Existing branch/image regressions and38 SQLite/PG image cases with0skips | 24 native checks on SQLite/PG; original parent/fork/image IDs and hashes retained |
| UAT387 / candidate11 | Reviewed; committed | 141 surrounding checks and recorded independent probes | 12 native title/image/reload checks on SQLite/PG |
| UAT388 / current source | Reviewed | 166 tests across9 suites; TypeScript426→426/no new diagnostics; lint0errors/no additions | Not packaged or natively verified; task stays open |

These are targeted repair receipts, not complete A/B/C passes and not results for a future latest-dev candidate. The running tracker contains the other open findings and their individual obligations.

## 2. Coverage denominator and scope

The playbook has **33 core workflows** (A12/B9/C12), **4 release-wide workflows** and **6 supplemental families**. Historical twelve-journey coverage is a subset. The [machine-readable inventory](UAT_ENGINEERING_SWEEP_INVENTORY_2026_09_21.json) records every family, specification line, candidate automation references and source hashes.

**The executable case manifest is not frozen.** Family counts and static test declarations cannot serve as executed-case totals. Each workflow still needs explicit variants, dependencies, actor, surface, cell, phase, modes, fixtures and canonical oracles. Required advertised supplemental features cannot become N/A simply because setup or automation is missing.

| Cell | New-sweep planned case count | Code-review coverage | Regression coverage | Native UAT |
| --- | --- | --- | --- | --- |
| sqlite-single | Not finalized | Not run comprehensively | Not run on integrated candidate | NOT_RUN |
| sqlite-multi | Not finalized | Not run comprehensively | Targeted historical repairs only | NOT_RUN for new sweep |
| pg-single | Not finalized | Not run comprehensively | Not run on integrated candidate | NOT_RUN |
| pg-multi | Not finalized | Not run comprehensively | Targeted historical repairs only | NOT_RUN for new sweep |

Primary surface: WebUI. Map extension options/sidepanel and every live caller of shared mechanisms explicitly; X-04 includes cross-surface continuity. Separate D deterministic integration, L actual provider integration and U UX review. Preserve installation/upgrade obligations: reused packages/model caches establish fresh application state, not clean-machine installation. Exact supported previous/oldest upgrade starting points and published installation profiles must be identified before the case manifest is frozen.

## 3. Confirmed assertion and discovery gaps

These are source-review or test-harness findings. They are not newly reproduced native product failures.

| Finding | Evidence | Required correction |
| --- | --- | --- |
| UAT389 / TASK13260.278.2 | [Content Review](../../apps/tldw-frontend/e2e/workflows/tier-2-features/content-review.spec.ts): required-draft/action early returns, swallowed expected API failures, response<500 rather than successful intended result | Own the fixture; assert edit/diff/commit content, identity/version and reload. Distinguish empty-state coverage and explicit prerequisite failure. |
| UAT390 / TASK13260.278.3 | [Ingest/search/Chat](../../apps/tldw-frontend/e2e/workflows/journeys/ingest-search-chat.spec.ts): returned mediaId is not matched in search; a generic answer containing Playwright satisfies the final assertion | Require exact source/citation/handoff IDs, supported facts, saved conversation and reload; use distractors and deterministic downstream controls. |
| UAT391 / TASK13260.278.4 | [Notes/flashcards](../../apps/tldw-frontend/e2e/workflows/journeys/notes-flashcards.spec.ts): conditional skips/absent-button no-op; positive generated/global count is sufficient | Require exactly five distinct supported saved cards, original Note/source identity, distinct reviews and scheduling/analytics; label direct content-paste adaptation. |
| UAT392 / TASK13260.278.5 | [Package selection](../../apps/tldw-frontend/package.json), [Playwright config](../../apps/tldw-frontend/playwright.config.ts), recorded collection failures | Declare complete case identities; repair collection boundaries/setup; use owned production artifacts, retries0 and exact reconciliation. |

The sidebar [queue contract test](../../apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.queue.contract.test.tsx) checks source strings. It remains useful structural evidence; the real-store, save-wrapper and queue regressions are separate, and native queue recovery is still UAT375. No additional defect is invented merely because a structural test is structural.

### Collection attempts and their limits

1. Unmodified full `--list`, with WebUI autostart disabled, exits1 and emits0suites. It discovers two incompatible Vitest files (`e2e/utils/page-objects/__tests__/SourcesPage.test.ts`, `e2e/workflows/__tests__/companion-home.web-mocks.test.ts`) and import-time requirements for onboarding and Skills configuration.
2. A diagnostic `.spec`-only selection with reserved `.invalid` collection URLs also exits1. The existing remote-auth guard correctly rejects the absent explicit key. This is a collection setup mistake, not a product defect or permission bypass.
3. The `.spec` selection with loopback port1 collection placeholders reaches the Skills module's next required environment value and exits1/0suites. After three attempts, collection was stopped and the requirements traced: Skills also requires a key, skill name and output path. Future collection needs one complete, explicit collection profile rather than repeated missing-variable retries. No endpoints were exercised or browser launched.

Raw reports/stderr remain under `.tmp/uat-engineering-sweep-20260921/`. Static source inventory is used as a limited audit input while registered-case collection is unresolved. It cannot certify the planned manifest or any workflow result.

### Reuse existing tools

- [assert-playwright-no-skips.mjs](../../apps/tldw-frontend/scripts/assert-playwright-no-skips.mjs) already rejects zero executions, skipped cases, unexpected failures and flakes. Reuse it; add exact identity reconciliation around it rather than another pass-count validator.
- [live-tier report helpers](../../apps/tldw-frontend/scripts/live-tier-uat/report.mjs) reconcile project counts and unexpected projects; exact required variant identities remain a separate obligation.
- [live-tier runner](../../apps/tldw-frontend/scripts/live-tier-uat/run.mjs) has owned-runtime helpers and retries0, but only supports tiers1–3 and uses a controlled OpenAI downstream. It is not a complete A/B/C four-cell live runner.
- [API interception inventory](../../apps/tldw-frontend/scripts/live-tier-uat/inventory-api-mocks.mjs) is reused for static source inspection. Browser-fulfilled application APIs prove UI contracts, not integration through real application persistence/workers.
- Default Playwright config starts a dev server, permits reuse and retries twice in CI. Release execution must supply and verify owned production artifacts. Preserve first-attempt and retry outcomes separately.

## 4. Review and execution gates

First trace auth/ownership, draft preservation, ordinary/Character Chat, restoration, queue dispatch, cancellation, late responses, image Retry and all save callers. Then trace linked ingestion→retrieval→Chat→Notes/study and remaining B/C workers, artifacts and permissions. Review X-01…X-04 and all shipped supplemental variants. A page load, fired request, positive count or outer completed job is insufficient.

Before another full UAT: finish the baseline/current-dev isolation work; retain exact causal regressions and appropriate official PostgreSQL results; freeze one production candidate and executable manifest; complete integrated/targeted recovery gates. Require no untriaged failures, missing required results or unresolved critical privacy/data-loss/startup defects. Record explicitly accepted exceptions without changing their failed outcome. Report every planned/passed/failed/blocked/not-run/N/A identity by workflow/variant/cell/surface/phase/revision.

Complete final upstream adoption before freezing the integrated/full-UAT candidate. Any later adoption or repair that changes source content or built artifact hashes requires a new freeze and the required integrated plus complete four-cell UAT gates before merge. A partial retest cannot certify the changed candidate. The inventory JSON has an exact `.gitignore` exception so normal staging includes this durable metadata while private Playwright packets remain ignored.

**Current external blocker:** Automatic approval review failed because Codex usage is exhausted. Native browser actions and remote/integration operations requiring escalation have not been retried through other permission paths. The PostgreSQL browser action that triggered the block never executed. Source inventory and documentation work are unaffected. No new full UAT or merge has been performed.

## 5. Complete family inventory

All rows below are **unverified for the future integrated sweep candidate**. References are reuse candidates, not coverage certification. Exact variant/case counts are still unresolved.

| Workflow | Goal | Existing automation candidates |
| --- | --- | --- |
| A-01 | Fresh setup to first successful Chat | [scenarios.ts](../../apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts), [setup-happy-path.spec.ts](../../apps/tldw-frontend/e2e/onboarding-uat/setup-happy-path.spec.ts), [single-user-cookie-lifecycle.spec.ts](../../apps/tldw-frontend/e2e/single-user-cookie-lifecycle.spec.ts), [settings-core.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts) |
| A-02 | Login, persistence, logout and connection recovery | [scenarios.ts](../../apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts), [setup-happy-path.spec.ts](../../apps/tldw-frontend/e2e/onboarding-uat/setup-happy-path.spec.ts), [single-user-cookie-lifecycle.spec.ts](../../apps/tldw-frontend/e2e/single-user-cookie-lifecycle.spec.ts), [settings-core.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts) |
| A-03 | Multi-turn Chat, stream controls and Retry | [chat-cockpit.real-server.spec.ts](../../apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts) |
| A-04 | Image attachments and capability recovery | [chat-cockpit.real-server.spec.ts](../../apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts) |
| A-05 | Ingest files/URLs and follow background progress | [ingest-search-chat.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/ingest-search-chat.spec.ts), [real-server-workflows.ts](../../apps/test-utils/real-server-workflows.ts) |
| A-06 | Media search, batch management, delete and restore | [real-server-workflows.spec.ts](../../apps/tldw-frontend/e2e/real-server-workflows.spec.ts) |
| A-07 | Search/RAG to cited answer and grounded Chat | [ingest-search-chat.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/ingest-search-chat.spec.ts), [real-server-workflows.ts](../../apps/test-utils/real-server-workflows.ts) |
| A-08 | Notes lifecycle, search and export | [notes.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-1-critical/notes.spec.ts) |
| A-09 | Chat answer to Note/card with provenance | [real-server-workflows.spec.ts](../../apps/tldw-frontend/e2e/real-server-workflows.spec.ts) |
| A-10 | Generate, review and manage five study cards | [notes-flashcards.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/notes-flashcards.spec.ts), [flashcards.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts) |
| A-11 | Study, scheduling, practice and accurate analytics | [notes-flashcards.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/notes-flashcards.spec.ts), [flashcards.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts) |
| A-12 | Settings and provider readiness reflect real behavior | [scenarios.ts](../../apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts), [setup-happy-path.spec.ts](../../apps/tldw-frontend/e2e/onboarding-uat/setup-happy-path.spec.ts), [single-user-cookie-lifecycle.spec.ts](../../apps/tldw-frontend/e2e/single-user-cookie-lifecycle.spec.ts), [settings-core.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts) |
| B-01 | Character create/edit/import/export lifecycle | [characters.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/characters.spec.ts), [character-chat.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts), [character-chat-phase7-readiness.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/character-chat-phase7-readiness.spec.ts) |
| B-02 | Character conversation and ordinary-Chat transitions | [characters.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/characters.spec.ts), [character-chat.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts), [character-chat-phase7-readiness.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/character-chat-phase7-readiness.spec.ts) |
| B-03 | World books and dictionaries affect only intended context | [world-books.spec.ts](../../apps/tldw-frontend/e2e/workflows/world-books.spec.ts), [dictionaries.spec.ts](../../apps/tldw-frontend/e2e/workflows/dictionaries.spec.ts) |
| B-04 | Text to speech, voice selection and playback | [tts-synthesis.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts), [stt-transcription.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts), [audio-studio.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/audio-studio.spec.ts) |
| B-05 | File transcription and microphone dictation | [tts-synthesis.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts), [stt-transcription.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts), [audio-studio.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/audio-studio.spec.ts) |
| B-06 | Audio Studio project to rendered export | [tts-synthesis.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts), [stt-transcription.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts), [audio-studio.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/audio-studio.spec.ts) |
| B-07 | Watchlist source to scheduled item and notification | [watchlist-ingest-notify.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/watchlist-ingest-notify.spec.ts), [watchlists-items.spec.ts](../../apps/tldw-frontend/e2e/workflows/watchlists-items.spec.ts) |
| B-08 | Collections organize and share the intended sources | [collections-stage3.spec.ts](../../apps/tldw-frontend/e2e/workflows/collections-stage3.spec.ts) |
| B-09 | Content draft review and source reanalysis | [content-review.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/content-review.spec.ts), [ContentReviewPage.ts](../../apps/tldw-frontend/e2e/utils/page-objects/ContentReviewPage.ts) |
| C-01 | Prompt library to applied Chat behavior | [prompts-chat.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/prompts-chat.spec.ts), [prompts-workspace.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/prompts-workspace.spec.ts) |
| C-02 | Prompt Studio variables, tests and revisions | [prompts-chat.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/prompts-chat.spec.ts), [prompts-workspace.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/prompts-workspace.spec.ts) |
| C-03 | Evaluation dataset to interpretable single/batch results | [evaluations.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts), [ingest-evaluate-review.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/ingest-evaluate-review.spec.ts) |
| C-04 | Agent registry to completed task and retained result | [agent-registry.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/agent-registry.spec.ts), [agent-tasks.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/agent-tasks.spec.ts), [acp-playground.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/acp-playground.spec.ts) |
| C-05 | ACP session, tool permission and workspace boundaries | [agent-registry.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/agent-registry.spec.ts), [agent-tasks.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/agent-tasks.spec.ts), [acp-playground.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/acp-playground.spec.ts) |
| C-06 | Writing session to reviewed, saved/exported revision | [writing-playground.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/writing-playground.spec.ts), [repo2txt.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-5-specialized/repo2txt.spec.ts) |
| C-07 | Repository to text bundle | [writing-playground.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/writing-playground.spec.ts), [repo2txt.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-5-specialized/repo2txt.spec.ts) |
| C-08 | Administrator account, role and quota lifecycle | [tier-4-admin](../../apps/tldw-frontend/e2e/workflows/tier-4-admin), [llamacpp-runtime-admin.spec.ts](../../apps/tldw-frontend/e2e/workflows/llamacpp-runtime-admin.spec.ts) |
| C-09 | Health, backup/restore and maintenance recovery | [tier-4-admin](../../apps/tldw-frontend/e2e/workflows/tier-4-admin), [llamacpp-runtime-admin.spec.ts](../../apps/tldw-frontend/e2e/workflows/llamacpp-runtime-admin.spec.ts) |
| C-10 | Model/runtime administration to real inference | [tier-4-admin](../../apps/tldw-frontend/e2e/workflows/tier-4-admin), [llamacpp-runtime-admin.spec.ts](../../apps/tldw-frontend/e2e/workflows/llamacpp-runtime-admin.spec.ts) |
| C-11 | MCP discovery and scoped tool execution | [mcp-hub.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts), [workflow-editor.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/workflow-editor.spec.ts), [chat-workflows.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/chat-workflows.spec.ts) |
| C-12 | Workflow/scheduled task to monitored result | [mcp-hub.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts), [workflow-editor.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/workflow-editor.spec.ts), [chat-workflows.spec.ts](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/chat-workflows.spec.ts) |
| X-01 | Upgrade an existing installation without losing work | No direct automation mapping established |
| X-02 | Reciprocal user, organization and browser-state isolation | No direct automation mapping established |
| X-03 | Usability, accessibility and responsive review | [stage4-axe-high-risk-routes.spec.ts](../../apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts), [playwright.config.ts](../../apps/tldw-frontend/playwright.config.ts) |
| X-04 | Extension capture, sidepanel and cross-surface continuity | [stage4-axe-high-risk-routes.spec.ts](../../apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts), [playwright.config.ts](../../apps/tldw-frontend/playwright.config.ts) |
| S-01 | Quiz and study assistance | No direct automation mapping established |
| S-02 | Chatbooks and portable backups | No direct automation mapping established |
| S-03 | Artifact editors | No direct automation mapping established |
| S-04 | Research and source workspaces | No direct automation mapping established |
| S-05 | Moderation and claims review | No direct automation mapping established |
| S-06 | Specialized tools and discovery | No direct automation mapping established |

### Static workflow-source inventory

The101 workflow spec files contain624 statically identifiable test declarations and413 browser-interception records using the existing inventory helper. These are source-audit counts, not registered/executed cases. Dynamic test expansion and other E2E directories remain separate.

| Project family | Spec files | Static declarations | Interception records |
| --- | ---: | ---: | ---: |
| chromium | 42 | 342 | 330 |
| journeys | 8 | 17 | 2 |
| tier-1 | 2 | 12 | 0 |
| tier-2 | 21 | 113 | 16 |
| tier-3 | 5 | 38 | 5 |
| tier-4 | 11 | 66 | 52 |
| tier-5 | 12 | 36 | 8 |

### Preserved native artifact hashes

These hashes describe the existing loaded extension directories, including their documented local host permissions. They are recorded now to preserve provenance; no browser acceptance was rerun during this audit. The JSON inventory records the tree-hash algorithm and original source-overlay hashes.

| Candidate | Native extension tree SHA-256 | Files |
| --- | --- | ---: |
| 10 | `8f137cac6e6f4285b61e6113768206207e89519dfa450c0bc1f92679dd70ba0d` | 1386 |
| 11 | `965867aa9562e989286bb6d5e97ff57dfa87d3e51dba887fb431083ac19149b8` | 1386 |
| 12 | `200ce9ff4c9a724e0d76d811d5859a0864dbc0eac23131fc1270ef71342e009f` | 1386 |
