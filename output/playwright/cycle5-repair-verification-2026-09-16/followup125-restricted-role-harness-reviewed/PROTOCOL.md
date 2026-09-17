# Next fresh matrix — recovered protocol, not execution

Prepared 2026-09-16 under parent TASK13260. Read-only source/evidence review only; no test, inference, browser, runtime, worktree, task/tracker or production changes. This document does not open the [mandatory full-UAT entry gate](../../IMPLEMENTATION_PLAN_uat_cycle_5.md#mandatory-entry-gate-for-another-full-uat). Parent must finish repairs and original-scenario acceptance, reconcile the ledger, and record the released frozen revision first.

Repository HEAD observed at handoff: `f29ffd480255b4ec4e2a50ff22931e0dd392fff0`; this is preparation provenance, not the next run's frozen revision. All35 local report links were checked for existing targets without executing their contents.

## 1. A/B/C naming: preserve the uncertainty

No explicit mapping from A/B/C to three workflow loops was found in the searched current frontend UAT/E2E/integration sources. This is a bounded search result, not a claim that the user's intended testset cannot exist elsewhere. The accepted working protocol already uses concrete journey names; no clarification is needed to prepare it.

The verified historical A/B/C definition is **coverage tiers**, from [E2E coverage design, Priority Tiers](../../Docs/Plans/2026-03-12-e2e-test-coverage-expansion-design.md#priority-tiers):

| Historical label | Exact feature set | Original style |
|---|---|---|
| A, daily use | Chat+RAG, Media, Settings, Notes, Flashcards | Page objects + workflow tests |
| B, regular use | Characters, Audio, Watchlists, Collections, Content Review | Inline workflow scripts |
| C, occasional | Evaluations, Prompt Studio, Agents/ACP, Writing, Admin | Minimal smoke-plus scripts |

Current [Playwright projects](../../apps/tldw-frontend/playwright.config.ts) use numeric tiers1–5 plus `journeys`, not a documented equivalent mapping. [Onboarding scenarios](../../apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts) define only `tierAScenarios`, containing10 first-run/recovery scenarios; they do not define B/C loops.

Search scope: current `apps/tldw-frontend`, `apps/packages`, `apps/test-utils`, frontend/extension Markdown documentation, `apps/Testing_Guide.md`, frontend UAT runners, E2E specs and integration test sources; cross-checked the historical coverage design, running tracker, and retained cycle5 controller/single/multi reports. Case-insensitive patterns covered `(loop|workflow|journey|test.?set|tier)[ _:-]*[ABC]`, `tier[ABC]Scenarios`, and `A/B/C` over `.ts/.tsx/.mjs/.md`. Current relevant hits were `tierAScenarios` and its runner tests; an extension PRD's A/B/C/D/F/K prompt-budget labels were unrelated. See the [tracker's executable inventory and explicit uncertainty](../../Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md#executable-workflow-inventory-for-the-next-uat) and [retained execution checklist](../../output/playwright/cycle5-full-uat-2026-09-16/multi/cycle5-multi-native-execution-checklist.md).

## 2. Four fresh cells and the accepted twelve rows

Apply the same named rows to **SQLite single-user, SQLite multi-user, PostgreSQL single-user, PostgreSQL multi-user**. Each cell needs fresh configuration/data/browser state; account-isolation rows use Alice/Bob/admin only in multi-user cells. Record Pass/Fail/Blocked/Partial for every applicable boundary, with source revision and evidence. The following summarizes the existing [cycle5 twelve-row protocol](../../output/playwright/cycle5-full-uat-2026-09-16/multi/cycle5-multi-native-execution-checklist.md#twelve-execution-rows), not new acceptance requirements.

| Row | Recovered checklist | Authoritative source |
|---|---|---|
| 1 | Fresh visible setup/provider discovery/first real ordinary Chat. Multi uses documented operator bootstrap/configuration, actual admin UI user creation and normal login; record adaptations/restarts. | Cycle5 rows1–2; [setup specs](../../apps/tldw-frontend/e2e/onboarding-uat/setup-happy-path.spec.ts) |
| 2 | Auth reload, logout/Disconnect, offline/reconnect and owned-state recovery. Multi: natural token expiry from observed issuance plus `expires_in`, distinct from active proactive refresh; separate expiry context as already specified. | Cycle5 row2 and its five preparation steps |
| 3 | Two ordinary turns, canonical persistence/reload; controlled real provider failure→Retry uses the same user identity, correct context, one answer and stable reload. Retain image/visibility controls and any outstanding limits separately. | Cycle5 row3; [real Chat cockpit](../../apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts) |
| 4 | Fresh public synthetic file→ingest/chunks→full-text search→useful cited QA→numbered citation/source excerpt→actual Media-to-Chat handoff with grounded final answer. Include minimize/resume and honest warning/failure behavior. | Cycle5 row4; [shared workflows](../../apps/test-utils/real-server-workflows.ts) |
| 5 | Exact URL `https://en.wikipedia.org/wiki/Playwright_(software)`; confirm article rather than denial-page content; search `Playwright`; ask `What is Playwright? Use the ingested content to answer.` If external access is denied, record it and block dependent steps; no substitute counts as this row. | [ingest-search-chat.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/ingest-search-chat.spec.ts) |
| 6 | Biology Note→exactly5 grounded generated/saved cards→five distinct revealed/rated cards→completion/scheduling/session count5→normal reload. Wait for each settled next identity; retain actual event counts. | [notes-flashcards.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/notes-flashcards.spec.ts); cycle5 row6 tightens native evidence beyond the spec's positive-count assertions |
| 7 | Save/sync pirate Prompt, actual Use in Chat/System Instruction, verify actual request application, real pirate answer with literal `ARRR`, and reload. | [prompts-chat.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/prompts-chat.spec.ts); cycle5 row7 supplies the explicit apply/payload acceptance missing in the executable spec |
| 8 | Create TestBot, library Chat entry/character replacement from prior context, real `complete-v2` with intended character, final `BEEP BOOP`, canonical reload; ordinary/character transitions and truthful provider recovery. | [character-chat.spec.ts](../../apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts); cycle5 row8 |
| 9 | Completed Chat answer→Note with clean final text and exact source→open linked conversation; Chat→reviewed card→Study; mixed deck/undecked totals, source link, reload. Exercise exposed practice/scheduled/early-End/re-rate controls with exact event deltas. | Active shared tests at [3511](../../apps/test-utils/real-server-workflows.ts#L3511), [3970](../../apps/test-utils/real-server-workflows.ts#L3970); cycle5 row9 |
| 10 | Source→real analysis→Multi-Item Review→changed reanalysis/save/reload; failure retains prior analysis and shows an error. | Active shared test [4425](../../apps/test-utils/real-server-workflows.ts#L4425) |
| 11 | Permission-aware sole-item soft delete→empty active list→dated Trash→restore exact original. Ordinary user's denial remains honest; use admin for authorized deletion, without changing roles to bypass policy. | Active shared test [4274](../../apps/test-utils/real-server-workflows.ts#L4274); cycle5 row11 |
| 12 | Reciprocal Alice/Bob browser metadata/content/draft/handoff isolation through logout/login/reload/Back; own positive reads/writes then valid foreign denials; media UUID/content comparison; Chat/job/QA ownership; confidential synthetic negative with public retrieval positive. | Cycle5 row12; [controller's limits and adaptations](../../output/playwright/cycle5-full-uat-2026-09-16/controller/cycle5-controller-FINAL_REPORT.md#limits-and-adaptations) |

Exact row6 source facts, blank-line separated: `The mitochondria is the powerhouse of the cell.` / `DNA stands for deoxyribonucleic acid.` / `Photosynthesis converts light energy into chemical energy.` / `The human body has 206 bones.` / `Water boils at 100 degrees Celsius at sea level.`

Exact row7 instruction: `You are a pirate. Respond to everything in pirate speak. Always say ARRR at least once.` Question: `Tell me about the weather today.` Exact row8 instruction: `You are E2E-TestBot. Always respond with exactly: BEEP BOOP.` Use fresh unique record names/IDs rather than old native IDs.

Optional STT/TTS/MCP/Evaluations/Watchlists/extensions and clean-machine installation were not certified by cycle5. They remain separately identified coverage, not silently added to or passed by these twelve rows. Likewise historical vision/hidden-tab/Wikipedia limits are historical observations: report the next released environment's actual result, without copying old blocked/pass status.

## 3. Existing executable sets and mock boundaries

Commands below are references for later execution, **not run during this preparation**. CWD is `apps/tldw-frontend`; see [package scripts](../../apps/tldw-frontend/package.json), [testing guide](../../apps/Testing_Guide.md), and actual config rather than the guide's older default ports.

| Existing command | Exact set / important limit |
|---|---|
| `bun run e2e:tier1` … `e2e:tier5` | Corresponding `tier-1-critical`, `tier-2-features`, `tier-3-automation`, `tier-4-admin`, `tier-5-specialized` directories. These are mixed suites; project membership alone does not prove unmocked backend/provider use. |
| `bun run e2e:journeys` | All8 files listed below. Core named journeys use configured backend/model, but `authedPage` seeds single-user authentication/onboarding and a test bypass; standard title-settings stubbing may apply. Watchlist journey explicitly fulfills watchlist/notification APIs. Server/model/UI-unavailable skip branches exist. |
| `bun run e2e:critical` / `e2e:features` / `e2e:admin` | Exactly tier1+journeys / tier2+3 / tier4+5. Historical A/B/C cannot be inferred from these aliases. |
| `bun run e2e:all-tiers` | Numeric tiers1–5 plusjourneys; it does not include all root `e2e/workflows/*.spec.ts` files or the shared real-server wrapper. |
| `bunx playwright test e2e/real-server-workflows.spec.ts --project=chromium --workers=1` | **Four active shared tests:** Chat→Note/backlink; Chat→card/review; Media trash; ingestion→analysis→Review→reanalysis. The17-item `LEGACY_REAL_SERVER_WORKFLOW_TITLES` constant is not17 registered tests. Uses the configured server/provider without API fulfillment in this implementation; downstream can still be mock if configured that way. Requires `TLDW_E2E_SERVER_URL` and private `TLDW_E2E_API_KEY`, seeds single-user/device/onboarding state, and can skip unavailable config/capability. Does not certify fresh multi-user login unchanged. |
| `bun run e2e:onboarding:uat -- --viewport all` | Real app/backend path with **mock OpenAI downstream**. Ten `tierAScenarios`, selected via `--scenario`; desktop/mobile support varies by scenario. Runner creates SQLite single-user profiles and launches the repository mock server. It is not real-model or PostgreSQL/multi-user certification. |
| `bun run uat:live-tiers -- --projects=tier-1,tier-2,tier-3 --workers=1` | Despite “live”, [runner](../../apps/tldw-frontend/scripts/live-tier-uat/run.mjs) launches `mock_openai.server`, aliases providers to it, and uses a deterministic ACP subprocess. It inventories application API mocks and rejects skips by default. `--grep`, partial projects or `--allow-skips` are explicitly non-certifying. `--list-only` is inventory, not acceptance. |
| `bun run e2e:chat-cockpit:real:focused` | Five selected live cockpit/model/stream-stop-regenerate cases, one worker, explicit no-skip report check. Useful bounded provider controls, not the whole matrix. |

[Shared seed](../../apps/tldw-frontend/e2e/real-server-workflows.spec.ts), [authedPage fixture](../../apps/tldw-frontend/e2e/utils/fixtures.ts), [auth seeding/stubs](../../apps/tldw-frontend/e2e/utils/helpers.ts), and [Watchlist fulfillment](../../apps/tldw-frontend/e2e/workflows/journeys/watchlist-ingest-notify.spec.ts) explain why these executable successes cannot replace native fresh setup/ownership/provenance checks. Character journey can accept provider-configuration recovery without a successful answer; the native TestBot row still needs the actual answer. The exact Wikipedia and Notes/Prompt assertions are weaker than the accepted native protocol as noted above. Skips/blocked steps never count as passes.

The `test:integration` package alias points at [run-frontend-integration.sh](../../Helper_Scripts/run-frontend-integration.sh), whose static `FRONTEND_DIR` is still `admin-ui`. It is a separate backend/unit/smoke helper, not the recovered A/B/C matrix runner; do not launch it as an assumed full-UAT equivalent.

### Exact current project file inventory

All filenames below are under [e2e/workflows](../../apps/tldw-frontend/e2e/workflows); `.spec.ts` suffix omitted for compactness. This is filesystem/config inventory, not collected/running test counts.

- **tier1 (2 files):** notes; settings-core.
- **tier2 (21):** audio-alias; audio-studio; audiobook-studio; characters; chatbooks-full-account-roundtrip; chatbooks; content-review; data-tables; document-workspace; documentation; evaluations; flashcards; kanban; mcp-hub; prompts-workspace; quiz; sources; speech-playground; stt-transcription; tts-synthesis; writing-playground.
- **tier3 (5):** acp-playground; agent-registry; agent-tasks; chat-workflows; workflow-editor.
- **tier4 (11):** admin-data-ops; admin-llamacpp; admin-maintenance; admin-mlx; admin-orgs; admin-overview; admin-server; notifications; privileges; profile-companion; settings-full.
- **tier5 (12):** chunking-playground; claims-review; journalists; model-playground; moderation-responsive; moderation-review-power-user; moderation-review; moderation-routes; osint; repo2txt; researchers; skills.
- **journeys (8):** character-chat-phase6; character-chat-phase7-readiness; character-chat; ingest-evaluate-review; ingest-search-chat; notes-flashcards; prompts-chat; watchlist-ingest-notify.

## 4. Isolation before the next freeze

The existing [recovery audit](../fresh-uat-recovery-20260916/RECOVERY-AUDIT.md#material-isolation-limits) and [entry gate](../../IMPLEMENTATION_PLAN_uat_cycle_5.md) already require resolving repository-relative state before full fresh coverage.

- Freeze an isolated **source root**, not only CWD/configuration. [get_project_root](../../tldw_Server_API/app/core/Utils/Utils.py#L149) walks from resolved `__file__`; changing CWD cannot redirect it. Consequently concurrent matrix cells sharing one imported source root can still share the following state. Use separately isolated frozen roots for concurrent cells, or an explicitly isolated serial arrangement; do not import the mutable original checkout through a dependency/source alias. This is the direct consequence of the inspected path resolution, not a new product requirement.
- [System operations](../../tldw_Server_API/app/services/admin_system_ops_service.py#L40): `<source-root>/Databases/system_ops.json` plus lock; ordinary authentication consults maintenance state. An inactive existing store was accepted only for targeted recovery, not full fresh isolation.
- [Document upload drafts](../../tldw_Server_API/app/core/Ingestion_Media_Processing/document_upload_drafts.py#L75): `<source-root>/Databases/document_upload_drafts.db`; no inspected environment override. [Scraper cookies/hash state](../../tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py#L462): `<source-root>/Databases/webscraper/`; no automatic redirection from the per-user DB setting.
- [Workflow artifacts](../../tldw_Server_API/app/core/Workflows/adapters/_common.py#L471): set supported `WORKFLOWS_ARTIFACTS_DIR` (or existing singular alias) to the private profile; otherwise defaults to source-root `Databases/artifacts`.
- Keep the already audited private configuration/dotenv allowlist, Auth/content/per-user DBs, logs/temp/cache/source/upload paths, distinct browser storage and unique real Next build directory. The recovery audit records why a cache symlink broke module resolution. Reused Python/frontend dependencies and external model files must be disclosed; this certifies fresh state/workflows, not a clean dependency/OS install.
- PostgreSQL cells use the official fixture lifecycle with separate fresh auth/content databases per profile, retained by the existing fixture-holder pattern; application environments do not inherit pytest/test/mock flags. `TLDW_TEST_POSTGRES_REQUIRED=1` belongs to fixture/test execution. A skip or unavailable PG is a gap, not coverage. Do not recycle the targeted-acceptance profiles as fresh cells.
- Parent owns exact ports/PIDs, lifecycle/outage restoration and one-at-a-time real-provider work. Record real provider/model/capability metadata and actual successful inference; an inventory or health200 alone does not certify inference. Native application env must omit `CHAT_FORCE_MOCK` and inherited test switches. Do not expose private credentials in reports.

## 5. Evidence discipline carried forward

For each row/cell retain exact inputs/settings, timestamps, request status and safe body evidence, canonical IDs/counts, final rendered answer separately from reasoning, normal reload, screenshots when useful, console attribution and mutations/adaptations. Preserve original failures and any blocked dependency; continue independent rows. Keep source frozen during native execution; later repairs follow review. Use the existing secret-scan/allowlist/manifest practice before retention. Three failed attempts trigger investigation, not repeated random inference. Targeted repair evidence remains distinct from the next full matrix.
