# Targeted preserved multi-user repair verification

TASK13260 · code f83a8ba456 · started 2026-09-16 UTC.

This is targeted verification on the preserved cycle4 multi profile, not a fresh full UAT or signoff. No inference authorized for this runner without the parent's shared slot. No profile initialization/reset/config edits or product edits.

## Runtime

- Preserved isolated API 18301 and WebUI 18381 restarted with the existing launcher.
- Initial sandbox launch failed with localhost listen EPERM; restarted through authorized escalation. No product finding.
- Secrets remain in the existing 0600 runtime-private.json and are entered only by the redacting browser wrapper.

## Checks

| Check | State | Evidence / limits |
| --- | --- | --- |
| UAT058 Prompts / Characters titles | PASS | prompts-title.txt; characters-cold.txt: expected page titles |
| UAT067 cold New Character close / reopen warning | PASS bounded | character-drawer-open/reopened.txt; characters-console.txt: no disconnected-useForm warning; startup transient auth/collection401s retained |
| UAT112 create / review accessible names | PASS | Manual synthetic Create & Add Another200; clean Create/Create & Add Another names afterward; Review all due clean after save. flashcard-created.txt and flashcard-after-create-another.txt; retained card UUID6209bde3-d6c8-4eec-b6a4-ca02804b14bf in flashcard-retained-id.txt |
| UAT114 six tabs hidden stream release / visible Prompt save | NATIVE UNVERIFIED — harness blocked | Six real headed Playwright tabs/streams200 observed, but every document remained visible. Installed Playwright enables focus emulation. Both detached and retained CDP override removal changed focus but not visibility; no page visibility/event fake or source edits. Native CUA Chrome provider unavailable; cua-driver daemon failed supported relaunch; external permission probe reports Accessibility not granted with inaccurate-context caveat. No hidden-release or catch-up pass claimed. After closing only five extra test tabs, ordinary visible-tab Prompt save POST201 and Synced #2 passed (prompt-visible-save.txt/prompt-after-save.txt). Six-tab evidence files and native-harness-limitations.md |
| UAT117 existing reasoning-only character transcript | PASS for restored transcript | Actual owned saved Cedar chat bd564030-70fe-41bc-9c91-4b661408f8e3 shows missing-final-answer warning + Retry/Continue, retains expanded reasoning. reasoning-recovery.png visually inspected. Reload200 retains five canonical rows incl assistant pa_e1d1-0ac7-dcb-ae58 (14297 chars reasoning only). No inference or recovery action invoked |
| UAT102 admin Create user contextual feedback | PASS | Actual POST200; contextual “User created”; new ID4 cycle4_repair_20260916_0348 visible as default user, verified. Original admin remains unverified. admin-created.txt, admin-after-create.txt, admin-create-console.txt (0 errors/0 warnings) |

## Data impact

- Created one manual Alice flashcard: UUID6209bde3-d6c8-4eec-b6a4-ca02804b14bf, synthetic Amber question.
- Created one default-role synthetic user: ID4, username cycle4_repair_20260916_0348, example.com email; password retained only in private0600 file.
- No character created; no inference or recovery invoked; existing accounts/data preserved.
- Created one manual synthetic prompt: server ID2 (visible Synced #2), browser local IDpa_d03d-c8db-cb0-5381; title Cycle4 repair visible-tab prompt 20260916_0355. No generation.

## Follow-up 108 / 113 (2026-09-16 UTC)

- UAT113: In progress; fresh character chat entry through Alice’s existing Cedar character; real generation waiting on parent inference lease.
- UAT108: Pending; fresh ordinary chat, actual unavailable-provider failure then exact configured Gemma Retry. No simulated responses or backend configuration changes.

- UAT108 negative leg established: actual Ollama/gemma3:1b chat completion502, one synthetic user request, fresh canonical conversation81b8e878-931c-436a-ac54-37f0bf38c1cd. UI shows recoverable server error and Retry chat. Exact LLaMa.cpp/Gemma restored; Retry awaits parent inference lease. Evidence retry-negative-request-108.txt and retry-ready-working-model-108.txt.

- UAT108 follow-up FAIL canonical persistence: actual Retry200 submitted original user exactly once and no errorJSON, rendered AMBER RETRY. Normal reload GET200 returned duplicate user rows7f16c00c-19b4-4add-9e2a-8b05ad5d3e18 and9be643bf-4a42-4f6f-951e-be2aded6a78f, assistantc68f5688-ec3a-4a5c-8b52-c2f080d9541e, plus system. Outbound fix passes; same-user-once persistence fails. Parent notified immediately; no product edits.

- UAT113 PASS: fresh existing Cedar character entry had mode=character&characterId=4 without chatId; real complete-v2/providerllama/exactGemma200 +persist200 promoted chatId0a871f1b-fb60-4089-9efc-d17f5287f74a. Normal reload retained exact URL and canonical three rows: greeting57037eb4-bb7a-4470-8d5f-9a020efc4680, user4de9779e-748d-430d-a6c3-db70978eca89, assistantpa_0671-b4c6-867-55af. UI final CEDAR GUIDE: CEDAR READY retained separately from reasoning. Evidence character-response-113.txt, character-canonical-reload-113.txt, character-reloaded-113.txt.
- Shared inference lease granted after parent ingest terminal, used sequentially for108 Retry then113 first reply, and released once both terminal before reload. One actual Ollama502 negative request; two actual Gemma200 requests. New conversations81b8e878-931c-436a-ac54-37f0bf38c1cd and0a871f1b-fb60-4089-9efc-d17f5287f74a retained; original fixtures untouched.
