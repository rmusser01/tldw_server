# PostgreSQL Study: Hard counted as an analytics lapse

## Decision

**Evidence supports a new analytics-semantics issue.** The daily query classifies Hard as a lapse using `rating < 3`, even though the active schedulers record Hard as successful recall (`was_lapse=False`) and leave the card's lapse count unchanged. No current documentation located in the bounded Flashcards guide/plans and parent-tracker search defines an intentional Hard-as-lapse metric.

Important correction: **1–4 are keyboard shortcuts, not the API rating scale.** `useFlashcardShortcuts.ts:3–8` maps keys 1/2/3/4 to Again **0**, Hard **2**, Good **3**, Easy **5**. The public guide (Ratings and Scheduling Basics) documents those values; the API schema accepts integers 0–5. The supplied captures directly contain Good=3 and Hard=2; they do not include the earlier Easy POST, so no Easy=4 API claim is made.

## Native evidence

Card `c780acb5-a76d-4d60-983a-d49d3d508161`, frozen PG-single revision `8f8774e6c868b304a96d95ab82e28389c129a78b`:

- **14:49:59.405 UTC:** review POST rating=3; HTTP200 at **.434** returns version3, repetitions2, lapses0, interval10, queue review, scheduler sm2_plus. Analytics **14:49:59.503** reports reviewed2, retention100%, lapse0%.
- **14:51:00.773:** review POST rating=2; HTTP200 at **.798** returns version4, repetitions3, lapses0, interval14. Analytics **14:51:00.846** reports reviewed3, retention66.6667%, lapse33.3333%.
- `auth-outage-recovered.txt`: after recovery, **14:56:46.076** analytics still has those same three-review metrics; canonical card GETs retain version4/lapses0. This is persisted server output, not merely stale UI formatting.

Reviewed3 is expected: a scheduled re-rate adds another review event, as accepted in UAT128/TASK13260.68. The defect is classifying the Hard event as forgotten, not the count or an undo requirement. A lifetime card lapse count and today's event rate are not generally numerically equal; here the isolated Hard transition and formula establish the discrepancy.

## Source and contract

- `ChaChaNotes_DB.py:37456` computes `SUM(CASE WHEN rating < 3 THEN 1 ELSE 0 END) AS lapses_today`; 37476–37478 divides by reviewed_today and defines retention as its complement. That exact formula explains 1/3 lapse and 2/3 retention.
- The actual review path calls `_simulate_scheduler_review_transition` (37315 onward; dispatch at 278–289) and stores its **`was_lapse`** on each `flashcard_reviews` row (37348–37374). Analytics ignores this existing scheduler outcome.
- Active SM-2+ initializes `was_lapse=False`, increments lapses only for rating0 in the mature review branch (`scheduler_sm2.py:408–430`), and handles Hard2 separately by increasing repetitions/interval without adding a lapse (469 onward). FSRS likewise sets a lapse for0 and separately accepts Hard2 (`scheduler_fsrs.py:169–190`).
- The guide says Hard means recalled with effort, while Relearns/lapses counts forgetting after learning (`Flashcards_Study_Guide.md:160–177`). The UI simply formats the backend retention/lapse fields (`ReviewAnalyticsSummary.tsx:63–74`).
- A legacy `_srs_sm2_update` method still documents q<3 as lapse (ChaChaNotes_DB.py:36692 onward); it is not the scheduler invoked by the inspected current review path. This explains a plausible legacy threshold, not a current product promise that overrides the active scheduler/guide.
- Existing analytics tests use rating1 as their lapse sample, or seed raw rating1 review rows without exercising the current scheduler outcome. They preserve the old threshold but lack a real Hard transition control. Prior UAT177/TASK13260.114 repaired PostgreSQL date-expression/read-lifetime failures; its bounded acceptance did not validate Hard semantics. No existing parent-tracker finding for this mismatch was found.

## Bounded follow-up

Prefer calculating lapse events from the already persisted `was_lapse` outcome, with an explicit historical-row/null compatibility decision before edits; do not blindly rewrite old ratings or change scheduler behavior. Add actual review→analytics PostgreSQL/SQLite tests for all supported button ratings and both schedulers, mature vs learning states, repeated scheduled events, date/deck/owner filters, and historical rows. Preserve reviewed_today counting and prior transaction/date controls. Existing rating1 assumptions need classification against the accepted 0–5 API and historical data contract rather than silently changing expectations.

The adjacent session `correct_count` also uses rating>=3 (ChaChaNotes_DB.py:37378), so its intended definition should be checked during design; this audit does not assert a second native defect or broaden repair scope. No tests, source changes, native/runtime/DB actions, task changes, or acceptance claims were made. Only this ignored audit is written; parent owns tracking.

## Hash binding

All 12 selected source/history files match original PG-single archive-manifest entries. Manifest SHA-256: `f9a6d30e6a8faef5635df40d5ee026e18ebc225d344bf29b62a8bcca7f2b2f4f`.

Paths relative to `sources/pg-single`:

| Path | SHA-256 |
|---|---|
| `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` | `33f987c9e4c6acc3b1502c9f6e10dc66c6e17929f2258c4d67fa40e87b7fe06b` |
| `tldw_Server_API/app/core/Flashcards/scheduler_sm2.py` | `b009c9bd5109bfe9354285c34f48aadc4bc2c587cffe37e3e90b82e789c4609b` |
| `tldw_Server_API/app/core/Flashcards/scheduler_fsrs.py` | `3b3b0dae382d682e8d597c844bde5acbc7609c5acb830c39c91d5176285e4892` |
| `tldw_Server_API/app/api/v1/schemas/flashcards.py` | `921d6f32fe1ac399d5f89c266bc8cd2d30681e5571fd3702f0cf9489fd05c425` |
| `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardShortcuts.ts` | `79d029a4d264eafcde966e6c9c364362d1493db2513d923e33410ebb28749856` |
| `apps/packages/ui/src/components/Flashcards/components/ReviewAnalyticsSummary.tsx` | `c9adc3d60008465c148bb8c51bd3b73e118802e0e3cab890139dbf7a5e5215d5` |
| `Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md` | `2f731ab928abd289ed2672e94b392a68ade9431c790e2261747d6c72add0e477` |
| `Docs/Product/Completed/Plans/IMPLEMENTATION_PLAN_flashcards_hci_01_visibility_status_2026_02_18.md` | `bdd31c1a8909dc99136db6e2437041f7d0c345410d523e54acd3fca20145efb2` |
| `tldw_Server_API/tests/DB_Management/test_flashcard_analytics_backends.py` | `e23e82ac69b0f94754214709153f29aecc0668125748c739df7bf2dd61b5c9d5` |
| `tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py` | `a574278d41482ab8ebcee93d1ff649cde203749db2351608ce57872dc0abbca3` |
| `backlog/tasks/task-13260.114 - Repair-PostgreSQL-Study-analytics-and-review-queue-read-failures.md` | `f078f64ef835d584413d4741b488321a9c0527c7058e52676719484a56990540` |
| `backlog/tasks/task-13260.68 - Keep-scheduled-Cram-progression-stable-through-rating-and-re-rating.md` | `4b2dde00d28cdba645f36afe6b0718040dc8581d30f453cf4eb58eaa830d9275` |

Inputs relative to `native/pg-single`:

| Path | SHA-256 |
|---|---|
| `study-rerate-reloaded.txt` | `23bc30d0a787872cb68d3aba8741669bf4fa44bd647aabf409ad935266a3289f` |
| `auth-outage-recovered.txt` | `e35e66bcbd76c5f41cd081642f1ba7e8212c9ce96ec96fef228abcb4db642c00` |
