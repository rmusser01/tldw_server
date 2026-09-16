# Stage3 reconciliation follow-up:095 acceptance scope and retained evidence

2026-09-16T14:52:30.306Z. Product under controller acceptance remains`3c30685611`. Read-only: no repository/task edits, tests, browser/runtime actions or inference. Original125-row audit remains intact; this report corrects two recommendations.

## UAT095: verified for its original acceptance scope

**My prior missing-native criterion was too strong.** UAT095 was discovered by a synthetic response through the actual client → proxy → request core → fetch boundary. No native backend disclosure was originally observed. [TASK13260.36](</Users/macbook-dev/Documents/GitHub/tldw_server2/backlog/tasks/task-13260.36 - Align-Chat-completion-error-handling-and-stale-regression-coverage.md>) AC1–3 require byte-preserving successful content; actual failed transport retaining status/cancellation/actionable guidance while excluding diagnostic paths; meaningful rejection controls and independent review. Neither those ACs nor the task-specific entry in [cycle3 plan](</Users/macbook-dev/Documents/GitHub/tldw_server2/IMPLEMENTATION_PLAN_uat_cycle_3.md>) requires a launched browser to receive a private-path sentinel. The design contains no additional095-specific native-sentinel requirement.

The evidence bundle's statements that native/extension acceptance was not performed are truthful coverage limits. They do not by themselves establish a new mandatory reproduction surface. The defensible disposition is **verified for the originally observed transport defect**, not a native-disclosure pass. This is an explicit criteria assessment supported by real-boundary regressions and independent review, not closure from a stale Done label or generic successful Chat.

### Evidence against each criterion

| Criterion | Retained evidence | What it establishes |
|---|---|---|
| Preserve successful output | [repair-report.md](</Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle3-repair-verification-2026-09-15/chat-errors095/repair-report.md>) and [permanent proxy tests](</Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/__tests__/background-proxy.test.ts>) | Exact successful JSON/assistant path/code/error-like content remains intact; no failure diagnostic is written. |
| Actual failed transport | [red-confirmed.log](</Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle3-repair-verification-2026-09-15/chat-errors095/red-confirmed.log>) and [independent-final-tests.log](</Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle3-repair-verification-2026-09-15/chat-errors095/independent-final-tests.log>) | HTTP500 direct/extension rejection and warning/stored-diagnostic paths redact sentinels;422 returnResponse retains actionable validation/nested data/status/Retry-After; exact endpoint/method negatives retain their established contract. |
| User-visible sink | [quick-test-red.log](</Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle3-repair-verification-2026-09-15/chat-errors095/quick-test-red.log>) → [quick-test-green.log](</Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle3-repair-verification-2026-09-15/chat-errors095/quick-test-green.log>); [actual Quick Test hook control](</Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Prompt/__tests__/usePromptInteractions.quick-test-errors.test.tsx>) | Real hook/client/proxy/parser passes sanitized actionable text to the notification sink; original source reproduces the leak; successful output positive control preserves bytes. |
| Cancellation and independent review | [independent-review.md](</Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle3-repair-verification-2026-09-15/chat-errors095/independent-review.md>) and [independent-cancellation-final.log](</Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle3-repair-verification-2026-09-15/chat-errors095/independent-cancellation-final.log>) | The review's real499 multiline cancellation regression was corrected; unchanged2-case probe preserves AbortError/REQUEST_ABORTED and no ordinary warning/diagnostic. Independent217/7 pass; counts overlap. |

The [original private sentinel investigation](</Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle3-repair-verification-2026-09-15/chat-errors095/initial-sanitization-investigation.md>) is historical characterization that deliberately **expects the old leak**. It must not be misrepresented as a final negative test. Permanent failure controls and original-source Quick Test RED → corrected GREEN supply the repaired expectation. Current source still has the failed canonical POST gate and pre-sanitation cancellation handling at [background-proxy.ts](</Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/background-proxy.ts>) lines1001–1002 and1109–1154. No new run was performed for this follow-up.

Remaining limits: synthetic HTTP rather than a live server; mocked extension messaging rather than a launched worker; actual hook/notification arguments rather than native toast pixels. No native disclosure, general successful-body sanitation or popup-lifetime certification is claimed. These are not newly added095 gates.

## UAT056: exact retained native acceptance found

**Change awaiting acceptance → verified.** [cycle4 single running tracker](</Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-full-uat-2026-09-16/single/RUNNING_TRACKER.md>) line31 records actual Next with blank analysis provider staying on Configure and focusing the field, followed by successful configured-provider continuation. [exact provider-required snapshot](</Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-full-uat-2026-09-16/single/evidence/uat-cycle4-single-provider-required.txt>) independently shows:

- Configure current (line225), Review disabled (line228).
- Blank Analysis provider invalid/active (lines274–277).
- Explicit “Choose an analysis provider before running ingest analysis” alert (line279).

This is the original negative transition after`4a21a23540`, on cycle4 product`7c9409fad2`; it does not rely on the filename alone or just a valid-provider ingestion. Controller additionally reports a fresh equivalent`3c30685611` control (single136-ingest-provider-blocked); it is supplementary and not needed to recover this missed retained proof.

## Other pending records: bounded search result

| Records | Defensible status from this follow-up |
|---|---|
|012,016,068,074,104|Controller reports new native passes, including127 terminal resume for104. Those fresh artifacts should supply their own acceptance rows; this follow-up does not invent paths or independently certify unseen captures.|
|013,015,055|Controller is collecting Home producer, Temp/settings-request and Manage controls. No duplicate browser work or substituted proof here.|
|020|[fresh setup snapshot](</Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle5-full-uat-2026-09-16/single/cycle5-single-native-001-fresh.txt>) and cycle4 initial snapshot show collapsed warning-detail groups. Neither proves the explanatory warning text; keep this specific presentation gap until expanded details or exact native response evidence is retained.|
|024|No post-repair native rejected Flashcard verification/raw-JSON control found. Successful generation does not replace it.|
|031|No exact native failed tracked-greeting save/guard control found. Later normal Retry/export identity evidence is related but not the same action.|
|064|Clean URL/exact Note transfer/generation passes. [task13](</Users/macbook-dev/Documents/GitHub/tldw_server2/backlog/tasks/task-13260.13 - Protect-Notes-to-Flashcards-content-transfer-across-navigation-and-accounts.md>) explicitly leaves live cross-account transfer pending; cycle3 Bob history artifacts are pre-repair failure evidence. Do not close from unrelated Notes/QA account isolation.|
|114,118|Remain blocked. Current true-hidden-tab tooling limit and stored-image recovery limitation are distinct from095's synthetic-origin criterion correction.|

These two corrections alone change the prior count from109/14/2 to **111 verified /12 awaiting acceptance /2 blocked** (0 unresolved implementations), before controller's fresh native results. The companion JSON provides exact paths and SHA256 snapshots of the19 inputs read for this follow-up. This is not a new full UAT or a gate-open decision.
