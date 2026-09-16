# Cycle5 bounded multi-user repair acceptance — complete

Frozen product: **3c30685611**. API18501/UI18581 retain the original cycle5 server data and configuration. The original browser profile was unrecoverable, so this run uses a new persistent browser profile and normal existing-Alice UI login at14:06:32Z. This is targeted repair acceptance, not full fresh UAT. No source, server configuration, runtime, browser token, storage or clock changes. Root coordinated real inference and, after two natural rotations, revoked only Alice’s current session through the canonical admin endpoint.

## Results

| Repair | Result | Native evidence |
|---|---|---|
| UAT129 Prompt collections | PASS | Valid Alice GET collections200; valid empty collection list and Collections UI. Artifacts004–006. No old local Prompt mirror recovery claim. |
| UAT132 hyphen filtering | PASS | Exact hyphenated Aurora title returns only one owned match, excluding unrelated Indigo. Unmatched hyphen marker returns zero. Actual POST200 payload/results010–016. |
| UAT127 completed ingest/resume | PASS after explicitly authorized reassessment | Correct-provider job8: actual Minimize → usable Notes → resume active job → terminal Warning Results in59s. One POST; sourceid5 intact, safe truncation warning, completed Results reopen. Artifacts111–127. Original lost browser attachment not recovered. |
| UAT131 first saved greeting | PASS | Fresh ordinary saved Pirate chat16679409-c76c-4fde-8b38-b993939711be. Real LLaMa.cpp /chat/completions200 returnsARRR. Reloaded canonical response has four rows: system, one greeting, one user and one answer. Artifacts056–065. |
| UAT068 saved-character replacement | PASS | Exact original saved TestBot4 route changes to Default Assistant1, removing old chatId. Real complete-v2 creates0735df9e-26ae-48a9-89ab-447cb1cd0e4b and answersMAPLE READY. Reload retains new messages; original TestBot IDs, versions and content hashes are unchanged. Artifacts068–084. |
| UAT016 owned Note admin probe | PASS | Alice Biology note4ce0fe3d-f7cb-4acc-bee3-410681ded957 loads GET200, savedVersion1. Narrow44-request window has no admin/title-policy requests. Artifacts087–094. |
| UAT012 unavailable Billing feature | PASS, scoped UI/request check | No Billing control in Settings; no billing requests in26-request window. OpenAPI lists only billing/subscriptions and admin/billing/analytics, not the complete feature surface. Artifacts095–100. |
| UAT134 natural active expiry and terminal invalidation | PASS, initiator limitation retained | Two natural research-runs401 → refresh200 → readiness sessions200 sequences, at14:36Z and15:07Z; signed-in Chat preserved. Canonical revocation of only Alice session12 caused refresh401 and automatic Sign in. Authenticated request window stayed unchanged for approximately90s. Evidence108–109,136–141; parent admin revocation evidence separate. Neither successful rotation was initiated by readiness itself. |

## Interpretation and adaptations

The original ingestion attachment was local to the lost browser profile; no Recent Jobs/resume control was visible in the new wizard. Three new attempts were retained honestly: quick success, correct duplicate skip, then a provider warning. The final warning is expected configuration failure: custom-openai-api was explicitly selected, but the preserved multi profile configures real inference under llama.cpp. Root confirmed custom upstream401. Results correctly surfaced the warning and preserved the source; this does not pass active Minimize/reattach. Testing stopped after three attempts. Root then explicitly redispatched one reassessed attempt with the correct llama.cpp provider in a separate tab. That new job8 passed actual Minimize, active resume, terminal Warning projection, source handoff and completed-result reopening. The UI script had a final observation-only URL-constructor error after the awaited UI actions completed; follow-up captures separately establish the hidden dialog and usable Notes. The main session remained intact.

For UAT131, the normal UI sequence was Default Assistant selection → runtime Clear assistant → ordinary saved conversation → Conversation Settings Pirate instruction → first send. Canonical greetingd1e5d53b-c60a-463a-8881-81bbe8086a7f occurs exactly once. Old damaged conversations were not changed.

For UAT068, the original saved URL from retained evidence was used after an ordinary history/reload control. Replacement changed identity and route before the real send. The old conversationedaa65df-dd08-4668-a4cd-affadd033995 was reopened afterward; its canonical message IDs, sender, version and content SHA256 values match the before capture exactly (artifact084).

New-profile Notes onboarding briefly intercepted a click; visible Skip tour resolved it. A Next development-tool portal overlapped the sidebar Settings button; the header Open settings action worked. Opening Chat Settings also emitted one nonblocking AntD InputNumber addonBefore deprecation, recorded by root as UAT140/task13260.79. None was silently treated as a successful click or repaired during acceptance.

## Natural expiry and terminal invalidation details

Alice signed in through the UI at14:06:32Z. The first expired research-runs request724 returned401; refresh725 returned200 and sessions736/752 returned200. The second expired research-runs request1712 returned401; refresh1713 returned200 and sessions1735 returned200. The original Chat remained signed in with Notifications active and its two messages intact. This proves repeated natural renewal in the active browser context; it does not independently prove readiness apiSend initiated either refresh.

Root then used the canonical admin endpoint to revoke only Alice user2 session12 (created14:06:28). The DELETE returned200; older sessions were left untouched. The browser’s buddies1761/1762 requests returned401, refresh1763 returned401, and the app automatically navigated to `/login`. Captures139 and140 are byte-identical across approximately90seconds, with no additional sessions, refresh or research-runs requests. At15:09:54Z, the visible page remained Sign in. No reload, storage injection, token manipulation or clock change was used. Parent evidence: `/private/tmp/cycle5-repair-admin-session-revoke.json`.

## Evidence integrity

Artifacts use `/private/tmp/cycle5-repair-native-multi-*`. The running chronology is in `cycle5-repair-native-multi-RUNNING_TRACKER.md`. A scoped scan of144 current artifacts found no runtime-secret or JWT-shaped values. Public synthetic fixtures and canonical IDs are retained; no request headers, passwords or authentication token bodies were captured. Browser/provider acceptance uses actual UI and real inference, not mock responses.

Acceptance finished at15:09:54Z. Browser remains on Sign in after scoped session revocation. Multi source-freeze ownership was released to root; no further browser actions are planned.
