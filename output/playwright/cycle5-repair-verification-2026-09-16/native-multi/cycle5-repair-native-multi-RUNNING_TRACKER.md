# Cycle5 bounded multi-user repair acceptance

- Frozen product revision: 3c30685611.
- Preserved cycle5 server data/config; **new browser profile**, because original profiles expired and were unrecoverable. This is targeted repair acceptance, not full fresh UAT.
- Isolated API18501/UI18581; existing Alice login through UI. Exclusive browser session cycle5-repair-multi-20260916.
- Scope: UAT129 collections,132 hyphen search,127 completed ingest/resume,131 clean saved Pirate greeting,068 actual saved-character replacement,134 natural token expiry and terminal invalidation. Root owns outages/runtime and inference coordination.
- No injected auth/storage/clock state, no source or repo changes.

## Running outcomes
Existing Alice login completed through the UI at14:06:32Z. Evidence prefix /private/tmp/cycle5-repair-native-multi-*.

## 14:08Z checkpoint
- Login: PASS existing Alice through UI; home14:06:32Z.
- UAT129: PASS original boundary GET collections200 (request243), valid empty collection list. Evidence004–006. New browser has no old local Prompt mirror; no saved-prompt recovery claim.
- UAT132: PASS positive full-text hyphen search exact owned Aurora title returns1of1 (request310), unrelated Indigo excluded; unmatched hyphen marker returns0of0(request327). Evidence008–013. No other-owner read was attempted; this checks filtering, not new isolation coverage.
- UAT127: New browser Quick Ingest opens blank Add step with no Recent Jobs/resume control visible (014). Original local job attachment is unavailable. Proceeding with parent-authorized small public no-analysis file; distinguish this from original Warning result.

## 14:20Z checkpoint
- UAT127 partial: new Quick job5 completed1s and reopened Results, Open in Media id3 preservessource. Standard job6 skipped duplicatecontent correctly; job7 unique Maple content saved with provider warning2s. Explicit custom-openai-api selected, but parent confirms this profile's real configured provider is llama.cpp; custom upstream401 is expected configuration failure, not new regression. Results warning surfaced correctly. All three jobs ended before available Minimize click; no successful active reattach claimed; original local attachment unavailable. Stop after3attempts. Evidence020–042.
- UAT131 PASS: freshDefaultAssistant→runtime Clear assistant creates saved ordinary conversation16679409-c76c-4fde-8b38-b993939711be with one neutral greeting; setPirateprompt viaConversationSettings; actual /chat/completions200(request880) LLaMa.cpp answersARRR; normalreload preserves3visible messages. Canonical GET1116 has system+one greeting+one user+one answer, total4. Greeting d1e5d53b-c60a-463a-8881-81bbe8086a7f remains once. Evidence056–065. Old damaged chats unchanged.

## 14:24Z checkpoint
- UAT068 PASS at original saved route `/chat?mode=character&characterId=4&chatId=edaa65df-dd08-4668-a4cd-affadd033995`. Actual pickerDefaultAssistant switches immediately tocharacter1 with oldchatId removed. ActualLLaMa.cpp complete-v2 request260200 createsnew0735df9e-26ae-48a9-89ab-447cb1cd0e4b, real MAPLE READY answer; normalreload retainsnew3messages. OriginalTestBot canonicalmessageids/sender/version/contenthashes identicalbefore/after. Evidence068–084.
- Real inference lease released toparent14:24Z.
- UAT134 pending: active ordinary session from14:06Z, all observed readiness sessions200 through14:23Z. No token/storage/clock manipulation; no natural expiry evidence yet. Page navigations reset CLI request inventory; snapshots preserved perphase. Remaining window will avoid hardnavigation until expiry evidence captured.
- Incidental nonblocking diagnostic during ChatSettings→Conversation: AntD InputNumber `addonBefore` deprecation. No overlay or failed workflow; retained original browserconsole line31. No source repair attempted.

## Additional acceptance requested by parent, 14:28–14:30Z
- UAT016 PASS: existing Alice Biology note4ce0fe3d-f7cb-4acc-bee3-410681ded957 opened, savedVersion1/sourcecontentintact. ActualGET200. Narrow44-requestwindow afterNotesentry has zero admin/title-policy requests. Evidence087–094. Newprofilefirst-use tour was dismissed using visibleSkipTour beforeopeningNote.
- UAT012 PASS scoped UI/request boundary: SettingsnoBillingcontrol and zero billingrequests in26-requestwindow. OpenAPI200lists onlysubscriptions/adminanalytics billingroutes; featurecapabilityis incomplete, notallroutesabsent. Evidence095–100. SidebarSettingsbutton overlappedNextdevtoolsportal; headerOpenSettingsworked. Noauthstatechange.

## First natural rotation, 14:36Z
- Request724 GET original Chat research-runs returned401; request725 POST auth/refresh returned200; subsequent readiness sessions736/752 returned200.
- Visible Chat remains signed in, original BEEP BOOP transcript intact, Notifications active. No reload, login, token/storage/clock manipulation used to recover.
- Evidence108/109. Initiating request was research-runs, not readiness apiSend itself; this distinction is retained. Second natural rotation expected around15:06Z. Terminal invalidation still pending.

## Reassessed UAT127 acceptance, parent explicitly redispatched at14:42Z
- Correct preserved provider llama.cpp selected through visible UI in a separate tab; unique public Willow fixture. One new attempt authorized after earlier configuration diagnosis and three-attempt stop.
- Actual Start processing → wait visible Minimize → click Minimize → wait hidden dialog executed in one UI script. Only the final return expression failed (`URL` unavailable in CLI sandbox); no product error. Follow-up observation proves dialog hidden and Notes visible. Existing Biology note opens normally.
- Quick Ingest resumes the same job8 at elapsed0:39; terminal Results shows Saved with warnings after59s. Actual completed job8(owner2) at14:45:10/progress100, Warning/media5. Real local analysis was truncated; safe warning displayed, no raw envelope. Source opens intact inMedia; no bad analysis saved.
- Media→Notes→QuickIngest reopens completed Warning Results. Request inventory has exactly one ingest POST. Evidence111–127. UAT127 now PASS for this newly attached actual terminal-warning resume; original lost browser attachment is still not claimed recovered.
- Main Chat tab restored visible14:46:34Z; completed auxiliary tab closed. No authentication changes. Inference lease released toparent. Second natural expiry still expected~15:06Z.

## Second natural rotation, 15:07Z
- Actual request1712 research-runs401 →1713 auth/refresh200 →1735 auth/sessions200. Chat remains signed in; Notifications active and original BEEP BOOP transcript preserved.
- Evidence136/137. Initiator again was research-runs, not readiness apiSend; no token/storage/clock manipulation, forced reload or manual refresh.
- Parent notified to revoke only current Alice session12 (created14:06:28) using canonical privileged admin endpoint. Terminal UI/polling acceptance remains pending.

## Terminal invalidation and completion, 15:08–15:10Z
- Parent canonical admin DELETE revoked only Alice user2/session12,200. Browser buddies1761/1762 returned401, refresh1763 returned401, then automatically navigated to Sign in.
- Authenticated request windows139/140 are byte-identical after approximately90s; no repeated sessions/refresh/research-runs requests. Final visible page15:09:54Z remains /login (141). No browser reload or auth injection.
- UAT134 PASS with both natural refresh initiators explicitly research-runs. All eight assigned rows have scoped PASS outcomes. No additional browser actions; multi native freeze released to root.
