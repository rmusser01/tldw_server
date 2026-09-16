# Read-only route/mode observation classification

Stable authenticated preserved Alice session, source e7bf2184d3, no new inference, sends, config/auth/store mutation, runtime restart or repo edit. Browser checks finished06:58UTC; root notified it can stop Next. Only actual navigation/history/Note clicks were used.

##1. Bare ordinary chatId URL — reproducible unsupported-route gap, not authentication

From the successfully loaded Character416f935c-4848-4725-9d3f-29cd6348bf34, navigated to /chat?chatId=c244a779-afcd-4f3d-8d89-a63065971a9b. After settling, URL namesc244 but title, displayed three rows and canonical GET200 requests still belong to416f. Authentication is stable; no401 in this probe. Thus initial idle authentication did not cause the mismatch.

Source: utils/character-chat-mode-intent.ts:33-47 returns null unless mode=character BEFORE reading chatId aliases. Playground.tsx:721-740 derives the only route chat ID from that Character-specific intent; its effect955-962 therefore ignores barechatId. pages/chat/index.tsx mounts Playground without another ordinary-ID parser. Existing persisted conversation restoration wins.

Important scope distinction: read-only search found the product's URL producer only in Playground.tsx:208-220, which includes mode=character. Ordinary history and Note consumers navigate /chat after setting owned state, not /chat?chatId=… . The bareordinary URL was introduced by the native harness earlier; no evidence that a current product link generates this URL. Classify as a confirmed unsupported ordinary deep-link contract/gap, not a newly introduced111/113 regression or failed product-generated link. If normal saved-chat deep links are required, root should explicitly track that scope.

Minimal optional design: parse owned savedconversation intent independently of Character-mode intent, giving explicit validchatId priority over persisted restoration, using existing cancellation and canonical loader; leave Character entry params and extension hash handling intact. Tests must include ordinaryURL from previousCharacter, previousordinary, malformedID, delayed identity switch, actual canonical IDs and title. Do not indiscriminately add mode=character to ordinarylinks.

##2. Character workflow remains active on canonical ordinary history/Note — definite product issue

Clicked actual visible Recentconversations row for c244. It correctly loaded ordinary c244 title, system/user/recovered answer and the prior local displayerror, and URL became /chat. Yet UI displayed Character Chat and Choose a character to start character chat. Then actual existing Note1eea11bb-c31d-448e-a83d-74004f7d540e → More/Openconversation again correctly restored c244 and retained the same false Character workflow banner.111 navigation itself remains passing. No attempted ordinarysend; no claim of native request failure.

Source: Playground.tsx:455-459 persists global chatWorkflowMode; Character route effect938-941 sets it tocharacter. characterWorkflowActive824-828 ORs this preference and local characterModeIntentActive into actual selectedassistant state, even after loading canonical ordinary metadata. Canonical loader useServerChatLoader.ts:980-998 correctly resolves no trackedassistant and clears selection. Neither that resolution nor Note handoff reconciles workflow mode. Only explicit starter selection913-914 and clearAssistant2692-2693 reset it. The mode feeds readiness and characterChatSendBlocker2934 onward and is passed to real composer4271-4272, so the wrong banner is backed by stale workflow gating, not just a stale title.

Minimal scoped design: once an owned existing saved chat's canonical metadata is resolved, derive workflow from that chat (trackedCharacter→character, ordinary/persona→standard as existing semantics require); do not let the global new-chat preference override an existing ordinary conversation. Scope/epoch guard any setter; leave explicit freshCharacter route before metadata and actualunsaved Character drafts intact. Prefer deriving existing-chat workflow over global preference writes if possible. Cover Character→ordinary via real history and Notes, coldrestoredordinary with persistedCharacter preference, knownCharacter, freshCharacter entry, delayedaccount/source resolution, metadata-pending and trueunsaved draft controls. Likely production ownership only Playground.tsx with coordinator integration tests; avoid backend/ChatTldw/human formatter.

## Evidence

/private/tmp/cycle4-route-mode-probe-direct-settled.txt and direct-requests.txt: mismatched bareURL with successful416f reads. history-result.txt: actual ordinaryhistory result with wrongCharacter banner. note-result.txt and final-requests.txt: correctordinaryNote result with samebanner and actualcanonical reads. All files use this prefix. No mocks, seed records, forced visibility or hidden UI state writes. Previous native111/121/117 acceptance bundle remains separate.
