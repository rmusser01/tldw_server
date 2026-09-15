# UAT034 cached-title follow-up — review requires one correction

The matching `server_chat_id` check and early cached-title publication are appropriately bounded. One restore/loader ordering issue is confirmed when combined with the approved UAT093 removal of pre-metadata selection/write waiting.

## P2: publish cached metadata before awaiting selection persistence

At `apps/packages/ui/src/hooks/usePlaygroundSessionPersistence.tsx:614–655`, restore sets the server target, awaits assistant persistence, then publishes cached assistant identity and `serverChatMetaLoaded=false`. A real server loader can finish its canonical metadata and messages during that await. The resumed restore then replaces current canonical identity with the cache and marks an already loaded conversation's metadata incomplete. Its loader can remain `loaded` with `metaLoaded=false` because the load has already completed/has messages.

Confirmed by `/private/tmp/uat034-session-loader-race.config.ts` and `/private/tmp/uat034-session-loader-race-red.log`: **1 failed /9 filtered**. This probe runs actual session persistence, actual selected-assistant hook, actual WebUI Storage/hook aliases, actual option store and actual server loader. Only HTTP, surrounding unrelated dependencies and the existing Storage write boundary are controlled. The loader transform removes only the already-approved .33 pre-metadata minimal selection/write waiter; the production loader had not yet been edited when this probe ran. It is a confirmed integration finding against that approved combined direction, not a claim that the existing global waiter permits this exact ordering.

Sequence: hold legacy selectedCharacter persistence for cached character5; canonical server response returns character6, title `Canonical title`, messages; observe canonical `metaLoaded=true` and loaded messages; release held restore; observe `{meta:false,id:'5',title:'Canonical title',load:'loaded'}`.

Small correction: publish the cached title, assistant identity and pending flag synchronously immediately after selecting the saved target, before awaiting shared assistant persistence. Retain the existing restore revision/current-owner checks after that await. Then canonical results can only replace the cache, never be overwritten by it afterward.

The original probe/config is frozen for re-review. Parent notified; no production/tests/runtime/browser/commit/global-document changes by reviewer. Other title matching, cancellation and offline-cache assessment will be finalized after correction.
