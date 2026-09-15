# UAT034 cached-title follow-up — independent re-review clear

The matching `server_chat_id` check and early cached-title publication are appropriately bounded. The ordering finding below is now fixed: all cached identity/pending metadata is published synchronously before either assistant-persistence await. No remaining material issue found in this bounded follow-up.

## P2: publish cached metadata before awaiting selection persistence

At `apps/packages/ui/src/hooks/usePlaygroundSessionPersistence.tsx:614–655`, restore sets the server target, awaits assistant persistence, then publishes cached assistant identity and `serverChatMetaLoaded=false`. A real server loader can finish its canonical metadata and messages during that await. The resumed restore then replaces current canonical identity with the cache and marks an already loaded conversation's metadata incomplete. Its loader can remain `loaded` with `metaLoaded=false` because the load has already completed/has messages.

Confirmed by `/private/tmp/uat034-session-loader-race.config.ts` and `/private/tmp/uat034-session-loader-race-red.log`: **1 failed /9 filtered**. This probe runs actual session persistence, actual selected-assistant hook, actual WebUI Storage/hook aliases, actual option store and actual server loader. Only HTTP, surrounding unrelated dependencies and the existing Storage write boundary are controlled. The loader transform removes only the already-approved .33 pre-metadata minimal selection/write waiter; the production loader had not yet been edited when this probe ran. It is a confirmed integration finding against that approved combined direction, not a claim that the existing global waiter permits this exact ordering.

Sequence: hold legacy selectedCharacter persistence for cached character5; canonical server response returns character6, title `Canonical title`, messages; observe canonical `metaLoaded=true` and loaded messages; release held restore; observe `{meta:false,id:'5',title:'Canonical title',load:'loaded'}`.

Small correction: publish the cached title, assistant identity and pending flag synchronously immediately after selecting the saved target, before awaiting shared assistant persistence. Retain the existing restore revision/current-owner checks after that await. Then canonical results can only replace the cache, never be overwritten by it afterward.

The original probe/config is frozen for re-review. Parent notified; no production/tests/runtime/browser/commit/global-document changes by reviewer. Other title matching, cancellation and offline-cache assessment will be finalized after correction.

## Final correction verification

- Original held-storage scenario/assertions now **1 passed** against the current production loader: `/private/tmp/uat034-session-loader-race-green.log`. Final trace is `{meta:true,id:6,title:"Canonical title",load:"loaded"}`. The original config remains untouched; `/private/tmp/uat034-session-loader-race-current.config.ts` only bypasses its obsolete loader transformation because .33 has now removed that scaffold in production. The actual WebUI storage/session/loader test itself is unchanged.
- **24 passed /3 suites**, `/private/tmp/uat034-session-title-independent-green.log`: session persistence10, local conversation load13 and server-sidebar context reset1. Covers matched/foreign cached title, missing local history, late explicit selection and cancellation during assistant persistence.
- Additional actual formatter/session/loader offline control **1 passed**, `/private/tmp/uat034-session-offline.config.ts` and `-green.log`: canonical metadata/messages/profile requests all fail; the matching cached title, saved target and actual formatted cached answer remain available with metadata correctly pending. It uses actual WebUI selection/storage modules and actual message formatters; Dexie read and network results remain controlled. The existing optional-profile warning is visible and is not masked.
- Scoped diff-check passed. Root owns final lint and combined compiler; no clean-typecheck or native acceptance claim.

The title is derived only from an exact local `server_chat_id` match, and publication happens under the existing restore revision checks. All cached metadata now precedes async selection persistence, so later canonical loading remains authoritative. Cancellation checks after the await remain intact. No new auth/network prerequisite or offline cache-clearing behavior was introduced. No further production/testing changes requested.

Latest test-only formatting/callback-type annotation landed just after the24-suite run started. I reran the final session persistence suite afterward: **10 passed**, `/private/tmp/uat034-session-title-final-formatted-test.log`. No production change occurred between the actual storage race/offline probes and this final source check. The original pre-correction review text is separately retained at `/private/tmp/uat034-session-title-independent-initial-red-review.md`.
