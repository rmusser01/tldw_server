# A04 imported local history consent and read-only recovery

Task: TASK-13260.278.18.83.45; parent TASK13260. A04 only, A05 not started.
Candidate frozen523ef323e9f56bc31559f6c5b8e506c27e599dcb plus preserved WIP; not latest dev.
ADR assessment: existing ADR049 governs owner-bound history, explicit legacy review and readable history; no new durable ownership rule. ADR050 native lifecycle and all native STOPs remain.

## Stage1: Diagnose and define the ownership boundary
**Goal:** Reproduce silent imported-local open and identify the supported explicit consent path.
**Success Criteria:** Real normal import is committed; unowned record refuses before consent without claimed image fetch; all foreign/native guards remain.
**Tests:** Native JSON chooser/import scope red/green; public settingsHistoryId target; source/ADR assessment.
**Status:** Complete

Normal importer scope repaired separately; exact helper live green on sv1AMfmx32rLQGBRU7GEG. A sanitized fresh local import cannot open: loader rejects absent server scope, controller independently refuses absent local owner, and no local binding action exists. Actual target15s/0rows/query retained; unavailable image not reached. Existing HTTP asset send/fork admission refusal is intentional and must survive.

## Stage2: Add explicit current-profile consent with lease fences
**Goal:** Offer normal Use this local conversation action for the exact refused local target.
**Success Criteria:** Private epoch-fenced candidate, original caller lease and guarded metadata-resume callback; consume once and clear on invalidation. Reuse existing atomic transaction/abort helper to derive current browser owner only after consent, require all trust/server markers absent, validate profile/record/pending writes/lease. Native bind path unchanged; no automatic authority from imports.
**Tests:** Gateway positive plus foreign/native/pending/stale/abort controls; controller no-bind-before-consent, exact target, navigation/account/unmount refusal; composed loader read-only fallback and history/server/model/prompt/files/title adoption; recovery UI consent control.
**Status:** Complete

## Stage3: Qualify the normal imported asset path and preserve wider gates
**Goal:** Build reviewed minimal repair and observe explicit binding then genuine missing-image retrieval/error UI.
**Success Criteria:** One existing window/tab, normal UI, same profile/data/keys/API/provider; actualHTTP404 and honest readable unavailable asset with unsupported Send retained. Exact scoped accounting, no fourth unchanged import replay, all failed histories retained.
**Tests:** Configured UI/WebUI affected suites, compile/lint/security, independent actual source, one supported owned build/120startup15stop, real local consent/asset/composer workflow. No fake UAT or imported authority injection.
**Status:** In Progress

Repository/staged review/hooks/rebase/publication, other A04 variants/modes, human UX and broader43/native/installer/release criteria remain open. Heartbeat remains paused. No tracking-only commit or proof artifact.

## Implementation verification — 2026-10-13
Explicit unowned local consent now uses the existing awaited abort-aware transaction, a private exact target/lease/epoch candidate and guarded loader resume; incoming JSON cannot erase an existing destination ownership namespace. Transcript reads wait for consent. Both binding labels require a live candidate. Gateway/controller and deferred-loader regressions had causal reds; current configured WebUI five suites178pass. UI corrected full loader-history17/load15/recovery25 pass alongside prior unchanged gateway/controller passes. The initial full unit run166pass11fail is retained: missing current mock export/router/store context were corrected with original assertions retained. No actual IndexedDB commit/live missing-asset acceptance is claimed yet. New functional24-binding build/transition is source-reviewed CLEAR; all retained caps/identity/mutex/finally restoration remain.

Configured WebUI TypeScript natural0; scoped diff whitespace0. New24-binding local-consent build executing once; build-result/source-restoration/frontend transition/live consent remain unaccepted until observed.

### Actual consent / new recovery fixes
Build taQIl1dhkuJW_VVvjRIgO/compiler-token-bundle natural0/155.318s; all24bindings restored, oldowned52222 stop0.102s, new19110/owner19095 canonicalbuildready0.483s. One same tab/public target:0rows before Use; explicit consent opens1exactquestion/title and realmissingimageHTTP404. Ownership commit and readableloader acceptance is scoped live, not arbitrary realIDB rollback qualification. Missing thumbnail offered no image-specific error: new shared Antd renderer/error status, sameURL incarnation fences had realunit reds and6green. One actual unsupportedHTTP Send0completion/1question but draftblank: originalguardresultFAILED for draft preservation. Minimal currentcontroller unsupported guard BEFOREoptimisticreset, realcontroller/composer red1 thenUI30/WebUI36affectedgreen; setupfailedhandoff partial-real-iconfixture13UI/13WebUIgreen preservesassertions. TypeScript0; scopedlint0errors, inherited17Any. New27-binding combinedbuild follows independent product/operator CLEAR and actual pureoperator newtarget-staging red/green; failure returns originalOSError/restores earlierbindings/absentnewtarget. All normalcurrentcandidate/runtime/profile guards remain, no repeated import/provider redraw. Combinedliveimage message/draftretention acceptance still pending.
