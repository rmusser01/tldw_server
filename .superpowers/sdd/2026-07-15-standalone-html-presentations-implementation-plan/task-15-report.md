# Task 15 report: inert standalone HTML presentation workspace

## Status and scope

Implemented the approved Task 15 WebUI workspace on base `45a507666d21ede1955d451aef52853f2eec5b14`, then completed the receiving-review correction package on top of `c81fcc32e11c07d1afbb15b9c2a1537791586e35`. V1 remains inert: it has no preview, execution, HTML insertion, source-derived navigation, popup, source-derived worker/module, or resource surface.

The implementation stays inside the Task 15 brief plus the controller-authorized presentation-client hardening seam and the explicitly authorized shared-router/Next-shim navigation-guard seam. No request-core, background proxy, backend, OpenAPI, dependency, Backlog, CSP, or protected-artifact change was made.

Binding preflight used throughout:

`IMPECCABLE_PREFLIGHT: context=pass product=pass command_reference=pass shape=pass image_gate=skipped:no imagery belongs in the inert editor workflow mutation=open`

## Analogue and design findings

The implementation followed these concrete repository analogues before tests were written:

1. `PresentationStudioPage.tsx` for source-free, content-kind-first detail dispatch and the existing structured project handoff.
2. `TemplateCodeEditor.tsx` for lazy Monaco loading, a render-failure boundary, textarea fallback, and local editor/model ownership.
3. `standalone-html-session-records.ts` and `useStandaloneHtmlGeneration.ts` for principal/origin-scoped 24-hour session records and lifecycle fencing.
4. `useSlidesCapabilities.ts` for authoritative current-config/current-subject derivation and fail-closed capability behavior.
5. `usePresentationStudioAutosave.tsx` and the Presentation Studio store as structured-only negative patterns that must not receive standalone source or drive HTML save.
6. `diff-worker-client.ts` / `diff.worker.ts` for an application-owned static worker and explicit controller disposal.
7. `download-blob.ts` for temporary anchor handoff, tightened here to the standalone attachment contract, one fixed anchor, and bounded early revocation.

The approved product shape was preserved: stable desktop Code/Outline columns, explicit narrow-screen tabs, shared Button/tokens, inline recovery/conflict confirmations, visible source/safe-outline labels, and no imagery or fidelity preview.

## TDD evidence

Production remained untouched until all five canonical files existed and a genuine complete canonical RED had been captured.

- Initial sandbox attempt: invalid harness run because Vitest could not create its temporary directory (`EPERM`); it collected zero tests and was not counted.
- Test-only harness correction: the required `.ts` outline test used `React.createElement` rather than JSX. This was separated from product failures.
- Canonical RED, exact five-file command: **5 failed files; 55 failed tests; all 55 collected**.
- Authorized attachment-client RED, separate from canonical work: **1 file; 6 failed / 86 passed (92 collected)**. The failures covered optional abort plumbing and the five additional exact successful-response security policies.
- Attachment-client GREEN: **1 file; 92/92 passed**.
- First canonical GREEN: **5 files; 55/55 passed**.

Additional test-first hardening remained separately ledgered:

- Focused Unicode/label/DTO/origin hardening RED: **3 files; 4 failed / 37 passed**; then **41/41 passed**.
- Canonical after that tranche: **5 files; 58/58 passed**.
- Source-boundary hardening RED after one query-only harness correction: **5 files; 10 failed / 57 passed (67 collected)**; then **67/67 passed**.
- Monaco fallback/lifecycle/download cancellation/tab semantics RED: **3 files; 5 failed / 30 passed**; then **35/35 passed**.
- Final worker-state fencing RED: **1 file; 2 failed / 21 passed**; then **23/23 passed**.

Initial pre-review canonical result: **5 files; 71/71 passed**.

### Receiving-review correction rounds

Production remained unchanged from the initial Task 15 commit until the first coherent review-fix RED was complete:

- Exact canonical five-file RED: **5 files; 44 failed / 67 passed (111 collected)**. It covered synchronous identity mismatch fencing; settled capability gates and live revocation; same-scope quarantine/recovery; pending-pagehide persistence; compatible route blocking; real Monaco 0.55.1 initialization order and rollback; save/reconciliation races; and safe-outline depth/node/control/marker bounds.
- Focused GREEN tranches: source editor **18/18**, safe outline **30/30**, workspace **40/40**.
- First review-fix canonical GREEN: **5 files; 111/111 passed**.

A final diff audit found route-instance reuse, pending-digest reconciliation, middle-click pre-`auxclick`, and exact marker-cap gaps. A second coherent test-only round was completed before those production changes:

- Exact canonical RED: **3 failed / 2 passed files; 11 failed / 104 passed (115 collected)**. Failures mapped to Monaco (**3**), workspace conflict/recovery races (**6**), and outline bounds/markers (**2**).
- Associated parent-route regression: an initial lifecycle-only assertion false-passed because React unmounted after a transient reused render. A render-time private-state sentinel corrected that harness and then captured a genuine **1 failed / 14 passed (15 collected)** RED: presentation B rendered once with A's private component state.
- First security tranche GREEN: real Monaco, outline, and parent route **65/65**.
- Workspace implementation attempt 1: **41/42**; the fresh ETag completed correctly, but an edit during the GET had changed status to `Not saved`, hiding the ready confirmation. Restoring `Conflict` only after verified GET fixed that root cause.
- Workspace GREEN: **42/42**.
- Final canonical GREEN: **5 files; 115/115 passed**.

The final acceptance audit found that the synchronous preflight candidate was still missing from navigation/action authority, forged outline DTOs could misrepresent truncation metadata or invalid text, Monaco modifier-left gestures reached mouse-up before the click guard, and an aborted ambiguous-save reconciliation could publish late state. A third coherent test-only round preceded those production changes:

- Exact canonical five-file RED: **3 failed / 2 passed files; 13 failed / 113 passed (126 collected)**. Failures mapped to SourceEditor (**3**: pending callback plus both real-Monaco initialization orders), workspace (**3**: pending actions, immediate navigation, and mismatch-during-reconciliation), and outline (**7**: lone surrogate, empty text, four truncation-chain forgeries, and a natural marker suffix).
- Focused SourceEditor/outline GREEN: **2 files; 58/58 passed**.
- Focused workspace GREEN: **45/45 passed** on the first state-machine implementation attempt.
- A same-turn `beforeunload` refinement then captured **1 failed / 44 skipped** before passing **1/1**. Strengthening the same test for same-turn invalid rollback produced the same narrow RED once more; the final ref-authoritative dirty check passed **1/1**. The root cause was resolved on the second refinement attempt, below the three-attempt stop limit.
- Final canonical GREEN after the refinement: **5 files; 126/126 passed**.

A post-amend final-diff audit then identified two lifecycle/rendering risks, and both were reproduced before production changed:

- Focused two-file RED: **2 failed / 64 passed (66 collected)**. The real workspace remained stuck on principal confirmation when React StrictMode replayed mount effects, and the forced textarea fallback visibly reverted a preflight-valid candidate while SHA-256 was deferred.
- Minimal implementation attempt 1 re-armed the mounted fence in effect setup and gave only the fallback textarea a local candidate buffer synchronized on actual external accepted-value changes, with explicit invalid/read-only rollback.
- Focused GREEN: **2 files; 66/66 passed**.
- Final canonical GREEN: **5 files; 128/128 passed**.

The final receiving-review package was again test-first. Production stayed unchanged until one coherent four-file focused run completed:

- Focused editor/workspace/outline/recovery RED: **4 failed files; 29 failed / 120 passed (149 collected)**, plus the expected unhandled `SecurityError` proving the remaining bare `sessionStorage` getter escaped its boundary. Failures covered storage acquisition/read/write/cleanup, unconditional mismatch/logout/pagehide scrubbing, pending editor transitions, realistic deferred fallback typing, synchronous worker setup/dispatch exceptions, chrome/active subtree exclusion, DTO block ceilings, rendered marker ceilings, stale recovery on exact-base reversion, and empty-candidate authority.
- The strengthened StrictMode regression used one real deferred detail request spanning effect replay, then resolved that request and proved source publication. It remained GREEN against the already-correct mount fence; it no longer relies on an impossible second detail call.
- A narrow old-scope cleanup regression produced a genuine **1 failed / 54 skipped** RED after one test-readiness correction: successful new-scope persistence incorrectly cleared the warning while old scoped recovery remained unresolved.
- Empty-source authority regressions produced a genuine **4 failed / 55 skipped** RED after one test-readiness correction. They covered same-scope reauthentication, pagehide, save rebase, and lost-response rebase without treating `""` as absence.
- Focused editor/workspace/recovery GREEN: **3 files; 97/97 passed**. Focused outline/controller/render GREEN: **1 file; 57/57 passed**.
- A final worker-identity addendum captured **1 failed / 57 skipped**: retired worker A's second error terminated active replacement B. Fencing `onerror` to its owning worker passed **1/1** and the full outline suite passed **58/58**.
- Final canonical GREEN after all review amendments: **5 files; 163/163 passed**.

The closing review extension also remained regression-first:

- Real Hash/Memory router StrictMode tests reproduced unbalanced router-owned listeners in both wrappers (Hash added four tracked listeners but removed two; Memory added two and removed one). Effect-owned construction/disposal closed that lifecycle. A separate **4/4 focused RED** proved the shared route-guard interface was absent. The first native `unstable_usePrompt` approach reached **24/24 behavioral assertions** but still exited 1 with React Router's uncancellable deferred `Invalid blocker state transition`; after the mandated three-attempt pause, the reviewer-authorized architecture switched real data routers to `useBlocker` plus an identity-fenced, cleanup-cancellable application timer while leaving the synchronous Next shim prompt active only outside data-router context.
- A coherent editor/workspace/outline/download cleanup and source-policy run produced **4 failed files; 14 failed / 175 passed (189 collected)**, plus four expected unhandled disposer errors. It covered independently throwing editor/model/worker/download cleanup, anchor removal and URL revocation, persistent dirty authority through offline/quarantined shells, malformed save-response retention, and symmetric URL-like text rejection in worker output and forged DTOs. Its full tranche GREEN was **4 files; 207/207 passed** after later focused additions.
- The final parser/ETag review was test-only first: **2 files; 10 failed / 8 passed / 151 skipped (18 selected)**. It proved that syntactic `<div/>` incorrectly bypassed the HTML depth preflight and that tag lists, concatenated weak tags, internal quotes, controls, and non-byte code points could become reusable ETag authority across initial detail, save, ambiguous reconciliation, and fresh overwrite reads. The exact focused GREEN was **18/18**.
- The concrete `5 × 1,333` inline-marker plus 25 structural-card-marker DTO produced a genuine **1 failed / 73 skipped** render-budget RED. Main-thread validation now rejects marker reservations above 100,000 rendered scalars; focused GREEN was **1/1**.
- A final callback-time mismatch regression produced a genuine **1 failed / 96 skipped** RED: recovered source remained in `workspaceStateRef` after the principal-boundary callback returned but before React committed. The aggregate ref is now synchronously replaced with source-free state on quarantine, mismatch/logout scrub, pagehide, and unmount; focused GREEN was **1/1** and the full workspace suite passed **97/97**.
- Final canonical GREEN after every closing amendment: **5 files; 227/227 passed**.

The last receiving-review correction was also isolated behind focused REDs before production changed:

- The shared conflict controls initially reproduced **2 failed / 97 skipped** tests: a newer discard action did not abort an older overwrite preflight. Strengthening the coherent interleaving matrix produced **4 failed / 97 skipped**, covering both last-started actions and both response orders, including an older discard response replacing a newly overwritten/saved source. One conflict-action controller/epoch then made the full workspace suite **101/101 passed**.
- The post-mount Monaco failure regression produced **2 failed / 25 skipped**: fallback retained the model/editor resources and `saveViewState` was not disabled. The editor now retires its scoped guard/model/editor independently on the surface transition without cancelling a valid pending digest, and the full SourceEditor suite passed **27/27**.
- The shared-router compatibility check used a real `MemoryRouterWithFuture` under StrictMode. Two preliminary behavior-level probes did not reproduce the persona hook race; the third and final probe observed the live blocker's `proceed()` directly and produced a genuine **1 failed / 1 skipped** RED because it ran synchronously. The existing persona hook now uses the same cleanup-cancellable, exact-blocker-fenced next-task transition and its full blocker suite passed **2/2**.
- Final exact canonical GREEN after these corrections: **5 files; 231/231 passed**.

The routed-offline review was reopened test-first without touching production:

- After one Monaco-mock readiness correction, the coherent Page/router/outline/persona run produced **2 failed / 1 passed files; 17 failed / 64 passed (81 collected)**. The real Page plus real data-router integration accounted for five failures: accepted and deferred-digest offline retention, same-scope config reauthentication, and source/recovery scrubbing before origin and principal kind changes. Twelve exact copy failures covered the fixed worker message and visible outline heading. The strengthened persona regression remained GREEN and additionally proved its retired blocker still had zero `proceed()` calls after the 10 ms task wait.
- The first implementation run closed every product/copy assertion; its only two failures were one intentionally repeated structured slide title matching four valid structured surfaces. Narrowing that test-only query to the unique deck heading produced **3 files; 81/81 passed**.
- The first canonical rerun was **229/231 passed** solely because two existing Workspace assertions still expected the replaced em dash. Updating those exact-copy assertions produced the final **5 files; 231/231 passed**.

The final parent/child authority-settlement review also remained test-first:

- Five real Page/data-router regressions first produced **5 failed / 5 passed (10 collected)**. They inverted the usual response order so structured or unsupported metadata (and structured detail) resolved before the retained Workspace finished principal confirmation. They proved that same-scope restoration, origin/principal scrubbing, the global capability-unavailable branch, and failed old-scope recovery removal could otherwise evict the only cleanup owner too early.
- The first handshake implementation made that focused suite **10/10 passed** by buffering the exact epoch-bound Page result and requiring a source-free child settlement callback before adoption. Its first canonical run was **228/231 passed**: three pre-existing direct-Workspace storage-failure cases showed that unresolved cleanup had been made too broad and blocked a verified reload even when no parent kind transition existed. Narrowing the block to pending parent authority made those four directly related lifecycle assertions **4/4 passed** and restored the canonical **231/231** result.
- A final capability-deadlock addendum produced a genuine **2 failed / 10 skipped** RED: when same-scope identity resolved but standalone capability discovery ended in error or read-unavailable, valid buffered structured authority never committed. Settlement now derives from trusted principal resolution plus required recovery cleanup, independently of standalone read capability. The focused pair passed **2/2**; restoring the established same-scope quarantine order after one full-suite regression made the complete routed lifecycle suite **13/13 passed**. It also proves a buffered standalone result retains its exact in-memory quarantine across a capability error and restores it when capability becomes ready.
- The handoff error/durability addendum then produced **5 failed / 1 passed / 13 skipped (6 selected)**. Both same-scope and mismatch request-error outcomes deadlocked behind the retained Workspace; sessionStorage getter and setter failures allowed a dirty structured handoff to discard the only copy; and a preflight-valid digest-pending candidate was not persisted before unmount. The already-present kind-first gate passed its new deferred-metadata/new-origin regression: the second HTML detail call remained forbidden until exact standalone metadata authority arrived. An epoch-owned success/error outcome buffer plus the source-free `(authorityEpoch, releaseSafe)` callback made the selected set **6/6 passed**. New-kind and error handoffs now require the active/quarantined authority to be clean or its exact latest preflight candidate to be synchronously durable; failure retains the warning, route guard, and mounted guarded owner until retry.
- The final identity-ready regression produced **2/2 failed**: direct account switch and scope mismatch could report cleanup completion before a new principal was trusted. Requiring a ready principal fixed switch; the first focused rerun was **1/2** because the durability guard masked the explicit mismatch shell. Preserving the guarded-principal branch made the second refinement **2/2 passed**. Logout/mismatch now cannot release any buffered success/error outcome until a later successful reauthentication for the matching Page epoch.
- A final settled-capability-denial regression produced a genuine **2 failed / 21 skipped** RED: after loading had quarantined a dirty digest-pending candidate and scoped recovery remained unavailable, `ready` with `canReadStandalone=false` scrubbed that quarantine and falsely released the buffered structured result. The first production attempt made the focused pair **2/2 passed**. Settled denial now retires owned work but preserves pending/quarantined same-scope authority and its existing durability result; destructive capability-loss cleanup remains limited to an active non-handoff workspace, while confirmed scope changes retain their existing scrub semantics.
- The final URL-free/stale-release review produced one coherent two-file RED: **17 failed / 1 passed / 95 skipped (113 collected)**. Worker extraction and the shared policy admitted domain, `www`, IPv4/IPv6/localhost, path, and previously unlisted RFC-scheme tokens; 13 matching forged DTO cases reached the controller. In the routed Page, both accepted/getter/structured and digest-pending/setter/error interleavings trusted an earlier `releaseSafe=true` after the editor remounted during delayed metadata. The first production attempt made all **18/18 selected assertions pass**, and the complete outline plus routed lifecycle pair passed **113/113**.
- The complete real routed lifecycle suite finished **25/25 passed**.
- Final exact canonical GREEN after the complete handshake round: **5 files; 245/245 passed**.
- The closing Unicode-host correction remained test-first. One focused outline run produced a genuine **10 failed / 14 passed / 72 skipped (96 collected)** RED: worker extraction retained eight fully Unicode, mixed U-label, IDNA-separator, and suffix-delimiter host references; the shared policy and matching forged DTO boundary accepted the same cases. Existing ASCII cases and ordinary international prose controls remained GREEN. A single shared linear scanner addition made the focused selection **24/24 passed**, the complete outline suite **96/96 passed**, and the exact canonical suite **253/253 passed** on its first production attempt.

## Implementation

- Added one shared source preflight/validator that manually rejects U+0000, every unpaired UTF-16 surrogate (including terminal high surrogates), and UTF-8 size above exactly 1 MiB before `TextEncoder`; accepted values carry exact bytes, scalar count, byte length, and SHA-256.
- Added a lazy inert plaintext Monaco editor and parity textarea. Both are visibly labelled, unnamed, non-autofill/spellcheck inputs. Monaco has editor-scoped `links: false`, hover/context actions disabled, the non-opening default middle-click option, and root capture for click/auxclick/contextmenu, middle pointer/mouse down/up, modifier-left pointer/mouse down/up, and navigation keys. It performs no global service/provider override or mutation. Real pinned Monaco tests prove an explicit open-link action cannot reach the opener while an unrelated editor retains its opener/options in both initialization orders. One editor-local draft buffer synchronously feeds Monaco and textarea, so pending valid text survives Suspense/error/fallback transitions; invalid/read-only/external-adoption paths alone roll it back and external-value epochs fence stale digests. A post-mount render failure independently retires the scoped navigation guard, editor, and model without cancelling valid pending validation, and `saveViewState={false}` prevents the pinned wrapper from retaining view state in its module-global map.
- Added the static outline worker/controller. Lexical preflight is linear and capped before lazy `cheerio/slim`; bogus closers and syntactic self-closing non-void HTML elements cannot hide apparent depth. Parsed traversal iteratively counts all descendants, including suppressed forbidden subtrees; active/URL-bearing elements and application chrome classes are discarded consistently during slide discovery, trusted-text collection, and extraction. One worker-safe bounded linear scanner rejects every immediate-content ASCII RFC scheme plus scheme-relative, `www`, domain, IPv4/IPv6/localhost, absolute, and dot-relative references symmetrically in worker output and main-thread DTO validation. The same pass conservatively rejects fully Unicode and mixed U-label hosts followed by a port, path, query, or fragment, recognizing ASCII full stop and the three IDNA-equivalent Unicode separators without normalizing or constructing a URL. It uses no backtracking expression and preserves ordinary international prose without host shape. Exact-key/digest-bound DTOs reject C0/C1 and bidi controls, lone surrogates, empty block text, more than 50,000 blocks, false truncation markers, inconsistent block→slide→outline truncation, and application-marker reservations above the rendered ceiling while allowing outline-only structural truncation. Under-cap text stays exact, and application-owned structural markers are fitted inside both 20,000-scalar slide and 100,000-scalar total rendered ceilings. Main-thread rendering uses React text nodes, `dir="auto"`, and bidi isolation only. The source-free fixed failure message and visible Safe Outline heading use the required colon/period copy. The controller contains synchronous factory/handler/`postMessage`/termination failures, keeps one active and one replaceable pending source, and identity-fences retired worker errors as well as stale/hung/errored results.
- Added a component-local workspace with settled capability and trusted-principal confirmation before HTML detail/recovery, synchronous scope-mismatch disposal, same-scope in-memory quarantine, abort/epoch fences, exact anchored strong-ETag raw save authority, and digest-only lost-response reconciliation. Only one HTTP strong entity-tag containing legal `etagc` bytes can become a base or `If-Match`; wildcard, weak, list, quote/control, whitespace, and non-byte shapes fail closed at every source-bearing response boundary. The mount fence is re-armed in effect setup so React StrictMode replay cannot permanently invalidate the live instance. A synchronously preflight-valid candidate immediately drives dirty navigation protection and disables/guards source-consuming actions until that exact candidate is accepted or rejected; the route/beforeunload/pagehide authority stays mounted through offline and same-scope quarantine shells. Saves adopt the server base for candidate A without replacing accepted B or a pending preflight-valid C and rewrite recovery against the new base. Discard fences both accepted digest and synchronous candidate epoch. Overwrite fetches and validates a fresh ETag before displaying confirmation, captures the latest accepted candidate only at confirmation, and preserves post-confirm edits and a second 412. One identity-fenced conflict-action controller/epoch spans overwrite preparation, overwrite save, and discard refresh: starting any action aborts its competitor, and every successful adoption invalidates older work before it can validate or publish a late response. Aborted reconciliation is fenced again after its inner catch. It never calls the structured store or autosave path.
- Added closed, capped, 24-hour `sessionStorage` recovery keyed by canonical origin, trusted nonsecret principal, and presentation ID. Storage acquisition itself is no-throw; getter and operation failures return explicit unavailable results. Divergent recovery never autoapplies. Pagehide synchronously preflights the latest candidate even before SHA-256 completes, writes it when divergent or clears stale recovery when it exactly equals base, and then unconditionally disposes Monaco, worker, request, download URL, UI, source refs, and source-bearing aggregate state before its callback returns. Empty source remains an initialized candidate throughout reauthentication, save/lost-response rebase, and pagehide. Same-scope pageshow/focus/visibility/config reauthentication restores quarantined memory without refetch/overwrite; mismatch/logout/switch scrubs it and clears matching old-scope recovery. Unresolved cleanup failures are tracked per scope/operation, retried on later storage access, and cannot be hidden by a successful new-scope write.
- Keyed the standalone workspace by presentation ID in the metadata-first parent, preventing even a transient reused render of presentation B with presentation A's private component state while preserving A's already-scoped recovery on unmount.
- Added the authenticated draft download manager. It rejects invalid source before encoding/dispatch, stops if disposed during digest validation, requires the client-returned bytes to exactly match the accepted draft, creates one `application/octet-stream` URL only for `<a download="presentation.html">`, removes the anchor in `finally`, and revokes on the next task or synchronously on failure/pagehide/dispose. Throwing abort, anchor removal, timer, listener, or URL revocation primitives cannot prevent the remaining cleanup or retain an owned URL reference.
- Updated Presentation Studio to decide from metadata before detail or structured initialization. WebUI mounts the isolated HTML workspace; structured remains compatible; unknown stays read-only; extension runtime makes no HTML detail call, including the legacy metadata/capability-unavailable fallback. Once exact metadata selects standalone HTML, the Page retains that keyed Workspace through offline and authority-revalidation shells instead of replacing it with the generic Studio shell. Exact project/authority epochs fence both successful metadata/detail results and request-error outcomes. The Workspace reports only the source-free authority epoch plus a release-safe boolean, and only after a trusted principal is ready and any required old-scope recovery removal succeeds. Same-scope quarantine restores the exact draft; mismatches synchronously scrub memory and retry failed old-key cleanup without surrendering the mounted owner. Before a new-kind/error handoff can unmount a dirty same-scope owner, the exact latest preflight-valid candidate is synchronously written to scoped recovery; getter/write failure keeps the guarded shell, warning, route protection, and buffered outcome until retry. Any preflight-valid or newly accepted edit made after a same-scope settlement synchronously revokes that exact epoch's cached release authority before delayed metadata/error can adopt, with the callback fenced to the ready principal and active scope. Settlement remains independent of standalone read-capability discovery, so a verified structured/unsupported result cannot deadlock behind an error or legacy read denial, while a verified standalone result keeps its quarantine for later capability recovery. A settled read denial aborts owned work without erasing pending/quarantined same-scope authority or changing a failed durability result. New-scope source detail remains forbidden until exact standalone kind authority. Offline editor remount resumes the exact preflight-valid pending candidate under its existing epoch/scope fences.
- Converted the shared Hash/Memory wrappers to effect-owned data routers and added one route-agnostic leave guard: real data routers use `useBlocker` with one cleanup-cancellable confirmation timer, while the Next shim synchronously fences push/replace/Link/hash/POP transitions and suppresses intentional-cancellation fallback/logging. StrictMode setup/cleanup balances exact router-owned listeners, child updates remain live, clean navigation stays unprompted, and inline confirmed Back asks only once. The pre-existing persona dirty-navigation hook was aligned to the same exact-blocker, cleanup-cancellable next-task transition so converting the wrappers does not introduce a synchronous `proceed()` teardown race.
- In the pre-authorized client seam only, added optional `AbortSignal` options to detail/save/draft calls and required successful attachment `ok/status`, MIME, disposition, `nosniff`, `noopen`, `private, no-store`, `no-referrer`, and same-origin resource policy before returning bytes. Request/response shapes were preserved.

## Verification evidence

### Passing gates

- Final focused Unicode URL-policy selection: **1 file / 24 tests passed** with 72 unrelated tests skipped in 0.49 seconds.
- Final complete Safe Outline suite: **1 file / 96 tests passed** in 0.70 seconds.
- Final exact canonical five-file Vitest command: **5 files / 253 tests passed** in 7.42 seconds. A preceding sandbox-only attempt collected zero tests because Vitest could not create its temporary directory; it is not counted.
- Parent metadata-first route regression: **1 file / 24 tests passed** in 2.38 seconds.
- Final noncanonical directly impacted matrix: **19 files / 239 tests passed** in 9.07 seconds. It covers Presentation Studio bootstrap/create/index/new/generation/form/style/store/readiness, capability/generation/autosave hooks, route guards, presentation normalization/client boundaries, and shared router navigation.
- Fresh correction directly impacted matrix: **4 files / 121 tests passed** in 21.57 seconds, covering the parent metadata-first route, shared router guards, and the full persona state/blocker integration.
- Fresh routed Page/router/persona matrix: **5 files / 146 tests passed** in 24.72 seconds, including the 25-case real Page/data-router lifecycle suite, authority-settlement inversion, error/durability, stale-release revocation, identity-ready, capability-deadlock, settled-denial quarantine retention, and kind-first coverage, plus the strengthened retired-persona-blocker assertion.
- Real navigation guards: shared Hash/Memory data-router suite **16/16 passed** and the final Next shim transition suite **12/12 passed** in 0.82 seconds, including StrictMode cleanup and allowed/denied Link, programmatic, replace, hash, and POP paths with no late teardown errors.
- Focused auth/config lifecycle matrix: **6 files / 78 tests passed** in 3.58 seconds.
- Standalone presentation client boundary: **92/92 passed**.
- Fresh OpenAPI verifier: exit 0; **317 ClientPath entries** and **49 media fallback fields** verified, with the same 10 reviewed repository exceptions.
- `git diff --check`: exit 0 before the report update and again after final staging.
- Static source-sink audit: no `dangerouslySetInnerHTML`, browser `DOMParser`, `srcdoc`, `innerHTML`, `insertAdjacentHTML`, popup/navigation assignment, source-derived URL/worker/module/function/import, iframe/object/embed/script resource sink, analytics, global source cache/store, or extension-message sink in Task 15 production. No standalone source reaches logging or `localStorage`. Reviewed positive hits are limited to:
  - the fixed static worker URL;
  - the fixed lazy Monaco and `cheerio/slim` imports;
  - canonical-origin reads;
  - scoped `sessionStorage` recovery;
  - the fixed temporary download Blob/anchor and its revocation calls;
  - pre-existing persona-preference `localStorage` access in the narrowly adjusted compatibility hook, which never receives standalone source;
  - guarded pre-existing Next-shim diagnostics for non-intentional routing failures, which never log standalone source;
  - forbidden-tag strings used only by the outline discard set.
- Bandit: not applicable; Task 15 touched no Python.

### Typecheck audit

Exactly one package audit was run with `NODE_OPTIONS=--max-old-space-size=8192`:

`bunx tsc --noEmit -p tsconfig.json`

It exited 2 on the inherited repository baseline. The output named three Task 15 discriminant-narrowing diagnostics (`standalone-html-source.ts`, `StandaloneHtmlSourceEditor.tsx`, and `StandaloneHtmlWorkspace.tsx`) and no other Task 15 path. All three were fixed immediately with explicit `ok === false` discriminants; subsequent canonical and broad runtime/transform gates passed. Per the brief's one-audit instruction, the 8 GiB package command was not repeated.

Inherited diagnostics remain in Notes tests, AudioStudio, ResearchWorkspace tests, Scheduled Tasks, Setup tests, Skills, a Dexie test, extension background, scheduled-task control-plane, MCP hub, and voice cloning. They were not changed or represented as clean.

### Design-system audit

`bun run verify:design-system-state` exits 1 on **11 blocked repository findings**, **62 allowed Ant Design baseline exceptions**, and **1 stale `AvailableModelsList.tsx` baseline entry**. The blocked set is in Skills, Scheduled Tasks, and the pre-existing Task 14 `StandaloneHtmlGenerationForm.tsx`; it names no Task 15 file changed by this correction. Task 15 itself uses shared Button primitives and product tokens and introduces no Ant Design product-state primitive.

## Visual and accessibility self-review

- Loading, guarded, offline, bounded load-error, dirty, saving, saved, conflict, recovery, storage-quota, validator-unavailable, and leave-confirmation states are present and source-free where required.
- Save state uses `role="status"` plus polite live announcement; bounded errors use alerts. Recovery and conflict choices are semantic labelled regions with destructive actions confirmed inline.
- Desktop remains a stable two-column workbench. Narrow Code/Outline controls are real tabs with selected state, controls/tabpanel linkage, preserved mounted content, native keyboard activation, visible focus rings, and approximately 44px targets.
- The source and safe outline have programmatic visible labels. Outline text uses automatic direction plus bidi isolation and never renders markup or links.
- Shared controls and editor styles use repository tokens; motion is limited to existing transitions with reduced-motion handling. No gradients, glass, decorative animation, side stripes, modal-first routine flow, or imagery were added.
- Browser/multi-engine visual and no-execution automation remains intentionally deferred to Task 17 by the approved plan.

## Files and scope deviations

Primary Task 15 production/test files match the brief. Directly associated additions beyond its original file list are:

- `PresentationStudioPage.test.tsx`, for the metadata-first WebUI handoff and extension source-free regression.
- `PresentationStudioPage.offline-standalone.test.tsx`, for the controller-requested real Page/data-router offline, reauthentication, and identity-change integration coverage.
- `PresentationStudioPage.tsx`, for the correctness-required presentation-ID component key that prevents route-instance source reuse.
- `services/tldw/domains/presentations.ts` and `tldw-api-client.presentations-standalone.test.ts`, under the controller's explicit narrow authorization for abort plumbing and exact attachment security headers.
- `standalone-html-outline-text-policy.ts`, a Task-15-local bounded scalar/text classifier shared by the worker and main-thread validator so URL-token policy cannot drift.
- `entries/shared/router-utils.tsx`, its new navigation-guard suite, the Next `react-router-dom` shim, and its transition suite, under the controller's explicit narrow authorization to make dirty internal navigation guardable in both runtime families.
- `routes/hooks/usePersonaStateDocs.ts` and `routes/__tests__/sidepanel-persona.blocker.test.tsx`, under the controller's explicit compatibility authorization to keep existing dirty numeric-Back behavior safe after those shared wrappers became data routers.

No other scope deviation occurred. The protected `apps/packages/ui/node_modules/antd` artifact and both Watchlist templates were left untouched and will not be staged.

## Remaining concerns

- Repository-wide typecheck and design-system verification still have inherited failures described above. The final 20-path Prettier invocation checked the report plus 19 changed TypeScript/TSX files and reported the existing formatting baseline in all 19 code files; it is not represented as a clean formatting gate. `git diff --check` is clean. Frontend-root ESLint ignores shared-package files, while package-root ESLint has no `eslint.config.*`, so neither invocation is represented as a lint pass.
- Pinned Monaco 0.55.1 exposes only a global first-initializer `StandaloneServices.initialize` override, not a literal editor-scoped opener override. Task 15 therefore performs no service/provider mutation and achieves the security objective through editor-local plaintext/link/provider/action/gesture prevention; real Monaco tests prove zero opener calls for the standalone editor and preserve an unrelated editor in both initialization orders.
- React Router documents rough edges for rapid repeated POP navigation. The application guard owns and cancels only its current confirmation/proceed timer and identity-fences stale blockers; the verified single-transition Link/programmatic/replace/Back contracts are green.
- Browser/multi-engine execution-sentinel, bfcache, responsive visual, and download-flow automation is Task 17 scope.
- No push was performed.

Initial commit: `c81fcc32e11c07d1afbb15b9c2a1537791586e35 feat(webui): add inert HTML presentation workspace (TASK-12115)`

Receiving-review correction commit subject: `fix(webui): harden inert HTML workspace boundaries (TASK-12115)`
