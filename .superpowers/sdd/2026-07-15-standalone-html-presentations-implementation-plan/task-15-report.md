# Task 15 report: inert standalone HTML presentation workspace

## Status and scope

Implemented the approved Task 15 WebUI workspace on base `45a507666d21ede1955d451aef52853f2eec5b14`. V1 remains inert: it has no preview, execution, HTML insertion, source-derived navigation, popup, worker code, module, or resource surface.

The implementation stays inside the Task 15 brief plus the controller-authorized presentation-client hardening seam. No request-core, background proxy, backend, OpenAPI, dependency, Backlog, CSP, or protected-artifact change was made.

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

Final canonical result: **5 files; 71/71 passed**.

## Implementation

- Added one shared source preflight/validator that manually rejects U+0000, every unpaired UTF-16 surrogate (including terminal high surrogates), and UTF-8 size above exactly 1 MiB before `TextEncoder`; accepted values carry exact bytes, scalar count, byte length, and SHA-256.
- Added a lazy inert plaintext Monaco editor and parity textarea. Both are visibly labelled, unnamed, non-autofill/spellcheck inputs. Monaco has `links: false`, hover/suggestions disabled, a per-editor rejecting opener service override, a render-failure fallback, and an imperative local-only disposal handle.
- Added the static outline worker/controller. Lexical preflight is linear and capped before lazy `cheerio/slim`; parsed traversal is iterative; active and URL-bearing elements, attributes, and CSS are discarded; DTOs are exact-key/digest-bound and enforce card/block/slide/total caps. Main-thread rendering uses React text nodes, `dir="auto"`, and bidi isolation only. The controller keeps one active and one replaceable pending source, ignores stale duplicate results, and terminates/replaces hung or errored workers without endless retry or stale failure state.
- Added a component-local workspace with source-free principal confirmation before HTML detail, source-minimal base state, abort/fence checks, explicit strong-ETag raw save, digest-only lost-response reconciliation, and the three confirmed 412 choices. It never calls the structured store or autosave path.
- Added closed, capped, 24-hour `sessionStorage` recovery keyed by canonical origin, trusted nonsecret principal, and presentation ID. Divergent recovery never autoapplies. Pagehide flushes the last accepted buffer and synchronously disposes Monaco, worker, request, download URL, and source refs; pageshow/focus/visibility reauthenticate first; logout/switch/origin mismatch clears matching old-scope recovery.
- Added the authenticated draft download manager. It rejects invalid source before encoding/dispatch, stops if disposed during digest validation, requires the client-returned bytes to exactly match the accepted draft, creates one `application/octet-stream` URL only for `<a download="presentation.html">`, removes the anchor in `finally`, and revokes on the next task or synchronously on failure/pagehide/dispose.
- Updated Presentation Studio to decide from metadata before detail or structured initialization. WebUI mounts the isolated HTML workspace; structured remains compatible; unknown stays read-only; extension runtime makes no HTML detail call, including the legacy metadata/capability-unavailable fallback.
- In the pre-authorized client seam only, added optional `AbortSignal` options to detail/save/draft calls and required successful attachment `ok/status`, MIME, disposition, `nosniff`, `noopen`, `private, no-store`, `no-referrer`, and same-origin resource policy before returning bytes. Request/response shapes were preserved.

## Verification evidence

### Passing gates

- Final canonical five-file Vitest command: **5 files / 71 tests passed** in 4.58 seconds.
- Final directly impacted matrix: **25 files / 312 tests passed** in 21.50 seconds. This includes Presentation Studio page/index/new/form/store, capability/generation/autosave hooks, route guard, auth, normalization, and the full **92/92** standalone presentation-client suite.
- OpenAPI verifier: exit 0; **317 ClientPath entries** and **49 media fallback fields** verified, with the same 10 reviewed repository exceptions.
- `git diff --check`: exit 0.
- Static source-sink audit: no `dangerouslySetInnerHTML`, browser `DOMParser`, `srcdoc`, `innerHTML`, `insertAdjacentHTML`, popup/navigation assignment, source-derived URL/worker/module/function/import, iframe/object/embed/script resource sink, logging, analytics, global cache/store, localStorage, or extension-message sink in Task 15 production. Reviewed positive hits are limited to:
  - the fixed static worker URL;
  - the fixed lazy Monaco and `cheerio/slim` imports;
  - canonical-origin reads;
  - scoped `sessionStorage` recovery;
  - the fixed temporary download Blob/anchor and its revocation calls;
  - forbidden-tag strings used only by the outline discard set.
- Bandit: not applicable; Task 15 touched no Python.

### Typecheck audit

Exactly one package audit was run with `NODE_OPTIONS=--max-old-space-size=8192`:

`bunx tsc --noEmit -p tsconfig.json`

It exited 2 on the inherited repository baseline. The output named three Task 15 discriminant-narrowing diagnostics (`standalone-html-source.ts`, `StandaloneHtmlSourceEditor.tsx`, and `StandaloneHtmlWorkspace.tsx`) and no other Task 15 path. All three were fixed immediately with explicit `ok === false` discriminants; subsequent canonical and broad runtime/transform gates passed. Per the brief's one-audit instruction, the 8 GiB package command was not repeated.

Inherited diagnostics remain in Notes tests, AudioStudio, ResearchWorkspace tests, Scheduled Tasks, Setup tests, Skills, a Dexie test, extension background, scheduled-task control-plane, MCP hub, and voice cloning. They were not changed or represented as clean.

### Design-system audit

`bun run verify:design-system-state` exits 1 on existing blocked entries in Skills, Scheduled Tasks, and the Task 14 `StandaloneHtmlGenerationForm.tsx`, plus its stale baseline entry. It names no Task 15 file. Task 15 itself uses shared Button primitives and product tokens and introduces no Ant Design product-state primitive.

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
- `services/tldw/domains/presentations.ts` and `tldw-api-client.presentations-standalone.test.ts`, under the controller's explicit narrow authorization for abort plumbing and exact attachment security headers.

No other scope deviation occurred. The protected `apps/packages/ui/node_modules/antd` artifact and both Watchlist templates were left untouched and will not be staged.

## Remaining concerns

- Repository-wide typecheck and design-system verification still have inherited failures described above; no Task 15 failure remains in the final functional/static gates.
- Browser/multi-engine execution-sentinel, bfcache, responsive visual, and download-flow automation is Task 17 scope.
- No push was performed.

Requested commit subject: `feat(webui): add inert HTML presentation workspace (TASK-12115)`
