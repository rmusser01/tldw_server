# H2 native fork implementation-plan review — pass 2

Date: 2026-09-18. Independent requesting-code-review seat.

Reviewed worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design`, branch `codex/chatbook-h2-native-fork`, source baseline `ac76c4bc5b035561bf816009c1326a114e87def9`.

**Design readiness: ready to implement in the reviewed scope.**

**Implementation-plan readiness: ready to implement in the reviewed scope.**

**Open findings in this seat: 0 P1, 0 P2.** All three P2 findings and both minor corrections from `/private/tmp/chatbook-h2-plan-review-1.md` are resolved at the specification/plan level. No new P1/P2 was identified in the repairs. This verdict is not application qualification, production readiness, or broader parity completion.

## Scope and method

Reread the revised 188-line design and relevant portions of the revised 426-line plan, concentrating on the three previous findings and adjacent authority/consumer boundaries. Rechecked the current source for summary admission/CAS, image-reference representation, both image client serializers, FileArtifacts normalization/persistence/export order, the image adapter request context and provider reference payload, and actual shared transports. The source remains the unimplemented H1 baseline.

No repository edits, agents, applications, runtime tests, database actions, builds or Git mutations were performed. Only this report was written outside the repository. Receipt/GC/quota findings remain the other review seat's responsibility; this report does not substitute for its closure.

## Prior-finding disposition

### P2-1 — Summary history fence: resolved

- `IMPLEMENTATION_PLAN_chatbook_h2_native_fork.md:72–85` now carries the admitted post-input `history_revision` in `FrozenHistoryBehaviorPlan` and `expected_history_revision` in `NativeChildSummaryUpdate`, alongside the settings revision.
- Task 3.2 at `:299` and `:312` captures the post-append version in the same admission transaction and compares both versions under the child lock. The tests hold composition while an actual edit/delete/append changes history without changing settings.
- Design `Docs/Design/2026-09-18-chatbook-h2-native-fork-design.md:129` explicitly discards the stale update and prohibits installing it under a freshly reread fence.

This matches the actual two-fence obligation at `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:4209–4218`, while accounting for current-input append at `core/Chat/chat_service.py:3965–3974`. Frozen effect validation still occurs before input persistence; only the version carried out for later summary persistence is obtained after that append. The repair does not introduce a live-card materialization dependency.

### P2-2 — Canonical unavailable-asset resolution: resolved

- Design `:49` adds an exact scoped `PATCH /conversations/{id}/retained-assets/context` contract with expected manifest/target revisions and explicit remove/replace/restore actions.
- Design `:161` defines the transaction and action semantics. Removal disables only the selected request reference while retaining historical display and bytes. Replacement uses an already retained live claim in the same conversation. Restoration requires known original content identity; differing or unknown identity requires explicit replacement. Wrong owner/stale state writes nothing.
- The same paragraph requires canonical reload/reconciliation after a lost result, forbids blind retry with a newly fetched revision, and excludes browser hiding and source-job cancellation as resolution.
- Plan `:249` names `update_native_asset_context` and its canonical manifest/revision response. Task 2.3 at `:264` implements the server mutation and tests stale/cross-owner claims, restore hash, lost result and actual admission unblocking.
- Task 4.2 at `:357` and `:376` names the real `useFileUpload.ts` native branch, exact scoped PATCH client and canonical-result handling.

The contract now supplies the missing authoritative transition; it no longer depends on `useFileUpload.ts:173–186`'s browser-only filtering or `chat-document-processing.ts:760–819`'s Media job/draft cancellation. Preserving claims during a context-only removal is explicit and does not accidentally invoke purge/GC semantics.

### P2-3 — Native reference-image consumer: resolved

- Design `:145` defines the typed `reference_image` discriminator, exact conversation/asset/revision/scope address and mutual exclusion with legacy `reference_file_id`.
- It carries the type through request snapshots, both client serializers, `/api/v1/files/create`, normalization and export, then injects an authenticated resolver via FileArtifacts/image context. The resolver verifies exact owned native bytes and uses the existing detached `ResolvedReferenceImage` / `ImageGenRequest.reference_image`; no MediaFiles/URL fallback is allowed.
- New Task 3.3 at plan `:315–323` names the actual endpoint, schema, FileArtifacts service, image adapter and image validation/resolver files. Its production-handler test deletes the original MediaFiles resource and inspects the image backend payload; negative cases cover both reference forms, missing/wrong-owner/wrong-scope/revision, backend/model capability, projection header and unsupported export modes.
- Task 4.2 at `:357` and `:376` names both image serializers and the request snapshot/controller consumers, includes the held-owner test, and correctly separates the text-only “Refine with LLM” helper from reference-image admission.

The proposed detached byte interface is compatible with the existing `ResolvedReferenceImage` (`core/Image_Generation/capabilities.py:16–33`), whose identity already permits a string. The plan correctly addresses both context installation sites (`FileArtifactsService.create` normalization and `_export_sync`) and the old `reference_file_id`-only export branch at `image_adapter.py:181–199`. Inline synchronous use is the qualified target; asynchronous jobs and later re-export explicitly reject native references before creating work until they have their own current authorized resolver. This is an honest bounded capability gate, not a claim of asynchronous native-reference support.

## Minor corrections and adjacent checks

- **Transport path correction: resolved.** Plan `:357` identifies actual shared `entries/background.ts`, `services/background-proxy.ts` and `services/tldw/request-core.ts`; it explicitly recognizes the extension entrypoint as a re-export and requires testing the WebUI's actual shim resolution without inventing another transport.
- **Result discriminator: resolved.** Design `:53` explicitly retains H1 `state`, `operation_id`, `owner_key`, `child_id` and `message_map`; the deletion example at plan `:166` now uses `result.state`.
- **Neutral-child cold recognition:** design `:171–173` now gives every H2 child a bounded `native_bundle`, not only snapshot characters or rich assets. Plan `:359`, `:375` covers neutral empty children and poisoned ambient settings without a local receipt. The optional character binding cannot become the test for canonical settings authority.
- **Protocol and boundaries:** plan `:359` explicitly separates atomic request/digest validation from the legacy `historyDigest` path. The revised scope continues to exclude H3/H4/F02 and unsupported persona/multi-participant/comparison behavior. Native-reference async/re-export is explicitly unqualified; successful image display is not accepted as provider-use proof.
- **Dependency/test placement:** the new reference-image task depends on the native stores/read service from Stages 1–2, then has its client integration in Task 4.2. A4/A6 coverage at plan `:409`, `:411` includes Task 3.3. Required implementation tests remain planned, not recorded as passing.

## Remaining implementation obligations

The proposed fixes still need implementation and the named negative/race/provider-payload tests, both-shell qualification, real SQLite/PostgreSQL proof and final independent runtime review. The plan's own completion gates preserve those obligations. No additional specification blocker was found by this seat.

## Supplemental final interface check — 2026-09-18

Reviewed the final narrow additions in the design and plan. **No P1/P2 introduced; both readiness verdicts above remain unchanged.**

- Design `:48` now supplies the authenticated/scoped/projection-gated collection GET for actual canonical manifest reload after a lost PATCH result. Plan `:249` names `read_native_asset_context` and the GET route. This completes the previously described reconciliation path without rebasing or redispatching a mutation automatically.
- Design `:50` identifies `target_reference_id` and expected manifest/reference/asset revisions. Plan `:249` explicitly makes removal affect only the selected reference when two references share one claim; bytes and the other reference remain. The revision CAS therefore controls context use rather than accidentally treating claim identity as the mutation target.
- The canonical request tuple at design `:65–66` now includes reference ID/revision, role and `context_enabled`, in addition to immutable asset identity and fidelity. A changed context-reference choice cannot reuse an accepted operation digest merely because its bytes are unchanged. This agrees with capture/commit using the same semantic projection.
- Plan `:60`, `:94` makes the pure preallocation projector return `NativeForkBindingTemplateV1`. Task 3.1 at `:278–281` binds it to the fresh child and emits `NativeAssistantBindingV1`. The retained projection is consequently independent of an unallocated child ID, while the committed identity remains child-owned.

No repository edits or runtime work were performed; only this supplemental report text was appended. These remain reviewed interface contracts, not passing implementation evidence.
