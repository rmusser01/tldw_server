---
id: TASK-13525
title: 'WebUI perf batch W5: extension background and content scripts'
status: Done
created_date: 2026-10-07 04:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WebUI/extension perf program. Index: Docs/Plans/2026-10-06-webui-perf-remediation-coordination-index.md. Plan: Docs/Plans/2026-10-06-webui-perf-batch-5-extension-scripts-implementation-plan.md. Scope: Web-clipper tiered capture, copilot popup streaming appends, chunked base64, sidepanel-open guard, warm alarm hygiene. WP-56..63.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 1 (web-clipper tiered capture, WP-56/WP-57): overlap check found TASK-13511 (in dev) had made the parser lazily loaded, but capture still eagerly serialized outerHTML + ran the parser + forced innerText on every clip; requestedType never reached the extraction. Stage 1 threads requestedType into extractClipPageTextFromDocument: selection clips with a live selection return the selection immediately (no outerHTML serialize, no parser, no innerText forced layout — spy-pinned mechanism test), article tier next, fullPageText only as the final fallback for selection/article requests; other clip types keep the eager general flow (characterization trio pins selection-present / selection-absent / article-empty tiers; no captured-content regression). parser/default.ts per-attribute removeAttr loop replaced by one in-place attribs pass preserving the exact kept-set {href, src} (pinned via link title / image alt behavior). Tests: services/web-clipper/__tests__/content-extract.test.ts (6) + parser/__tests__/default.test.ts (1); scoped suites green (30 tests / 6 files incl. existing), apps/tldw-frontend bun run typecheck exit 0. ADR assessment: none required — scoped perf change under the approved plan (Docs/Plans/2026-10-06-webui-perf-batch-5-extension-scripts-implementation-plan.md); no new durable architecture rule.
Stage 2 (WP-58, WP-59) complete: copilot popup stream loop appends one text node per chunk (appendResponseText) and setResponseText skips the full rewrite when the append path already rendered the accumulated text, so stop/done no-op instead of rewriting O(total) per call; rAF position coalescing unchanged (zero synchronous layout reads per chunk, one updatePosition per frame — pinned by getBoundingClientRect spy). arrayBufferToBase64 now converts in 8 KB String.fromCharCode.apply slices; output byte-identical to btoa (property test, sizes 0/1/7/8191/8192/8193/65536 + round-trip) and the STT WebSocket JSON wire format unchanged (multi-slice frame pinned in background.stt-protocol.test.ts; the content-script port already carries binary, so the base64 hop stays protocol-required JSON). Tests: entries/__tests__/copilot-popup.content.test.ts (new, MutationObserver append proof over 1000 chunks), utils/__tests__/compress.test.ts (new). Scoped: entries + voice-consumer suites 294 green; extension bun copilot tests green; FE typecheck exit 0. ADR: none required (scoped perf change under approved plan).
Stage 3 (WP-60/WP-61/WP-62) complete: emitBackgroundMessage extracted to createBackgroundStatusBroadcaster (open sidepanel once per funnelId/sessionId, first emit still opens from closed, message never suppressed); funnel metrics moved to coalescing serialize-on-drain writer (hydrate-once merge, bounded 200, one write per drain window); syncWebClipperContextMenu gated on server-URL change in the tldwConfig watcher. Overlap verdicts: alarm non-forcing ALREADY-DONE (W2 Stage 3, verified); capabilities TTL ALREADY-DONE (W2); context-menu URL gate was the GAP (now closed); WP-63 effective-config reads NOT landed anywhere, remains W4 Stage 5 scope (no duplicate work). Scoped suites: entries (286) + entries/shared (70) + background-helpers.broadcast (13) green; FE tsc --noEmit exit 0; eslint touched paths clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
W5 extension background/content-scripts batch complete on codex/webui-perf-w5-ext-scripts (b2b7d5d2b0..c0393fe1f3, 3 stage commits, zero fix rounds — the only batch in the program review-clean at every stage). S1 (486c653ee6, WP-56/57): web-clipper tiered capture — selection clips return selection text immediately (no outerHTML serialize, no parser, no innerText forced layout; property-getter spy pins with positive controls), article tier next, fullPageText final fallback only; parser attr-strip single in-place pass (kept-set {href,src} exact); intended fullExtract delta contained (all four consumers traced; visibleBody/clipType/actualType/fallbackPath pinned unchanged). S2 (b1511ccf9c, WP-58/59): copilot popup streams one text-node append per chunk (MutationObserver proof: 1000 chunks, zero removal records vs 1009+ at base) with same-rendered-length skip so stop/done calls no-op; zero synchronous layout reads during the stream (one rAF-coalesced position pass per frame); arrayBufferToBase64 chunked in 8 KB slices, byte-identical to btoa (property test 0..65536), STT WebSocket wire format pinned unchanged — port hop already binary at HEAD, base64 protocol-required only on the WS JSON hop (design ruling on evidence). S3 (c0393fe1f3, WP-60/61/62): emitBackgroundMessage → createBackgroundStatusBroadcaster opening the sidepanel at most once per funnelId/sessionId (first emit still opens from closed; runtime message never suppressed; capacity 500); funnel metrics moved to a coalescing serialize-on-drain writer (5-event burst → 1 write vs 5 at base; 205-event lost-update reproduced red at base — now lossless, bounded 200; correctness bonus); context-menu re-sync gated on server-URL change; alarm non-forcing + capabilities TTL verified ALREADY-DONE by W2 (no duplicate work); WP-63 verified not-landed-anywhere (W4 Stage 5 scope). Batch-final: Family B canned e2e re-run 3x at HEAD plus one base run proving count metrics bit-identical (longTasks 0 everywhere; the 19→21-notification / 49–51→30-mutation drift vs W1's record attributed to interim dev below the branch base); M3 closed done-by-mechanism (no canned copilot harness exists in-repo — both copilot e2e specs need a live backend; append-only + zero-layout-read pins; wall-clock left to a future human DevTools or canned-harness pass); scoped suites 403/403 green (29 files); FE typecheck exit 0; eslint exit 0 from apps/ with the workspace binary (corrected invocation); git diff --check clean over the range; Bandit N/A (TS-only, per program precedent). Baseline append: Batch W5 section in Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md. Open gaps recorded there: cross-frame selection still runs the article tier; fullExtract delta extends to default article clips; same-length-skip fallback branch untested (no reachable caller); eslint must be invoked from apps/ with the workspace binary; scheduleModelWarmAlarm still runs on same-URL config writes (idempotent); test:e2e:perf beyond the Family B spec left to CI. ADR: none required — scoped perf changes under the approved plan, no new durable architecture rule (assessed each stage).
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
