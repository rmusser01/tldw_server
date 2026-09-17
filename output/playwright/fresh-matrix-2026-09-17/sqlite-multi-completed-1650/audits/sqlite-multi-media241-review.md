# UAT241 Media full-content handoff: independent audit

Task **TASK13260.183**. Frozen SQLite-multi revision `8f8774e6c868b304a96d95ab82e28389c129a78b`.

## Verdict

**The ID-only handoff defect is observed. A loading race is strongly source-supported, but its initial timing is not directly captured.** The first supplied observation starts after navigation to Chat; it does not show pre-click `detailLoading` or the matching detail-request completion. Preserve that confidence boundary instead of treating inferred timing as an instrumented reproduction.

## Native chronology

- **15:34:36.316 UTC:** `media-chat-handoff-actual.txt` shows composer **“Let's talk about media 1.”** and “Prepared chat with this media in the composer.”
- `media-handoff-settled.txt` again shows the same draft on `/chat`. It has no observation timestamp. No initial Send or generated-answer claim is supported by these inputs.
- **15:40:49.938:** `media-handoff-ready.txt` shows Media1 selected, its source paragraphs visibly loaded, and the full-content Chat action.
- **15:41:09.967:** `media-handoff-after-content.txt` shows the resulting composer collapsed as **13 lines / 1,971 chars**. `source-composer-label-click.txt` records the actual textarea value: **1,971 characters**, comprising header/separator plus **1,914 source characters**.
- `source-chat-result.txt` includes `/chat/completions` HTTP200 at **15:43:47.190**, and at **15:44:23.660** visibly shows the source-containing user message plus a question and the answer **Dr. Mira Vale, Cedar Ridge, 18:00 every Friday**. The completion event is a response marker, not a separately observed outgoing request body.

This later positive control demonstrates content availability and successful full-content handoff for the same item. It makes persistent source loss unlikely, but does not exclude an initial detail error, content-extraction state, or another ID-only trigger. Exact first-click loading timing remains unproven.

## Frozen source explanation

1. `useMediaNavigationState.ts:46–48` initializes empty content. A matching result may be selected before detail resolution (217); the selection effect (292–308) starts `loadSelectedDetails`, which sets loading true and clears content (109–112), then fills it after the response (115–119). Failure also leaves it empty (134–141).
2. `ContentViewer.tsx:711–726` renders the full-content Chat button for a non-note selection/callback, without a loading/disabled guard. Its tooltip promises full content. The loading indicator is separately rendered in the content body (767 onward), leaving the action available. The Actions menu forwards the same callback without readiness gating (`useContentMetadata.tsx:395–400`).
3. `ViewMediaPage.tsx:893–930` checks only selection and serializes `nav.selectedContent || ''`. It emits/persists a **normal-mode** payload, clears RAG selection, navigates and reports success. It does not await content or reject loading/error/empty state. That payload is a snapshot and does not update when the Media fetch finishes.
4. `media-chat-handoff.ts:20–39` drops empty content during normalization; its hint builder (57–72) deliberately converts an ID-only payload into **“Let's talk about media 1.”** This precisely matches the initial draft. The fallback itself is an existing tested compatibility behavior.
5. `PlaygroundForm.tsx:2523–2559` applies normal mode and the hint; it does not fetch content for a normal ID-only handoff. Waiting in Chat therefore does not repair the producer's omitted content.

These facts establish a reachable unsafe action and explain the symptom. A deferred-detail causal test is still required before claiming the exact race reproduced. No test was run here.

## Prior Home UAT013

TASK13260.3/UAT013 concerned a **Home `rag_media`** handoff: effective retrieval was not enabled and asynchronous session restoration could replace source intent. Its bounded repair enables retrieval for accepted RAG handoffs and protects newer source intent, explicitly preserving ordinary content handoffs.

UAT241 concerns the **Media full-content producer**, which intentionally selects normal mode. It must not be repaired by forcing RAG or weakening Home owner/restore guards. The separate “Chat about media” action already uses `rag_media` (ViewMediaPage 940 onward). This audit does not claim a regression in UAT013's accepted contract.

## Bounded repair/tests

- Gate both button/menu and handler on usable content belonging to the **current selection**. Loading, failed and empty detail must not navigate with an empty fallback or announce successful full-content preparation. Preserve the separate RAG action and generic ID-only consumers.
- If awaiting content, capture selection/authority and reject obsolete completion. A loading boolean alone may miss selection-transition frames or stale content; test identity as well as availability.
- Use actual navigation hook, ContentViewer action and handoff consumer with a held detail promise: no empty full-content handoff before resolution; exact source after resolution. Cover failed/empty detail, A→B selection with late A completion, menu/direct-handler paths and ready normal/RAG positives.
- The existing permalink handoff test (1055 onward) checks mode/ID rather than requiring full content and uses a mocked viewer. Existing viewer loading accessibility coverage tests the loading announcement. Neither establishes this live action boundary.

No production, tests, browser/runtime, DB, tracker or task changes. Only the two designated ignored audit files are written. No repair or full-matrix acceptance is claimed.

## Hash binding

All 10 selected source/history hashes match the original SQLite-multi archive manifest. Complete source/input hashes and machine-readable facts are in the sibling JSON.

Archive manifest SHA-256: `703f3cd208450440ee4ce31ba51526200257f363fc7fa3fb07c4811d4fa9cb36`.
Loaded draft SHA-256: `e0ee0a607f0627dcccd8946c3e0cd425d8bad732529ccc988cef0ece2a89b30e`.
Source body SHA-256: `a94b1e966d89b7b94e0cd69dafe9ab1c554dc81accf43e57957276b08294225c`.

Inputs relative to `native/sqlite-multi`:

| Path | SHA-256 |
|---|---|
| `media-chat-handoff-actual.txt` | `bb57a1e13e7fea7b3dd61b8b8de4b694e1376f427d0e5e876de40f5fb536c7b7` |
| `media-handoff-settled.txt` | `927b9c980b3bea8d0c9f55be650fec9c540e15ad9241fb85fe03d2971b746e60` |
| `media-handoff-ready.txt` | `c19e67d84c863343097f3060c849a665e2b61b2a6fcd2fad5969e832c8b17c6f` |
| `media-handoff-after-content.txt` | `fb24316d5dcd30f36a14e8ad26ed1c8a993939235fcfc8a7f50f493eb7aff4a2` |
| `source-composer-label-click.txt` | `7bd1127de8e5550943e718e0bcb9bc8aa6758b29d7b7bf71d020fa370101b3d4` |
| `source-chat-result.txt` | `365a97d68e4dd3d18206d06c11d136ce79dc4628477053b85c32ec58a29117ac` |
