# TASK13260.213 / UAT272 — World Book attachment hydration

## Scope

Repair only the World Books Manager/detail-panel attachment hydration path, the duplicated character collection path order, the two maintained character-world-book readers, and focused maintained tests. Do not change backend attachment routes, Character CRUD, native runtime, or task tracking.

## Causal finding

The retained native evidence establishes that the character-to-world-book association endpoint returns an enabled association after reload. The World Books Manager nevertheless obtains its attachment candidates through `tldwClient.listCharacters()`. In the retained native run, that legacy list route redirected and then produced rate-limit responses, so the Manager had no character candidates from which to hydrate reciprocal attachment state. The Manager currently starts this legacy candidate request before attachment UI is requested.

The existing per-character `listCharacterWorldBooks` result is already the intended reciprocal data source. Its response is an array and its `world_book_id` matches the selected book; no reverse endpoint is required or added.

The endpoint's ordinary relationship read defaults to `enabled_only=true`, which hides a persisted disabled attachment after reload. This is valid for ordinary callers, but not for the World Books Manager's membership and metadata matrix: it must retain disabled links so it can display and re-enable them.

## Design

Correct the existing character collection client’s candidate order to start with the backend’s canonical trailing-slash route while preserving its alternate-path fallback. This must be made in both maintained client implementations. Keep the Manager’s existing character-list operation but delay it until attachment hydration is requested. Keep sequential per-character relationship reads and owner-scoped client calls. A missing or inaccessible relationship (`404` or `403`) is treated as no association; transport and server failures reject the query so the UI remains unavailable rather than falsely empty. The retry control invalidates the two existing attachment query keys. Do not infer authentication failure from the retained request evidence.

Add only an optional `includeDisabled` argument to the two existing `listCharacterWorldBooks` implementations. It leaves no-argument callers on the endpoint default and emits `enabled_only=false` only for the Manager's relationship hydration read.

## Tests

1. Focused client tests prove both maintained collection implementations prefer the canonical path, and prove direct Base and chat-RAG relationship readers preserve the default path while explicitly requesting disabled links when asked.
2. Manager tests prove lazy candidate hydration, a successful association map retaining disabled attachment metadata, a rejected transport/server relationship read, and a `404` no-association control.
3. Detail and Manager tests prove the retry button is actionable and invalidates the two existing query keys.
4. Existing attachment controls continue to cover attach, detach, matrix, and metadata behavior.

## Stages

### Stage 1: Causal test — complete

Capture the legacy-route failure and write a focused failing Manager test.

### Stage 2: Minimal hydration correction — complete

Correct the shared character collection path order and defer the existing Manager collection read until attachment hydration is requested, preserving the current selection and matrix entry points.

### Stage 3: Verification — complete

The focused client, Manager, detail-panel, matrix, attach/detach, metadata, and quick-attach suites pass. The frontend typecheck preserves the known 90 unrelated diagnostics and reports none in changed paths. Static receipts are retained under `.tmp/uat-repairs-231-246/worldbook272/`.
