# UAT221 bounded design — TASK13260.159

Approved by root before implementation. Native Notes Graph403 is an intentional notes.graph.read permission gate; current generic load failure obscures this and a later403 leaves prior graph pages visible. Preserve server policy and the separate Connections implementation.

Use the existing numeric error.status convention in the actual graph hook. Hide graph-derived data and actions when the current authority receives403, including the separate cursor expansion request. Suppress automatic retry/reconnect for that denial. Explain permission-unavailable using the existing i18n path, while allowing explicit Refresh graph after an administrator changes permissions; successful manual refresh or replacement authority restores normal graph behavior. Transient errors and offline mode keep existing stale-data behavior. No global capability architecture or cache-domain rewrite.

Own two production TSX paths (workspace + its components/Notes/hooks hook), one English option resource, and a new mounted regression test. Tests use real workspace/hook/service/query/i18n, mock only API transport and canvas rendering unavailable in jsdom. Cover cold denial, warm denial, cursor denial, no automatic reconnect retry, manual recovery, authority replacement, loading and transient/offline controls. Existing graph/authority/Connections tests remain adjacent controls.

Stages: (1) baseline + causal RED; (2) minimal GREEN; (3) adjacent/static + frozen independent handoff. Native denied/admin controls remain root's gate. Generated public locale copies are build artifacts and stay unchanged; shared assets/locale/en is the active source.

## Independent lifetime finding and revised boundary

Reviewer retained a real-hook counterexample (1 fail / 12 pass): radius 1 starts, radius 2 receives 403, radius 1 resolves 200 late, then reopening radius 1 renders cached private data. Clearing only present pages does not revoke pending cache writers. Original frozen candidate is preserved under pre-lifetime-review/.

Installed TanStack query-core 5.90.20 cancelQueries synchronously cancels retryers; their later transport fulfillments cannot write Query data. On a non-canceled 403, cancel other Notes Graph queries matching the reporting authority, excluding the reporting query, with revert:false; then clear cached pages. Revert:false prevents canceled cursor fetches from resolving with previous data into the expansion success callback. Carry the Query context key and AbortSignal to identify the reporting query and ignore late canceled errors, without requiring background transport cancellation. Expected TanStack cancellation resolves the expansion command to null; unrelated failures still reject. Explicit refresh can create a fresh request after denial, and other authorities retain their own requests/cache. No service/schema/global cache change.

Permanent controls: the exact reviewer late-success/reopen, active concurrent same-authority base and cursor writers, late canceled 403 after successful manual recovery, and an unrelated authority still completing. Existing reconnect, authority change, transient/offline and explicit recovery tests remain. Root approved this boundary before production revision.
