# Frozen-reference retention candidate admission — TASK-13609

The private candidate store could reserve arbitrary hash/representation staging metadata under a generic retention lease, without binding it to the exact reviewed reference. The new typed reserve_retention_candidate boundary strict-revalidates the request, reuses original request/source-incarnation admission, selects an exact retained reference, derives its hash and representation, then calls the existing generation-bound candidate store. Compatible role/context aliases can reuse the same immutable candidate. Size/MIME and storage IDs remain immutable staging metadata; replay cannot replace them.

One local savepoint contains source admission and candidate insertion. Caller-caught failure rolls back only this work. No quota intent, claim, reference, byte or ready-state is created by this method. Generic controlled-clock house fixtures retain their existing behavior; future production retention preparation must use the typed boundary.

Independent review found two additional roots, both reproduced:
- Lease time was sampled before generic path/candidate lock waits. A final existing live-attempt check now samples UTC after reservation inside the savepoint.
- NativeScopeV1 serialization could silently normalize unchecked global scope with a foreign workspace field. The shared serializer now validates original fields before emitting the closed wire shape, rejecting malformed data across candidate, reservation and lease callers while preserving valid output.

Evidence:
- Initial causal15 failed/15 official PostgreSQL skips,12 warnings6.66s; focused green15/15 skips8.58s.
- Initial connected six suites157 passed/158 skipped,16 warnings131.93s, before review fixes.
- Review regressions4 failed/4 skipped/30 deselected,12 warnings2.74s; focused fixed19 passed/19 skipped10.36s.
- Strengthened original-lease clock control and malformed-scope cases5 passed/5 skipped/30 deselected,12 warnings3.94s. This uses actual reservation with an unchanged stored lease and controlled time advancement after insertion, both still-live and expired. It simulates the timing boundary; it does not establish a physical PostgreSQL lock wait.
- Three-file Ruff0; changed production files HEAD Ruff0. Actual two-production-file Bandit0 findings/0 errors; coverage checked. New test Black, new method AST-preserving Black formatting, compile and diff checks pass.
- Both final independent source reviews return bounded CLEAR.

ADR required:no. Existing ADR049/050 and proposed071 govern ownership, lifecycle and source incarnation; no persistence/public API changes. Approved H2 Task2.3 remains broader than this boundary.

No runtime, browser, profile database or send changes. Actual typed upload producer, AuthNZ quota authority, retention claim/reference publication, native-source bridge, Sync/public admission and broader NN/G/H2 acceptance remain open. Private evidence /private/tmp/chat-retention-candidates-13609-*; raw logs are not published.

Final connected nine suites:363 passed/236 skipped,16 warnings121.48s. Official unavailable PostgreSQL controls and existing backend-specific skips remain explicit. Full canonical fingerprint/path checks pass unchanged:2129 paths/3301 schemas, SHA256 bd7997199762d42481de4c4b32536f3c82f279d695c86ff44afa427b36771212. Initial normal exact-owned six-file hooks pass; house Ruff/Black filters skip native paths, meaningful manual checks above pass. All three bounded stages complete; only own plan retires. Final hooks/local commit follow; master TASK13458 and H2 parent13261.4 remain In Progress.
