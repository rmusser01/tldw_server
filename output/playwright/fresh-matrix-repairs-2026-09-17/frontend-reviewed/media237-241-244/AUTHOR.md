# UAT237/241/244 Media repair candidate

Source frozen in source-freeze.json (six production and six test paths); root owns tasks13260.179/.183/.186, integration and native acceptance. No browser, runtime, DB, inference, task, staging or commit changes by this unit.

## Changes
- **237:** Offline initial verification renders checking-connection guidance without mounting protected Media requests. Search, type, keyword and detail callbacks retire at the existing synchronous connection-store boundary and unmount. Real QueryClient cache is partitioned by mounted lifetime; the page remounts private state even across a batched disconnected→connected authority change. Valid connected background checks and genuine configured outage/retry behavior remain covered.
- **241:** Full-content Chat requires successfully loaded, nonempty content matching current selection and a resolved current account. Button/menu explain loading/unavailable state. Stale detail responses cannot replace newer selection. Both full-content and adjacent RAG producers use the existing ownerScope handoff contract, await storage, and check captured lifetime/owner/selection before event/navigation/mode changes. Normal and RAG modes remain unchanged. The existing real Chat consumer owner checks are independently exercised; no consumer/schema edits.
- **244:** The actual terminal Quick Ingest session emits saved positive media IDs once through its existing current-operation guard. The mounted real search hook validates operation/lifetime and refreshes current filters/page immediately. Error and process-only completions emit no catalogue refresh. Completion before Media mounts remains supported by its initial query; legacy completion event compatibility remains.

## Verification
- **193 tests / 15 Media and Quick Ingest suites passed, zero skipped, zero unhandled errors:** green-final-after-review.log.
- **37 tests / 1 existing real Form/action/RAG consumer suite passed, zero skipped:** consumer-ui-controls.log (UI Vitest config required).
- Actual-root ESLint: **0 errors,151 existing warnings**, versus HEAD owned-source baseline0/154; **zero introduced warning signatures/multiplicity**. eslint-comparison.json plus both full JSON outputs.
- Full identical-options compiler API comparison: **90 baseline /90 current diagnostics, zero added or removed**, compiler-comparison.json. Baseline overlays only these owned12 paths with HEAD bytes while preserving all other current shared changes/dependencies. Full project is not compiler-clean at baseline.
- Scoped git diff --check passed. Bandit not applicable: TypeScript/TSX only.
- Exact reproducible commands in commands.txt; scripts and source hashes retained.

## RED and harness receipts
- red237.log: initial connection gating and late resolve/reject after unmount,3 causal failures. red237-authority.log: synchronous authority invalidation1RED. red237-remount.log: batched owner replacement1RED.
- red241.log: pending/empty/failed full-content handoff3RED; red241-selection.log: lateA causes redundantB reload1RED; red241-viewer.log: actual ContentViewer disabled-state1RED.
- red241-owner.log: normal/RAG pending persistence handoffs and unresolved owner3RED. First green run's remaining assertion included unrelated last-selected-ID storage; corrected to identify the actual discuss-media setting, without weakening navigation/payload assertions.
- red244-causal.log: actual runtime completion→wizard session→real QueryClient mounted catalogue1RED; before-mount control already passed. Earlier red244.log used nonexistent fixture updateSession and is explicitly **noncausal**.
- Adjacent initial185pass/2fail/24unhandled came from the old search-experience custom query fixture calling queryFn without a real TanStack context. Approved test-only replacement uses actual QueryClient/Provider and preserves all original metadata/filter/count assertions.
- Other corrected test expectations: extractor intentionally trims full source text; pagination is in actual URL rather than JSON body. No production workaround was added.
- green-final.log initially included the consumer under the frontend config, causing pa-tesseract import resolution failure while191 tests passed; consumer rerun uses its established UI config and37 pass. Final media-only receipt is green-final-media.log.
- Compiler initially found the new request wrapper's generic path mismatch; fixed using the original typed bgRequest generic argument, with no path widening. Final comparison is exact90/90.

## Limits
Component/integration controls establish these code boundaries, not native acceptance. Root must perform the retained native UAT237/241/244 checks. Other shared changes are excluded from this ownership/review unit. No new authority framework, permission changes, provider requests or backend edits.

## Independent review follow-up: stale-selection interval
Root identified the remaining unscoped interval request/continuation. Two permanent actual navigation-hook tests reproduce a late404 warning after synchronous authority loss/recovery and a late recovery refetch replacing a newly selected source (red237-interval-review.log:2RED). The existing lifetime signal is now passed to the interval request; callbacks verify that lifetime and current selection before warning/refetch and after awaiting refetch. Cleanup stops new timer dispatch. No new authority abstraction. The initial attempt also treated every React effect cleanup as invalidation, which broke valid deletion recovery after its own warning/refetch rendered (green237-interval-review.log:2failed/35pass). Final guard uses actual lifetime/current selection after awaits; both existing next-item and clear-selection controls pass (green237-interval-final.log:37/2). Final adjacent suite:193/15 pass0skip, green-final-after-review.log. Prior author report/source freeze/evidence/static comparisons preserved with before-interval prefix.
