# Candidate c5453d frontend CI repair

Tracking TASK-13263 (new CI follow-up, not reopening completed Qodo subtask). Base c5453d43856d311ffa0904c51e7f327bbd9a929f. Run35542860122, failed shard4job106163963096 and shard8job106163963048. Raw job logs `/tmp/candidate-c545-shard4.log` and `/tmp/candidate-c545-shard8.log` downloaded directly because overall run remains active.

## Root cause and repair

Both media-page suites completely mock @tanstack/react-query with useQuery only. The account-authority cache cleanup now correctly calls useQueryClient; omitted mock export makes30 permalink tests and all10 bulk-action tests fail before page rendering. Add a stable hoisted queryClient object with removeQueries and export useQueryClient in both fixtures; do not modify production cache invalidation.

The remaining stale-deletion callback test expected selected source1 retained after synchronous disconnect/reconnect. That contradicts the new security contract that authority retirement clears selection and content. Strengthen the timing check: after disconnect/reconnect assert selection null before resolving the pending404; then resolve and assert selection remains null, still no warning and no refetch. This verifies immediate cleanup and stale-response suppression separately; no assertion skipped or timeout increased.

Changed files:
- apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx
- apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.stage14.bulk-actions.test.tsx
- Backlog TASK-13263 notes via official CLI.
No production source, workflow, version or manifest edit. Parent owns commit/push/protected-record refresh.

## Verification

From candidate apps/tldw-frontend, installed Vitest4 binary:

`node_modules/.bin/vitest run ../packages/ui/src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx ../packages/ui/src/components/Review/__tests__/ViewMediaPage.stage14.bulk-actions.test.tsx`

Original RED:41failed/1passed, bothfiles; `/tmp/candidate-c545-media-red.log`.

GREEN command adds `../packages/ui/src/components/Review/hooks/__tests__/useMediaSearch.outage.test.tsx ../packages/ui/src/components/Review/__tests__/useMediaNavigationState.permalink-hydration.test.tsx`:58passed across4files in2.79s, `/tmp/candidate-c545-media-green.log`. The real-query tests retain account invalidation/cache-purge coverage beyond the updated page fixture mocks.

ESLint on both actual test files with root config and exactHEAD lintText baseline:10before/10after, no new diagnostics; `/tmp/candidate-c545-media-eslint.json`. Root Next pages-directory lookup warning is preexisting. Diffcheckclean; own review confirms only fixture contract/security expectation changes. Bandit N/A TypeScript-only test files; no Python changed.

Full WebUI `NODE_OPTIONS=--max-old-space-size=8192 node_modules/.bin/tsc --noEmit --incremental false` exits0, `/tmp/candidate-c545-media-tsc.log`. All owned files ready for parent review and commit; no commit/push performed.
