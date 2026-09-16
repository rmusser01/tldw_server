# Independent review: UAT137 / UAT138

**Verdict: approved within the frozen scope; no actionable findings.**

Reviewed the eight manifest paths against base 3c30685611. All 8 SHA256 hashes still match /private/tmp/cycle5-uat137-138-owned.json; independent audit is /private/tmp/cycle5-uat137-138-independent-hash-audit.json.

## Assessment

- UAT137: the remaining count reaches both English ICU messages, with matching visible singular/plural wording and accessible status. The permanent test mounts the actual ReviewProgress with the actual ICU adapter and English resources; it is not a mocked translator assertion. Empty queues remain hidden. English fallback strings correctly retain interpolated remaining/reviewed values; the existing ICU adapter converts these placeholders, and installed ICU defaults do not memoize missing-resource fallbacks. The public English resource mirror matches. Only presentation/resources changed, with queue, Cram and re-rating controls passing; no scheduling or queue mutation change is present.
- UAT138: uses the existing pure request-failure classifier, without changing authentication, request dispatch, endpoint-missing gating or search mapping. Known transport/HTTP failures produce a neutral warning and existing user feedback; unknown exceptions still use console.error with the original error. The actual hook runs under a real QueryClient with only storage and API transport mocked. Recovery refetch succeeds on the same mounted hook and displays the preserved source. Endpoint-missing and unexpected-error controls are meaningful. This is a bounded console-overlay repair, not a new outage recovery mechanism.
- Examined retained RED logs: the real ICU singular case failed before the change; the corrected request-path outage fixture failed specifically because the transport error reached console.error. The invalid earlier 404 fixture was accurately excluded from the claimed product RED.

## Independent verification

From apps/packages/ui, using only the installed local binary:

```sh
node_modules/.bin/vitest run src/components/Flashcards/components/__tests__/ReviewProgress.test.tsx src/components/Flashcards/components/__tests__/ReviewProgress.plurals.test.tsx src/components/Flashcards/components/__tests__/ReviewProgress.responsive.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.queue-state.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx src/components/Review/hooks/__tests__/useMediaSearch.outage.test.tsx src/components/Review/__tests__/MediaReviewPage.stage3.search-filter-sort.test.tsx src/services/__tests__/backend-unreachable.test.ts --maxWorkers=2 > /private/tmp/cycle5-uat137-138-independent-tests.log 2>&1
```

**65 passed / 9 suites, exit 0, no skips.** Independent comparison of author baseline/current lint JSON confirms identical 42 rule/severity/message diagnostics. No additional lint run was needed for the unchanged frozen files.

## Limits

No source/test/task edits, browser/runtime/inference actions or commits. No native acceptance claim; native recheck remains pending. Whole compiler validation is root-owned. Existing localStorage and adjacent mocked-control React warnings remain in the test log. Review excludes other agents' UAT013/140 changes and unrelated working-tree files. Python Bandit does not apply to these TypeScript/JSON changes.
