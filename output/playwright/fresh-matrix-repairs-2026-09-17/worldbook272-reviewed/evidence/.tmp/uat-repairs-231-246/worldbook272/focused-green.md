# UAT272 focused GREEN receipt

Command 1:

```sh
cd apps/packages/ui && bun run test -- src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts src/components/Option/WorldBooks/__tests__/WorldBooksManager.attachmentStage1.test.tsx src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx --maxWorkers=1 --no-file-parallelism
```

Exit: `0`.

Result: 3 files, 56 passed, 0 failed, 0 skipped. Duration: 21.18 seconds.

Command 2:

```sh
cd apps/packages/ui && bun run test -- src/components/Option/WorldBooks/__tests__/WorldBooksManager.attachmentStage2.test.tsx src/components/Option/WorldBooks/__tests__/WorldBooksManager.attachmentStage3.test.tsx src/components/Option/WorldBooks/__tests__/WorldBooksManager.attachmentStage4.test.tsx --maxWorkers=1 --no-file-parallelism
```

Exit: `0`.

Result: 3 files, 6 passed, 0 failed, 0 skipped. Duration: 28.22 seconds. Vitest emitted its existing CSS stylesheet parse notices, which did not fail a test.

Combined focused result: 6 files, 62 passed, 0 failed, 0 skipped.

## Disabled-link follow-up GREEN receipts

```sh
cd apps/packages/ui && bun run test -- src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts src/components/Option/WorldBooks/__tests__/WorldBooksManager.attachmentStage1.test.tsx src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx --maxWorkers=1 --no-file-parallelism
```

Exit: `0`. Result: 3 files, 57 passed, 0 failed, 0 skipped. Retained full output: `focused-green-review2-main.log`.

```sh
cd apps/packages/ui && bun run test -- src/components/Option/WorldBooks/__tests__/WorldBooksManager.attachmentStage2.test.tsx src/components/Option/WorldBooks/__tests__/WorldBooksManager.attachmentStage3.test.tsx src/components/Option/WorldBooks/__tests__/WorldBooksManager.attachmentStage4.test.tsx --maxWorkers=1 --no-file-parallelism
```

Exit: `0`. Result: 3 files, 6 passed, 0 failed, 0 skipped. Retained full output: `focused-green-review2-controls.log`.

Combined follow-up result: 6 files, 63 passed, 0 failed, 0 skipped.
