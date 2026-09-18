# UAT272 causal RED receipt

Command:

```sh
cd apps/packages/ui && bun run test -- src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts src/components/Option/WorldBooks/__tests__/WorldBooksManager.attachmentStage1.test.tsx src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx --maxWorkers=1 --no-file-parallelism
```

Exit: `1`.

Result: 3 files, 52 tests; 49 passed and 3 expected failures before the repair:

1. the collection request used `/api/v1/characters?limit=5` instead of the canonical trailing-slash route;
2. the Manager enabled its character candidate query before attachment tooling was requested;
3. the detail panel omitted a retryable attachment-load error.

This was the second and final planned red execution. No source or test was skipped.

## Review-revision RED receipt

The same three-file command was run after adding the review-requested causal tests but before changing production code. Exit: `1`. The three expected failures established that a `503` relationship read was silently converted to an empty attachment map, and that neither the Manager nor detail-panel error state supplied an actionable **Try again** control. The subsequent production change only preserves `403`/`404` as the established no-association result; it rethrows other errors and wires retry to the two existing query keys.

## Disabled-link causal RED receipt

Command:

```sh
cd apps/packages/ui && bun run test -- src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts src/components/Option/WorldBooks/__tests__/WorldBooksManager.attachmentStage1.test.tsx --maxWorkers=1 --no-file-parallelism
```

Exit: `1`. Result: 2 files, 45 passed and 2 expected failures. The full tool output is retained in `disabled-links-causal-red.log`.

The direct Base/client assertion received `/api/v1/characters/7/world-books` after an explicit disabled-link request, missing `?enabled_only=false`. The real Manager query function then received the endpoint-default filtered fixture and returned `{}` instead of the persisted disabled attachment metadata. No endpoint default, backend route, or fixture was changed for this reproduction.
