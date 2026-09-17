# UAT249 / TASK13260.191 — speech timeout test fixture

Test-only candidate frozen in source-freeze.json. Root owns task/docs/integration/native acceptance. Exactly one existing test file changed; no production, timeout, auth, transport or sanitizer implementation changes.

## Cause and minimal correction
The original fixture directly assigned private client.config while mocked storage always returned null. Actual WebUI requestWithCurrentConfig reloads configuration; both speech tests therefore failed as unconfigured before dispatch. The fixture now retains get/set/remove data in a resettable in-memory map and uses awaited public updateConfig, exercising normal credential persistence and configuration validation.

After fixing that, the old bgRequest-only speech mock was also stale: current WebUI dispatch uses request-core.tldwRequest. The fixture now mocks that actual transport boundary (successful response envelope) and inspects its /audio/speech init. It does not force an extension surface, bypass ensureConfigForRequest, replace requestWithCurrentConfig, or suppress errors. Chat completion sanitizer cases retain their existing bgRequest boundary.

All original assertion lines are unchanged, including default timeout>=120000, explicit5000, and successful error-looking assistant content. assertion-preservation.json binds the unchanged lines. Four scenarios retained; none skipped or removed.

## Receipts
- red.log: original **2 failed speech /2 passed sanitizer**; causal failure is unconfigured client.
- intermediate-web-transport-failure.log: configuration repair reaches the WebUI path; speech's stale bgRequest mock allowed fetch and both fail with fetch failed, while39 adjacent controls pass. No runtime was started; this is retained as a fixture correction, not product regression.
- green-focused.log: **9 passed/2 files, zero skips** (all4 sanitizer/speech plus5 configuration-guidance cases).
- adjacent-concurrent-scope-red.log: **43 passed/1 failed across3 files**. Newly added Character stream captured-scope test belongs concurrently active UAT248; root confirmed ownership. It remains unchanged, not deselected or weakened. Root will run final combined adjacency after248 is green. Historical green.log retains this same non-green run; its filename does not establish acceptance.
- Scoped actual-root ESLint: **0 errors/4 warnings**, baseline0/5, no added warning signatures. Scoped git diff --check exit0.
- Bandit not applicable to this TypeScript-only test fixture. No global compiler run; root owns final compiler check.

## Exact commands
From apps/packages/ui:

    node ../../tldw-frontend/node_modules/vitest/vitest.mjs run src/services/tldw/__tests__/TldwApiClient.sanitizer.test.ts --reporter=dot
    node ../../tldw-frontend/node_modules/vitest/vitest.mjs run src/services/tldw/__tests__/TldwApiClient.sanitizer.test.ts src/services/tldw/__tests__/TldwApiClient.configuration-guidance.test.ts --reporter=dot
    node ../../tldw-frontend/node_modules/vitest/vitest.mjs run src/services/tldw/__tests__/TldwApiClient.sanitizer.test.ts src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts src/services/tldw/__tests__/TldwApiClient.configuration-guidance.test.ts --reporter=dot

From root: node .tmp/uat-repairs-231-246/speech-fixture249/lint.mjs. This uses actual root ESLint configuration, with baseline original bytes and current bytes, not a file-ignore shortcut.

No native speech/provider acceptance claim follows from these isolated mocked-transport tests.
