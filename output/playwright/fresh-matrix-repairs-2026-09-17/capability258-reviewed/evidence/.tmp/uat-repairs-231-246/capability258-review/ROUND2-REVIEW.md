# UAT258 round-two review — CLEAR

The round-one tests-only finding is resolved. The three actual-client cases now live in the maintained quickstart-auth suite and are included by the repository UI Vitest configuration.

- Foreign-origin and wrong-mode cookie markers each dispatch zero protected capability requests.
- A valid exact-origin single-user cookie dispatches one protected request.
- Each case verifies real stored-config normalization and request readiness, plus continued public OpenAPI/docs-info discovery.

The tests use the real singleton, initialization, `getConfig`, `ensureConfigForRequest`, capability discovery and active-cookie predicate. Only storage and network boundaries are mocked. Setup and teardown reset stored/session state, runtime overrides, cookie invalidation, environment state and singleton initialization; spies are restored. The exact diff against the prepared prior test consists of the singleton import and this three-case block.

**Independent verification: 25 passed, 0 failed**, including all three new cases, using the repository UI Vitest configuration. Both production files and the existing capability suite remain byte-identical to round one. The earlier causal red evidence and both prior review reports remain unchanged.

The author’s 120 adjacent passes and lint result (0 errors, one existing warning) were not redundantly rerun. Bandit’s TypeScript parser error means it provides no TypeScript security assurance. No native acceptance, compiler rerun or full matrix claim is made. No product, maintained test, Git, tracker, browser, runtime, model or database state was changed by this review.

Final maintained-test SHA-256: `385bc1afbd8f8d2fa36d2cfe51ea18458e263e14c55cbedf84b40f92aaa0b8c1`.

The prior production correction remains accepted as reviewed in `ROUND1-REVIEW.md`; this round closes its outstanding regression-retention gap.
