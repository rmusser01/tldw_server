# UAT246 proxy diagnostic — independent review

**CLEAR as bounded causal proxy coverage; original native UAT246 remains open.** No production change is introduced by this diagnostic extension.

Reviewed `quickstart-proxy-timeout.test.mjs` SHA `19643c1987d996d0f109d0eb877ce1f2b2dbbcc97969b3de78a26e90183546b2`, using the installed Next server and application's actual quickstart configuration. The only additions are a controlled SSE upstream case and its timing/content assertion; existing forwarding/body/auth, upstream503, cancellation and delayed-generation controls are unchanged.

## Independent verification

Full real-wall-clock fixture: **5 passed, 0 failed, 0 skipped**, 63.7569 seconds; owned temporary upstream/Next processes and fixture use the test's existing cleanup. Scoped ESLint: **0 errors, 0 warnings**. Node syntax passes. No Python production was touched; Bandit is not meaningful JavaScript security analysis.

The SSE control receives HTTP200 at **3ms**, then the complete expected role/content/DONE stream at **31,505ms**. Upstream emits content at **31,500ms** and closes normally at **31,501ms**. Delayed JSON generation also completes after **31,503ms**. The test's 40-second client deadline and 45-second test limit bound the run.

The author default-setting control was independently source-reviewed: it changes only fixture app/config paths and removes `experimental.proxyTimeout` from the otherwise current real config. Retained timed result: early headers at **3ms**, upstream close at **30,003ms** without content emission, downstream read finally aborting at its **40,003ms** client deadline. This establishes that the default proxy can return early200 and then leave the reader idle after closing upstream, while the configured180000ms proxy survives the tested silence. The independent run uses the corrected config; the default control was inspected, not rerun.

## Evidence quality and limits

The first headers-only attempt incorrectly assumed flushed upstream headers would reach the client immediately. Both original snapshots/logs remain bound in the reviewer manifest; the final test emits an actual role frame before the silence. It asserts early response establishment and final full content; it does not timestamp each semantic token or test every possible idle interval up to180 seconds.

This is real Node HTTP → installed Next external rewrite → controlled HTTP/SSE upstream. It is not the complete-v2 backend, real provider, browser fetch, production build, native archive, or a reproduction of the original TestBot45-second symptom. No real authentication credentials are used; the authorization assertion uses a fixed synthetic token. No new timeout policy is proposed here. The original native246 cause remains a plausible hypothesis requiring the planned passive correlation; these passing controls do not close it.

Source/config/package hashes stayed stable through the independent run. This review made no product/test source, task, tracker or Git edits and did not touch held native runtimes or databases.
