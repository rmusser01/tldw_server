# UAT231/232/236/243 — frozen author handoff

Tasks: TASK13260.173 / .174 / .178 / .185. Parent owns task records, integration, and native acceptance. This unit changes six frontend production files and ten tests; the exact sixteen files and review snapshots are bound by owned-manifest.json (SHA256 43713006f45a461a49d7c6a5e202c98c94838c2da60c6e14ca9bdf3df4a97c8c). No backend, browser, runtime, provider, or timeout changes.

## Problem and resulting behavior

- **231:** Missing configuration previously always instructed WebUI users to open the extension. TldwApiClient now uses the existing browser-surface detector; extension guidance remains available on extension protocols.
- **232:** Direct transport retains model_not_available inside Error.details, while generic message formatting discards the code. The shared error formatter now recognizes that actual envelope and uses the existing actionable unavailable-model copy and model-selector action. The existing sanitizer limits displayed details. Real useChatActions → streamCharacterChatCompletion → bgStream → controlled fetch proves failed HTTP400 then successful Retry keeps the conversation, greeting, acknowledged user ID, and request context; no new conversation or duplicate user write.
- **236:** Existing llama catalog, picker, and setup aliases disagreed. Closed known aliases now resolve to llama.cpp without altering the model suffix, and Settings uses the existing consolidated useSelectedModel owner. A mounted Settings control proved the old storage-only setter was overwritten by live store state. Wrong-provider, explicitly unconfigured, and catalog-only negatives remain blocked.
- **243:** Ordinary chat creation publishes its server ID and resolved metadata during the turn. The success closure still captured a null initial ID and re-selected that same ID, clearing the resolved metadata. The finalizer now avoids re-selecting an already-current ID inside the existing authority guard. Real action and Playground coordinator controls prove the newly created ordinary chat immediately gets ordinary presentation after a controlled account transition, while the intentional empty-chat Character preference survives.

No production Playground preference, stream cancellation, request serialization, auth ownership, or Retry branching behavior changed. ModelsBody's existing test fixture now mocks the consolidated hook with the same previous selection and setter; its existing assertions remain.

## Causal evidence

All receipts remain in this packet, including unsuccessful harness attempts.

- utility-contract-red.log: 10 failed / 93 passed before utility corrections. One undefined-provider-flag expectation contradicted existing legacy eligibility and was corrected to the explicit catalog-only negative; it was not used to change the product contract.
- models-owner-confirmed-red.log: mounted real Settings/store selection remained on the stale model before the consolidated hook change. Earlier models-owner logs preserve fixture corrections for grouped options and actual null Auto representation.
- ordinary-metadata-causal-red.log and ordinary-coordinator-causal-red.log: actual action metadata and immediate coordinator presentation each failed on original source. Earlier similarly named logs preserve missing fixture collaborators rather than being counted as product proof.
- actual-boundaries-baseline-red.log: first nonmutating original-source replay reproduced four failures. The final replay, final-actual-boundaries-baseline-red.log, reproduces **five expected failures** across actual transport, action, coordinator, and both banner variants; 206 unrelated cases are deselected by the explicit test-name filter. No source rollback was performed.
- Initial real-transport and banner failures caused by legacy credential fixture shape or the incorrect inline-Retry button expectation were corrected in tests and retained. The final test uses normal synthetic device credential metadata and existing mounted banner actions.

## Final verification

Exact argv arrays are in commands.json; run them from the repository root. vitest.config.ts uses the existing frontend config and aliases an already installed Bun pa-tesseract.js package because its UI package link is absent. This is a private harness adaptation, not a dependency install or production config change. vitest.baseline.config.ts loads the six retained original production files at their actual logical paths.

| Verification | Result / receipt |
| --- | --- |
| Focused final | **327 passed, 0 skipped, 10 files**, 39.22s; final-focused-green.log |
| Pertinent adjacent | **103 passed, 2 failed, 8 files**; adjacent-green.log |
| Adjacent failure baseline | Same **2 failed / 2 passed** in unchanged TldwApiClient.sanitizer.test.ts on original client source; adjacent-sanitizer-baseline.log |
| Final actual-boundary baseline | **5 expected failures**, 206 deselected, 4 files; final-actual-boundaries-baseline-red.log |
| Scoped ESLint | **0 errors, 623 unchanged warnings**, all sixteen logical paths parsed; eslint-comparison.json has zero added/removed messages |
| Full frontend TypeScript comparison | **90 baseline / 90 current, zero added/removed diagnostics**; tsc-comparison.json |
| Bandit | 0 findings, **16 TypeScript/TSX parse errors**; unsupported-language limitation, not security assurance |
| Whitespace | diff-check.txt passes |
| Freeze | freeze-verification.json confirms all sixteen live sources and snapshots match |

The two adjacent speech tests assign a private client config field while getConfig resolves mocked storage returning null. They fail before speech transport on both original and current code. No assertions were weakened and no unowned test was edited. All seven other adjacent files pass. The initial compiler run exceeded its default 4GB heap; the retained expanded-heap run completes at 8GB. Both receipts are retained.

## Source provenance

Retained source baseline is commit 84f1d4e2511e7c52a7bb4128999192fbb3342aa7. The manifest captures freeze HEAD 4236986b145451ea30743f64a500f29f3bc18d87; parent integrated disjoint work meanwhile. owned.patch and the exact retained baseline files identify this unit independently of branch movement. The current sixteen source hashes were checked after compiler completion. Source and test freeze is final; report/manifest packaging can proceed without code changes.

## UAT246 and remaining limits

Five controlled lifetime tests were added to the real Character transport harness as diagnostics only. They qualify the configured 45-second idle budget, explicit 12-second budget, role/reasoning bytes resetting idle time across 80 seconds, caller abort, and caller abort together with snapshot invalidation. They do not establish original native TestBot causality or justify a timeout change.

DIAGNOSTIC246-frontend-lifetime.md separately preserves an invalidation-only candidate and its exact intermediate test. It uses mocked snapshot acquisition; it is not an established actual-auth lifecycle defect and this patch does not repair it. The misleading historical transport-lifetime-green.log name is retained although that intermediate run contains a failure. Parent owns separate backend246 proof.

Independent review and targeted native acceptance remain pending. This author packet does not close the matrix gate or claim real-provider/native acceptance.
