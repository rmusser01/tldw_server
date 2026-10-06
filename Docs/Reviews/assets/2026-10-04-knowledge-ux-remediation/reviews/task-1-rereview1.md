# Task 1 scoped re-review — fix round 1

Scope: the prior Important arrival-before-hydration finding and new breakage introduced by `2a293bc6bbca347a0eef18d841925fe4730811db..f0d34148b4c4ec95e343c47944bc807bf008068a`. Read the exact supplied package once, the Task 1 brief, prior review, and appended fix report. Product and Git state were read-only; this review report is the only write.

## Prior finding

**ADDRESSED — Block queries until an explicit arrival has been applied.**

- `apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx:1786`: applied arrival is committed state. Pending derives from the current explicit route key/search versus that committed arrival, so it is true on the initial loading render and again when a mounted route receives a different arrival.
- `KnowledgeQAProvider.tsx:2414`: the shared query gate returns before search state, effective query scope, thread creation, message persistence, or private RAG requests. Search, follow-up, token-limit rerun, and history rerun all reuse this gate.
- `KnowledgeQAProvider.tsx:2026`: shared thread creation also rejects pending arrivals. The direct `startNewTopic` path at `:3087` cannot create the thread that formerly stranded defaults hydration. Branch/retry callers also reuse this shared creation guard.
- `KnowledgeQAProvider.tsx:1873`: the effect commits the arrival together with the existing clear/reset and exact-scope dispatches. The pending render remains gated, and the applied render supplies the new exact settings. Existing invalid rejection at `:2421` remains after application; deliberate source selection recovery at `:3498` remains usable.
- `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.scope-handoff.test.tsx:117`: new regressions retain loading while asking on exact/invalid arrivals, assert no thread/messages/answer, release hydration, and derive the eventual answer from API media IDs (`Scoped 3,7` or deliberately recovered `Scoped 9`). The new-topic regression at `:162` and same-route exact/invalid regressions at `:178` cover the other identified entry paths. Existing cancellation/replacement coverage at `:209` remains intact.

## New breakage within the fix

- Critical: none found.
- Important: none found.
- Minor: none found.

The change uses two shared entry gates and existing hydration/reset behavior. No public helper, route, provider-authority contract, or downstream caller signature changes were introduced. Callback/effect dependencies include the new committed/pending state, preventing the normal rerender path from retaining a pre-arrival entry gate.

## Evidence and limits

Reviewed `/private/tmp/task1-fix1-red.log`: five outcome regressions failed before the product fix (early thread creation and old-scope answers), with 11 existing tests passing. Reviewed `/private/tmp/task1-fix1-final-green.log`: the scope-handoff suite and authority suite pass, 50 tests across two files. Reviewed format and pre-commit logs: formatting and applicable hooks pass. No covered suites were rerun; source inspection left no specific unresolved code doubt requiring another focused test.

Unchanged out-of-scope observations remain non-blocking for this round: baseline Node/i18next harness warnings; repository-wide type diagnostics; Task 2 ownerless prefill/error semantics; controller-owned real-browser verification. The fix report correctly states that Bandit analyzes zero Python lines in this TypeScript scope rather than claiming TypeScript coverage. This round makes no browser, clean whole-repository typecheck, push, merge, or original-checkout-cleanliness claim.

**Spec Compliance: PASS for amended Task 1 within the assigned re-review scope.** The previously failing explicit exact/invalid arrival contract is now enforced through both query and direct topic entry paths, and same-route replacement plus deliberate invalid recovery are preserved. Earlier passing portions were not re-audited.

**Task quality: PASS within the assigned re-review scope.** The minimal shared fix has meaningful RED/GREEN outcome evidence and introduces no identified new blocking regression. Previously recorded integration/baseline concerns remain assigned to their existing owners.

**Round verdict: PASS — prior Important finding addressed; no new Critical/Important/Minor breakage identified.**
