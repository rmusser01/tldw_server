# UAT024 generation HTTP error presentation

Task: TASK13260.6 (existing UAT024 reopened). Parent owns task records, native acceptance, runtime/config restoration and integration.

## Evidence and cause to establish
The actual authenticated generation fixture rejected an unsupported numerical claim with HTTP422. GeneratePanel already catches this and maps it to actionable source guidance, but its mutation logs the same Error with console.error. Next Pages development error handling promotes that log to its runtime overlay. The existing UAT175 save regression observes the installed Next handler and real panel/service/request stack.

## Approved bounded design
Extend the existing mounted Next regression with generation422 through fetch, preserving source/count/type/difficulty/focus/provider/model, no drafts or save writes after rejection, actionable inline feedback, no raw JSON or runtime overlay, and subsequent successful generation. Preserve real TypeError diagnostic behavior as a negative classification control.
After causal RED, reuse the existing local expected-HTTP classifier for generation. Rename its create-specific name/comment only as needed; preserve both create handlers and their behavior. Keep the rejected promise, error object, HTTP status and console warning available. No global console suppression, verifier changes, GeneratePanel changes or additional retry behavior.

## Stages
1. Diagnosis and causal test: complete (1 meaningful RED, 7 controls PASS). Meaningful RED must isolate Next overlay dispatch, with previous save/unexpected-error controls passing.
2. Minimal hook change: complete. Wire generation onError through the existing classifier.
3. Verification/review packet: complete (97 tests / 9 suites PASS; native pending). Focused mounted/service/hooks suites, scoped ESLint, compiler comparison, Bandit applicability, diff and exact hashes. Native reacceptance remains parent-owned.
