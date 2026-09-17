# Independent review — UAT232 ordinary Chat cause-chain correction

**CLEAR for the bounded three-file implementation. Native acceptance remains pending.** No actionable source or test finding remains after removing the new test's unnecessary `any` annotation.

Task TASK-13260.174; baseline `5aea3524e69179ef6f48edce333a9c477a70d71c`. Reviewed uncommitted production formatter and two test files only. The reviewer changed no product/test source, Git state, task, runtime, browser or database.

## Source and behavior

The recorded native ordinary completion returns an actual HTTP 400 `model_not_available`, while its UI still gives generic server/Health guidance. The direct transport retains the structured envelope, but `TldwChatService.streamMessage` wraps it in `Error('Stream completion failed', { cause })`. The former formatter checks only the outer envelope.

The correction walks the existing `cause` chain, checking the same structured `error_code` locations at each object. A visited-object set terminates ordinary cycles. When it finds a model-unavailable envelope, it sends that matched error through the existing server-message sanitizer before encoding the existing model-selection action. It preserves the outer-message handling for unrelated errors and retains the existing top-level image-support and API-key precedence. No request, retry, auth, timeout, persistence or provider logic changes.

The integration test keeps the real ordinary Chat service wrapper, domain streaming adapter, direct HTTP error parsing, SSE parser and assistant-error formatter. Its HTTP boundary and configuration storage are controlled. The first response is a structured HTTP 400; the second is a successful SSE completion. Assertions bind the real endpoint, unchanged conversation/client-message IDs, unchanged request contents and explicit failed-turn retry metadata. This establishes the service/formatter boundary; it is not a full native React workflow.

The utility tests additionally check sanitized nested detail and an unrelated cyclic chain. Existing sanitizer, local image-support provenance, abort, banner, saved ordinary Chat and actual Character-lease controls remain exercised. No broader arbitrary error-unwrapping behavior is introduced.

## Independent verification

| Check | Result |
| --- | --- |
| Final formatter and actual ordinary transport focused tests | **20 passed**, 0 skipped, 2 files |
| Character/lease, error banner, TldwChat abort, sanitizer neighbors | **84 passed**, 0 skipped, 4 files |
| Saved ordinary Chat pipeline | **99 passed**, 0 skipped, 1 file |
| Final tests with only production formatter replaced by baseline via Vite transform | **2 expected guidance failures, 18 passing controls** |
| Scoped ESLint, final three files | **0 errors, 0 warnings** |
| TypeScript baseline/current, project plus touched test roots | **90 / 90 existing diagnostics**, no additions/removals |
| Scoped Bandit, all three TS files | 3 AST parse errors; **no meaningful TypeScript security coverage** |
| Scoped whitespace diff check | Passed |
| Final source hashes | All three stable after verification |

Total successful focused/adjacent coverage is **203 tests across seven files**. This is not an all-repository test or clean-typecheck claim. The baseline replay replaces only the production formatter in the Vite transform; it does not edit or revert working source.

## Failed launches and corrections retained

- The author's initial RED had i18n/storage setup gaps. The corrected author RED and the independent baseline replay both isolate the two actual guidance failures, with 18 passing controls.
- The author's broad UI-package configuration cannot resolve `@web/lib/auth` for the Character suite. The independent frontend-config launch passes the four relevant suites but cannot collect saved-normal tests because that config lacks the installed OCR alias. Saved-normal was therefore run separately with its existing UI-package config and passed all 99 tests. Initial failed launches remain in their logs; they are not hidden or counted as green runs.
- The first combined compiler invocation exhausted Node's default heap after producing baseline diagnostics. Running current mode separately with an 8 GiB limit completed normally and matched all 90 baseline diagnostics. The OOM log is retained.
- Review found one added `no-explicit-any` warning in the new request-body test annotation. The author changed it to `Record<string, unknown>`; final focused tests and scoped lint were rerun on those bytes and pass.

## Boundaries and evidence

The earlier native failed response is diagnostic input, not acceptance of this correction. The original visible unavailable-model interaction, successful Retry and canonical reload must still be repeated on the newly accepted source. The deterministic successful retry in this test does not establish actual model availability or database persistence.

`audit.json` binds source, relevant supporting code/configuration, native diagnostic evidence, author RED/green receipts, and independent commands/logs. Full native content and private runtime data are not projected. Exact scoped hashes are recorded there. The retained TypeScript/lint results are differential; Bandit's inability to parse TypeScript is not represented as a security pass.
