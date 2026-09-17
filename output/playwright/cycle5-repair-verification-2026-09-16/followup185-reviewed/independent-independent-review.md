# UAT185 independent review

TASK13260.122. Verdict: **clear bounded source/test review; ready for parent-owned native acceptance**. No production/test edits, browser/runtime actions, provider inference, task changes or git mutations by this reviewer.

## Fresh independent test

From the repository root, using the existing official isolated PostgreSQL runner with mandatory PostgreSQL and SQLite controls:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat185-independent-review node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/Flashcards/test_study_assistant_adapter_contract.py -q --tb=short
```

**65 passed, zero skipped,5 warnings,68.63s; exit0.** The exact command receipt and redacted log are copied beside this report. This is13 generator/target controls and52 actual route/storage controls:26 PostgreSQL and26 SQLite. No live application DB or real provider was used; only terminal inference/configuration sources are controlled, with the real target resolver, Chat request builder, routers, storage and serialization retained.

The wider author159-case run and21 existing endpoint guard run were inspected as retained evidence, not rerun or represented as independent executions. I also inspected the causal50-failure/11-control expanded RED and separate4-failure policy-store RED receipts. Production Bandit has0 findings/errors; the test B106 on the pre-existing `token_type="access"` fixture exactly matches baseline. These static receipts are author executions.

## Source and contracts reviewed

The three production files and two test files match the author's frozen copies and exact hashes, independently checked before the run and rechecked afterward. `source-review.json` records all hashes and snapshot equality; `verification-manifest.json` records retained evidence hashes.

1. **Target resolution:** `generate_study_assistant_reply` now calls the existing canonical `resolve_chat_target` before dispatch. That resolver handles omitted/partial values, inline provider-qualified model IDs and aliases, applies existing configured-default precedence, checks registered adapters and validates provider/model overrides. Study introduces no alternative resolver or fabricated provider/model identity. The returned resolved target is the same target dispatched and persisted on the assistant message. Unit controls cover explicit, default, partial, alias and qualified choices plus administrative/environment/config default precedence.
2. **Policy failure:** the resolver's override-validation lookup can raise `ByokResolutionError` outside its own generic candidate catch. Study catches only this typed target-resolution error and raises the existing safe configuration class with its cause retained internally. It does not disable policy, change the global resolver or proceed with an unavailable policy store. Both routes/backends prove400 and zero provider calls for this case, as well as missing model, disabled provider, forbidden model and unknown provider.
3. **Error privacy and login:** the existing `ProviderCallPolicy(privacy_safe_errors=True)` affects provider error normalization without adding deadlines, endpoint overrides, retries or other policy settings. I inspected the actual async adapter and `privacy_safe_chat_error` implementation: terminal provider failures become bounded typed errors. Both routes map configuration, provider credentials and bad provider requests to fixed400 text, rate limits to429, other provider failures to502. No upstream body/path/message is interpolated into public details. This keeps provider authentication out of the frontend401 application-refresh path inspected in `request-core.ts`/`background-proxy.ts`; no native logout behavior is claimed. Actual route tests reject the seeded private marker in every failure response. Unexpected local endpoint errors retain the existing generic500 path; only failures raised at the provider boundary use provider normalization.
4. **History and owner boundaries:** generation still completes before either message append. Tested target/provider failures leave exactly the original prior message ID; stale versions return409 before dispatch or append. Success on both routes/backends persists one ordered user/assistant pair with the same thread and resolved assistant identity, and GET reload retains those IDs, answer and timestamp strings. Existing context lookup, owner DB dependency, guidance snapshot and append/version code are unchanged. This does not claim atomicity for a later database append failure after successful generation.
5. **Fixture integrity:** I independently parsed the old/current owner-guidance test AST: all36 assertions are unchanged and ordered identically. The only fixture adjustment supplies healthy deterministic provider settings required by newly real resolution. Owner-specific guidance storage, action selection, edit/reset, immutable in-flight owner/guidance snapshot and same-worker cleanup assertions remain present. No production auth dependency was changed.

## Limits

This review certifies the bounded default-target/typed-error repair and the65 focused controls. It does not certify real model answer quality, native response/reload UX, full Chat BYOK admission/routing, every provider implementation, concurrent append atomicity, or unrelated Study rating behavior. Native acceptance remains the parent's next step. No actionable remaining defect found in the reviewed scope.
