# UAT185 / TASK13260.122 implementation

## Result and scope
Ready for independent review: final combined suite159 passed, plus21 existing endpoint guards, zero skips. Three production files and two test files only; exact copies are in `review-snapshot/`, hashes in `source-freeze-manifest.json`, diff in `owned.patch`. No runtime/browser/provider/config/task/tracker/staging/commit actions were performed by this author. Native real-provider response/reload acceptance remains root-owned.

## Cause and repair
The native request omitted both optional provider and model. The real Chat adapter accepts `api_provider` and `messages`; the initial keyword-mismatch hypothesis was false. Study passed the absent values straight to that adapter instead of resolving server defaults.

The shared Study generator now calls the existing canonical `resolve_chat_target`, preserving aliases, configured default precedence and provider/model policy. It passes resolved identity to the actual adapter and returns it for assistant persistence. Grounding, action guidance, context/history, fact-check normalization, token/temperature settings and successful append order remain unchanged. Existing `ProviderCallPolicy(privacy_safe_errors=True)` normalizes terminal provider diagnostics without introducing a new transport policy.

Only the Flashcards and Quiz Study respond handlers map typed failures: configuration or upstream credentials400, bad provider request400, rate limit429, other provider failure502. Fixed public text avoids upstream details and distinguishes provider credentials from application login. Generation still precedes both message appends; tested failures leave the prior message IDs intact. This is not a claim of general atomicity if a database append fails after successful generation.

An adjacent test exposed `ByokResolutionError` escaping the canonical resolver's override-policy lookup. Four additional actual-route cases proved500. Study now translates only that typed target-resolution error locally to the existing safe configuration error, retaining its cause internally. No global resolver changes or policy bypass.

## Test boundary and fixture adjustment
The65 new cases call the real generator, canonical target resolver, actual async Chat builder and real router/storage/serialization. Only config sources and the terminal provider adapter are controlled. SQLite and official PostgreSQL fixtures exercise both Flashcard and Quiz response/reload paths. They cover explicit/default/qualified/aliased identities; existing default precedence; grounded fact-check output; missing/disabled/disallowed/unknown targets; policy-store failure; provider auth/request/rate/other failures; prior-history preservation; and stale version rejection before dispatch. No real model inference or model-quality claim.

The existing owner-guidance fixture had mocked terminal dispatch without initializing the now-used provider settings. Its healthy empty override snapshot and deterministic OpenAI/default-model sources are now explicit, while real target/policy resolution remains active. All36 existing assertions are AST-identical (`assertion-security-check.json`); owner storage, guidance edits/reset, immutable prompt snapshot and worker cleanup coverage remain intact.

## Verification receipts
| Run | Outcome | Meaning |
| --- | --- | --- |
| Initial adapter RED |25 failed /6 passed /0 skipped,30.82s|Original31 causal cases; supported aliases/explicit dispatch pass.|
| First expanded RED |54 failed /7 passed,59.34s|50 causal failures plus4 new fixture omissions (required input_modality). Retained separately.|
| Corrected expanded RED |50 failed /11 passed /0 skipped,56.88s|All61 cases have valid fixture semantics.|
| Unchanged adjacent baseline |94 passed /0 skipped,58.18s|Before production edits.|
| Initial focused GREEN |61 passed /0 skipped,89.48s|Approved initial implementation.|
| Initial adjacent GREEN attempt |14 failed /80 passed,82.86s|Exposed unavailable policy fixture/default settings; retained, not dismissed.|
| Policy-store RED |4 failed /61 deselected /0 skipped,7.78s|Actual resolver throws typed storage error; both routes/backends return500.|
| Final combined GREEN |159 passed /0 skipped,129.18s|65 new plus94 adjacent cases; corrected fixture and typed error mapping.|
| Final endpoint guards |21 passed /234 deselected /0 skipped,38.30s|Existing missing context/input/DB/conflict/persistence behavior.|

The only change during final combined execution was formatter insertion of one blank line in the new fixture; its test AST and production source were unchanged. Exact final source copies are frozen for independent replay.

Five-file Ruff:0 findings. New test formatter check, compileall and scoped diff check pass. Production Bandit:0 findings/0 errors. Test Bandit excludes pytest assertion ruleB101; oneB106 on the pre-existing `AuthPrincipal(token_type="access")` fixture matches baseline exactly, zero added findings. Baseline and final JSON receipts are retained.

## Reproduce
Activate the repository virtual environment first. The existing private runner supplies `TLDW_TEST_POSTGRES_REQUIRED=1` and uses official temporary database fixtures on the owned cluster; it creates no manual SQL databases and runs no live application/provider requests.

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat185-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/Flashcards/test_study_assistant_adapter_contract.py tldw_Server_API/tests/Flashcards/test_study_assistant_service.py tldw_Server_API/tests/Flashcards/test_study_assistant_service_prompts.py tldw_Server_API/tests/Flashcards/test_study_assistant_db.py tldw_Server_API/tests/Flashcards/test_study_response_timestamp_contract.py tldw_Server_API/tests/Chat/test_chat_target_resolution.py -q --tb=short
TLDW_UAT_EVIDENCE_LABEL=uat185-independent-guards node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py -k assistant -q --tb=short
```

For the smaller independent causal subset, run only `test_study_assistant_adapter_contract.py` (65 cases) with the same runner. Dedicated new tests override database/guidance dependencies; authenticated ownership is additionally covered by the existing owner-guidance suite. These checks do not newly certify full Chat BYOK admission, provider endpoint routing, or unrelated Study rating behavior (UAT187).
