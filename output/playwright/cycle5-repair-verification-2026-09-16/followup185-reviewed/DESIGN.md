# UAT185 / TASK13260.122: Study provider dispatch

## Diagnosis
Native Flashcards Study omitted both optional provider and model. The actual async Chat builder accepts api_provider and messages aliases; it requires a provider and does not resolve Study server defaults. Native therefore raises ChatConfigurationError before inference. Changing the keyword alone cannot repair this request.

## Bounded repair
Resolve one target in generate_study_assistant_reply through the existing resolve_chat_target helper. Preserve aliases, grounding, owner guidance, context/history, fact-check normalization and sampling values. Pass resolved provider/model and ProviderCallPolicy(privacy_safe_errors=True) to the actual adapter. Return resolved identity for assistant persistence. No endpoint override, routing, fallback, credential admission framework or inference behavior changes.

Translate typed Chat failures only in the two shared Flashcards/Quiz Study respond routes. Configuration and rejected upstream credentials get safe actionable400 (upstream credentials must not look like application401). Bad provider request400, rate limiting429, provider/unexpected terminal failure502. Details are fixed public strings; no raw diagnostics. Both routes generate before appending, so these failures add no user/assistant rows. This does not certify general atomicity of the two successful append operations or new authentication/BYOK admission semantics.

## Causal tests
Real shared service -> canonical resolver -> actual async Chat request builder -> registry terminal test adapter. Controlled config sources and terminal inference only. Real SQLite and official PostgreSQL fixtures; actual Flashcard and Quiz routers and actual response serialization/reload. Dependency overrides choose test DB/guidance; not authentication acceptance. No live provider calls.

Initial31 cases:25 expected failures,6 passing controls,0 skips. Explicit targets including original aliases pass; default target/normalization5 failures; HTTP omitted defaults4 failures; missing-model/disabled-policy8 failures; upstream/rate-limit8 failures. Initial source/tests/log retained before further controls.

## Ownership and limits
Only core/Flashcards/study_assistant.py, endpoints/flashcards.py, endpoints/quizzes.py and new tests/Flashcards/test_study_assistant_adapter_contract.py. Production remains held until root completes native capture. Root owns tasks/shared plans/browser/runtime/config/commits. UAT187 rating-response datetime serialization is separate.

## Expanded RED and baseline checks
The first expanded run had50 causal failures,7 passing controls and4 test-setup failures (the newly seeded prior-message control omitted required input_modality). That fixture omission was corrected; the retained complete rerun is50 failed/11 passed/0 skipped in56.88s. All failures are default/target-policy/safe-error contract assertions. Current31-case initial RED remains independently valid. Four stale-version controls now pass before repair, along with the explicit adapter/route and fact-check positives.

Unchanged adjacent baseline:94 passed/0 skipped in58.18s across Study service, owner guidance, DB, timestamp response and canonical target suites. Three proposed production paths have clean baseline Ruff and Bandit scans (zero findings/errors). Proposed patch is private only. Local target imports preserve the existing context-only limited-import behavior; target resolution follows prompt/action validation, preserving error order.
