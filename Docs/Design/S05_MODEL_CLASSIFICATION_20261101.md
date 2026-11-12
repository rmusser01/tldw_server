# Explicit moderation model classification

Task: TASK-13260.278.18.83.45. Human selected “Implement classification, then S06.”

## Behavior

Add a Model classification panel to the existing Admin Rules sandbox. An administrator supplies one criterion, a fictional sample and an explicitly selected configured provider/model. Explain before submission that the sample and result are saved to the staff Review queue. One bounded nonstreaming call returns a strict benign/flagged result, short reason and exact sample evidence. No confidence score is fabricated. Both outcomes remain needs_review; only existing human controls decide them. Deterministic testing and live Chat enforcement remain unchanged.

Inputs are limited to 4,000 sample characters, 500 criterion characters and bounded provider/model identifiers. Provider URLs/keys, actor identity, verdicts and provenance cannot come from the client. A client request UUID identifies a saved result for recovery; returning an already-saved result does not call the provider again. Reusing an ID with different inputs is a conflict. Concurrent requests can spend more than one inference but create one item; no cross-worker exactly-once inference promise is made.

Reject malformed, truncated, refused, ungrounded or oversized output. Provider timeout/unavailability never means benign. Keep the form and request ID after uncertainty; explicit retry first reads the saved receipt. A changed form creates a new request. No automatic retry/fallback or tool use.

## Existing boundaries

Use rules_router admin + SYSTEM_CONFIGURE and the authenticated request's credential runtime/accounting/egress context. Do not recurse through HTTP Chat. Use perform_chat_api_call_async with immutable single-attempt policy, a 120-second provider deadline, bounded output and exact configured origin. Support only audited transports; initially the tested strict native-async llama.cpp transport. Other providers receive a clear unsupported error before dispatch. Native async cancellation must close the owned HTTP connection; normal llama.cpp traffic retains its existing dispatcher behavior.

Save through ModerationReviewService/ModerationReviewStore. The generated item carries server-authored provider/model/criterion/reason/evidence provenance, input digest, authenticated actor and request ID. Item insertion and item.created audit remain atomic in the existing SQLite store; new classification creation is attributed. No raw samples or provider error text enter operational audit/logs. Human decisions use the unchanged audit/Undo flow.

The UI uses house controls/model catalog, captured account authority and request retirement. Stop/unmount/account change aborts the owned request; late completions cannot publish into another account. A server may finish persistence after a disconnect, so recovery checks the same receipt rather than claiming rollback.

## Scope and verification

Reach: API schemas/router, small classification service, existing Review insertion attribution, strict llama.cpp transport/dispatch certification, shared UI service/panel, API generated contracts, tests and user guidance. No dependency, new worker framework, schema migration, automatic policy hook, model download or provider-service changes.

Causal tests cover trust boundaries, strict JSON/evidence, one-call/failure/cancel, receipt recovery/conflict, authentic saved/audit provenance and permissions. Transport tests require real socket retirement and exact-origin/one-attempt bounds. UI tests cover result, uncertain retry/Stop and owner retirement in both clients. Then normal lint/types/scoped baseline-current Bandit, canonical export/types, independent actual/staged review, hooks/local commit, offline package/build and actual configured-provider benign/flagged plus recovery checks in supported modes. Screenshot human UX review and disposal finish S05 before S06.

ADR assessment: no new durable architecture rule. Reuse ADR018 authority, ADR025 provider dispatch, ADR026/030 egress/exact configured origin, ADR067 accounting and ADR059 task editing. Existing synchronous providers are not silently certified as cancel-safe.


## Governed admission

Classification is cost-sensitive. An exact path before the broad moderation glob reuses chat.default request admission; other Rules/Review routes retain core.default. In DB policy-store mode, operators retain ownership of the existing chat.default DB policy (already used by Chat); the file route map takes precedence and ADR056's bounded default fallback still applies. Middleware reserves request units only. LLM token/cost quotas and best-effort accounting remain in the provider dispatcher; no token reservation or committed usage-receipt guarantee is invented here.

Governing records: [ADR018](../ADR/018-resource-governance-endpoint-policy-and-route-map.md), [ADR025](../ADR/025-llm-provider-adapter-routing-and-overrides.md), [ADR026](../ADR/026-security-outbound-egress-and-ssrf-policy.md), [ADR030](../ADR/030-configured-local-llm-egress-policy.md), [ADR056](../ADR/056-resource-governor-safety-net.md), [ADR067](../ADR/067-llm-calls-accounted-in-provider-base-class.md), [ADR059](../ADR/059-backlog-py-task-editor-cutover.md).
