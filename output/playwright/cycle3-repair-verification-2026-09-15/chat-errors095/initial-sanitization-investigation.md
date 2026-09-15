# UAT034 broader-suite sanitization investigation

Read-only investigation, 2026-09-15. No repository edits, browser/runtime actions, installs or commits.

## Conclusion

The reported `tldw-api-client.chat-mutations` failure is an obsolete success-body assertion, not a .34 regression. It mocks `bgRequest.mockResolvedValue(...)`, i.e. the successful transport contract, and asks `createChatCompletion` to remove error-like content. The committed TASK-12091 / FRONTEND_AUDIT C1 repair deliberately removed that behavior because it corrupted legitimate assistant replies discussing errors and file paths. The companion `tldw-api-client.chat-sanitization-regression` suite verifies preservation and passes all five controls.

There is a separate error-only sanitation gap, confirmed by a synthetic real transport probe: a non-2xx Chat error containing a private path remains in the thrown error message, console request warning, and stored request diagnostic. This is conditional on the API returning such content; no live backend disclosure was observed in this investigation. Reintroducing success-body sanitation would not repair this path because the promise rejects before the success wrapper runs.

## Evidence

- Existing tests command from `apps/packages/ui`:
  `npx vitest run src/services/__tests__/tldw-api-client.chat-mutations.test.ts src/services/__tests__/tldw-api-client.chat-sanitization-regression.test.ts --maxWorkers=1`
- `/private/tmp/uat034-sanitization-existing-tests.log`: 9 pass, 1 fail (the obsolete assertion only).
- `/private/tmp/uat034-sanitization-probe.config.ts`: temporary transform appends one test to the existing proxy harness; repository files unchanged.
- Probe command from `apps/packages/ui`:
  `npx vitest run --config /private/tmp/uat034-sanitization-probe.config.ts --maxWorkers=1 -t 'UAT034 actual completion transport'`
- `/private/tmp/uat034-sanitization-probe.log`: 1 pass, 109 unrelated tests skipped by name filter.
- Probe executes real `TldwApiClient.createChatCompletion -> bgRequest -> tldwRequest -> fetch` with only browser/storage boundary stubs and a synthetic fetch response (no network). Manual device credential metadata is valid. The real request core returns non-2xx 500; the proxy rejects; the client does not return a synthetic successful Response. Exactly one fetch occurs. It also verifies the private-path sentinel survives in the rejection, warning, and `__tldwLastRequestError`.
- Two probe setup errors were resolved before the valid run: existing proxy-only storage mock lacked `safeStorageSerde`; initial synthetic single-user config lacked the required credential metadata and correctly stopped at 401 before fetch. Neither was counted as product evidence.

## Exact boundaries

- `services/__tests__/tldw-api-client.chat-mutations.test.ts:168-195`: resolved success payload expected to be scrubbed; no non-2xx transport modeled.
- `services/tldw/TldwApiClient.ts:3298`: resolved bgRequest data intentionally preserved; non-2xx rejection escapes before `createJsonResponseLike`.
- `services/tldw/domains/chat-rag.ts:268,388`: same documented success-preservation contract.
- `services/tldw/request-core.ts:728`: formats non-2xx detail/error/message; does not redact string content.
- `services/background-proxy.ts:1045,1098-1147`: generic error message retained; RAG has an explicit sanitizer opt-in, normal Chat does not. Generic `sanitizeResponseData` only sanitizes sensitive object fields, not the error message before logging.
- `services/tldw/TldwChat.ts:422-427`: normal non-stream send wraps failure with generic `Chat completion failed` but retains cause and logs the raw error. Thus I do not claim the ordinary Chat toast visibly renders the path.
- `components/Option/Prompt/hooks/usePromptInteractions.tsx:693-706`: Quick Test calls this client directly and renders `error.message` in notification description. This is a source-traced user-visible sink for the confirmed transport message; no native Quick Test failure was driven here.

## Minimal proposed scope (not implemented)

1. Replace the obsolete test with the intended successful-payload preservation plus genuine rejected-transport behavior. Retain the existing five success-preservation controls; never scrub successful assistant content based on words/paths.
2. If parent includes the independently confirmed error-only gap, scope it to failed Chat completion transport before console/diagnostic persistence, preserving HTTP status/code/cancellation and safe actionable validation. Reuse the existing failure-message sanitizer contract where appropriate; add real direct/extension non-2xx tests and a Quick Test notification control. Do not change generic successful response semantics or broadly rewrite auth/transport.

Current .34 client/domain/scope-error files remain owned by the title repair agent and were not edited.
