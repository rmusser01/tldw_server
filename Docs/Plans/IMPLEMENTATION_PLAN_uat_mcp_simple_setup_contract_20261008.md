# C11 MCP simple setup contract

Associated task: TASK-13260.278.18.83.45. Parent recorded BEFORE-edit unit and source design ruling under standing human all43 fix/validation authorization. Root owns parent task/runtime/profile/auth/transports/live/Git. Prior Backup/Admin/C02/C03 units remain frozen.

## Verified source and approved design

ExternalServersTab getStdioServerDraft729–737 emits flattened command/args/env/cwd. Actual service sends this config unchanged; McpHubExternalRegistryService54–80 spreads config into canonical ExternalMCPServerConfig. Canonical federation schema supports only stdio/websocket and requires nested stdio/websocket; actual stdio adapter reads server_config.stdio. Therefore flat config cannot load as executable stdio. HTTP draft sends transport=sse/flat URL+headers, which no canonical runtime enum or adapter supports. Managed create stores metadata; validate endpoint returns readiness rather than repairing config. No working HTTP/SSE transport is removed or reinterpreted.

Approved frontend-only correction: wrap the identical stdio fields in config.stdio; HTTP/SSE choice disabled with precise supported-transport explanation. Existing Advanced/manual transport options already stdio/websocket only and valid nested manual/import config stays unchanged. Replace unsupported SSE import placeholder with a valid nested supported example; retain raw import/JSON error compatibility. No backend, transport/parser, auth/network/dependency/permission/default/owner/Enabled/Save-vs-discover change. No tool/server launch, credential/model/profile/runtime action. Existing stdlib smoke_stdio_server.py is read-only context, not launched here.

## ADR assessment

ADR required: no. Docs/ADR/README.md and canonical runtime/model/adapters/managed consumers reviewed. This aligns the existing client with established transport wire/schema requirements and truthful supported options, with no new durable architecture/security decision. ADR-052 MCP admin predicate and ADR-017 owner/RBAC semantics remain unchanged. Root owns task association per ADR-002/059.

## Stage 1: Trace and causal mounted request tests
**Goal**: Retain exact pre-unit source/test bytes in memory; demonstrate flat stdio/unsupported guided HTTP request defects on unchanged production.
**Success Criteria**: Real mounted ExternalServersTab and actual mcp-hub service at external bgRequestClient seam; exact canonical outgoing request fields, no false live/schema-validation/auth claim. Source-supported unsupported attribution and request refusal tests precede production.
**Tests**: New request/validation controls for stdio nested full/minimal fields; owner/enabled; Save versus discovery; invalid fields/env; unsupported HTTP choice; manual nested stdio/WebSocket/import compatibility; persistent failures/credential masking retained by existing affected tests.
**Status**: Complete

## Stage 2: Minimal source and affected qualification
**Goal**: Correct only approved simple setup mapping/refusal and valid example.
**Success Criteria**: Causal red before code; normal configured UI/Web genuinely affected new/existing component tests and mcp-hub service consumers; preserve original assertions except genuinely wrong flat guided expectation and unsupported executable HTTP expectation. Scoped baseline/current ESLint, TSX syntax; maximum three declared attempts with fixture/setup failures distinct from causal red and STOP/reassessment after cap. Node20.19.5/Vitest4.0.18/8192MB/outer600/case300 normal plugins.
**Tests**: New mounted module, existing ExternalServersTab suite and genuinely affected service create/discovery controls as needed. No old passing unrelated replay. Bandit N/A: TSX/TS/Markdown only.
**Status**: Complete

## Stage 3: Freeze and independent/root acceptance
**Goal**: Full changed-body/new-test self-review, independent review, exact source/test/plan hashes and limits.
**Success Criteria**: No new static findings; owner/field/error/mask/discovery gates intact; root owns combined types/build/actual controlled stdio readiness acceptance. Source/test freeze before live.
**Tests**: Root type/build/live; no actual transport/process/tool/API/browser action here.
**Status**: In Progress

## Evidence and attempts

- Parent source ruling approves nested Local stdio and explicit HTTP/SSE unavailability, preserving supported Advanced/manual WebSocket. Two pre-unit component source/test byte inventories retained only in memory. No production/test edits yet; no diagnostic/proof artifacts.

- Attempt1 unchanged production: natural1/6.362s,4 causal failures/10 passing controls/zero fixture failures/skips. Exact outgoing full/minimal stdio nested-body assertions failed; HTTP/SSE choice was enabled; import example transport was unsupported sse. Missing identity/command, malformed env, manual supported choices/nested stdio+WebSocket, JSON-object guard, create refusal and discovery refusal controls passed. Causal red is outgoing client contract/refusal only, not actual model/schema/transport execution.
- Minimal production patch: identical stdio fields nested under stdio; HTTP/SSE choice disabled with actual runtime support text; supported nested WebSocket import example. All Advanced/manual/import parser/owner/Enabled/Save/discovery/error/credential code unchanged. Existing tests change only guided stdio expectation to canonical nesting and wrong executable HTTP/SSE case to truthful no-create/no-discovery refusal. Next declared current-source normal affected UI/Web phase; maximum three issue attempts retained.

- Attempt2 current UI affected scope natural1/21.174s: all14 new cases passed,19/20 existing passed, zero skips; retain33 passes without replay. Sole failed untouched existing `opens the managed server editor from a drill target` uses immediate getByDisplayValue("stdio") after findByDisplayValue("Docs Managed"). Native Server select already exposes selected option Docs Managed before the separate drill edit effect opens the editor. Same-module focus/legacy drill cases await actual focused markers. Proposed narrow fixture correction awaits the SAME stdio display assertion; preserves original onDrillHandled/body/oracle, no source/modal/style/event/timeout/mock change. No Web command or fixture correction yet. Leftover static Delete Credential Slot portal in failure output is not treated as a proven cause or cleanup justification.
- Current scoped static: all3 TSX parse checks0, source lint0 errors/2 inherited exhaustive-deps warnings with exact baseline fingerprints; both test modules0 errors/0 warnings. Source48b457abac98c6329ff11a28adf9faddc20178401a1c912736a8c9f2b3577903, original affected testf154a1dde582d5b3c3ebfd7a077191dd1eac6225b06f2b11edb5307f1e284b5d, new14testaea2a16274bd1651c8bb83267dd0db1e3d20456d90436bba798a95c6f8051c3b. Bandit N/A. Full before/current bytes/static results retained in memory only. Production frozen; no stdio server/transport/API/runtime action.

- Root authorized remaining third declared qualification phase BEFORE fixture edit: only immediate getByDisplayValue("stdio")→await findByDisplayValue("stdio") in the sole failed existing drill edit case. This awaits the same actual editor oracle after the selected Server dropdown ambiguity; preserves original name/onDrillHandled assertions, no source/mock/style/event/timeout increase. Qualify only that UI case once retaining33 passes, then first affected Web34. No cap reset or unchanged UI replay; stop/reassess if the remaining phase fails.

- Third declared qualification phase: exact original drill assertion changed only to `await findByDisplayValue("stdio")`, preserving the value and onDrillHandled assertion. Sole failed UI case passed naturally in 2.573s, retaining the earlier 33 passes without replay (34 distinct UI cases qualified, no single 34-case green execution). First affected Web command passed all34 in 19.347s, natural0, no skips. This phase closes the proven asynchronous fixture issue; no timeout/mock/production change or cap reset.
- Final scoped existing-test ESLint0 errors/0 warnings and TSX syntax0. An initial static-result JSON parser encountered the configured ESLint Pages-directory prefix after the scan; output framing was corrected for the existing-test-only result, not a behavioral defect/red or test retry. Production retains0 errors/2 inherited exact-baseline exhaustive-deps warnings; new14 module0/0; all touched TSX syntax0. Bandit N/A for TSX/Markdown.
- Full changed-body/new14-body self-review and root independent actual-source review CLEAR: three production hunks, all14 new bodies, changed original expectation/refusal/await, connected setup/Save/reset/registry/canonical schema/adapter reviewed. No finding in this narrow outgoing request/refusal scope. Production48b457abac98c6329ff11a28adf9faddc20178401a1c912736a8c9f2b3577903; new14testaea2a16274bd1651c8bb83267dd0db1e3d20456d90436bba798a95c6f8051c3b; existing20test26742fa0b875db7ad6b4d963140faa4dd1a70aabe4ba03bc8be9052ddf20250a.
- Source/tests frozen. Root combined types/build and actual controlled stdio readiness remain open; no real process/tool/server launch, schema-execution, HTTP/AuthNZ/ownership lifecycle or full C11/all43 acceptance claim. Unsupported private HTTP branch remains unreachable through current setup setters; no SSE runtime support is inferred. Prior Admin/Backup/DSR/C02/C03 scope remains unchanged.
