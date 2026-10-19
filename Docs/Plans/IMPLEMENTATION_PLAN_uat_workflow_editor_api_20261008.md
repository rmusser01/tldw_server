# C12: Connect the workflow editor to the existing Workflow API

Related task: TASK-13260.278.18.83.45. The parent recorded the live C12 cause before edits and approved this design; the frontend build freeze was released before this unit began.

## Design and ADR assessment

Save currently clears local dirty state and prints the graph while reporting success. Run creates a timestamp-based local ID without submitting inputs or executing steps; controls change local state only. Connect these actions to the existing owner-scoped definition, run, status, diagnostics, and control endpoints.

Governing decisions: [ADR-003](../ADR/003-jobs-vs-scheduler-default.md) and [existing Workflow design](../Design/Workflows.md). Reuse immutable Workflow definitions and the existing Scheduler. Editor layout and Start input schema remain presentation metadata; the stored DSL is authoritative. No new runner, auth rule, public API, or architecture decision is introduced, so no new ADR is required.

Compile an unambiguous connected Start → executable steps → End chain to existing registered server step types, preserving IDs, labels, templates, and config. Refuse unsupported wiring, cycles, disconnected executable nodes, routing overrides and nonempty End output mapping. Store a versioned canvas snapshot in existing metadata and verify it against the stored DSL when reopening. Plain linear server definitions can be represented as a canvas; unsupported definitions fail explicitly.

Save preflights in block mode, creates a private definition or immutable version, and acknowledges its returned ID/version. A late save cannot mark changed or replacement state clean. Open lists and reads saved definitions. Export is a pure local snapshot. Run requires the acknowledged current saved snapshot and validated object inputs. Inject the already installed WebUI Ajv helper through the thin page/shared route boundary; strict workflow validation refuses invalid or unsupported schemas and validation unavailability. No new dependency or remote schema loading.

Dispatch one retained idempotency key per explicit run attempt. Adopt only the server's run ID/status. Poll actual run and step state without overlapping requests, normalize `succeeded` for display, and render final output/errors. Controls and approvals call existing APIs and refresh; failures retain last verified state. Reuse diagnostics/attempt inspection. Owner leases, abort signals, and view generations fence save/open/dispatch/status/diagnostics across reset, replacement, unmount, or auth changes. Schedules remain a separate user flow.

## Stage 1: Contracts and causal regression
**Goal**: Demonstrate the actual fake Save/Run seam in mounted integration tests before production edits.
**Success Criteria**: Save fails to reach the API on unchanged source; missing required input produces a synthetic run on unchanged source. Existing assertions and suites remain intact.
**Tests**: Mounted editor/store/services with only transport, canonical owner-resolution, and canvas-layout seams; required-input guard and successful save boundary.
**Status**: Complete

## Stage 2: Existing API integration
**Goal**: Implement Save/reopen and real input/dispatch/status/control behavior with the existing abstractions.
**Success Criteria**: Acknowledged saved versions round-trip; invalid input and unsupported graphs refuse; actual server IDs/outcomes drive the UI; owner/view races preserve replacement state; controls never fabricate success.
**Tests**: Definition compiler/round-trip and unsupported graph cases; strict Ajv input cases; mounted save/run/control/error/late-response cases; service protocol and request-scope controls.
**Status**: Complete

## Stage 3: Qualification and freeze
**Goal**: Qualify only affected scope and provide an independently reviewable production freeze.
**Success Criteria**: Normal UI/Web affected suites and types pass, touched lint has no new findings, source review completes, exact hashes and limits are reported. The parent loads the combined build and verifies actual Save/reopen/required-input refusal/real execution and server outcomes before closing live C12; browser UAT remains the parent's responsibility.
**Tests**: Sequential normal configured UI/Web Vitest runs with 600-second outer/300-second case caps; affected existing editor/store/registry/inspector/service tests; TypeScript and lint. Bandit is N/A for TypeScript-only changes. Maximum three bounded attempts per issue.
**Status**: In Progress

## Verification record

Unchanged-source causal baseline: normal UI Vitest, 600-second outer/300-second case cap, natural exit 1 in 3.05 seconds. Two expected failures: Save made zero API calls; required-input Run produced a synthetic timestamp ID. No live acceptance claimed.

Additional causal controls: held Open overwrote a newer draft (one expected failure); immutable active-run binding incorrectly adopted the latest saved version on remount (one expected failure); stale waiting approval and foreign investigation projected into a failed run (two expected failures). Each focused corrected run passed naturally. Six independently identified unsupported saved-settings cases (plain/editor completion webhook, adapter metadata, default inputs, tags, future editor metadata) failed before explicit refusal and passed after it. Unknown acknowledged-or-lost dispatch across remount failed before owner/view-scoped store retention and passed after it. These are integration boundary controls with transport/owner seams; they do not establish actual worker or browser acceptance.

Strict installed Ajv helper: normal Web configuration, four cases passed naturally. The existing non-strict default remains compatible; workflow opt-in rejects unknown/remote/async constraints without coercing inputs.

First complete affected UI run: 108 passed and two new mounted control fixtures failed because they seeded only the editable definition version, omitting the newly separated immutable run binding. Both fixtures now seed actual run identity/version fields, preserving every original assertion. Qualification continues on the final source; no test or auth behavior was disabled.

Production candidate frozen 2026-10-08T22:11:06Z for independent review (ten TypeScript paths; no backend production edits). Pending submissions retain their original idempotency key and inputs across editor remount under the same owner/view. Only acknowledged run adoption or an explicit reset clears the attempt; the UI warns that forgetting an unknown outcome may repeat accepted work. Server definitions with settings the canvas cannot preserve fail Open explicitly. Schedules remain separate.

Final configured affected UI: 14 files / 111 cases, natural exit 0 / 14.62 seconds. Final matching Web: 15 files / 115 cases (same 111 plus four strict Ajv cases), natural exit 0 / 14.55 seconds. The scope contains 10 mounted editor, 26 compiler, 17 service, eight hook integration cases, and 50 existing editor/registry/inspector/service/route cases. All original assertions remain. Integration tests replace only transport, canonical owner resolution, and canvas-layout boundaries; worker execution and browser outcomes remain unqualified until parent UAT.

Touched ESLint: 15 production/test paths, natural exit 0, zero errors / seven pre-existing unused-symbol warnings verified against HEAD (five editor, two store); zero new findings. Inherited Timeline prop deprecations and test-environment CSS/config warnings are reported without claiming browser acceptance. Bandit is not applicable: the unit changes only TypeScript and uses existing authenticated service/owner lease abstractions; no Python auth or backend code changed. Independent final actual-source review read all changed functions and all new test bodies and returned CLEAR with no remaining Critical/Important/Minor finding. Both reported semantic-loss and timeout/remount P2 findings have causal controls and are resolved.

Final configured UI TypeScript check started 2026-10-08T22:13:22Z and exited naturally 0 with 8 GB heap, 600-second outer cap, `--noEmit --incremental false`. An early overlapping Web type process was cancelled (exit 130) and excluded; the qualifying Web check starts only after UI completion. Before/after hashes include the frozen A10 taxonomy/GeneratePanel/test and root Retry detail source/test in addition to this unit.

Qualifying sequential Web TypeScript check started 2026-10-08T22:15:01Z after UI natural completion, and exited naturally 0 by the recorded 22:16:19Z boundary with the same configured heap/cap/options. Exact finish instants were not captured; use these observed boundaries rather than inferred durations. After-check hashes at 22:16:19Z match all ten C12 production and five new test paths frozen at 22:11:06Z, plus the five requested frozen A10/Retry files. The shared source was not globally frozen: concurrent Writing and document-flow edits may postdate these type checks' reads, so their final combined qualification remains with the parent.

Additional frozen type scope: A10 `error-taxonomy.ts` SHA256 `0475af576a29d6e1ee7b68c007325756eda5bf3d9b8b6886adf5de775fe4583f`, `GeneratePanel.tsx` `8596004bc21a86b78487ca9cce997202aadf1b53609053d6f930165ded71e0cd`, its new grounding test `9a579db1ab30d400c49c8461d78bbdb8ca20266bdd036b2dd50006f3fb4f6ccb`; Retry `HistorySelectionReview.tsx` `28e22f70d2be28c2da08630b06d680c0a429d42e2ff20597fa0ac5f190b2a7b8`, its test `0b35c8a146c0bd8b477994650e3d553eff0f20699b9507f99a3a9d2d98f6e522`. All five match before and after. Scoped whitespace check passes. Source qualification and independent freeze are complete; Stage 3 stays in progress for the parent's combined build/live acceptance. No runtime, browser, worker, model, index, commit, or publication action was performed by this unit.

## Production freeze hashes

| Path | SHA256 |
| --- | --- |
| `apps/packages/ui/src/types/workflow-editor.ts` | `0c09c5025951ea2eba5f0473a848036d80e74fe663b391b464d5086f61ed001f` |
| `apps/packages/ui/src/services/tldw/workflows.ts` | `62e7cafff1b518ee64667f78d6faaf8ece1aab9ce53ed02b5a460b8bb26b1f1e` |
| `apps/packages/ui/src/services/workflow-editor-definition.ts` | `5b4173f7adc1438b5084fb4a172f9bc67a40c563243d79e5c4899ae957213f49` |
| `apps/packages/ui/src/hooks/useWorkflowEditorApi.ts` | `7634e5f24f83874a10a97b08dc1716a632bc2f19343f99a14e16cd071fc09efa` |
| `apps/packages/ui/src/store/workflow-editor.ts` | `218416684f891748daf9ac6bdd80c7a750914e97364976a00752f5e4651465b0` |
| `apps/packages/ui/src/components/WorkflowEditor/WorkflowEditor.tsx` | `701a25db6ad6c41ae2cdb976941f2f30f95dad1cea3337d676ad5d7660f3c151` |
| `apps/packages/ui/src/components/WorkflowEditor/ExecutionPanel.tsx` | `337c7fe2040a54eac2b07b1bc457d022a8d5670b63bc1f0d628bf9057576658f` |
| `apps/packages/ui/src/routes/option-workflow-editor.tsx` | `fae714e4dab0b2b065b1ab0cf24accf71ed1b78db3c53bcf256a7c960c469807` |
| `apps/tldw-frontend/lib/ajv.ts` | `5d1140d62cb614462aa5446095c525cc9121d049b6182be9cf16720a07557a34` |
| `apps/tldw-frontend/pages/workflow-editor.tsx` | `6155e3e52382042226a8526b3137cb47a75c8fe93738cda99c59272d7661bce2` |

- Root corrected final combined configured semantic types: UI natural0, no diagnostics,66.245 seconds outer (finished2026-10-08T22:47:13.773923Z); sequential WebUI natural0, no diagnostics,42.918 seconds outer (finished22:47:56.986458Z). Both use8GB/600second cap/noEmit/nonincremental. All7293 inventoried apps files match before/after; freeze digest13bf4be7bcfd0bfd39be60868cbadcbe78756b28661c26334cdafb97a99fc18c. This qualifies current Writing/C02/C12/A10/Retry source together after five ignored Testing Library option corrections. Root final source reviews CLEAR; combined production build/live UAT remains open.
