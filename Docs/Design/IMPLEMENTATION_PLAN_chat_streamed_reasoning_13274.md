# Streamed reasoning in chat — TASK-13274

Implements the approved all-ten remediation. Reasoning-only output must show the existing Thinking disclosure while generation continues, collapsed by default. Answer text keeps the lightweight streaming renderer. Full WebUI, extension Options, compact chat and character paths reuse the same accumulator and presentation.

ADR assessment: no new ADR. This repairs existing optional reasoning transport and presentation without changing HTTP APIs, persistence schemas, history admission or ownership. ADR-049 governs selected-history ownership; ADR-059 governs task editing.

## Stage 1: Qualify each lost-progress boundary
**Goal**: Reproduce reasoning-only loss and disclosure expansion in the real consumers.
**Success Criteria**: Physical SSE/worker frames reach ChatTldw; actual Collapse/rendered message tests distinguish reasoning from answer. Role/usage frames do not close reasoning; combined frames close before answer; later reasoning preserves earlier answer. Native transient progress is tested with real admission and retirement.
**Tests**: Existing dispatch integration plus held reasoning frames; shared accumulator cases; actual AntD disclosure and full/compact streaming messages. Preserve string-only direct service callers, plain streaming/Mermaid controls, error/sentinel/Stop behavior.
**Status**: Complete

## Stage 2: Repair shared transport and rendering
**Goal**: Connect the existing reasoning fields to the existing disclosure.
**Success Criteria**: ChatTldw explicitly opts into reasoning frames without changing default TldwChat string callers. Existing onChunk metadata/tool/sentinel behavior stays intact. Reasoning is collapsed by default, can be opened during generation, and does not expose raw think tags or trigger live Mermaid rendering.
**Tests**: Stage 1 regressions plus existing transport/model/rendering suites and both client typechecks.
**Status**: In Progress

## Stage 3: Reach character paths without adding authority
**Goal**: Publish reasoning-only transient updates in legacy and native character chat.
**Success Criteria**: Full and compact update on accumulated-text changes. Native character display reuses the accumulator only after actual admission; Stop, detached response and account retirement retain existing publication/settlement rules. Transient reasoning cannot certify saved history.
**Tests**: Actual native admission fixture with held reasoning progress, combined answer, Stop and source retirement; existing native scene/history and legacy character suites.
**Status**: In Progress

## Stage 4: Verify, review and record evidence
**Goal**: Close this finding while continuing the complete queue.
**Success Criteria**: Affected suites and both compilers pass, independent original-source review is clear, authored static checks/tracking/hooks pass, and a current-bundle reasoning walkthrough distinguishes real display evidence from participant research.
**Tests**: Connected regressions, owned formatting/static/diff/task checks and current-bundle browser checks. No Python is authored in this task; Bandit is not applicable to its TS/Markdown scope.
**Status**: Not Started

2026-10-10 shared transport/presentation evidence: actual physical model15, actual disclosure19, shared accumulator7, existing per-call abort13, startup/idle policy14 and extension startup1 all pass (six suites69 checks). Original model/disclosure RED5 is causal; plain/string-only positives preserved. Independent shared-source reasoning review is CLEAR. Native/legacy publication and real cookie operation retirement remain separate qualified work; both client types/current-bundle checks await writer freeze. Evidence `/private/tmp/chat-reasoning-qualified-green-setup-refresh-red.log`; its setup8 failures are TASK-13272 and do not count as reasoning acceptance.

2026-10-10 native/character evidence: five suites73 pass. Strengthened held full legacy reasoning RED retains live account/source/controller and user ACK yet receives empty text; its React updater reads cleared mutable buffers, so publication snapshots text/time before scheduling. Compact provider progress is green after correcting its model fixture shape. Reusable deferred updater, complete consumer suites, both types and current-bundle proof remain pending. Cookie operation account retirement is tracked13549, not reasoning authority.
