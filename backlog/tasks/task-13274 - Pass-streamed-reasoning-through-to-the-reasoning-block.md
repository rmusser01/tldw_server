---
id: TASK-13274
title: Pass streamed reasoning through to the reasoning block
status: In Progress
assignee: []
created_date: 2026-09-19 14:47
updated_date: 2026-09-19 14:47
labels:
- chat
- webui
- streaming
- ux-audit
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reasoning models appear frozen. Verified against live llama.cpp: the model emitted 57 reasoning chunks in a 60-token stream, while the UI showed a generating placeholder for 15.5 seconds with zero disclosure elements before any text appeared. The streaming parser handles the reasoning field and a reasoning block component exists to render it; the transport layer yields only the plain token and discards reasoning, so the two never meet. A thinking-budget control already exists in model settings, so reasoning is a first-class concept in the product.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Streamed reasoning content reaches the reasoning block on the default chat path.
- [ ] #2 Reasoning is visible to the user while the response is still generating.
- [ ] #3 Reasoning is collapsed by default and does not displace the answer.
- [ ] #4 A non-reasoning model shows no empty reasoning affordance.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Continuing the approved all-ten chat remediation. Scope is shared default chat reasoning progress, real streaming disclosure with collapsed default, correct reasoning/answer interleaving, and equivalent character publication in full/compact/native paths. Preserve string-only default for three direct TldwChat sibling callers; ChatTldw opts into typed reasoning frames. Reuse consumeStreamingChunk and ReasoningBlock without adding stores/history authority or persisting unacknowledged native reasoning. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_streamed_reasoning_13274.md. ADR assessment: no new ADR; this repairs the existing optional reasoning presentation/transport contract and retains existing history admission and settlement rules. Physical direct/worker SSE, actual disclosure, parser edge cases and native ownership controls required before completion.
2026-10-10 qualified RED: /private/tmp/chat-reasoning-and-legacy-dispatch-qualified-red.log has five causal failures: physical direct/extension ChatTldw drop held reasoning before answer (2), actual shared Message bypasses Thinking (2), actual AntD Collapse forcibly expands after user closes while streaming (1). String-only sibling and exact plain whitespace controls pass. Previous unqualified accessibility-name and literal newline fixtures corrected before this RED. Shared accumulator7 pass after reasoning remains open across role/usage frames, mixed frame closes before answer, and later reasoning starts its own segment. Root shared transport/presentation repair written; legacy/native publication regressions and broader review/type/current-bundle evidence still open.
2026-10-10 shared reasoning repair qualified green: six unique suites69 checks pass (actual physical model15, actual Collapse/shared message19, accumulator7, per-call abort13, startup/idle14 and extension startup1). Independent connected source review CLEAR, preserving default string-only siblings, exact mixed-frame answer, metadata/tool callbacks and single interruption sentinel. Native fixture real admission/held reasoning reached its gate before four causal failures (lost transient reasoning and unsaved recovery text); New omitted the real caller source fence and is unqualified until corrected. Reusable legacy held-reasoning failure is causal; full legacy did not reach declared provider gate and needs fixture qualification. Cookie New/principal fixtures likewise fail before the declared progress gate and are excluded. /private/tmp/chat-character-reasoning-cookie-retirement-qualified-red.log is not nine causal source bugs; final types/native walkthrough remain open.
2026-10-10 character reasoning qualification: native actual admission/Stop/account/detached progress, reusable character, physical model, real disclosure and parser five suites73 pass. Full legacy strengthened fixture keeps original owner/controller alive, provider held, user ACK and unique IDs yet publishes an empty bot row: qualified RED at deferred updater. Root found mutable text/time captured by React callback after buffers reset; full publisher now snapshots queued payload, reusable has same source defect pending an independently delayed updater control. Compact fixture initially lacked real model getState and is excluded from causal RED; corrected shape produces qualified compact Thinking GREEN. Real cookie New positive passes but principal retirement causal RED belongs13549, repaired in existing snapshot lease. Shared transport/presentation and native publisher source reviews CLEAR; remaining publication/compilers/native evidence open. /private/tmp/chat-character-reasoning-admission-qualified-green.log and /private/tmp/chat-cookie-lease-and-character-publication-qualified-red.log.
2026-10-10 final connected publication checkpoint: real full and compact paths snapshot progress before deferred React publication; shared ordinary and native character reasoning keep original command/account proof. Latest nine connected suites828/828 and both client compilers pass; prior physical reasoning/disclosure69 and native/character74 remain separately qualified. Two independent connected source reviewers CLEAR with own earlier authorship excluded. Static/format/hooks and current-bundle reasoning display remain pending. In Progress; no participant, physical AT or packaged extension inference. No new ADR;049/059 govern.
Final original-command consumer checkpoint: 20 connected suites / 1436 tests pass after fixture typing and production cleanup; both WebUI and extension compilers pass; all 112 authored TypeScript scopes introduce zero ESLint findings; owned formatter 21 is runtime-AST equivalent; git diff --check passes. Final independent source-only reviewers A and B CLEAR for the production cleanup, excluding their own earlier SDK/fixture authorship. Evidence: /private/tmp/chat-final-consumer-static-repair-connected.log, /private/tmp/chat-final-consumer-{web,extension}-types-qualified.log, /private/tmp/chat-final-consumer-static-qualified.jsonl, /private/tmp/chat-final-consumer-format.json. Latest fetched origin/dev 0c53973f17eede4074f4aa7894d0b42322340614 remains included by 2e47e3aa01c6bfc7cb61dea5f1cf47c15b7c35f0. Normal incremental commit/hooks are next. The current native bundle is stale; packaged/physical AT/uncoached participant and all-ten acceptance remain open. Three scene components contain four remaining unqualified operations (two selection preflights and Drawer hydration/Apply), to migrate next with original qualified receipts and CAS. No task/stage/namespace closure; ADR049/050/059 and approved D7 govern, no new durable rule. Bandit N/A for authored TS/Markdown-only scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
