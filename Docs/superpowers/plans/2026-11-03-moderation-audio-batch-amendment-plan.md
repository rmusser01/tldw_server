# Moderation Audio Batch Amendment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. This is part of the existing SINGLE final fix wave; do not dispatch another implementer or reviewer.

**Goal:** Remove interpreter-per-segment overhead without changing regex semantics or approved execution budgets.
**Architecture:** Add one bounded independent-text redaction variant to the existing child protocol. Reuse PolicyEvaluator and existing runner/reporting ownership; assemble complete Audio results only after full validation.
**Tech Stack:** Python>=3.12, stdlib re/subprocess/JSON, existing pytest/Bun toolchain.
**Spec:** `Docs/superpowers/specs/2026-11-03-moderation-audio-batch-amendment.md` (requester-approved short design).

## Global Constraints

- At most4,096 independent transcript/segment text fields per child batch.
- Aggregate input AND aggregate successful output each remain at most1,000,000 characters.
- Keep existing256-rule,4096-character pattern/replacement,262144-byte policy,33554432-byte request and8388608-byte response frame ceilings, including four-byte prefix.
- Keep one inherited two-second deadline and four active workers per API PID; no queue, inline matching, timeout reset/widening, concatenation or pool.
- One result per field or complete failure; preserve independent regex/anchor/replacement semantics, override dispatch, metadata privacy and cancellation ownership.

## Review Focus

1. A modest100-segment transcript succeeds without one launch per segment.
2. Anchored/cross-boundary patterns cannot match across independent fields.
3. Many empty fields or expanding replacements cannot bypass aggregate caps.
4. Cancellation or a late field failure withholds the whole payload and reports safely.
5. Disabled/partial opt-out, text/Text keys and trusted overrides retain existing behavior.

## Stage 1: Bounded Child Variant
**Goal:** Add the smallest typed redaction batch to existing execution data/protocol/worker.
**Success Criteria:** Strict valid round trips, independent semantics and aggregate caps.
**Tests:** Protocol/worker unit and schema-aware property coverage.
**Status:** Not Started

**Files:** `Moderation/execution_models.py`, `execution_protocol.py`, `regex_worker.py`, owned protocol/worker tests.
**Interfaces:** Add `redact_many` to Operation, not the single-text BaseOperation. Add constant `MAX_TEXT_FIELDS=4096`. Extend WorkerRequest.text to `str | tuple[str, ...]`; only redact_many uses a tuple internally/list on the wire, keeping existing single-text variants strict. Add `request_input_chars(request: WorkerRequest) -> int` for accurate aggregate ownership/reporting metadata. Add `snapshot_redaction_batch_request(texts: Sequence[str], policy: ModerationPolicy, phase: str | None, limits: EvaluationLimits, bounds: ExecutionBounds) -> WorkerRequest`. Batch responses contain one string per input field in order, strictly validated against count and aggregate limits. Existing private version/field structure stays otherwise unchanged; incompatible operation/type combinations fail closed.

- [ ] Write/run RED tests for independent anchors, empty/Unicode fields, raw/structured rules, literal replacements, count4096/4097, aggregate input/output limits and malformed result count/types.
- [ ] Implement snapshot validation before copied encoding, compile/reconstruct shared policy once in the child, and run existing redaction separately per field. Enforce remaining aggregate output allowance before growth; zero remaining allowance permits empty results only. No matching in parent.
- [ ] Run protocol/worker and relevant existing evaluator tests GREEN; verify single-text and supervised/custom variants remain strict and unchanged.

## Stage 2: Service And Audio Integration
**Goal:** Batch checked Audio fields under one existing outer scope.
**Success Criteria:** Atomic Audio results, supported dispatch and bounded launches.
**Tests:** Service/payload/HTTP/WS and real many-segment success regressions.
**Status:** Not Started

**Files:** `Moderation/moderation_service.py`, runner/reporting metadata sites as required by the new variant, `Audio/stt_policy.py`, existing Audio HTTP aggregate call site, owned service/Audio tests.
**Interface:** `ModerationService.redact_texts(self, texts: Sequence[str], policy: ModerationPolicy, phase: str | None = None) -> list[str]`. The default path submits one shared-policy child batch and inherits the reporting/deadline scope. Existing public single-text signatures and tuple dispatch stay intact. A trusted overridden `redact_text` must still be honored rather than silently bypassed; preserve its dispatch without introducing a default inline evaluator fallback.

- [ ] Write/run RED for100 six-second segments with benign text/full transcript, one bounded default batch, independent segment anchors, retained metadata/Text key, partial opt-out, trusted override and enabled/no-op behavior.
- [ ] Collect only actually checked fields, create one PII policy snapshot, redact in a batch, and map results back only after the entire response succeeds. Route HTTP aggregate transcript/segments through the same approach; no per-field fresh deadline.
- [ ] Add late-field output failure and cancellation tests asserting no original/partial payload/save/downstream success; preserve one sanitized incident and real owner cleanup.
- [ ] Run new service/payload/HTTP/WS tests plus existing Audio retention, realtime, streaming and owned protocol/runner/reporting regressions GREEN.

## Stage 3: Evidence And Single Fix-Wave Handoff
**Goal:** Prove the representative workload is repaired under unchanged budgets.
**Success Criteria:** Actual isolated execution, updated packaging/platform/security evidence and one final scoped re-review.
**Tests:** Representative real batch and affected regression/security/metadata gates.
**Status:** Not Started

- [ ] After independent containment prerequisites, run the warmed actual100-segment5099-character workload, record elapsed time, actual batch count, aggregate counts, result correctness and settled ownership. Keep earlier failure evidence. Do not widen caps if it still fails.
- [ ] Update installed-worker/flags/import tests and actual Linux3.12 source snapshot/probes for the new variant; preserve Windows unverified status. Update README/validation evidence and TASK-15417 notes with amendment/ADR links.
- [ ] Compile, lint and Bandit touched production/tests (only B101 blanket test exclusion); inspect new findings. Run metadata/ID/shard/whitespace gates and relevant broader focused regressions scaled to the shared protocol change.
- [ ] Commit working code normally under TASK-15417 and append all F1-F3/M1-M2 evidence/interfaces/SHA to final-fix-report.md. Parent owns exactly ONE scoped independent re-review for the complete final fix wave.

## Self-Review And Handoff

Spec coverage: field count, aggregate characters/bytes, child-only semantics,
deadline/ownership, all-or-nothing Audio/HTTP/WS and compatibility each have named
test steps above. ADR assessment: existing accepted ADR-086/ADR-021 govern, no
new durable architecture or accepted-rationale edits. Requester explicitly chose
subagent-driven execution and approved this short amendment on2026-11-03. These
implementation details preserve that approved contract; same final fix agent
continues after this written plan, with no second implementation wave.
