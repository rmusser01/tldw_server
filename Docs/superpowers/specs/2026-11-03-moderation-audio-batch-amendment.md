# Moderation Audio Batch Amendment

Status: approved by the requester on2026-11-03 through the explicit batching
approval question. Associated work: TASK-15466 (previously TASK-15417). This records the approved short
design, not a relaxation of execution limits.

## Evidence And Intent

Whole-branch review F1 reproduced a warmed benign100-segment,5099-character
transcript failing its shared two-second deadline after approximately50 execute
attempts. Launching one interpreter per segment makes ordinary long-form Audio
unusable despite trivial regex work. Preserve independent segment matching and
atomic fail-closed results while eliminating that startup amplification.

## Approaches

- Selected and approved: one bounded fresh child handles independent text fields
  against one policy snapshot, compiling that policy once. Each field retains
  its own anchors, boundaries, replacements, ordering and per-field evaluator
  limits; results remain associated by index.
- Rejected: concatenate text, because patterns could cross segment boundaries
  and anchors would change meaning.
- Rejected: reset or widen deadlines, run inline, or introduce a persistent pool;
  these conflict with the approved containment and ownership design.

## Contract

- At most4,096 independent transcript/segment text fields per child batch.
- Aggregate input AND aggregate successful output each remain at most1,000,000
  characters, including full transcript and segment duplicates actually checked.
- Keep existing256-rule,4096-character pattern/replacement,262144-byte policy,
  33554432-byte request and8388608-byte response frame ceilings. Frame caps include
  the four-byte prefix; encoded bytes remain independent of character admission.
- Keep one inherited two-second deadline across admission, serialization, child
  startup and all fields; four active workers per API PID, no queue or retry.
- One child evaluates fields independently, not by concatenation. It executes
  existing PolicyEvaluator redaction semantics, not another regex algorithm.
- Check count/aggregate sizes before encoding and prospective output growth
  before materialization. Empty fields cannot evade the field-count ceiling.
- Return exactly one result per admitted field, or fail the complete batch. No
  partial text/segment payload is exposed, persisted or forwarded on failure.
- Cancellation, kill/reap, quarantine and sanitized report-once handling remain
  mandatory. Metadata describes the aggregate operation, never content.
- Preserve existing service method signatures, trusted override dispatch, Audio
  opt-out/no-op guards, segment metadata/key handling and public response shapes.
  New default batching does not authorize an in-parent regex fallback.

## Builder Decisions

The implementation plan defines the private operation tag and data shape. This
is a focused Audio redaction batch, not a general task framework or process pool.
No admin dashboard, new external dependency, LLM coverage expansion or public
API endpoint is added. Existing compiler/admin-lint scope remains unchanged.

## ADR Assessment

ADR required: existing accepted ADR-096 governs stdlib re in killable batch
children, and ADR-021 governs lifecycle ownership. This extends a batch's bounded
data shape without changing engine, ownership, persistence authority or durable
architecture rules. No new ADR or accepted-rationale rewrite is required.

## Acceptance

A real warmed100-segment payload completes under the existing shared scope with
bounded child count, semantic parity covers anchors and zero-length/literal
replacements per field, aggregate/item/wire caps reject safely, and failures or
cancellation never return a partial payload. Actual platform evidence and timing
limitations remain explicit; representative failure cannot justify silent tuning.
