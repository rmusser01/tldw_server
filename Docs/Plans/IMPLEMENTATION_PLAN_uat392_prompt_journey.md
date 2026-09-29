# UAT392: saved Prompt to canonical Chat journey

Backlog: TASK-13260.278.5.1. PR: #2979. Full/native UAT remains paused.

## Stage 1: Trace the gap and establish causal controls
**Goal**: Prove a missing saved system instruction cannot qualify the C-01 journey.
**Success Criteria**: Controlled provider accepts the exact saved instruction/question and rejects missing, replaced, user-role-only, and changed-question inputs.
**Tests**: Real provider HTTP in streaming and non-streaming modes; positive control fails before fixture addition.
**Status**: Complete

## Stage 2: Use the real UI and durable records
**Goal**: Replace optional API evidence with actual saved Prompt application and canonical conversation checks.
**Success Criteria**: Server Prompt ID/name/instructions persist; normal Use as System Instruction action; exact completion request; successful response; saved user/assistant IDs/content survive reload. No DOM deletion or application API fulfillment.
**Tests**: Existing prompts-chat browser journey, strict Playwright collection, types/lint, provider controls.
**Status**: In Progress

## Stage 3: Review and publish bounded evidence
**Goal**: Record verified engineering coverage and remaining UAT392 gates without inflating release acceptance.
**Success Criteria**: Review diff, Bandit touched Python, update Backlog/tracker/PR; preserve completed current CI before next push. SQLite/PostgreSQL/native acceptance remains separately stated.
**Tests**: Exact-head CI artifacts and results; do not claim browser execution from collection.
**Status**: In Progress

Causal controls: missing/changed/role-shifted/prefixed/suffixed/extra-turn instructions fail the bounded Pirate match; actual Chat templating and OpenAI adapter payloads are exercised in both stream modes.32Prompt provider cases pass within70combined provider checks. The shared Prompt creation helper no longer deletes UI portals; static Chromium before/after control proves an unrelated drawer is preserved. Source journey requires saved Prompt IDs, normal Use as System Instruction, exact submitted messages, canonical saved turn and reload. Types/lint and collection complete before publication; real-app browser execution still pending.
