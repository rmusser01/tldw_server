# UAT native Character prompt delivery diagnostic — 2026-10-08

Associated parent task: `TASK-13260.278.18.83.45`. The parent recorded the observed TestBot response and authorized this focused diagnostic before these edits. The parent owns the task record and live UAT.

## Goal and scope

Determine whether the exact saved instruction `You are E2E-TestBot. Always respond with exactly: BEEP BOOP.` reaches the first system message of the Ollama streaming request for a first native Character send. Join the real Character store and conversation factory, owner-bound history snapshot/selection/admission, saved behavior projection, prompt templating, call preparation, provider registry, and Ollama adapter. Use the existing `chacha_db` SQLite fixture and transport injection points; no model or network request.

Only one new integration module and this plan are owned by this diagnostic. Production remains unchanged. Test configuration/catalog inputs are explicit, isolated fixtures; prompt, persistence, selection, adapter, and payload logic remain real. A passing diagnostic establishes source-level prompt delivery for this case, not live model compliance or UAT acceptance. A source failure requires reporting the causal seam to the parent before a production change.

## ADR check

**ADR required: no.** This test characterizes existing behavior without changing a durable rule. Governing decisions: [ADR-025](../ADR/025-llm-provider-adapter-routing-and-overrides.md), [ADR-030](../ADR/030-configured-local-llm-egress-policy.md), and [ADR-049](../ADR/049-chat-history-selection-ownership.md). The parent receives this assessment for the associated task record.

## Stage 1: Assemble one connected diagnostic

**Goal**: Add one integration test with the exact saved TestBot instruction and greeting, then capture the registered Ollama adapter's outbound streaming payload.
**Success Criteria**: Real database creation, behavior snapshot, owner-bound selection, admission, projection, templating, call preparation, and adapter transport all participate; no replacement of prompt or history logic.
**Tests**: `tldw_Server_API/tests/Chat_NEW/integration/test_native_character_prompt_delivery.py`.
**Status**: Complete

## Stage 2: Run the unchanged-production baseline

**Goal**: Run the single diagnostic naturally in the existing qualified Python 3.12 environment with normal pytest plugins and private supported fixture paths.
**Success Criteria**: Report exact outcome and natural exit within the 600-second command / 300-second case bounds, with at most three bounded attempts per issue. Preserve a passing diagnostic; report any source failure before proposing a minimal connected fix.
**Tests**: `python -m pytest tldw_Server_API/tests/Chat_NEW/integration/test_native_character_prompt_delivery.py -q -p no:cacheprovider --timeout=300`.
**Status**: Complete

## Stage 3: Verify test scope and freeze

**Goal**: Read the complete test, run touched-file static checks and Bandit, and freeze the diagnostic for independent parent review.
**Success Criteria**: No new static/security finding; exact test/plan paths and verification limits reported; production and all original tests/assertions preserved. No unrelated suite replay, runtime action, build, index, or parent task edit.
**Tests**: Touched-module Ruff, static AST-only Bandit in the existing project environment, and scoped diff review.
**Status**: Complete

## Verification record

The first unchanged-production run completed naturally: 1 failed, 36 warnings, pytest 1.61 seconds, outer exit 1 in 3.584 seconds. Its only failure was the diagnostic's greeting/user expectation: the real adapter retains text-part arrays while the initial expectation used strings. The first system message already matched the exact TestBot instruction. This is a test expectation correction, not evidence of a production failure or a red-to-green production repair. The assertion is retained with the actual OpenAI content shape for the next single-case baseline run.

The corrected baseline passed naturally: 1 passed, 36 warnings, pytest 1.37 seconds, outer exit 0 in 2.907 seconds. After adding fixture/transport return type annotations and using the standard `# nosec B101` form for intentional test assertions, the final frozen module passed naturally again: 1 passed, 36 warnings, pytest 1.38 seconds, outer exit 0 in 2.905 seconds. Both runs used activated `.venv-uat-fixall-py312-20261007`, Python 3.12.13, normal configured pytest plugins, `-p no:cacheprovider`, a 300-second case timeout, and a 600-second outer subprocess limit. Database, user-content, and Jobs paths were task-private supported paths with synthetic command-scoped auth/encryption values; official fixtures owned storage and teardown. No command timed out or required forced cleanup.

The captured request used model `qwen2.5:0.5b`, streaming mode, one configured endpoint dispatch, and exactly these messages:

1. System: `You are E2E-TestBot.\n\nYou are E2E-TestBot. Always respond with exactly: BEEP BOOP.`
2. Assistant greeting: one text part, `BEEP BOOP.`
3. User: one text part, `Hello`.

Coverage is the first-send service integration for a saved Character named `E2E-TestBot`, its frozen default prompt preset, seeded greeting, current owner-bound native snapshot/selection/admission, raw prompt templating, real call preparation, genuine runtime-issued provider endpoint capability, real provider registry, and actual Ollama streaming payload construction. Configuration/catalog and credential-resolution inputs are scoped synthetic fixtures; the HTTP streamer returns only a terminal SSE marker. This does not test live model inference/compliance, HTTP auth/CSRF handling, frontend selection state, explicit system-prompt overrides, or every saved Character field/preset. No claim about the actual live conversation's provider payload or answer quality follows solely from this test.

Final Ruff: exit 0, no findings. AST parse: passed. Bandit 1.9.3 in the primary project environment, static AST only: exit 0, 0 findings / 0 errors over 152 lines, with 11 intentional pytest assertions marked `# nosec B101`; no other check suppression. Scoped diff check passed. No production file or original test/assertion was edited; production Bandit baseline/current comparison is not applicable to this test-only addition. Full-body self-review found no actionable diagnostic defect. No unrelated cookie/auth suite was replayed and no runtime, model, build, index, task-record, or publication action occurred.

Frozen scope is this plan and the single integration module. The parent owns independent review and any next live diagnostic or acceptance decision. Existing source evidence and this passing baseline do not substantiate a production prompt-drop fix.
