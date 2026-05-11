# Persona Chat Judge Evaluation Contract

Date checked: 2026-05-11

## Summary

This artifact defines the optional calibrated Persona Chat judge layer for Stage 2 follow-up issue [#1566](https://github.com/rmusser01/tldw_server/issues/1566). It builds on the deterministic trace/error taxonomy and fixture records from `Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md` and `tldw_Server_API/tests/fixtures/persona_chat_quality_cases.json`.

The judge layer is offline only for this slice. It does not change Persona Chat runtime behavior, block responses, moderate live chat, or replace deterministic fixture tests.

## Input Contract

The normalized judge input is derived from one Persona Chat quality fixture case:

| Field | Source | Purpose |
| --- | --- | --- |
| `case_id` | Fixture `case_id` | Stable trace key for calibration reports. |
| `assistant_kind` | Fixture `assistant_kind` | Confirms the case is persona-backed ordinary chat. |
| `assistant_id` | Fixture `assistant_id` | Keeps persona identity visible in judge artifacts. |
| `persona_memory_mode` | Fixture `persona_memory_mode` | Required for memory expectation checks. |
| `user_input` | Fixture `input` | User turn under evaluation. |
| `expected_context` | Fixture `expected_context` | Redaction-safe effective context and prompt-preview diagnostics. |
| `response_observation` | Fixture `response_observation` | Redaction-safe assistant output and selected/rejected exemplar metadata. |
| `labels` | Fixture `labels` | Failure labels represented by the case. Label presence is the expected fail class for that judge dimension; absence is the expected pass class. |
| `expected_evidence` | Fixture `expected_evidence` | Trace references the judge should cite when making a decision. |

The helper defensively copies nested dictionaries so tests and callers can mutate their working copies without corrupting fixture state.

## Judge Dimensions

Each dimension is binary and maps to one narrow failure family:

| Dimension key | Failure labels | What it checks |
| --- | --- | --- |
| `boundary_refusal` | `PC-BOUND-001` | Prompt-reveal or instruction-override refusal. |
| `boundary_style` | `PC-BOUND-002` | Refusal that keeps explicit persona constraints. |
| `capability_truthfulness` | `PC-CAP-001` | No unsupported live tools, automation, memory, visual rendering, or data access claims. |
| `memory_expectation_alignment` | `PC-MEM-003` | Memory claims match `persona_memory_mode`. |
| `exemplar_synthesis` | `PC-EX-001` | Selected exemplar guidance is synthesized instead of copied. |

There is intentionally no holistic Persona Chat quality score. If a future quality signal needs another subjective criterion, it should be added as another binary dimension with its own labels and calibration evidence.

## Prompt Contract

`build_persona_chat_judge_prompt()` creates an offline prompt for one dimension at a time. The prompt requires:

- A specific dimension criterion.
- Explicit `PASS` and `FAIL` definitions.
- Case evidence from fixture fields only.
- JSON output with `critique`, `result`, and `evidence`.
- Critique before verdict.

The prompt does not request numeric ratings or aggregate scores. It also does not include fixture `labels`; labels are calibration ground truth and must not be shown to the judge.

## Calibration Contract

`calibrate_persona_chat_judge_predictions()` compares predicted judge results to expected fixture labels before outputs can be treated as useful quality signals.

Expected result is derived per dimension:

- If the fixture labels include that dimension's failure label, expected result is `Fail`.
- If the fixture labels do not include that dimension's failure label, expected result is `Pass`.

The report includes:

- True pass, true fail, false pass, and false fail counts.
- TPR: pass-labeled cases predicted as pass.
- TNR: fail-labeled cases predicted as fail.
- Missing predictions for dimensions represented by predictions or fixture labels in the batch.
- Unknown predictions for unregistered dimensions or unknown case ids.
- Warnings when class counts are too small for production calibration.

The default production threshold is 20 pass and 20 fail cases per dimension. The current synthetic fixture set is useful for contract and smoke calibration, but it is not enough to claim production-calibrated judge accuracy.

## Non-Goals

- No live LLM execution.
- No API endpoint or recipe-run wiring in this slice.
- No runtime Persona Chat gating.
- No moderation gate.
- No replacement for deterministic fixture checks.
- No Persona Live visual-pack, renderer, VN/CYOA, native companion, or wake-word work.

## Current Verification Surface

Focused tests live in `tldw_Server_API/tests/Evaluations/test_persona_chat_judge.py` and cover:

- Fixture-to-judge input normalization and defensive copies.
- Binary prompt shape and structured output requirements.
- One positive and one negative fixture-label calibration case for `exemplar_synthesis`.
- Missing and unknown prediction reporting.
