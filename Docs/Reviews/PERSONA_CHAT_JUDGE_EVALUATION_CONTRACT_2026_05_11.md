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

## Review Policy Contract

`evaluate_persona_chat_judge_report_policy()` classifies offline harness reports as `advisory` or `blocked` before any report is treated as a calibration signal. The helper accepts the harness dataclass or the JSON object produced by the offline review command.

The policy blocks reports with malformed report fields, malformed case rows, invalid candidate envelopes, missing candidates, extra candidate ids, verdict agreement below the configured threshold, or flag agreement below the configured threshold. Reports that otherwise match the fixture can still be `advisory` rather than production-calibrated when aggregate pass/fail sample counts are below threshold or when per-dimension sample counts are unavailable. The current synthetic fixture remains advisory because it is a smoke/contract surface, not a statistically meaningful held-out calibration set.

The existing `calibrate_persona_chat_judge_predictions()` report is still the source of per-dimension calibration evidence. The V1 offline review-command report only exposes aggregate `verdict_counts`; therefore the policy helper must not mark a report `production_calibrated` unless a future report shape also supplies per-dimension pass/fail counts that meet threshold.

The policy result is intentionally trace-safe. It contains status fields, stable reason keys, and per-case `case_id` / `source_case_id` links only. It does not echo prompts, assistant text, exemplar bodies, memory content, RAG snippets, candidate payloads, rationales, filesystem paths, secrets, or database content.

## Known Failure Modes And Residual Risk

- Insufficient labeled data: the current fixture set is a smoke and contract surface, not a statistically meaningful production calibration set. Raw pass rates must not be used as production quality claims.
- Label leakage: fixture `labels` are calibration ground truth only. The prompt builder must continue excluding labels, and future callers must not pass calibration labels into judge-visible evidence.
- Corrupted calibration keys: missing or duplicate `case_id` values and duplicate `(case_id, dimension_key)` predictions can silently overwrite data. The helper rejects these before computing metrics.
- Invalid judge result parsing: only exact `Pass` and `Fail` values are accepted. Parser or model drift that emits variants such as `PASS`, `fail`, or explanatory prose must be treated as invalid output.
- Unknown prediction scope: unregistered dimensions and unknown case IDs are reported as unknown predictions rather than counted as calibration evidence.
- Model and prompt drift: any model, prompt, dimension, or fixture-schema change requires a fresh calibration run before comparing results across revisions.
- Report trust drift: review-command reports can be generated from malformed or partial candidate files. The policy blocks malformed reports, malformed case rows, missing candidates, extra candidates, invalid candidates, and low-agreement reports before maintainers treat them as calibrated.
- Trace leakage: judge review artifacts can become privacy risks if they copy raw prompt or response content. The policy result is restricted to bounded identifiers and reason keys.
- Subjective boundary cases: the V1 dimensions cover selected Persona Chat failure families only. They do not replace human review for nuanced style, safety, or intent disagreements outside the registered dimensions.
- Grounding and factuality limits: this judge evaluates observed Persona Chat behavior against fixture evidence. It does not prove factual correctness, retrieval grounding, or broader RAG quality.
- Runtime limits: this layer is optional and offline. It does not gate live chat responses, enforce moderation, or protect runtime Persona Chat behavior.

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
- Required identity-field validation, duplicate calibration-key rejection, and invalid judge-result rejection.

Policy tests live in `tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py` and cover advisory classification for the synthetic fixture, blocked invalid/missing/extra/low-agreement reports, dict report input compatibility, stable JSON serialization, and raw-text exclusion.
