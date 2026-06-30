# Persona Chat Judge Evaluation Contract

Date checked: 2026-05-11

## Summary

This document defines the V1 contract for the optional Persona Chat judge evaluation tracked in [#1566](https://github.com/rmusser01/tldw_server/issues/1566). It includes contract fixtures, an offline review harness, and a narrow explicit execution adapter boundary. It does not change normal Persona Chat runtime behavior, add production scoring, or gate chat responses.

The contract applies to ordinary persona-backed chat in the Buddy/Persona system. Persona Live rendering, avatar behavior, VN/CYOA flows, and native background interaction remain outside this scope.

## V1 Behavior

- Judge evaluation is offline-only.
- Judge output is advisory until a human-reviewed calibration set exists.
- Runtime gating is not allowed in V1.
- Deterministic checks remain the first gate for identity, memory mode, exemplar selection, preview parity, and trace shape.
- Fixture records must be synthetic or explicitly redacted and must not be mined from user-owned local databases.

## Input Envelope

Each judge case uses a bounded input envelope:

| Field | Required | Purpose |
| --- | --- | --- |
| `assistant_kind` | Yes | Must be `persona` for this V1 contract. |
| `assistant_id` | Yes | Stable persona identifier used by ordinary Persona Chat. |
| `persona_memory_mode` | Yes | Must be `read_only` or `read_write`. |
| `user_input` | Yes | Synthetic or redacted current user turn. |
| `effective_context.persona_context` | Yes | Bounded facts from prompt preview or runtime debug context. |
| `response_observation.assistant_text` | Yes | Synthetic or redacted assistant response being reviewed. |
| `response_observation.selected_exemplar_ids` | Yes | Selected exemplar ids or an empty list. |
| `response_observation.rejected_exemplar_reasons` | Yes | Bounded rejection reasons keyed by fixture-safe ids. |
| `deterministic_labels` | Yes | Existing `PC-*` labels identified by deterministic setup or checks. |

The input envelope must separate context facts from observed response text. A judge must not infer hidden state that is absent from `effective_context`.

## Output Envelope

Each expected judge output uses this shape:

| Field | Required | Purpose |
| --- | --- | --- |
| `verdict` | Yes | One of `pass`, `fail`, or `inconclusive`. |
| `scores.role_adherence` | Yes | 0.0 to 1.0 or `null` when not judged. |
| `scores.boundary_behavior` | Yes | 0.0 to 1.0 or `null` when not judged. |
| `scores.memory_semantics` | Yes | 0.0 to 1.0 or `null` when not judged. |
| `scores.exemplar_use` | Yes | 0.0 to 1.0 or `null` when not judged. |
| `scores.grounding_separation` | Yes | 0.0 to 1.0 or `null` when not judged. |
| `expected_flags` | Yes | Zero or more failure labels from the Persona Chat taxonomy. |
| `rationale` | Yes | Short bounded rationale for review display. |
| `evidence` | Yes | Bounded input keys that support the verdict. |

V1 fixtures require `fail` outputs to include at least one expected failure label. `pass` outputs must have no expected failure labels.

## Calibration Rules

A future executable judge must satisfy these rules before its output is considered usable:

1. Run against the golden contract fixture in `tldw_Server_API/app/core/Evaluations/data/persona_chat_judge_contract_cases.json`.
2. Run against a separate held-out set before any threshold tuning is accepted.
3. Report agreement measures for each judged axis, including false-positive and false-negative examples.
4. Preserve deterministic checks as hard preconditions before subjective scoring.
5. Document known bias cases, especially generic-assistant bias, over-rewarding theatrical style, and under-penalizing memory-mode claims.

Production use of judge execution remains deferred until these calibration requirements are implemented and reviewed.

## Executable Adapter Boundary

The first executable adapter boundary is `execute_persona_chat_judge()` in `tldw_Server_API/app/core/Evaluations/persona_chat_judge_execution.py`. It is still offline and explicit: callers provide normalized `PersonaChatJudgeInput` rows, selected dimension keys, provider/model metadata, and a completion callable. The adapter builds prompts with the existing prompt helper, calls only the injected callable, and returns sanitized `PersonaChatJudgePrediction` rows plus bounded failures.

The adapter is not a provider integration. It does not resolve credentials, call provider SDKs directly, persist reports, enqueue Jobs, expose API endpoints, update WebUI state, mutate chat output, or allow runtime gating. `runtime_gating_allowed` remains `false`.

Judge responses must be strict JSON objects with exactly `critique`, `result`, and `evidence`. `result` must be exact `Pass` or `Fail`; evidence values must be trace-field references that are actually present in the prompt envelope. Parsed predictions redact critique text to the marker `provided`, and provider/model metadata plus case/dimension identifiers are bounded before they are included in outputs or completion-call metadata. Failures use stable keys only, including `malformed_json`, `invalid_response_shape`, `missing_result`, `invalid_result`, `invalid_evidence`, `unknown_dimension`, `duplicate_prediction`, and `provider_call_failed`, and must not echo prompts, responses, exemplar text, filesystem paths, secrets, or database content.

Adapter predictions feed the existing per-dimension calibration helper. The harness, review command, and policy helper below remain the separate report-review path for already-produced V1 contract candidate outputs.

## Offline Harness

The offline report-review harness lives in `tldw_Server_API/app/core/Evaluations/persona_chat_judge_harness.py`. It compares already-produced candidate judge outputs against the checked-in V1 contract fixture and returns a bounded report with case counts, per-verdict counts, verdict agreement, flag agreement, score schema validity, missing candidates, invalid candidates, extra candidates, and per-case mismatch keys.

The harness does not call model providers, persist evaluation runs, enqueue Jobs, expose API endpoints, or gate Persona Chat responses. Future judge adapters should feed their outputs into this helper before any output is treated as calibrated.

## Offline Review Command

The unified evaluations CLI exposes an offline review command over the harness:

```bash
tldw-evals persona-chat-judge review --candidates candidate_outputs.json --output persona_chat_judge_report.json
```

`candidate_outputs.json` must be a JSON object keyed by `PC-JUDGE-###` case id. By default, the command loads the packaged V1 contract fixture from `tldw_Server_API/app/core/Evaluations/data/persona_chat_judge_contract_cases.json`; `--fixture` may point at an explicit local fixture during review. The command prints the bounded report JSON to stdout and, when `--output` is provided, writes the same JSON to that explicit file path.

This file output is the V1 offline report persistence location. It is intentionally user-selected and file-based only: the command does not call providers, write databases, enqueue Jobs, expose API endpoints, update WebUI state, gate Persona Chat responses, or mutate chat output.

## Calibration Policy

The review-only policy helper in `tldw_Server_API/app/core/Evaluations/persona_chat_judge_policy.py` classifies offline harness reports before maintainers treat them as useful calibration signals. It consumes either a `PersonaChatJudgeHarnessReport` or the bounded JSON shape emitted by the review command, then returns:

- `status`: `advisory` or `blocked`.
- `production_calibrated`: `false` unless the report is clean, aggregate pass/fail fixture counts meet the configured threshold, and per-dimension pass/fail counts are available and meet the same threshold.
- `runtime_gating_allowed`: always `false` in V1.
- `reason_keys`: stable machine-readable reasons such as `sample_too_small`, `dimension_sample_counts_unavailable`, `dimension_sample_too_small`, `invalid_candidates`, `missing_candidates`, `extra_candidates`, `verdict_agreement_below_threshold`, `flag_agreement_below_threshold`, and `invalid_report`.
- `case_issues`: bounded case summaries containing only `case_id`, `source_case_id`, and mismatch/reason keys.

The default policy keeps the current synthetic fixture advisory because it has fewer than 20 pass and 20 fail cases and because the V1 review-command report has no per-dimension sample counts. Invalid candidate envelopes, missing candidates, extra candidates, malformed reports, malformed case rows, or agreement below configured thresholds block trust in the report. Blocked status means the report cannot be used as a calibrated quality signal; it still does not gate live Persona Chat behavior.

Policy output must stay trace-safe. It must not include raw prompts, assistant responses, exemplar text, memory bodies, RAG snippets, filesystem paths, secrets, candidate payloads, database content, or rationale text. Linkage is limited to bounded fixture identifiers and reason keys.

## Privacy And Redaction

Contract fixtures must not contain:

- Secrets, tokens, credentials, or local filesystem paths.
- Raw content from user-owned local databases.
- Raw prompt content that was not written for this fixture.
- External source material that cannot be included in a test fixture.
- Unbounded chat transcripts, exemplar bodies, memory bodies, or RAG snippets.

Fixture ids and exemplar ids may be synthetic. Rationale and evidence fields should cite bounded keys such as `assistant_text`, `persona_memory_mode`, `selected_exemplar_ids`, and `user_input`.

## Taxonomy Mapping

`expected_flags` must use labels from `Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md`. V1 contract fixtures intentionally start with two calibration examples:

- A passing prompt-reveal refusal case linked to `PC-CASE-008`.
- A failing read-only memory promise case linked to `PC-CASE-015` and `PC-MEM-003`.

Additional fixture cases should preserve the same envelope and continue mapping back to the `PC-CASE-###` corpus.

## Non-Goals

- No direct live judge provider calls from the adapter itself.
- No Persona Chat endpoint, worker, or recipe execution changes.
- No runtime chat gating, blocking, retrying, or response rewriting.
- No Persona Live, avatar, visual pack, VN/CYOA, or native companion changes.
- No parallel evaluation subsystem outside the existing Evaluations and Jobs direction.

## Remaining Executable Harness Prerequisites

Before treating executable judge output as calibrated or production-usable, follow-up PRs should still define:

- How any future persisted judge reports link back to deterministic Persona Chat trace ids without storing sensitive content.
- Provider-specific completion adapters and configuration, if needed, outside the offline boundary.
- Held-out calibration data and thresholds for any production quality claims.
