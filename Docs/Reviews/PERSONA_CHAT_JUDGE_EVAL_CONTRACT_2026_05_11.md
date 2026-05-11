# Persona Chat Judge Evaluation Contract

Date checked: 2026-05-11

## Summary

This document defines the V1 contract for the optional Persona Chat judge evaluation tracked in [#1566](https://github.com/rmusser01/tldw_server/issues/1566). It is a contract and fixture slice only. It does not add an executable judge, change normal Persona Chat runtime behavior, add production scoring, or gate chat responses.

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

1. Run against the golden contract fixture in `tldw_Server_API/tests/fixtures/persona_chat_judge_contract_cases.json`.
2. Run against a separate held-out set before any threshold tuning is accepted.
3. Report agreement measures for each judged axis, including false-positive and false-negative examples.
4. Preserve deterministic checks as hard preconditions before subjective scoring.
5. Document known bias cases, especially generic-assistant bias, over-rewarding theatrical style, and under-penalizing memory-mode claims.

Judge execution remains deferred until these calibration requirements are implemented and reviewed.

## Offline Harness

The first executable layer is the offline harness in `tldw_Server_API/app/core/Evaluations/persona_chat_judge_harness.py`. It compares already-produced candidate judge outputs against the checked-in V1 contract fixture and returns a bounded report with case counts, verdict agreement, flag agreement, score schema validity, missing candidates, invalid candidates, extra candidates, and per-case mismatch keys.

The harness does not call model providers, persist evaluation runs, enqueue Jobs, expose API endpoints, or gate Persona Chat responses. Future judge adapters should feed their outputs into this helper before any output is treated as calibrated.

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

- No live judge provider calls.
- No Persona Chat endpoint, worker, or recipe execution changes.
- No runtime chat gating, blocking, retrying, or response rewriting.
- No Persona Live, avatar, visual pack, VN/CYOA, or native companion changes.
- No parallel evaluation subsystem outside the existing Evaluations and Jobs direction.

## Future Executable Harness Prerequisites

Before adding executable judge code, the next PR should define:

- The exact prompt and model input shape derived from this envelope.
- The persistence location for offline reports.
- The review command that can run without configured commercial providers.
- The calibration report schema and threshold policy.
- How judge reports link back to deterministic Persona Chat trace ids without storing sensitive content.
