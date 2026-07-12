# Research Workspace Artifact Claim Verification Design

Task: TASK-12146
Date: 2026-07-04
Status: Draft

## Goal

Move Research Workspace generated artifacts from "structurally real and non-placeholder" to "factually good enough" by verifying generated artifact claims against the exact selected source excerpts during backend generation.

The gate uses the existing Claims module as the source of truth. It must not call the public `/api/v1/claims/*` HTTP API from the frontend or from backend generation code.

## Constraints

- Verification runs as an internal backend call inside the generation process.
- Verification uses the exact Research Workspace source excerpts selected for the artifact.
- Verification does not re-retrieve broader media DB context for the default gate.
- Claims verification uses existing `ClaimsEngine`, `ClaimVerification`, `VerificationStatus`, and `VerificationReport` semantics.
- FVA is not the default gate. It remains a later high-risk/contested follow-up path.
- Frontend-only generation paths must move backend-side before they can use this gate.
- Provider/model settings from artifact generation must be propagated into claim extraction and verification by default so local llama.cpp workflows do not silently switch providers.
- Users may configure a separate claims-verification provider/model for this gate. When that non-default verifier is used, the backend response must make it explicit so the frontend can show which LLM verified the artifact.

## Non-Goals

- Do not add a public artifact verification endpoint for this workflow.
- Do not create a separate frontend `factCheck` vocabulary that duplicates Claims module statuses.
- Do not calibrate a full evaluator or dashboard before gating generation.
- Do not require perfect factuality. The target is conservative high-confidence failure blocking plus review marking for uncertainty.

## Existing Reuse Points

- `tldw_Server_API/app/core/Claims_Extraction/claims_engine.py`
  - `Claim`
  - `Evidence`
  - `ClaimVerification`
  - `ClaimsEngine.extract_claims_only(...)`
  - `ClaimsEngine.verify_claims_only(..., doc_only_mode=True)`
- `tldw_Server_API/app/core/Claims_Extraction/verification_report.py`
  - `generate_verification_report(...)`
- `tldw_Server_API/app/core/RAG/rag_service/types.py`
  - `VerificationStatus`
  - `Document`

## Design

Add one internal helper:

`tldw_Server_API/app/core/Claims_Extraction/artifact_verification.py`

Primary function:

```python
async def verify_generated_artifact_against_sources(
    *,
    artifact_type: str,
    units: list[ArtifactVerificationUnit],
    source_documents: list[Document],
    generation_provider: str | None,
    generation_model: str | None,
    verification_provider: str | None = None,
    verification_model: str | None = None,
    generation_context: dict[str, Any] | None = None,
) -> ArtifactVerificationResult:
    ...
```

The helper:

1. Converts artifact units into claim text.
2. Uses deterministic unit claims where the artifact structure already contains facts.
3. Uses `ClaimsEngine.extract_claims_only(...)` only for prose-heavy units.
4. Uses `ClaimsEngine.verify_claims_only(..., doc_only_mode=True)` against the provided `source_documents`.
5. Builds the existing `VerificationReport`.
6. Applies an artifact-level policy verdict:
   - `grounded`
   - `needs_revision`
   - `failed`
7. Returns report data plus unit mappings.

## Artifact Units

Each claim must retain a source artifact unit ID so users can see what failed.

Examples:

- Quiz: `quiz:q3:question`, `quiz:q3:answer`, `quiz:q3:explanation`
- Flashcards: `flashcard:7:front`, `flashcard:7:back`
- Audio summaries: `audio_script:paragraph:4`
- Data tables: `table:row:3`, optionally `table:row:3:column:revenue`
- Slides: `slide:2:title`, `slide:2:bullet:1`, `slide:2:speaker_notes`
- Mindmaps: `mindmap:node:research_gap`, `mindmap:edge:research_gap->method`

Flattening is acceptable only as an input to ClaimsEngine. The result must preserve the unit ID.

## Verdict Policy

The default policy is mixed.

`failed`:

- High-confidence `refuted`
- `hallucination`
- `numerical_error`
- `misquoted`
- Strong `citation_not_found`
- No usable selected source documents

`needs_revision`:

- `unverified`
- `contested`
- Low verification rate
- Weak evidence
- Budget or provider limits prevented verification

`grounded`:

- No failed claims
- Verification rate meets the artifact threshold
- Review-only statuses stay below the artifact threshold

Artifact thresholds should be conservative and hard-coded first. No settings UI.

Recommended defaults:

- Quiz and flashcards: stricter, because answers/back sides are direct factual outputs.
- Data tables: strict on rows with entities, numbers, dates, or relationships.
- Slides: strict on titles, bullets, and speaker notes that contain named entities, numbers, dates, or causal claims.
- Audio summaries: strict on central claims; review for broad unsupported wording.
- Mindmaps: strict only on named entities, numbers, dates, and causal edges; review for vague unsupported nodes.

## Generation Flow

Backend generation should follow this order:

1. Load exact selected Research Workspace sources.
2. Generate artifact draft.
3. Normalize draft into artifact units.
4. Verify units internally through Claims module.
5. If `failed`, mark generation failed and retain failure report.
6. If `needs_revision`, persist artifact with `reviewStatus: "needs_revision"` and `claimVerification`.
7. If `grounded`, persist artifact with `claimVerification`.
8. Only then mark the artifact successful/complete.

For generators that currently persist before content inspection, keep the record in a draft/failed state until verification finishes. Do not present it as a completed asset if the claims gate failed.

## Provider and Model Selection

The internal helper must not rely on global Claims settings for this gate.

It receives generation provider/model and optional claims-verification provider/model. If no claims-verification override is configured, the helper uses the generation provider/model. If a claims-verification override is configured, the helper uses that provider/model for claim extraction and verification and records that the verifier differs from generation.

The callable may ignore ClaimsEngine's default provider/model arguments when necessary so the gate uses the requested verifier instead of a global Claims default.

The result metadata should record:

- generation provider/model
- verification provider/model actually used
- whether an explicit claims-verification provider/model override was applied
- whether the verification model differs from the generation model
- budget or provider fallback state

This is required for llama.cpp at `127.0.0.1:9099` and similar local-provider workflows.

The frontend settings UI should expose the claims-verification provider/model as an advanced Research Workspace generation option. Artifact details should display the verifier when it differs from the generation LLM.

## Backend Integration Targets

First slice:

- Data tables: hook worker/service after generated rows exist and before success status.
- Slides: hook presentation generation after draft slides exist and before completed presentation state.
- Flashcards/quizzes: hook backend generation/save services where generated drafts are available.

Second slice:

- Audio summaries: move script generation backend-side, verify script before TTS.
- Mindmaps: move generation backend-side, verify Mermaid node/edge labels before returning.
- Any remaining frontend `createChatCompletion` artifact path: move behind backend orchestration before adding this gate.

The frontend can still call backend generation endpoints. The forbidden path is calling Claims HTTP endpoints for verification.

## Persistence Shape

Store verification under generated artifact data as:

```json
{
  "claimVerification": {
    "verdict": "grounded | needs_revision | failed",
    "report": {},
    "unitResults": [],
    "metadata": {}
  }
}
```

`report` is the existing `VerificationReport.to_dict()` output.

Do not store a parallel `factCheck.claims` schema.

## Cost Control

Use caps before any model call:

- Max units per artifact type.
- Prefer deterministic claims for structured artifacts.
- Skip low-information units like headings without facts.
- Verify high-risk units first: numbers, dates, named entities, causal claims, answers, explanations, and table rows.

If the cap is hit, return `needs_revision`, not `grounded`.

## Tests

Backend unit tests:

- Builds `Document` objects from explicit source excerpts.
- Maps Claims statuses to artifact verdicts.
- Preserves artifact unit IDs.
- Marks refuted/hallucinated/numerical-error claims as failed.
- Marks unverified/contested as needs revision.
- Does not retrieve from media DB for the default helper.

Backend integration tests:

- Data table generation stores failed/needs_revision/grounded claim verification results.
- Slides generation does not mark a presentation complete when verification fails.
- Quiz and flashcard generation verify answers and explanations before success.

Property tests:

- Generated units containing numbers/dates/entities absent from source text cannot be `grounded`.
- Unit IDs round-trip through normalization and report mapping.

Frontend tests:

- Completed artifacts show `claimVerification`.
- `needs_revision` maps to existing review UI state.
- Failed verification surfaces the backend error/report without creating a clean completed artifact.

## Rollout

1. Add internal helper and tests.
2. Wire the helper into one backend-generated artifact type with persistence gating.
3. Extend to the remaining backend-generated artifact types.
4. Move frontend-only artifact generation backend-side.
5. Wire audio summaries and mindmaps.
6. Add optional FVA follow-up for high-risk or contested artifacts.

## Open Implementation Notes

- The first implementation should avoid changing public Claims endpoints.
- If ClaimsEngine needs first-class provider override support, add it narrowly and preserve existing callers.
- If a generator writes DB rows before verification, prefer status updates over deleting drafts.
