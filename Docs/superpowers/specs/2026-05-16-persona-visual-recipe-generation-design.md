# Persona Visual Recipe-Backed Generation Workflow Design

## Status

Draft design for GitHub issue #1765 and Backlog TASK-406.

## Purpose

Persona Visual starter packs now expose `production_recipe` metadata for the bundled starter catalog. That metadata describes the authored-asset workflow: identity brief, neutral anchor, static sheet guidance, animation output targets, and review checks. The next backend step is to let generation requests reference those recipe outputs while reusing the existing Persona Visual generation Jobs and generated-candidate review flow.

This design is backend-only. It does not add WebUI behavior, final art generation, automatic activation, runtime renderer support, MCP provider execution, marketplace behavior, shared library behavior, or VN/CYOA behavior.

## Existing System

The current backend already has the right primitives:

- `PersonaVisualGenerationRequest` accepts a prompt, target state, and optional backend.
- `create_generate_candidate_job()` creates a `persona_visual_generate_candidate` Jobs row in the `persona_visuals` domain.
- `PersonaVisualGenerationWorker` resolves an image backend, generates one image, persists it as a generated asset, and creates a review-gated visual candidate.
- Generated candidates are listed, fetched, accepted, rejected, or failed through existing review endpoints.
- Starter catalog responses expose bounded `production_recipe` metadata after PR #1762.

The gap is that the generation request only carries free-form prompt text. It does not have a backend contract for saying “generate the `static_talking_reaction_sheet` output described by starter `x` using this starter’s recipe metadata.”

## Goals

1. Add a backend contract for recipe-backed generation intent.
2. Validate requested recipe outputs against bundled starter metadata.
3. Build bounded recipe context for the existing generation job payload.
4. Preserve the existing generated-candidate review gate.
5. Keep implementation slices small and reversible.

## Non-Goals

- No WebUI or extension changes.
- No automatic pack activation.
- No automatic candidate acceptance.
- No final default-asset production or bundled authored art.
- No renderer expansion beyond existing `sprite_frames` candidate behavior.
- No new MCP provider execution or resource download.
- No new marketplace/shared-library behavior.
- No VN/CYOA behavior.
- No new parallel job system.

## Proposed Approach

Extend the existing generation request and job payload with optional recipe intent fields. A request may remain prompt-only, preserving the current behavior. When recipe intent is present, the backend validates it against the starter catalog, derives bounded recipe context, and queues the same `persona_visual_generate_candidate` job type.

Recommended request shape:

```json
{
  "request_id": "client-or-server-correlation-id",
  "prompt": "User direction layered onto the recipe",
  "target_state": "speaking",
  "backend": "configured-image-backend",
  "starter_pack_id": "search-lens-basic",
  "recipe_output": "required_state_loops"
}
```

`starter_pack_id` and `recipe_output` are optional together. If one is present without the other, reject the request with a validation error. If both are absent, keep current prompt-only generation behavior.

## Backend Contract

### Request Fields

- `starter_pack_id`: optional string, bounded to the same identifier expectations as starter catalog IDs.
- `recipe_output`: optional string, bounded to the same item text limit used by production recipe metadata.
- `prompt`: remains required for V1 unless implementation chooses a later explicit recipe-only mode. For the first slice, keeping it required avoids silent generation from generic recipes.
- `request_id`: optional client-provided correlation identifier. If omitted, the backend should generate one before validation/enqueue and return it with the job response in the implementation slice.

### Validation Rules

- The target persona exists and belongs to the user.
- The visual pack exists and belongs to that persona/user.
- If recipe fields are present:
  - `starter_pack_id` resolves in the bundled starter catalog.
  - `recipe_output` is included in `starter.production_recipe.animation_outputs`.
  - Recipe text and output values are already bounded by starter catalog validation.
  - The request does not mutate starter metadata.
- Unsupported starter IDs or recipe outputs return `400` with machine-readable detail.

### Job Payload

Add a `recipe_intent` object to the existing generation payload:

```json
{
  "user_id": "1",
  "persona_id": "persona-id",
  "pack_id": "pack-id",
  "request_id": "client-or-server-correlation-id",
  "prompt": "Effective bounded prompt sent to the image adapter",
  "target_state": "speaking",
  "backend": "configured-image-backend",
  "recipe_intent": {
    "starter_pack_id": "search-lens-basic",
    "recipe_output": "required_state_loops",
    "correlation_id": "client-or-server-correlation-id",
    "user_prompt": "User direction layered onto the recipe",
    "identity_brief": "bounded recipe identity text",
    "neutral_anchor": "bounded neutral anchor text",
    "static_sheet": "bounded static sheet text",
    "review_checks": ["neutral_identity_consistency"]
  }
}
```

The payload should not include raw external provider output, secrets, file paths, or mutable starter objects. It should contain only bounded recipe strings and identifiers needed for trace-safe replay and review.

`request_id` and `recipe_intent.correlation_id` should carry the same normalized value for recipe-backed requests. Plain prompt-only requests may use the same request ID behavior without `recipe_intent`.

## Prompt Construction

For the first implementation slice, prompt construction should happen before enqueueing, not in the worker. This keeps idempotency and request validation deterministic.

Recommended prompt composition:

1. A short system-owned prefix that states the selected recipe output.
2. The bounded starter recipe fields.
3. The user prompt as an additional direction.
4. Review checks as explicit constraints.

The payload `prompt` should be the effective prompt sent to the image adapter. For prompt-only requests it is the normalized user prompt, preserving current behavior. For recipe-backed requests it is the backend-composed prompt. The original user prompt should be retained as `recipe_intent.user_prompt` so downstream review tooling can distinguish authored recipe context from user direction without parsing the effective prompt.

The final prompt should be bounded by the existing prompt max length after composition. If the composed prompt would exceed the max, fail closed with a validation error rather than truncating recipe context silently.

The implementation should avoid introducing prompt-template configurability in this slice. The composition rules should be deterministic backend code, with tests asserting stable output for a sample recipe-backed request.

## Idempotency

Current idempotency keys include user, persona, pack, target state, prompt, and backend. Recipe-backed requests should include a digest of:

- request ID/correlation ID
- `starter_pack_id`
- `recipe_output`
- bounded recipe context used at enqueue time
- user prompt
- target state
- backend

This prevents a prompt-only request and a recipe-backed request with similar text from collapsing into the same job. It also avoids accidental reuse when starter recipe metadata changes in a future PR. The correlation ID should not replace idempotency; it exists for audit/debug linkage, while idempotency still covers payload equivalence.

## Worker Behavior

The worker should remain mostly unchanged:

- It receives the existing job type.
- It validates required identity fields and pack ownership.
- It uses the already-composed prompt for image generation.
- It persists one generated asset.
- It creates one generated candidate with a proposed manifest patch.

For the first implementation, the worker should not require a DB schema migration. Keep recipe intent in the Jobs payload/result unless an existing candidate metadata field can store it without changing persistence. If durable candidate provenance becomes necessary, create a follow-up slice with a migration and API response design.

## Review Gate

The current candidate review gate remains authoritative:

- Generated recipe-backed assets are `generated_candidate` assets.
- Candidates remain review-only until accepted.
- Accepting a candidate remains explicit and separate from generation.
- Rejection/failure handling remains unchanged.
- No recipe-backed job may activate a pack or alter the active visual pack directly.

## Error Handling

Use existing endpoint error style and add specific machine-readable messages where possible:

- `starter_pack_id_required_with_recipe_output`
- `recipe_output_required_with_starter_pack_id`
- `starter_pack_not_found`
- `recipe_output_not_found`
- `recipe_prompt_too_long`
- `invalid_recipe_generation_payload`

Backend failures from image providers continue to surface through existing job failure semantics.

## Traceability and Audit Events

Recipe-backed generation should define one correlation chain from request validation through review outcome:

- `request_id`: client-provided or server-generated identifier returned by the generation-job response.
- `correlation_id`: same normalized value copied into `recipe_intent` and log/audit metadata.
- `job_id`: Jobs row ID returned by the existing generation job creation path.
- `candidate_id`: generated-candidate ID created by the worker.

The implementation should emit structured, bounded log or audit events at these points:

- `persona_visual.recipe_generation.request_validated`: after persona, pack, starter, and recipe output validation.
- `persona_visual.recipe_generation.job_created`: after the Jobs row is created; include `request_id`, `correlation_id`, `job_id`, `persona_id`, `pack_id`, `starter_pack_id`, and `recipe_output`.
- `persona_visual.recipe_generation.candidate_created`: after the worker stores the generated candidate; include `request_id` or `correlation_id` when available, plus `job_id`, `candidate_id`, `persona_id`, and `pack_id`.
- `persona_visual.recipe_generation.candidate_reviewed`: when an existing candidate review endpoint accepts, rejects, or fails a recipe-backed candidate; include `job_id`, `candidate_id`, review status, and correlation identifiers when available.

Events must not include raw provider credentials, generated image bytes, local filesystem paths, unbounded prompts, or raw exception bodies. If durable candidate provenance is deferred in Slice 1, the review event may fall back to `job_id` and `candidate_id` correlation until a later slice adds durable `recipe_intent` lookup.

## Trace Safety

Recipe intent is safe to store and return only if it remains bounded and non-secret:

- Do not store API keys, provider credentials, local file paths, or downloaded resource URLs.
- Do not store generated image binary data in job payloads.
- Do not store unbounded prompt expansions.
- Do not echo unknown starter metadata fields.
- Keep recipe context derived only from validated bundled starter catalog data.

## Implementation Slices

### Slice 1: Backend Contract and Job Payload

- Extend request schema with optional `starter_pack_id` and `recipe_output`.
- Add service/helper logic to validate starter recipe intent.
- Compose bounded generation prompt.
- Add `recipe_intent` to the existing generation job payload.
- Add `request_id`/correlation ID handling and trace/audit event requirements.
- Update idempotency digest.
- Add unit/API tests.
- Do not change worker persistence or candidate response shape.

### Slice 2: Worker and Candidate Provenance, If Needed

- Preserve recipe intent in job result or candidate metadata only if an existing durable metadata path exists.
- Add tests that recipe-backed jobs still create review-gated candidates.
- Avoid DB migrations unless there is a concrete review need that cannot be met from existing Jobs payload/result storage.

### Slice 3: Follow-Up UX Planning

Out of scope for this design implementation. A later frontend slice can read starter recipes and queue recipe-backed generation jobs, but this design does not specify UI layout, copy, controls, or routes.

## Testing Strategy

Focused tests should cover:

- Prompt-only generation remains unchanged.
- Request with only `starter_pack_id` fails.
- Request with only `recipe_output` fails.
- Request without prompt fails because prompt remains required for V1.
- Unknown starter ID fails.
- Unknown recipe output fails.
- Composed prompt exceeding the max length fails closed.
- Valid recipe-backed request queues the existing job type.
- Job payload includes bounded `recipe_intent`.
- Job payload and response include request/correlation identifiers.
- Idempotency distinguishes recipe-backed and prompt-only requests.
- Worker still creates generated candidates without activation.

## Open Questions

1. Should recipe intent be durably copied onto generated candidates in the first implementation slice, or is Jobs payload/result traceability enough for V1?
2. Should recipe-backed generation require a user prompt in V1, or allow recipe-only requests later once backend prompt templates are proven?
3. Should `recipe_output` be constrained to an enum generated from bundled starters at runtime, or remain a bounded string validated by service logic?

## Recommendation

Implement Slice 1 first. It gives the backend a clear recipe-backed generation contract while preserving the existing Jobs and generated-candidate review architecture. It avoids new persistence and UI commitments until the request/job semantics are proven.
