# Planned Flashcard Generation Controls Design

Task: TASK-12170

## Goal

Add planned flashcard generation so users can request exact counts of supported flashcard styles from one source selection.

This design reuses the existing flashcard generation endpoint, deck/card storage, preview flow, media asset paths, and spaced-repetition scheduler. It does not add a new card model, scheduler model, visual card extraction, or automatic spaced-repetition behavior.

## Current State

The flashcard generation API accepts a single `num_cards` value and one `card_type`: `basic`, `basic_reverse`, or `cloze`. The WebUI `/flashcards` Generate panel exposes the same single-type shape. Other callers, including selected-text sidepanel generation and Research Workspace artifact generation, call the same client service with a fixed or local default.

The backend stores generated cards using existing flashcard model types. There is no persisted true/false flashcard type today, and the scheduler should not need to know how a generated card was requested.

## API Contract

Add an optional `card_plan` field to `FlashcardGenerateRequest`.

Example:

```json
{
  "text": "...",
  "num_cards": 10,
  "card_plan": [
    { "card_type": "basic", "count": 5 },
    { "card_type": "basic_reverse", "count": 2 },
    { "card_type": "cloze", "count": 2 },
    { "card_type": "true_false", "count": 1 }
  ]
}
```

Plan row fields:

- `card_type`: `basic`, `basic_reverse`, `cloze`, or `true_false`.
- `count`: integer from `1` to the existing total card cap.

Validation rules:

- `card_plan` and explicit `card_type` are mutually exclusive.
- `num_cards` must be explicitly present when `card_plan` is present.
- Sum of plan row counts must equal `num_cards`.
- Duplicate `card_type` rows are rejected.
- Unsupported card types, zero counts, and unknown row fields are rejected.
- Existing total generation caps still apply.

Legacy requests without `card_plan` remain backward compatible. The schema should not let the `card_type` default conflict with planned requests: treat `card_type` as optional at validation time, then default to `basic` only for non-plan requests.

`true_false` is only valid inside `card_plan` in v1. Legacy single-type requests keep the existing `basic`, `basic_reverse`, and `cloze` values.

## Response Metadata

Add response-only `generation_type` to generated flashcard previews:

```ts
generation_type?: "basic" | "basic_reverse" | "cloze" | "true_false"
```

This field exists only so the backend can validate planned output and the UI can label preview cards. It is not persisted to deck cards, scheduling state, or card review records. Save/create calls should drop it before persistence.

For legacy requests, `generation_type` can mirror the requested `card_type`. For planned requests, validation uses `generation_type` before any storage normalization.

## Generation Service

Refactor generation around a normalized internal plan while preserving current legacy behavior.

Legacy path:

- Build a single-row internal plan from `num_cards` and resolved `card_type`.
- Keep current prompt behavior and compatible output normalization.

Planned path:

- Pass `card_plan` through the endpoint and workflow adapter config explicitly.
- Update `FlashcardGenerateConfig` to include `card_type` and `card_plan`; do not keep relying on the current `format` field while the adapter reads `card_type`.
- Prompt the model for exact counts and require each returned object to include `generation_type`.
- Validate exact counts by `generation_type` before returning.
- Fail the whole planned generation if counts or generated shapes do not match.
- Do not add automatic retries in v1.

True/false handling:

- `true_false` is a generation style, not a stored model type.
- Generated true/false cards return `generation_type: "true_false"` and `model_type: "basic"`.
- The front should read like a true/false prompt, for example `True or false: ...`.
- The back should contain the answer and a short explanation.

Stable failure message shape:

```text
Generated flashcards did not satisfy requested card plan: cloze expected 2, got 1
```

Deterministic test mode must support planned requests so endpoint tests exercise the same validation path instead of bypassing it.

## WebUI

Update `/flashcards` Generate panel first.

Default mode remains the current simple flow:

- `Number of cards`.
- `Card type`.
- Existing difficulty, focus topics, provider/model, and deck controls.
- Submit the legacy payload shape.

Add a visible `Advanced mix` toggle next to the existing card count/type controls. When enabled:

- Show fixed rows for `basic`, `basic_reverse`, `cloze`, and `true_false`.
- Do not allow duplicate rows.
- Use count inputs only; no separate total input.
- Omit disabled or zero-count rows from `card_plan`.
- Derive `num_cards` from enabled/count rows and submit it with `card_plan`.
- Default mix is `5 basic`, `2 basic_reverse`, `2 cloze`, `1 true_false`.
- Disable generation when derived total is `0` or above the existing API cap.
- Preserve all user inputs after a generation failure.
- Preview cards use `generation_type` labels, including `True/False` for true/false cards.
- Save strips `generation_type` and persists existing flashcard card shapes.

## Other Callers

Shared frontend service types and generation hooks should accept optional `card_plan`, but callers should not change hidden defaults.

- Quiz companion flashcards keep their current default unless a visible mix selection is added.
- Sidepanel selected-text generation keeps its current quick-generate default unless the sidepanel exposes a mix selector.
- Research Workspace artifact generation keeps current defaults unless the artifact flow exposes a selected mix.
- Any caller that does expose a selected mix should send the same `card_plan` payload and use `generation_type` for preview labels.

This satisfies extension/WebUI compatibility without surprising users by silently changing generated card distributions.

## Tests

Backend schema and endpoint tests:

- Legacy request still works without `card_plan`.
- Valid mixed `card_plan` is accepted.
- `card_plan` plus explicit `card_type` is rejected.
- Missing explicit `num_cards` with `card_plan` is rejected.
- Sum mismatch is rejected.
- Duplicate plan rows are rejected.
- Unknown plan row fields are rejected.
- Zero-count plan rows are rejected.
- Unsupported plan card types are rejected.
- Legacy `card_type: "true_false"` without `card_plan` is rejected.
- `true_false` is accepted in planned requests.

Generation tests:

- Planned deterministic generation returns exact counts.
- Planned validation fails on wrong counts.
- `true_false` validates by `generation_type` and returns stored `model_type: "basic"`.
- Prompt/template coverage includes requested counts and all four generation types.

Frontend tests:

- Advanced mix off sends the legacy payload.
- Advanced mix on sends derived `num_cards` plus `card_plan`.
- Derived total updates as row counts change.
- Generate is disabled for total `0` and over-cap totals.
- True/false preview labels use `generation_type`.
- Save payloads do not persist `generation_type`.

## Out Of Scope

- Visual flashcards from images or source figures.
- Persisting generated-style metadata.
- New scheduler or spaced-repetition models.
- Automatic retry/repair loops for bad LLM output.
- Full sidepanel or Research Workspace mix-selection redesigns.
- Changing flashcard review, import/export, or deck storage schemas.

## Implementation Notes

- Keep this as an API-boundary feature first.
- Use strict validation only when `card_plan` is present.
- Keep legacy generated behavior backward compatible for clients that only send `num_cards` and `card_type`.
- Prefer fixed-row UI over a dynamic builder; duplicate rows are not needed for this task.
