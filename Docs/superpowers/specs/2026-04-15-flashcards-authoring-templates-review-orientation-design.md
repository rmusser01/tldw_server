# Flashcards Authoring Templates and Review Orientation Design

Date: 2026-04-15
Status: Approved in brainstorming, pending spec review and implementation planning
Owner: Codex brainstorming session

## Summary

Add two related flashcards improvements:

1. User-level authoring templates that help heavy card authors create many similar cards quickly.
2. A deck-level review orientation default that can show non-cloze cards back-first instead of front-first during study.

The design keeps these concerns separate from the existing flashcard schema:

- `basic`, `basic_reverse`, and `cloze` remain card models, not authoring templates.
- Authoring templates are reusable presets that materialize normal card drafts.
- Review orientation changes study presentation only; it does not change stored flashcard content or scheduling behavior.

## Problem

The current flashcards create flow in [FlashcardCreateDrawer.tsx](../../../apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx) supports manual creation well enough for one-off cards, but it is slower than it needs to be for users building many structurally similar cards.

Current friction:

- the UI already uses `template` language for card model selection, which is ambiguous for users who also want reusable authoring presets
- authors must repeatedly re-enter deck, tags, model choice, and recurring front/back scaffolding
- there is no first-class place to save and reuse flashcard-specific creation patterns

The current review flow in [ReviewTab.tsx](../../../apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx) always presents the `front` side first.

Current limitation:

- users who want to practice inverse recall from answer to prompt cannot set a deck to study back-first
- `basic_reverse` exists as a card model, but the live review UI does not have a deck-level presentation setting that flips the prompt side for the whole deck

## Goals

- Add reusable, user-owned flashcard authoring templates.
- Allow templates to prefill deck, tags, card model, and front/back/notes/extra scaffolding.
- Support explicit placeholder definitions with labels, help text, defaults, and required flags.
- Let users apply a template through a lightweight fill step before the card lands in the normal create drawer.
- Let users save the current draft as a reusable template.
- Provide a dedicated Flashcards `Templates` surface as the primary template library home.
- Add a deck-level review orientation default for non-cloze cards.
- Allow a session-level review override without changing deck settings.
- Preserve current scheduling, undo, analytics, and review semantics.

## Non-Goals

- Create custom flashcard schemas or arbitrary user-defined field structures.
- Replace `basic`, `basic_reverse`, or `cloze` with a new note-type system.
- Auto-create cards immediately when a template is applied.
- Infer placeholders from arbitrary `{{token}}` text without explicit template field definitions.
- Change how review scheduling works.
- Make cloze cards study back-first.
- Introduce a global app-wide review orientation preference as the primary behavior.

## Confirmed Product Decisions

- Authoring templates are user-level, not deck-owned.
- Templates may prefill:
  - default deck
  - default tags
  - default card model
  - starter content for front, back, notes, and extra
- Templates support explicit named placeholder fields with labels/help/defaults.
- Applying a template uses a short fill step before values are written into the existing create drawer.
- Templates are managed both:
  - inline from the create drawer
  - from a dedicated Flashcards templates library
- Deck review orientation applies to all review sessions for that deck.
- Deck review orientation applies to all non-cloze cards, not only `basic_reverse`.
- Separate template management should live primarily inside Flashcards, not global settings.

## Current State

### Flashcard authoring already has the right extension point

[FlashcardCreateDrawer.tsx](../../../apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx) already manages:

- deck selection
- inline deck creation
- card model selection
- front/back/notes/extra inputs
- tag selection
- create mutations through existing shared hooks

That makes it the natural place to apply templates and to offer `Save as template`.

### The current term `template` is overloaded

The existing create and edit drawers use `model_type` with labels like `Basic`, `Basic (reverse)`, and `Cloze`. That is a card-behavior choice, not an authoring preset.

Design implication:

- rename UI copy from `Card template` to `Card model` or `Card type`
- reserve `template` language for reusable authoring presets

### Review already routes through one active-card presentation path

[ReviewTab.tsx](../../../apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx) derives a single `activeCard` and always renders:

- `front` as the prompt
- `back` as the revealed answer

Because due review and cram review share that rendering path, a deck-level orientation setting can be added as presentation logic without changing review mutations or scheduling.

### Deck settings already have a natural home

[DeckSchedulerSettingsEditor.tsx](../../../apps/packages/ui/src/components/Flashcards/components/DeckSchedulerSettingsEditor.tsx) and the deck create/update flow already support deck-level behavioral configuration. Review orientation should live alongside deck settings, not as an isolated ad hoc flag in Review.

### The repo already contains a template CRUD precedent

The writing/playground feature already uses user-owned template persistence and CRUD patterns in:

- [writing.py](../../../tldw_Server_API/app/api/v1/endpoints/writing.py)
- `ChaChaNotes_DB` writing template helpers

Design implication:

- flashcards templates should follow a similar persistence and optimistic-locking pattern rather than being stored as generic UI settings

## Approaches Considered

### Approach 1: Separate authoring templates from card models

Keep card model selection as-is conceptually, rename its UI label, and add a distinct authoring-template layer with its own CRUD, template-apply step, and library.

Pros:

- clear mental model
- fits explicit placeholder definitions
- scales without overloading the drawer
- aligns with existing deck settings and template CRUD patterns

Cons:

- needs new persistence and APIs
- template application adds one intentional step

### Approach 2: Keep almost all template behavior inside the create drawer

Make the drawer both the template application tool and the main template management surface, with the separate library used only secondarily.

Pros:

- efficient for heavy authors
- minimal navigation

Cons:

- increases create-drawer complexity
- higher state-management and responsive-layout risk

### Approach 3: Treat templates as lightweight settings-only presets

Store template-like presets in settings or client-only UI state without a first-class flashcards template domain.

Pros:

- smaller initial implementation

Cons:

- weak fit for named placeholder fields
- poor long-term sync/versioning semantics
- feels bolted on compared with other template systems in the repo

## Recommendation

Use Approach 1.

Authoring templates should be a first-class flashcards concept, distinct from card models and distinct from deck settings. Deck review orientation should be a deck-level study setting with a temporary session override in Review.

## Proposed Design

### 1. Flashcards authoring templates become a first-class flashcards resource

Add a new user-owned flashcards template entity.

Each template stores:

- `name`
- optional `description`
- optional default `deck_id`
- default `tags`
- default `model_type`
- scaffold text for `front`, `back`, `notes`, and `extra`
- explicit placeholder definitions
- standard metadata such as `version`, `created_at`, `last_modified`, `deleted`, `client_id`

Placeholder definitions should include:

- stable key
- user-facing label
- optional help text
- optional default value
- required flag
- target field mapping or mappings

Templates produce ordinary flashcard drafts. They do not create a new flashcard schema and do not alter the persisted flashcard model.

### 2. Create drawer flow adds template apply and save actions

In [FlashcardCreateDrawer.tsx](../../../apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx):

- rename `Card template` copy to `Card model` or `Card type`
- add `Apply template`
- add `Save as template`

Applying a template should use this flow:

1. user opens template picker
2. user chooses a template
3. system opens a compact `Fill Template Values` step
4. user fills required and optional named inputs
5. system resolves placeholders into scaffold text
6. resolved values are written into the normal create form
7. user can still edit everything before creating the flashcard

Saving as template should use the current draft as the starting point, but the UX should make it explicit which draft content becomes:

- literal default text
- placeholder-backed scaffold

That avoids silently freezing one-off draft content into a reusable template.

### 3. Templates library lives inside Flashcards

Add a dedicated `Templates` tab to [FlashcardsManager.tsx](../../../apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx).

This becomes the primary management home for:

- listing templates
- searching/filtering templates
- creating templates from scratch
- editing metadata, scaffolding, and placeholder definitions
- duplicating templates
- soft deleting templates

The create drawer remains the fast-entry point for applying and saving templates, but deeper management belongs in the Templates tab.

Routing and visibility rules:

- add `templates` to flashcards tab parsing so `?tab=templates` opens the library directly
- keep the Templates tab visible even when the user has zero decks, because templates are user-level and can be created before a first deck exists
- keep the existing no-deck startup default unchanged unless later product work intentionally revisits flashcards FTUE; this feature should not broaden startup-routing scope unnecessarily

### 4. Review orientation becomes a deck-level study setting

Extend the deck model with a new field such as:

- `review_prompt_side: "front" | "back"`

Default:

- `front`

Behavior:

- if a deck is `front`, review works exactly as it does today
- if a deck is `back`, non-cloze cards show the `back` as the initial prompt and reveal the `front` as the answer
- cloze cards ignore the setting and remain front-first

This setting belongs in deck create/edit settings rather than scheduler settings JSON because it is a separate study-orientation concern, not a scheduling algorithm concern.

### 5. Review sessions allow temporary override

Add a session-level review orientation control to [ReviewTab.tsx](../../../apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx).

Precedence:

1. session override
2. deck default
3. app default (`front`)

Expected behavior:

- changing the session override changes prompt/answer presentation only
- it does not persist back to the deck automatically
- it resets when the existing review scope key changes, matching the current review-session lifecycle

For implementation, `reviewScopeKey` should be treated as the concrete reset boundary. In current terms, the override resets when any of these change:

- review mode (`due` vs `cram`)
- selected deck
- cram tag filter

It should not reset for ordinary within-scope card progression such as:

- moving to the next card in the same queue
- revealing the answer
- submitting a rating
- using undo inside the same review scope

### 6. Scheduling and review semantics stay unchanged

The review-orientation feature is a presentation change only.

It must not change:

- review rating meaning
- answer-time capture
- undo behavior
- queue progression
- session analytics
- scheduler calculations
- flashcard storage

This is critical because users are asking to reverse recall direction, not to create a second scheduler or a second card record.

## Data Model

### Proposed flashcards template schema

Suggested API/resource shape:

- `id`
- `name`
- `description`
- `deck_id`
- `tags`
- `model_type`
- `front_template`
- `back_template`
- `notes_template`
- `extra_template`
- `placeholder_definitions`
- `created_at`
- `last_modified`
- `deleted`
- `client_id`
- `version`

`placeholder_definitions` should be structured JSON rather than inferred from raw template text.

Identity rules:

- `id` is the canonical stable identifier used by API routes and UI mutations
- `name` is user-editable data, not the resource locator
- `name` should remain unique per user among non-deleted flashcards templates

Example conceptual shape:

```json
{
  "name": "Vocabulary Definition",
  "deck_id": 42,
  "tags": ["vocab", "language"],
  "model_type": "basic",
  "front_template": "What does {{term}} mean?",
  "back_template": "{{definition}}\n\nExample: {{example}}",
  "notes_template": "Source: {{source}}",
  "extra_template": null,
  "placeholder_definitions": [
    {
      "key": "term",
      "label": "Term",
      "help_text": "Word or phrase being learned",
      "default_value": "",
      "required": true,
      "targets": ["front_template"]
    }
  ]
}
```

### Proposed deck schema extension

Add a deck field such as:

- `review_prompt_side`

This should be carried through:

- `DeckCreate`
- `DeckUpdate`
- `Deck`
- deck persistence in `ChaChaNotes_DB`
- UI deck create/edit flows

## API Design

Follow the existing flashcards endpoint structure with a dedicated templates resource:

- `POST /api/v1/flashcards/templates`
- `GET /api/v1/flashcards/templates`
- `GET /api/v1/flashcards/templates/{template_id}`
- `PATCH /api/v1/flashcards/templates/{template_id}`
- `DELETE /api/v1/flashcards/templates/{template_id}`

Behavioral expectations:

- optimistic locking on update/delete
- soft delete by default
- names validated for non-empty input
- names remain unique per user
- payload validated server-side
- rename is supported via `PATCH` without changing the resource identifier

Deck APIs should be extended so deck create/update/read includes `review_prompt_side`.

## UX Details

### Template picker and apply flow

The apply flow should be lightweight:

- searchable template picker
- small placeholder form
- resolved draft drops into the normal drawer

The user should remain in control of the final card draft. Applying a template is a shortcut, not an automatic submission path.

### Template library expectations

The Templates tab should support:

- clear list of templates
- create new template
- duplicate existing template
- edit placeholder definitions
- delete template

It does not need advanced version-tree UX in v1, even if backend metadata supports versioning or optimistic locking.

### Review orientation UI

Deck default belongs in deck settings.

Session override belongs near the review controls where users already choose:

- deck
- review mode
- cram options

The review card chrome should reflect the active prompt side clearly so `Front` and `Back` labels remain accurate from the user’s perspective.

## Validation and Edge Cases

### Templates

- unresolved required placeholders block application
- optional placeholders may fall back to defaults or empty strings
- missing default deck should not block application; warn and continue without applying that deck
- template application never auto-creates a flashcard
- template payload validation should reject malformed placeholder definitions
- template save/update validation should cross-check scaffold tokens and definitions:
  - every `{{token}}` referenced in scaffold text must have a matching placeholder definition
  - every placeholder definition must target at least one supported template field
  - every declared placeholder should be referenced in at least one targeted field so the fill step does not ask for dead inputs caused by typos or stale config

### Review orientation

- cloze cards always remain front-first
- `back-first` should work in both due review and cram review
- review history and analytics should not treat `back-first` as a different flashcard type
- temporary session override should not silently persist to deck settings

## Testing Strategy

### Backend

- flashcards template schemas
- flashcards template CRUD endpoints
- optimistic locking for template updates/deletes
- deck schema persistence for `review_prompt_side`
- template payload validation

### Shared UI and WebUI

- create drawer template apply flow
- create drawer save-as-template flow
- template library CRUD interactions
- deck settings editing for review orientation
- review tab front-first rendering
- review tab back-first rendering
- cloze cards ignoring deck back-first defaults
- session override precedence over deck default
- no scheduling/regression differences caused by orientation changes

### Extension

- flashcards tab routing for `?tab=templates`
- templates tab visibility when no decks exist
- create drawer template apply/save flows where shared UI is reused by the extension
- review orientation rendering regressions on extension flashcards study surfaces

### Regression focus

- existing create flow without templates
- existing `basic_reverse` and `cloze` card model behavior
- deck create/edit flows
- due review and cram review queue progression

## Terminology Cleanup

As part of this work:

- rename existing flashcards UI copy from `Card template` to `Card model` or `Card type`
- reserve `Template` for reusable authoring presets

This avoids user confusion and keeps the feature vocabulary coherent.

## Risks

- create drawer crowding if too many template controls are added inline
- placeholder-definition UX becoming too complex if v1 tries to become a full form builder
- accidental coupling of review orientation to scheduling if implementation touches the wrong abstractions

Mitigation:

- keep template application as a compact step
- keep the Templates tab as the primary management surface
- keep review orientation in presentation logic and deck configuration only

## Implementation Notes for Planning

- Reuse the existing flashcards shared UI package so web and extension behavior stay aligned.
- Follow the writing-template CRUD pattern where it fits, but do not force flashcards templates into generic settings storage.
- Treat `review_prompt_side` as a deck field, not scheduler settings JSON.
- Keep the implementation incremental:
  1. terminology cleanup and deck schema extension
  2. template data model and CRUD
  3. create-drawer apply/save flow
  4. templates management tab
  5. review orientation override UI and regressions
