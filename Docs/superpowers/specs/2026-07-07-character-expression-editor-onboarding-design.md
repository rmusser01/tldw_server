# Character Expression Editor And Onboarding PRD

Date: 2026-07-07
Status: Ready for spec review
Backlog: TASK-12167
Related: TASK-12164

## Summary

Add a real configuration and discovery surface for character expression images
used by the existing explicit `Emote: <state>` character-chat directive feature.

The first slice belongs in the shared character editor. It replaces the current
"Mood images coming soon" placeholder with compact expression rows, starter
states, per-state image controls, and a small preview/test surface. Character
chat gets a dismissible nudge only when the selected character has no expression
images. The browser extension links users to the shared editor instead of
duplicating the editor in the sidepanel.

This keeps the implementation useful and small while leaving the broader
Persona Visual runtime unification to `TASK-12164`.

## Goals

- Let new users discover where character expression images are configured.
- Let users add, edit, remove, preview, and test expression images from the
  shared character editor.
- Reuse existing avatar image input patterns for URL, upload, and generation.
- Store expression images in the existing character metadata format consumed by
  the emote directive feature.
- Preserve unrelated character extension metadata.
- Keep WebUI and browser extension behavior aligned through shared UI and links.
- Avoid adding a second storage model before the Persona Visual follow-up.

## Non-Goals

- Do not build a shared `set_emote` or `set_visual_state` tool in this slice.
- Do not implement Persona Visual runtime unification in this slice.
- Do not build a chat-side setup wizard.
- Do not add bulk import.
- Do not add a backend database migration.
- Do not replace the existing `Emote: <state>` directive parser.
- Do not duplicate the full editor inside the browser extension sidepanel.

## Existing Context

The merged emote directive feature lets character-chat responses emit standalone
lines such as:

```text
Emote: thinking
```

The frontend strips those lines from visible and stored assistant text, updates
the character portrait during streaming, and stores the final state in existing
mood metadata. The backend prompt helper reads available states from character
metadata and uses the same slug rule:

```text
^[a-z0-9][a-z0-9_-]{0,39}$
```

The shared UI already contains the relevant editor and image patterns:

- `apps/packages/ui/src/components/Option/Characters/CharacterEditorForm.tsx`
  contains the character editor and currently shows a "Mood images (coming
  soon)" placeholder.
- `apps/packages/ui/src/components/Option/Characters/AvatarField.tsx` already
  handles URL, upload, generation, size limits, and MIME validation for avatar
  images.
- `apps/packages/ui/src/components/Option/Characters/utils.ts` already parses
  and merges character extension metadata.
- `apps/packages/ui/src/utils/character-mood.ts` reads mood image maps from
  current and legacy extension locations.

The WebUI and extension both consume shared package UI, so the editor should be
implemented once in shared UI.

## Recommended Approach

Build a shared **Expression Images** section inside the existing character
editor.

The section should expose compact expression rows, not a full repeated avatar
field per row. It should reuse the avatar field's source behavior and validation
rules, but render controls in a denser row layout suitable for several states.

Starter rows are shown for:

- `neutral`
- `happy`
- `sad`
- `angry`
- `thinking`
- `surprised`

Empty starter rows are suggestions only and are not saved. Users can add custom
rows for any valid state slug.

This approach is recommended because it configures the feature where character
metadata already lives, avoids chat/editor duplication, and keeps the future
Persona Visual unification as a migration path instead of a prerequisite.

## Alternatives Considered

### Chat-First Setup Wizard

Show a setup wizard from character chat whenever a character has no expression
images.

This is highly discoverable, but it makes chat responsible for editing character
assets and would duplicate validation, image handling, and metadata merge logic.

### Persona Visual-First Unification

Implement the shared visual state runtime before adding an editor for character
emotes.

This is the right long-term direction, but it is broader than the immediate
configuration problem. `TASK-12164` should handle that follow-up.

## Editor UX

Replace the placeholder in the character editor's metadata/advanced area with an
**Expression Images** section.

Each expression row contains:

- state slug input
- compact image source control for URL, upload, and generation
- thumbnail preview
- row-level validation or warning text
- remove button

The section also includes:

- `Add expression` action for custom states
- preview picker for configured states
- portrait preview that swaps to the selected expression image
- copy action for the selected preview state, such as `Emote: thinking`

The preview falls back to the base avatar when the selected state has no image
or the image fails to load. The copy action is enabled only when a valid preview
state is selected.

## Data Contract

The editor reads current and legacy mood image locations:

- `extensions.tldw.mood_images`
- `extensions.tldw.moodImages`
- top-level `mood_images`
- top-level `moodImages`

When saving, the editor writes canonically to:

```json
{
  "extensions": {
    "tldw": {
      "mood_images": {
        "happy": "data:image/png;base64,...",
        "thinking": "https://example.test/thinking.png"
      }
    }
  }
}
```

Unrelated `extensions` data must be preserved.

If more than one mood image location exists, canonical
`extensions.tldw.mood_images` wins. This gives deterministic loading and avoids
surprising overwrites from stale legacy data. Precedence is whole-map
precedence: when canonical mood images exist, legacy maps are ignored rather
than merged per state.

Saving writes the current editor rows to `extensions.tldw.mood_images` and
removes legacy aliases for the same data:

- `extensions.tldw.moodImages`
- top-level `mood_images`
- top-level `moodImages`

If no configured expressions remain, saving removes the mood image keys instead
of writing an empty object. Nudge detection should treat missing or empty mood
image maps as no configured expression images.

## Validation

Expression state slugs must match the backend rule:

```text
^[a-z0-9][a-z0-9_-]{0,39}$
```

Rules:

- Empty starter rows are not saved.
- Empty custom rows are invalid once created.
- Custom rows with a valid state but no image are incomplete and block save
  until the user adds an image or removes the row.
- Duplicate states are blocked.
- Renaming a state moves the image value to the new key and does not leave a
  stale key behind.
- Upload size, MIME validation, and supported image value checks match the
  existing avatar field.
- Remote image preview load failures do not block save; they show a warning and
  the preview falls back to the base avatar.
- Invalid metadata JSON keeps the existing editor behavior and blocks expression
  merges until the metadata is fixed.

## Onboarding And Discovery

Character chat shows a small dismissible nudge when the selected character has
no configured expression images.

The nudge:

- appears near the character/portrait area
- links to the character editor and opens or anchors the Expression Images
  section when routing supports it
- falls back to the Characters page if direct edit routing is unavailable
- is dismissible per server/user/character scope when that context is available

Dismissal keys should include every available stable scope value in this order:
server, user, character id. If no stable character identifier exists, the nudge
can be dismissed only for the current page session instead of using a broad
global key.

Nudge dismissal is a client-side UI preference in V1. Do not add backend
persistence for it.

The browser extension should expose an **Edit character expressions** entry that
opens the shared WebUI editor route or the Characters fallback. It should not
embed the editor in the extension sidepanel.

No modal, wizard, or repeated prompt is needed.

## Error Handling

- Bad state slugs show inline row validation and block save.
- Duplicate states show inline row validation and block save.
- Bad upload or generation results show row-level errors and preserve the prior
  row value until replacement succeeds.
- Per-row generation has loading and error state so users cannot accidentally
  overwrite another row mid-request.
- Broken preview images fall back to the base avatar.
- Clipboard copy shows visible success or failure because clipboard APIs can
  fail in browser and extension contexts.
- Invalid metadata JSON shows one clear message telling the user to fix metadata
  before expression changes can be saved.

## Accessibility

- State inputs, image actions, remove actions, preview picker, and copy action
  have accessible labels.
- Icon-only controls have tooltips and `aria-label` values.
- Keyboard users can add, edit, remove, preview, and copy expressions.
- Thumbnail previews have useful alt text, such as `Happy expression preview`.
- Validation must be visible inline, not hidden behind hover-only UI.

## Testing

Focused tests should cover the risky behavior rather than retesting existing
avatar internals.

Metadata helper tests:

- read all supported mood image locations
- apply deterministic precedence when multiple locations exist
- save only non-empty rows to `extensions.tldw.mood_images`
- remove legacy mood image aliases when saving canonical expression rows
- remove mood image keys instead of writing an empty object when no configured
  expressions remain
- preserve unrelated extension data
- reject duplicate and invalid states
- preserve arbitrary valid custom state slugs such as `smirk`
- rename state without leaving stale keys

Editor/component tests:

- starter rows render
- custom row can be added and removed
- URL/upload/generate source behavior updates only the target row
- create/edit round trip preserves expression rows after save and reload
- preview picker swaps image and handles image `onError` fallback
- copy selected preview state reports success and failure
- invalid metadata blocks expression merge with a clear message
- role/name queries cover the main controls to catch missing labels

Chat UI tests:

- setup nudge appears only when the selected character has no expression images
- dismissal persists for the server/user/character scope
- dismissal key builder does not hide nudges for other users, servers, or
  characters

Extension test:

- **Edit character expressions** opens the WebUI editor route or the Characters
  fallback.

Backend tests are not required unless implementation changes backend validation
or directive parsing.

## Implementation Boundaries

The implementation should stay close to existing files:

- Add a small shared expression image editor component under the existing
  character editor area.
- Add or extend character metadata helpers near existing character editor utils.
- Reuse existing image validation and generation behavior from avatar editing.
- Add the chat nudge near the character portrait/header code path.
- Add only the extension link or route handoff needed to reach the shared
  editor.

Avoid new dependencies, a new asset storage abstraction, or a parallel
expression runtime.

## Open Risks

- Large base64 expression images can bloat character metadata. V1 mitigates this
  by matching the avatar upload limits instead of inventing new limits.
- Direct edit deep links may not exist for every host. V1 must have a Characters
  page fallback.
- Generation APIs may fail or be disabled. V1 treats generation as one source
  option, not a required path.
- The future Persona Visual bridge may change the long-term storage contract.
  V1 keeps all writes in the existing `extensions.tldw.mood_images` location so
  migration remains straightforward.

## Success Criteria

- A user can configure expression images without editing raw JSON.
- A new user can discover the feature from character chat.
- The extension routes users to the same editor instead of maintaining a second
  editor.
- Existing character extension metadata survives expression edits.
- Empty starter rows do not pollute saved metadata.
- Configured expression states are available to the existing emote directive
  prompt/runtime path.
