# Character Chat Streaming Emote Directives PRD

Date: 2026-07-06
Status: Ready for spec review
Backlog: TASK-12163
Follow-up: TASK-12164

## Summary

Add explicit streaming emote control to character chat portraits.

Today, character mood is primarily inferred from assistant text after a response
finishes. That cannot reproduce the reference behavior: a character portrait
acts through a single streaming response, changing expression at dialogue beats.

V1 lets the assistant emit standalone control directives such as:

```text
Emote: annoyed
```

The character-chat frontend consumes those directives, updates the portrait as
the response streams, removes the directives from visible and persisted chat
text, and stores compact metadata for the final emote and future replay.

This is a character-chat portrait feature first. A shared Persona Visual runtime
integration is intentionally tracked as follow-up work in `TASK-12164`.

## Goals

- Let a character portrait change expression multiple times during one
  streaming assistant response.
- Keep raw `Emote:` directives out of rendered chat and persisted assistant
  content.
- Preserve clean assistant text for history, search, export, and replay.
- Reuse existing character mood image resolution and `mood_label` persistence
  where possible.
- Make explicit emotes win over heuristic mood detection.
- Parse non-streaming character responses too, so directives are still stripped
  when streaming is disabled.
- Store optional `emote_events` metadata for future replay or shared visual
  runtime integration.
- Fail gracefully when an emote state has no matching image asset.

## Non-Goals

- Do not build the shared Persona Visual runtime integration in this PR.
- Do not expose an agentic `set_emote` or `set_visual_state` tool in this PR.
- Do not add a database migration for emote events.
- Do not add fuzzy matching between arbitrary emote labels and available
  assets.
- Do not replay historic emote beats on message reload in V1.
- Do not replace the existing character mood image map format.
- Do not build a full animation or lip-sync system.

## Existing Context

Relevant current code paths:

- `apps/packages/ui/src/utils/character-mood.ts`
  - defines current mood labels
  - infers mood through `detectCharacterMood()`
  - resolves character mood images from card extensions, currently limited to
    built-in mood aliases
- `apps/packages/ui/src/components/Common/Playground/Message.tsx`
  - chooses explicit mood, inferred mood, and mood-specific portrait image
- `apps/packages/ui/src/hooks/chat/useChatActions.ts`
  - streams character chat responses
  - persists final assistant messages
  - currently runs final-text mood inference in character chat mode
- `apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts`
  - duplicate/older character chat mode flow with similar mood handling
- `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts`
  - reloads `mood_label`, `mood_confidence`, and `mood_topic`
- `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
  - already accepts `mood_label`, `mood_confidence`, and `mood_topic`
- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
  - stores mood metadata for non-streaming and persisted streaming messages
- Persona Visual runtime files already support safe named visual states, but
  that broader integration is deferred to `TASK-12164`.

The character editor currently does not expose full mood-image management,
while the parser already understands mood image extensions. This PR may rely on
existing character-card mood image data, but should not expand the character
asset editor unless required for the core flow.

## Recommended Approach

Use streaming text directives in character chat:

1. Add a small parser for standalone `Emote: <state>` lines.
2. Add a streaming line buffer that keeps only the current incomplete line.
3. Strip directives before text reaches rendered chat or persistence.
4. Fire portrait updates immediately for accepted directives.
5. Persist clean assistant content, final `mood_label`, and optional
   `emote_events` in existing metadata.

This approach is recommended because it matches the reference video and works
with ordinary chat streaming. It does not require provider-specific tool calling,
a database migration, or a new animation runtime. It does require small request
schema additions so the existing message metadata can carry `emote_events`.

## Alternatives Considered

### Final Emote Line Only

The model ends with one line such as `Emote: irritated`, the frontend strips it,
and the final `mood_label` is stored.

This is simpler, but it misses the core behavior from the reference video:
expression changes during one assistant message.

### Agentic Tool First

Expose a tool such as:

```json
{ "name": "set_emote", "arguments": { "state": "smug" } }
```

This is cleaner for future agentic UIs, but it pushes V1 into provider/tool
orchestration and still does not help plain chat models unless text directives
also exist.

### Shared Persona Visual Runtime First

Unify character portraits, Persona Buddy, and future agent surfaces around a
common visual-state contract before shipping character emotes.

This is the right long-term direction, but too broad for the first PR. It is
tracked as `TASK-12164`.

## Directive Contract

The only V1 directive grammar is a standalone line:

```text
Emote: <state>
```

Rules:

- Prefix matching is case-insensitive.
- Inline prose is not parsed. `She says Emote: smug` remains visible text.
- Directives inside fenced code blocks are ignored and remain visible code.
- State labels are trimmed, lowercased, and internal whitespace is converted to
  `-`.
- A normalized state is valid only when it matches
  `^[a-z0-9][a-z0-9_-]{0,39}$`.
- Empty, unsafe, or longer-than-40-character normalized state labels are
  invalid.
- Invalid standalone directives are still stripped, but do not fire or store an
  event.
- After the per-response event cap is reached, later standalone directives are
  still stripped, but do not fire or store new events.
- Consecutive duplicate states are stripped but ignored as events.

V1 cap: 5 accepted emote events per assistant response.

V1 accepts arbitrary safe state slugs. It does not restrict directives to the
current built-in classifier labels. Built-in mood labels remain the fallback
classifier taxonomy, not the full emote taxonomy.

These directives are app-level presentation controls. They are not a security
boundary. A user can prompt a model to emit them, and that should only affect
the character portrait state.

## Prompt Contract

Character-chat prompting should tell the model to emit `Emote: <state>` only
when the character expression should change. It should not emit an emote after
every sentence.

Example:

```text
Emote: annoyed
What the hell is that supposed to mean?

Emote: smug
I'm feeling great because I just saved us.

Emote: irritated
You should be thanking me.
```

The exact prompt wording can live near the existing character-chat prompt
assembly. The parser remains defensive because models will sometimes ignore or
misformat instructions.

## Streaming Data Flow

1. Character-chat stream starts.
2. Response chunks are appended to a parser buffer that holds only the current
   incomplete line.
3. Complete lines are classified as visible text, directives, or fenced code.
4. Visible text is emitted to the assistant message.
5. Accepted directives update the active portrait state immediately and append
   an event with an offset in sanitized visible text.
6. Invalid, duplicate, and over-cap directives are stripped without firing an
   event.
7. On stream end, the parser flushes the final unterminated line.
8. The frontend persists the sanitized assistant text, the last accepted emote
   as `mood_label`, and optional `emote_events` metadata.

The parser must normalize `\n`, `\r\n`, and final text without a trailing
newline. It must also track fenced-code state across chunks.

## Non-Streaming Data Flow

Non-streaming character responses use the same directive contract in
final-content mode.

The UI should never show raw directive lines. Persistence should receive only
sanitized content and emote metadata.

If the WebUI owns the non-streaming response before persistence, it should parse
and strip in the frontend before saving. If a backend endpoint persists a
non-streaming character response server-side, that backend path must run an
equivalent parser before saving and before returning `assistant_content`. V1 does
not need to simulate intermediate beat timing for a non-streaming response; the
portrait can update to the last accepted emote after parsing.

## Metadata Contract

Reuse `mood_label` as the final emote so existing message rendering and history
loading keep working.

The final emote is the last accepted emote event in sanitized response order. If
there are no accepted emote events, no explicit final emote exists and the
existing heuristic mood fallback may run.

Store optional event metadata under existing message metadata, for example:

```json
{
  "mood_label": "irritated",
  "emote_events": [
    { "at_char": 0, "state": "annoyed" },
    { "at_char": 94, "state": "smug" },
    { "at_char": 212, "state": "irritated" }
  ]
}
```

`at_char` is the JavaScript string offset in sanitized assistant text after
visible text emitted so far. It is for ordering and future best-effort replay,
not exact grapheme or animation timing.

Persistence path:

- Add optional `emote_events` support to the character stream persist request
  and include it in `_build_stream_persist_metadata_extra()`.
- For backend-persisted non-streaming character completions, store parser output
  directly in `metadata_extra`.
- Do not add a database migration. The existing message metadata storage remains
  the durable location.
- Frontend message objects keep carrying this through `metadataExtra`.

Metadata loading should be best-effort. Missing or malformed `emote_events`
must not block chat history rendering.

On history reload, V1 restores only the final emote through the existing
`mood_label` path. It does not replay beat events.

## Portrait Resolution

Use the existing character mood image extension locations, but broaden the
frontend mood image parser/resolver to support safe custom state slugs instead
of only built-in `CharacterMoodLabel` values.

The implementation should keep `detectCharacterMood()` constrained to the
current built-in labels. Only the image map and explicit emote resolver need to
accept arbitrary safe slugs.

If an exact matching asset exists for the normalized state slug, show it. If
not, keep the current portrait or base character image. Do not add fuzzy
matching in V1.

Explicit emote directives override `detectCharacterMood()`. The heuristic
classifier runs only when no valid explicit directive exists for the assistant
message.

## Error Handling

- Malformed directives are stripped and ignored.
- Unsafe state labels are stripped and ignored.
- Duplicate consecutive states are stripped and ignored.
- Over-cap directives are stripped and ignored.
- Asset misses keep the current/base portrait.
- Malformed metadata on load is ignored.
- Parser errors should fall back to visible sanitized text rather than breaking
  chat rendering.

The UI should not log full assistant content while reporting parser errors.

## Testing Requirements

### Parser Unit Tests

- Strips standalone directives and returns clean text.
- Ignores inline `Emote:` prose.
- Ignores directives inside fenced code blocks.
- Tracks fenced code state across split streaming chunks.
- Normalizes safe state labels.
- Rejects labels that are empty, unsafe, or longer than 40 characters after
  normalization.
- Strips invalid standalone directives without returning events.
- Drops duplicate consecutive states.
- Strips but ignores events after the cap.
- Handles `\n`, `\r\n`, and final text without trailing newline.
- Records `at_char` against sanitized visible text.
- Uses the exact 40-character slug limit and 5-event cap.

### Streaming Buffer Tests

- A directive split across chunks never appears in visible output.
- Normal text split across chunks is preserved.
- The buffer holds only the current incomplete line.
- The final unterminated line flushes correctly on stream completion.

### Character Chat Integration Tests

- Explicit final emote persists as `mood_label`.
- `emote_events` persists under metadata.
- The last accepted emote event becomes `mood_label`.
- `detectCharacterMood()` is bypassed when at least one valid directive exists.
- Non-streaming character responses are parsed and stripped before
  display/persist.
- History reload restores the final emote without replaying events.

### UI Behavior Test

One minimal browser or component test should stream chunks containing an emote
directive, then assert:

- the portrait changes when a matching mood image exists
- raw `Emote:` text never appears in rendered chat

## Acceptance Criteria

- Streaming character chat can change the character portrait multiple times
  within one assistant response when valid `Emote:` directives arrive.
- Raw `Emote:` directive lines never appear in rendered chat or persisted
  assistant content, including partial/chunked streaming cases.
- Explicit emote directives override heuristic mood detection.
- `detectCharacterMood()` only runs when no valid directive exists.
- Non-streaming character responses are also parsed and stripped.
- Invalid, unsafe, duplicate consecutive, and over-cap directives are stripped
  but do not fire or store emote events.
- Missing emote image assets do not break rendering.
- Final emote, defined as the last accepted event, persists as `mood_label`.
- Optional `emote_events` persist in metadata.
- History reload restores the final emote and does not replay beats.
- Parser, streaming-buffer, integration, and minimal UI behavior tests cover the
  directive flow.

## Follow-Up: Shared Persona Visual Runtime

`TASK-12164` should evaluate the broader design after this V1 lands.

Follow-up scope:

- Review existing Persona Visual runtime state handling and character mood image
  resolution for a shared boundary.
- Define how character emote states map to Persona Visual runtime/custom states.
- Decide whether a future agentic control should be `set_emote` or
  `set_visual_state`.
- Define how agentic tool control coexists with text directives.

This follow-up should be planned separately. It should not block the V1
character-chat portrait behavior.
