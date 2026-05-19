# Character Chat Terminology Taxonomy

Date: 2026-05-09

Purpose: keep character-chat labels consistent at decision points without rebranding internal modules or API terms.

## User-Facing Terms

| Term | Meaning in the WebUI | Use When | Avoid |
| --- | --- | --- | --- |
| Character | A reusable speaking identity with character fields, prompt, greeting, and optional metadata. | The user is choosing, creating, importing, editing, favoriting, or chatting as a saved character. | Do not call characters personas. |
| Character chat | A conversation using a selected character. | The user is starting, resuming, blocking, or retrying a character-backed chat. | Do not call it generic chat when the selected character matters. |
| Scene | Optional context layered onto a character chat. | The user can add mood, setting, goals, notes, or roleplay framing after a character is selected. | Do not expose Actor as the primary runtime label. |
| Persona | A persistent behavior/profile concept distinct from character cards. | The user is in Persona Garden or choosing a profile separate from a saved character. | Do not use Persona as a synonym for Character. |
| Assistant | The generic AI identity when no character is selected. | The user is in a model/chat flow without a selected character. | Do not use Assistant as the picker label when the picker contains characters and personas. |
| Companion Home | The broader home/workspace surface. | The user is on the dashboard or shell-level home surface. | Do not use Companion as a chat identity. |

## Label Rules

- Entry points for character-backed conversations should say `Character chat`.
- Pickers containing both characters and personas should say `Select character or persona`.
- Search controls in that picker should say `Search characters and personas`.
- Scene controls should be framed as optional context, not as a prerequisite before character selection.
- Runtime UI should use short labels. Longer explanations belong in settings, docs, or audits.

## Review Checklist

- Does the label describe the user's decision rather than the implementation object?
- If the user is choosing a saved character, does the UI say `Character`?
- If the user is choosing a persistent profile, does the UI say `Persona`?
- If the user is adding scene/context metadata, does the UI make that optional?
- Does the change preserve existing API names and routes unless the user-facing label is the source of confusion?
