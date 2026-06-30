# Writing Playground Document-First Revisions Design

Date: 2026-05-22
Backlog: TASK-443
Status: Approved for planning; design hardening review applied

## Purpose

The Writing Playground should become a document-first creative drafting workspace for fiction,
essays, scripts, blog posts, and other long-form writing. The current page already has a large
set of writing controls, but the core loop still behaves mostly like prompt generation into an
editor plus separate analysis panels. The approved direction is to make AI assistance feel like
reviewable document work:

1. Write in the document.
2. Ask the assistant for a concrete change.
3. Review a proposed edit.
4. Apply, reject, copy, or regenerate it.

Comments and annotations are valuable, but they are a later phase. The first slice should make
proposed text changes trustworthy before adding persistent marginal commentary.

## Current Evidence

The design is grounded in the current shared WebUI and extension surface:

- WebUI route wrapper: `apps/packages/ui/src/routes/option-writing-playground.tsx`.
- Next page wrapper: `apps/tldw-frontend/pages/writing-playground.tsx`.
- Extension route wrapper: `apps/tldw-frontend/extension/routes/option-writing-playground.tsx`.
- Extension registry path: `apps/tldw-frontend/extension/routes/route-registry.tsx`.
- Shared component root: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`.
- Existing shell/sidebar component: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundShell.tsx`.
- Existing editor panel: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundEditorPanel.tsx`.
- Existing inspector tabs: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundInspectorPanel.tsx`.
- Existing rich editor: `apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx`.
- Existing writing store: `apps/packages/ui/src/store/writing-playground.tsx`.
- Existing service layer: `apps/packages/ui/src/services/writing-playground.ts`.
- Prior PRD: `apps/extension/docs/Product/WIP/Writing-Playground-PRD.md`.

The current Writing Playground already includes:

- session and manuscript navigation
- plain and rich editor modes
- edit, preview, and split editor views
- generation, stop, undo, redo, search, replace, and read-aloud actions
- sampling, context, setup, analysis, characters, research, agent, and feedback tabs
- writing capabilities gating
- server-backed sessions, templates, themes, and snapshots where supported
- WebUI and extension parity tests around route and writing mode behavior

This means the improvement should extend the existing shared component tree instead of creating a
new route or separate extension-only experience.

## Product Direction

### Primary User Goal

The user wants help drafting and revising long-form creative work while staying in control of the
document. The page should support broad creative writing, not only manuscript fiction.

Examples:

- continue this essay in the same voice
- rewrite this passage to be less generic
- expand this paragraph with sensory detail
- tighten this scene without losing tone
- turn this rough outline into a blog post section
- make this script beat more conversational
- revise this selected section for clarity

### Product Model

The editor is the primary workspace. AI actions are document operations, not just chat messages.

The assistant can still use the existing context, model, template, character, research, and
analysis infrastructure, but the visible output of common creative-writing actions should be a
reviewable proposal against the current document.

### Non-Goals

- Do not build a separate Writing page.
- Do not replace the existing `WritingPlayground` route with a chat-first clone.
- Do not implement Google Docs style collaboration.
- Do not require full rich-text operational transforms in the first slice.
- Do not add persistent annotations or comment threads in the first slice.
- Do not remove existing power-user controls for sampling, context, templates, themes, tokens, or
  manuscript tools.
- Do not introduce backend revision-history APIs until the client workflow proves useful.

## Approaches Considered

### 1. Document-first revision workflow

Keep the editor as the center of gravity. Add compact writing actions and a revision queue that
shows proposed edits with diffs and apply or reject controls.

Strengths:

- fits broad long-form creative writing
- uses the existing shared route and editor infrastructure
- makes AI changes reviewable instead of silently mutating the document
- keeps comments and annotations as a clean later extension

Weaknesses:

- requires careful range handling and drift detection
- needs a proposal response contract around a probabilistic model output

### 2. Chat-led writing assistant

Make conversation the main control surface, with the document beside it like a side-by-side chat
and document tool.

Strengths:

- resembles the reference image closely
- makes tool calls and reasoning visible
- could reuse chat-style interaction patterns

Weaknesses:

- risks making writing feel like chatting around a document instead of writing in it
- can bury the draft behind assistant transcript noise
- duplicates capabilities that already belong in the Writing Playground

### 3. Mode-based writing studio

Add explicit modes such as Draft, Revise, Structure, Research, and Polish, each with tailored
controls and panels.

Strengths:

- powerful mental model for advanced users
- can organize the existing broad feature set

Weaknesses:

- too broad for the first slice
- risks more navigation and mode confusion before the edit loop is validated

## Recommended Approach

Use approach 1: document-first revision workflow.

The first implementation plan should extend the existing shared `WritingPlayground` with a small
AI action bar, a proposed-edit generation adapter, a revision queue, and conflict-safe apply logic.
The feature should work in both WebUI and extension routes because both consume the shared UI
component.

## Target UX

### Editor-First Layout

The default visible state should feel like a serious writing desk:

- The editor and current document are central.
- Sessions and manuscript navigation stay available on the left.
- Sampling, context, setup, analysis, characters, research, agent, and feedback stay available in
  the inspector, but should not be required for the basic drafting loop.
- The editor should remain useful when the inspector is closed.

### AI Action Bar

Add a compact action bar near the editor. Initial actions:

- Continue
- Rewrite
- Expand
- Tighten
- Tone
- Outline
- Custom

Tone should prompt for a requested direction such as warmer, sharper, more formal, more playful,
or preserve voice. Outline should produce a structured outline or next-section plan by default,
not replace the draft unless the user explicitly asks to apply outline text.

Targeting rules:

- If text is selected, target the selection.
- If no text is selected, Continue targets the insertion point or end of the draft.
- If no text is selected, Rewrite, Expand, Tighten, and Tone target the current paragraph when
  the editor can resolve it.
- If the current paragraph cannot be resolved, fall back to whole-document targeting only after
  clear user confirmation for destructive or large edits.
- Outline can target the whole document by default.
- Custom should show the resolved target before the request is sent.

The action bar should not expose all prompt engineering details. It should translate a writing
intent into a structured request while still honoring the existing model, template, context, and
generation settings.

Important boundary: the first implementation should distinguish text-changing actions from
planning or advisory actions. Continue, Rewrite, Expand, Tighten, Tone, and Custom can create
replacement or insertion proposals. Outline should default to a non-mutating advisory proposal
unless the user explicitly asks to insert or replace text with the outline.

### Revision Queue

Add a queue for AI proposals. It can live below the editor at first or inside a new inspector tab
named `Revisions`. The implementation plan should choose the least disruptive placement based on
current layout constraints.

Each proposal should show:

- action type
- operation type
- instruction
- target summary
- short rationale when available
- before and after preview
- diff view
- status: pending, applied, rejected, conflict, raw suggestion, or advisory
- actions: Apply, Reject, Copy, Regenerate

Applied proposals should remain visible briefly enough to support user confidence and undo. The
existing generation undo stack can remain the first undo mechanism if it can cover applied proposal
changes safely.

Advisory proposals should not show Apply unless the user converts them into an insertion or
replacement request. This prevents Outline or critique-like actions from being treated as document
patches by accident.

### Document-Aware Status

Improve the status bar around writing state:

- word count
- selected word count when text is selected
- save state
- active model
- pending revisions count
- generation state

Token counts, logprobs, word clouds, and tokenizer details belong in Analysis, not the default
writing status line.

### Workflow Presets

Add a small set of creative-writing presets that shape instructions:

- Draft freely
- Polish prose
- Developmental edit
- Preserve voice
- Make concise
- Expand sensory detail

These presets should map to visible instruction text or templates. They should not become hidden
magic settings that the user cannot inspect or override.

## Architecture

### Component Boundaries

The implementation should add small, testable units instead of growing the already large
`WritingPlayground/index.tsx`.

Suggested units:

- `WritingActionBar`: renders actions, custom instruction input, and targeting summary.
- `writing-revision-types.ts`: defines proposal, target, status, and action types.
- `writing-revision-utils.ts`: range resolution, word counts, drift checks, and apply planning.
- `writing-revision-prompt-utils.ts`: builds structured prompts for proposed edits.
- `WritingRevisionQueue`: renders proposal cards and queue actions.
- `WritingRevisionDiff`: renders before/after and inline or block-level text diff.
- `useWritingRevisions`: owns local proposal state, session-payload sync, and apply/reject handlers.

The existing `WritingPlayground` root should orchestrate these units but not own all internal
logic directly.

### Draft Editor

The current plain and rich editor modes remain. The editor adapter introduced by earlier Writing
work should be the boundary for selection and range behavior:

- get selected text
- get selected range
- focus editor
- set selection
- apply replacement where safe

Plain text remains the canonical generation and diff input. Rich editor mode can initially derive
plain text for proposals and apply text replacements when the target still matches.

### Generation Adapter

The proposed-edit flow should wrap the existing chat-completions service rather than introduce a
new provider path.

Request inputs:

- full document text
- target range and target text
- action type
- user instruction
- writing preset if selected
- active model and provider state
- existing context, template, world info, author note, and memory settings

Expected model output:

```json
{
  "title": "Short proposal title",
  "replacement": "The proposed replacement text",
  "rationale": "Brief reason for the change",
  "notes": ["Optional concise notes"]
}
```

For text-changing proposals, `title` and `notes` are optional display fields. The proposal can be
valid with only `replacement` and an optional `rationale`; missing display fields should not block
the user from reviewing the proposed text. Advisory and raw-suggestion proposals may use `rawText`,
`rationale`, `title`, and `notes` without a `replacement`. The first implementation can request
JSON through prompt discipline and client validation. If the server later exposes structured output
support for the selected provider, the adapter can use it behind the same client boundary.

Proposal generation should not stream partial edits into the editor. For structured proposed-edit
requests, default to non-streaming generation, or accumulate the stream invisibly and validate only
after the complete response is available. Partial JSON or partial replacement text must never
appear as an applyable proposal.

### Proposal Shape

Client-side proposal shape:

```ts
type WritingRevisionProposal = {
  id: string
  sessionId: string
  action: "continue" | "rewrite" | "expand" | "tighten" | "tone" | "outline" | "custom"
  operation: "insert" | "replace" | "advisory"
  instruction: string
  target: {
    mode: "selection" | "paragraph" | "cursor" | "document"
    start: number
    end: number
    beforeText: string
    anchor: {
      documentFingerprint: string
      prefix: string
      suffix: string
    }
  }
  replacementText?: string
  rawText?: string
  rationale?: string
  title?: string
  notes?: string[]
  regeneratedFromId?: string
  createdAt: string
  status: "pending" | "applied" | "rejected" | "conflict" | "raw_suggestion" | "advisory"
}
```

Continue can use a zero-length target at the cursor or document end. The queue should still render
it as an insertion proposal. Because zero-length targets cannot rely on `beforeText` matching, the
proposal must store an insertion anchor:

- `documentFingerprint`: a stable hash or equivalent fingerprint of the document text at proposal
  creation time
- `prefix`: a bounded text window before the insertion point
- `suffix`: a bounded text window after the insertion point

The first implementation can compute the fingerprint client-side from the canonical plain text. It
does not need a backend document-version API.

Replacement and insertion proposals require `replacementText`. Advisory and raw-suggestion
proposals can omit `replacementText` and instead use `rawText`, `rationale`, `title`, and `notes`
for display-only review.

### Persistence

Initial persistence should be session-payload based:

- Store pending, applied, rejected, conflict, and raw-suggestion proposals in the active writing
  session payload under a schema-versioned revisions field.
- The canonical document remains the existing prompt/editor text.
- Applying a proposal updates the canonical document through the same dirty-state and autosave path
  as direct edits.
- Proposal history can be pruned by count or age in the client if payload size becomes an issue.

This avoids a new backend revision-history API in the first slice while preserving refresh safety.

The implementation must not assume the current dirty-state helper already tracks proposal-only
changes. Today the Writing session save path primarily compares prompt, rich prompt, settings,
template, theme, and chat-mode state. The revision feature needs one of these explicit save
contracts:

- extend the session dirty baseline and save helper to include the schema-versioned revisions
  payload, or
- add a narrow proposal-save helper that merges revisions with the latest pending editor payload and
  uses the same expected-version conflict handling.

Either way, proposal-only changes must persist, and proposal saves must not overwrite unsaved prompt
or settings edits by merging against a stale `activeSessionDetail.payload`.

## Data Flow

1. User writes or selects text in the editor.
2. User chooses an action or enters a custom instruction.
3. The UI resolves a target:
   - selection
   - current paragraph
   - cursor insertion
   - whole document
4. The UI builds a proposed-edit request from the document, target, action, instruction, selected
   model, and existing writing context.
5. The adapter sends the request through the existing chat-completions path.
6. The client validates the response into a `WritingRevisionProposal`.
7. The queue renders the proposal and diff.
8. Apply checks that the current target still matches `beforeText`, or validates the insertion
   anchor for zero-length targets.
9. If the target matches and the proposal operation is `replace` or `insert`, apply replaces or
   inserts text and marks the proposal applied.
10. If the target drifted, mark the proposal conflict and offer copy or manual apply.
11. If the proposal operation is `advisory`, do not mutate the document; allow copy or a follow-up
    request that turns it into an insertion or replacement proposal.
12. Reject marks the proposal rejected without changing the document.
13. Regenerate creates a replacement proposal linked by `regeneratedFromId`, instruction, and
    target.

## Error Handling

### Model Output Cannot Be Parsed

If the model response is not valid structured output, create a `raw_suggestion` proposal. Show the
raw text, allow copy, and do not offer automatic apply unless a safe replacement can be derived.

### Target Drift

Before applying, compare the current document slice at the saved range with `beforeText`.

- If it matches, apply.
- If it does not match, attempt a conservative exact-text search for `beforeText`.
- If there is exactly one match, offer to retarget before apply.
- If there are zero or multiple matches, mark conflict and offer copy/manual apply.

Never silently overwrite text when the target has drifted.

### Insertion Anchor Drift

Continue and other insertion proposals have `start === end` and `beforeText === ""`, so they need
separate validation:

- If the current document fingerprint is unchanged, apply at the saved insertion offset.
- If the fingerprint changed, search for the saved `prefix` and `suffix` as a local insertion
  anchor.
- If the prefix and suffix identify exactly one insertion point, offer to retarget before apply.
- If the anchor cannot be found or is ambiguous, mark conflict and offer copy/manual apply.

Never treat an empty `beforeText` match as sufficient validation for insertion proposals.

### Missing Selection

If an action expects a selected passage but none exists:

- use current paragraph when available
- otherwise show a target confirmation state before whole-document edits

### Rich Editor Apply Limitation

If rich editor mode cannot safely apply a range replacement while preserving structure, the proposal
should degrade to copy/manual apply. This is acceptable for the first slice; correctness is more
important than pretending rich-text patching is solved.

### Backend Or Model Unavailable

Reuse existing offline, unsupported, and model-missing states. The action bar should disable
generation actions with the same reasons as the existing Generate button.

## Testing Strategy

Unit coverage:

- range resolution for selected text, cursor insertion, current paragraph, and whole document
- word-count and selected-word-count helpers
- proposal response validation
- non-streaming or complete-response validation for structured proposed edits
- raw-suggestion fallback
- advisory proposal behavior
- exact-match apply
- zero-length insertion anchor validation
- drift conflict
- unique exact-text retargeting
- unique insertion-anchor retargeting
- ambiguous exact-text retargeting
- ambiguous insertion-anchor retargeting
- session-payload serialization and pruning

Component coverage:

- action bar renders disabled/enabled states correctly
- action selection creates a pending proposal
- revision queue renders pending, applied, rejected, conflict, raw-suggestion, and advisory states
- Apply and Reject update UI state
- advisory proposals do not render Apply until converted into a text-changing request
- conflict state does not mutate editor text

Integration and parity coverage:

- WebUI route still renders `/writing-playground`.
- Extension route still renders `#/writing-playground`.
- Plain editor proposal apply works.
- Rich editor proposal generation works and either applies safely or falls back honestly.
- Responsive layouts remain usable with the revision queue visible.

Manual/browser verification:

- create a session
- draft a short passage
- select a paragraph
- request Rewrite or Tighten
- inspect diff
- reject one proposal
- apply one proposal
- edit the target before applying and confirm conflict handling
- verify the same core flow in the extension options route

Bandit:

- Not applicable for a documentation-only design task.
- If a later implementation touches backend Python, run Bandit on touched backend paths.

## Rollout Plan

### Stage 1: Client Proposal Loop

Add action bar, proposal generation adapter, queue state, diff preview, and conflict-safe apply for
plain text.

Success criteria:

- user can generate a proposed edit
- user can apply or reject it
- target drift is detected
- proposal state survives session refresh through payload persistence

Stage 1 should be planned as small implementation commits rather than one large patch:

1. revision types and pure utilities
2. proposal validation and prompt-building utilities
3. plain-text apply/conflict tests
4. action bar and queue UI
5. session-payload persistence integration
6. WebUI/extension parity and responsive verification

### Stage 2: Rich Editor And Layout Hardening

Improve TipTap range handling and decide whether the queue belongs below the editor, in the
inspector, or in a responsive hybrid layout.

Success criteria:

- rich editor mode does not misrepresent unsupported apply operations
- queue remains usable on WebUI desktop, narrow WebUI, and extension options layout

### Stage 3: Presets And Writing Status

Add creative-writing presets and document-aware status bar improvements.

Success criteria:

- presets produce inspectable instructions
- status bar reports writing-focused state without displacing analysis tools

### Stage 4: Comment And Annotation Design

Design comments/annotations after the proposed-edit workflow is stable.

Candidate actions:

- explain why
- mark weak spots
- flag inconsistency
- reader reaction
- style note

This stage should reuse revision targets and proposal metadata but should not be included in the
first implementation plan unless explicitly re-scoped.

## Open Questions For Implementation Planning

- Should the first queue placement be below the editor or in a new inspector `Revisions` tab?
- How many proposals should be persisted per session before pruning?
- Should Regenerate replace the existing pending proposal or append a linked alternative?
- Should whole-document edits require confirmation every time or only for destructive actions?
- How much of the existing `WritingPlayground/index.tsx` should be extracted before adding the new
  units?

## Definition Of Done For This Design

- Spec captures the document-first creative drafting workflow.
- Spec preserves WebUI and extension shared-surface parity.
- Spec defines proposed edits, revision queue, data flow, conflict handling, and testing.
- Spec scopes comments and annotations to a later phase.
- Spec review loop has no blocking findings.
