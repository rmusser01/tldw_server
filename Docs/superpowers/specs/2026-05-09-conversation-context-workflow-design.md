# Conversation Context Workflow Design

Date: 2026-05-09
Status: Approved brainstorming design, pending implementation planning
Backlog: TASK-185

## Purpose

Design a conversation-first workflow for first-time and power users who want to
chat with reusable context assets. The design corrects a key framing risk from
the character-card audit: worldbooks and chat dictionaries are not
character-exclusive features. They are reusable conversation context assets that
can apply to blank chats, character chats, workspace chats, and other
non-character-focused conversations.

This is a design artifact only. Implementation sequencing, file ownership, API
changes, and tests belong in a later implementation plan.

## Evidence Base

This design is grounded in
`Docs/Reviews/CHARACTER_CARD_WORLDBOOK_DICTIONARY_UX_AUDIT_2026_05_09.md`.
The audit found:

- Worldbook attachment and prompt-preview injection work for character-focused
  flows.
- Chat dictionaries process terms reliably for chat sessions, including global
  and workspace-scoped sessions.
- The UI risks implying that lore and dictionaries are character-owned because
  worldbooks appear on character cards while dictionaries are managed through
  chat assignment surfaces.
- Prompt preview exposes worldbook diagnostics but not equivalent dictionary
  diagnostics.
- Workspace-scoped context exists at the API level, but Workspace Playground
  does not make worldbooks or dictionaries visible and manageable.
- Missing or invalid workspace/provider state can produce confusing blockers,
  including a prior invalid workspace path that returned a server error instead
  of a recoverable setup error.

## Product Model

The parent concept is **Conversation Context**.

A chat session is the runtime container where context is assembled. A
conversation may include:

- A character card, if the conversation is character-focused.
- One or more worldbooks, whether or not a character is present.
- One or more chat dictionaries, whether or not a character is present.
- Workspace scope, if the conversation belongs to a workspace.
- Provider and model settings.
- Prompt preview and diagnostics showing what context will be injected,
  transformed, skipped, or blocked.

Character chat is a specialized workflow on top of this broader context system.
Characters can prefill the character slot and recommend related context, but
they do not own worldbooks or dictionaries. Worldbooks and dictionaries remain
reusable assets that can be attached to conversations, workspaces, global chat
defaults, or character-started chat shortcuts.

The user-facing model is:

1. Start or open a conversation.
2. Choose the context it should use.
3. Verify the assembled runtime context.
4. Chat.

## Design Principles

- Conversation-first: every route that starts or manages chat context should
  eventually land in a conversation with visible effective context.
- Optional character: character cards are one context input, not a prerequisite
  for worldbooks or dictionaries.
- Reusable assets: worldbooks and dictionaries should feel attachable to many
  chat surfaces, not trapped inside one page.
- Runtime legibility: users should always be able to answer, "What context will
  this chat actually use?"
- Progressive disclosure: first-time users get plain readiness states; power
  users can inspect diagnostics, matches, transformations, and scope.
- Preserve existing capability: underlying storage and APIs can remain separate
  where that matches the codebase. The UX should assemble them coherently at the
  conversation boundary.

## Personas And Primary Jobs

### First-Time User

Primary job: start a character or blank conversation and understand how to add
usable context without learning separate feature islands first.

Expected flow:

1. Start from New Chat, Character Cards, a Worldbook, a Dictionary, or a
   Workspace.
2. See a clear setup path with optional context slots.
3. Add or confirm character, worldbooks, dictionaries, workspace, and provider.
4. See whether the chat is ready or blocked.
5. Start chatting with confidence that the selected context is active.

### Regular Power User

Primary job: reuse context assets across many conversations and quickly debug
why a chat did or did not use specific lore, terminology, provider settings, or
workspace scope.

Expected flow:

1. Open an existing chat or workspace.
2. Inspect effective context without leaving the conversation.
3. Attach, remove, or switch worldbooks and dictionaries.
4. Verify matched entries, dictionary transformations, skipped assets, and
   provider blockers.
5. Reuse or bulk-assign assets without losing conversation state.

## Workflow Shape

The primary workflow is conversation-first, with character chat as one branch of
the same system.

### Entry Routes

Users may enter from:

- New Chat
- Character Cards
- World Books
- Chat Dictionaries
- Workspace Playground
- Existing chat history

Each route should preserve its intent but converge on the same conversation
context model.

Examples:

- Character card row action: opens New Conversation with the character slot
  prefilled.
- Worldbook detail action: opens New Conversation with the worldbook slot
  prefilled.
- Dictionary quick assign: offers chat selection and a "start chat with this
  dictionary" route.
- Workspace Playground: starts a workspace-scoped conversation with workspace
  scope prefilled and context slots visible.

### Setup Flow

Use a lightweight setup sequence:

1. Intent
2. Context
3. Provider
4. Start

Intent examples:

- Blank chat
- Character chat
- Workspace chat
- Research chat

Context slots:

- Character: selected or none
- Worldbooks: selected, active, and match-ready
- Dictionaries: selected, active, and transform-ready
- Workspace: selected or none
- Provider/model: selected, missing, or blocked

The setup flow should not force character selection. If the user enters from a
character card, character is simply prefilled.

### In-Chat Flow

Once chat starts, the conversation should keep a compact context inspector
available. Users should be able to edit context without abandoning the
conversation. The inspector is the shared confirmation surface for first-time
users and the shared debugging surface for power users.

## Proposed Work Packages

### 1. Conversation Context Panel

Add a reusable panel available from chat, character-chat routes, and workspace
chat surfaces.

The panel should show:

- Character: selected or none.
- Worldbooks: attached, active, matched entries, and diagnostics.
- Dictionaries: attached or active, scope, and replacement status.
- Workspace: selected or none.
- Provider/model: available, missing, or blocked.

First-time value: reduces uncertainty about whether lore, dictionaries, and
provider setup are ready.

Power-user value: creates a single audit surface for context composition.

### 2. New Conversation Setup Flow

Create a lightweight setup surface for starting chat:

`Intent -> Context -> Provider -> Start`

The flow should support blank chats, character chats, workspace chats, and
context-first entry from worldbooks or dictionaries. It should make character an
optional context slot and should avoid copy that suggests worldbooks or
dictionaries belong only to characters.

### 3. Prompt Preview And Diagnostics Unification

Prompt preview should show worldbook and dictionary diagnostics together.

It should answer:

- Which worldbook entries matched?
- Which dictionary terms were transformed?
- Which context assets were skipped, and why?
- Which provider, model, workspace, or context blocker prevents a real chat?
- Does the preview match the payload that will be used for the next message?

Worldbook diagnostics already have a stronger visible path than dictionary
diagnostics. The unified preview should close that parity gap.

### 4. Cross-Asset Assignment Clarity

Worldbooks and dictionaries should expose assignment as attachment to reusable
conversation scopes, not as ambiguous activation.

Recommended terms:

- Attach to chat
- Attach to workspace
- Use with character-started chats
- Active globally

Avoid terms that imply character ownership when the target is actually a chat,
workspace, or global chat setting.

### 5. Workspace Context Controls

Workspace Playground should expose the same context slots as chat:

- Character
- Worldbooks
- Dictionaries
- Provider/model
- Effective context diagnostics

If workspace-scoped chat can carry dictionary settings by API, users should be
able to see and manage that state in the workspace UI.

## Reliability States

The workflow should make these states explicit.

| State | Meaning | First-time copy | Power-user detail |
| --- | --- | --- | --- |
| Configured | Asset is attached to the chat, workspace, global defaults, or character-start shortcut | Added | Scope and source |
| Active | Asset is eligible for this conversation | Ready | Effective runtime scope |
| Matched | Asset applied to the current message or preview | Used in this message | Keys, terms, entry IDs, replacement counts |
| Skipped | Asset is present but did not apply | No matches yet | Skip reason and matching criteria |
| Blocked | Chat cannot run as configured | Needs setup | Provider, model, workspace, permission, or data error |

## Error Handling

Errors should be recoverable and tied to user action.

- Missing provider/model should be a local blocker in the setup flow and context
  inspector, not a late surprise after send.
- Invalid workspace IDs should return and display a recoverable setup error, not
  an unexplained server failure.
- Missing or deleted worldbooks and dictionaries should show as skipped or
  unavailable with a source reference.
- Dictionary processing and worldbook matching should expose zero-match states
  separately from failures.
- Prompt preview should distinguish "no context matched" from "context could
  not be evaluated."

## Data Flow Concept

The UI should assemble effective context at the conversation boundary:

1. User chooses or opens a chat.
2. UI loads conversation metadata and scope.
3. UI loads selected context assets by type: character, worldbooks,
   dictionaries, workspace, provider/model.
4. UI requests or computes preview diagnostics for the next message.
5. User sees effective context before sending.
6. Send path uses the same effective context contract as preview.
7. Post-send diagnostics remain inspectable for debugging and confidence.

The design goal is parity between "what preview says" and "what send uses."

## Terminology Guidance

Preferred language:

- Conversation context
- Attached to chat
- Attached to workspace
- Active for this conversation
- Matched this message
- No matches yet
- Blocked by provider setup

Use carefully:

- Character chat: only when a character card is selected.
- Active globally: only for context that truly applies beyond one chat.
- Use with character-started chats: for shortcuts or defaults that prefill
  context when a character initiates a conversation.

Avoid:

- Copy that implies worldbooks or dictionaries are character-card-only.
- "Activate" without saying where it activates.
- Diagnostics that use backend-only terms without a plain-language equivalent.

## Validation Scenarios

Future implementation should be validated against these scenarios:

- Blank chat with worldbook only.
- Blank chat with dictionary only.
- Blank chat with worldbook and dictionary.
- Character chat with no worldbook or dictionary.
- Character chat with worldbook and dictionary.
- Workspace chat with worldbook only.
- Workspace chat with dictionary only.
- Workspace chat with worldbook and dictionary.
- Context-first entry from a worldbook into a new chat.
- Context-first entry from a dictionary into a new chat.
- Existing chat with changed context assets.
- Prompt preview parity with actual chat payload.
- Missing provider/model blocker.
- Invalid workspace/context ID recoverability.
- Zero-match worldbook and dictionary states.

## Success Criteria

The workflow succeeds when:

- First-time users can start a blank or character conversation with reusable
  context without visiting unrelated pages first.
- Power users can verify effective context in one place for chat and workspace
  flows.
- Worldbooks and dictionaries are visibly reusable across conversation types.
- Prompt preview and diagnostics cover both lore injection and dictionary
  transformations.
- The UI clearly distinguishes configured, active, matched, skipped, and blocked
  states.
- Workspace context controls expose capabilities that already exist at the API
  level.

## Open Questions For Implementation Planning

- Should Conversation Context Panel live as a right rail, drawer, popover, or
  tabbed panel in the current WebUI layout?
- Which backend endpoint should become the source of truth for effective
  context preview, especially dictionary diagnostics?
- Should assignment controls support bulk operations in the first implementation
  slice, or should bulk assignment be deferred after parity is established?
- How should global context defaults be represented so users understand when a
  chat inherited context versus when it was explicitly attached?
- What should be the minimum viable workspace context control set for the first
  implementation tranche?

## Out Of Scope

- Redesigning the full Chat, Character Cards, World Books, Dictionaries, or
  Workspace modules.
- Changing the persistence model before an implementation plan reviews current
  APIs and database boundaries.
- Requiring users to choose a character before using worldbooks or dictionaries.
- Treating prompt preview as a separate feature island from send-time behavior.

## Next Step

After this spec is reviewed and accepted, create an implementation plan that
breaks the work into independently reviewable packages. The first likely plan
should prioritize conversation context inspection and preview parity before
larger assignment or workspace-control expansions.
