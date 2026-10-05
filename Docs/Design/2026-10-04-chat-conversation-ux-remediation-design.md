# Chat conversation UX remediation design

Date: 2026-10-04

Status: Proposed; written design awaiting user review. No application changes have begun.

Program: TASK-13453; delivery stages TASK-13454–TASK-13458.

Baseline: latest `origin/dev` verified at `75ab224081bf140ef52017c1a9b0a04f6878d488`.

## Outcome and evidence

Make `/chat` dependable and understandable for first-time users, returning users, experienced power users, character roleplayers, and people using assistive technology. Give those audiences equal weight. Sequence work by dependencies: establish valid conversation state, expose it clearly, improve roleplay setup, complete responsive/accessibility and extension behavior, then verify complete journeys.

All ten finding groups and the enhancements from the [independent NN/G audit](/Users/macbook-dev/.codex/worktrees/chat-nng-ux-audit/tldw_server2/Docs/Reviews/CHAT_NNG_UX_AUDIT_2026_10_04.md) are in scope. The audit is preserved in the attached audit worktree at commit `942efa7da9`. This document is self-contained about the proposed changes; screenshots and detailed reproductions remain in the audit.

The baseline evidence distinguishes demonstrated failures from risks. Standard sending worked, and a locally saved standard conversation survived browser restart, but the primary history views did not expose it. Fresh explicit-provider character sessions saved and reopened; a local-standard-to-character transition failed ownership checks. The audit did not establish data loss, setting leakage, an older-character-session omission, or successful transfer of every handoff field. Those distinctions remain in the regression evidence.

Apply [NN/G's ten usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/) as review criteria, particularly accurate status, understandable language, predictable controls, prevention/recovery, and visible ways to continue work. Use [progressive disclosure](https://www.nngroup.com/articles/progressive-disclosure/) to retain expert capability while reducing optional decisions in everyday chat. These are our product design decisions, not an NN/G certification or an official scoring system.

## Selected approach

**Recommended: repair the shared flows and reorganize the existing components in five stages.** Keep current visual tokens, React Query, Zustand, Dexie, Ant Design, WebUI wrappers, and WXT extension APIs. Reuse the existing history loaders, roleplay preview, saved setups, greeting picker, command palette, and handoff service. This gives both surfaces the same behavior without a new conversation framework.

Two alternatives were considered:

| Approach | Benefit | Cost / decision |
|---|---|---|
| Shared flow repair and progressive disclosure | Addresses the audited tasks directly; preserves existing contracts and expert controls | Selected. Requires careful integration with concurrent lifecycle fixes. |
| Replace `/chat` with a new setup wizard and workbench | Allows a fresh information architecture | Adds another setup/settings system and migration effort before saving and ownership are dependable. Do not introduce it for this scope. |
| Make every chat server-only immediately | Could simplify cross-origin identity eventually | Changes persistence, offline behavior, and migration policy. The separate D7 proposal is awaiting owner decision; this client design does not decide that architecture. |

The existing [D7 proposal, PR #3165](https://github.com/rmusser01/tldw_server/pull/3165), and [honest saving labels, PR #3147](https://github.com/rmusser01/tldw_server/pull/3147), [sidepanel integrity, PR #3152](https://github.com/rmusser01/tldw_server/pull/3152), and [content search, PR #3178](https://github.com/rmusser01/tldw_server/pull/3178) overlap parts of the scope. These were open when this design was prepared. Refresh dev before each stage and verify merged behavior; reuse those changes rather than create competing implementations. An open PR or its claimed checks do not close an audit finding. Do not merge or cherry-pick unrelated work as part of design preparation.

## Conversation behavior

### Honest saving and a reachable history

Derive the status from acknowledged writes for the current conversation and turn. Connectivity means the server can be contacted; it does not prove persistence. A server ID can identify an older saved conversation without proving the latest reply was saved.

| Current state | Visible status and action |
|---|---|
| Empty saved-mode view, no durable content | **New chat**; no saved-success claim |
| Temporary mode | **Temporary** and the existing precise retention explanation |
| Local write acknowledged, no server acknowledgement | **Saved on this browser**; **Find conversations** can reopen it |
| Write pending | **Saving…** with destination in details; retain the last acknowledged state |
| Server confirms current content, or owner-validated server load returns it | **Saved on server** |
| Server write fails and a local copy is acknowledged | **Saved on this browser · Server save failed**; offer the existing safe retry/save operation only when its outcome can be reconciled |
| Local write fails | **Could not save on this browser**; retain the visible draft/content and offer a valid recovery/export path |
| Reload cannot establish the latest server write outcome | Explain unconfirmed changes; reconcile through the existing owner-bound loader before claiming server success |

Use the existing write/recovery records and, when merged, PR #3147's shared status resolver. Extend gaps in that flow rather than add a parallel status store. Do not infer local durability merely from a generated history ID. Do not automatically replay an uncertain provider request or silently import local histories into the server.

The main sidebar gains one **Find conversations** field and an **All** view with separately labelled **Server** and **On this browser** result groups. Reuse each group's existing pagination and loader. Keep Server, On this browser, Folders, and Trash filters available; group-local page controls avoid inventing a combined cursor protocol. Suppress an identical server mirror only when its scoped server identity is proven; duplicate titles remain distinct. On server failure, readable local results remain available and the failed group has Retry. Origin-local history cannot become another account's mutation authority.

### Starting or changing a conversation

Use one consistent **New chat** menu everywhere: **New saved chat**, **New temporary chat**, and **New character chat**. Where space calls for a single mode-preserving shortcut, its name states the resulting mode, such as **New temporary chat**. “Saved” describes the requested mode; empty views still show **New chat** until a write is acknowledged.

An explicit new conversation releases the prior selected-history lease, clears prior IDs and ownership through the existing controller/reset path, and opens a fresh view. Late loads, saves, greeting callbacks, and account changes cannot rebind that view to the previous conversation. Keep the previous saved conversation reachable. Preserve the source draft while a transition is pending or fails, using existing draft/tab state. Temporary content is not written to history as a side effect of switching modes; explain any action that would discard unsaved temporary content.

For **New character chat**, reuse the setup drawer. Apply validates character selection and a provider-qualified usable model, prepares the fresh view, and shows **Ready to start** rather than implying a saved server session exists. The first explicit greeting/turn uses the existing native creation and owner-bound load path; publish the server identity only after an acknowledgement is valid for the current view/account. A suitable automatic model may resolve from the existing catalog when it has one authoritative provider; ambiguous or unavailable selection asks the user to choose a model before sending.

Applying a different character to a bound incompatible conversation offers **Start a new character chat**, preserving the source chat and draft. It does not convert a local owner to native ownership or reinterpret existing rows as authoritative ancestry. **Use no prior messages** describes excluding prior messages; if the bound native path cannot do that safely, replace the action there with **Start a new character chat**. Do not present an action as recovery unless it actually establishes a supported state.

For an unconfigured first visit, show the missing prerequisite with a direct link to the existing connection/provider setup. Distinguish connecting the server, selecting a usable model, and selecting a character. The user can write a draft while configuring; readiness prevents a provider request that is already known to fail. Reuse the current setup/connection UI rather than introduce another wizard. Test this state separately from the provisioned connection used in the audit.

Preflight the same prerequisites the sender enforces: selected owner, model/provider, temporary/native limitations, and unsupported prompt/scene/files/tools/settings. Reuse or extract the shared validation used by sending instead of maintaining an independent readiness rule. Keep send-time checks for races. Errors give a plain reason and concrete action—Choose model, Start a new character chat, Reconnect, or Review unsupported settings—and retain the draft. Internal codes can remain in diagnostic details. Retry generation only after the previous outcome is known to permit it.

## Information architecture and roleplay

### Casual and Pro

Casual makes the composer the focal point: message, model, Attach, Send, and **Customize chat**. Character selection remains a clearly named entry inside Customize; an active character has a visible chip that reopens setup. Active context/tools are summarized compactly so hiding their controls cannot hide their effect. Queue/Stop and response recovery remain visible when relevant.

Customize uses the existing popover/drawer/rail controls grouped by task: **Research & context**, **Character & your role**, and **Response settings**. Existing configuration opens directly rather than duplicating editors. Pro retains its current direct controls, comparison and expert tools, shortcut/slash/mention entry points, focus mode, and settings depth. Mode changes preserve the draft and active settings.

Recent conversations and Find conversations precede platform shortcuts in the sidebar. History is expanded on first use; thereafter preserve the explicit expansion preference when closing/reopening the sidebar. The command palette is labelled **Search commands**; **Find conversations** is the history entry point. Keep existing accelerators and help, without changing a working shortcut merely to add a new convention. Escape and close return focus to the initiating control.

### The character opening

When a card is selected and no turns exist, replace the generic starter with a compact character introduction and the selected greeting. **Start with this greeting** is the primary action; **Write your own opening** focuses the composer. Alternate greetings and contextual controls are secondary, with an explanation of which speaker begins. Expression-image configuration appears as a secondary affordance after the opening rather than above it.

Reuse `ChatGreetingPicker` and the valid server persistence path. Viewing a greeting does not by itself acknowledge a saved turn. Starting it writes at most one greeting and binds it to the correct character/session; switching cards or views invalidates stale callbacks. If a card has no greeting, show **Write the first message**. The introduction and primary action must be visible without scrolling through generic onboarding at desktop and narrow widths; exceptionally long greetings may scroll within the transcript.

### Language, scope, and commitment

Use **AI character**, **Your role**, **Scene**, **Inherited lore**, and **Additional pinned context**. Preserve distinctions where an existing Persona acts as an assistant rather than the user's role: label the actual behavior, not every Persona as “you.” The zero-pinned state reads **No additional pinned context**. It does not describe inherited character instructions, worldbooks, or dictionaries as absent.

Every editor states its scope beside its heading: **This conversation**, **Character library**, or **Provider-model default**. Apply these commitment rules:

| Interaction | Commitment |
|---|---|
| Direct composer selection/toggle | Takes effect for subsequent turns; communicates saving/pending/error when persistence applies |
| Multi-field roleplay or conversation/settings editor | Edit a staged draft; **Apply to this conversation** commits changed fields through the existing scoped APIs; **Cancel** discards unapplied changes |
| Character library editor | Existing explicit save updates the reusable card; conversation Apply never edits the card implicitly |
| Provider/model default editor | Explicit **Save model default** updates that scope; conversation controls never write global defaults implicitly |
| Save reusable roleplay setup | Stores a named setup using the existing preset service; shows scope and preview before later Apply |

Remove per-field autosave from an editor that claims Apply/Cancel controls the whole draft; retain direct immediate controls outside staged editors. Dirty state is visible, closing a dirty editor offers a clear discard route, and navigation cannot silently discard it. If multiple existing operations partially succeed, acknowledge which changed, retain the remaining draft, and retry only unresolved operations. An owner change invalidates the pending Apply rather than applying to a new chat.

Reuse saved setups and the existing before/after preview. A preset shows character/assistant role, behavior, scene, generation settings, and additional context changes before Apply. Modified setups can be saved as a new named setup through the current service. Revert never claims to roll back a server write it cannot undo.

### What the next turn includes

Reuse the current state/compatibility and prompt-assembly previews to show a collapsed **Next turn** summary: assistant/character, user's role where applicable, model/provider, history selection, scene, inherited lore, and additional sources. Expanding it exposes concise provenance and links to the owning editor. Avoid a second independently assembled prompt.

Distinguish **Selected**, **Planned for this turn**, **Confirmed by server**, **Excluded**, and **Unknown**. Server-side conditional lore may be configured without being activated; show that distinction. A local preview cannot claim the provider received particular context. Do not show **Included** simply because a UI field was populated.

Current native character sends reject client actor/scene, custom prompt/context and other overrides (`native-history-character-send.ts` and its caller). The initial client remediation explains and blocks incompatible combinations before Send. Snapshot-based native resume requires the existing **TASK-13160** capability and snapshot contract; route presence or a speculative preview cannot enable it. Its accepted [behavior snapshot decision](../../backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md) permits only supported server-owned, versioned behavior-setting mutations. It does not automatically make every client scene/prompt overlay supported. Use authoritative snapshot/settings values when available; leave any consuming enhancement open until its exact contract and tests pass, and assess a missing durable API separately before extending it. The client must not bypass native history through an older, less guarded sender to make setup appear successful.

### Complete character-session discovery

The current overview fetches five character sessions before partitioning by character. Test interleaved sessions, and use the existing `/chats/` `character_id` filter for the active character in `useServerChatHistory`, including it in the query key and scope checks. Provide **All sessions for this character** with normal paging. The compact five-row preview is a preview, not the entire history. Other-character recents can keep their own bounded list. A partial/stale fetch says so; an absent item from the first five results cannot establish “No sessions.”

## Responsive, accessible, and extension behavior

Constrain the existing model dropdown to the viewport and use its placement/overflow handling. The search and model rows remain reachable at 320px, 390px, desktop, and 200% zoom; long labels wrap/truncate with an accessible full name. Keep transcript/composer usable with customization open and with the software keyboard where the test environment supports it.

Give greeting switches and history search persistent accessible names/descriptions. Use valid textarea/autocomplete semantics without adding unsupported ARIA. Each surface has a coherent main landmark and heading structure. Keep a visible focus indicator, predictable tab order, Escape dismissal, and focus return; announce meaningful loading/saving/error changes without announcing every streamed token.

Correct shared semantic color tokens for both themes rather than apply unrelated per-component colors. Verify normal text at 4.5:1 and the relevant non-text/focus requirements against [WCAG 2.2](https://www.w3.org/TR/WCAG22/). Assess touch targets with spacing/exception rules; do not treat the observed 16×16 icon alone as proof of a failure. Preserve the product's visual identity and CSP.

The compact sidepanel keeps a truncated but readable conversation title visible, with the full title available to assistive technology and title editing/details. Title takes precedence over redundant branding on small widths. Show mode and accurate save status without expanding a full cockpit.

Label destinations **Open in full extension** and **Continue in WebUI**. Reuse the existing expansion controller and handoff payload; expose a concise explanation of carried state before opening:

| State | Full extension | WebUI |
|---|---|---|
| Valid selected history | Existing bounded expansion with validated selection | Reopen a server reference only through destination owner/account/workspace validation |
| Local-only conversation | Reopen in the same extension origin through its local loader | Local ID is not portable; carry the supported draft/settings only, explain that history stays in the extension |
| Unsent draft / model / supported page context | Preserve through existing tab/draft mechanisms; describe actual fields | Use existing allowlisted payload fields and destination context resolution; unavailable items are explained |
| Character/roleplay behavior | Restore acknowledged session state when owned | Restore from an owned server session only; unsaved card/persona/scene settings absent from the payload are not claimed transferred |
| Active streaming | Continues in its source panel under the current contract | Do not imply the in-flight generation has moved or silently re-send it |

An expired payload, destination mismatch, failed open, or unavailable context preserves the source. Retain current payload freshness/validation and avoid placing credentials or full histories in URLs. Successful handoff means the destination acknowledges the available state, not merely that a new tab opened. If the current mechanism cannot carry a field safely, the UI names the limitation and offers a path in the source/full extension; adding a new cross-origin transfer protocol requires a separate architecture assessment.

## Stages and validation gates

Each stage delivers a reviewable behavior change. This is a design sequence, not the detailed implementation plan. After design review, write the unique implementation plan, then use the repository's TDD, review, security, and verification workflow. Refresh the latest dev baseline and record it at each stage. Preserve red-first reproductions and turn known-defect wrappers into ordinary assertions only after independent behavior checks.

### Stage 1 — Saving and safe conversation transitions

**Task:** TASK-13454. **Status:** Not Started.

**Goal:** dependable start, saving, new-chat mode, and character ownership. Findings 1, 2, 8.

**Success criteria:** acknowledged statuses; standard local chat reachable; fresh character transition/provider readiness; explicit mode semantics; failures preserve drafts/source.

**Tests:** unconfigured first visit → existing connection/provider setup → ready; create → name → reload → new chat → find → reopen for local/server; local standard → new character → greeting/first turn; ambiguous default provider; unsupported context; late create/load acknowledgement; account/workspace switch; save failure/reconnect; Temporary at every new-chat entry point.

**Source areas:** `usePersistenceMode`, `usePlaygroundPersistence`, `useClearChat`, `useHistorySelection`, `useChatActions`, `native-history-character-send`, `ChatHeader`, shared sidebar and compact route. Reuse current persistence transactions/recovery.

### Stage 2 — Focused composer and reachable history

**Task:** TASK-13455. **Status:** Not Started.

**Goal:** simplify everyday setup and make every relevant saved chat findable. Findings 3, 10; navigation portion of 1.

**Success criteria:** Casual disclosure with recognizable active settings; Pro capability retained; history-first sidebar with retained expansion; distinct command/history search; complete paged active-character sessions.

**Tests:** Casual/Pro toggle preserves draft/settings; keyboard access to every moved action; collapsed/reopened sidebar; local/server query and duplicate titles; server failure with local results; more than five interleaved sessions; page boundaries and large histories.

**Source areas:** `ComposerToolbar`/overflow, `ChatSidebar`, `LocalChatList`, `useLoadLocalConversation`, `useServerChatHistory`, `CharacterChatSessionsPanel`, existing command palette.

### Stage 3 — Character opening, terminology, and scoped setup

**Task:** TASK-13456. **Status:** Not Started.

**Goal:** understand and control character behavior and the next turn. Findings 4, 6, 7; presets and next-turn summary enhancements.

**Success criteria:** visible character greeting; accurate identity/context concepts; Apply/Cancel and scope rules; reusable setup preview; truthful context provenance/capability state.

**Tests:** greeting/no-greeting/long-greeting; repeated click/switch during persistence; imported PNG/JSON card; inherited worldbook/dictionary; selected vs activated lore; preset modify/apply/cancel; partial settings save and navigation; scene/prompt/context restoration and unsupported-native recovery; historical isolation once TASK-13160 supports native resume.

**Source areas:** `PlaygroundChat`, `ChatGreetingPicker`, `RolePlaySetupDrawer`/preview, roleplay state/scene/compatibility helpers, `useChatSettingsRecord`, conversation/model settings, saved setups and prompt-assembly preview.

**Dependency:** TASK-13160 owns the server contract for unsupported native extensions. Deliver other client improvements independently; keep its consuming enhancement open until supported. Do not hold unrelated accessibility repairs behind this backend dependency.

### Stage 4 — Accessible narrow layouts and extension continuity

**Task:** TASK-13457. **Status:** Not Started.

**Goal:** equivalent usable controls and identifiable conversations across surfaces. Findings 5, 9.

**Success criteria:** viewport-safe dropdown, accessible names/semantics/contrast/focus, compact title/status, accurate destinations and carried state.

**Tests:** 320/390px, 200% zoom, desktop, both themes; keyboard and VoiceOver/manual screen-reader walkthrough; targeted axe; model popup geometry; long title/labels; both handoffs with drafts and scoped server/local histories; expired/mismatched payloads; native Chrome docking/reopen.

**Source areas:** `ChatModelSelectorDropdown`, `ComposerTextarea`, greeting/history inputs, layout landmarks, shared color/runtime tokens, `SidepanelHeaderSimple`, `ControlRow`, compact route/sidebar and `sidepanel-chat-webui-handoff`.

**Scheduling:** after Stage 1/2 state/navigation contracts; may proceed while Stage 3's external backend dependency remains open.

### Stage 5 — Complete journeys and evidence-based polish

**Task:** TASK-13458. **Status:** Not Started.

**Goal:** verify all findings and enhancements together and publish the revised walkthrough.

**Success criteria:** supported cross-surface journeys pass, ownership/privacy invariants hold, meaningful checks pass, and limitations are explicit. No whole-program completion while a scoped enhancement remains blocked/unverified.

**Tests:** the matrix below against the real backend lifecycle with a deterministic provider; relevant unit/integration/e2e regressions, typecheck/lint, WebUI and packaged-extension builds, touched-scope security checks, before/after screenshots and NN/G review.

**Output:** verified walkthrough and issue closure matrix; bounded polish driven by failures; a short uncoached usability-study script. Report participant results only if actually collected. No telemetry or automatic recruitment is introduced.

## Coverage and proof

| Audit finding / enhancement | Stage | Required evidence |
|---|---|---|
| 1 Saving meaning and reopen access | 1, 2, 5 | Actual local/server writes and correct reload/search/reopen; failed/latest-turn saving status |
| 2 Native character/provider/owner transition | 1, 5 | Fresh and local-to-new-native paths, preflight and stale-owner failures |
| 3 Casual overload and buried history | 2, 5 | First-send hierarchy; Pro/tool/shortcut parity; expansion preference |
| 4 Character greeting hierarchy | 3, 5 | Visible opening and exactly-once correct-session greeting |
| 5 Responsive and accessibility defects | 4, 5 | Popup geometry, names/landmarks, contrast, keyboard and manual screen reader |
| 6 Identity, scene, context language | 3, 5 | Accurate preview across inherited and pinned context; task comprehension review |
| 7 Editing scope/commit semantics | 3, 5 | Apply/cancel/retry/navigation; no unintended library/default changes |
| 8 New-chat mode inconsistency | 1, 5 | All entry points and temporary/privacy behavior |
| 9 Extension identity/handoff differences | 4, 5 | Native panel plus full-extension/WebUI carried-state matrix |
| 10 Search boundaries/session coverage | 2, 5 | Commands vs history; interleaved sessions and paging |
| Casual/Pro task-based disclosure | 2, 5 | Draft/active-state preservation and direct expert access |
| Unified find/reopen and origin badges | 1, 2, 5 | Scoped local/server groups, mirrors, duplicate titles, outage |
| Existing roleplay presets with preview/scope | 3, 5 | Save/reuse/modify/apply/cancel and restore |
| Next-turn summary/provenance | 3, 5 | Planned vs acknowledged context; no unsupported-inclusion claims |
| Consistent handoff/context carry | 4, 5 | Correct destination, origin-local limitations, failure recovery |
| Contextual help and keyboard/focus | 2, 3, 4, 5 | Short nearby explanations and complete keyboard flows |

Test the following journeys on WebUI, full-page MV3 extension, and compact sidepanel where its bounded capabilities apply:

1. **Standard:** start → first message → name → leave/restart → find → correct chat → continue, for both local and server origins.
2. **Roleplay:** choose/import card → inspect scope/context → greeting or own opening → two turns → reopen → change supported scene/setup → continue. Include worldbooks/dictionaries and capability-gated custom context.
3. **Power:** reuse/modify preset, switch model/provider, inspect context/history, use commands and shortcuts, stop/regenerate where allowed, and maintain current expert tools.
4. **Failure:** provider refusal, interrupted stream, offline/reconnect, failed save/load, conflicting/ambiguous history, stale lease/account/workspace change, and failed handoff; preserve recoverable content without silent replay or cross-owner mutation.
5. **Range/access:** empty/long greeting, long/duplicate title, over five interleaved character sessions, 2,000 seeded histories with paged results, narrow/zoom/light/dark, keyboard and manual screen reader. Record fixture size and machine; avoid an invented performance budget before measuring.

The usability-study script uses uncoached standard and roleplay tasks for novice and experienced participants equally. Record completion, wrong-chat choices, assistance requests, and post-task ease. Two minutes to first standard message after a usable connection/provider is available, and thirty seconds to reopen the correct chat, remain proposed product targets, not measured results or automatic release claims. Record unconfigured connection/setup time separately. Automated checks and expert walkthroughs do not substitute for recruited participant results.

## Architecture and delivery constraints

**ADR check:** ADR required: **no** for this client design. [ADR-049](../ADR/049-chat-history-selection-ownership.md) governs owner-bound selection; [ADR-050](../ADR/050-native-chat-fork-storage-lifecycle.md) governs native fork storage and unfinished public execution; [ADR-059](../ADR/059-backlog-py-task-editor-cutover.md) governs task records. The accepted [behavior snapshot decision](../../backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md) governs snapshot-based roleplay resume and versioned settings. We change presentation and callers within those rules, not persistence authority, public API, or fork lifecycle. TASK-13160 remains the existing server contract owner. If implementation needs a new durable API, transfer protocol, or migration decision, record a new assessment and design that change before coding it.

Preserve owner/account/workspace fences and explicit legacy-history review. Do not infer ancestry from displayed rows/timestamps, silently promote local owners, expose unfinished native forks, weaken CSP, or alter temporary retention accidentally. Keep scope-specific settings and privacy decisions intact. Use existing dependencies, locale source files and mirror tooling, shared components, and current test harnesses. Avoid a new settings framework, onboarding wizard, analytics collection, search index, or generic history aggregation service.

This branch is isolated from the dirty primary checkout and from the completed audit branch. It contains design/tracking only. The detailed implementation plan and application edits follow written design approval. Completion is evidence-based by stage; unmet backend capabilities, native-shell checks, and human-study checks remain explicitly identified rather than disappearing from scope.
