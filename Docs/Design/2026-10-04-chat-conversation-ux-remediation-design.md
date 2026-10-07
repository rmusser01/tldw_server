# Chat conversation UX remediation design

Date: 2026-10-04

Status: Approved by the user on 2026-10-04, with the pre-implementation review corrections below. Stage 1 implementation is in progress. The accepted D7 owner decisions integrated on 2026-10-05 govern the remaining persistence work.

Program: TASK-13453; delivery stages TASK-13454–TASK-13458.

Initial baseline: `origin/dev` at `75ab224081bf140ef52017c1a9b0a04f6878d488`. Current verified baseline: live remote `dev` at `7ba48f251ec47a1e0bb680f49f9b7d86ec2b988d` (includes previously checked `1047ce10ecc191f78b0bee7e1b8ad19780c68015`), integrated in `2f7fbb8653` on 2026-10-06. Incoming domain/session authority changes were checked with the affected chat, catalog and API-client suites; both client type gates pass. Reviewed implementation checkpoints below record the frontend evidence. Remaining staged acceptance is still open.

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
| Server identity for new saved chats, with explicit legacy import | Simplifies cross-surface identity and acknowledged persistence | Adopted in the separate accepted D7 design. Implement after its backend prerequisites; legacy local chats remain usable and are never migrated automatically. |

The accepted [D7 design](2026-10-04-d7-conversation-model.md), merged by [PR #3165](https://github.com/rmusser01/tldw_server/pull/3165), and [honest saving labels, PR #3147](https://github.com/rmusser01/tldw_server/pull/3147), [sidepanel integrity, PR #3152](https://github.com/rmusser01/tldw_server/pull/3152), and [content search, PR #3178](https://github.com/rmusser01/tldw_server/pull/3178) overlap parts of the scope. These were open when this design was prepared; D7 is now accepted and merged as a design, not as shipped backend capability. Refresh dev before each stage and verify merged behavior; reuse those changes rather than create competing implementations. An open PR or its claimed checks do not close an audit finding. Do not merge or cherry-pick unrelated work as part of design preparation.

## Accepted D7 reconciliation (2026-10-05)

D7 §8 records the owner decisions: new non-Temporary chats use native server identity; cached chats are readable offline while Send waits for connection; existing local chats stay usable and import only on an explicit per-chat or bulk action to the signed-in account; Temporary chats survive reload on this device until closed or cleared and never reach the server; Sync v2 users remain in scope. Its P1–P10 contracts and migration gates control the corresponding work below. Do not extend auto-promotion as the long-term first-send design or add an offline outbox. Use durable records and server acknowledgements for saving status, with no competing status store.

The draft-transition checkpoints preserve recoverable unsent text and qualified owner identity. Their earlier Temporary transcript exclusion is superseded by Q3 and needs a follow-up across all surfaces; it is not evidence that the new reload policy is delivered. Backend idempotent creation, settlement metadata/partials, lossless import, freshness and Sync v2 support remain unimplemented until their contracts and tests exist. D7 acceptance does not enable TASK-13160 custom context or close any unsupported-capability gate.

## Pre-implementation review checkpoint

The user approved the direction and requested a critical review before continuing. Two independent reviewers and a source review checked the design against dev; a fresh fetch confirmed the baseline above. This checkpoint did not launch the application or run application tests. Source facts and derived failure risks below are implementation gates, not newly reproduced incidents.

| Priority | Problem found in the design or its assumed helpers | Correction and gate |
|---|---|---|
| P1 | Recovery by loading a known ID cannot resolve a creation response lost before that ID reaches the client. Retry can create another chat. | Separate unknown creation from known-ID reconciliation; retain uncertainty across reload using existing recovery records, and do not replay creation without an applicable replay contract. Stage 1. |
| P1 | The native sender requires user input; the current greeting picker inserts local state and cannot create a fresh server session. | Separate preview from commitment. Use supported greeting seeding at creation, validate owned rows, and carry supported provider/model/sampling fields. Greeting persistence belongs in Stage 1; hierarchy in Stage 3. |
| P1 | Character/model-only setup validation can announce readiness while the sender rejects scene, prompts, tools, context, or settings. | Run the shared complete preflight before leaving the source chat and again before dispatch; retain source and drafts on failure. Stage 1. |
| P1 | Ordinary settings helpers can return locally saved data after a server PUT fails, without captured request scope. | Return explicit local/server outcomes, fence Apply, and retry unresolved operations only. Never infer server commitment from a non-null record. Stage 3. |
| P2 | Universal conversation Apply wording misrepresents creation, model defaults, and existing live settings. | Use action labels for their actual purpose, map every field to supported scope, and distinguish staged forms from explicitly live sections. Stage 3. |
| P2 | Deduplicating every server-linked local row can hide the only reachable copy during outages or partial paging; local DB failures currently resemble empty results. | Deduplicate only a proven duplicate visible in current successful results; distinguish unavailable, empty, no matches, and partial coverage in both groups. Stage 2. |
| P2 | Different handoff mechanisms carry different fields; an existing “Continue in WebUI” path opens extension options. Opening a tab provides no import receipt. | Map each entry point to its actual destination and fields; report Opened in the source and Imported/Partially imported in the destination. Preserve draft conflict recovery. Stage 4. |
| P2 | Nested disclosure can unmount the initiating focus target, and successful transitions also need source-draft preservation. | Test focus return to a visible control, one-layer Escape, and successful as well as failed transitions with drafts and temporary content. Stages 1–4. |

The minimal corrections reuse the current controller, scoped API options, create endpoint, recovery records, settings helpers, paged loaders, and transfer mechanisms. No additional conversation framework, universal transaction system, or cross-origin acknowledgement protocol is introduced.

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
| Creation was dispatched but its response/identity is unknown | **Server save unconfirmed**; retain the device copy/draft and explain how to review server history; no blind create retry |
| Server write fails and a local copy is acknowledged | **Saved on this browser · Server save failed**; offer the existing safe retry/save operation only when its outcome can be reconciled |
| Local write fails | **Could not save on this browser**; retain the visible draft/content and offer a valid recovery/export path |
| Reload cannot establish the latest server write outcome | Explain unconfirmed changes; reconcile through the existing owner-bound loader before claiming server success |

Use the existing write/recovery records and, when merged, PR #3147's shared status resolver. Extend gaps in that flow rather than add a parallel status store. Do not infer local durability merely from a generated history ID. Do not automatically replay an uncertain provider request or silently import local histories into the server.

Treat known-ID append reconciliation and an unknown first-create outcome separately. The current promotion closure can reconcile a known conversation's stored prefix; it cannot look up a conversation whose create response was lost. Extend existing saved-mode recovery records minimally to retain the dispatched/unconfirmed state before creation and across reload. A pending marker describes uncertainty, not server ownership or acknowledged saving. Temporary content is never written there. Do not offer create replay merely because an endpoint accepts an idempotency header: current native creation only gives stable identity under particular server modes. Require the applicable contract and tests, including a lost response after server commit. Otherwise preserve the local content/draft, allow history review/export, and explain that starting again may create another conversation. Known-ID retries also reconcile before repeating unacknowledged appends.

The main sidebar gains one **Find conversations** field and an **All** view with separately labelled **Server** and **On this browser** result groups. Reuse each group's existing pagination and loader. Keep Server, On this browser, Folders, and Trash filters available; group-local page controls avoid inventing a combined cursor protocol. Suppress a mirror only when a duplicate with the same proven scoped identity is visible in the current successful server results. An ID alone, a stale page, or an item absent from the current filter/page does not justify hiding the device copy. During outages keep readable cached copies reachable, identify their stale/cached state, and keep mutation authority unavailable until validated. Duplicate titles remain distinct and show origin plus character/mode/date or excerpt sufficient to choose the right chat. Origin-local history cannot become another account's mutation authority.

Both groups distinguish **Loading**, **No conversations**, **No matches**, **Unavailable**, and **Partial/stale results**. Keep the query and unaffected group usable when one fails, and offer Retry for the failed operation. A Dexie/storage failure must not be converted into a successful zero-result page.

### Starting or changing a conversation

Use one consistent **New chat** menu everywhere: **New saved chat**, **New temporary chat**, and **New character chat**. Where space calls for a single mode-preserving shortcut, its name states the resulting mode, such as **New temporary chat**. “Saved” describes the requested mode; empty views still show **New chat** until a write is acknowledged.

An explicit new conversation releases the prior selected-history lease, clears prior IDs and ownership through the existing controller/reset path, and opens a fresh view. Late loads, saves, greeting callbacks, and account changes cannot rebind that view to the previous conversation. Scope reset to the active view; compact tabs retain their independently owned snapshots. Keep the previous saved conversation and its unsent draft reachable after a successful transition as well as while pending or failed, using existing draft/tab state. Temporary content is not written to history as a side effect of switching modes; before discarding it, offer an explicit retain/copy or discard choice through the existing navigation flow. Do not redirect an in-flight turn's completion to a newly selected view.

For **New character chat**, reuse the setup drawer. **Use setup for new chat** runs the full shared compatibility preflight before changing conversation, then prepares the fresh view and shows **Ready to start** only when no known blocker remains. Evaluate the proposed fresh target and its intended settings; the source's local owner is preserved, not treated as the target's owner or implicitly converted. Actual dispatch still validates the active target owner. **Apply to this conversation** labels supported edits to an existing conversation. Cancel or a failed preflight preserves source owner, draft, and settings; incompatible fields are explained beside their controls without silently discarding them. A suitable automatic model may resolve from the existing catalog when it has one authoritative provider; ambiguous or unavailable selection asks the user to choose a model before sending.

The first explicit user turn uses the existing native create/load path. Greeting-only startup extends that existing create caller with the endpoint's supported `seed_first_message`, `greeting_strategy`, and `alternate_index` query options; it does not send an empty user turn or call the provider. Send the displayed default or exact alternate index rather than choosing a second random greeting at creation. Both paths capture and pass supported provider, model, and explicit sampling fields. Validate the created conversation and loaded greeting rows under the current owner/view/account/workspace before publishing identity or a saved greeting. A requested seed, create response, or seed-status header alone is insufficient proof; check the actual owned history. If the card changed after preview, show the actual accepted opening and a review action rather than silently overwriting it or creating another session. Edited greetings that cannot be represented by existing seed options require a separately verified owner-bound append path or an explicit limitation. Unknown creation follows the unconfirmed recovery rule above; rapid repeated activation cannot issue duplicate creation.

Applying a different character to a bound incompatible conversation offers **Start a new character chat**, preserving the source chat and draft. It does not convert a local owner to native ownership or reinterpret existing rows as authoritative ancestry. **Use no prior messages** describes excluding prior messages; if the bound native path cannot do that safely, replace the action there with **Start a new character chat**. Do not present an action as recovery unless it actually establishes a supported state.

For an unconfigured first visit, show the missing prerequisite with a direct link to the existing connection/provider setup. Distinguish connecting the server, selecting a usable model, and selecting a character. The user can write a draft while configuring; readiness prevents a provider request that is already known to fail. Reuse the current setup/connection UI rather than introduce another wizard. Test this state separately from the provisioned connection used in the audit.

Preflight the same prerequisites the sender enforces: selected owner, model/provider, temporary/native limitations, and unsupported prompt/scene/files/tools/settings. Reuse or extract the shared validation used by sending instead of maintaining an independent readiness rule. Keep send-time checks for races. Errors give a plain reason and concrete action—Choose model, Start a new character chat, Reconnect, or Review unsupported settings—and retain the draft. Internal codes can remain in diagnostic details. Retry generation only after the previous outcome is known to permit it.

## Information architecture and roleplay

### Casual and Pro

Casual makes the composer the focal point: message, model, Attach, Send, and **Customize chat**. Character selection remains a clearly named entry inside Customize; an active character has a visible chip that reopens setup. Active context/tools are summarized compactly so hiding their controls cannot hide their effect. Queue/Stop and response recovery remain visible when relevant.

Customize uses the existing popover/drawer/rail controls grouped by task: **Research & context**, **Character & your role**, and **Response settings**. Existing configuration opens directly rather than duplicating editors. Pro retains its current direct controls, comparison and expert tools, shortcut/slash/mention entry points, focus mode, and settings depth. Mode changes preserve the draft and active settings.

Recent conversations and Find conversations precede platform shortcuts in the sidebar. History is expanded on first use; thereafter preserve the explicit expansion preference when closing/reopening the sidebar. The command palette is labelled **Search commands**; **Find conversations** is the history entry point. Keep existing accelerators and help, without changing a working shortcut merely to add a new convention. Escape dismisses one disclosure layer. Final close returns focus to the visible initiating control, or **Customize chat** if a nested trigger has unmounted. Casual/Pro changes, validation errors, and dirty-close recovery cannot strand focus.

### The character opening

When a card is selected and no turns exist, replace the generic starter with a compact character introduction and the selected greeting. **Start with this greeting** is the primary action; **Start with your message** focuses the composer and creates no assistant greeting turn. Alternate greetings and contextual controls are secondary, with an explanation of which speaker begins. Expression-image configuration appears as a secondary affordance after the opening rather than above it.

Reuse `ChatGreetingPicker` and the owner-bound creation path established in Stage 1. Preview, alternate selection, and reroll do not insert transcript rows or request history. The current `useCharacterGreeting` automatically inserts both; separate that preview behavior before claiming explicit commitment. Starting with a greeting commits at most one assistant row without generation and binds it to the correct character/session; switching cards or views invalidates stale callbacks. Starting with the user's message excludes the preview from transcript and client request history; any server-owned card instructions or lore remain separately identified rather than being described as removed. Verify stored rows and actual assembled context in both paths. If a card has no greeting, show **Write the first message**. The introduction and primary action must be visible without scrolling through generic onboarding at desktop and narrow widths; exceptionally long greetings may scroll within the transcript.

### Language, scope, and commitment

Use **AI character**, **Your role**, **Scene**, **Inherited lore**, and **Additional pinned context**. Preserve distinctions where an existing Persona acts as an assistant rather than the user's role: label the actual behavior, not every Persona as “you.” The zero-pinned state reads **No additional pinned context**. It does not describe inherited character instructions, worldbooks, or dictionaries as absent.

Every editor states its scope beside its heading: **This conversation**, **Character library**, or **Provider-model default**. Apply these commitment rules:

| Interaction | Commitment |
|---|---|
| Direct composer selection/toggle | Takes effect for subsequent turns; communicates saving/pending/error when persistence applies |
| New-character setup form | Edit a staged draft; **Use setup for new chat** prepares a new conversation after full preflight; **Cancel** preserves the source |
| Supported conversation edit form | Edit a staged draft; **Apply to this conversation** commits supported changed fields through scoped APIs; **Cancel** discards only unapplied changes |
| Existing live settings section | Explicitly states **Changes take effect immediately** and its actual persistence destination; closing does not claim to undo them |
| Character library editor | Existing explicit save updates the reusable card; conversation Apply never edits the card implicitly |
| Provider/model default editor | Explicit **Save model default** updates that scope; conversation controls never write global defaults implicitly |
| Save reusable roleplay setup | Stores a named setup using the existing preset service; shows scope and preview before later Apply |

Do not claim a whole-draft Cancel in a section that already committed live changes. Retain existing live controls where their timing and scope are useful; remove per-field autosave only inside a form explicitly staged behind Apply/Cancel. Dirty state is visible, closing a dirty editor offers a clear discard route, and navigation cannot silently discard it. If multiple operations partially succeed, acknowledge which changed, retain the remaining draft, and retry only unresolved operations. Revert cannot undo a confirmed write through a local form reset.

Before changing an editor, map each field to its actual storage scope and completion support. The current model settings Save writes provider/model defaults; renaming it does not make those values conversation overrides. The supported conversation generation override subset is `temperature`, `top_p`, `repetition_penalty`, and `stop`; arbitrary fields stored in JSON are not thereby supported native behavior. Conversation Apply must not call model-default mutations. Keep unsupported/default-only fields in their correctly labelled section, and verify two conversations sharing a model remain isolated on switch/reopen and Cancel.

Extend the existing ordinary settings helper's result to distinguish local acknowledgement, server acknowledgement, failed/unknown server outcome, and unresolved operations. Its current non-null result after a failed server PUT proves only local saving. Thread captured request scope, scope and cancellation through the existing API options; an owner/view/account/workspace change invalidates pending Apply and prevents stale completion from publishing. Native versioned behavior updates still require the applicable server contract; a generic settings PUT is not evidence of supported snapshot mutation.

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

Label destinations **Open in full extension** and **Continue in WebUI** only when the actual URL has that destination. The current bounded-package action in `ControlRow` opens `options.html`; its WebUI label must change to the full-extension destination. An actual WebUI entry uses the configured HTTP(S) destination and its existing allowlisted transfer mechanism. Do not infer identical capabilities from similar button placement.

Map these existing mechanisms separately, and expose a concise carried-state explanation before opening:

| Mechanism / entry point | Actual destination | Verified carried state and limits |
|---|---|---|
| Selected-history expansion (`prepareExpansionPath`) | Full extension | Existing bounded owner-validated selection. Local history uses the same-origin local loader; no assumption that another transport also carries it. |
| One-shot package (`createSidepanelChatHandoff`) | Full extension, same origin | Draft, bounded page-context snippets, and route intent; no model/settings/history payload. Current limits: 10-minute lifetime, 32,000-character draft, up to four 4,000-character snippets. Show truncation and preserve the original. |
| URL-fragment package (`sidepanel-chat-webui-handoff`) | Configured WebUI when an entry point actually invokes it | Existing allowlisted model/settings/context references and draft. Local IDs are origin-specific; card/persona/scene and full history are absent. It carries no established source account/server/workspace identity, so freshness alone cannot authorize continuation. |

The WebUI receiver ignores foreign local IDs without blocking transferable drafts, resolves any sufficiently identified server reference through its owner-bound loader, and never assigns a payload server ID directly as mutation authority. If the current payload cannot prove the necessary provenance, leave that reference unconfirmed and offer explicit destination history selection. Missing or unmatched document/RAG context is explained, not silently described as restored. Model preservation from the bounded package is claimed only if an independent existing tab-state path proves it.

At the source report **Opened — confirm transferred state in the destination** after a successful open, or an actionable open failure. The destination reports **Imported** or **Partially imported**, listing what arrived and what remains unavailable. An existing destination draft offers explicit insert/replace/cancel choices through the current conflict flow before consuming the package; `ifEmptyOnly` followed by URL cleanup is insufficient. Opening a tab is not an import receipt. Do not add a cross-surface acknowledgement protocol to imply source confirmation.

An expired payload, destination mismatch, failed open, or unavailable context preserves the source. Active streaming remains in its source panel; do not imply the generation moved or silently re-send it. Restore character behavior only from an owned acknowledged session; unsaved card/persona/scene fields absent from a payload are not claimed transferred. Retain current freshness/validation and avoid credentials or full histories in URLs. If a mechanism cannot carry a field safely, name the limitation and offer the source/full-extension path; new portable identity or cross-origin transfer contracts require a separate architecture assessment.

## Stages and validation gates

Each stage delivers a reviewable behavior change. This is a design sequence, not the detailed implementation plan. After design review, write the unique implementation plan, then use the repository's TDD, review, security, and verification workflow. Refresh the latest dev baseline and record it at each stage. Preserve red-first reproductions and turn known-defect wrappers into ordinary assertions only after independent behavior checks.

### Stage 1 — Saving and safe conversation transitions

**Task:** TASK-13454. **Status:** In Progress.

**Goal:** dependable start, saving, new-chat mode, and character ownership. Findings 1, 2, 8, plus the persistence prerequisite for finding 4.

**Success criteria:** acknowledged statuses including unknown creation; standard local chat reachable; fresh character transition with complete preflight; supported greeting-only creation with owned rows/model settings; explicit mode semantics; successful and failed transitions preserve source drafts.

**Tests:** unconfigured first visit → existing connection/provider setup → ready; create → name → reload → new chat → find → reopen for local/server; local standard → new character → greeting/first turn; selected alternate greeting and rapid repeated start; greeting before any user turn with provider-call count zero; ambiguous default provider; unsupported scene/prompt/files/tools/settings before transition; late create/load acknowledgement; lost create and append responses then reload/retry; account/workspace switch; save failure/reconnect; successful/failed/cancelled transitions with drafts and temporary content at every entry point.

**Source areas:** `usePersistenceMode`, `usePlaygroundPersistence`, `pending-chat-promotion`, `useClearChat`, `useHistorySelection`, `useChatActions`, `native-history-character-send`, greeting hook/picker, existing create client, `ChatHeader`, shared sidebar and compact route. Reuse current persistence transactions/recovery.

### Stage 2 — Focused composer and reachable history

**Task:** TASK-13455. **Status:** In Progress.

**Goal:** simplify everyday setup and make every relevant saved chat findable. Findings 3, 10; navigation portion of 1.

**Success criteria:** Casual disclosure with recognizable active settings; Pro capability retained; history-first sidebar with retained expansion; distinct command/history search; complete paged active-character sessions.

**Tests:** Casual/Pro toggle preserves draft/settings; keyboard access to every moved action and return to a visible trigger; collapsed/reopened sidebar; local/server query and duplicate titles; server outage, stale/partial pages, and filtering do not hide cached mirrors; local DB failure is unavailable rather than empty; more than five interleaved sessions; page boundaries and large histories.

**Source areas:** `ComposerToolbar`/overflow, `ChatSidebar`, `LocalChatList`, `useLoadLocalConversation`, `useServerChatHistory`, `CharacterChatSessionsPanel`, existing command palette.

### Stage 3 — Character opening, terminology, and scoped setup

**Task:** TASK-13456. **Status:** In Progress.

**Goal:** understand and control character behavior and the next turn. Findings 4, 6, 7; presets and next-turn summary enhancements.

**Success criteria:** visible non-mutating character greeting preview; accurate identity/context concepts; field-to-scope support map, staged/live commitment and explicit Apply results; reusable setup preview; truthful context provenance/capability state.

**Tests:** greeting/no-greeting/long-greeting preview leaves transcript/request history unchanged; own opening vs greeting verifies stored rows/context; repeated click/switch during persistence; imported PNG/JSON card; inherited worldbook/dictionary; selected vs activated lore; preset modify/apply/cancel; local settings success/server failure or unknown outcome; partial retry does not repeat acknowledged operations; held PUT across owner/account/workspace/view changes; two conversations sharing a model and Cancel leave defaults isolated; nested disclosure focus, dirty-close and error focus; scene/prompt/context restoration and unsupported-native recovery; historical isolation once TASK-13160 supports native resume.

**Source areas:** `PlaygroundChat`, `ChatGreetingPicker`, `RolePlaySetupDrawer`/preview, roleplay state/scene/compatibility helpers, `useChatSettingsRecord`, conversation/model settings, saved setups and prompt-assembly preview.

**Dependency:** TASK-13160 owns the server contract for unsupported native extensions. Deliver other client improvements independently; keep its consuming enhancement open until supported. Do not hold unrelated accessibility repairs behind this backend dependency.

### Stage 4 — Accessible narrow layouts and extension continuity

**Task:** TASK-13457. **Status:** In Progress.

**Goal:** equivalent usable controls and identifiable conversations across surfaces. Findings 5, 9.

**Success criteria:** viewport-safe dropdown, accessible names/semantics/contrast/focus, compact title/status, accurate destinations and carried state.

**Tests:** 320/390px, 200% zoom, desktop, both themes; keyboard and VoiceOver/manual screen-reader walkthrough; targeted axe; model popup geometry; long title/labels; each of the three transfer mechanisms with actual destination URLs and supported field matrix; nonempty destination draft insert/replace/cancel; bounded/truncated drafts; cross-origin local IDs, missing context, unconfirmed provenance and scoped server/local histories; expired/mismatched payloads; source Opened vs destination Imported/Partial; native Chrome docking/reopen.

**Source areas:** `ChatModelSelectorDropdown`, `ComposerTextarea`, greeting/history inputs, layout landmarks, shared color/runtime tokens, `SidepanelHeaderSimple`, `ControlRow`, compact route/sidebar, `sidepanel-chat-handoff`/import hook, selected-history expansion, and `sidepanel-chat-webui-handoff`/WebUI receiver.

**Scheduling:** after Stage 1/2 state/navigation contracts; may proceed while Stage 3's external backend dependency remains open.

### Stage 5 — Complete journeys and evidence-based polish

**Task:** TASK-13458. **Status:** In Progress.

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
| 4 Character greeting hierarchy | 1, 3, 5 | Non-mutating visible preview, supported greeting creation and at-most-one correct-session row, including unknown outcomes |
| 5 Responsive and accessibility defects | 4, 5 | Popup geometry, names/landmarks, contrast, keyboard and manual screen reader |
| 6 Identity, scene, context language | 3, 5 | Accurate preview across inherited and pinned context; task comprehension review |
| 7 Editing scope/commit semantics | 3, 5 | Field-to-scope support; local/server acknowledgements; partial retry/cancel/navigation; no unintended library/default changes |
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

This branch is isolated from the dirty primary checkout and from the completed audit branch. Written design approval is recorded, and Stage 1 application work is underway in reviewed incremental deliveries. Detailed plans follow the corrected contracts and accepted D7 reconciliation above. Completion is evidence-based by stage; unmet backend capabilities, native-shell checks, and human-study checks remain explicitly identified rather than disappearing from scope.

## Live opening contract correction — TASK-13454.13

Current dev8aeddeee36 is integrated at88a2d65aee after the passing frontend opening checkpoint a9822ac3a0 and notification fix1ba4415c2c. Live non-Sync creation exposed a canonical role gap: the transactional factory seeds the character display name into sender, while H1 uses sender as the protocol role. The assistant-only qualifier correctly refuses that row; known-ID reopening incorrectly attributes it to You. One create and zero provider calls were observed, and the original multiline draft remained recoverable.

Correct only new factory-generated seed messages to sender assistant and explicit null parent; keep the accepted raw greeting unchanged. Reuse the existing owner-checked unique root-chain H1 interpretation. Do not fabricate user admission/settlement, normalize arbitrary named senders, rewrite older rows, or claim Sync support. Both production callers use the shared factory without other initial messages; test module startup/reopen as well as actual API create/messages/H1. Existing qualified character labels remain; immutable historical display names are a separate limitation because API identity currently reads the live card. No metadata/display framework is added to this bounded protocol fix.

The actual preview/alternate choices do not mutate history. Own-message focus preserves exact input and trailing spaces. Opening actions fit390px and320px without horizontal overflow; screenshots are in /private/tmp/chat-ux-fresh-opening-390.jpg and /private/tmp/chat-ux-fresh-opening-320.jpg. Complete keyboard/zoom/screen-reader/package and other all-entry-point journeys remain required parent checks.

Verified correction: relevant Python suites292 passed, actual module reopening passed, Bandit0 and no new Ruff diagnostics; both independent source reviews found no blockers. One explicit live save produced one owned settled assistant root with exact raw text and zero provider/user-turn requests. Cold reload returned the same chat and exact focused draft (/private/tmp/chat-greeting-canonical-owned-proof.json; /private/tmp/chat-ux-saved-greeting-reloaded.jpg). TASK-13454.11 also qualifies the canonical H1 message id for display-only greeting projection; no synthetic server ID/version or raw-content mutation is introduced. Its real formatter-to-renderer regression failed before that correction and now passes with the related checks (3suites32); both type gates pass and no new frontend lint diagnostics are introduced. User placeholders remain visible when no user display name is configured.

## Explicit model selection checkpoint — TASK-13454.14

Checkpoint `61da94ba19` makes actual provider/model selection and settings ownership truthful across full, Research and compact chat. Catalog matches cannot qualify a providerless choice; canonical menu actions retain the selected provider; owned aliases and overrides preserve their settings. Deliberate settings-dialog A→B changes activate B before edits and Save. Native readiness follows the unchanged strict preflight route. Latest dev is integrated in `f460db867c`, preserving both chat Restore/Notes organization and upstream bounded Media routes with their negative method/path tests.

Post-merge verification: 421 relevant regressions pass, both client type gates pass, no new touched ESLint diagnostics, diff checks pass, and both independent source reviews are clear. Pre-merge live fixture evidence shows Choose provider without Current, then an explicit Custom OpenAI choice with exact draft and focus retained. The post-merge browser check preserves explicit provider identity and exact multiline/trailing-space text (`/private/tmp/chat-model-latest-dev-choice-draft-retained.jpg`); its fixture shows No chat models configured, so positive readiness and focus return are not claimed for that check. Packaged extension, assistive-technology and real-provider generation proof remain separate stage gates.


## Compact character selection checkpoint — TASK-13503

Checkpoint `220ca10129` keeps the source chat, assistant overlay, exact draft and resume marker while a tracked picker is opened or cancelled. The real picker delegates preparation to the compact form: supported native configuration, exact draft retention and accepted navigation precede the acknowledged identity write and targeted destination reset. Text/image edit revisions, picker intents, account/source changes and connection changes retire delayed work. Response settings retain their existing scope. Unavailable saved-persona behavior is explicit and preserves the source.

The shared owned preference writer now acknowledges only the current accepted write. Its temporary per-owner queue baseline prevents optimistic/watched candidates from becoming rollback state; private-key compensation cannot establish a history or server owner, and cannot overwrite a newer same-key choice. Real WebUI and installed Plasmo hook tests using the test Storage shim cover watch-before-ACK, separate-hook cancellation, queued-write rejection, warm/cold same-owner logout/login and newer-choice preservation. Failed rollback reports an explicit error and preserves the accepted RAM baseline for retry; it does not claim durable restoration while storage IO fails. Clear overlay respects newer intents, source changes and unmount. Verification failure has a visible account Retry without fetching an unqualified private catalog.

Latest dev integration `153bc4e00d` retains both the initial-chat setup type and upstream bounded cache in the sole import conflict. Post-integration evidence: 495 combined chat/cache/PDF/diff checks plus 68 request-scope/security checks pass; WebUI types, extension compilation, official English locale sync, no-new touched/integration lint and diff checks pass. Bandit is not applicable to this TS/TSX/JSON/documentation scope. Both independent source reviews cleared the compact fixes; integration spot-checks preserve the account/request/cache contracts.

This closes the bounded compact selection unit. Library and Quick Chat entry, full setup/history improvements, narrow/native extension continuity and complete staged acceptance remain open. The callback adapter is source-boundary evidence: its mocked retention/H1 context does not establish actual whole-form hydration or native modal focus. Physical browser-tab persistence, packaged extension, assistive-technology and real-provider generation remain separate parent-program gates.


## Character Library entry checkpoint — TASK-13504

Reviewed checkpoint `445847c992` and latest-dev merge `34f678d9b1` keep current/new-tab Library actions as route intent. Opening or dismissing a blocker leaves source identity, transcript, owner, exact draft and model/prompt settings intact. The destination verifies the real owned card and reuses native preflight, exact draft retention, accepted navigation and the acknowledged identity writer before releasing source ownership. Missing or rejected setup offers explicit recovery; newer input, account, route, settings or source changes retire delayed preparation. Accepted unsaved route commands are consumed, and a saved source URL retires using its pre-reset identity.

Pending entry blocks central submission, direct queue Run and automatic drain. The existing assistant action revision retires held queue preparation even when setup appears and disappears; source queue items remain available for deliberate review with localized explanation. No new transition or queue framework is introduced.

Final merged evidence: 29 suites and 902 tests pass; both client type gates pass; official English locale sync, no-new precheckpoint touched lint and diff checks pass. Remote dev `7226596cca5b875ad6e121a15133f69600152258` was verified unchanged after the checks. Both independent source reviewers cleared this bounded change and did not run checks. Bandit is not applicable to this TS/TSX/JSON/documentation scope.

Actual Form/composer tests include the real draft hook over acknowledged Map IO. Actual root/H1/Core coordinator tests explicitly stub destination Form acceptance. These separate proofs do not establish whole root plus Form native composition, private cold-tab durability, native popup/focus, packaged-extension or assistive-technology behavior, or positive provider generation. Quick Chat creation/promotion, full setup/history improvements, narrow/native continuity and complete staged acceptance remain open; this checkpoint does not close the all-ten program.


## Character Quick Chat recovery checkpoint — TASK-13505

Checkpoint `fb0cf6ca79` uses an independent verified account/controller and the existing native creation, admission, settlement and recovery protocol. Its unique private draft remains separate from the full-chat destination. Current model/provider choices are explicit. Popup/account/connection replacement retires delayed preparation, while a qualified saved owner keeps its independent lifetime. Unknown creation or turn outcomes require Review rather than replaying a POST. Closing does not implicitly delete a saved chat.

Saved continuation qualifies H1 and accepts navigation before closing. Nonempty text requires acknowledged exact retention and an explicit Copy action; destination text and metadata remain intact. Cold discovery attaches known creation records, and original unresolved text is distinct from a settled retained next message. A create ACK is shown as Chat already created even if history qualification fails. Preparation changes and foreign ownership stay fenced; current owned qualification errors identify the actual failed read.

The opt-in owned HTTP fixture passes with two turns, four canonical roles, one root, two selected members on turn two and exact cold NEXT input. The verified principal, HTTP/SSE bridge and profile/bookmark IO use declared adapters; it does not prove the real account watcher, progressive SSE timing, physical IndexedDB durability or packaged extension behavior. The runtime's chat/H1/character source matches current dev; two unrelated runtime files differ. Existing shared send sanitizers trim outer whitespace, so successful saved messages are not claimed byte-for-byte identical.

The actual browser created a chat and then failed history qualification before sending a provider turn. Its original full-chat draft survived Library navigation, and its exact Quick Chat opening remained reachable through Recover drafts after popup retirement. Explicit saved-chat Review failed without clearing either text or recovery. Native successful browser generation is still unverified. The same walkthrough exposed an empty Library after failed reads and a Saved mode badge beside pending recovery. Their UI causes can be reviewed independently; the browser transport cause is not yet established. Both source reviewers clear the bounded recovery corrections. The39-file relevant matrix plus five-suite final reruns verifies1180 passes and one opt-in skip; the owned opt-in HTTP check passes separately. Final WebUI types, extension entrypoint/config compile, no-new touched lint, English locale agreement and diff checks pass. The bounded task is complete; no whole-program completion is implied.


## Honest header mode and Library failures — TASK-13506

The shared header now reuses the existing History on/Temporary mode copy. The temporary flag establishes whether new messages are eligible for history; it does not establish a successful write. Temporary help also acknowledges retained device drafts. This change introduces no persistence tracker or save behavior. Related open PR3147 was reviewed as a reference; it is not modified or incorporated by this checkpoint.

Failed main Character Library reads now reject after the existing notification, reaching the existing error/Retry renderer. Successful empty results retain onboarding, and supported legacy route fallback remains available. The original read failure remains identifiable. The production query client retains its existing one-retry policy; the new actual-query tests use IO doubles and disable automatic retries for deterministic state checks. They do not establish broader account/cache ownership or network timing.

Four behavior checks failed before the change. After the fix,55 checks in five query/header/renderer/mode suites pass, plus four targeted Manager fallback/error regressions. Both independent source reviewers clear the bounded changes. Final WebUI types, extension entrypoint/config compile, no-new touched lint and diff checks pass. This bounded task is complete. The real browser header displays History on beside retained acknowledged creation, and Library reads currently display all four owned cards. A failing post-change browser read was not imposed, and native successful generation/physical storage/assistive/packaged-extension gates remain open.


## Accessible shared model picker — TASK-13507

The existing Dropdown now uses viewport bounds, wrapped controls/rows and scrollable content. Search and Sort retain accessible names; the trigger advertises its actual menu and retains the full provider/model name beside readiness warnings. The native multiline textarea no longer exposes unsupported expansion semantics. Selection, favorite storage, exact unsent input and existing composer behavior remain intact. No popup framework or dependency is added.

Keyboard checks cover both control variants, nested Sort, Search/menu/Help Tab reachability and departure, Escape, Favorite Enter/Space without model selection, replacement Favorite focus as unselected rows move between groups, and native row selection. Opening focus uses the repository's cancellable frame pattern. Native trigger clicks close once without premature blur/reopening. The actual full Form consumes an external return request on departure and qualifies/cancels deliberate delayed return on changed focus, reopen or unmount.

Source failures were reproduced before each correction, including the retained default-motion opening-focus race, Favorite group movement, actual Form external click/Tab return and trigger pointer ordering. The final nine-suite run passes182 checks (/private/tmp/chat-accessible-model-final-regressions.log). WebUI shared types, extension entrypoint/config compilation, official English locale mirror agreement, no-new touched ESLint and diff checks pass. Both authorized independent source reviewers clear the final changes; they did not execute tests. Remote dev94854ca3db6ec3eaa0c27b4e9537bdd47d222611 is unchanged and remains integrated at292ea526d1. Bandit is not applicable to TS/TSX/JSON/docs; no Python implementation was changed.

The Form focus harness uses the actual Form callback and installed Dropdown/Tooltip, with declared model-state, storage, other controls and service doubles. The browser AX walkthrough confirms Search/Sort names, open focus and Escape return while retaining the exact draft. It does not establish native geometry: the current IAB reports0x0 viewport and cannot capture screenshots after documented resize/reset/presentation attempts. Actual320/390px,200%zoom, visual motion/themes, touch, VoiceOver, packaged extension, positive native browser generation and complete handoff journeys remain parent gates. This closes the bounded model-picker unit, not the all-ten program.


## Explicit sidepanel destinations and copy outcomes — TASK-13508

Existing native extension, configured WebUI callback and owned selected-history expansion now have destination-specific names and descriptions. The bounded action is named **Copy draft & page references**: the actual producer carries draft text plus page/document titles and URLs, excluding page content, history and chat settings. It uses the existing short-lived local package and opaque URL identifier; no transfer fields, ownership authority or routing protocol are added. A successfully opened page is labelled Opened, never Imported, with source retention and a request to review the copy there. Outcomes remain visible outside the closed More popover.

The two independent reviews caught overstated page-context copy and lost active-page read failures. The existing Form producer now lives in the existing context helper, allowing its real combined-document/read-failure branch to be exercised at the browser IO boundary. It marks missing requested references through the existing partial flag, preserved by the shared storage builder. Surviving references produce Partial copy; a context-only failure offers Retry without creating/opening an empty copy. Size shortening and blocked opens retain explicit outcomes. No body content or credentials are added to the URL.

The final13-suite matrix passes141 checks (/private/tmp/chat-handoff-copy-final-regressions.log), including route branches, opaque URLs, duplicate-click protection, actual extracted producer, existing storage read-back, import contracts, installed AntD Popover closure and related Form checks. The broader first run caught a stale full-Form source-extraction test dependency; the harness now executes the actual getSubmitDisabledReason callback with its declared blocker ref instead of leaving that callback undefined. Assertions remain intact; no production dispatch behavior was changed by this harness repair. Browser/storage/other control IO doubles and source-extraction boundaries are declared: these checks do not prove native navigation, physical storage, assistive announcements or destination load/import ACK.

Official English locale synchronization was narrowed to13 intended generated keys and unrelated mirrors restored byte-for-byte. Final WebUI shared types, extension entrypoint/config compilation, no-new touched ESLint and diff checks pass. Bandit is not applicable to this TS/TSX/JSON/docs scope. Both independent source reviews clear the bounded corrections; they inspected test evidence and did not execute it. Inspected overlaps with PR3152 full-title rename, PR3147 persistence and PR2738 quick-ingest remain independent references, without integration. Remote dev94854ca remains current and included through292ea526d1.

Source account/navigation retirement, receiver ownership/provenance/conflict/freshness/ACK, cross-origin local-ID limits and qualified server references remain parent continuity work. Compact full-title editing, packaged extension, native geometry/zoom/VoiceOver and complete successful browser journeys also remain open. This checkpoint closes a bounded UI delivery, not the all-ten program.


## Exact copied drafts and fresh conflict choices — TASK-13509

The existing receiver preserves all unsent whitespace and treats a whitespace-only destination as an existing draft. Explicit Insert and Replace reread the short-lived package before applying it; the pending choice retires on route departure, unmount or destination edits. The actual form's synchronous revision getter also detects edit followed by Undo, including a batched return to the same text. Cancel remains available while the read runs; duplicate activation issues one read. No transfer format, server mutation or ownership authority is added.

Explicit choices opt into read-error propagation from the existing package reader. Missing, malformed, expired and consumed packages retain their unavailable behavior; the reader's default callers retain their existing contract. The Retry regression exercises the actual reader and hook with a fresh valid package and declared storage IO rejection, then a successful retry, preserving the destination draft and URL while the read fails. All three import/submission harnesses now use the real useSimpleForm rather than duplicating form behavior.

The review additions fail four checks before correction and pass afterward. Final13-suite verification passes158 checks (/private/tmp/chat-copy-import-final-regressions.log); WebUI shared types, extension entrypoint/config compile, no-new touched ESLint and diff checks pass. Four intended English keys were synchronized using the official locale script and unrelated mirrors restored byte-for-byte. Both independent source reviewers clear the bounded changes; they inspected evidence and did not execute tests. Bandit is not applicable to TS/TSX/JSON/docs. Remote dev94854ca3db6ec3eaa0c27b4e9537bdd47d222611 remains unchanged and integrated through292ea526d1.

The hook/router/form/reader tests use declared storage and unrelated service IO doubles. They do not establish native storage durability, full root/Form hydration or full browser/packaged-extension composition. Initial auto-import and cold draft readiness, source/destination account/view ownership and unknown provenance, source action retirement, cross-origin IDs and qualified server references, imported/partial receipts, compact full-title editing, native geometry/zoom/VoiceOver and complete successful browser journeys remain parent gates. This closes the bounded explicit-choice unit, not the all-ten program. The own completed plan was removed as required.


## Authoritative model discovery and account/setup refresh — TASK-13511

Shared discovery converted metadata failures to cache or empty, cache scopes represented credential presence rather than authorization, and QueryClient retained successful empty setup results. Retired direct reads could borrow a newer owner's result even during configuration lookup. Native cookie reconciliation cleared its own in-flight pointer. Failure-first checks reproduced these shared causes.

The existing service now preserves strict failure semantics alongside advisory offline reads, scopes cache to the existing authorization revision, fences configuration/network retirement, and coalesces raw requests with each caller's own error policy. A shared catalog hook follows existing setup/account invalidation and withholds failed or pending results from readiness and dispatch. Main/compact pickers, readiness, queue dispatch and setup consumers reuse it. Models refresh, onboarding, Quick Chat helper/popout and character generation offer explicit Retry; successful empty remains a configuration state. Existing H1 ownership/provider validation and transport security remain intact; no dependency was added.

Final18-suite verification passes296 checks (/private/tmp/chat-authoritative-final-regressions.log). Shared WebUI types, extension entrypoint/config compilation, no-added touched ESLint, diff and official English locale mirror checks pass. Both authorized independent source reviewers clear this unit; neither ran the final checks. The first concurrent run had two Settings reset timeouts; both pass in isolation and the final sequential suite passes unchanged. The older compact offline AST harness now supplies the already-present greeting preparation input. Bandit is not applicable to TS/TSX/JSON/docs. The completed temporary three-stage plan is retired.

One owned IAB tab completed a selected greeting plus synthetic user turn and mock-provider reply, then reopened all three messages through Server history. Explicit Retry recovered a visible metadata failure without replaying the turn. Saved server identity: a9ca478f-ba5c-446a-a8d2-cb292f0a1026. The raw read with render_placeholders=false preserves exact placeholders/outer whitespace and complete user/reply text. Default GET expands placeholders using User; H1 remains raw. The UI's blank display-name preference leaves {{user}} visible, and Markdown naturally renders paragraph whitespace. A consistent display-name fallback is a Stage3 usability follow-up, not a corrupted-save finding.

The earlier standard completion400/Failed to fetch remains unproven. Native screenshots/geometry, packaged extension/docking, VoiceOver, narrow widths/zoom/themes, receiver cold hydration/owner retirement and D7 prerequisites remain open. This closes the catalog root repair, not the all-ten program. Latest origin/dev1fc353 was re-fetched unchanged after checks.


## Stage 4 destination ownership root repair checkpoint — TASK-13512

The copied-draft receiver now waits for the actual owner-bound composer to finish restoring. Its initial and explicit reads retain the existing owner and selected-history fence plus draft IDs/address/revision. Cancel uses the same displayed ownership, and the shared consume service rechecks it after its storage read, immediately before removal. Imported page context retires when the destination changes without clearing unrelated text.

Canceled discovery can restart after callback rerenders and StrictMode. A displayed conflict remains explicitly reviewable after readiness recovery, same-URL navigation and history-fence recapture; an old held choice cannot keep a fresh choice disabled or affect it later. The import guard covers destination ownership rather than unrelated assistant preparation or file-retrieval context. The Form previews the exact copied plaintext before Insert/Replace/Cancel. Two English keys were synced officially; unrelated locale bytes were restored.

Failure-first checks reproduced cold restoration, held read retirement, context carryover, the canceled-read latch, stale Cancel, readiness/route recovery, overbroad setup guards and held-choice recapture. Final latest-dev18-suite verification passes567 checks, with66 catalog-service checks separately. WebUI types, extension entrypoint/config compilation, no-added touched lint, locale and diff checks pass. Both authorized independent reviewers clear the bounded source; the merged account/cache authority integration is also source-reviewed without a blocker. No Python implementation changed; Bandit does not apply to this TS/TSX/JSON/docs scope. Implementation commitac925515a3; latest-dev integration2f7fbb8653. The completed temporary plan is retired.

The receiver tests use the actual simple form and Router with declared storage IO doubles. Composite checks use the real composer, registry and account watcher with a declared principal resolver; production Form guard/JSX checks extract the actual source and do not mount the full Form. This evidence does not establish physical native storage, browser geometry, packaged extension/docking or full end-to-end transfer. Source action retirement, payload account provenance/unknown owner review, cross-origin qualified references, receipts and D7/backend gates remain parent work. The all-ten remediation is still in progress.


## Stage 4 source ownership root repair checkpoint — TASK-13519

Copy draft & page references now waits for the actual owned composer to restore and captures the source shown in the current render: composer/account, draft address and monotonic revision, history IDs, input revision and H1 selection epoch. A stale displayed control cannot acquire a new owner after synchronous New, edit/Undo or history selection before React rerenders. The existing Form supplies these fences; no second owner registry is introduced.

The shared ControlRow action retains an attempt identity across page reads, package creation and native/fallback opening. Source retirement releases old pending state without affecting a newer attempt, prevents stale opening/fallback/feedback/focus, and keeps the original text. Completed attempts retain ownership of their visible Opened/Partial result until the source changes, while allowing another explicit Copy. The existing package service checks ownership after cleanup, storage write and read-back; a retired creation removes its unique unpublished copy. Cleanup remains best effort under storage failure. Calls already issued to storage or native tab creation cannot be undone.

Failure-first checks reproduce five ControlRow races, three storage boundaries, the missing production capture, three stale-render starts, an H1 epoch change before IDs/commit, and receipt loss after a harmless callback rerender. The final20-suite matrix passes600 tests. WebUI types, extension entrypoint/config compilation, no-added touched lint and diff checks pass. Both authorized independent reviewers clear the bounded final change. Latest remote dev remains7ba48f251ec47a1e0bb680f49f9b7d86ec2b988d and is an ancestor of this branch. No Python implementation changed; Bandit does not apply to this TS/TSX/docs scope. The own completed temporary implementation plan is retired.

ControlRow checks mount its real action with declared page/storage/open IO doubles. Production Form capture checks extract the actual callback and use the real chat store, including a real H1 controller starting a load before React commit; they do not mount the full Form. Existing real-composer/account restoration suites run in the matrix. Native storage/opening, packaged extension/docking, browser geometry and VoiceOver remain unverified. Payload provenance/unknown-owner review, cross-origin references, destination receipt details, compact title and D7/backend prerequisites remain parent work. The all-ten remediation remains In Progress.
