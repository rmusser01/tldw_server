# UAT111 / TASK13260.51: normal saved Chat Note backlink

## Finding

The menu dispatch and normal-chat navigation support exist. The earliest demonstrated blocker is the Chat unsaved-work guard: normal saved user messages can lack `serverMessageId` (UAT103), so Notes classifies them as unsaved and returns before loading/navigating to the linked conversation.

This is a source-confirmed path reproduced through the actual Note menu with the earlier captured same-conversation mirror. The runner has since selected a new character chat and did not mutate browser state to reconstruct the original attempt. Therefore there is no exact contemporaneous blocked-row capture or retained warning toast from the two live clicks. Keep that distinction explicit.

## Live evidence

- `output/playwright/cycle4-full-uat-2026-09-16/multi/chat-note-save.json`: Note `b89c7b7d-daa6-438a-9fa3-ef4bd6b8b9f9`, conversation `a9feabc7-38aa-4b06-b08d-0b673b069819`, message `22c2476d-06f7-4262-991b-3d177cbd9e4d`.
- `chat-note-backlink-opened.txt`: remains `/notes`; the Note shows the correct public pirate answer and conversation/message association. Runner reports two menu clicks, no new tab/navigation, only Note GET200.
- Earlier `uat103-read-only-mirror.txt`, same active conversation/history: two saved local user rows have no `serverMessageId`: `pa_c2f0-84a4-fe2-10dd` (71 chars) and `pa_c743-b596-7e5-f183` (66 chars). The five server rows include their canonical persisted counterparts. Subsequent new normal turns use the same missing-user-ack path, but this report does not invent a later live mirror capture.

## Exact source path

1. `apps/packages/ui/src/components/Notes/NotesEditorHeader.tsx:391-393`: menu key `open-conversation` calls `onOpenLinkedConversation`.
2. `NotesEditorPane.tsx:588-590`: forwards to `openLinkedConversation()`.
3. `NotesManagerPage.tsx:1924-1935`: validates backlink ID, then rejects streaming/processing or any nonempty non-greeting message without `serverMessageId`, showing `Finish or save the current chat before opening the linked conversation.` The return precedes scope acquisition, Chat GET, transcript GET, assistant selection, and navigation.
4. The same predicate also aborts a pending backlink read if such messages appear (`1942-1945`). A durable repair must preserve both initial and late guards.
5. On the accepted path, `1966-1970` supports neutral normal chats by selecting `null` when there is no tracked assistant. `1981` calls `selectServerChat(chat)`.
6. `hooks/chat/useSelectServerChat.ts:111-116` navigates `/chat` in Web (or `/` in the extension sidepanel). No character-only navigation condition exists.

## Private behavioral probe

`/private/tmp/uat111-note-backlink-probe.config.ts` appends tests only in memory to the existing `NotesManagerPage.stage26.backlink-labels.test.tsx` fixture. It renders the actual Notes page/editor/menu, uses the real store and selection/navigation hook, and mocks its existing server/authority boundaries. Captured IDs/roles/acknowledgment metadata drive the first case; content is length-matched synthetic text because the guard only checks nonemptiness. This is not a full Chat pipeline or real browser test.

The focused run (`uat111-note-backlink-probe-red.log`) produced **1 expected failure, 2 passes; 13 existing tests were unselected by the test-name filter**:

- Captured saved-mirror shape: the real action emits the exact warning, requests no transcript, and never navigates. Expected `/chat` assertion fails.
- Acknowledged normal saved rows: real menu reaches `/chat`, preserves the conversation ID, and selects neutral assistant context.
- Genuinely unsent user draft: warning/no navigation/no transcript request, preserving the safety guard.

The separate full-probe run (`uat111-note-backlink-full-probe-red.log`) includes all original fixture controls: **1 expected failure / 15 passes**, including the existing delayed work/account guards. These totals overlap the focused run and must not be added together. No tests were disabled or edited on disk.

## Minimal repair boundary after full-run freeze

Coordinate TASK13260.51 with UAT103/TASK13260.44. Establish canonical user acknowledgment for successful normal saved turns and conservative reconciliation for already-created mirrors. Then the existing guard can distinguish confirmed saved turns from real unsent work. Do not simply remove the guard, ignore every user row, deduplicate by content, or permit overwrite whenever a server conversation ID exists: those changes can discard a genuine draft, including one attached to the same conversation.

Use the actual Note menu test as a downstream acceptance test alongside the real normal send → local persistence → reload/mirror regression. Preserve tracked-character positive controls, neutral normal context, same-tab navigation, Note dirty confirmation, failed/foreign Chat reads, account/server changes during scope acquisition/transcript fetch, and a genuinely unsent row arriving during the fetch. If upstream IDs solve this path, no Notes production change is required; any Notes guard change needs evidence that canonical acknowledgment alone is insufficient.

Native follow-up must save a real normal reply as a Note, click the actual More actions backlink, assert the exact conversation/transcript/neutral mode, then reload. Keep a truly unsent-draft negative control. This read-only diagnosis does not claim that acceptance.

## Scope / reproduction

No application/test-source/browser/runtime changes, inference, or commits. Only private probes/report and official task notes.

```sh
node apps/node_modules/.bun/vitest@4.0.18+08ee8852a9d25cb0/node_modules/vitest/vitest.mjs run --config /private/tmp/uat111-note-backlink-probe.config.ts -t UAT111
node apps/node_modules/.bun/vitest@4.0.18+08ee8852a9d25cb0/node_modules/vitest/vitest.mjs run --config /private/tmp/uat111-note-backlink-probe.config.ts
```
