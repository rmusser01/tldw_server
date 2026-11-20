# Document Workspace Conversation Tabs Implementation Plan

> **For agentic workers** Use superpowers:executing-plans and test-driven development.

**Goal** Give the Chat sidebar independent browser-scoped conversations for the open document.

**Design** [Approved conversation tabs design](../specs/2026-11-20-document-workspace-conversation-tabs-design.md)

**ADR assessment** [ADR-049](../../ADR/049-chat-history-selection-ownership.md) governs owner-bound reads and writes. The browser reuses the current document index and local history identity. That requires no new ADR. TASK-15586 tracks cross-device discovery.

## Stage 1 — Verify and restore saved conversations

**Status** Complete

**Goal** Query `getAllHistoriesByDocId(docId, requestScope)` for ordered, owned rows and open the selected ID.

**Success criteria** History from another account or document never populates the list or the transcript. The newest owned chat opens by default. A failed read offers retry.

- [x] Extend `recent-chat-ownership.test.ts` and `useDocumentChat.ownership.test.tsx` with a mixed-owner list, a deferred load, and a mismatched history header. Observe red using `bun run test:run` from `apps/tldw-frontend`.
- [x] Scope the query in `db/dexie/helpers.ts`. Verify loaded headers and use the shared message formatter in `useDocumentChat.ts`.

## Stage 2 — Isolate chat state and pending replies

**Status** Complete

**Goal** Keep each conversation's input and document context while the workspace is open. Adopt the first saved ID in its originating tab.

**Success criteria** Switching cancels the reply, invalidates its visible source, and leaves retained messages with the originating conversation. Rapid selection and account changes reject stale reads and writes.

- [x] Add hook tests for independent drafts, first-save adoption, reuse after remount, and late persistence from an inactive tab.
- [x] Keep the per-document cache in the verified workspace. Have `useDocumentChat` expose `selectConversation`, `newConversation`, `retryConversation`, and the active conversation's draft, key, and load status. Report completed document-history writes through `useMessageOption` and `useChatActions` so a stopped turn can still identify where it saved.
- [x] Run the hook, Dexie ownership, and shared response-dispatch suites. Confirm the same document supplies RAG when switching chats.

## Stage 3 — Render, review, and verify

**Status** In Progress

**Goal** Give the sidebar keyboard-accessible, horizontally scrollable conversation tabs, a labeled New chat control, and loading and retry states.

**Success criteria** Keyboard and pointer selection work at drawer width. The selected tab remains identifiable and messages restore without duplicated saved identities.

- [x] Add a red sidebar interaction test. Use existing workspace tab styles, roving focus, and readable title truncation in `DocumentChat.tsx`.
- [x] Run the targeted suites, frontend typecheck, scoped lint, and full diff review. Record the results in TASK-15584 and commit the implementation independently.
- [ ] Verify overflow at desktop and drawer widths in a browser. Check both follow-ups before updating the open Dev PR or preparing Hosted.
