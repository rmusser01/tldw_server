import { beforeEach, describe, expect, it, vi } from "vitest"
import type { HistoryInfo, Message } from "../types"

// Only IndexedDB's storage/transaction boundary is replaced; reconciliation is real.
const state = vi.hoisted(() => ({ histories: new Map<string, HistoryInfo>(), messages: new Map<string, Message>() }))
function table<T extends { id: string }>(rows: Map<string, T>) { return ({
  get: async (id: string) => rows.get(id),
  add: async (row: T) => { if (rows.has(row.id)) throw new Error("duplicate primary key"); rows.set(row.id, structuredClone(row)); return row.id },
  put: async (row: T) => { rows.set(row.id, structuredClone(row)); return row.id },
  delete: async (id: string) => { rows.delete(id) },
  update: async (id: string, changes: Partial<T>) => { if (!rows.has(id)) return 0; rows.set(id, { ...rows.get(id)!, ...changes }); return 1 },
  where: (field: string) => ({ equals: (value: unknown) => ({ toArray: async () => [...rows.values()].filter(row => (row as Record<string, unknown>)[field] === value) }) })
}) }
vi.mock("../schema", () => ({ db: {
  chatHistories: table(state.histories), messages: table(state.messages), modelNickname: {}, sessionFiles: {},
  transaction: async (_mode: string, _tables: unknown[], operation: (transaction: { abort: () => void }) => Promise<unknown>) => operation({ abort: vi.fn() })
} }))
vi.mock("../helpers", () => ({ generateID: () => `history-${state.histories.size + 1}` }))
import { acknowledgePromotedChatMessage, linkServerChatMirror, reconcileServerChatMirror, reconcileServerChatMessages, removeAcknowledgedServerMirrorMessage } from "../server-chat-mirror"

const history = (id: string, owner?: string): HistoryInfo => ({ id, title: "Cedar", is_rag: false, createdAt: 1, server_chat_id: "chat-1", ...(owner ? { server_scope_key: owner } : {}) })
const row = (id: string, historyId: string, content: string, serverMessageId?: string): Message => ({ id, history_id: historyId, name: "You", role: "user" as const, content, images: [], createdAt: 1, serverMessageId })
const incoming = (id: string, content: string, version = 1) => ({ id, serverMessageId: id, serverMessageVersion: version, role: "user" as const, isBot: false, name: "You", message: content, images: [], sources: [], createdAt: 1 })
const link = (ownerKey: string, extra = {}) => linkServerChatMirror({ chatId: "chat-1", title: "Cedar", ownerKey, ...extra })

describe("owned server Chat mirror", () => {
  beforeEach(() => { state.histories.clear(); state.messages.clear() })
  it("removes only the exact acknowledged server row after canonical deletion", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("target", row("target", "alice", "Repeat", "server-target"))
    state.messages.set("same-text", row("same-text", "alice", "Repeat", "server-other"))
    await removeAcknowledgedServerMirrorMessage({ historyId: "alice", chatId: "chat-1", localMessageId: "target", serverMessageId: "server-target" })
    expect([...state.messages.keys()]).toEqual(["same-text"])
  })

  it("rejects a mismatched canonical deletion receipt without changing the mirror", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("target", row("target", "alice", "Repeat", "server-target"))
    await expect(removeAcknowledgedServerMirrorMessage({ historyId: "alice", chatId: "chat-1", localMessageId: "target", serverMessageId: "server-other" })).rejects.toThrow()
    expect(state.messages.has("target")).toBe(true)
  })
  it.each(["unchanged", "edited"])("acknowledges the exact promotion source and preserves %s content", async kind => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("source", { ...row("source", "alice", kind === "edited" ? "Later local edit" : "Greeting"), role: "assistant" as const })
    state.messages.set("same-text", { ...row("same-text", "alice", "Greeting"), role: "assistant" as const })
    await acknowledgePromotedChatMessage({ historyId: "alice", chatId: "chat-1", ownerKey: "A", source: { ...incoming("source", "Greeting"), serverMessageId: undefined, role: "assistant" as const, isBot: true }, serverMessageId: "canonical", version: 1, signal: new AbortController().signal, isCurrent: () => true })
    expect(state.messages.get("source")).toMatchObject({ content: kind === "edited" ? "Later local edit" : "Greeting", serverMessageId: "canonical" })
    expect(state.messages.get("same-text")?.serverMessageId).toBeUndefined()
  })
  it.each(["owner", "history", "role", "ack", "cancelled", "target"])("rejects a promotion acknowledgement with changed %s", async kind => {
    state.histories.set("alice", history("alice", kind === "owner" ? "B" : "A"))
    state.messages.set("source", { ...row("source", kind === "history" ? "bob" : "alice", "Greeting", kind === "ack" ? "other-server" : undefined), role: kind === "role" ? "user" : "assistant" })
    const signal = new AbortController()
    if (kind === "cancelled") signal.abort()
    await expect(acknowledgePromotedChatMessage({ historyId: "alice", chatId: "chat-1", ownerKey: "A", source: { ...incoming("source", "Greeting"), role: "assistant" as const, isBot: true }, serverMessageId: "canonical", signal: signal.signal, isCurrent: () => kind !== "target" })).rejects.toThrow()
    expect(state.messages.get("source")?.serverMessageId).toBe(kind === "ack" ? "other-server" : undefined)
  })
  it("persists an exact acknowledged synthetic source ID while preserving a distinct equal row", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("distinct", { ...row("distinct", "alice", "Greeting"), role: "assistant" as const })
    const remote = { ...incoming("canonical", "Greeting"), role: "assistant" as const, isBot: true }
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [remote], localMessages: [{ ...remote, id: "original-local" }] })
    expect(state.messages.get("original-local")?.serverMessageId).toBe("canonical")
    expect(state.messages.get("distinct")?.serverMessageId).toBeUndefined()
    expect(state.messages.size).toBe(2)
  })

  it.each([
    { role: "user" as const, isBot: false, expectedName: "You" },
    { role: "assistant" as const, isBot: true, expectedName: "Assistant" }
  ])("uses a role-appropriate name for unnamed $role messages", async ({ role, isBot, expectedName }) => {
    state.histories.set("alice", history("alice", "A"))
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [{ ...incoming("saved", "Hello"), role, isBot, name: undefined }] })
    expect([...state.messages.values()][0]).toMatchObject({ role, name: expectedName })
  })

  it("links an unanswered user by exact client correlation while retaining equal-text drafts", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("local-q", { ...row("local-q", "alice", "Repeat"), images: [""] })
    state.messages.set("draft", row("draft", "alice", "Repeat"))
    const remote = [{ ...incoming("question", "Repeat"), metadataExtra: { client_message_id: "local-q" } }]
    const current = ["local-q", "draft"].map(id => ({ ...incoming(id, "Repeat"), serverMessageId: undefined, images: [""] }))
    expect(reconcileServerChatMessages(current, remote).map(item => [item.id, item.serverMessageId])).toEqual([["local-q", "question"], ["draft", undefined]])
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: remote })
    expect([...state.messages.values()].map(item => [item.id, item.serverMessageId])).toEqual([["local-q", "question"], ["draft", undefined]])
  })

  it("links an image-only unanswered user without treating another image as identity", async () => {
    state.histories.set("alice", history("alice", "A"))
    const image = "data:image/png;base64,owned"
    state.messages.set("image-user", { ...row("image-user", "alice", ""), images: [image] })
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [{ ...incoming("saved-image", ""), images: [image], metadataExtra: { client_message_id: "image-user" } }] })
    expect([...state.messages.values()]).toMatchObject([{ id: "image-user", content: "", images: [image], serverMessageId: "saved-image" }])
  })

  it.each(["missing", "other-id", "edited-content", "changed-image", "wrong-role", "duplicate-claim", "foreign-history", "already-acknowledged"])("preserves unmatched correlation work: %s", async kind => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("local-q", { ...row("local-q", kind === "foreign-history" ? "bob" : "alice", kind === "edited-content" ? "Edited" : "Repeat", kind === "already-acknowledged" ? "different-server" : undefined),
      images: kind === "changed-image" ? ["data:image/png;base64,other"] : [], role: kind === "wrong-role" ? "assistant" : "user" })
    const remote = [{ ...incoming("question", "Repeat"), metadataExtra: kind === "missing" ? {} : { client_message_id: kind === "other-id" ? "other" : "local-q" } }]
    if (kind === "duplicate-claim") remote.push({ ...remote[0], id: "question-2", serverMessageId: "question-2" })
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: remote })
    expect(state.messages.get("local-q")?.serverMessageId).toBe(kind === "already-acknowledged" ? "different-server" : undefined)
  })

  it.each([{ images: [] }, { images: [""] }])("recovers an explicitly parented legacy user with images $images and keeps equal-text drafts", async ({ images }) => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("local-q", { ...row("local-q", "alice", "Repeat"), images })
    state.messages.set("local-a", { ...row("local-a", "alice", "Reply", "answer"), role: "assistant" as const, parent_message_id: "local-q" })
    state.messages.set("draft", { ...row("draft", "alice", "Repeat"), images })
    const remote = [incoming("question", "Repeat"), { ...incoming("answer", "Reply"), isBot: true, role: "assistant" as const, parentMessageId: "question" }]
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: remote })
    expect([...state.messages.values()].map(r => [r.id, r.serverMessageId])).toEqual([
      ["local-q", "question"], ["local-a", "answer"], ["draft", undefined]
    ])
    expect(state.messages.get("local-a")?.parent_message_id).toBe("question")
    const current = [
      { ...incoming("local-q", "Repeat"), images, serverMessageId: undefined },
      { ...remote[1], id: "local-a", parentMessageId: "local-q" },
      { ...incoming("draft", "Repeat"), images, serverMessageId: undefined }
    ]
    expect(reconcileServerChatMessages(current, remote).map(r => [r.id, r.serverMessageId])).toEqual([
      ["local-q", "question"], ["local-a", "answer"], ["draft", undefined]
    ])
  })
  it.each([
    { localImages: ["data:image/png;base64,local"], remoteImages: [] },
    { localImages: [""], remoteImages: ["data:image/png;base64,remote"] },
    { localImages: ["data:image/png;base64,local"], remoteImages: ["data:image/png;base64,remote"] }
  ])("preserves anchored work when substantive images differ ($localImages / $remoteImages)", async ({ localImages, remoteImages }) => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("local-q", { ...row("local-q", "alice", "Repeat"), images: localImages })
    state.messages.set("local-a", { ...row("local-a", "alice", "Reply", "answer"), role: "assistant" as const, parent_message_id: "local-q" })
    const remote = [
      { ...incoming("question", "Repeat"), images: remoteImages },
      { ...incoming("answer", "Reply"), isBot: true, role: "assistant" as const, parentMessageId: "question" }
    ]
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: remote })
    expect(state.messages.get("local-q")).toMatchObject({ serverMessageId: undefined, images: localImages })
    expect(state.messages.size).toBe(3)
  })
  it.each(["missing-parent", "edited-user", "already-mirrored", "wrong-reply", "conflicting-parent"])("keeps ambiguous legacy work: %s", async kind => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("local-q", row("local-q", "alice", kind === "edited-user" ? "Edited" : "Repeat"))
    state.messages.set("local-a", { ...row("local-a", "alice", "Reply", kind === "wrong-reply" ? "other-answer" : "answer"), role: "assistant" as const, parent_message_id: kind === "missing-parent" ? null : "local-q" })
    if (kind === "already-mirrored") state.messages.set("canonical-q", row("canonical-q", "alice", "Repeat", "question"))
    const remote = [incoming("question", "Repeat"), { ...incoming("answer", "Reply"), isBot: true, role: "assistant" as const, parentMessageId: kind === "conflicting-parent" ? "other-user" : "question" }]
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: remote })
    expect(state.messages.get("local-q")).toMatchObject({ serverMessageId: undefined, content: kind === "edited-user" ? "Edited" : "Repeat" })
  })
  it("keeps identical conversation IDs in different owners' histories", async () => {
    state.histories.set("alice", history("alice", "A"))
    const bob = await link("B", { currentHistoryId: "alice" })
    expect(bob).not.toBe("alice")
    expect(state.histories.get("alice").server_scope_key).toBe("A")
    expect(state.histories.get(bob).server_scope_key).toBe("B")
  })
  it("adopts only the explicitly validated current legacy history", async () => {
    state.histories.set("legacy", history("legacy"))
    expect(await link("A", { currentHistoryId: "legacy", legacyHistoryId: "legacy" })).toBe("legacy")
    expect(state.histories.get("legacy").server_scope_key).toBe("A")
  })
  it("leaves an unvalidated legacy history untouched", async () => {
    state.histories.set("legacy", history("legacy"))
    expect(await link("A", { currentHistoryId: "legacy" })).not.toBe("legacy")
    expect(state.histories.get("legacy").server_scope_key).toBeUndefined()
  })
  it("does not repurpose an owned history from another conversation", async () => {
    state.histories.set("other", { ...history("other", "A"), server_chat_id: "chat-other" })
    expect(await link("A", { currentHistoryId: "other", legacyHistoryId: "other" })).not.toBe("other")
    expect(state.histories.get("other").server_chat_id).toBe("chat-other")
  })
  it("fills a nonempty old mirror, retaining canonical IDs across repeated reads", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("question", row("question", "alice", "When?"))
    const args = { historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [incoming("question", "When?"), incoming("answer", "08:30")] }
    await reconcileServerChatMirror(args)
    await reconcileServerChatMirror(args)
    const saved = [...state.messages.values()]
    expect(saved).toHaveLength(2)
    expect(saved.map(r => [r.serverMessageId, r.content])).toEqual([["question", "When?"], ["answer", "08:30"]])
    expect(saved[1].id).not.toBe("answer")
  })
  it("retains normalized metadata through mirror storage", async () => {
    state.histories.set("alice", history("alice", "A"))
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [{ ...incoming("answer", "08:30"), metadataExtra: { speaker_character_id: 4, mood: "calm" } }] })
    expect([...state.messages.values()][0].metadataExtra).toEqual({ speaker_character_id: 4, mood: "calm" })
  })
  it("adds missing replies without replacing an unsynced visible draft or a newer edit", () => {
    const result = reconcileServerChatMessages([
      { ...incoming("question", "Newer question", 3), id: "local-question" },
      { ...incoming("draft", "Unsent"), serverMessageId: undefined }
    ], [incoming("question", "Old question", 2), incoming("answer", "08:30")])
    expect(result.map(message => [message.id, message.message, message.serverMessageId])).toEqual([
      ["local-question", "Newer question", "question"], ["answer", "08:30", "answer"], ["draft", "Unsent", undefined]
    ])
  })
  it.each(["acknowledged", "canonical ID only"])("does not reappend an original answer already in a collapsed variant group: %s", identity => {
    const user = incoming("question", "Describe the icon")
    const original = { ...incoming("answer-1", "A speech bubble"), role: "assistant", isBot: true, parentMessageId: user.id,
      ...(identity === "canonical ID only" ? { serverMessageId: undefined } : {}) }
    const latest = { ...incoming("answer-2", "An icon with dots"), role: "assistant", isBot: true, parentMessageId: user.id }
    const grouped = { ...latest, variants: [{ ...original, serverMessageId: "answer-1" }, latest], activeVariantIndex: 1 }
    const result = reconcileServerChatMessages([user, original, latest], [user, grouped])
    expect(result.map(message => message.id)).toEqual(["question", "answer-2"])
    expect(result[1].variants?.map(variant => variant.id)).toEqual(["answer-1", "answer-2"])
  })
  it.each(["text", "image", "distinct ID"])("preserves local variant work that the collapsed server group does not represent: %s", change => {
    const original = { ...incoming("answer-1", "Original answer"), role: "assistant", isBot: true, parentMessageId: "question" }
    const latest = { ...incoming("answer-2", "New answer"), role: "assistant", isBot: true, parentMessageId: "question" }
    const local = { ...original,
      ...(change === "text" ? { message: "Unsynced edit" } : {}),
      ...(change === "image" ? { images: ["data:image/png;base64,local"] } : {}),
      ...(change === "distinct ID" ? { id: "local-draft", serverMessageId: undefined } : {}) }
    const result = reconcileServerChatMessages([local, latest], [{ ...latest, variants: [original, latest], activeVariantIndex: 1 }])
    expect(result).toContainEqual(local)
  })
  it.each(["draft", "edited text", "edited image"])("preserves an inactive local variant while a saved original is selected: %s", change => {
    const original = { ...incoming("answer-1", "Original answer"), role: "assistant", isBot: true, parentMessageId: "question" }
    const latest = { ...incoming("answer-2", "New answer"), role: "assistant", isBot: true, parentMessageId: "question" }
    const inactive = { ...latest,
      ...(change === "draft" ? { id: "local-draft", serverMessageId: undefined } : {}),
      ...(change === "edited text" ? { message: "Unsynced edit" } : {}),
      ...(change === "edited image" ? { images: ["data:image/png;base64,local"] } : {}) }
    const local = { ...original, variants: [original, inactive], activeVariantIndex: 0 }
    const result = reconcileServerChatMessages([local], [{ ...latest, variants: [original, latest], activeVariantIndex: 1 }])
    expect(result).toContainEqual(local)
  })
  it("collapses an original selected locally when all its variants are saved unchanged", () => {
    const original = { ...incoming("answer-1", "Original answer"), role: "assistant", isBot: true, parentMessageId: "question" }
    const latest = { ...incoming("answer-2", "New answer"), role: "assistant", isBot: true, parentMessageId: "question" }
    const local = { ...original, variants: [original, latest], activeVariantIndex: 0 }
    const grouped = { ...latest, variants: [original, latest], activeVariantIndex: 1 }
    expect(reconcileServerChatMessages([local], [grouped])).toEqual([grouped])
  })
  it("preserves unsynced drafts and newer acknowledged rows", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("draft", row("draft", "alice", "Unsent"))
    state.messages.set("local-q", { ...row("local-q", "alice", "Newer edit", "question"), serverMessageVersion: 3 })
    state.messages.set("late", row("late", "alice", "Acknowledged after GET", "late"))
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [incoming("question", "Old", 2), incoming("answer", "08:30")] })
    expect([...state.messages.values()].map(r => r.content)).toEqual(["Unsent", "Newer edit", "Acknowledged after GET", "08:30"])
  })
  it("retains an unknown-version local edit when acknowledging an old visible row", () => {
    const merged = reconcileServerChatMessages([{ id: "question", role: "user" as const, isBot: false, name: "You", message: "My unsynced edit", sources: [] }], [incoming("question", "Original server text"), incoming("answer", "08:30")])
    expect(merged.map(message => message.message)).toEqual(["My unsynced edit", "08:30"])
    expect(merged[0].serverMessageId).toBe("question")
  })
  it("retains an unknown-version local edit in the old persistent mirror", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("question", row("question", "alice", "My unsynced edit"))
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [incoming("question", "Original server text"), incoming("answer", "08:30")] })
    expect(state.messages.get("question")).toMatchObject({ content: "My unsynced edit", serverMessageId: "question" })
    expect(state.messages.size).toBe(2)
  })
  it("never moves another history's global message ID", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("answer", row("answer", "bob", "Bob secret", "answer"))
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [incoming("answer", "Alice answer")] })
    expect(state.messages.get("answer")).toMatchObject({ history_id: "bob", content: "Bob secret" })
    expect([...state.messages.values()].filter(r => r.history_id === "alice")).toMatchObject([{ content: "Alice answer", serverMessageId: "answer" }])
  })
  it("refuses a qualified-ID collision with another history", async () => {
    state.histories.set("alice", history("alice", "A"))
    state.messages.set("alice:server:answer", row("alice:server:answer", "bob", "Do not move"))
    await expect(reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: [incoming("answer", "Alice answer")] })).rejects.toThrow()
    expect(state.messages.get("alice:server:answer").history_id).toBe("bob")
  })
  it.each(["wrong-owner", "aborted"])("rejects a %s mirror write", async mode => {
    state.histories.set("alice", history("alice", "A"))
    const controller = new AbortController(); if (mode === "aborted") controller.abort()
    await expect(reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: mode === "wrong-owner" ? "B" : "A", signal: controller.signal, messages: [incoming("answer", "Late")] })).rejects.toThrow()
    expect(state.messages.size).toBe(0)
  })
})

describe('H1 canonical cursor translation', () => {
  beforeEach(() => { state.histories.clear(); state.messages.clear() })
  it('maps only an acknowledged local ID in its verified owner, never equal text or another account', async () => {
    const { resolveServerMirrorCursor } = await import('../server-chat-mirror')
    state.histories.set('alice', history('alice', 'A'))
    state.histories.set('bob', history('bob', 'B'))
    state.messages.set('alice-local', row('alice-local', 'alice', 'same', 'shared-server-id'))
    state.messages.set('bob-local', row('bob-local', 'bob', 'same', 'shared-server-id'))
    const request = { historyId: 'alice', chatId: 'chat-1', ownerKey: 'A', cursor: { kind: 'after_message' as const, message_id: 'alice-local' } }
    expect(await resolveServerMirrorCursor(request)).toEqual({ kind: 'after_message', message_id: 'shared-server-id' })
    await expect(resolveServerMirrorCursor({ ...request, cursor: { kind: 'after_message', message_id: 'bob-local' } })).rejects.toThrow()
    await expect(resolveServerMirrorCursor({ ...request, ownerKey: 'B' })).rejects.toThrow()
  })
  it('rejects absent or ambiguous canonical mappings and keeps explicit empty', async () => {
    const { resolveServerMirrorCursor } = await import('../server-chat-mirror')
    state.histories.set('alice', history('alice', 'A'))
    state.messages.set('local', row('local', 'alice', 'same'))
    const request = { historyId: 'alice', chatId: 'chat-1', ownerKey: 'A', cursor: { kind: 'after_message' as const, message_id: 'local' } }
    await expect(resolveServerMirrorCursor(request)).rejects.toThrow()
    state.messages.set('local', row('local', 'alice', 'same', 'canonical'))
    state.messages.set('duplicate', row('duplicate', 'alice', 'same', 'canonical'))
    await expect(resolveServerMirrorCursor(request)).rejects.toThrow()
    expect(await resolveServerMirrorCursor({ ...request, cursor: { kind: 'empty' } })).toEqual({ kind: 'empty' })
  })
})

describe("authoritative parent recovery for H1 cursors", () => {
  beforeEach(() => {
    state.histories.clear()
    state.messages.clear()
  })
  it.each(["missing-parent", "explicit-parent-reordered"])(
    "never chooses the adjacent same-text branch: %s",
    async (kind) => {
      const { resolveServerMirrorCursor } =
        await import("../server-chat-mirror")
      state.histories.set("alice", history("alice", "A"))
      state.messages.set("local-u", row("local-u", "alice", "Repeat"))
      state.messages.set("local-a", {
        ...row("local-a", "alice", "Reply", "a"),
        role: "assistant",
        parent_message_id: "local-u"
      })
      const remote = [
        incoming("u1", "Repeat"),
        incoming("u2", "Repeat"),
        {
          ...incoming("a", "Reply"),
          role: "assistant" as const,
          isBot: true,
          ...(kind === "explicit-parent-reordered"
            ? { parentMessageId: "u1" }
            : {})
        }
      ]
      await reconcileServerChatMirror({
        historyId: "alice",
        chatId: "chat-1",
        ownerKey: "A",
        messages: remote
      })
      const translate = () =>
        resolveServerMirrorCursor({
          historyId: "alice",
          chatId: "chat-1",
          ownerKey: "A",
          cursor: { kind: "after_message", message_id: "local-u" }
        })
      if (kind === "missing-parent") {
        expect(state.messages.get("local-u")?.serverMessageId).toBeUndefined()
        await expect(translate()).rejects.toThrow(
          "missing_canonical_message_id"
        )
      } else {
        expect(await translate()).toEqual({
          kind: "after_message",
          message_id: "u1"
        })
      }
      expect(state.messages.get("local-u")?.serverMessageId).not.toBe("u2")
    }
  )
})
