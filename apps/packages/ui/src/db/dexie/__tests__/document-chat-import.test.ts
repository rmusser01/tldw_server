import { describe, expect, it, vi } from "vitest"
import { rememberDocumentChatMedia, reserveDocumentChatImport, verifyDocumentChatImport } from "../document-chat-import"
import { serverChatMirrorOwnerKey } from "../server-chat-mirror"

const local = vi.hoisted(() => {
  const rows: Array<Record<string, unknown>> = []
  const history: Record<string, unknown> = {}
  let tail = Promise.resolve<unknown>(null)
  return {
    rows, history, db: {
      chatHistories: {
        get: async () => structuredClone(history),
        update: async (_key: string, patch: Record<string, unknown>) => Object.assign(history, structuredClone(patch)),
        where: () => ({ equals: (id: string) => ({ and: (accept: (row: Record<string, unknown>) => boolean) => ({
          modify: async (patch: Record<string, unknown>) => {
            if (id !== history.id || !accept(history)) return 0
            Object.assign(history, structuredClone(patch))
            return 1
          }
        }) }) })
      },
      messages: { where: () => ({ equals: () => ({ toArray: async () => structuredClone(rows) }) }) },
      transaction: (_mode: string, ...args: unknown[]) => {
        const run = tail.then(args.at(-1) as () => Promise<unknown>)
        tail = run.catch(() => undefined)
        return run
      }
    }
  }
})
vi.mock("../schema", () => ({ db: local.db }))

const target = { mediaId: 1, mediaUuid: "b679f16d-b053-4f9d-9f39-a6153d2d3cab" }
const scope = { config: { serverUrl: "https://chat.test", authMode: "multi-user" as const }, userId: "alice" }
const historyId = "pa_local"
const source = { name: "A chapter", pageContent: "This excerpt stays with the answer.", metadata: { media_id: 1 } }

describe("browser chat import reservation", () => {
  it("reuses a browser chat import after a lost response and retains later turns", async () => {
    Object.assign(local.history, {
      id: historyId, doc_id: "document:1", title: "Existing EPUB chat", createdAt: 5,
      server_scope_key: serverChatMirrorOwnerKey({ requestScope: scope })
    })
    local.rows.push(
      { id: "pa_q", history_id: historyId, role: "user", content: "Question", createdAt: 10, images: [""] },
      { id: "pa_a", history_id: historyId, role: "assistant", content: "Answer", parent_message_id: "pa_q", createdAt: 11, sources: [source] }
    )
    await expect(reserveDocumentChatImport(historyId, target, scope)).rejects.toThrow("Confirm this document")
    expect(local.history.document_chat_import).toBeUndefined()
    const [first, second] = await Promise.all([
      reserveDocumentChatImport(historyId, target, scope, true),
      reserveDocumentChatImport(historyId, target, scope, true)
    ])
    expect(second.payload).toEqual(first.payload)
    expect(first.payload.messages[0].images).toEqual([])
    expect(first.payload.messages[1].document_chat_sources_v1).toEqual([
      { media_uuid: target.mediaUuid, title: source.name, excerpt: source.pageContent }
    ])
    local.rows.push({
      id: "pa_late", history_id: historyId, role: "user", content: "Kept on this browser", parent_message_id: "pa_a", createdAt: 12
    })
    expect(await reserveDocumentChatImport(historyId, target, scope, true)).toEqual(first)
    expect(await verifyDocumentChatImport(historyId, target, scope, first)).toBe(false)
    expect(local.rows.at(-1)?.content).toBe("Kept on this browser")
  })

  it("stamps the confirmed document UUID only on the original owned mirror", async () => {
    Object.assign(local.history, {
      id: "pa_mirror", doc_id: "document:1", document_media_uuid: undefined, server_chat_id: "saved-chat",
      server_scope_key: serverChatMirrorOwnerKey({ requestScope: scope })
    })
    await expect(rememberDocumentChatMedia("pa_mirror", "replacement", target, scope)).resolves.toBe(false)
    await expect(rememberDocumentChatMedia("pa_mirror", "saved-chat", target, { ...scope, userId: "bob" })).resolves.toBe(false)
    expect(local.history.document_media_uuid).toBeUndefined()
    await expect(rememberDocumentChatMedia("pa_mirror", "saved-chat", target, scope)).resolves.toBe(true)
    expect(await local.db.chatHistories.get()).toHaveProperty("document_media_uuid", target.mediaUuid)
  })
})
