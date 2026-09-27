import { describe, expect, it, vi } from "vitest"
import { fetchAllServerChatMessages, mapServerChatMessagesToPlaygroundMessages } from "@/hooks/chat/useServerChatLoader"
import { reconcileServerChatMessages } from "@/db/dexie/server-chat-mirror"
import type { ServerChatMessage } from "@/services/tldw/TldwApiClient"

const image = "data:image/png;base64,aW1hZ2U="
const server = (content = "Question", version = 1): ServerChatMessage & { images: string[] } => ({
  id: "saved", role: "user", content, version, images: [image], created_at: "2026-09-16T00:00:00Z",
  metadata_extra: { client_message_id: "local", content_placeholder_reason: "image_attachment" }
})

it.each(["Question", ""])("maps complete canonical attachments before correlation: %s", text => {
  const local = { id: "local", role: "user" as const, isBot: false, name: "You", message: text, images: [image], sources: [] }
  const mapped = mapServerChatMessagesToPlaygroundMessages({ serverMessages: [server(text || "<Image attachment x1>")], assistantName: "Assistant", characterId: null })
  const rows = reconcileServerChatMessages([local], mapped)
  expect(rows).toHaveLength(1)
  expect(rows[0]).toMatchObject({ id: "local", serverMessageId: "saved", message: text, images: [image] })
})

it.each([2, 3, undefined])("keeps edited/unknown-version attachment-placeholder text at version %s", version => {
  const row = server("<Image attachment x1>")
  row.version = version
  const [mapped] = mapServerChatMessagesToPlaygroundMessages({ serverMessages: [row], assistantName: "Assistant", characterId: null })
  expect(mapped.message).toBe(row.content)
  expect(mapped.images).toEqual([image])
})

it("keeps literal placeholder text without trusted provenance", () => {
  const row = server("<Image attachment x1>")
  row.metadata_extra = { client_message_id: "local" }
  const [mapped] = mapServerChatMessagesToPlaygroundMessages({ serverMessages: [row], assistantName: "Assistant", characterId: null })
  expect(mapped.message).toBe(row.content)
})

describe("bounded complete image history", () => {
  it("aborts after an awaited page without returning a partial image history", async () => {
    const controller = new AbortController()
    const fetch = vi.fn(async () => { controller.abort(); return [server()] })
    await expect(fetchAllServerChatMessages(fetch, { limit: 1, signal: controller.signal })).rejects.toThrow()
    expect(fetch).toHaveBeenCalledTimes(1)
  })
  it("rejects cumulative attachment expansion across pages", async () => {
    const payload = "a".repeat(33 * 1024 * 1024)
    const fetch = vi.fn(async ({ offset }) => offset < 2 ? [{ ...server(), id: String(offset), images: [payload] }] : [])
    await expect(fetchAllServerChatMessages(fetch, { limit: 1 })).rejects.toThrow(/image|attachment/i)
    expect(fetch).toHaveBeenCalledTimes(2)
  })
  it("keeps exact offsets and complete images across normal pages", async () => {
    const fetch = vi.fn(async ({ offset }) => offset < 3 ? [{ ...server(), id: String(offset) }] : [])
    const rows = await fetchAllServerChatMessages(fetch, { limit: 1 })
    expect(rows.map(row => row.id)).toEqual(["0", "1", "2"])
    expect(fetch.mock.calls.map(([arg]) => arg.offset)).toEqual([0, 1, 2, 3])
  })
})
