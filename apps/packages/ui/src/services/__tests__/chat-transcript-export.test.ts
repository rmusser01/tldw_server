import { beforeEach, describe, expect, it, vi } from "vitest"

const client = vi.hoisted(() => ({ getChat: vi.fn(), listChatMessages: vi.fn() }))
const download = vi.hoisted(() => ({ transcript: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: client }))
vi.mock("@/utils/chat-transcript", async (original) => ({
  ...(await original<typeof import("@/utils/chat-transcript")>()),
  downloadTranscript: download.transcript
}))

import { exportServerChat, readServerChatTranscript } from "../chat-transcript-export"

const message = (id: string, parent: string | null, role: "user" | "assistant", content: string, second: number) => ({
  id, parent_message_id: parent, role, content, created_at: `2026-11-01T09:00:0${second}Z`
})

describe("server chat export (CS-09)", () => {
  beforeEach(() => {
    client.getChat.mockReset()
    client.listChatMessages.mockReset()
    download.transcript.mockReset()
  })

  it("reads every saved message, alternatives included, in creation order", async () => {
    client.getChat.mockResolvedValue({ id: "chat-1", title: "Trip plan", created_at: "2026-11-01T09:00:00Z" })
    const rows = [
      message("u1", null, "user", "Where first?", 0),
      { ...message("a1b", "u1", "assistant", "Lisbon.", 2), version: 2,
        metadata_extra: { model_id: "model-b", provider: "openai", generation_status: "complete" } },
      { ...message("a1", "u1", "assistant", "Porto.", 1), sender: "Navigator", metadata_extra: { model_id: "model-a" } }
    ]
    client.listChatMessages.mockImplementation(async (_id: string, params: { limit: number; offset: number }) =>
      rows.slice(params.offset, params.offset + params.limit))
    const transcript = await readServerChatTranscript("chat-1")
    expect(transcript).toEqual({
      id: "chat-1",
      title: "Trip plan",
      createdAt: "2026-11-01T09:00:00Z",
      updatedAt: null,
      messages: [
        { id: "u1", parentId: null, role: "user", content: "Where first?", timestamp: "2026-11-01T09:00:00Z" },
        { id: "a1", parentId: "u1", role: "assistant", sender: "Navigator", content: "Porto.",
          timestamp: "2026-11-01T09:00:01Z", model: "model-a", metadata: { model_id: "model-a" } },
        { id: "a1b", parentId: "u1", role: "assistant", content: "Lisbon.", timestamp: "2026-11-01T09:00:02Z",
          version: 2, model: "model-b", provider: "openai",
          metadata: { model_id: "model-b", provider: "openai", generation_status: "complete" } }
      ]
    })
    expect(client.listChatMessages).toHaveBeenCalledWith(
      "chat-1",
      expect.objectContaining({ include_deleted: "false", include_metadata: "true", offset: 0 }),
      expect.anything()
    )
  })

  it("names an export by the title the caller knows when the chat's details can't be read", async () => {
    client.getChat.mockRejectedValue(new Error("HTTP 500"))
    client.listChatMessages.mockResolvedValue([])
    expect((await readServerChatTranscript("chat-1", { title: "From the list" })).title).toBe("From the list")
  })

  it("fails when the messages can't be read, instead of saving an empty file", async () => {
    client.getChat.mockResolvedValue({ id: "chat-1", title: "Trip plan" })
    client.listChatMessages.mockRejectedValue(new Error("HTTP 503"))
    await expect(exportServerChat("chat-1", "markdown")).rejects.toThrow("HTTP 503")
    expect(download.transcript).not.toHaveBeenCalled()
  })

  it("downloads the server transcript in the chosen format", async () => {
    client.getChat.mockResolvedValue({ id: "chat-1", title: "Trip plan" })
    client.listChatMessages.mockResolvedValue([message("u1", null, "user", "Hi", 0)])
    await exportServerChat("chat-1", "json")
    expect(download.transcript).toHaveBeenCalledWith(expect.objectContaining({ id: "chat-1", title: "Trip plan" }), "json")
  })
})
