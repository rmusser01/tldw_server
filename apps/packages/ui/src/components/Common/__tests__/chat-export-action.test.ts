import { beforeEach, describe, expect, it, vi } from "vitest"

const antdMessage = vi.hoisted(() => ({ loading: vi.fn(), success: vi.fn(), error: vi.fn(), destroy: vi.fn() }))
vi.mock("antd", () => ({ message: antdMessage }))

import { runChatExport } from "../chat-export-action"
import { ChatExportTooLargeError } from "@/services/chat-transcript-export"

const t = (_key: string, fallback: string) => fallback

describe("runChatExport (CS-09 review)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("shows it is exporting, then that it worked", async () => {
    await runChatExport({ chatId: "chat-1", t, run: async () => {} })
    expect(antdMessage.loading).toHaveBeenCalledWith(expect.objectContaining({ key: "chat-export:chat-1", content: "Exporting chat…" }))
    expect(antdMessage.success).toHaveBeenCalledWith(expect.objectContaining({ key: "chat-export:chat-1", content: "Chat exported." }))
  })

  it("ignores a second click while the same chat is still exporting", async () => {
    let release!: () => void
    const run = vi.fn(() => new Promise<void>((resolve) => { release = resolve }))
    const first = runChatExport({ chatId: "chat-1", t, run })
    await runChatExport({ chatId: "chat-1", t, run })
    expect(run).toHaveBeenCalledTimes(1)
    release()
    await first
    await runChatExport({ chatId: "chat-1", t, run: async () => {} })
    expect(antdMessage.success).toHaveBeenCalledTimes(2)
  })

  it("says a chat over the reading limit is too large to export here, without Try again", async () => {
    await runChatExport({ chatId: "chat-1", t, run: async () => { throw new ChatExportTooLargeError() } })
    const content = String(antdMessage.error.mock.calls[0][0].content)
    expect(content).toMatch(/too large to export here/)
    expect(content).not.toMatch(/Try again/)
  })

  it("says an ordinary failure can be tried again, including a device export that reports false", async () => {
    await runChatExport({ chatId: "chat-1", t, run: async () => false })
    expect(antdMessage.error).toHaveBeenCalledWith(expect.objectContaining({ content: "Couldn't export this chat. Try again." }))
  })
})
