import { beforeEach, describe, expect, it, vi } from "vitest"
import type { HistoryInfo } from "../types"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { getRecentChatFromWebUI, getRecentChatFromCopilot, getHistoryByDocId, saveHistory } from "../helpers"

const io = vi.hoisted(() => ({ histories: [] as HistoryInfo[] }))
vi.mock("../chat", () => ({ PageAssistDatabase: class {
  async getChatHistories() { return io.histories }
  async getHistoryByDocId(id: string) { return io.histories.find(history => history.doc_id === id) ?? null }
  async getAllHistoriesByDocId(id: string) { return io.histories.filter(history => history.doc_id === id) }
  async addChatHistory(history: HistoryInfo) { io.histories.unshift(history) }
  async getChatHistory(id: string) { return [{ content: `${id} private transcript` }] }
} }))

const aliceOwner = '["http://chat.test","multi-user","manual",null,"alice",null]'
const bobOwner = '["http://chat.test","multi-user","manual",null,"bob",null]'
const snapshot = { requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "bob" } } as ServicePromptSnapshot
const history = (id: string, server_scope_key?: string): HistoryInfo => ({
  id, title: `${id} title`, createdAt: 1, is_rag: false, message_source: "web-ui",
  server_chat_id: `${id}-server`, server_scope_key
})

describe("recent Chat cache ownership", () => {
  beforeEach(() => { io.histories = [] })

  it("resumes only the verified owner's Copilot history, skipping foreign and unowned rows", async () => {
    io.histories = [history("alice", aliceOwner), history("legacy"), history("bob", bobOwner)]
      .map(row => ({ ...row, message_source: "copilot" }))
    expect((await getRecentChatFromCopilot(snapshot))?.history.id).toBe("bob")
    io.histories.pop()
    expect(await getRecentChatFromCopilot(snapshot)).toBeNull()
  })

  it("can resume a newly saved Chat only for its verified owner", async () => {
    await saveHistory("Bob saved chat", false, "web-ui", undefined, undefined, snapshot.requestScope)
    expect((await getRecentChatFromWebUI(snapshot))?.history.title).toBe("Bob saved chat")
    const alice = { ...snapshot, requestScope: { ...snapshot.requestScope, userId: "alice" } }
    expect(await getRecentChatFromWebUI(alice)).toBeNull()
  })

  it("skips a newer foreign Chat and resumes the current owner's Chat", async () => {
    io.histories = [history("alice", aliceOwner), history("bob", bobOwner)]
    const result = await getRecentChatFromWebUI(snapshot)
    expect(result?.history.id).toBe("bob")
    expect(result?.messages).toEqual([{ content: "bob private transcript" }])
  })

  it("finds the owner's document Chat when another account has a newer matching document ID", async () => {
    io.histories = [
      { ...history("alice", aliceOwner), doc_id: "document:1", createdAt: 3 },
      { ...history("bob", bobOwner), doc_id: "document:1", createdAt: 2 }
    ]
    expect((await getHistoryByDocId("document:1", snapshot.requestScope))?.id).toBe("bob")
  })

  it.each([aliceOwner, undefined])("does not adopt another owner's or unowned legacy cache (%s)", async owner => {
    io.histories = [history("foreign", owner)]
    expect(await getRecentChatFromWebUI(snapshot)).toBeNull()
  })
})
