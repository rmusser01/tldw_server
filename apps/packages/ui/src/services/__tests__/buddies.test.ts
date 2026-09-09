import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  getBuddyAttachment,
  listBuddies,
  listBuddyConversations,
  listBuddyActivity,
  listBuddyTurns,
  putBuddyAttachment,
  readBuddyConversation
} from "../buddies"

const mocks = vi.hoisted(() => ({
  fetchWithAuth: vi.fn(),
  getChat: vi.fn(),
  listChatMessages: vi.fn(),
  getConfig: vi.fn(),
  invalidateChatMessagesCache: vi.fn()
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: mocks }))

beforeEach(() => {
  vi.clearAllMocks()
  mocks.getConfig.mockResolvedValue({
    serverUrl: "https://first.invalid",
    accessToken: "first"
  })
})
describe("independent Buddy authority", () => {
  it("sends attachment version and scope without interpreting the client slot as authority", async () => {
    mocks.fetchWithAuth.mockResolvedValue({
      ok: true,
      json: async () => ({ version: 4 })
    })
    await putBuddyAttachment({
      expected_version: 3,
      buddy_id: "duck",
      scope_type: "workspace",
      scope_id: "research"
    })
    expect(JSON.parse(mocks.fetchWithAuth.mock.calls[0][1].body)).toEqual({
      expected_version: 3,
      buddy_id: "duck",
      scope_type: "workspace",
      scope_id: "research"
    })
  })
  it("does not treat a revoked attachment as an empty successful response", async () => {
    mocks.fetchWithAuth.mockResolvedValue({
      ok: false,
      status: 403,
      json: async () => ({ detail: "Access revoked" })
    })
    await expect(getBuddyAttachment()).rejects.toThrow("Access revoked")
  })
  it("rechecks workspace membership before reading a cached transcript", async () => {
    mocks.getChat.mockResolvedValue({
      id: "conversation",
      workspace_id: "elsewhere",
      scope_type: "workspace"
    })
    await expect(
      readBuddyConversation(
        { buddy_id: "duck", scope_type: "workspace", scope_id: "research" },
        "conversation"
      )
    ).rejects.toThrow("no longer belongs")
    expect(mocks.listChatMessages).not.toHaveBeenCalled()
  })
  it("never reads a different conversation through a conversation attachment", async () => {
    await expect(
      readBuddyConversation(
        { buddy_id: "duck", scope_type: "conversation", scope_id: "one" },
        "two"
      )
    ).rejects.toThrow("different conversation")
    expect(mocks.getChat).not.toHaveBeenCalled()
  })
  it("passes bounded profile and conversation pages to the server", async () => {
    mocks.fetchWithAuth.mockResolvedValue({
      ok: true,
      json: async () => ({ buddies: [], conversations: [] })
    })
    await listBuddies({ limit: 100, offset: 100 })
    await listBuddyConversations({ limit: 100, offset: 200 })
    expect(mocks.fetchWithAuth.mock.calls.map(([path]) => path)).toEqual([
      "/api/v1/buddies?limit=100&offset=100",
      "/api/v1/buddies/attachment/conversations?client_slot=default&limit=100&offset=200"
    ])
  })
  it("reads later activity pages even when earlier conversations have no result", async () => {
    mocks.fetchWithAuth.mockImplementation(async (path) => ({
      ok: true,
      json: async () => ({
        items: path.endsWith("offset=100")
          ? [{ conversation_id: "later", result: { id: "reply" } }]
          : []
      })
    }))
    expect((await listBuddyActivity({ conversationPages: 2 })).items).toEqual([
      { conversation_id: "later", result: { id: "reply" } }
    ])
    expect(mocks.fetchWithAuth).toHaveBeenCalledTimes(2)
  })
  it("keeps active turn queries separate from paged recent history", async () => {
    mocks.fetchWithAuth.mockResolvedValue({
      ok: true,
      json: async () => ({ turns: [] })
    })
    await listBuddyTurns({ pages: 2 })
    await listBuddyTurns({ status: "active" })
    expect(mocks.fetchWithAuth.mock.calls.map(([path]) => path)).toEqual([
      "/api/v1/buddies/turns?client_slot=default&limit=100&offset=0",
      "/api/v1/buddies/turns?client_slot=default&limit=100&offset=100",
      "/api/v1/buddies/turns?client_slot=default&limit=100&offset=0&status=active"
    ])
  })
  it("does not continue a transcript read under changed credentials", async () => {
    mocks.getChat.mockImplementation(async () => {
      mocks.getConfig.mockResolvedValue({
        serverUrl: "https://second.invalid",
        accessToken: "second"
      })
      return { id: "conversation", scope_type: "global" }
    })
    await expect(
      readBuddyConversation(
        {
          buddy_id: "duck",
          scope_type: "conversation",
          scope_id: "conversation"
        },
        "conversation"
      )
    ).rejects.toThrow("Connection changed")
    expect(mocks.listChatMessages).not.toHaveBeenCalled()
  })
  it("checks caller lifetime before issuing the follow-up message read", async () => {
    let active = true
    mocks.getChat.mockImplementation(async () => {
      active = false
      return { id: "conversation", scope_type: "global" }
    })
    await expect(
      readBuddyConversation(
        {
          buddy_id: "duck",
          scope_type: "conversation",
          scope_id: "conversation"
        },
        "conversation",
        null,
        { isCurrent: () => active }
      )
    ).rejects.toThrow("Interaction changed")
    expect(mocks.listChatMessages).not.toHaveBeenCalled()
  })
})
