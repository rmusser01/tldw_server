import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  getBuddyAttachment,
  importBuddyArchive,
  listBuddies,
  listBuddyConversations,
  listBuddyActivity,
  listBuddyTurns,
  previewBuddyArchive,
  putBuddyAttachment,
  readBuddyConversation
} from "../buddies"

const mocks = vi.hoisted(() => ({
  fetchWithAuth: vi.fn(),
  upload: vi.fn(),
  initialize: vi.fn(),
  ensureConfigForRequest: vi.fn(),
  requestWithCurrentConfig: vi.fn(),
  getChat: vi.fn(),
  listChatMessages: vi.fn(),
  getConfig: vi.fn(),
  invalidateChatMessagesCache: vi.fn()
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: mocks }))

beforeEach(() => {
  vi.clearAllMocks()
  mocks.initialize.mockResolvedValue(undefined)
  mocks.ensureConfigForRequest.mockResolvedValue({
    serverUrl: "https://first.invalid",
    authMode: "single-user",
    authSource: "manual",
    apiKey: "synthetic-first-key"
  })
  mocks.getConfig.mockResolvedValue({
    serverUrl: "https://first.invalid",
    accessToken: "first"
  })
})
describe("independent Buddy authority", () => {
  it("keeps the native file on the direct upload transport for review and import", async () => {
    const file = new File(["archive"], "moss.tldw-persona-vpack")
    const digest = "a".repeat(64)
    mocks.upload.mockResolvedValueOnce({ archive_sha256: digest }).mockResolvedValueOnce({ id: "moss" })
    const reviewed = await previewBuddyArchive(file)
    expect(reviewed.preview).toEqual({ archive_sha256: digest })
    mocks.ensureConfigForRequest.mockResolvedValue({ serverUrl: "https://second.invalid", authMode: "single-user", apiKey: "synthetic-new-key" })
    await expect(importBuddyArchive(reviewed)).resolves.toEqual({ id: "moss" })
    expect(mocks.upload).toHaveBeenNthCalledWith(1, expect.objectContaining({
      path: "/api/v1/buddies/import-preview",
      directFile: file,
      fileFieldName: "archive"
    }))
    expect(mocks.upload).toHaveBeenNthCalledWith(2, expect.objectContaining({
      path: "/api/v1/buddies/import",
      directFile: file,
      fileFieldName: "archive",
      fields: { reviewed_sha256: digest },
      servicePromptConfig: expect.objectContaining({ serverUrl: "https://first.invalid", expectedUserId: null }),
      abortSignal: expect.any(AbortSignal)
    }))
    expect(mocks.fetchWithAuth).not.toHaveBeenCalled()
  })

  it("cancels archive review when credentials change during scope discovery", async () => {
    const file = new File(["archive"], "moss.tldw-persona-vpack")
    let finishInitialization!: () => void
    mocks.initialize.mockReturnValueOnce(new Promise<void>((resolve) => { finishInitialization = resolve }))
    const pending = previewBuddyArchive(file)
    const cancelled = expect(pending).rejects.toMatchObject({ name: "AbortError" })
    window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
    finishInitialization()
    await cancelled
    expect(mocks.upload).not.toHaveBeenCalled()
  })

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

it("loads effective Buddy reply settings only after the authorized transcript read", async () => {
  mocks.getChat.mockResolvedValue({
    id: "conversation",
    scope_type: "workspace",
    workspace_id: "research"
  })
  mocks.listChatMessages.mockResolvedValue([])
  const dispatched = vi.fn()
  mocks.requestWithCurrentConfig.mockImplementation(async (createRequest) => {
    dispatched(createRequest(await mocks.getConfig()))
    return { provider: "custom-openai-api", model: "saved-model" }
  })
  const result = await readBuddyConversation(
    { buddy_id: "duck", scope_type: "workspace", scope_id: "research" },
    "conversation",
    "research",
    { includeReplySettings: true }
  )
  expect(dispatched).toHaveBeenCalledWith({
    path: "/api/v1/buddies/conversation-targets/conversation/reply-settings?client_slot=default",
    method: "GET"
  })
  expect(result.replySettings).toEqual({
    provider: "custom-openai-api",
    model: "saved-model"
  })
})

it("does not read reply settings after a connection change during the transcript request", async () => {
  mocks.getChat.mockResolvedValue({
    id: "conversation",
    scope_type: "workspace",
    workspace_id: "research"
  })
  mocks.listChatMessages.mockImplementation(async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "https://second.invalid",
      accessToken: "second"
    })
    return []
  })
  await expect(
    readBuddyConversation(
      { buddy_id: "duck", scope_type: "workspace", scope_id: "research" },
      "conversation",
      "research",
      { includeReplySettings: true }
    )
  ).rejects.toThrow("Connection changed")
  expect(mocks.fetchWithAuth).not.toHaveBeenCalled()
})

it("does not dispatch reply settings to a connection changed during transport configuration", async () => {
  mocks.getChat.mockResolvedValue({
    id: "conversation",
    scope_type: "workspace",
    workspace_id: "research"
  })
  mocks.listChatMessages.mockResolvedValue([])
  let release: (config: { serverUrl: string; accessToken: string }) => void
  const configRead = new Promise<{ serverUrl: string; accessToken: string }>(
    (resolve) => {
      release = resolve
    }
  )
  const dispatch = vi.fn()
  mocks.fetchWithAuth.mockImplementation(async () => {
    const config = await configRead
    dispatch(config)
    return {
      ok: true,
      json: async () => ({ model: "wrong", provider: "wrong" })
    }
  })
  mocks.requestWithCurrentConfig.mockImplementation(async (createRequest) => {
    const config = await configRead
    const request = createRequest(config)
    dispatch(config, request)
    return { model: "wrong", provider: "wrong" }
  })
  const pending = readBuddyConversation(
    { buddy_id: "duck", scope_type: "workspace", scope_id: "research" },
    "conversation",
    "research",
    { includeReplySettings: true }
  )
  const rejection = expect(pending).rejects.toThrow("Connection changed")
  await vi.waitFor(() =>
    expect(
      mocks.fetchWithAuth.mock.calls.length +
        mocks.requestWithCurrentConfig.mock.calls.length
    ).toBe(1)
  )
  const changed = { serverUrl: "https://second.invalid", accessToken: "second" }
  mocks.getConfig.mockResolvedValue(changed)
  release!(changed)
  await rejection
  expect(dispatch).not.toHaveBeenCalled()
})
