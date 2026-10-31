import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  saveHistory: vi.fn(async (..._args: unknown[]) => ({ id: "new-history" })),
  saveMessage: vi.fn(async (..._args: unknown[]) => undefined),
  getSetting: vi.fn(async () => false),
  pageAssistModel: vi.fn()
}))

vi.mock("@/db/dexie/helpers", () => ({
  saveHistory: mocks.saveHistory,
  saveMessage: mocks.saveMessage,
  updateLastUsedModel: vi.fn(),
  updateLastUsedPrompt: vi.fn(),
  updateChatHistoryCreatedAt: vi.fn()
}))
vi.mock("@/db/dexie/chat-persistence-transaction", () => ({
  runChatPersistenceTransaction: async (
    _signal: AbortSignal | undefined,
    operation: () => Promise<unknown>
  ) => operation()
}))
vi.mock("@/models", () => ({ pageAssistModel: mocks.pageAssistModel }))
vi.mock("@/services/settings/registry", async () => ({
  ...await vi.importActual<typeof import("@/services/settings/registry")>(
    "@/services/settings/registry"
  ),
  getSetting: mocks.getSetting
}))
vi.mock("@/store/option", () => ({
  useStoreMessageOption: { getState: () => ({ setHistory: vi.fn() }) }
}))

import { saveMessageOnSuccess } from "../index"

const image = "data:image/png;base64,aW1hZ2U="
const payload = {
  historyId: null,
  setHistoryId: vi.fn(),
  isRegenerate: false,
  selectedModel: "model-1",
  message: "",
  image,
  fullText: "Three dots",
  source: []
}

describe("successful Chat titles with title generation disabled", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    document.title = "Previous title"
  })

  it.each([
    { message: "", expected: "Untitled Chat", image },
    { message: " \n ", expected: "Untitled Chat", image },
    { message: "Describe this image", expected: "Describe this image", image },
    { message: "Text-only question", expected: "Text-only question", image: "" }
  ])("persists a readable title for $expected", async ({ message, expected, image }) => {
    await saveMessageOnSuccess({ ...payload, message, image })

    expect(mocks.saveHistory).toHaveBeenCalledWith(
      expected, false, "web-ui", undefined, undefined, undefined
    )
    expect(document.title).toBe(expected)
    expect(mocks.saveMessage).toHaveBeenCalledWith(expect.objectContaining({
      role: "user", content: message, images: [image]
    }))
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
  })

  it("keeps the authored title when an image is sent to existing history", async () => {
    document.title = "My authored title"
    await saveMessageOnSuccess({ ...payload, historyId: "existing-history" })

    expect(mocks.saveHistory).not.toHaveBeenCalled()
    expect(mocks.getSetting).not.toHaveBeenCalled()
    expect(document.title).toBe("My authored title")
  })
})
