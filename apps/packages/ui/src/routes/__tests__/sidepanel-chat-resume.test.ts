import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  storageGet: vi.fn<(key: string) => Promise<unknown>>(),
  copilotResumeLastChat: vi.fn<() => Promise<boolean>>(),
  getRecentChatFromCopilot: vi.fn<() => Promise<unknown>>(),
  sendMessage: vi.fn<(message: { type: string }) => Promise<{ tabId?: unknown }>>()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: (key: string) => mocks.storageGet(key)
  }),
  safeStorageSerde: {}
}))

vi.mock("@/services/app", () => ({
  copilotResumeLastChat: () => mocks.copilotResumeLastChat()
}))

vi.mock("@/db/dexie/helpers", () => ({
  getRecentChatFromCopilot: () => mocks.getRecentChatFromCopilot()
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      sendMessage: (message: { type: string }) => mocks.sendMessage(message)
    }
  }
}))

import { hasResumableSidepanelChat } from "../sidepanel-chat-resume"

describe("hasResumableSidepanelChat", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.copilotResumeLastChat.mockResolvedValue(false)
    mocks.getRecentChatFromCopilot.mockResolvedValue(null)
    mocks.sendMessage.mockResolvedValue({ tabId: 7 })
  })

  it("treats a stored tabs snapshot with real chat state as resumable", async () => {
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "sidepanelChatTabsState:tab-7") {
        return {
          tabs: [{ id: "tab-1" }],
          activeTabId: "tab-1",
          snapshotsById: {
            "tab-1": {
              history: [
                {
                  role: "user",
                  content: "hello"
                }
              ],
              messages: [],
              chatMode: "normal",
              historyId: null,
              webSearch: false,
              toolChoice: "none",
              selectedModel: null,
              selectedSystemPrompt: null,
              selectedQuickPrompt: null,
              temporaryChat: false,
              useOCR: false,
              serverChatId: null,
              serverChatState: null,
              serverChatTopic: null,
              serverChatClusterId: null,
              serverChatSource: null,
              serverChatExternalRef: null,
              queuedMessages: [],
              modelSettings: {}
            }
          }
        }
      }
      return null
    })

    await expect(hasResumableSidepanelChat()).resolves.toBe(true)
  })

  it("does not treat a blank persisted tab scaffold as resumable", async () => {
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "sidepanelChatTabsState:tab-7") {
        return {
          tabs: [{ id: "tab-1" }],
          activeTabId: "tab-1",
          snapshotsById: {
            "tab-1": {
              history: [],
              messages: [],
              chatMode: "normal",
              historyId: null,
              webSearch: false,
              toolChoice: "none",
              selectedModel: null,
              selectedSystemPrompt: null,
              selectedQuickPrompt: null,
              temporaryChat: false,
              useOCR: false,
              serverChatId: null,
              serverChatState: null,
              serverChatTopic: null,
              serverChatClusterId: null,
              serverChatSource: null,
              serverChatExternalRef: null,
              queuedMessages: [],
              modelSettings: {}
            }
          }
        }
      }
      return null
    })

    await expect(hasResumableSidepanelChat()).resolves.toBe(false)
  })

  it("treats a blank persisted tab scaffold with a matching per-tab overlay marker as resumable", async () => {
    const matchingOverlayResumeKey =
      "sidepanelChatOverlayResume:tldw:sidepanelChatDraft:tab-1"
    const unrelatedOverlayResumeKey =
      "sidepanelChatOverlayResume:tldw:sidepanelChatDraft:tab-2"

    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "sidepanelChatTabsState:tab-7") {
        return {
          tabs: [{ id: "tab-1" }],
          activeTabId: "tab-1",
          snapshotsById: {
            "tab-1": {
              history: [],
              messages: [],
              chatMode: "normal",
              historyId: null,
              webSearch: false,
              toolChoice: "none",
              selectedModel: null,
              selectedSystemPrompt: null,
              selectedQuickPrompt: null,
              temporaryChat: false,
              useOCR: false,
              serverChatId: null,
              serverChatState: null,
              serverChatTopic: null,
              serverChatClusterId: null,
              serverChatSource: null,
              serverChatExternalRef: null,
              queuedMessages: [],
              modelSettings: {}
            }
          }
        }
      }
      if (key === matchingOverlayResumeKey) {
        return {
          updatedAt: "2026-05-22T12:00:00.000Z"
        }
      }
      if (key === unrelatedOverlayResumeKey) return null
      return null
    })

    await expect(hasResumableSidepanelChat()).resolves.toBe(true)
    expect(mocks.storageGet).toHaveBeenCalledWith(matchingOverlayResumeKey)
    expect(mocks.storageGet).not.toHaveBeenCalledWith(unrelatedOverlayResumeKey)
  })

  it("does not treat unrelated scratch overlays as resumable for a blank persisted tab scaffold", async () => {
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "sidepanelChatTabsState:tab-7") {
        return {
          tabs: [{ id: "tab-1" }],
          activeTabId: "tab-1",
          snapshotsById: {
            "tab-1": {
              history: [],
              messages: [],
              chatMode: "normal",
              historyId: null,
              webSearch: false,
              toolChoice: "none",
              selectedModel: null,
              selectedSystemPrompt: null,
              selectedQuickPrompt: null,
              temporaryChat: false,
              useOCR: false,
              serverChatId: null,
              serverChatState: null,
              serverChatTopic: null,
              serverChatClusterId: null,
              serverChatSource: null,
              serverChatExternalRef: null,
              queuedMessages: [],
              modelSettings: {}
            }
          }
        }
      }
      if (key === "chatSettings:scratch") {
        return {
          assistantOverlay: {
            kind: "persona",
            id: "persona-1",
            name: "Guide Persona",
            system_prompt_snapshot: "Stay in character.",
            updatedAt: "2026-05-22T12:00:00.000Z"
          }
        }
      }
      return null
    })

    await expect(hasResumableSidepanelChat()).resolves.toBe(false)
  })

  it("treats a legacy snapshot with an empty messages array as resumable", async () => {
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "sidepanelChatState:tab-7") {
        return {
          history: [],
          messages: [],
          chatMode: "normal",
          historyId: null
        }
      }
      return null
    })

    await expect(hasResumableSidepanelChat()).resolves.toBe(true)
  })

  it("returns false when there is no stored state and copilot resume is disabled", async () => {
    mocks.storageGet.mockResolvedValue(null)

    await expect(hasResumableSidepanelChat()).resolves.toBe(false)
  })
})
