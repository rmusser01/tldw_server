import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  storageGet: vi.fn<(key: string) => Promise<unknown>>(),
  copilotResumeLastChat: vi.fn<() => Promise<boolean>>(),
  getRecentChatFromCopilot: vi.fn<() => Promise<unknown>>(),
  loadSnapshot: vi.fn(),
  sendMessage: vi.fn<(message: { type: string }) => Promise<{ tabId?: unknown }>>()
}))

vi.mock("@/services/service-prompts", () => ({ loadServicePromptSnapshot: () => mocks.loadSnapshot() }))

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

import { durableSidepanelTabs, hasResumableSidepanelChat, type SidepanelTabsState } from "../sidepanel-chat-resume"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import type { SidepanelChatSnapshot } from "@/store/sidepanel-chat-tabs"

const draftOnlyTabs = (temporary: boolean) => ({
  version: 2,
  ownerKey: '["http://chat.test","multi-user","manual",null,"alice",null]',
  tabs: [{ id: "logical-draft", label: "New chat", historyId: null, serverChatId: null, serverChatTopic: null, updatedAt: 1 }],
  activeTabId: "logical-draft",
  snapshotsById: {
    "logical-draft": {
      chatDraftHandle: { ownerHash: chatDraftOwnerKey({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" }), address: { kind: "fresh", id: "unsent", scope: { type: "global" }, temporary } },
      history: [], messages: [], chatMode: "normal", historyId: null, serverChatId: null,
      webSearch: false, toolChoice: "none", selectedModel: null, selectedSystemPrompt: null,
      selectedQuickPrompt: null, temporaryChat: temporary, useOCR: false,
      serverChatState: null, serverChatTopic: null, serverChatClusterId: null,
      serverChatSource: null, serverChatExternalRef: null, queuedMessages: [], modelSettings: {}
    } as SidepanelChatSnapshot
  }
})

describe("hasResumableSidepanelChat", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.copilotResumeLastChat.mockResolvedValue(false)
    mocks.getRecentChatFromCopilot.mockResolvedValue(null)
    mocks.sendMessage.mockResolvedValue({ tabId: 7 })
    mocks.loadSnapshot.mockImplementation(async () => ({
      requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" },
      scopeSignal: new AbortController().signal, scopeInvalidatedSignal: new AbortController().signal, release: () => {}
    }))
  })

  it("does not offer another account's unowned cached tabs", async () => {
    mocks.storageGet.mockResolvedValue({ tabs: [{ id: "old-tab", historyId: "alice-private" }], activeTabId: "old-tab", snapshotsById: {} })
    await expect(hasResumableSidepanelChat()).resolves.toBe(false)
  })

  it.each([false, true])("offers resume for a verified draft-only view with no saved messages (Temporary %s)", async temporary => {
    mocks.storageGet.mockResolvedValue(draftOnlyTabs(temporary))
    await expect(hasResumableSidepanelChat()).resolves.toBe(true)
    expect(mocks.copilotResumeLastChat).not.toHaveBeenCalled()
  })

  it("does not offer another principal's draft handle from an otherwise owner-keyed scaffold", async () => {
    const saved = draftOnlyTabs(false)
    saved.snapshotsById["logical-draft"].chatDraftHandle!.ownerHash = "b".repeat(64)
    mocks.storageGet.mockImplementation(async key => key.startsWith("sidepanelChatTabsState:v2:") ? saved : null)
    await expect(hasResumableSidepanelChat()).resolves.toBe(false)
  })

  it("ignores a held resume read after an account boundary", async () => {
    let finish!: (value: unknown) => void
    mocks.storageGet.mockImplementationOnce(() => new Promise(resolve => { finish = resolve })).mockResolvedValue(null)
    const pending = hasResumableSidepanelChat()
    await vi.waitFor(() => expect(finish).toBeTypeOf("function"))
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
    finish({ version: 2, ownerKey: '["http://chat.test","multi-user","manual",null,"alice",null]', tabs: [{ id: "old-tab", historyId: "alice-private" }], activeTabId: "old-tab", snapshotsById: {} })
    await expect(pending).resolves.toBe(false)
  })

  it.each([
    [7, "tab-7"],
    [0, "tab-0"],
    [null, "global"],
    [undefined, "global"]
  ])("treats stored real chat state as resumable for runtime tab %s", async (tabId, storageSuffix) => {
    mocks.sendMessage.mockResolvedValue({ tabId })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === `sidepanelChatTabsState:v2:${encodeURIComponent('["http://chat.test","multi-user","manual",null,"alice",null]')}:${storageSuffix}`) {
        return {
          version: 2, ownerKey: '["http://chat.test","multi-user","manual",null,"alice",null]',
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
      if (key === `sidepanelChatTabsState:v2:${encodeURIComponent('["http://chat.test","multi-user","manual",null,"alice",null]')}:tab-7`) {
        return {
          version: 2, ownerKey: '["http://chat.test","multi-user","manual",null,"alice",null]',
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
      if (key === `sidepanelChatTabsState:v2:${encodeURIComponent('["http://chat.test","multi-user","manual",null,"alice",null]')}:tab-7`) {
        return {
          version: 2, ownerKey: '["http://chat.test","multi-user","manual",null,"alice",null]',
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
      if (key === `sidepanelChatTabsState:v2:${encodeURIComponent('["http://chat.test","multi-user","manual",null,"alice",null]')}:tab-7`) {
        return {
          version: 2, ownerKey: '["http://chat.test","multi-user","manual",null,"alice",null]',
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

  it("does not adopt an unowned legacy snapshot", async () => {
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

    await expect(hasResumableSidepanelChat()).resolves.toBe(false)
  })

  it("returns false when there is no stored state and copilot resume is disabled", async () => {
    mocks.storageGet.mockResolvedValue(null)

    await expect(hasResumableSidepanelChat()).resolves.toBe(false)
  })
})


it.each([false, true])("does not durably retain an active insertion command in a saved tab (Temporary %s)", temporary => {
  const saved = draftOnlyTabs(temporary)
  const original = saved.snapshotsById["logical-draft"]
  original.selectedQuickPrompt = "one-time command"
  original.selectedSystemPrompt = "system context"
  const durable = durableSidepanelTabs(saved as SidepanelTabsState, original.chatDraftHandle!.ownerHash)
  expect({ quick: durable.snapshotsById["logical-draft"].selectedQuickPrompt, system: durable.snapshotsById["logical-draft"].selectedSystemPrompt, originalQuick: original.selectedQuickPrompt })
    .toEqual({ quick: null, system: "system context", originalQuick: "one-time command" })
})
