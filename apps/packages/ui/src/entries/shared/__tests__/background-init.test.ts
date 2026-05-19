import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getInitialConfig: vi.fn(),
  getServerCapabilities: vi.fn(),
  contextMenusRemoveAll: vi.fn(),
  contextMenusCreate: vi.fn(),
  contextMenusRemove: vi.fn(),
  alarmsClear: vi.fn(),
  alarmsCreate: vi.fn()
}))

vi.mock("@/services/action", () => ({
  getInitialConfig: (...args: unknown[]) => mocks.getInitialConfig(...args)
}))

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: (...args: unknown[]) =>
    mocks.getServerCapabilities(...args)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    contextMenus: {
      removeAll: (...args: unknown[]) => mocks.contextMenusRemoveAll(...args),
      create: (...args: unknown[]) => mocks.contextMenusCreate(...args),
      remove: (...args: unknown[]) => mocks.contextMenusRemove(...args)
    },
    alarms: {
      clear: (...args: unknown[]) => mocks.alarmsClear(...args),
      create: (...args: unknown[]) => mocks.alarmsCreate(...args)
    },
    i18n: {
      getMessage: (key: string) =>
        (
          {
            openOptionToChat: "Open Web UI to Chat",
            openSidePanelToChat: "Open Side Panel to Chat",
            contextSummarize: "Summarize",
            contextExplain: "Explain",
            contextRephrase: "Rephrase",
            contextTranslate: "Translate",
            contextCustom: "Custom",
            contextCopilotPopup: "Copilot Popup",
            contextNarrateSelection: "Narrate selection",
            contextSaveToNotes: "Save to Notes",
            contextSaveToClipper: "Save to Clipper",
            contextSaveToCompanion: "Save to Companion"
          } as Record<string, string>
        )[key] || key
    }
  }
}))

import { initBackground } from "../background-init"

const flushPromises = async () => {
  await Promise.resolve()
  await Promise.resolve()
}

describe("background clipper rollout guard", () => {
  const watchHandlers: Record<string, ((value: any) => void) | undefined> = {}
  const storage = {
    watch: vi.fn((handlers: Record<string, (value: any) => void>) => {
      Object.assign(watchHandlers, handlers)
    }),
    get: vi.fn(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "http://127.0.0.1:8000",
          authMode: "single-user"
        }
      }
      return null
    }),
    set: vi.fn(async () => undefined)
  }
  const warmModels = vi.fn(async () => null)

  beforeEach(() => {
    Object.keys(watchHandlers).forEach((key) => {
      delete watchHandlers[key]
    })
    vi.clearAllMocks()
    mocks.getInitialConfig.mockResolvedValue({
      contextMenuClick: "sidePanel",
      actionIconClick: "webui"
    })
    mocks.getServerCapabilities.mockResolvedValue({
      hasWebClipper: true
    })
    mocks.contextMenusRemove.mockResolvedValue(undefined)
    mocks.contextMenusRemoveAll.mockResolvedValue(undefined)
    mocks.alarmsClear.mockResolvedValue(undefined)
    mocks.alarmsCreate.mockResolvedValue(undefined)
  })

  it("creates the clipper menu only when the connected server advertises support", async () => {
    mocks.getServerCapabilities.mockResolvedValueOnce({
      hasWebClipper: false
    })

    await initBackground({
      storage: storage as never,
      contextMenuId: { webui: "open-web-ui-pa", sidePanel: "open-side-panel-pa" },
      saveToClipperMenuId: "save-to-clipper-pa",
      saveToCompanionMenuId: "save-to-companion-pa",
      saveToNotesMenuId: "save-to-notes-pa",
      narrateSelectionMenuId: "narrate-selection-pa",
      transcribeMenuId: {
        transcribe: "transcribe-media-pa",
        transcribeAndSummarize: "transcribe-and-summarize-media-pa"
      },
      warmModels,
      capabilities: {
        sendToTldw: false,
        processLocal: false,
        transcribe: false,
        openApiCheck: false
      },
      onActionIconClickChange: vi.fn(),
      onContextMenuClickChange: vi.fn()
    })

    expect(mocks.contextMenusCreate).not.toHaveBeenCalledWith(
      expect.objectContaining({ id: "save-to-clipper-pa" })
    )
  })

  it("removes the clipper menu after config changes to a server without clipper support", async () => {
    await initBackground({
      storage: storage as never,
      contextMenuId: { webui: "open-web-ui-pa", sidePanel: "open-side-panel-pa" },
      saveToClipperMenuId: "save-to-clipper-pa",
      saveToCompanionMenuId: "save-to-companion-pa",
      saveToNotesMenuId: "save-to-notes-pa",
      narrateSelectionMenuId: "narrate-selection-pa",
      transcribeMenuId: {
        transcribe: "transcribe-media-pa",
        transcribeAndSummarize: "transcribe-and-summarize-media-pa"
      },
      warmModels,
      capabilities: {
        sendToTldw: false,
        processLocal: false,
        transcribe: false,
        openApiCheck: false
      },
      onActionIconClickChange: vi.fn(),
      onContextMenuClickChange: vi.fn()
    })

    expect(mocks.contextMenusCreate).toHaveBeenCalledWith(
      expect.objectContaining({ id: "save-to-clipper-pa" })
    )

    mocks.contextMenusCreate.mockClear()
    mocks.contextMenusRemove.mockClear()
    mocks.getServerCapabilities.mockResolvedValueOnce({
      hasWebClipper: false
    })

    watchHandlers.tldwConfig?.({
      oldValue: { serverUrl: "http://127.0.0.1:8000" },
      newValue: { serverUrl: "http://127.0.0.1:9000" }
    })
    await flushPromises()

    expect(mocks.contextMenusRemove).toHaveBeenCalledWith("save-to-clipper-pa")
    expect(mocks.contextMenusCreate).not.toHaveBeenCalledWith(
      expect.objectContaining({ id: "save-to-clipper-pa" })
    )
  })
})
