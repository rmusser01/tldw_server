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

vi.mock("@/services/action", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/action")>(),
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
  // Let all immediately resolved storage reads settle before restoring globals.
  await new Promise<void>((resolve) => setTimeout(resolve, 0))
}

describe("background clipper rollout guard", () => {
  const watchHandlers: Record<string, ((value: any) => void) | undefined> = {}
  let storedTldwConfig: Record<string, unknown>
  const storage = {
    watch: vi.fn((handlers: Record<string, (value: any) => void>) => {
      Object.assign(watchHandlers, handlers)
    }),
    get: vi.fn(async (key: string) => {
      if (key === "tldwConfig") {
        return storedTldwConfig
      }
      return null
    })
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
    storedTldwConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user"
    }
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
        transcribe: false
      },
      onActionIconClickChange: vi.fn(),
      onContextMenuClickChange: vi.fn()
    })
    await flushPromises()

    expect(mocks.contextMenusCreate).not.toHaveBeenCalledWith(
      expect.objectContaining({ id: "save-to-clipper-pa" })
    )
  })

  it.each([window.location.origin, "https://api.example.test"])(
    "does not send a separate OpenAPI drift probe at startup for %s",
    async (serverUrl) => {
      const fetchSpy = vi.fn(async () => ({ ok: false }))
      vi.stubGlobal("fetch", fetchSpy)
      storedTldwConfig = {
        serverUrl,
        authMode: "single-user",
        apiKey: "test-api-key",
        credentialSource: "manual",
        apiKeyPersistence: "device",
        apiKeyServerOrigin: serverUrl
      }
      try {
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
            transcribe: false
          },
          onActionIconClickChange: vi.fn(),
          onContextMenuClickChange: vi.fn()
        })
        await flushPromises()

        expect(fetchSpy).not.toHaveBeenCalled()
      } finally {
        vi.unstubAllGlobals()
      }
    }
  )

  it("does not block startup on capability refreshes or model warmup", async () => {
    let resolveCapabilities: (value: { hasWebClipper: boolean }) => void = () => {}
    mocks.getServerCapabilities.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveCapabilities = resolve
        })
    )
    warmModels.mockImplementationOnce(
      () => new Promise(() => undefined)
    )

    const result = await Promise.race([
      initBackground({
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
          transcribe: false
        },
        onActionIconClickChange: vi.fn(),
        onContextMenuClickChange: vi.fn()
      }),
      new Promise((resolve) => setTimeout(() => resolve("timeout"), 25))
    ])

    expect(result).toEqual(
      expect.objectContaining({ modelWarmAlarmName: "tldw:model-warm" })
    )

    resolveCapabilities({ hasWebClipper: true })
    await flushPromises()
  })

  it("cold start warms models non-forced so fresh-cache wakes are no-ops", async () => {
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
        transcribe: false
      },
      onActionIconClickChange: vi.fn(),
      onContextMenuClickChange: vi.fn()
    })

    // WP-61: the cold-start warm must NOT force a refresh_openrouter=true
    // round trip on every worker wake; warmModels(false) returns the
    // 15-min TTL catalog without network traffic while fresh.
    expect(warmModels).toHaveBeenCalledTimes(1)
    expect(warmModels).toHaveBeenCalledWith(false)

    // The periodic warm alarm stays scheduled (relaxed 6h cadence); the
    // alarm handler's isCatalogFresh() gate plus the TTL keep idle fires
    // no-ops.
    expect(mocks.alarmsCreate).toHaveBeenCalledWith(
      "tldw:model-warm",
      expect.objectContaining({ periodInMinutes: 360 })
    )
  })

  it("no-ops the clipper context-menu sync when a config write keeps the server URL", async () => {
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
        transcribe: false
      },
      onActionIconClickChange: vi.fn(),
      onContextMenuClickChange: vi.fn()
    })
    await flushPromises()

    expect(mocks.contextMenusCreate).toHaveBeenCalledWith(
      expect.objectContaining({ id: "save-to-clipper-pa" })
    )

    mocks.contextMenusCreate.mockClear()
    mocks.contextMenusRemove.mockClear()
    mocks.getServerCapabilities.mockClear()

    // WP-62: an unrelated config write (e.g. an API-key refresh) must not
    // rebuild the clipper context menu or re-resolve server capabilities.
    watchHandlers.tldwConfig?.({
      oldValue: { serverUrl: "http://127.0.0.1:8000", apiKey: "old-key" },
      newValue: { serverUrl: "http://127.0.0.1:8000", apiKey: "new-key" }
    })
    await flushPromises()

    expect(mocks.getServerCapabilities).not.toHaveBeenCalled()
    expect(mocks.contextMenusRemove).not.toHaveBeenCalledWith(
      "save-to-clipper-pa"
    )
    expect(mocks.contextMenusCreate).not.toHaveBeenCalledWith(
      expect.objectContaining({ id: "save-to-clipper-pa" })
    )
  })

  it("re-syncs the clipper context menu when the server URL changes", async () => {
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
        transcribe: false
      },
      onActionIconClickChange: vi.fn(),
      onContextMenuClickChange: vi.fn()
    })
    await flushPromises()

    mocks.getServerCapabilities.mockClear()

    watchHandlers.tldwConfig?.({
      oldValue: { serverUrl: "http://127.0.0.1:8000" },
      newValue: { serverUrl: "http://127.0.0.1:9000" }
    })
    await flushPromises()

    expect(mocks.getServerCapabilities).toHaveBeenCalledTimes(1)
    expect(mocks.contextMenusRemove).toHaveBeenCalledWith("save-to-clipper-pa")
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
        transcribe: false
      },
      onActionIconClickChange: vi.fn(),
      onContextMenuClickChange: vi.fn()
    })
    await flushPromises()

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
