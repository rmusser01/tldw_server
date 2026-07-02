import fs from "node:fs"
import path from "node:path"
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
    }),
    set: vi.fn(async () => undefined)
  }
  const warmModels = vi.fn(async () => null)

  it("reads quickstart deployment mode from extension-safe import.meta.env", () => {
    const source = fs.readFileSync(
      path.resolve(__dirname, "../background-init.ts"),
      "utf8"
    )

    expect(source).toContain("import.meta.env")
    expect(source).toContain("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE")
  })

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
        transcribe: false,
        openApiCheck: false
      },
      onActionIconClickChange: vi.fn(),
      onContextMenuClickChange: vi.fn()
    })
    await flushPromises()

    expect(mocks.contextMenusCreate).not.toHaveBeenCalledWith(
      expect.objectContaining({ id: "save-to-clipper-pa" })
    )
  })

  it("skips same-origin WebUI OpenAPI drift probes", async () => {
    const previousFetch = globalThis.fetch
    const previousMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    storedTldwConfig = {
      serverUrl: window.location.origin,
      authMode: "single-user",
      apiKey: "test-api-key"
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
          transcribe: false,
          openApiCheck: true
        },
        onActionIconClickChange: vi.fn(),
        onContextMenuClickChange: vi.fn()
      })
      await flushPromises()

      expect(fetchSpy).not.toHaveBeenCalled()
    } finally {
      vi.stubGlobal("fetch", previousFetch)
      if (previousMode === undefined) {
        delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
      } else {
        process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = previousMode
      }
    }
  })

  it("keeps advanced same-origin OpenAPI drift probes enabled", async () => {
    const previousFetch = globalThis.fetch
    const previousMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    const fetchSpy = vi.fn(async () => ({ ok: false }))
    vi.stubGlobal("fetch", fetchSpy)
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "advanced"
    storedTldwConfig = {
      serverUrl: window.location.origin,
      authMode: "single-user",
      apiKey: "test-api-key"
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
          transcribe: false,
          openApiCheck: true
        },
        onActionIconClickChange: vi.fn(),
        onContextMenuClickChange: vi.fn()
      })
      await flushPromises()

      expect(fetchSpy).toHaveBeenCalledWith(
        `${window.location.origin}/openapi.json`,
        expect.objectContaining({
          headers: { "X-API-KEY": "test-api-key" }
        })
      )
    } finally {
      vi.stubGlobal("fetch", previousFetch)
      if (previousMode === undefined) {
        delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
      } else {
        process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = previousMode
      }
    }
  })

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
          transcribe: false,
          openApiCheck: false
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
