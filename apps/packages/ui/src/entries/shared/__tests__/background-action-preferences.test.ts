import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const browserApi = vi.hoisted(() => ({
  contextMenus: { removeAll: vi.fn(), create: vi.fn(), remove: vi.fn() },
  alarms: { clear: vi.fn(), create: vi.fn() },
  i18n: { getMessage: (key: string) => key }
}))
vi.mock("wxt/browser", () => ({ browser: browserApi }))
vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasWebClipper: false }))
}))

beforeEach(() => {
  vi.resetModules()
  vi.clearAllMocks()
  window.localStorage.clear()
  window.sessionStorage.clear()
  const listeners = new Set<
    (changes: Record<string, unknown>, area: string) => void
  >()
  const area = (name: string) => {
    const values: Record<string, unknown> = {}
    return {
      get: async (keys?: string | string[]) =>
        Object.fromEntries(
          (keys === undefined
            ? Object.keys(values)
            : Array.isArray(keys)
              ? keys
              : [keys]
          )
            .filter((key) => key in values)
            .map((key) => [key, values[key]])
        ),
      set: async (items: Record<string, unknown>) => {
        const changes = Object.fromEntries(
          Object.entries(items).map(([key, newValue]) => [
            key,
            { oldValue: values[key], newValue }
          ])
        )
        Object.assign(values, items)
        for (const listener of listeners) listener(changes, name)
      },
      remove: async (key: string) => {
        const oldValue = values[key]
        delete values[key]
        for (const listener of listeners)
          listener({ [key]: { oldValue } }, name)
      }
    }
  }
  const storage = {
    local: area("local"),
    sync: area("sync"),
    session: area("session"),
    onChanged: {
      addListener: (
        listener: (changes: Record<string, unknown>, area: string) => void
      ) => listeners.add(listener),
      removeListener: (
        listener: (changes: Record<string, unknown>, area: string) => void
      ) => listeners.delete(listener)
    }
  }
  vi.stubGlobal("chrome", { storage })
  vi.stubGlobal("browser", { storage })
})

afterEach(async () => {
  await new Promise((resolve) => setTimeout(resolve, 0))
  vi.unstubAllGlobals()
})

async function initialize() {
  const { initBackground } = await import("../background-init")
  const { createSafeStorage } = await import("@/utils/safe-storage")
  const action = vi.fn(),
    contextMenu = vi.fn(),
    warmModels = vi.fn(async () => [])
  const storage = createSafeStorage({ area: "local" })
  await initBackground({
    storage,
    contextMenuId: { webui: "web-menu", sidePanel: "side-menu" },
    saveToClipperMenuId: "clipper",
    saveToCompanionMenuId: "companion",
    saveToNotesMenuId: "notes",
    narrateSelectionMenuId: "narrate",
    transcribeMenuId: {
      transcribe: "transcribe",
      transcribeAndSummarize: "summary"
    },
    warmModels,
    capabilities: {
      sendToTldw: false,
      processLocal: false,
      transcribe: false,
      openApiCheck: false
    },
    onActionIconClickChange: action,
    onContextMenuClickChange: contextMenu
  })
  return { action, contextMenu, storage, warmModels }
}

describe("background action preferences across browser storage areas", () => {
  it("applies the saved toolbar choice without restarting the worker", async () => {
    const { action } = await initialize()
    const { ACTION_ICON_CLICK_SETTING } = await import("@/services/action")
    const { setSetting } = await import("@/services/settings/registry")
    expect(action).toHaveBeenLastCalledWith("webui")
    await setSetting(ACTION_ICON_CLICK_SETTING, "sidePanel")
    await vi.waitFor(() => expect(action).toHaveBeenLastCalledWith("sidePanel"))
  })

  it("updates the existing context menu after its saved choice changes", async () => {
    const { contextMenu } = await initialize()
    const { CONTEXT_MENU_CLICK_SETTING } = await import("@/services/action")
    const { setSetting } = await import("@/services/settings/registry")
    await setSetting(CONTEXT_MENU_CLICK_SETTING, "webui")
    await vi.waitFor(() =>
      expect(contextMenu).toHaveBeenLastCalledWith("webui")
    )
    expect(browserApi.contextMenus.remove).toHaveBeenCalledWith("side-menu")
    expect(browserApi.contextMenus.create).toHaveBeenCalledWith(
      expect.objectContaining({ id: "web-menu" })
    )
  })

  it.each(["actionIconClick", "contextMenuClick"] as const)(
    "restores %s's declared default when the saved choice is removed",
    async (key) => {
      const callbacks = await initialize()
      const { ACTION_ICON_CLICK_SETTING, CONTEXT_MENU_CLICK_SETTING } =
        await import("@/services/action")
      const { setSetting, clearSetting } = await import(
        "@/services/settings/registry"
      )
      const setting =
        key === "actionIconClick"
          ? ACTION_ICON_CLICK_SETTING
          : CONTEXT_MENU_CLICK_SETTING
      const callback =
        key === "actionIconClick" ? callbacks.action : callbacks.contextMenu
      const changed = setting.defaultValue === "webui" ? "sidePanel" : "webui"
      await setSetting(setting, changed)
      await vi.waitFor(() => expect(callback).toHaveBeenLastCalledWith(changed))
      await clearSetting(setting)
      await vi.waitFor(() =>
        expect(callback).toHaveBeenLastCalledWith(setting.defaultValue)
      )
    }
  )

  it.each([undefined, "local"] as const)(
    "invalidates a cached preference in the actual %s storage area",
    async (area) => {
      const { defineSetting, getSetting, getStorageForSetting } = await import(
        "@/services/settings/registry"
      )
      const setting = defineSetting("x04CachedChoice", "webui", { area })
      const storage = getStorageForSetting(setting)
      await storage.set(setting.key, "webui")
      expect(await getSetting(setting)).toBe("webui")
      await storage.set(setting.key, "sidePanel")
      expect(await getSetting(setting)).toBe("sidePanel")
    }
  )

  it("keeps connection changes on the local authentication store", async () => {
    const { storage, warmModels } = await initialize()
    await storage.set("tldwConfig", { serverUrl: "http://127.0.0.1:8000" })
    await vi.waitFor(() => expect(warmModels).toHaveBeenCalledWith(true))
  })
})
