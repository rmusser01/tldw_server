import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const storageBacking = {
  values: new Map<string, unknown>(),
  getCalls: [] as string[],
  setCalls: [] as Array<{ key: string; value: unknown }>,
  removeCalls: [] as string[],
  changeListeners: new Set<
    (changes: Record<string, { oldValue?: unknown; newValue?: unknown }>, area: string) => void
  >()
}

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: ({ area = "local" }: { area?: string } = {}) => ({
    get: async (key: string) => {
      storageBacking.getCalls.push(`${area}::${key}`)
      return storageBacking.values.get(`${area}::${key}`)
    },
    set: async (key: string, value: unknown) => {
      storageBacking.setCalls.push({ key: `${area}::${key}`, value })
      const previous = storageBacking.values.get(`${area}::${key}`)
      storageBacking.values.set(`${area}::${key}`, value)
      for (const listener of storageBacking.changeListeners) {
        listener({ [key]: { oldValue: previous, newValue: value } }, area)
      }
    },
    remove: async (key: string) => {
      storageBacking.removeCalls.push(`${area}::${key}`)
      const previous = storageBacking.values.get(`${area}::${key}`)
      storageBacking.values.delete(`${area}::${key}`)
      if (previous !== undefined) {
        for (const listener of storageBacking.changeListeners) {
          listener({ [key]: { oldValue: previous, newValue: undefined } }, area)
        }
      }
    }
  })
}))

// chrome.storage.onChanged stub so tests can observe cross-context invalidation.
vi.stubGlobal("chrome", {
  storage: {
    onChanged: {
      addListener: (listener: (typeof storageBacking.changeListeners) extends Set<infer L> ? L : never) =>
        storageBacking.changeListeners.add(listener)
    }
  }
})

import {
  clearSetting,
  defineSetting,
  getSetting,
  setSetting
} from "@/services/settings/registry"

beforeEach(() => {
  storageBacking.values.clear()
  storageBacking.getCalls.length = 0
  storageBacking.setCalls.length = 0
  storageBacking.removeCalls.length = 0
  window.localStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("settings registry cache + mirror (WP-20)", () => {
  it("issues no writes for 100 reads after the first mirror", async () => {
    const setting = defineSetting("cachedTheme", "dark" as string, {
      localStorageKey: "cachedTheme",
      mirrorToLocalStorage: true
    })
    storageBacking.values.set("local::cachedTheme", "light")
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem")
    const removeItemSpy = vi.spyOn(Storage.prototype, "removeItem")

    for (let i = 0; i < 100; i += 1) {
      expect(await getSetting(setting)).toBe("light")
    }

    // First read: one chrome-side read; the mirror write happens once.
    expect(storageBacking.getCalls.filter((key) => key === "local::cachedTheme")).toHaveLength(1)
    expect(storageBacking.setCalls).toEqual([])
    expect(setItemSpy).toHaveBeenCalledTimes(1)
    expect(removeItemSpy).not.toHaveBeenCalled()
    expect(window.localStorage.getItem("cachedTheme")).toBe("light")
  })

  it("skips the mirror write when localStorage already holds the serialized value", async () => {
    const setting = defineSetting("premirrored", "dark" as string, {
      localStorageKey: "premirrored",
      mirrorToLocalStorage: true
    })
    storageBacking.values.set("local::premirrored", "light")
    // Another context already wrote the mirror (string settings mirror raw).
    window.localStorage.setItem("premirrored", "light")
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem")

    expect(await getSetting(setting)).toBe("light")
    expect(await getSetting(setting)).toBe("light")

    expect(setItemSpy).not.toHaveBeenCalled()
  })

  it("mirrors on set only when the normalized value changes", async () => {
    const setting = defineSetting("mirrorOnSet", "a" as string, {
      localStorageKey: "mirrorOnSet",
      mirrorToLocalStorage: true
    })
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem")

    await setSetting(setting, "b" as never)
    await setSetting(setting, "b" as never)
    await setSetting(setting, "c" as never)

    expect(setItemSpy).toHaveBeenCalledTimes(2)
    expect(window.localStorage.getItem("mirrorOnSet")).toBe("c")
  })

  it("invalidates the cache when another context writes via storage.onChanged", async () => {
    const setting = defineSetting("crossContext", "dark" as string)
    storageBacking.values.set("local::crossContext", "dark")
    expect(await getSetting(setting)).toBe("dark")
    expect(await getSetting(setting)).toBe("dark")
    expect(storageBacking.getCalls.filter((key) => key === "local::crossContext")).toHaveLength(1)

    // A different context (options page) writes the same key.
    storageBacking.values.set("local::crossContext", "light")
    for (const listener of storageBacking.changeListeners) {
      listener({ crossContext: { newValue: "light" } }, "local")
    }

    expect(await getSetting(setting)).toBe("light")
    expect(storageBacking.getCalls.filter((key) => key === "local::crossContext")).toHaveLength(2)
  })

  it("still migrates a legacy localStorage value into chrome storage exactly once", async () => {
    const setting = defineSetting("legacyMigrated", "default" as string, {
      localStorageKey: "legacyMigrated",
      mirrorToLocalStorage: true
    })
    window.localStorage.setItem("legacyMigrated", '"legacy-value"')

    expect(await getSetting(setting)).toBe("legacy-value")
    expect(storageBacking.setCalls).toEqual([{ key: "local::legacyMigrated", value: "legacy-value" }])

    // The second read is served from the cache: no further writes.
    expect(await getSetting(setting)).toBe("legacy-value")
    expect(storageBacking.setCalls).toHaveLength(1)
  })

  it("drops the cached value on clearSetting", async () => {
    const setting = defineSetting("clearedSetting", "x" as string)
    await setSetting(setting, "y" as never)
    expect(await getSetting(setting)).toBe("y")
    expect(storageBacking.getCalls.filter((key) => key === "local::clearedSetting")).toHaveLength(0)

    await clearSetting(setting)
    expect(await getSetting(setting)).toBe("x")
  })
})
