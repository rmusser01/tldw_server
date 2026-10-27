import {
  spyOnStorageGet,
  spyOnStorageSet,
  storageOriginal
} from "@/hooks/chat/__tests__/native-plasmo-storage-fixture"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"
import {
  defineSetting,
  getSetting,
  getStorageForSetting
} from "@/services/settings/registry"
import { useSetting } from "../useSetting"

const setting = defineSetting("fixture:hydration:disabled", false, {
  area: "local",
  localStorageKey: "fixture:hydration:disabled",
  mirrorToLocalStorage: true,
  coerce: (value) => value === true || value === "true"
})
const storage = getStorageForSetting(setting)
beforeEach(async () => {
  localStorage.clear()
  await storage.set(setting.key, false)
})

const holdFirstRead = () => {
  let release!: (value: unknown) => void
  let admitted = false
  const held = new Promise((resolve) => {
    release = resolve
  })
  spyOnStorageGet().mockImplementation(function (key: string) {
    if (key === setting.key && !admitted) {
      admitted = true
      return held
    }
    return storageOriginal.get.call(this, key)
  })
  return { release, admitted: () => admitted }
}

describe("owned setting hydration", () => {
  it("does not publish an old initial value after an explicit choice", async () => {
    const read = holdFirstRead()
    const hook = renderHook(() => useSetting(setting))
    await waitFor(() => expect(read.admitted()).toBe(true))
    await act(async () => {
      await hook.result.current[1](true)
    })
    expect(hook.result.current[0]).toBe(true)
    await act(async () => {
      read.release(false)
    })
    await waitFor(() => expect(hook.result.current[2].isLoading).toBe(false))
    expect(hook.result.current[0]).toBe(true)
    expect(await storageOriginal.get.call(storage, setting.key)).toBe(true)
    expect(localStorage.getItem(setting.localStorageKey!)).toBe("true")
  })

  it("does not let a retired normalization read overwrite a newer stored choice", async () => {
    const read = holdFirstRead()
    let current = true
    const pending = getSetting(setting, { isCurrent: () => current })
    await waitFor(() => expect(read.admitted()).toBe(true))
    current = false
    await storage.set(setting.key, true)
    localStorage.setItem(setting.localStorageKey!, "true")
    read.release("false")
    await pending
    expect(await storageOriginal.get.call(storage, setting.key)).toBe(true)
    expect(localStorage.getItem(setting.localStorageKey!)).toBe("true")
  })

  it("keeps later watch changes after refusing a superseded initial read", async () => {
    const read = holdFirstRead()
    const hook = renderHook(() => useSetting(setting))
    await waitFor(() => expect(read.admitted()).toBe(true))
    await act(async () => {
      await storage.set(setting.key, true)
    })
    await waitFor(() => expect(hook.result.current[0]).toBe(true))
    await act(async () => {
      read.release(false)
    })
    expect(hook.result.current[0]).toBe(true)
    await act(async () => {
      await storage.set(setting.key, false)
    })
    await waitFor(() => expect(hook.result.current[0]).toBe(false))
  })
  it("hydrates a replacement key and refuses the previous held read", async () => {
    const replacement = defineSetting("fixture:hydration:replacement", false, {
      area: "local"
    })
    await getStorageForSetting(replacement).set(replacement.key, true)
    const read = holdFirstRead()
    const hook = renderHook(({ current }) => useSetting(current), {
      initialProps: { current: setting }
    })
    await waitFor(() => expect(read.admitted()).toBe(true))
    hook.rerender({ current: replacement })
    await waitFor(() => expect(hook.result.current[0]).toBe(true))
    await act(async () => {
      read.release(false)
    })
    expect(hook.result.current[0]).toBe(true)
  })

  it("keeps an explicit choice through an equivalent definition rerender", async () => {
    const read = holdFirstRead()
    const hook = renderHook(({ current }) => useSetting(current), {
      initialProps: { current: setting }
    })
    await waitFor(() => expect(read.admitted()).toBe(true))
    hook.rerender({ current: { ...setting } })
    spyOnStorageSet().mockRejectedValueOnce(
      new Error("Preference write unavailable")
    )
    await act(async () => {
      await expect(hook.result.current[1](true)).rejects.toThrow(
        "Preference write unavailable"
      )
    })
    await act(async () => {
      read.release(false)
    })
    expect(hook.result.current[0]).toBe(true)
  })

  it("finishes legacy normalization and mirror cleanup after its own watch event", async () => {
    const legacy = defineSetting("fixture:hydration:legacy", false, {
      area: "local",
      localStorageKey: "fixture:hydration:old",
      coerce: (value) => value === true || value === "true"
    })
    localStorage.setItem(legacy.localStorageKey!, "true")
    const hook = renderHook(() => useSetting(legacy))
    await waitFor(() => expect(hook.result.current[0]).toBe(true))
    await waitFor(() => expect(hook.result.current[2].isLoading).toBe(false))
    expect(await storageOriginal.get.call(storage, legacy.key)).toBe(true)
    expect(localStorage.getItem(legacy.localStorageKey!)).toBeNull()
  })

  it("preserves a same-value normalized watch after removing a stored preference", async () => {
    await storage.set(setting.key, true)
    const read = holdFirstRead()
    const hook = renderHook(() => useSetting(setting))
    await waitFor(() => expect(read.admitted()).toBe(true))
    await act(async () => {
      await storage.remove(setting.key)
    })
    await act(async () => {
      read.release(true)
    })
    await waitFor(() => expect(hook.result.current[2].isLoading).toBe(false))
    expect(hook.result.current[0]).toBe(false)
  })

  it("retires a normalization read on unmount before any derived write", async () => {
    await storage.set(setting.key, "false")
    const read = holdFirstRead()
    const hook = renderHook(() => useSetting(setting))
    await waitFor(() => expect(read.admitted()).toBe(true))
    hook.unmount()
    await act(async () => {
      read.release("false")
    })
    expect(await storageOriginal.get.call(storage, setting.key)).toBe("false")
  })

  it("moves the current read and watcher to a replacement storage area", async () => {
    const replacement = defineSetting("fixture:hydration:sync", false, {
      area: "sync"
    })
    const sync = getStorageForSetting(replacement)
    await sync.set(replacement.key, true)
    const hook = renderHook(({ current }) => useSetting(current), {
      initialProps: { current: setting }
    })
    await waitFor(() => expect(hook.result.current[2].isLoading).toBe(false))
    hook.rerender({ current: replacement })
    await waitFor(() => expect(hook.result.current[0]).toBe(true))
    await act(async () => {
      await sync.set(replacement.key, false)
    })
    await waitFor(() => expect(hook.result.current[0]).toBe(false))
    await act(async () => {
      await storage.set(setting.key, true)
    })
    expect(hook.result.current[0]).toBe(false)
  })

  it("resolves consecutive functional choices from the latest explicit value", async () => {
    const hook = renderHook(() => useSetting(setting))
    await waitFor(() => expect(hook.result.current[2].isLoading).toBe(false))
    await act(async () => {
      const first = hook.result.current[1]((previous) => !previous)
      const second = hook.result.current[1]((previous) => !previous)
      await Promise.all([first, second])
    })
    expect(hook.result.current[0]).toBe(false)
    expect(await storageOriginal.get.call(storage, setting.key)).toBe(false)
  })
  it("preserves watches when its implicit default area becomes explicit", async () => {
    const implicit = defineSetting("fixture:hydration:implicit-area", false)
    const sync = getStorageForSetting(implicit)
    await sync.set(implicit.key, false)
    const hook = renderHook(({ current }) => useSetting(current), {
      initialProps: { current: implicit }
    })
    await waitFor(() => expect(hook.result.current[2].isLoading).toBe(false))
    const explicit = { ...implicit, area: sync.area }
    hook.rerender({ current: explicit })
    await act(async () => {
      await getStorageForSetting(explicit).set(implicit.key, true)
    })
    await waitFor(() => expect(hook.result.current[0]).toBe(true))
  })
})
