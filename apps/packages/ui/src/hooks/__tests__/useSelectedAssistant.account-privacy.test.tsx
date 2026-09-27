// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Storage } from "@plasmohq/storage"

const authority = vi.hoisted(() => ({ user: "alice" as string | null, pending: null as Promise<void> | null, plasmo: false }))
vi.mock("@plasmohq/storage", () => import("../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@plasmohq/storage/hook", async importOriginal => {
  const actual = await importOriginal<typeof import("@plasmohq/storage/hook")>()
  const web = await import("../../../../../tldw-frontend/extension/shims/plasmo-storage-hook")
  return { useStorage: (...args: Parameters<typeof web.useStorage>) => authority.plasmo ? actual.useStorage(...args) : web.useStorage(...args) }
})
vi.mock("@/services/service-prompts", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/service-prompts")>(),
  resolveServicePromptScope: async () => {
    await authority.pending
    if (!authority.user) throw new Error("Signed out")
    return { scopeKey: authority.user, userId: authority.user,
      config: { serverUrl: "http://localhost:8000", authMode: "multi-user", authSource: "manual", orgId: 2 } }
  }
}))

import { useSelectedAssistant } from "../useSelectedAssistant"

const alice = { kind: "character" as const, id: "4", name: "Alice private TestBot", system_prompt: "Alice private prompt", greeting: "Alice private greeting" }
const bob = { kind: "character" as const, id: "5", name: "Bob private helper", system_prompt: "Bob private prompt" }
const settle = async () => { await act(async () => { await vi.dynamicImportSettled() }) }
const switchTo = async (user: string) => {
  act(() => { authority.user = null; window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")) })
  await act(async () => { authority.user = user; window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } })) })
  await settle()
}
const mount = async () => {
  const view = renderHook(() => useSelectedAssistant())
  await waitFor(() => expect(view.result.current[2].isLoading).toBe(false))
  return view
}

describe("private assistant account ownership with native storage adapters", () => {
  beforeEach(() => { localStorage.clear(); authority.user = "alice"; authority.pending = null; authority.plasmo = false })
  afterEach(() => vi.restoreAllMocks())

  it.each(["selectedAssistant", "plasmo-sync:selectedAssistant", "selectedCharacter", "plasmo-sync:selectedCharacter"])("rejects an unowned legacy %s snapshot", async key => {
    localStorage.setItem(key, JSON.stringify(alice))
    const view = await mount()
    await settle()
    expect(view.result.current[0]).toBeNull()
    await waitFor(() => expect(localStorage.getItem(key)).toBeNull())
  })

  it("clears the private selection on logout before resolving the next account", async () => {
    const view = await mount()
    await act(async () => { await view.result.current[1](alice) })
    expect(view.result.current[0]?.name).toBe(alice.name)
    act(() => { authority.user = null; window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")) })
    expect(view.result.current[0]).toBeNull()
    await switchTo("bob")
    expect(view.result.current[0]).toBeNull()
  })

  it("recovers only the same owner's selection on reload and reciprocal login", async () => {
    let view = await mount()
    await act(async () => { await view.result.current[1](alice) })
    view.unmount()
    view = await mount()
    expect(view.result.current[0]).toMatchObject(alice)
    await switchTo("bob")
    expect(view.result.current[0]).toBeNull()
    await act(async () => { await view.result.current[1](bob) })
    await switchTo("alice")
    await waitFor(() => expect(view.result.current[0]).toMatchObject(alice))
  })

  it("does not let a delayed Alice write become Bob's active selection", async () => {
    const view = await mount()
    const originalSet = Storage.prototype.set
    let release!: () => void
    let started = false
    const barrier = new Promise<void>(resolve => { release = resolve })
    vi.spyOn(Storage.prototype, "set").mockImplementation(async function (key, value) {
      if (JSON.stringify(value).includes(alice.name)) { started = true; await barrier }
      return originalSet.call(this, key, value)
    })
    let pending!: Promise<void>
    act(() => { pending = view.result.current[1](alice) })
    await waitFor(() => expect(started).toBe(true))
    await switchTo("bob")
    await act(async () => { release(); await pending })
    expect(view.result.current[0]).toBeNull()
    view.unmount()
    const reloaded = await mount()
    expect(reloaded.result.current[0]).toBeNull()
  })

  it("retains the same user's selection on an ordinary config refresh", async () => {
    const view = await mount()
    await act(async () => { await view.result.current[1](alice) })
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } })))
    await settle()
    expect(view.result.current[0]).toMatchObject(alice)
  })

  it("rejects a delayed owned read after the account changes", async () => {
    const view = await mount()
    await act(async () => { await view.result.current[1](alice) })
    const originalGet = Storage.prototype.get
    let release!: () => void
    let started = false
    const barrier = new Promise<void>(resolve => { release = resolve })
    vi.spyOn(Storage.prototype, "get").mockImplementation(async function (key) {
      const value = await originalGet.call(this, key)
      if (key.startsWith("selectedAssistant:owner:") && JSON.stringify(value)?.includes(alice.name)) {
        started = true
        await barrier
      }
      return value
    })
    const pending = view.result.current[2].readSelection()
    await waitFor(() => expect(started).toBe(true))
    await switchTo("bob")
    await act(async () => { release(); expect(await pending).toBeNull() })
    expect(view.result.current[0]).toBeNull()
  })

  it("does not apply a delayed prior-owner hydration to the next account", async () => {
    const first = await mount()
    await act(async () => { await first.result.current[1](alice) })
    first.unmount()
    const originalGet = Storage.prototype.get
    let release!: () => void
    let started = false
    const barrier = new Promise<void>(resolve => { release = resolve })
    vi.spyOn(Storage.prototype, "get").mockImplementation(async function (key) {
      const value = await originalGet.call(this, key)
      if (key.startsWith("selectedAssistant:owner:") && JSON.stringify(value)?.includes(alice.name)) {
        started = true
        await barrier
      }
      return value
    })
    const view = renderHook(() => useSelectedAssistant())
    await waitFor(() => expect(started).toBe(true))
    await switchTo("bob")
    await act(async () => { release() })
    await settle()
    expect(view.result.current[0]).toBeNull()
  })

  it("rejects selection before the first account verification instead of acknowledging a no-op", async () => {
    let release!: () => void
    authority.pending = new Promise<void>(resolve => { release = resolve })
    const view = renderHook(() => useSelectedAssistant())
    await act(async () => {
      await expect(view.result.current[1](alice)).rejects.toThrow(/account/i)
      release()
    })
    await waitFor(() => expect(view.result.current[2].isLoading).toBe(false))
    expect(view.result.current[0]).toBeNull()
  })

  it("waits for the current owned key with the actual extension storage hook", async () => {
    authority.plasmo = true
    let verify!: () => void
    authority.pending = new Promise<void>(resolve => { verify = resolve })
    const originalGet = Storage.prototype.get
    let hydrate!: () => void
    let ownedReadStarted = false
    const barrier = new Promise<void>(resolve => { hydrate = resolve })
    vi.spyOn(Storage.prototype, "get").mockImplementation(async function (key) {
      const value = await originalGet.call(this, key)
      if (key.startsWith("selectedAssistant:owner:") && !key.endsWith(":unresolved")) {
        ownedReadStarted = true
        await barrier
      }
      return value
    })
    const view = renderHook(() => useSelectedAssistant())
    await settle()
    await act(async () => { verify() })
    await waitFor(() => expect(ownedReadStarted).toBe(true))
    expect(view.result.current[2].isLoading).toBe(true)
    await act(async () => { hydrate() })
    await waitFor(() => expect(view.result.current[2].isLoading).toBe(false))
    expect(view.result.current[0]).toBeNull()
  })

  it("keeps an explicit clear after reload", async () => {
    const view = await mount()
    await act(async () => { await view.result.current[1](alice) })
    await act(async () => { await view.result.current[1](null) })
    view.unmount()
    const reloaded = await mount()
    await settle()
    expect(reloaded.result.current[0]).toBeNull()
  })
})
