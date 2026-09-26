// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: () => () => {}
}))

const makeTabStorage = () => {
  const items = new Map<string, string>()
  return {
    getItem: (key: string) => items.get(key) ?? null,
    setItem: (key: string, value: string) => { items.set(key, value) },
    removeItem: (key: string) => { items.delete(key) }
  }
}

const openTab = async (storage = makeTabStorage()) => {
  vi.stubGlobal("sessionStorage", storage)
  vi.resetModules()
  return { storage, store: (await import("../playground-session")).usePlaygroundSessionStore }
}

const normal = { historyId: "normal-local", serverChatId: "normal-server", scopeKey: "account-a:org-2", chatMode: "normal" as const }
const character = { historyId: "character-local", serverChatId: "character-server", scopeKey: "account-a:org-2", trackedAssistantKind: "character" as const, trackedAssistantId: "testbot", trackedCharacterId: "testbot", trackedAssistantDisplayName: "TestBot" }

describe("tab-owned playground session persistence", () => {
  beforeEach(() => localStorage.clear())
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("restores each saved normal and character chat after the other tab changes", async () => {
    const first = await openTab()
    first.store.getState().saveSession(normal)
    const second = await openTab()
    second.store.getState().saveSession(character)
    await first.store.persist.rehydrate()
    expect(first.store.getState()).toMatchObject({ ...normal, trackedAssistantKind: null })
    first.store.getState().saveSession({ webSearch: true })
    await second.store.persist.rehydrate()
    expect(second.store.getState()).toMatchObject(character)
  })

  it("retains the durable last-session fallback for a new tab", async () => {
    const first = await openTab()
    first.store.getState().saveSession(character)
    const reopened = await openTab()
    expect(reopened.store.getState()).toMatchObject(character)
  })

  it("pins a restored legacy session before another tab changes the durable fallback", async () => {
    localStorage.setItem("tldw-playground-session", JSON.stringify({ version: 1, state: { ...normal, lastUpdated: Date.now() } }))
    const first = await openTab()
    const second = await openTab()
    second.store.getState().saveSession(character)
    await first.store.persist.rehydrate()
    expect(first.store.getState()).toMatchObject(normal)
  })

  it("keeps scope validation and cleared-session reloads intact", async () => {
    const tab = await openTab()
    tab.store.getState().saveSession(character)
    expect(tab.store.getState().isSessionValid("account-b:org-3")).toBe(false)
    tab.store.getState().clearSession()
    await tab.store.persist.rehydrate()
    expect(tab.store.getState().isSessionValid("account-a:org-2")).toBe(false)
  })

  it.each(["getter", "read", "pin"])("recovers the durable session when tab storage fails during %s", async (failure) => {
    const saved = await openTab()
    saved.store.getState().saveSession(character)
    const storage = makeTabStorage()
    vi.stubGlobal("sessionStorage", storage)
    if (failure === "getter") {
      vi.spyOn(window, "sessionStorage", "get").mockImplementation(() => { throw new Error("Blocked") })
    } else {
      vi.spyOn(storage, failure === "read" ? "getItem" : "setItem").mockImplementation(() => { throw new Error("Blocked") })
    }
    vi.resetModules()
    const store = (await import("../playground-session")).usePlaygroundSessionStore
    expect(store.getState()).toMatchObject(character)
    expect(() => store.getState().saveSession(normal)).not.toThrow()
    expect(JSON.parse(localStorage.getItem("tldw-playground-session")!).state).toMatchObject(normal)
  })

  it("recovers the new durable session instead of an old tab pin after a quota failure", async () => {
    const tab = await openTab()
    tab.store.getState().saveSession(normal)
    const write = vi.spyOn(tab.storage, "setItem").mockImplementation(() => { throw new Error("Quota exceeded") })
    expect(() => tab.store.getState().saveSession(character)).not.toThrow()
    write.mockRestore()
    const reloaded = await openTab(tab.storage)
    expect(reloaded.store.getState()).toMatchObject(character)
  })

  it("retains tab recovery when durable storage writes fail", async () => {
    const tab = await openTab()
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("Quota exceeded") })
    expect(() => tab.store.getState().saveSession(character)).not.toThrow()
    const reloaded = await openTab(tab.storage)
    expect(reloaded.store.getState()).toMatchObject(character)
  })

  it("clears the durable session even when tab storage rejects writes", async () => {
    const tab = await openTab()
    tab.store.getState().saveSession(character)
    vi.spyOn(tab.storage, "setItem").mockImplementation(() => { throw new Error("Quota exceeded") })
    expect(() => tab.store.getState().clearSession()).not.toThrow()
    const reloaded = await openTab()
    expect(reloaded.store.getState().isSessionValid("account-a:org-2")).toBe(false)
  })
})
