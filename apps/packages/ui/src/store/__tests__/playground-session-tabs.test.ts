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
  afterEach(() => vi.unstubAllGlobals())

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
})
