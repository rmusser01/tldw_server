import React from "react"
import { act, cleanup, render, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const { router, handlers, pop, nativeHistory } = vi.hoisted(() => {
  type State = { url: string; as: string; options: object; __N: boolean; key: string; extra?: string }
  type Entry = { href: string; state: State }
  const handlers = new Map<string, Set<(...args: unknown[]) => void>>()
  const pop = { handler: (_state: { url: string; as: string }) => true }
  const nativeHistory = {
    entries: [] as Entry[], index: 1, key: "source-key", nextKey: 0, isSsr: false,
    push: null as typeof window.history.pushState | null,
    replace: null as typeof window.history.replaceState | null,
    onPop: null as ((event: PopStateEvent) => void) | null
  }
  function move(href: string, method: "pushState" | "replaceState", options: object = {}): Promise<boolean> {
    try { router.events.emit("routeChangeStart", href, { shallow: false }) } catch (error) { return Promise.reject(error) }
    // Next16.1.4 coerces a push of the current asPath into replaceState.
    if (href === router.asPath) method = "replaceState"
    if (method === "pushState") nativeHistory.key = "next-key-" + ++nativeHistory.nextKey
    const state = { url: href, as: href, options, __N: true, key: nativeHistory.key }
    router.asPath = href
    window.history[method](state, "", href)
    router.events.emit("routeChangeComplete", href)
    return Promise.resolve(true)
  }
  const router = {
    asPath: "/media?id=1", pathname: "/media", query: { id: "1" },
    events: {
      on: (name: string, fn: (...args: unknown[]) => void) => { const set = handlers.get(name) || new Set(); set.add(fn); handlers.set(name, set) },
      off: (name: string, fn: (...args: unknown[]) => void) => { handlers.get(name)?.delete(fn) },
      emit: (name: string, ...args: unknown[]) => { for (const fn of [...(handlers.get(name) || [])]) fn(...args) }
    },
    beforePopState: (fn: typeof pop.handler) => { pop.handler = fn },
    push: (href: string): Promise<boolean> => move(href, "pushState"),
    replace: (href: string, options?: object): Promise<boolean> => move(href, "replaceState", options)
  }
  return { router, handlers, pop, nativeHistory }
})
vi.mock("next/router", () => ({ useRouter: () => router }))
vi.mock("react-router-dom", async () => await import("@web/extension/shims/react-router-dom"))
import { RouteLeavePrompt } from "@/entries/shared/route-leave-prompt"

beforeEach(() => {
  handlers.clear(); router.asPath = "/media?id=1"
  pop.handler = () => true
  nativeHistory.index = 1; nativeHistory.key = "source-key"; nativeHistory.nextKey = 0; nativeHistory.isSsr = false
  nativeHistory.entries = [
    { href: "/notes", state: { url: "/notes", as: "/notes", options: {}, __N: true, key: "notes-key" } },
    { href: "/media?id=1", state: { url: "/media?id=1", as: "/media?id=1", options: { shallow: true }, __N: true, key: "source-key", extra: "retained" } }
  ]
  nativeHistory.push = window.history.pushState.bind(window.history)
  nativeHistory.replace = window.history.replaceState.bind(window.history)
  nativeHistory.replace(nativeHistory.entries[1].state, "", "/media?id=1")
  vi.spyOn(window.history, "pushState").mockImplementation((state, title, href) => {
    nativeHistory.entries.splice(nativeHistory.index + 1)
    nativeHistory.entries.push({ href: String(href), state })
    nativeHistory.index += 1
    nativeHistory.push!(state, title, href)
  })
  vi.spyOn(window.history, "replaceState").mockImplementation((state, title, href) => {
    nativeHistory.entries[nativeHistory.index] = { href: String(href), state }
    nativeHistory.replace!(state, title, href)
  })
  nativeHistory.onPop = (event: PopStateEvent) => {
    const state = event.state as typeof nativeHistory.entries[number]["state"]
    nativeHistory.key = state.key
    // Next can restore the key and return before beforePopState on the SSR source.
    if (nativeHistory.isSsr && state.as === router.asPath) return
    if (pop.handler(state)) void router.replace(state.as, state.options)
  }
  window.addEventListener("popstate", nativeHistory.onPop)
})
afterEach(() => {
  cleanup()
  if (nativeHistory.onPop) window.removeEventListener("popstate", nativeHistory.onPop)
  vi.restoreAllMocks()
})

const browserBack = () => {
  nativeHistory.index = 0
  const entry = nativeHistory.entries[0]
  // Native traversal changes the current URL/state before Next may handle it.
  nativeHistory.replace!(entry.state, "", entry.href)
  window.dispatchEvent(new PopStateEvent("popstate", { state: entry.state }))
}

describe("Media dirty draft on the actual Next prompt boundary", () => {
  it("restores the source URL after the user declines Back", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
    render(<RouteLeavePrompt when message="Discard unsaved content?" />)
    await act(async () => browserBack())
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce())
    expect(router.asPath).toBe("/media?id=1")
    await waitFor(() => expect(window.location.pathname + window.location.search).toBe("/media?id=1"))
  })

  it("retains the original Back destination and source state after declining", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true)
    const source = window.history.state
    const destination = nativeHistory.entries[0].state
    render(<RouteLeavePrompt when message="Discard unsaved content?" />)
    await act(async () => browserBack())
    await waitFor(() => expect(window.location.pathname + window.location.search).toBe("/media?id=1"))
    expect(nativeHistory.entries[0]).toEqual({ href: "/notes", state: destination })
    expect(window.history.pushState).toHaveBeenCalledWith(source, "", "/media?id=1")
    expect(window.history.state).toEqual(source)
    expect(nativeHistory.key).toBe("source-key")
    await act(async () => browserBack())
    await waitFor(() => expect(router.asPath).toBe("/notes"))
    expect(window.history.state).toEqual(destination)
    expect(nativeHistory.key).toBe("notes-key")
    expect(confirm).toHaveBeenCalledTimes(2)
  })

  it("retires restoration bypass even when Next returns before beforePopState", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
    nativeHistory.isSsr = true
    render(<RouteLeavePrompt when message="Discard unsaved content?" />)
    await act(async () => browserBack())
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce())
    expect(nativeHistory.key).toBe("source-key")
    nativeHistory.isSsr = false
    await act(async () => {
      await router.push("/media?id=1").catch(() => undefined)
    })
    await waitFor(() => expect(confirm).toHaveBeenCalledTimes(2))
    expect(nativeHistory.entries[0].href).toBe("/notes")
  })

  it("continues Back to the requested source after explicit discard", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true)
    render(<RouteLeavePrompt when message="Discard unsaved content?" />)
    await act(async () => browserBack())
    await waitFor(() => expect(router.asPath).toBe("/notes"))
    expect(window.location.pathname + window.location.search).toBe("/notes")
    expect(confirm).toHaveBeenCalledOnce()
  })

  it("does not fabricate source state when no trusted entry was captured", async () => {
    nativeHistory.replace!({ unexpected: true }, "", "/media?id=1")
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
    render(<RouteLeavePrompt when message="Discard unsaved content?" />)
    await act(async () => browserBack())
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce())
    expect(nativeHistory.entries).toHaveLength(2)
    expect(window.history.state).toEqual(nativeHistory.entries[0].state)
    expect(router.asPath).toBe("/media?id=1")
  })

  it("allows clean Back without asking", async () => {
    const confirm = vi.spyOn(window, "confirm")
    render(<RouteLeavePrompt when={false} message="Discard unsaved content?" />)
    await act(async () => browserBack())
    expect(router.asPath).toBe("/notes")
    expect(confirm).not.toHaveBeenCalled()
  })
})
