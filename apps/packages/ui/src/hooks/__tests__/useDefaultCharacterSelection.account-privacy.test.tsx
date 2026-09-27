// @vitest-environment jsdom
import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Storage } from "@plasmohq/storage"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
const mocks = vi.hoisted(() => ({ user: "alice" as string | null, read: vi.fn(), write: vi.fn() }))
vi.mock("@plasmohq/storage", () => import("../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@plasmohq/storage/hook", () => import("../../../../../tldw-frontend/extension/shims/plasmo-storage-hook"))
vi.mock("@/services/service-prompts", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/service-prompts")>(),
  resolveServicePromptScope: async () => {
    if (!mocks.user) throw new Error("Signed out")
    return { scopeKey: mocks.user, userId: mocks.user, clientPrincipalVerified: true,
      config: { serverUrl: "http://localhost:8000", authMode: "multi-user", authSource: "manual", orgId: 2 } }
  }
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  getDefaultCharacterPreference: (...args: unknown[]) => mocks.read(...args),
  setDefaultCharacterPreference: (...args: unknown[]) => mocks.write(...args)
} }))
import { useDefaultCharacterSelection } from "../useDefaultCharacterSelection"
const alice = { id: "4", name: "Alice private default", system_prompt: "Alice private instructions" }
const bob = { id: "5", name: "Bob default" }
let queryClient: QueryClient
const wrapper = ({ children }: React.PropsWithChildren) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
const mount = async () => {
  const view = renderHook(() => useDefaultCharacterSelection(), { wrapper })
  await waitFor(() => expect(view.result.current[2].isLoading).toBe(false))
  return view
}
const switchTo = async (user: string) => {
  act(() => { mocks.user = null; window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")) })
  await act(async () => { mocks.user = user; window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } })) })
}
beforeEach(() => {
  localStorage.clear(); mocks.user = "alice"
  mocks.read.mockReset().mockResolvedValue(null); mocks.write.mockReset().mockResolvedValue({ applied: [], skipped: [] })
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
})
afterEach(() => { queryClient.clear(); vi.restoreAllMocks() })
describe("default Character account ownership", () => {
  it("rejects the unowned global payload", async () => {
    localStorage.setItem("defaultCharacterSelection", JSON.stringify(alice))
    const view = await mount()
    expect(view.result.current[0]).toBeNull()
    expect(localStorage.getItem("defaultCharacterSelection")).toBeNull()
  })
  it("keeps defaults private through reciprocal login and reload", async () => {
    let view = await mount()
    await act(async () => { await view.result.current[1](alice) })
    await switchTo("bob")
    await waitFor(() => expect(view.result.current[2].isLoading).toBe(false))
    expect(view.result.current[0]).toBeNull()
    await act(async () => { await view.result.current[1](bob) })
    await switchTo("alice")
    await waitFor(() => expect(view.result.current[0]).toMatchObject(alice))
    view.unmount(); view = await mount()
    expect(view.result.current[0]).toMatchObject(alice)
  })
  it("does not adopt a cached or late Alice server preference for Bob", async () => {
    let release!: (id: string) => void
    mocks.read.mockImplementation(({ requestScope }) => requestScope.userId === "alice" ? new Promise(resolve => { release = resolve }) : Promise.resolve("5"))
    const view = await mount()
    await waitFor(() => expect(release).toBeDefined())
    await switchTo("bob")
    await waitFor(() => expect(view.result.current[2].preference?.defaultCharacterId).toBe("5"))
    await act(async () => { release("4") })
    expect(view.result.current[2].preference?.defaultCharacterId).toBe("5")
    expect(view.result.current[0]).toBeNull()
  })
  it("preserves a local-only default when the server still reports no default", async () => {
    const view = await mount()
    await waitFor(() => expect(view.result.current[2].preference?.defaultCharacterId).toBeNull())
    mocks.write.mockRejectedValueOnce(new Error("Temporary server failure"))
    await expect(view.result.current[2].writePreference("4")).rejects.toThrow("Temporary server failure")
    await act(async () => { await view.result.current[1](alice, { localOnly: true }) })
    expect(view.result.current[0]).toMatchObject(alice)
    expect(view.result.current[2].preference).toBeUndefined()
    view.unmount()
    const reloaded = await mount()
    expect(reloaded.result.current[0]).toMatchObject(alice)
    expect(reloaded.result.current[2].preference).toBeUndefined()
  })
  it("rejects an old default writer after the account switches", async () => {
    const view = await mount()
    const oldWriter = view.result.current[1]
    await switchTo("bob")
    await waitFor(() => expect(view.result.current[2].isLoading).toBe(false))
    await expect(oldWriter(alice)).rejects.toThrow(/account/)
    expect(view.result.current[0]).toBeNull()
  })
  it.each(["4", null])("publishes a successful default preference write before local synchronization: %s", async next => {
    mocks.read.mockResolvedValue("5")
    const view = await mount()
    await waitFor(() => expect(view.result.current[2].preference?.defaultCharacterId).toBe("5"))
    await act(async () => {
      await view.result.current[2].writePreference(next)
      await view.result.current[1](next ? alice : null)
    })
    expect(view.result.current[2].preference?.defaultCharacterId).toBe(next)
    expect(view.result.current[0]).toEqual(next ? alice : null)
  })
  it("does not let an older server read undo a successful default write", async () => {
    let release!: (id: string) => void
    mocks.read.mockImplementation(() => new Promise(resolve => { release = resolve }))
    const view = await mount()
    await waitFor(() => expect(release).toBeDefined())
    await act(async () => {
      await view.result.current[2].writePreference("4")
      await view.result.current[1](alice)
    })
    await act(async () => { release("5") })
    await waitFor(() => expect(queryClient.isFetching()).toBe(0))
    expect(view.result.current[2].preference?.defaultCharacterId).toBe("4")
    expect(view.result.current[0]).toMatchObject(alice)
  })
  it("withholds a cached server default until the local-only record has hydrated", async () => {
    mocks.read.mockResolvedValue("5")
    const original = await mount()
    await waitFor(() => expect(original.result.current[2].preference?.defaultCharacterId).toBe("5"))
    await act(async () => { await original.result.current[1](alice, { localOnly: true }) })
    original.unmount()
    const originalGet = Storage.prototype.get
    let release!: () => void
    let started = false
    const barrier = new Promise<void>(resolve => { release = resolve })
    vi.spyOn(Storage.prototype, "get").mockImplementation(async function (key) {
      const result = await originalGet.call(this, key)
      if (key.startsWith("defaultCharacterSelection:owner:") && !key.endsWith(":unresolved")) { started = true; await barrier }
      return result
    })
    const view = renderHook(() => useDefaultCharacterSelection(), { wrapper })
    await waitFor(() => expect(started).toBe(true))
    expect(view.result.current[2].preference).toBeUndefined()
    await act(async () => { release() })
    await waitFor(() => expect(view.result.current[0]).toMatchObject(alice))
    expect(view.result.current[2].preference).toBeUndefined()
  })
  it("pins a server write to its captured owner and rejects its late completion", async () => {
    let release!: () => void
    mocks.write.mockImplementation((_id, { requestScope }) => {
      expect(requestScope.userId).toBe("alice")
      return new Promise<void>(resolve => { release = resolve })
    })
    const view = await mount()
    const pending = view.result.current[2].writePreference("4")
    const rejected = expect(pending).rejects.toThrow(/account/)
    await waitFor(() => expect(release).toBeDefined())
    await switchTo("bob")
    await act(async () => { release(); await rejected })
    expect(view.result.current[0]).toBeNull()
  })
})
