import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useDocumentOperation } from "../useDocumentOperation"
import { useAnnotations } from "../useAnnotations"
import { useDocumentWorkspaceStore as store } from "@/store/document-workspace"
import { useConnectionStore } from "@/store/connection"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
const api = vi.hoisted(() => ({ listAnnotations: vi.fn() }))
vi.mock(
  "@plasmohq/storage",
  async () =>
    import("../../../../../../tldw-frontend/extension/shims/plasmo-storage")
)
vi.mock("@/services/tldw", () => ({
  tldwClient: api,
  tldwModels: {
    subscribeInvalidation: vi.fn(),
    clearCache: vi.fn(async () => {})
  }
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => {},
    ensureConfigForRequest: async () =>
      JSON.parse(localStorage.getItem("tldwConfig") || "null")
  }
}))
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: async () => ({ id: 1 }) }
}))
vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))
vi.mock("@/utils/browser-runtime", () => ({ isExtensionRuntime: () => false }))
const original = {
  serverUrl: "https://document-a.test",
  authMode: "single-user",
  apiKey: "synthetic-document-key"
}
const change = (config: unknown) => {
  const oldValue = localStorage.getItem("tldwConfig"),
    newValue = JSON.stringify(config)
  localStorage.setItem("tldwConfig", newValue)
  window.dispatchEvent(
    new StorageEvent("storage", { key: "tldwConfig", oldValue, newValue })
  )
}
let client: QueryClient
let connection: ReturnType<typeof useConnectionStore.getState>
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={client}>{children}</QueryClientProvider>
)
beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  sessionStorage.clear()
  localStorage.setItem("tldwConfig", JSON.stringify(original))
  store.getState().reset()
  connection = useConnectionStore.getState()
  useConnectionStore.setState({
    state: { ...connection.state, isConnected: true, mode: "normal" }
  })
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
})
afterEach(() => {
  cleanup()
  client.clear()
  store.getState().reset()
  useConnectionStore.setState(connection, true)
})
describe("Document real verified lease", () => {
  it("survives a new ingestion draft and retires across server A to B to A", async () => {
    const hook = renderHook(() => useDocumentOperation(), { wrapper })
    await waitFor(() => expect(hook.result.current).not.toBeNull())
    const first = hook.result.current!
    act(() => useQuickIngestSessionStore.getState().createDraftSession())
    expect(first.isCurrent()).toBe(true)
    act(() => {
      change({ ...original, serverUrl: "https://document-b.test" })
      change(original)
    })
    await waitFor(() => expect(hook.result.current?.isCurrent()).toBe(true))
    expect({
      oldCurrent: first.isCurrent(),
      oldAborted: first.signal.aborted,
      newRevision:
        hook.result.current!.authorityRevision === first.authorityRevision
    }).toEqual({ oldCurrent: false, oldAborted: true, newRevision: false })
  })
  it.each(["server", "credential", "logout"])(
    "aborts held document reads on real %s retirement",
    async (boundary) => {
      let finish!: (value: unknown) => void
      api.listAnnotations.mockReturnValue(
        new Promise((resolve) => {
          finish = resolve
        })
      )
      store.setState({ activeDocumentId: 1 })
      renderHook(() => useAnnotations(1), { wrapper })
      await waitFor(() => expect(api.listAnnotations).toHaveBeenCalledTimes(1))
      const options = api.listAnnotations.mock.calls[0][1]
      act(() => {
        if (boundary === "server")
          change({ ...original, serverUrl: "https://document-b.test" })
        else if (boundary === "credential")
          change({ ...original, apiKey: "synthetic-replacement-key" })
        else window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
      })
      expect(options.signal.aborted).toBe(true)
      await act(async () => {
        finish({
          media_id: 1,
          annotations: [
            {
              id: "old",
              media_id: 1,
              location: "1",
              text: "Old private",
              color: "yellow",
              annotation_type: "highlight",
              created_at: new Date(0).toISOString(),
              updated_at: new Date(0).toISOString()
            }
          ],
          total_count: 1
        })
        await Promise.resolve()
      })
      expect(store.getState().annotations).toEqual([])
    }
  )
})
