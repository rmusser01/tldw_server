import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useAnnotationSyncOnClose } from "@/hooks/document-workspace/useAnnotationSync"
import { tldwClient } from "@/services/tldw"
import { useConnectionStore } from "@/store/connection"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"

vi.mock("@/services/tldw", () => ({
  tldwClient: {
    syncAnnotations: vi.fn(),
    getConfig: vi.fn(),
  },
  tldwModels: {
    subscribeInvalidation: vi.fn(),
  },
}))

const SERVER_URL = "http://127.0.0.1:8000"
const SYNC_URL = "/api/v1/media/42/annotations/sync"

const buildWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

const pendingAnnotation = (text = "Important") => ({
  id: "ann-1",
  documentId: 1,
  location: "1",
  text,
  color: "yellow" as const,
  annotationType: "highlight" as const,
  createdAt: new Date("2026-05-25T00:00:00Z"),
  updatedAt: new Date("2026-05-25T00:00:00Z"),
})

const storedValues = () =>
  Array.from({ length: localStorage.length }, (_, i) =>
    localStorage.getItem(localStorage.key(i) as string)
  ).join("\n")

const renderOnClose = (forceSync = vi.fn()) => {
  const rendered = renderHook(() => useAnnotationSyncOnClose(42, forceSync), {
    wrapper: buildWrapper(),
  })
  return { ...rendered, forceSync }
}

// The unload headers are prepared asynchronously after render; let that settle.
const settle = () =>
  act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })

const closePage = () =>
  act(() => {
    window.dispatchEvent(new Event("beforeunload"))
  })

describe("useAnnotationSyncOnClose", () => {
  const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  const originalNavigator = globalThis.navigator
  let initialConnectionStore: ReturnType<typeof useConnectionStore.getState>
  let initialWorkspaceStore: ReturnType<typeof useDocumentWorkspaceStore.getState>
  let fetchMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    initialConnectionStore = useConnectionStore.getState()
    initialWorkspaceStore = useDocumentWorkspaceStore.getState()
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    localStorage.clear()
    fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 200 }))
    vi.stubGlobal("fetch", fetchMock)
    vi.mocked(tldwClient.getConfig).mockResolvedValue({
      serverUrl: SERVER_URL,
      authMode: "single-user",
      apiKey: "unload-test-key",
    })
    vi.mocked(tldwClient.syncAnnotations).mockResolvedValue({
      media_id: 42,
      synced_count: 1,
      annotations: [],
    })

    useConnectionStore.setState({
      state: {
        ...initialConnectionStore.state,
        serverUrl: SERVER_URL,
      },
    })
    useDocumentWorkspaceStore.setState({
      ...initialWorkspaceStore,
      annotationSyncStatus: "pending",
      pendingAnnotations: [pendingAnnotation()],
    })
  })

  afterEach(() => {
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
    Object.defineProperty(globalThis, "navigator", {
      value: originalNavigator,
      configurable: true,
    })
    document.cookie = "csrf_token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/"
    localStorage.clear()
    useConnectionStore.setState(initialConnectionStore, true)
    useDocumentWorkspaceStore.setState(initialWorkspaceStore, true)
    vi.unstubAllGlobals()
    vi.clearAllMocks()
    vi.restoreAllMocks()
  })

  it("sends quickstart unload syncs to the webui origin with fetch keepalive, not sendBeacon", async () => {
    const sendBeacon = vi.fn().mockReturnValue(true)
    Object.defineProperty(globalThis, "navigator", {
      value: { ...originalNavigator, sendBeacon },
      configurable: true,
    })

    const { forceSync } = renderOnClose()
    await settle()
    closePage()

    expect(sendBeacon).not.toHaveBeenCalled()
    expect(fetchMock).toHaveBeenCalledWith(
      SYNC_URL,
      expect.objectContaining({ method: "POST", keepalive: true })
    )
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      annotations: [
        {
          location: "1",
          text: "Important",
          color: "yellow",
          annotation_type: "highlight",
        },
      ],
      client_ids: ["ann-1"],
    })
    expect(forceSync).not.toHaveBeenCalled()
  })

  it.each([
    [
      "single-user API key",
      { serverUrl: SERVER_URL, authMode: "single-user", apiKey: "unload-test-key" },
      "X-API-KEY",
      "unload-test-key",
    ],
    [
      "multi-user bearer",
      { serverUrl: SERVER_URL, authMode: "multi-user", accessToken: "unload-access-token" },
      "Authorization",
      "Bearer unload-access-token",
    ],
    [
      "cookie session",
      { serverUrl: SERVER_URL, authMode: "single-user", authSource: "cookie-session" },
      "X-CSRF-Token",
      "csrf-from-cookie",
    ],
  ])("authenticates the unload request in %s mode", async (_mode, config, header, value) => {
    document.cookie = "csrf_token=csrf-from-cookie; path=/"
    vi.mocked(tldwClient.getConfig).mockResolvedValue(config as any)

    renderOnClose()
    await settle()
    closePage()

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][1].headers).toMatchObject({ [header]: value })
  })

  it("persists pending annotations locally before sending", async () => {
    let storedAtSend = ""
    fetchMock.mockImplementation(() => {
      storedAtSend = storedValues()
      return Promise.resolve(new Response(null, { status: 200 }))
    })

    renderOnClose()
    await settle()
    closePage()

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(storedAtSend).toContain('"ann-1"')
  })

  it("defers payloads over 60KB: persists them and makes no request", async () => {
    useDocumentWorkspaceStore.setState({
      pendingAnnotations: [pendingAnnotation("x".repeat(61_000))],
    })

    const { forceSync } = renderOnClose()
    await settle()
    closePage()

    expect(fetchMock).not.toHaveBeenCalled()
    expect(storedValues()).toContain('"ann-1"')
    expect(forceSync).toHaveBeenCalled()
  })

  it("sends via fetch keepalive when sendBeacon is unavailable", async () => {
    Object.defineProperty(globalThis, "navigator", {
      value: { ...originalNavigator, sendBeacon: undefined },
      configurable: true,
    })

    renderOnClose()
    await settle()
    closePage()

    expect(fetchMock).toHaveBeenCalledWith(
      SYNC_URL,
      expect.objectContaining({ method: "POST", keepalive: true })
    )
  })

  it("replays persisted annotations on the next load and clears them once synced", async () => {
    const firstPage = renderOnClose()
    await settle()
    closePage()
    firstPage.unmount()

    // Next load: the in-memory store starts empty.
    useDocumentWorkspaceStore.setState({
      annotationSyncStatus: "synced",
      pendingAnnotations: [],
    })
    renderOnClose()
    await settle()

    expect(tldwClient.syncAnnotations).toHaveBeenCalledWith(
      42,
      [{ location: "1", text: "Important", color: "yellow", annotation_type: "highlight" }],
      ["ann-1"]
    )
    expect(storedValues()).not.toContain('"ann-1"')
  })
})
