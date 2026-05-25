import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useAnnotationSyncOnClose } from "@/hooks/document-workspace/useAnnotationSync"
import { useConnectionStore } from "@/store/connection"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"

vi.mock("@/services/tldw", () => ({
  tldwClient: {
    syncAnnotations: vi.fn(),
  },
}))

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

describe("useAnnotationSyncOnClose", () => {
  const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  const originalNavigator = globalThis.navigator
  let initialConnectionStore: ReturnType<typeof useConnectionStore.getState>
  let initialWorkspaceStore: ReturnType<typeof useDocumentWorkspaceStore.getState>

  beforeEach(() => {
    initialConnectionStore = useConnectionStore.getState()
    initialWorkspaceStore = useDocumentWorkspaceStore.getState()
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"

    useConnectionStore.setState({
      state: {
        ...initialConnectionStore.state,
        serverUrl: "http://127.0.0.1:8000",
      },
    })
    useDocumentWorkspaceStore.setState({
      ...initialWorkspaceStore,
      annotationSyncStatus: "pending",
      pendingAnnotations: [
        {
          id: "ann-1",
          documentId: 1,
          location: "1",
          text: "Important",
          color: "yellow",
          annotationType: "highlight",
          createdAt: new Date("2026-05-25T00:00:00Z"),
          updatedAt: new Date("2026-05-25T00:00:00Z"),
        },
      ],
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
    useConnectionStore.setState(initialConnectionStore, true)
    useDocumentWorkspaceStore.setState(initialWorkspaceStore, true)
    vi.restoreAllMocks()
  })

  it("uses the webui origin for quickstart annotation beacons", () => {
    const sendBeacon = vi.fn().mockReturnValue(true)
    Object.defineProperty(globalThis, "navigator", {
      value: {
        ...originalNavigator,
        sendBeacon,
      },
      configurable: true,
    })

    const forceSync = vi.fn()
    renderHook(() => useAnnotationSyncOnClose(42, forceSync), {
      wrapper: buildWrapper(),
    })

    act(() => {
      window.dispatchEvent(new Event("beforeunload"))
    })

    expect(sendBeacon).toHaveBeenCalledWith(
      "/api/v1/media/42/annotations/sync",
      expect.any(Blob)
    )
    expect(forceSync).not.toHaveBeenCalled()
  })
})
