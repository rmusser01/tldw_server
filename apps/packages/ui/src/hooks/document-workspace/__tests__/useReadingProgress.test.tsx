import React from "react"
import { act, cleanup, renderHook } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  useUpdateReadingProgress,
  useReadingProgressAutoSave,
  useReadingProgressSaveOnClose,
} from "@/hooks/document-workspace/useReadingProgress"
import { useConnectionStore } from "@/store/connection"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"

const mocks = vi.hoisted(() => ({
  updateReadingProgress: vi.fn(),
}))

const owner = {authorityKey:"progress-owner",authorityRevision:1,requestScope:{config:{serverUrl:"https://a.test",authMode:"single-user"},userId:null},signal:new AbortController().signal,isCurrent:()=>true,assertCurrent:()=>{}}
vi.mock("@/hooks/document-workspace/useDocumentOperation",()=>({useDocumentOperation:()=>owner}))
vi.mock("@/services/tldw", () => ({
  tldwClient: {
    updateReadingProgress: mocks.updateReadingProgress,
  },
  tldwModels: {
    subscribeInvalidation: vi.fn(),
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

describe("useReadingProgressAutoSave", () => {
  let initialConnectionStore: ReturnType<typeof useConnectionStore.getState>
  let initialWorkspaceStore: ReturnType<typeof useDocumentWorkspaceStore.getState>

  beforeEach(() => {
    vi.useFakeTimers()
    initialConnectionStore = useConnectionStore.getState()
    initialWorkspaceStore = useDocumentWorkspaceStore.getState()

    useConnectionStore.setState({
      state: {
        ...initialConnectionStore.state,
        isConnected: true,
        mode: "normal",
      },
    })

    useDocumentWorkspaceStore.setState({
      currentPage: 1,
      totalPages: 10,
      zoomLevel: 100,
      viewMode: "single",
      currentCfi: null,
      currentPercentage: 0,
    })

    mocks.updateReadingProgress.mockReset()
  })

  afterEach(() => {
    cleanup()
    useConnectionStore.setState(initialConnectionStore, true)
    useDocumentWorkspaceStore.setState(initialWorkspaceStore, true)
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("keeps the public non-promise progress mutation bound to its owner",async()=>{
    mocks.updateReadingProgress.mockResolvedValue({media_id:152,current_page:2,total_pages:10})
    const hook=renderHook(()=>useUpdateReadingProgress(),{wrapper:buildWrapper()})
    await act(async()=>{hook.result.current.mutate({mediaId:152,progress:{current_page:2,total_pages:10}});await vi.advanceTimersByTimeAsync(0)})
    expect(mocks.updateReadingProgress).toHaveBeenCalledWith(152,{current_page:2,total_pages:10},owner)
    expect(hook.result.current.isError).toBe(false)
    hook.unmount()
  })

  it("does not overwrite defaults before the caller's cold read settles",async()=>{
    mocks.updateReadingProgress.mockResolvedValue(undefined)
    const hook=renderHook(()=>useReadingProgressAutoSave(152,50,false),{wrapper:buildWrapper()})
    await act(async()=>{await vi.advanceTimersByTimeAsync(100)})
    expect(mocks.updateReadingProgress).not.toHaveBeenCalled()
    hook.unmount()
  })

  it("stops autosave retries for a media id after a 404 response", async () => {
    mocks.updateReadingProgress.mockRejectedValue({ status: 404 })

    renderHook(() => useReadingProgressAutoSave(152, 50), {
      wrapper: buildWrapper(),
    })

    await act(async () => {
      vi.advanceTimersByTime(50)
      await Promise.resolve()
    })

    expect(mocks.updateReadingProgress).toHaveBeenCalledTimes(1)

    act(() => {
      useDocumentWorkspaceStore.setState({ currentPage: 2 })
    })

    await act(async () => {
      vi.advanceTimersByTime(50)
      await Promise.resolve()
    })

    expect(mocks.updateReadingProgress).toHaveBeenCalledTimes(1)
  })

  it("serializes reading writes so an older save cannot finish last", async () => {
    let finish!:()=>void
    mocks.updateReadingProgress.mockReturnValueOnce(new Promise<void>(resolve=>{finish=resolve})).mockResolvedValue(undefined)
    const hook=renderHook(()=>useReadingProgressAutoSave(152,50),{wrapper:buildWrapper()})
    await act(async()=>{await vi.advanceTimersByTimeAsync(50)})
    act(()=>useDocumentWorkspaceStore.setState({currentPage:2}))
    await act(async()=>{await vi.advanceTimersByTimeAsync(50)})
    expect(mocks.updateReadingProgress).toHaveBeenCalledTimes(1)
    await act(async()=>{finish();await vi.advanceTimersByTimeAsync(0)})
    expect(mocks.updateReadingProgress.mock.calls.map(args=>args[1].current_page)).toEqual([1,2])
    hook.unmount()
  })

  it("keeps attempting autosave for non-404 failures", async () => {
    mocks.updateReadingProgress.mockRejectedValue({ status: 500 })

    renderHook(() => useReadingProgressAutoSave(152, 50), {
      wrapper: buildWrapper(),
    })

    await act(async () => {
      vi.advanceTimersByTime(50)
      await Promise.resolve()
    })

    expect(mocks.updateReadingProgress).toHaveBeenCalledTimes(1)

    act(() => {
      useDocumentWorkspaceStore.setState({ currentPage: 3 })
    })

    await act(async () => {
      vi.advanceTimersByTime(50)
      await Promise.resolve()
    })

    expect(mocks.updateReadingProgress).toHaveBeenCalledTimes(2)
  })
})

describe("useReadingProgressSaveOnClose", () => {
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
      currentPage: 3,
      totalPages: 12,
      zoomLevel: 125,
      viewMode: "single",
      currentCfi: null,
      currentPercentage: 25,
    })
  })

  afterEach(() => {
    cleanup()
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

  it("attempts the captured authenticated save without an unauthenticated beacon", async () => {
    const sendBeacon = vi.fn().mockReturnValue(true)
    Object.defineProperty(globalThis, "navigator", {
      value: {
        ...originalNavigator,
        sendBeacon,
      },
      configurable: true,
    })

    const forceSave = vi.fn()
    renderHook(() => useReadingProgressSaveOnClose(42, forceSave))

    await act(async () => {
      window.dispatchEvent(new Event("beforeunload"))
      await Promise.resolve()
    })
    expect(sendBeacon).not.toHaveBeenCalled()
    expect(forceSave).toHaveBeenCalledTimes(1)
  })
})
