import React, { StrictMode } from "react"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { initialState, useWorkspaceStore } from "@/store/workspace"
import {
  createOwnedWorkspaceDraftStore,
  ownedWorkspaceDraftKey
} from "@/store/owned-workspace-state"
import type { OwnedWorkspaceBundle } from "@/store/workspace-api"
import { OwnedWorkspaceLoadError } from "@/store/workspace-api"
import { useOwnedWorkspaceOpening } from "../useOwnedWorkspaceOpening"

const mocks = vi.hoisted(() => ({ context: vi.fn() }))
vi.mock("@/services/owned-workspace-opening", async (original) => ({
  ...(await original<typeof import("@/services/owned-workspace-opening")>()),
  createOwnedWorkspaceReadContext: mocks.context
}))

const scope = {
  serverBase: "https://research.example",
  principalId: "3",
  organizationId: "7"
}
const bundle = (id: string): OwnedWorkspaceBundle => ({
  workspace: {
    id,
    name: `Workspace ${id}`,
    archived: false,
    deleted: false,
    workspace_profile: "research",
    study_materials_policy: "workspace",
    version: 2,
    banner_title: null,
    banner_subtitle: null,
    banner_color: null,
    audio_provider: null,
    audio_model: null,
    audio_voice: null,
    audio_speed: null,
    created_at: "2026-09-13T12:00:00Z",
    last_modified: "2026-09-13T12:00:00Z"
  },
  sources: [],
  artifacts: [],
  notes: []
})
function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}
const state = () => useWorkspaceStore.getState()

beforeEach(() => {
  localStorage.clear()
  useWorkspaceStore.setState({ ...initialState, storeHydrated: true })
  mocks.context.mockReset().mockImplementation(async (id: string) => ({
    scope,
    load: async () => bundle(id)
  }))
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe("owned workspace opening lifecycle", () => {
  it("does not overwrite a populated legacy index while hydration is delayed", async () => {
    const { WORKSPACE_STORAGE_KEY } = await import("@/store/workspace-events")
    state().initializeWorkspace("Precious legacy workspace")
    await new Promise((resolve) => setTimeout(resolve, 0))
    const storage = useWorkspaceStore.persist.getOptions().storage!
    const before = localStorage.getItem(WORKSPACE_STORAGE_KEY)
    const envelope = await storage.getItem(WORKSPACE_STORAGE_KEY)
    const disk = Object.fromEntries(
      Object.keys(localStorage).map((key) => [key, localStorage.getItem(key)])
    )
    useWorkspaceStore.persist.setOptions({
      storage: { ...storage, setItem: () => {} }
    })
    useWorkspaceStore.setState({ ...initialState, storeHydrated: false })
    const gate = deferred<typeof envelope>()
    useWorkspaceStore.persist.setOptions({
      storage: { ...storage, getItem: () => gate.promise }
    })
    const hydration = useWorkspaceStore.persist.rehydrate()
    const rendered = renderHook(() => useOwnedWorkspaceOpening("target"))
    try {
      await new Promise((resolve) => setTimeout(resolve, 0))
      expect(localStorage.getItem(WORKSPACE_STORAGE_KEY)).toBe(before)
      expect(
        Object.fromEntries(
          Object.keys(localStorage).map((key) => [
            key,
            localStorage.getItem(key)
          ])
        )
      ).toEqual(disk)
      rendered.unmount()
      gate.resolve(envelope)
      await hydration
      expect(state().workspaceName).toBe("Precious legacy workspace")
    } finally {
      rendered.unmount()
      gate.resolve(envelope)
      await hydration
      useWorkspaceStore.persist.setOptions({ storage })
    }
  })
  it("times out stalled hydration and retries storage before opening", async () => {
    vi.useFakeTimers()
    useWorkspaceStore.setState({ storeHydrated: false })
    const rehydrate = vi
      .spyOn(useWorkspaceStore.persist, "rehydrate")
      .mockImplementation(async () => {
        useWorkspaceStore.setState({ storeHydrated: true })
      })
    try {
      const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
      await act(async () => {
        await vi.advanceTimersByTimeAsync(30_000)
      })
      expect(result.current).toMatchObject({
        status: "error",
        reason: "timeout"
      })
      expect(mocks.context).not.toHaveBeenCalled()
      await act(async () => {
        if (result.current.status === "error") result.current.retry()
      })
      expect(rehydrate).toHaveBeenCalledTimes(1)
      expect(result.current.status).toBe("ready")
    } finally {
      rehydrate.mockRestore()
    }
  })

  it("does not restart the 30-second budget after slow hydration", async () => {
    vi.useFakeTimers()
    useWorkspaceStore.setState({ storeHydrated: false })
    mocks.context.mockImplementation(() => new Promise(() => {}))
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(25_000)
    })
    await act(async () => {
      useWorkspaceStore.setState({ storeHydrated: true })
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })
    expect(result.current).toMatchObject({ status: "error", reason: "timeout" })
    expect(mocks.context.mock.calls[0][1].aborted).toBe(true)
  })
  it.each([
    "tldwConfig",
    "tldwManualSessionApiKey",
    "tldwRefreshRotation",
    "tldwCookieSessionConfig",
    null
  ])("revalidates on cross-tab identity storage changes: %s", async (key) => {
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() => window.dispatchEvent(new StorageEvent("storage", { key })))
    await waitFor(() => expect(mocks.context).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(result.current.status).toBe("ready"))
  })

  it("ignores cross-tab draft writes instead of interrupting editing", async () => {
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() =>
      window.dispatchEvent(
        new StorageEvent("storage", { key: ownedWorkspaceDraftKey(scope, "a") })
      )
    )
    expect(result.current.status).toBe("ready")
    expect(mocks.context).toHaveBeenCalledTimes(1)
  })

  it("clears readiness on pagehide and verifies again on restoration", async () => {
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() => window.dispatchEvent(new Event("pagehide")))
    expect(result.current.status).toBe("loading")
    expect(state().workspaceId).toBe("")
    act(() => window.dispatchEvent(new Event("pageshow")))
    await waitFor(() => expect(result.current.status).toBe("ready"))
    expect(mocks.context).toHaveBeenCalledTimes(2)
  })

  it.each(["invalid-response", "unavailable"] as const)(
    "preserves the loader's %s error",
    async (reason) => {
      mocks.context.mockResolvedValue({
        scope,
        load: async () => {
          throw new OwnedWorkspaceLoadError(reason, "notes")
        }
      })
      const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
      await waitFor(() =>
        expect(result.current).toMatchObject({ status: "error", reason })
      )
      expect(state().workspaceId).not.toBe("a")
      expect(mocks.context.mock.calls[0][1].aborted).toBe(true)
    }
  )

  it("never remains ready when the store's activation is invalidated elsewhere", async () => {
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() => {
      state().invalidateOwnedWorkspace()
    })
    expect(result.current.status).toBe("loading")
  })
  it("waits for hydration before verifying or activating", async () => {
    useWorkspaceStore.setState({ storeHydrated: false })
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    expect(result.current.status).toBe("loading")
    expect(mocks.context).not.toHaveBeenCalled()
    await act(async () => {
      useWorkspaceStore.setState({ storeHydrated: true })
    })
    await waitFor(() => expect(result.current.status).toBe("ready"))
    expect(state().workspaceId).toBe("a")
    expect(state().activeWorkspaceOrigin).toEqual({
      kind: "server-owned",
      scope
    })
  })

  it("requires verified context even when a recovered draft exists", async () => {
    const attempt = state().beginOwnedWorkspace(scope, "a")
    state().activateOwnedWorkspace(attempt, bundle("a"))
    state().updateNoteContent("Private draft")
    state().invalidateOwnedWorkspace()
    mocks.context.mockRejectedValue({ status: 401 })
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await waitFor(() =>
      expect(result.current).toMatchObject({
        status: "error",
        reason: "denied"
      })
    )
    expect(state().workspaceId).toBe("")
    expect(state().currentNote.content).not.toContain("Private draft")
    expect(
      createOwnedWorkspaceDraftStore(() => localStorage).load(scope, "a")
    ).toMatchObject({
      status: "ready",
      durable: true,
      draft: { currentNote: { content: "Private draft" } }
    })
  })

  it("never reports the prior target ready during a route change", async () => {
    const renders: string[] = []
    const { result, rerender } = renderHook(
      ({ id }) => {
        const opening = useOwnedWorkspaceOpening(id)
        renders.push(`${id}:${opening.status}`)
        return opening
      },
      { initialProps: { id: "a" } }
    )
    await waitFor(() => expect(result.current.status).toBe("ready"))
    mocks.context.mockImplementation(() => new Promise(() => {}))
    rerender({ id: "b" })
    expect(renders).not.toContain("b:ready")
    expect(result.current.status).toBe("loading")
  })

  it("discards a late bundle from an old route even when transport ignores abort", async () => {
    const old = deferred<OwnedWorkspaceBundle>()
    mocks.context.mockImplementation(async (id: string) => ({
      scope,
      load: () => (id === "a" ? old.promise : Promise.resolve(bundle(id)))
    }))
    const { result, rerender } = renderHook(
      ({ id }) => useOwnedWorkspaceOpening(id),
      { initialProps: { id: "a" } }
    )
    await waitFor(() =>
      expect(state().ownedWorkspaceAttempt?.workspaceId).toBe("a")
    )
    rerender({ id: "b" })
    await waitFor(() => expect(result.current.status).toBe("ready"))
    await act(async () => old.resolve(bundle("a")))
    expect(state().workspaceId).toBe("b")
  })

  it.each([
    [401, "denied"],
    [403, "denied"],
    [404, "unavailable"],
    [410, "unavailable"],
    [412, "connection"],
    [0, "connection"],
    [503, "connection"]
  ])(
    "classifies HTTP %i and permits an explicit retry",
    async (status, reason) => {
      mocks.context.mockRejectedValueOnce({ status })
      const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
      await waitFor(() =>
        expect(result.current).toMatchObject({ status: "error", reason })
      )
      act(() => {
        if (result.current.status === "error") result.current.retry()
      })
      await waitFor(() => expect(result.current.status).toBe("ready"))
    }
  )

  it("classifies an account-bound 412 as denial and re-verifies on retry", async () => {
    mocks.context.mockRejectedValueOnce({
      status: 412,
      detail: { code: "request_config_scope_changed" }
    })
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await waitFor(() =>
      expect(result.current).toMatchObject({
        status: "error",
        reason: "denied"
      })
    )
    expect(state().activeWorkspaceOrigin.kind).toBe("legacy-local")
    act(() => {
      if (result.current.status === "error") result.current.retry()
    })
    await waitFor(() => expect(result.current.status).toBe("ready"))
    expect(mocks.context).toHaveBeenCalledTimes(2)
  })

  it("bounds the whole opening, including a stalled config/auth lookup, to 30 seconds", async () => {
    vi.useFakeTimers()
    const stalled = deferred<{
      scope: typeof scope
      load: () => Promise<OwnedWorkspaceBundle>
    }>()
    mocks.context.mockReturnValueOnce(stalled.promise)
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000)
    })
    expect(result.current).toMatchObject({ status: "error", reason: "timeout" })
    expect(mocks.context.mock.calls[0][1].aborted).toBe(true)
    await act(async () =>
      stalled.resolve({ scope, load: async () => bundle("a") })
    )
    expect(state().workspaceId).not.toBe("a")
    await act(async () => {
      if (result.current.status === "error") result.current.retry()
    })
    expect(result.current.status).toBe("ready")
  })

  it("invalidates visible content synchronously on connection changes and isolates the new account draft", async () => {
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() => state().updateNoteContent("Account A draft"))
    const verification = deferred<{
      scope: typeof scope
      load: () => Promise<OwnedWorkspaceBundle>
    }>()
    mocks.context.mockReturnValueOnce(verification.promise)
    act(() => {
      window.dispatchEvent(new Event("tldw:config-updated"))
      expect(state().workspaceId).toBe("")
    })
    expect(result.current.status).toBe("loading")
    await act(async () =>
      verification.resolve({
        scope: { ...scope, principalId: "4", organizationId: "8" },
        load: async () => bundle("a")
      })
    )
    expect(result.current.status).toBe("ready")
    expect(state().currentNote.content).not.toContain("Account A draft")
  })

  it("re-verifies token refresh and preserves drafts for the same verified scope", async () => {
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await waitFor(() => expect(result.current.status).toBe("ready"))
    act(() => state().updateNoteContent("Unsent draft"))
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", {
          detail: { kind: "refresh" }
        })
      )
    )
    await waitFor(() => expect(result.current.status).toBe("ready"))
    expect(mocks.context).toHaveBeenCalledTimes(2)
    expect(state().currentNote.content).toBe("Unsent draft")
  })

  it("does not restart network reads on logout or resurrect a late response", async () => {
    const pending = deferred<OwnedWorkspaceBundle>()
    mocks.context.mockResolvedValue({ scope, load: () => pending.promise })
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await waitFor(() => expect(state().ownedWorkspaceAttempt).not.toBeNull())
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", {
          detail: { kind: "logout" }
        })
      )
    )
    await act(async () => pending.resolve(bundle("a")))
    expect(result.current).toMatchObject({ status: "error", reason: "denied" })
    expect(state().workspaceId).not.toBe("a")
    expect(mocks.context).toHaveBeenCalledTimes(1)
  })

  it("aborts on unmount and cannot activate afterward", async () => {
    const pending = deferred<OwnedWorkspaceBundle>()
    mocks.context.mockResolvedValue({ scope, load: () => pending.promise })
    const { unmount } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await waitFor(() => expect(state().ownedWorkspaceAttempt).not.toBeNull())
    unmount()
    await act(async () => pending.resolve(bundle("a")))
    expect(mocks.context.mock.calls[0][1].aborted).toBe(true)
    expect(state().ownedWorkspaceAttempt).toBeNull()
    expect(state().workspaceId).not.toBe("a")
  })

  it("activates only the current StrictMode run", async () => {
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"), {
      wrapper: ({ children }) => <StrictMode>{children}</StrictMode>
    })
    await waitFor(() => expect(result.current.status).toBe("ready"))
    expect(mocks.context).toHaveBeenCalledTimes(1)
    expect(mocks.context.mock.calls[0][1].aborted).toBe(false)
    expect(state().ownedWorkspaceBundle?.workspace.id).toBe("a")
  })

  it("reports draft conflict without replacing quarantined server edits", async () => {
    const attempt = state().beginOwnedWorkspace(scope, "a")
    state().activateOwnedWorkspace(attempt, bundle("a"))
    state().setWorkspaceName("Unsaved rename")
    const { result } = renderHook(() => useOwnedWorkspaceOpening("a"))
    await waitFor(() =>
      expect(result.current).toMatchObject({
        status: "error",
        reason: "draft-conflict"
      })
    )
    expect(
      createOwnedWorkspaceDraftStore(() => localStorage).load(scope, "a")
    ).toMatchObject({
      status: "ready",
      durable: true,
      draft: { pendingChanges: { workspaceName: "Unsaved rename" } }
    })
  })
})
