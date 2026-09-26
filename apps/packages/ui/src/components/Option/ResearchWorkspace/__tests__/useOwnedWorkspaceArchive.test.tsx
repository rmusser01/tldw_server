import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { initialState, useWorkspaceStore } from "@/store/workspace"
import { createOwnedWorkspaceDraftStore } from "@/store/owned-workspace-state"
import { useOwnedWorkspaceArchive } from "../hooks/useOwnedWorkspaceArchive"

const api = vi.hoisted(() => ({
  context: vi.fn(),
  get: vi.fn(),
  patch: vi.fn(),
  navigate: vi.fn()
}))
vi.mock("@/services/owned-workspace-opening", () => ({
  createOwnedWorkspaceLifecycleContext: api.context
}))
vi.mock("react-router-dom", () => ({ useNavigate: () => api.navigate }))
const scope = {
  serverBase: "https://research.test",
  principalId: "2",
  organizationId: null
}
const workspace = {
  id: "owned",
  name: "Original",
  version: 3,
  archived: false,
  deleted: false,
  created_at: "2026-09-13",
  last_modified: "2026-09-13",
  study_materials_policy: "general" as const,
  workspace_profile: "research" as const,
  banner_title: null,
  banner_subtitle: null,
  banner_color: null,
  audio_provider: null,
  audio_model: null,
  audio_voice: null,
  audio_speed: null
}
const archived = { ...workspace, archived: true, version: 4 }
const state = () => useWorkspaceStore.getState()
function activate(id = "owned", principalId = "2") {
  const attempt = state().beginOwnedWorkspace({ ...scope, principalId }, id)
  state().activateOwnedWorkspace(attempt, {
    workspace: { ...workspace, id },
    notes: [],
    sources: [],
    artifacts: []
  })
}
function draft(name: string) {
  const s = state()
  s.setOwnedWorkspaceRenameDraft({
    origin: s.activeWorkspaceOrigin,
    workspaceId: s.workspaceId,
    expectedDraft: s.ownedWorkspaceRenameDraft,
    draft: { name, baseVersion: 3 }
  })
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
const context = () => ({ scope, get: api.get, setArchived: api.patch })
beforeEach(() => {
  vi.resetAllMocks()
  localStorage.clear()
  useWorkspaceStore.setState({ ...initialState, storeHydrated: true })
  activate()
  api.context.mockResolvedValue(context())
  api.patch.mockResolvedValue(archived)
  api.get.mockResolvedValue({ ...workspace, name: "Reviewed", version: 5 })
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("owned archive lifecycle", () => {
  it("pins account and version, retains the latest draft, and navigates only after accepted completion", async () => {
    const pending = deferred<typeof archived>()
    api.patch.mockReturnValue(pending.promise)
    const { result } = renderHook(useOwnedWorkspaceArchive)
    expect(api.context).not.toHaveBeenCalled()
    act(() => {
      draft("Before")
      result.current.open()
    })
    let saving!: Promise<void>
    await act(async () => {
      saving = result.current.confirm()
    })
    expect(api.context).toHaveBeenCalledWith(
      "owned",
      scope,
      expect.any(AbortSignal)
    )
    expect(api.patch).toHaveBeenCalledExactlyOnceWith(true, 3)
    expect(api.get).not.toHaveBeenCalled()
    act(() => draft("Latest typing"))
    await act(async () => {
      pending.resolve(archived)
      await saving
    })
    expect(
      createOwnedWorkspaceDraftStore(() => localStorage).load(scope, "owned")
    ).toMatchObject({
      status: "ready",
      durable: true,
      draft: { renameDraft: { name: "Latest typing", baseVersion: 3 } }
    })
    expect(state().activeWorkspaceOrigin.kind).toBe("legacy-local")
    expect(api.navigate).toHaveBeenCalledExactlyOnceWith("/workspaces")
  })

  it("locks dispatch synchronously before context creation completes", async () => {
    const pending = deferred<ReturnType<typeof context>>()
    api.context.mockReturnValue(pending.promise)
    const { result } = renderHook(useOwnedWorkspaceArchive)
    act(() => result.current.open())
    let saving!: Promise<void>
    act(() => {
      saving = result.current.confirm()
      void result.current.confirm()
    })
    expect(api.context).toHaveBeenCalledTimes(1)
    await act(async () => {
      pending.resolve(context())
      await saving
    })
    expect(api.patch).toHaveBeenCalledTimes(1)
  })

  it.each([false, true])(
    "rejects stale confirmation before context creation (reopen=%s)",
    async (reopen) => {
      const { result } = renderHook(useOwnedWorkspaceArchive)
      act(() => result.current.open())
      const stale = result.current.confirm
      act(() => result.current.cancel())
      if (reopen) act(() => result.current.open())
      await act(stale)
      expect(api.context).not.toHaveBeenCalled()
    }
  )

  it.each([false, true])(
    "rejects stale confirmation after context creation (reopen=%s)",
    async (reopen) => {
      const pending = deferred<ReturnType<typeof context>>()
      api.context.mockReturnValue(pending.promise)
      const { result } = renderHook(useOwnedWorkspaceArchive)
      act(() => result.current.open())
      let saving!: Promise<void>
      act(() => {
        saving = result.current.confirm()
      })
      act(() => result.current.cancel())
      if (reopen) act(() => result.current.open())
      await act(async () => {
        pending.resolve(context())
        await saving
      })
      expect(api.patch).not.toHaveBeenCalled()
    }
  )

  it.each([409, 412, 500, undefined])(
    "blocks repeated PATCH after %s, including cancel/reopen, until explicit successful inspection",
    async (status) => {
      api.patch.mockRejectedValueOnce({ status })
      const { result } = renderHook(useOwnedWorkspaceArchive)
      act(() => {
        draft("Preserved")
        result.current.open()
      })
      await act(() => result.current.confirm())
      expect(result.current.error).toBe(
        status === 409
          ? "conflict"
          : status === 412
            ? "accountChanged"
            : "uncertain"
      )
      const oldConfirmation = result.current.confirm
      act(() => result.current.cancel())
      act(() => result.current.open())
      await act(() => result.current.confirm())
      expect(api.patch).toHaveBeenCalledTimes(1)
      expect(api.get).not.toHaveBeenCalled()
      expect(state().ownedWorkspaceRenameDraft?.name).toBe("Preserved")
      await act(() => result.current.checkStatus())
      expect(result.current.remote?.name).toBe("Reviewed")
      expect(result.current.remote?.archived).toBe(false)
      expect(api.patch).toHaveBeenCalledTimes(1)
      await act(oldConfirmation)
      expect(api.patch).toHaveBeenCalledTimes(1)
      api.patch.mockResolvedValue({ ...archived, version: 6 })
      await act(() => result.current.confirm())
      expect(api.patch).toHaveBeenLastCalledWith(true, 5)
    }
  )

  it("failed or regressing status reads do not release uncertainty", async () => {
    api.patch.mockRejectedValue(new Error("lost response"))
    const { result } = renderHook(useOwnedWorkspaceArchive)
    act(() => result.current.open())
    await act(() => result.current.confirm())
    api.get.mockRejectedValueOnce(new Error("offline"))
    await act(() => result.current.checkStatus())
    expect(result.current.error).toBe("reviewFailed")
    api.get.mockResolvedValue({ ...workspace, version: 2 })
    await act(() => result.current.checkStatus())
    await act(() => result.current.confirm())
    expect(api.patch).toHaveBeenCalledTimes(1)
    expect(api.navigate).not.toHaveBeenCalled()
  })

  it.each([3, 4])(
    "reconciles an archived status read at version %s without another mutation",
    async (version) => {
      api.patch.mockRejectedValueOnce(new Error("response lost"))
      api.get.mockResolvedValue({ ...archived, version })
      const { result } = renderHook(useOwnedWorkspaceArchive)
      act(() => {
        draft("Retained")
        result.current.open()
      })
      await act(() => result.current.confirm())
      await act(() => result.current.checkStatus())
      expect(api.patch).toHaveBeenCalledTimes(1)
      expect(api.navigate).toHaveBeenCalledWith("/workspaces")
      expect(
        createOwnedWorkspaceDraftStore(() => localStorage).load(scope, "owned")
      ).toMatchObject({
        status: "ready",
        durable: true,
        draft: { renameDraft: { name: "Retained", baseVersion: 3 } }
      })
    }
  )

  it("checks draft storage immediately before dispatch, after context creation", async () => {
    const pending = deferred<ReturnType<typeof context>>()
    api.context.mockReturnValue(pending.promise)
    const { result } = renderHook(useOwnedWorkspaceArchive)
    act(() => {
      draft("Keep")
      result.current.open()
    })
    let saving!: Promise<void>
    act(() => {
      saving = result.current.confirm()
    })
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("quota")
    })
    await act(async () => {
      pending.resolve(context())
      await saving
    })
    expect(api.patch).not.toHaveBeenCalled()
    expect(result.current.error).toBe("storageUnavailable")
    expect(state().ownedWorkspaceRenameDraft?.name).toBe("Keep")
  })

  it("blocks cancel and retries completion, not mutation, after post-archive storage failure", async () => {
    const pending = deferred<typeof archived>()
    api.patch.mockReturnValue(pending.promise)
    const { result } = renderHook(useOwnedWorkspaceArchive)
    act(() => {
      draft("Keep")
      result.current.open()
    })
    let saving!: Promise<void>
    await act(async () => {
      saving = result.current.confirm()
    })
    const storage = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new Error("quota")
      })
    await act(async () => {
      pending.resolve(archived)
      await saving
    })
    expect(result.current.completionBlocked).toBe(true)
    expect(result.current.error).toBe("completionFailed")
    act(() => result.current.cancel())
    expect(result.current.isOpen).toBe(true)
    await act(() => result.current.confirm())
    expect(api.patch).toHaveBeenCalledTimes(1)
    expect(api.navigate).not.toHaveBeenCalled()
    expect(state().ownedWorkspaceRenameDraft?.name).toBe("Keep")
    storage.mockRestore()
    act(() => result.current.finish())
    expect(api.navigate).toHaveBeenCalledWith("/workspaces")
    expect(api.patch).toHaveBeenCalledTimes(1)
    expect(api.get).not.toHaveBeenCalled()
  })

  it.each(["context", "patch", "get"])(
    "suppresses late %s after cancel/reopen and keeps uncertainty after dispatch",
    async (phase) => {
      const gate = deferred<ReturnType<typeof context> | typeof archived>()
      if (phase === "context") api.context.mockReturnValue(gate.promise)
      if (phase === "patch") api.patch.mockReturnValue(gate.promise)
      if (phase === "get") {
        api.patch.mockRejectedValue(new Error("offline"))
        api.get.mockReturnValue(gate.promise)
      }
      const { result } = renderHook(useOwnedWorkspaceArchive)
      act(() => result.current.open())
      let operation!: Promise<void>
      if (phase === "get") {
        await act(() => result.current.confirm())
        await act(async () => {
          operation = result.current.checkStatus()
        })
      } else
        await act(async () => {
          operation = result.current.confirm()
        })
      act(() => result.current.cancel())
      act(() => result.current.open())
      await act(async () => {
        gate.resolve(phase === "context" ? context() : archived)
        await operation
      })
      expect(api.navigate).not.toHaveBeenCalled()
      expect(state().workspaceId).toBe("owned")
      expect(result.current.remote).toBeNull()
      if (phase !== "context") {
        await act(() => result.current.confirm())
        expect(api.patch).toHaveBeenCalledTimes(1)
      } else expect(api.patch).not.toHaveBeenCalled()
    }
  )

  it.each(["workspace", "account", "attempt", "unmount"])(
    "invalidates old callbacks and late receipts across %s changes",
    async (kind) => {
      const pending = deferred<typeof archived>()
      api.patch.mockReturnValue(pending.promise)
      const { result, unmount } = renderHook(useOwnedWorkspaceArchive)
      act(() => result.current.open())
      const old = result.current
      let saving!: Promise<void>
      await act(async () => {
        saving = old.confirm()
      })
      act(() => {
        if (kind === "workspace") {
          activate("other")
          activate()
        }
        if (kind === "account") {
          activate("owned", "9")
          activate()
        }
        if (kind === "attempt") activate()
        if (kind === "unmount") unmount()
      })
      await act(async () => {
        pending.resolve(archived)
        await saving
        old.open()
        old.cancel()
        await old.confirm()
        await old.checkStatus()
        old.finish()
      })
      expect(api.context).toHaveBeenCalledTimes(1)
      expect(api.navigate).not.toHaveBeenCalled()
      expect(state().workspaceId).toBe("owned")
    }
  )

  it("does nothing for a legacy local workspace", async () => {
    useWorkspaceStore.setState({
      ...initialState,
      workspaceId: "local",
      storeHydrated: true
    })
    const { result } = renderHook(useOwnedWorkspaceArchive)
    act(() => result.current.open())
    await act(() => result.current.confirm())
    expect(result.current.enabled).toBe(false)
    expect(api.context).not.toHaveBeenCalled()
  })

  it("requires a new confirmation callback for the reviewed version even in the same modal", async () => {
    api.patch.mockRejectedValueOnce({ status: 409 })
    const { result } = renderHook(useOwnedWorkspaceArchive)
    act(() => result.current.open())
    const stale = result.current.confirm
    await act(stale)
    await act(() => result.current.checkStatus())
    await act(stale)
    expect(api.patch).toHaveBeenCalledTimes(1)
    api.patch.mockResolvedValue({ ...archived, version: 6 })
    await act(() => result.current.confirm())
    expect(api.patch).toHaveBeenLastCalledWith(true, 5)
  })

  it.each(["workspace", "account", "unmount"])(
    "does not dispatch if %s changes during context creation",
    async (kind) => {
      const pending = deferred<ReturnType<typeof context>>()
      api.context.mockReturnValue(pending.promise)
      const { result, unmount } = renderHook(useOwnedWorkspaceArchive)
      act(() => result.current.open())
      let saving!: Promise<void>
      act(() => {
        saving = result.current.confirm()
      })
      act(() => {
        if (kind === "workspace") activate("other")
        if (kind === "account") activate("owned", "9")
        if (kind === "unmount") unmount()
      })
      await act(async () => {
        pending.resolve(context())
        await saving
      })
      expect(api.patch).not.toHaveBeenCalled()
      expect(api.navigate).not.toHaveBeenCalled()
    }
  )

  it("does not navigate when completion rejects a receipt behind current metadata", async () => {
    const pending = deferred<typeof archived>()
    api.patch.mockReturnValue(pending.promise)
    const { result } = renderHook(useOwnedWorkspaceArchive)
    act(() => result.current.open())
    let saving!: Promise<void>
    await act(async () => {
      saving = result.current.confirm()
    })
    act(() => {
      useWorkspaceStore.setState({
        ownedWorkspaceBundle: {
          ...state().ownedWorkspaceBundle!,
          workspace: { ...workspace, version: 5 }
        }
      })
    })
    await act(async () => {
      pending.resolve(archived)
      await saving
    })
    expect(api.navigate).not.toHaveBeenCalled()
    expect(state().workspaceId).toBe("owned")
    expect(result.current.error).toBe("reviewFailed")
    expect(result.current.needsReview).toBe(true)
    api.get.mockResolvedValue({ ...archived, version: 5 })
    await act(() => result.current.checkStatus())
    expect(api.patch).toHaveBeenCalledTimes(1)
    expect(api.navigate).toHaveBeenCalledWith("/workspaces")
  })
})
