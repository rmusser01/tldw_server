import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { initialState, useWorkspaceStore } from "@/store/workspace"
import type { OwnedWorkspaceBundle } from "@/store/workspace-api"
import { createOwnedWorkspaceDraftStore } from "@/store/owned-workspace-state"
import { useOwnedWorkspaceRename } from "../hooks/useOwnedWorkspaceRename"

const api = vi.hoisted(() => ({
  context: vi.fn(),
  get: vi.fn(),
  patch: vi.fn()
}))
vi.mock("@/services/owned-workspace-opening", () => ({
  createOwnedWorkspaceMetadataContext: api.context
}))
const state = () => useWorkspaceStore.getState()
const scope = {
  serverBase: "https://research.test",
  principalId: "2",
  organizationId: null
}
const bundle: OwnedWorkspaceBundle = {
  workspace: {
    id: "owned",
    name: "Original",
    version: 3,
    archived: false,
    deleted: false,
    created_at: "2026-09-13",
    last_modified: "2026-09-13",
    study_materials_policy: "general",
    workspace_profile: "research",
    banner_title: null,
    banner_subtitle: null,
    banner_color: null,
    audio_provider: null,
    audio_model: null,
    audio_voice: null,
    audio_speed: null
  },
  notes: [],
  sources: [],
  artifacts: []
}
const remote = { ...bundle.workspace, name: "Remote", version: 4 }
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}
function activate(id = "owned", principalId = "2") {
  const attempt = state().beginOwnedWorkspace({ ...scope, principalId }, id)
  state().activateOwnedWorkspace(attempt, {
    ...bundle,
    workspace: { ...bundle.workspace, id }
  })
}
beforeEach(() => {
  vi.resetAllMocks()
  localStorage.clear()
  useWorkspaceStore.setState({ ...initialState, storeHydrated: true })
  activate()
  api.context.mockResolvedValue({ get: api.get, patch: api.patch })
  api.patch.mockResolvedValue({ ...remote, name: "Mine" })
  api.get.mockResolvedValue(remote)
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  const current = state()
  current.setOwnedWorkspaceRenameDraft({
    origin: current.activeWorkspaceOrigin,
    workspaceId: current.workspaceId,
    expectedDraft: current.ownedWorkspaceRenameDraft,
    draft: null
  })
})

describe("owned workspace rename", () => {
  it.each(["keep", "server"])(
    "recovers a same-version GET after an unknown save failure with %s",
    async (choice) => {
      api.patch.mockRejectedValueOnce({ status: 500 })
      api.get.mockResolvedValue(bundle.workspace)
      const { result } = renderHook(useOwnedWorkspaceRename)
      act(() => result.current.start())
      act(() => result.current.change("Mine"))
      await act(() => result.current.save())
      await act(() => result.current.review())
      expect(result.current.remote?.name).toBe("Original")
      expect(result.current.error).toBeNull()
      expect(api.patch).toHaveBeenCalledTimes(1)
      if (choice === "keep") {
        act(() => result.current.keepMine())
        expect(result.current.canSave).toBe(true)
        expect(state().ownedWorkspaceRenameDraft).toEqual({
          name: "Mine",
          baseVersion: 3
        })
        await act(() => result.current.save())
        expect(api.patch).toHaveBeenLastCalledWith({ name: "Mine", version: 3 })
        expect(state().workspaceName).toBe("Mine")
      } else {
        act(() => result.current.useServer())
        expect(state().ownedWorkspaceRenameDraft).toBeNull()
        expect(state().workspaceName).toBe("Original")
        expect(state().ownedWorkspaceBundle?.workspace.version).toBe(3)
      }
    }
  )

  it.each([false, true])(
    "does not dispatch after cancel during context setup (reopen=%s)",
    async (reopen) => {
      const pending = deferred<{
        get: typeof api.get
        patch: typeof api.patch
      }>()
      api.context.mockReturnValue(pending.promise)
      const { result } = renderHook(useOwnedWorkspaceRename)
      act(() => result.current.start())
      let saving!: Promise<void>
      act(() => {
        saving = result.current.save()
      })
      act(() => result.current.cancel())
      if (reopen) act(() => result.current.start())
      const draft = state().ownedWorkspaceRenameDraft
      await act(async () => {
        pending.resolve({ get: api.get, patch: api.patch })
        await saving
      })
      expect(api.patch).not.toHaveBeenCalled()
      expect(state().ownedWorkspaceRenameDraft).toBe(draft)
    }
  )

  it("handles a rejected inspected receipt inline without throwing or losing the draft", async () => {
    api.patch.mockRejectedValue({ status: 409 })
    api.get.mockResolvedValue({ ...remote, deleted: true })
    const { result } = renderHook(useOwnedWorkspaceRename)
    act(() => result.current.start())
    await act(() => result.current.save())
    await act(() => result.current.review())
    const draft = state().ownedWorkspaceRenameDraft
    act(() => result.current.useServer())
    expect(result.current.error).toBe("reviewFailed")
    expect(state().ownedWorkspaceRenameDraft).toBe(draft)
  })

  it("does not request metadata on mount or edit and persists the entered base version", () => {
    const { result } = renderHook(useOwnedWorkspaceRename)
    act(() => result.current.start())
    act(() => result.current.change("Mine"))
    expect(state().ownedWorkspaceRenameDraft).toEqual({
      name: "Mine",
      baseVersion: 3
    })
    expect(
      createOwnedWorkspaceDraftStore(() => localStorage).load(scope, "owned")
    ).toMatchObject({
      status: "ready",
      durable: true,
      draft: { renameDraft: { name: "Mine", baseVersion: 3 } }
    })
    expect(api.context).not.toHaveBeenCalled()
  })

  it("saves the entered version without a GET even after canonical metadata advances", async () => {
    const { result } = renderHook(useOwnedWorkspaceRename)
    act(() => result.current.start())
    act(() => result.current.change("Mine"))
    act(() => {
      useWorkspaceStore.setState({
        ownedWorkspaceBundle: { ...bundle, workspace: remote }
      })
    })
    api.patch.mockResolvedValue({ ...remote, name: "Mine", version: 5 })
    await act(() => result.current.save())
    expect(api.patch).toHaveBeenCalledWith({ name: "Mine", version: 3 })
    expect(api.get).not.toHaveBeenCalled()
    expect(state().workspaceName).toBe("Mine")
    expect(state().ownedWorkspaceRenameDraft).toBeNull()
    expect(api.context).toHaveBeenCalledWith(
      "owned",
      scope,
      expect.any(AbortSignal)
    )
  })

  it("locks repeated saves immediately, before context setup resolves", async () => {
    const pending = deferred<Awaited<ReturnType<typeof api.context>>>()
    api.context.mockReturnValue(pending.promise)
    const { result } = renderHook(useOwnedWorkspaceRename)
    act(() => result.current.start())
    let first!: Promise<void>
    act(() => {
      first = result.current.save()
      void result.current.save()
    })
    expect(api.context).toHaveBeenCalledTimes(1)
    await act(async () => {
      pending.resolve({ get: api.get, patch: api.patch })
      await first
    })
    expect(api.patch).toHaveBeenCalledTimes(1)
  })

  it("keeps the draft on 409 until explicit review and keep-my-name, without retry", async () => {
    api.patch.mockRejectedValueOnce({ status: 409 })
    const { result } = renderHook(useOwnedWorkspaceRename)
    act(() => result.current.start())
    act(() => result.current.change("Mine"))
    await act(() => result.current.save())
    expect(result.current.error).toBe("conflict")
    expect(api.get).not.toHaveBeenCalled()
    await act(() => result.current.save())
    expect(api.patch).toHaveBeenCalledTimes(1)
    await act(() => result.current.review())
    expect(result.current.remote?.name).toBe("Remote")
    expect(state().ownedWorkspaceRenameDraft?.baseVersion).toBe(3)
    act(() => result.current.change("Newer typing"))
    act(() => result.current.keepMine())
    expect(state().ownedWorkspaceRenameDraft).toEqual({
      name: "Newer typing",
      baseVersion: 4
    })
    expect(api.patch).toHaveBeenCalledTimes(1)
    api.patch.mockResolvedValue({
      ...remote,
      name: "Newer typing",
      version: 5
    })
    await act(() => result.current.save())
    expect(api.patch).toHaveBeenLastCalledWith({
      name: "Newer typing",
      version: 4
    })
  })

  it("explicitly uses the inspected server name and discards the draft", async () => {
    api.patch.mockRejectedValue({ status: 409 })
    const { result } = renderHook(useOwnedWorkspaceRename)
    act(() => result.current.start())
    act(() => result.current.change("Mine"))
    await act(() => result.current.save())
    await act(() => result.current.review())
    act(() => result.current.useServer())
    expect(state().workspaceName).toBe("Remote")
    expect(state().ownedWorkspaceRenameDraft).toBeNull()
    expect(api.patch).toHaveBeenCalledTimes(1)
  })

  it("retains newer typing and advances its version after a save receipt", async () => {
    const pending = deferred<typeof remote>()
    api.patch.mockReturnValue(pending.promise)
    const { result } = renderHook(useOwnedWorkspaceRename)
    act(() => result.current.start())
    act(() => result.current.change("Mine"))
    let saving!: Promise<void>
    await act(async () => {
      saving = result.current.save()
    })
    act(() => result.current.change("Still typing"))
    await act(async () => {
      pending.resolve({ ...remote, name: "Mine" })
      await saving
    })
    expect(state().ownedWorkspaceRenameDraft).toEqual({
      name: "Still typing",
      baseVersion: 4
    })
    expect(state().workspaceName).toBe("Mine")
  })

  it("does not overwrite a cancelled and reopened editor on a late receipt", async () => {
    const pending = deferred<typeof remote>()
    api.patch.mockReturnValue(pending.promise)
    const { result } = renderHook(useOwnedWorkspaceRename)
    act(() => result.current.start())
    let saving!: Promise<void>
    await act(async () => {
      saving = result.current.save()
    })
    act(() => result.current.cancel())
    act(() => result.current.start())
    act(() => result.current.change("Reopened"))
    await act(async () => {
      pending.resolve(remote)
      await saving
    })
    expect(state().ownedWorkspaceRenameDraft).toEqual({
      name: "Reopened",
      baseVersion: 3
    })
    expect(result.current.error).toBeNull()
  })

  it.each([412, 500, undefined])(
    "preserves drafts on failure %s",
    async (status) => {
      api.patch.mockRejectedValue({ status })
      const { result } = renderHook(useOwnedWorkspaceRename)
      act(() => result.current.start())
      act(() => result.current.change("Mine"))
      await act(() => result.current.save())
      expect(result.current.error).toBe(
        status === 412 ? "accountChanged" : "saveFailed"
      )
      expect(state().ownedWorkspaceRenameDraft).toEqual({
        name: "Mine",
        baseVersion: 3
      })
      expect(api.patch).toHaveBeenCalledTimes(1)
    }
  )

  it("explains unavailable draft storage and preserves the in-memory editor", async () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("quota")
    })
    const { result } = renderHook(useOwnedWorkspaceRename)
    act(() => result.current.start())
    act(() => result.current.change("Mine"))
    expect(result.current.storageUnavailable).toBe(true)
    expect(state().ownedWorkspaceRenameDraft?.name).toBe("Mine")
  })

  it("retains the persisted draft and original version across reactivation", () => {
    const { result } = renderHook(useOwnedWorkspaceRename)
    act(() => result.current.start())
    act(() => result.current.change("Persisted"))
    act(() => {
      activate("other")
      activate()
    })
    expect(result.current.draft).toEqual({ name: "Persisted", baseVersion: 3 })
  })

  it.each(["workspace", "account", "attempt", "unmount"])(
    "rejects old callbacks across %s lifetimes including A-B-A",
    async (kind) => {
      const { result, unmount } = renderHook(useOwnedWorkspaceRename)
      act(() => result.current.start())
      const old = result.current
      act(() => {
        if (kind === "workspace") {
          activate("other")
          activate()
        }
        if (kind === "account") {
          activate("owned", "9")
          activate()
        }
        if (kind === "attempt") {
          const attempt = state().beginOwnedWorkspace(scope, "owned")
          state().activateOwnedWorkspace(attempt, bundle)
        }
        if (kind === "unmount") unmount()
      })
      const draft = state().ownedWorkspaceRenameDraft
      await act(async () => {
        old.change("Wrong")
        old.cancel()
        old.start()
        await old.save()
        await old.review()
        old.keepMine()
        old.useServer()
      })
      expect(state().ownedWorkspaceRenameDraft).toBe(draft)
      expect(api.context).not.toHaveBeenCalled()
    }
  )

  it("discards a late GET when the editor is cancelled and reopened", async () => {
    const pending = deferred<typeof remote>()
    api.patch.mockRejectedValue({ status: 409 })
    api.get.mockReturnValue(pending.promise)
    const { result } = renderHook(useOwnedWorkspaceRename)
    act(() => result.current.start())
    await act(() => result.current.save())
    let reviewing!: Promise<void>
    await act(async () => {
      reviewing = result.current.review()
    })
    act(() => result.current.cancel())
    act(() => result.current.start())
    await act(async () => {
      pending.resolve(remote)
      await reviewing
    })
    expect(result.current.remote).toBeNull()
    expect(state().ownedWorkspaceRenameDraft?.baseVersion).toBe(3)
  })
})
