import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { initialState, useWorkspaceStore } from "@/store/workspace"
import type { OwnedWorkspaceBundle } from "@/store/workspace-api"
import { createOwnedWorkspaceDraftStore } from "@/store/owned-workspace-state"
import { useOwnedWorkspaceBanner } from "../hooks/useOwnedWorkspaceBanner"

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
    banner_title: "Original title",
    banner_subtitle: "Original subtitle",
    banner_color: "#123456",
    audio_provider: null,
    audio_model: null,
    audio_voice: null,
    audio_speed: null,
    assistantDefaults: null
  },
  notes: [],
  sources: [],
  artifacts: []
}
const remote = {
  ...bundle.workspace,
  version: 4,
  banner_title: "Server title",
  banner_subtitle: "Server subtitle"
}
const context = () => ({ get: api.get, patch: api.patch })
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((yes) => {
    resolve = yes
  })
  return { promise, resolve }
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
  api.context.mockResolvedValue(context())
  api.get.mockResolvedValue(bundle.workspace)
  api.patch.mockResolvedValue(remote)
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("owned banner editor", () => {
  it("trims and caps restored text at the existing banner limits without altering the recovery draft", async () => {
    const { result } = renderHook(useOwnedWorkspaceBanner)
    await act(() => result.current.open())
    act(() => {
      result.current.changeTitle("  " + "t".repeat(90) + "  ")
      result.current.changeSubtitle("  " + "s".repeat(190) + "  ")
    })
    const draft = state().ownedWorkspaceBannerDraft
    api.patch.mockRejectedValue({ status: 409 })
    await act(() => result.current.save())
    expect(api.patch).toHaveBeenCalledWith({
      version: 3,
      banner_title: "t".repeat(80),
      banner_subtitle: "s".repeat(180)
    })
    expect(state().ownedWorkspaceBannerDraft).toBe(draft)
  })

  it("does not request on mount and loads canonical text using the frozen account context", async () => {
    const { result } = renderHook(useOwnedWorkspaceBanner)
    expect(api.context).not.toHaveBeenCalled()
    await act(() => result.current.open())
    expect(api.context).toHaveBeenCalledExactlyOnceWith(
      "owned",
      scope,
      expect.any(AbortSignal)
    )
    expect(state().ownedWorkspaceBannerDraft).toEqual({
      title: "Original title",
      subtitle: "Original subtitle",
      baseVersion: 3
    })
  })

  it.each([false, true])(
    "writes only versioned text, committing only after receipt (reset=%s)",
    async (reset) => {
      const pending = deferred<typeof remote>()
      api.patch.mockReturnValue(pending.promise)
      const { result } = renderHook(useOwnedWorkspaceBanner)
      await act(() => result.current.open())
      act(() => {
        result.current.changeTitle("Mine")
        result.current.changeSubtitle("My subtitle")
      })
      let saving!: Promise<void>
      await act(async () => {
        saving = result.current.save(reset)
      })
      expect(api.patch).toHaveBeenCalledWith({
        version: 3,
        banner_title: reset ? "" : "Mine",
        banner_subtitle: reset ? "" : "My subtitle"
      })
      expect(state().workspaceBanner.title).toBe("Original title")
      await act(async () => {
        pending.resolve({
          ...remote,
          banner_title: reset ? "" : "Mine",
          banner_subtitle: reset ? "" : "My subtitle"
        })
        await saving
      })
      expect(state().workspaceBanner.title).toBe(reset ? "" : "Mine")
      expect(state().ownedWorkspaceBundle?.workspace.banner_color).toBe(
        "#123456"
      )
      expect(state().ownedWorkspaceBundle?.workspace.name).toBe("Original")
      expect(state().ownedWorkspaceBannerDraft).toBeNull()
      expect(result.current.isOpen).toBe(false)
    }
  )

  it("persists text and its original base across unmount and requires explicit review on reopen", async () => {
    const first = renderHook(useOwnedWorkspaceBanner)
    await act(() => first.result.current.open())
    act(() => first.result.current.changeTitle("Mine"))
    first.unmount()
    const loaded = createOwnedWorkspaceDraftStore(() => localStorage).load(
      scope,
      "owned"
    )
    expect(loaded).toMatchObject({ status: "ready", durable: true })
    if (loaded.status !== "ready") throw new Error("Missing durable draft")
    const persisted = loaded.draft
    expect(persisted.bannerDraft).toEqual({
      title: "Mine",
      subtitle: "Original subtitle",
      baseVersion: 3
    })
    expect(JSON.stringify(persisted.bannerDraft)).not.toMatch(/image|reset/)
    api.get.mockResolvedValue(remote)
    const second = renderHook(useOwnedWorkspaceBanner)
    await act(() => second.result.current.open())
    expect(state().ownedWorkspaceBannerDraft?.baseVersion).toBe(3)
    expect(second.result.current.canSave).toBe(false)
    expect(second.result.current.remote?.version).toBe(4)
  })

  it.each([409, 412, 500])(
    "preserves draft and blocks retries after save error %s",
    async (status) => {
      api.patch.mockRejectedValue({ status })
      const { result } = renderHook(useOwnedWorkspaceBanner)
      await act(() => result.current.open())
      act(() => result.current.changeTitle("Mine"))
      const draft = state().ownedWorkspaceBannerDraft
      await act(() => result.current.save())
      await act(() => result.current.save())
      expect(state().ownedWorkspaceBannerDraft).toBe(draft)
      expect(result.current.error).toBe(
        status === 409
          ? "conflict"
          : status === 412
            ? "accountChanged"
            : "saveFailed"
      )
      expect(api.patch).toHaveBeenCalledTimes(1)
      if (status === 412) {
        await act(() => result.current.review())
        expect(api.get).toHaveBeenCalledTimes(1)
      }
    }
  )

  it("does not replay reset intent after conflict review and Keep my text", async () => {
    api.patch.mockRejectedValueOnce({ status: 409 })
    const { result } = renderHook(useOwnedWorkspaceBanner)
    await act(() => result.current.open())
    act(() => result.current.changeTitle("Mine"))
    await act(() => result.current.save(true))
    api.get.mockResolvedValue(remote)
    await act(() => result.current.review())
    expect(state().ownedWorkspaceBannerDraft?.baseVersion).toBe(3)
    act(() => result.current.keepMine())
    expect(api.patch).toHaveBeenCalledTimes(1)
    api.patch.mockResolvedValue({ ...remote, version: 5 })
    await act(() => result.current.save())
    expect(api.patch).toHaveBeenLastCalledWith({
      version: 4,
      banner_title: "Mine",
      banner_subtitle: "Original subtitle"
    })
  })

  it.each(["keep", "server"])(
    "allows explicit same-version review after uncertain save: %s",
    async (choice) => {
      api.patch.mockRejectedValue(new Error("response lost"))
      const { result } = renderHook(useOwnedWorkspaceBanner)
      await act(() => result.current.open())
      act(() => result.current.changeTitle("Mine"))
      await act(() => result.current.save())
      await act(() => result.current.review())
      expect(result.current.remote?.version).toBe(3)
      expect(result.current.error).toBeNull()
      act(() =>
        choice === "keep"
          ? result.current.keepMine()
          : result.current.useServer()
      )
      expect(result.current.canSave).toBe(choice === "keep")
      expect(result.current.isOpen).toBe(choice === "keep")
      expect(state().workspaceBanner.title).toBe("Original title")
    }
  )

  it("locks pending synchronously before resolving context", async () => {
    const { result } = renderHook(useOwnedWorkspaceBanner)
    await act(() => result.current.open())
    const pending = deferred<ReturnType<typeof context>>()
    api.context.mockReturnValueOnce(pending.promise)
    let saving!: Promise<void>
    act(() => {
      saving = result.current.save()
      void result.current.save(true)
    })
    expect(result.current.pending).toBe(true)
    expect(api.context).toHaveBeenCalledTimes(2)
    await act(async () => {
      pending.resolve(context())
      await saving
    })
    expect(api.patch).toHaveBeenCalledTimes(1)
  })

  it.each(["cancel", "edit"])(
    "suppresses dispatch when %s occurs before context resolves",
    async (action) => {
      const { result } = renderHook(useOwnedWorkspaceBanner)
      await act(() => result.current.open())
      const pending = deferred<ReturnType<typeof context>>()
      api.context.mockReturnValueOnce(pending.promise)
      let saving!: Promise<void>
      act(() => {
        saving = result.current.save()
      })
      act(() =>
        action === "cancel"
          ? result.current.cancel()
          : result.current.changeTitle("New")
      )
      if (action === "cancel") await act(() => result.current.open())
      const draft = state().ownedWorkspaceBannerDraft
      await act(async () => {
        pending.resolve(context())
        await saving
      })
      expect(api.patch).not.toHaveBeenCalled()
      expect(state().ownedWorkspaceBannerDraft).toBe(draft)
      expect(result.current.isOpen).toBe(true)
    }
  )

  it.each([false, true])(
    "accepts dispatched receipt without replacing newer edits (reopened=%s)",
    async (reopen) => {
      const { result } = renderHook(useOwnedWorkspaceBanner)
      await act(() => result.current.open())
      const pending = deferred<typeof remote>()
      api.patch.mockReturnValue(pending.promise)
      let saving!: Promise<void>
      await act(async () => {
        saving = result.current.save()
      })
      if (reopen) {
        act(() => result.current.cancel())
        await act(() => result.current.open())
      }
      act(() => result.current.changeTitle("Newer"))
      const draft = state().ownedWorkspaceBannerDraft
      await act(async () => {
        pending.resolve(remote)
        await saving
      })
      expect(state().workspaceBanner.title).toBe("Server title")
      expect(state().ownedWorkspaceBannerDraft).toEqual({
        ...draft,
        baseVersion: reopen ? 3 : 4
      })
      if (reopen) expect(state().ownedWorkspaceBannerDraft).toBe(draft)
      expect(result.current.isOpen).toBe(true)
    }
  )

  it.each(["unmount", "account", "aba"])(
    "guards loads, writes and old confirmation callbacks after %s",
    async (transition) => {
      const view = renderHook(useOwnedWorkspaceBanner)
      await act(() => view.result.current.open())
      const old = view.result.current
      const pending = deferred<typeof remote>()
      api.patch.mockReturnValue(pending.promise)
      let saving!: Promise<void>
      await act(async () => {
        saving = old.save()
      })
      if (transition === "unmount") view.unmount()
      else
        act(() => {
          activate("other", "9")
          if (transition === "aba") activate()
        })
      const workspace = state().ownedWorkspaceBundle?.workspace
      const draft = state().ownedWorkspaceBannerDraft
      await act(async () => {
        pending.resolve(remote)
        await saving
        await old.save(true)
      })
      act(() => {
        old.changeTitle("Leaked")
        old.cancel()
      })
      expect(state().ownedWorkspaceBundle?.workspace).toBe(workspace)
      expect(state().ownedWorkspaceBannerDraft).toBe(draft)
      expect(api.patch).toHaveBeenCalledTimes(1)
    }
  )

  it("does not let late GET or an old reset callback change a reopened editor", async () => {
    const pending = deferred<typeof remote>()
    api.get.mockReturnValueOnce(pending.promise)
    const { result } = renderHook(useOwnedWorkspaceBanner)
    let opening!: Promise<void>
    await act(async () => {
      opening = result.current.open()
    })
    const old = result.current
    act(() => result.current.cancel())
    await act(() => result.current.open())
    act(() => result.current.changeTitle("New"))
    const draft = state().ownedWorkspaceBannerDraft
    await act(async () => {
      pending.resolve(remote)
      await opening
      await old.save(true)
    })
    expect(state().ownedWorkspaceBannerDraft).toBe(draft)
    expect(api.patch).not.toHaveBeenCalled()
    expect(result.current.isOpen).toBe(true)
  })

  it.each([412, 500])(
    "keeps the modal unsavable on GET failure %s",
    async (status) => {
      api.get.mockRejectedValue({ status })
      const { result } = renderHook(useOwnedWorkspaceBanner)
      await act(() => result.current.open())
      expect(result.current.error).toBe(
        status === 412 ? "accountChanged" : "loadFailed"
      )
      await act(() => result.current.save(true))
      expect(api.patch).not.toHaveBeenCalled()
    }
  )

  it("rejects superseded review and requires a fresh inspection", async () => {
    const { result } = renderHook(useOwnedWorkspaceBanner)
    await act(() => result.current.open())
    api.get.mockResolvedValue(remote)
    await act(() => result.current.review())
    act(() => {
      useWorkspaceStore.setState({
        ownedWorkspaceBundle: {
          ...bundle,
          workspace: { ...remote, version: 5 }
        }
      })
    })
    act(() => result.current.keepMine())
    expect(result.current.error).toBe("reviewFailed")
    expect(result.current.remote).toBeNull()
    expect(state().ownedWorkspaceBannerDraft?.baseVersion).toBe(3)
  })

  it("retains text when review acceptance rejects invalid metadata", async () => {
    const { result } = renderHook(useOwnedWorkspaceBanner)
    await act(() => result.current.open())
    api.get.mockResolvedValue({ ...remote, deleted: true })
    await act(() => result.current.review())
    const draft = state().ownedWorkspaceBannerDraft
    act(() => result.current.useServer())
    expect(result.current.error).toBe("reviewFailed")
    expect(state().ownedWorkspaceBannerDraft).toBe(draft)
  })

  it("reports unavailable persistence without losing in-memory text", async () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("quota")
    })
    const { result } = renderHook(useOwnedWorkspaceBanner)
    await act(() => result.current.open())
    act(() => result.current.changeTitle("Mine"))
    expect(result.current.storageUnavailable).toBe(true)
    expect(state().ownedWorkspaceBannerDraft?.title).toBe("Mine")
  })
})
