import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { initialState, useWorkspaceStore } from "@/store/workspace"
import type { OwnedWorkspaceBundle } from "@/store/workspace-api"
import { createOwnedWorkspaceDraftStore } from "@/store/owned-workspace-state"
import { useOwnedWorkspaceAssistant } from "../hooks/useOwnedWorkspaceAssistant"

const api = vi.hoisted(() => ({
  context: vi.fn(),
  get: vi.fn(),
  patch: vi.fn(),
  listPersonas: vi.fn()
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
const defaults = (
  assistantId = "one",
  personaMemoryMode: "read_only" | "read_write" = "read_only"
) => ({
  assistantKind: "persona" as const,
  assistantId,
  personaMemoryMode,
  voice: null,
  style: null,
  toolPolicyProfileId: null
})
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
    audio_speed: null,
    assistantDefaults: defaults()
  },
  notes: [],
  sources: [],
  artifacts: []
}
const remote = {
  ...bundle.workspace,
  version: 4,
  assistantDefaults: defaults("two")
}
const context = () => ({
  get: api.get,
  patch: api.patch,
  listPersonas: api.listPersonas
})
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
  api.context.mockResolvedValue(context())
  api.get.mockResolvedValue(bundle.workspace)
  api.listPersonas.mockResolvedValue([
    { id: "one", name: "First" },
    { id: "two", name: "Second" }
  ])
  api.patch.mockResolvedValue(remote)
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("owned default assistant", () => {
  it("requests nothing on mount; open loads metadata and catalog through one frozen context", async () => {
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    expect(api.context).not.toHaveBeenCalled()
    await act(() => result.current.open())
    expect(api.context).toHaveBeenCalledExactlyOnceWith(
      "owned",
      scope,
      expect.any(AbortSignal)
    )
    expect(api.get).toHaveBeenCalledTimes(1)
    expect(api.listPersonas).toHaveBeenCalledTimes(1)
    expect(result.current.personas).toEqual([
      { id: "one", name: "First" },
      { id: "two", name: "Second" }
    ])
    expect(state().ownedWorkspaceAssistantDraft).toEqual({
      assistantId: "one",
      personaMemoryMode: "read_only",
      baseVersion: 3
    })
  })

  it.each([false, true])(
    "saves versioned defaults (clear=%s) only after the receipt",
    async (clear) => {
      const pending = deferred<typeof remote>()
      api.patch.mockReturnValue(pending.promise)
      const { result } = renderHook(useOwnedWorkspaceAssistant)
      await act(() => result.current.open())
      act(() => result.current.changePersona("two"))
      let saving!: Promise<void>
      await act(async () => {
        saving = result.current.save(clear)
      })
      expect(api.patch).toHaveBeenCalledWith({
        version: 3,
        assistantDefaults: clear ? null : defaults("two")
      })
      expect(state().assistantDefaults).toEqual(defaults())
      await act(async () => {
        pending.resolve({
          ...remote,
          assistantDefaults: clear ? null : defaults("two")
        })
        await saving
      })
      expect(state().assistantDefaults).toEqual(clear ? null : defaults("two"))
      expect(state().ownedWorkspaceAssistantDraft).toBeNull()
      expect(result.current.isOpen).toBe(false)
    }
  )

  it("retains the original draft version across unmount/reopen and never persists consent", async () => {
    const first = renderHook(useOwnedWorkspaceAssistant)
    await act(() => first.result.current.open())
    act(() => {
      first.result.current.changePersona("two")
      first.result.current.changeMode("read_write")
    })
    act(() => first.result.current.confirm(true))
    first.unmount()
    const loaded = createOwnedWorkspaceDraftStore(() => localStorage).load(
      scope,
      "owned"
    )
    expect(loaded).toMatchObject({ status: "ready", durable: true })
    if (loaded.status !== "ready") throw new Error("Missing durable draft")
    const persisted = loaded.draft
    expect(persisted.assistantDraft).toMatchObject({
      assistantId: "two",
      personaMemoryMode: "read_write",
      baseVersion: 3
    })
    expect(JSON.stringify(persisted)).not.toMatch(/confirm|consent/)
    api.get.mockResolvedValue(remote)
    const second = renderHook(useOwnedWorkspaceAssistant)
    expect(api.get).toHaveBeenCalledTimes(1)
    await act(() => second.result.current.open())
    expect(state().ownedWorkspaceAssistantDraft).toEqual({
      assistantId: "two",
      personaMemoryMode: "read_write",
      baseVersion: 3
    })
    expect(second.result.current.confirmed).toBe(false)
    expect(second.result.current.canSave).toBe(false)
  })

  it("requires fresh consent after persona, mode, and review changes", async () => {
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    act(() => result.current.changeMode("read_write"))
    await act(() => result.current.save())
    expect(api.patch).not.toHaveBeenCalled()
    act(() => result.current.confirm(true))
    expect(result.current.canSave).toBe(true)
    act(() => result.current.changePersona("two"))
    expect(result.current.confirmed).toBe(false)
    act(() => result.current.confirm(true))
    act(() => result.current.changeMode("read_only"))
    act(() => result.current.changeMode("read_write"))
    expect(result.current.confirmed).toBe(false)
    act(() => result.current.confirm(true))
    await act(() => result.current.review())
    expect(result.current.confirmed).toBe(false)
    act(() => result.current.keepMine())
    expect(result.current.canSave).toBe(false)
    act(() => result.current.confirm(true))
    await act(() => result.current.save())
    expect(api.patch).toHaveBeenCalledWith({
      version: 3,
      assistantDefaults: defaults("two", "read_write"),
      confirmReadWriteAssistantDefault: true
    })
  })

  it.each([false, true])(
    "requires explicit review and a new click after conflict (clear=%s)",
    async (clear) => {
      api.patch.mockRejectedValueOnce({ status: 409 })
      const { result } = renderHook(useOwnedWorkspaceAssistant)
      await act(() => result.current.open())
      await act(() => result.current.save(clear))
      expect(result.current.error).toBe("conflict")
      await act(() => result.current.save(clear))
      expect(api.patch).toHaveBeenCalledTimes(1)
      api.get.mockResolvedValue(remote)
      await act(() => result.current.review())
      expect(result.current.remote?.assistantDefaults).toEqual(defaults("two"))
      expect(state().ownedWorkspaceAssistantDraft?.baseVersion).toBe(3)
      act(() => result.current.keepMine())
      expect(state().ownedWorkspaceAssistantDraft?.baseVersion).toBe(4)
      expect(api.patch).toHaveBeenCalledTimes(1)
      api.patch.mockResolvedValue({ ...remote, version: 5 })
      await act(() => result.current.save(clear))
      expect(api.patch).toHaveBeenLastCalledWith({
        version: 4,
        assistantDefaults: clear ? null : defaults()
      })
    }
  )

  it.each(["keep", "server"])(
    "accepts same-version explicit review after an ambiguous save: %s",
    async (choice) => {
      api.patch.mockRejectedValue({ status: 500 })
      const { result } = renderHook(useOwnedWorkspaceAssistant)
      await act(() => result.current.open())
      act(() => result.current.changePersona("two"))
      await act(() => result.current.save())
      await act(() => result.current.save())
      expect(api.patch).toHaveBeenCalledTimes(1)
      await act(() => result.current.review())
      expect(result.current.remote?.version).toBe(3)
      expect(result.current.error).toBeNull()
      act(() =>
        choice === "keep"
          ? result.current.keepMine()
          : result.current.useServer()
      )
      expect(result.current.canSave).toBe(choice === "keep")
      expect(state().assistantDefaults).toEqual(defaults())
      expect(result.current.isOpen).toBe(choice === "keep")
    }
  )

  it("locks duplicate saves synchronously before context setup", async () => {
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    const pending = deferred<ReturnType<typeof context>>()
    api.context.mockReturnValue(pending.promise)
    let saving!: Promise<void>
    act(() => {
      saving = result.current.save()
      void result.current.save()
    })
    expect(api.context).toHaveBeenCalledTimes(2)
    await act(async () => {
      pending.resolve(context())
      await saving
    })
    expect(api.patch).toHaveBeenCalledTimes(1)
  })

  it("suppresses dispatch after Cancel/reopen during context setup", async () => {
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    const pending = deferred<ReturnType<typeof context>>()
    api.context.mockReturnValueOnce(pending.promise)
    let saving!: Promise<void>
    act(() => {
      saving = result.current.save()
    })
    act(() => result.current.cancel())
    await act(() => result.current.open())
    const draft = state().ownedWorkspaceAssistantDraft
    await act(async () => {
      pending.resolve(context())
      await saving
    })
    expect(api.patch).not.toHaveBeenCalled()
    expect(state().ownedWorkspaceAssistantDraft).toBe(draft)
    expect(result.current.isOpen).toBe(true)
  })

  it("applies an already dispatched receipt without closing or rewriting a reopened modal", async () => {
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    const pending = deferred<typeof remote>()
    api.patch.mockReturnValue(pending.promise)
    let saving!: Promise<void>
    await act(async () => {
      saving = result.current.save()
    })
    act(() => result.current.cancel())
    await act(() => result.current.open())
    act(() => result.current.changePersona("two"))
    const draft = state().ownedWorkspaceAssistantDraft
    await act(async () => {
      pending.resolve(remote)
      await saving
    })
    expect(state().assistantDefaults).toEqual(defaults("two"))
    expect(state().ownedWorkspaceAssistantDraft).toBe(draft)
    expect(result.current.isOpen).toBe(true)
  })

  it("retains newer typing after an accepted save receipt", async () => {
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    const pending = deferred<typeof remote>()
    api.patch.mockReturnValue(pending.promise)
    let saving!: Promise<void>
    await act(async () => {
      saving = result.current.save()
    })
    act(() => result.current.changePersona(""))
    await act(async () => {
      pending.resolve(remote)
      await saving
    })
    expect(state().ownedWorkspaceAssistantDraft).toEqual({
      assistantId: "",
      personaMemoryMode: "read_only",
      baseVersion: 4
    })
    expect(result.current.isOpen).toBe(true)
  })

  it.each(["unmount", "account", "aba"])(
    "rejects old load results and callbacks after %s",
    async (transition) => {
      const pending = deferred<typeof remote>()
      api.get.mockReturnValueOnce(pending.promise)
      const view = renderHook(useOwnedWorkspaceAssistant)
      let opening!: Promise<void>
      act(() => {
        opening = view.result.current.open()
      })
      const old = view.result.current
      if (transition === "unmount") view.unmount()
      else
        act(() => {
          activate("other", "9")
          if (transition === "aba") activate()
        })
      await act(async () => {
        pending.resolve(remote)
        await opening
      })
      act(() => {
        old.changePersona("two")
        old.confirm(true)
        old.cancel()
      })
      await act(() => old.save())
      expect(state().ownedWorkspaceAssistantDraft).toBeNull()
      expect(api.patch).not.toHaveBeenCalled()
    }
  )

  it("does not let a late open overwrite a second modal", async () => {
    const pending = deferred<typeof remote>()
    api.get.mockReturnValueOnce(pending.promise)
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    let opening!: Promise<void>
    await act(async () => {
      opening = result.current.open()
    })
    const old = result.current
    act(() => result.current.cancel())
    await act(() => result.current.open())
    act(() => result.current.changePersona("two"))
    const draft = state().ownedWorkspaceAssistantDraft
    await act(async () => {
      pending.resolve(remote)
      await opening
    })
    act(() => old.cancel())
    expect(state().ownedWorkspaceAssistantDraft).toBe(draft)
    expect(result.current.isOpen).toBe(true)
  })

  it.each([412, 500])(
    "blocks saving after a metadata load failure %s",
    async (status) => {
      api.get.mockRejectedValue({ status })
      const { result } = renderHook(useOwnedWorkspaceAssistant)
      await act(() => result.current.open())
      expect(result.current.error).toBe(
        status === 412 ? "accountChanged" : "loadFailed"
      )
      expect(result.current.canSave).toBe(false)
      await act(() => result.current.save())
      expect(api.patch).not.toHaveBeenCalled()
    }
  )

  it.each([404, 500])(
    "allows explicit clear after a catalog failure %s but not saving a choice",
    async (status) => {
      api.listPersonas.mockRejectedValue({ status })
      const { result } = renderHook(useOwnedWorkspaceAssistant)
      await act(() => result.current.open())
      expect(result.current.catalogError).toBe(true)
      expect(result.current.workspace).toEqual(bundle.workspace)
      expect(result.current.canSave).toBe(false)
      await act(() => result.current.save())
      expect(api.patch).not.toHaveBeenCalled()
      expect(result.current.canClear).toBe(true)
      api.patch.mockResolvedValue({ ...remote, assistantDefaults: null })
      await act(() => result.current.save(true))
      expect(api.patch).toHaveBeenCalledWith({
        version: 3,
        assistantDefaults: null
      })
      expect(state().assistantDefaults).toBeNull()
    }
  )

  it("blocks even clearing after a catalog account mismatch", async () => {
    api.listPersonas.mockRejectedValue({ status: 412 })
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    expect(result.current.error).toBe("accountChanged")
    await act(() => result.current.save(true))
    expect(api.patch).not.toHaveBeenCalled()
  })

  it("refuses a selected Persona absent from the frozen catalog", async () => {
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    act(() => result.current.changePersona("missing"))
    await act(() => result.current.save())
    expect(result.current.canSave).toBe(false)
    expect(api.patch).not.toHaveBeenCalled()
  })

  it("keeps catalog-failed clear conflicts explicit without preserving a hidden clear intent", async () => {
    api.listPersonas.mockRejectedValue({ status: 404 })
    api.patch.mockRejectedValueOnce({ status: 409 })
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    await act(() => result.current.save(true))
    api.get.mockResolvedValue(remote)
    await act(() => result.current.review())
    expect(result.current.catalogError).toBe(true)
    expect(result.current.remote?.assistantDefaults).toEqual(defaults("two"))
    act(() => result.current.keepMine())
    expect(result.current.canSave).toBe(false)
    expect(result.current.canClear).toBe(true)
    expect(api.patch).toHaveBeenCalledTimes(1)
    api.patch.mockResolvedValue({
      ...remote,
      version: 5,
      assistantDefaults: null
    })
    await act(() => result.current.save(true))
    expect(api.patch).toHaveBeenLastCalledWith({
      version: 4,
      assistantDefaults: null
    })
  })

  it.each(["unmount", "account", "aba"])(
    "discards an old save receipt after %s",
    async (transition) => {
      const view = renderHook(useOwnedWorkspaceAssistant)
      await act(() => view.result.current.open())
      const pending = deferred<typeof remote>()
      api.patch.mockReturnValue(pending.promise)
      let saving!: Promise<void>
      await act(async () => {
        saving = view.result.current.save()
      })
      if (transition === "unmount") view.unmount()
      else
        act(() => {
          activate("other", "9")
          if (transition === "aba") activate()
        })
      const workspace = state().ownedWorkspaceBundle?.workspace
      const draft = state().ownedWorkspaceAssistantDraft
      await act(async () => {
        pending.resolve(remote)
        await saving
      })
      expect(state().ownedWorkspaceBundle?.workspace).toBe(workspace)
      expect(state().ownedWorkspaceAssistantDraft).toBe(draft)
    }
  )

  it("requires review rather than silently rebasing after canonical metadata advances", async () => {
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    act(() => {
      useWorkspaceStore.setState({
        ownedWorkspaceBundle: { ...bundle, workspace: remote }
      })
    })
    api.patch.mockRejectedValueOnce({ status: 409 })
    await act(() => result.current.save())
    expect(api.patch).toHaveBeenCalledWith({
      version: 3,
      assistantDefaults: defaults()
    })
    expect(api.get).toHaveBeenCalledTimes(1)
    expect(result.current.error).toBe("conflict")
  })

  it("retains the draft when a reviewed receipt is rejected", async () => {
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    api.get.mockResolvedValue({ ...remote, deleted: true })
    await act(() => result.current.review())
    const draft = state().ownedWorkspaceAssistantDraft
    act(() => result.current.useServer())
    expect(result.current.error).toBe("reviewFailed")
    expect(result.current.remote).toBeNull()
    expect(state().ownedWorkspaceAssistantDraft).toBe(draft)
    expect(result.current.isOpen).toBe(true)
  })

  it("invalidates a review superseded by another metadata receipt and allows fresh review", async () => {
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    api.get.mockResolvedValue(remote)
    await act(() => result.current.review())
    const latest = { ...remote, version: 5 }
    act(() => {
      useWorkspaceStore.setState({
        ownedWorkspaceBundle: { ...bundle, workspace: latest }
      })
    })
    act(() => result.current.keepMine())
    expect(result.current.error).toBe("reviewFailed")
    expect(result.current.remote).toBeNull()
    api.get.mockResolvedValue(latest)
    await act(() => result.current.review())
    expect(result.current.remote?.version).toBe(5)
  })

  it("suppresses a prepared save when its draft changes before dispatch", async () => {
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    const pending = deferred<ReturnType<typeof context>>()
    api.context.mockReturnValueOnce(pending.promise)
    let saving!: Promise<void>
    act(() => {
      saving = result.current.save()
    })
    act(() => result.current.changePersona("two"))
    await act(async () => {
      pending.resolve(context())
      await saving
    })
    expect(api.patch).not.toHaveBeenCalled()
    expect(state().ownedWorkspaceAssistantDraft?.assistantId).toBe("two")
  })

  it("reports unavailable recovery storage without losing the in-memory settings", async () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("quota")
    })
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    act(() => result.current.changePersona("two"))
    expect(result.current.storageUnavailable).toBe(true)
    expect(state().ownedWorkspaceAssistantDraft?.assistantId).toBe("two")
  })

  it("opens a new modal after an external editor-session change", async () => {
    const { result } = renderHook(useOwnedWorkspaceAssistant)
    await act(() => result.current.open())
    act(() => {
      const current = state()
      current.setOwnedWorkspaceAssistantDraft({
        origin: current.activeWorkspaceOrigin,
        workspaceId: current.workspaceId,
        expectedDraft: current.ownedWorkspaceAssistantDraft,
        draft: null
      })
    })
    expect(result.current.isOpen).toBe(false)
    await act(() => result.current.open())
    expect(result.current.isOpen).toBe(true)
    expect(api.get).toHaveBeenCalledTimes(2)
  })
})
