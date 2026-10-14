// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { captureCurrentActorSceneTarget } from "@/hooks/chat/actor-scene-source"
import { useStoreMessageOption } from "@/store/option"
import { useActorStore } from "@/store/actor"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { createDefaultActorSettings } from "@/types/actor"
import { installSettingsLocks } from "@/services/__tests__/chat-settings-fixture"
import {
  RolePlaySetupDrawer,
  type RolePlayApplyContext,
  type RolePlaySetupApplyPayload
} from "../RolePlaySetupDrawer"

const io = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
  get: vi.fn<(key: string) => Promise<unknown>>(),
  set: vi.fn<(key: string, value: unknown) => Promise<void>>(),
  accountRevision: 0,
  accountAvailable: true,
  controller: null as HistorySelectionController | null
}))
// Account/H1 acceptance and the storage engine are declared doubles. The actual
// target capture, component, scene service, bucket CAS and Applied store are used.
vi.mock("@/utils/safe-storage", async (original) => ({
  ...(await original<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: () => ({
    hasPersistentBackend: true,
    get: io.get,
    set: io.set
  })
}))
vi.mock("@/hooks/chat/useHistorySelection", () => ({
  useHistorySelectionContext: () => io.controller
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback })
}))
vi.mock("@/components/Common/AssistantSelect", () => ({
  AssistantSelect: ({
    onSelectionChange
  }: {
    onSelectionChange: (
      selection: import("@/types/assistant-selection").AssistantSelection
    ) => void
  }) => (
    <button
      type="button"
      onClick={() =>
        onSelectionChange({
          kind: "character",
          id: "card-b",
          name: "Character B"
        })
      }>
      Choose Character B
    </button>
  )
}))
vi.mock("../SavedRolePlaySetupsPanel", () => ({
  SavedRolePlaySetupsPanel: () => null
}))
vi.mock("../playground-features", () => ({
  PRESETS: [],
  SystemPromptTemplatesModal: () => null
}))

const scope: ServicePromptRequestScope = {
  config: { serverUrl: "https://scene-owner.example", authMode: "single-user" },
  userId: 1
}
const originalScene = {
  ...createDefaultActorSettings(),
  notes: "Owned original scene."
}
const foreignScene = {
  ...createDefaultActorSettings(),
  notes: "Foreign scratch scene."
}
const ownedKey = (
  target: NonNullable<ReturnType<typeof captureCurrentActorSceneTarget>>
) => "actorSettings:owned:" + target.key
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
const currentTarget = () => {
  const target = captureCurrentActorSceneTarget(io.controller)
  if (!target) throw new Error("Missing declared original target")
  return target
}
const renderDrawer = (
  onApply = vi.fn<
    (
      payload: RolePlaySetupApplyPayload,
      context?: RolePlayApplyContext
    ) => boolean | Promise<boolean>
  >(() => true),
  props: Partial<React.ComponentProps<typeof RolePlaySetupDrawer>> = {}
) => {
  const onClose = vi.fn()
  const view = render(
    <RolePlaySetupDrawer
      open
      beforeState={{
        active: false,
        identity: null,
        behavior: null,
        scene: null,
        generationStyle: null,
        context: { pinnedCount: 0, hasExternalContext: false }
      }}
      historyId={useStoreMessageOption.getState().historyId}
      serverChatId={useStoreMessageOption.getState().serverChatId}
      onApply={onApply}
      onClose={onClose}
      {...props}
    />
  )
  return { ...view, onApply, onClose }
}
const stageAndApply = async (notes = "Frozen proposed scene.") => {
  const input = await screen.findByRole("textbox", { name: "Scene notes" })
  await waitFor(() => expect(input).toHaveValue(originalScene.notes))
  fireEvent.change(input, { target: { value: notes } })
  fireEvent.click(screen.getByRole("button", { name: "Apply", exact: true }))
}

beforeEach(() => {
  installSettingsLocks()
  usePlaygroundSessionStore.getState().clearSession()
  useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  useStoreMessageOption.getState().startNewChatDraft(false)
  useActorStore.getState().reset()
  io.accountRevision = 0
  io.accountAvailable = true
  const revision = io.accountRevision
  const signal = new AbortController().signal
  io.controller = {
    draftAccount: {
      requestScope: scope,
      isCurrent: () => io.accountAvailable && io.accountRevision === revision,
      retryVerification: () => {}
    },
    getCurrent: () => ({ owner: null, view: null }),
    getSignal: () => signal,
    fence: () => () => true
  } as unknown as HistorySelectionController
  io.values.clear()
  io.values.set(ownedKey(currentTarget()), {
    value: structuredClone(originalScene),
    updatedAt: 10
  })
  io.values.set("actorSettings:scratch", structuredClone(originalScene))
  io.get.mockReset().mockImplementation(async (key) => io.values.get(key))
  io.set.mockReset().mockImplementation(async (key, value) => {
    io.values.set(key, value)
  })
  vi.spyOn(console, "error").mockImplementation(() => {})
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe("actual role-play Scene ownership and original-read Save", () => {
  it("loads the original qualified fresh Scene without adopting ambiguous scratch", async () => {
    io.values.set("actorSettings:scratch", structuredClone(foreignScene))
    renderDrawer()
    expect(
      await screen.findByRole("textbox", { name: "Scene notes" })
    ).toHaveValue(originalScene.notes)
    expect(io.get).not.toHaveBeenCalledWith("actorSettings:scratch")
    expect(useActorStore.getState().applied).toBeNull()
  })

  it("disables Scene editing after its acknowledged original account retires", async () => {
    const f = renderDrawer()
    const notes = await screen.findByRole("textbox", { name: "Scene notes" })
    await waitFor(() => expect(notes).toHaveValue(originalScene.notes))
    expect(notes).toBeEnabled()
    io.accountRevision++
    f.rerender(<RolePlaySetupDrawer open beforeState={{ active: false, identity: null,
      behavior: null, scene: null, generationStyle: null,
      context: { pinnedCount: 0, hasExternalContext: false } }}
      historyId={null} serverChatId={null} onApply={f.onApply} onClose={f.onClose} />)
    expect(screen.getByRole("textbox", { name: "Scene notes" })).toBeDisabled()
    expect(io.set).not.toHaveBeenCalled()
  })

  it("refuses an unverified account before Scene storage IO", async () => {
    io.accountAvailable = false
    const f = renderDrawer()
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Scene settings could not be loaded"
    )
    expect(io.get).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Apply", exact: true }))
    expect(f.onApply).not.toHaveBeenCalled()
  })

  it("retains the original record for CAS and preserves staged edits after conflict", async () => {
    const target = currentTarget()
    const newer = {
      value: { ...originalScene, notes: "Independently saved scene." },
      updatedAt: 20
    }
    const onApply = vi.fn(() => {
      io.values.set(ownedKey(target), newer)
      return true
    })
    const f = renderDrawer(onApply)
    await stageAndApply()
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "scene settings could not be saved"
    )
    expect(io.values.get(ownedKey(target))).toEqual(newer)
    expect(screen.getByRole("textbox", { name: "Scene notes" })).toHaveValue(
      "Frozen proposed scene."
    )
    expect(useActorStore.getState().applied).toBeNull()
    expect(f.onClose).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Apply", exact: true }))
    await waitFor(() =>
      expect(
        io.get.mock.calls.filter(([key]) => key === ownedKey(target))
      ).toHaveLength(3)
    )
    expect(onApply).toHaveBeenCalledOnce()
  })

  it.each([false, true])(
    "reviews the saved Scene explicitly without silently adopting a later record (changed again: %s)",
    async (changedAgain) => {
      const target = currentTarget()
      const reviewed = {
        value: { ...originalScene, notes: "Saved scene to review." },
        updatedAt: 20
      }
      const newest = {
        value: { ...originalScene, notes: "Another independent scene." },
        updatedAt: 30
      }
      const onApply = vi.fn(() => {
        io.values.set(ownedKey(target), reviewed)
        return true
      })
      const f = renderDrawer(onApply)
      await stageAndApply()
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "scene settings could not be saved"
      )
      fireEvent.click(
        screen.getByRole("button", { name: "Review saved scene", exact: true })
      )
      await screen.findByRole("button", { name: "Use my staged edits" })
      expect(screen.getByText(/Saved scene to review/)).toBeInTheDocument()
      expect(screen.getByRole("textbox", { name: "Scene notes" })).toHaveValue(
        "Frozen proposed scene."
      )
      if (changedAgain) io.values.set(ownedKey(target), newest)
      fireEvent.click(
        screen.getByRole("button", { name: "Use my staged edits" })
      )
      await waitFor(() =>
        expect(screen.queryByRole("alert")).not.toBeInTheDocument()
      )
      fireEvent.click(
        screen.getByRole("button", { name: "Apply", exact: true })
      )
      if (changedAgain) {
        expect(await screen.findByRole("alert")).toHaveTextContent(
          "scene settings could not be saved"
        )
        expect(io.values.get(ownedKey(target))).toEqual(newest)
        expect(f.onClose).not.toHaveBeenCalled()
      } else {
        await waitFor(() => expect(f.onClose).toHaveBeenCalledOnce())
        expect(io.values.get(ownedKey(target))).toMatchObject({
          value: { notes: "Frozen proposed scene." }
        })
        expect(useActorStore.getState().applied?.target.key).toBe(target.key)
      }
      expect(onApply).toHaveBeenCalledOnce()
    }
  )

  it("atomically applies the acknowledged Scene and preserves its owner after drawer unmount", async () => {
    const target = currentTarget()
    const f = renderDrawer()
    await stageAndApply()
    await waitFor(() => expect(f.onClose).toHaveBeenCalledOnce())
    const state = useActorStore.getState()
    expect(state.applied?.target.key).toBe(target.key)
    expect(state.applied).toMatchObject({
      settings: { notes: "Frozen proposed scene." },
      preview: state.preview,
      tokenCount: state.tokenCount
    })
    expect(io.values.get(ownedKey(target))).toMatchObject({
      value: { notes: "Frozen proposed scene." }
    })
    expect(io.values.get("actorSettings:scratch")).toEqual(originalScene)
    f.unmount()
    expect(state.applied?.target.isCurrent()).toBe(true)
  })

  it("keeps the frozen Scene proposal if the Apply callback mutates its payload", async () => {
    const onApply = vi.fn((payload: RolePlaySetupApplyPayload) => {
      payload.sceneSettings!.notes = "Mutated caller proposal."
      return true
    })
    const f = renderDrawer(onApply)
    await stageAndApply()
    await waitFor(() => expect(f.onClose).toHaveBeenCalledOnce())
    expect(io.values.get(ownedKey(currentTarget()))).toMatchObject({
      value: { notes: "Frozen proposed scene." }
    })
  })

  it.each([false, true])(
    "saves only to an explicitly acknowledged new fresh target (conflict: %s)",
    async (conflict) => {
      const original = currentTarget()
      let destination: ReturnType<typeof currentTarget> | undefined
      const onApply = vi.fn(
        (
          _payload: RolePlaySetupApplyPayload,
          context?: RolePlayApplyContext
        ) => {
          const source = useStoreMessageOption.getState()
          const address = {
            kind: "fresh" as const,
            id: "issued-fresh-destination",
            scope: { type: "global" as const },
            temporary: false
          }
          expect(
            source.setChatDraftAddress(address, {
              sourceAddress: source.chatDraftAddress,
              expectedRevision: source.chatDraftRevision
            })
          ).toBe(true)
          destination = currentTarget()
          if (conflict)
            io.values.set(ownedKey(destination), {
              value: foreignScene,
              updatedAt: 30
            })
          context?.onDestination?.({
            historyId: null,
            serverChatId: null,
            isCurrent: () => destination!.isCurrent(),
            sceneTarget: destination
          })
          return true
        }
      )
      const f = renderDrawer(onApply)
      await stageAndApply()
      if (conflict) {
        expect(await screen.findByRole("alert")).toHaveTextContent(
          "scene settings could not be saved"
        )
        expect(io.values.get(ownedKey(destination!))).toMatchObject({
          value: foreignScene,
          updatedAt: 30
        })
        expect(f.onClose).not.toHaveBeenCalled()
        expect(useActorStore.getState().applied).toBeNull()
      } else {
        await waitFor(() => expect(f.onClose).toHaveBeenCalledOnce())
        expect(io.values.get(ownedKey(destination!))).toMatchObject({
          value: { notes: "Frozen proposed scene." }
        })
        expect(useActorStore.getState().applied?.target.key).toBe(
          destination!.key
        )
      }
      expect(io.values.get(ownedKey(original))).toEqual({
        value: originalScene,
        updatedAt: 10
      })
    }
  )

  it("does not adopt a silent caller's changed fresh draft from ambient null IDs", async () => {
    const onApply = vi.fn(() => {
      useStoreMessageOption.getState().startNewChatDraft(false)
      return true
    })
    const f = renderDrawer(onApply)
    await stageAndApply()
    await waitFor(() => expect(onApply).toHaveBeenCalledOnce())
    await act(async () => {})
    expect(io.set).not.toHaveBeenCalled()
    expect(f.onClose).not.toHaveBeenCalled()
    expect(useActorStore.getState().applied).toBeNull()
  })

  it("preserves the independently verified account when its native owner retires during acknowledged fresh Apply", async () => {
    const store = useStoreMessageOption.getState()
    const address = {
      kind: "native" as const,
      id: "native-original",
      ownerKey: "owner-a",
      scope: { type: "global" as const }
    }
    store.setChatDraftAddress(address)
    store.setServerChatId(address.id)
    const owner = {
      kind: "native" as const,
      conversation_id: address.id,
      owner_key: address.ownerKey,
      request_scope: scope,
      scope: address.scope,
      validate_lease: () =>
        useStoreMessageOption.getState().chatDraftAddress === address
    }
    io.controller!.getCurrent = (() => ({
      owner,
      view: { owner_key: address.ownerKey, conversation_id: address.id }
    })) as HistorySelectionController["getCurrent"]
    const original = currentTarget()
    io.values.set(ownedKey(original), { value: originalScene, updatedAt: 10 })
    io.values.set("actorSettings:server:" + address.id, originalScene)
    let destination: ReturnType<typeof currentTarget> | undefined
    const onApply = vi.fn(
      (_payload: RolePlaySetupApplyPayload, context?: RolePlayApplyContext) => {
        useStoreMessageOption.getState().startNewChatDraft(false)
        useStoreMessageOption.getState().setServerChatId(null)
        destination = currentTarget()
        context?.onDestination?.({
          historyId: null,
          serverChatId: null,
          sceneTarget: destination,
          isCurrent: () => destination!.isCurrent()
        })
        return true
      }
    )
    const f = renderDrawer(onApply)
    await stageAndApply()
    await waitFor(() => expect(f.onClose).toHaveBeenCalledOnce())
    expect(original.account!.isCurrent()).toBe(false)
    expect(io.values.get(ownedKey(destination!))).toMatchObject({
      value: { notes: "Frozen proposed scene." }
    })
    expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
  })

  it.each([
    "uninterrupted",
    "source selection",
    "same-ID H1",
    "guard replacement",
    "foreign character",
    "own character"
  ] as const)(
    "retains only its acknowledged source during held fresh Scene CAS: %s",
    async (change) => {
      let epoch = 0
      io.controller!.fence = () => {
        const original = epoch
        return () => original === epoch
      }
      const held = deferred<void>()
      let destination: ReturnType<typeof currentTarget> | undefined
      const onApply = vi.fn(
        (
          _payload: RolePlaySetupApplyPayload,
          context?: RolePlayApplyContext
        ) => {
          useStoreMessageOption.getState().startNewChatDraft(false)
          destination = currentTarget()
          io.get.mockImplementation(async (key) => {
            if (key === ownedKey(destination!)) await held.promise
            return io.values.get(key)
          })
          context?.onDestination?.({
            historyId: null,
            serverChatId: null,
            sceneTarget: destination,
            isCurrent: () => destination!.isCurrent()
          })
          return true
        }
      )
      const props: Partial<React.ComponentProps<typeof RolePlaySetupDrawer>> = {
        characterId: "card-a"
      }
      const f = renderDrawer(onApply, props)
      await waitFor(() =>
        expect(
          screen.getByRole("textbox", { name: "Scene notes" })
        ).toHaveValue(originalScene.notes)
      )
      if (change === "own character")
        fireEvent.click(
          screen.getByRole("button", { name: "Choose Character B" })
        )
      await stageAndApply()
      await waitFor(() =>
        expect(
          destination &&
            io.get.mock.calls.some(([key]) => key === ownedKey(destination!))
        ).toBe(true)
      )
      if (change === "source selection")
        act(() =>
          usePlaygroundSessionStore.setState((state) => ({
            sourceSelectionRevision: state.sourceSelectionRevision + 1
          }))
        )
      if (change === "same-ID H1") epoch++
      if (
        change === "guard replacement" ||
        change === "foreign character" ||
        change === "own character"
      ) {
        f.rerender(
          <RolePlaySetupDrawer
            open
            beforeState={{
              active: false,
              identity: null,
              behavior: null,
              scene: null,
              generationStyle: null,
              context: { pinnedCount: 0, hasExternalContext: false }
            }}
            historyId={null}
            serverChatId={null}
            characterId={change === "guard replacement" ? "card-a" : "card-b"}
            captureApplyGuard={
              change === "guard replacement" ? () => () => true : undefined
            }
            onApply={onApply}
            onClose={f.onClose}
          />
        )
      }
      await act(async () => {
        held.resolve()
        await new Promise((resolve) => setTimeout(resolve, 20))
      })
      if (change === "uninterrupted" || change === "own character") {
        await waitFor(() => expect(f.onClose).toHaveBeenCalledOnce())
        expect(io.set).toHaveBeenCalledOnce()
        expect(useActorStore.getState().applied?.target.key).toBe(
          destination!.key
        )
      } else {
        expect(io.set).not.toHaveBeenCalled()
        expect(useActorStore.getState().applied).toBeNull()
        expect(f.onClose).not.toHaveBeenCalled()
      }
    }
  )

  it("keeps unavailable character defaults distinct from an editable empty Scene", async () => {
    io.values.delete(ownedKey(currentTarget()))
    io.get.mockImplementation(async (key) => {
      if (key.startsWith("actorProfile:owned:"))
        throw new Error("default storage unavailable")
      return io.values.get(key)
    })
    renderDrawer(undefined, { characterId: "card-a" })
    expect(
      await screen.findByText(
        "Character defaults are unavailable. You can still edit and save this scene."
      )
    ).toHaveAttribute("role", "status")
    expect(screen.getByRole("textbox", { name: "Scene notes" })).toHaveValue("")
    expect(
      screen.getByRole("button", { name: "Apply", exact: true })
    ).toBeEnabled()
    expect(useActorStore.getState().applied).toBeNull()
  })

  it("cancels an unchanged staged Scene without applying or persisting it", async () => {
    const f = renderDrawer()
    await waitFor(() =>
      expect(screen.getByRole("textbox", { name: "Scene notes" })).toHaveValue(
        originalScene.notes
      )
    )
    fireEvent.click(screen.getByRole("button", { name: "Cancel", exact: true }))
    expect(f.onClose).toHaveBeenCalledOnce()
    expect(f.onApply).not.toHaveBeenCalled()
    expect(io.set).not.toHaveBeenCalled()
    expect(useActorStore.getState().applied).toBeNull()
  })

  it("refuses a retired account's held Scene without displaying or applying it", async () => {
    const held = deferred<void>()
    io.get.mockImplementationOnce(async (key) => {
      await held.promise
      return io.values.get(key)
    })
    const f = renderDrawer()
    await waitFor(() => expect(io.get).toHaveBeenCalledOnce())
    io.accountRevision++
    await act(async () => {
      held.resolve()
    })
    expect(
      screen.queryByDisplayValue(originalScene.notes)
    ).not.toBeInTheDocument()
    expect(io.set).not.toHaveBeenCalled()
    expect(f.onClose).not.toHaveBeenCalled()
  })
})
