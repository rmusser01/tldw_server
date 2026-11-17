import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createDefaultActorSettings } from "@/types/actor"
import { installSettingsLocks } from "./chat-settings-fixture"

const io = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
  localPersistent: true,
  legacyPersistent: true,
  get: vi.fn(),
  set: vi.fn(),
  verifyProfile: vi.fn()
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: (options?: { area?: string }) => ({
    get hasPersistentBackend() {
      return options?.area === "local"
        ? io.localPersistent
        : io.legacyPersistent
    },
    get: io.get,
    set: io.set
  })
}))
vi.mock("@/db/dexie/history-selection", () => ({
  ensureLocalProfileId: io.verifyProfile
}))
import {
  getActorSettingsForChat,
  getActorSettingsForChatWithCharacterFallback,
  getActorSceneForChat,
  saveActorSettingsForChat,
  getActorProfileForCharacter,
  saveActorProfileForCharacter,
  copyActorSceneToBoundChat
} from "../actor-settings"
import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import {
  bindActorSceneAfterLoad,
  captureActorSceneSend,
  readActorSceneForSend
} from "@/hooks/chat/actor-scene-source"
import { useActorStore } from "@/store/actor"
import { useStoreMessageOption } from "@/store/option"
import type { ChatScope } from "@/types/chat-scope"
import { captureChatDraftLoad } from "@/hooks/chat/chat-draft-transition"
import type {
  HistorySelectionController,
  HistoryLoadReceipt
} from "@/hooks/chat/useHistorySelection"

const accountA = "a".repeat(64),
  accountB = "b".repeat(64)
const sceneA = {
  ...createDefaultActorSettings(),
  notes: "Original account and conversation scene."
}
const sceneB = {
  ...createDefaultActorSettings(),
  notes: "Independent account and conversation scene."
}
// Owner/lease qualification is a declared fixture boundary; these tests exercise actual storage routing/CAS.
const target = (parts: unknown[], accountKey: string | null = accountA) => ({
  key: JSON.stringify(parts),
  accountKey,
  isCurrent: () => true,
  profileIsCurrent: () => accountKey !== null
})
const native = target(["native", accountA, "global", null, "collision"])
const source = { historyId: null, serverChatId: "collision", target: native }
const sceneKey = (value: typeof native) => `actorSettings:owned:${value.key}`
const profile = (accountKey: string) => ({
  key: JSON.stringify([accountKey, "card-one"]),
  isCurrent: () => true
})
const profileA = profile(accountA),
  profileB = profile(accountB)

const isolated = [
  {
    label: "fresh address",
    historyId: null,
    serverChatId: null,
    a: target(["fresh", accountA, "global", null, "fresh-a"]),
    b: target(["fresh", accountA, "global", null, "fresh-b"])
  },
  {
    label: "native account",
    historyId: null,
    serverChatId: "collision",
    a: native,
    b: target(["native", accountB, "global", null, "collision"], accountB)
  },
  {
    label: "native workspace",
    historyId: null,
    serverChatId: "collision",
    a: target(["native", accountA, "workspace", "workspace-a", "collision"]),
    b: target(["native", accountA, "workspace", "workspace-b", "collision"])
  },
  {
    label: "genuine local profile",
    historyId: "collision",
    serverChatId: null,
    a: target(["local", "profile-a", "collision"], null),
    b: target(["local", "profile-b", "collision"], null)
  }
]

describe("qualified scene and character-default records", () => {
  beforeEach(() => {
    installSettingsLocks()
    io.localPersistent = true
    io.legacyPersistent = true
    io.values.clear()
    io.verifyProfile.mockReset().mockImplementation(async (expected: string) => {
      if (expected !== "profile-a") throw new Error("owner_mismatch")
      return expected
    })
    io.get
      .mockReset()
      .mockImplementation(async (key: string) => io.values.get(key))
    io.set
      .mockReset()
      .mockImplementation(async (key: string, value: unknown) => {
        io.values.set(key, value)
      })
    vi.spyOn(console, "error").mockImplementation(() => undefined)
  })
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it.each(isolated)(
    "isolates $label across scene Save and read",
    async ({ historyId, serverChatId, a, b }) => {
      const first = { historyId, serverChatId, target: a },
        second = { historyId, serverChatId, target: b }
      expect(
        await saveActorSettingsForChat({
          ...first,
          settings: sceneA,
          expectedRecord: null
        })
      ).toBe(true)
      expect(
        await saveActorSettingsForChat({
          ...second,
          settings: sceneB,
          expectedRecord: null
        })
      ).toBe(true)
      expect(await getActorSettingsForChat(first)).toEqual(sceneA)
      expect(await getActorSettingsForChat(second)).toEqual(sceneB)
    }
  )

  it("retains an ambiguous bare conversation scene without adopting or rewriting it", async () => {
    const legacyKey = "actorSettings:server:collision"
    io.values.set(legacyKey, sceneA)
    expect(await getActorSettingsForChat(source)).toEqual(
      createDefaultActorSettings()
    )
    expect(io.values.get(legacyKey)).toEqual(sceneA)
    expect(io.get).not.toHaveBeenCalledWith(legacyKey)
    expect(io.set).not.toHaveBeenCalled()
  })

  describe("frozen fresh scene copy after an explicitly accepted bind", () => {
    const requestScope = {
      config: {
        serverUrl: "https://bind-scene.example",
        authMode: "single-user" as const
      },
      userId: 1
    }
    const arrangeBind = async (
      kind: "native" | "local" = "native",
      saved = true,
      scope: ChatScope = { type: "global" }
    ) => {
      let sourceCurrent = true,
        accountCurrent = true,
        bindingCurrent = true
      const fresh = captureActorSettingsTarget({
        address: { kind: "fresh", id: "fresh-copy", scope, temporary: false },
        requestScope,
        accountIsCurrent: () => accountCurrent,
        isCurrent: () => sourceCurrent
      })!
      const record = { value: structuredClone(sceneA), updatedAt: 7 }
      if (saved) io.values.set(sceneKey(fresh), record)
      const read = await getActorSceneForChat({ target: fresh })
      const owner =
        kind === "native"
          ? {
              kind,
              conversation_id: "bound-one",
              owner_key: "accepted-owner",
              request_scope: requestScope,
              scope,
              validate_lease: () => bindingCurrent
            }
          : {
              kind,
              conversation_id: "bound-one",
              owner_key: "accepted-owner",
              profile_id: "profile-a",
              validate_lease: () => bindingCurrent
            }
      const destination = captureActorSettingsTarget({
        address:
          kind === "native"
            ? { kind, id: "bound-one", ownerKey: "accepted-owner", scope }
            : { kind, id: "bound-one", profileId: "profile-a" },
        owner,
        view: {
          owner_key: "accepted-owner",
          conversation_id: "bound-one"
        } as HistoryLoadReceipt["view"],
        requestScope,
        accountIsCurrent: () => accountCurrent,
        isCurrent: () => bindingCurrent
      })!
      sourceCurrent = false
      io.set.mockClear()
      return {
        fresh,
        destination,
        read,
        record,
        retireAccount: () => {
          accountCurrent = false
        },
        retireBinding: () => {
          bindingCurrent = false
        }
      }
    }
    it.each(["native", "local"] as const)(
      "copies an original saved scene to accepted %s ownership and retains its source",
      async (kind) => {
        const bound = await arrangeBind(kind)
        expect(bound.fresh.isCurrent()).toBe(false)
        expect(
          await copyActorSceneToBoundChat(bound.read, bound.destination)
        ).toBe(true)
        expect(io.values.get(sceneKey(bound.fresh))).toEqual(bound.record)
        expect(io.values.get(sceneKey(bound.destination))).toEqual(bound.record)
      }
    )
    it("publishes an owned atomic scene only after its accepted copy is acknowledged", async () => {
      const bound = await arrangeBind()
      const receipt: HistoryLoadReceipt = {
        owner: {
          kind: "native",
          owner_key: "accepted-owner",
          conversation_id: "bound-one",
          request_scope: requestScope,
          validate_lease: () => true
        },
        view: {
          owner_key: "accepted-owner",
          conversation_id: "bound-one"
        } as HistoryLoadReceipt["view"]
      }
      useActorStore.getState().reset()
      expect(await bindActorSceneAfterLoad(bound.read, receipt, () => true)).toBe(
        true
      )
      expect(useActorStore.getState().applied?.target.key).toBe(
        bound.destination.key
      )
      expect(useActorStore.getState().applied?.settings).toEqual(sceneA)
      expect(io.values.get(sceneKey(bound.destination))).toEqual(bound.record)
    })
    it("keeps inherited defaults staged after an accepted bind", async () => {
      const bound = await arrangeBind("native", false)
      const receipt: HistoryLoadReceipt = {
        owner: {
          kind: "native",
          owner_key: "accepted-owner",
          conversation_id: "bound-one",
          request_scope: requestScope,
          validate_lease: () => true
        },
        view: {
          owner_key: "accepted-owner",
          conversation_id: "bound-one"
        } as HistoryLoadReceipt["view"]
      }
      useActorStore.getState().reset()
      expect(
        await bindActorSceneAfterLoad(
          { ...bound.read, origin: "profile", settings: sceneA },
          receipt,
          () => true
        )
      ).toBe(true)
      expect(useActorStore.getState().applied).toBeNull()
      expect(io.set).not.toHaveBeenCalled()
    })
    it("does not publish Applied after preparation retires during its copy", async () => {
      const bound = await arrangeBind()
      const receipt: HistoryLoadReceipt = {
        owner: {
          kind: "native",
          owner_key: "accepted-owner",
          conversation_id: "bound-one",
          request_scope: requestScope,
          validate_lease: () => true
        },
        view: {
          owner_key: "accepted-owner",
          conversation_id: "bound-one"
        } as HistoryLoadReceipt["view"]
      }
      useActorStore.getState().reset()
      let current = true
      io.set.mockImplementationOnce(async (key: string, value: unknown) => {
        io.values.set(key, value)
        current = false
      })
      expect(
        await bindActorSceneAfterLoad(
          bound.read,
          receipt,
          () => true,
          () => current
        )
      ).toBe(false)
      expect(useActorStore.getState().applied).toBeNull()
      expect(io.values.get(sceneKey(bound.fresh))).toEqual(bound.record)
    })
    it("reads the original qualified send source without inheriting character defaults", async () => {
      useStoreMessageOption.getState().startNewChatDraft(false)
      useStoreMessageOption.setState({ historyId: null, serverChatId: null })
      const controller = {
        getCurrent: () => ({ status: "idle", owner: null, view: null }),
        draftAccount: { requestScope, isCurrent: () => true }
      } as unknown as HistorySelectionController
      const read = await readActorSceneForSend(controller, () => true)
      expect(read.originalRecord).toBeNull()
      expect(read.origin).toBe("default")
      expect(read.target.account?.requestScope).toEqual(requestScope)
      expect(
        io.get.mock.calls.every(([key]) => key.startsWith("actorSettings:owned:"))
      ).toBe(true)
    })
    it("refuses a send source before IO when its original backend account is unavailable", async () => {
      useStoreMessageOption.getState().startNewChatDraft(false)
      useStoreMessageOption.setState({ historyId: null, serverChatId: null })
      const controller = {
        getCurrent: () => ({ status: "idle", owner: null, view: null }),
        draftAccount: { requestScope, isCurrent: () => false }
      } as unknown as HistorySelectionController
      await expect(readActorSceneForSend(controller, () => true)).rejects.toThrow(
        "actor_settings_unavailable"
      )
      expect(io.get).not.toHaveBeenCalled()
    })
    it.each(["preparation", "account"] as const)(
      "does not acknowledge retirement during Applied publication (%s)",
      async (retired) => {
        const bound = await arrangeBind("local")
        const receipt: HistoryLoadReceipt = {
          owner: {
            kind: "local",
            owner_key: "accepted-owner",
            conversation_id: "bound-one",
            profile_id: "profile-a"
          },
          view: {
            owner_key: "accepted-owner",
            conversation_id: "bound-one"
          } as HistoryLoadReceipt["view"]
        }
        useActorStore.getState().reset()
        let current = true
        const unsubscribe = useActorStore.subscribe((state) => {
          if (state.applied) {
            if (retired === "preparation") current = false
            else bound.retireAccount()
          }
        })
        try {
          expect(
            await bindActorSceneAfterLoad(
              bound.read,
              receipt,
              () => true,
              () => current
            )
          ).toBe(false)
          expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
          expect(io.values.get(sceneKey(bound.destination))).toEqual(bound.record)
        } finally {
          unsubscribe()
        }
      }
    )
    it.each(["absent", "retired"] as const)(
      "reads a genuine-local scene with %s optional backend verification",
      async (account) => {
        useStoreMessageOption.getState().startNewChatDraft(false)
        useStoreMessageOption.setState({ historyId: null, serverChatId: null })
        const receipt: HistoryLoadReceipt = {
          owner: {
            kind: "local",
            owner_key: "local-owner",
            conversation_id: "owned-local",
            profile_id: "profile-a"
          },
          view: {
            owner_key: "local-owner",
            conversation_id: "owned-local"
          } as HistoryLoadReceipt["view"]
        }
        expect(captureChatDraftLoad("switch")(receipt)).toBe(true)
        useStoreMessageOption.setState({
          historyId: "owned-local",
          serverChatId: null
        })
        const controller = {
          getCurrent: () => ({ status: "ready", ...receipt }),
          ...(account === "retired"
            ? { draftAccount: { requestScope, isCurrent: () => false } }
            : {})
        } as unknown as HistorySelectionController
        expect((await readActorSceneForSend(controller)).origin).toBe("default")
        expect(io.verifyProfile).toHaveBeenCalledWith("profile-a")
      }
    )
    it.each([
      [{ type: "global" }, { type: "workspace", workspaceId: "workspace-b" }],
      [{ type: "workspace", workspaceId: "workspace-a" }, { type: "global" }],
      [
        { type: "workspace", workspaceId: "workspace-a" },
        { type: "workspace", workspaceId: "workspace-b" }
      ]
    ] as [ChatScope, ChatScope][])(
      "refuses a qualified native destination in another scope (%j to %j)",
      async (sourceScope, destinationScope) => {
        const bound = await arrangeBind("native", true, sourceScope)
        const destination = captureActorSettingsTarget({
          address: {
            kind: "native",
            id: "other-scope",
            ownerKey: "other-owner",
            scope: destinationScope
          },
          owner: {
            kind: "native",
            conversation_id: "other-scope",
            owner_key: "other-owner",
            request_scope: requestScope,
            scope: destinationScope,
            validate_lease: () => true
          },
          view: {
            owner_key: "other-owner",
            conversation_id: "other-scope"
          } as HistoryLoadReceipt["view"],
          requestScope,
          accountIsCurrent: () => true,
          isCurrent: () => true
        })!
        io.get.mockClear()
        expect(await copyActorSceneToBoundChat(bound.read, destination)).toBe(
          false
        )
        expect(io.get).not.toHaveBeenCalled()
        expect(io.set).not.toHaveBeenCalled()
      }
    )
    it("settles an exact prior copy without changing its raw record", async () => {
      const bound = await arrangeBind()
      io.values.set(sceneKey(bound.destination), structuredClone(bound.record))
      expect(await copyActorSceneToBoundChat(bound.read, bound.destination)).toBe(
        true
      )
      expect(io.values.get(sceneKey(bound.destination))).toEqual(bound.record)
      expect(io.set).toHaveBeenCalledWith(
        sceneKey(bound.destination),
        bound.record
      )
    })
    it.each(["source", "destination"] as const)(
      "preserves independently changed %s data instead of refreshing the frozen receipt",
      async (changed) => {
        const bound = await arrangeBind()
        const newer = { value: sceneB, updatedAt: 8 }
        const key = sceneKey(
          changed === "source" ? bound.fresh : bound.destination
        )
        io.values.set(key, newer)
        expect(
          await copyActorSceneToBoundChat(bound.read, bound.destination)
        ).toBe(false)
        expect(io.values.get(key)).toEqual(newer)
        expect(io.set).not.toHaveBeenCalled()
      }
    )
    it.each(["account", "binding"] as const)(
      "refuses a retired original %s before copy IO",
      async (retired) => {
        const bound = await arrangeBind()
        if (retired === "account") bound.retireAccount()
        else bound.retireBinding()
        io.get.mockClear()
        expect(
          await copyActorSceneToBoundChat(bound.read, bound.destination)
        ).toBe(false)
        expect(io.get).not.toHaveBeenCalled()
        expect(io.set).not.toHaveBeenCalled()
      }
    )
    it("refuses a non-fresh source even when its scene is saved", async () => {
      const read = await getActorSceneForChat({ target: native })
      expect(await copyActorSceneToBoundChat(read, native)).toBe(false)
      expect(io.set).not.toHaveBeenCalled()
    })
    it("does not persist inherited defaults as an applied destination scene", async () => {
      const bound = await arrangeBind("native", false)
      const inherited = {
        ...bound.read,
        origin: "profile" as const,
        settings: sceneA
      }
      expect(await copyActorSceneToBoundChat(inherited, bound.destination)).toBe(
        false
      )
      expect(io.set).not.toHaveBeenCalled()
      expect(io.values.has(sceneKey(bound.destination))).toBe(false)
    })
    it("requires the same existing cross-context source lock for a saved copy", async () => {
      const bound = await arrangeBind()
      vi.stubGlobal(
        "navigator",
        Object.create(window.navigator, { locks: { value: undefined } })
      )
      expect(await copyActorSceneToBoundChat(bound.read, bound.destination)).toBe(
        false
      )
      expect(io.set).not.toHaveBeenCalled()
    })
    it("holds the original source lock through destination CAS", async () => {
      const bound = await arrangeBind()
      let release!: () => void
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      io.set.mockImplementationOnce(async (key: string, value: unknown) => {
        await held
        io.values.set(key, value)
      })
      const copying = copyActorSceneToBoundChat(bound.read, bound.destination)
      await vi.waitFor(() => expect(io.set).toHaveBeenCalledOnce())
      let replacementSettled = false
      const replacing = saveActorSettingsForChat({
        historyId: null,
        serverChatId: null,
        target: { ...bound.fresh, isCurrent: () => true },
        expectedRecord: bound.record,
        settings: sceneB
      }).then((result) => {
        replacementSettled = true
        return result
      })
      await Promise.resolve()
      expect(replacementSettled).toBe(false)
      release()
      expect(await copying).toBe(true)
      expect(await replacing).toBe(true)
      expect(io.values.get(sceneKey(bound.destination))).toEqual(bound.record)
    })
    it("does not acknowledge a copy whose accepted destination retires during its write", async () => {
      const bound = await arrangeBind()
      let release!: () => void
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      io.set.mockImplementationOnce(async (key: string, value: unknown) => {
        await held
        io.values.set(key, value)
      })
      const copying = copyActorSceneToBoundChat(bound.read, bound.destination)
      await vi.waitFor(() => expect(io.set).toHaveBeenCalledOnce())
      bound.retireBinding()
      release()
      expect(await copying).toBe(false)
      expect(io.values.get(sceneKey(bound.fresh))).toEqual(bound.record)
    })
  })

  describe("original scene command capture before asynchronous preparation", () => {
    const requestScope = { config: { serverUrl: "https://original-command.example", authMode: "single-user" as const }, userId: 1 }
    const arrangeCommand = () => {
      useStoreMessageOption.getState().startNewChatDraft()
      useStoreMessageOption.setState({ historyId: null, serverChatId: null })
      let accountCurrent = true, epoch = 1, commandCurrent = true
      const controller = { getCurrent: () => ({ status: "idle", owner: null, view: null }),
        fence: () => { const original = epoch; return () => epoch === original },
        draftAccount: { requestScope, isCurrent: () => accountCurrent } } as unknown as HistorySelectionController
      return { controller, capture: () => captureActorSceneSend(controller, () => commandCurrent),
        retireAccount: () => { accountCurrent = false }, reload: () => { epoch++ },
        retireCommand: () => { commandCurrent = false } }
    }
    it.each(["New", "account", "H1 epoch", "command"] as const)("refuses the originally captured %s retirement without reading a replacement", async changed => {
      const original = arrangeCommand(), captured = original.capture()!
      expect(captured.isCurrent()).toBe(true)
      if (changed === "New") useStoreMessageOption.getState().startNewChatDraft()
      if (changed === "account") {
        original.retireAccount()
        original.controller.draftAccount = { requestScope: { ...requestScope, userId: 2 }, isCurrent: () => true, retryVerification: () => {} }
      }
      if (changed === "H1 epoch") original.reload()
      if (changed === "command") original.retireCommand()
      expect(captured.isCurrent()).toBe(false)
      await expect(captured.read()).rejects.toThrow("actor_settings_scope_changed")
      expect(io.get).not.toHaveBeenCalled()
    })
    it("retains a cloned original backend scope and one saved scene receipt through later preparation", async () => {
      const original = arrangeCommand()
      const target = captureActorSettingsTarget({ address: useStoreMessageOption.getState().chatDraftAddress,
        requestScope, accountIsCurrent: () => true, isCurrent: () => true })!
      const record = { value: sceneA, updatedAt: 11 }
      io.values.set(sceneKey(target), record)
      const captured = original.capture()!
      const send = await captured.read()
      original.controller.draftAccount = { requestScope: { ...requestScope, userId: 2 }, isCurrent: () => true, retryVerification: () => {} }
      io.values.set(sceneKey(target), { value: sceneB, updatedAt: 12 })
      expect(send.requestScope).toEqual(requestScope)
      expect(send.requestScope).not.toBe(requestScope)
      expect(send.scene.originalRecord).toEqual(record)
      expect(send.scene.settings).toEqual(sceneA)
      expect(send.isCurrent()).toBe(true)
      expect(io.get).toHaveBeenCalledOnce()
    })
    it("uses accepted native transport ownership despite an unavailable unrelated fresh-account verifier", async () => {
      const original = arrangeCommand()
      const receipt: HistoryLoadReceipt = { owner: { kind: "native", conversation_id: "native-command", owner_key: "native-owner",
        request_scope: requestScope, validate_lease: () => true },
        view: { conversation_id: "native-command", owner_key: "native-owner" } as HistoryLoadReceipt["view"] }
      expect(captureChatDraftLoad("switch")(receipt)).toBe(true)
      useStoreMessageOption.setState({ historyId: null, serverChatId: "native-command" })
      original.controller.getCurrent = () => ({ status: "ready", ...receipt }) as ReturnType<HistorySelectionController["getCurrent"]>
      original.retireAccount()
      expect((await original.capture()!.read()).requestScope).toEqual(requestScope)
    })
  })

  it("isolates the same character card's defaults by captured account", async () => {
    expect(
      await saveActorProfileForCharacter({
        characterId: "card-one",
        target: profileA,
        settings: sceneA
      })
    ).toBe(true)
    expect(
      await saveActorProfileForCharacter({
        characterId: "card-one",
        target: profileB,
        settings: sceneB
      })
    ).toBe(true)
    const first = { throwOnError: true, target: profileA },
      second = { throwOnError: true, target: profileB }
    expect(await getActorProfileForCharacter("card-one", first)).toEqual(sceneA)
    expect(await getActorProfileForCharacter("card-one", second)).toEqual(
      sceneB
    )
  })

  it("retains an ambiguous bare character default without calling it this account's default", async () => {
    const legacyKey = "actorProfile:card-one"
    io.values.set(legacyKey, sceneA)
    const options = { throwOnError: true, target: profileB }
    expect(await getActorProfileForCharacter("card-one", options)).toBeNull()
    expect(io.values.get(legacyKey)).toEqual(sceneA)
    expect(io.get).not.toHaveBeenCalledWith(legacyKey)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("refuses a scene Save without an original record or certified absence", async () => {
    expect(
      await saveActorSettingsForChat({ ...source, settings: sceneA })
    ).toBe(false)
    expect(io.get).not.toHaveBeenCalled()
    expect(io.set).not.toHaveBeenCalled()
  })

  it("preserves a competing scene instead of refreshing the original absence at Save time", async () => {
    expect(
      await saveActorSettingsForChat({
        ...source,
        settings: sceneB,
        expectedRecord: null
      })
    ).toBe(true)
    const before = structuredClone(io.values.get(sceneKey(native)))
    io.set.mockClear()
    expect(
      await saveActorSettingsForChat({
        ...source,
        settings: sceneA,
        expectedRecord: null
      })
    ).toBe(false)
    expect(io.values.get(sceneKey(native))).toEqual(before)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("preserves a newer scene when the original hydrated record has changed", async () => {
    const original = { value: sceneA, updatedAt: 1 }
    const newer = { value: sceneB, updatedAt: 2 }
    io.values.set(sceneKey(native), newer)
    expect(
      await saveActorSettingsForChat({
        ...source,
        settings: sceneA,
        expectedRecord: original
      })
    ).toBe(false)
    expect(io.values.get(sceneKey(native))).toEqual(newer)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("writes a replacement timestamp above the original even when the clock moves backward", async () => {
    const original = { value: sceneA, updatedAt: 100 }
    io.values.set(sceneKey(native), original)
    vi.spyOn(Date, "now").mockReturnValue(1)
    expect(
      await saveActorSettingsForChat({
        ...source,
        settings: sceneB,
        expectedRecord: original
      })
    ).toBe(true)
    expect(io.values.get(sceneKey(native))).toEqual({
      value: sceneB,
      updatedAt: 101
    })
  })

  it.each(["read", "save"])(
    "refuses a retired scene target before %s IO",
    async (kind) => {
      const retired = {
        ...source,
        target: { ...native, isCurrent: () => false }
      }
      if (kind === "read")
        await expect(getActorSettingsForChat(retired)).rejects.toThrow(
          "actor_settings_unavailable"
        )
      else
        expect(
          await saveActorSettingsForChat({
            ...retired,
            settings: sceneA,
            expectedRecord: null
          })
        ).toBe(false)
      expect(io.get).not.toHaveBeenCalled()
      expect(io.set).not.toHaveBeenCalled()
    }
  )

  it("refuses a held scene read after its original qualified target retires", async () => {
    let current = true,
      release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    io.get.mockImplementationOnce(async () => {
      await held
      return undefined
    })
    const pending = getActorSettingsForChat({
      ...source,
      target: { ...native, isCurrent: () => current }
    })
    await vi.waitFor(() => expect(io.get).toHaveBeenCalledOnce())
    current = false
    release()
    await expect(pending).rejects.toThrow("actor_settings_unavailable")
    expect(io.set).not.toHaveBeenCalled()
  })

  it("refuses conditional scene Save without cross-context locks", async () => {
    vi.stubGlobal(
      "navigator",
      Object.create(window.navigator, { locks: { value: undefined } })
    )
    expect(
      await saveActorSettingsForChat({
        ...source,
        settings: sceneA,
        expectedRecord: null
      })
    ).toBe(false)
    expect(io.set).not.toHaveBeenCalled()
  })

  it.each([false, { ...sceneA, version: 0 }])(
    "preserves an invalid qualified scene without certifying neutral defaults (%j)",
    async (value) => {
      const stored = { value, updatedAt: 1 }
      io.values.set(sceneKey(native), stored)
      await expect(getActorSettingsForChat(source)).rejects.toThrow(
        "actor_settings_unavailable"
      )
      expect(io.values.get(sceneKey(native))).toEqual(stored)
      expect(io.set).not.toHaveBeenCalled()
    }
  )

  it("retains raw scene metadata separately from the normalized editor proposal", async () => {
    const raw = {
      ...sceneA,
      appendable: undefined,
      aspects: [{ ...sceneA.aspects[0], key: " User Role " }]
    }
    const original = { value: raw, updatedAt: 1 }
    io.values.set(sceneKey(native), original)
    const read = await getActorSceneForChat({ target: native })
    expect(read.originalRecord).toEqual(original)
    expect(read.settings.appendable).toBe(false)
    expect(read.settings.aspects[0].key).not.toBe(raw.aspects[0].key)
    read.settings.notes = "Only staged here."
    expect(io.values.get(sceneKey(native))).toEqual(original)
  })
  it("keeps original scene absence as Save expectation when inheriting a profile", async () => {
    io.values.set(`actorProfile:owned:${profileA.key}`, {
      value: sceneA,
      updatedAt: 1
    })
    const read = await getActorSceneForChat({
      target: native,
      characterId: "card-one"
    })
    expect(read.origin).toBe("profile")
    expect(read.originalRecord).toBeNull()
    expect(read.settings).toEqual(sceneA)
    expect(
      await saveActorSettingsForChat({
        ...source,
        settings: sceneB,
        expectedRecord: read.originalRecord
      })
    ).toBe(true)
    expect(io.values.get(`actorProfile:owned:${profileA.key}`)).toEqual({
      value: sceneA,
      updatedAt: 1
    })
  })
  it("keeps certified genuine-local absence editable with unverified optional defaults", async () => {
    const local = target(["local", "profile-a", "collision"], null)
    const read = await getActorSceneForChat({
      target: local,
      characterId: "card-one"
    })
    expect(read.originalRecord).toBeNull()
    expect(read.profileUnavailable).toBe(true)
    expect(read.origin).toBe("default")
    expect(
      await saveActorSettingsForChat({
        historyId: "collision",
        serverChatId: null,
        target: local,
        settings: sceneA,
        expectedRecord: read.originalRecord
      })
    ).toBe(true)
    expect(io.get).not.toHaveBeenCalledWith(
      `actorProfile:owned:${profileA.key}`
    )
  })
  it("reports unavailable optional defaults separately from certified absence", async () => {
    io.get.mockImplementation(async (key: string) => {
      if (key.startsWith("actorProfile:")) throw new Error("unavailable")
      return io.values.get(key)
    })
    const read = await getActorSceneForChat({
      target: native,
      characterId: "card-one"
    })
    expect(read.originalRecord).toBeNull()
    expect(read.profileUnavailable).toBe(true)
    expect(io.set).not.toHaveBeenCalled()
  })
  it("refuses a held read after editor retirement while the destination stays valid", async () => {
    let operationCurrent = true,
      release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    io.get.mockImplementationOnce(async () => {
      await held
      return undefined
    })
    const pending = getActorSceneForChat({
      target: native,
      validate_lease: () => operationCurrent
    })
    await vi.waitFor(() => expect(io.get).toHaveBeenCalledOnce())
    operationCurrent = false
    release()
    await expect(pending).rejects.toThrow("actor_settings_unavailable")
    expect(native.isCurrent()).toBe(true)
  })
  it("freezes the scene proposal before a held conditional write", async () => {
    let release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    io.get.mockImplementationOnce(async () => {
      await held
      return undefined
    })
    const proposal = structuredClone(sceneA)
    const pending = saveActorSettingsForChat({
      ...source,
      settings: proposal,
      expectedRecord: null
    })
    await vi.waitFor(() => expect(io.get).toHaveBeenCalledOnce())
    proposal.notes = "Changed while pending."
    release()
    expect(await pending).toBe(true)
    expect(io.values.get(sceneKey(native))).toMatchObject({
      value: { notes: sceneA.notes }
    })
  })
  it("refuses success after a held write loses its original destination", async () => {
    let current = true,
      release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    io.set.mockImplementationOnce(async (key: string, value: unknown) => {
      io.values.set(key, value)
      await held
    })
    const pending = saveActorSettingsForChat({
      ...source,
      target: { ...native, isCurrent: () => current },
      settings: sceneA,
      expectedRecord: null
    })
    await vi.waitFor(() => expect(io.set).toHaveBeenCalledOnce())
    current = false
    release()
    expect(await pending).toBe(false)
    expect(io.values.get(sceneKey(native))).toMatchObject({
      value: { notes: sceneA.notes }
    })
  })
  it("retains an invalid qualified character default instead of replacing it on explicit Save", async () => {
    const raw = { value: { ...sceneA, version: 0 }, updatedAt: 1 }
    io.values.set(`actorProfile:owned:${profileA.key}`, raw)
    expect(
      await saveActorProfileForCharacter({
        characterId: "card-one",
        target: profileA,
        settings: sceneB
      })
    ).toBe(false)
    expect(io.values.get(`actorProfile:owned:${profileA.key}`)).toEqual(raw)
    expect(io.set).not.toHaveBeenCalled()
  })

  it.each(["scene", "profile"])(
    "keeps qualified %s Save independent of the unrelated legacy storage area",
    async (kind) => {
      io.legacyPersistent = false
      const saved =
        kind === "scene"
          ? await saveActorSettingsForChat({
              ...source,
              settings: sceneA,
              expectedRecord: null
            })
          : await saveActorProfileForCharacter({
              characterId: "card-one",
              target: profileA,
              settings: sceneA
            })
      expect(saved).toBe(true)
    }
  )

  it.each([
    { label: "false", original: false },
    { label: "zero", original: 0 },
    { label: "empty string", original: "" },
    { label: "empty array", original: [] },
    {
      label: "array record",
      original: Object.assign([], { value: sceneA, updatedAt: 1 })
    },
    {
      label: "invalid date timestamp",
      original: { value: sceneA, updatedAt: Number.MAX_VALUE }
    }
  ])(
    "refuses present $label as a scene CAS expectation before IO",
    async ({ original }) => {
      if (original && typeof original === "object")
        io.values.set(sceneKey(native), original)
      const before = io.values.get(sceneKey(native))
      // Runtime/storage boundary inputs must fail closed despite the static record type.
      const params = {
        ...source,
        settings: sceneB,
        expectedRecord: original as never
      }
      expect(await saveActorSettingsForChat(params)).toBe(false)
      expect(io.values.get(sceneKey(native))).toEqual(before)
      expect(io.get).not.toHaveBeenCalled()
      expect(io.set).not.toHaveBeenCalled()
    }
  )

  it.each(["unavailable", "malformed", "retired"])(
    "fails a qualified profile read closed without an extra strict flag (%s)",
    async (kind) => {
      const options = {
        target:
          kind === "retired"
            ? { ...profileA, isCurrent: () => false }
            : profileA
      }
      if (kind === "unavailable")
        io.get.mockRejectedValueOnce(new Error("unavailable"))
      if (kind === "malformed")
        io.values.set(`actorProfile:owned:${profileA.key}`, {
          value: false,
          updatedAt: 1
        })
      await expect(
        getActorProfileForCharacter("card-one", options)
      ).rejects.toThrow("actor_settings_unavailable")
      expect(io.set).not.toHaveBeenCalled()
    }
  )

  it("refuses an array wrapper masquerading as an owned scene record", async () => {
    const stored = Object.assign([], { value: sceneA, updatedAt: 1 })
    io.values.set(sceneKey(native), stored)
    await expect(getActorSettingsForChat(source)).rejects.toThrow(
      "actor_settings_unavailable"
    )
    expect(io.values.get(sceneKey(native))).toEqual(stored)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("keeps the settings-only fallback wrapper from hiding unavailable inherited defaults", async () => {
    io.get.mockImplementation(async (key: string) => {
      if (key.startsWith("actorProfile:")) throw new Error("unavailable")
      return io.values.get(key)
    })
    await expect(
      getActorSettingsForChatWithCharacterFallback({
        ...source,
        characterId: "card-one"
      })
    ).rejects.toThrow("actor_settings_unavailable")
    expect(io.set).not.toHaveBeenCalled()
  })
  it.each(["scene", "profile"])(
    "preserves a %s record when its monotonic replacement would exceed the valid Date range",
    async (kind) => {
      const original = { value: sceneA, updatedAt: 8_640_000_000_000_000 }
      const key =
        kind === "scene"
          ? sceneKey(native)
          : `actorProfile:owned:${profileA.key}`
      io.values.set(key, original)
      const saved =
        kind === "scene"
          ? await saveActorSettingsForChat({
              ...source,
              settings: sceneB,
              expectedRecord: original
            })
          : await saveActorProfileForCharacter({
              characterId: "card-one",
              target: profileA,
              settings: sceneB
            })
      expect(saved).toBe(false)
      expect(io.values.get(key)).toEqual(original)
      expect(io.set).not.toHaveBeenCalled()
    }
  )
  it.each(["present", "absent", "save"] as const)(
    "refuses a profile replaced during held %s scene IO before acknowledgement",
    async (kind) => {
      const local = {
        ...target(["local", "profile-a", "collision"], null),
        localProfileId: "profile-a"
      }
      const key = sceneKey(local),
        raw = { value: sceneA, updatedAt: 1 }
      if (kind !== "absent") io.values.set(key, raw)
      let release!: () => void
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      let profileCurrent = true
      io.verifyProfile.mockImplementation(async () => {
        if (!profileCurrent) throw new Error("owner_mismatch")
      })
      if (kind === "save")
        io.set.mockImplementationOnce(async (address: string, value: unknown) => {
          await held
          io.values.set(address, value)
        })
      else
        io.get.mockImplementationOnce(async () => {
          const observed = io.values.get(key)
          await held
          return observed
        })
      const result =
        kind === "save"
          ? saveActorSettingsForChat({
              historyId: "collision",
              serverChatId: null,
              target: local,
              expectedRecord: raw,
              settings: sceneB
            })
          : getActorSceneForChat({ target: local })
      const rejected =
        kind === "save"
          ? expect(result).resolves.toBe(false)
          : expect(result).rejects.toThrow("actor_settings_unavailable")
      await vi.waitFor(() =>
        expect(kind === "save" ? io.set : io.get).toHaveBeenCalled()
      )
      profileCurrent = false
      release()
      await rejected
      // A refused acknowledgement is not rollback of the owned write that already completed.
      if (kind === "save")
        expect(io.values.get(key)).toMatchObject({ value: sceneB })
    }
  )

  it.each(["read", "save"])(
    "verifies the original genuine-local profile before %s scene IO",
    async (kind) => {
      const local = {
        ...target(["local", "profile-a", "collision"], null),
        localProfileId: "profile-a"
      }
      const params = { historyId: "collision", serverChatId: null, target: local }
      if (kind === "read") await getActorSettingsForChat(params)
      else
        expect(
          await saveActorSettingsForChat({
            ...params,
            settings: sceneA,
            expectedRecord: null
          })
        ).toBe(true)
      expect(io.verifyProfile).toHaveBeenCalledWith("profile-a")
      expect(io.verifyProfile).not.toHaveBeenCalledWith()
    }
  )
  it.each(["read", "save"])(
    "refuses a replaced persistent local profile before %s scene IO",
    async (kind) => {
      const local = {
        ...target(["local", "profile-b", "collision"], null),
        localProfileId: "profile-b"
      }
      const params = { historyId: "collision", serverChatId: null, target: local }
      if (kind === "read")
        await expect(getActorSettingsForChat(params)).rejects.toThrow(
          "actor_settings_unavailable"
        )
      else
        expect(
          await saveActorSettingsForChat({
            ...params,
            settings: sceneA,
            expectedRecord: null
          })
        ).toBe(false)
      expect(io.get).not.toHaveBeenCalled()
      expect(io.set).not.toHaveBeenCalled()
    }
  )

  describe("missing qualified targets refuse every public storage wrapper", () => {
    const unqualifiedSources = [
      { label: "native ID", historyId: null, serverChatId: "collision" },
      { label: "local ID", historyId: "collision", serverChatId: null },
      { label: "scratch IDs", historyId: null, serverChatId: null }
    ]
    const readCases = unqualifiedSources.flatMap((source) =>
      ["rich Scene", "plain Scene", "Scene profile fallback"].map((kind) => ({
        ...source,
        kind
      }))
    )
    beforeEach(() => {
      // Valid legacy data must not make an unchecked caller qualified. Include
      // owned records too: refusal must precede any physical record discovery.
      for (const key of [
        "actorSettings:server:collision",
        "actorSettings:local:collision",
        "actorSettings:scratch",
        "actorProfile:card-one"
      ])
        io.values.set(key, structuredClone(sceneA))
      io.values.set(sceneKey(native), { value: sceneA, updatedAt: 11 })
      io.values.set(`actorProfile:owned:${profileA.key}`, {
        value: sceneA,
        updatedAt: 11
      })
    })

    it.each(readCases)(
      "refuses missing target $kind with $label before any storage IO",
      async ({ kind, historyId, serverChatId }) => {
        const before = structuredClone([...io.values])
        const params = { historyId, serverChatId }
        const pending =
          kind === "rich Scene"
            ? // Exercise an unchecked runtime caller without fabricating a target
              // or weakening the rich reader's required static parameter type.
              Reflect.apply(getActorSceneForChat, undefined, [params])
            : kind === "plain Scene"
              ? getActorSettingsForChat(params)
              : getActorSettingsForChatWithCharacterFallback({
                  ...params,
                  characterId: "card-one"
                })
        await expect(pending).rejects.toThrow("actor_settings_unavailable")
        expect([...io.values]).toEqual(before)
        expect(io.get).not.toHaveBeenCalled()
        expect(io.set).not.toHaveBeenCalled()
        expect(io.verifyProfile).not.toHaveBeenCalled()
      }
    )

    it.each(
      unqualifiedSources.flatMap((source) =>
        ["certified absence", "original raw record"].map((expectation) => ({
          ...source,
          expectation
        }))
      )
    )(
      "refuses missing target Scene Save with $label and otherwise valid $expectation",
      async ({ historyId, serverChatId, expectation }) => {
        const before = structuredClone([...io.values])
        const original = { value: structuredClone(sceneA), updatedAt: 11 }
        expect(
          await saveActorSettingsForChat({
            historyId,
            serverChatId,
            settings: sceneB,
            expectedRecord: expectation === "certified absence" ? null : original,
            validate_lease: () => true
          })
        ).toBe(false)
        expect(original).toEqual({ value: sceneA, updatedAt: 11 })
        expect([...io.values]).toEqual(before)
        expect(io.get).not.toHaveBeenCalled()
        expect(io.set).not.toHaveBeenCalled()
        expect(io.verifyProfile).not.toHaveBeenCalled()
      }
    )

    it.each(["advisory", "strict"])(
      "refuses a missing target %s Profile read before any storage IO",
      async (kind) => {
        const before = structuredClone([...io.values])
        await expect(
          getActorProfileForCharacter(
            "card-one",
            kind === "strict"
              ? { throwOnError: true, validate_lease: () => true }
              : undefined
          )
        ).rejects.toThrow("actor_settings_unavailable")
        expect([...io.values]).toEqual(before)
        expect(io.get).not.toHaveBeenCalled()
        expect(io.set).not.toHaveBeenCalled()
        expect(io.verifyProfile).not.toHaveBeenCalled()
      }
    )

    it("refuses missing target Profile Save with an otherwise valid proposal and lease", async () => {
      const before = structuredClone([...io.values])
      expect(
        await saveActorProfileForCharacter({
          characterId: "card-one",
          settings: sceneB,
          validate_lease: () => true
        })
      ).toBe(false)
      expect([...io.values]).toEqual(before)
      expect(io.get).not.toHaveBeenCalled()
      expect(io.set).not.toHaveBeenCalled()
      expect(io.verifyProfile).not.toHaveBeenCalled()
    })
  })

})
