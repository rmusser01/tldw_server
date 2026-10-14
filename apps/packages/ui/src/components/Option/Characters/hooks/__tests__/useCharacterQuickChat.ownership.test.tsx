import type { TFunction } from "i18next"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useCharacterQuickChat } from "../useCharacterQuickChat"
import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessageOption } from "@/store/option"
import { useActorStore } from "@/store/actor"
import { createDefaultActorSettings } from "@/types/actor"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"

const mocks = vi.hoisted(() => ({
  create: vi.fn(), complete: vi.fn(), deleteChat: vi.fn(), nativeSend: vi.fn(),
  navigate: vi.fn(), selectCharacter: vi.fn(), retain: vi.fn(),
  ownerGeneration: 0, ownerReady: true, retryOwner: vi.fn(), getCharacter: vi.fn(), preparePath: vi.fn(), requestScope: { config: { serverUrl: "https://private-scene-command.example", authMode: "single-user" as const }, userId: 1 }, scene: vi.fn<typeof import("@/services/actor-settings").getActorSceneForChat>(), accountClear: undefined as undefined | (() => void),
  selection: null as null | Record<string, unknown>, reset: vi.fn(), getReference: vi.fn(), copy: vi.fn(), requestNavigation: vi.fn(),
}))
vi.mock("react-router-dom", () => ({ useNavigate: () => mocks.navigate }))
vi.mock("@/hooks/useSelectedCharacter", () => ({ useSelectedCharacter: () => [null, mocks.selectCharacter] }))
vi.mock("@/hooks/useComposerFocus", () => ({ focusComposer: vi.fn() }))
vi.mock("@/services/actor-settings", () => ({ getActorSceneForChat: mocks.scene }))
vi.mock("@/hooks/useChatDraftOwner", async original => ({
  ...await original<typeof import("@/hooks/useChatDraftOwner")>(),
  useChatDraftOwner: (clear: () => void) => {
    mocks.accountClear = clear
    const generation = mocks.ownerGeneration
    return { ownerKey: mocks.ownerReady ? chatDraftOwnerKey(mocks.requestScope) : null, requestScope: mocks.ownerReady ? mocks.requestScope : null, serverScopeKey: "owner-a",
      isCurrent: () => mocks.ownerReady && generation === mocks.ownerGeneration, verificationState: mocks.ownerReady ? "ready" : "pending", retryVerification: mocks.retryOwner }
  }
}))
vi.mock("@/services/pending-chat-promotion", () => ({ listChatCreationRecovery: vi.fn(async () => ({ attempts: [], unavailable: false })), readChatCreationRecovery: vi.fn(async () => null) }))
vi.mock("@/db/dexie/history-selection", async original => ({ ...await original<typeof import("@/db/dexie/history-selection")>(), loadHistoryTurnRecoveries: vi.fn(async () => []) }))
vi.mock("@/utils/settings-return", () => ({ requestSettingsNavigation: mocks.requestNavigation }))
vi.mock("@/hooks/useDraftPersistence", () => ({ useDraftPersistence: () => ({
  draftReady: true, draftSaved: true, retainDraft: mocks.retain, flushDraft: vi.fn(), clearDraft: vi.fn(), listDrafts: vi.fn()
}) }))
vi.mock("@/hooks/chat/useHistorySelection", () => ({ useHistorySelection: () => ({
  reset: mocks.reset, getReference: mocks.getReference,
  getCurrent: () => mocks.selection ?? ({ status: "idle", owner: null, view: null }),
  getSignal: () => undefined, fence: () => () => true, prepareExpansionPath: mocks.preparePath, loadConversation: vi.fn(), recoveries: [], recoveryError: null
}) }))
vi.mock("@/hooks/chat/native-history-character-send", () => ({ sendNativeHistoryCharacter: mocks.nativeSend }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  createChat: mocks.create, completeCharacterChatTurn: mocks.complete, deleteChat: mocks.deleteChat,
  getCharacter: mocks.getCharacter, getChat: vi.fn(async (id: string) => ({ id, character_id: 4 }))
} }))

const t = (key: string, options?: Record<string, unknown> | string) => typeof options === "string" ? options : String(options?.defaultValue ?? key)
const card = (id: number) => ({ id, name: `Guide ${id}`, greeting: "Preview opening" })
const deps = { t: t as TFunction, activeQuickChatModel: "openai:fixture-model", isServerConnected: true,
  availableModels: [{ model: "fixture-model", provider: "openai", is_configured: true }] }
const held = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done }); return { promise, resolve } }

beforeEach(() => {
  vi.clearAllMocks()
  useStoreChatModelSettings.getState().reset()
  mocks.ownerGeneration = 0
  mocks.selection = null
  mocks.reset.mockImplementation(() => { mocks.selection = null })
  useStoreMessageOption.getState().startNewChatDraft()
  useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  useActorStore.getState().reset()
  mocks.scene.mockReset().mockImplementation(async ({ target }) => ({ target: target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false }))
  mocks.ownerReady = true
  mocks.getCharacter.mockReset().mockImplementation(async (id: string) => ({ id, name: "Guide" }))
  mocks.preparePath.mockResolvedValue("/chat?chatId=quick-session&client_session_id=qualified-popup-view")
  mocks.accountClear = undefined
  mocks.copy.mockResolvedValue(undefined)
  mocks.requestNavigation.mockReturnValue(true)
  Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: mocks.copy } })
  mocks.retain.mockResolvedValue(true)
  mocks.create.mockResolvedValue({ id: "quick-session" })
  mocks.complete.mockResolvedValue({ assistant_content: "Owned reply", saved: true })
  mocks.nativeSend.mockImplementation(async (params) => {
    try {
      const created = await mocks.create()
      mocks.selection = { status: "ready", owner: { kind: "native", conversation_id: created.id, owner_key: "popup-owner", request_scope: mocks.requestScope, validate_lease: () => true },
        view: { owner_key: "popup-owner", conversation_id: created.id }, capture: { status: "captured" } }
      params.setServerChatId(created.id)
      const response = await mocks.complete()
      params.setMessages([{ id: "assistant-result", isBot: true, name: "Assistant", message: response.assistant_content, createdAt: 1, sources: [] }])
    } finally { params.releaseActivity() }
  })
})

const start = async (result: ReturnType<typeof renderHook<ReturnType<typeof useCharacterQuickChat>, unknown>>["result"], id = 4) => {
  await act(async () => { await result.current.openQuickChat(card(id)) })
  act(() => result.current.setQuickChatDraft("  exact draft\ntrailing spaces  "))
}

describe("Character Quick Chat owned lifetime", () => {
  it.each(["current", "retired before review", "retired during review"])("shows guarded roleplay provider guidance for a %s popup failure", async boundary => {
    const { NativeHistoryRoleplaySendError } = await import("@/hooks/chat/native-character-preflight")
    const { loadHistoryTurnRecoveries } = await import("@/db/dexie/history-selection")
    const turns = held<Awaited<ReturnType<typeof loadHistoryTurnRecoveries>>>()
    let sourceCurrent = boundary !== "retired before review"
    mocks.nativeSend.mockImplementationOnce(async params => {
      mocks.selection = { status: "ready", bookmarkScope: { profile_id: "p", client_session_id: "client" },
        view: { conversation_id: "quick-session", owner_key: "popup-owner" }, owner: { kind: "native", conversation_id: "quick-session", validate_lease: () => true }, capture: { status: "captured" } }
      params.setServerChatId("quick-session")
      throw new NativeHistoryRoleplaySendError(Object.assign(new Error("HTTP 401"), { status: 401,
        details: { detail: { error: { code: "provider_authentication_failed", type: "provider_authentication_failed", message: "The selected provider credentials could not be authenticated." } } } }), () => sourceCurrent)
    })
    const hook = renderHook(() => useCharacterQuickChat(deps))
    await start(hook.result)
    vi.mocked(loadHistoryTurnRecoveries).mockClear().mockImplementationOnce(() => turns.promise)
    let sending!: Promise<void>
    try {
      act(() => { sending = hook.result.current.sendQuickChatMessage() })
      await waitFor(() => expect(mocks.nativeSend).toHaveBeenCalledOnce())
      if (boundary !== "retired before review") {
        await waitFor(() => expect(loadHistoryTurnRecoveries).toHaveBeenCalledOnce())
        if (boundary === "retired during review") sourceCurrent = false
      }
      await act(async () => { turns.resolve([]); await sending })
      expect(hook.result.current.quickChatDraft).toBe("  exact draft\ntrailing spaces  ")
      if (boundary === "current") expect(hook.result.current.quickChatError).toContain("Open model settings")
      else expect(hook.result.current.quickChatError).toBeNull()
      if (boundary === "retired before review") expect(loadHistoryTurnRecoveries).not.toHaveBeenCalled()
      expect(mocks.nativeSend).toHaveBeenCalledOnce()
      expect(mocks.deleteChat).not.toHaveBeenCalled()
    } finally { turns.resolve([]); await sending; hook.unmount(); vi.mocked(loadHistoryTurnRecoveries).mockReset().mockResolvedValue([]) }
  })

  it("freezes the popup's own scene before retention while main New remains irrelevant", async () => {
    const original = { ...createDefaultActorSettings(), notes: "Private original scene" }
    let stored = original
    mocks.scene.mockImplementation(async ({ target }) => ({ target: target!, settings: structuredClone(stored), originalRecord: { value: structuredClone(stored), updatedAt: 1 }, origin: "scene", profileUnavailable: false }))
    const pending = held<boolean>()
    mocks.retain.mockReturnValueOnce(pending.promise)
    const view = renderHook(() => useCharacterQuickChat(deps))
    await start(view.result)
    let send!: ReturnType<typeof view.result.current.sendQuickChatMessage>
    act(() => { send = view.result.current.sendQuickChatMessage() })
    await waitFor(() => expect(mocks.retain).toHaveBeenCalledOnce())
    const readsBeforeRetention = mocks.scene.mock.calls.length
    const mainAddress = useStoreMessageOption.getState().chatDraftAddress!
    act(() => { useStoreMessageOption.getState().startNewChatDraft() })
    stored = { ...original, notes: "Private newer scene" }
    await act(async () => { pending.resolve(true); await send })
    expect(readsBeforeRetention).toBe(1)
    expect(mocks.nativeSend).toHaveBeenCalledWith(expect.objectContaining({ actorSceneSend: expect.objectContaining({ requestScope: mocks.requestScope,
      scene: expect.objectContaining({ settings: original, target: expect.objectContaining({ address: expect.objectContaining({ kind: "fresh" }) }) }) }) }))
    expect(mocks.nativeSend.mock.calls[0][0].actorSceneSend.scene.target.address.id).not.toBe(mainAddress.kind === "fresh" ? mainAddress.id : null)
    expect(useActorStore.getState().applied).toBeNull()
  })

  it("reads a later popup turn from its accepted native owner instead of its retained creation handle", async () => {
    const view = renderHook(() => useCharacterQuickChat(deps))
    await start(view.result)
    await act(async () => { await view.result.current.sendQuickChatMessage() })
    act(() => view.result.current.setQuickChatDraft("Second private question"))
    await act(async () => { await view.result.current.sendQuickChatMessage() })
    expect(mocks.scene.mock.calls.map(([args]) => args.target!.address?.kind)).toEqual(["fresh", "native"])
    expect(mocks.scene.mock.calls[1][0].target?.address).toMatchObject({ kind: "native", id: "quick-session", ownerKey: "popup-owner" })
    expect(mocks.nativeSend.mock.calls[1][0].privateDraft.handle.address.kind).toBe("fresh")
  })

  it("shows pending account verification and retries the owned card without silently dropping the intent", async () => {
    mocks.ownerReady = false
    const { result, rerender } = renderHook(() => useCharacterQuickChat(deps))
    await act(async () => { await result.current.openQuickChat(card(4)) })
    expect(result.current.quickChatCharacter).toMatchObject({ id: 4 })
    expect(mocks.getCharacter).not.toHaveBeenCalled()
    await act(async () => { await result.current.retryQuickChat() })
    expect(mocks.retryOwner).toHaveBeenCalledOnce()
    mocks.ownerReady = true
    rerender()
    await waitFor(() => expect(result.current.quickChatCharacter?.name).toBe("Guide"))
    expect(mocks.getCharacter).toHaveBeenCalledWith("4", expect.objectContaining({ requestScope: mocks.requestScope }))
  })

  it("retries a failed scoped card read without submitting or losing owned text", async () => {
    mocks.getCharacter.mockRejectedValueOnce(new Error("offline"))
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await act(async () => { await result.current.openQuickChat(card(4)) })
    expect(result.current.quickChatReadiness.canStart).toBe(false)
    act(() => result.current.setQuickChatDraft("KEPT WHILE RETRYING  "))
    await act(async () => { await result.current.retryQuickChat() })
    expect(result.current.quickChatCharacter?.name).toBe("Guide")
    expect(result.current.quickChatDraft).toBe("KEPT WHILE RETRYING  ")
    expect(mocks.nativeSend).not.toHaveBeenCalled()
  })

  it("offers exact Copy while draft storage is unavailable and keeps the popup", async () => {
    mocks.retain.mockResolvedValue(false)
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    await act(async () => { await result.current.closeQuickChat() })
    await act(async () => { await result.current.copyQuickChatDraft() })
    expect(mocks.copy).toHaveBeenCalledWith("  exact draft\ntrailing spaces  ")
    expect(result.current.quickChatCharacter?.name).toBe("Guide")
    expect(result.current.quickChatDraft).toBe("  exact draft\ntrailing spaces  ")
    expect(mocks.navigate).not.toHaveBeenCalled()
  })

  it("continues a saved session using its qualified view after exact Copy", async () => {
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    await act(async () => { await result.current.sendQuickChatMessage() })
    act(() => result.current.setQuickChatDraft("NEXT UNSENT INPUT  "))
    await act(async () => { await result.current.handlePromoteQuickChat() })
    expect(mocks.copy).toHaveBeenCalledWith("NEXT UNSENT INPUT  ")
    expect(mocks.navigate).toHaveBeenCalledWith("/chat?chatId=quick-session&client_session_id=qualified-popup-view")
    expect(mocks.create).toHaveBeenCalledTimes(1)
    expect(mocks.selectCharacter).not.toHaveBeenCalled()
  })

  it("keeps zero-valued settings owned by an internal model alias", async () => {
    const store = useStoreChatModelSettings.getState()
    store.updateScopedSetting("openai:tldw:fixture-model", "temperature", 0)
    store.updateScopedSetting("openai:tldw:fixture-model", "topP", 0)
    store.setActiveSettingsScope("openai:tldw:fixture-model")
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    await act(async () => { await result.current.sendQuickChatMessage() })
    expect(mocks.nativeSend).toHaveBeenCalledWith(expect.objectContaining({ settings: expect.objectContaining({ temperature: 0, topP: 0 }) }))
    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe("openai:tldw:fixture-model")
  })

  it("blocks an unsupported setting owned by the selected alias before retention or dispatch", async () => {
    const store = useStoreChatModelSettings.getState()
    store.updateScopedSetting("openai:tldw:fixture-model", "numCtx", 4096)
    store.setActiveSettingsScope("openai:tldw:fixture-model")
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    await act(async () => { await result.current.sendQuickChatMessage() })
    expect(mocks.nativeSend).not.toHaveBeenCalled()
    expect(mocks.retain).not.toHaveBeenCalled()
    expect(result.current.quickChatDraft).toBe("  exact draft\ntrailing spaces  ")
  })

  it("retires settings changed while exact draft retention is held", async () => {
    const pending = held<boolean>()
    mocks.retain.mockReturnValueOnce(pending.promise)
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    let sending!: Promise<void>
    act(() => { sending = result.current.sendQuickChatMessage() })
    await waitFor(() => expect(mocks.retain).toHaveBeenCalled())
    act(() => useStoreChatModelSettings.getState().updateSetting("temperature", 0.9))
    await act(async () => { pending.resolve(true); await sending })
    expect(mocks.nativeSend).not.toHaveBeenCalled()
    expect(result.current.quickChatDraft).toBe("  exact draft\ntrailing spaces  ")
  })

  it("uses strict readiness with catalog availability rather than a bare provider guess", async () => {
    const { result } = renderHook(() => useCharacterQuickChat({ ...deps, activeQuickChatModel: "fixture-model" }))
    await start(result)
    expect(result.current.quickChatReadiness).toMatchObject({ canStart: false, reason: "explicit-model-provider-required" })
    await act(async () => { await result.current.sendQuickChatMessage() })
    expect(mocks.nativeSend).not.toHaveBeenCalled()
  })

  it("keeps a configured-false model blocked in the displayed and dispatched readiness", async () => {
    const { result } = renderHook(() => useCharacterQuickChat({ ...deps, availableModels: [{ model: "fixture-model", provider: "openai", is_configured: false }] }))
    await start(result)
    expect(result.current.quickChatReadiness.canStart).toBe(false)
    await act(async () => { await result.current.sendQuickChatMessage() })
    expect(mocks.nativeSend).not.toHaveBeenCalled()
  })

  it("clears only the acknowledged unchanged input after readiness disconnect and reconnect", async () => {
    const pending = held<{ assistant_content: string }>()
    mocks.complete.mockReturnValueOnce(pending.promise)
    const { result, rerender } = renderHook(props => useCharacterQuickChat(props), { initialProps: deps })
    await start(result)
    let operation!: Promise<void>
    act(() => { operation = result.current.sendQuickChatMessage() })
    await waitFor(() => expect(mocks.complete).toHaveBeenCalled())
    rerender({ ...deps, isServerConnected: false })
    rerender(deps)
    await act(async () => { pending.resolve({ assistant_content: "Settled reply" }); await operation })
    expect(result.current.quickChatDraft).toBe("")
    expect(mocks.create).toHaveBeenCalledTimes(1)
  })

  it("preserves newer input when an older send settles", async () => {
    const pending = held<{ assistant_content: string }>()
    mocks.complete.mockReturnValueOnce(pending.promise)
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    let operation!: Promise<void>
    act(() => { operation = result.current.sendQuickChatMessage() })
    await waitFor(() => expect(mocks.complete).toHaveBeenCalled())
    act(() => result.current.setQuickChatDraft("new next message  "))
    await act(async () => { pending.resolve({ assistant_content: "Settled reply" }); await operation })
    expect(result.current.quickChatDraft).toBe("new next message  ")
  })

  it("reserves departure synchronously so a held clipboard cannot race Send", async () => {
    const pending = held<void>()
    mocks.copy.mockReturnValueOnce(pending.promise)
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    let departing!: Promise<void>
    act(() => { departing = result.current.handlePromoteQuickChat() })
    await waitFor(() => expect(mocks.copy).toHaveBeenCalled())
    await act(async () => { await result.current.sendQuickChatMessage() })
    const sendsBeforeAck = mocks.create.mock.calls.length
    await act(async () => { pending.resolve(); await departing })
    expect(sendsBeforeAck).toBe(0)
  })

  it.each(["refused", "unknown", "dispatching"])(
    "Quick retains exact input and classifies %s send recovery without automatic replay",
    async (state) => {
      const acceptedSelection = {
        status: "ready",
        bookmarkScope: { profile_id: "p", client_session_id: "client" },
        view: { conversation_id: "quick-session", owner_key: "popup-owner" },
        owner: {
          kind: "native",
          conversation_id: "quick-session", owner_key: "popup-owner", request_scope: mocks.requestScope,
          validate_lease: () => true
        },
        capture: { status: "captured" }
      }
      const { loadHistoryTurnRecoveries } =
        await import("@/db/dexie/history-selection")
      const { readChatCreationRecovery } =
        await import("@/services/pending-chat-promotion")
      vi.mocked(loadHistoryTurnRecoveries).mockResolvedValueOnce([
        {
          scope: { profile_id: "p", client_session_id: "old" },
          turn: {
            operation_id: "attempt",
            state,
            input_text: "  exact draft\ntrailing spaces  "
          }
        } as unknown as Awaited<
          ReturnType<typeof loadHistoryTurnRecoveries>
        >[number]
      ])
      vi.mocked(readChatCreationRecovery).mockResolvedValueOnce({
        state: "bound",
        chatId: "quick-session",
        inputText: "  exact draft\ntrailing spaces  "
      } as unknown as NonNullable<
        Awaited<ReturnType<typeof readChatCreationRecovery>>
      >)
      const { NativeHistoryContextRefusal } =
        await import("@/hooks/chat/native-character-preflight")
      mocks.nativeSend.mockImplementationOnce(async (params) => {
        mocks.selection = acceptedSelection
        params.setServerChatId("quick-session")
        throw new NativeHistoryContextRefusal(new Error("HTTP 409"), () => true)
      })
      const { result } = renderHook(() => useCharacterQuickChat(deps))
      await start(result)
      await act(async () => {
        await result.current.sendQuickChatMessage()
      })
      expect(result.current.quickChatDraft).toBe(
        "  exact draft\ntrailing spaces  "
      )
      expect(result.current.quickChatNeedsReview).toBe(state !== "refused")
      expect(result.current.quickChatError).toContain(
        "Your message was not added"
      )
      expect(mocks.nativeSend).toHaveBeenCalledOnce()
      expect(mocks.deleteChat).not.toHaveBeenCalled()
    }
  )

  it("does not label a native preparation failure as an uncertain send without recovery facts", async () => {
    mocks.nativeSend.mockRejectedValueOnce(new Error("stale_selection"))
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    await act(async () => { await result.current.sendQuickChatMessage() })
    expect(result.current.quickChatNeedsReview).toBe(false)
    expect(result.current.quickChatDraft).toBe("  exact draft\ntrailing spaces  ")
  })

  it("keeps A until replacement draft retention is acknowledged", async () => {
    const pending = held<boolean>()
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    mocks.retain.mockReturnValueOnce(pending.promise)
    let replacing!: Promise<void>
    act(() => { replacing = result.current.openQuickChat(card(5)) })
    const beforeAck = { id: String(result.current.quickChatCharacter?.id), draft: result.current.quickChatDraft }
    await act(async () => { pending.resolve(true); await replacing })
    expect(beforeAck).toEqual({ id: "4", draft: "  exact draft\ntrailing spaces  " })
    expect(String(result.current.quickChatCharacter?.id)).toBe("5")
  })

  it("refuses card replacement when A cannot be retained", async () => {
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    mocks.retain.mockResolvedValueOnce(false)
    await act(async () => { await result.current.openQuickChat(card(5)) })
    expect(String(result.current.quickChatCharacter?.id)).toBe("4")
    expect(result.current.quickChatDraft).toBe("  exact draft\ntrailing spaces  ")
  })

  it("does not let a closed A creation install its session or reply into B", async () => {
    const pending = held<{ id: string }>()
    mocks.create.mockReturnValueOnce(pending.promise)
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    let operation!: Promise<void>
    act(() => { operation = result.current.sendQuickChatMessage() })
    await waitFor(() => expect(mocks.create).toHaveBeenCalledTimes(1))
    await act(async () => { await result.current.closeQuickChat() })
    await act(async () => { await result.current.openQuickChat(card(5)) })
    await act(async () => { pending.resolve({ id: "late-A" }); await operation })
    expect(result.current.quickChatSessionId).toBeNull()
    expect(result.current.quickChatMessages.some(row => row.content === "Owned reply")).toBe(false)
  })

  it("does not implicitly delete a saved conversation when closing the popup", async () => {
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    await act(async () => { await result.current.sendQuickChatMessage() })
    await act(async () => { await result.current.closeQuickChat() })
    expect(mocks.deleteChat).not.toHaveBeenCalled()
  })

  it("retains exact editable input when a send outcome is unconfirmed", async () => {
    mocks.complete.mockRejectedValueOnce(new Error("native_history_persistence_unacknowledged"))
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    await act(async () => { await result.current.sendQuickChatMessage() })
    expect(result.current.quickChatDraft).toBe("  exact draft\ntrailing spaces  ")
  })

  it("uses the independent native controller and private handle without writing the full-chat assistant", async () => {
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    await act(async () => { await result.current.sendQuickChatMessage() })
    expect(mocks.nativeSend).toHaveBeenCalledWith(expect.objectContaining({
      privateDraft: { handle: { ownerHash: chatDraftOwnerKey(mocks.requestScope), address: expect.objectContaining({ kind: "fresh", temporary: false }) }, requestScope: mocks.requestScope, isCurrent: expect.any(Function) },
      toolChoice: "none", model: "openai:fixture-model"
    }))
    expect(mocks.selectCharacter).not.toHaveBeenCalled()
  })

  it("blocks a second Send synchronously before React publishes busy state", async () => {
    const pending = held<{ id: string }>()
    mocks.create.mockReturnValue(pending.promise)
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    let operation!: Promise<void>
    act(() => { operation = result.current.sendQuickChatMessage(); void result.current.sendQuickChatMessage() })
    await waitFor(() => expect(mocks.create).toHaveBeenCalled())
    expect(mocks.create).toHaveBeenCalledTimes(1)
    await act(async () => { pending.resolve({ id: "quick-session" }); await operation })
  })

  it("clears private popup state on account change and fences a held reply across account ABA", async () => {
    const pending = held<{ assistant_content: string }>()
    mocks.complete.mockReturnValueOnce(pending.promise)
    const { result, rerender } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    let operation!: Promise<void>
    act(() => { operation = result.current.sendQuickChatMessage() })
    await waitFor(() => expect(mocks.complete).toHaveBeenCalledTimes(1))
    act(() => { mocks.ownerGeneration++; mocks.accountClear?.() })
    rerender()
    act(() => { mocks.ownerGeneration++; mocks.accountClear?.() })
    rerender()
    await act(async () => { await result.current.openQuickChat(card(5)) })
    await act(async () => { pending.resolve({ assistant_content: "Old account reply" }); await operation })
    expect(result.current.quickChatMessages.some(row => row.content === "Old account reply")).toBe(false)
    expect(result.current.quickChatSessionId).toBeNull()
  })

  it("keeps the popup and makes no send request when exact draft retention fails", async () => {
    mocks.retain.mockResolvedValue(false)
    const { result } = renderHook(() => useCharacterQuickChat(deps))
    await start(result)
    await act(async () => { await result.current.sendQuickChatMessage() })
    expect(mocks.create).not.toHaveBeenCalled()
    expect(result.current.quickChatDraft).toBe("  exact draft\ntrailing spaces  ")
  })
})


it("permits a deliberate new popup after an earlier preparation never dispatched", async () => {
  const { listChatCreationRecovery } = await import("@/services/pending-chat-promotion")
  const { freshChatDraftAddress } = await import("@/utils/chat-draft-address")
  vi.mocked(listChatCreationRecovery).mockResolvedValueOnce({ unavailable: false, attempts: [{
    version: 1, operationId: "old-prepared", state: "prepared", characterId: "4", inputText: "OLD UNSENT",
    surface: "playground", createdAt: 1, handle: { address: freshChatDraftAddress(false), ownerHash: chatDraftOwnerKey(mocks.requestScope) }
  }] })
  const { result } = renderHook(() => useCharacterQuickChat(deps))
  await start(result)
  await act(async () => { await result.current.sendQuickChatMessage() })
  expect(mocks.nativeSend).toHaveBeenCalledOnce()
  expect(result.current.quickChatNeedsReview).toBe(false)
})


it("requires created-chat Review when a native owner exists without qualified history", async () => {
  const { readChatCreationRecovery } = await import("@/services/pending-chat-promotion")
  const { freshChatDraftAddress } = await import("@/utils/chat-draft-address")
  vi.mocked(readChatCreationRecovery).mockResolvedValueOnce({ version: 1, operationId: "known-id", state: "acknowledged", chatId: "quick-session",
    characterId: "4", inputText: "ORIGINAL", surface: "playground", createdAt: 1,
    handle: { address: freshChatDraftAddress(false), ownerHash: chatDraftOwnerKey(mocks.requestScope) } })
  mocks.nativeSend.mockRejectedValueOnce(new Error("history_selection_not_ready"))
  const { result } = renderHook(() => useCharacterQuickChat(deps))
  await start(result)
  await act(async () => { await result.current.sendQuickChatMessage() })
  expect(result.current.quickChatNeedsReview).toBe(true)
  expect(result.current.quickChatDraft).toBe("  exact draft\ntrailing spaces  ")
  await act(async () => { await result.current.sendQuickChatMessage() })
  expect(mocks.nativeSend).toHaveBeenCalledOnce()
  await act(async () => { await result.current.handlePromoteQuickChat() })
  expect(mocks.navigate).toHaveBeenCalledWith("/chat")
})
