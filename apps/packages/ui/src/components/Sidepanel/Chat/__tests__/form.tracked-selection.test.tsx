// @vitest-environment jsdom
import {
  nativeCharacterPreflightMessage,
  resolveNativeCharacterConfiguration
} from "@/hooks/chat/native-character-preflight"
import { getSelectedAssistantOperationRevision } from "@/hooks/useSelectedAssistant"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import type { AssistantSelection } from "@/types/assistant-selection"
import { createDefaultActorSettings } from "@/types/actor"
import { freshChatDraftAddress } from "@/utils/chat-draft-address"
import { buildChatModelUsability } from "@/utils/chat-model-availability"
import { resolveMessageSteering } from "@/utils/message-steering"
import { act, renderHook } from "@testing-library/react"
import { readFileSync } from "node:fs"
import path from "node:path"
import React from "react"
import ts from "typescript"
import { beforeEach, describe, expect, it, vi } from "vitest"

// Exercise the actual compact callback with real store/preflight boundaries.
// This adapter does not claim a mounted native extension or whole form journey.
const ast = ts.createSourceFile(
  "form.tsx",
  readFileSync(path.resolve(__dirname, "../form.tsx"), "utf8"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX
)
const callbacks = new Map<string, ts.Expression>()
const visit = (node: ts.Node) => {
  if (ts.isVariableDeclaration(node) && node.initializer)
    callbacks.set(node.name.getText(ast), node.initializer)
  ts.forEachChild(node, visit)
}
visit(ast)
const callback =
  callbacks.get("handleTrackedAssistantSelection") ??
  callbacks.get("handlePrepareTrackedStart")
if (!callback) throw new Error("Missing compact tracked-selection callback")
const evaluate = new Function(
  "React",
  "scope",
  ts.transpileModule(`with (scope) { return ${callback.getText(ast)}; }`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 }
  }).outputText
) as (
  react: typeof React,
  scope: Record<string, unknown>
) => (
  selection: AssistantSelection,
  current?: () => boolean
) => Promise<boolean>
const character: AssistantSelection = {
  kind: "character",
  id: "4",
  name: "Cedar",
  metadata: { selectionMode: "tracked" }
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const fixture = () => {
  let ownerCurrent = true
  let historyCurrent = true
  const view = {
    open: true,
    ready: true,
    streaming: false,
    model: "custom-openai-api:gpt-4o-mini",
    models: [
      {
        model: "gpt-4o-mini",
        provider: "custom-openai-api",
        is_configured: true
      }
    ],
    chatMode: "normal",
    documents: [] as unknown[],
    files: [] as unknown[],
    pageContext: false,
    messageRevision: 1,
    message: "KEEP SOURCE\nTRAILING SPACES  ",
    image: ""
  }
  const retained: string[] = []
  const retainCurrentDraft = vi.fn(async () => {
    retained.push(view.message)
    return true
  })
  const setSelectedAssistant = vi.fn(
    async (
      _selection: AssistantSelection,
      options: { isCurrent: () => boolean }
    ) => {
      if (!options.isCurrent()) return false
      return true
    }
  )
  const snapshot = {
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
    release: vi.fn()
  }
  const historySelection = {
    fence: () => () => historyCurrent,
    reset: vi.fn(),
    getCurrent: () => ({ owner: null })
  }
  const notification = { error: vi.fn() }
  const requestSettingsNavigation = vi.fn(() => true)
  const connectionGenerationRef = { current: 0 }
  const scope = {
    useStoreMessageOption,
    getSelectedAssistantOperationRevision,
    useStoreChatModelSettings,
    usePlaygroundSessionStore,
    useMcpToolsStore,
    resolveNativeCharacterConfiguration,
    nativeCharacterPreflightMessage,
    buildChatModelUsability,
    resolveMessageSteering,
    trackedSelectionViewRef: { current: view },
    trackedSelectionPendingRef: {
      current: null as { isCurrent: () => boolean } | null
    },
    connectionGenerationRef,
    characterMeta: { isCurrent: () => ownerCurrent },
    greetingHistorySelection: historySelection,
    retainCurrentDraft,
    setSelectedAssistant,
    createDefaultActorSettings,
    freshChatDraftAddress,
    getActorSettingsForChat: vi
      .fn<typeof import("@/services/actor-settings").getActorSettingsForChat>()
      .mockResolvedValue(createDefaultActorSettings()),
    loadServicePromptSnapshot: vi.fn(async () => snapshot),
    requestSettingsNavigation,
    notification,
    t: (_key: string, fallback: string) => fallback,
    sidepanelOverlayResumeStorageRef: {
      current: { remove: vi.fn(async () => {}) }
    },
    overlayResumeMarkerKey: "source-marker"
  }
  const hook = renderHook(() => {
    const form = useSimpleForm({
      initialValues: { message: view.message, image: view.image }
    })
    Object.assign(scope, { form })
    return { select: evaluate(React, scope), form }
  })
  return {
    hook,
    scope,
    view,
    retained,
    retainCurrentDraft,
    setSelectedAssistant,
    snapshot,
    historySelection,
    notification,
    requestSettingsNavigation,
    revokeOwner: () => {
      ownerCurrent = false
    },
    revokeHistory: () => {
      historyCurrent = false
    }
  }
}

beforeEach(() => {
  useStoreChatModelSettings.getState().reset()
  useMcpToolsStore.setState({ tools: [] })
  usePlaygroundSessionStore.getState().clearSession()
  useStoreMessageOption.getState().startNewChatDraft(false)
  useStoreMessageOption.setState({
    selectedModel: "custom-openai-api:gpt-4o-mini",
    historyId: "local-source",
    serverChatId: "source",
    history: [{ role: "user", content: "SOURCE TURN" }],
    messages: [],
    temporaryChat: false,
    selectedSystemPrompt: "",
    webSearch: false,
    toolChoice: "none",
    replyTarget: null,
    messageSteeringMode: "none",
    messageSteeringForceNarrate: false,
    serverChatCharacterId: "2",
    serverChatAssistantKind: "character",
    serverChatMetaLoaded: true,
    isProcessing: false,
    streaming: false
  })
})

const apply = async (
  f: ReturnType<typeof fixture>,
  selection = character,
  current = () => true
) => {
  let accepted: boolean | undefined
  await act(async () => {
    accepted = await f.hook.result.current.select(selection, current)
  })
  return accepted
}

describe("actual compact tracked-selection transition", () => {
  it("retains exact source text before identity, then binds a fresh destination without erasing response settings", async () => {
    useStoreChatModelSettings
      .getState()
      .setActiveSettingsScope("custom-openai-api:gpt-4o-mini")
    useStoreChatModelSettings.getState().setTemperature(0.6)
    const settings = useStoreChatModelSettings.getState()
    const source = useStoreMessageOption.getState().chatDraftAddress
    const f = fixture()
    expect(await apply(f)).toBe(true)
    expect(f.retained).toEqual(["KEEP SOURCE\nTRAILING SPACES  "])
    expect(f.retainCurrentDraft.mock.invocationCallOrder[0]).toBeLessThan(
      f.setSelectedAssistant.mock.invocationCallOrder[0]
    )
    expect(useStoreMessageOption.getState()).toMatchObject({
      historyId: null,
      serverChatId: null,
      history: [],
      messages: [],
      temporaryChat: false,
      serverChatAssistantKind: "character",
      serverChatCharacterId: "4",
      serverChatMetaLoaded: false
    })
    expect(useStoreMessageOption.getState().chatDraftAddress).not.toBe(source)
    expect(useStoreChatModelSettings.getState()).toBe(settings)
    expect(f.historySelection.reset).toHaveBeenCalledOnce()
    expect(f.requestSettingsNavigation).toHaveBeenCalledOnce()
    expect(f.snapshot.release).toHaveBeenCalledOnce()
  })

  it.each(["enabled", "unavailable"] as const)(
    "Scene selection isolates the fresh compact destination from %s legacy scratch",
    async (legacy) => {
      const f = fixture()
      const source = useStoreMessageOption.getState()
      if (legacy === "enabled")
        f.scope.getActorSettingsForChat.mockResolvedValue({
          ...createDefaultActorSettings(),
          isEnabled: true,
          notes: "Departing scratch must stay private"
        })
      else
        f.scope.getActorSettingsForChat.mockRejectedValue(
          new Error("actor_settings_unavailable")
        )
      try {
        expect(await apply(f)).toBe(true)
        expect(f.scope.getActorSettingsForChat).not.toHaveBeenCalled()
        expect(f.retained).toEqual(["KEEP SOURCE\nTRAILING SPACES  "])
        expect(f.setSelectedAssistant).toHaveBeenCalledOnce()
        expect(f.historySelection.reset).toHaveBeenCalledOnce()
        expect(useStoreMessageOption.getState().chatDraftAddress).not.toBe(
          source.chatDraftAddress
        )
      } finally {
        f.hook.unmount()
      }
    }
  )

  it.each([
    "New",
    "New then same raw IDs",
    "New then same issued address"
  ] as const)(
    "Scene selection refuses compact destination publication retired by %s",
    async (retirement) => {
      const f = fixture()
      const original = useStoreMessageOption.getState()
      let armed = true
      let replacement:
        | ReturnType<typeof useStoreMessageOption.getState>
        | undefined
      const release = useStoreMessageOption.subscribe((state) => {
        if (!armed || state.chatDraftAddress === original.chatDraftAddress) return
        armed = false
        const issuedAddress = state.chatDraftAddress
        state.startNewChatDraft(false)
        if (retirement === "New then same issued address" && issuedAddress)
          useStoreMessageOption.getState().setChatDraftAddress(issuedAddress)
        useStoreMessageOption.setState({
          historyId:
            retirement === "New then same raw IDs"
              ? original.historyId
              : "replacement-local",
          serverChatId:
            retirement === "New then same raw IDs"
              ? original.serverChatId
              : "replacement-server",
          history: [{ role: "user", content: "REPLACEMENT TURN" }],
          messages: [
            {
              id: "replacement-message",
              isBot: false,
              name: "You",
              message: "REPLACEMENT VIEW",
              sources: []
            }
          ]
        })
        replacement = useStoreMessageOption.getState()
      })
      try {
        expect(await apply(f)).toBe(false)
        expect(armed).toBe(false)
        expect(replacement).toBeDefined()
        expect(useStoreMessageOption.getState()).toBe(replacement)
        expect(f.historySelection.reset).not.toHaveBeenCalled()
        expect(f.retained).toEqual(["KEEP SOURCE\nTRAILING SPACES  "])
        expect(f.setSelectedAssistant).toHaveBeenCalledOnce()
      } finally {
        release()
        f.hook.unmount()
      }
    }
  )

  it.each(["navigation", "retention", "identity"])(
    "preserves the source after %s fails",
    async (failure) => {
      const f = fixture()
      const source = useStoreMessageOption.getState()
      if (failure === "navigation")
        f.requestSettingsNavigation.mockReturnValue(false)
      if (failure === "retention") f.retainCurrentDraft.mockResolvedValue(false)
      if (failure === "identity")
        f.setSelectedAssistant.mockRejectedValue(new Error("storage failed"))
      expect(await apply(f)).toBe(false)
      expect(useStoreMessageOption.getState()).toBe(source)
      expect(f.historySelection.reset).not.toHaveBeenCalled()
      if (failure !== "identity")
        expect(f.setSelectedAssistant).not.toHaveBeenCalled()
    }
  )

  it.each([
    "unqualified model",
    "unconfigured provider",
    "image",
    "files",
    "prompt",
    "sampling",
    "tools",
    "persona",
    "steering",
    "in-flight send"
  ])("rejects unsupported %s before leaving source", async (unsupported) => {
    const f = fixture()
    if (unsupported === "unqualified model") f.view.model = "tldw:gpt-4o-mini"
    if (unsupported === "unconfigured provider")
      f.view.models[0].is_configured = false
    if (unsupported === "image") f.view.image = "data:image/png;base64,review"
    if (unsupported === "files") f.view.files = [{}]
    if (unsupported === "prompt")
      useStoreMessageOption.getState().setSelectedSystemPrompt("source prompt")
    if (unsupported === "sampling")
      useStoreChatModelSettings.getState().setTopK(3)
    if (unsupported === "tools") {
      useMcpToolsStore.setState({ tools: [{}] as never })
      useStoreMessageOption.getState().setToolChoice("required")
    }
    if (unsupported === "steering")
      useStoreMessageOption
        .getState()
        .setMessageSteeringMode("continue_as_user")
    if (unsupported === "in-flight send")
      useStoreMessageOption.getState().setIsProcessing(true)
    const source = useStoreMessageOption.getState()
    expect(
      await apply(
        f,
        unsupported === "persona"
          ? { kind: "persona", id: "p", name: "Persona" }
          : character
      )
    ).toBe(false)
    expect(useStoreMessageOption.getState()).toBe(source)
    expect(f.setSelectedAssistant).not.toHaveBeenCalled()
  })

  it("preserves the source when the owned setter declines its identity write", async () => {
    const f = fixture()
    const source = useStoreMessageOption.getState()
    f.setSelectedAssistant.mockResolvedValueOnce(false)
    expect(await apply(f)).toBe(false)
    expect(useStoreMessageOption.getState()).toBe(source)
    expect(f.historySelection.reset).not.toHaveBeenCalled()
  })

  it("reports preference recovery failure without claiming it was restored", async () => {
    const f = fixture()
    const source = useStoreMessageOption.getState()
    f.setSelectedAssistant.mockRejectedValueOnce(
      new Error("assistant_selection_restore_failed")
    )
    expect(await apply(f)).toBe(false)
    expect(useStoreMessageOption.getState()).toBe(source)
    expect(f.notification.error).toHaveBeenCalledWith(
      expect.objectContaining({
        description:
          "Assistant preferences could not be restored. Your chat and draft are still here; try choosing the assistant again."
      })
    )
  })

  it("retires held retention after live tool choice changes", async () => {
    useMcpToolsStore.setState({ tools: [{}] as never })
    const f = fixture()
    const held = deferred<boolean>()
    f.retainCurrentDraft.mockReturnValueOnce(held.promise)
    let pending!: Promise<boolean>
    act(() => {
      pending = f.hook.result.current.select(character)
    })
    await vi.waitFor(() => expect(f.retainCurrentDraft).toHaveBeenCalledOnce())
    useStoreMessageOption.getState().setToolChoice("required")
    const source = useStoreMessageOption.getState()
    await act(async () => {
      held.resolve(true)
      expect(await pending).toBe(false)
    })
    expect(useStoreMessageOption.getState()).toBe(source)
    expect(f.setSelectedAssistant).not.toHaveBeenCalled()
  })

  it.each(["image", "message"] as const)(
    "retires actual batched %s edits that return to the original value",
    async (field) => {
      const f = fixture()
      const held = deferred<boolean>()
      f.setSelectedAssistant.mockReturnValueOnce(held.promise)
      let pending!: Promise<boolean>
      act(() => {
        pending = f.hook.result.current.select(character)
      })
      await vi.waitFor(() =>
        expect(f.setSelectedAssistant).toHaveBeenCalledOnce()
      )
      const form = f.hook.result.current.form
      const original = form.getValues()[field]
      act(() => {
        form.setFieldValue(field, "changed")
        form.setFieldValue(field, original)
      })
      expect(form.getValues()[field]).toBe(original)
      const source = useStoreMessageOption.getState()
      await act(async () => {
        held.resolve(true)
        expect(await pending).toBe(false)
      })
      expect(useStoreMessageOption.getState()).toBe(source)
      expect(f.historySelection.reset).not.toHaveBeenCalled()
    }
  )

  it.each(["retention", "identity"])(
    "retires held %s work after close/reopen without overwriting a newer choice",
    async (stage) => {
      const f = fixture()
      const held = deferred<boolean>()
      let pickerCurrent = true
      if (stage === "retention")
        f.retainCurrentDraft.mockImplementationOnce(() => held.promise)
      else
        f.setSelectedAssistant.mockImplementationOnce(async () => {
          return await held.promise
        })
      let pending!: Promise<boolean>
      act(() => {
        pending = f.hook.result.current.select(character, () => pickerCurrent)
      })
      await vi.waitFor(() =>
        expect(
          stage === "retention" ? f.retainCurrentDraft : f.setSelectedAssistant
        ).toHaveBeenCalledOnce()
      )
      pickerCurrent = false
      f.scope.trackedSelectionPendingRef.current = null
      const current = useStoreMessageOption.getState()
      let retired: boolean | undefined
      await act(async () => {
        held.resolve(true)
        retired = await pending
      })
      expect(retired).toBe(false)
      expect(useStoreMessageOption.getState()).toBe(current)
      expect(f.historySelection.reset).not.toHaveBeenCalled()
    }
  )

  it.each(
    (["snapshot", "retention"] as const).flatMap((stage) =>
      (
        [
          "picker",
          "owner",
          "history",
          "New",
          "restore",
          "input ABA",
          "model settings",
          "connection ABA"
        ] as const
      ).map((boundary) => ({ stage, boundary }))
    )
  )(
    "retires held $stage preparation after $boundary changes",
    async ({ stage, boundary }) => {
      const f = fixture()
      const snapshotGate = deferred<typeof f.snapshot>()
      const retentionGate = deferred<boolean>()
      if (stage === "snapshot")
        f.scope.loadServicePromptSnapshot.mockReturnValueOnce(
          snapshotGate.promise
        )
      else f.retainCurrentDraft.mockReturnValueOnce(retentionGate.promise)
      let pickerCurrent = true
      let pending!: Promise<boolean>
      act(() => {
        pending = f.hook.result.current.select(character, () => pickerCurrent)
      })
      try {
        await vi.waitFor(() =>
          expect(
            stage === "snapshot"
              ? f.scope.loadServicePromptSnapshot
              : f.retainCurrentDraft
          ).toHaveBeenCalledOnce()
        )
        if (boundary === "picker") pickerCurrent = false
        if (boundary === "owner") f.revokeOwner()
        if (boundary === "history") f.revokeHistory()
        if (boundary === "New")
          useStoreMessageOption.getState().startNewChatDraft()
        if (boundary === "restore")
          usePlaygroundSessionStore.getState().cancelPendingRestore()
        if (boundary === "input ABA") f.view.messageRevision += 2
        if (boundary === "model settings")
          useStoreChatModelSettings.getState().setTemperature(0.9)
        if (boundary === "connection ABA")
          f.scope.connectionGenerationRef.current += 1
        const current = useStoreMessageOption.getState()
        await act(async () => {
          snapshotGate.resolve(f.snapshot)
          retentionGate.resolve(true)
          expect(await pending).toBe(false)
        })
        expect(useStoreMessageOption.getState()).toBe(current)
        expect(f.setSelectedAssistant).not.toHaveBeenCalled()
      } finally {
        snapshotGate.resolve(f.snapshot)
        retentionGate.resolve(true)
        await pending
        f.hook.unmount()
      }
    }
  )
})


// The existing adapter executes the production callback, not a mounted form or
// actual H1 controller. These observers delegate each publication to its real
// store setter; identity, prompt, retention, navigation and H1 IO retain the
// explicitly declared fixture boundaries above.
const observeTrackedSelectionPublications = () => {
  const chat = useStoreMessageOption.getState()
  const session = usePlaygroundSessionStore.getState()
  const original = {
    setHistoryId: chat.setHistoryId,
    setServerChatId: chat.setServerChatId,
    setHistory: chat.setHistory,
    setMessages: chat.setMessages
  }
  const publications = {
    setHistoryId: vi.fn(original.setHistoryId),
    setServerChatId: vi.fn(original.setServerChatId),
    setHistory: vi.fn(original.setHistory),
    setMessages: vi.fn(original.setMessages),
    clearSession: vi.fn(session.clearSession)
  }
  useStoreMessageOption.setState({
    setHistoryId: publications.setHistoryId,
    setServerChatId: publications.setServerChatId,
    setHistory: publications.setHistory,
    setMessages: publications.setMessages
  })
  usePlaygroundSessionStore.setState({
    clearSession: publications.clearSession
  })
  return {
    publications,
    restore: () => {
      useStoreMessageOption.setState(original)
      usePlaygroundSessionStore.setState({ clearSession: session.clearSession })
    }
  }
}

const seedTrackedSourcePresentation = () => {
  useStoreMessageOption
    .getState()
    .setHistory([{ role: "user", content: "SOURCE TURN" }])
  useStoreMessageOption.getState().setMessages([
    {
      id: "source-message",
      isBot: false,
      name: "You",
      message: "SOURCE VIEW",
      sources: []
    }
  ])
  useStoreChatModelSettings
    .getState()
    .setActiveSettingsScope("custom-openai-api:gpt-4o-mini")
  useStoreChatModelSettings.getState().setTemperature(0.6)
}

describe("actual compact tracked-selection intent handoff", () => {
  it.each([
    "unchanged",
    "foreign model choice",
    "same-value model revision",
    "changed tool choice"
  ] as const)(
    "fresh destination subscriber handoff preserves original intent after %s",
    async (mutation) => {
      seedTrackedSourcePresentation()
      const observer = observeTrackedSelectionPublications()
      const original = useStoreMessageOption.getState()
      const session = usePlaygroundSessionStore.getState()
      const settings = useStoreChatModelSettings.getState()
      const tools = useMcpToolsStore.getState()
      const f = fixture()
      let issued:
        | {
            address: NonNullable<typeof original.chatDraftAddress>
            revision: number
          }
        | undefined
      let handoff: typeof original | undefined
      const release = useStoreMessageOption.subscribe((state) => {
        if (
          issued ||
          !state.chatDraftAddress ||
          state.chatDraftAddress === original.chatDraftAddress
        )
          return
        issued = {
          address: state.chatDraftAddress,
          revision: state.chatDraftRevision
        }
        if (mutation === "foreign model choice")
          useStoreMessageOption
            .getState()
            .setSelectedModel("custom-openai-api:gpt-4.1")
        if (mutation === "same-value model revision")
          useStoreMessageOption
            .getState()
            .setSelectedModel(original.selectedModel)
        if (mutation === "changed tool choice")
          useStoreMessageOption.getState().setToolChoice("required")
        handoff = useStoreMessageOption.getState()
      })
      try {
        const accepted = await apply(f)
        // Qualify the actual callback's preflight, identity ACK and own issued
        // address before checking whether the continuation was refused.
        expect(f.scope.loadServicePromptSnapshot).toHaveBeenCalledOnce()
        expect(f.retained).toEqual(["KEEP SOURCE\nTRAILING SPACES  "])
        expect(f.setSelectedAssistant).toHaveBeenCalledOnce()
        expect(issued?.address.kind).toBe("fresh")
        expect(issued?.revision).toBe(original.chatDraftRevision + 1)
        expect(handoff?.chatDraftAddress).toBe(issued?.address)
        expect(handoff?.chatDraftRevision).toBe(issued?.revision)
        expect(handoff?.history).toBe(original.history)
        expect(handoff?.messages).toBe(original.messages)
        expect(handoff?.historyId).toBe(original.historyId)
        expect(handoff?.serverChatId).toBe(original.serverChatId)
        const current = useStoreMessageOption.getState()
        expect(current.chatDraftAddress).toBe(issued?.address)
        expect(current.chatDraftRevision).toBe(issued?.revision)
        expect(current.selectedModel).toBe(
          mutation === "foreign model choice"
            ? "custom-openai-api:gpt-4.1"
            : original.selectedModel
        )
        expect(current.selectedModelRevision).toBe(
          original.selectedModelRevision +
            (mutation === "foreign model choice" ||
            mutation === "same-value model revision"
              ? 1
              : 0)
        )
        expect(current.toolChoice).toBe(
          mutation === "changed tool choice" ? "required" : original.toolChoice
        )
        expect(useStoreChatModelSettings.getState()).toBe(settings)
        expect(useMcpToolsStore.getState()).toBe(tools)
        expect(f.hook.result.current.form.getValues()).toEqual({
          message: "KEEP SOURCE\nTRAILING SPACES  ",
          image: ""
        })
        expect(accepted).toBe(mutation === "unchanged")
        if (mutation === "unchanged") {
          expect(observer.publications.clearSession).toHaveBeenCalledOnce()
          expect(f.historySelection.reset).toHaveBeenCalledOnce()
          for (const publish of [
            observer.publications.setHistoryId,
            observer.publications.setServerChatId,
            observer.publications.setHistory,
            observer.publications.setMessages
          ])
            expect(publish).toHaveBeenCalledOnce()
          expect(current).toMatchObject({
            historyId: null,
            serverChatId: null,
            history: [],
            messages: [],
            serverChatCharacterId: character.id,
            serverChatAssistantKind: character.kind,
            serverChatMetaLoaded: false
          })
          expect(usePlaygroundSessionStore.getState().restoreRevision).toBe(
            session.restoreRevision + 1
          )
        } else {
          expect(observer.publications.clearSession).not.toHaveBeenCalled()
          expect(f.historySelection.reset).not.toHaveBeenCalled()
          for (const publish of [
            observer.publications.setHistoryId,
            observer.publications.setServerChatId,
            observer.publications.setHistory,
            observer.publications.setMessages
          ])
            expect(publish).not.toHaveBeenCalled()
          expect(current).toBe(handoff)
          expect(current.history).toBe(original.history)
          expect(current.messages).toBe(original.messages)
          expect(current.historyId).toBe(original.historyId)
          expect(current.serverChatId).toBe(original.serverChatId)
          expect(current.serverChatCharacterId).toBe(
            original.serverChatCharacterId
          )
          expect(usePlaygroundSessionStore.getState()).toBe(session)
        }
        expect(f.scope.getActorSettingsForChat).not.toHaveBeenCalled()
        expect(
          f.scope.sidepanelOverlayResumeStorageRef.current.remove
        ).not.toHaveBeenCalled()
        expect(f.notification.error).not.toHaveBeenCalled()
        expect(f.snapshot.release).toHaveBeenCalledOnce()
        expect(f.scope.trackedSelectionPendingRef.current).toBeNull()
      } finally {
        release()
        f.hook.unmount()
        observer.restore()
      }
    }
  )

  it("same-value model revision retires held preparation before identity", async () => {
    seedTrackedSourcePresentation()
    const observer = observeTrackedSelectionPublications()
    const original = useStoreMessageOption.getState()
    const session = usePlaygroundSessionStore.getState()
    const settings = useStoreChatModelSettings.getState()
    const tools = useMcpToolsStore.getState()
    const f = fixture()
    const held = deferred<boolean>()
    f.retainCurrentDraft.mockReturnValueOnce(held.promise)
    let pending!: Promise<boolean>
    act(() => {
      pending = f.hook.result.current.select(character)
    })
    try {
      await vi.waitFor(() => expect(f.retainCurrentDraft).toHaveBeenCalledOnce())
      expect(f.scope.loadServicePromptSnapshot).toHaveBeenCalledOnce()
      expect(f.setSelectedAssistant).not.toHaveBeenCalled()
      act(() => {
        useStoreMessageOption.getState().setSelectedModel(original.selectedModel)
      })
      const current = useStoreMessageOption.getState()
      expect(current.selectedModel).toBe(original.selectedModel)
      expect(current.selectedModelRevision).toBe(
        original.selectedModelRevision + 1
      )
      expect(current.chatDraftAddress).toBe(original.chatDraftAddress)
      expect(current.chatDraftRevision).toBe(original.chatDraftRevision)
      expect(current.toolChoice).toBe(original.toolChoice)
      expect(current.history).toBe(original.history)
      expect(current.messages).toBe(original.messages)
      expect(useStoreChatModelSettings.getState()).toBe(settings)
      expect(useMcpToolsStore.getState()).toBe(tools)
      let accepted: boolean | undefined
      await act(async () => {
        held.resolve(true)
        accepted = await pending
      })
      expect(accepted).toBe(false)
      expect(f.setSelectedAssistant).not.toHaveBeenCalled()
      expect(f.requestSettingsNavigation).not.toHaveBeenCalled()
      expect(f.historySelection.reset).not.toHaveBeenCalled()
      expect(observer.publications.clearSession).not.toHaveBeenCalled()
      for (const publish of [
        observer.publications.setHistoryId,
        observer.publications.setServerChatId,
        observer.publications.setHistory,
        observer.publications.setMessages
      ])
        expect(publish).not.toHaveBeenCalled()
      expect(useStoreMessageOption.getState()).toBe(current)
      expect(usePlaygroundSessionStore.getState()).toBe(session)
      expect(f.hook.result.current.form.getValues()).toEqual({
        message: "KEEP SOURCE\nTRAILING SPACES  ",
        image: ""
      })
      expect(f.snapshot.release).toHaveBeenCalledOnce()
    } finally {
      held.resolve(true)
      await pending
      f.hook.unmount()
      observer.restore()
    }
  })
})
