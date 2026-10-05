import { createChatWithRecovery, markChatCreationBound, readChatCreationRecovery, markChatCreationSettled } from "@/services/pending-chat-promotion"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import { chatDraftAddressKey, type ChatDraftHandle } from "@/utils/chat-draft-address"
import type { Dispatch, SetStateAction } from "react"
import { captureChatDraftLoad } from "./chat-draft-transition"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import type { TFunction } from "i18next"
import { buildCharacterChatAssistantErrorContent } from "./useCharacterChatMode"
import type { Message, ToolChoice } from "@/store/option"
import type { ChatModelSettings } from "@/store/model"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import type { ChatScope } from "@/types/chat-scope"
import type {
  HistoryLoadReceipt,
  HistorySelectionController
} from "./useHistorySelection"
import type { HistoryAdmissionV1 } from "@/types/history-selection"
import type { HistoryTurnRecovery } from "@/db/dexie/types"
import {
  saveHistoryTurnRecovery,
  dismissHistoryTurnRecovery
} from "@/db/dexie/history-selection"
import { formatSelectedHistory } from "@/db/dexie/helpers"
import { getActorSettingsForChat } from "@/services/actor-settings"
import { loadServicePromptSnapshot } from "@/services/service-prompts"
import {
  assertHistoryConversationTarget,
  captureHistorySnapshot,
  prepareHistoryContext,
  finalizeHistorySelection,
  parseNativeHistoryAdmission,
  historyAdmissionReference
} from "@/services/chat-history-selection"
import {
  tldwClient,
  type ChatCompletionRequest
} from "@/services/tldw/TldwApiClient"
import { prepareChatCompletionRequest } from "@/services/tldw/TldwChat"
import { validateNativeCharacterConfiguration } from "./native-character-preflight"
import { extractStreamTransportInterruption } from "@/utils/extract-token-from-chunk"

export type NativeHistoryCharacterSendParams = {
  controller: HistorySelectionController
  surface: "playground" | "sidepanel"
  originIsCurrent: () => boolean
  signal: AbortSignal
  temporary: boolean
  historyId?: string | null
  serverChatId?: string | null
  scope?: ChatScope
  characterId?: string | number | null
  model: string
  currentModel?: string | null
  toolChoice?: ToolChoice
  settings: ChatModelSettings
  message: string
  image: string
  t: TFunction
  unsupportedContext?: boolean
  setServerChatId: (id: string) => void
  onCreated?: (characterId: string | number) => void
  setMessages: Dispatch<SetStateAction<Message[]>>
  releaseActivity: () => void
}

/** One native dispatch owns admission and settlement. Browser cancellation is never a rollback receipt. */
export const sendNativeHistoryCharacter = async (
  params: NativeHistoryCharacterSendParams
): Promise<void> => {
  const { controller, signal } = params
  const sourceAddress = useStoreMessageOption.getState().chatDraftAddress
  const sourceDraft = sourceAddress ? structuredClone(sourceAddress) : null
  const creationScope = params.scope ? structuredClone(params.scope) : undefined
  const acceptDraftBind = !params.historyId && !params.serverChatId && useStoreMessageOption.getState().chatDraftAddress?.kind === "fresh" ? captureChatDraftLoad("bind") : null
  if (!params.message.trim()) throw new Error("native_history_input_required")
  const settingsState = useStoreChatModelSettings.getState()
  const toolsState = useMcpToolsStore.getState()
  const capturedSettings = { ...params.settings }
  const actor = await getActorSettingsForChat({
    historyId: params.historyId ?? null,
    serverChatId: params.serverChatId ?? null
  })
  const { model, provider, settings } = validateNativeCharacterConfiguration({
    ...params,
    settings: capturedSettings,
    temporary: params.temporary || params.historyId === "temp",
    clientToolCount: toolsState.tools.length,
    actorEnabled: actor?.isEnabled
  })
  if (!params.originIsCurrent()) throw new Error("stale_selection")
  const initialOwner = controller.getCurrent().owner
  if (initialOwner?.kind === "local")
    throw new Error("native_history_requires_server_owner")
  if (initialOwner?.kind === "unavailable") throw new Error(initialOwner.code)
  const snapshot = await loadServicePromptSnapshot([], {
    signal,
    requestScope:
      initialOwner?.kind === "native" ? initialOwner.request_scope : undefined
  })
  const authValid = () => !snapshot.scopeInvalidatedSignal.aborted
  const leaseValid = () =>
    acceptDraftBind?.isCurrent() !== false &&
    authValid() &&
    !signal.aborted &&
    useStoreChatModelSettings.getState() === settingsState &&
    useMcpToolsStore.getState() === toolsState
  try {
    if (!params.originIsCurrent() || !leaseValid())
      throw new Error("stale_selection")
    let creationHandle: ChatDraftHandle | undefined
    let current = controller.getCurrent()
    let selectionCurrent = params.originIsCurrent
    let targetChatId = params.serverChatId
    if (!current.owner && current.status === "idle") {
      if (!params.originIsCurrent() || !leaseValid())
        throw new Error("stale_selection")
      let chatId = params.serverChatId
      let createdCharacterId: string | number | undefined
      if (!chatId && !params.historyId) {
        if (!params.characterId)
          throw new Error("native_history_character_required")
        if (!sourceDraft || sourceDraft.kind !== "fresh") throw new Error("chat_creation_source_invalid")
        creationHandle = { address: sourceDraft, ownerHash: chatDraftOwnerKey(snapshot.requestScope) }
        const created = await createChatWithRecovery({
          snapshot, handle: creationHandle, scope: creationScope, surface: params.surface,
          inputText: params.message, setup: { characterId: params.characterId, provider, model,
            temperature: settings.temperature, topP: settings.topP },
          isCurrent: () => params.originIsCurrent() && leaseValid(),
          create: () => tldwClient.createChat(
          {
            character_id: params.characterId,
            provider,
            model,
            ...(settings.temperature != null ? { temperature: settings.temperature } : {}),
            ...(settings.topP != null ? { top_p: settings.topP } : {})
          },
          {
            scope: creationScope,
            signal: snapshot.scopeSignal,
            requestScope: snapshot.requestScope
          }
          )
        })
        // A late creation ACK must never bind the newly opened view or account.
        if (!params.originIsCurrent() || !leaseValid())
          throw new Error("stale_selection")
        chatId = created.id
        createdCharacterId = params.characterId
        if (!chatId) throw new Error("native_history_creation_unacknowledged")
      }
      let receipt: HistoryLoadReceipt | undefined
      let loadPhase: "preparing" | "accepted" | "rejected" = "preparing"
      const loaded = await controller.loadConversation(
        {
          historyId: params.historyId,
          serverChatId: chatId,
          scope: params.scope,
          // Preparation ends at the qualified bind; later settings or Stop
          // must not revoke the saved owner's independent account lease.
          isCurrent: () => acceptDraftBind?.isCurrent() !== false &&
            (loadPhase === "accepted" || (loadPhase === "preparing" && leaseValid()))
        },
        null,
        (value) => {
          loadPhase = "rejected"
          const now = controller.getCurrent()
          if (!leaseValid() || now.owner !== value.owner || now.view !== value.view ||
            (chatId && (value.owner.kind !== "native" || value.owner.conversation_id !== chatId))) return
          // The binding receipt validates the owner through this same guard.
          loadPhase = "preparing"
          if (acceptDraftBind && !acceptDraftBind(value)) {
            loadPhase = "rejected"
            return
          }
          receipt = value
          loadPhase = "accepted"
          if (chatId) params.setServerChatId(chatId)
          if (createdCharacterId != null) params.onCreated?.(createdCharacterId)
        }
      )
      current = controller.getCurrent()
      if (
        !loaded ||
        !receipt ||
        current.owner !== receipt.owner ||
        current.view !== receipt.view ||
        !leaseValid()
      )
        throw new Error("stale_selection")
      if (
        chatId &&
        (receipt.owner.kind !== "native" ||
          receipt.owner.conversation_id !== chatId)
      )
        throw new Error("owner_conversation_mismatch")
      selectionCurrent = controller.fence()
      if (creationHandle && chatId) await markChatCreationBound(creationHandle, chatId)
      if (!leaseValid() || !selectionCurrent() || controller.getCurrent().owner !== receipt.owner || controller.getCurrent().view !== receipt.view) throw new Error("stale_selection")
      targetChatId = chatId
    }
    if (current.owner?.kind !== "native")
      throw new Error("native_history_requires_server_owner")
    if (!current.owner.validate_lease())
      throw new Error("request_config_scope_changed")
    if (
      current.status !== "ready" ||
      !current.view ||
      !current.bookmarkScope ||
      current.capture?.status !== "captured"
    )
      throw new Error(current.error || "history_selection_not_ready")
    await assertHistoryConversationTarget(current.owner, { historyId: params.historyId, serverChatId: targetChatId })
    if (!selectionCurrent() || !leaseValid() || controller.getCurrent().owner !== current.owner || controller.getCurrent().view !== current.view)
      throw new Error("stale_selection")
    const view = structuredClone(current.view)
    const scope = { ...current.bookmarkScope }
    const owner = {
      ...current.owner,
      scope: current.owner.scope ? { ...current.owner.scope } : undefined,
      owner_key: view.owner_key,
      request_scope: snapshot.requestScope,
      validate_lease: authValid
    }
    if (params.serverChatId && params.serverChatId !== owner.conversation_id)
      throw new Error("owner_conversation_mismatch")
    const sameView = () => {
      const now = controller.getCurrent().view
      return (
        authValid() &&
        !!now &&
        now.owner_key === view.owner_key &&
        now.conversation_id === view.conversation_id &&
        now.view_session_id === view.view_session_id &&
        now.selection_revision === view.selection_revision
      )
    }
    const capture = await captureHistorySnapshot(
      owner,
      view,
      "send",
      snapshot.scopeSignal
    )
    if (capture.status !== "captured") throw new Error(capture.code)
    if (!sameView()) throw new Error("stale_selection")
    const request = prepareChatCompletionRequest(
      [
        ...(settings.systemPrompt?.trim()
          ? [{ role: "system" as const, content: settings.systemPrompt }]
          : []),
        { role: "user", content: params.message }
      ],
      {
        model,
        apiProvider: provider,
        saveToDb: true,
        conversationId: owner.conversation_id,
        temperature: settings.temperature,
        maxTokens: settings.numPredict,
        topP: settings.topP,
        frequencyPenalty: settings.frequencyPenalty,
        presencePenalty: settings.presencePenalty,
        reasoningEffort: settings.reasoningEffort as
          | "low"
          | "medium"
          | "high"
          | undefined,
        thinkingBudgetTokens: settings.llamaThinkingBudgetTokens,
        grammarMode: settings.llamaGrammarMode,
        grammarId: settings.llamaGrammarId,
        grammarInline: settings.llamaGrammarInline,
        grammarOverride: settings.llamaGrammarOverride,
        jsonMode: settings.jsonMode,
        slashCommandInjectionMode: settings.slashCommandInjectionMode
      }
    )
    const prepared = prepareHistoryContext(request, leaseValid)
    const selection = finalizeHistorySelection(
      owner,
      capture,
      prepared,
      controller.getCurrent().view!
    )
    const operationId = crypto.randomUUID()
    const displayId = `native-draft-${operationId}`
    const createdAt = Date.now()
    const base = formatSelectedHistory(capture)
    let admission: HistoryAdmissionV1 | undefined
    let resultId: string | undefined
    let content = ""
    let dispatched = false
    let followAttempted = false
    const record = (
      state: HistoryTurnRecovery["state"]
    ): HistoryTurnRecovery => ({
      operation_id: operationId,
      persistence: "server",
      origin_view: view,
      owner_key: view.owner_key,
      conversation_id: view.conversation_id,
      selection_digest: selection.selection_digest,
      request_context_digest: selection.request_context_digest,
      created_at: createdAt,
      input_text: params.message,
      input_images: [],
      result_text: content,
      state,
      ...(admission
        ? {
            input_id: admission.input_message_id,
            admission: historyAdmissionReference(admission)
          }
        : {}),
      ...(resultId ? { assistant_id: resultId } : {})
    })
    await saveHistoryTurnRecovery(scope, view, record("dispatching"))
    try {
      if (!sameView() || !leaseValid()) throw new Error("stale_selection")
      dispatched = true
      // The transport sets stream=true. Never give it the frozen provenance object.
      const wire = {
        ...(structuredClone(prepared.payload) as ChatCompletionRequest),
        tldw_history_selection_v1: structuredClone(selection)
      }
      for await (const chunk of tldwClient.streamChatCompletion(wire, {
        signal: snapshot.scopeSignal,
        requestScope: snapshot.requestScope,
        scope: owner.scope,
        streamIdleTimeoutMs: 60_000
      })) {
        if (!authValid() || signal.aborted)
          throw new Error("native_history_consumption_interrupted")
        if (extractStreamTransportInterruption(chunk))
          throw new Error("native_history_stream_interrupted")
        if (chunk?.error)
          throw new Error("native_history_stream_interrupted", { cause: chunk.error })
        if (chunk?.tldw_history_admission_v1 !== undefined) {
          const accepted = parseNativeHistoryAdmission(
            owner,
            selection,
            chunk.tldw_history_admission_v1
          )
          if (
            admission &&
            JSON.stringify(admission) !== JSON.stringify(accepted)
          )
            throw new Error("invalid_history_admission")
          admission = accepted
          await saveHistoryTurnRecovery(scope, view, record("accepted_unsent"))
        }
        if (chunk?.tldw_message_id !== undefined) {
          if (
            !admission ||
            typeof chunk.tldw_message_id !== "string" ||
            !chunk.tldw_message_id.trim() ||
            chunk.tldw_conversation_id !== owner.conversation_id ||
            chunk.tldw_message_id === admission.input_message_id ||
            (resultId && resultId !== chunk.tldw_message_id)
          )
            throw new Error("invalid_history_settlement")
          resultId = chunk.tldw_message_id
        }
        const delta = chunk?.choices?.[0]?.delta?.content
        if (typeof delta === "string") content += delta
        if (sameView() && admission && content)
          params.setMessages([
            ...base.messages,
            {
              id: admission.input_message_id,
              isBot: false,
              name: "You",
              message: params.message,
              sources: [],
              createdAt
            },
            {
              id: displayId,
              isBot: true,
              name: "Assistant",
              message: content,
              sources: [],
              createdAt
            }
          ])
      }
      if (!admission || !resultId)
        throw new Error("native_history_persistence_unacknowledged")
      await dismissHistoryTurnRecovery(scope, view, operationId, "completed")
      if (creationHandle) await markChatCreationSettled(creationHandle, owner.conversation_id)
      if (sameView()) {
        followAttempted = true
        await controller.followResult(view, resultId)
      }
    } catch (error) {
      if (admission && resultId) {
        // A later disconnect cannot erase an already verified owner settlement.
        await dismissHistoryTurnRecovery(scope, view, operationId, "completed")
        if (creationHandle) await markChatCreationSettled(creationHandle, owner.conversation_id)
        if (sameView() && !followAttempted)
          await controller.followResult(view, resultId)
        return
      }
      if (dispatched) {
        await saveHistoryTurnRecovery(
          scope,
          view,
          record(
            content
              ? "generated_unsaved"
              : admission
                ? "accepted_unsent"
                : "unknown"
          )
        )
        if (sameView()) {
          // This error is display-only; the retained receipt owns any uncertain outcome.
          params.setMessages([...base.messages, {
            id: displayId, isBot: true, name: "Assistant", sources: [], createdAt,
            message: buildCharacterChatAssistantErrorContent(undefined, error, params.t)
          }])
          await controller.refreshRecovery()
        }
      } else await dismissHistoryTurnRecovery(scope, view, operationId)
      throw error
    }
  } finally {
    snapshot.release()
    params.releaseActivity()
  }
}


/** Explicit recovery only loads actual owned rows; it never creates or submits retained input. */
export const reviewNativeChatCreation = async (params: {
  controller: HistorySelectionController
  handle: ChatDraftHandle
  isCurrent: () => boolean
  onAccepted: (chat: Awaited<ReturnType<typeof tldwClient.getChat>>) => void
  retainInput?: () => Promise<boolean>
}) => {
  const source = useStoreMessageOption.getState()
  const address = source.chatDraftAddress
  if (!address || address.kind !== "fresh" || chatDraftAddressKey(address) !== chatDraftAddressKey(params.handle.address)) return false
  const session = usePlaygroundSessionStore.getState()
  if (session.serverChatSelectionIntent) return false
  const sourceFence = params.controller.fence()
  const navigationCurrent = () => useStoreMessageOption.getState().serverChatId === source.serverChatId &&
    usePlaygroundSessionStore.getState().serverChatSelectionIntent === session.serverChatSelectionIntent &&
    usePlaygroundSessionStore.getState().restoreRevision === session.restoreRevision &&
    usePlaygroundSessionStore.getState().sourceSelectionRevision === session.sourceSelectionRevision
  const bind = captureChatDraftLoad("bind")
  if (params.retainInput && !(await params.retainInput())) return false
  if (!bind.isCurrent() || !params.isCurrent() || !sourceFence() || !navigationCurrent()) return false
  const snapshot = await loadServicePromptSnapshot([])
  const authCurrent = () => !snapshot.scopeSignal.aborted && !snapshot.scopeInvalidatedSignal.aborted
  const current = () => bind.isCurrent() && params.isCurrent() && authCurrent() && navigationCurrent()
  const preparing = () => sourceFence() && current()
  try {
    if (!preparing() || chatDraftOwnerKey(snapshot.requestScope) !== params.handle.ownerHash || params.handle.address.kind !== "fresh") return false
    const attempt = await readChatCreationRecovery(params.handle)
    if (!attempt?.chatId || attempt.state === "unconfirmed" || !preparing()) return false
    const scope = structuredClone(params.handle.address.scope)
    const chat = await tldwClient.getChat(attempt.chatId, { scope, requestScope: snapshot.requestScope, signal: snapshot.scopeSignal })
    if (!preparing() || chat.id !== attempt.chatId) return false
    let acceptedReceipt: HistoryLoadReceipt | undefined
    let phase: "preparing" | "accepted" | "rejected" = "preparing"
    const loaded = await params.controller.loadConversation({ serverChatId: attempt.chatId, scope, isCurrent: () => bind.isCurrent() && (phase === "accepted" || (phase === "preparing" && current())) }, null, receipt => {
      phase = "rejected"
      const now = params.controller.getCurrent()
      if (!current() || receipt.owner !== now.owner || receipt.view !== now.view || receipt.owner.kind !== "native" ||
        receipt.owner.conversation_id !== attempt.chatId || chatDraftOwnerKey(receipt.owner.request_scope) !== params.handle.ownerHash) return
      phase = "preparing"
      if (!receipt.owner.validate_lease() || !bind(receipt)) { phase = "rejected"; return }
      acceptedReceipt = receipt
      phase = "accepted"
      params.onAccepted(chat)
    })
    if (!loaded || !acceptedReceipt || !bind.isCurrent()) return false
    const destination = params.controller.fence()
    await markChatCreationBound(attempt.handle, attempt.chatId)
    const now = params.controller.getCurrent()
    return destination() && bind.isCurrent() && now.owner === acceptedReceipt.owner && now.view === acceptedReceipt.view
  } finally { snapshot.release() }
}
