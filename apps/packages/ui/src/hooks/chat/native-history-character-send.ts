import type { GreetingSeed } from "@/utils/character-greetings"
import { createChatWithRecovery, acknowledgeChatCreation, markChatCreationBound, readChatCreationRecovery, markChatCreationSettled } from "@/services/pending-chat-promotion"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import { chatDraftAddressKey, isChatDraftHandle, type ChatDraftHandle } from "@/utils/chat-draft-address"
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
import { requireServerSettingsResponse } from "@/services/chat-settings"
import { NativeHistoryContextRefusal, NativeHistorySettingsReadError, validateNativeCharacterConfiguration } from "./native-character-preflight"
import { extractStreamTransportInterruption } from "@/utils/extract-token-from-chunk"

export type NativeHistoryCharacterSendParams = {
  controller: HistorySelectionController
  /** An independent popup owns its draft and never binds the full composer. */
  privateDraft?: { handle: ChatDraftHandle; isCurrent: () => boolean }
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
  /** Explicit assistant-only startup; never submitted as a user turn. */
  greetingSeed?: GreetingSeed
  greetingIsCurrent?: () => boolean
  t: TFunction
  unsupportedContext?: boolean
  setServerChatId: (id: string) => void
  onCreated?: (characterId: string | number) => void
  setMessages: Dispatch<SetStateAction<Message[]>>
  releaseActivity: () => void
}

// These current backend context refusals precede input admission and provider dispatch.
// A generic conflict, future code, or observed stream evidence cannot prove refusal.
const isSavedContextRefusal = (error: unknown): boolean => {
  const response = error as {
    status?: unknown
    details?: { detail?: { status?: unknown; code?: unknown } }
  } | null
  const detail = response?.details?.detail
  return (
    response?.status === 409 &&
    detail?.status === "unsupported_history_capability" &&
    typeof detail.code === "string" &&
    [
      "invalid_materialized_behavior",
      "saved_settings",
      "materialized_binding",
      "behavior_controls",
      "author_note",
      "assistant_overlay",
      "world_books",
      "memory",
      "greeting",
      "unbound_snapshot",
      "persona",
      "snapshot_missing",
      "snapshot_invalid",
      "participants",
      "source_identity",
      "exemplars",
      "default_memory",
      "generation_defaults",
      "prompt_extensions",
      "character_extensions",
      "prompt_preset",
      "unbound_sampling",
      "skills_registry",
      "skills_unresolved",
      "skills_recovery",
      "skills_installed",
      "prompt_template"
    ].some((suffix) => detail.code === `unsupported_history_context_${suffix}`)
  )
}

/** One native dispatch owns admission and settlement. Browser cancellation is never a rollback receipt. */
export const sendNativeHistoryCharacter = async (
  params: NativeHistoryCharacterSendParams
): Promise<{ content: string; matchesPreview: boolean } | void> => {
  const { controller, signal } = params
  const greetingSeed = params.greetingSeed ? structuredClone(params.greetingSeed) : undefined
  const privateHandle = params.privateDraft ? structuredClone(params.privateDraft.handle) : undefined
  if (privateHandle && (!isChatDraftHandle(privateHandle) || privateHandle.address.kind !== "fresh" || privateHandle.address.temporary))
    throw new Error("chat_creation_source_invalid")
  const privateCurrent = () => params.privateDraft?.isCurrent() !== false
  const sourceAddress = privateHandle ? privateHandle.address : useStoreMessageOption.getState().chatDraftAddress
  const sourceDraft = sourceAddress ? structuredClone(sourceAddress) : null
  const creationScope = params.scope ? structuredClone(params.scope) : (privateHandle || greetingSeed) && sourceDraft?.kind === "fresh" ? structuredClone(sourceDraft.scope) : undefined
  if (privateHandle && params.scope && chatDraftAddressKey({ ...privateHandle.address, scope: params.scope } as ChatDraftHandle["address"]) !== chatDraftAddressKey(privateHandle.address))
    throw new Error("chat_creation_source_invalid")
  const acceptDraftBind = !privateHandle && !params.historyId && !params.serverChatId && sourceDraft?.kind === "fresh" ? captureChatDraftLoad("bind") : null
  if (!greetingSeed && !params.message.trim()) throw new Error("native_history_input_required")
  if (greetingSeed && (params.message || !greetingSeed.rawText?.trim() ||
      params.historyId || params.serverChatId || sourceDraft?.kind !== "fresh" ||
      controller.getCurrent().status !== "idle" || controller.getCurrent().owner))
    throw new Error("native_history_greeting_requires_fresh")
  if (!params.originIsCurrent()) throw new Error("stale_selection")
  const greetingCurrent = () => !greetingSeed || params.greetingIsCurrent?.() !== false
  if (!greetingCurrent()) throw new Error("stale_selection")
  const settingsState = useStoreChatModelSettings.getState()
  const toolsState = useMcpToolsStore.getState()
  const capturedSettings = { ...params.settings }
  const actor = privateHandle ? null : await getActorSettingsForChat({
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
  if (privateHandle && privateHandle.ownerHash !== chatDraftOwnerKey(snapshot.requestScope)) {
    snapshot.release()
    throw new Error("request_config_scope_changed")
  }
  const leaseValid = () =>
    acceptDraftBind?.isCurrent() !== false && privateCurrent() &&
    (!privateHandle || params.originIsCurrent()) &&
    authValid() &&
    !signal.aborted &&
    useStoreChatModelSettings.getState() === settingsState &&
    useMcpToolsStore.getState() === toolsState
  let greetingResult: { content: string; matchesPreview: boolean } | undefined
  try {
    if (!params.originIsCurrent() || !greetingCurrent() || !leaseValid())
      throw new Error("stale_selection")
    let creationHandle: ChatDraftHandle | undefined
    let current = controller.getCurrent()
    let selectionCurrent = params.originIsCurrent
    let targetChatId = params.serverChatId
    if (!current.owner && current.status === "idle") {
      if (!params.originIsCurrent() || !greetingCurrent() || !leaseValid())
        throw new Error("stale_selection")
      let chatId = params.serverChatId
      let createdCharacterId: string | number | undefined
      if (!chatId && !params.historyId) {
        if (!params.characterId)
          throw new Error("native_history_character_required")
        if (!sourceDraft || sourceDraft.kind !== "fresh") throw new Error("chat_creation_source_invalid")
        creationHandle = privateHandle ?? { address: sourceDraft, ownerHash: chatDraftOwnerKey(snapshot.requestScope) }
        const created = await createChatWithRecovery({
          snapshot,
          handle: creationHandle,
          scope: creationScope,
          surface: params.surface,
          inputText: params.message,
          setup: {
            characterId: params.characterId,
            provider,
            model,
            temperature: settings.temperature,
            topP: settings.topP
          },
          isCurrent: () =>
            params.originIsCurrent() && greetingCurrent() && leaseValid(),
          create: (id) =>
            tldwClient.createChat(
              {
                id,
                character_id: params.characterId,
                provider,
                model,
                ...(settings.temperature != null
                  ? { temperature: settings.temperature }
                  : {}),
                ...(settings.topP != null ? { top_p: settings.topP } : {})
              },
              {
                scope: creationScope,
                signal: snapshot.scopeSignal,
                requestScope: snapshot.requestScope,
                ...(greetingSeed
                  ? {
                      seedFirstMessage: true,
                      greetingStrategy: greetingSeed.greetingStrategy,
                      ...(greetingSeed.greetingStrategy === "alternate_index"
                        ? { alternateIndex: greetingSeed.alternateIndex }
                        : {})
                    }
                  : {})
              }
            )
        })

        // A late creation ACK must never bind the newly opened view or account.
        if (!params.originIsCurrent() || !greetingCurrent() || !leaseValid())
          throw new Error("stale_selection")
        chatId = created.id
        createdCharacterId = params.characterId
        if (!chatId) throw new Error("native_history_creation_unacknowledged")
      }
      if (greetingSeed) {
        const chat = await tldwClient.getChat(chatId!, { scope: creationScope, requestScope: snapshot.requestScope, signal: snapshot.scopeSignal })
        if (!params.originIsCurrent() || !greetingCurrent() || !leaseValid()) throw new Error("stale_selection")
        if (chat.id !== chatId || String(chat.character_id) !== String(params.characterId))
          throw new Error("owner_conversation_mismatch")
      }
      let receipt: HistoryLoadReceipt | undefined
      let loadPhase: "preparing" | "accepted" | "rejected" = "preparing"
      const loaded = await controller.loadConversation(
        {
          historyId: params.historyId,
          serverChatId: chatId,
          scope: creationScope,
          ...(greetingSeed ? { qualifyCapture: (capture, owner) => {
            if (!greetingCurrent() || !leaseValid()) throw new Error("stale_selection")
            if (owner.kind !== "native" || owner.conversation_id !== chatId || !owner.validate_lease() ||
                chatDraftOwnerKey(owner.request_scope) !== chatDraftOwnerKey(snapshot.requestScope))
              throw new Error("owner_conversation_mismatch")
            if (capture.status !== "captured") throw new Error(capture.code)
            const row = capture.rows[0]
            const content = capture.selected_content[0]
            if (capture.snapshot.nodes.length !== 1 || capture.rows.length !== 1 || capture.selected_content.length !== 1 ||
                row.role !== "assistant" || !row.settled || row.parent_id !== null ||
                (row.conversation_id != null && row.conversation_id !== chatId) || content.id !== row.id ||
                content.revision !== row.revision || !content.message.trim() || content.images.length)
              throw new Error("native_history_greeting_not_stored")
            greetingResult = { content: content.message, matchesPreview: content.message === greetingSeed.rawText }
          } } : {}),
          // Preparation ends at the qualified bind; later settings or Stop
          // must not revoke the saved owner's independent account lease.
          isCurrent: () => acceptDraftBind?.isCurrent() !== false && privateCurrent() &&
            (loadPhase === "accepted" || (loadPhase === "preparing" && greetingCurrent() && leaseValid() &&
              (!privateHandle || params.originIsCurrent())))
        },
        null,
        (value) => {
          loadPhase = "rejected"
          const now = controller.getCurrent()
          if (!greetingCurrent() || !leaseValid() || (privateHandle && (!params.originIsCurrent() ||
              value.owner.kind !== "native" || chatDraftOwnerKey(value.owner.request_scope) !== privateHandle.ownerHash)) ||
            now.owner !== value.owner || now.view !== value.view ||
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
        throw new Error(
          !receipt && current.status !== "ready" && current.error &&
          params.originIsCurrent() && greetingCurrent() && leaseValid() &&
          current.owner?.kind === "native" && current.owner.validate_lease() &&
          (!chatId || current.owner.conversation_id === chatId) &&
          chatDraftOwnerKey(current.owner.request_scope) === chatDraftOwnerKey(snapshot.requestScope)
            ? current.error
            : "stale_selection"
        )
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
    if (greetingSeed) {
      if (!creationHandle || !greetingResult) throw new Error("native_history_greeting_not_stored")
      await markChatCreationSettled(creationHandle, current.owner.conversation_id)
      if (!selectionCurrent() || !leaseValid() || controller.getCurrent().owner !== current.owner || controller.getCurrent().view !== current.view)
        throw new Error("stale_selection")
      return greetingResult
    }
    const originOwner = current.owner
    const originView = current.view
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
    if (privateHandle && !creationHandle) {
      const attempt = await readChatCreationRecovery(privateHandle)
      if (attempt?.state === "bound" && attempt.chatId === owner.conversation_id && attempt.inputText === params.message)
        creationHandle = privateHandle
    }
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
    const preparingIsCurrent = () => sameView() && selectionCurrent() && leaseValid() &&
      controller.getCurrent().owner === originOwner && controller.getCurrent().view === originView && originOwner.validate_lease()
    if (!preparingIsCurrent()) throw new Error("stale_selection")
    let savedSettings: Record<string, unknown>
    try {
      savedSettings = requireServerSettingsResponse(owner.conversation_id,
        await tldwClient.getChatSettings(owner.conversation_id, {
          scope: owner.scope, requestScope: owner.request_scope, signal: snapshot.scopeSignal
        }))
    } catch (cause) {
      throw new NativeHistorySettingsReadError(cause, preparingIsCurrent)
    }
    if (!preparingIsCurrent()) throw new Error("stale_selection")
    const preferred = savedSettings.chatGenerationOverride
    const legacy = savedSettings.generationOverrides
    const overrides = preferred && typeof preferred === "object" && !Array.isArray(preferred)
      ? preferred as Record<string, unknown>
      : legacy && typeof legacy === "object" && !Array.isArray(legacy)
        ? legacy as Record<string, unknown> : null
    const usesSavedSampler = (field: "temperature" | "top_p") =>
      overrides?.enabled !== false && typeof overrides?.[field] === "number" && Number.isFinite(overrides[field])
    const capture = await captureHistorySnapshot(
      owner,
      view,
      "send",
      snapshot.scopeSignal
    )
    if (capture.status !== "captured") throw new Error(capture.code)
    if (!preparingIsCurrent()) throw new Error("stale_selection")
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
        temperature: usesSavedSampler("temperature") ? undefined : settings.temperature,
        maxTokens: settings.numPredict,
        topP: usesSavedSampler("top_p") ? undefined : settings.topP,
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
    const prepared = prepareHistoryContext(request, preparingIsCurrent)
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
    let streamStarted = false
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
      if (!preparingIsCurrent()) throw new Error("stale_selection")
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
        streamStarted = true
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
      if (
        dispatched &&
        !streamStarted &&
        !admission &&
        !resultId &&
        content === "" &&
        isSavedContextRefusal(error)
      ) {
        const acceptedOwner = current.owner
        const refusal = new NativeHistoryContextRefusal(
          error,
          () =>
            selectionCurrent() &&
            privateCurrent() &&
            sameView() &&
            acceptedOwner.validate_lease()
        )
        try {
          // Composer retention is not an acknowledged write; keep the only durable input.
          await saveHistoryTurnRecovery(scope, view, record("refused"))
          if (refusal.isCurrent()) await controller.refreshRecovery()
        } catch {
          // The earlier dispatch record remains recoverable if storage or its account retired.
          console.warn(
            "Could not record or display a confirmed native context refusal."
          )
        }
        throw refusal
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
  expectedOperationId: string
  expectedClientChatId?: string
  isCurrent: () => boolean
  onAccepted: (chat: Awaited<ReturnType<typeof tldwClient.getChat>>) => void
  retainInput?: () => Promise<boolean>
  copyInput?: () => Promise<boolean>
  inputText?: string
}) => {
  const source = useStoreMessageOption.getState()
  const address = source.chatDraftAddress
  if (
    !address ||
    address.kind !== "fresh" ||
    chatDraftAddressKey(address) !== chatDraftAddressKey(params.handle.address)
  )
    return false
  const session = usePlaygroundSessionStore.getState()
  if (session.serverChatSelectionIntent) return false
  const sourceFence = params.controller.fence()
  const navigationCurrent = () =>
    useStoreMessageOption.getState().serverChatId === source.serverChatId &&
    usePlaygroundSessionStore.getState().serverChatSelectionIntent ===
      session.serverChatSelectionIntent &&
    usePlaygroundSessionStore.getState().restoreRevision ===
      session.restoreRevision &&
    usePlaygroundSessionStore.getState().sourceSelectionRevision ===
      session.sourceSelectionRevision
  let bind = captureChatDraftLoad("bind")
  if (params.retainInput && !(await params.retainInput())) return false
  if (
    !bind.isCurrent() ||
    !params.isCurrent() ||
    !sourceFence() ||
    !navigationCurrent()
  )
    return false
  const snapshot = await loadServicePromptSnapshot([])
  const authCurrent = () =>
    !snapshot.scopeSignal.aborted && !snapshot.scopeInvalidatedSignal.aborted
  const current = () =>
    bind.isCurrent() &&
    params.isCurrent() &&
    authCurrent() &&
    navigationCurrent()
  const preparing = () => sourceFence() && current()
  try {
    if (
      !preparing() ||
      chatDraftOwnerKey(snapshot.requestScope) !== params.handle.ownerHash ||
      params.handle.address.kind !== "fresh"
    )
      return false
    const attempt = await readChatCreationRecovery(params.handle)
    if (
      !attempt ||
      attempt.handle.address.kind !== "fresh" ||
      attempt.operationId !== params.expectedOperationId ||
      attempt.clientChatId !== params.expectedClientChatId ||
      !preparing()
    )
      return false
    const chatId =
      attempt.chatId ??
      (attempt.state === "unconfirmed" ? attempt.clientChatId : undefined)
    if (!chatId) return false
    const sameAttempt = async () => {
      const now = await readChatCreationRecovery(params.handle)
      return (
        now?.operationId === attempt.operationId &&
        now.clientChatId === attempt.clientChatId &&
        chatDraftAddressKey(now.handle.address) ===
          chatDraftAddressKey(attempt.handle.address) &&
        (now.chatId ??
          (now.state === "unconfirmed" ? now.clientChatId : undefined)) ===
          chatId
      )
    }
    if (
      attempt.state === "settled" ||
      (params.inputText !== undefined && params.inputText !== attempt.inputText)
    ) {
      // Different retained input is a separate draft, not this chat's original first-ID bind.
      if (!params.copyInput || !(await params.copyInput()) || !preparing())
        return false
      bind = captureChatDraftLoad("switch")
    }
    if (!(await sameAttempt()) || !preparing()) return false
    const scope = structuredClone(attempt.handle.address.scope)
    const chat = await tldwClient.getChat(chatId, {
      scope,
      requestScope: snapshot.requestScope,
      signal: snapshot.scopeSignal
    })
    if (
      !preparing() ||
      chat.id !== chatId ||
      !(await sameAttempt()) ||
      !preparing()
    )
      return false
    if (
      !(await acknowledgeChatCreation(attempt, chatId)) ||
      !(await sameAttempt()) ||
      !preparing()
    )
      return false
    let acceptedReceipt: HistoryLoadReceipt | undefined
    let phase: "preparing" | "accepted" | "rejected" = "preparing"
    const loaded = await params.controller.loadConversation(
      {
        serverChatId: chatId,
        scope,
        isCurrent: () =>
          bind.isCurrent() &&
          (phase === "accepted" || (phase === "preparing" && current()))
      },
      null,
      (receipt) => {
        phase = "rejected"
        const now = params.controller.getCurrent()
        if (
          !current() ||
          receipt.owner !== now.owner ||
          receipt.view !== now.view ||
          receipt.owner.kind !== "native" ||
          receipt.owner.conversation_id !== chatId ||
          chatDraftOwnerKey(receipt.owner.request_scope) !==
            params.handle.ownerHash
        )
          return
        phase = "preparing"
        if (!receipt.owner.validate_lease() || !bind(receipt)) {
          phase = "rejected"
          return
        }
        acceptedReceipt = receipt
        phase = "accepted"
        params.onAccepted(chat)
      }
    )
    if (!loaded || !acceptedReceipt || !bind.isCurrent()) return false
    const destination = params.controller.fence()
    await markChatCreationBound(attempt.handle, chatId, attempt.operationId)
    const now = params.controller.getCurrent()
    return (
      destination() &&
      bind.isCurrent() &&
      now.owner === acceptedReceipt.owner &&
      now.view === acceptedReceipt.view
    )
  } finally {
    snapshot.release()
  }
}
