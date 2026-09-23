import type { Dispatch, SetStateAction } from "react"
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
import {
  parseProviderQualifiedModelSelection,
  resolveExplicitProviderForSelectedModel
} from "@/utils/resolve-api-provider"
import { extractStreamTransportInterruption } from "@/utils/extract-token-from-chunk"

export type NativeHistoryCharacterSendParams = {
  controller: HistorySelectionController
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
  if (params.temporary || params.historyId === "temp")
    throw new Error("temporary_history_unavailable")
  if (params.image) throw new Error("native_history_assets_unsupported")
  if (params.unsupportedContext)
    throw new Error("native_history_explicit_context_unsupported")
  if (!params.message.trim()) throw new Error("native_history_input_required")
  const settingsState = useStoreChatModelSettings.getState()
  const toolsState = useMcpToolsStore.getState()
  if (
    params.toolChoice === "required" ||
    (toolsState.tools.length && params.toolChoice !== "none")
  )
    throw new Error("native_history_client_tools_unsupported")
  // Preserve saved sampling defaults when the composer has no explicit override.
  const settings = { ...params.settings }
  const unsupportedSettings: Array<keyof ChatModelSettings> = [
    "f16KV",
    "keepAlive",
    "logitsAll",
    "mirostat",
    "mirostatEta",
    "mirostatTau",
    "numBatch",
    "numCtx",
    "numGpu",
    "numGqa",
    "numKeep",
    "numThread",
    "penalizeNewline",
    "repeatLastN",
    "repeatPenalty",
    "ropeFrequencyBase",
    "ropeFrequencyScale",
    "tfsZ",
    "topK",
    "typicalP",
    "useMLock",
    "useMMap",
    "useMlock",
    "vocabOnly",
    "seed",
    "minP",
    "thinking",
    "historyMessageLimit",
    "historyMessageOrder"
  ]
  const unsupportedSetting = unsupportedSettings.find(
    (key) => settings[key] !== undefined && settings[key] !== null
  )
  if (unsupportedSetting)
    throw new Error(`native_history_setting_unsupported:${unsupportedSetting}`)
  if (
    settings.reasoningEffort &&
    !["low", "medium", "high"].includes(settings.reasoningEffort)
  )
    throw new Error("native_history_reasoning_effort_unsupported")
  if (settings.extraBody?.trim() || settings.extraHeaders?.trim())
    throw new Error("native_history_extra_parameters_unsupported")
  if (settings.systemPromptTemplateId)
    throw new Error("native_history_prompt_template_unsupported")
  const actor = await getActorSettingsForChat({
    historyId: params.historyId ?? null,
    serverChatId: params.serverChatId ?? null
  })
  if (actor?.isEnabled)
    throw new Error("native_history_actor_context_unsupported")
  const selected = parseProviderQualifiedModelSelection(params.model)
  const model = selected.modelId.replace(/^tldw:/, "").trim()
  const provider = resolveExplicitProviderForSelectedModel({
    currentSelectedModel: params.currentModel ?? params.model,
    requestedSelectedModel: params.model,
    explicitProvider: settings.apiProvider
  })
  if (!model || !provider)
    throw new Error("native_history_explicit_model_provider_required")
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
    authValid() &&
    !signal.aborted &&
    useStoreChatModelSettings.getState() === settingsState &&
    useMcpToolsStore.getState() === toolsState
  try {
    if (!params.originIsCurrent() || !leaseValid())
      throw new Error("stale_selection")
    let current = controller.getCurrent()
    if (!current.owner && current.status === "idle") {
      if (!params.originIsCurrent() || !leaseValid())
        throw new Error("stale_selection")
      let chatId = params.serverChatId
      let createdCharacterId: string | number | undefined
      if (!chatId && !params.historyId) {
        if (!params.characterId)
          throw new Error("native_history_character_required")
        const created = await tldwClient.createChat(
          { character_id: params.characterId },
          {
            scope: params.scope,
            signal: snapshot.scopeSignal,
            requestScope: snapshot.requestScope
          }
        )
        // A late creation ACK must never bind the newly opened view or account.
        if (!params.originIsCurrent() || !leaseValid())
          throw new Error("stale_selection")
        chatId = created.id
        createdCharacterId = params.characterId
        if (!chatId) throw new Error("native_history_creation_unacknowledged")
      }
      let receipt: HistoryLoadReceipt | undefined
      const loaded = await controller.loadConversation(
        {
          historyId: params.historyId,
          serverChatId: chatId,
          scope: params.scope
        },
        null,
        (value) => {
          receipt = value
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
      if (chatId) params.setServerChatId(chatId)
      if (createdCharacterId != null) params.onCreated?.(createdCharacterId)
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
        if (extractStreamTransportInterruption(chunk) || chunk?.error)
          throw new Error("native_history_stream_interrupted")
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
      if (sameView()) {
        followAttempted = true
        await controller.followResult(view, resultId)
      }
    } catch (error) {
      if (admission && resultId) {
        // A later disconnect cannot erase an already verified owner settlement.
        await dismissHistoryTurnRecovery(scope, view, operationId, "completed")
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
          params.setMessages(base.messages)
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
