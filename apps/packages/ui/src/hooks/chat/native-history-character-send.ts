import { isSavedRoleplayConversation } from "./roleplay-resume-source"
import type { NativeHistoryEditResend, NativeHistoryRegenerate } from "@/types/chat-modes"
import type { GreetingSeed } from "@/utils/character-greetings"
import { createChatWithRecovery, acknowledgeChatCreation, markChatCreationBound, readChatCreationRecovery, markChatCreationSettled } from "@/services/pending-chat-promotion"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import { chatDraftAddressKey, isChatDraftHandle, type ChatDraftHandle } from "@/utils/chat-draft-address"
import type { Dispatch, SetStateAction } from "react"
import { captureChatDraftLoad, captureChatDraftSource } from "./chat-draft-transition"
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
import { copyActorSceneToBoundChat } from "@/services/actor-settings"
import { captureActorSceneSend, captureBoundActorSceneTarget, bindActorSceneAfterLoad, type ActorSceneSend } from "./actor-scene-source"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { loadServicePromptSnapshot } from "@/services/service-prompts"
import {
  assertHistoryConversationTarget,
  captureHistorySnapshot,
  nativeHistoryNeedsBranchChoice,
  requireNativeHistoryEditAdmission,
  prepareHistoryContext,
  finalizeHistorySelection,
  parseNativeHistoryAdmission,
  historyAdmissionReference,
  isDefinitiveNativeBranchRejection,
  isDefinitiveNativeEditRejection,
  nativeEditResendView,
  nativeRegenerateView,
  nativeRegenerationMessage,
  parseNativeRegenerationAdmission
} from "@/services/chat-history-selection"
import {
  tldwClient,
  type ChatCompletionRequest
} from "@/services/tldw/TldwApiClient"
import { prepareChatCompletionRequest } from "@/services/tldw/TldwChat"
import { requireServerSettingsResponse } from "@/services/chat-settings"
import { NativeHistoryBranchChoice, NativeHistoryBranchRefusal, NativeHistoryContextRefusal, NativeHistoryEditRefusal, NativeHistorySettingsReadError, NativeHistoryRoleplaySendError, validateNativeCharacterConfiguration } from "./native-character-preflight"
import { extractStreamTransportInterruption } from "@/utils/extract-token-from-chunk"
import { consumeStreamingChunk } from "@/utils/streaming-chunks"

export type NativeHistoryCharacterSendParams = {
  controller: HistorySelectionController
  /** An independent popup owns its draft and never binds the full composer. */
  privateDraft?: { handle: ChatDraftHandle; requestScope: ServicePromptRequestScope; isCurrent: () => boolean }
  actorSceneSend?: ActorSceneSend
  surface: "playground" | "sidepanel"
  originIsCurrent: () => boolean
  sourceIsCurrent?: () => boolean
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
  historyEditResend?: NativeHistoryEditResend
  historyRegenerate?: NativeHistoryRegenerate
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
  const editResend = params.historyEditResend
  const regenerate = params.historyRegenerate
  const explicitAction = editResend ?? regenerate
  if (regenerate && (editResend || params.greetingSeed || params.privateDraft ||
      controller.getCurrent().status !== "ready" || controller.getCurrent().owner?.kind !== "native"))
    throw new Error("unsupported_history_regeneration")
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
  const capturedDraftIsCurrent = privateHandle ? privateCurrent : acceptDraftBind?.isCurrent ?? captureChatDraftSource().isCurrent
  const draftIsCurrent = () => params.sourceIsCurrent?.() !== false && capturedDraftIsCurrent()
  if (!greetingSeed && !regenerate && !params.message.trim()) throw new Error("native_history_input_required")
  if (greetingSeed && (params.message || !greetingSeed.rawText?.trim() ||
      params.historyId || params.serverChatId || sourceDraft?.kind !== "fresh" ||
      controller.getCurrent().status !== "idle" || controller.getCurrent().owner))
    throw new Error("native_history_greeting_requires_fresh")
  if (!params.originIsCurrent()) throw new Error("stale_selection")
  const greetingCurrent = () => !greetingSeed || params.greetingIsCurrent?.() !== false
  if (!greetingCurrent()) throw new Error("stale_selection")
  const resumeOrigin = controller.getCurrent()
  if (
    !explicitAction &&
    !greetingSeed &&
    resumeOrigin.owner?.kind === "native" &&
    (isSavedRoleplayConversation(controller) || params.privateDraft)
  ) {
    const owner = resumeOrigin.owner, view = resumeOrigin.view
    const originalPreparationIsCurrent = () => params.originIsCurrent() && owner.validate_lease() &&
      controller.getCurrent().owner === owner && controller.getCurrent().view === view
    try {
      const { sendSavedRoleplay } = await import("./roleplay-resume-send")
      if (!originalPreparationIsCurrent()) throw new Error("stale_selection")
      if (privateHandle && (privateHandle.ownerHash !== chatDraftOwnerKey(owner.request_scope) ||
          privateHandle.ownerHash !== chatDraftOwnerKey(params.privateDraft!.requestScope)))
        throw new Error("request_config_scope_changed")
      const attempt = privateHandle ? await readChatCreationRecovery(privateHandle) : undefined
      if (!originalPreparationIsCurrent()) throw new Error("stale_selection")
      const saved = await sendSavedRoleplay({
        controller,
        signal,
        originIsCurrent: originalPreparationIsCurrent,
        sourceIsCurrent: () =>
          draftIsCurrent() && params.sourceIsCurrent?.() !== false,
        serverChatId: params.serverChatId,
        historyId: params.historyId,
        message: params.message,
        image: params.image,
        unsupportedContext: params.unsupportedContext,
        setMessages: params.setMessages
      })
      if (saved && privateHandle && attempt?.state === "bound" &&
          attempt.chatId === owner.conversation_id && attempt.inputText === params.message)
        await markChatCreationSettled(privateHandle, owner.conversation_id, attempt.operationId)
      return
    } finally {
      params.releaseActivity()
    }
  }
  const settingsState = useStoreChatModelSettings.getState()
  const toolsState = useMcpToolsStore.getState()
  const capturedSettings = { ...params.settings }
  const initialOwner = controller.getCurrent().owner
  const sceneCapture = params.actorSceneSend || privateHandle ? null : captureActorSceneSend(controller, () => params.originIsCurrent() && draftIsCurrent())
  const actorSceneSend = params.actorSceneSend ?? await sceneCapture?.read()
  if (!actorSceneSend) throw new Error("actor_settings_unavailable")
  if (!actorSceneSend.isCurrent()) throw new Error("actor_settings_scope_changed")
  const { model, provider, settings } = validateNativeCharacterConfiguration({
    ...params,
    image: regenerate ? "" : params.image,
    settings: capturedSettings,
    temporary: params.temporary || params.historyId === "temp",
    clientToolCount: toolsState.tools.length,
    actorEnabled: actorSceneSend.scene.settings.isEnabled
  })
  if (!params.originIsCurrent()) throw new Error("stale_selection")
  if (initialOwner?.kind === "local")
    throw new Error("native_history_requires_server_owner")
  if (initialOwner?.kind === "unavailable") throw new Error(initialOwner.code)
  const snapshot = await loadServicePromptSnapshot([], {
    signal,
    requestScope: actorSceneSend.requestScope
  })
  const authValid = () => !snapshot.scopeInvalidatedSignal.aborted
  if (!actorSceneSend.isCurrent() || chatDraftOwnerKey(actorSceneSend.requestScope) !== chatDraftOwnerKey(snapshot.requestScope) ||
      (privateHandle && (privateHandle.ownerHash !== chatDraftOwnerKey(snapshot.requestScope) ||
        chatDraftOwnerKey(params.privateDraft!.requestScope) !== chatDraftOwnerKey(snapshot.requestScope)))) {
    snapshot.release()
    throw new Error("request_config_scope_changed")
  }
  const leaseValid = () =>
    draftIsCurrent() && acceptDraftBind?.isCurrent() !== false && privateCurrent() &&
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
      let adoptionIsCurrent: (() => boolean) | undefined
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
          adoptionIsCurrent = controller.fence()
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
      if ((acceptDraftBind || privateHandle) && actorSceneSend.scene.target.address?.kind === "fresh") {
        const acceptedReceipt = receipt
        const bindingIsCurrent = () => (privateHandle ? privateCurrent() : acceptDraftBind!.isCurrent()) &&
          actorSceneSend.scene.target.account?.isCurrent() === true &&
          controller.getCurrent().owner === acceptedReceipt.owner &&
          controller.getCurrent().view?.owner_key === acceptedReceipt.view.owner_key &&
          controller.getCurrent().view?.conversation_id === acceptedReceipt.view.conversation_id
        const scenePreparationIsCurrent = () => bindingIsCurrent() && leaseValid() &&
          adoptionIsCurrent?.() === true && controller.getCurrent().view === acceptedReceipt.view
        let copied: boolean
        if (privateHandle) {
          const destination = captureBoundActorSceneTarget(actorSceneSend.scene.target, acceptedReceipt, bindingIsCurrent)
          copied = Boolean(destination && scenePreparationIsCurrent() &&
            (actorSceneSend.scene.originalRecord === null ||
              await copyActorSceneToBoundChat(actorSceneSend.scene, destination, scenePreparationIsCurrent)))
        } else copied = await bindActorSceneAfterLoad(actorSceneSend.scene, acceptedReceipt, bindingIsCurrent, scenePreparationIsCurrent)
        if (!copied || !scenePreparationIsCurrent()) throw new Error("actor_settings_bind_not_committed")
      }
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
    const originalCapture = current.capture
    const originView = current.view
    const historyBranch = Boolean(explicitAction || controller.takeBranchIntent?.(originView))
    const historyEdit = editResend ? Object.freeze({ version: 1 as const, message_id: editResend.message_id, message_revision: editResend.message_revision }) : undefined
    const view = editResend
      ? nativeEditResendView(
          current.owner,
          originView,
          current.capture,
          editResend
        )
      : regenerate
        ? nativeRegenerateView(current.owner, originView, current.capture, regenerate)
        : structuredClone(originView)
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
        authValid() && params.sourceIsCurrent?.() !== false &&
        !!now &&
        now.owner_key === originView.owner_key &&
        now.conversation_id === originView.conversation_id &&
        now.view_session_id === originView.view_session_id &&
        now.selection_revision === originView.selection_revision
      )
    }
    const preparingIsCurrent = () =>
      sameView() &&
      selectionCurrent() &&
      leaseValid() &&
      (!explicitAction ||
        (explicitAction.isCurrent() && explicitAction.isOriginCurrent())) &&
      controller.getCurrent().owner === originOwner &&
      controller.getCurrent().view === originView &&
      originOwner.validate_lease()
    if (!preparingIsCurrent()) throw new Error("stale_selection")
    if (
      !explicitAction &&
      (isSavedRoleplayConversation(controller) ||
        privateHandle ||
        (creationHandle && owner.scope?.type !== "workspace"))
    ) {
      const { sendSavedRoleplay } = await import("./roleplay-resume-send")
      let saved: boolean
      try {
        saved = await sendSavedRoleplay({
          controller,
          signal: snapshot.scopeSignal,
          originIsCurrent: preparingIsCurrent,
          sourceIsCurrent: () =>
            authValid() && params.sourceIsCurrent?.() !== false && draftIsCurrent(),
          serverChatId: owner.conversation_id,
          message: params.message,
          image: params.image,
          unsupportedContext: params.unsupportedContext,
          setMessages: params.setMessages
        })
      } catch (error) {
        if (error instanceof NativeHistoryBranchRefusal || error instanceof NativeHistoryContextRefusal || error instanceof NativeHistorySettingsReadError) throw error
        throw new NativeHistoryRoleplaySendError(error, () => preparingIsCurrent() && draftIsCurrent())
      }
      if (saved && creationHandle)
        await markChatCreationSettled(creationHandle, owner.conversation_id)
      return
    }
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
    const overrides =
      preferred && typeof preferred === "object" && !Array.isArray(preferred)
        ? (preferred as Record<string, unknown>)
        : legacy && typeof legacy === "object" && !Array.isArray(legacy)
          ? (legacy as Record<string, unknown>)
          : null
    const usesSavedSampler = (field: "temperature" | "top_p") =>
      overrides?.enabled !== false && typeof overrides?.[field] === "number" && Number.isFinite(overrides[field])
    let inputVersion: number | undefined
    if (regenerate) {
      const input = originalCapture.rows.at(-2)!
      const content = originalCapture.selected_content.at(-2)!
      if (params.message !== content.message || params.image !== (content.images[0] ?? ""))
        throw new Error("message_changed")
      const row = await tldwClient.getMessage(input.id, {
        scope: owner.scope, requestScope: owner.request_scope, signal: snapshot.scopeSignal
      })
      if (!preparingIsCurrent()) throw new Error("stale_selection")
      if (!row || row.id !== input.id || row.conversation_id !== owner.conversation_id || row.sender !== "user" ||
          row.parent_message_id !== input.parent_id || row.content !== content.message ||
          row.has_image !== Boolean(content.images.length) || !Number.isSafeInteger(row.version) ||
          row.version <= 0 || row.version >= Number.MAX_SAFE_INTEGER)
        throw new Error("message_changed")
      inputVersion = row.version
    }
    const capture = await captureHistorySnapshot(
      owner,
      view,
      "send",
      snapshot.scopeSignal
    )
    if (capture.status !== "captured") throw new Error(capture.code)
    if (!preparingIsCurrent()) throw new Error("stale_selection")
    if (editResend) nativeEditResendView(owner, originView, capture, editResend, originalCapture)
    if (regenerate) {
      nativeRegenerateView(owner, originView, capture, regenerate)
      for (const id of [regenerate.input_message_id, regenerate.message_id]) {
        if (JSON.stringify(originalCapture.selected_content.find(row => row.id === id)) !==
            JSON.stringify(capture.selected_content.find(row => row.id === id)))
          throw new Error("message_changed")
      }
    }
    const userMessage = regenerate ? nativeRegenerationMessage(capture, regenerate, inputVersion!)
      : { role: "user" as const, content: params.message }
    const request = prepareChatCompletionRequest(
      [
        ...(settings.systemPrompt?.trim()
          ? [{ role: "system" as const, content: settings.systemPrompt }]
          : []),
        userMessage
      ],
      {
        model,
        apiProvider: provider,
        saveToDb: true,
        conversationId: owner.conversation_id,
        regenerateFromMessageId: regenerate?.message_id,
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
      explicitAction ? view : controller.getCurrent().view!
    )
    const operationId = crypto.randomUUID()
    const displayId = `native-draft-${operationId}`
    const createdAt = Date.now()
    const base = formatSelectedHistory(capture)
    let admission: HistoryAdmissionV1 | undefined
    let resultId: string | undefined
    let content = ""
    let apiReasoning = false
    let dispatched = false
    let streamStarted = false
    let followAttempted = false
    let resultFollowed = false
    const record = (
      state: HistoryTurnRecovery["state"]
    ): HistoryTurnRecovery => ({
      operation_id: operationId,
      ...(editResend ? { action: "edit_resend" as const } : regenerate ? { action: "regenerate" as const } : {}),
      persistence: "server",
      origin_view: view,
      owner_key: view.owner_key,
      conversation_id: view.conversation_id,
      selection_digest: selection.selection_digest,
      request_context_digest: selection.request_context_digest,
      created_at: createdAt,
      input_text: params.message,
      input_images: regenerate ? [...capture.selected_content.at(-2)!.images] : [],
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
    const finishSettled = async () => {
      if (explicitAction) {
        // A saved result remains recoverable when newer editor input prevents following it.
        await saveHistoryTurnRecovery(scope, view, {
          ...record("generated_unsaved"),
          outcome: "complete",
          settled_message_id: resultId!
        })
        if (
          sameView() &&
          selectionCurrent() &&
          explicitAction.isOriginCurrent() &&
          explicitAction.isCurrent() &&
          !followAttempted
        ) {
          followAttempted = true
          resultFollowed =
            (await controller.followResult(
              originView,
              resultId!,
              explicitAction.isCurrent
            )) !== false
        }
        if (resultFollowed)
          await dismissHistoryTurnRecovery(
            scope,
            view,
            operationId,
            "completed"
          )
        else if (sameView() && explicitAction.isOriginCurrent())
          await controller.refreshRecovery()
      } else {
        await dismissHistoryTurnRecovery(scope, view, operationId, "completed")
        if (sameView() && !followAttempted) {
          followAttempted = true
          await controller.followResult(originView, resultId!)
        }
      }
      if (creationHandle) await markChatCreationSettled(creationHandle, owner.conversation_id)
    }
    try {
      if (!preparingIsCurrent()) throw new Error("stale_selection")
      if (historyEdit) await requireNativeHistoryEditAdmission(owner, capture, "completion", {
        signal: snapshot.scopeSignal, validate_lease: preparingIsCurrent
      })
      if (
        !historyBranch &&
        (await nativeHistoryNeedsBranchChoice(owner, capture, "completion", {
          signal: snapshot.scopeSignal,
          validate_lease: preparingIsCurrent
        }))
      )
        throw new NativeHistoryBranchChoice(
          () =>
            sameView() &&
            selectionCurrent() &&
            privateCurrent() &&
            draftIsCurrent() &&
            originOwner.validate_lease()
        )

      await saveHistoryTurnRecovery(scope, view, record("dispatching"))
      if (!preparingIsCurrent()) throw new Error("stale_selection")
      dispatched = true
      // The transport sets stream=true. Never give it the frozen provenance object.
      const wire = {
        ...(structuredClone(prepared.payload) as ChatCompletionRequest),
        tldw_history_selection_v1: structuredClone(selection),
        tldw_history_branch: historyBranch,
        ...(historyEdit ? { tldw_history_edit_v1: historyEdit } : {})
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
          const accepted = regenerate
            ? parseNativeRegenerationAdmission(owner, capture, inputVersion!, chunk.tldw_history_admission_v1)
            : parseNativeHistoryAdmission(owner, selection, chunk.tldw_history_admission_v1)
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
            (regenerate && originalCapture.snapshot.nodes.some(row => row.id === chunk.tldw_message_id)) ||
            (resultId && resultId !== chunk.tldw_message_id)
          )
            throw new Error("invalid_history_settlement")
          resultId = chunk.tldw_message_id
        }
        const accumulated = consumeStreamingChunk(
          { fullText: content, contentToSave: content, apiReasoning },
          chunk
        )
        content = accumulated.fullText
        apiReasoning = accumulated.apiReasoning
        if (!explicitAction && sameView() && admission && content)
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
      await finishSettled()
    } catch (error) {
      if (admission && resultId) {
        // A later disconnect cannot erase an already verified owner settlement.
        await finishSettled()
        return
      }
      if (
        (error instanceof NativeHistoryBranchChoice || dispatched) &&
        !streamStarted &&
        !admission &&
        !resultId &&
        content === "" &&
        (error instanceof NativeHistoryBranchChoice ||
          isSavedContextRefusal(error) ||
          (!historyBranch && isDefinitiveNativeBranchRejection(error, selection)) ||
          (historyEdit && isDefinitiveNativeEditRejection(error, selection, historyEdit)))
      ) {
        const acceptedOwner = current.owner
        const branchRefusal =
          !historyBranch &&
          (error instanceof NativeHistoryBranchChoice ||
            isDefinitiveNativeBranchRejection(error, selection))
        const Refusal = historyEdit && isDefinitiveNativeEditRejection(error, selection, historyEdit)
          ? NativeHistoryEditRefusal
          : branchRefusal ? NativeHistoryBranchRefusal : NativeHistoryContextRefusal
        const refusal =
          error instanceof NativeHistoryBranchChoice
            ? error
            : new Refusal(
                error,
                () =>
                  selectionCurrent() &&
                  privateCurrent() &&
                  draftIsCurrent() &&
                  sameView() &&
                  acceptedOwner.validate_lease()
              )
        try {
          // Composer retention is not an acknowledged write; keep the only durable input.
          await saveHistoryTurnRecovery(scope, view, record("refused"))
        } catch {
          // The earlier dispatch record remains recoverable if storage or its account retired.
          console.warn("Could not retain a native input that was not admitted.")
        }
        if (refusal.isCurrent()) {
          if (branchRefusal) controller.holdStaleSend?.(originView, draftIsCurrent)
          try {
            await controller.refreshRecovery()
          } catch {
            console.warn("Could not display retained native input.")
          }
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
          if (!explicitAction)
            params.setMessages([
              ...base.messages,
              {
                id: displayId,
                isBot: true,
                name: "Assistant",
                sources: [],
                createdAt,
                message: buildCharacterChatAssistantErrorContent(
                  undefined,
                  error,
                  params.t
                )
              }
            ])
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
