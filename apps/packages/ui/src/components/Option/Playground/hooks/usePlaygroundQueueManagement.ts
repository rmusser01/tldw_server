import React from "react"
import { isSavedRoleplayConversation } from "@/hooks/chat/roleplay-resume-source"
import { captureChatDispatchSource } from "@/hooks/chat/chat-dispatch-source"
import { useStoreChatModelSettings } from "@/store/model"
import { resolveModelSelectorSelection, type ModelSelectorDescriptor } from "@/hooks/playground/modelSelectorUtils"
import { captureActorSceneSend } from "@/hooks/chat/actor-scene-source"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import {
  buildAvailableChatModelIds,
  findUnavailableChatModel,
  normalizeChatModelId
} from "@/utils/chat-model-availability"
import { useComposerQueue } from "@/components/Chat/composer/hooks/useComposerQueue"
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  IMAGE_GENERATION_USER_MESSAGE_TYPE
} from "@/utils/image-generation-chat"
import {
  throwIfChatSubmitUnsuccessful,
  type ChatSubmitResult
} from "@/hooks/chat/chat-action-utils"
import { projectTokenBudget } from "../usage-metrics"
import type { QueuedRequest, QueuedRequestSnapshot } from "@/utils/chat-request-queue"
import type { ChatDocuments } from "@/models/ChatTypes"
import type { UploadedFile } from "@/db/dexie/types"
import {
  prepareChatDocumentAttachmentsForSend
} from "@/services/chat-document-processing"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type PlaygroundQueuedSourceContext = {
  documents?: ChatDocuments
  uploadedFiles?: UploadedFile[]
  imageBackendOverride?: string
  isImageCommand?: boolean
  requestOverrides?: Record<string, unknown>
}

type SubmissionIntent = {
  message: string
  isImageCommand: boolean
  imageBackendOverride?: string
  handled?: boolean
  invalidImageCommand?: boolean
  imageCommandMissingProvider?: boolean
}

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UsePlaygroundQueueManagementDeps {
  composerModels: unknown[] | undefined
  composerModelsLoading?: boolean
  isConnectionReady: boolean
  isSending: boolean
  getDispatchDisabledReason?: () => string | null
  captureDispatchFence?: () => () => boolean
  historySelection?: HistorySelectionController | null
  selectedModel: string | null
  chatMode: string
  webSearch: boolean
  compareMode: boolean
  compareModeActive: boolean
  compareSelectedModels: string[]
  selectedSystemPrompt: string
  selectedQuickPrompt: string | null
  toolChoice: string
  useOCR: boolean
  selectedDocuments: Array<{
    id: number
    title?: string
    url?: string
    favIconUrl?: string
  }>
  uploadedFiles: any[]
  contextFiles: any[]
  documentContext: any[]
  queuedMessages: QueuedRequest[]
  setQueuedMessages: (value: QueuedRequest[] | ((prev: QueuedRequest[]) => QueuedRequest[])) => void
  historyId: string | null
  serverChatId: string | null
  conversationTokenCount: number
  resolvedMaxContext: number
  estimateTokensForText: (text: string) => number
  characterContextTokenEstimate: number
  pinnedSourceTokenEstimate: number
  currentContextSnapshot: Record<string, any>
  setLastSubmittedContext: (value: Record<string, any>) => void
  setSelectedModel: (model: string) => void
  setChatMode: (mode: string) => void
  setWebSearch: (value: boolean) => void
  setCompareMode: (value: boolean) => void
  setCompareSelectedModels: (models: string[]) => void
  setSelectedSystemPrompt: (value: string) => void
  setToolChoice: (value: string) => void
  setUseOCR: (value: boolean) => void
  compareModelsSupportCapability: (
    models: string[],
    capability: string
  ) => boolean
  canCommitDispatchResult?: () => boolean
  sendMessage: (payload: Record<string, any>) => Promise<ChatSubmitResult | void>
  stopStreamingRequest: (options?: { discardTurn?: boolean }) => void
  form: {
    setFieldError: (field: string, error: string) => void
    reset: () => void
  }
  clearSelectedDocuments: () => void
  clearUploadedFiles: () => void
  textAreaFocus: () => void
  onEnqueueSuccess?: () => void
  notificationApi: {
    error: (opts: Record<string, any>) => void
    warning: (opts: Record<string, any>) => void
    info: (opts: Record<string, any>) => void
  }
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
}

const mergePreparedDocumentOverrides = (
  baseOverrides: Record<string, unknown>,
  prepared: Awaited<ReturnType<typeof prepareChatDocumentAttachmentsForSend>>,
  promptText: string
) => {
  const {
    documentProcessing,
    documentSnippetForModel,
    userMetadataExtra,
    ...preparedOverrides
  } = (prepared.requestOverrides ?? {}) as Record<string, unknown>
  const messageForModel =
    typeof documentSnippetForModel === "string" && documentSnippetForModel.trim()
      ? [
          typeof baseOverrides.messageForModel === "string"
            ? baseOverrides.messageForModel
            : promptText,
          documentSnippetForModel
        ]
          .filter(Boolean)
          .join("\n\n")
      : undefined
  return {
    ...baseOverrides,
    ...preparedOverrides,
    ...(typeof messageForModel === "string" ? { messageForModel } : {}),
    userMetadataExtra: {
      ...((baseOverrides.userMetadataExtra as Record<string, unknown> | undefined) ?? {}),
      ...(userMetadataExtra && typeof userMetadataExtra === "object"
        ? userMetadataExtra
        : {}),
      documentProcessing: documentProcessing ?? prepared.turnMetadata
    }
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function usePlaygroundQueueManagement(
  deps: UsePlaygroundQueueManagementDeps
) {
  const {
    composerModels,
    composerModelsLoading = false,
    isConnectionReady,
    isSending,
    getDispatchDisabledReason,
    captureDispatchFence,
    historySelection,
    selectedModel,
    chatMode,
    webSearch,
    compareModeActive,
    compareSelectedModels,
    selectedSystemPrompt,
    selectedQuickPrompt,
    toolChoice,
    useOCR,
    selectedDocuments,
    uploadedFiles,
    contextFiles,
    documentContext,
    queuedMessages,
    setQueuedMessages,
    historyId,
    serverChatId,
    conversationTokenCount,
    resolvedMaxContext,
    estimateTokensForText,
    currentContextSnapshot,
    setLastSubmittedContext,
    setSelectedModel,
    setChatMode,
    setWebSearch,
    setCompareMode,
    setCompareSelectedModels,
    setSelectedSystemPrompt,
    setToolChoice,
    setUseOCR,
    compareModelsSupportCapability,
    sendMessage,
    canCommitDispatchResult,
    stopStreamingRequest,
    form,
    clearSelectedDocuments,
    clearUploadedFiles,
    textAreaFocus,
    onEnqueueSuccess,
    notificationApi,
    t
  } = deps

  const availableChatModelIds = React.useMemo(
    () =>
      buildAvailableChatModelIds(
        Array.isArray(composerModels) ? (composerModels as any[]) : []
      ),
    [composerModels]
  )

  const buildQueuedDocuments = React.useCallback(
    (): ChatDocuments =>
      selectedDocuments.map((doc) => ({
        type: "tab",
        tabId: doc.id,
        title: doc.title,
        url: doc.url,
        favIconUrl: doc.favIconUrl
      })),
    [selectedDocuments]
  )

  const buildQueuedRequestSnapshot = React.useCallback(
    (): Partial<QueuedRequestSnapshot> => ({
      selectedModel,
      chatMode:
        chatMode === "rag" || chatMode === "vision" ? chatMode : "normal",
      webSearch,
      compareMode: compareModeActive,
      compareSelectedModels,
      selectedSystemPrompt,
      selectedQuickPrompt,
      toolChoice,
      useOCR
    }),
    [
      chatMode,
      compareModeActive,
      compareSelectedModels,
      selectedModel,
      selectedQuickPrompt,
      selectedSystemPrompt,
      toolChoice,
      useOCR,
      webSearch
    ]
  )

  const isQueuedDispatchBlockedByComposerState = React.useMemo(() => {
    const uploadedFileIds = new Set(uploadedFiles.map((file) => file.id))
    const hasUnreplayableUploads = uploadedFiles.some(
      (file) => !file.content || !file.processingMode
    )
    const hasUntrackedContextFiles = contextFiles.some(
      (file) => !uploadedFileIds.has(file.id)
    )
    return (
      hasUnreplayableUploads ||
      hasUntrackedContextFiles ||
      (Array.isArray(documentContext) && documentContext.length > 0)
    )
  }, [contextFiles, documentContext, uploadedFiles])

  const validateQueuedRequest = React.useCallback(
    (item: QueuedRequest) => {
      if (isQueuedDispatchBlockedByComposerState) {
        return t(
          "playground:composer.queue.currentDraftAttachmentConflict",
          "Clear the current draft attachments/context before sending queued requests."
        )
      }

      const sourceContext = (item.sourceContext ??
        null) as PlaygroundQueuedSourceContext | null

      if (!sourceContext?.isImageCommand && !isSavedRoleplayConversation(historySelection)) {
        if (composerModelsLoading) return t("playground:composer.modelsLoading", "Loading models…")
        if (!item.snapshot.compareMode) {
          const normalizedSelectedModel = normalizeChatModelId(
            item.snapshot.selectedModel
          )
          if (!normalizedSelectedModel) {
            return t("formError.noModel")
          }
          const unavailableModel = findUnavailableChatModel(
            [normalizedSelectedModel],
            availableChatModelIds,
          true
          )
          if (unavailableModel) {
            return t(
              "playground:composer.validationModelUnavailableInline",
              "Selected model is not available on this server. Refresh models or choose a different model."
            )
          }
        } else if (
          !item.snapshot.compareSelectedModels ||
          item.snapshot.compareSelectedModels.length < 2
        ) {
          return t(
            "playground:composer.validationCompareMinModelsInline",
            "Select at least two models for Compare mode."
          )
        } else {
          const unavailableModel = findUnavailableChatModel(
            item.snapshot.compareSelectedModels,
            availableChatModelIds,
          true
          )
          if (unavailableModel) {
            return t(
              "playground:composer.validationModelUnavailableInline",
              "Selected model is not available on this server. Refresh models or choose a different model."
            )
          }
        }

        if (
          item.snapshot.compareMode &&
          item.image.length > 0 &&
          !compareModelsSupportCapability(
            item.snapshot.compareSelectedModels,
            "vision"
          )
        ) {
          return t(
            "playground:composer.validationCompareVisionInline",
            "One or more selected compare models do not support image input."
          )
        }
      }

      return null
    },
    [
      availableChatModelIds,
      composerModelsLoading,
      compareModelsSupportCapability,
      historySelection,
      isQueuedDispatchBlockedByComposerState,
      t
    ]
  )

  const sendQueuedRequest = React.useCallback(
    async (item: QueuedRequest, assertDispatchReady: () => void) => {
      let sceneCapture: ReturnType<typeof captureActorSceneSend> = null
      let dispatchIsCurrent: (() => boolean) | null = null
      const sourceIsCurrent = captureDispatchFence?.() ?? captureChatDraftSource().isCurrent
      const assertReady = () => {
        assertDispatchReady()
        const reason = getDispatchDisabledReason?.()
        if (reason || sourceIsCurrent?.() === false || dispatchIsCurrent?.() === false || sceneCapture?.isCurrent() === false) throw new Error(reason || t(
          "playground:composer.queueChatChanged", "This chat or setup changed. Review the queued request before running it again."
        ))
      }
      assertReady()
      const validationError = validateQueuedRequest(item)
      if (validationError) {
        form.setFieldError("message", validationError)
        throw new Error(validationError)
      }

      const sourceContext = (item.sourceContext ??
        null) as PlaygroundQueuedSourceContext | null
      const documents = Array.isArray(sourceContext?.documents)
        ? sourceContext.documents
        : []
      const queuedUploadedFiles = Array.isArray(sourceContext?.uploadedFiles)
        ? sourceContext.uploadedFiles
        : []

      if (
        isSavedRoleplayConversation(historySelection) &&
        !sourceContext?.isImageCommand
      ) {
        if (
          item.image ||
          sourceContext?.documents?.length ||
          sourceContext?.uploadedFiles?.length ||
          sourceContext?.requestOverrides ||
          item.snapshot.compareMode ||
          item.snapshot.chatMode !== "normal" ||
          item.snapshot.webSearch ||
          item.snapshot.selectedSystemPrompt ||
          item.snapshot.toolChoice === "required"
        )
          throw new Error(
            t(
              "playground:composer.roleplayResumeContext",
              "This conversation uses its saved roleplay setup. Start a new chat to add extra context, tools or another mode. Your message is kept."
            )
          )
        assertReady()
        const result = await sendMessage({
          message: item.promptText,
          image: "",
          docs: []
        })
        throwIfChatSubmitUnsuccessful(result)
        return
      }

      const modelSettings = useStoreChatModelSettings.getState()
      const queuedModelSelection = resolveModelSelectorSelection((composerModels ?? []) as ModelSelectorDescriptor[], item.snapshot.selectedModel, modelSettings.apiProvider, {
        scope: modelSettings.activeSettingsScope,
        providerOverride: modelSettings.activeSettingsScope ? modelSettings.scopedSettingsByModelKey[modelSettings.activeSettingsScope]?.apiProvider : undefined
      })
      setSelectedModel(item.snapshot.selectedModel)
      assertReady()
      modelSettings.setActiveSettingsScope(queuedModelSelection.settingsScope ?? null)
      assertReady()
      if (useStoreChatModelSettings.getState().activeSettingsScope !== (queuedModelSelection.settingsScope ?? undefined)) throw new Error(t(
        "playground:composer.queueChatChanged", "This chat or setup changed. Review the queued request before running it again."
      ))
      dispatchIsCurrent = captureChatDispatchSource({
        selectedModel: item.snapshot.selectedModel ?? "",
        toolChoice: item.snapshot.toolChoice === "auto" || item.snapshot.toolChoice === "required" || item.snapshot.toolChoice === "none" ? item.snapshot.toolChoice : undefined,
        selectionSource: { model: "explicit", toolChoice: "explicit" }
      })

      sceneCapture = sourceContext?.isImageCommand ? null : captureActorSceneSend(historySelection ?? null, () => sourceIsCurrent?.() !== false && dispatchIsCurrent?.() !== false)
      if (!sourceContext?.isImageCommand && !sceneCapture) throw new Error("actor_settings_unavailable")
      const actorSceneSend = sceneCapture ? await sceneCapture.read() : undefined
      assertReady()

      let queuedRequestOverrides: Record<string, unknown> = {
        ...(sourceContext?.requestOverrides ?? {})
      }
      if (queuedUploadedFiles.length > 0) {
        const preparedDocuments = await prepareChatDocumentAttachmentsForSend({
          files: queuedUploadedFiles,
          historyId: historyId ?? undefined,
          sessionId: serverChatId ?? undefined
        })
        assertReady()
        const firstBlockedOrFailed =
          preparedDocuments.blockedFiles[0] || preparedDocuments.failedFiles[0]
        if (firstBlockedOrFailed || !preparedDocuments.requestOverrides) {
          const message =
            firstBlockedOrFailed?.processingBlockedReason ||
            firstBlockedOrFailed?.processingError ||
            t(
              "playground:documentProcessing.sendBlockedBody",
              "Resolve the document processing issue before sending."
            )
          notificationApi.error({
            message: t(
              "playground:documentProcessing.sendBlockedTitle",
              "Document processing blocked"
            ),
            description: message
          })
          throw new Error(String(message))
        }
        queuedRequestOverrides = mergePreparedDocumentOverrides(
          queuedRequestOverrides,
          preparedDocuments,
          item.promptText
        )
      }

      assertReady()
      setChatMode(item.snapshot.chatMode)
      assertReady()
      setWebSearch(item.snapshot.webSearch)
      assertReady()
      setCompareMode(item.snapshot.compareMode)
      assertReady()
      setCompareSelectedModels(item.snapshot.compareSelectedModels)
      assertReady()
      setSelectedSystemPrompt(item.snapshot.selectedSystemPrompt ?? "")
      assertReady()
      if (
        item.snapshot.toolChoice === "auto" ||
        item.snapshot.toolChoice === "required" ||
        item.snapshot.toolChoice === "none"
      ) {
        setToolChoice(item.snapshot.toolChoice)
        assertReady()
      }
      setUseOCR(item.snapshot.useOCR)
      assertReady()

      const projectedForSubmission = projectTokenBudget({
        conversationTokens: conversationTokenCount,
        draftTokens: estimateTokensForText(item.promptText),
        maxTokens: resolvedMaxContext
      })
      if (
        projectedForSubmission.isOverLimit ||
        projectedForSubmission.isNearLimit
      ) {
        notificationApi.warning({
          message: t(
            "playground:tokens.preSendWarningTitle",
            "Context budget warning"
          ),
          description: projectedForSubmission.isOverLimit
            ? t(
                "playground:tokens.preSendOverLimit",
                "Projected send exceeds the model context window. Consider trimming prompt/context before sending."
              )
            : t(
                "playground:tokens.preSendNearLimit",
                "Projected send is near the context window limit."
              )
        })
      }

      setLastSubmittedContext(currentContextSnapshot)
      assertReady()
      const submitResult = await sendMessage({
        actorSceneSend,
        image: sourceContext?.isImageCommand ? "" : item.image,
        message: item.promptText,
        docs: sourceContext?.isImageCommand ? [] : documents,
        requestOverrides: {
          selectedModel: item.snapshot.selectedModel,
          selectedSystemPrompt: item.snapshot.selectedSystemPrompt,
          ...queuedRequestOverrides,
          toolChoice:
            item.snapshot.toolChoice === "auto" ||
            item.snapshot.toolChoice === "required" ||
            item.snapshot.toolChoice === "none"
              ? item.snapshot.toolChoice
              : undefined,
          useOCR: item.snapshot.useOCR,
          webSearch: item.snapshot.webSearch
        },
        imageBackendOverride: sourceContext?.isImageCommand
          ? sourceContext.imageBackendOverride
          : undefined,
        userMessageType: sourceContext?.isImageCommand
          ? IMAGE_GENERATION_USER_MESSAGE_TYPE
          : undefined,
        assistantMessageType: sourceContext?.isImageCommand
          ? IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
          : undefined,
        imageGenerationSource: sourceContext?.isImageCommand
          ? "slash-command"
          : undefined
      })
      throwIfChatSubmitUnsuccessful(submitResult)
    },
    [
      captureDispatchFence,
      composerModels,
      historySelection,
      getDispatchDisabledReason,
      conversationTokenCount,
      currentContextSnapshot,
      estimateTokensForText,
      form,
      historyId,
      notificationApi,
      resolvedMaxContext,
      sendMessage,
      serverChatId,
      setChatMode,
      setCompareMode,
      setCompareSelectedModels,
      setLastSubmittedContext,
      setSelectedModel,
      setSelectedSystemPrompt,
      setToolChoice,
      setUseOCR,
      setWebSearch,
      t,
      validateQueuedRequest
    ]
  )

  const resolveConversationId = React.useCallback(
    () => historyId ?? serverChatId ?? null,
    [historyId, serverChatId]
  )

  const handleEnqueueBlocked = React.useCallback(() => {
    notificationApi.warning({
      message: t(
        "playground:composer.queue.attachmentsNeedManualRepairTitle",
        "Queue needs a simpler draft"
      ),
      description: t(
        "playground:composer.queue.attachmentsNeedManualRepairBody",
        "Queued requests currently support text, images, and tab mentions. Clear attached files/context before queueing this draft."
      )
    })
  }, [notificationApi, t])

  const handleEnqueueSuccess = React.useCallback(
    (isStreamingAtEnqueue: boolean) => {
      onEnqueueSuccess?.()
      clearSelectedDocuments()
      clearUploadedFiles()
      textAreaFocus()
      notificationApi.info({
        message: t(
          "playground:composer.queue.requestQueued",
          "Request queued"
        ),
        description: isStreamingAtEnqueue
          ? t(
              "playground:composer.queue.requestQueuedWhileBusy",
              "We'll run it after the current response finishes."
            )
          : t(
              "playground:composer.queue.requestQueuedWhileOffline",
              "We'll send it when your tldw server reconnects."
            )
      })
    },
    [
      clearSelectedDocuments,
      clearUploadedFiles,
      form,
      notificationApi,
      onEnqueueSuccess,
      t,
      textAreaFocus
    ]
  )

  const cancelCurrentAndRunDisabledReasonText =
    isSending && serverChatId
      ? t(
          "playground:composer.queue.cancelAndRunDisabled",
          "Cancel current & run now is not available for server-backed turns yet."
        )
      : null

  const queue = useComposerQueue({
    canCommitDispatchResult,
    isConnectionReady,
    isStreaming: isSending,
    queuedMessages,
    setQueuedMessages,
    sendQueuedRequest,
    stopStreamingRequest,
    resolveConversationId,
    buildQueuedDocuments,
    buildQueuedRequestSnapshot,
    isQueuedDispatchBlocked: isQueuedDispatchBlockedByComposerState || Boolean(getDispatchDisabledReason?.()),
    onEnqueueBlocked: handleEnqueueBlocked,
    onEnqueueSuccess: handleEnqueueSuccess,
    cancelCurrentAndRunDisabledReasonText
  })

  const queueSubmission = React.useCallback(
    ({
      promptText,
      image,
      intent,
      requestOverrides
    }: {
      promptText: string
      image: string
      intent: SubmissionIntent
      requestOverrides?: {
        messageForModel?: string
      } & Record<string, unknown>
    }) => {
      const documents = buildQueuedDocuments()
      return queue.enqueue({
        promptText,
        image: intent.isImageCommand ? "" : image,
        attachments: documents,
        sourceContext: {
          documents,
          uploadedFiles: uploadedFiles.map((file) => ({ ...file })),
          imageBackendOverride: intent.isImageCommand
            ? intent.imageBackendOverride
            : undefined,
          isImageCommand: intent.isImageCommand,
          requestOverrides
        },
        blockedReason: isQueuedDispatchBlockedByComposerState
          ? "draft-attachments-conflict"
          : null
      })
    },
    [
      buildQueuedDocuments,
      isQueuedDispatchBlockedByComposerState,
      queue,
      uploadedFiles
    ]
  )

  const validateSelectedChatModelsAvailability = React.useCallback(
    (modelsToCheck: string[]) => {
      if (composerModelsLoading) {
        form.setFieldError("message", t("playground:composer.modelsLoading", "Loading models…"))
        return false
      }
      const unavailableModel = findUnavailableChatModel(
        modelsToCheck,
        availableChatModelIds,
          true
      )
      if (!unavailableModel) return true
      form.setFieldError(
        "message",
        t(
          "playground:composer.validationModelUnavailableInline",
          "Selected model is not available on this server. Refresh models or choose a different model."
        )
      )
      return false
    },
    [availableChatModelIds, composerModelsLoading, form, t]
  )

  return {
    availableChatModelIds,
    isQueuedDispatchBlockedByComposerState,
    queuedRequestActions: queue.queuedRequestActions,
    queueSubmission,
    cancelCurrentAndRunDisabledReason: queue.cancelCurrentAndRunDisabledReason,
    handleRunQueuedRequest: queue.handleRunQueuedRequest,
    handleRunNextQueuedRequest: queue.handleRunNextQueuedRequest,
    validateSelectedChatModelsAvailability,
    validateQueuedRequest,
    buildQueuedDocuments,
    buildQueuedRequestSnapshot
  }
}

export type UsePlaygroundQueueManagementReturn = ReturnType<
  typeof usePlaygroundQueueManagement
>
