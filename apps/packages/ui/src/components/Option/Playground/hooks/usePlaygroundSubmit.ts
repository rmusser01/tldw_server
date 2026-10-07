import React from "react"
import { useStoreMessageOption } from "@/store/option"
import { defaultEmbeddingModelForRag } from "~/services/tldw-server"
import { getIsSimpleInternetSearch } from "@/services/search"
import { formatPinnedResults } from "@/utils/rag-format"
import { normalizeChatModelId } from "@/utils/chat-model-availability"
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  IMAGE_GENERATION_USER_MESSAGE_TYPE
} from "@/utils/image-generation-chat"
import {
  projectTokenBudget
} from "../usage-metrics"
import { useComposerSubmit } from "@/components/Chat/composer/hooks/useComposerSubmit"
import {
  isChatSubmitSuccess,
  normalizeChatSubmitResult
} from "@/hooks/chat/chat-action-utils"
import type { ChatResearchContext } from "@/services/tldw/TldwApiClient"
import {
  buildSidepanelHandoffMessageForModel,
  type SidepanelChatHandoffPageContext
} from "@/services/sidepanel-chat-handoff"
import {
  documentProcessingSelectionKey,
  prepareChatDocumentAttachmentsForSend
} from "@/services/chat-document-processing"
import type {
  DocumentProcessingTurnMetadata,
  UploadedFile
} from "@/db/dexie/types"

type PlaygroundQueueSubmissionArgs = {
  promptText: string
  image: string
  intent: any
  requestOverrides?: Record<string, unknown>
}

type DocumentProcessingTurnReservation = {
  message: string
  metadata: DocumentProcessingTurnMetadata
}

export type UsePlaygroundSubmitDeps = {
  form: any
  beginPromptAssistReset: () => number
  restorePromptAssistAttempt: (attemptId: number) => void
  markPromptAssistAttemptSaved: (attemptId: number, acknowledgedDelivery?: boolean) => void
  isSending: boolean
  isConnectionReady: boolean
  draftReady: boolean
  webSearch: boolean
  compareModeActive: boolean
  compareSelectedModels: string[]
  selectedModel: string | null | undefined
  historyId?: string | null
  serverChatId?: string | null
  fileRetrievalEnabled: boolean
  ragPinnedResults: any[]
  selectedDocuments: any[]
  uploadedFiles: UploadedFile[]
  currentContextSnapshot: any
  conversationTokenCount: number
  characterContextTokenEstimate: number
  pinnedSourceTokenEstimate: number
  resolvedMaxContext: number
  jsonMode: boolean
  openUIRequestMode: boolean
  researchContext?: ChatResearchContext
  importedSidepanelContext?: SidepanelChatHandoffPageContext | null
  clearImportedSidepanelContext?: () => void
  captureCompletionGuard?: () => () => boolean
  sendMessage: (args: any) => Promise<any>
  clearOpenUIRequestMode: () => void
  clearSelectedDocuments: () => void
  clearUploadedFiles: () => void
  reserveDocumentProcessingTurn?: (
    reservation: DocumentProcessingTurnReservation
  ) => string | undefined
  updateDocumentProcessingTurn?: (
    userMessageId: string,
    metadata: DocumentProcessingTurnMetadata
  ) => void
  textAreaFocus: () => void
  setLastSubmittedContext: (ctx: any) => void
  estimateTokensForText: (text: string) => number
  resolveSubmissionIntent: (message: string) => any
  queueSubmission: (args: PlaygroundQueueSubmissionArgs) => unknown
  validateSelectedChatModelsAvailability: (models: string[]) => boolean
  compareModelsSupportCapability: (models: string[], cap: string) => boolean
  notificationApi: any
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
}

const buildPendingDocumentProcessingMetadata = (
  files: UploadedFile[],
  status: "waiting_for_files" | "processing"
): DocumentProcessingTurnMetadata => ({
  status,
  files: files.map((file) => ({
    id: file.id,
    filename: file.filename,
    mode: file.processingMode,
    status:
      status === "processing"
        ? "processing"
        : file.processingStatus ?? "pending",
    summary: file.processingSummary ?? file.processingBlockedReason,
    error: file.processingError
  }))
})

const toSendingPromptMetadata = (
  metadata: unknown,
  fallback: DocumentProcessingTurnMetadata
): DocumentProcessingTurnMetadata => {
  const base =
    metadata && typeof metadata === "object" && !Array.isArray(metadata)
      ? (metadata as DocumentProcessingTurnMetadata)
      : fallback
  return {
    ...base,
    status: "sending_prompt"
  }
}

const toCancelledDocumentProcessingMetadata = (
  files: UploadedFile[]
): DocumentProcessingTurnMetadata => ({
  status: "cancelled",
  files: files.map((file) => ({
    id: file.id,
    filename: file.filename,
    mode: file.processingMode,
    status: "cancelled"
  }))
})

export function usePlaygroundSubmit(deps: UsePlaygroundSubmitDeps) {
  const {
    form,
    beginPromptAssistReset,
    restorePromptAssistAttempt,
    markPromptAssistAttemptSaved,
    isSending,
    isConnectionReady,
    draftReady,
    webSearch,
    compareModeActive,
    compareSelectedModels,
    selectedModel,
    historyId,
    serverChatId,
    fileRetrievalEnabled,
    ragPinnedResults,
    selectedDocuments,
    uploadedFiles,
    currentContextSnapshot,
    conversationTokenCount,
    characterContextTokenEstimate,
    pinnedSourceTokenEstimate,
    resolvedMaxContext,
    jsonMode,
    openUIRequestMode,
    researchContext,
    importedSidepanelContext,
    clearImportedSidepanelContext,
    sendMessage,
    captureCompletionGuard,
    clearOpenUIRequestMode,
    clearSelectedDocuments,
    clearUploadedFiles,
    reserveDocumentProcessingTurn,
    updateDocumentProcessingTurn,
    textAreaFocus,
    setLastSubmittedContext,
    estimateTokensForText,
    resolveSubmissionIntent,
    queueSubmission,
    validateSelectedChatModelsAvailability,
    compareModelsSupportCapability,
    notificationApi,
    t
  } = deps
  const connectionReadyRef = React.useRef(isConnectionReady)
  const connectionGenerationRef = React.useRef(0)
  React.useLayoutEffect(() => {
    if (connectionReadyRef.current && !isConnectionReady) connectionGenerationRef.current += 1
    connectionReadyRef.current = isConnectionReady
  }, [isConnectionReady])
  const draftReadyRef = React.useRef(draftReady)
  draftReadyRef.current = draftReady

  const submitFormRef = React.useRef<
    (options?: { ignorePinnedResults?: boolean }) => void
  >(() => undefined)
  const [isPreparingDocuments, setIsPreparingDocuments] = React.useState(false)
  const currentDocumentSelectionKey =
    documentProcessingSelectionKey(uploadedFiles)
  const currentDocumentSelectionKeyRef = React.useRef(
    currentDocumentSelectionKey
  )
  const activeDocumentPreparationRef = React.useRef<{
    controller: AbortController
    selectionKey: string
  } | null>(null)
  currentDocumentSelectionKeyRef.current = currentDocumentSelectionKey

  React.useEffect(() => {
    const active = activeDocumentPreparationRef.current
    if (active && (active.selectionKey !== currentDocumentSelectionKey || !isConnectionReady)) {
      active.controller.abort()
    }
  }, [currentDocumentSelectionKey, isConnectionReady])

  React.useEffect(
    () => () => activeDocumentPreparationRef.current?.controller.abort(),
    []
  )

  // Route through the shared composer dispatch so cross-cutting concerns
  // (metrics, error handling) have one home if we need them later.
  const { dispatch } = useComposerSubmit({ sendMessage, captureCompletionGuard })

  const buildPinnedMessage = React.useCallback(
    (message: string, options?: { ignorePinnedResults?: boolean }) => {
      if (options?.ignorePinnedResults) return message
      if (fileRetrievalEnabled) return message
      if (!ragPinnedResults || ragPinnedResults.length === 0) return message
      const pinnedText = formatPinnedResults(ragPinnedResults, "markdown")
      return message ? `${message}\n\n${pinnedText}` : pinnedText
    },
    [fileRetrievalEnabled, ragPinnedResults]
  )

  const submitForm = (options?: { ignorePinnedResults?: boolean }) => {
    form.onSubmit(async (value: any) => {
      if (!connectionReadyRef.current) {
        notificationApi.info({ message: t("playground:composer.connectionRequired", "Connect to your tldw server before sending. Your draft has not been queued.") })
        return
      }
      if (!draftReadyRef.current) {
        notificationApi.info({ message: t("playground:draftLoading", "Draft still loading"),
          description: t("playground:draftLoadingWait", "Wait for this conversation's draft to finish loading before sending.") })
        return
      }
      const connectionGeneration = connectionGenerationRef.current
      const draftRevision = useStoreMessageOption.getState().chatDraftRevision
      const currentDraft = () => connectionReadyRef.current && connectionGenerationRef.current === connectionGeneration && draftReadyRef.current && useStoreMessageOption.getState().chatDraftRevision === draftRevision
      if (activeDocumentPreparationRef.current) return
      let documentPreparation: {
        controller: AbortController
        selectionKey: string
      } | null = null
      try {
      const intent = resolveSubmissionIntent(value.message)
      if (intent.handled && !intent.invalidImageCommand) {
        form.setFieldValue("message", intent.message)
      }
      if (intent.invalidImageCommand) {
        notificationApi.error({
          message: t("error", { defaultValue: "Error" }),
          description: intent.imageCommandMissingProvider
            ? t(
                "imageCommand.missingProvider",
                "Pick an Image provider in More tools or use /generate-image:<provider> <prompt>."
              )
            : t(
                "imageCommand.invalidUsage",
                "Use /generate-image:<provider> <prompt>."
              )
        })
        return
      }
      const nextMessage = intent.message
      const combinedMessage = intent.isImageCommand
        ? nextMessage
        : buildPinnedMessage(nextMessage, options)
      const trimmed = combinedMessage.trim()
      const visiblePrompt =
        !intent.isImageCommand &&
        trimmed.length === 0 &&
        importedSidepanelContext
          ? t(
              "playground:sidepanelHandoff.contextOnlyDraft",
              "Summarize this page."
            )
          : trimmed
      const messageForModel =
        !intent.isImageCommand && importedSidepanelContext
          ? buildSidepanelHandoffMessageForModel(
              visiblePrompt,
              importedSidepanelContext
            )
          : undefined
      const requestOverrides: Record<string, unknown> | undefined = messageForModel
        ? { messageForModel }
        : undefined
      const openUIRequestOverrides: Record<string, unknown> | undefined =
        openUIRequestMode && !intent.isImageCommand
          ? { dynamicUIRequest: { renderer: "openui" } }
          : undefined
      let mergedRequestOverrides: Record<string, unknown> | undefined =
        requestOverrides || openUIRequestOverrides
          ? {
              ...(requestOverrides ?? {}),
              ...(openUIRequestOverrides ?? {})
            }
          : undefined
      if (
        !intent.isImageCommand &&
        visiblePrompt.length === 0 &&
        value.image.length === 0 &&
        selectedDocuments.length === 0 &&
        uploadedFiles.length === 0
      ) {
        return
      }
      const shouldQueueInsteadOfSend = isSending
      if (!intent.isImageCommand) {
        if (!compareModeActive) {
          const normalizedSelectedModel = normalizeChatModelId(selectedModel)
          if (!normalizedSelectedModel) {
            form.setFieldError("message", t("formError.noModel"))
            return
          }
          if (!validateSelectedChatModelsAvailability([normalizedSelectedModel])) {
            return
          }
        } else if (
          !compareSelectedModels ||
          compareSelectedModels.length < 2
        ) {
          form.setFieldError(
            "message",
            t(
              "playground:composer.validationCompareMinModelsInline",
              "Select at least two models for Compare mode."
            )
          )
          return
        } else if (
          !validateSelectedChatModelsAvailability(compareSelectedModels)
        ) {
          return
        }
        if (
          compareModeActive &&
          value.image.length > 0 &&
          !compareModelsSupportCapability(compareSelectedModels, "vision")
        ) {
          form.setFieldError(
            "message",
            t(
              "playground:composer.validationCompareVisionInline",
              "One or more selected compare models do not support image input."
            )
            )
          return
        }
      }

      if (intent.isImageCommand && trimmed.length === 0) {
        notificationApi.error({
          message: t("error", { defaultValue: "Error" }),
          description: t(
            "imageCommand.missingPrompt",
            "Image prompt is required."
          )
        })
        return
      }

      if (shouldQueueInsteadOfSend) {
        const queuedItem = queueSubmission({
          promptText: visiblePrompt,
          image: value.image,
          intent,
          ...(mergedRequestOverrides
            ? { requestOverrides: mergedRequestOverrides }
            : {})
        })
        if (queuedItem && messageForModel) {
          clearImportedSidepanelContext?.()
        }
        if (queuedItem && openUIRequestMode) {
          clearOpenUIRequestMode()
        }
        return
      }

      if (!intent.isImageCommand && uploadedFiles.length > 0) {
        documentPreparation = {
          controller: new AbortController(),
          selectionKey: currentDocumentSelectionKeyRef.current
        }
        activeDocumentPreparationRef.current = documentPreparation
        setIsPreparingDocuments(true)
      }

      if (!intent.isImageCommand && webSearch) {
        const defaultEM = await defaultEmbeddingModelForRag()
        const simpleSearch = await getIsSimpleInternetSearch()
        if (!defaultEM && !simpleSearch) {
          form.setFieldError("message", t("formError.noEmbeddingModel"))
          return
        }
      }
      if (!intent.isImageCommand && uploadedFiles.length > 0) {
        if (!documentPreparation) return
        if (!currentDraft()) return
        const waitingMetadata = buildPendingDocumentProcessingMetadata(
          uploadedFiles,
          "waiting_for_files"
        )
        const reservedDocumentUserMessageId =
          reserveDocumentProcessingTurn?.({
            message: visiblePrompt,
            metadata: waitingMetadata
          })
        if (reservedDocumentUserMessageId) {
          updateDocumentProcessingTurn?.(
            reservedDocumentUserMessageId,
            buildPendingDocumentProcessingMetadata(uploadedFiles, "processing")
          )
        }
        let preparedDocuments
        try {
          preparedDocuments = await prepareChatDocumentAttachmentsForSend({
            files: uploadedFiles,
            historyId: historyId ?? undefined,
            sessionId: serverChatId ?? undefined,
            signal: documentPreparation.controller.signal
          })
        } catch (error) {
          if (
            documentPreparation.controller.signal.aborted ||
            (error instanceof Error && error.name === "AbortError")
          ) {
            if (reservedDocumentUserMessageId) {
              updateDocumentProcessingTurn?.(
                reservedDocumentUserMessageId,
                toCancelledDocumentProcessingMetadata(uploadedFiles)
              )
            }
            return
          }
          throw error
        }
        if (
          documentPreparation.controller.signal.aborted ||
          currentDocumentSelectionKeyRef.current !==
            documentPreparation.selectionKey
        ) {
          if (reservedDocumentUserMessageId) {
            updateDocumentProcessingTurn?.(
              reservedDocumentUserMessageId,
              toCancelledDocumentProcessingMetadata(uploadedFiles)
            )
          }
          return
        }
        const firstBlockedOrFailed =
          preparedDocuments.blockedFiles[0] || preparedDocuments.failedFiles[0]
        if (
          firstBlockedOrFailed ||
          !preparedDocuments.requestOverrides
        ) {
          if (reservedDocumentUserMessageId) {
            updateDocumentProcessingTurn?.(
              reservedDocumentUserMessageId,
              preparedDocuments.turnMetadata
            )
          }
          notificationApi.error({
            message: t(
              "playground:documentProcessing.sendBlockedTitle",
              "Document processing blocked"
            ),
            description:
              firstBlockedOrFailed?.processingBlockedReason ||
              firstBlockedOrFailed?.processingError ||
              t(
                "playground:documentProcessing.sendBlockedBody",
                "Resolve the document processing issue before sending."
              )
          })
          return
        }

        const {
          documentProcessing,
          documentSnippetForModel,
          userMetadataExtra,
          ...preparedRequestOverrides
        } = preparedDocuments.requestOverrides as Record<string, unknown>
        const preparedMessageForModel =
          typeof documentSnippetForModel === "string" &&
          documentSnippetForModel.trim()
            ? [
                typeof mergedRequestOverrides?.messageForModel === "string"
                  ? mergedRequestOverrides.messageForModel
                  : visiblePrompt,
                documentSnippetForModel
              ]
                .filter(Boolean)
                .join("\n\n")
            : undefined
        const sendingPromptMetadata = toSendingPromptMetadata(
          documentProcessing,
          preparedDocuments.turnMetadata
        )
        if (reservedDocumentUserMessageId) {
          updateDocumentProcessingTurn?.(
            reservedDocumentUserMessageId,
            sendingPromptMetadata
          )
        }
        mergedRequestOverrides = {
          ...(mergedRequestOverrides ?? {}),
          ...preparedRequestOverrides,
          ...(typeof preparedMessageForModel === "string"
            ? { messageForModel: preparedMessageForModel }
            : {}),
          ...(reservedDocumentUserMessageId
            ? { userMessageId: reservedDocumentUserMessageId }
            : {}),
          userMetadataExtra: {
            ...((mergedRequestOverrides as any)?.userMetadataExtra ?? {}),
            ...(userMetadataExtra && typeof userMetadataExtra === "object"
              ? userMetadataExtra
              : {}),
            documentProcessing: sendingPromptMetadata
          }
        }
      }
      if (!currentDraft()) return
      let promptAssistAttemptId!: number

      const payload = {
        image: intent.isImageCommand ? "" : value.image,
        message: visiblePrompt,
        docs: intent.isImageCommand
          ? []
          : selectedDocuments.map((doc: any) => ({
              type: "tab",
              tabId: doc.id,
              title: doc.title,
              url: doc.url,
              favIconUrl: doc.favIconUrl
            })),
        imageBackendOverride: intent.isImageCommand
          ? intent.imageBackendOverride
          : undefined,
        userMessageType: intent.isImageCommand
          ? IMAGE_GENERATION_USER_MESSAGE_TYPE
          : undefined,
        assistantMessageType: intent.isImageCommand
          ? IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
          : undefined,
        imageGenerationSource: intent.isImageCommand
          ? "slash-command"
          : undefined,
        researchContext:
          intent.isImageCommand || compareModeActive
            ? undefined
            : researchContext,
        ...(mergedRequestOverrides ? { requestOverrides: mergedRequestOverrides } : {})
      }

      const completionIsCurrent = captureCompletionGuard?.() ?? (() => true)
      if (!completionIsCurrent()) return
      const result = await dispatch(payload, {
        beforeSend: () => {
          promptAssistAttemptId = beginPromptAssistReset()
          form.reset()
          clearSelectedDocuments()
          clearUploadedFiles()
          textAreaFocus()
          const projectedForSubmission = projectTokenBudget({
            conversationTokens:
              conversationTokenCount +
              characterContextTokenEstimate +
              pinnedSourceTokenEstimate,
            draftTokens: estimateTokensForText(messageForModel ?? visiblePrompt),
            maxTokens: resolvedMaxContext
          })
          if (projectedForSubmission.isOverLimit || projectedForSubmission.isNearLimit) {
            notificationApi.warning({
              message: t("playground:tokens.preSendWarningTitle", "Context budget warning"),
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
        },
        afterSend: (result) => {
          if (isChatSubmitSuccess(normalizeChatSubmitResult(result as any))) {
            markPromptAssistAttemptSaved(promptAssistAttemptId, result !== undefined)
            if (messageForModel) {
              clearImportedSidepanelContext?.()
            }
          } else {
            restorePromptAssistAttempt(promptAssistAttemptId)
          }
        },
        onSendError: () => restorePromptAssistAttempt(promptAssistAttemptId)
      })
      if (openUIRequestMode && completionIsCurrent() && isChatSubmitSuccess(normalizeChatSubmitResult(result))) {
        clearOpenUIRequestMode()
      }
      } finally {
        if (
          documentPreparation &&
          activeDocumentPreparationRef.current === documentPreparation
        ) {
          activeDocumentPreparationRef.current = null
          setIsPreparingDocuments(false)
        }
      }
    })()
  }

  React.useEffect(() => {
    submitFormRef.current = submitForm
  })

  return {
    submitForm,
    submitFormRef,
    isPreparingDocuments
  }
}
