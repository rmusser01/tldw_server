import React from "react"
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
  isSending: boolean
  isConnectionReady: boolean
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

export function usePlaygroundSubmit(deps: UsePlaygroundSubmitDeps) {
  const {
    form,
    isSending,
    isConnectionReady,
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

  const submitFormRef = React.useRef<
    (options?: { ignorePinnedResults?: boolean }) => void
  >(() => undefined)

  // Route through the shared composer dispatch so cross-cutting concerns
  // (metrics, error handling) have one home if we need them later.
  const { dispatch } = useComposerSubmit({ sendMessage })

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
      const shouldQueueInsteadOfSend = isSending || !isConnectionReady
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

      if (!intent.isImageCommand && webSearch) {
        const defaultEM = await defaultEmbeddingModelForRag()
        const simpleSearch = await getIsSimpleInternetSearch()
        if (!defaultEM && !simpleSearch) {
          form.setFieldError("message", t("formError.noEmbeddingModel"))
          return
        }
      }

      if (!intent.isImageCommand && uploadedFiles.length > 0) {
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
        const preparedDocuments = await prepareChatDocumentAttachmentsForSend({
          files: uploadedFiles,
          historyId: historyId ?? undefined,
          sessionId: serverChatId ?? undefined
        })
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
          userMetadataExtra,
          ...preparedRequestOverrides
        } = preparedDocuments.requestOverrides as Record<string, unknown>
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

      await dispatch(payload, {
        afterSend: (result) => {
          if (
            messageForModel &&
            isChatSubmitSuccess(normalizeChatSubmitResult(result as any))
          ) {
            clearImportedSidepanelContext?.()
          }
        }
      })
      if (openUIRequestMode) {
        clearOpenUIRequestMode()
      }
    })()
  }

  React.useEffect(() => {
    submitFormRef.current = submitForm
  })

  return {
    submitForm,
    submitFormRef
  }
}
