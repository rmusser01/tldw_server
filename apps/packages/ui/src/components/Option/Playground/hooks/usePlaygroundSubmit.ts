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
import type { ChatResearchContext } from "@/services/tldw/TldwApiClient"

export type UsePlaygroundSubmitDeps = {
  form: any
  isSending: boolean
  isConnectionReady: boolean
  webSearch: boolean
  compareModeActive: boolean
  compareSelectedModels: string[]
  selectedModel: string | null | undefined
  fileRetrievalEnabled: boolean
  ragPinnedResults: any[]
  selectedDocuments: any[]
  uploadedFiles: any[]
  currentContextSnapshot: any
  conversationTokenCount: number
  characterContextTokenEstimate: number
  pinnedSourceTokenEstimate: number
  resolvedMaxContext: number
  jsonMode: boolean
  openUIRequestMode: boolean
  researchContext?: ChatResearchContext
  sendMessage: (args: any) => Promise<any>
  clearOpenUIRequestMode: () => void
  clearSelectedDocuments: () => void
  clearUploadedFiles: () => void
  textAreaFocus: () => void
  setLastSubmittedContext: (ctx: any) => void
  estimateTokensForText: (text: string) => number
  resolveSubmissionIntent: (message: string) => any
  queueSubmission: (args: any) => void
  validateSelectedChatModelsAvailability: (models: string[]) => boolean
  compareModelsSupportCapability: (models: string[], cap: string) => boolean
  notificationApi: any
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
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
    sendMessage,
    clearOpenUIRequestMode,
    clearSelectedDocuments,
    clearUploadedFiles,
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
      if (
        !intent.isImageCommand &&
        trimmed.length === 0 &&
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
        queueSubmission({
          promptText: trimmed,
          image: value.image,
          intent
        })
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
      form.reset()
      clearSelectedDocuments()
      clearUploadedFiles()
      textAreaFocus()
      const projectedForSubmission = projectTokenBudget({
        conversationTokens:
          conversationTokenCount +
          characterContextTokenEstimate +
          pinnedSourceTokenEstimate,
        draftTokens: estimateTokensForText(trimmed),
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
      await dispatch({
        image: intent.isImageCommand ? "" : value.image,
        message: trimmed,
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
        requestOverrides:
          openUIRequestMode && !intent.isImageCommand
            ? { dynamicUIRequest: { renderer: "openui" } }
            : undefined,
        researchContext:
          intent.isImageCommand || compareModeActive
            ? undefined
            : researchContext
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
