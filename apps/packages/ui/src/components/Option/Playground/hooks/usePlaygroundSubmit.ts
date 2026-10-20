import { isSavedRoleplayConversation } from "@/hooks/chat/roleplay-resume-source"
import { captureChatDispatchSource } from "@/hooks/chat/chat-dispatch-source"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import React from "react"
import { captureActorSceneSend, type ActorSceneSend } from "@/hooks/chat/actor-scene-source"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { nativeCharacterPreflightMessage } from "@/hooks/chat/native-character-preflight"
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
import type { ComposerDraftOwnership } from "@/components/Chat/composer/hooks/useComposerText"
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
  captureDraftOwnership: () => ComposerDraftOwnership
  captureHistorySelection: () => () => boolean
  isSending: boolean
  isConnectionReady: boolean
  draftReady: boolean
  webSearch: boolean
  chatMode: string
  systemPrompt?: string
  selectedSystemPrompt?: string | null
  systemPromptTemplateId?: string
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
  historySelection?: HistorySelectionController | null
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
  getSubmitDisabledReason?: () => string | null
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
    captureDraftOwnership,
    captureHistorySelection,
    isSending,
    isConnectionReady,
    draftReady,
    webSearch,
    chatMode,
    systemPrompt,
    selectedSystemPrompt,
    systemPromptTemplateId,
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
    historySelection,
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
    getSubmitDisabledReason,
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
  const sourceDraftRevision = useStoreMessageOption(state => state.chatDraftRevision)

  const historySelection = useHistorySelectionContext()

  const submitFormRef = React.useRef<
    (options?: { ignorePinnedResults?: boolean }) => void
  >(() => undefined)
  const [isPreparingDocuments, setIsPreparingDocuments] = React.useState(false)
  const activeSubmissionRef = React.useRef<symbol | null>(null)
  const currentDocumentSelectionKey =
    documentProcessingSelectionKey(uploadedFiles)
  const contextKeyFor = (searchEnabled: boolean, mode = chatMode) => JSON.stringify([
    historyId ?? null,
    serverChatId ?? null,
    selectedModel ?? null,
    mode,
    systemPrompt ?? null,
    selectedSystemPrompt ?? null,
    systemPromptTemplateId ?? null,
    compareModeActive,
    compareSelectedModels,
    searchEnabled,
    fileRetrievalEnabled,
    selectedDocuments.map((doc: any) => [doc.id, doc.title, doc.url]),
    currentDocumentSelectionKey,
    Boolean(currentContextSnapshot?.temporaryChat),
    jsonMode,
    importedSidepanelContext ?? null,
    openUIRequestMode,
    researchContext ?? null,
    ragPinnedResults
  ])
  const contextKey = contextKeyFor(webSearch)
  const contextRevisionRef = React.useRef({ key: contextKey, revision: 0 })
  const pageContext = importedSidepanelContext ?? null
  const pageContextRevisionRef = React.useRef({ value: pageContext, revision: 0 })
  React.useLayoutEffect(() => {
    if (contextRevisionRef.current.key !== contextKey) {
      contextRevisionRef.current = {
        key: contextKey,
        revision: contextRevisionRef.current.revision + 1
      }
    }
    if (pageContextRevisionRef.current.value !== pageContext) {
      pageContextRevisionRef.current = {
        value: pageContext,
        revision: pageContextRevisionRef.current.revision + 1
      }
    }
  }, [contextKey, pageContext])
  const currentDocumentSelectionKeyRef = React.useRef(
    currentDocumentSelectionKey
  )
  const activeDocumentPreparationRef = React.useRef<{
    controller: AbortController
    selectionKey: string
    isCurrent: () => boolean
  } | null>(null)
  currentDocumentSelectionKeyRef.current = currentDocumentSelectionKey

  React.useEffect(() => {
    const active = activeDocumentPreparationRef.current
    if (active && (active.selectionKey !== currentDocumentSelectionKey || !isConnectionReady || !active.isCurrent())) {
      active.controller.abort()
      if (activeDocumentPreparationRef.current === active) {
        activeDocumentPreparationRef.current = null
        setIsPreparingDocuments(false)
      }
    }
  }, [currentDocumentSelectionKey, isConnectionReady, sourceDraftRevision, historySelection])

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
    if (getSubmitDisabledReason?.()) return
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
      const activePreparation = activeDocumentPreparationRef.current
      if (activePreparation) {
        if (activePreparation.isCurrent()) return
        activePreparation.controller.abort()
        if (activeDocumentPreparationRef.current === activePreparation) {
          activeDocumentPreparationRef.current = null
          setIsPreparingDocuments(false)
        }
      }
      if (activeSubmissionRef.current) return
      if (historySelection?.getCurrent().status === "unsupported_history_capability") {
        notificationApi.error({
          message: t("playground:historyReview.unavailable", "Selected history unavailable"),
          description: t(
            "playground:historyReview.unsupportedDraftRetained",
            "Review the selected conversation or start a new chat. Your draft is retained."
          )
        })
        return
      }
      const submission = Symbol("playground-submit")
      activeSubmissionRef.current = submission
      let documentPreparation: {
        controller: AbortController
        selectionKey: string
        isCurrent: () => boolean
      } | null = null
      try {
      const draftOwnership = captureDraftOwnership()
      if (!draftOwnership.isOwnerCurrent()) {
        notificationApi.warning({
          message: t("playground:composer.preparingChat", "Preparing chat"),
          description: t("playground:composer.preparingChatDraftRetained", "Your draft is retained. Try Send again in a moment.")
        })
        return
      }
      const historyIsCurrent = captureHistorySelection()
      const contextRevision = contextRevisionRef.current.revision
      const pageContextRevision = pageContextRevisionRef.current.revision
      let ownSlashToggleKey: string | null = null
      const isCurrent = () =>
        draftOwnership.isCurrent() &&
        historyIsCurrent() &&
        pageContextRevisionRef.current.revision === pageContextRevision &&
        (contextRevisionRef.current.revision === contextRevision ||
          (ownSlashToggleKey !== null &&
            contextRevisionRef.current.revision === contextRevision + 1 &&
            contextRevisionRef.current.key === ownSlashToggleKey))
      if (!isCurrent()) return
      const intent = resolveSubmissionIntent(value.message)
      if (intent.handled && /^\/web(?:\s|$)/i.test(value.message.trimStart())) {
        ownSlashToggleKey = contextKeyFor(!webSearch)
      } else if (
        intent.handled && /^\/search(?:\s|$)/i.test(value.message.trimStart())
      ) {
        ownSlashToggleKey = contextKeyFor(
          webSearch,
          chatMode === "rag" ? "normal" : "rag"
        )
      }
      if (intent.handled && !intent.invalidImageCommand) {
        if (!draftOwnership.normalizeMessage(intent.message)) return
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
      const savedRoleplay = isSavedRoleplayConversation(historySelection)
      if (
        savedRoleplay &&
        !intent.isImageCommand &&
        (value.image ||
          selectedDocuments.length ||
          uploadedFiles.length ||
          webSearch ||
          fileRetrievalEnabled ||
          (!options?.ignorePinnedResults && ragPinnedResults.length) ||
          compareModeActive ||
          openUIRequestMode ||
          importedSidepanelContext ||
          researchContext)
      ) {
        form.setFieldError(
          "message",
          t(
            "playground:composer.roleplayResumeContext",
            "This conversation uses its saved roleplay setup. Start a new chat to add extra context, tools or another mode. Your message is kept."
          )
        )
        notificationApi.error({
          message: t("error"),
          description: t(
            "playground:composer.roleplayResumeContext",
            "This conversation uses its saved roleplay setup. Start a new chat to add extra context, tools or another mode. Your message is kept."
          )
        })
        return
      }
      const nextMessage = intent.message
      const combinedMessage = intent.isImageCommand
        ? nextMessage
        : buildPinnedMessage(nextMessage, options)
      const trimmed = savedRoleplay ? combinedMessage : combinedMessage.trim()
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
      if (!intent.isImageCommand && !savedRoleplay) {
        if (!compareModeActive) {
          const normalizedSelectedModel = normalizeChatModelId(selectedModel)
          if (!normalizedSelectedModel) {
            form.setFieldError("message", t("formError.noModel"))
            return
          }
          if (!validateSelectedChatModelsAvailability([normalizedSelectedModel])) {
            return
          }
          if (
            value.image.length > 0 &&
            !compareModelsSupportCapability([String(selectedModel)], "vision")
          ) {
            form.setFieldError(
              "message",
              t(
                "playground:composer.validationVisionInline",
                "Image support is not confirmed for this model. Choose a model that supports images, or remove the attachment."
              )
            )
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

      const completionIsCurrent = captureCompletionGuard?.() ?? (() => true)
      const inputRevision = form.getRevision?.()
      const dispatchIsCurrent = savedRoleplay
        ? () => true
        : captureChatDispatchSource({
            selectedModel: selectedModel ?? "",
            selectionSource: { model: "global", toolChoice: "global" }
          })
      const preparationIsCurrent = () => currentDraft() && completionIsCurrent() && form.getRevision?.() === inputRevision && dispatchIsCurrent() &&
        (!documentPreparation || (!documentPreparation.controller.signal.aborted && currentDocumentSelectionKeyRef.current === documentPreparation.selectionKey))
      const sceneCapture =
        intent.isImageCommand || savedRoleplay
          ? null
          : captureActorSceneSend(
              historySelection ?? null,
              () => currentDraft() && dispatchIsCurrent()
            )
      let actorSceneSend: ActorSceneSend | undefined
      if (!intent.isImageCommand) {
        documentPreparation = {
          controller: new AbortController(),
          selectionKey: currentDocumentSelectionKeyRef.current,
          isCurrent: () => preparationIsCurrent() && sceneCapture?.isCurrent() !== false
        }
        activeDocumentPreparationRef.current = documentPreparation
        if (uploadedFiles.length > 0) setIsPreparingDocuments(true)

        try {
          if (!savedRoleplay) {
            if (!sceneCapture) throw new Error("actor_settings_unavailable")
            actorSceneSend = await sceneCapture.read()
          }
        } catch (error) {
          if (preparationIsCurrent() && sceneCapture?.isCurrent() !== false) notificationApi.error({ message: t("error", { defaultValue: "Error" }),
            description: nativeCharacterPreflightMessage(error, t) ?? t("playground:composer.sceneLoadError", "Scene settings could not be loaded. Existing chat setup was left unchanged.") })
          return
        }
        if (!currentDraft() || !preparationIsCurrent() || documentPreparation.controller.signal.aborted) return
      }

      if (!intent.isImageCommand && webSearch) {
        const defaultEM = await defaultEmbeddingModelForRag()
        if (!isCurrent()) return
        const simpleSearch = await getIsSimpleInternetSearch()
        if (!isCurrent() || !currentDraft() || !preparationIsCurrent()) return
        if (!defaultEM && !simpleSearch) {
          form.setFieldError("message", t("formError.noEmbeddingModel"))
          return
        }
      }
      if (!intent.isImageCommand && uploadedFiles.length > 0) {
        if (!documentPreparation) return
        if (!isCurrent() || !currentDraft()) return
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
            !isCurrent() ||
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
          !isCurrent() ||
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
        if (!currentDraft() || !preparationIsCurrent()) return
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
      if (!isCurrent() || !currentDraft()) return
      let promptAssistAttemptId!: number

      const payload = {
        actorSceneSend,
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

      if (!preparationIsCurrent() || actorSceneSend?.isCurrent() === false) return
      if (documentPreparation && activeDocumentPreparationRef.current === documentPreparation) {
        activeDocumentPreparationRef.current = null
        setIsPreparingDocuments(false)
      }
      const result = await dispatch(payload, {
        beforeSend: () => {
          promptAssistAttemptId = beginPromptAssistReset()
          form.reset()
          if (activeSubmissionRef.current === submission) activeSubmissionRef.current = null
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
          const outcome = normalizeChatSubmitResult(result)
          if (outcome.status === "failed" && outcome.restoreDraft) {
            draftOwnership.restoreAfterReset?.(promptAssistAttemptId)
          }
          if (draftOwnership.isOwnerCurrent() && isChatSubmitSuccess(normalizeChatSubmitResult(result as any))) {
            markPromptAssistAttemptSaved(promptAssistAttemptId, result !== undefined)
            if (messageForModel && pageContextRevisionRef.current.revision === pageContextRevision) {
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
        if (activeSubmissionRef.current === submission) {
          activeSubmissionRef.current = null
        }
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
