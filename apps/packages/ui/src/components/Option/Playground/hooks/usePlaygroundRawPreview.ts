import React from "react"
import { resolveApiProviderForModel } from "@/utils/resolve-api-provider"
import {
  captureChatRequestDebugSnapshot,
  type ChatRequestDebugSnapshot,
  type ChatToolRequestDebugMetadata
} from "@/services/tldw/chat-request-debug"
import type {
  ChatCompletionRequest,
  ChatMessage,
  ChatResearchContext
} from "@/services/tldw/TldwApiClient"
import { parseJsonObject } from "./utils"
import { formatPinnedResults } from "@/utils/rag-format"
import { resolveChatToolRequest } from "@/utils/chat-tools"

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UsePlaygroundRawPreviewDeps {
  composerModels: unknown[] | undefined
  selectedModel: string | null
  compareModeActive: boolean
  compareSelectedModels: string[]
  compareMaxModels: number
  currentChatModelSettings: {
    temperature?: number
    numPredict?: number
    topP?: number
    frequencyPenalty?: number
    presencePenalty?: number
    reasoningEffort?: string
    historyMessageLimit?: number
    historyMessageOrder?: string
    slashCommandInjectionMode?: string
    apiProvider?: string
    extraHeaders?: string
    extraBody?: string
    llamaThinkingBudgetTokens?: number
    llamaGrammarMode?: string
    llamaGrammarId?: string
    llamaGrammarInline?: string
    llamaGrammarOverride?: string
    jsonMode?: boolean
  }
  history: Array<{ role: string; content?: string; image?: string }>
  systemPrompt: string | undefined
  hasMcp: boolean
  mcpHealthState: string
  mcpTools: any[]
  toolChoice: string
  temporaryChat: boolean
  serverChatId: string | null
  serverChatState: string | null
  serverChatSource: string | null
  selectedCharacter: { id?: string | number; name?: string } | null
  messageSteeringMode: string | undefined
  messageSteeringForceNarrate: boolean | undefined
  ragMediaIds: number[] | null
  selectedKnowledge: unknown
  ragPinnedResults?: any[]
  fileRetrievalEnabled?: boolean
  contextFiles: any[]
  documentContext: any[]
  selectedDocuments: any[]
  imageBackendDefaultTrimmed: string
  resolveSubmissionIntent: (message: string) => {
    message: string
    isImageCommand: boolean
    imageBackendOverride?: string
    handled?: boolean
    invalidImageCommand?: boolean
    imageCommandMissingProvider?: boolean
  }
  formImage: string
  formMessage: string
  researchContext?: ChatResearchContext
  notificationApi: {
    error: (opts: { message: string; description?: string }) => void
    success?: (opts: { message: string; description?: string }) => void
  }
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
  setToolsPopoverOpen: (open: boolean) => void
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

const omitUndefinedFields = <T extends Record<string, unknown>>(payload: T): T =>
  Object.fromEntries(
    Object.entries(payload).filter(([, value]) => value !== undefined)
  ) as T

type PreviewChatRequestBuild = {
  request: ChatCompletionRequest
  metadata: ChatToolRequestDebugMetadata
}

export function usePlaygroundRawPreview(deps: UsePlaygroundRawPreviewDeps) {
  const {
    composerModels,
    selectedModel,
    compareModeActive,
    compareSelectedModels,
    compareMaxModels,
    currentChatModelSettings,
    history,
    systemPrompt,
    hasMcp,
    mcpHealthState,
    mcpTools,
    toolChoice,
    temporaryChat,
    serverChatId,
    serverChatState,
    serverChatSource,
    selectedCharacter,
    messageSteeringMode,
    messageSteeringForceNarrate,
    ragMediaIds,
    selectedKnowledge,
    ragPinnedResults,
    fileRetrievalEnabled,
    contextFiles,
    documentContext,
    selectedDocuments,
    imageBackendDefaultTrimmed,
    resolveSubmissionIntent,
    formImage,
    formMessage,
    researchContext,
    notificationApi,
    t,
    setToolsPopoverOpen
  } = deps

  const [rawRequestModalOpen, setRawRequestModalOpen] = React.useState(false)
  const [rawRequestSnapshot, setRawRequestSnapshot] =
    React.useState<ChatRequestDebugSnapshot | null>(null)

  const getComposerModelMeta = React.useCallback(
    (modelId: string) => {
      const normalized = String(modelId || "").replace(/^tldw:/, "")
      const models = Array.isArray(composerModels) ? (composerModels as any[]) : []
      return models.find((entry) => {
        const candidate =
          String(entry?.id || entry?.model || entry?.name || "").replace(
            /^tldw:/,
            ""
          )
        return candidate === normalized
      })
    },
    [composerModels]
  )

  const supportsCapability = React.useCallback(
    (modelId: string, capability: string) => {
      const meta = getComposerModelMeta(modelId)
      const caps = Array.isArray(meta?.capabilities) ? meta.capabilities : []
      return caps.includes(capability)
    },
    [getComposerModelMeta]
  )

  const toPreviewUserContent = React.useCallback(
    (text: string, image: string, modelId: string): ChatMessage["content"] => {
      const cleanedText = text ?? ""
      const trimmedImage = typeof image === "string" ? image.trim() : ""
      const canUseImages = supportsCapability(modelId, "vision")
      if (trimmedImage.length > 0 && canUseImages) {
        return [
          { type: "image_url", image_url: { url: trimmedImage } },
          { type: "text", text: cleanedText }
        ]
      }
      return cleanedText
    },
    [supportsCapability]
  )

  const toPreviewHistoryMessages = React.useCallback(
    (modelId: string, draftMessage: string, draftImage: string): ChatMessage[] => {
      const requestMessages: ChatMessage[] = []
      for (const entry of history || []) {
        if (!entry || typeof entry.content !== "string") continue
        if (entry.role === "system") {
          requestMessages.push({ role: "system", content: entry.content })
          continue
        }
        if (entry.role === "assistant") {
          requestMessages.push({ role: "assistant", content: entry.content })
          continue
        }
        if (entry.role === "user") {
          requestMessages.push({
            role: "user",
            content: toPreviewUserContent(
              entry.content,
              typeof entry.image === "string" ? entry.image : "",
              modelId
            )
          })
        }
      }

      if (draftMessage.trim().length > 0 || draftImage.trim().length > 0) {
        requestMessages.push({
          role: "user",
          content: toPreviewUserContent(draftMessage, draftImage, modelId)
        })
      }

      const trimmedSystemPrompt = String(systemPrompt || "").trim()
      if (
        trimmedSystemPrompt.length > 0 &&
        requestMessages[0]?.role !== "system"
      ) {
        requestMessages.unshift({ role: "system", content: trimmedSystemPrompt })
      }

      return requestMessages
    },
    [history, systemPrompt, toPreviewUserContent]
  )

  const buildNormalPreviewRequest = React.useCallback(
    async (modelId: string, draftMessage: string, draftImage: string) => {
      const normalizedModel = String(modelId || "").replace(/^tldw:/, "").trim()
      const resolvedProvider = await resolveApiProviderForModel({
        modelId: normalizedModel,
        explicitProvider: currentChatModelSettings.apiProvider
      })
      const modelSupportsTools = supportsCapability(normalizedModel, "tools")
      const toolRequest = resolveChatToolRequest({
        tools: mcpTools,
        toolChoice,
        modelSupportsTools,
        mcpHealthState,
        hasMcp
      })
      const parsedExtraHeaders = parseJsonObject(currentChatModelSettings.extraHeaders)
      const previewExtraHeaders = {
        ...(parsedExtraHeaders ?? {})
      }
      if (toolRequest.tools) {
        previewExtraHeaders["X-TLDW-Loop-Compat"] = "1"
      }
      const request = omitUndefinedFields({
        messages: toPreviewHistoryMessages(normalizedModel, draftMessage, draftImage),
        model: normalizedModel,
        stream: true,
        temperature: currentChatModelSettings.temperature,
        max_tokens: currentChatModelSettings.numPredict,
        top_p: currentChatModelSettings.topP,
        frequency_penalty: currentChatModelSettings.frequencyPenalty,
        presence_penalty: currentChatModelSettings.presencePenalty,
        reasoning_effort:
          currentChatModelSettings.reasoningEffort === "low" ||
          currentChatModelSettings.reasoningEffort === "medium" ||
          currentChatModelSettings.reasoningEffort === "high"
            ? currentChatModelSettings.reasoningEffort
            : undefined,
        ...(toolRequest.tools
          ? {
              tools: toolRequest.tools,
              ...(toolRequest.toolChoice
                ? { tool_choice: toolRequest.toolChoice }
                : {})
            }
          : {}),
        save_to_db: !temporaryChat && Boolean(serverChatId),
        conversation_id: !temporaryChat && serverChatId ? serverChatId : undefined,
        history_message_limit: currentChatModelSettings.historyMessageLimit,
        history_message_order: currentChatModelSettings.historyMessageOrder,
        slash_command_injection_mode:
          currentChatModelSettings.slashCommandInjectionMode,
        api_provider: resolvedProvider || undefined,
        extra_headers:
          Object.keys(previewExtraHeaders).length > 0
            ? previewExtraHeaders
            : undefined,
        extra_body: parseJsonObject(currentChatModelSettings.extraBody),
        thinking_budget_tokens:
          currentChatModelSettings.llamaThinkingBudgetTokens,
        grammar_mode:
          currentChatModelSettings.llamaGrammarMode === "library" ||
          currentChatModelSettings.llamaGrammarMode === "inline"
            ? currentChatModelSettings.llamaGrammarMode
            : "none",
        grammar_id: currentChatModelSettings.llamaGrammarId,
        grammar_inline: currentChatModelSettings.llamaGrammarInline,
        grammar_override: currentChatModelSettings.llamaGrammarOverride,
        response_format: currentChatModelSettings.jsonMode
          ? { type: "json_object" }
          : undefined,
        research_context: compareModeActive ? undefined : researchContext
      } satisfies Record<string, unknown>) as ChatCompletionRequest
      return {
        request,
        metadata: {
          model: normalizedModel,
          toolChoice: toolRequest.toolChoice,
          toolOmissionReason: toolRequest.omittedReason,
          toolCounts: toolRequest.counts
        }
      } satisfies PreviewChatRequestBuild
    },
    [
      currentChatModelSettings,
      hasMcp,
      mcpHealthState,
      mcpTools,
      serverChatId,
      temporaryChat,
      toPreviewHistoryMessages,
      toolChoice,
      supportsCapability
    ]
  )

  const buildCurrentRawRequestSnapshot = React.useCallback(async () => {
    const intent = resolveSubmissionIntent(formMessage || "")
    let draftMessage = intent.message.trim()
    // Append pinned source expansion to match submitForm behavior
    if (!intent.isImageCommand && !fileRetrievalEnabled && ragPinnedResults && ragPinnedResults.length > 0) {
      const pinnedText = formatPinnedResults(ragPinnedResults, "markdown")
      draftMessage = draftMessage ? `${draftMessage}\n\n${pinnedText}` : pinnedText
    }
    const draftImage = intent.isImageCommand ? "" : String(formImage || "")
    const hasScopedRagMediaIds =
      Array.isArray(ragMediaIds) && ragMediaIds.length > 0
    const shouldUseRag =
      Boolean(selectedKnowledge) || (fileRetrievalEnabled && hasScopedRagMediaIds)
    const hasContextFiles = Array.isArray(contextFiles) && contextFiles.length > 0
    const hasDocs =
      (Array.isArray(selectedDocuments) && selectedDocuments.length > 0) ||
      (Array.isArray(documentContext) && documentContext.length > 0)
    const isCharacterFlow =
      !compareModeActive &&
      !intent.isImageCommand &&
      !hasContextFiles &&
      !hasDocs &&
      !shouldUseRag &&
      Boolean(selectedCharacter?.id)

    if (intent.isImageCommand) {
      const backend =
        intent.imageBackendOverride || imageBackendDefaultTrimmed || "image-generation"
      return {
        endpoint: "/api/v1/files/create",
        method: "POST",
        mode: "non-stream" as const,
        sentAt: new Date().toISOString(),
        body: {
          file_type: "image",
          payload: {
            backend,
            prompt: draftMessage
          },
          export: {
            format: "png",
            mode: "inline",
            async_mode: "sync"
          },
          options: {
            persist: true
          }
        }
      } satisfies ChatRequestDebugSnapshot
    }

    if (isCharacterFlow) {
      const resolvedModel = String(selectedModel || "").replace(/^tldw:/, "").trim()
      const provider = await resolveApiProviderForModel({
        modelId: resolvedModel,
        explicitProvider: currentChatModelSettings.apiProvider
      })
      const chatIdHint = serverChatId || "<new-chat-id>"
      const messagePayload: Record<string, unknown> = {
        role: "user",
        content: draftMessage
      }
      const normalizedImage = String(formImage || "")
      if (normalizedImage.startsWith("data:")) {
        const b64 = normalizedImage.includes(",")
          ? normalizedImage.split(",")[1]
          : normalizedImage
        if (b64 && b64.length > 0) {
          messagePayload.image_base64 = b64
        }
      }
      return {
        endpoint: `/api/v1/chats/${chatIdHint}/complete-v2`,
        method: "POST",
        mode: "stream",
        sentAt: new Date().toISOString(),
        body: {
          sequence: [
            !serverChatId
              ? {
                  endpoint: "/api/v1/chats/",
                  method: "POST",
                  body: {
                    character_id: selectedCharacter?.id,
                    state: serverChatState || "in-progress",
                    source: serverChatSource || undefined
                  }
                }
              : null,
            {
              endpoint: `/api/v1/chats/${chatIdHint}/messages`,
              method: "POST",
              body: messagePayload
            },
            {
              endpoint: `/api/v1/chats/${chatIdHint}/complete-v2`,
              method: "POST",
              body: {
                include_character_context: true,
                model: resolvedModel,
                provider: provider || undefined,
                save_to_db: !temporaryChat,
                continue_as_user: messageSteeringMode === "continue_as_user",
                impersonate_user: messageSteeringMode === "impersonate_user",
                force_narrate: Boolean(messageSteeringForceNarrate),
                stream: true
              }
            }
          ].filter(Boolean)
        }
      } satisfies ChatRequestDebugSnapshot
    }

    const modelsForPreview = compareModeActive
      ? Array.from(
          new Set(
            compareSelectedModels.length > 0
              ? compareSelectedModels
              : selectedModel
                ? [selectedModel]
                : []
          )
        )
      : selectedModel
        ? [selectedModel]
        : []
    const limitedModels =
      compareModeActive && compareMaxModels > 0
        ? modelsForPreview.slice(0, compareMaxModels)
        : modelsForPreview

    if (limitedModels.length === 0) {
      return {
        endpoint: "/api/v1/chat/completions",
        method: "POST",
        mode: "stream",
        sentAt: new Date().toISOString(),
        body: {
          error: "No model selected"
        }
      } satisfies ChatRequestDebugSnapshot
    }

    if (limitedModels.length === 1) {
      const preview = await buildNormalPreviewRequest(
        limitedModels[0],
        draftMessage,
        draftImage
      )
      return {
        endpoint: "/api/v1/chat/completions",
        method: "POST",
        mode: "stream",
        sentAt: new Date().toISOString(),
        body: preview.request,
        metadata: preview.metadata
      } satisfies ChatRequestDebugSnapshot
    }

    const previews = await Promise.all(
      limitedModels.map((modelId) =>
        buildNormalPreviewRequest(modelId, draftMessage, draftImage)
      )
    )
    return {
      endpoint: "/api/v1/chat/completions",
      method: "POST",
      mode: "stream",
      sentAt: new Date().toISOString(),
      body: {
        compare_mode: true,
        requests: previews.map((preview) => preview.request)
      },
      metadata: {
        toolRequests: previews.map((preview) => preview.metadata)
      }
    } satisfies ChatRequestDebugSnapshot
  }, [
    buildNormalPreviewRequest,
    compareMaxModels,
    compareModeActive,
    compareSelectedModels,
    contextFiles,
    currentChatModelSettings.apiProvider,
    documentContext,
    fileRetrievalEnabled,
    formImage,
    formMessage,
    imageBackendDefaultTrimmed,
    messageSteeringForceNarrate,
    messageSteeringMode,
    ragMediaIds,
    ragPinnedResults,
    resolveSubmissionIntent,
    selectedCharacter?.id,
    selectedKnowledge,
    selectedModel,
    selectedDocuments,
    researchContext,
    serverChatId,
    serverChatSource,
    serverChatState,
    temporaryChat
  ])

  const rawRequestJson = React.useMemo(
    () =>
      rawRequestSnapshot
        ? JSON.stringify(rawRequestSnapshot.body, null, 2)
        : "",
    [rawRequestSnapshot]
  )

  const refreshRawRequestSnapshot = React.useCallback(async () => {
    try {
      const snapshot = await buildCurrentRawRequestSnapshot()
      setRawRequestSnapshot(snapshot)
      captureChatRequestDebugSnapshot({
        endpoint: snapshot.endpoint,
        method: snapshot.method,
        mode: snapshot.mode,
        body: snapshot.body,
        metadata: snapshot.metadata
      })
    } catch (error) {
      console.error("Failed to build current request preview", error)
      setRawRequestSnapshot(null)
      notificationApi.error({
        message: t("error", { defaultValue: "Error" }),
        description: t(
          "playground:tools.rawChatRequestBuildFailed",
          "Failed to generate request preview from current input."
        )
      })
    }
  }, [buildCurrentRawRequestSnapshot, notificationApi, t])

  const openRawRequestModal = React.useCallback(() => {
    setToolsPopoverOpen(false)
    setRawRequestSnapshot(null)
    setRawRequestModalOpen(true)
    void refreshRawRequestSnapshot()
  }, [refreshRawRequestSnapshot, setToolsPopoverOpen])

  const copyRawRequestJson = React.useCallback(async () => {
    if (!rawRequestJson) return
    try {
      await navigator.clipboard.writeText(rawRequestJson)
      notificationApi.success?.({
        message: t("common:copied", "Copied"),
        description: t(
          "playground:tools.rawChatRequestCopied",
          "Copied request JSON to clipboard."
        )
      })
    } catch {
      notificationApi.error({
        message: t("error", { defaultValue: "Error" }),
        description: t(
          "playground:tools.rawChatRequestCopyFailed",
          "Unable to copy request JSON."
        )
      })
    }
  }, [notificationApi, rawRequestJson, t])

  return {
    rawRequestModalOpen,
    setRawRequestModalOpen,
    rawRequestSnapshot,
    rawRequestJson,
    refreshRawRequestSnapshot,
    openRawRequestModal,
    copyRawRequestJson
  }
}

export type UsePlaygroundRawPreviewReturn = ReturnType<typeof usePlaygroundRawPreview>
