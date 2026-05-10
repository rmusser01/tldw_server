import { ChatTldw } from "./ChatTldw"
import type { ChatResearchContext } from "@/services/tldw/TldwApiClient"
import {
  getAllDefaultModelSettings,
  getModelSettings
} from "@/services/model-settings"
import { getDefaultApiProvider } from "@/services/tldw-server"
import { tldwModels } from "@/services/tldw"
import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessageOption, type ToolChoice } from "@/store/option"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { resolveChatToolRequest } from "@/utils/chat-tools"
import { resolveApiProviderForModel } from "@/utils/resolve-api-provider"

const isValidReasoningEffort = (
  value: unknown
): value is "low" | "medium" | "high" => {
  return value === "low" || value === "medium" || value === "high"
}

type PageAssistModelOptions = {
  model: string
  toolChoice?: ToolChoice
  tools?: Record<string, unknown>[]
  saveToDb?: boolean
  conversationId?: string
  historyMessageLimit?: number
  historyMessageOrder?: string
  slashCommandInjectionMode?: string
  apiProvider?: string
  extraHeaders?: string
  extraBody?: string
  researchContext?: ChatResearchContext
}

const parseJsonObject = (value?: string) => {
  if (!value || typeof value !== "string") return undefined
  const trimmed = value.trim()
  if (!trimmed) return undefined
  try {
    const parsed = JSON.parse(trimmed)
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
  } catch {
    return undefined
  }
  return undefined
}

export const pageAssistModel = async ({
  model,
  toolChoice,
  tools,
  saveToDb,
  conversationId,
  historyMessageLimit,
  historyMessageOrder,
  slashCommandInjectionMode,
  apiProvider,
  extraHeaders,
  extraBody,
  researchContext
}: PageAssistModelOptions): Promise<ChatTldw> => {
  const currentChatModelSettings = useStoreChatModelSettings.getState()
  const {
    toolChoice: storedToolChoice,
    serverChatId,
    temporaryChat
  } = useStoreMessageOption.getState()
  const {
    tools: storedTools,
    chatTools: storedChatTools,
    toolCounts,
    healthState: mcpHealthState
  } = useMcpToolsStore.getState()
  const resolvedToolChoice = toolChoice ?? storedToolChoice
  const normalizedModelId = String(model || "").replace(/^tldw:/, "")
  let modelSupportsTools = false
  let modelSupportsMultimodal = false
  try {
    const modelInfo = await tldwModels.getModel(normalizedModelId)
    modelSupportsTools = Boolean(modelInfo?.capabilities?.includes("tools"))
    modelSupportsMultimodal = Boolean(
      modelInfo?.capabilities?.includes("vision")
    )
  } catch {
    modelSupportsTools = false
    modelSupportsMultimodal = false
  }
  const resolvedToolCandidates = tools ?? storedChatTools ?? storedTools
  const hasStoredFilterCounts = Object.values(toolCounts).some(
    (count) => count > 0
  )
  const toolRequest = resolveChatToolRequest({
    tools: resolvedToolCandidates,
    toolChoice: resolvedToolChoice,
    modelSupportsTools,
    mcpHealthState,
    hasMcp: true,
    counts: !tools && hasStoredFilterCounts ? toolCounts : undefined
  })
  const resolvedConversationId =
    conversationId && conversationId.trim().length > 0
      ? conversationId.trim()
      : serverChatId ?? undefined
  const resolvedSaveToDb =
    typeof saveToDb === "boolean"
      ? saveToDb
      : Boolean(resolvedConversationId) && !temporaryChat
  const finalConversationId = resolvedSaveToDb
    ? resolvedConversationId
    : undefined
  const resolvedHistoryMessageLimit =
    typeof historyMessageLimit === "number"
      ? historyMessageLimit
      : currentChatModelSettings.historyMessageLimit
  const normalizedHistoryMessageLimit =
    typeof resolvedHistoryMessageLimit === "number" &&
    resolvedHistoryMessageLimit > 0
      ? resolvedHistoryMessageLimit
      : undefined
  const resolvedHistoryMessageOrder =
    historyMessageOrder ?? currentChatModelSettings.historyMessageOrder
  const normalizedHistoryMessageOrder =
    resolvedHistoryMessageOrder && resolvedHistoryMessageOrder.trim().length > 0
      ? resolvedHistoryMessageOrder.trim()
      : undefined
  const resolvedSlashInjectionMode =
    slashCommandInjectionMode ??
    currentChatModelSettings.slashCommandInjectionMode
  const normalizedSlashInjectionMode =
    resolvedSlashInjectionMode && resolvedSlashInjectionMode.trim().length > 0
      ? resolvedSlashInjectionMode.trim()
      : undefined
  const rawProvider = apiProvider ?? currentChatModelSettings.apiProvider
  const explicitProvider =
    typeof rawProvider === "string" && rawProvider.trim().length > 0
      ? rawProvider.trim()
      : undefined
  const defaultApiProvider =
    !explicitProvider ? await getDefaultApiProvider() : null
  const normalizedApiProvider = await resolveApiProviderForModel({
    modelId: model,
    explicitProvider: explicitProvider ?? defaultApiProvider ?? undefined
  })
  const resolvedExtraHeadersBase = parseJsonObject(
    extraHeaders ?? currentChatModelSettings.extraHeaders
  )
  const resolvedExtraHeaders = (() => {
    const headers = {
      ...(resolvedExtraHeadersBase ?? {})
    } as Record<string, unknown>
    if (toolRequest.tools) {
      headers["X-TLDW-Loop-Compat"] = "1"
    }
    return Object.keys(headers).length > 0 ? headers : undefined
  })()
  const resolvedExtraBody = parseJsonObject(
    extraBody ?? currentChatModelSettings.extraBody
  )
  const userDefaultModelSettings = await getAllDefaultModelSettings()

  const {
    keepAlive,
    temperature,
    topK,
    topP,
    numCtx,
    seed,
    numGpu,
    numPredict,
    useMMap,
    minP,
    repeatLastN,
    repeatPenalty,
    tfsZ,
    numKeep,
    numThread,
    useMlock,
    reasoningEffort
  } = {
    keepAlive:
      currentChatModelSettings?.keepAlive ??
      userDefaultModelSettings?.keepAlive,
    temperature:
      currentChatModelSettings?.temperature ??
      userDefaultModelSettings?.temperature,
    topK: currentChatModelSettings?.topK ?? userDefaultModelSettings?.topK,
    topP: currentChatModelSettings?.topP ?? userDefaultModelSettings?.topP,
    numCtx:
      currentChatModelSettings?.numCtx ?? userDefaultModelSettings?.numCtx,
    seed: currentChatModelSettings?.seed,
    numGpu:
      currentChatModelSettings?.numGpu ?? userDefaultModelSettings?.numGpu,
    numPredict:
      currentChatModelSettings?.numPredict ??
      userDefaultModelSettings?.numPredict,
    useMMap:
      currentChatModelSettings?.useMMap ?? userDefaultModelSettings?.useMMap,
    minP: currentChatModelSettings?.minP ?? userDefaultModelSettings?.minP,
    repeatLastN:
      currentChatModelSettings?.repeatLastN ??
      userDefaultModelSettings?.repeatLastN,
    repeatPenalty:
      currentChatModelSettings?.repeatPenalty ??
      userDefaultModelSettings?.repeatPenalty,
    tfsZ: currentChatModelSettings?.tfsZ ?? userDefaultModelSettings?.tfsZ,
    numKeep:
      currentChatModelSettings?.numKeep ?? userDefaultModelSettings?.numKeep,
    numThread:
      currentChatModelSettings?.numThread ??
      userDefaultModelSettings?.numThread,
    useMlock:
      currentChatModelSettings?.useMlock ?? userDefaultModelSettings?.useMlock,
    reasoningEffort:
      currentChatModelSettings?.reasoningEffort ??
      userDefaultModelSettings?.reasoningEffort
  }

  const modelSettings = await getModelSettings(model)

  const _keepAlive = modelSettings?.keepAlive || keepAlive || ""
  const payload = {
    keepAlive: _keepAlive.length > 0 ? _keepAlive : undefined,
    temperature: modelSettings?.temperature || temperature,
    topK: modelSettings?.topK || topK,
    topP: modelSettings?.topP || topP,
    numCtx: modelSettings?.numCtx || numCtx,
    numGpu: modelSettings?.numGpu || numGpu,
    numPredict: modelSettings?.numPredict || numPredict,
    useMMap: modelSettings?.useMMap || useMMap,
    minP: modelSettings?.minP || minP,
    repeatPenalty: modelSettings?.repeatPenalty || repeatPenalty,
    repeatLastN: modelSettings?.repeatLastN || repeatLastN,
    tfsZ: modelSettings?.tfsZ || tfsZ,
    numKeep: modelSettings?.numKeep || numKeep,
    numThread: modelSettings?.numThread || numThread,
    useMlock: modelSettings?.useMlock || useMlock
  }

  // Default to tldw_server chat model
  return new ChatTldw({
    model,
    temperature: payload.temperature,
    topP: payload.topP,
    maxTokens: payload.numPredict,
    streaming: true,
    reasoningEffort: isValidReasoningEffort(modelSettings?.reasoningEffort)
      ? modelSettings.reasoningEffort
      : isValidReasoningEffort(reasoningEffort)
        ? reasoningEffort
        : undefined,
    toolChoice: toolRequest.toolChoice,
    tools: toolRequest.tools,
    supportsMultimodal: modelSupportsMultimodal,
    saveToDb: resolvedSaveToDb,
    conversationId: finalConversationId,
    historyMessageLimit: normalizedHistoryMessageLimit,
    historyMessageOrder: normalizedHistoryMessageOrder,
    slashCommandInjectionMode: normalizedSlashInjectionMode,
    apiProvider: normalizedApiProvider,
    extraHeaders: resolvedExtraHeaders,
    extraBody: resolvedExtraBody,
    researchContext,
    chatDebugMetadata: {
      toolChoice: toolRequest.toolChoice,
      toolOmissionReason: toolRequest.omittedReason,
      toolCounts: toolRequest.counts
    }
  })
}
