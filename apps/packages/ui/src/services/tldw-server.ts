import { Storage } from "@plasmohq/storage"
import { tldwClient, tldwModels } from "./tldw"
import { setNoOfRetrievedDocs, setTotalFilePerKB } from "./app"
import { createSafeStorage } from "@/utils/safe-storage"
import {
  resolveWebUiQuickstartServerUrl,
  type BrowserSurface
} from "@/services/tldw/browser-networking"

const storage = createSafeStorage()

// Default local tldw_server endpoint
const DEFAULT_TLDW_URL = "http://127.0.0.1:8000"

const getCurrentBrowserSurface = (): BrowserSurface => {
  if (typeof window === "undefined") {
    return "extension"
  }

  try {
    const protocol = String(window.location?.protocol || "").trim().toLowerCase()
    if (protocol === "chrome-extension:" || protocol === "moz-extension:") {
      return "extension"
    }
    if (protocol === "http:" || protocol === "https:") {
      return "webui-page"
    }
  } catch {
    // Fall through to the browser-app default.
  }

  return "browser-app"
}

const getQuickstartWebUiServerUrl = (
): string | null => {
  try {
    return resolveWebUiQuickstartServerUrl({
      surface: getCurrentBrowserSurface(),
      deploymentMode: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
      pageOrigin:
        typeof window === "undefined" ? null : String(window.location?.origin || "").trim(),
      apiOrigin: process.env.NEXT_PUBLIC_API_URL
    })
  } catch {
    return null
  }
}

// Read API key from environment variables
export const DEFAULT_TLDW_API_KEY = import.meta?.env?.VITE_TLDW_API_KEY || ""

/**
 * Read any previously stored tldw server URL from extension storage,
 * without falling back to the hard-coded default.
 *
 * This is used by connection bootstrap code to distinguish a true
 * first-run (no URL configured anywhere) from a misconfigured server.
 */
export const getStoredTldwServerURL = async (): Promise<string | null> => {
  try {
    const url = await storage.get("tldwServerUrl")
    if (typeof url === "string") {
      const trimmed = url.trim()
      if (trimmed.length > 0) {
        return trimmed
      }
    }
  } catch {
    // Ignore storage read failures; caller will treat as "no URL".
  }
  return null
}

export const getTldwServerURL = async () => {
  const config = await tldwClient.getConfig()
  const quickstartWebUiServerUrl = getQuickstartWebUiServerUrl()
  if (quickstartWebUiServerUrl) {
    return quickstartWebUiServerUrl
  }
  if (config?.serverUrl) {
    return config.serverUrl
  }
  // Fallback to stored URL or default
  const stored = await getStoredTldwServerURL()
  const quickstartStoredServerUrl = getQuickstartWebUiServerUrl()
  if (quickstartStoredServerUrl) {
    return quickstartStoredServerUrl
  }
  return stored || DEFAULT_TLDW_URL
}

export const setTldwServerURL = async (url: string) => {
  await storage.set("tldwServerUrl", url)
  await tldwClient.updateConfig({ serverUrl: url })
  clearChatModelsCache()
}

export const isTldwServerRunning = async () => {
  try {
    const config = await tldwClient.getConfig()
    if (!config) return false

    const health = await tldwClient.healthCheck()
    return health
  } catch (e) {
    console.error("tldw server not running:", e)
    return false
  }
}

const mapTldwModelToUi = (model: any) => ({
  name: `tldw:${model.id}`,
  model: `tldw:${model.id}`,
  provider: String(model.provider || "unknown").toLowerCase(),
  nickname: model.name || model.id,
  context_length: model.contextLength,
  avatar: undefined,
  modified_at: new Date().toISOString(),
  size: 0,
  digest: "",
  is_configured:
    typeof model.isConfigured === "boolean" ? model.isConfigured : undefined,
  provider_is_configured:
    typeof model.providerIsConfigured === "boolean"
      ? model.providerIsConfigured
      : undefined,
  catalog_only:
    typeof model.catalogOnly === "boolean" ? model.catalogOnly : undefined,
  details: {
    provider: model.provider,
    capabilities: model.capabilities,
    type: model.type,
    modalities: model.modalities,
    is_configured:
      typeof model.isConfigured === "boolean" ? model.isConfigured : undefined,
    provider_is_configured:
      typeof model.providerIsConfigured === "boolean"
        ? model.providerIsConfigured
        : undefined,
    catalog_only:
      typeof model.catalogOnly === "boolean" ? model.catalogOnly : undefined
  }
})

const CHAT_MODELS_CACHE_TTL_MS = 60_000
let chatModelsCache: { value: any[]; expiresAt: number } | null = null
let chatModelsInFlight: Promise<any[]> | null = null

const dedupeChatModelsByModel = (models: any[]) => {
  const unique: any[] = []
  const indexByModel = new Map<string, number>()
  const duplicates: string[] = []

  for (const model of models) {
    const modelId = String(model?.model || model?.name || "").trim()
    const provider = String(model?.provider || "unknown").trim().toLowerCase()
    const key = modelId ? `${provider}:${modelId}` : ""
    if (!key) {
      unique.push(model)
      continue
    }
    const existingIndex = indexByModel.get(key)
    if (existingIndex == null) {
      indexByModel.set(key, unique.length)
      unique.push(model)
      continue
    }
    duplicates.push(key)
    const existing = unique[existingIndex] || {}
    const merged: any = { ...existing }
    if (!merged.nickname && model?.nickname) merged.nickname = model.nickname
    if (!merged.name && model?.name) merged.name = model.name
    if (!merged.provider && model?.provider) merged.provider = model.provider
    if (!merged.details && model?.details) merged.details = model.details
    if (!merged.modified_at && model?.modified_at) {
      merged.modified_at = model.modified_at
    }

    const existingDetails =
      merged.details && typeof merged.details === "object"
        ? merged.details
        : {}
    const incomingDetails =
      model?.details && typeof model.details === "object"
        ? model.details
        : {}
    const mergedDetails: any = { ...incomingDetails, ...existingDetails }
    const capabilities = new Set<string>()
    const existingCaps = existingDetails.capabilities
    const incomingCaps = incomingDetails.capabilities
    if (Array.isArray(existingCaps)) {
      existingCaps.forEach((cap) => capabilities.add(String(cap)))
    }
    if (Array.isArray(incomingCaps)) {
      incomingCaps.forEach((cap) => capabilities.add(String(cap)))
    }
    if (capabilities.size > 0) {
      mergedDetails.capabilities = Array.from(capabilities)
    }
    if (Object.keys(mergedDetails).length > 0) {
      merged.details = mergedDetails
    }
    unique[existingIndex] = merged
  }

  if (import.meta.env?.DEV && duplicates.length > 0) {
    const uniqueDupes = Array.from(new Set(duplicates))
    console.debug("tldw_server: deduped chat models", {
      duplicates: uniqueDupes,
      total: models.length,
      unique: unique.length
    })
  }

  return unique
}

export const clearChatModelsCache = () => {
  chatModelsCache = null
  chatModelsInFlight = null
}

export const getAllModels = async ({ returnEmpty = false }: { returnEmpty?: boolean }) => {
  try {
    // If no config, avoid network calls when returnEmpty requested
    try {
      const cfg = await tldwClient.getConfig()
      if (!cfg) {
        if (returnEmpty) return []
      }
    } catch {
      if (returnEmpty) return []
    }
    // Use the richer tldwModels API (backed by /api/v1/llm/models/metadata)
    const models = await tldwModels.getModels()
    return models.map(mapTldwModelToUi)
  } catch (e) {
    if (!returnEmpty) console.error("Failed to fetch tldw models:", e)
    if (returnEmpty) return []
    throw e
  }
}

export const fetchChatModels = async ({
  returnEmpty = false,
  forceRefresh = false,
  refreshOpenRouter = false,
  allowNetwork = true
}: {
  returnEmpty?: boolean
  forceRefresh?: boolean
  refreshOpenRouter?: boolean
  allowNetwork?: boolean
} = {}) => {
  const now = Date.now()
  if (!forceRefresh && chatModelsCache && chatModelsCache.expiresAt > now) {
    return chatModelsCache.value
  }
  if (!forceRefresh && !allowNetwork) {
    if (chatModelsCache?.value) {
      return chatModelsCache.value
    }

    const cachedChatModels = await tldwModels.getCachedChatModels()
    const resolved = dedupeChatModelsByModel(cachedChatModels.map(mapTldwModelToUi))
    if (resolved.length > 0) {
      chatModelsCache = {
        value: resolved,
        expiresAt: Date.now() + CHAT_MODELS_CACHE_TTL_MS
      }
    }
    return resolved
  }
  if (!forceRefresh && chatModelsInFlight) {
    return await chatModelsInFlight
  }

  try {
    const fetchPromise = (async () => {
      // Primary: tldw_server aggregated models
      const chatModels = await tldwModels.getChatModels(forceRefresh, {
        refreshOpenRouter: refreshOpenRouter || forceRefresh
      })
      const tldw = chatModels.map(mapTldwModelToUi)

      // Only tldw_server models are exposed as chat models
      const combined = [...tldw]

      if (import.meta.env?.DEV) {
        console.debug("tldw_server: fetchChatModels resolved", {
          tldwCount: tldw.length,
          total: combined.length
        })
      }

      const resolved = dedupeChatModelsByModel(combined)
      chatModelsCache = {
        value: resolved,
        expiresAt: Date.now() + CHAT_MODELS_CACHE_TTL_MS
      }
      return resolved
    })()

    chatModelsInFlight = fetchPromise
    return await fetchPromise
  } catch (e) {
    console.error("Failed to fetch chat models:", e)
    if (chatModelsCache?.value?.length) {
      return chatModelsCache.value
    }
    if (returnEmpty) return []
    throw e
  } finally {
    chatModelsInFlight = null
  }
}

export const fetchImageModels = async ({
  returnEmpty = false
}: { returnEmpty?: boolean } = {}) => {
  try {
    const models = await tldwModels.getImageModels()
    return models
  } catch (e) {
    if (!returnEmpty) {
      console.error("Failed to fetch image models:", e)
    }
    return []
  }
}

// Compatibility aliases
export const getOllamaURL = getTldwServerURL
export const setOllamaURL = setTldwServerURL
export const isOllamaRunning = isTldwServerRunning

// ─────────────────────────────────────────────────────────────────────────────
// Settings formerly in ollama.ts - now consolidated here
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_PAGE_SHARE_URL = "https://pageassist.xyz"

const DEFAULT_RAG_QUESTION_PROMPT =
  "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.   Chat History: {chat_history} Follow Up Input: {question} Standalone question:"

const DEFAULT_RAG_SYSTEM_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know. DO NOT try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.  {context}  Question: {question} Helpful answer:`

const DEFAULT_WEBSEARCH_PROMPT = `You are an AI model who is expert at searching the web and answering user's queries.

Generate a response that is informative and relevant to the user's query based on provided search results. the current date and time are {current_date_time}.

\`search-results\` block provides knowledge from the web search results. You can use this information to generate a meaningful response.

<search-results>
 {search_results}
</search-results>
`

const DEFAULT_WEBSEARCH_FOLLOWUP_PROMPT = `You will rephrase follow-up questions into concise, standalone search queries optimized for internet search engines. Transform conversational questions into keyword-focused search terms by removing unnecessary words, question formats, and context dependencies while preserving the core information need.

ONLY RETURN QUERY WITHOUT ANY TEXT

Examples:
Follow-up question: What are the symptoms of a heart attack?
heart attack symptoms

Follow-up question: Where is the upcoming Olympics being held?
upcoming Olympics

Follow-up question: Taylor Swift's latest album?
Taylor Swift latest album ${new Date().getFullYear()}

Follow-up question: How does photosynthesis work in plants?
photosynthesis process plants

Follow-up question: What's the current stock price of Apple?
Apple stock price today

Previous Conversation:
{chat_history}

Follow-up question: {question}
`

export const getPageShareUrl = async () => {
  const pageShareUrl = await storage.get("pageShareUrl")
  if (!pageShareUrl || typeof pageShareUrl !== "string" || pageShareUrl.length === 0) {
    return DEFAULT_PAGE_SHARE_URL
  }
  return pageShareUrl
}

export const setPageShareUrl = async (pageShareUrl: string) => {
  await storage.set("pageShareUrl", pageShareUrl)
}

export const systemPromptForNonRag = async () => {
  const prompt = await storage.get("systemPromptForNonRag")
  if (typeof prompt !== "string" || prompt.length === 0) {
    return undefined
  }
  return prompt
}

export const setSystemPromptForNonRag = async (prompt: string) => {
  await storage.set("systemPromptForNonRag", prompt)
}

export const systemPromptForNonRagOption = async () => {
  const prompt = await storage.get("systemPromptForNonRagOption")
  if (typeof prompt !== "string" || prompt.length === 0) {
    return undefined
  }
  return prompt
}

export const setSystemPromptForNonRagOption = async (prompt: string) => {
  await storage.set("systemPromptForNonRagOption", prompt)
}

export const promptForRag = async () => {
  const prompt = await storage.get("systemPromptForRag")
  const questionPrompt = await storage.get("questionPromptForRag")

  let ragPrompt = typeof prompt === "string" ? prompt : undefined
  let ragQuestionPrompt = typeof questionPrompt === "string" ? questionPrompt : undefined

  if (!ragPrompt || ragPrompt.length === 0) {
    ragPrompt = DEFAULT_RAG_SYSTEM_PROMPT
  }

  if (!ragQuestionPrompt || ragQuestionPrompt.length === 0) {
    ragQuestionPrompt = DEFAULT_RAG_QUESTION_PROMPT
  }

  return {
    ragPrompt,
    ragQuestionPrompt
  }
}

export const setPromptForRag = async (
  prompt: string,
  questionPrompt: string
) => {
  await storage.set("systemPromptForRag", prompt)
  await storage.set("questionPromptForRag", questionPrompt)
}

export const getWebSearchPrompt = async () => {
  const prompt = await storage.get("webSearchPrompt")
  if (!prompt || typeof prompt !== "string" || prompt.length === 0) {
    return DEFAULT_WEBSEARCH_PROMPT
  }
  return prompt
}

export const setWebSearchPrompt = async (prompt: string) => {
  await storage.set("webSearchPrompt", prompt)
}

export const geWebSearchFollowUpPrompt = async () => {
  const prompt = await storage.get("webSearchFollowUpPrompt")
  if (!prompt || typeof prompt !== "string" || prompt.length === 0) {
    return DEFAULT_WEBSEARCH_FOLLOWUP_PROMPT
  }
  return prompt
}

export const setWebSearchFollowUpPrompt = async (prompt: string) => {
  await storage.set("webSearchFollowUpPrompt", prompt)
}

export const setWebPrompts = async (prompt: string, followUpPrompt: string) => {
  await setWebSearchPrompt(prompt)
  await setWebSearchFollowUpPrompt(followUpPrompt)
}

export const defaultEmbeddingModelForRag = async () => {
  const embeddingMode = await storage.get("defaultEmbeddingModel")
  if (embeddingMode && typeof embeddingMode === "string" && embeddingMode.length > 0) {
    return embeddingMode
  }

  // Fallback: derive from tldw_server embeddings providers-config, if available
  try {
    const cfg =
      typeof (tldwClient as any).getEmbeddingProvidersConfig === "function"
        ? await (tldwClient as any).getEmbeddingProvidersConfig()
        : null

    if (cfg) {
      const provider = cfg?.default_provider
      const model = cfg?.default_model

      if (provider && model) {
        const id = `${provider}/${model}`
        await storage.set("defaultEmbeddingModel", id)
        return id
      }
    }
  } catch (e) {
    if (import.meta.env?.DEV) {
      console.warn("tldw_server: unable to resolve default embedding model from providers-config", e)
    }
  }

  return null
}

export const setDefaultEmbeddingModelForRag = async (model: string) => {
  await storage.set("defaultEmbeddingModel", model)
}

export const defaultEmbeddingChunkSize = async () => {
  const embeddingChunkSize = await storage.get("defaultEmbeddingChunkSize")
  if (!embeddingChunkSize || (embeddingChunkSize as string).length === 0) {
    return 1000
  }
  const parsed = parseInt(embeddingChunkSize as string, 10)
  return Number.isNaN(parsed) ? 1000 : parsed
}

export const setDefaultEmbeddingChunkSize = async (size: number) => {
  await storage.set("defaultEmbeddingChunkSize", size.toString())
}

export const defaultEmbeddingChunkOverlap = async () => {
  const embeddingChunkOverlap = await storage.get("defaultEmbeddingChunkOverlap")
  if (!embeddingChunkOverlap || (embeddingChunkOverlap as string).length === 0) {
    return 200
  }
  const parsed = parseInt(embeddingChunkOverlap as string, 10)
  return Number.isNaN(parsed) ? 200 : parsed
}

export const setDefaultEmbeddingChunkOverlap = async (overlap: number) => {
  await storage.set("defaultEmbeddingChunkOverlap", overlap.toString())
}

export const defaultSplittingStrategy = async () => {
  const splittingStrategy = await storage.get("defaultSplittingStrategy")
  if (!splittingStrategy || typeof splittingStrategy !== "string" || splittingStrategy.length === 0) {
    return "RecursiveCharacterTextSplitter"
  }
  return splittingStrategy
}

export const setDefaultSplittingStrategy = async (strategy: string) => {
  await storage.set("defaultSplittingStrategy", strategy)
}

export const defaultSplittingSeparator = async () => {
  const splittingSeparator = await storage.get("defaultSplittingSeparator")
  if (!splittingSeparator || typeof splittingSeparator !== "string" || splittingSeparator.length === 0) {
    return "\\n\\n"
  }
  return splittingSeparator
}

export const setDefaultSplittingSeparator = async (separator: string) => {
  await storage.set("defaultSplittingSeparator", separator)
}

export const saveForRag = async (
  model: string,
  chunkSize: number,
  overlap: number,
  totalFilePerKB: number,
  noOfRetrievedDocs?: number,
  strategy?: string,
  separator?: string
) => {
  await setDefaultEmbeddingModelForRag(model)
  await setDefaultEmbeddingChunkSize(chunkSize)
  await setDefaultEmbeddingChunkOverlap(overlap)
  await setTotalFilePerKB(totalFilePerKB)
  if (noOfRetrievedDocs) {
    await setNoOfRetrievedDocs(noOfRetrievedDocs)
  }
  if (strategy) {
    await setDefaultSplittingStrategy(strategy)
  }
  if (separator) {
    await setDefaultSplittingSeparator(separator)
  }
}

export const sendWhenEnter = async () => {
  const sendWhenEnterVal = await storage.get("sendWhenEnter")
  if (!sendWhenEnterVal || typeof sendWhenEnterVal !== "string" || sendWhenEnterVal.length === 0) {
    return true
  }
  return sendWhenEnterVal === "true"
}

export const setSendWhenEnter = async (sendWhenEnterVal: boolean) => {
  await storage.set("sendWhenEnter", sendWhenEnterVal.toString())
}

export const getSelectedModel = async (): Promise<string | null> => {
  const model = await storage.get("selectedModel")
  return typeof model === "string" ? model : null
}

export const setSelectedModel = async (model: string) => {
  await storage.set("selectedModel", model)
}

export const getDefaultApiProvider = async (): Promise<string | null> => {
  const provider = await storage.get("defaultApiProvider")
  if (typeof provider !== "string") return null
  const trimmed = provider.trim()
  if (!trimmed || trimmed.toLowerCase() === "auto") return null
  return trimmed
}

export const setDefaultApiProvider = async (
  provider: string | null
): Promise<void> => {
  const trimmed = typeof provider === "string" ? provider.trim() : ""
  if (!trimmed || trimmed.toLowerCase() === "auto") {
    await storage.remove("defaultApiProvider")
    return
  }
  await storage.set("defaultApiProvider", trimmed)
}

export const getEmbeddingModels = async () => {
  try {
    const models = await tldwModels.getEmbeddingModels(false)
    return models.map((m) => ({
      name: m.name || m.id,
      model: m.id,
      provider: m.provider,
      nickname: m.name || m.id,
      avatar: undefined
    }))
  } catch (e) {
    console.error("Failed to fetch embedding models:", e)
    return []
  }
}
