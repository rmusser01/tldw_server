import {
  type ChatHistory as ChatHistoryType,
  type Message as MessageType,
  type MessageVariant
} from "~/store/option"
import { ChatDocuments } from "@/models/ChatTypes"
import { normalizeChatRole } from "@/utils/normalize-chat-role"
import { isDatabaseClosedError } from "@/utils/ff-error"
import {
  type HistoryInfo,
  type CompareState,
  type MessageHistory,
  type Message,
  type Prompts,
  type UploadedFile,
  type SessionFiles,
  type Webshare,
  Prompt,
  LastUsedModelType,
  ModelNicknames,
  Models
} from "./types"
import { PageAssistDatabase } from "./chat"
import { db as chatDB } from "./schema"
import {
  deletePromptByIdFB,
  getAllPromptsFB,
  getPromptByIdFB,
  savePromptFB,
  updatePromptFB
} from ".."
import { ModelNickname } from "./nickname"
import { ModelDb } from "./models"

// Helper function to generate IDs (keeping the same format)
export const generateID = () => {
  return "pa_xxxx-xxxx-xxx-xxxx".replace(/[x]/g, () => {
    const r = Math.floor(Math.random() * 16)
    return r.toString(16)
  })
}

// Chat History Functions
export const saveHistory = async (
  title: string,
  is_rag?: boolean,
  message_source?: "copilot" | "web-ui" | "branch" | "server",
  doc_id?: string,
  server_chat_id?: string
) => {
  const id = generateID()
  const createdAt = Date.now()
  const history: HistoryInfo = {
    id,
    title: title?.trim()?.length > 0 ? title : "Untitled Chat",
    createdAt,
    is_rag: is_rag || false,
    message_source,
    doc_id,
    server_chat_id
  }
  const db = new PageAssistDatabase()
  await db.addChatHistory(history)
  return history
}

export const getHistoryByServerChatId = async (serverChatId: string) => {
  const db = new PageAssistDatabase()
  return await db.getHistoryByServerChatId(serverChatId)
}

export const getHistoryByDocId = async (docId: string) => {
  const db = new PageAssistDatabase()
  return await db.getHistoryByDocId(docId)
}

export const getAllHistoriesByDocId = async (docId: string) => {
  const db = new PageAssistDatabase()
  return await db.getAllHistoriesByDocId(docId)
}
export const updateChatHistoryCreatedAt = async (history_id: string) => {
  const createdAt = Date.now()
  const db = new PageAssistDatabase()
  await db.updateChatHistoryCreatedAt(history_id, createdAt)
}

export const setHistoryServerChatId = async (
  historyId: string,
  serverChatId: string
) => {
  const db = new PageAssistDatabase()
  await db.setHistoryServerChatId(historyId, serverChatId)
}

export const updateMessage = async (
  history_id: string,
  message_id: string,
  content: string
) => {
  const db = new PageAssistDatabase()
  await db.updateMessage(history_id, message_id, content)
}

export const updateMessageMedia = async (
  message_id: string,
  updates: { images?: string[]; generationInfo?: any }
) => {
  const db = new PageAssistDatabase()
  await db.updateMessageMedia(message_id, updates)
}

export const updateMessageDiscoSkillComment = async (
  message_id: string,
  discoSkillComment: Message["discoSkillComment"] | null
) => {
  const db = new PageAssistDatabase()
  await db.updateMessageDiscoSkillComment(message_id, discoSkillComment)
}

export const saveMessage = async ({
  id,
  content,
  history_id,
  name,
  role,
  images,
  source,
  discoSkillComment,
  generationInfo,
  metadataExtra,
  message_type,
  clusterId,
  modelId,
  modelImage,
  modelName,
  parent_message_id,
  reasoning_time_taken,
  time,
  documents,
  serverMessageId,
  createdAt: createdAtOverride
}: {
  id?: string
  history_id: string
  name: string
  role: string
  content: string
  images: string[]
  source?: any[]
  time?: number
  message_type?: string
  discoSkillComment?: Message["discoSkillComment"]
  clusterId?: string
  modelId?: string
  generationInfo?: any
  metadataExtra?: Record<string, unknown>
  reasoning_time_taken?: number
  modelName?: string
  modelImage?: string
  parent_message_id?: string | null
  documents?: ChatDocuments
  serverMessageId?: string | null
  createdAt?: number
}) => {
  const messageId = id ?? generateID()
  let createdAt =
    typeof createdAtOverride === "number"
      ? createdAtOverride
      : Date.now()
  if (typeof createdAtOverride !== "number" && time) {
    createdAt += time
  }
  const message: Message = {
    id: messageId,
    history_id,
    name,
    role,
    content,
    images,
    createdAt,
    sources: source,
    messageType: message_type,
    clusterId,
    modelId,
    generationInfo: generationInfo,
    metadataExtra,
    reasoning_time_taken,
    modelName,
    modelImage,
    parent_message_id: parent_message_id ?? null,
    documents,
    discoSkillComment,
    serverMessageId: serverMessageId ?? undefined
  }
  const db = new PageAssistDatabase()
  await db.addMessage(message)
  return message
}

export const getCompareState = async (
  history_id: string
): Promise<CompareState | null> => {
  const db = new PageAssistDatabase()
  return await db.getCompareState(history_id)
}

export const saveCompareState = async (state: CompareState) => {
  const db = new PageAssistDatabase()
  await db.setCompareState(state)
  return state
}

export const deleteCompareState = async (history_id: string) => {
  const db = new PageAssistDatabase()
  await db.deleteCompareState(history_id)
}

const shouldGroupVariants = (message: Message): boolean => {
  if (normalizeChatRole(message.role) !== "assistant") return false
  if (!message.parent_message_id) return false
  const messageType = message.messageType || ""
  if (messageType.startsWith("compare:")) return false
  if (message.clusterId) return false
  return true
}

const buildVariantFromHistory = (message: Message): MessageVariant => ({
  id: message.id,
  message: message.content,
  sources: message.sources ?? [],
  images: message.images ?? [],
  generationInfo: message.generationInfo,
  metadataExtra: message.metadataExtra as MessageVariant["metadataExtra"],
  reasoning_time_taken: message.reasoning_time_taken,
  createdAt: message.createdAt,
  serverMessageId: message.serverMessageId,
  serverMessageVersion: message.serverMessageVersion
})

const collapseVariantMessages = (messages: MessageHistory) => {
  const sorted = [...messages].sort((a, b) => a.createdAt - b.createdAt)
  const variantsByParent = new Map<string, Message[]>()
  const lastIdByParent = new Map<string, string>()

  for (const message of sorted) {
    if (!shouldGroupVariants(message)) continue
    const parentId = message.parent_message_id || ""
    const existing = variantsByParent.get(parentId) || []
    existing.push(message)
    variantsByParent.set(parentId, existing)
    lastIdByParent.set(parentId, message.id)
  }

  const collapsed = sorted.filter((message) => {
    if (!shouldGroupVariants(message)) return true
    const parentId = message.parent_message_id || ""
    return lastIdByParent.get(parentId) === message.id
  })

  return { collapsed, variantsByParent }
}

export const formatToChatHistory = (
  messages: MessageHistory
): ChatHistoryType => {
  const { collapsed } = collapseVariantMessages(messages)
  return collapsed.map((message) => {
    return {
      content: message.content,
      role: normalizeChatRole(message.role),
      images: message.images
    }
  })
}

export const formatToMessage = (messages: MessageHistory): MessageType[] => {
  const { collapsed, variantsByParent } = collapseVariantMessages(messages)
  return collapsed.map((message) => {
    const normalizedRole = normalizeChatRole(message.role)
    const mapped: MessageType = {
      isBot: normalizedRole === "assistant",
      message: message.content,
      name: message.name,
      role: normalizedRole,
      sources: message?.sources || [],
      images: message.images || [],
      messageType: message?.messageType,
      clusterId: message?.clusterId,
      modelId: message?.modelId,
      parentMessageId: message?.parent_message_id ?? null,
      generationInfo: message?.generationInfo,
      metadataExtra: message?.metadataExtra as MessageType["metadataExtra"],
      reasoning_time_taken: message?.reasoning_time_taken,
      modelName: message?.modelName,
      modelImage: message?.modelImage,
      createdAt: message?.createdAt,
      id: message.id,
      serverMessageId: message.serverMessageId ?? undefined,
      serverMessageVersion: message.serverMessageVersion ?? undefined,
      documents: message?.documents,
      discoSkillComment: message?.discoSkillComment
    }
    if (shouldGroupVariants(message)) {
      const parentId = message.parent_message_id || ""
      const grouped = variantsByParent.get(parentId) || []
      if (grouped.length > 1) {
        const variants = grouped.map(buildVariantFromHistory)
        mapped.variants = variants
        mapped.activeVariantIndex = variants.length - 1
      }
    }
    return mapped
  })
}

export const deleteByHistoryId = async (history_id: string) => {
  const db = new PageAssistDatabase()
  await db.deleteMessage(history_id)
  await db.removeChatHistory(history_id)
  await db.deleteCompareState(history_id)
  return history_id
}

/**
 * Get full chat data including history info and all messages.
 * Used for undo functionality - captures state before deletion.
 */
export const getFullChatData = async (history_id: string) => {
  const historyInfo = await chatDB.chatHistories.get(history_id)
  if (!historyInfo) return null

  const db = new PageAssistDatabase()
  const messages = await db.getChatHistory(history_id)

  return {
    historyInfo,
    messages
  }
}

/**
 * Restore a deleted chat with its messages.
 * Used by undo functionality to bring back deleted conversations.
 */
export const restoreChat = async (data: {
  historyInfo: HistoryInfo
  messages: Message[]
}) => {
  const db = new PageAssistDatabase()

  // Restore the history record
  await db.addChatHistory(data.historyInfo)

  // Restore all messages
  for (const msg of data.messages) {
    await db.addMessage(msg)
  }

  return data.historyInfo.id
}

export const updateHistory = async (id: string, title: string) => {
  await chatDB.chatHistories.update(id, { title })
}

export const pinHistory = async (id: string, is_pinned: boolean) => {
  await chatDB.chatHistories.update(id, { is_pinned })
}

export const removeMessageUsingHistoryId = async (history_id: string) => {
  const db = new PageAssistDatabase()
  const chatHistory = await db.getChatHistory(history_id)
  if (chatHistory.length > 0) {
    const firstMessage = chatHistory.sort(
      (a, b) => b.createdAt - a.createdAt
    )[0]
    await db.removeMessage(history_id, firstMessage.id)
  }
}

export const updateMessageByIndex = async (
  history_id: string,
  index: number,
  message: string
) => {
  try {
    const db = new PageAssistDatabase()
    const chatHistory = await db.getChatHistory(history_id)
    const sortedHistory = chatHistory.sort((a, b) => a.createdAt - b.createdAt)
    if (sortedHistory[index]) {
      await db.updateMessage(history_id, sortedHistory[index].id, message)
    }
  } catch (e) {
    // temp chat will break
  }
}

export const removeMessageByIndex = async (
  history_id: string,
  index: number
) => {
  try {
    const db = new PageAssistDatabase()
    const chatHistory = await db.getChatHistory(history_id)
    const sortedHistory = chatHistory.sort((a, b) => a.createdAt - b.createdAt)
    const target = sortedHistory[index]
    if (target) {
      await db.removeMessage(history_id, target.id)
    }
  } catch {
    // temp chat will break
  }
}

export const deleteChatForEdit = async (history_id: string, index: number) => {
  const db = new PageAssistDatabase()
  const chatHistory = await db.getChatHistory(history_id)
  const sortedHistory = chatHistory.sort((a, b) => a.createdAt - b.createdAt)

  // Delete messages after the specified index
  const messagesToDelete = sortedHistory.slice(index + 1)
  for (const message of messagesToDelete) {
    await db.removeMessage(history_id, message.id)
  }
}

// Prompt Functions
export const getAllPrompts = async () => {
  try {
    const db = new PageAssistDatabase()
    return await db.getAllPrompts()
  } catch (e) {
    try {
      return await getAllPromptsFB()
    } catch {
      if (!isDatabaseClosedError(e)) {
        console.error("Failed to load prompts from Dexie and fallback storage:", e)
      }
      return []
    }
  }
}

export const savePrompt = async ({
  content,
  title,
  name,
  author,
  details,
  system_prompt,
  user_prompt,
  fewShotExamples,
  modulesConfig,
  promptFormat,
  promptSchemaVersion,
  structuredPromptDefinition,
  syncPayloadVersion,
  versionNumber,
  changeDescription,
  parentVersionId,
  serverParentVersionId,
  is_system = false,
  tags = [],
  keywords,
  favorite = false
}: {
  title: string
  name?: string
  content?: string
  author?: string
  details?: string
  system_prompt?: string
  user_prompt?: string
  fewShotExamples?: Prompt["fewShotExamples"]
  modulesConfig?: Prompt["modulesConfig"]
  promptFormat?: Prompt["promptFormat"]
  promptSchemaVersion?: Prompt["promptSchemaVersion"]
  structuredPromptDefinition?: Prompt["structuredPromptDefinition"]
  syncPayloadVersion?: Prompt["syncPayloadVersion"]
  versionNumber?: Prompt["versionNumber"]
  changeDescription?: Prompt["changeDescription"]
  parentVersionId?: Prompt["parentVersionId"]
  serverParentVersionId?: Prompt["serverParentVersionId"]
  is_system?: boolean
  tags?: string[]
  keywords?: string[]
  favorite?: boolean
}) => {
  const db = new PageAssistDatabase()
  const id = generateID()
  const createdAt = Date.now()
  const promptName = name || title
  const resolvedKeywords = keywords ?? tags
  const resolvedContent =
    content ??
    (is_system ? system_prompt : user_prompt) ??
    system_prompt ??
    user_prompt ??
    ""

  const prompt = {
    id,
    title: promptName,
    name: promptName,
    content: resolvedContent,
    is_system: !!is_system,
    createdAt,
    updatedAt: createdAt,
    tags: resolvedKeywords,
    keywords: resolvedKeywords,
    favorite,
    usageCount: 0,
    lastUsedAt: null,
    author,
    details,
    system_prompt: system_prompt ?? (is_system ? resolvedContent : undefined),
    user_prompt: user_prompt ?? (!is_system ? resolvedContent : undefined),
    promptFormat: promptFormat ?? 'legacy',
    promptSchemaVersion: promptSchemaVersion ?? null,
    structuredPromptDefinition: structuredPromptDefinition ?? null,
    syncPayloadVersion: syncPayloadVersion ?? 1,
    fewShotExamples: fewShotExamples ?? null,
    modulesConfig: modulesConfig ?? null,
    versionNumber: versionNumber ?? null,
    changeDescription: changeDescription ?? null,
    parentVersionId: parentVersionId ?? null,
    serverParentVersionId: serverParentVersionId ?? null,
    // Default sync values for new prompts
    syncStatus: 'local' as const,
    sourceSystem: 'workspace' as const
  }
  await db.addPrompt(prompt)
  await savePromptFB(prompt)
  return prompt
}

export const deletePromptById = async (id: string) => {
  // Soft delete: moves prompt to trash
  const db = new PageAssistDatabase()
  await db.deletePrompt(id)
  // Note: Firefox storage doesn't support soft delete, so we keep it there until permanent delete
  return id
}

export const permanentlyDeletePrompt = async (id: string) => {
  // Hard delete: removes from both Dexie and Firefox storage
  const db = new PageAssistDatabase()
  await db.permanentlyDeletePrompt(id)
  await deletePromptByIdFB(id)
  return id
}

export const restorePrompt = async (id: string) => {
  // Restore from trash
  const db = new PageAssistDatabase()
  await db.restorePrompt(id)
  return id
}

export const getDeletedPrompts = async () => {
  try {
    const db = new PageAssistDatabase()
    return await db.getDeletedPrompts()
  } catch (e) {
    if (isDatabaseClosedError(e)) {
      // Firefox storage doesn't support soft delete tracking
      return []
    }
    return []
  }
}

export const emptyTrash = async () => {
  const db = new PageAssistDatabase()
  const deletedPrompts = await db.getDeletedPrompts()
  // Also remove from Firefox storage
  for (const prompt of deletedPrompts) {
    await deletePromptByIdFB(prompt.id)
  }
  return await db.emptyTrash()
}

export const autoCleanupTrash = async (maxAgeDays: number = 30) => {
  const db = new PageAssistDatabase()
  // Get prompts that will be deleted
  const cutoff = Date.now() - (maxAgeDays * 24 * 60 * 60 * 1000)
  const deletedPrompts = await db.getDeletedPrompts()
  const expiredPrompts = deletedPrompts.filter(p => p.deletedAt && p.deletedAt < cutoff)
  // Remove from Firefox storage
  for (const prompt of expiredPrompts) {
    await deletePromptByIdFB(prompt.id)
  }
  return await db.autoCleanupTrash(maxAgeDays)
}

export const updatePrompt = async ({
  content,
  id,
  title,
  name,
  author,
  details,
  system_prompt,
  user_prompt,
  fewShotExamples,
  modulesConfig,
  promptFormat,
  promptSchemaVersion,
  structuredPromptDefinition,
  syncPayloadVersion,
  versionNumber,
  changeDescription,
  parentVersionId,
  serverParentVersionId,
  is_system,
  tags = [],
  keywords,
  favorite,
  usageCount,
  lastUsedAt
}: {
  id: string
  title?: string
  name?: string
  content?: string
  author?: string
  details?: string
  system_prompt?: string
  user_prompt?: string
  fewShotExamples?: Prompt["fewShotExamples"]
  modulesConfig?: Prompt["modulesConfig"]
  promptFormat?: Prompt["promptFormat"]
  promptSchemaVersion?: Prompt["promptSchemaVersion"]
  structuredPromptDefinition?: Prompt["structuredPromptDefinition"]
  syncPayloadVersion?: Prompt["syncPayloadVersion"]
  versionNumber?: Prompt["versionNumber"]
  changeDescription?: Prompt["changeDescription"]
  parentVersionId?: Prompt["parentVersionId"]
  serverParentVersionId?: Prompt["serverParentVersionId"]
  is_system?: boolean
  tags?: string[]
  keywords?: string[]
  favorite?: boolean
  usageCount?: number
  lastUsedAt?: number | null
}) => {
  const db = new PageAssistDatabase()
  const resolvedKeywords = keywords ?? tags
  const resolvedContent =
    content ??
    (is_system ? system_prompt : user_prompt) ??
    system_prompt ??
    user_prompt

  const payload = {
    title: name || title,
    name: name || title,
    content: resolvedContent,
    is_system,
    tags: resolvedKeywords,
    keywords: resolvedKeywords,
    favorite,
    usageCount,
    lastUsedAt,
    author,
    details,
    system_prompt,
    user_prompt,
    promptFormat,
    promptSchemaVersion,
    structuredPromptDefinition,
    syncPayloadVersion,
    fewShotExamples,
    modulesConfig,
    versionNumber,
    changeDescription,
    parentVersionId,
    serverParentVersionId
  }

  await db.updatePrompt(id, payload)
  await updatePromptFB({
    id,
    ...payload
  })
  return id
}

export const incrementPromptUsage = async (id: string) => {
  if (!id || id.trim().length === 0) return null
  const db = new PageAssistDatabase()
  return db.incrementPromptUsage(id)
}

export const getPromptById = async (id: string) => {
  try {
    if (!id || id.trim() === "") return null
    const db = new PageAssistDatabase()
    return await db.getPromptById(id)
  } catch (e) {
    if (isDatabaseClosedError(e)) {
      return await getPromptByIdFB(id)
    }
    return null
  }
}

// Webshare Functions
export const getAllWebshares = async () => {
  try {
    const db = new PageAssistDatabase()
    return await db.getAllWebshares()
  } catch (e) {
    return []
  }
}

export const deleteWebshare = async (id: string) => {
  const db = new PageAssistDatabase()
  await db.deleteWebshare(id)
  return id
}

export const saveWebshare = async ({
  title,
  url,
  api_url,
  share_id
}: {
  title: string
  url: string
  api_url: string
  share_id: string
}) => {
  const db = new PageAssistDatabase()
  const id = generateID()
  const createdAt = Date.now()
  const webshare: Webshare = { id, title, url, share_id, createdAt, api_url }
  await db.addWebshare(webshare)
  return webshare
}

// User Functions
export const getUserId = async () => {
  const db = new PageAssistDatabase()
  const id = await db.getUserID()
  if (!id || id?.trim() === "") {
    const user_id = "user_xxxx-xxxx-xxx-xxxx-xxxx".replace(/[x]/g, () => {
      const r = Math.floor(Math.random() * 16)
      return r.toString(16)
    })
    await db.setUserID(user_id)
    return user_id
  }
  return id
}

// Export/Import Functions
export const exportChatHistory = async () => {
  const db = new PageAssistDatabase()
  const chatHistories = await db.getChatHistories()
  const results = await Promise.allSettled(
    chatHistories.map(async (history) => {
      const messages = await db.getChatHistory(history.id)
      return { history, messages }
    })
  )
  return results
    .filter(
      (r): r is PromiseFulfilledResult<{ history: (typeof chatHistories)[0]; messages: MessageHistory }> =>
        r.status === "fulfilled"
    )
    .map((r) => r.value)
}

export const importChatHistory = async (
  data: {
    history: HistoryInfo
    messages: MessageHistory
  }[]
) => {
  // Use bulk operations to reduce storage quota pressure (especially for Firefox fallback)
  const histories = data.map(d => d.history)
  const allMessages = data.flatMap(d => d.messages)

  // Dexie path: use bulk operations via importChatHistoryV2
  const db = new PageAssistDatabase()
  await db.importChatHistoryV2(data, { mergeData: true })
}

export const exportPrompts = async () => {
  const db = new PageAssistDatabase()
  return await db.getAllPrompts()
}

export const exportOAIConfigs = async () => {
  // OpenAI-compatible provider configs are no longer used; keep the
  // export shape but always return an empty list.
  return []
}

export const exportNicknames = async () => {
  const modelNickname = new ModelNickname()
  const data = await modelNickname.getAllModelNicknames()
  return data
}

export const exportModels = async () => {
  const db = new ModelDb()
  return db.getAll()
}

export const importNicknamesV2 = async (
  nicknames: ModelNicknames,
  options: {
    replaceExisting?: boolean
    mergeData?: boolean
  } = {}
) => {
  const db = new ModelNickname()
  await db.importDataV2(nicknames, options)
}

export const importModelsV2 = async (
  models: Models,
  options: {
    replaceExisting?: boolean
    mergeData?: boolean
  } = {}
) => {
  const db = new ModelDb()
  await db.importDataV2(models, options)
}

export const importPrompts = async (prompts: Prompts) => {
  // Use bulk operations to reduce storage quota pressure
  const db = new PageAssistDatabase()
  await db.importPromptsV2(prompts, { mergeData: true })
}

export const importOAIConfigs = async (configs: any[]) => {
  // Legacy OpenAI provider configs are ignored now that the
  // extension is tldw_server-only.
  void configs
}

// Utility Functions
export const getRecentChatFromCopilot = async () => {
  const db = new PageAssistDatabase()
  const chatHistories = await db.getChatHistories()
  if (chatHistories.length === 0) return null
  const history = chatHistories.find(
    (history) => history.message_source === "copilot"
  )
  if (!history) return null

  const messages = await db.getChatHistory(history.id)

  return { history, messages }
}

export const getRecentChatFromWebUI = async () => {
  const db = new PageAssistDatabase()
  const chatHistories = await db.getChatHistories()
  if (chatHistories.length === 0) return null
  const history = chatHistories.find(
    (history) => history.message_source === "web-ui"
  )
  if (!history) return null

  const messages = await db.getChatHistory(history.id)

  return { history, messages }
}

export const getTitleById = async (id: string) => {
  const db = new PageAssistDatabase()
  const title = await db.getChatHistoryTitleById(id)
  return title
}

export const getLastChatHistory = async (history_id: string) => {
  const db = new PageAssistDatabase()
  const messages = await db.getChatHistory(history_id)
  messages.sort((a, b) => a.createdAt - b.createdAt)
  const lastMessage = messages[messages.length - 1]
  return normalizeChatRole(lastMessage?.role) === "assistant"
    ? lastMessage
    : messages.findLast((m) => normalizeChatRole(m.role) === "assistant")
}

export const deleteHistoriesByDateRange = async (
  rangeLabel: string
): Promise<string[]> => {
  const db = new PageAssistDatabase()
  const allHistories = await db.getChatHistories()
  const now = new Date()
  const today = new Date(now.setHours(0, 0, 0, 0))
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  const lastWeek = new Date(today)
  lastWeek.setDate(lastWeek.getDate() - 7)
  let historiesToDelete: HistoryInfo[] = []

  switch (rangeLabel) {
    case "today":
      historiesToDelete = allHistories.filter(
        (item) => !item.is_pinned && new Date(item?.createdAt) >= today
      )
      break
    case "yesterday":
      historiesToDelete = allHistories.filter(
        (item) =>
          !item.is_pinned &&
          new Date(item?.createdAt) >= yesterday &&
          new Date(item?.createdAt) < today
      )
      break
    case "last7Days":
      historiesToDelete = allHistories.filter(
        (item) =>
          !item.is_pinned &&
          new Date(item?.createdAt) >= lastWeek &&
          new Date(item?.createdAt) < yesterday
      )
      break
    case "older":
      historiesToDelete = allHistories.filter(
        (item) => !item.is_pinned && new Date(item?.createdAt) < lastWeek
      )
      break
    case "pinned":
      historiesToDelete = allHistories.filter((item) => item.is_pinned)
      break
    default:
      return []
  }

  const deletedIds: string[] = []
  for (const history of historiesToDelete) {
    await db.deleteMessage(history.id)
    await db.removeChatHistory(history.id)
    await db.deleteCompareState(history.id)
    deletedIds.push(history.id)
  }

  return deletedIds
}

// Session Files Helper Functions
export const getSessionFiles = async (
  sessionId: string
): Promise<UploadedFile[]> => {
  const db = new PageAssistDatabase()
  return await db.getSessionFiles(sessionId)
}

export const addFileToSession = async (
  sessionId: string,
  file: UploadedFile
) => {
  const db = new PageAssistDatabase()
  await db.addFileToSession(sessionId, file)
}

export const removeFileFromSession = async (
  sessionId: string,
  fileId: string
) => {
  const db = new PageAssistDatabase()
  await db.removeFileFromSession(sessionId, fileId)
}

export const updateFileInSession = async (
  sessionId: string,
  fileId: string,
  updates: Partial<UploadedFile>
) => {
  const db = new PageAssistDatabase()
  await db.updateFileInSession(sessionId, fileId, updates)
}

export const setRetrievalEnabled = async (
  sessionId: string,
  enabled: boolean
) => {
  const db = new PageAssistDatabase()
  await db.setRetrievalEnabled(sessionId, enabled)
}

export const getSessionFilesInfo = async (
  sessionId: string
): Promise<SessionFiles | null> => {
  const db = new PageAssistDatabase()
  return await db.getSessionFilesInfo(sessionId)
}

export const clearSessionFiles = async (sessionId: string) => {
  const db = new PageAssistDatabase()
  await db.clearSessionFiles(sessionId)
}
export const importChatHistoryV2 = async (
  data: any[],
  options: {
    replaceExisting?: boolean
    mergeData?: boolean
  } = {}
) => {
  const chatDb = new PageAssistDatabase()
  return chatDb.importChatHistoryV2(data, options)
}

export const importPromptsV2 = async (
  data: Prompt[],
  options: {
    replaceExisting?: boolean
    mergeData?: boolean
  } = {}
) => {
  const chatDb = new PageAssistDatabase()
  return chatDb.importPromptsV2(data, options)
}

export const importOAIConfigsV2 = async (
  data: any[],
  options: {
    replaceExisting?: boolean
    mergeData?: boolean
  } = {}
) => {
  // Legacy OpenAI provider configs are ignored.
  void data
  void options
}

export const updateLastUsedModel = async (
  history_id: string,
  model_id: string
) => {
  const chatDb = new PageAssistDatabase()
  return chatDb.updateLastUsedModel(history_id, model_id)
}

export const updateLastUsedPrompt = async (
  history_id: string,
  usedPrompt: LastUsedModelType
) => {
  const chatDb = new PageAssistDatabase()
  return chatDb.updateLastUsedPrompt(history_id, usedPrompt)
}

export const getHistoriesWithMetadata = async (historyIds: string[]) => {
  const db = new PageAssistDatabase()
  return db.getHistoriesWithMetadata(historyIds)
}
