import type {
  ConversationContextComposition,
  ConversationContextPiece,
  ConversationContextProviderMessage,
  ConversationContextSelection,
  ConversationContextSource
} from "@/types/conversation-context"
import { normalizeConversationContextIdList } from "./conversationContextSettings"

export type DictionaryProcessRequest = {
  text: string
  dictionary_id?: number | string
  dictionary_ids?: Array<number | string>
  max_iterations?: number
  token_budget?: number
  chat_id?: string
}

export type DictionaryProcessResponse = {
  original_text: string
  processed_text: string
  replacements: number
  iterations: number
  entries_used: number[]
  token_budget_exceeded?: boolean
  token_budget_used?: number | null
  processing_time_ms?: number | null
}

export type WorldBookProcessRequest = {
  text: string
  world_book_ids?: number[]
  character_id?: number
  scan_depth?: number
  token_budget?: number
  recursive_scanning?: boolean
}

export type WorldBookProcessDiagnostic = {
  entry_id?: number | null
  world_book_id?: number | null
  activation_reason?: string
  keyword?: string | null
  [key: string]: unknown
}

export type WorldBookProcessResponse = {
  injected_content: string
  entries_matched: number
  tokens_used: number
  books_used: number
  entry_ids: number[]
  token_budget?: number | null
  budget_exhausted?: boolean | null
  skipped_entries_due_to_budget?: number | null
  diagnostics: WorldBookProcessDiagnostic[]
}

export interface ConversationContextPrimitiveClient {
  processDictionary: (
    request: DictionaryProcessRequest
  ) => Promise<DictionaryProcessResponse>
  processWorldBookContext: (
    request: WorldBookProcessRequest
  ) => Promise<WorldBookProcessResponse>
}

export type ComposeConversationContextInput = {
  inputText: string
  selection: ConversationContextSelection
  inheritedWorldBookIds?: Array<number | string | null | undefined>
  primitives: ConversationContextPrimitiveClient
  dictionaryMaxIterations?: number
  dictionaryTokenBudget?: number
  worldBookScanDepth?: number
  worldBookTokenBudget?: number
  recursiveWorldBookScanning?: boolean
}

const normalizeSelection = (
  selection: ConversationContextSelection
): ConversationContextSelection => ({
  ...selection,
  characterId: selection.characterId ?? null,
  worldBookIds: normalizeConversationContextIdList(selection.worldBookIds),
  dictionaryIds: normalizeConversationContextIdList(selection.dictionaryIds)
})

const uniqueIds = (...lists: number[][]): number[] =>
  normalizeConversationContextIdList(lists.flat())

const numericCharacterId = (
  value: ConversationContextSelection["characterId"]
): number | undefined => {
  if (typeof value === "number" && Number.isInteger(value) && value > 0) {
    return value
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value)
    if (Number.isInteger(parsed) && parsed > 0) return parsed
  }
  return undefined
}

const buildContextSystemMessage = (
  sections: ConversationContextComposition["previewSections"]
): ConversationContextProviderMessage | null => {
  if (sections.length === 0) return null
  const content = sections
    .map((section) => `${section.name}:\n${section.content}`)
    .join("\n\n")
  return { role: "system", content }
}

const sourceForWorldBook = (
  id: number,
  explicitWorldBookIds: Set<number>,
  inheritedWorldBookIds: Set<number>
): ConversationContextSource => {
  if (explicitWorldBookIds.has(id)) return "explicit_chat"
  if (inheritedWorldBookIds.has(id)) return "character_inherited"
  return "request"
}

export const composeConversationContext = async ({
  inputText,
  selection,
  inheritedWorldBookIds,
  primitives,
  dictionaryMaxIterations = 5,
  dictionaryTokenBudget,
  worldBookScanDepth,
  worldBookTokenBudget,
  recursiveWorldBookScanning
}: ComposeConversationContextInput): Promise<ConversationContextComposition> => {
  const normalizedSelection = normalizeSelection(selection)
  const normalizedInheritedWorldBookIds =
    normalizeConversationContextIdList(inheritedWorldBookIds)
  const combinedWorldBookIds = uniqueIds(
    normalizedSelection.worldBookIds,
    normalizedInheritedWorldBookIds
  )
  const explicitWorldBookIds = new Set(normalizedSelection.worldBookIds)
  const inheritedWorldBookIdSet = new Set(normalizedInheritedWorldBookIds)
  const pieces: ConversationContextPiece[] = []
  const previewSections: ConversationContextComposition["previewSections"] = []
  const warnings: string[] = []
  let transformedInputText = inputText

  if (normalizedSelection.characterId !== null) {
    pieces.push({
      kind: "character",
      id: normalizedSelection.characterId,
      source: "explicit_chat",
      status: "configured"
    })
  }

  if (normalizedSelection.dictionaryIds.length > 0) {
    const dictionaryResult = await primitives.processDictionary({
      text: transformedInputText,
      dictionary_ids: normalizedSelection.dictionaryIds,
      max_iterations: dictionaryMaxIterations,
      token_budget: dictionaryTokenBudget,
      chat_id: normalizedSelection.chatId
    })
    transformedInputText =
      dictionaryResult.processed_text ?? transformedInputText

    for (const dictionaryId of normalizedSelection.dictionaryIds) {
      pieces.push({
        kind: "dictionary",
        id: dictionaryId,
        source: "explicit_chat",
        status: dictionaryResult.replacements > 0 ? "active" : "configured",
        diagnostics: {
          entries_used: dictionaryResult.entries_used,
          replacements: dictionaryResult.replacements,
          iterations: dictionaryResult.iterations
        }
      })
    }

    if (dictionaryResult.replacements > 0) {
      previewSections.push({
        name: "Dictionaries",
        content: transformedInputText,
        source: "explicit_chat"
      })
    }
    if (dictionaryResult.token_budget_exceeded) {
      warnings.push("Dictionary processing exceeded the token budget.")
    }
  }

  if (combinedWorldBookIds.length > 0) {
    const worldBookResult = await primitives.processWorldBookContext({
      text: transformedInputText,
      world_book_ids: combinedWorldBookIds,
      character_id: numericCharacterId(normalizedSelection.characterId),
      scan_depth: worldBookScanDepth,
      token_budget: worldBookTokenBudget,
      recursive_scanning: recursiveWorldBookScanning
    })
    const matchedWorldBookIds = new Set(
      (worldBookResult.diagnostics || [])
        .map((diagnostic) => diagnostic.world_book_id)
        .filter((id): id is number => typeof id === "number" && id > 0)
    )

    for (const worldBookId of combinedWorldBookIds) {
      const source = sourceForWorldBook(
        worldBookId,
        explicitWorldBookIds,
        inheritedWorldBookIdSet
      )
      pieces.push({
        kind: "worldbook",
        id: worldBookId,
        source,
        status: matchedWorldBookIds.has(worldBookId)
          ? "matched"
          : "configured",
        content: worldBookResult.injected_content,
        diagnostics: (worldBookResult.diagnostics || []).filter(
          (diagnostic) => diagnostic.world_book_id === worldBookId
        )
      })
    }

    if (worldBookResult.injected_content) {
      previewSections.push({
        name: "Worldbooks",
        content: worldBookResult.injected_content,
        source: "explicit_chat"
      })
    }
    if (worldBookResult.budget_exhausted) {
      warnings.push("Worldbook processing exhausted the token budget.")
    }
  }

  const contextMessage = buildContextSystemMessage(previewSections)
  const providerMessages = [
    ...(contextMessage ? [contextMessage] : []),
    { role: "user", content: transformedInputText }
  ]

  return {
    selection: normalizedSelection,
    inputText,
    transformedInputText,
    pieces,
    previewSections,
    providerMessages,
    readiness: warnings.length > 0 ? "partial" : "ready",
    warnings
  }
}
